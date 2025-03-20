import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from models.multimodal_model import MultiModalNet
from models.kg_builder import MedicalKG
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import BertTokenizer, BertModel
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
MODEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\models\multimodal_model.pth"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型和知识图谱
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

try:
    kg = MedicalKG(uri="bolt://localhost:7687", user="neo4j", password="120190333")
    kg.disease_cols = disease_cols
    kg_embeddings = kg.generate_disease_embeddings().to(device)
    A = kg.get_adjacency_matrix().to(device)
    logger.info("从 Neo4j 加载知识图谱嵌入完成")
except Exception as e:
    logger.error(f"知识图谱嵌入加载失败: {str(e)[:500]}")
    raise RuntimeError(f"知识图谱嵌入加载失败: {str(e)[:500]}")

try:
    model = MultiModalNet(disease_cols=disease_cols, kg_embeddings=kg_embeddings, adjacency_matrix=A).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.kg_embeddings = checkpoint['kg_embeddings'].to(device)
    model.eval()
    logger.info(f"模型已从 {MODEL_PATH} 加载")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)[:500]}")
    raise RuntimeError(f"模型加载失败: {str(e)[:500]}")

# 加载 BERT 模型和分词器
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_encoder = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
    logger.info("BERT 模型和分词器加载完成")
except Exception as e:
    logger.error(f"BERT 模型加载失败: {str(e)[:500]}")
    raise RuntimeError(f"BERT 模型加载失败: {str(e)[:500]}")

# 定义 WrappedModel 用于 Grad-CAM
class WrappedModel(nn.Module):
    def __init__(self, model, use_text=False, text_feature=None):
        super(WrappedModel, self).__init__()
        self.model = model
        self.use_text = use_text
        self.text_feature = text_feature

    def forward(self, paired_img):
        batch_size = paired_img.size(0)
        device = paired_img.device
        meta = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
        return self.model(paired_img, self.text_feature, meta, use_text=self.use_text)[0]

# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 512)),  # 调整为模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image

# 处理关键词，转换为 text_feature
def process_keywords(keywords):
    inputs = tokenizer(keywords, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 移动到设备
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    text_feature = outputs.last_hidden_state.mean(1).squeeze(0)  # [768]
    return text_feature

# 将 numpy 图像转换为 base64 字符串
def numpy_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

# 预测接口
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 接收前端数据
        if 'image' not in request.files:
            return jsonify({'error': '未提供图像文件'}), 400
        if 'age' not in request.form or 'gender' not in request.form:
            return jsonify({'error': '未提供年龄或性别'}), 400

        # 读取图像
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')

        # 读取年龄和性别
        age = float(request.form['age'])
        gender = int(request.form['gender'])  # 0: 女, 1: 男
        if gender not in [0, 1]:
            return jsonify({'error': '性别必须为 0（女）或 1（男）'}), 400

        # 读取关键词（可选）
        keywords = request.form.get('keywords', None)
        use_text = True if keywords else False
        text_feature = None
        if keywords:
            logger.info(f"接收到关键词: {keywords}，启用文本特征 (use_text=True)")
            text_feature = process_keywords(keywords).to(device, dtype=torch.float32)
        else:
            logger.info("未提供关键词，使用 use_text=False")

        # 2. 预处理图像和元数据
        paired_img = preprocess_image(image).to(device, dtype=torch.float32)
        meta = torch.tensor([age, gender], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 2]

        # 3. 模型预测
        with torch.no_grad():
            with autocast():
                logits, _, _, _, _ = model(paired_img, text_feature, meta, use_text=use_text)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()[0]  # [8]

        # 强制执行 'N' 的排他性
        if preds[0] == 1:  # 如果 'N' 为 1，其他疾病置为 0
            preds[1:] = 0
        elif preds[1:].sum() == 0:  # 如果没有其他疾病，'N' 置为 1
            preds[0] = 1

        # 4. 生成 Grad-CAM 热力图
        wrapped_model = WrappedModel(model, use_text=use_text, text_feature=text_feature)
        target_layer = model.feature_extractor.efficientnet._conv_head
        grad_cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

        # 准备 Grad-CAM 输入
        grad_input = paired_img.clone().detach().requires_grad_(True)
        positive_indices = torch.where(torch.tensor(preds) == 1)[0]

        # 准备原始图像用于叠加热力图
        img = paired_img[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean).clip(0, 1)

        heatmaps = []
        if len(positive_indices) > 0:
            for idx in positive_indices:
                target_category = idx.item()
                disease_name = disease_cols[target_category]
                targets = [ClassifierOutputTarget(target_category)]
                grayscale_cam = grad_cam(input_tensor=grad_input, targets=targets)
                grayscale_cam_clipped = np.clip(grayscale_cam[0], 0, np.percentile(grayscale_cam[0], 95))
                grayscale_cam_normalized = (grayscale_cam_clipped - grayscale_cam_clipped.min()) / (grayscale_cam_clipped.max() - grayscale_cam_clipped.min() + 1e-8)
                visualization = show_cam_on_image(img, grayscale_cam_normalized, use_rgb=True, image_weight=0.4)
                visualization = (visualization * 255).astype(np.uint8)
                heatmap_base64 = numpy_to_base64(visualization)
                heatmaps.append({
                    'disease': disease_name,
                    'heatmap': f"data:image/png;base64,{heatmap_base64}"
                })
        else:
            logger.warning("没有预测到任何疾病，未生成热力图")

        # 5. 构造返回结果
        result = {
            'predictions': preds.tolist(),  # [0, 1, 0, ...]
            'diseases': disease_cols,
            'heatmaps': heatmaps
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"预测过程中发生错误: {str(e)[:500]}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)