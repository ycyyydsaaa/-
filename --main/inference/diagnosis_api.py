from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
from models.multimodal_model import MultiModalNet
from models.kg_builder import MedicalKG
from torchvision import transforms

# 全局配置
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
MODEL_PATH = "models/multimodal_model.pth"
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalNet(disease_cols)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 初始化知识图谱
kg = MedicalKG()


@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        # 获取上传数据：左右眼图像、年龄、性别、诊断关键词
        left_file = request.files.get('left_image')
        right_file = request.files.get('right_image')
        age = float(request.form.get('age'))
        gender = request.form.get('gender')  # "Female" 或 "Male"
        keywords = request.form.get('keywords', '')  # 关键词字符串，支持空格或逗号分隔
        if not all([left_file, right_file, age, gender]):
            return jsonify({"error": "缺少必要参数"}), 400

        # 图像处理
        left_img = Image.open(io.BytesIO(left_file.read())).convert('RGB')
        right_img = Image.open(io.BytesIO(right_file.read())).convert('RGB')
        left_img = IMG_TRANSFORM(left_img).unsqueeze(0).to(device)
        right_img = IMG_TRANSFORM(right_img).unsqueeze(0).to(device)

        # 文本处理
        text = [keywords]
        # 元数据处理
        meta = {
            'age': torch.tensor([age / 100.0], dtype=torch.float).to(device),
            'gender': torch.tensor([0 if gender.lower() == 'female' else 1], dtype=torch.float).to(device)
        }

        with torch.no_grad():
            logits, seg_output, decoupled_features = model(left_img, right_img, text, meta)
        probs = torch.sigmoid(logits)

        # 知识图谱校验（将关键词按空格分割）
        validation = kg.validate_prediction(logits, keywords.split(), disease_cols)

        # 生成诊断路径树
        diag_path = model.generate_diagnostic_path(logits)

        response = {
            "diagnosis": {d: validation[d]["adjusted_prob"] for d in disease_cols},
            "details": validation,
            "diagnostic_path": diag_path
        }
        return jsonify(response)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "服务器内部错误"}), 500


if __name__ == '__main__':
    app.run(debug=True)
