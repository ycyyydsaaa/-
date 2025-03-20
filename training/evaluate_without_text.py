import os
import torch
from torch.utils.data import DataLoader
from utils.data_loader import FundusDataset
from models.multimodal_model import MultiModalNet
from models.kg_builder import MedicalKG
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
EXCEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\labels.xlsx"
TEST_EXCEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\labels_test.xlsx"
IMG_ROOT = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\paired_dir"
MODEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\models\multimodal_model.pth"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 1

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def validate_neo4j_graph(kg, disease_cols):
    all_diseases = {record['d.name'] for record in kg.graph.run("MATCH (d:Disease) RETURN d.name").data()}
    expected_diseases = set(disease_cols)
    if all_diseases != expected_diseases:
        raise ValueError(f"Neo4j 中的疾病节点 {all_diseases} 与预期 {expected_diseases} 不一致")
    for d in disease_cols:
        symptoms = kg.query_disease_symptoms(d)
        if not symptoms:
            logger.warning(f"疾病 {d} 在 Neo4j 中没有关联症状，可能图数据不完整")
    logger.info("Neo4j 图数据验证通过")

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, paired_img):
        batch_size = paired_img.size(0)
        device = paired_img.device
        meta = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
        return self.model(paired_img, None, meta, use_text=False)[0]

def evaluate_without_text(model, dataloader, device, disease_cols):
    model.eval()
    correct = 0
    total = 0
    correct_per_disease = torch.zeros(len(disease_cols), device=device)
    all_preds = []
    all_labels = []
    all_probs = []

    eval_bar = tqdm(total=len(dataloader), desc="Evaluation (No Text)", position=0, leave=True)
    wrapped_model = WrappedModel(model)
    target_layer = model.feature_extractor.efficientnet._conv_head
    grad_cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                eval_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)

            with autocast():
                logits, _, _, _, _ = model(paired_img, None, meta, use_text=False)
                probs = torch.sigmoid(logits)
                preds = probs > 0.5

            # 强制执行 'N' 的排他性
            mask_n1 = preds[:, 0] == 1
            preds[mask_n1, 1:] = 0
            mask_no_disease = (preds[:, 1:].sum(dim=1) == 0) & (preds[:, 0] == 0)
            preds[mask_no_disease, 0] = 1

            if batch_idx == 0:
                try:
                    grad_input = paired_img[0:1].clone().detach().requires_grad_(True)
                    with torch.enable_grad():
                        with autocast():
                            grad_logits, _, _, _, _ = model(grad_input, None, meta[0:1], use_text=False)
                        img = paired_img[0].cpu().permute(1, 2, 0).numpy()
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img = (img * std + mean).clip(0, 1)

                        height, width = img.shape[:2]
                        img_left = img[:, :width // 2, :]
                        img_right = img[:, width // 2:, :]

                        positive_indices = torch.where(labels[0] == 1)[0]
                        if len(positive_indices) > 0:
                            for idx in positive_indices:
                                target_category = idx.item()
                                if target_category >= len(disease_cols):
                                    logger.warning(f"无效目标类别 {target_category}，跳过")
                                    continue
                                disease_name = disease_cols[target_category]
                                targets = [ClassifierOutputTarget(target_category)]
                                grayscale_cam = grad_cam(input_tensor=grad_input, targets=targets)
                                grayscale_cam_clipped = np.clip(grayscale_cam[0], 0, np.percentile(grayscale_cam[0], 95))
                                grayscale_cam_normalized = (grayscale_cam_clipped - grayscale_cam_clipped.min()) / (grayscale_cam_clipped.max() - grayscale_cam_clipped.min() + 1e-8)

                                cam_left = grayscale_cam_normalized[:, :width // 2]
                                cam_right = grayscale_cam_normalized[:, width // 2:]

                                vis_left = show_cam_on_image(img_left, cam_left, use_rgb=True, image_weight=0.4)
                                vis_right = show_cam_on_image(img_right, cam_right, use_rgb=True, image_weight=0.4)
                                vis_combined = np.hstack((vis_left, vis_right))
                                cv2.imwrite(f"gradcam_no_text_batch_{batch_idx}_disease_{disease_name}_split.png", vis_combined * 255)
                                logger.info(f"成功生成热力图: gradcam_no_text_batch_{batch_idx}_disease_{disease_name}_split.png")
                except Exception as e:
                    logger.error(f"Batch 0 GradCAM 生成失败: {str(e)[:500]}，继续评估")

            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            correct_per_disease += (preds == labels).float().sum(dim=0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            eval_bar.update(1)

    eval_bar.close()
    del grad_cam
    torch.cuda.empty_cache()

    if total == 0:
        logger.warning("测试数据集为空，返回零值指标")
        return 0.0, torch.zeros(len(disease_cols)), 0.0, 0.0, 0.0

    full_match_accuracy = correct / total
    accuracy_per_disease = correct_per_disease / total
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    optimal_thresholds = np.zeros(len(disease_cols))
    for i in range(len(disease_cols)):
        fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_probs[:, i])
        precision, recall, f1 = [], [], []
        for thresh in thresholds:
            preds_temp = (all_probs[:, i] > thresh).astype(int)
            p, r, f, _ = precision_recall_fscore_support(all_labels[:, i], preds_temp, average='binary', zero_division=0)
            precision.append(p)
            recall.append(r)
            f1.append(f)
        optimal_idx = np.argmax(f1)
        optimal_thresholds[i] = thresholds[optimal_idx]
        logger.info(f"疾病 {disease_cols[i]} 的最优阈值: {optimal_thresholds[i]:.4f}")

    preds_dynamic = np.zeros_like(all_probs)
    for i in range(len(disease_cols)):
        preds_dynamic[:, i] = (all_probs[:, i] > optimal_thresholds[i]).astype(int)

    mask_n1 = preds_dynamic[:, 0] == 1
    preds_dynamic[mask_n1, 1:] = 0
    mask_no_disease = (preds_dynamic[:, 1:].sum(axis=1) == 0) & (preds_dynamic[:, 0] == 0)
    preds_dynamic[mask_no_disease, 0] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    micro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)[2]
    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[2]
    weighted_f1 = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[2]
    weighted_f1_dynamic = precision_recall_fscore_support(all_labels, preds_dynamic, average='weighted', zero_division=0)[2]

    table_data = [[disease, f"{accuracy_per_disease[i].item():.4f}", f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", f"{int(correct_per_disease[i].item())}/{total}"]
                  for i, (disease, p, r, f) in enumerate(zip(disease_cols, precision, recall, f1))]
    headers = ["疾病", "准确率", "精确率", "召回率", "F1 分数", "正确预测/总样本"]
    logger.info("\n测试集评估结果（无文本特征）：")
    logger.info(f"全匹配准确度: {full_match_accuracy:.4f}")
    logger.info(f"Micro F1 分数: {micro_f1:.4f}")
    logger.info(f"Macro F1 分数: {macro_f1:.4f}")
    logger.info(f"Weighted F1 分数: {weighted_f1:.4f}")
    logger.info(f"动态阈值 Weighted F1 分数: {weighted_f1_dynamic:.4f}")
    logger.info("\n逐疾病评估指标：")
    logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))

    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(disease_cols))
        width = 0.2
        ax.bar([i - 1.5 * width for i in x], accuracy_per_disease.cpu().numpy(), width, label='Accuracy', color='skyblue')
        ax.bar([i - 0.5 * width for i in x], precision, width, label='Precision', color='lightgreen')
        ax.bar([i + 0.5 * width for i in x], recall, width, label='Recall', color='salmon')
        ax.bar([i + 1.5 * width for i in x], f1, width, label='F1 Score', color='gold')
        ax.set_xticks(x)
        ax.set_xticklabels(disease_cols, rotation=45, ha='right')
        ax.set_title("Per-Disease Evaluation Metrics (No Text Feature)")
        ax.set_xlabel("Disease")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("evaluation_metrics_no_text.png", dpi=300)
        plt.close()
        logger.info("评估指标柱状图已保存: evaluation_metrics_no_text.png")
    except Exception as e:
        logger.error(f"绘制柱状图失败: {str(e)[:500]}，评估结果已生成")

    return full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1, weighted_f1_dynamic

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    if not os.path.exists(TEST_EXCEL_PATH):
        logger.warning(f"测试集文件 {TEST_EXCEL_PATH} 不存在，正在从原始数据集中拆分...")
        from utils.split_dataset import split_dataset
        train_excel_path, test_excel_path = split_dataset(
            excel_path=EXCEL_PATH,
            test_size=0.2,
            random_state=42,
            disease_cols=disease_cols
        )
        logger.info(f"已生成测试集文件: {test_excel_path}")
    else:
        logger.info(f"找到测试集文件: {TEST_EXCEL_PATH}")

    try:
        test_dataset = FundusDataset(excel_path=TEST_EXCEL_PATH, img_root=IMG_ROOT, disease_cols=disease_cols, phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
        logger.info(f"测试数据集大小: {len(test_dataset)}")
    except Exception as e:
        raise RuntimeError(f"测试数据集加载失败: {str(e)[:500]}")

    try:
        kg = MedicalKG(uri="bolt://localhost:7687", user="neo4j", password="120190333")
        kg.disease_cols = disease_cols
        validate_neo4j_graph(kg, disease_cols)
        kg_embeddings = kg.generate_disease_embeddings().to(device)
        A = kg.get_adjacency_matrix().to(device)
        logger.info("从 Neo4j 加载知识图谱嵌入完成")
    except Exception as e:
        raise RuntimeError(f"知识图谱嵌入加载失败: {str(e)[:500]}")

    try:
        model = MultiModalNet(disease_cols=disease_cols, kg_embeddings=kg_embeddings, adjacency_matrix=A).to(device)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.kg_embeddings = checkpoint['kg_embeddings'].to(device)
        logger.info(f"模型已从 {MODEL_PATH} 加载")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)[:500]}")

    full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1, weighted_f1_dynamic = evaluate_without_text(model, test_dataloader, device, disease_cols)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"评估过程中发生错误: {str(e)[:500]}")
        import traceback
        traceback.print_exc()