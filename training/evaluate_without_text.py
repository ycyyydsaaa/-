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
EXCEL_PATH = r"/data/eye/pycharm_project_257/data/Training_Dataset/labels.xlsx"
TEST_EXCEL_PATH = r"/data/eye/pycharm_project_257/data/Training_Dataset/labels_test.xlsx"
IMG_ROOT = r"/data/eye/pycharm_project_257/data/Training_Dataset/paired_dir"
MODEL_PATH = r"/data/eye/pycharm_project_257/models/multimodal_model.pth"
KG_DATA_DIR = r"/data/eye/pycharm_project_257/kg_data"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 8

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, paired_img):
        batch_size = paired_img.size(0)
        device = paired_img.device
        meta = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
        return self.model(paired_img, None, meta, use_text=False)[0]

def evaluate(model, dataloader, device, disease_cols):
    model.eval()
    correct = 0
    total = 0
    correct_per_disease = torch.zeros(len(disease_cols), device=device)
    all_preds = []
    all_labels = []
    all_probs = []

    eval_bar = tqdm(total=len(dataloader), desc="Evaluation (No Text)", position=0, leave=True)
    wrapped_model = WrappedModel(model)
    grad_cam = GradCAM(model=wrapped_model, target_layers=[model.feature_extractor.efficientnet._conv_head])

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                eval_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)

            # 检查输入是否包含 NaN
            if torch.isnan(paired_img).any() or torch.isinf(paired_img).any():
                logger.warning(f"Batch {batch_idx}: paired_img contains NaN or Inf")
            if torch.isnan(meta).any() or torch.isinf(meta).any():
                logger.warning(f"Batch {batch_idx}: meta contains NaN or Inf")

            with autocast():
                logits, _, _, _, _ = model(paired_img, None, meta, use_text=False)
                # 检查 logits 是否包含 NaN 或 Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.warning(f"Batch {batch_idx}: logits contains NaN or Inf, min={logits.min().item()}, max={logits.max().item()}")
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                probs = torch.sigmoid(logits)
                if torch.isnan(probs).any():
                    logger.warning(f"Batch {batch_idx}: probs contains NaN after sigmoid, min={probs.min().item()}, max={probs.max().item()}")
                    probs = torch.nan_to_num(probs, nan=0.5)
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
                            grad_logits = wrapped_model(grad_input)
                        positive_indices = torch.where(labels[0] == 1)[0]
                        if len(positive_indices) > 0:
                            img = paired_img[0].cpu().permute(1, 2, 0).numpy()
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            img = (img * std + mean).clip(0, 1)
                            for idx in positive_indices:
                                target_category = idx.item()
                                if target_category >= len(disease_cols):
                                    logger.warning(f"无效目标类别 {target_category}，跳过")
                                    continue
                                disease_name = disease_cols[target_category]
                                targets = [ClassifierOutputTarget(target_category)]
                                grayscale_cam = grad_cam(input_tensor=grad_input, targets=targets)
                                visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)
                                cv2.imwrite(f"gradcam_no_text_{disease_name}.png", visualization * 255)
                                logger.info(f"成功生成热力图: gradcam_no_text_{disease_name}.png")
                    del grad_input, grad_logits, grayscale_cam, visualization
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Batch 0 GradCAM 生成失败: {str(e)[:500]}，继续评估")

            # 使用布尔张量计算真阳性
            preds_bool = preds.bool()
            labels_bool = labels.bool()
            tp = (preds_bool & labels_bool).float().sum(dim=0)

            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            correct_per_disease += (preds == labels).float().sum(dim=0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            eval_bar.update(1)

    eval_bar.close()
    del grad_cam, wrapped_model
    torch.cuda.empty_cache()

    if total == 0:
        logger.warning("测试数据集为空，返回零值指标")
        return 0.0, torch.zeros(len(disease_cols)), 0.0, 0.0, 0.0

    full_match_accuracy = correct / total
    accuracy_per_disease = correct_per_disease / total
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # 检查 all_probs 是否包含 NaN
    if np.isnan(all_probs).any():
        logger.warning("all_probs contains NaN, replacing with 0.5")
        all_probs = np.nan_to_num(all_probs, nan=0.5)

    # 计算动态阈值
    optimal_thresholds = np.zeros(len(disease_cols))
    for i in range(len(disease_cols)):
        try:
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
        except ValueError as e:
            logger.warning(f"疾病 {disease_cols[i]} 的 ROC 曲线计算失败: {str(e)[:500]}，使用默认阈值 0.5")
            optimal_thresholds[i] = 0.5

    # 使用动态阈值生成预测，确保类型为 int32
    preds_dynamic = np.zeros_like(all_probs, dtype=np.int32)
    for i in range(len(disease_cols)):
        preds_dynamic[:, i] = (all_probs[:, i] > optimal_thresholds[i]).astype(np.int32)

    # 强制执行 'N' 的排他性
    mask_n1 = preds_dynamic[:, 0] == 1
    preds_dynamic[mask_n1, 1:] = 0
    mask_no_disease = (preds_dynamic[:, 1:].sum(axis=1) == 0) & (preds_dynamic[:, 0] == 0)
    preds_dynamic[mask_no_disease, 0] = 1

    # 确保 all_labels 和 preds_dynamic 类型一致
    all_labels = all_labels.astype(np.int32)
    all_preds = all_preds.astype(np.int32)

    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    micro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)[2]
    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[2]
    weighted_f1 = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[2]
    weighted_f1_dynamic = precision_recall_fscore_support(all_labels, preds_dynamic, average='weighted', zero_division=0)[2]

    # 输出结果
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

    # 绘制柱状图
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
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate)
        logger.info(f"测试数据集大小: {len(test_dataset)}")
    except Exception as e:
        raise RuntimeError(f"测试数据集加载失败: {str(e)[:500]}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        kg_embeddings = checkpoint['kg_embeddings'].to(device)
        A = torch.load(os.path.join(KG_DATA_DIR, "adjacency_matrix.pt"), map_location=device)
        model = MultiModalNet(disease_cols=disease_cols, kg_embeddings=kg_embeddings, adjacency_matrix=A).to(device)
        # 检查模型权重是否包含 NaN
        for name, param in model.state_dict().items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"模型参数 {name} 包含 NaN 或 Inf")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.initialize_kg_logits()  # 确保 kg_logits 被正确初始化（会跳过重新计算）
        logger.info(f"模型已从 {MODEL_PATH} 加载，kg_logits 已从 state_dict 恢复")
        logger.info(f"加载后的 kg_logits: {model.kg_logits}")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)[:500]}")

    full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1, weighted_f1_dynamic = evaluate(model, test_dataloader, device, disease_cols)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"评估过程中发生错误: {str(e)[:500]}")
        import traceback
        traceback.print_exc()