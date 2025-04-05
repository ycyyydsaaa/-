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
    def __init__(self, model, eye='both'):
        super(WrappedModel, self).__init__()
        self.model = model
        self.eye = eye

    def forward(self, img):
        batch_size = img.size(0)
        device = img.device
        meta = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
        if self.eye == 'left':
            img = img[:, :, :, :img.size(3)//2]
        elif self.eye == 'right':
            img = img[:, :, :, img.size(3)//2:]
        return self.model(img, None, meta, use_text=False)[0]

def evaluate(model, dataloader, device, disease_cols, thresholds=None):
    model.eval()
    correct = 0
    total = 0
    correct_per_disease = torch.zeros(len(disease_cols), device=device)
    all_preds = []
    all_labels = []
    all_probs = []

    # 初始化阈值
    if thresholds is None:
        thresholds = torch.full((len(disease_cols),), 0.5, device=device)
    else:
        thresholds = torch.tensor(thresholds, device=device)

    eval_bar = tqdm(total=len(dataloader), desc="Evaluation (No Text)", position=0, leave=True)

    # Grad-CAM初始化
    wrapped_model_left = WrappedModel(model, eye='left')
    wrapped_model_right = WrappedModel(model, eye='right')
    grad_cam_left = GradCAM(model=wrapped_model_left,
                          target_layers=[model.feature_extractor.efficientnet._conv_head])
    grad_cam_right = GradCAM(model=wrapped_model_right,
                           target_layers=[model.feature_extractor.efficientnet._conv_head])

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                eval_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)

            # 输入检查
            if torch.isnan(paired_img).any() or torch.isinf(paired_img).any():
                logger.warning(f"Batch {batch_idx}: paired_img contains NaN or Inf")
            if torch.isnan(meta).any() or torch.isinf(meta).any():
                logger.warning(f"Batch {batch_idx}: meta contains NaN or Inf")

            with autocast():
                logits, _, _, _, _ = model(paired_img, None, meta, use_text=False)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                probs = torch.sigmoid(logits)
                probs = torch.nan_to_num(probs, nan=0.5)
                preds = probs > thresholds

            # N排他性处理
            mask_n1 = preds[:, 0] == 1
            preds[mask_n1, 1:] = 0
            mask_no_disease = (preds[:, 1:].sum(dim=1) == 0) & (preds[:, 0] == 0)
            preds[mask_no_disease, 0] = 1

            # Grad-CAM可视化
            if batch_idx == 0:
                try:
                    grad_input = paired_img[0:1].clone().detach().requires_grad_(True)
                    img = paired_img[0].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_denorm = (img * std + mean).clip(0, 1)

                    positive_indices = torch.where(labels[0] == 1)[0]
                    if len(positive_indices) > 0:
                        for idx in positive_indices:
                            target_category = idx.item()
                            if target_category >= len(disease_cols):
                                continue
                            disease_name = disease_cols[target_category]
                            targets = [ClassifierOutputTarget(target_category)]

                            # 左眼热力图
                            with torch.enable_grad():
                                with autocast():
                                    grayscale_cam_left = grad_cam_left(input_tensor=grad_input, targets=targets)
                                    grayscale_cam_left = cv2.resize(grayscale_cam_left[0], (256, 256))
                                    vis_left = show_cam_on_image(img_denorm[:, :256, :], grayscale_cam_left, use_rgb=True)

                            # 右眼热力图
                            with torch.enable_grad():
                                with autocast():
                                    grayscale_cam_right = grad_cam_right(input_tensor=grad_input, targets=targets)
                                    grayscale_cam_right = cv2.resize(grayscale_cam_right[0], (256, 256))
                                    vis_right = show_cam_on_image(img_denorm[:, 256:, :], grayscale_cam_right, use_rgb=True)

                            # 合并热力图
                            vis_combined = np.hstack((vis_left, vis_right))
                            cv2.imwrite(f"gradcam_{disease_name}_combined.png", vis_combined * 255)
                            logger.info(f"生成热力图: gradcam_{disease_name}_combined.png")

                except Exception as e:
                    logger.error(f"GradCAM生成失败: {str(e)}")

            # 指标计算
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
    del grad_cam_left, grad_cam_right, wrapped_model_left, wrapped_model_right
    torch.cuda.empty_cache()

    if total == 0:
        return 0.0, torch.zeros(len(disease_cols)), 0.0, 0.0, 0.0

    # 合并结果
    full_match_accuracy = correct / total
    accuracy_per_disease = correct_per_disease / total
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
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
        except:
            optimal_thresholds[i] = 0.5

    # 动态阈值预测
    preds_dynamic = np.zeros_like(all_probs, dtype=np.int32)
    for i in range(len(disease_cols)):
        preds_dynamic[:, i] = (all_probs[:, i] > optimal_thresholds[i]).astype(np.int32)

    # N排他性处理
    mask_n1 = preds_dynamic[:, 0] == 1
    preds_dynamic[mask_n1, 1:] = 0
    mask_no_disease = (preds_dynamic[:, 1:].sum(axis=1) == 0) & (preds_dynamic[:, 0] == 0)
    preds_dynamic[mask_no_disease, 0] = 1

    # 计算最终指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    micro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)[2]
    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[2]
    weighted_f1_dynamic = precision_recall_fscore_support(all_labels, preds_dynamic, average='weighted', zero_division=0)[2]

    # 结果输出
    table_data = [[disease, f"{accuracy_per_disease[i].item():.4f}", f"{p:.4f}", f"{r:.4f}", f"{f:.4f}",
                   f"{int(correct_per_disease[i].item())}/{total}"]
                  for i, (disease, p, r, f) in enumerate(zip(disease_cols, precision, recall, f1))]
    headers = ["疾病", "准确率", "精确率", "召回率", "F1分数", "正确预测/总样本"]
    logger.info("\n评估结果：")
    logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))

    # 绘制图表
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(disease_cols))
        width = 0.2
        ax.bar([i - 1.5*width for i in x], accuracy_per_disease.cpu().numpy(), width, label='Accuracy')
        ax.bar([i - 0.5*width for i in x], precision, width, label='Precision')
        ax.bar([i + 0.5*width for i in x], recall, width, label='Recall')
        ax.bar([i + 1.5*width for i in x], f1, width, label='F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels(disease_cols)
        ax.legend()
        plt.savefig("evaluation_metrics.png")
        plt.close()
    except Exception as e:
        logger.error(f"图表生成失败: {str(e)}")

    return full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1, weighted_f1_dynamic

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    if not os.path.exists(TEST_EXCEL_PATH):
        from utils.split_dataset import split_dataset
        train_excel_path, test_excel_path = split_dataset(
            excel_path=EXCEL_PATH,
            test_size=0.2,
            random_state=42,
            disease_cols=disease_cols
        )
        logger.info(f"生成测试集: {test_excel_path}")

    test_dataset = FundusDataset(excel_path=TEST_EXCEL_PATH, img_root=IMG_ROOT, disease_cols=disease_cols, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                               collate_fn=custom_collate)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    kg_embeddings = checkpoint['kg_embeddings'].to(device)
    A = torch.load(os.path.join(KG_DATA_DIR, "adjacency_matrix.pt"), map_location=device)
    model = MultiModalNet(disease_cols=disease_cols, kg_embeddings=kg_embeddings, adjacency_matrix=A).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.initialize_kg_logits()

    # 从checkpoint加载阈值（如果存在）
    thresholds = checkpoint.get('thresholds', None)
    full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1, weighted_f1_dynamic = evaluate(
        model, test_dataloader, device, disease_cols, thresholds=thresholds
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"评估错误: {str(e)}")
        import traceback
        traceback.print_exc()