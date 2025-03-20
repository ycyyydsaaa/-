import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from utils.data_loader import FundusDataset
from utils.split_dataset import split_dataset
from models.multimodal_model import MultiModalNet
from models.kg_builder import MedicalKG
import math
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.WARNING)

# 数据路径和超参数定义
EXCEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\labels.xlsx"
IMG_ROOT = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\paired_dir"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 1
EPOCHS = 30
LR = 2e-5
WARMUP_EPOCHS = 5

# 自定义数据整理函数，用于过滤无效样本
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# 学习率调度器：Warmup + Cosine Decay
class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=30):
        self.warmup = warmup_epochs
        self.total = total_epochs
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [base_lr * (self.last_epoch + 1) / self.warmup for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup) / (self.total - self.warmup)
        return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

# 评估函数
def evaluate(model, dataloader, device, disease_cols):
    model.eval()
    correct = 0
    total = 0
    correct_per_disease = torch.zeros(len(disease_cols), device=device)
    all_preds = []
    all_labels = []

    eval_bar = tqdm(total=len(dataloader), desc="Evaluation", position=1, leave=False)
    grad_cam = GradCAM(model=model, target_layers=[model.feature_extractor.efficientnet._conv_head])

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                eval_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            text_feature = None
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)

            with autocast():
                logits, _, _, _, _ = model(paired_img, text_feature, meta)
                preds = torch.sigmoid(logits) > 0.5

            if batch_idx == 0:
                grad_input = paired_img[0:1].clone().detach().requires_grad_(True)
                with torch.enable_grad():
                    with autocast():
                        grad_logits, _, _, _, _ = model(grad_input, None, meta[0:1])
                    positive_indices = torch.where(labels[0] == 1)[0]
                    if len(positive_indices) > 0:
                        img = paired_img[0].cpu().permute(1, 2, 0).numpy()
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img = (img * std + mean).clip(0, 1)
                        for idx in positive_indices:
                            target_category = idx.item()
                            if target_category >= len(disease_cols):
                                tqdm.write(f"警告: 无效目标类别 {target_category}, 跳过")
                                continue
                            disease_name = disease_cols[target_category]
                            targets = [ClassifierOutputTarget(target_category)]
                            grayscale_cam = grad_cam(input_tensor=grad_input, targets=targets)
                            visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)
                            cv2.imwrite(f"gradcam_{disease_name}.png", visualization * 255)
                            tqdm.write(f"成功生成 {disease_name} 热力图")

            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            correct_per_disease += (preds == labels).float().sum(dim=0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            eval_bar.update(1)

    eval_bar.close()
    torch.cuda.empty_cache()

    if total == 0:
        return 0.0, torch.zeros(len(disease_cols)), 0.0, 0.0

    full_match_accuracy = correct / total
    accuracy_per_disease = correct_per_disease / total
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    micro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='micro')[2]
    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, average='macro')[2]

    table_data = [[disease, f"{accuracy_per_disease[i].item():.4f}", f"{precision[i]:.4f}",
                   f"{recall[i]:.4f}", f"{f1[i]:.4f}", f"{int(correct_per_disease[i].item())}/{total}"]
                  for i, disease in enumerate(disease_cols)]

    headers = ["疾病", "准确率", "精确率", "召回率", "F1 分数", "正确预测/总样本"]
    tqdm.write("\n测试集评估结果：")
    tqdm.write(f"全匹配准确度: {full_match_accuracy:.4f}")
    tqdm.write(f"Micro F1 分数: {micro_f1:.4f}")
    tqdm.write(f"Macro F1 分数: {macro_f1:.4f}")
    tqdm.write("\n逐疾病评估指标：")
    tqdm.write(tabulate(table_data, headers=headers, tablefmt="grid"))

    return full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1

# 训练函数
def train():
    train_excel_path, test_excel_path = split_dataset(
        excel_path=EXCEL_PATH,
        test_size=0.2,
        random_state=42,
        disease_cols=disease_cols
    )

    train_dataset = FundusDataset(excel_path=train_excel_path, img_root=IMG_ROOT, disease_cols=disease_cols, phase='train')
    test_dataset = FundusDataset(excel_path=test_excel_path, img_root=IMG_ROOT, disease_cols=disease_cols, phase='test')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kg = MedicalKG(uri="bolt://localhost:7687", user="neo4j", password="120190333")
    df = pd.read_excel(train_excel_path)
    kg.build_kg(df, disease_cols)
    kg_embeddings = kg.generate_disease_embeddings().to(device)
    A = kg.get_adjacency_matrix().to(device)
    print("Adjacency Matrix A:", A.cpu().numpy())

    model = MultiModalNet(disease_cols=disease_cols, kg_embeddings=kg_embeddings, adjacency_matrix=A).to(device)
    model = model.float()

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=None):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.bce = nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, inputs, targets):
            with autocast():
                bce_loss = self.bce(inputs, targets)
                bce_loss = torch.clamp(bce_loss, max=100.0)
                p_t = torch.exp(-bce_loss)
                focal_loss = ((1 - p_t) ** self.gamma) * bce_loss
                if self.alpha is not None:
                    alpha_t = self.alpha[targets.long()]
                    focal_loss = alpha_t * focal_loss
                return focal_loss.mean()

    pos_weight = torch.tensor([10.0 / df[col].mean() if df[col].mean() > 0.01 else 1.0 for col in disease_cols], device=device).clamp(max=100.0)
    criterion_cls = FocalLoss(gamma=2.0, alpha=pos_weight)

    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-5},
        {'params': model.text_proj.parameters(), 'lr': 3e-4},
        {'params': model.feat_adapter.parameters()},
        {'params': [p for n, p in model.named_parameters() if not n.startswith(('feature_extractor', 'text_proj', 'feat_adapter'))]}
    ], lr=LR)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS)
    scaler = GradScaler()

    accum_steps = 1  # 确保梯度累积步数为 1

    for epoch in range(EPOCHS):
        model.train()
        epoch_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}")
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            try:
                if batch is None:
                    epoch_bar.update(1)
                    continue

                paired_img = batch['paired_image'].to(device, dtype=torch.float32)
                meta = batch['meta'].to(device, dtype=torch.float32)
                labels = batch['labels'].to(device, dtype=torch.float32)

                if batch_idx == 0:
                    tqdm.write(f"paired_img device: {paired_img.device}, type: {paired_img.dtype}")
                    tqdm.write(f"model device: {next(model.parameters()).device}")
                    tqdm.write(f"Batch {batch_idx} - GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

                text_feature = batch.get('text_feature', None)
                if text_feature is not None:
                    text_feature = text_feature.to(device, dtype=torch.float32)
                if torch.rand(1).item() < 0.8:
                    text_feature = None

                with autocast():
                    logits, _, kg_logits, _, _ = model(paired_img, text_feature, meta, use_text=True)
                    loss_cls = criterion_cls(logits, labels)
                    total_loss = loss_cls

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                running_loss += total_loss.item()

                del paired_img, meta, labels, logits, kg_logits, loss_cls, total_loss
                if text_feature is not None:
                    del text_feature
                torch.cuda.empty_cache()

                epoch_bar.update(1)

            except Exception as e:
                tqdm.write(f"Error at Batch {batch_idx}: {str(e)}")
                raise

        epoch_bar.close()
        avg_loss = running_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        scheduler.step()
        tqdm.write(f"Epoch {epoch + 1}/{EPOCHS} Loss: {avg_loss:.4f}")

    tqdm.write("开始测试集评估...")
    evaluate(model, test_dataloader, device, disease_cols)

    save_path = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\models\multimodal_model.pth"
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'kg_embeddings': model.kg_embeddings,
        'disease_cols': disease_cols
    }, save_path)
    tqdm.write(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    train()