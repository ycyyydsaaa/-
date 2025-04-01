import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from models.kg_builder import MedicalKG
from utils.data_loader import FundusDataset, logger
from utils.split_dataset import split_dataset
from models.multimodal_model import MultiModalNet
import math
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from tqdm import tqdm
import logging
import psutil
import gc
import sys
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)

EXCEL_PATH = r"/data/eye/pycharm_project_257/data/Training_Dataset/labels.xlsx"
IMG_ROOT = r"/data/eye/pycharm_project_257/data/Training_Dataset/paired_dir"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
TEXT_REG_LAMBDA = 0.005

# 定义评估指标权重
PRECISION_WEIGHT = 0.5
RECALL_WEIGHT = 0.3
MICRO_F1_WEIGHT = 0.1
MACRO_F1_WEIGHT = 0.1


def print_tensor_info(tensor, name):
    if tensor is not None:
        memory_mb = tensor.element_size() * tensor.nelement() / 1024 ** 2
        ref_count = sys.getrefcount(tensor)
        print(f"Tensor {name}: Memory = {memory_mb:.2f} MB, Ref Count = {ref_count}")
    else:
        print(f"Tensor {name}: None")


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {
            'paired_image': torch.empty(0, 3, 256, 512),
            'text_feature': None,
            'meta': torch.empty(0, 2),
            'labels': torch.empty(0, len(disease_cols))
        }
    return torch.utils.data.dataloader.default_collate(batch)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2


def contrastive_loss(img_feat, text_feat, temperature=0.07):
    img_feat = F.normalize(img_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    logits = img_feat @ text_feat.T / temperature
    labels = torch.arange(img_feat.size(0), device=img_feat.device)
    return F.cross_entropy(logits, labels)


def evaluate(model, dataloader, device, disease_cols):
    model.eval()
    correct = 0
    total = 0
    correct_per_disease = torch.zeros(len(disease_cols), device=device)
    precision_sum = torch.zeros(len(disease_cols), device=device)
    recall_sum = torch.zeros(len(disease_cols), device=device)
    f1_sum = torch.zeros(len(disease_cols), device=device)
    true_positives = torch.zeros(len(disease_cols), device=device)
    predicted_positives = torch.zeros(len(disease_cols), device=device)
    actual_positives = torch.zeros(len(disease_cols), device=device)

    # 初始化进度条，不频繁更新
    eval_bar = tqdm(total=len(disease_cols), desc="Evaluation", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)

            logits, global_feat_weighted, kg_logits, _, _ = model(paired_img, None, meta, use_text=False,
                                                                  batch_idx=batch_idx)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            preds = torch.sigmoid(logits) > 0.5

            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            correct_per_disease += (preds == labels).float().sum(dim=0)

            tp = (preds.float() * labels).sum(dim=0)
            pred_pos = preds.float().sum(dim=0)
            actual_pos = labels.float().sum(dim=0)

            true_positives += tp
            predicted_positives += pred_pos
            actual_positives += actual_pos

            precision = tp / (pred_pos + 1e-6)
            recall = tp / (actual_pos + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1

            del paired_img, meta, labels, logits, preds, global_feat_weighted
            torch.cuda.empty_cache()
            gc.collect()

    # 一次性更新进度条到完成状态
    eval_bar.update(len(dataloader))
    eval_bar.close()
    torch.cuda.empty_cache()
    gc.collect()

    if total == 0:
        return 0.0, torch.zeros(len(disease_cols)), 0.0, 0.0, 0.0, 0.0, 0.0

    full_match_accuracy = correct / total
    accuracy_per_disease = correct_per_disease / total
    precision_avg = precision_sum / len(dataloader)
    recall_avg = recall_sum / len(dataloader)
    f1_avg = f1_sum / len(dataloader)

    micro_tp = true_positives.sum()
    micro_pred_pos = predicted_positives.sum()
    micro_actual_pos = actual_positives.sum()
    micro_precision = micro_tp / (micro_pred_pos + 1e-6)
    micro_recall = micro_tp / (micro_actual_pos + 1e-6)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-6)

    macro_f1 = f1_avg.mean()

    table_data = [[disease, f"{accuracy_per_disease[i].item():.4f}", f"{precision_avg[i].item():.4f}",
                   f"{recall_avg[i].item():.4f}", f"{f1_avg[i].item():.4f}",
                   f"{int(correct_per_disease[i].item())}/{total}"]
                  for i, disease in enumerate(disease_cols)]

    headers = ["疾病", "准确率", "精确率", "召回率", "F1 分数", "正确预测/总样本"]
    tqdm.write("\n测试集评估结果：")
    tqdm.write(f"全匹配准确度: {full_match_accuracy:.4f}")
    tqdm.write(f"Micro Precision: {micro_precision.item():.4f}")
    tqdm.write(f"Micro Recall: {micro_recall.item():.4f}")
    tqdm.write(f"Micro F1 分数: {micro_f1.item():.4f}")
    tqdm.write(f"Macro F1 分数: {macro_f1.item():.4f}")
    tqdm.write("\n逐疾病评估指标：")
    tqdm.write(tabulate(table_data, headers=headers, tablefmt="grid"))

    minority_classes = ['A', 'H', 'D', 'G']
    minority_indices = [disease_cols.index(cls) for cls in minority_classes]
    minority_f1 = f1_avg[minority_indices]
    tqdm.write("少数类 F1 分数：")
    for cls, f1 in zip(minority_classes, minority_f1):
        tqdm.write(f"{cls}: {f1.item():.4f}")

    del correct_per_disease, precision_sum, recall_sum, f1_sum, true_positives, predicted_positives, actual_positives
    torch.cuda.empty_cache()
    gc.collect()

    return (
        full_match_accuracy,
        accuracy_per_disease,
        micro_f1.item(),
        macro_f1.item(),
        micro_precision.item(),
        micro_recall.item(),
        minority_f1.mean().item()
    )


def plot_learning_rate_curve(learning_rates, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, marker='o', linestyle='-', color='b')
    plt.title('Learning Rate Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    tqdm.write(f"学习率曲线已保存到 {save_path}")


def compute_alpha_weights(df, disease_cols, device):
    """计算 Focal Loss 的 alpha 权重，确保值在 (0,1) 范围内"""
    alpha_weights = []
    for col in disease_cols:
        mean_pos = df[col].mean()  # 正样本比例
        alpha_i = 1 / (1 + math.exp(-(1 / (mean_pos + 1e-3) - 1)))  # 使用 sigmoid 映射到 (0,1)
        alpha_weights.append(alpha_i)
    return torch.tensor(alpha_weights, device=device)


class StableFocalLoss(nn.Module):
    """修正后的Focal Loss实现，确保数值稳定性"""

    def __init__(self, gamma=3.0, alpha=None):  # 修改 gamma 为 3.0
        super(StableFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        with torch.no_grad():
            p = torch.sigmoid(inputs)
            p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
            p_t = p * targets + (1 - p) * (1 - targets)
            modulating_factor = (1.0 - p_t + 1e-6).pow(self.gamma)
        focal_loss = modulating_factor * bce_loss
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def train():
    logger.info("开始执行 train 函数")
    torch.cuda.empty_cache()
    gc.collect()
    initial_memory = get_memory_usage()
    tqdm.write(f"初始内存使用量: {initial_memory:.2f} MB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"使用的设备: {device}")
    if device.type == "cuda":
        tqdm.write(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
        tqdm.write(f"CUDA 版本: {torch.version.cuda}")
    else:
        tqdm.write("警告: 未检测到 GPU，程序将使用 CPU 运行")

    # 数据加载和预处理
    train_excel_path, test_excel_path = split_dataset(
        excel_path=EXCEL_PATH, test_size=0.2, random_state=42, disease_cols=disease_cols)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop((256, 512), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FundusDataset(excel_path=train_excel_path, img_root=IMG_ROOT,
                                  disease_cols=disease_cols, transform=train_transform)
    test_dataset = FundusDataset(excel_path=test_excel_path, img_root=IMG_ROOT,
                                 disease_cols=disease_cols, transform=test_transform)

    df = pd.read_excel(train_excel_path)
    labels = torch.tensor(df[disease_cols].values, dtype=torch.float32)
    class_counts = labels.sum(dim=0)
    class_weights = torch.log(1.0 / (class_counts + 1e-6))  # 修改为对数缩放
    class_weights = class_weights / class_weights.max()
    sample_weights = torch.zeros(len(train_dataset))
    for idx in range(len(train_dataset)):
        label = labels[idx]
        weight = (label * class_weights).sum()
        sample_weights[idx] = weight * 2
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=min(4, os.cpu_count() - 1), pin_memory=True,
                                  collate_fn=custom_collate, persistent_workers=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=min(4, os.cpu_count() - 1), pin_memory=True,
                                 collate_fn=custom_collate, persistent_workers=False)

    # 知识图谱构建
    kg = MedicalKG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="120190333",
        local_dir="/data/eye/pycharm_project_257/kg_data"
    )
    kg.build_kg(df, disease_cols, batch_size=1000)
    kg_embeddings = kg.generate_disease_embeddings().to(device)
    A = kg.get_adjacency_matrix().to(device)
    tqdm.write(
        f"kg_embeddings shape: {kg_embeddings.shape}, size: {kg_embeddings.element_size() * kg_embeddings.nelement() / 1024 ** 2:.2f} MB")
    tqdm.write(f"adjacency_matrix shape: {A.shape}, size: {A.element_size() * A.nelement() / 1024 ** 2:.2f} MB")
    kg._save_local_data()

    # 模型初始化
    model = MultiModalNet(disease_cols=disease_cols, kg_embeddings=kg_embeddings, adjacency_matrix=A).to(device)
    model = model.float()
    model.initialize_kg_logits()

    # 损失函数
    alpha_weights = compute_alpha_weights(df, disease_cols, device)
    criterion_cls = StableFocalLoss(gamma=3.0, alpha=alpha_weights)  # 修改 gamma

    # 优化器和学习率调度器
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 5e-5},
        {'params': model.text_proj.parameters(), 'lr': 1e-4},
        {'params': model.feat_adapter.parameters(), 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters()
                    if not n.startswith(('feature_extractor', 'text_proj', 'feat_adapter'))], 'lr': 3e-4}
    ], weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[1e-4, 1e-4, 1e-4, 2e-4],  # 降低 max_lr
        total_steps=EPOCHS * len(train_dataloader),
        pct_start=0.5,  # 延长上升期
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=10.0,
        final_div_factor=1e3
    )

    # 训练变量初始化
    best_score = 0.0
    patience = 15
    epochs_no_improve = 0
    learning_rates = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}")
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None or len(batch['paired_image']) == 0:
                epoch_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)
            text_feature = batch.get('text_feature', None)

            if text_feature is not None and torch.rand(1).item() < 0.9:
                text_feature = None
            use_text = text_feature is not None
            if use_text:
                text_feature = text_feature.to(device, dtype=torch.float32)
                if text_feature.dim() == 1:
                    text_feature = text_feature.unsqueeze(0)
                elif text_feature.dim() == 3:
                    text_feature = text_feature.squeeze(1)
                expected_batch_size = paired_img.size(0)
                if text_feature.size(0) != expected_batch_size:
                    text_feature = text_feature.repeat(expected_batch_size, 1)

            logits, global_feat_weighted, kg_logits, _, _ = model(paired_img, text_feature, meta, use_text=use_text,
                                                                  batch_idx=batch_idx)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Invalid logits detected at batch {batch_idx}, skipping...")
                continue

            loss_cls = criterion_cls(logits, labels)
            text_reg_loss = 0.0
            if use_text:
                for name, param in model.text_proj.named_parameters():
                    text_reg_loss += torch.norm(param, p=2)
                text_reg_loss = TEXT_REG_LAMBDA * text_reg_loss
                text_feat = model.text_proj(text_feature)
                text_feat = model.feat_adapter(text_feat)
                contrastive_loss_val = 0.1 * contrastive_loss(global_feat_weighted, text_feat)
                total_loss = (loss_cls + text_reg_loss + contrastive_loss_val)
            else:
                total_loss = loss_cls

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 增大 max_norm
            if batch_idx % 50 == 0:
                tqdm.write(f"Batch {batch_idx}, Grad Norm: {grad_norm:.4f}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += total_loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # 中间验证
            if batch_idx % 144 == 0:
                model.eval()
                with torch.no_grad():
                    val_results = evaluate(model, test_dataloader, device, disease_cols)
                    micro_f1 = val_results[2]
                    tqdm.write(f"Epoch {epoch + 1}, Batch {batch_idx}, Micro F1: {micro_f1:.4f}")
                model.train()

            epoch_bar.set_postfix({'loss': f"{running_loss / (batch_idx + 1):.4f}", 'lr': f"{current_lr:.2e}"})
            epoch_bar.update(1)

            del paired_img, meta, labels, logits, kg_logits, loss_cls, total_loss, global_feat_weighted
            if text_feature is not None:
                del text_feature
            torch.cuda.empty_cache()
            gc.collect()

        epoch_bar.close()
        avg_loss = running_loss / len(train_dataloader)
        tqdm.write(f"Epoch {epoch + 1}/{EPOCHS} Loss: {avg_loss:.4f}")

        val_results = evaluate(model, test_dataloader, device, disease_cols)
        full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1, micro_precision, micro_recall, minority_f1 = val_results

        score = (PRECISION_WEIGHT * micro_precision +
                 RECALL_WEIGHT * micro_recall +
                 MICRO_F1_WEIGHT * micro_f1 +
                 MACRO_F1_WEIGHT * macro_f1)

        if score > best_score:
            best_score = score
            epochs_no_improve = 0
            save_path = "/data/eye/pycharm_project_257/models/best_multimodal_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'kg_embeddings': model.kg_embeddings,
                'disease_cols': disease_cols,
                'scheduler_state': scheduler.state_dict()
            }, save_path)
            tqdm.write(f"保存最佳模型，Total Score: {score:.4f}, Minority F1: {minority_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {epoch + 1}, best Total Score: {best_score:.4f}")
                break

        current_memory = get_memory_usage()
        tqdm.write(f"Epoch {epoch + 1} 结束后内存使用量: {current_memory:.2f} MB")
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
            tqdm.write(f"GPU 内存使用量: {gpu_mem:.2f} MB")
        torch.cuda.empty_cache()
        gc.collect()

    save_path = "/data/eye/pycharm_project_257/models/multimodal_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'kg_embeddings': model.kg_embeddings,
        'disease_cols': disease_cols,
        'scheduler_state': scheduler.state_dict()
    }, save_path)
    tqdm.write(f"模型已保存到 {save_path}")

    plot_learning_rate_curve(learning_rates,
                             "/data/eye/pycharm_project_257/plots/learning_rate_curve.png")

    model.clear_resources()
    kg.clear_cache()
    train_dataset.clear_cache()
    test_dataset.clear_cache()
    del model, optimizer, scheduler, train_dataset, train_dataloader, test_dataset, test_dataloader, kg, df
    if 'kg_embeddings' in locals():
        del kg_embeddings
    if 'A' in locals():
        del A
    torch.cuda.empty_cache()
    gc.collect()

    final_memory = get_memory_usage()
    tqdm.write(f"最终内存使用量: {final_memory:.2f} MB")
    tqdm.write(f"内存变化: {final_memory - initial_memory:.2f} MB")


if __name__ == "__main__":
    train()