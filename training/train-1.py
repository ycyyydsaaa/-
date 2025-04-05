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
from sklearn.metrics import precision_recall_curve
from tabulate import tabulate
from tqdm import tqdm
import logging
import psutil
import gc
import sys
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from memory_profiler import profile

logging.basicConfig(level=logging.WARNING)

EXCEL_PATH = r"/data/eye/pycharm_project_257/data/Training_Dataset/labels.xlsx"
IMG_ROOT = r"/data/eye/pycharm_project_257/data/Training_Dataset/paired_dir"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
TEXT_REG_LAMBDA = 0.005

# 定义评估指标权重
PRECISION_WEIGHT = 0.5
RECALL_WEIGHT = 0.3
MICRO_F1_WEIGHT = 0.1
MACRO_F1_WEIGHT = 0.1

def print_tensor_info(tensor, name):
    """打印张量的内存使用情况和引用计数"""
    if tensor is not None:
        memory_mb = tensor.element_size() * tensor.nelement() / 1024 ** 2
        ref_count = sys.getrefcount(tensor)
        print(f"Tensor {name}: Memory = {memory_mb:.2f} MB, Ref Count = {ref_count}")
    else:
        print(f"Tensor {name}: None")

def custom_collate(batch):
    """自定义批处理函数，处理空批次"""
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
    """获取当前进程的内存使用量 (MB)"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2

def contrastive_loss(img_feat, text_feat, temperature=0.07):
    """计算对比损失"""
    img_feat = F.normalize(img_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    logits = img_feat @ text_feat.T / temperature
    labels = torch.arange(img_feat.size(0), device=img_feat.device)
    return F.cross_entropy(logits, labels)

def evaluate(model, dataloader, device, disease_cols):
    """评估模型性能"""
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
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化
    # 用于动态阈值的收集
    all_probs = []  # 存储概率值（而不是原始 logits）
    all_labels = []

    # 初始化进度条
    # 修改进度条初始化（添加position和leave参数）
    eval_bar = tqdm(total=len(dataloader), desc="Evaluating", position=1, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)

            logits, global_feat_weighted, kg_logits, _, _ = model(paired_img, None, meta, use_text=False,
                                                                  batch_idx=batch_idx)
            # 将 logits 转换为概率
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)  # 处理 NaN 和无穷值

            # 收集概率和标签用于动态阈值计算
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # 更新进度条
            eval_bar.update(1)

            # 清理
            del paired_img, meta, labels, logits, global_feat_weighted, kg_logits, probs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
            eval_bar.set_postfix({'gpu_mem': f"{gpu_mem:.2f} MB"})

    eval_bar.close()

    # 将所有批次的概率和标签合并
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算每个类别的动态阈值并打印
    thresholds = []
    tqdm.write("\n动态阈值（基于 Precision-Recall 曲线）：")
    beta = 0.5  # 可调参数，越小越重视精确率
    for i, disease in enumerate(disease_cols):
        precision, recall, thresh = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        fbeta_scores = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-6)
        optimal_idx = np.argmax(fbeta_scores)
        optimal_threshold = thresh[optimal_idx] if optimal_idx < len(thresh) else 0.5
        optimal_threshold = np.clip(optimal_threshold, 0, 1)
        thresholds.append(optimal_threshold)
        tqdm.write(f"{disease}: {optimal_threshold:.4f}")

    # 使用动态阈值生成预测
    preds = torch.tensor(all_probs > np.array(thresholds), dtype=torch.float32, device=device)
    labels = torch.tensor(all_labels, dtype=torch.float32, device=device)

    # 计算指标
    correct = (preds == labels).all(dim=1).sum().item()
    total = labels.size(0)
    correct_per_disease = (preds == labels).float().sum(dim=0)

    tp = (preds * labels).sum(dim=0)
    pred_pos = preds.sum(dim=0)
    actual_pos = labels.sum(dim=0)

    true_positives = tp
    predicted_positives = pred_pos
    actual_positives = actual_pos

    precision = tp / (pred_pos + 1e-6)
    recall = tp / (actual_pos + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    precision_sum = precision
    recall_sum = recall
    f1_sum = f1

    if total == 0:
        return 0.0, torch.zeros(len(disease_cols)), 0.0, 0.0, 0.0, 0.0, 0.0

    full_match_accuracy = correct / total
    accuracy_per_disease = correct_per_disease / total
    precision_avg = precision_sum
    recall_avg = recall_sum
    f1_avg = f1_sum

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
    tqdm.write("\n测试集评估结果（基于动态阈值）：")
    tqdm.write(f"全匹配准确度: {full_match_accuracy:.4f}")
    tqdm.write(f"Micro Precision: {micro_precision.item():.4f}")
    tqdm.write(f"Micro Recall: {micro_recall.item():.4f}")
    tqdm.write(f"Micro F1 分数: {micro_f1.item():.4f}")
    tqdm.write(f"Macro F1 分数: {macro_f1.item():.4f}")
    tqdm.write("\n逐疾病评估指标：")
    tqdm.write(tabulate(table_data, headers=headers, tablefmt="grid"))

    minority_classes = ['H', 'A', 'G', 'C', 'M']  # 包含所有出现频率<10%的疾病
    minority_indices = [disease_cols.index(cls) for cls in minority_classes]
    minority_f1 = f1_avg[minority_indices]
    tqdm.write("少数类 F1 分数（基于动态阈值）：")
    for cls, f1 in zip(minority_classes, minority_f1):
        tqdm.write(f"{cls}: {f1.item():.4f}")

    # 清理
    del correct_per_disease, precision_sum, recall_sum, f1_sum, true_positives, predicted_positives, actual_positives
    del all_probs, all_labels, preds, labels
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
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
    """绘制学习率曲线并保存"""
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
    """计算类别权重"""
    alpha_weights = []
    for col in disease_cols:
        mean_pos = df[col].mean()
        alpha_i = 1 / (1 + math.exp(-(1 / (mean_pos + 1e-3) - 1)))
        alpha_weights.append(alpha_i)
    return torch.tensor(alpha_weights, device=device)

class StableFocalLoss(nn.Module):
    """稳定的 Focal Loss 实现"""
    def __init__(self, gamma=3.0, alpha=None, thresholds=None, minority_mask=None):
        super(StableFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.thresholds = thresholds
        self.minority_mask = minority_mask  # 新增少数类标识
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        if self.thresholds is not None:
            logit_adjust = torch.log(torch.tensor([t / (1 - t) for t in self.thresholds], device=inputs.device))
            inputs = inputs - logit_adjust

        bce_loss = self.bce(inputs, targets)
        if self.minority_mask is not None:
            minority_weights = torch.where(self.minority_mask, 2.0, 1.0)
            bce_loss = bce_loss * minority_weights
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

def compute_pr_thresholds(probs, labels, disease_cols):
    """基于 Precision-Recall 曲线计算动态阈值"""
    thresholds = []
    for i in range(len(disease_cols)):
        precision, recall, thresh = precision_recall_curve(labels[:, i], probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        optimal_idx = np.argmax(f1_scores)
        thresholds.append(thresh[optimal_idx] if optimal_idx < len(thresh) else 0.5)
    return thresholds

@profile
def train():
    """训练主函数"""
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

    # 数据集准备
    train_excel_path, test_excel_path = split_dataset(
        excel_path=EXCEL_PATH, test_size=0.2, random_state=42, disease_cols=disease_cols)

    # 数据增强
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

    # 数据集加载
    train_dataset = FundusDataset(excel_path=train_excel_path, img_root=IMG_ROOT,
                                  disease_cols=disease_cols, transform=train_transform)
    test_dataset = FundusDataset(excel_path=test_excel_path, img_root=IMG_ROOT,
                                 disease_cols=disease_cols, transform=test_transform)

    # 样本权重计算
    df = pd.read_excel(train_excel_path)
    labels = torch.tensor(df[disease_cols].values, dtype=torch.float32)
    class_counts = labels.sum(dim=0)
    class_weights = torch.log(1.0 / (class_counts + 1e-6))
    class_weights = class_weights / class_weights.max()
    sample_weights = torch.zeros(len(train_dataset))
    for idx in range(len(train_dataset)):
        label = labels[idx]
        weight = (label * class_weights).sum()
        sample_weights[idx] = weight * 2
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 数据加载器
    num_workers = 8  # 减少 worker 数量，降低内存压力
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                  collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                 collate_fn=custom_collate)

    # 知识图谱构建
    kg = MedicalKG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="120190333",
        local_dir="/data/eye/pycharm_project_257/kg_data"
    )
    kg.build_kg(df, disease_cols, batch_size=1000)
    # kg._save_local_data()

    # 直接生成 kg_embeddings 和 A_norm
    kg_embeddings = kg.generate_disease_embeddings().to(device)
    A = kg.get_adjacency_matrix().to(device)
    A = A.to_sparse()
    eye_matrix = torch.eye(A.size(0), device=device).to_sparse()
    A = eye_matrix + A
    D_values = 1 / torch.sqrt(A.to_dense().sum(dim=1) + 1e-6)
    D = torch.diag(D_values).to_sparse()
    A_norm = D @ A @ D

    # 模型初始化，直接传入 kg_embeddings 和 A_norm
    model = MultiModalNet(disease_cols=disease_cols, kg_embeddings=kg_embeddings, adjacency_matrix=A_norm).to(device)
    model = model.float()
    model.initialize_kg_logits()  # 确保此方法存在

    # 损失函数和优化器
    alpha_weights = compute_alpha_weights(df, disease_cols, device)
    minority_classes = ['H', 'A', 'G', 'C', 'M']
    minority_mask = torch.tensor([d in minority_classes for d in disease_cols], dtype=torch.bool, device=device)
    criterion_cls = StableFocalLoss(gamma=3.0, alpha=alpha_weights, minority_mask=minority_mask)

    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('feature_extractor')], 'lr': 3e-4}
    ], weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[1e-4, 2e-4],
        total_steps=EPOCHS * len(train_dataloader),
        pct_start=0.5,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=10.0,
        final_div_factor=1e3
    )

    # 训练循环
    best_score = 0.0
    patience = 10
    epochs_no_improve = 0
    learning_rates = []
    train_thresholds = torch.full((len(disease_cols),), 0.5, device=device)

    for epoch in range(EPOCHS):
        all_train_probs = []
        all_train_labels = []

        model.train()
        epoch_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}", position=0, leave=True)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None or len(batch['paired_image']) == 0:
                epoch_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device, dtype=torch.float32)
            meta = batch['meta'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.float32)

            logits, global_feat_weighted, kg_logits, _, _ = model(paired_img, None, meta, use_text=False, batch_idx=batch_idx)

            with torch.no_grad():
                train_probs = torch.sigmoid(logits.detach()).cpu().numpy()
                all_train_probs.append(train_probs)
                all_train_labels.append(labels.cpu().numpy())

            loss_cls = criterion_cls(logits, labels)
            total_loss = loss_cls

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None  # 确保梯度清空

            scheduler.step()

            running_loss += total_loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # 内存清理
            del paired_img, meta, labels, logits, global_feat_weighted, kg_logits, loss_cls, total_loss, train_probs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
            epoch_bar.set_postfix({'loss': f"{running_loss / (batch_idx + 1):.4f}", 'lr': f"{current_lr:.2e}", 'gpu_mem': f"{gpu_mem:.2f} MB"})
            epoch_bar.update(1)

        epoch_bar.close()

        # 计算动态阈值
        all_train_probs = np.concatenate(all_train_probs)
        all_train_labels = np.concatenate(all_train_labels)
        train_thresholds = compute_pr_thresholds(all_train_probs, all_train_labels, disease_cols)
        criterion_cls.thresholds = train_thresholds
        del all_train_probs, all_train_labels
        torch.cuda.empty_cache()
        gc.collect()

        tqdm.write("\n训练集动态阈值：")
        for disease, thresh in zip(disease_cols, train_thresholds):
            tqdm.write(f"{disease}: {thresh:.4f}")

        # 评估
        model.eval()
        with torch.no_grad():
            val_results = evaluate(model, test_dataloader, device, disease_cols)
            full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1, micro_precision, micro_recall, minority_f1 = val_results
            tqdm.write(f"Epoch {epoch + 1} 评估完成，Micro F1: {micro_f1:.4f}")

        model.train()

        # 早停
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
                'scheduler_state': scheduler.state_dict(),
                'thresholds': train_thresholds
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

        train_dataset.clear_cache()
        test_dataset.clear_cache()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    # 保存模型
    save_path = "/data/eye/pycharm_project_257/models/multimodal_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if train_thresholds is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'kg_embeddings': model.kg_embeddings,
            'disease_cols': disease_cols,
            'scheduler_state': scheduler.state_dict(),
            'thresholds': train_thresholds
        }, save_path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'kg_embeddings': model.kg_embeddings,
            'disease_cols': disease_cols,
            'scheduler_state': scheduler.state_dict()
        }, save_path)
    tqdm.write(f"模型已保存到 {save_path}")

    # 学习率曲线
    plot_learning_rate_curve(learning_rates,
                             "/data/eye/pycharm_project_257/plots/learning_rate_curve.png")

    # 资源清理
    model.clear_resources()
    kg.clear_cache()
    train_dataset.clear_cache()
    test_dataset.clear_cache()
    train_dataloader._shutdown_workers()  # 显式关闭 worker 进程
    test_dataloader._shutdown_workers()
    del model, optimizer, scheduler, train_dataset, train_dataloader, test_dataset, test_dataloader, kg, df
    if 'kg_embeddings' in locals():
        del kg_embeddings
    if 'A' in locals():
        del A
    if 'A_norm' in locals():
        del A_norm
    if 'eye_matrix' in locals():
        del eye_matrix
    if 'D' in locals():
        del D
    if 'D_values' in locals():
        del D_values
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    final_memory = get_memory_usage()
    tqdm.write(f"最终内存使用量: {final_memory:.2f} MB")
    tqdm.write(f"内存变化: {final_memory - initial_memory:.2f} MB")

if __name__ == "__main__":
    train()