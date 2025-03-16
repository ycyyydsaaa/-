import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from utils.data_loader import FundusDataset
from models.multimodal_model import MultiModalNet
from models.kg_builder import MedicalKG
import math
from torch.optim.lr_scheduler import LRScheduler
from utils.split_dataset import split_dataset
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2
import logging
import sys

# 配置日志级别
logging.basicConfig(level=logging.WARNING)

# 配置参数
EXCEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\labels.xlsx"
IMG_ROOT = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\paired_dir"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']  # N 表示正常
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
WARMUP_EPOCHS = 5

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

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

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def evaluate(model, dataloader, device, disease_cols):
    model.eval()
    correct = 0
    total = 0
    correct_per_disease = torch.zeros(len(disease_cols), device=device)
    all_preds = []
    all_labels = []

    eval_bar = tqdm(total=len(dataloader), desc="Evaluation", position=1, leave=False, dynamic_ncols=True)
    target_layer = model.fpn.output_convs[-1]
    grad_cam = GradCAM(model=model, target_layers=[target_layer])

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                eval_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device)
            text_feature = batch['text_feature'].to(device)
            meta = batch['meta'].to(device)
            labels = batch['labels'].to(device)

            logits, _, kg_logits, _, _ = model(paired_img, text_feature, meta, None)
            preds = torch.sigmoid(logits) > 0.5

            if batch_idx == 0:
                grad_input = paired_img[0:1].clone().detach().requires_grad_(True)
                with torch.enable_grad():
                    grad_logits, _, _, feature_maps, _ = model(grad_input, text_feature[0:1], meta[0:1], None)
                    if labels[0][0] == 1:
                        if labels[0][1:].sum() > 0:
                            tqdm.write(f"警告: Batch 0 标记为正常（N=1），但其他疾病标签非零: {labels[0].tolist()}")
                        tqdm.write("Batch 0 为正常（N=1），未生成 GradCAM 热力图")
                    else:
                        positive_indices = torch.where(labels[0] == 1)[0]
                        if len(positive_indices) > 0:
                            img = paired_img[0].cpu().permute(1, 2, 0).numpy()
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            img = (img * std + mean).clip(0, 1)
                            for idx in positive_indices:
                                try:  # ========== 异常捕获开始 ==========
                                    target_category = idx.item()
                                    disease_name = disease_cols[target_category]
                                    # 生成GradCAM热力图
                                    targets = [ClassifierOutputTarget(target_category)]
                                    grayscale_cam = grad_cam(input_tensor=grad_input, targets=targets)
                                    visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)
                                    cv2.imwrite(f"gradcam_batch_{batch_idx}_disease_{disease_name}.png", visualization * 255)
                                except Exception as e:
                                    # 打印错误信息但继续执行
                                    tqdm.write(f"生成疾病 {disease_name} 热力图失败: {str(e)}")
                                    import traceback
                                    traceback.print_exc()  # 打印详细堆栈信息（调试时使用）
                                # ========== 异常捕获结束 ==========
                        else:
                            tqdm.write("Batch 0 无疾病标签，未生成 GradCAM 热力图")

            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            correct_per_disease += (preds == labels).float().sum(dim=0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            eval_bar.update(1)

    eval_bar.close()

    del grad_cam
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

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(disease_cols))
    width = 0.2
    ax.bar([i - 1.5 * width for i in x], accuracy_per_disease.cpu().numpy(), width, label='Accuracy', color='skyblue')
    ax.bar([i - 0.5 * width for i in x], precision, width, label='Precision', color='lightgreen')
    ax.bar([i + 0.5 * width for i in x], recall, width, label='Recall', color='salmon')
    ax.bar([i + 1.5 * width for i in x], f1, width, label='F1 Score', color='gold')
    ax.set_xticks(x)
    ax.set_xticklabels(disease_cols)
    ax.set_title("Per-Disease Evaluation Metrics")
    ax.set_xlabel("Disease")
    ax.set_ylabel("Score")
    ax.legend()
    plt.savefig("evaluation_metrics.png")
    plt.close()

    return full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1

def train():
    train_excel_path, test_excel_path = split_dataset(EXCEL_PATH, test_size=0.2, random_state=42, disease_cols=disease_cols)
    train_dataset = FundusDataset(excel_path=train_excel_path, img_root=IMG_ROOT, disease_cols=disease_cols, phase='train')
    test_dataset = FundusDataset(excel_path=test_excel_path, img_root=IMG_ROOT, disease_cols=disease_cols, phase='test')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True, collate_fn=custom_collate)

    tqdm.write(f"训练数据集大小: {len(train_dataset)}")
    tqdm.write(f"测试数据集大小: {len(test_dataset)}")

    tqdm.write("检查训练数据中的 'N' 排他性...")
    for batch in train_dataloader:
        if batch is not None:
            labels = batch['labels']
            if labels[:, 0].sum() > 0 and labels[labels[:, 0] == 1, 1:].sum() > 0:
                tqdm.write(f"数据警告: 'N' = 1 时其他标签非零: {labels[labels[:, 0] == 1].tolist()}")
            break
    tqdm.write("检查完成")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_excel(train_excel_path)
    missing_cols = [col for col in disease_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"训练数据 DataFrame 缺少以下列: {missing_cols}")

    kg = MedicalKG(uri="bolt://localhost:7687", user="neo4j", password="120190333")
    kg.build_kg(df, disease_cols, frequency_threshold_ratio=0.6)
    if kg.disease_cols is None:
        raise ValueError("知识图谱构建失败，disease_cols 未初始化")

    kg_embeddings = kg.generate_disease_embeddings().to(device)
    A = kg.get_adjacency_matrix().to(device)
    assert A.dim() == 2 and A.shape[0] == A.shape[1] == len(disease_cols), f"邻接矩阵形状 {A.shape} 与疾病数量 {len(disease_cols)} 不匹配"

    model = MultiModalNet(disease_cols, kg_embeddings).to(device)
    pseudo_labels = kg.generate_pseudo_labels(df, train_dataloader, model, device=device)
    pseudo_dataset = FundusDataset(excel_path=train_excel_path, img_root=IMG_ROOT, disease_cols=disease_cols, phase='train')
    pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=4, pin_memory=True, collate_fn=custom_collate)

    model_pseudo = MultiModalNet(disease_cols, kg_embeddings).to(device)
    pos_weights = torch.tensor([min(1 / (1 - df[d].mean()), 50.0) for d in disease_cols], device=device)  # 调整正类权重
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer_pseudo = optim.Adam(model_pseudo.parameters(), lr=LR)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS)
    scheduler_pseudo = WarmupCosineLR(optimizer_pseudo, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS)
    ema = EMA(model, decay=0.999)
    ema.register()
    scaler = GradScaler()
    torch.cuda.empty_cache()

    model.train()
    model_pseudo.train()
    for epoch in range(EPOCHS):
        epoch_bar = tqdm(total=len(train_dataloader) + len(pseudo_dataloader),
                         desc=f"Epoch {epoch + 1}/{EPOCHS}",
                         position=0,
                         leave=True,
                         dynamic_ncols=True)
        running_loss = 0.0
        running_cls_loss = 0.0
        running_graph_loss = 0.0
        running_penalty = 0.0
        running_text_weight_loss = 0.0
        valid_batches_main = 0
        valid_batches_pseudo = 0

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                epoch_bar.update(1)
                continue

            try:
                optimizer.zero_grad(set_to_none=True)
                paired_img = batch['paired_image'].to(device)
                text_feature = batch['text_feature'].to(device)
                meta = batch['meta'].to(device)
                labels = batch['labels'].to(device).float()

                # 随机丢弃文本特征（30% 概率）
                if torch.rand(1).item() < 0.5:
                    text_feature = torch.zeros_like(text_feature)

                with autocast():
                    logits, seg_output, kg_logits, _, _ = model(paired_img, text_feature, meta, None)
                    if logits is None or kg_logits is None:
                        raise ValueError(f"Model output is None: logits={logits}, kg_logits={kg_logits}")
                    loss_cls = criterion_cls(logits, labels)
                    penalty = torch.tensor(0.0, device=device)
                    if labels[:, 0].sum() > 0:
                        normal_samples = labels[:, 0] == 1
                        penalty = torch.mean(torch.sigmoid(logits[normal_samples, 1:]) ** 2)
                        loss_cls += 0.1 * penalty

                    loss_graph = torch.tensor(0.0, device=device)
                    valid_pairs = [(m, n) for m in range(len(disease_cols)) for n in range(m + 1, len(disease_cols)) if A[m, n]]
                    if valid_pairs:
                        m_indices, n_indices = zip(*valid_pairs)
                        m_indices, n_indices = torch.tensor(m_indices, device=device), torch.tensor(n_indices, device=device)
                        diff_logits = logits[:, m_indices] - logits[:, n_indices]
                        diff_kg = kg_logits[:, m_indices] - kg_logits[:, n_indices]
                        loss_graph = ((diff_logits - diff_kg).pow(2).mean(dim=0)).sum() * 0.01 / len(valid_pairs)

                    # 增强正则化：增大 text_weight 的惩罚
                    text_weight_penalty = 0.8* torch.abs(model.text_weight)  # L1 正则化，系数从 0.01 增至 0.1
                    total_loss = loss_cls + loss_graph + text_weight_penalty

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update()

                running_loss += total_loss.item()
                running_cls_loss += loss_cls.item()
                running_graph_loss += loss_graph.item()
                running_penalty += penalty.item() if penalty > 0 else 0.0
                running_text_weight_loss += text_weight_penalty.item()

                epoch_bar.set_postfix({
                    "Batch Loss": f"{total_loss.item():.4f}",
                    "Cls Loss": f"{loss_cls.item():.4f}",
                    "Graph Loss": f"{loss_graph.item():.4f}",
                    "Penalty": f"{penalty.item():.4f}",
                    "Text Weight Loss": f"{text_weight_penalty.item():.4f}",
                    "Text Weight": f"{model.text_weight.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.6f}"
                })
                epoch_bar.update(1)

            except Exception as e:
                tqdm.write(f"Error in main loop at epoch {epoch}, batch {batch_idx}: {str(e)}")
                epoch_bar.update(1)
                raise

        for batch_idx, batch in enumerate(pseudo_dataloader):
            if batch is None:
                epoch_bar.update(1)
                continue

            try:
                optimizer_pseudo.zero_grad(set_to_none=True)
                paired_img = batch['paired_image'].to(device)
                text_feature = batch['text_feature'].to(device)
                meta = batch['meta'].to(device)
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(pseudo_labels))
                labels = pseudo_labels[start_idx:end_idx].to(device)

                # 随机丢弃文本特征（30% 概率）
                if torch.rand(1).item() < 0.3:
                    text_feature = torch.zeros_like(text_feature)

                with autocast():
                    logits, seg_output, kg_logits, _, _ = model_pseudo(paired_img, text_feature, meta, None)
                    if logits is None or kg_logits is None:
                        raise ValueError(f"Model output is None: logits={logits}, kg_logits={kg_logits}")
                    loss_cls = criterion_cls(logits, labels) * 0.5
                    penalty = torch.tensor(0.0, device=device)
                    if labels[:, 0].sum() > 0:
                        normal_samples = labels[:, 0] == 1
                        penalty = torch.mean(torch.sigmoid(logits[normal_samples, 1:]) ** 2)
                        loss_cls += 0.1 * penalty

                    # 对伪标签模型也应用正则化
                    text_weight_penalty = 0.1 * torch.abs(model_pseudo.text_weight)
                    total_loss = loss_cls + text_weight_penalty

                scaler.scale(total_loss).backward()
                scaler.step(optimizer_pseudo)
                scaler.update()

                running_loss += total_loss.item()
                running_cls_loss += loss_cls.item()
                running_penalty += penalty.item() if penalty > 0 else 0.0
                running_text_weight_loss += text_weight_penalty.item()
                valid_batches_pseudo += 1
                epoch_bar.update(1)

            except Exception as e:
                tqdm.write(f"Error in pseudo loop at epoch {epoch}, batch {batch_idx}: {str(e)}")
                epoch_bar.update(1)
                raise

        epoch_bar.close()
        total_valid_batches = valid_batches_main + valid_batches_pseudo
        if total_valid_batches > 0:
            avg_loss = running_loss / total_valid_batches
            avg_cls_loss = running_cls_loss / total_valid_batches
            avg_graph_loss = running_graph_loss / valid_batches_main if valid_batches_main > 0 else 0.0
            avg_penalty = running_penalty / total_valid_batches
            avg_text_weight_loss = running_text_weight_loss / total_valid_batches
        else:
            avg_loss = avg_cls_loss = avg_graph_loss = avg_penalty = avg_text_weight_loss = 0.0

        tqdm.write(f"Epoch [{epoch + 1}/{EPOCHS}] 完成, Average Loss: {avg_loss:.4f}, "
                   f"Classification Loss: {avg_cls_loss:.4f}, Graph Loss: {avg_graph_loss:.4f}, "
                   f"Penalty: {avg_penalty:.4f}, Text Weight Loss: {avg_text_weight_loss:.4f}, "
                   f"Text Weight: {model.text_weight.item():.4f}")

        scheduler.step()
        scheduler_pseudo.step()
        torch.cuda.empty_cache()

    tqdm.write("开始测试集评估（无文本特征）...")
    ema.apply_shadow()
    full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1 = evaluate(model, test_dataloader, device, disease_cols)
    ema.restore()

    project_root = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing"
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "multimodal_model.pth")
    ema.apply_shadow()
    torch.save(model.state_dict(), save_path)
    ema.restore()
    tqdm.write(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    try:
         train()
    except Exception as e:
        import traceback
        tqdm.write(f"训练过程中发生错误: {str(e)[:500]}")
        traceback.print_exc()