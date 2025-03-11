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

# 配置参数
EXCEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\labels.xlsx"
IMG_ROOT = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\paired_dir"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
WARMUP_EPOCHS = 5

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

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            paired_img = batch['paired_image'].to(device)
            text = batch['keywords']
            meta = batch['meta'].to(device)
            labels = batch['labels'].to(device)
            logits, _, _ = model(paired_img, text, meta, None)
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            print(f"Evaluate - Batch {batch_idx}, Correct: {correct}, Total: {total}")
    accuracy = correct / total
    print(f"测试集评估完成，准确度: {accuracy:.4f}")
    return accuracy

def train():
    # 数据集分割
    train_excel_path, test_excel_path = split_dataset(EXCEL_PATH, test_size=0.2, random_state=42,
                                                      disease_cols=disease_cols)
    train_dataset = FundusDataset(excel_path=train_excel_path, img_root=IMG_ROOT,
                                  disease_cols=disease_cols, phase='train')
    test_dataset = FundusDataset(excel_path=test_excel_path, img_root=IMG_ROOT,
                                 disease_cols=disease_cols, phase='test')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"测试数据集大小: {len(test_dataset)}")

    # 初始化知识图谱和邻接矩阵
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_excel(train_excel_path)
    kg = MedicalKG()
    kg.build_kg(df, disease_cols)
    kg_embeddings = kg.generate_disease_embeddings().to(device)
    A = kg.get_adjacency_matrix().to(device)
    assert A.dim() == 2, "邻接矩阵必须是二维张量"
    assert A.shape[0] == A.shape[1] == len(disease_cols), f"邻接矩阵形状 {A.shape} 与疾病数量 {len(disease_cols)} 不匹配"
    print(f"邻接矩阵形状: {A.shape}, 非零元素数量: {A.sum().item()}")

    model = MultiModalNet(disease_cols, kg_embeddings).to(device)

    # 模型预验证
    with torch.no_grad():
        dummy_batch = next(iter(train_dataloader))
        dummy_logits, _, _ = model(
            dummy_batch['paired_image'].to(device),
            dummy_batch['keywords'],
            dummy_batch['meta'].to(device),
            None
        )
        print(f"模型预验证 - Logits shape: {dummy_logits.shape}")
        assert dummy_logits.size(1) == len(disease_cols), "模型输出维度与疾病数量不匹配"

    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS)
    ema = EMA(model, decay=0.999)
    ema.register()

    torch.cuda.empty_cache()
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_dataloader):
            try:
                paired_img = batch['paired_image'].to(device, non_blocking=True)
                text = batch['keywords']
                meta = batch['meta'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                # 打印批次信息
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_dataloader)}]")
                print(f"  Paired_img shape: {paired_img.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Meta shape: {meta.shape}")
                print(f"  GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                logits, seg_output, _ = model(paired_img, text, meta, None)
                loss_cls = criterion_cls(logits, labels)

                # 图正则化损失
                loss_graph = torch.tensor(0.0, device=device)
                valid_pairs = [(m, n) for m in range(len(disease_cols))
                               for n in range(m + 1, len(disease_cols))
                               if A[m, n]]
                if valid_pairs:
                    loss_graph = sum((logits[:, m] - logits[:, n]).pow(2).mean() for m, n in valid_pairs)
                    loss_graph = loss_graph * 0.01 / len(valid_pairs)

                loss = loss_cls + loss_graph

                if not torch.isfinite(loss):
                    print(f"  Loss 非有限: {loss.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update()

                running_loss += loss.item()
                print(f"  Batch Loss: {loss.item():.4f}, Classification Loss: {loss_cls.item():.4f}, Graph Loss: {loss_graph.item():.4f}")

            except Exception as e:
                print(f"  批次 {i} 出错: {str(e)[:200]}")
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] 完成, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
        torch.cuda.empty_cache()

    # 测试评估
    print("开始测试集评估...")
    ema.apply_shadow()
    test_accuracy = evaluate(model, test_dataloader, device)
    ema.restore()

    # 保存模型
    project_root = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing"
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "multimodal_model.pth")
    ema.apply_shadow()  # 使用 EMA 的影子参数保存模型
    torch.save(model.state_dict(), save_path)
    ema.restore()
    print(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)[:500]}")