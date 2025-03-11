import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from utils.data_loader import FundusDataset
from models.multimodal_model import MultiModalNet
from models.kg_builder import MedicalKG
from torch.cuda.amp import autocast, GradScaler
import math
from torch.optim.lr_scheduler import LRScheduler
from utils.split_dataset import split_dataset

# 配置参数
EXCEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\labels.xlsx"
IMG_ROOT = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\images"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
ACCUMULATION_STEPS = 2
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
    # EMA类保持不变
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
    # evaluate函数保持不变
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            left_img = batch['left_image'].to(device)
            right_img = batch['right_image'].to(device)
            text = batch['keywords']
            meta = {k: v.to(device) for k, v in batch['meta'].items()}
            labels = batch['labels'].to(device)
            seg_target = None
            with autocast():
                logits, _, _ = model(left_img, right_img, text, meta, seg_target)
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds == labels).all(dim=1).sum().item()
                total += labels.size(0)
    return correct / total

def train():
    # 数据集分割
    train_excel_path, test_excel_path = split_dataset(EXCEL_PATH, test_size=0.2, random_state=42,
                                                      disease_cols=disease_cols)
    train_dataset = FundusDataset(excel_path=train_excel_path, img_root=IMG_ROOT, disease_cols=disease_cols,
                                  phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_dataset = FundusDataset(excel_path=test_excel_path, img_root=IMG_ROOT, disease_cols=disease_cols, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 初始化模型和相关组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_excel(train_excel_path)
    kg = MedicalKG()
    kg.build_kg(df, disease_cols)
    kg_embeddings = kg.generate_disease_embeddings().to(device)
    print(f"kg_embeddings shape: {kg_embeddings.shape}")
    A = kg.get_adjacency_matrix().to(device)
    print(f"A shape: {A.shape}")

    # ========== 新增邻接矩阵校验 ==========
    print(f"\n邻接矩阵有效性检查:")
    print(f"1. 维度验证: {A.dim()}D张量 (应为2D矩阵)")
    print(f"2. 形状验证: {A.shape} (应与疾病数量{len(disease_cols)}一致)")
    print(f"3. 非零元素数量: {A.sum().item()} (至少应有1个关联关系)")
    print(f"4. 数据类型: {A.dtype}\n")

    assert A.dim() == 2, "邻接矩阵必须是二维张量"
    assert A.shape[0] == A.shape[1] == len(disease_cols), "邻接矩阵形状与疾病数量不匹配"
    assert A.sum() > 0, "邻接矩阵不能全为0，请检查知识图谱构建"

    # 可选：检查矩阵对称性（如果是无向图）
    if not torch.allclose(A, A.T):
        print("[警告] 邻接矩阵不对称，当前按有向图处理")

    model = MultiModalNet(disease_cols, kg_embeddings).to(device)
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS)
    scaler = GradScaler()
    ema = EMA(model, decay=0.999)
    ema.register()

    torch.cuda.empty_cache()
    model.train()

    # train.py (关键修改部分)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # 初始化梯度

        for i, batch in enumerate(train_dataloader):
            try:
                left_img = batch['left_image'].to(device, non_blocking=True)
                right_img = batch['right_image'].to(device, non_blocking=True)
                text = batch['keywords']
                meta = {k: v.to(device, non_blocking=True) for k, v in batch['meta'].items()}
                labels = batch['labels'].to(device, non_blocking=True)

                # ========== 新增变量初始化 ==========
                loss = torch.tensor(0.0, device=device)  # 确保loss变量存在
                loss_cls = torch.tensor(0.0, device=device)
                loss_graph = torch.tensor(0.0, device=device)

                with autocast():
                    logits, seg_output, decoupled_features = model(left_img, right_img, text, meta, None)
                    loss_cls = criterion_cls(logits, labels)
                    # ========== 修改图正则化计算 ==========
                    loss_graph = torch.tensor(0.0, device=device)
                    num_diseases = len(disease_cols)
                    if num_diseases > 1 and A.sum() > 0:  # 增加有效性检查
                        for m in range(num_diseases):
                            for n in range(m + 1, num_diseases):
                                if m < logits.size(1) and n < logits.size(1):  # 维度检查
                                    if A[m, n]:
                                        diff = (logits[:, m] - logits[:, n]).pow(2).mean()
                                        loss_graph += diff
                        loss_graph /= A.sum()  # 仅在有效时计算

                    loss = loss_cls + 0.01 * loss_graph  # 确保loss被赋值

                # ========== 梯度累积逻辑优化 ==========
                accumulation_progress = (i % ACCUMULATION_STEPS) + 1
                is_last_batch = (i + 1) == len(train_dataloader)
                # 反向传播（修正retain_graph条件）
                scaler.scale(loss / ACCUMULATION_STEPS).backward(
                    retain_graph=not (accumulation_progress == ACCUMULATION_STEPS or is_last_batch)
                )

                # 参数更新（增加同步和清理）
                if accumulation_progress == ACCUMULATION_STEPS or is_last_batch:
                    torch.cuda.synchronize()  # 确保所有操作完成
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update()

                    # 显式释放资源
                    del logits, loss, seg_output, decoupled_features
                    torch.cuda.empty_cache()

                running_loss += loss.item()

            except Exception as e:  # 修改为捕获所有异常

                print(f"批次 {i} 出错: {e}")

                optimizer.zero_grad(set_to_none=True)

                torch.cuda.empty_cache()

                # 新增错误处理逻辑

                if 'loss' in locals():
                    del loss

                continue

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

        # 每隔5个epoch打印诊断路径
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                with autocast():
                    sample_logits, _, _ = model(left_img, right_img, text, meta, None)
                    diag_path = model.generate_diagnostic_path(sample_logits)
                    print("诊断路径树:", diag_path)

        torch.cuda.empty_cache()

    # 应用EMA并评估
    ema.apply_shadow()
    test_accuracy = evaluate(model, test_dataloader, device)
    print(f"测试集准确度: {test_accuracy:.4f}")
    ema.restore()

    # 保存模型
    project_root = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing"
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "multimodal_model.pth")
    ema.apply_shadow()
    torch.save(model.state_dict(), save_path)
    ema.restore()
    print(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"训练过程中发生错误: {e}")