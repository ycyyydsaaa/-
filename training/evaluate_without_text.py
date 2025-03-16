import os
import torch
from torch.utils.data import DataLoader
from utils.data_loader import FundusDataset
from models.multimodal_model import MultiModalNet
from models.kg_builder import MedicalKG
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2

# 配置参数
TEST_EXCEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\labels_test.xlsx"
IMG_ROOT = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\data\Training_Dataset\paired_dir"
MODEL_PATH = r"D:\Toolbox App\PyCharm Professional\jbr\bin\D\寒风冷雨不知眠\PycharmProjects\eye_image_processing\models\multimodal_model.pth"
disease_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
BATCH_SIZE = 2


# 自定义数据整理函数
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# 验证 Neo4j 图数据完整性（修复版本）
def validate_neo4j_graph(kg, disease_cols):
    """
    检查 Neo4j 中的知识图谱是否完整，与预期疾病类别一致。

    Args:
        kg: MedicalKG 实例
        disease_cols: 预期疾病类别列表

    Raises:
        ValueError: 如果图数据不完整或不一致
    """
    # 修复：正确提取疾病名称集合
    all_diseases = {record['d.name'] for record in kg.graph.run("MATCH (d:Disease) RETURN d.name").data()}
    expected_diseases = set(disease_cols)

    if all_diseases != expected_diseases:
        raise ValueError(f"Neo4j 中的疾病节点 {all_diseases} 与预期 {expected_diseases} 不一致")

    for d in disease_cols:
        symptoms = kg.query_disease_symptoms(d)
        if not symptoms:
            raise ValueError(f"疾病 {d} 在 Neo4j 中没有关联症状，可能图数据不完整")

    tqdm.write("Neo4j 图数据验证通过")


# 评估函数，不使用文本特征
def evaluate_without_text(model, dataloader, device, disease_cols):
    model.eval()
    correct = 0
    total = 0
    correct_per_disease = torch.zeros(len(disease_cols), device=device)
    all_preds = []
    all_labels = []

    eval_bar = tqdm(total=len(dataloader), desc="Evaluation (No Text)", position=0, leave=True, dynamic_ncols=True)
    try:
        target_layer = model.fpn.output_convs[-1]
        grad_cam = GradCAM(model=model, target_layers=[target_layer])
    except Exception as e:
        tqdm.write(f"GradCAM 初始化失败: {str(e)[:500]}，将跳过热力图生成")
        grad_cam = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                eval_bar.update(1)
                continue

            paired_img = batch['paired_image'].to(device)
            meta = batch['meta'].to(device)
            labels = batch['labels'].to(device)

            logits, _, kg_logits, _, _ = model(paired_img, None, meta, None)
            preds = torch.sigmoid(logits) > 0.5

            # 强制 'N' 排他性
            mask_n1 = preds[:, 0] == 1
            preds[mask_n1, 1:] = 0
            mask_no_disease = (preds[:, 1:].sum(dim=1) == 0) & (preds[:, 0] == 0)
            preds[mask_no_disease, 0] = 1
            if batch_idx == 0 and grad_cam is not None:
                try:
                    grad_input = paired_img[0:1].clone().detach().requires_grad_(True)
                    with torch.enable_grad():
                        grad_logits, _, _, feature_maps, _ = model(grad_input, None, meta[0:1], None)
                        if labels[0][0] == 1:
                            if labels[0][1:].sum() > 0:
                                tqdm.write(f"警告: Batch 0 标记为正常（N=1），但其他疾病标签非零: {labels[0].tolist()}")
                            tqdm.write("Batch 0 为正常（N=1），跳过 GradCAM 热力图生成")
                        else:
                            positive_indices = torch.where(labels[0] == 1)[0]
                            if len(positive_indices) > 0:
                                img = paired_img[0].cpu().permute(1, 2, 0).numpy()
                                mean = np.array([0.485, 0.456, 0.406])
                                std = np.array([0.229, 0.224, 0.225])
                                img = (img * std + mean).clip(0, 1)
                                for idx in positive_indices:
                                    target_category = idx.item()
                                    disease_name = disease_cols[target_category]
                                    targets = [ClassifierOutputTarget(target_category)]
                                    grayscale_cam = grad_cam(input_tensor=grad_input, targets=targets)
                                    visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)
                                    cv2.imwrite(f"gradcam_no_text_batch_{batch_idx}_disease_{disease_name}.png",
                                                visualization * 255)
                                    tqdm.write(
                                        f"成功生成热力图: gradcam_no_text_batch_{batch_idx}_disease_{disease_name}.png")
                            else:
                                tqdm.write("Batch 0 无疾病标签，跳过 GradCAM 热力图生成")
                except Exception as e:
                    tqdm.write(f"Batch 0 GradCAM 生成失败: {str(e)[:500]}，继续评估")

            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            correct_per_disease += (preds == labels).float().sum(dim=0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            eval_bar.update(1)

    eval_bar.close()
    if grad_cam is not None:
        del grad_cam
    torch.cuda.empty_cache()

    if total == 0:
        tqdm.write("警告: 测试数据集为空，返回零值指标")
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
    tqdm.write("\n测试集评估结果（无文本特征）：")
    tqdm.write(f"全匹配准确度: {full_match_accuracy:.4f}")
    tqdm.write(f"Micro F1 分数: {micro_f1:.4f}")
    tqdm.write(f"Macro F1 分数: {macro_f1:.4f}")
    tqdm.write("\n逐疾病评估指标：")
    tqdm.write(tabulate(table_data, headers=headers, tablefmt="grid"))

    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(disease_cols))
        width = 0.2
        ax.bar([i - 1.5 * width for i in x], accuracy_per_disease.cpu().numpy(), width, label='Accuracy',
               color='skyblue')
        ax.bar([i - 0.5 * width for i in x], precision, width, label='Precision', color='lightgreen')
        ax.bar([i + 0.5 * width for i in x], recall, width, label='Recall', color='salmon')
        ax.bar([i + 1.5 * width for i in x], f1, width, label='F1 Score', color='gold')
        ax.set_xticks(x)
        ax.set_xticklabels(disease_cols)
        ax.set_title("Per-Disease Evaluation Metrics (No Text Feature)")
        ax.set_xlabel("Disease")
        ax.set_ylabel("Score")
        ax.legend()
        plt.savefig("evaluation_metrics_no_text.png")
        plt.close()
        tqdm.write("评估指标柱状图已保存: evaluation_metrics_no_text.png")
    except Exception as e:
        tqdm.write(f"绘制柱状图失败: {str(e)[:500]}，评估结果已生成")

    return full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"使用设备: {device}")

    try:
        test_dataset = FundusDataset(excel_path=TEST_EXCEL_PATH, img_root=IMG_ROOT, disease_cols=disease_cols,
                                     phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=4, pin_memory=True, collate_fn=custom_collate)
        tqdm.write(f"测试数据集大小: {len(test_dataset)}")
    except Exception as e:
        raise RuntimeError(f"测试数据集加载失败: {str(e)[:500]}")

    try:
        kg = MedicalKG(uri="bolt://localhost:7687", user="neo4j", password="120190333")
        kg.disease_cols = disease_cols
        validate_neo4j_graph(kg, disease_cols)
        kg_embeddings = kg.generate_disease_embeddings().to(device)
        tqdm.write("从 Neo4j 加载知识图谱嵌入完成")
    except Exception as e:
        raise RuntimeError(f"知识图谱嵌入加载失败: {str(e)[:500]}")

    try:
        model = MultiModalNet(disease_cols, kg_embeddings).to(device)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        tqdm.write(f"模型已从 {MODEL_PATH} 加载")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)[:500]}")

    full_match_accuracy, accuracy_per_disease, micro_f1, macro_f1 = evaluate_without_text(
        model, test_dataloader, device, disease_cols
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        tqdm.write(f"评估过程中发生错误: {str(e)[:500]}")
        traceback.print_exc()