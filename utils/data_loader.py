import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from transformers import BertTokenizer, BertModel
import logging
import gc
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_tensor_info(tensor, name):
    if tensor is not None:
        memory_mb = tensor.element_size() * tensor.nelement() / 1024**2
        ref_count = sys.getrefcount(tensor)
        print(f"Tensor {name}: Memory = {memory_mb:.2f} MB, Ref Count = {ref_count}")
    else:
        print(f"Tensor {name}: None")

class FundusDataset(Dataset):
    def __init__(self, excel_path, img_root, disease_cols, phase='train', transform=None):
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel 文件 {excel_path} 不存在")
        if not os.path.exists(img_root):
            raise FileNotFoundError(f"图像根目录 {img_root} 不存在")

        df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.disease_cols = disease_cols
        self.phase = phase
        self.transform = transform  # 使用传入的 transform

        # 验证和清理数据
        self._validate_and_clean(df)

        # 将 DataFrame 转换为轻量的数据结构（字典列表）
        self.data = df.to_dict('records')
        del df  # 释放 DataFrame
        gc.collect()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_path = "/data/eye/pycharm_project_257/models/bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
        self.text_encoder = None
        self.preprocess_text_features()

    def _validate_and_clean(self, df):
        missing_or_corrupt = []
        for idx, row in df.iterrows():
            img_file = str(row['paired_image']) if 'paired_image' in row and pd.notna(
                row['paired_image']) else f"{row['id']}.png"
            img_path = os.path.join(self.img_root, img_file)
            if not os.path.exists(img_path):
                missing_or_corrupt.append(idx)
            else:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    logger.warning(f"图像 {img_path} 损坏: {str(e)}")
                    missing_or_corrupt.append(idx)

        if missing_or_corrupt:
            logger.warning(f"移除缺失或损坏图像的样本索引: {missing_or_corrupt}")
            df.drop(missing_or_corrupt, inplace=True)
            df.reset_index(drop=True, inplace=True)

        invalid_labels = df[self.disease_cols].apply(lambda x: ~x.isin([0, 1])).any(axis=1)
        if invalid_labels.any():
            invalid_indices = df[invalid_labels].index.tolist()
            logger.warning(f"移除无效标签的样本索引: {invalid_indices}")
            df = df[~invalid_labels].reset_index(drop=True)

        for col in self.disease_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        if 'Patient Age' not in df.columns:
            df['Patient Age'] = 0.0
        else:
            df['Patient Age'] = pd.to_numeric(df['Patient Age'], errors='coerce').fillna(0.0).astype('float32')
        if 'Patient Sex' not in df.columns:
            df['Patient Sex'] = 0.0
        else:
            df['Patient Sex'] = df['Patient Sex'].map({'Male': 1.0, 'Female': 0.0}).fillna(0.0).astype('float32')

    def load_text_encoder(self):
        if self.text_encoder is None:
            bert_path = "/data/eye/pycharm_project_257/models/bert-base-uncased"
            self.text_encoder = BertModel.from_pretrained(bert_path, local_files_only=True).to(self.device).eval()

    def preprocess_text_features(self):
        self.load_text_encoder()
        for idx, row in enumerate(self.data):
            text_feature_path = os.path.join(self.img_root, f"{row['id']}_text.pt")
            if not os.path.exists(text_feature_path):
                left_keywords = row.get('Left-Diagnostic Keywords', '无关键词')
                right_keywords = row.get('Right-Diagnostic Keywords', '无关键词')
                keywords = f"{left_keywords} {right_keywords}"
                inputs = self.tokenizer(keywords, return_tensors='pt', padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.text_encoder(**inputs)
                    text_feature = outputs.last_hidden_state.mean(1).squeeze(0).cpu()
                torch.save(text_feature, text_feature_path)
                logger.info(f"保存文本特征: {text_feature_path}, 形状: {text_feature.shape}")
                del inputs, outputs, text_feature
                torch.cuda.empty_cache()
        self.text_encoder = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()

    def load_image(self, path):
        if not os.path.exists(path):
            logger.error(f"图像文件不存在: {path}")
            return None
        try:
            with Image.open(path) as img:
                return img.convert('RGB')
        except Exception as e:
            logger.error(f"加载图像失败 {path}: {str(e)}")
            return None

    def __getitem__(self, idx):
        row = self.data[idx]
        try:
            img_file = str(row['paired_image']) if 'paired_image' in row and pd.notna(row['paired_image']) else f"{row['id']}.png"
            img_path = os.path.join(self.img_root, img_file)
            img = self.load_image(img_path)

            if img is None:
                logger.error(f"图像加载失败: {img_path} (样本 {idx})")
                return None

            img = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)

            text_feature_path = os.path.join(self.img_root, f"{row['id']}_text.pt")
            text_feature = None
            if os.path.exists(text_feature_path):
                text_feature = torch.load(text_feature_path, map_location='cpu')
                if text_feature.dim() == 1:
                    pass
                elif text_feature.dim() == 2 and text_feature.shape[0] == 1:
                    text_feature = text_feature.squeeze(0)
                else:
                    logger.error(f"样本 {idx} - text_feature 维度异常: {text_feature.shape}")
                    text_feature = torch.zeros(768)
            else:
                text_feature = torch.zeros(768)  # 默认值

            age = float(row['Patient Age'])
            gender = float(row['Patient Sex'])
            meta = torch.tensor([age, gender], dtype=torch.float32)
            labels = torch.tensor([row[col] for col in self.disease_cols], dtype=torch.float32)

            return {
                'paired_image': img,
                'text_feature': text_feature,
                'meta': meta,
                'labels': labels
            }
        except Exception as e:
            logger.error(f"样本 {idx} 出错: {str(e)}")
            return None

    def __len__(self):
        return len(self.data)