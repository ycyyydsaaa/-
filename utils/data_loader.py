import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from transformers import BertTokenizer, BertModel
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FundusDataset(Dataset):
    def __init__(self, excel_path, img_root, disease_cols, phase='train'):
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel 文件 {excel_path} 不存在")
        if not os.path.exists(img_root):
            raise FileNotFoundError(f"图像根目录 {img_root} 不存在")

        self.df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.disease_cols = disease_cols
        self.phase = phase

        self._validate_and_clean()

        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),  # 保持左右眼拼接格式
            transforms.RandomHorizontalFlip() if phase == 'train' else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(30) if phase == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased').eval()

    def _validate_and_clean(self):
        if 'paired_image' not in self.df.columns and 'id' not in self.df.columns:
            raise ValueError("Excel 文件中必须包含 'paired_image' 或 'id' 列.")

        missing_or_corrupt = []
        for idx, row in self.df.iterrows():
            img_file = str(row['paired_image']) if 'paired_image' in row and pd.notna(row['paired_image']) else f"{row['id']}.png"
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
            self.df.drop(missing_or_corrupt, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        invalid_labels = self.df[self.disease_cols].apply(lambda x: ~x.isin([0, 1])).any(axis=1)
        if invalid_labels.any():
            invalid_indices = self.df[invalid_labels].index.tolist()
            logger.warning(f"移除无效标签的样本索引: {invalid_indices}")
            self.df = self.df[~invalid_labels].reset_index(drop=True)

        if 'Patient Age' not in self.df.columns:
            logger.warning("警告: 列 'Patient Age' 不存在，将使用默认值 0.0")
            self.df['Patient Age'] = 0.0
        else:
            self.df['Patient Age'] = pd.to_numeric(self.df['Patient Age'], errors='coerce').fillna(0.0)

        if 'Patient Sex' not in self.df.columns:
            logger.warning("警告: 列 'Patient Sex' 不存在，将使用默认值 0.0")
            self.df['Patient Sex'] = 0.0
        else:
            self.df['Patient Sex'] = self.df['Patient Sex'].map({'Male': 1.0, 'Female': 0.0}).fillna(0.0)

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
        row = self.df.iloc[idx]
        try:
            img_file = str(row['paired_image']) if 'paired_image' in row and pd.notna(row['paired_image']) else f"{row['id']}.png"
            img_path = os.path.join(self.img_root, img_file)
            img = self.load_image(img_path)

            if img is None:
                logger.warning(f"跳过损坏的图像 {img_path} (样本 {idx})")
                return None

            img = self.transform(img)

            left_keywords = row.get('Left-Diagnostic Keywords', '无关键词')
            right_keywords = row.get('Right-Diagnostic Keywords', '无关键词')
            keywords = f"{left_keywords} {right_keywords}"
            inputs = self.tokenizer(keywords, return_tensors='pt', padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
            text_feature = outputs.last_hidden_state.mean(1).squeeze(0)  # [768]

            age = float(row['Patient Age'])
            gender = float(row['Patient Sex'])
            meta = torch.tensor([age, gender], dtype=torch.float32)  # [2]
            labels = torch.tensor(row[self.disease_cols].values.astype(float), dtype=torch.float32)  # [len(disease_cols)]

            return {
                'paired_image': img,  # [3, 256, 512]
                'text_feature': text_feature,  # [768]
                'meta': meta,  # [2]
                'labels': labels  # [len(disease_cols)]
            }
        except Exception as e:
            logger.error(f"样本 {idx} 出错: {str(e)}")
            return None

    def __len__(self):
        return len(self.df)