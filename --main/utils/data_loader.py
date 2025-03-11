import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class FundusDataset(Dataset):
    def __init__(self, excel_path, img_root, disease_cols, phase='train'):
        self.df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.disease_cols = disease_cols
        self.phase = phase
        self._validate_and_clean()

        # 数据增强
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]) if phase == 'train' else None

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _validate_and_clean(self):
        # 检查图像文件名列
        if 'paired_image' not in self.df.columns and 'id' not in self.df.columns:
            raise ValueError("Excel 文件中必须包含 'paired_image' 或 'id' 列。")

        # 检查图像存在性
        missing_images = []
        for idx, row in self.df.iterrows():
            img_file = (
                str(row['paired_image'])
                if 'paired_image' in row and pd.notna(row['paired_image'])
                else f"paired_{row['id']}.png" if 'id' in row else None
            )
            if not img_file or not os.path.exists(os.path.join(self.img_root, img_file)):
                missing_images.append(idx)
        if missing_images:
            print(f"移除缺失图像的样本索引: {missing_images}")
            self.df.drop(missing_images, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        # 检查标签有效性
        invalid_labels = self.df[self.disease_cols].apply(lambda x: ~x.isin([0, 1])).any(axis=1)
        if invalid_labels.any():
            print(f"移除无效标签的样本索引: {self.df[invalid_labels].index.tolist()}")
            self.df = self.df[~invalid_labels].reset_index(drop=True)

        # 检查和处理 "Patient Age" 和 "Patient Sex" 列，修复 FutureWarning
        if 'Patient Age' in self.df.columns:
            self.df['Patient Age'] = pd.to_numeric(self.df['Patient Age'], errors='coerce')
            self.df['Patient Age'] = self.df['Patient Age'].fillna(0.0)  # 修改赋值方式
        else:
            print("警告: 列 'Patient Age' 不存在，将使用默认值 0.0")
            self.df['Patient Age'] = 0.0

        if 'Patient Sex' in self.df.columns:
            self.df['Patient Sex'] = self.df['Patient Sex'].map({'Male': 1.0, 'Female': 0.0}).fillna(0.0)
        else:
            print("警告: 列 'Patient Sex' 不存在，将使用默认值 0.0")
            self.df['Patient Sex'] = 0.0

    def load_image(self, path):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {path}: {str(e)}")
            return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # 获取图像路径
            img_file = (
                str(row['paired_image'])
                if 'paired_image' in row and pd.notna(row['paired_image'])
                else f"paired_{row['id']}.png"
            )
            img_path = os.path.join(self.img_root, img_file)
            img = self.load_image(img_path)
            if img is None:
                raise FileNotFoundError(f"图像 {img_path} 不存在")

            # 数据增强
            if self.phase == 'train' and self.augment:
                img = self.augment(img)

            # 预处理
            img = self.transform(img)

            # 元数据处理
            age = float(row['Patient Age'])
            gender = float(row['Patient Sex'])
            meta = torch.tensor([age, gender], dtype=torch.float32)

            # 标签处理
            labels = torch.tensor(row[self.disease_cols].values.astype(float), dtype=torch.float32)

            # 关键词处理
            left_keywords = row.get('Left-Diagnostic Keywords', '无关键词')
            right_keywords = row.get('Right-Diagnostic Keywords', '无关键词')
            keywords = f"{left_keywords} {right_keywords}"

            return {
                'paired_image': img,
                'keywords': keywords,
                'meta': meta,
                'labels': labels
            }
        except Exception as e:
            print(f"样本 {idx} 出错: {str(e)}")
            return {
                'paired_image': torch.zeros((3, 256, 512)),
                'keywords': '错误样本',
                'meta': torch.zeros(2),
                'labels': torch.zeros(len(self.disease_cols))
            }

    def __len__(self):
        return len(self.df)