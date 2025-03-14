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
        """
        初始化 FundusDataset 类。

        参数:
            excel_path (str): Excel 文件路径，包含图像元数据和标签
            img_root (str): 图像文件根目录
            disease_cols (list): 疾病标签列名列表
            phase (str): 数据集阶段，'train' 或其他（如 'val', 'test'）
        """
        # 验证输入参数
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel 文件 {excel_path} 不存在")
        if not os.path.exists(img_root):
            raise FileNotFoundError(f"图像根目录 {img_root} 不存在")

        self.df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.disease_cols = disease_cols
        self.phase = phase

        # 数据清理和验证
        self._validate_and_clean()

        # 定义变换流水线
        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
                transforms.RandomApply([transforms.Lambda(self.random_pixel_scale)], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # 初始化 BERT 分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased').eval()

    def random_pixel_scale(self, img):
        """
        随机缩放图像像素值，确保结果在 [0, 1] 范围内。
        """
        scale = 0.8 + torch.rand(1).item() * 0.4
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor * scale
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return transforms.ToPILImage()(img_tensor)

    def _validate_and_clean(self):
        """验证和清理数据集"""
        # 检查必要列是否存在
        if 'paired_image' not in self.df.columns and 'id' not in self.df.columns:
            raise ValueError("Excel 文件中必须包含 'paired_image' 或 'id' 列.")

        # 检查图像文件是否存在或是否损坏
        missing_or_corrupt = []
        for idx, row in self.df.iterrows():
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
            self.df.drop(missing_or_corrupt, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        # 检查标签有效性
        invalid_labels = self.df[self.disease_cols].apply(lambda x: ~x.isin([0, 1])).any(axis=1)
        if invalid_labels.any():
            invalid_indices = self.df[invalid_labels].index.tolist()
            logger.warning(f"移除无效标签的样本索引: {invalid_indices}")
            self.df = self.df[~invalid_labels].reset_index(drop=True)

        # 验证和处理元数据
        if 'Patient Age' not in self.df.columns:
            logger.warning("警告: 列 'Patient Age' 不存在，将使用默认值 0.0")
            self.df['Patient Age'] = 0.0
        else:
            self.df['Patient Age'] = pd.to_numeric(self.df['Patient Age'], errors='coerce').fillna(0.0)
            if self.df['Patient Age'].isnull().any():
                logger.warning("警告: 'Patient Age' 包含无效值，已填充为 0.0")

        if 'Patient Sex' not in self.df.columns:
            logger.warning("警告: 列 'Patient Sex' 不存在，将使用默认值 0.0")
            self.df['Patient Sex'] = 0.0
        else:
            self.df['Patient Sex'] = self.df['Patient Sex'].map({'Male': 1.0, 'Female': 0.0}).fillna(0.0)
            if self.df['Patient Sex'].isnull().any():
                logger.warning("警告: 'Patient Sex' 包含无效值，已填充为 0.0")

    def load_image(self, path):
        # logger.info(f"尝试加载图像: {path}")
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
        """获取单个样本"""
        row = self.df.iloc[idx]
        try:
            # 获取图像路径
            img_file = str(row['paired_image']) if 'paired_image' in row and pd.notna(
                row['paired_image']) else f"{row['id']}.png"
            img_path = os.path.join(self.img_root, img_file)
            # logger.info(f"尝试加载图像: {img_path}")
            img = self.load_image(img_path)

            # 检查图像是否加载成功
            if img is None:
                logger.warning(f"跳过损坏的图像 {img_path} (样本 {idx})")
                return None

            # 应用变换
            img = self.transform(img)

            # 动态计算文本特征
            left_keywords = row.get('Left-Diagnostic Keywords', '无关键词')
            right_keywords = row.get('Right-Diagnostic Keywords', '无关键词')
            keywords = f"{left_keywords} {right_keywords}"
            inputs = self.tokenizer(keywords, return_tensors='pt', padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
            text_feature = outputs.last_hidden_state.mean(1).squeeze(0)

            # 获取元数据和标签
            age = float(row['Patient Age'])
            gender = float(row['Patient Sex'])
            meta = torch.tensor([age, gender], dtype=torch.float32)
            labels = torch.tensor(row[self.disease_cols].values.astype(float), dtype=torch.float32)

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
        """返回数据集大小"""
        return len(self.df)

if __name__ == "__main__":
    # 示例用法（用于测试）
    import tempfile
    import numpy as np

    # 创建临时 Excel 文件用于测试
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
        df = pd.DataFrame({
            'id': [1, 2],
            'paired_image': ['image1.png', 'image2.png'],
            'Patient Age': [30, 40],
            'Patient Sex': ['Male', 'Female'],
            'Left-Diagnostic Keywords': ['diabetes', 'normal'],
            'Right-Diagnostic Keywords': ['normal', 'glaucoma'],
            'N': [0, 1], 'D': [1, 0], 'G': [0, 1], 'C': [0, 0], 'A': [0, 0], 'H': [0, 0], 'M': [0, 0], 'O': [0, 0]
        })
        df.to_excel(f.name, index=False)
        excel_path = f.name

    # 创建临时图像目录
    with tempfile.TemporaryDirectory() as img_root:
        # 创建 dummy 图像文件
        for i in range(1, 3):
            img = Image.fromarray(np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8))
            img.save(os.path.join(img_root, f"image{i}.png"))

        # 初始化数据集
        dataset = FundusDataset(
            excel_path=excel_path,
            img_root=img_root,
            disease_cols=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'],
            phase='train'
        )

        # 测试数据集
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is None:
                print(f"样本 {i} 无效，跳过")
                continue
            print(f"样本 {i}: 图像形状 {sample['paired_image'].shape}, 文本特征形状 {sample['text_feature'].shape}, "
                  f"元数据 {sample['meta']}, 标签 {sample['labels']}")

    # 清理临时文件
    os.unlink(excel_path)