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
import numpy as np
from functools import lru_cache

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_tensor_info(tensor, name):
    """打印张量内存信息"""
    if tensor is not None:
        memory_mb = tensor.element_size() * tensor.nelement() / 1024 ** 2
        ref_count = sys.getrefcount(tensor)
        print(f"Tensor {name}: Memory = {memory_mb:.2f} MB, Ref Count = {ref_count}")
    else:
        print(f"Tensor {name}: None")


class FundusDataset(Dataset):
    """
    改进后的数据集类，修复内存泄漏问题
    关键改进：
    1. BERT模型单例化
    2. 图像加载内存优化
    3. 文本特征缓存管理
    """

    # 类共享的BERT模型（单例）
    _bert_model = None
    _tokenizer = None

    def __init__(self, excel_path, img_root, disease_cols, phase='train', transform=None):
        """
        初始化数据集
        Args:
            excel_path: 标签文件路径
            img_root: 图像根目录
            disease_cols: 疾病标签列名列表
            phase: 数据集阶段（train/test）
            transform: 图像变换
        """
        # 路径验证
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel文件 {excel_path} 不存在")
        if not os.path.exists(img_root):
            raise FileNotFoundError(f"图像根目录 {img_root} 不存在")

        # 读取数据并验证
        df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.disease_cols = disease_cols
        self.phase = phase
        self.transform = transform

        # 数据清洗和验证
        self._validate_and_clean(df)

        # 转换为轻量级数据结构（避免保留DataFrame引用）
        self.data = [dict(row) for _, row in df.iterrows()]
        del df  # 立即释放DataFrame内存
        gc.collect()

        # 预加载文本特征（内存安全方式）
        self._load_text_encoder()
        self.preprocess_text_features()
        self._unload_text_encoder()

    def _load_text_encoder(self):
        """安全加载BERT模型和分词器"""
        if FundusDataset._bert_model is None:
            bert_path = "/data/eye/pycharm_project_257/models/bert-base-uncased"
            FundusDataset._bert_model = BertModel.from_pretrained(
                bert_path, local_files_only=True
            ).eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if FundusDataset._tokenizer is None:
            bert_path = "/data/eye/pycharm_project_257/models/bert-base-uncased"
            FundusDataset._tokenizer = BertTokenizer.from_pretrained(
                bert_path, local_files_only=True
            )

    def _unload_text_encoder(self):
        """释放BERT资源但不删除单例"""
        if hasattr(self, 'text_encoder'):
            del self.text_encoder
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()

    def _validate_and_clean(self, df):
        """数据验证和清洗"""
        # 验证图像文件
        missing_or_corrupt = []
        for idx, row in df.iterrows():
            img_file = str(row['paired_image']) if 'paired_image' in row and pd.notna(
                row['paired_image']) else f"{row['id']}.png"
            img_path = os.path.join(self.img_root, img_file)

            if not os.path.exists(img_path):
                missing_or_corrupt.append(idx)
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify()  # 验证图像完整性
            except Exception as e:
                logger.warning(f"图像 {img_path} 损坏: {str(e)}")
                missing_or_corrupt.append(idx)

        if missing_or_corrupt:
            logger.warning(f"移除缺失或损坏图像的样本索引: {missing_or_corrupt}")
            df.drop(missing_or_corrupt, inplace=True)
            df.reset_index(drop=True, inplace=True)

        # 验证标签
        invalid_labels = df[self.disease_cols].apply(lambda x: ~x.isin([0, 1])).any(axis=1)
        if invalid_labels.any():
            invalid_indices = df[invalid_labels].index.tolist()
            logger.warning(f"移除无效标签的样本索引: {invalid_indices}")
            df.drop(invalid_indices, inplace=True)
            df.reset_index(drop=True, inplace=True)

        # 处理元数据
        for col in self.disease_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        df['Patient Age'] = pd.to_numeric(df.get('Patient Age', 0), errors='coerce').fillna(0).astype('float32')
        df['Patient Sex'] = df.get('Patient Sex', 0).map({'Male': 1.0, 'Female': 0.0}).fillna(0).astype('float32')

    @lru_cache(maxsize=1000)  # 控制缓存大小
    def _load_cached_text_feature(self, feature_path):
        """
        带缓存的文本特征加载
        Args:
            feature_path: 特征文件路径
        Returns:
            torch.Tensor: 文本特征向量
        """
        try:
            feature = torch.load(feature_path, map_location='cpu')
            return feature.squeeze().clone()  # 确保返回副本
        except:
            return torch.zeros(768)  # 默认值

    def preprocess_text_features(self):
        """预生成文本特征文件（内存安全方式）"""
        for idx, row in enumerate(self.data):
            text_feature_path = os.path.join(self.img_root, f"{row['id']}_text.pt")
            if not os.path.exists(text_feature_path):
                left_keywords = row.get('Left-Diagnostic Keywords', '无关键词')
                right_keywords = row.get('Right-Diagnostic Keywords', '无关键词')
                keywords = f"{left_keywords} {right_keywords}"

                inputs = FundusDataset._tokenizer(
                    keywords,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(FundusDataset._bert_model.device)

                with torch.no_grad():
                    outputs = FundusDataset._bert_model(**inputs)
                    text_feature = outputs.last_hidden_state.mean(1).squeeze(0).cpu()

                torch.save(text_feature, text_feature_path)
                logger.debug(f"保存文本特征: {text_feature_path}, 形状: {text_feature.shape}")

                del inputs, outputs, text_feature
                torch.cuda.empty_cache()

    def load_image(self, path):
        """
        安全加载图像方法
        Args:
            path: 图像路径
        Returns:
            np.ndarray: RGB图像数组
        """
        try:
            with Image.open(path) as img:
                # 转换为numpy数组并立即关闭文件
                img_array = np.array(img.convert('RGB'))
                return img_array
        except Exception as e:
            logger.error(f"加载图像失败 {path}: {str(e)}")
            return None

    def __getitem__(self, idx):
        """
        获取样本（内存安全实现）
        Args:
            idx: 样本索引
        Returns:
            dict: 包含图像、文本特征、元数据和标签的字典
        """
        row = self.data[idx]
        try:
            # 加载图像
            img_file = str(row.get('paired_image', f"{row['id']}.png"))
            img_path = os.path.join(self.img_root, img_file)
            img_array = self.load_image(img_path)

            if img_array is None:
                return None

            # 转换图像张量
            img = self.transform(img_array) if self.transform else \
                torch.from_numpy(img_array).permute(2, 0, 1).float() / 255
            del img_array  # 立即释放numpy数组

            # 加载文本特征
            text_feature_path = os.path.join(self.img_root, f"{row['id']}_text.pt")
            text_feature = self._load_cached_text_feature(text_feature_path)

            # 元数据
            meta = torch.tensor([
                float(row.get('Patient Age', 0)),
                float(row.get('Patient Sex', 0))
            ], dtype=torch.float32)

            # 标签
            labels = torch.tensor([row[col] for col in self.disease_cols], dtype=torch.float32)

            return {
                'paired_image': img,
                'text_feature': text_feature,
                'meta': meta,
                'labels': labels
            }

        except Exception as e:
            logger.error(f"样本 {idx} 处理出错: {str(e)}", exc_info=True)
            return None

    def __len__(self):
        return len(self.data)

    def clear_cache(self):
        """清理所有缓存"""
        self._load_cached_text_feature.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()