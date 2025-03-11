import faiss
import numpy as np

class FeatureRetrieval:
    def __init__(self, feature_dim, index_path=None):
        self.feature_dim = feature_dim
        self.index = faiss.IndexFlatL2(feature_dim)
        if index_path:
            self.index = faiss.read_index(index_path)

    def add_features(self, features):
        """
        features: numpy 数组，形状 (n_samples, feature_dim)
        """
        self.index.add(features)

    def search(self, query_feature, k=5):
        """
        query_feature: numpy 数组，形状 (feature_dim,)
        返回最相似的 k 个样本的索引和距离
        """
        query_feature = np.expand_dims(query_feature, axis=0)
        distances, indices = self.index.search(query_feature, k)
        return distances, indices
