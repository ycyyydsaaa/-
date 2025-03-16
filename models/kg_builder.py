# from py2neo import Graph, Node, Relationship
# import torch
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
#
# class MedicalKG:
#     def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="120190333"):
#         try:
#             self.graph = Graph(uri, auth=(user, password))
#         except Exception as e:
#             print(f"Neo4j 连接失败: {e}")
#             raise
#         self.disease_cols = None
#
#     def build_kg(self, df, disease_cols):
#         """
#         使用 Left-Diagnostic Keywords 和 Right-Diagnostic Keywords 两列来构建知识图谱。
#         """
#         self.graph.delete_all()
#         self.disease_cols = disease_cols
#
#         # 创建疾病节点
#         disease_nodes = {}
#         for d in disease_cols:
#             node = Node("Disease", name=d)
#             self.graph.create(node)
#             disease_nodes[d] = node
#
#         symptom_cache = {}
#         for index, row in df.iterrows():
#             for side in ['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords']:
#                 if side not in row:
#                     continue
#                 keywords = row[side]
#                 if isinstance(keywords, str):
#                     keywords_list = [kw.strip().lower() for kw in keywords.replace('，', ',').split(',') if kw.strip()]
#                 else:
#                     keywords_list = []
#                 for kw in keywords_list:
#                     if kw not in symptom_cache:
#                         symptom_node = Node("Symptom", name=kw)
#                         self.graph.merge(symptom_node, "Symptom", "name")
#                         symptom_cache[kw] = symptom_node
#                     else:
#                         symptom_node = symptom_cache[kw]
#                     for d in disease_cols:
#                         if row[d] == 1:
#                             rel = Relationship(disease_nodes[d], "HAS_SYMPTOM", symptom_node)
#                             self.graph.merge(rel)
#
#     def query_disease_symptoms(self, disease):
#         query = "MATCH (d:Disease {name:$disease})-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name AS symptom"
#         result = self.graph.run(query, disease=disease)
#         return [record["symptom"] for record in result]
#
#     def query_symptom_diseases(self, symptom):
#         query = "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom {name:$symptom}) RETURN d.name AS disease"
#         result = self.graph.run(query, symptom=symptom)
#         return [record["disease"] for record in result]
#
#     def validate_prediction(self, predicted_labels, keywords, disease_cols, threshold=0.5):
#         from torch import sigmoid
#         probs = sigmoid(predicted_labels)
#         validation = {}
#         for i, d in enumerate(disease_cols):
#             expected_symptoms = set(self.query_disease_symptoms(d))
#             common = expected_symptoms & set(k.lower() for k in keywords)
#             coverage = len(common) / len(expected_symptoms) if expected_symptoms else 1.0
#             validation[d] = {
#                 "original_prob": probs[0][i].item(),
#                 "coverage": coverage,
#                 "adjusted_prob": probs[0][i].item() * coverage
#             }
#         return validation
#
#     def _get_edge_index(self):
#         edges = []
#         all_symptoms = self.get_all_symptoms()
#         for d in self.disease_cols:
#             symptoms = self.query_disease_symptoms(d)
#             for s in symptoms:
#                 d_idx = self.disease_cols.index(d)
#                 s_idx = all_symptoms.index(s)
#                 edges.append([d_idx, len(self.disease_cols) + s_idx])
#                 edges.append([len(self.disease_cols) + s_idx, d_idx])
#         return torch.tensor(edges, dtype=torch.long).t()
#
#     def get_all_symptoms(self):
#         query = "MATCH (s:Symptom) RETURN s.name"
#         result = self.graph.run(query)
#         return list(set([record["s.name"] for record in result]))
#
#     def generate_disease_embeddings(self, embedding_dim=128):
#         num_diseases = len(self.disease_cols)
#         all_symptoms = self.get_all_symptoms()
#         num_symptoms = len(all_symptoms)
#         num_nodes = num_diseases + num_symptoms
#         print(f"num_diseases: {num_diseases}, num_symptoms: {num_symptoms}, num_nodes: {num_nodes}")
#
#         x = torch.eye(num_nodes)
#         edge_index = self._get_edge_index()
#         print(f"edge_index shape: {edge_index.shape}")
#
#         data = Data(x=x, edge_index=edge_index)
#         gcn = GCNConv(num_nodes, embedding_dim)
#         embeddings = gcn(data.x, data.edge_index)
#         print(f"embeddings shape: {embeddings.shape}")
#
#         disease_embeddings = embeddings[:num_diseases]
#         print(f"disease_embeddings shape: {disease_embeddings.shape}")
#         return disease_embeddings
#
#     def get_adjacency_matrix(self):
#         num_diseases = len(self.disease_cols)
#         A = torch.zeros((num_diseases, num_diseases))
#         disease_symptom_map = {}
#         for d in self.disease_cols:
#             disease_symptom_map[d] = set(self.query_disease_symptoms(d))
#         for i, d1 in enumerate(self.disease_cols):
#             for j, d2 in enumerate(self.disease_cols):
#                 if i != j and disease_symptom_map[d1].intersection(disease_symptom_map[d2]):
#                     A[i, j] = 1
#         print(f"adjacency matrix shape: {A.shape}")
#         return A
from py2neo import Graph, Node, Relationship
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.optim as optim

class MedicalKG:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="120190333"):
        try:
            self.graph = Graph(uri, auth=(user, password))
        except Exception as e:
            print(f"Neo4j 连接失败: {e}")
            raise
        self.disease_cols = None

    def build_kg(self, df, disease_cols, frequency_threshold_ratio=0.9):
        try:
            self.graph.delete_all()
            if not disease_cols:
                raise ValueError("disease_cols 不能为空")
            self.disease_cols = disease_cols
            print(f"build_kg: 设置 disease_cols 为 {self.disease_cols}")

            disease_nodes = {}
            for d in disease_cols:
                node = Node("Disease", name=d)
                self.graph.create(node)
                disease_nodes[d] = node

            keyword_disease_count = {}
            for index, row in df.iterrows():
                for side in ['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords']:
                    if side not in row:
                        continue
                    keywords = row[side]
                    if isinstance(keywords, str):
                        keywords_list = [kw.strip().lower() for kw in keywords.replace('，', ',').split(',') if kw.strip()]
                    else:
                        keywords_list = []
                    for kw in keywords_list:
                        if kw not in keyword_disease_count:
                            keyword_disease_count[kw] = {d: 0 for d in disease_cols}
                        for d in disease_cols:
                            if row[d] == 1:
                                keyword_disease_count[kw][d] += 1

            symptom_cache = {}
            for index, row in df.iterrows():
                for side in ['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords']:
                    if side not in row:
                        continue
                    keywords = row[side]
                    if isinstance(keywords, str):
                        keywords_list = [kw.strip().lower() for kw in keywords.replace('，', ',').split(',') if kw.strip()]
                    else:
                        keywords_list = []
                    for kw in keywords_list:
                        if kw not in symptom_cache:
                            symptom_node = Node("Symptom", name=kw)
                            self.graph.merge(symptom_node, "Symptom", "name")
                            symptom_cache[kw] = symptom_node
                        else:
                            symptom_node = symptom_cache[kw]

                        max_count = max(keyword_disease_count[kw].values())
                        if max_count == 0:
                            continue
                        threshold = max_count * frequency_threshold_ratio
                        associated_diseases = [
                            d for d, count in keyword_disease_count[kw].items()
                            if count >= threshold and count > 0
                        ]
                        for d in associated_diseases:
                            if row[d] == 1:
                                rel = Relationship(disease_nodes[d], "HAS_SYMPTOM", symptom_node)
                                self.graph.merge(rel)

            for d in disease_cols:
                if not self.graph.match((disease_nodes[d], None), "HAS_SYMPTOM"):
                    dummy_symptom = Node("Symptom", name=f"{d}_dummy")
                    self.graph.merge(dummy_symptom, "Symptom", "name")
                    rel = Relationship(disease_nodes[d], "HAS_SYMPTOM", dummy_symptom)
                    self.graph.merge(rel)
        except Exception as e:
            print(f"构建知识图谱失败: {str(e)}")
            self.disease_cols = None
            raise

    def query_disease_symptoms(self, disease):
        query = "MATCH (d:Disease {name:$disease})-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name AS symptom"
        result = self.graph.run(query, disease=disease)
        return [record["symptom"] for record in result]

    def query_symptom_diseases(self, symptom):
        query = "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom {name:$symptom}) RETURN d.name AS disease"
        result = self.graph.run(query, symptom=symptom)
        return [record["disease"] for record in result]

    def validate_prediction(self, predicted_labels, keywords, disease_cols, threshold=0.5):
        from torch import sigmoid
        probs = sigmoid(predicted_labels)
        validation = {}
        for i, d in enumerate(disease_cols):
            expected_symptoms = set(self.query_disease_symptoms(d))
            common = expected_symptoms & set(k.lower() for k in keywords)
            coverage = len(common) / len(expected_symptoms) if expected_symptoms else 1.0
            validation[d] = {
                "original_prob": probs[0][i].item(),
                "coverage": coverage,
                "adjusted_prob": probs[0][i].item() * coverage
            }
        return validation

    def _get_edge_index(self):
        edges = []
        all_symptoms = self.get_all_symptoms()
        for d in self.disease_cols:
            symptoms = self.query_disease_symptoms(d)
            for s in symptoms:
                d_idx = self.disease_cols.index(d)
                s_idx = all_symptoms.index(s)
                edges.append([d_idx, len(self.disease_cols) + s_idx])
                edges.append([len(self.disease_cols) + s_idx, d_idx])
        return torch.tensor(edges, dtype=torch.long).t()

    def get_all_symptoms(self):
        query = "MATCH (s:Symptom) RETURN s.name"
        result = self.graph.run(query)
        return list(set([record["s.name"] for record in result]))

    def generate_disease_embeddings(self, embedding_dim=128):
        if self.disease_cols is None:
            raise ValueError("disease_cols 未初始化，请先调用 build_kg 方法")
        num_diseases = len(self.disease_cols)
        all_symptoms = self.get_all_symptoms()
        num_symptoms = len(all_symptoms)
        num_nodes = num_diseases + num_symptoms
        print(f"num_diseases: {num_diseases}, num_symptoms: {num_symptoms}, num_nodes: {num_nodes}")

        x = torch.eye(num_nodes)
        edge_index = self._get_edge_index()
        print(f"edge_index shape: {edge_index.shape}")

        data = Data(x=x, edge_index=edge_index)
        gcn = GCNConv(num_nodes, embedding_dim)
        embeddings = gcn(data.x, data.edge_index)
        print(f"embeddings shape: {embeddings.shape}")

        disease_embeddings = embeddings[:num_diseases]
        print(f"disease_embeddings shape: {disease_embeddings.shape}")
        return disease_embeddings

    def get_adjacency_matrix(self):
        if self.disease_cols is None:
            raise ValueError("disease_cols 未初始化，请先调用 build_kg 方法")
        num_diseases = len(self.disease_cols)
        A = torch.zeros((num_diseases, num_diseases))
        disease_symptom_map = {}
        for d in self.disease_cols:
            disease_symptom_map[d] = set(self.query_disease_symptoms(d))
        for i, d1 in enumerate(self.disease_cols):
            for j, d2 in enumerate(self.disease_cols):
                if i != j and disease_symptom_map[d1].intersection(disease_symptom_map[d2]):
                    A[i, j] = 1
        print(f"adjacency matrix shape: {A.shape}")
        return A

    def generate_pseudo_labels(self, df, dataloader, model, num_steps=100, lr=1e-3, device='cuda'):
        print("Running sample-level generate_pseudo_labels - Version 2025-03-14")
        if self.disease_cols is None:
            raise ValueError("disease_cols 未初始化，请先调用 build_kg 方法")

        num_diseases = len(self.disease_cols)
        num_samples = len(df)
        print(f"num_diseases: {num_diseases}, num_samples: {num_samples}")

        # 获取图结构
        all_symptoms = self.get_all_symptoms()
        num_symptoms = len(all_symptoms)
        num_nodes = num_diseases + num_symptoms
        edge_index = self._get_edge_index().to(device)

        # 计算疾病共现矩阵
        pos_weights = torch.tensor([min(1 / (df[d].mean() + 1e-6), 10.0) for d in self.disease_cols], device=device)
        pos_weights = pos_weights / pos_weights.mean()
        disease_cooccur = torch.zeros((num_diseases, num_diseases), device=device)
        for idx, row in df.iterrows():
            labels = torch.tensor(row[self.disease_cols].values.astype(float), device=device)
            weighted_labels = labels * pos_weights
            disease_cooccur += weighted_labels.outer(weighted_labels)
        smooth_factor = 0.1
        denominator = disease_cooccur.sum() + num_diseases * smooth_factor
        if denominator == 0:
            denominator = torch.tensor(num_diseases * 1e-6, device=device)
        disease_cooccur = (disease_cooccur + smooth_factor) / denominator
        disease_cooccur.fill_diagonal_(1.0)

        # 初始化 GCN
        gcn = GCNConv(num_nodes, num_diseases).to(device)
        optimizer = optim.Adam(gcn.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # 提取样本特征（关键修正部分）
        model.eval()
        pseudo_labels = torch.zeros((num_samples, num_diseases), device=device)
        true_labels = torch.tensor(df[self.disease_cols].values.astype(float), device=device)
        feature_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue
                paired_img = batch['paired_image'].to(device)
                text_feature = batch['text_feature'].to(device)
                meta = batch['meta'].to(device)
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(start_idx + paired_img.size(0), num_samples)

                # ========== 修正特征提取逻辑 ==========
                # 提取中间特征 (FPN输出)
                _, _, _, feature_maps, _ = model(paired_img, text_feature, meta, None)
                # 全局平均池化 [batch_size, 256, H, W] → [batch_size, 256]
                img_features = feature_maps.mean([2, 3])
                # =====================================

                feature_list.append(img_features)
                pseudo_labels[start_idx:end_idx] = true_labels[start_idx:end_idx]

        features = torch.cat(feature_list, dim=0)  # [num_samples, 256]

        # 投影特征到疾病维度
        feature_projector = nn.Linear(256, num_diseases).to(device)  # 输入维度256 → 输出维度8
        sample_features = feature_projector(features)  # [num_samples, 8]

        # GCN 训练（保持不变）
        x = torch.eye(num_nodes).to(device)
        data = Data(x=x, edge_index=edge_index)
        for step in range(num_steps):
            optimizer.zero_grad()
            out = gcn(data.x, data.edge_index)  # [num_nodes, 8]
            loss = criterion(out[:num_diseases], disease_cooccur)
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print(f"Step {step}, loss: {loss.item()}")

        # 生成样本级伪标签（保持不变）
        with torch.no_grad():
            no_label_count = (true_labels.sum(dim=1) == 0).sum().item()
            print(f"无标签样本数量: {no_label_count} / {num_samples}")
            for idx in range(num_samples):
                if true_labels[idx].sum() == 0:  # 无标签样本
                    feature_vector = sample_features[idx]  # [8]
                    x_sample = disease_cooccur + 0.3 * feature_vector.unsqueeze(0)  # [1,8]
                    out = gcn(x_sample, edge_index[:num_diseases, :])  # [1,8]
                    probs = torch.sigmoid(out.diag())  # [8]
                    thresholds = torch.clamp(0.5 / pos_weights, min=0.1, max=0.9)
                    pseudo_labels[idx] = (probs > thresholds).float()
                    if pseudo_labels[idx, 0] == 1:  # 'N' 排他性
                        pseudo_labels[idx, 1:] = 0

        print(f"Pseudo labels shape: {pseudo_labels.shape}")
        print(f"Sample pseudo labels (first 5): {pseudo_labels[:5].tolist()}")
        torch.cuda.empty_cache()
        return pseudo_labels