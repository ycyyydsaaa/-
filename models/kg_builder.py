from py2neo import Graph, Node, Relationship
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd  # 添加 pandas 导入
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
                    if side not in row or pd.isna(row[side]):
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
                    if side not in row or pd.isna(row[side]):
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