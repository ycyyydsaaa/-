import gc
import os
import pickle
import torch
from py2neo import Graph, Node, Relationship
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import torch.nn.functional as F
import logging
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

class MedicalKG:
    def __init__(self, uri="bolt://0.tcp.ap.ngrok.io:16358", user="neo4j", password="120190333", local_dir="/data/coding/eye/pycharm_project_257/kg_data"):
        self.uri = uri
        self.user = user
        self.password = password
        self.local_dir = local_dir
        self.graph = None
        self.disease_cols = None
        self.symptom_cache = {}
        self.all_symptoms_cache = None
        self.disease_symptom_map = {}
        self.kg_embeddings = None
        self.A = None

        os.makedirs(self.local_dir, exist_ok=True)
        self.nodes_path = os.path.join(self.local_dir, "kg_nodes.pkl")
        self.embeddings_path = os.path.join(self.local_dir, "kg_embeddings.pt")
        self.adjacency_path = os.path.join(self.local_dir, "adjacency_matrix.pt")

    def _connect_to_neo4j(self):
        try:
            self.graph = Graph(self.uri, auth=(self.user, self.password))
            logger.info("成功连接到 Neo4j 数据库")
            result = self.graph.run("RETURN 1")
            logger.info(f"Neo4j 测试查询结果: {list(result)}")
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {str(e)}")
            self.graph = None

    def build_kg(self, df, disease_cols, frequency_threshold_ratio=0.9, batch_size=1000):
        if not disease_cols:
            logger.error("disease_cols 为空，无法构建知识图谱")
            raise ValueError("disease_cols 不能为空")
        if df.empty:
            logger.error("输入的 DataFrame 为空，无法构建知识图谱")
            raise ValueError("输入的 DataFrame 为空")

        if self._load_local_data():
            logger.info("从本地加载知识图谱数据")
            return

        self._connect_to_neo4j()
        if self.graph is None:
            logger.warning("Neo4j 连接失败，将使用默认数据")
            self.disease_cols = disease_cols
            self.disease_symptom_map = {d: set() for d in disease_cols}
            for d in disease_cols:
                self.disease_symptom_map[d].add(f"{d}_dummy")
            return

        try:
            self.graph.delete_all()
            self.disease_cols = disease_cols
            logger.info(f"build_kg: 设置 disease_cols 为 {self.disease_cols}")

            disease_nodes = {}
            for d in disease_cols:
                node = Node("Disease", name=d)
                self.graph.create(node)
                disease_nodes[d] = node

            keyword_disease_count = {}
            for start in range(0, len(df), batch_size):
                batch_df = df[start:start + batch_size]
                for index, row in batch_df.iterrows():
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
            for start in range(0, len(df), batch_size):
                batch_df = df[start:start + batch_size]
                for index, row in batch_df.iterrows():
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

            for d in self.disease_cols:
                self.disease_symptom_map[d] = set(self.query_disease_symptoms(d))

        except Exception as e:
            logger.error(f"构建知识图谱失败: {str(e)}")
            self.disease_cols = None
            raise

    def query_disease_symptoms(self, disease):
        if disease in self.symptom_cache:
            return self.symptom_cache[disease]
        if disease in self.disease_symptom_map:
            self.symptom_cache[disease] = list(self.disease_symptom_map[disease])
            return self.symptom_cache[disease]
        if self.graph is None:
            raise ValueError("Neo4j 数据库未连接，且本地数据不可用")
        query = "MATCH (d:Disease {name:$disease})-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name AS symptom"
        result = self.graph.run(query, disease=disease)
        symptoms = [record["symptom"] for record in result]
        self.symptom_cache[disease] = symptoms
        return symptoms

    def query_symptom_diseases(self, symptom):
        if self.disease_symptom_map:
            diseases = [d for d, symptoms in self.disease_symptom_map.items() if symptom in symptoms]
            return diseases
        if self.graph is None:
            raise ValueError("Neo4j 数据库未连接，且本地数据不可用")
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
        if self.all_symptoms_cache is not None:
            return self.all_symptoms_cache
        if self.graph is None:
            all_symptoms = set()
            for symptoms in self.disease_symptom_map.values():
                all_symptoms.update(symptoms)
            self.all_symptoms_cache = list(all_symptoms)
            return self.all_symptoms_cache
        query = "MATCH (s:Symptom) RETURN s.name"
        result = self.graph.run(query)
        self.all_symptoms_cache = list(set([record["s.name"] for record in result]))
        return self.all_symptoms_cache

    def generate_disease_embeddings(self, embedding_dim=128):
        if self.kg_embeddings is not None:
            return self.kg_embeddings
        if self.disease_cols is None:
            logger.error("disease_cols 未初始化，请先调用 build_kg 方法")
            raise ValueError("disease_cols 未初始化，请先调用 build_kg 方法")
        num_diseases = len(self.disease_cols)
        all_symptoms = self.get_all_symptoms()
        num_symptoms = len(all_symptoms)
        num_nodes = num_diseases + num_symptoms

        x = torch.zeros(num_nodes, embedding_dim)
        for i in range(num_nodes):
            x[i, i % embedding_dim] = 1.0
        edge_index = self._get_edge_index()

        data = Data(x=x, edge_index=edge_index)
        gcn1 = GCNConv(embedding_dim, 256)
        gcn2 = GCNConv(256, embedding_dim)
        with torch.no_grad():
            x = F.relu(gcn1(data.x, data.edge_index))
            embeddings = gcn2(x, data.edge_index)

        self.kg_embeddings = embeddings[:num_diseases]
        print_tensor_info(self.kg_embeddings, "kg_embeddings_after_generation")

        del x, edge_index, data, embeddings, gcn1, gcn2
        torch.cuda.empty_cache()
        gc.collect()

        return self.kg_embeddings

    def get_adjacency_matrix(self):
        if self.A is not None:
            return self.A
        if self.disease_cols is None:
            logger.error("disease_cols 未初始化，请先调用 build_kg 方法")
            raise ValueError("disease_cols 未初始化，请先调用 build_kg 方法")
        num_diseases = len(self.disease_cols)
        A = torch.zeros((num_diseases, num_diseases))
        for i, d1 in enumerate(self.disease_cols):
            for j, d2 in enumerate(self.disease_cols):
                if i != j and self.disease_symptom_map[d1].intersection(self.disease_symptom_map[d2]):
                    A[i, j] = 1
        self.A = A.to_sparse()
        print_tensor_info(self.A, "A_after_generation")
        return self.A

    def _save_local_data(self):
        kg_nodes = {
            'disease_cols': self.disease_cols,
            'all_symptoms': self.get_all_symptoms(),
            'disease_symptom_map': self.disease_symptom_map,
            'symptom_cache': self.symptom_cache
        }
        with open(self.nodes_path, 'wb') as f:
            pickle.dump(kg_nodes, f)
        logger.info(f"知识图谱节点数据已保存到 {self.nodes_path}")

        if self.kg_embeddings is not None:
            torch.save(self.kg_embeddings, self.embeddings_path)
            logger.info(f"疾病嵌入已保存到 {self.embeddings_path}")
        else:
            logger.warning("kg_embeddings 为 None，未保存疾病嵌入")
        if self.A is not None:
            torch.save(self.A, self.adjacency_path)
            logger.info(f"邻接矩阵已保存到 {self.adjacency_path}")
        else:
            logger.warning("A 为 None，未保存邻接矩阵")

    def _load_local_data(self):
        if not (os.path.exists(self.nodes_path) and os.path.exists(self.embeddings_path) and os.path.exists(self.adjacency_path)):
            logger.warning("本地数据文件缺失，无法加载")
            return False

        try:
            with open(self.nodes_path, 'rb') as f:
                kg_nodes = pickle.load(f)
            self.disease_cols = kg_nodes['disease_cols']
            self.all_symptoms_cache = kg_nodes['all_symptoms']
            self.disease_symptom_map = kg_nodes['disease_symptom_map']
            self.symptom_cache = kg_nodes['symptom_cache']
            logger.info(f"从 {self.nodes_path} 加载知识图谱节点数据")

            self.kg_embeddings = torch.load(self.embeddings_path, map_location='cpu')
            self.A = torch.load(self.adjacency_path, map_location='cpu')
            logger.info(f"从 {self.embeddings_path} 加载疾病嵌入")
            logger.info(f"从 {self.adjacency_path} 加载邻接矩阵")
            return True
        except Exception as e:
            logger.error(f"加载本地数据失败: {str(e)}")
            return False

    def clear_cache(self):
        print("Before clearing KG cache:")
        print_tensor_info(self.kg_embeddings, "kg_embeddings")
        print_tensor_info(self.A, "A")

        self.symptom_cache.clear()
        self.all_symptoms_cache = None
        self.disease_symptom_map.clear()
        self.kg_embeddings = None
        self.A = None
        self.graph = None
        gc.collect()
        torch.cuda.empty_cache()

        print("After clearing KG cache:")
        print_tensor_info(self.kg_embeddings, "kg_embeddings")
        print_tensor_info(self.A, "A")
        logger.info("知识图谱缓存已清空")