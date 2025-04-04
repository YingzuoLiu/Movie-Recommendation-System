import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
import faiss
import os
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FaissIndex:
    """Faiss索引包装器，用于高效的近邻搜索"""
    
    def __init__(self, dimension: int, index_type: str = "IVFFlat", 
                 metric: str = "cosine", n_lists: int = 100):
        """
        初始化Faiss索引
        
        Args:
            dimension: 向量维度
            index_type: 索引类型 ("Flat", "IVFFlat", "IVFPQ")
            metric: 距离度量 ("cosine", "l2")
            n_lists: IVF聚类中心数量
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.n_lists = min(n_lists, 100000)  # 防止聚类中心过多
        self.index = None
        self.id_mapping = {}  # 外部ID到索引位置的映射
        
    def _create_index(self, n_vectors: int) -> faiss.Index:
        """创建适合数据规模的索引"""
        if self.metric == "cosine":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2
            
        if self.index_type == "Flat":
            return faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" \
                else faiss.IndexFlatL2(self.dimension)
                
        elif self.index_type == "IVFFlat":
            # 根据数据量调整聚类中心数量
            n_lists = min(self.n_lists, max(int(np.sqrt(n_vectors)), 10))
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 
                                    n_lists, faiss_metric)
                                    
        elif self.index_type == "IVFPQ":
            # 对于大规模数据使用PQ压缩
            n_lists = min(self.n_lists, max(int(np.sqrt(n_vectors)), 10))
            quantizer = faiss.IndexFlatL2(self.dimension)
            # 使用4位编码，32字节子向量
            return faiss.IndexIVFPQ(quantizer, self.dimension, n_lists, 
                                  8, 4, faiss_metric)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """如果使用余弦相似度，对向量进行L2归一化"""
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms
        return vectors
        
    def build(self, vectors: np.ndarray, ids: List[int]) -> None:
        """构建索引"""
        try:
            n_vectors = len(vectors)
            if n_vectors == 0:
                raise ValueError("Empty vector list")
                
            # 创建ID映射
            self.id_mapping = {id: idx for idx, id in enumerate(ids)}
            
            # 预处理向量
            vectors = self._normalize(vectors.astype(np.float32))
            
            # 创建并训练索引
            self.index = self._create_index(n_vectors)
            
            if isinstance(self.index, faiss.IndexIVF):
                logger.info(f"Training index with {n_vectors} vectors...")
                self.index.train(vectors)
                
            # 添加向量到索引
            self.index.add(vectors)
            logger.info(f"Built index with {n_vectors} vectors")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
            
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最近邻"""
        if self.index is None:
            raise ValueError("Index not built")
            
        # 预处理查询向量
        query = self._normalize(query.reshape(1, -1).astype(np.float32))
        
        # 执行搜索
        distances, indices = self.index.search(query, k)
        
        # 转换回原始ID
        reverse_mapping = {v: k for k, v in self.id_mapping.items()}
        original_ids = np.array([reverse_mapping[idx] for idx in indices[0]])
        
        return distances[0], original_ids
        
    def add_items(self, vectors: np.ndarray, ids: List[int]) -> None:
        """添加新项目到索引"""
        try:
            if self.index is None:
                self.build(vectors, ids)
                return
                
            # 更新ID映射
            start_idx = len(self.id_mapping)
            for i, id in enumerate(ids):
                self.id_mapping[id] = start_idx + i
                
            # 预处理向量
            vectors = self._normalize(vectors.astype(np.float32))
            
            # 添加到索引
            self.index.add(vectors)
            logger.info(f"Added {len(vectors)} vectors to index")
            
        except Exception as e:
            logger.error(f"Error adding items: {e}")
            raise
            
    def save(self, path: str) -> None:
            """保存索引"""
            try:
                if self.index is not None:
                    faiss.write_index(self.index, f"{path}.index")
                    np.save(f"{path}.mapping.npy", self.id_mapping)
                    logger.info(f"Successfully saved index to {path}")
            except Exception as e:
                logger.error(f"Error saving index: {e}")
                raise
                
    def load(self, path: str) -> None:
        """加载索引"""
        try:
            mapping_path = f"{path}.mapping.npy"
            index_path = f"{path}.index"
            
            if not os.path.exists(mapping_path) or not os.path.exists(index_path):
                raise FileNotFoundError(f"Index files not found at {path}")
                
            self.index = faiss.read_index(index_path)
            self.id_mapping = np.load(mapping_path, allow_pickle=True).item()
            logger.info(f"Successfully loaded index from {path}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise