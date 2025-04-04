import numpy as np
import torch
from typing import List, Dict, Optional
from models.wide_deep import WideAndDeepRecommender
from models.faiss_search import FaissIndex
import logging

logger = logging.getLogger(__name__)

class HybridRecommender:
    """结合神经网络和近邻搜索的混合推荐系统"""
    
    def __init__(self, embedding_dim: int = 32,
                 hidden_layers: List[int] = [64, 32],
                 index_type: str = "IVFFlat"):
        self.model = WideAndDeepRecommender(
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers
        )
        self.faiss_index = FaissIndex(
            dimension=embedding_dim,
            index_type=index_type
        )
        self.embedding_dim = embedding_dim
        
    def _get_item_embeddings(self) -> np.ndarray:
        """获取所有物品的嵌入向量"""
        self.model.model.eval()
        with torch.no_grad():
            item_ids = torch.arange(len(self.model.item_mapping)).to(self.model.device)
            embeddings = self.model.model.item_embedding(item_ids).cpu().numpy()
        return embeddings
        
    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """获取用户的嵌入向量"""
        if user_id not in self.model.user_mapping:
            return None
            
        self.model.model.eval()
        with torch.no_grad():
            user_idx = self.model.user_mapping[user_id]
            user_tensor = torch.LongTensor([user_idx]).to(self.model.device)
            embedding = self.model.model.user_embedding(user_tensor).cpu().numpy()
        return embedding
        
    def fit(self, user_movie_ratings: Dict[int, Dict[int, float]]) -> None:
        """训练模型并构建索引"""
        logger.info("Training Wide & Deep model...")
        self.model.fit(user_movie_ratings)
        
        logger.info("Building Faiss index...")
        item_embeddings = self._get_item_embeddings()
        item_ids = list(self.model.item_mapping.keys())
        self.faiss_index.build(item_embeddings, item_ids)
        
    def predict(self, user_id: int, n_recommendations: int = 5,
                alpha: float = 0.5) -> List[int]:
        """生成混合推荐"""
        # 获取用户嵌入
        user_embedding = self._get_user_embedding(user_id)
        
        if user_embedding is None:
            # 对于新用户，使用Wide & Deep模型的冷启动策略
            return self.model.predict(user_id, n_recommendations)
            
        # 获取近邻推荐
        try:
            distances, neighbor_ids = self.faiss_index.search(
                user_embedding,
                k=n_recommendations * 2  # 获取更多候选项
            )
        except Exception as e:
            logger.error(f"Faiss search error: {e}")
            neighbor_ids = []
            
        # 获取模型推荐
        model_recommendations = self.model.predict(
            user_id,
            n_recommendations * 2
        )
        
        # 合并推荐结果
        all_recommendations = []
        seen_items = set()
        
        # 按照混合权重交替添加推荐
        i, j = 0, 0
        while len(all_recommendations) < n_recommendations and \
              (i < len(neighbor_ids) or j < len(model_recommendations)):
            
            # 添加近邻推荐
            if i < len(neighbor_ids):
                item_id = int(neighbor_ids[i])
                if item_id not in seen_items:
                    all_recommendations.append((item_id, 1 - alpha))
                    seen_items.add(item_id)
                i += 1
                
            # 添加模型推荐
            if j < len(model_recommendations):
                item_id = model_recommendations[j]
                if item_id not in seen_items:
                    all_recommendations.append((item_id, alpha))
                    seen_items.add(item_id)
                j += 1
                
        # 返回推荐结果
        return [item_id for item_id, _ in all_recommendations[:n_recommendations]]
        
    def save(self, path: str) -> None:
            """保存模型和索引"""
            try:
                logger.info(f"Saving hybrid model to {path}")
                self.model.save(f"{path}_model.pt")
                self.faiss_index.save(f"{path}_index")
                logger.info("Successfully saved hybrid model")
            except Exception as e:
                logger.error(f"Error saving hybrid model: {e}")
                raise
        
    def load(self, path: str) -> None:
        """加载模型和索引"""
        try:
            logger.info(f"Loading hybrid model from {path}")
            self.model.load(f"{path}_model.pt")
            self.faiss_index.load(f"{path}_index")
            logger.info("Successfully loaded hybrid model")
        except Exception as e:
            logger.error(f"Error loading hybrid model: {e}")
            raise