import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
import pickle
from models.base_recommender import BaseRecommender

class MatrixFactorization(BaseRecommender):
    """基于矩阵分解的推荐系统"""
    
    def __init__(self, n_factors: int = 100, regularization: float = 0.1):
        """
        初始化矩阵分解推荐器
        
        Args:
            n_factors: 隐因子数量
            regularization: 正则化参数
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}  # 用户ID到矩阵索引的映射
        self.item_mapping = {}  # 物品ID到矩阵索引的映射
        self.mean_rating = 0.0
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """创建用户和物品的ID映射"""
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(set(user_ids)))}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(sorted(set(item_ids)))}
        
    def _create_rating_matrix(self, ratings: Dict[int, Dict[int, float]]) -> csr_matrix:
        """创建评分矩阵"""
        # 提取所有用户和物品ID
        user_ids = list(ratings.keys())
        item_ids = []
        for user_ratings in ratings.values():
            item_ids.extend(user_ratings.keys())
            
        # 创建映射
        self._create_mappings(user_ids, item_ids)
        
        # 构建评分矩阵
        rows, cols, data = [], [], []
        for user_id, user_ratings in ratings.items():
            user_idx = self.user_mapping[user_id]
            for item_id, rating in user_ratings.items():
                if item_id in self.item_mapping:  # 确保物品在映射中
                    item_idx = self.item_mapping[item_id]
                    rows.append(user_idx)
                    cols.append(item_idx)
                    data.append(rating)
                    
        return csr_matrix((data, (rows, cols)), 
                         shape=(len(self.user_mapping), len(self.item_mapping)))
    
    def fit(self, user_movie_ratings: Dict[int, Dict[int, float]]) -> None:
        """训练模型"""
        # 创建评分矩阵
        rating_matrix = self._create_rating_matrix(user_movie_ratings)
        
        # 计算全局平均评分
        self.mean_rating = rating_matrix.data.mean()
        
        # 中心化评分矩阵
        centered_matrix = rating_matrix.copy()
        centered_matrix.data -= self.mean_rating
        
        # 执行SVD
        U, sigma, Vt = svds(centered_matrix, k=min(self.n_factors, 
                                                  min(rating_matrix.shape) - 1))
        
        # 调整奇异值
        sigma_diag = np.diag(sigma)
        
        # 保存用户和物品因子
        self.user_factors = U @ np.sqrt(sigma_diag)
        self.item_factors = np.sqrt(sigma_diag) @ Vt
        
    def predict(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """生成推荐"""
        if user_id not in self.user_mapping:
            # 对于新用户，返回平均评分最高的物品
            if hasattr(self, 'average_ratings'):
                sorted_items = sorted(self.average_ratings.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
                return [item_id for item_id, _ in sorted_items[:n_recommendations]]
            else:
                # 如果没有平均评分，返回随机推荐
                return list(self.item_mapping.keys())[:n_recommendations]
        
        # 获取用户的隐因子
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_factors[user_idx]
        
        # 计算预测评分
        predictions = self.mean_rating + user_vector @ self.item_factors
        
        # 获取已评分的物品
        rated_items = set()
        if user_id in self.user_ratings:
            rated_items = set(self.user_ratings[user_id].keys())
        
        # 找到评分最高的未评分物品
        item_scores = []
        for item_id, item_idx in self.item_mapping.items():
            if item_id not in rated_items:
                item_scores.append((item_id, predictions[item_idx]))
        
        # 排序并返回推荐
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [int(item_id) for item_id, _ in item_scores[:n_recommendations]]
    
    def save(self, path: str) -> None:
        """保存模型"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'mean_rating': self.mean_rating,
            'n_factors': self.n_factors,
            'regularization': self.regularization
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str) -> None:
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            self.user_factors = model_data['user_factors']
            self.item_factors = model_data['item_factors']
            self.user_mapping = model_data['user_mapping']
            self.item_mapping = model_data['item_mapping']
            self.mean_rating = model_data['mean_rating']
            self.n_factors = model_data['n_factors']
            self.regularization = model_data['regularization']