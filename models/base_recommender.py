from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRecommender(ABC):
    """推荐系统的基类，定义了推荐器的基本接口"""
    
    @abstractmethod
    def fit(self, user_movie_ratings: Dict[int, Dict[int, float]]) -> None:
        """
        训练推荐模型
        
        Args:
            user_movie_ratings: 用户-电影评分矩阵
                格式: {user_id: {movie_id: rating}}
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """
        为指定用户生成推荐
        
        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            
        Returns:
            推荐的电影ID列表
        """
        pass
    
    def save(self, path: str) -> None:
        """保存模型"""
        pass
    
    def load(self, path: str) -> None:
        """加载模型"""
        pass