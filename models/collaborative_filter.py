import numpy as np
from typing import List, Dict, Tuple
import pickle
from models.base_recommender import BaseRecommender

class CollaborativeFilter(BaseRecommender):
    """基于用户的协同过滤推荐器"""
    
    def __init__(self):
        self.user_ratings = {}  # 用户评分数据
        self.movie_means = {}   # 电影平均分
        self.user_means = {}    # 用户平均分
        self.movie_users = {}   # 每部电影的用户列表
        
    def fit(self, user_movie_ratings: Dict[int, Dict[int, float]]) -> None:
        """训练推荐模型"""
        self.user_ratings = user_movie_ratings
        
        # 计算每部电影的平均分
        for user_id, ratings in user_movie_ratings.items():
            for movie_id, rating in ratings.items():
                if movie_id not in self.movie_means:
                    self.movie_means[movie_id] = []
                self.movie_means[movie_id].append(rating)
                
                if movie_id not in self.movie_users:
                    self.movie_users[movie_id] = set()
                self.movie_users[movie_id].add(user_id)
        
        self.movie_means = {
            movie_id: np.mean(ratings)
            for movie_id, ratings in self.movie_means.items()
        }
        
        # 计算每个用户的平均分
        self.user_means = {
            user_id: np.mean(list(ratings.values()))
            for user_id, ratings in user_movie_ratings.items()
        }
    
    def _calculate_similarity(self, user1: int, user2: int) -> float:
        """计算两个用户之间的相似度"""
        # 获取两个用户共同评分的电影
        user1_movies = set(self.user_ratings[user1].keys())
        user2_movies = set(self.user_ratings[user2].keys())
        common_movies = user1_movies.intersection(user2_movies)
        
        if not common_movies:
            return 0.0
        
        # 计算皮尔逊相关系数
        user1_ratings = np.array([self.user_ratings[user1][m] for m in common_movies])
        user2_ratings = np.array([self.user_ratings[user2][m] for m in common_movies])
        
        user1_mean = self.user_means[user1]
        user2_mean = self.user_means[user2]
        
        numerator = np.sum((user1_ratings - user1_mean) * (user2_ratings - user2_mean))
        denominator = np.sqrt(np.sum((user1_ratings - user1_mean)**2) * 
                            np.sum((user2_ratings - user2_mean)**2))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    def predict(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """生成推荐"""
        if user_id not in self.user_ratings:
            # 对于新用户，返回评分最高的电影
            sorted_movies = sorted(
                self.movie_means.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [movie_id for movie_id, _ in sorted_movies[:n_recommendations]]
        
        # 计算用户相似度
        similarities = []
        for other_user in self.user_ratings:
            if other_user != user_id:
                similarity = self._calculate_similarity(user_id, other_user)
                similarities.append((other_user, similarity))
        
        # 获取最相似的用户
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = similarities[:5]  # 取top5相似用户
        
        # 计算预测评分
        user_rated_movies = set(self.user_ratings[user_id].keys())
        predictions = []
        
        for movie_id in self.movie_means:
            if movie_id in user_rated_movies:
                continue
                
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user, similarity in similar_users:
                if movie_id in self.user_ratings[similar_user]:
                    rating = self.user_ratings[similar_user][movie_id]
                    weighted_sum += similarity * (rating - self.user_means[similar_user])
                    similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predicted_rating = self.user_means[user_id] + weighted_sum / similarity_sum
                predictions.append((movie_id, predicted_rating))
        
        # 返回预测评分最高的电影
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in predictions[:n_recommendations]]
    
    def save(self, path: str) -> None:
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'user_ratings': self.user_ratings,
                'movie_means': self.movie_means,
                'user_means': self.user_means,
                'movie_users': self.movie_users
            }, f)
    
    def load(self, path: str) -> None:
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.user_ratings = data['user_ratings']
            self.movie_means = data['movie_means']
            self.user_means = data['user_means']
            self.movie_users = data['movie_users']