import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import pickle
from models.base_recommender import BaseRecommender

class MovieDataset(Dataset):
    """电影推荐数据集"""
    def __init__(self, user_movie_ratings: Dict[int, Dict[int, float]], 
                 user_mapping: Dict[int, int],
                 item_mapping: Dict[int, int]):
        self.ratings = []
        self.users = []
        self.items = []
        
        # 构建训练数据
        for user_id, ratings in user_movie_ratings.items():
            user_idx = user_mapping[user_id]
            for item_id, rating in ratings.items():
                if item_id in item_mapping:  # 确保物品在映射中
                    item_idx = item_mapping[item_id]
                    self.users.append(user_idx)
                    self.items.append(item_idx)
                    self.ratings.append(rating)
        
        self.users = torch.LongTensor(self.users)
        self.items = torch.LongTensor(self.items)
        self.ratings = torch.FloatTensor(self.ratings)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return {
            'user_id': self.users[idx],
            'item_id': self.items[idx],
            'rating': self.ratings[idx]
        }

class WideAndDeepModel(nn.Module):
    """Wide & Deep 神经网络模型"""
    def __init__(self, n_users: int, n_items: int, 
                 embedding_dim: int = 32, hidden_layers: List[int] = [64, 32]):
        super().__init__()
        
        # Wide部分：直接特征交叉
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Deep部分：多层神经网络
        deep_layers = []
        input_dim = embedding_dim * 2  # 用户和物品嵌入的拼接
        
        for hidden_dim in hidden_layers:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
            
        self.deep_layers = nn.Sequential(*deep_layers)
        
        # Wide和Deep的组合层
        self.final_layer = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
        
    def forward(self, user_ids, item_ids):
        # Wide部分
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        wide_feature = user_emb * item_emb  # 元素级别的乘法
        
        # Deep部分
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        deep_feature = self.deep_layers(deep_input)
        
        # 组合Wide和Deep
        combined = torch.cat([deep_feature, wide_feature], dim=1)
        output = self.final_layer(combined)
        
        return output.squeeze()

class WideAndDeepRecommender(BaseRecommender):
    """Wide & Deep推荐系统"""
    
    def __init__(self, embedding_dim: int = 32, 
                 hidden_layers: List[int] = [64, 32],
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 n_epochs: int = 10):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        
    def _create_mappings(self, ratings: Dict[int, Dict[int, float]]) -> None:
        """创建用户和物品的ID映射"""
        unique_users = set()
        unique_items = set()
        
        for user_id, user_ratings in ratings.items():
            unique_users.add(user_id)
            for item_id in user_ratings:
                unique_items.add(item_id)
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(unique_users))}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(sorted(unique_items))}
        
        print(f"Created mappings for {len(self.user_mapping)} users and {len(self.item_mapping)} items")
        
    def fit(self, user_movie_ratings: Dict[int, Dict[int, float]]) -> None:
        """训练模型"""
        print("Creating mappings...")
        self._create_mappings(user_movie_ratings)
        
        print("Creating dataset...")
        dataset = MovieDataset(
            user_movie_ratings,
            self.user_mapping,
            self.item_mapping
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        print("Initializing model...")
        self.model = WideAndDeepModel(
            n_users=len(self.user_mapping),
            n_items=len(self.item_mapping),
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        print("Starting training...")
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch in dataloader:
                user_ids = batch['user_id'].to(self.device)
                item_ids = batch['item_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
            
    def predict(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """生成推荐"""
        if not self.model or user_id not in self.user_mapping:
            # 处理新用户
            return list(self.item_mapping.keys())[:n_recommendations]
            
        self.model.eval()
        user_idx = self.user_mapping[user_id]
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        
        # 计算所有物品的预测评分
        predictions = []
        with torch.no_grad():
            for item_id, item_idx in self.item_mapping.items():
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                prediction = self.model(user_tensor, item_tensor)
                predictions.append((item_id, prediction.item()))
                
        # 排序并返回top-N推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in predictions[:n_recommendations]]
        
    def save(self, path: str) -> None:
        """保存模型"""
        if self.model:
            model_data = {
                'state_dict': self.model.state_dict(),
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'embedding_dim': self.embedding_dim,
                'hidden_layers': self.hidden_layers
            }
            torch.save(model_data, path)
        
    def load(self, path: str) -> None:
        """加载模型"""
        model_data = torch.load(path)
        self.user_mapping = model_data['user_mapping']
        self.item_mapping = model_data['item_mapping']
        self.embedding_dim = model_data['embedding_dim']
        self.hidden_layers = model_data['hidden_layers']
        
        self.model = WideAndDeepModel(
            n_users=len(self.user_mapping),
            n_items=len(self.item_mapping),
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers
        ).to(self.device)
        
        self.model.load_state_dict(model_data['state_dict'])