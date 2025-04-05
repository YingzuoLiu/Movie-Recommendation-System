import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class WideAndDeepModel(nn.Module):
    """Wide & Deep 神经网络模型"""
    def __init__(self, n_users: int, n_items: int, 
                 embedding_dim: int = 32, hidden_layers: list = [64, 32]):
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

    def predict(self, user_id, n_recommendations=5):
        """为指定用户生成推荐"""
        device = next(self.parameters()).device
        # 确保用户ID是张量
        if not isinstance(user_id, torch.Tensor):
            user_id = torch.tensor([user_id], device=device)
        
        self.eval()
        with torch.no_grad():
            predictions = []
            # 对每个物品生成预测分数
            for item_id in range(1000):  # 假设有1000个物品
                item_tensor = torch.tensor([item_id], device=device)
                pred = self(user_id, item_tensor)
                predictions.append((item_id, pred.item()))
            
            # 排序并返回top-N推荐
            recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in recommendations[:n_recommendations]]

    def model_fn(model_dir):
        """加载模型"""
        print(f"Loading model from {model_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(os.path.join(model_dir, 'model.pth'))
        model.eval()
        print("Model loaded successfully")
        return model.to(device)

    def input_fn(request_body, request_content_type):
        """处理输入数据"""
        print(f"Processing input: {request_body}")
        if request_content_type == 'application/json':
            data = json.loads(request_body)
            return data
        raise ValueError(f'Unsupported content type: {request_content_type}')

    def predict_fn(input_data, model):
        """生成预测"""
        print(f"Generating predictions for input: {input_data}")
        try:
            # 默认返回一些推荐
            return [1, 2, 3, 4, 5]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return []

    def output_fn(prediction, response_content_type):
        """处理输出数据"""
        print(f"Formatting output: {prediction}")
        if response_content_type == 'application/json':
            return json.dumps({"recommendations": prediction})
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # 模型参数
    parser.add_argument('--num-users', type=int, default=1000)
    parser.add_argument('--num-items', type=int, default=1000)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # 加载数据
    train_data = np.load(os.path.join(args.train, 'train.npz'))
    
    # 创建数据集
    user_ids = torch.from_numpy(train_data['user_ids']).long()
    item_ids = torch.from_numpy(train_data['item_ids']).long()
    ratings = torch.from_numpy(train_data['ratings']).float()
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideAndDeepModel(
        n_users=args.num_users,
        n_items=args.num_items,
        embedding_dim=args.embedding_dim
    ).to(device)
    
    # 训练模型
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(user_ids, item_ids, ratings)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True
    )
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_user_ids, batch_item_ids, batch_ratings in dataloader:
            batch_user_ids = batch_user_ids.to(device)
            batch_item_ids = batch_item_ids.to(device)
            batch_ratings = batch_ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_user_ids, batch_item_ids)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")