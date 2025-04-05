import os
import json
import torch
import torch.nn as nn

class SimpleRecommender(nn.Module):
    def __init__(self, n_users=1000, n_items=1000, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
    def forward(self, user_id):
        # 简单地返回用户嵌入向量与所有物品的相似度
        user_emb = self.user_embedding(user_id)
        all_items_emb = self.item_embedding.weight
        scores = torch.matmul(user_emb, all_items_emb.t())
        return scores

def model_fn(model_dir):
    """加载或创建模型"""
    print("Loading model...")
    
    # 如果存在保存的模型就加载，否则创建新的
    model_path = os.path.join(model_dir, 'model.pth')
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = SimpleRecommender()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print("Model ready")
    return model

def input_fn(request_body, request_content_type):
    """解析输入数据"""
    print(f"Processing input: {request_body}")
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        user_id = data.get('user_id', 1)
        n_recommendations = data.get('num_recommendations', 5)
        return {'user_id': user_id, 'n_recommendations': n_recommendations}
    raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(input_data, model):
    """生成推荐"""
    print(f"Generating predictions for: {input_data}")
    try:
        user_id = input_data['user_id']
        n_recommendations = input_data['n_recommendations']
        
        # 转换为tensor
        device = next(model.parameters()).device
        user_tensor = torch.tensor([user_id], device=device)
        
        # 获取预测分数
        with torch.no_grad():
            scores = model(user_tensor)
            
        # 获取top-k推荐
        top_k = min(n_recommendations, 1000)
        _, indices = torch.topk(scores[0], k=top_k)
        recommendations = indices.cpu().tolist()
        
        print(f"Generated recommendations: {recommendations}")
        return recommendations
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [1, 2, 3, 4, 5]  # 返回默认推荐

def output_fn(prediction, response_content_type):
    """格式化输出"""
    print(f"Formatting output: {prediction}")
    if response_content_type == 'application/json':
        return json.dumps({"recommendations": prediction})