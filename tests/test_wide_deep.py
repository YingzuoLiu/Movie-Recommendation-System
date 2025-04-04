import torch
from models.wide_deep import WideAndDeepRecommender

def test_wide_deep():
    # 创建测试数据
    user_ratings = {
        1: {1: 5.0, 2: 3.0, 3: 4.0, 4: 2.0},
        2: {1: 3.0, 2: 1.0, 3: 2.0, 4: 3.0},
        3: {1: 4.0, 2: 3.0, 3: 4.0, 5: 5.0},
        4: {2: 5.0, 3: 4.0, 4: 5.0, 5: 4.0},
    }
    
    # 创建模型
    model = WideAndDeepRecommender(
        embedding_dim=16,
        hidden_layers=[32, 16],
        n_epochs=2  # 测试时使用较少的训练轮数
    )
    
    # 训练模型
    print("\nTraining Wide & Deep model...")
    model.fit(user_ratings)
    
    # 测试推荐
    recommendations = model.predict(user_id=1, n_recommendations=2)
    print(f"\nRecommendations for user 1: {recommendations}")
    
    # 测试模型保存和加载
    model.save("test_wd_model.pt")
    
    new_model = WideAndDeepRecommender()
    new_model.load("test_wd_model.pt")
    new_recommendations = new_model.predict(user_id=1, n_recommendations=2)
    
    print(f"Recommendations after model reload: {new_recommendations}")
    
    # 验证推荐的合理性
    assert len(recommendations) == 2
    assert all(isinstance(r, int) for r in recommendations)
    assert len(set(recommendations)) == len(recommendations)  # 确保没有重复
    
    # 清理测试文件
    import os
    if os.path.exists("test_wd_model.pt"):
        os.remove("test_wd_model.pt")

if __name__ == "__main__":
    test_wide_deep()