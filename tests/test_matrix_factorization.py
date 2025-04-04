import numpy as np
from models.matrix_factorization import MatrixFactorization

def test_matrix_factorization():
    # 创建测试数据
    user_ratings = {
        1: {1: 5.0, 2: 3.0, 3: 4.0, 4: 2.0},
        2: {1: 3.0, 2: 1.0, 3: 2.0, 4: 3.0},
        3: {1: 4.0, 2: 3.0, 3: 4.0, 5: 5.0},
        4: {2: 5.0, 3: 4.0, 4: 5.0, 5: 4.0},
    }
    
    # 创建并训练模型
    model = MatrixFactorization(n_factors=3)
    model.fit(user_ratings)
    
    # 测试推荐
    recommendations = model.predict(user_id=1, n_recommendations=2)
    print(f"\nRecommendations for user 1: {recommendations}")
    
    # 测试模型保存和加载
    model.save("test_mf_model.pkl")
    
    new_model = MatrixFactorization()
    new_model.load("test_mf_model.pkl")
    new_recommendations = new_model.predict(user_id=1, n_recommendations=2)
    
    print(f"Recommendations after model reload: {new_recommendations}")
    
    # 验证推荐的合理性
    assert len(recommendations) == 2
    assert all(isinstance(r, (int, np.integer)) for r in recommendations)
    assert len(set(recommendations)) == len(recommendations)  # 确保没有重复
    
    # 清理测试文件
    import os
    if os.path.exists("test_mf_model.pkl"):
        os.remove("test_mf_model.pkl")

if __name__ == "__main__":
    test_matrix_factorization()