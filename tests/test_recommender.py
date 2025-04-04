from models.collaborative_filter import CollaborativeFilter
import os

def test_basic_recommendation():
    # 创建测试数据
    user_ratings = {
        1: {1: 5.0, 2: 3.0, 3: 4.0},
        2: {1: 3.0, 2: 1.0, 4: 5.0},
        3: {1: 4.0, 2: 2.0, 3: 4.0, 4: 1.0},
    }
    
    # 创建并训练模型
    recommender = CollaborativeFilter()
    recommender.fit(user_ratings)
    
    # 测试推荐
    recommendations = recommender.predict(user_id=1, n_recommendations=2)
    print(f"为用户1推荐的电影: {recommendations}")
    
    # 测试模型保存和加载
    recommender.save("test_model.pkl")
    
    new_recommender = CollaborativeFilter()
    new_recommender.load("test_model.pkl")
    new_recommendations = new_recommender.predict(user_id=1, n_recommendations=2)
    
    print(f"加载模型后的推荐: {new_recommendations}")
    assert recommendations == new_recommendations
    
    # 清理测试文件
    if os.path.exists("test_model.pkl"):
        os.remove("test_model.pkl")

if __name__ == "__main__":
    test_basic_recommendation()