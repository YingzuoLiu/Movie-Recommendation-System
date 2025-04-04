from models.hybrid_recommender import HybridRecommender
import os
import time
import numpy as np

def test_hybrid_recommender():
    # 创建测试数据
    user_ratings = {
        1: {1: 5.0, 2: 3.0, 3: 4.0, 4: 2.0},
        2: {1: 3.0, 2: 1.0, 3: 2.0, 4: 3.0},
        3: {1: 4.0, 2: 3.0, 3: 4.0, 5: 5.0},
        4: {2: 5.0, 3: 4.0, 4: 5.0, 5: 4.0},
    }
    
    # 创建推荐器
    recommender = HybridRecommender(
        embedding_dim=16,
        hidden_layers=[32, 16],
        index_type="Flat"  # 使用简单的索引类型进行测试
    )
    
    # 训练模型和构建索引
    print("\nTraining hybrid recommender...")
    start_time = time.time()
    recommender.fit(user_ratings)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # 测试推荐性能
    print("\nTesting recommendation performance...")
    latencies = []
    for user_id in range(1, 5):
        start_time = time.time()
        recommendations = recommender.predict(user_id, n_recommendations=2)
        latency = time.time() - start_time
        latencies.append(latency)
        print(f"User {user_id} recommendations: {recommendations}, "
              f"latency: {latency*1000:.2f}ms")
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f"\nAverage latency: {avg_latency*1000:.2f}ms")
    print(f"P95 latency: {p95_latency*1000:.2f}ms")
    
    # 测试不同混合权重的效果
    print("\nTesting different hybrid weights...")
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    test_user_id = 1
    
    for alpha in alphas:
        recommendations = recommender.predict(
            user_id=test_user_id,
            n_recommendations=2,
            alpha=alpha
        )
        print(f"Alpha={alpha:.1f}: {recommendations}")
    
    # 测试冷启动场景
    print("\nTesting cold start scenario...")
    new_user_id = 10  # 未知用户
    cold_start_recommendations = recommender.predict(new_user_id, n_recommendations=2)
    print(f"Cold start recommendations: {cold_start_recommendations}")
    
    # 测试模型持久化
    print("\nTesting model persistence...")
    recommender.save("test_hybrid")
    
    new_recommender = HybridRecommender()
    new_recommender.load("test_hybrid")
    
    # 验证加载后的模型性能
    original_recommendations = recommender.predict(test_user_id, n_recommendations=2)
    loaded_recommendations = new_recommender.predict(test_user_id, n_recommendations=2)
    
    print(f"Original recommendations: {original_recommendations}")
    print(f"Loaded model recommendations: {loaded_recommendations}")
    
    # 清理测试文件
    for suffix in ['_model.pt', '_index.index', '_index.mapping']:
        if os.path.exists(f"test_hybrid{suffix}"):
            os.remove(f"test_hybrid{suffix}")
    
    # 验证结果
    assert len(cold_start_recommendations) == 2, "Cold start recommendations failed"
    assert len(original_recommendations) == len(loaded_recommendations), \
           "Model persistence test failed"
    assert avg_latency < 0.1, f"Average latency ({avg_latency*1000:.2f}ms) exceeds threshold"

if __name__ == "__main__":
    test_hybrid_recommender()