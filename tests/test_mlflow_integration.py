from models.model_trainer import ModelTrainer
import os

def test_model_training():
    # 创建测试数据
    user_ratings = {
        1: {1: 5.0, 2: 3.0, 3: 4.0, 4: 2.0, 5: 5.0},
        2: {1: 3.0, 2: 1.0, 4: 5.0, 5: 2.0, 6: 4.0},
        3: {1: 4.0, 2: 2.0, 3: 4.0, 4: 1.0, 5: 3.0},
        4: {2: 4.0, 3: 3.0, 4: 5.0, 5: 4.0, 6: 2.0},
    }
    
    # 设置MLflow本地存储
    os.environ['MLFLOW_TRACKING_URI'] = 'mlruns'
    
    # 创建训练器
    trainer = ModelTrainer(experiment_name="test_experiment")
    
    # 训练并评估模型
    metrics = trainer.train_and_evaluate(
        user_ratings,
        model_params={"n_neighbors": 5}
    )
    
    print("\nTraining metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 获取最佳模型
    best_model = trainer.get_best_model()
    
    # 测试推荐
    recommendations = best_model.predict(user_id=1, n_recommendations=3)
    print(f"\nRecommendations for user 1: {recommendations}")
    
    # 验证指标存在且合理
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1

if __name__ == "__main__":
    test_model_training()