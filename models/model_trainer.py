import mlflow
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from models.collaborative_filter import CollaborativeFilter

class ModelTrainer:
    def __init__(self, experiment_name: str = "movie_recommendations"):
        """
        初始化模型训练器
        
        Args:
            experiment_name: MLflow实验名称
        """
        mlflow.set_experiment(experiment_name)
        self.model = CollaborativeFilter()
        
    def _split_data(self, user_ratings: Dict[int, Dict[int, float]], 
                    test_size: float = 0.2) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        """
        将数据分割为训练集和测试集
        """
        train_data = {}
        test_data = {}
        
        for user_id, ratings in user_ratings.items():
            # 将用户评分转换为列表
            movies = list(ratings.keys())
            rates = list(ratings.values())
            
            # 随机分割
            if len(movies) > 1:  # 确保用户至少有2个评分
                train_movies, test_movies, train_rates, test_rates = train_test_split(
                    movies, rates, test_size=test_size, random_state=42
                )
                
                # 重建字典
                train_data[user_id] = dict(zip(train_movies, train_rates))
                test_data[user_id] = dict(zip(test_movies, test_rates))
            else:
                train_data[user_id] = ratings  # 如果评分太少，全部放入训练集
                
        return train_data, test_data
    
    def _calculate_metrics(self, model: CollaborativeFilter, 
                         test_data: Dict[int, Dict[int, float]]) -> Dict[str, float]:
        """
        计算模型性能指标
        """
        total_precision = 0
        total_recall = 0
        total_users = 0
        
        for user_id, actual_ratings in test_data.items():
            if not actual_ratings:  # 跳过没有测试数据的用户
                continue
                
            # 获取推荐
            recommended_movies = set(model.predict(user_id, n_recommendations=10))
            actual_movies = set(actual_ratings.keys())
            
            # 计算指标
            if recommended_movies:
                precision = len(recommended_movies.intersection(actual_movies)) / len(recommended_movies)
                recall = len(recommended_movies.intersection(actual_movies)) / len(actual_movies)
                
                total_precision += precision
                total_recall += recall
                total_users += 1
        
        # 计算平均值
        avg_precision = total_precision / total_users if total_users > 0 else 0
        avg_recall = total_recall / total_users if total_users > 0 else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": f1_score
        }
    
    def train_and_evaluate(self, user_ratings: Dict[int, Dict[int, float]], 
                          model_params: Dict = None) -> None:
        """
        训练并评估模型，记录到MLflow
        
        Args:
            user_ratings: 用户评分数据
            model_params: 模型参数
        """
        with mlflow.start_run():
            # 记录参数
            if model_params:
                mlflow.log_params(model_params)
            
            # 分割数据
            train_data, test_data = self._split_data(user_ratings)
            
            # 训练模型
            self.model.fit(train_data)
            
            # 评估模型
            metrics = self._calculate_metrics(self.model, test_data)
            
            # 记录指标
            mlflow.log_metrics(metrics)
            
            # 保存模型
            mlflow.sklearn.log_model(self.model, "model")
            
            return metrics
            
    def get_best_model(self) -> CollaborativeFilter:
        """
        获取性能最好的模型
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("movie_recommendations")
        
        if experiment is None:
            return self.model
            
        # 获取所有运行记录
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_score DESC"]
        )
        
        if runs:
            best_run = runs[0]
            # 加载最佳模型
            best_model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
            return best_model
            
        return self.model