import time
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from api.database import ModelMetrics
import logging

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, db: Session):
        self.db = db
        self.latency_window = []
        self.max_window_size = 1000
        
    def record_prediction_latency(self, latency_ms: float):
        """记录预测延迟"""
        self.latency_window.append(latency_ms)
        if len(self.latency_window) > self.max_window_size:
            self.latency_window.pop(0)
            
    def get_latency_metrics(self) -> Dict[str, float]:
        """获取延迟指标"""
        if not self.latency_window:
            return {
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0
            }
            
        return {
            "avg_latency_ms": np.mean(self.latency_window),
            "p95_latency_ms": np.percentile(self.latency_window, 95),
            "p99_latency_ms": np.percentile(self.latency_window, 99)
        }
        
    def record_model_metrics(self, model_version: str, 
                           metrics: Dict[str, float],
                           parameters: Optional[Dict] = None):
        """记录模型指标"""
        try:
            model_metrics = ModelMetrics(
                model_version=model_version,
                precision=metrics.get("precision", 0),
                recall=metrics.get("recall", 0),
                f1_score=metrics.get("f1_score", 0),
                latency_ms=self.get_latency_metrics()["avg_latency_ms"],
                parameters=parameters
            )
            self.db.add(model_metrics)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error recording model metrics: {e}")
            self.db.rollback()
            
    def check_model_drift(self, threshold: float = 0.1) -> bool:
        """检查模型是否发生drift"""
        try:
            # 获取最近一周的指标
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_metrics = self.db.query(ModelMetrics)\
                .filter(ModelMetrics.timestamp > week_ago)\
                .order_by(ModelMetrics.timestamp.desc())\
                .all()
                
            if len(recent_metrics) < 2:
                return False
                
            # 比较最新和最早的F1分数
            latest_f1 = recent_metrics[0].f1_score
            oldest_f1 = recent_metrics[-1].f1_score
            
            return abs(latest_f1 - oldest_f1) > threshold
            
        except Exception as e:
            logger.error(f"Error checking model drift: {e}")
            return False
            
    def should_retrain_model(self) -> bool:
        """检查是否需要重新训练模型"""
        # 检查模型drift
        if self.check_model_drift():
            return True
            
        # 检查性能下降
        latency_metrics = self.get_latency_metrics()
        if latency_metrics["p95_latency_ms"] > 500:  # 如果95%请求延迟超过500ms
            return True
            
        return False