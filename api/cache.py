import redis
import json
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True  # 自动将响应解码为字符串
        )
        self.default_ttl = 3600  # 默认缓存1小时
        
    def get_recommendations(self, user_id: int) -> Optional[List[int]]:
        """获取缓存的推荐结果"""
        try:
            key = f"recommendations:user:{user_id}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting recommendations from cache: {e}")
            return None
            
    def set_recommendations(self, user_id: int, recommendations: List[int], 
                          ttl: int = None) -> bool:
        """缓存推荐结果"""
        try:
            key = f"recommendations:user:{user_id}"
            ttl = ttl or self.default_ttl
            return self.redis_client.setex(
                key,
                ttl,
                json.dumps(recommendations)
            )
        except Exception as e:
            logger.error(f"Error setting recommendations in cache: {e}")
            return False
            
    def get_model_metrics(self, model_version: str) -> Optional[Dict]:
        """获取缓存的模型指标"""
        try:
            key = f"metrics:model:{model_version}"
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting model metrics from cache: {e}")
            return None
            
    def set_model_metrics(self, model_version: str, metrics: Dict,
                         ttl: int = None) -> bool:
        """缓存模型指标"""
        try:
            key = f"metrics:model:{model_version}"
            ttl = ttl or self.default_ttl
            return self.redis_client.setex(
                key,
                ttl,
                json.dumps(metrics)
            )
        except Exception as e:
            logger.error(f"Error setting model metrics in cache: {e}")
            return False
            
    def clear_user_cache(self, user_id: int) -> bool:
        """清除用户的缓存数据"""
        try:
            key = f"recommendations:user:{user_id}"
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error clearing user cache: {e}")
            return False