from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import mlflow
import os
import logging
from prometheus_client import Counter, Histogram, generate_latest

from models.hybrid_recommender import HybridRecommender
from api.database import get_db, init_db
from api.cache import CacheService
from api.monitoring import ModelMonitor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="Movie Recommendation System",
    description="A hybrid movie recommendation API with MLflow integration",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置MLflow
os.environ['MLFLOW_TRACKING_URI'] = 'mlruns'

# 初始化服务组件
recommender = HybridRecommender(
    embedding_dim=32,
    hidden_layers=[64, 32]
)

cache = CacheService(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379))
)

# Prometheus指标
RECOMMENDATIONS_COUNTER = Counter('recommendations_total', 'Total number of recommendations made')
RECOMMENDATION_LATENCY = Histogram('recommendation_latency_seconds', 'Recommendation latency')
CACHE_HIT_COUNTER = Counter('cache_hits_total', 'Total number of cache hits')
CACHE_MISS_COUNTER = Counter('cache_misses_total', 'Total number of cache misses')

# Pydantic模型
class MovieBase(BaseModel):
    title: str
    genres: List[str]

class MovieCreate(MovieBase):
    pass

class Movie(MovieBase):
    id: int
    
    class Config:
        orm_mode = True

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 5
    alpha: float = 0.5  # 混合权重参数

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Movie]
    model_version: str
    latency_ms: float

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化操作"""
    try:
        logger.info("Initializing application...")
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

# API路由
@app.get("/", include_in_schema=True)
def read_root():
    """API根路由，返回API概述和可用端点信息"""
    return {
        "name": "Movie Recommendation System API",
        "version": "1.0.0",
        "description": "A hybrid movie recommendation system with MLOps integration",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "movies": "/movies/{movie_id}",
            "recommendations": "/recommendations/",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "database": "operational",
            "cache": "operational",
            "recommender": "operational"
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    return generate_latest()

@app.get("/movies/{movie_id}", response_model=Movie)
async def get_movie(movie_id: int, db = Depends(get_db)):
    """获取电影详情"""
    try:
        movie = db.query(Movie).filter(Movie.id == movie_id).first()
        if movie is None:
            raise HTTPException(status_code=404, detail="Movie not found")
        return movie
    except Exception as e:
        logger.error(f"Error fetching movie {movie_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/recommendations/", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    db = Depends(get_db),
    monitor: ModelMonitor = Depends(lambda: ModelMonitor(next(get_db())))
):
    """获取混合推荐"""
    start_time = time.time()
    RECOMMENDATIONS_COUNTER.inc()
    
    try:
        # 检查缓存
        cache_key = f"recommendations:{request.user_id}:{request.num_recommendations}:{request.alpha}"
        cached_recommendations = cache.get_recommendations(cache_key)
        
        if cached_recommendations:
            CACHE_HIT_COUNTER.inc()
            latency = time.time() - start_time
            RECOMMENDATION_LATENCY.observe(latency)
            monitor.record_prediction_latency(latency * 1000)
            
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[
                    get_movie(movie_id, db)
                    for movie_id in cached_recommendations
                ],
                model_version="1.0.0",
                latency_ms=latency * 1000
            )
            
        CACHE_MISS_COUNTER.inc()
        
        # 生成推荐
        recommendations = recommender.predict(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations,
            alpha=request.alpha
        )
        
        # 缓存结果
        cache.set_recommendations(cache_key, recommendations)
        
        # 记录性能指标
        latency = time.time() - start_time
        RECOMMENDATION_LATENCY.observe(latency)
        monitor.record_prediction_latency(latency * 1000)
        
        # 检查是否需要重新训练
        if monitor.should_retrain_model():
            background_tasks.add_task(retrain_model, db)
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=[
                get_movie(movie_id, db)
                for movie_id in recommendations
            ],
            model_version="1.0.0",
            latency_ms=latency * 1000
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

async def retrain_model(db):
    """后台重新训练模型"""
    try:
        logger.info("Starting model retraining")
        # 获取最新数据
        ratings = get_latest_ratings(db)
        
        # 训练新模型
        with mlflow.start_run():
            recommender.fit(ratings)
            
            # 记录训练指标
            metrics = {
                "training_time": time.time() - mlflow.active_run().info.start_time,
            }
            mlflow.log_metrics(metrics)
            
            # 保存模型
            mlflow.sklearn.log_model(recommender, "model")
            
        logger.info("Model retraining completed successfully")
            
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")

def get_latest_ratings(db) -> Dict[int, Dict[int, float]]:
    """从数据库获取最新的评分数据"""
    try:
        # TODO: 实现数据库查询逻辑
        pass
    except Exception as e:
        logger.error(f"Error fetching ratings: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting recommendation service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)