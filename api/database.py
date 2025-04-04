from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Movie(Base):
    __tablename__ = "movies"
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    genres = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    ratings = relationship("Rating", back_populates="movie")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    ratings = relationship("Rating", back_populates="user")

class Rating(Base):
    __tablename__ = "ratings"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    movie_id = Column(Integer, ForeignKey("movies.id"))
    rating = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String, nullable=False)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    latency_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    parameters = Column(JSON)

# 数据库连接
SQLALCHEMY_DATABASE_URL = "sqlite:///./movie_rec.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建数据库表
def init_db():
    Base.metadata.create_all(bind=engine)

# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()