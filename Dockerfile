FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY api/ api/
COPY models/ models/
COPY data/ data/
COPY mlruns/ mlruns/

# 创建必要的目录
RUN mkdir -p /app/data /app/mlruns

# 设置环境变量
ENV PYTHONPATH=/app
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]