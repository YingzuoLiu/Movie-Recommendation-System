#!/bin/bash

# 确保目录存在
mkdir -p data mlruns

# 构建和启动服务
docker-compose build
docker-compose up -d

# 等待服务启动
echo "Waiting for services to start..."
sleep 10

# 检查服务状态
echo "Checking service health..."
curl -f http://localhost:8000/health

echo "Services are running. Access the API at http://localhost:8000/docs"
echo "Access Prometheus at http://localhost:9090"