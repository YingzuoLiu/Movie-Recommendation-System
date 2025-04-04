#!/bin/bash

# 确保minikube正在运行
minikube status || minikube start

# 构建Docker镜像
eval $(minikube docker-env)
docker build -t movie-rec-api:latest .

# 创建命名空间
kubectl create namespace movie-rec

# 应用Kubernetes配置
kubectl apply -f k8s/deployment.yaml -n movie-rec
kubectl apply -f k8s/service.yaml -n movie-rec
kubectl apply -f k8s/redis.yaml -n movie-rec
kubectl apply -f k8s/monitoring.yaml -n movie-rec

# 等待所有Pod就绪
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=movie-rec-api -n movie-rec --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n movie-rec --timeout=300s

# 获取服务URL
echo "Service URL:"
minikube service movie-rec-api -n movie-rec --url

# 显示部署状态
echo "Deployment status:"
kubectl get all -n movie-rec