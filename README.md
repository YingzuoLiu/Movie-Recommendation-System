# Movie Recommendation System

A production-ready movie recommendation system with MLOps integration and real-time prediction capabilities.

## Features

- Multiple recommendation algorithms:
  - Collaborative Filtering
  - Matrix Factorization
  - Wide & Deep Neural Network
- Real-time prediction with Faiss similarity search
- MLflow experiment tracking and model management
- Redis caching for high performance
- Kubernetes deployment support
- Comprehensive monitoring with Prometheus
- Automated performance testing

## Performance

- Average response time: 22.32ms
- 95th percentile latency: 28.29ms
- Success rate: 100%

## Tech Stack

- FastAPI
- PyTorch
- MLflow
- Redis
- Kubernetes
- Docker
- Prometheus
- Faiss

## Getting Started

### Prerequisites

- Python 3.9+
- Docker
- Kubernetes (optional)
- Redis

### Installation

1. Clone the repository
```bash
git clone github.com/YingzuoLiu/Movie-Recommendation-System
cd movie-rec
```

2. Create virtual environment
```bash
conda create -n movie_rec python=3.9
conda activate movie_rec
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running Locally

1. Start the API
```bash
python api/main.py
```

2. Run performance tests
```bash
python tests/performance_test_report.py
```

### Docker Deployment

```bash
docker-compose up -d
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Project Structure

```
movie_rec/
├── api/                  # FastAPI application
├── models/              # Recommendation models
├── data/                # Data storage
├── mlflow/              # MLflow tracking
├── tests/               # Test files
├── docker/              # Docker configuration
└── k8s/                 # Kubernetes configuration
```

## API Documentation

Access the API documentation at `http://localhost:8000/docs` after starting the server.

## Performance Testing

The system includes automated performance testing that generates a comprehensive report including:
- Request latency distribution
- Success rate over time
- Key performance metrics

