import numpy as np
import boto3
import os
from datetime import datetime

def create_sample_data():
    """创建示例训练数据"""
    print("Creating sample training data...")
    
    # 创建示例用户-物品评分矩阵
    n_users = 1000
    n_items = 1000
    n_ratings = 10000
    
    # 随机生成一些评分数据
    np.random.seed(42)  # 设置随机种子以保证可重复性
    user_ids = np.random.randint(0, n_users, n_ratings)
    item_ids = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.normal(3.5, 1.0, n_ratings)
    ratings = np.clip(ratings, 1, 5)
    
    # 创建目录
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # 分割训练集和测试集
    train_idx = np.random.choice(n_ratings, int(n_ratings * 0.8), replace=False)
    test_idx = np.setdiff1d(np.arange(n_ratings), train_idx)
    
    # 保存训练集
    train_path = 'data/train/train.npz'
    np.savez(train_path,
             user_ids=user_ids[train_idx],
             item_ids=item_ids[train_idx],
             ratings=ratings[train_idx])
    
    # 保存测试集
    test_path = 'data/test/test.npz'
    np.savez(test_path,
             user_ids=user_ids[test_idx],
             item_ids=item_ids[test_idx],
             ratings=ratings[test_idx])
    
    print("Sample data created successfully")
    return train_path, test_path

def upload_to_s3(bucket_name: str, train_path: str, test_path: str):
    """上传数据到S3"""
    try:
        session = boto3.Session(region_name='ap-southeast-1')
        s3 = session.client('s3')
        
        print(f"Uploading data to bucket: {bucket_name}")
        
        # 上传训练数据
        s3.upload_file(
            train_path,
            bucket_name,
            'data/train/train.npz'
        )
        print("Uploaded training data")
        
        # 上传测试数据
        s3.upload_file(
            test_path,
            bucket_name,
            'data/test/test.npz'
        )
        print("Uploaded test data")
        
        print("Data uploaded successfully")
        
    except Exception as e:
        print(f"Error uploading data: {e}")
        raise

if __name__ == "__main__":
    print("Preparing training data...")
    train_path, test_path = create_sample_data()
    
    bucket = 'sagemaker-ap-southeast-1-204529129889'
    print(f"Uploading to bucket: {bucket}")
    upload_to_s3(bucket, train_path, test_path)