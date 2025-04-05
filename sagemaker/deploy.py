import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from datetime import datetime
import time

def deploy_model():
    """部署模型到SageMaker"""
    session = boto3.Session(region_name='ap-southeast-1')
    sagemaker_session = sagemaker.Session(boto_session=session)
    
    # 设置角色ARN
    role = f'arn:aws:iam::{session.client("sts").get_caller_identity()["Account"]}:role/sagemaker-execution-role'
    
    # 创建PyTorch模型
    model = PyTorchModel(
        model_data='s3://sagemaker-ap-southeast-1-204529129889/dummy/model.tar.gz',  # 虚拟模型数据路径
        role=role,
        entry_point='inference.py',
        source_dir='sagemaker',
        framework_version='1.9.1',
        py_version='py38'
    )
    
    try:
        # 部署模型
        print("Deploying model...")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name='movie-rec-endpoint'
        )
        print("Model deployed successfully!")
        
        # 等待端点就绪
        print("\nWaiting for endpoint to be ready...")
        sagemaker_client = session.client('sagemaker')
        
        while True:
            response = sagemaker_client.describe_endpoint(
                EndpointName='movie-rec-endpoint'
            )
            status = response['EndpointStatus']
            print(f"Endpoint status: {status}")
            
            if status == 'InService':
                print("Endpoint is ready!")
                break
            elif status in ['Failed', 'OutOfService']:
                print("Endpoint deployment failed!")
                break
                
            time.sleep(30)
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    deploy_model()