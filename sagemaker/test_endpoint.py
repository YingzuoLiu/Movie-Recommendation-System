import boto3
import numpy as np
import time
import json
from botocore.config import Config
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_endpoint(endpoint_name="movie-rec-endpoint"):
    """测试SageMaker端点"""
    # 设置较长的超时时间和重试
    config = Config(
        read_timeout=60,  # 增加到60秒
        connect_timeout=60,
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    
    # 创建SageMaker运行时客户端
    runtime = boto3.client(
        'sagemaker-runtime', 
        region_name='ap-southeast-1',
        config=config
    )
    
    # 创建简单的测试数据
    test_data = {
        "user_id": 1,
        "num_recommendations": 5
    }
    
    logger.info("Starting endpoint test")
    logger.info(f"Test data: {json.dumps(test_data, indent=2)}")
    
    try:
        # 调用端点
        start_time = time.time()
        logger.info("Invoking endpoint...")
        
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(test_data)
        )
        
        # 计算延迟
        latency = (time.time() - start_time) * 1000
        logger.info(f"Request completed in {latency:.2f}ms")
        
        # 处理响应
        response_body = response['Body'].read().decode('utf-8')
        logger.info(f"Raw response: {response_body}")
        
        result = json.loads(response_body)
        logger.info(f"Parsed result: {json.dumps(result, indent=2)}")
        
        print("\nTest Results:")
        print(f"Latency: {latency:.2f}ms")
        print(f"Response: {json.dumps(result, indent=2)}")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        print(f"\nError testing endpoint: {str(e)}")
        
        # 获取端点状态
        try:
            sagemaker = boto3.client('sagemaker', region_name='ap-southeast-1')
            endpoint_desc = sagemaker.describe_endpoint(
                EndpointName=endpoint_name
            )
            status = endpoint_desc['EndpointStatus']
            logger.info(f"Current endpoint status: {status}")
            print(f"\nEndpoint status: {status}")
            
            # 如果端点已失败，输出失败原因
            if status == 'Failed' and 'FailureReason' in endpoint_desc:
                logger.error(f"Failure reason: {endpoint_desc['FailureReason']}")
                print(f"Failure reason: {endpoint_desc['FailureReason']}")
            
            # 输出变体状态
            print("\nProduction variants status:")
            for variant in endpoint_desc['ProductionVariants']:
                variant_info = (
                    f"- Variant: {variant['VariantName']}\n"
                    f"  Instance count: {variant['CurrentInstanceCount']}\n"
                    f"  Status: {variant.get('CurrentStatus', 'Unknown')}"
                )
                print(variant_info)
                logger.info(variant_info)
                
        except Exception as desc_error:
            logger.error(f"Error getting endpoint status: {str(desc_error)}")
            print(f"Error getting endpoint status: {str(desc_error)}")
            
        return False

def clean_up(endpoint_name="movie-rec-endpoint"):
    """清理资源"""
    logger.info("Starting cleanup...")
    sagemaker = boto3.client('sagemaker', region_name='ap-southeast-1')
    
    try:
        # 删除端点
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deleted endpoint: {endpoint_name}")
        
        # 删除端点配置
        sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_name)
        logger.info(f"Deleted endpoint configuration: {endpoint_name}")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        print(f"Error cleaning up: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting endpoint test script")
    
    try:
        sagemaker = boto3.client('sagemaker', region_name='ap-southeast-1')
        endpoint_desc = sagemaker.describe_endpoint(
            EndpointName="movie-rec-endpoint"
        )
        status = endpoint_desc['EndpointStatus']
        logger.info(f"Initial endpoint status: {status}")
        print(f"Endpoint status: {status}")
        
        if status == 'InService':
            print("Endpoint is ready. Starting test...")
            success = test_endpoint()
            
            if success:
                response = input("\nDo you want to clean up the deployed resources? (y/N): ")
                if response.lower() == 'y':
                    clean_up()
                else:
                    print("\nEndpoint will continue running. Remember to clean up resources when no longer needed.")
        else:
            print(f"Endpoint is not ready (status: {status})")
            
    except Exception as e:
        logger.error(f"Script error: {str(e)}")
        print(f"Error checking endpoint status: {str(e)}")