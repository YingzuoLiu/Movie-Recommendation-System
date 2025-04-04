import aiohttp
import asyncio
import time
import numpy as np
from typing import List, Dict
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    async def send_request(self, user_id: int) -> Dict:
        """发送单个推荐请求"""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/recommendations/",
                    json={"user_id": int(user_id), "num_recommendations": 5},
                    timeout=30  # 增加超时时间
                ) as response:
                    latency = time.time() - start_time
                    logger.debug(f"Request completed in {latency:.3f} seconds")
                    
                    data = await response.json() if response.status == 200 else None
                    
                    return {
                        "user_id": user_id,
                        "latency": latency,
                        "status": response.status,
                        "success": response.status == 200,
                        "data": data
                    }
                    
        except asyncio.TimeoutError:
            latency = time.time() - start_time
            logger.error(f"Request timeout for user {user_id}")
            return {
                "user_id": user_id,
                "latency": latency,
                "status": 408,
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Request error for user {user_id}: {str(e)}")
            return {
                "user_id": user_id,
                "latency": latency,
                "status": 500,
                "success": False,
                "error": str(e)
            }
                
    async def load_test(self, 
                       num_users: int = 100,
                       requests_per_second: int = 10,
                       duration: int = 30) -> None:
        """执行负载测试"""
        logger.info(f"Starting load test with {requests_per_second} RPS for {duration} seconds")
        
        user_ids = list(range(1, num_users + 1))
        total_requests = requests_per_second * duration
        
        # 创建请求队列
        requests = []
        for i in range(total_requests):
            user_id = np.random.choice(user_ids)
            requests.append(self.send_request(user_id))
            
        # 分批发送请求
        for i in range(0, len(requests), requests_per_second):
            batch = requests[i:i + requests_per_second]
            start_time = time.time()
            
            # 执行一批请求
            results = await asyncio.gather(*batch)
            self.results.extend(results)
            
            # 计算需要等待的时间
            elapsed = time.time() - start_time
            wait_time = max(0, 1 - elapsed)  # 确保至少等待到1秒
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
            # 打印进度
            progress = (i + len(batch)) / len(requests) * 100
            logger.info(f"Progress: {progress:.1f}%")
            
    def generate_report(self, output_file: str = "performance_report.html") -> None:
        """生成性能报告"""
        if not self.results:
            logger.error("No test results available")
            return
            
        df = pd.DataFrame(self.results)
        
        # 计算指标
        total_requests = len(df)
        successful_requests = df['success'].sum()
        success_rate = successful_requests / total_requests * 100
        avg_latency = df['latency'].mean() * 1000  # 转换为毫秒
        p95_latency = df['latency'].quantile(0.95) * 1000
        p99_latency = df['latency'].quantile(0.99) * 1000
        
        logger.info(f"""
        Performance Summary:
        - Total Requests: {total_requests}
        - Success Rate: {success_rate:.2f}%
        - Average Latency: {avg_latency:.2f}ms
        - 95th Percentile Latency: {p95_latency:.2f}ms
        - 99th Percentile Latency: {p99_latency:.2f}ms
        """)
        
        # 生成图表
        plt.figure(figsize=(10, 6))
        df['latency'].hist(bins=50)
        plt.title('Latency Distribution')
        plt.xlabel('Latency (s)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig('latency_distribution.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        df['success'].rolling(window=100, min_periods=1).mean().plot()
        plt.title('Success Rate Over Time')
        plt.xlabel('Request Number')
        plt.ylabel('Success Rate (100 request window)')
        plt.grid(True)
        plt.savefig('success_rate.png')
        plt.close()
        
        # 创建报告
        report = f"""
        <html>
        <head>
            <title>Performance Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Performance Test Report</h1>
            
            <h2>Summary</h2>
            <div class="metric">Total Requests: {total_requests}</div>
            <div class="metric">Success Rate: {success_rate:.2f}%</div>
            <div class="metric">Average Latency: {avg_latency:.2f}ms</div>
            <div class="metric">95th Percentile Latency: {p95_latency:.2f}ms</div>
            <div class="metric">99th Percentile Latency: {p99_latency:.2f}ms</div>
            
            <h2>Latency Distribution</h2>
            <div class="chart">
                <img src="latency_distribution.png" alt="Latency Distribution">
            </div>
            
            <h2>Success Rate Over Time</h2>
            <div class="chart">
                <img src="success_rate.png" alt="Success Rate">
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(report)

async def main():
    # 创建测试实例
    tester = PerformanceTester()
    
    try:
        # 执行负载测试
        logger.info("Starting load test...")
        await tester.load_test(
            num_users=100,
            requests_per_second=10,
            duration=30
        )
        
        # 生成报告
        logger.info("Generating performance report...")
        tester.generate_report()
        logger.info("Performance test completed. Check performance_report.html for results.")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())