import asyncio
import aiohttp
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
from typing import List, Dict
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []

    async def check_service_health(self) -> bool:
        """检查服务健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def send_recommendation_request(self, user_id: int) -> Dict:
        """发送推荐请求"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                params = {
                    "user_id": int(user_id),
                    "num_recommendations": 5
                }
                
                async with session.post(
                    f"{self.base_url}/recommendations/",
                    json=params,
                    timeout=10
                ) as response:
                    duration = time.time() - start_time
                    status = response.status
                    
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "user_id": user_id,
                        "latency": duration * 1000,  # 转为毫秒
                        "status": status,
                        "success": status == 200
                    }
                    
                    if status == 200:
                        logger.debug(f"Successful request for user {user_id}")
                    else:
                        logger.warning(f"Request failed for user {user_id} with status {status}")
                        
                    return result
                    
        except Exception as e:
            logger.error(f"Request error for user {user_id}: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "latency": 0,
                "status": 500,
                "success": False,
                "error": str(e)
            }

    async def run_load_test(self, 
                           num_users: int = 10,
                           requests_per_second: int = 5,
                           duration: int = 30) -> None:
        """运行负载测试"""
        # 首先检查服务是否在运行
        if not await self.check_service_health():
            logger.error("Service is not healthy, aborting test")
            return

        total_requests = requests_per_second * duration
        logger.info(f"Starting load test: {requests_per_second} RPS for {duration}s")
        logger.info(f"Total planned requests: {total_requests}")

        # 生成用户ID列表
        user_ids = list(range(1, num_users + 1))

        # 执行测试
        for i in range(0, total_requests, requests_per_second):
            # 创建一批请求
            batch_requests = []
            for _ in range(requests_per_second):
                user_id = np.random.choice(user_ids)
                batch_requests.append(self.send_recommendation_request(user_id))

            # 执行批次请求
            batch_results = await asyncio.gather(*batch_requests)
            self.results.extend(batch_results)

            # 计算并显示当前批次的成功率
            success_rate = sum(r['success'] for r in batch_results) / len(batch_results) * 100
            logger.info(f"Batch {i//requests_per_second + 1}/{duration}: "
                       f"Success rate {success_rate:.1f}%")

            # 等待下一秒
            await asyncio.sleep(1)

    def generate_report(self) -> None:
        """生成性能测试报告"""
        if not self.results:
            logger.error("No test results to generate report")
            return

        # 转换结果为DataFrame
        df = pd.DataFrame(self.results)
        
        # 计算关键指标
        total_requests = len(df)
        successful_requests = df['success'].sum()
        success_rate = (successful_requests / total_requests) * 100
        avg_latency = df['latency'].mean()
        p95_latency = df['latency'].quantile(0.95)
        p99_latency = df['latency'].quantile(0.99)

        # 生成延迟分布图
        plt.figure(figsize=(10, 6))
        plt.hist(df['latency'].dropna(), bins=50, alpha=0.75)
        plt.title('Response Latency Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig('performance_latency_distribution.png')
        plt.close()

        # 生成成功率时间序列图
        plt.figure(figsize=(10, 6))
        df['success'] = df['success'].astype(float)
        df.set_index(pd.to_datetime(df['timestamp']))['success'].rolling(
            window=min(20, len(df))).mean().plot()
        plt.title('Success Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Success Rate')
        plt.grid(True, alpha=0.3)
        plt.savefig('performance_success_rate.png')
        plt.close()

        # 生成HTML报告
        report = f"""
        <html>
        <head>
            <title>Performance Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metrics {{ 
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric {{ 
                    padding: 20px;
                    background: #f5f5f5;
                    border-radius: 5px;
                }}
                .charts {{ margin: 40px 0; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Performance Test Report</h1>
            <p>Test conducted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Request Statistics</h3>
                    <p>Total Requests: {total_requests}</p>
                    <p>Successful Requests: {int(successful_requests)}</p>
                    <p>Success Rate: {success_rate:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Latency Statistics</h3>
                    <p>Average Latency: {avg_latency:.2f}ms</p>
                    <p>95th Percentile: {p95_latency:.2f}ms</p>
                    <p>99th Percentile: {p99_latency:.2f}ms</p>
                </div>
            </div>
            
            <div class="charts">
                <div class="chart">
                    <h3>Latency Distribution</h3>
                    <img src="performance_latency_distribution.png" alt="Latency Distribution">
                </div>
                <div class="chart">
                    <h3>Success Rate Over Time</h3>
                    <img src="performance_success_rate.png" alt="Success Rate">
                </div>
            </div>
        </body>
        </html>
        """

        # 保存报告
        with open('performance_report.html', 'w') as f:
            f.write(report)

        logger.info(f"Report generated: {total_requests} requests, {success_rate:.1f}% success rate")

async def main():
    try:
        # 创建测试器实例
        tester = PerformanceTester()
        
        # 运行负载测试
        logger.info("Starting performance test...")
        await tester.run_load_test(
            num_users=10,        # 测试用户数
            requests_per_second=5,  # 每秒请求数
            duration=30          # 测试持续时间（秒）
        )
        
        # 生成报告
        logger.info("Generating performance report...")
        tester.generate_report()
        logger.info("Performance test completed. Check performance_report.html for results.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())