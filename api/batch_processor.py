from typing import List, Dict, Any
import numpy as np
from collections import defaultdict
import threading
import time
import logging
from queue import Queue

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, 
                 batch_size: int = 32,
                 max_latency: float = 0.1,  # 最大延迟100ms
                 timeout: float = 0.5):     # 批处理超时500ms
        self.batch_size = batch_size
        self.max_latency = max_latency
        self.timeout = timeout
        
        self.request_queue = Queue()
        self.results = {}
        self.lock = threading.Lock()
        
        # 启动批处理线程
        self.processing_thread = threading.Thread(
            target=self._process_batch_loop,
            daemon=True
        )
        self.processing_thread.start()
        
    def add_request(self, request_id: str, user_id: int) -> None:
        """添加新的预测请求到队列"""
        self.request_queue.put((request_id, user_id))
        
    def get_result(self, request_id: str, timeout: float = None) -> List[int]:
        """获取预测结果"""
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.results:
                    return self.results.pop(request_id)
            time.sleep(0.01)
        raise TimeoutError("Request timeout")
        
    def _process_batch_loop(self) -> None:
        """批处理主循环"""
        while True:
            batch = self._collect_batch()
            if batch:
                try:
                    self._process_batch(batch)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    
    def _collect_batch(self) -> List[tuple]:
        """收集一批请求"""
        batch = []
        start_time = time.time()
        
        while len(batch) < self.batch_size and \
              time.time() - start_time < self.timeout:
            try:
                # 等待新请求，但不要超过剩余时间
                remaining_time = self.timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                    
                request = self.request_queue.get(timeout=remaining_time)
                batch.append(request)
                
            except Exception:
                break
                
        return batch
        
    def _process_batch(self, batch: List[tuple]) -> None:
        """处理一批请求"""
        if not batch:
            return
            
        # 提取用户ID
        request_ids, user_ids = zip(*batch)
        
        try:
            # 这里应该调用实际的模型进行批量预测
            # 示例中返回随机推荐
            recommendations = {
                request_id: list(np.random.randint(1, 1000, 5))
                for request_id in request_ids
            }
            
            # 保存结果
            with self.lock:
                self.results.update(recommendations)
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # 返回空推荐
            with self.lock:
                self.results.update({
                    request_id: []
                    for request_id in request_ids
                })