"""
推理 Pipeline 模块
支持动态 batch、异步预处理的模块化推理
"""
import os
import time
import threading
import queue
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from configs import get_config


@dataclass
class InferenceResult:
    """推理结果"""
    class_id: int
    class_name: str
    confidence: float
    all_probs: np.ndarray
    inference_time: float


class AsyncPreprocessor:
    """
    异步预处理器
    在后台线程中进行图像预处理
    """
    
    def __init__(
        self,
        transform: Callable,
        queue_size: int = 4,
        num_workers: int = 2,
    ):
        """
        Args:
            transform: 预处理函数
            queue_size: 队列大小
            num_workers: 工作线程数
        """
        self.transform = transform
        self.queue_size = queue_size
        self.num_workers = num_workers
        
        # 输入输出队列
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        # 工作线程
        self.workers = []
        self.running = False
    
    def start(self):
        """启动工作线程"""
        self.running = True
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """停止工作线程"""
        self.running = False
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.workers = []
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                if item is None:
                    break
                
                idx, image = item
                processed = self.transform(image)
                self.output_queue.put((idx, processed))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Preprocessing error: {e}")
    
    def submit(self, idx: int, image: Image.Image):
        """
        提交图像进行预处理
        
        Args:
            idx: 图像索引
            image: PIL 图像
        """
        self.input_queue.put((idx, image))
    
    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[int, torch.Tensor]]:
        """
        获取预处理结果
        
        Args:
            timeout: 超时时间
        
        Returns:
            (索引, 预处理后的张量)
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class InferencePipeline:
    """
    模块化推理 Pipeline
    支持动态 batch 和异步预处理
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = 'cuda',
        batch_size: int = 1,
        async_preprocess: bool = True,
        num_preprocess_workers: int = 2,
    ):
        """
        Args:
            model: 模型
            class_names: 类别名称列表
            device: 设备
            batch_size: 批大小
            async_preprocess: 是否使用异步预处理
            num_preprocess_workers: 预处理工作线程数
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.batch_size = batch_size
        
        # 模型设置为评估模式
        self.model.eval()
        self.model.to(device)
        
        # 预处理
        self.transform = self._get_transform()
        
        # 异步预处理器
        self.async_preprocess = async_preprocess
        if async_preprocess:
            self.preprocessor = AsyncPreprocessor(
                transform=self.transform,
                num_workers=num_preprocess_workers,
            )
        else:
            self.preprocessor = None
        
        # 批处理缓冲区
        self.batch_buffer = []
        self.batch_indices = []
    
    def _get_transform(self) -> transforms.Compose:
        """获取预处理 transform"""
        config = get_config()
        return transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.data.normalize_mean,
                std=config.data.normalize_std
            ),
        ])
    
    def start(self):
        """启动 Pipeline"""
        if self.preprocessor is not None:
            self.preprocessor.start()
    
    def stop(self):
        """停止 Pipeline"""
        if self.preprocessor is not None:
            self.preprocessor.stop()
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: PIL 图像或 numpy 数组
        
        Returns:
            预处理后的张量
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image)
    
    @torch.no_grad()
    def infer_batch(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        批量推理
        
        Args:
            images: 图像张量 (B, C, H, W)
        
        Returns:
            输出概率, 推理时间
        """
        images = images.to(self.device)
        
        start_time = time.time()
        outputs = self.model(images)
        probs = torch.softmax(outputs, dim=1)
        inference_time = time.time() - start_time
        
        return probs, inference_time
    
    def postprocess(
        self,
        probs: torch.Tensor,
        inference_time: float,
    ) -> List[InferenceResult]:
        """
        后处理
        
        Args:
            probs: 概率张量
            inference_time: 推理时间
        
        Returns:
            推理结果列表
        """
        probs = probs.cpu().numpy()
        results = []
        
        for prob in probs:
            class_id = int(np.argmax(prob))
            confidence = float(prob[class_id])
            class_name = self.class_names[class_id]
            
            result = InferenceResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                all_probs=prob,
                inference_time=inference_time / len(probs),
            )
            results.append(result)
        
        return results
    
    def infer_single(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> InferenceResult:
        """
        单张图像推理
        
        Args:
            image: 输入图像
        
        Returns:
            推理结果
        """
        # 预处理
        tensor = self.preprocess(image).unsqueeze(0)
        
        # 推理
        probs, inference_time = self.infer_batch(tensor)
        
        # 后处理
        results = self.postprocess(probs, inference_time)
        
        return results[0]
    
    def infer_batch_async(
        self,
        images: List[Union[Image.Image, np.ndarray]],
    ) -> List[InferenceResult]:
        """
        异步批量推理
        
        Args:
            images: 图像列表
        
        Returns:
            推理结果列表
        """
        if self.preprocessor is None:
            # 同步处理
            tensors = torch.stack([self.preprocess(img) for img in images])
            probs, inference_time = self.infer_batch(tensors)
            return self.postprocess(probs, inference_time)
        
        # 异步预处理
        for idx, image in enumerate(images):
            self.preprocessor.submit(idx, image)
        
        # 收集结果
        tensors = [None] * len(images)
        for _ in range(len(images)):
            result = self.preprocessor.get_result()
            if result is not None:
                idx, tensor = result
                tensors[idx] = tensor
        
        # 批量推理
        tensors = torch.stack(tensors)
        probs, inference_time = self.infer_batch(tensors)
        
        return self.postprocess(probs, inference_time)
    
    def infer_stream(
        self,
        image_generator,
        callback: Optional[Callable] = None,
    ):
        """
        流式推理
        
        Args:
            image_generator: 图像生成器
            callback: 结果回调函数
        """
        batch_images = []
        batch_indices = []
        
        for idx, image in enumerate(image_generator):
            batch_images.append(image)
            batch_indices.append(idx)
            
            if len(batch_images) >= self.batch_size:
                results = self.infer_batch_async(batch_images)
                
                if callback is not None:
                    for result, img_idx in zip(results, batch_indices):
                        callback(img_idx, result)
                
                batch_images = []
                batch_indices = []
        
        # 处理剩余图像
        if batch_images:
            results = self.infer_batch_async(batch_images)
            
            if callback is not None:
                for result, img_idx in zip(results, batch_indices):
                    callback(img_idx, result)


class DynamicBatchInferencePipeline:
    """
    动态 Batch 推理 Pipeline
    根据输入动态调整 batch 大小
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = 'cuda',
        max_batch_size: int = 32,
        batch_timeout: float = 0.1,
    ):
        """
        Args:
            model: 模型
            class_names: 类别名称
            device: 设备
            max_batch_size: 最大 batch 大小
            batch_timeout: batch 等待超时
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        
        # 模型设置
        self.model.eval()
        self.model.to(device)
        
        # 预处理
        self.transform = self._get_transform()
        
        # 请求队列
        self.request_queue = queue.Queue()
        self.result_queues = {}
        self.queue_lock = threading.Lock()
        self.queue_counter = 0
        
        # 工作线程
        self.running = False
        self.worker_thread = None
    
    def _get_transform(self) -> transforms.Compose:
        """获取预处理 transform"""
        config = get_config()
        return transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.data.normalize_mean,
                std=config.data.normalize_std
            ),
        ])
    
    def start(self):
        """启动 Pipeline"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def stop(self):
        """停止 Pipeline"""
        self.running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)
    
    def _worker_loop(self):
        """工作线程循环"""
        batch_data = []
        batch_ids = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # 尝试获取请求
                try:
                    request_id, image = self.request_queue.get(timeout=0.01)
                    batch_data.append(image)
                    batch_ids.append(request_id)
                except queue.Empty:
                    pass
                
                # 检查是否应该处理 batch
                current_time = time.time()
                should_process = (
                    len(batch_data) >= self.max_batch_size or
                    (len(batch_data) > 0 and current_time - last_batch_time >= self.batch_timeout)
                )
                
                if should_process:
                    # 处理 batch
                    self._process_batch(batch_data, batch_ids)
                    batch_data = []
                    batch_ids = []
                    last_batch_time = current_time
                    
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    @torch.no_grad()
    def _process_batch(
        self,
        images: List,
        request_ids: List[int],
    ):
        """处理一个 batch"""
        # 预处理
        tensors = torch.stack([self.transform(img) for img in images])
        tensors = tensors.to(self.device)
        
        # 推理
        outputs = self.model(tensors)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # 返回结果
        for request_id, prob in zip(request_ids, probs):
            with self.queue_lock:
                if request_id in self.result_queues:
                    self.result_queues[request_id].put(prob)
    
    def submit(self, image: Union[Image.Image, np.ndarray]) -> int:
        """
        提交推理请求
        
        Args:
            image: 输入图像
        
        Returns:
            请求 ID
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        with self.queue_lock:
            request_id = self.queue_counter
            self.queue_counter += 1
            self.result_queues[request_id] = queue.Queue(maxsize=1)
        
        self.request_queue.put((request_id, image))
        
        return request_id
    
    def get_result(self, request_id: int, timeout: float = 1.0) -> Optional[InferenceResult]:
        """
        获取推理结果
        
        Args:
            request_id: 请求 ID
            timeout: 超时时间
        
        Returns:
            推理结果
        """
        with self.queue_lock:
            if request_id not in self.result_queues:
                return None
            result_queue = self.result_queues[request_id]
        
        try:
            prob = result_queue.get(timeout=timeout)
            
            class_id = int(np.argmax(prob))
            confidence = float(prob[class_id])
            class_name = self.class_names[class_id]
            
            result = InferenceResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                all_probs=prob,
                inference_time=0.0,
            )
            
            # 清理
            with self.queue_lock:
                del self.result_queues[request_id]
            
            return result
            
        except queue.Empty:
            return None


def create_inference_pipeline(
    model: nn.Module,
    class_names: Optional[List[str]] = None,
    device: str = 'cuda',
    batch_size: int = 1,
    async_preprocess: bool = True,
) -> InferencePipeline:
    """
    创建推理 Pipeline 的便捷函数
    
    Args:
        model: 模型
        class_names: 类别名称
        device: 设备
        batch_size: 批大小
        async_preprocess: 是否异步预处理
    
    Returns:
        推理 Pipeline
    """
    if class_names is None:
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    return InferencePipeline(
        model=model,
        class_names=class_names,
        device=device,
        batch_size=batch_size,
        async_preprocess=async_preprocess,
    )


if __name__ == "__main__":
    # 测试推理 Pipeline
    print("Testing inference pipeline...")
    
    from models import resnet18
    
    # 创建模型
    model = resnet18(num_classes=10)
    model.eval()
    
    # 类别名称
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # 创建 Pipeline
    pipeline = create_inference_pipeline(
        model=model,
        class_names=class_names,
        device='cpu',
        batch_size=4,
        async_preprocess=False,
    )
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # 测试单张推理
    result = pipeline.infer_single(test_image)
    print(f"Class: {result.class_name}, Confidence: {result.confidence:.4f}")
    
    # 测试批量推理
    test_images = [test_image for _ in range(4)]
    results = pipeline.infer_batch_async(test_images)
    print(f"Batch inference: {len(results)} results")
    
    print("Inference pipeline test passed!")