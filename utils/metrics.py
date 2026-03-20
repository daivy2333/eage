"""
评估指标模块
包含准确率、混淆矩阵等评估工具
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f'{self.name}: {self.avg:.4f}'


class AccuracyMeter:
    """
    准确率计算器
    """
    
    def __init__(self, topk: Tuple[int, ...] = (1, 5)):
        """
        Args:
            topk: 计算 top-k 准确率
        """
        self.topk = topk
        self.reset()
    
    def reset(self):
        self.correct = defaultdict(int)
        self.total = 0
    
    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        更新准确率
        
        Args:
            output: 模型输出 (B, C)
            target: 目标标签 (B,)
        """
        batch_size = target.size(0)
        
        # 获取 top-k 预测
        _, pred = output.topk(max(self.topk), dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        for k in self.topk:
            self.correct[k] += correct[:k].reshape(-1).float().sum(0).item()
        
        self.total += batch_size
    
    def get_accuracy(self, k: int) -> float:
        """
        获取 top-k 准确率
        
        Args:
            k: top-k
        
        Returns:
            准确率
        """
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct[k] / self.total
    
    def get_all_accuracies(self) -> Dict[int, float]:
        """
        获取所有 top-k 准确率
        
        Returns:
            准确率字典
        """
        return {k: self.get_accuracy(k) for k in self.topk}


class ConfusionMatrix:
    """
    混淆矩阵
    """
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: 类别数量
        """
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        更新混淆矩阵
        
        Args:
            pred: 预测标签
            target: 真实标签
        """
        for t, p in zip(target, pred):
            self.matrix[t, p] += 1
    
    def get_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return self.matrix
    
    def get_per_class_accuracy(self) -> np.ndarray:
        """
        获取每个类别的准确率
        
        Returns:
            各类别准确率
        """
        per_class_correct = np.diag(self.matrix)
        per_class_total = self.matrix.sum(axis=1)
        
        # 避免除零
        per_class_total = np.maximum(per_class_total, 1)
        
        return per_class_correct / per_class_total
    
    def get_precision(self) -> np.ndarray:
        """
        获取每个类别的精确率
        
        Returns:
            各类别精确率
        """
        per_class_correct = np.diag(self.matrix)
        per_class_predicted = self.matrix.sum(axis=0)
        
        per_class_predicted = np.maximum(per_class_predicted, 1)
        
        return per_class_correct / per_class_predicted
    
    def get_recall(self) -> np.ndarray:
        """
        获取每个类别的召回率
        
        Returns:
            各类别召回率
        """
        return self.get_per_class_accuracy()
    
    def get_f1_score(self) -> np.ndarray:
        """
        获取每个类别的 F1 分数
        
        Returns:
            各类别 F1 分数
        """
        precision = self.get_precision()
        recall = self.get_recall()
        
        # 避免除零
        denominator = precision + recall
        denominator = np.maximum(denominator, 1e-6)
        
        return 2 * precision * recall / denominator
    
    def get_overall_accuracy(self) -> float:
        """
        获取总体准确率
        
        Returns:
            总体准确率
        """
        total_correct = np.diag(self.matrix).sum()
        total = self.matrix.sum()
        
        if total == 0:
            return 0.0
        
        return total_correct / total
    
    def reset(self):
        """重置混淆矩阵"""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


def compute_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    num_classes: int = 10,
) -> Dict[str, float]:
    """
    计算模型评估指标
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数量
    
    Returns:
        评估指标字典
    """
    model.eval()
    
    acc_meter = AccuracyMeter(topk=(1, 5))
    conf_matrix = ConfusionMatrix(num_classes)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            acc_meter.update(outputs, labels)
            
            _, pred = outputs.max(1)
            conf_matrix.update(
                pred.cpu().numpy(),
                labels.cpu().numpy()
            )
    
    metrics = {
        'top1_accuracy': acc_meter.get_accuracy(1),
        'top5_accuracy': acc_meter.get_accuracy(5),
        'overall_accuracy': conf_matrix.get_overall_accuracy() * 100,
        'mean_per_class_accuracy': conf_matrix.get_per_class_accuracy().mean() * 100,
        'mean_precision': conf_matrix.get_precision().mean() * 100,
        'mean_recall': conf_matrix.get_recall().mean() * 100,
        'mean_f1': conf_matrix.get_f1_score().mean() * 100,
    }
    
    return metrics


def compute_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    计算模型大小
    
    Args:
        model: 模型
    
    Returns:
        模型大小信息
    """
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小 (假设 float32)
    param_size = total_params * 4 / (1024 * 1024)  # MB
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': param_size,
    }


def compute_inference_time(
    model: torch.nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = 'cuda',
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    计算推理时间
    
    Args:
        model: 模型
        input_size: 输入尺寸
        device: 设备
        num_runs: 运行次数
        warmup_runs: 预热次数
    
    Returns:
        推理时间信息
    """
    model.eval()
    model.to(device)
    
    # 创建输入
    dummy_input = torch.randn(input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # 同步
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time_ms': times.mean() * 1000,
        'std_time_ms': times.std() * 1000,
        'min_time_ms': times.min() * 1000,
        'max_time_ms': times.max() * 1000,
        'fps': 1.0 / times.mean(),
    }


def compare_models(
    original_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """
    比较原始模型和优化模型
    
    Args:
        original_model: 原始模型
        optimized_model: 优化模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        比较结果
    """
    # 计算指标
    original_metrics = compute_metrics(original_model, dataloader, device)
    optimized_metrics = compute_metrics(optimized_model, dataloader, device)
    
    # 计算模型大小
    original_size = compute_model_size(original_model)
    optimized_size = compute_model_size(optimized_model)
    
    # 计算推理时间
    original_time = compute_inference_time(original_model, device=device)
    optimized_time = compute_inference_time(optimized_model, device=device)
    
    # 计算改进
    size_reduction = (1 - optimized_size['model_size_mb'] / original_size['model_size_mb']) * 100
    speedup = original_time['mean_time_ms'] / optimized_time['mean_time_ms']
    accuracy_drop = original_metrics['top1_accuracy'] - optimized_metrics['top1_accuracy']
    
    return {
        'original': {
            **original_metrics,
            **original_size,
            **original_time,
        },
        'optimized': {
            **optimized_metrics,
            **optimized_size,
            **optimized_time,
        },
        'comparison': {
            'size_reduction_percent': size_reduction,
            'speedup': speedup,
            'accuracy_drop': accuracy_drop,
        }
    }


if __name__ == "__main__":
    # 测试评估指标
    print("Testing metrics...")
    
    # 测试 AverageMeter
    meter = AverageMeter('loss')
    meter.update(0.5, 10)
    meter.update(0.4, 10)
    print(f"AverageMeter: {meter}")
    
    # 测试 AccuracyMeter
    acc_meter = AccuracyMeter(topk=(1, 5))
    output = torch.randn(10, 10)
    target = torch.randint(0, 10, (10,))
    acc_meter.update(output, target)
    print(f"Top-1 Accuracy: {acc_meter.get_accuracy(1):.2f}%")
    print(f"Top-5 Accuracy: {acc_meter.get_accuracy(5):.2f}%")
    
    # 测试 ConfusionMatrix
    conf_matrix = ConfusionMatrix(num_classes=10)
    pred = torch.randint(0, 10, (100,)).numpy()
    target = torch.randint(0, 10, (100,)).numpy()
    conf_matrix.update(pred, target)
    print(f"Overall Accuracy: {conf_matrix.get_overall_accuracy():.4f}")
    
    print("Metrics test passed!")