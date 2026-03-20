"""
通道重要性评估模块
用于评估卷积层中各通道的重要性
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import OrderedDict


def compute_l1_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    计算 L1 范数重要性
    
    Args:
        weight: 卷积核权重 (out_channels, in_channels, kH, kW)
    
    Returns:
        各输出通道的重要性 (out_channels,)
    """
    # 对输入通道和空间维度求和
    importance = weight.abs().sum(dim=(1, 2, 3))
    return importance


def compute_l2_importance(weight: torch.Tensor) -> torch.Tensor:
    """
    计算 L2 范数重要性
    
    Args:
        weight: 卷积核权重 (out_channels, in_channels, kH, kW)
    
    Returns:
        各输出通道的重要性 (out_channels,)
    """
    importance = (weight ** 2).sum(dim=(1, 2, 3)).sqrt()
    return importance


def compute_geometric_median_importance(
    weight: torch.Tensor,
    num_iterations: int = 10
) -> torch.Tensor:
    """
    计算几何中位数重要性
    
    Args:
        weight: 卷积核权重 (out_channels, in_channels, kH, kW)
        num_iterations: 迭代次数
    
    Returns:
        各输出通道的重要性 (out_channels,)
    """
    # 展平权重
    weight_flat = weight.view(weight.size(0), -1)  # (out_channels, -1)
    
    # 计算几何中位数
    median = weight_flat.mean(dim=0, keepdim=True)  # (1, -1)
    
    for _ in range(num_iterations):
        distances = (weight_flat - median).norm(dim=1)
        mask = distances > 1e-6
        if mask.sum() == 0:
            break
        weights = 1.0 / distances[mask]
        median = (weight_flat[mask] * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        median = median.unsqueeze(0)
    
    # 计算到几何中位数的距离作为重要性
    importance = (weight_flat - median).norm(dim=1)
    
    return importance


def compute_bn_importance(bn_weight: torch.Tensor, bn_bias: torch.Tensor) -> torch.Tensor:
    """
    计算 BN 层重要性（基于缩放参数）
    
    Args:
        bn_weight: BN 缩放参数
        bn_bias: BN 偏置参数
    
    Returns:
        各通道的重要性
    """
    # 使用 BN 的 gamma 参数作为重要性
    return bn_weight.abs()


def compute_taylor_importance(
    model: nn.Module,
    layer_name: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    num_batches: int = 10
) -> torch.Tensor:
    """
    计算 Taylor 展开重要性（基于梯度）
    
    Args:
        model: 模型
        layer_name: 层名称
        dataloader: 数据加载器
        device: 设备
        num_batches: 计算批数
    
    Returns:
        各通道的重要性
    """
    model.eval()
    
    # 获取目标层
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found")
    
    importance = None
    
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        model.zero_grad()
        outputs = model(images)
        loss = outputs.mean()  # 简单的损失函数
        
        # 反向传播
        loss.backward()
        
        # 获取权重和梯度
        weight = target_layer.weight.data
        grad = target_layer.weight.grad.data
        
        # Taylor 重要性 = |weight * grad|
        batch_importance = (weight * grad).abs().sum(dim=(1, 2, 3))
        
        if importance is None:
            importance = batch_importance
        else:
            importance += batch_importance
    
    return importance / num_batches


class ChannelImportanceEvaluator:
    """
    通道重要性评估器
    """
    
    def __init__(
        self,
        method: str = 'l1',
        device: str = 'cuda'
    ):
        """
        Args:
            method: 评估方法 ('l1', 'l2', 'geometric_median', 'bn')
            device: 设备
        """
        self.method = method
        self.device = device
        
        self.importance_functions = {
            'l1': compute_l1_importance,
            'l2': compute_l2_importance,
            'geometric_median': compute_geometric_median_importance,
        }
    
    def evaluate_conv_layer(
        self,
        conv_layer: nn.Conv2d,
        bn_layer: Optional[nn.BatchNorm2d] = None
    ) -> torch.Tensor:
        """
        评估卷积层的通道重要性
        
        Args:
            conv_layer: 卷积层
            bn_layer: BN 层（可选）
        
        Returns:
            各通道的重要性
        """
        if self.method == 'bn' and bn_layer is not None:
            return compute_bn_importance(bn_layer.weight.data, bn_layer.bias.data)
        else:
            importance_fn = self.importance_functions.get(self.method, compute_l1_importance)
            return importance_fn(conv_layer.weight.data)
    
    def evaluate_model(
        self,
        model: nn.Module,
        skip_layers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        评估模型中所有卷积层的通道重要性
        
        Args:
            model: 模型
            skip_layers: 跳过的层名称列表
        
        Returns:
            各层的重要性字典
        """
        skip_layers = skip_layers or []
        importance_dict = {}
        
        # 获取所有卷积层和对应的 BN 层
        conv_layers = OrderedDict()
        bn_layers = OrderedDict()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers[name] = module
            elif isinstance(module, nn.BatchNorm2d):
                bn_layers[name] = module
        
        # 评估每个卷积层
        for name, conv_layer in conv_layers.items():
            if name in skip_layers:
                continue
            
            # 查找对应的 BN 层
            bn_layer = None
            bn_name = name.replace('conv', 'bn')
            if bn_name in bn_layers:
                bn_layer = bn_layers[bn_name]
            
            importance = self.evaluate_conv_layer(conv_layer, bn_layer)
            importance_dict[name] = importance
        
        return importance_dict
    
    def get_pruning_mask(
        self,
        importance: torch.Tensor,
        pruning_ratio: float,
        min_channels: int = 16
    ) -> torch.Tensor:
        """
        根据重要性生成剪枝掩码
        
        Args:
            importance: 重要性值
            pruning_ratio: 剪枝比例
            min_channels: 最小保留通道数
        
        Returns:
            剪枝掩码 (True 表示保留)
        """
        num_channels = importance.size(0)
        
        # 确保最小保留通道数不超过总通道数
        min_channels = min(min_channels, num_channels)
        
        # 计算要剪枝的通道数，确保不会剪枝过多
        num_prune = int(num_channels * pruning_ratio)
        num_prune = min(num_prune, num_channels - min_channels)
        num_prune = max(num_prune, 0)  # 确保非负
        
        # 如果不需要剪枝，返回全 True 掩码
        if num_prune == 0:
            return torch.ones_like(importance, dtype=torch.bool)
        
        # 获取阈值
        sorted_importance, _ = importance.sort()
        threshold = sorted_importance[num_prune]
        
        # 生成掩码
        mask = importance > threshold
        
        # 确保至少保留 min_channels 个通道
        if mask.sum() < min_channels:
            _, indices = importance.topk(min_channels)
            mask = torch.zeros_like(importance, dtype=torch.bool)
            mask[indices] = True
        
        return mask
    
    def get_layer_pruning_ratios(
        self,
        model: nn.Module,
        global_pruning_ratio: float,
        skip_layers: Optional[List[str]] = None,
        min_channels: int = 16
    ) -> Dict[str, float]:
        """
        计算各层的剪枝比例
        
        Args:
            model: 模型
            global_pruning_ratio: 全局剪枝比例
            skip_layers: 跳过的层
            min_channels: 最小通道数
        
        Returns:
            各层的剪枝比例
        """
        importance_dict = self.evaluate_model(model, skip_layers)
        
        # 计算全局重要性
        all_importance = torch.cat([imp for imp in importance_dict.values()])
        
        # 计算全局阈值
        sorted_importance, _ = all_importance.sort()
        num_prune = int(len(all_importance) * global_pruning_ratio)
        global_threshold = sorted_importance[num_prune]
        
        # 计算各层剪枝比例
        layer_ratios = {}
        for name, importance in importance_dict.items():
            num_channels = importance.size(0)
            num_prune = (importance <= global_threshold).sum().item()
            num_prune = min(num_prune, num_channels - min_channels)
            
            layer_ratios[name] = num_prune / num_channels
        
        return layer_ratios


if __name__ == "__main__":
    # 测试通道重要性评估
    print("Testing channel importance evaluation...")
    
    # 创建测试卷积层
    conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    
    # 测试 L1 重要性
    evaluator = ChannelImportanceEvaluator(method='l1')
    importance = evaluator.evaluate_conv_layer(conv)
    print(f"L1 importance shape: {importance.shape}")
    print(f"Importance range: [{importance.min():.4f}, {importance.max():.4f}]")
    
    # 测试剪枝掩码
    mask = evaluator.get_pruning_mask(importance, pruning_ratio=0.5, min_channels=16)
    print(f"Mask shape: {mask.shape}")
    print(f"Channels to keep: {mask.sum().item()}")
    
    print("Channel importance evaluation test passed!")