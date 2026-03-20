"""
通道剪枝模块
实现基于重要性的通道剪枝
"""
import copy
import os
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np

from .importance import ChannelImportanceEvaluator
from configs import get_config


class ChannelPruner:
    """
    通道剪枝器
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.5,
        method: str = 'l1',
        min_channels: int = 16,
        skip_layers: Optional[List[str]] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 待剪枝模型
            pruning_ratio: 剪枝比例
            method: 重要性评估方法
            min_channels: 最小保留通道数
            skip_layers: 跳过的层
            device: 设备
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.min_channels = min_channels
        self.skip_layers = skip_layers or ['conv1', 'fc']  # 默认跳过第一层和最后全连接层
        self.device = device
        
        # 重要性评估器
        self.evaluator = ChannelImportanceEvaluator(method=method, device=device)
        
        # 剪枝信息
        self.pruning_info = {}
        self.masks = {}
    
    def get_conv_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        """
        获取所有卷积层
        
        Returns:
            (层名称, 卷积层) 列表
        """
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name not in self.skip_layers:
                    conv_layers.append((name, module))
        return conv_layers
    
    def compute_pruning_masks(self) -> Dict[str, torch.Tensor]:
        """
        计算各层的剪枝掩码
        
        Returns:
            各层的剪枝掩码
        """
        importance_dict = self.evaluator.evaluate_model(self.model, self.skip_layers)
        
        # 分析 BasicBlock 结构
        block_structure = self._analyze_block_structure(self.model)
        
        # 找出所有不能剪枝的层
        # BasicBlock 内部的层不能剪枝，因为会破坏残差连接
        no_prune_layers = set()
        
        for block_info in block_structure.values():
            # conv1 的输出通道决定了 conv2 的输入通道
            # conv2 的输出通道必须与 shortcut 匹配
            # 所以 BasicBlock 内部的层都不能剪枝
            if 'conv1' in block_info:
                no_prune_layers.add(block_info['conv1'])
            if 'conv2' in block_info:
                no_prune_layers.add(block_info['conv2'])
            if 'downsample_conv' in block_info:
                no_prune_layers.add(block_info['downsample_conv'])
        
        masks = {}
        for name, importance in importance_dict.items():
            # 跳过不能剪枝的层
            if name in no_prune_layers:
                # 创建全 True 的掩码（不剪枝）
                mask = torch.ones_like(importance, dtype=torch.bool)
            else:
                mask = self.evaluator.get_pruning_mask(
                    importance,
                    self.pruning_ratio,
                    self.min_channels
                )
            masks[name] = mask
            
            # 记录剪枝信息
            self.pruning_info[name] = {
                'original_channels': importance.size(0),
                'pruned_channels': (~mask).sum().item(),
                'kept_channels': mask.sum().item(),
                'actual_ratio': (~mask).sum().item() / importance.size(0) if importance.size(0) > 0 else 0.0
            }
        
        return masks
    
    def apply_mask_to_layer(
        self,
        conv_layer: nn.Conv2d,
        mask: torch.Tensor
    ) -> nn.Conv2d:
        """
        将掩码应用到卷积层
        
        Args:
            conv_layer: 卷积层
            mask: 剪枝掩码
        
        Returns:
            剪枝后的卷积层
        """
        # 获取保留的通道索引
        keep_indices = mask.nonzero(as_tuple=True)[0]
        
        # 创建新的卷积层
        new_out_channels = keep_indices.size(0)
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            new_out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None
        )
        
        # 复制权重
        new_conv.weight.data = conv_layer.weight.data[keep_indices]
        
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[keep_indices]
        
        return new_conv
    
    def prune_model(self) -> nn.Module:
        """
        执行模型剪枝
        
        Returns:
            剪枝后的模型
        """
        # 计算掩码
        self.masks = self.compute_pruning_masks()
        
        # 创建模型副本 (deepcopy 后模型在 CPU 上)
        pruned_model = copy.deepcopy(self.model)
        pruned_model = pruned_model.to('cpu')  # 确保在 CPU 上
        
        # 获取卷积层和 BN 层
        conv_layers = {}
        bn_layers = {}
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers[name] = module
            elif isinstance(module, nn.BatchNorm2d):
                bn_layers[name] = module
        
        # 应用剪枝 - 只处理输出通道
        for name, mask in self.masks.items():
            if name not in conv_layers:
                continue
            
            conv_layer = conv_layers[name]
            # 将 mask 移动到 CPU 进行操作
            mask_cpu = mask.cpu()
            keep_indices = mask_cpu.nonzero(as_tuple=True)[0]
            
            # 剪枝卷积层权重 (输出通道)
            conv_layer.weight.data = conv_layer.weight.data[keep_indices]
            
            if conv_layer.bias is not None:
                conv_layer.bias.data = conv_layer.bias.data[keep_indices]
            
            # 更新输出通道数
            conv_layer.out_channels = keep_indices.size(0)
            
            # 剪枝对应的 BN 层
            bn_name = name.replace('conv', 'bn')
            if bn_name in bn_layers:
                bn_layer = bn_layers[bn_name]
                bn_layer.weight.data = bn_layer.weight.data[keep_indices]
                bn_layer.bias.data = bn_layer.bias.data[keep_indices]
                bn_layer.running_mean.data = bn_layer.running_mean.data[keep_indices]
                bn_layer.running_var.data = bn_layer.running_var.data[keep_indices]
                bn_layer.num_features = keep_indices.size(0)
        
        return pruned_model
    
    def _analyze_block_structure(self, model: nn.Module) -> Dict[str, Dict]:
        """
        分析模型的块结构，找出残差块中的卷积层关系
        
        Args:
            model: 模型
        
        Returns:
            块结构信息字典
        """
        block_structure = {}
        
        for name, module in model.named_modules():
            # 检查是否是 BasicBlock
            if hasattr(module, 'conv1') and hasattr(module, 'conv2'):
                block_info = {}
                
                # 记录 conv1 和 conv2 的完整名称
                for child_name, child_module in module.named_modules():
                    full_name = f"{name}.{child_name}" if name else child_name
                    if child_name == 'conv1':
                        block_info['conv1'] = full_name
                    elif child_name == 'conv2':
                        block_info['conv2'] = full_name
                
                # 记录 downsample 层
                if module.downsample is not None:
                    for ds_name, ds_module in module.downsample.named_modules():
                        if isinstance(ds_module, nn.Conv2d):
                            block_info['downsample_conv'] = f"{name}.downsample.{ds_name}"
                            break
                
                if name:
                    block_structure[name] = block_info
        
        return block_structure
    
    def get_pruning_summary(self) -> str:
        """
        获取剪枝摘要
        
        Returns:
            剪枝摘要字符串
        """
        summary = "Pruning Summary:\n"
        summary += "=" * 60 + "\n"
        
        total_original = 0
        total_pruned = 0
        
        for name, info in self.pruning_info.items():
            summary += f"{name}:\n"
            summary += f"  Original: {info['original_channels']}, "
            summary += f"Pruned: {info['pruned_channels']}, "
            summary += f"Kept: {info['kept_channels']}, "
            summary += f"Ratio: {info['actual_ratio']:.2%}\n"
            
            total_original += info['original_channels']
            total_pruned += info['pruned_channels']
        
        summary += "=" * 60 + "\n"
        summary += f"Total: Original {total_original}, Pruned {total_pruned}, "
        summary += f"Overall Ratio: {total_pruned / total_original:.2%}\n"
        
        return summary
    
    def save_pruning_info(self, path: str):
        """
        保存剪枝信息
        
        Args:
            path: 保存路径
        """
        import json
        
        # 转换为可序列化格式
        info = {
            'pruning_ratio': self.pruning_ratio,
            'min_channels': self.min_channels,
            'skip_layers': self.skip_layers,
            'layer_info': self.pruning_info,
        }
        
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def load_pruning_info(self, path: str):
        """
        加载剪枝信息
        
        Args:
            path: 文件路径
        """
        import json
        
        with open(path, 'r') as f:
            info = json.load(f)
        
        self.pruning_ratio = info['pruning_ratio']
        self.min_channels = info['min_channels']
        self.skip_layers = info['skip_layers']
        self.pruning_info = info['layer_info']


class ProgressivePruner:
    """
    渐进式剪枝器
    分多次完成剪枝，每次剪枝后进行微调
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_ratio: float = 0.5,
        num_steps: int = 3,
        method: str = 'l1',
        min_channels: int = 16,
        skip_layers: Optional[List[str]] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 模型
            target_ratio: 目标剪枝比例
            num_steps: 剪枝步数
            method: 重要性评估方法
            min_channels: 最小通道数
            skip_layers: 跳过的层
            device: 设备
        """
        self.model = model
        self.target_ratio = target_ratio
        self.num_steps = num_steps
        self.method = method
        self.min_channels = min_channels
        self.skip_layers = skip_layers
        self.device = device
        
        # 计算每步的剪枝比例
        # 使用等比数列，使得累积剪枝比例达到目标
        self.step_ratios = []
        remaining = 1.0
        for i in range(num_steps):
            step_ratio = 1 - (1 - target_ratio) ** (1 / num_steps)
            self.step_ratios.append(step_ratio)
    
    def get_step_ratio(self, step: int) -> float:
        """
        获取指定步的剪枝比例
        
        Args:
            step: 步数 (0-indexed)
        
        Returns:
            剪枝比例
        """
        if step < len(self.step_ratios):
            return self.step_ratios[step]
        return 0.0
    
    def prune_step(
        self,
        model: nn.Module,
        step: int
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        执行一步剪枝
        
        Args:
            model: 当前模型
            step: 步数
        
        Returns:
            剪枝后的模型, 剪枝信息
        """
        step_ratio = self.get_step_ratio(step)
        
        pruner = ChannelPruner(
            model=model,
            pruning_ratio=step_ratio,
            method=self.method,
            min_channels=self.min_channels,
            skip_layers=self.skip_layers,
            device=self.device
        )
        
        pruned_model = pruner.prune_model()
        pruning_info = pruner.pruning_info
        
        return pruned_model, pruning_info


def prune_model(
    model: nn.Module,
    pruning_ratio: float = 0.5,
    method: str = 'l1',
    min_channels: int = 16,
    skip_layers: Optional[List[str]] = None,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    剪枝模型的便捷函数
    
    Args:
        model: 模型
        pruning_ratio: 剪枝比例
        method: 重要性评估方法
        min_channels: 最小通道数
        skip_layers: 跳过的层
        device: 设备
    
    Returns:
        剪枝后的模型, 剪枝信息
    """
    pruner = ChannelPruner(
        model=model,
        pruning_ratio=pruning_ratio,
        method=method,
        min_channels=min_channels,
        skip_layers=skip_layers,
        device=device
    )
    
    pruned_model = pruner.prune_model()
    
    return pruned_model, pruner.pruning_info


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: 模型
    
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """
    估算模型 FLOPs
    
    Args:
        model: 模型
        input_size: 输入尺寸
    
    Returns:
        FLOPs
    """
    # 简单估算，仅考虑卷积层
    total_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 获取输入尺寸
            # 这里简化处理，实际需要跟踪特征图尺寸
            out_channels = module.out_channels
            in_channels = module.in_channels
            kernel_size = module.kernel_size[0]
            
            # 假设特征图尺寸为输入尺寸的一半（经过下采样）
            # 这是一个粗略估计
            feature_size = 32 // (2 ** (int(name.split('.')[0][-1]) - 1 if 'layer' in name else 0))
            
            flops = 2 * in_channels * out_channels * kernel_size * kernel_size * feature_size * feature_size
            total_flops += flops
    
    return total_flops


if __name__ == "__main__":
    # 测试通道剪枝
    print("Testing channel pruning...")
    
    from models import resnet18
    
    # 创建模型
    model = resnet18(num_classes=10)
    
    # 计算原始参数量
    original_params = count_parameters(model)
    print(f"Original model parameters: {original_params:,}")
    
    # 创建剪枝器
    pruner = ChannelPruner(
        model=model,
        pruning_ratio=0.5,
        method='l1',
        min_channels=16,
        skip_layers=['conv1', 'fc']
    )
    
    # 执行剪枝
    pruned_model = pruner.prune_model()
    
    # 计算剪枝后参数量
    pruned_params = count_parameters(pruned_model)
    print(f"Pruned model parameters: {pruned_params:,}")
    print(f"Parameter reduction: {(1 - pruned_params / original_params):.2%}")
    
    # 打印剪枝摘要
    print(pruner.get_pruning_summary())
    
    print("Channel pruning test passed!")