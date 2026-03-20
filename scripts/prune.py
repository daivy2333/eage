#!/usr/bin/env python
"""
剪枝脚本
对训练好的模型进行通道剪枝
"""
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs import get_config
from models import resnet18
from pruning import ChannelPruner, count_parameters


def main():
    parser = argparse.ArgumentParser(description='Prune ResNet-18 model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--pruning-ratio', type=float, default=0.5, help='Pruning ratio')
    parser.add_argument('--method', type=str, default='l1', choices=['l1', 'l2', 'geometric_median'], help='Pruning method')
    parser.add_argument('--min-channels', type=int, default=16, help='Minimum channels to keep')
    parser.add_argument('--output-path', type=str, default='weights/pruned_model.pth', help='Output path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    config = get_config()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Pruning ResNet-18 Model")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Pruning ratio: {args.pruning_ratio}")
    print(f"Pruning method: {args.method}")
    print(f"Min channels: {args.min_channels}")
    print("=" * 60)
    
    # 加载模型
    print("\nLoading model...")
    model = resnet18(num_classes=config.data.num_classes)
    
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    # 计算原始参数量
    original_params = count_parameters(model)
    print(f"Original model parameters: {original_params:,}")
    
    # 创建剪枝器
    print("\nCreating pruner...")
    pruner = ChannelPruner(
        model=model,
        pruning_ratio=args.pruning_ratio,
        method=args.method,
        min_channels=args.min_channels,
        skip_layers=['conv1', 'fc'],
        device=args.device
    )
    
    # 执行剪枝
    print("\nPruning model...")
    pruned_model = pruner.prune_model()
    
    # 计算剪枝后参数量
    pruned_params = count_parameters(pruned_model)
    print(f"Pruned model parameters: {pruned_params:,}")
    print(f"Parameter reduction: {(1 - pruned_params / original_params) * 100:.2f}%")
    
    # 打印剪枝摘要
    print("\n" + pruner.get_pruning_summary())
    
    # 保存剪枝模型
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({
        'model_state_dict': pruned_model.state_dict(),
        'pruning_info': pruner.pruning_info,
        'original_params': original_params,
        'pruned_params': pruned_params,
    }, args.output_path)
    
    print(f"\nPruned model saved to {args.output_path}")
    
    # 保存剪枝信息
    info_path = args.output_path.replace('.pth', '_info.json')
    pruner.save_pruning_info(info_path)
    print(f"Pruning info saved to {info_path}")


if __name__ == '__main__':
    main()