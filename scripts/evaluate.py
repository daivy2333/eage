#!/usr/bin/env python
"""
评估脚本
评估模型性能（准确率、模型大小、推理速度）
"""
import os
import sys
import argparse
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs import get_config
from models import resnet18
from data import get_cifar10_dataloaders
from utils import compute_metrics, compute_model_size, compute_inference_time, compare_models


def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of inference runs')
    parser.add_argument('--output-json', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()
    
    config = get_config()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Evaluating ResNet-18 Model")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
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
    
    # 获取数据加载器
    print("\nLoading dataset...")
    _, test_loader, _, _ = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
    )
    
    # 计算准确率
    print("\nComputing accuracy...")
    metrics = compute_metrics(model, test_loader, args.device, config.data.num_classes)
    
    print(f"\nAccuracy Metrics:")
    print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"  Mean Per-Class Accuracy: {metrics['mean_per_class_accuracy']:.2f}%")
    print(f"  Mean Precision: {metrics['mean_precision']:.2f}%")
    print(f"  Mean Recall: {metrics['mean_recall']:.2f}%")
    print(f"  Mean F1: {metrics['mean_f1']:.2f}%")
    
    # 计算模型大小
    print("\nComputing model size...")
    size_info = compute_model_size(model)
    
    print(f"\nModel Size:")
    print(f"  Total Parameters: {size_info['total_params']:,}")
    print(f"  Trainable Parameters: {size_info['trainable_params']:,}")
    print(f"  Model Size: {size_info['model_size_mb']:.2f} MB")
    
    # 计算推理时间
    print("\nComputing inference time...")
    time_info = compute_inference_time(
        model,
        input_size=(1, 3, 32, 32),
        device=args.device,
        num_runs=args.num_runs,
    )
    
    print(f"\nInference Time:")
    print(f"  Mean Time: {time_info['mean_time_ms']:.2f} ms")
    print(f"  Std Time: {time_info['std_time_ms']:.2f} ms")
    print(f"  Min Time: {time_info['min_time_ms']:.2f} ms")
    print(f"  Max Time: {time_info['max_time_ms']:.2f} ms")
    print(f"  FPS: {time_info['fps']:.2f}")
    
    # 汇总结果
    results = {
        'model_path': args.model_path,
        'device': args.device,
        'accuracy': metrics,
        'model_size': size_info,
        'inference_time': time_info,
    }
    
    # 保存结果
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()