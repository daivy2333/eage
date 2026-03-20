#!/usr/bin/env python
"""
主脚本
完整的模型训练 -> 剪枝 -> 蒸馏 -> 量化 -> 导出流程
"""
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import random
import numpy as np

from configs import get_config
from models import resnet18
from data import get_cifar10_dataloaders, get_calibration_dataloader
from training import Trainer
from pruning import ChannelPruner, count_parameters
from distillation import KnowledgeDistillationTrainer
from quantization import PostTrainingQuantizer, compare_model_size
from export import export_to_onnx, get_onnx_model_info
from utils import compute_metrics, compute_model_size, compute_inference_time


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Complete model optimization pipeline')
    parser.add_argument('--skip-training', action='store_true', help='Skip training')
    parser.add_argument('--skip-pruning', action='store_true', help='Skip pruning')
    parser.add_argument('--skip-distillation', action='store_true', help='Skip distillation')
    parser.add_argument('--skip-quantization', action='store_true', help='Skip quantization')
    parser.add_argument('--skip-export', action='store_true', help='Skip ONNX export')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--pruning-ratio', type=float, default=0.5, help='Pruning ratio')
    parser.add_argument('--distill-epochs', type=int, default=50, help='Distillation epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with fewer epochs')
    args = parser.parse_args()
    
    config = get_config()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 快速测试模式
    if args.quick_test:
        args.epochs = 10
        args.distill_epochs = 5
        print("Quick test mode enabled")
    
    set_seed(config.seed)
    
    print("=" * 70)
    print("  Embedded Object Detection Model Optimization Pipeline")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Training epochs: {args.epochs}")
    print(f"Pruning ratio: {args.pruning_ratio}")
    print(f"Distillation epochs: {args.distill_epochs}")
    print("=" * 70)
    
    # 创建目录
    os.makedirs(config.weights_dir, exist_ok=True)
    os.makedirs(config.export.onnx_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    
    # 获取数据加载器
    print("\n[1/6] Loading CIFAR-10 dataset...")
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_dataloaders(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # 阶段 1: 训练基础模型
    if not args.skip_training:
        print("\n[2/6] Training base ResNet-18 model...")
        config.training.epochs = args.epochs
        
        model = resnet18(num_classes=config.data.num_classes)
        print(f"  Parameters: {count_parameters(model):,}")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=args.device,
        )
        history = trainer.train()
        
        # 评估
        metrics = compute_metrics(trainer.model, test_loader, args.device)
        print(f"  Final Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        
        # 保存模型路径
        original_model_path = os.path.join(config.weights_dir, 'best_model.pth')
    else:
        print("\n[2/6] Skipping training...")
        original_model_path = os.path.join(config.weights_dir, 'best_model.pth')
        if not os.path.exists(original_model_path):
            print("  Warning: No trained model found, creating untrained model")
            model = resnet18(num_classes=config.data.num_classes)
            torch.save({'model_state_dict': model.state_dict()}, original_model_path)
    
    # 加载原始模型
    original_model = resnet18(num_classes=config.data.num_classes)
    checkpoint = torch.load(original_model_path, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        original_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        original_model.load_state_dict(checkpoint)
    original_model = original_model.to(args.device)
    original_model.eval()
    
    # 阶段 2: 通道剪枝
    if not args.skip_pruning:
        print("\n[3/6] Pruning model...")
        pruner = ChannelPruner(
            model=original_model,
            pruning_ratio=args.pruning_ratio,
            method='l1',
            min_channels=16,
            skip_layers=['conv1', 'fc'],
            device=args.device
        )
        pruned_model = pruner.prune_model()
        
        print(f"  Original params: {count_parameters(original_model):,}")
        print(f"  Pruned params: {count_parameters(pruned_model):,}")
        print(f"  Reduction: {(1 - count_parameters(pruned_model) / count_parameters(original_model)) * 100:.1f}%")
        
        # 保存剪枝模型
        pruned_model_path = os.path.join(config.weights_dir, 'pruned_model.pth')
        torch.save({
            'model_state_dict': pruned_model.state_dict(),
            'pruning_info': pruner.pruning_info,
        }, pruned_model_path)
    else:
        print("\n[3/6] Skipping pruning...")
        pruned_model = original_model  # 使用原始模型
    
    # 阶段 3: 知识蒸馏微调
    if not args.skip_distillation and not args.skip_pruning:
        print("\n[4/6] Knowledge distillation fine-tuning...")
        config.distillation.epochs = args.distill_epochs
        
        distill_trainer = KnowledgeDistillationTrainer(
            teacher_model=original_model,
            student_model=pruned_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=args.device,
        )
        distill_history = distill_trainer.train()
        
        # 保存蒸馏后的模型
        distilled_model_path = os.path.join(config.weights_dir, 'distilled_model.pth')
        torch.save({
            'model_state_dict': distill_trainer.student_model.state_dict(),
        }, distilled_model_path)
        
        final_model = distill_trainer.student_model
    else:
        print("\n[4/6] Skipping distillation...")
        final_model = pruned_model
    
    # 阶段 4: INT8 量化
    if not args.skip_quantization:
        print("\n[5/6] INT8 quantization...")
        calib_loader = get_calibration_dataloader(num_samples=500)
        
        # 创建模型副本用于量化
        quantize_model = resnet18(num_classes=config.data.num_classes)
        quantize_model.load_state_dict(final_model.state_dict())
        quantize_model.eval()
        
        quantizer = PostTrainingQuantizer(
            model=quantize_model,
            calibration_loader=calib_loader,
            qconfig='fbgemm',
            device='cpu',
        )
        quantized_model = quantizer.quantize(num_calibration_batches=20)
        
        # 比较大小
        size_info = compare_model_size(final_model, quantized_model)
        print(f"  Original size: {size_info['original_size_mb']:.2f} MB")
        print(f"  Quantized size: {size_info['quantized_size_mb']:.2f} MB")
        print(f"  Compression: {size_info['compression_ratio']:.2f}x")
        
        # 保存量化模型
        quantized_model_path = os.path.join(config.weights_dir, 'quantized_model.pth')
        torch.save(quantized_model.state_dict(), quantized_model_path)
    else:
        print("\n[5/6] Skipping quantization...")
        quantized_model = None
    
    # 阶段 5: ONNX 导出
    if not args.skip_export:
        print("\n[6/6] Exporting to ONNX...")
        
        # 导出原始模型
        original_onnx_path = os.path.join(config.export.onnx_dir, 'resnet18_original.onnx')
        export_to_onnx(
            model=original_model,
            output_path=original_onnx_path,
            input_size=(1, 3, 32, 32),
            dynamic_batch=True,
            optimize=False,
        )
        print(f"  Original model: {original_onnx_path}")
        
        # 导出优化模型
        final_onnx_path = os.path.join(config.export.onnx_dir, 'resnet18_optimized.onnx')
        export_to_onnx(
            model=final_model,
            output_path=final_onnx_path,
            input_size=(1, 3, 32, 32),
            dynamic_batch=True,
            optimize=False,
        )
        print(f"  Optimized model: {final_onnx_path}")
        
        # 获取模型信息
        info = get_onnx_model_info(original_onnx_path)
        if info:
            print(f"  ONNX model size: {info['model_size_mb']:.2f} MB")
    else:
        print("\n[6/6] Skipping ONNX export...")
    
    # 最终报告
    print("\n" + "=" * 70)
    print("  Optimization Complete!")
    print("=" * 70)
    
    # 计算最终指标
    print("\nFinal Model Comparison:")
    print("-" * 70)
    
    # 原始模型
    original_metrics = compute_metrics(original_model, test_loader, args.device)
    original_size = compute_model_size(original_model)
    original_time = compute_inference_time(original_model, device=args.device)
    
    print(f"Original Model:")
    print(f"  Accuracy: {original_metrics['top1_accuracy']:.2f}%")
    print(f"  Size: {original_size['model_size_mb']:.2f} MB")
    print(f"  Inference: {original_time['mean_time_ms']:.2f} ms")
    
    # 优化模型
    final_metrics = compute_metrics(final_model, test_loader, args.device)
    final_size = compute_model_size(final_model)
    final_time = compute_inference_time(final_model, device=args.device)
    
    print(f"\nOptimized Model:")
    print(f"  Accuracy: {final_metrics['top1_accuracy']:.2f}%")
    print(f"  Size: {final_size['model_size_mb']:.2f} MB")
    print(f"  Inference: {final_time['mean_time_ms']:.2f} ms")
    
    # 改进
    size_reduction = (1 - final_size['model_size_mb'] / original_size['model_size_mb']) * 100
    speedup = original_time['mean_time_ms'] / final_time['mean_time_ms']
    accuracy_drop = original_metrics['top1_accuracy'] - final_metrics['top1_accuracy']
    
    print(f"\nImprovements:")
    print(f"  Size Reduction: {size_reduction:.1f}%")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Accuracy Drop: {accuracy_drop:.2f}%")
    
    print("\n" + "=" * 70)
    print("All models saved to:", config.weights_dir)
    print("ONNX models saved to:", config.export.onnx_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()