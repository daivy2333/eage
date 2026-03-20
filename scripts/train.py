#!/usr/bin/env python
"""
训练脚本
训练 ResNet-18 模型在 CIFAR-10 数据集上
"""
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np

from configs import get_config, update_config
from models import resnet18
from data import get_cifar10_dataloaders
from training import Trainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--gradient-clip', type=float, default=5.0, help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval')
    parser.add_argument('--eval-interval', type=int, default=1, help='Evaluation interval')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 更新配置
    config = get_config()
    config.training.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.momentum = args.momentum
    config.training.warmup_epochs = args.warmup_epochs
    config.training.label_smoothing = args.label_smoothing
    config.training.gradient_clip = args.gradient_clip
    config.seed = args.seed
    config.device = args.device
    config.data.num_workers = args.num_workers
    config.training.save_interval = args.save_interval
    config.training.eval_interval = args.eval_interval
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = 'cpu'
    
    print("=" * 60)
    print("Training ResNet-18 on CIFAR-10")
    print("=" * 60)
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Weight decay: {config.training.weight_decay}")
    print(f"Warmup epochs: {config.training.warmup_epochs}")
    print(f"Label smoothing: {config.training.label_smoothing}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    # 创建数据加载器
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_dataloaders(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 创建模型
    print("\nCreating ResNet-18 model...")
    model = resnet18(num_classes=config.data.num_classes)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=config.device,
    )
    
    # 开始训练
    print("\nStarting training...")
    history = trainer.train()
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {history['best_acc']:.2f}%")
    print(f"Total training time: {history['total_time'] / 60:.2f} minutes")


if __name__ == '__main__':
    main()