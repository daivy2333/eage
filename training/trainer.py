"""
训练器模块
包含完整的训练循环、验证、模型保存等功能
"""
import os
import time
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import get_config
from .scheduler import get_scheduler
from .losses import get_loss_function


class Trainer:
    """
    模型训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 配置对象
            device: 设备
        """
        self.config = config or get_config()
        self.device = device or self.config.device
        
        # 模型
        self.model = model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 损失函数
        self.criterion = get_loss_function(
            loss_name='cross_entropy',
            label_smoothing=self.config.training.label_smoothing,
        )
        
        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
            nesterov=True,
        )
        
        # 学习率调度器
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_name=self.config.training.lr_scheduler,
            total_epochs=self.config.training.epochs,
            warmup_epochs=self.config.training.warmup_epochs,
            min_lr=self.config.training.min_lr,
        )
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.config.logs_dir,
                f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.learning_rates = []
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个 epoch
        
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        评估模型
        
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.test_loader, desc='Evaluating')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """
        完整训练流程
        
        Returns:
            训练历史
        """
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total epochs: {self.config.training.epochs}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            if (epoch + 1) % self.config.training.eval_interval == 0:
                test_loss, test_acc = self.evaluate()
            else:
                test_loss, test_acc = 0.0, 0.0
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)
            self.learning_rates.append(current_lr)
            
            # TensorBoard 记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 打印
            print(f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% - "
                  f"LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint('best_model.pth', is_best=True)
                print(f"New best accuracy: {self.best_acc:.2f}%")
            
            # 定期保存
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # 训练完成
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Training completed in {total_time / 60:.2f} minutes")
        print(f"Best test accuracy: {self.best_acc:.2f}%")
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.test_losses,
            'test_accs': self.test_accs,
            'learning_rates': self.learning_rates,
            'best_acc': self.best_acc,
            'total_time': total_time,
        }
        
        with open(os.path.join(self.config.logs_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        self.writer.close()
        
        return history
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        保存检查点
        
        Args:
            filename: 文件名
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
        }
        
        path = os.path.join(self.config.weights_dir, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config.weights_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best accuracy: {self.best_acc:.2f}%")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Optional[Any] = None,
    device: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    训练模型的便捷函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        config: 配置对象
        device: 设备
    
    Returns:
        model: 训练后的模型
        history: 训练历史
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    
    history = trainer.train()
    
    return trainer.model, history


if __name__ == "__main__":
    # 测试训练器
    print("Testing trainer...")
    
    from models import resnet18
    from data import get_cifar10_dataloaders
    
    # 创建模型
    model = resnet18(num_classes=10)
    
    # 获取数据加载器
    train_loader, test_loader, _, _ = get_cifar10_dataloaders(
        batch_size=64,
        num_workers=0,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
    )
    
    # 测试一个 epoch
    print("Testing one epoch...")
    train_loss, train_acc = trainer.train_epoch()
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    test_loss, test_acc = trainer.evaluate()
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("Trainer test passed!")