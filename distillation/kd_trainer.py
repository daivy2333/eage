"""
知识蒸馏训练器模块
实现教师-学生模型的知识蒸馏训练
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
from .losses import DistillationLoss, get_distillation_loss


class KnowledgeDistillationTrainer:
    """
    知识蒸馏训练器
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 配置对象
            device: 设备
        """
        self.config = config or get_config()
        self.device = device or self.config.device
        
        # 教师模型（冻结）
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 学生模型
        self.student_model = student_model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 蒸馏损失
        self.criterion = get_distillation_loss(
            temperature=self.config.distillation.temperature,
            alpha=self.config.distillation.alpha,
            label_smoothing=0.0
        )
        
        # 优化器
        self.optimizer = optim.SGD(
            self.student_model.parameters(),
            lr=self.config.distillation.learning_rate,
            momentum=0.9,
            weight_decay=self.config.distillation.weight_decay,
            nesterov=True,
        )
        
        # 学习率调度器
        from training.scheduler import WarmupCosineAnnealingLR
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5,
            total_epochs=self.config.distillation.epochs,
            min_lr=1e-6,
        )
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.config.logs_dir,
                f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.test_accs = []
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """
        训练一个 epoch
        
        Returns:
            avg_loss: 平均损失
            avg_soft_loss: 平均软标签损失
            avg_hard_loss: 平均硬标签损失
        """
        self.student_model.train()
        
        total_loss = 0.0
        total_soft_loss = 0.0
        total_hard_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 教师模型推理
            with torch.no_grad():
                teacher_logits = self.teacher_model(images)
            
            # 学生模型推理
            self.optimizer.zero_grad()
            student_logits = self.student_model(images)
            
            # 计算蒸馏损失
            loss, soft_loss, hard_loss = self.criterion(
                student_logits, teacher_logits, labels
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_soft_loss += soft_loss.item()
            total_hard_loss += hard_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'soft': total_soft_loss / (batch_idx + 1),
                'hard': total_hard_loss / (batch_idx + 1),
            })
        
        n = len(self.train_loader)
        return total_loss / n, total_soft_loss / n, total_hard_loss / n
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        评估学生模型
        
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.student_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        ce_loss = nn.CrossEntropyLoss()
        
        pbar = tqdm(self.test_loader, desc='Evaluating')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.student_model(images)
            loss = ce_loss(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """
        完整蒸馏训练流程
        
        Returns:
            训练历史
        """
        print(f"Starting knowledge distillation training on {self.device}")
        print(f"Teacher model: {self.teacher_model.__class__.__name__}")
        print(f"Student model: {self.student_model.__class__.__name__}")
        print(f"Temperature: {self.config.distillation.temperature}")
        print(f"Alpha: {self.config.distillation.alpha}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config.distillation.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, soft_loss, hard_loss = self.train_epoch()
            
            # 验证
            test_loss, test_acc = self.evaluate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录
            self.train_losses.append(train_loss)
            self.test_accs.append(test_acc)
            
            # TensorBoard 记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/soft', soft_loss, epoch)
            self.writer.add_scalar('Loss/hard', hard_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 打印
            print(f"Epoch {epoch + 1}/{self.config.distillation.epochs} - "
                  f"Train Loss: {train_loss:.4f} (soft: {soft_loss:.4f}, hard: {hard_loss:.4f}) - "
                  f"Test Acc: {test_acc:.2f}% - LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint('best_student_model.pth', is_best=True)
                print(f"New best accuracy: {self.best_acc:.2f}%")
        
        # 训练完成
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Distillation training completed in {total_time / 60:.2f} minutes")
        print(f"Best test accuracy: {self.best_acc:.2f}%")
        
        # 保存最终模型
        self.save_checkpoint('final_student_model.pth')
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'test_accs': self.test_accs,
            'best_acc': self.best_acc,
            'total_time': total_time,
        }
        
        with open(os.path.join(self.config.logs_dir, 'distillation_history.json'), 'w') as f:
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
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
        }
        
        path = os.path.join(self.config.weights_dir, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config.weights_dir, 'best_student_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.student_model.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best accuracy: {self.best_acc:.2f}%")


def distill_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Optional[Any] = None,
    device: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    知识蒸馏训练的便捷函数
    
    Args:
        teacher_model: 教师模型
        student_model: 学生模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        config: 配置对象
        device: 设备
    
    Returns:
        student_model: 训练后的学生模型
        history: 训练历史
    """
    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    
    history = trainer.train()
    
    return trainer.student_model, history


if __name__ == "__main__":
    # 测试知识蒸馏训练器
    print("Testing knowledge distillation trainer...")
    
    from models import resnet18
    from data import get_cifar10_dataloaders
    
    # 创建教师模型（假设已训练）
    teacher_model = resnet18(num_classes=10)
    
    # 创建学生模型（剪枝后）
    student_model = resnet18(num_classes=10)
    
    # 获取数据加载器
    train_loader, test_loader, _, _ = get_cifar10_dataloaders(
        batch_size=64,
        num_workers=0,
    )
    
    # 创建蒸馏训练器
    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        test_loader=test_loader,
    )
    
    # 测试一个 epoch
    print("Testing one epoch...")
    train_loss, soft_loss, hard_loss = trainer.train_epoch()
    print(f"Train Loss: {train_loss:.4f}, Soft: {soft_loss:.4f}, Hard: {hard_loss:.4f}")
    
    test_loss, test_acc = trainer.evaluate()
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("Knowledge distillation trainer test passed!")