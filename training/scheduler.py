"""
学习率调度器模块
包含 Warmup + CosineAnnealing 等调度策略
"""
import math
from typing import List, Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    带有 Warmup 的余弦退火学习率调度器
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr_init: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: Warmup 轮数
            total_epochs: 总训练轮数
            warmup_lr_init: Warmup 初始学习率
            min_lr: 最小学习率
            last_epoch: 上一轮 epoch
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine Annealing 阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupMultiStepLR(_LRScheduler):
    """
    带有 Warmup 的多步学习率调度器
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_lr_init: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: Warmup 轮数
            milestones: 学习率衰减的 epoch 列表
            gamma: 学习率衰减因子
            warmup_lr_init: Warmup 初始学习率
            last_epoch: 上一轮 epoch
        """
        self.warmup_epochs = warmup_epochs
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_lr_init = warmup_lr_init
        
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)
                for base_lr in self.base_lrs
            ]
        else:
            # MultiStep 阶段
            factor = 1.0
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    factor *= self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    """
    多项式学习率衰减
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        power: float = 0.9,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: 优化器
            total_epochs: 总训练轮数
            power: 多项式幂次
            min_lr: 最小学习率
            last_epoch: 上一轮 epoch
        """
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [
            max(self.min_lr, base_lr * factor)
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    milestones: Optional[List[int]] = None,
    **kwargs
) -> _LRScheduler:
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称
        total_epochs: 总训练轮数
        warmup_epochs: Warmup 轮数
        min_lr: 最小学习率
        milestones: MultiStep 的里程碑
        **kwargs: 其他参数
    
    Returns:
        学习率调度器
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine':
        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
        )
    elif scheduler_name == 'step':
        if milestones is None:
            milestones = [int(total_epochs * 0.5), int(total_epochs * 0.75)]
        return WarmupMultiStepLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            milestones=milestones,
            min_lr=min_lr,
        )
    elif scheduler_name == 'polynomial':
        return PolynomialLR(
            optimizer,
            total_epochs=total_epochs,
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


if __name__ == "__main__":
    # 测试学习率调度器
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建模拟优化器
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # 测试 Cosine Annealing
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=5,
        total_epochs=200,
        min_lr=1e-6,
    )
    
    lrs = []
    for epoch in range(200):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    print(f"Initial LR: {lrs[0]:.6f}")
    print(f"Peak LR: {max(lrs):.6f}")
    print(f"Final LR: {lrs[-1]:.6f}")
    
    # 绘制学习率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Warmup + Cosine Annealing Learning Rate Schedule')
    plt.grid(True)
    plt.savefig('logs/lr_schedule.png')
    print("Learning rate schedule saved to logs/lr_schedule.png")