"""
损失函数模块
包含 Label Smoothing Cross Entropy 等损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    带有 Label Smoothing 的交叉熵损失
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: 平滑系数
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            pred: 预测值 (N, C)
            target: 目标值 (N,)
        
        Returns:
            损失值
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 创建平滑标签
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    """
    标准交叉熵损失
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        """
        Args:
            label_smoothing: Label Smoothing 系数
        """
        super(CrossEntropyLoss, self).__init__()
        
        if label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(pred, target)


class MixupLoss(nn.Module):
    """
    Mixup 损失函数
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        """
        Args:
            label_smoothing: Label Smoothing 系数
        """
        super(MixupLoss, self).__init__()
        
        if label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        计算 Mixup 损失
        
        Args:
            pred: 预测值
            target_a: 目标 A
            target_b: 目标 B
            lam: 混合系数
        
        Returns:
            损失值
        """
        return lam * self.criterion(pred, target_a) + (1 - lam) * self.criterion(pred, target_b)


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 归约方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Focal Loss
        
        Args:
            pred: 预测值 (N, C)
            target: 目标值 (N,)
        
        Returns:
            损失值
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    loss_name: str = 'cross_entropy',
    label_smoothing: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    获取损失函数
    
    Args:
        loss_name: 损失函数名称
        label_smoothing: Label Smoothing 系数
        **kwargs: 其他参数
    
    Returns:
        损失函数
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'cross_entropy':
        return CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # 测试损失函数
    print("Testing loss functions...")
    
    # 创建测试数据
    batch_size = 4
    num_classes = 10
    
    pred = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # 测试标准交叉熵
    ce_loss = CrossEntropyLoss()
    loss = ce_loss(pred, target)
    print(f"Cross Entropy Loss: {loss.item():.4f}")
    
    # 测试 Label Smoothing
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = ls_loss(pred, target)
    print(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # 测试 Focal Loss
    focal_loss = FocalLoss()
    loss = focal_loss(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")
    
    print("Loss function tests passed!")