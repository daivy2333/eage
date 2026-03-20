"""
知识蒸馏损失函数模块
包含 KL Divergence 损失、特征蒸馏损失等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class KLDivergenceLoss(nn.Module):
    """
    KL 散度损失
    用于知识蒸馏中的软标签损失
    """
    
    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: 温度参数，用于软化概率分布
        """
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 KL 散度损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
        
        Returns:
            KL 散度损失
        """
        # 软化概率分布
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # 计算 KL 散度
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return kl_loss


class DistillationLoss(nn.Module):
    """
    知识蒸馏综合损失
    结合软标签损失和硬标签损失
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            temperature: 温度参数
            alpha: 软标签损失权重
            label_smoothing: 标签平滑系数
        """
        super(DistillationLoss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        
        # KL 散度损失
        self.kl_loss = KLDivergenceLoss(temperature=temperature)
        
        # 硬标签损失
        if label_smoothing > 0:
            self.ce_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        
        Returns:
            total_loss: 总损失
            soft_loss: 软标签损失
            hard_loss: 硬标签损失
        """
        # 软标签损失
        soft_loss = self.kl_loss(student_logits, teacher_logits)
        
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss


class FeatureDistillationLoss(nn.Module):
    """
    特征蒸馏损失
    用于中间层特征的蒸馏
    """
    
    def __init__(
        self,
        feature_layers: list = ['layer1', 'layer2', 'layer3', 'layer4'],
        weights: Optional[list] = None
    ):
        """
        Args:
            feature_layers: 需要蒸馏的特征层
            weights: 各层权重
        """
        super(FeatureDistillationLoss, self).__init__()
        
        self.feature_layers = feature_layers
        self.weights = weights or [1.0] * len(feature_layers)
        
        # 特征适配器（用于匹配学生和教师的特征维度）
        self.adapters = nn.ModuleDict()
    
    def add_adapter(self, layer_name: str, student_channels: int, teacher_channels: int):
        """
        添加特征适配器
        
        Args:
            layer_name: 层名称
            student_channels: 学生特征通道数
            teacher_channels: 教师特征通道数
        """
        if student_channels != teacher_channels:
            self.adapters[layer_name] = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, bias=False
            )
    
    def forward(
        self,
        student_features: dict,
        teacher_features: dict
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失
        
        Args:
            student_features: 学生模型特征
            teacher_features: 教师模型特征
        
        Returns:
            特征蒸馏损失
        """
        total_loss = 0.0
        
        for layer_name, weight in zip(self.feature_layers, self.weights):
            if layer_name not in student_features or layer_name not in teacher_features:
                continue
            
            student_feat = student_features[layer_name]
            teacher_feat = teacher_features[layer_name]
            
            # 特征适配
            if layer_name in self.adapters:
                student_feat = self.adapters[layer_name](student_feat)
            
            # 计算特征损失（MSE）
            loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += weight * loss
        
        return total_loss


class AttentionDistillationLoss(nn.Module):
    """
    注意力蒸馏损失
    基于注意力图的蒸馏
    """
    
    def __init__(self):
        super(AttentionDistillationLoss, self).__init__()
    
    def compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算注意力图
        
        Args:
            features: 特征图 (B, C, H, W)
        
        Returns:
            注意力图 (B, H, W)
        """
        # 计算通道注意力
        attention = features.pow(2).mean(dim=1)  # (B, H, W)
        
        # 归一化
        attention = attention / (attention.sum(dim=(1, 2), keepdim=True) + 1e-6)
        
        return attention
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力蒸馏损失
        
        Args:
            student_features: 学生特征
            teacher_features: 教师特征
        
        Returns:
            注意力蒸馏损失
        """
        student_attention = self.compute_attention_map(student_features)
        teacher_attention = self.compute_attention_map(teacher_features)
        
        # 计算注意力损失
        loss = F.mse_loss(student_attention, teacher_attention)
        
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    带有标签平滑的交叉熵损失
    """
    
    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_distillation_loss(
    temperature: float = 4.0,
    alpha: float = 0.7,
    label_smoothing: float = 0.0
) -> DistillationLoss:
    """
    获取蒸馏损失函数
    
    Args:
        temperature: 温度参数
        alpha: 软标签损失权重
        label_smoothing: 标签平滑系数
    
    Returns:
        蒸馏损失函数
    """
    return DistillationLoss(
        temperature=temperature,
        alpha=alpha,
        label_smoothing=label_smoothing
    )


if __name__ == "__main__":
    # 测试蒸馏损失
    print("Testing distillation losses...")
    
    batch_size = 4
    num_classes = 10
    
    # 创建测试数据
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 测试 KL 散度损失
    kl_loss = KLDivergenceLoss(temperature=4.0)
    loss = kl_loss(student_logits, teacher_logits)
    print(f"KL Divergence Loss: {loss.item():.4f}")
    
    # 测试综合蒸馏损失
    distill_loss = DistillationLoss(temperature=4.0, alpha=0.7)
    total_loss, soft_loss, hard_loss = distill_loss(student_logits, teacher_logits, labels)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Soft Loss: {soft_loss.item():.4f}")
    print(f"Hard Loss: {hard_loss.item():.4f}")
    
    print("Distillation loss tests passed!")