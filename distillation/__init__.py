"""知识蒸馏模块"""
from .losses import (
    KLDivergenceLoss,
    DistillationLoss,
    FeatureDistillationLoss,
    AttentionDistillationLoss,
    get_distillation_loss,
)
from .kd_trainer import (
    KnowledgeDistillationTrainer,
    distill_model,
)

__all__ = [
    'KLDivergenceLoss',
    'DistillationLoss',
    'FeatureDistillationLoss',
    'AttentionDistillationLoss',
    'get_distillation_loss',
    'KnowledgeDistillationTrainer',
    'distill_model',
]