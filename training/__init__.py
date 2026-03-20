"""训练模块"""
from .trainer import Trainer, train_model
from .scheduler import (
    WarmupCosineAnnealingLR,
    WarmupMultiStepLR,
    PolynomialLR,
    get_scheduler,
)
from .losses import (
    LabelSmoothingCrossEntropy,
    CrossEntropyLoss,
    MixupLoss,
    FocalLoss,
    get_loss_function,
)

__all__ = [
    'Trainer',
    'train_model',
    'WarmupCosineAnnealingLR',
    'WarmupMultiStepLR',
    'PolynomialLR',
    'get_scheduler',
    'LabelSmoothingCrossEntropy',
    'CrossEntropyLoss',
    'MixupLoss',
    'FocalLoss',
    'get_loss_function',
]