"""
项目配置文件
包含所有超参数和路径配置
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据集路径
    data_dir: str = "dataset/cifar-10-batches-py"
    # 类别数量
    num_classes: int = 10
    # 类别名称
    class_names: List[str] = field(default_factory=lambda: [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ])
    # 图像尺寸
    image_size: int = 32
    # 数据增强
    normalize_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    normalize_std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])
    # 数据加载
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """模型相关配置"""
    # 模型名称
    model_name: str = "resnet18"
    # 预训练权重
    pretrained: bool = False
    # 类别数量
    num_classes: int = 10
    # Dropout 比例
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 训练轮数
    epochs: int = 200
    # 学习率
    learning_rate: float = 0.1
    # 权重衰减
    weight_decay: float = 5e-4
    # 动量
    momentum: float = 0.9
    # 学习率调度
    lr_scheduler: str = "cosine"  # cosine, step, multistep
    # Warmup 轮数
    warmup_epochs: int = 5
    # 最小学习率
    min_lr: float = 1e-6
    # 梯度裁剪
    gradient_clip: float = 5.0
    # Label Smoothing
    label_smoothing: float = 0.1
    # 保存间隔
    save_interval: int = 10
    # 验证间隔
    eval_interval: int = 1


@dataclass
class PruningConfig:
    """剪枝相关配置"""
    # 剪枝比例
    pruning_ratio: float = 0.5
    # 剪枝方法
    pruning_method: str = "l1"  # l1, l2, geometric_median
    # 渐进式剪枝
    progressive: bool = True
    # 渐进式剪枝轮数
    progressive_steps: int = 3
    # 最小通道数
    min_channels: int = 16
    # 跳过剪枝的层
    skip_layers: List[str] = field(default_factory=lambda: ["first_conv", "last_fc"])


@dataclass
class DistillationConfig:
    """知识蒸馏相关配置"""
    # 温度参数
    temperature: float = 4.0
    # 蒸馏权重 (alpha)
    alpha: float = 0.7
    # 蒸馏训练轮数
    epochs: int = 100
    # 学习率
    learning_rate: float = 0.01
    # 权重衰减
    weight_decay: float = 1e-4


@dataclass
class QuantizationConfig:
    """量化相关配置"""
    # 量化类型
    qtype: str = "int8"  # int8, uint8
    # 量化方法
    qmethod: str = "static"  # static, dynamic
    # 校准数据集大小
    calibration_size: int = 1000
    # 校准批大小
    calibration_batch_size: int = 32


@dataclass
class ExportConfig:
    """导出相关配置"""
    # ONNX 导出路径
    onnx_dir: str = "onnx_models"
    # ONNX opset 版本
    opset_version: int = 14
    # 动态 batch
    dynamic_batch: bool = True
    # 优化 ONNX
    optimize: bool = True


@dataclass
class InferenceConfig:
    """推理相关配置"""
    # 批大小
    batch_size: int = 1
    # 异步预处理
    async_preprocess: bool = True
    # 预处理队列大小
    preprocess_queue_size: int = 4
    # 后处理队列大小
    postprocess_queue_size: int = 4
    # 线程数
    num_threads: int = 4


@dataclass
class Config:
    """总配置类"""
    # 项目名称
    project_name: str = "embedded_object_detection"
    # 随机种子
    seed: int = 42
    # 设备
    device: str = "cuda"  # cuda, cpu
    
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 路径配置
    weights_dir: str = "weights"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        """初始化后创建目录"""
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.export.onnx_dir, exist_ok=True)


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取全局配置"""
    return config


def update_config(**kwargs):
    """更新配置"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")