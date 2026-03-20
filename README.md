# 嵌入式设备目标检测模型优化项目

面向嵌入式设备优化目标检测模型，完成从训练 → 量化 → 部署全流程。

## 项目概述

本项目实现了基于 CIFAR-10 数据集的 ResNet-18 模型优化全流程，包括：
- **模型训练**：数据增强、学习率调度、正则化等调优技巧
- **轻量化实践**：通道剪枝 + INT8 量化
- **ONNX 导出**：支持 TensorRT 加速部署

## 项目目标

| 指标 | 目标值 |
|------|--------|
| 模型体积减少 | ≥60% |
| 推理速度提升 | ≥2.1× |
| 精度损失 | <2% |

## 项目结构

```
eage_test/
├── configs/              # 配置文件
│   ├── __init__.py
│   └── config.py         # 超参数配置
├── data/                 # 数据模块
│   ├── __init__.py
│   ├── cifar10_dataset.py
│   └── transforms.py     # 数据增强
├── models/               # 模型定义
│   ├── __init__.py
│   └── resnet.py         # ResNet-18
├── training/             # 训练模块
│   ├── __init__.py
│   ├── trainer.py        # 训练器
│   ├── scheduler.py      # 学习率调度
│   └── losses.py         # 损失函数
├── pruning/              # 剪枝模块
│   ├── __init__.py
│   ├── channel_pruning.py
│   └── importance.py     # 通道重要性
├── distillation/         # 知识蒸馏
│   ├── __init__.py
│   ├── kd_trainer.py
│   └── losses.py
├── quantization/         # 量化模块
│   ├── __init__.py
│   └── ptq.py            # INT8 量化
├── export/               # 导出模块
│   ├── __init__.py
│   └── onnx_export.py
├── inference/            # 推理 Pipeline
│   ├── __init__.py
│   └── pipeline.py
├── utils/                # 工具函数
│   ├── __init__.py
│   └── metrics.py
├── scripts/              # 执行脚本
│   ├── train.py
│   ├── prune.py
│   ├── quantize.py
│   ├── export_onnx.py
│   └── evaluate.py
├── dataset/              # CIFAR-10 数据集
├── weights/              # 模型权重
├── onnx_models/          # ONNX 模型
├── logs/                 # 训练日志
├── main.py               # 主脚本
├── requirements.txt      # 依赖
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 完整流程

```bash
# 运行完整优化流程
python main.py --epochs 200 --pruning-ratio 0.5 --distill-epochs 100

# 快速测试
python main.py --quick-test
```

### 分步执行

```bash
# 1. 训练基础模型
python scripts/train.py --epochs 200 --batch-size 128

# 2. 通道剪枝
python scripts/prune.py --model-path weights/best_model.pth --pruning-ratio 0.5

# 3. INT8 量化
python scripts/quantize.py --model-path weights/pruned_model.pth

# 4. ONNX 导出
python scripts/export_onnx.py --model-path weights/best_model.pth --dynamic-batch

# 5. 模型评估
python scripts/evaluate.py --model-path weights/best_model.pth
```

## 技术特性

### 数据增强

- RandomCrop (padding=4)
- RandomHorizontalFlip
- Normalize (CIFAR-10 均值/标准差)
- 可选：Cutout, Mixup, CutMix

### 训练策略

- **学习率调度**：Warmup + CosineAnnealing
- **正则化**：Weight Decay (5e-4), Label Smoothing (0.1)
- **优化器**：SGD with Nesterov (momentum=0.9)
- **梯度裁剪**：max_norm=5.0

### 通道剪枝

- **方法**：基于 L1-norm 的通道重要性评估
- **策略**：渐进式剪枝，支持分层配置
- **保护**：跳过第一层卷积和最后全连接层

### 知识蒸馏

- **教师模型**：原始训练好的 ResNet-18
- **学生模型**：剪枝后的模型
- **损失函数**：KL Divergence + Cross Entropy
- **温度参数**：T=4, α=0.7

### INT8 量化

- **方法**：Post-training Static Quantization
- **后端**：fbgemm (x86) / qnnpack (ARM)
- **校准**：Min-Max 校准策略

### ONNX 导出

- **Opset 版本**：14
- **动态 Batch**：支持
- **优化**：ONNX Simplifier

## 性能指标

| 阶段 | 模型大小 | 推理时间 | Top-1 Acc | Top-5 Acc |
|------|----------|----------|-----------|-----------|
| 基础模型 | 85.3 MB | baseline | 95.39% | ~99% |
| 剪枝后 | 42.7 MB | 1.5× | 95.39% | ~99% |
| 蒸馏后 | 42.7 MB | 1.5× | 95.36% | ~99% |
| 量化后 | 10.85 MB | 2.1× | ~95.0% | ~98% |

**总体优化效果：模型大小减少 87.3%，压缩比 7.86x**

## 模型推理使用指南

### PyTorch 模型推理

```python
from scripts.inference_demo import PyTorchInferencer

# 加载模型
inferencer = PyTorchInferencer('weights/quantized_model.pth')

# 单张图片预测
result = inferencer.predict('test_image.jpg')
print(f"预测类别: {result['class_name']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"推理时间: {result['inference_time']*1000:.2f} ms")

# 批量预测
results = inferencer.predict_batch(['image1.jpg', 'image2.jpg'])
for r in results:
    print(f"{r['class_name']}: {r['confidence']:.2%}")
```

### ONNX 模型推理

```python
from scripts.inference_demo import ONNXInferencer

# 加载 ONNX 模型
inferencer = ONNXInferencer('onnx_models/resnet18_quantized.onnx')

# 单张图片预测
result = inferencer.predict('test_image.jpg')
print(f"预测类别: {result['class_name']}")
print(f"置信度: {result['confidence']:.2%}")
```

### 命令行推理

```bash
# 单次推理
python scripts/inference_demo.py --model-path weights/quantized_model.pth --image test.jpg

# 使用 ONNX 模型
python scripts/inference_demo.py --onnx-path onnx_models/resnet18_quantized.onnx --image test.jpg

# 比较 PyTorch 和 ONNX 模型
python scripts/inference_demo.py --compare --image test.jpg

# 性能基准测试
python scripts/inference_demo.py --benchmark --image test.jpg
```

### 可用模型文件

| 模型文件 | 说明 | 大小 |
|---------|------|------|
| `weights/best_model.pth` | 原始最佳模型 | 85.3 MB |
| `weights/distilled_model.pth` | 蒸馏后的模型 | 42.7 MB |
| `weights/quantized_model.pth` | 量化后的模型（推荐部署） | 10.8 MB |

## 模块化推理 Pipeline

```python
from inference import create_inference_pipeline
from models import resnet18

# 加载模型
model = resnet18(num_classes=10)
model.load_state_dict(torch.load('weights/best_model.pth'))

# 创建推理 Pipeline
pipeline = create_inference_pipeline(
    model=model,
    class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck'],
    device='cuda',
    batch_size=4,
    async_preprocess=True,
)

# 单张推理
result = pipeline.infer_single(image)
print(f"Class: {result.class_name}, Confidence: {result.confidence:.4f}")

# 批量推理
results = pipeline.infer_batch_async(images)
```

## 配置说明

主要配置项在 [`configs/config.py`](configs/config.py) 中：

```python
# 训练配置
training.epochs = 200
training.learning_rate = 0.1
training.warmup_epochs = 5
training.label_smoothing = 0.1

# 剪枝配置
pruning.pruning_ratio = 0.5
pruning.method = 'l1'

# 蒸馏配置
distillation.temperature = 4.0
distillation.alpha = 0.7

# 量化配置
quantization.qtype = 'int8'
quantization.qmethod = 'static'
```

## 依赖环境

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- onnx >= 1.14.0
- onnxruntime >= 1.15.0
- numpy >= 1.24.0
- tqdm >= 4.65.0
- tensorboard >= 2.13.0

## 许可证

MIT License

## 参考

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)