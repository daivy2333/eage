# 量化模型问题分析报告

## 问题概述

从测试结果可以看到以下问题：

| 模型 | 预测结果 | 置信度 | 问题 |
|------|----------|--------|------|
| `best_model.pth` (原始模型) | airplane | 0.9064 | ✅ 正确 |
| `pruned_model.pth` (剪枝模型) | airplane | 0.9064 | ✅ 正确 |
| `quantized_model.pt` (量化模型) | automobile | 0.2269 | ❌ 错误，类似随机猜测 |
| ONNX 量化模型 | cat | 0.1551 | ❌ 错误，类似随机猜测 |

## 根本原因分析

### 1. 量化模型权重加载失败

**核心问题**：量化模型保存格式与加载格式不匹配。

从日志可以看到关键错误：
```
Warning: Could not load weights: 'fc.scale'
Using randomly initialized weights
```

**原因详解**：

1. **保存时**（[`scripts/quantize.py:99-102`](scripts/quantize.py:99)）：
   - 静态量化模型被 `QuantWrapper` 包装
   - state_dict 键名格式：`model.conv1.weight`, `model.bn1.weight`, `model.fc.weight` 等
   - 量化后的 fc 层包含 `scale`, `zero_point` 等量化参数

2. **加载时**（[`scripts/inference_demo.py:61-65`](scripts/inference_demo.py:61)）：
   - 由于 ResNet 残差连接兼容性问题，回退到动态量化
   - 动态量化只量化 `nn.Linear` 层
   - 动态量化后的模型结构不同，fc 层键名变为 `fc._packed_params._packed_params`

3. **结果**：权重键名不匹配，无法加载训练好的权重，使用随机初始化。

### 2. ResNet 残差连接与静态量化的兼容性问题

**问题**：ResNet 的残差连接 (`out += identity`) 在静态量化时存在数值问题。

```python
# models/resnet.py:85
out += identity  # 残差连接
```

**原因**：
- 静态量化要求所有张量有固定的 scale 和 zero_point
- 残差连接中两个分支的量化参数可能不同
- 直接相加会导致精度损失或数值溢出

**当前处理**（[`scripts/inference_demo.py:56-65`](scripts/inference_demo.py:56)）：
```python
if qmethod == 'static':
    print("Note: Static quantization has compatibility issues with ResNet residual connections.")
    print("Automatically falling back to dynamic quantization for better compatibility.")

# 动态量化 - 只量化全连接层
quantized_model = torch.quantization.quantize_dynamic(
    base_model,
    {nn.Linear},  # 只量化全连接层，避免残差连接问题
    dtype=torch.qint8
)
```

### 3. 动态量化对 ResNet-18 效果有限

**问题**：ResNet-18 只有一个 `fc` 层，动态量化收益极小。

**ResNet-18 参数分布**：
- 总参数量：约 11.2M
- `fc` 层参数：512 × 10 = 5,120（仅占 0.05%）
- 卷积层参数：约 11.2M（占 99.95%）

**结果**：
- 动态量化只量化了 0.05% 的参数
- 模型大小几乎不变
- 推理速度提升有限
- 如果权重加载失败，相当于用随机权重的 fc 层，预测完全错误

### 4. ONNX 推理参数被忽略

**问题**：`--onnx-path` 参数在没有 `--compare` 标志时被完全忽略。

**代码逻辑**（[`scripts/inference_demo.py:483-486`](scripts/inference_demo.py:483)）：
```python
# 单次推理
else:
    if os.path.exists(args.model_path):
        inferencer = PyTorchInferencer(args.model_path, args.device)  # 始终使用 model_path
```

当用户执行：
```bash
python scripts/inference_demo.py --onnx-path onnx_models/resnet18_quantized.onnx --image test.jpg
```

实际使用的是 `--model-path` 默认值 `weights/quantized_model.pt`，而不是指定的 ONNX 模型。

### 5. ONNX 量化模型导出问题

**问题**：如果 PyTorch 量化模型权重未正确加载，导出的 ONNX 模型也是随机权重。

**流程**：
1. 量化脚本保存模型 → TorchScript 失败 → 回退到 legacy 格式
2. 加载时权重不匹配 → 使用随机权重
3. 导出 ONNX 时使用随机权重模型
4. ONNX 模型推理结果错误

## 问题链路图

```
静态量化保存 (QuantWrapper 包装)
         ↓
TorchScript trace 失败 (残差连接问题)
         ↓
回退到 legacy 格式保存
         ↓
加载时检测到量化模型
         ↓
由于残差连接兼容性问题，回退到动态量化
         ↓
动态量化模型结构与保存的不匹配
         ↓
权重加载失败 ('fc.scale' 不存在)
         ↓
使用随机初始化权重
         ↓
推理结果错误 (随机猜测水平)
         ↓
导出 ONNX 也是随机权重
         ↓
ONNX 推理结果错误
```

## 修复建议

### 方案 1：修复权重加载逻辑（推荐）

修改 `_rebuild_quantized_model` 函数，正确处理权重映射：

```python
def _rebuild_quantized_model(checkpoint: dict, num_classes: int = 10) -> nn.Module:
    # 创建基础模型
    base_model = resnet18(num_classes=num_classes)
    base_model.eval()
    
    # 先加载原始权重到基础模型
    if 'model_state_dict' in checkpoint:
        # 提取 model. 前缀的权重
        state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('model.'):
                new_key = k[6:]  # 移除 'model.' 前缀
                # 跳过量化相关的键
                if 'scale' not in new_key and 'zero_point' not in new_key:
                    state_dict[new_key] = v
        base_model.load_state_dict(state_dict, strict=False)
    
    # 然后应用动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        base_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized_model
```

### 方案 2：使用 QAT（量化感知训练）

对于 ResNet 等有残差连接的模型，推荐使用 QAT：

```python
# 在训练过程中模拟量化
model = prepare_qat(model)
# 继续训练几个 epoch
model = train(model, train_loader, epochs=5)
# 转换为量化模型
model = convert(model)
```

### 方案 3：使用 FX 图模式量化

PyTorch 2.x 推荐使用 FX 图模式量化，自动处理残差连接：

```python
from torch.ao.quantization import get_default_qconfig, prepare_fx, convert_fx

qconfig = get_default_qconfig('fbgemm')
model_prepared = prepare_fx(model, qconfig)
# 校准
calibrate(model_prepared, calibration_loader)
# 转换
model_quantized = convert_fx(model_prepared)
```

### 方案 4：修复 ONNX 参数处理

修改 `main` 函数，正确处理 `--onnx-path` 参数：

```python
# 单次推理
else:
    # 优先使用 ONNX 模型
    if args.onnx_path and os.path.exists(args.onnx_path):
        inferencer = ONNXInferencer(args.onnx_path, args.device)
    elif os.path.exists(args.model_path):
        inferencer = PyTorchInferencer(args.model_path, args.device)
    else:
        print(f"Model not found")
        return
```

## 结论

用户的猜测部分正确，但核心问题不是 ResNet-18 太小，而是：

1. **量化模型保存/加载格式不兼容**：静态量化保存的权重无法加载到动态量化模型
2. **动态量化范围太小**：只量化了 fc 层（0.05% 参数），效果有限
3. **参数处理逻辑缺陷**：`--onnx-path` 参数被忽略

最关键的修复是**正确加载原始权重到基础模型，然后再应用动态量化**，这样可以保留训练好的权重。