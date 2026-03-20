#!/usr/bin/env python
"""
模型推理示例脚本
演示如何使用 PyTorch 和 ONNX 模型进行推理
"""
import os
import sys
import argparse
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from configs import get_config
from models import resnet18


# CIFAR-10 类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def _dequantize_tensor(tensor):
    """
    反量化张量
    
    Args:
        tensor: 可能是量化的张量
    
    Returns:
        反量化后的浮点张量
    """
    if hasattr(tensor, 'dequantize'):
        return tensor.dequantize()
    return tensor


def _extract_original_weights(checkpoint: dict) -> dict:
    """
    从量化模型检查点中提取原始权重
    
    Args:
        checkpoint: 包含量化信息的字典
    
    Returns:
        可用于加载到基础模型的 state_dict
    """
    if 'model_state_dict' not in checkpoint:
        return {}
    
    state_dict = checkpoint['model_state_dict']
    extracted = {}
    
    # 构建键名映射（处理 QuantWrapper 前缀和 BN 融合）
    for key, value in state_dict.items():
        # 移除 QuantWrapper 的 'model.' 前缀
        if key.startswith('model.'):
            new_key = key[6:]  # 移除 'model.' 前缀
        else:
            new_key = key
        
        # 跳过量化相关的键（scale, zero_point, _packed_params 等）
        if any(skip in new_key for skip in ['scale', 'zero_point', '_packed_params', 'weight_quant', 'activation_quant']):
            continue
        
        # 反量化张量（如果是量化张量）
        try:
            extracted[new_key] = _dequantize_tensor(value)
        except Exception:
            extracted[new_key] = value
    
    return extracted


def _remap_fused_bn_keys(state_dict: dict) -> dict:
    """
    重新映射融合后的 BN 键名
    
    静态量化时 Conv+BN 被融合，BN 的权重被吸收到 Conv 中。
    这个函数为缺失的 BN 层创建默认值。
    
    Args:
        state_dict: 原始 state_dict
    
    Returns:
        包含 BN 层默认值的 state_dict
    """
    import copy
    new_state_dict = copy.deepcopy(state_dict)
    
    # 需要添加 BN 层的层列表
    bn_layers = [
        'bn1',
        'layer1.0.bn1', 'layer1.0.bn2', 'layer1.1.bn1', 'layer1.1.bn2',
        'layer2.0.bn1', 'layer2.0.bn2', 'layer2.1.bn1', 'layer2.1.bn2',
        'layer3.0.bn1', 'layer3.0.bn2', 'layer3.1.bn1', 'layer3.1.bn2',
        'layer4.0.bn1', 'layer4.0.bn2', 'layer4.1.bn1', 'layer4.1.bn2',
        'layer2.0.downsample.1', 'layer3.0.downsample.1', 'layer4.0.downsample.1',
    ]
    
    # 为缺失的 BN 层添加默认值
    for bn_name in bn_layers:
        weight_key = f'{bn_name}.weight'
        bias_key = f'{bn_name}.bias'
        running_mean_key = f'{bn_name}.running_mean'
        running_var_key = f'{bn_name}.running_var'
        num_batches_tracked_key = f'{bn_name}.num_batches_tracked'
        
        if weight_key not in new_state_dict:
            # 尝试推断通道数
            # 从对应的 conv 层获取通道数
            conv_mapping = {
                'bn1': 'conv1',
                'layer1.0.bn1': 'layer1.0.conv1',
                'layer1.0.bn2': 'layer1.0.conv2',
                'layer1.1.bn1': 'layer1.1.conv1',
                'layer1.1.bn2': 'layer1.1.conv2',
                'layer2.0.bn1': 'layer2.0.conv1',
                'layer2.0.bn2': 'layer2.0.conv2',
                'layer2.1.bn1': 'layer2.1.conv1',
                'layer2.1.bn2': 'layer2.1.conv2',
                'layer3.0.bn1': 'layer3.0.conv1',
                'layer3.0.bn2': 'layer3.0.conv2',
                'layer3.1.bn1': 'layer3.1.conv1',
                'layer3.1.bn2': 'layer3.1.conv2',
                'layer4.0.bn1': 'layer4.0.conv1',
                'layer4.0.bn2': 'layer4.0.conv2',
                'layer4.1.bn1': 'layer4.1.conv1',
                'layer4.1.bn2': 'layer4.1.conv2',
                'layer2.0.downsample.1': 'layer2.0.downsample.0',
                'layer3.0.downsample.1': 'layer3.0.downsample.0',
                'layer4.0.downsample.1': 'layer4.0.downsample.0',
            }
            
            if bn_name in conv_mapping:
                conv_key = f'{conv_mapping[bn_name]}.weight'
                if conv_key in new_state_dict:
                    num_features = new_state_dict[conv_key].shape[0]
                    
                    # BN 默认值：weight=1, bias=0, running_mean=0, running_var=1
                    new_state_dict[weight_key] = torch.ones(num_features)
                    new_state_dict[bias_key] = torch.zeros(num_features)
                    new_state_dict[running_mean_key] = torch.zeros(num_features)
                    new_state_dict[running_var_key] = torch.ones(num_features)
                    new_state_dict[num_batches_tracked_key] = torch.tensor(0)
    
    return new_state_dict


def _rebuild_quantized_model(checkpoint: dict, num_classes: int = 10) -> nn.Module:
    """
    重建量化模型（用于旧格式模型）
    
    注意：由于 ResNet 的残差连接结构，静态量化存在兼容性问题。
    此函数会自动将静态量化模型转换为动态量化模型。
    
    Args:
        checkpoint: 包含量化信息的字典
        num_classes: 类别数量
    
    Returns:
        重建的量化模型
    """
    qconfig = checkpoint.get('qconfig', 'fbgemm')
    qmethod = checkpoint.get('qmethod', 'static')
    
    # 设置量化引擎
    torch.backends.quantized.engine = qconfig
    
    # 创建基础模型
    base_model = resnet18(num_classes=num_classes)
    base_model.eval()
    
    # 加载原始权重到基础模型
    # 新格式：保存的是原始模型权重，可以直接加载
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # 检查是否是量化后的权重（包含 _packed_params 等键）
        is_quantized_state = any('_packed_params' in k or 'scale' in k for k in state_dict.keys())
        
        if is_quantized_state:
            # 旧格式：量化后的权重，需要提取和反量化
            print("Extracting weights from quantized state_dict...")
            original_weights = _extract_original_weights(checkpoint)
            remapped_weights = _remap_fused_bn_keys(original_weights)
            
            try:
                base_model.load_state_dict(remapped_weights, strict=False)
                print("Loaded weights from quantized checkpoint (partial)")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
        else:
            # 新格式：原始权重，可以直接加载
            try:
                base_model.load_state_dict(state_dict, strict=True)
                print("Loaded original weights into base model")
            except Exception as e:
                print(f"Warning: Could not load all weights: {e}")
                base_model.load_state_dict(state_dict, strict=False)
                print("Loaded partial weights")
    
    # 由于 ResNet 残差连接的兼容性问题，统一使用动态量化
    if qmethod == 'static':
        print("Note: Static quantization has compatibility issues with ResNet residual connections.")
        print("Automatically falling back to dynamic quantization for better compatibility.")
    
    # 动态量化 - 更好的兼容性
    # 注意：动态量化会保留已加载的权重
    quantized_model = torch.quantization.quantize_dynamic(
        base_model,
        {nn.Linear},  # 只量化全连接层，避免残差连接问题
        dtype=torch.qint8
    )
    
    return quantized_model


class PyTorchInferencer:
    """PyTorch 模型推理器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        num_classes: int = 10
    ):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            device: 推理设备
            num_classes: 类别数量
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.is_quantized = False
        self.is_torchscript = False
        
        # 加载模型
        print(f"Loading model from {model_path}...")
        
        # 检查是否为 TorchScript 模型
        if model_path.endswith('.pt') or '_scripted' in model_path or '_traced' in model_path:
            try:
                self.model = torch.jit.load(model_path, map_location='cpu')
                self.is_torchscript = True
                self.is_quantized = True  # TorchScript 量化模型
                self.device = 'cpu'  # 量化模型在 CPU 上运行
                print(f"TorchScript model loaded successfully")
                print(f"Model running on CPU")
            except Exception as e:
                print(f"Failed to load as TorchScript: {e}")
                print("Trying to load as regular PyTorch model...")
                self._load_regular_model(model_path, num_classes)
        else:
            self._load_regular_model(model_path, num_classes)
        
        # 预处理
        config = get_config()
        self.transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.data.normalize_mean,
                std=config.data.normalize_std
            ),
        ])
        
        print(f"Model loaded on {self.device}")
    
    def _load_regular_model(self, model_path: str, num_classes: int):
        """加载常规 PyTorch 模型"""
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 检查是否为量化模型
        if isinstance(checkpoint, dict) and checkpoint.get('quantized', False):
            # 量化模型 - 需要重建
            print(f"Loading quantized model ({checkpoint.get('qmethod', 'static')})...")
            
            qconfig = checkpoint.get('qconfig', 'fbgemm')
            print(f"Rebuilding quantized model with {qconfig} backend...")
            
            self.model = _rebuild_quantized_model(checkpoint, num_classes)
            self.is_quantized = True
            self.device = 'cpu'
            print(f"Quantized model rebuilt successfully")
            
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 普通模型（带 model_state_dict 键）
            self.model = resnet18(num_classes=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
        else:
            # 普通模型（直接是 state_dict）
            self.model = resnet18(num_classes=num_classes)
            self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def preprocess(self, image_path: str) -> torch.Tensor:
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        单张图像预测
        
        Args:
            image_path: 图像路径
        
        Returns:
            预测结果字典
        """
        # 预处理
        tensor = self.preprocess(image_path).to(self.device)
        
        # 推理
        start_time = time.time()
        output = self.model(tensor)
        inference_time = time.time() - start_time
        
        # 后处理
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        
        return {
            'class_id': pred_class.item(),
            'class_name': CIFAR10_CLASSES[pred_class.item()],
            'confidence': confidence.item(),
            'inference_time': inference_time,
            'all_probs': probs.cpu().numpy().flatten()
        }
    
    @torch.no_grad()
    def predict_batch(self, image_paths: list) -> list:
        """
        批量图像预测
        
        Args:
            image_paths: 图像路径列表
        
        Returns:
            预测结果列表
        """
        # 预处理
        tensors = torch.stack([self.preprocess(path).squeeze(0) for path in image_paths])
        tensors = tensors.to(self.device)
        
        # 推理
        start_time = time.time()
        outputs = self.model(tensors)
        inference_time = time.time() - start_time
        
        # 后处理
        probs = torch.softmax(outputs, dim=1)
        confidences, pred_classes = torch.max(probs, dim=1)
        
        results = []
        for i in range(len(image_paths)):
            results.append({
                'class_id': pred_classes[i].item(),
                'class_name': CIFAR10_CLASSES[pred_classes[i].item()],
                'confidence': confidences[i].item(),
                'inference_time': inference_time / len(image_paths),
                'all_probs': probs[i].cpu().numpy()
            })
        
        return results


class ONNXInferencer:
    """ONNX 模型推理器"""
    
    def __init__(
        self,
        onnx_path: str,
        device: str = 'cuda'
    ):
        """
        初始化推理器
        
        Args:
            onnx_path: ONNX 模型路径
            device: 推理设备
        """
        import onnxruntime as ort
        
        self.device = device
        
        # 设置执行提供者
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if device == 'cpu' or not torch.cuda.is_available():
            providers = ['CPUExecutionProvider']
        
        # 创建推理会话
        print(f"Loading ONNX model from {onnx_path}...")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 预处理
        config = get_config()
        self.transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.data.normalize_mean,
                std=config.data.normalize_std
            ),
        ])
        
        print(f"ONNX model loaded with providers: {self.session.get_providers()}")
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0).numpy()
    
    def predict(self, image_path: str) -> dict:
        """
        单张图像预测
        
        Args:
            image_path: 图像路径
        
        Returns:
            预测结果字典
        """
        # 预处理
        input_tensor = self.preprocess(image_path)
        
        # 推理
        start_time = time.time()
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        inference_time = time.time() - start_time
        
        # 后处理
        output = output.squeeze()
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / exp_output.sum()
        
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        return {
            'class_id': int(pred_class),
            'class_name': CIFAR10_CLASSES[pred_class],
            'confidence': float(confidence),
            'inference_time': inference_time,
            'all_probs': probs
        }
    
    def predict_batch(self, image_paths: list) -> list:
        """
        批量图像预测
        
        Args:
            image_paths: 图像路径列表
        
        Returns:
            预测结果列表
        """
        # 预处理
        input_tensors = np.vstack([self.preprocess(path) for path in image_paths])
        
        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensors})[0]
        inference_time = time.time() - start_time
        
        # 后处理
        results = []
        for i in range(len(image_paths)):
            output = outputs[i]
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / exp_output.sum()
            
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            results.append({
                'class_id': int(pred_class),
                'class_name': CIFAR10_CLASSES[pred_class],
                'confidence': float(confidence),
                'inference_time': inference_time / len(image_paths),
                'all_probs': probs
            })
        
        return results


def compare_models(pytorch_path: str, onnx_path: str, test_image: str):
    """
    比较 PyTorch 和 ONNX 模型的推理结果
    
    Args:
        pytorch_path: PyTorch 模型路径
        onnx_path: ONNX 模型路径
        test_image: 测试图像路径
    """
    print("\n" + "=" * 60)
    print("Comparing PyTorch and ONNX models")
    print("=" * 60)
    
    # PyTorch 推理
    pytorch_inferencer = PyTorchInferencer(pytorch_path)
    pytorch_result = pytorch_inferencer.predict(test_image)
    
    # ONNX 推理
    onnx_inferencer = ONNXInferencer(onnx_path)
    onnx_result = onnx_inferencer.predict(test_image)
    
    # 打印结果
    print(f"\nTest image: {test_image}")
    print("\nPyTorch Result:")
    print(f"  Class: {pytorch_result['class_name']} (ID: {pytorch_result['class_id']})")
    print(f"  Confidence: {pytorch_result['confidence']:.4f}")
    print(f"  Inference time: {pytorch_result['inference_time']*1000:.2f} ms")
    
    print("\nONNX Result:")
    print(f"  Class: {onnx_result['class_name']} (ID: {onnx_result['class_id']})")
    print(f"  Confidence: {onnx_result['confidence']:.4f}")
    print(f"  Inference time: {onnx_result['inference_time']*1000:.2f} ms")
    
    # 检查一致性
    if pytorch_result['class_id'] == onnx_result['class_id']:
        print("\n✓ Predictions match!")
    else:
        print("\n✗ Predictions differ!")
    
    # 检查概率差异
    prob_diff = np.abs(pytorch_result['all_probs'] - onnx_result['all_probs']).max()
    print(f"Max probability difference: {prob_diff:.6f}")


def benchmark_model(inferencer, test_images: list, num_runs: int = 100):
    """
    模型性能基准测试
    
    Args:
        inferencer: 推理器实例
        test_images: 测试图像列表
        num_runs: 运行次数
    """
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    # 预热
    for _ in range(10):
        inferencer.predict(test_images[0])
    
    # 基准测试
    times = []
    for i in range(num_runs):
        image = test_images[i % len(test_images)]
        result = inferencer.predict(image)
        times.append(result['inference_time'])
    
    # 统计
    times = np.array(times)
    print(f"\nNumber of runs: {num_runs}")
    print(f"Average inference time: {times.mean()*1000:.2f} ms")
    print(f"Std deviation: {times.std()*1000:.2f} ms")
    print(f"Min time: {times.min()*1000:.2f} ms")
    print(f"Max time: {times.max()*1000:.2f} ms")
    print(f"Throughput: {1/times.mean():.1f} images/sec")


def main():
    parser = argparse.ArgumentParser(description='Model Inference Demo')
    parser.add_argument('--model-path', type=str, default='weights/quantized_model.pt',
                        help='Path to PyTorch model (.pt for TorchScript, .pth for legacy)')
    parser.add_argument('--onnx-path', type=str, default='onnx_models/resnet18_quantized.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--compare', action='store_true',
                        help='Compare PyTorch and ONNX models')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    args = parser.parse_args()
    
    # 检查是否有测试图像
    if args.image is None:
        # 使用 CIFAR-10 测试集中的图像
        test_dir = 'dataset/cifar-10-batches-py'
        print(f"No test image provided. Please specify --image argument.")
        print(f"Example: python scripts/inference_demo.py --image path/to/image.jpg")
        return
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    # 比较模型
    if args.compare:
        if os.path.exists(args.model_path) and os.path.exists(args.onnx_path):
            compare_models(args.model_path, args.onnx_path, args.image)
        else:
            print("Model files not found. Please check paths.")
    
    # 基准测试
    elif args.benchmark:
        if os.path.exists(args.model_path):
            inferencer = PyTorchInferencer(args.model_path, args.device)
            benchmark_model(inferencer, [args.image])
        else:
            print(f"Model not found: {args.model_path}")
    
    # 单次推理
    else:
        # 优先使用 ONNX 模型（如果指定了 --onnx-path 且文件存在）
        use_onnx = args.onnx_path and os.path.exists(args.onnx_path)
        use_pytorch = os.path.exists(args.model_path)
        
        if use_onnx:
            print(f"Using ONNX model: {args.onnx_path}")
            inferencer = ONNXInferencer(args.onnx_path, args.device)
        elif use_pytorch:
            print(f"Using PyTorch model: {args.model_path}")
            inferencer = PyTorchInferencer(args.model_path, args.device)
        else:
            print(f"Model not found.")
            print(f"  PyTorch model: {args.model_path} (exists: {use_pytorch})")
            print(f"  ONNX model: {args.onnx_path} (exists: {use_onnx})")
            return
        
        result = inferencer.predict(args.image)
        
        print("\n" + "=" * 60)
        print("Inference Result")
        print("=" * 60)
        print(f"Image: {args.image}")
        print(f"Predicted class: {result['class_name']} (ID: {result['class_id']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {result['inference_time']*1000:.2f} ms")
        
        # 打印 Top-5 预测
        probs = result['all_probs']
        top5_indices = np.argsort(probs)[-5:][::-1]
        print("\nTop-5 predictions:")
        for i, idx in enumerate(top5_indices):
            print(f"  {i+1}. {CIFAR10_CLASSES[idx]}: {probs[idx]:.4f}")


if __name__ == '__main__':
    main()