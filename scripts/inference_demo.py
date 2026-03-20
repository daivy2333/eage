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
        
        # 加载模型
        print(f"Loading model from {model_path}...")
        self.model = resnet18(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
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
    parser.add_argument('--model-path', type=str, default='weights/quantized_model.pth',
                        help='Path to PyTorch model')
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
        if os.path.exists(args.model_path):
            inferencer = PyTorchInferencer(args.model_path, args.device)
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
        else:
            print(f"Model not found: {args.model_path}")


if __name__ == '__main__':
    main()