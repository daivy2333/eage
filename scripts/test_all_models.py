#!/usr/bin/env python
"""
一键测试脚本
测试所有模型（原始、剪枝、蒸馏、量化、ONNX）并保存结果
"""
import os
import sys
import time
import argparse
from datetime import datetime

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


class ModelTester:
    """模型测试器"""
    
    def __init__(self, test_image: str, output_file: str = None):
        """
        初始化测试器
        
        Args:
            test_image: 测试图像路径
            output_file: 输出文件路径
        """
        self.test_image = test_image
        self.output_file = output_file
        self.results = []
        
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
        
        # 预处理图像
        self.input_tensor = self._preprocess_image(test_image)
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def _get_model_info(self, model: nn.Module) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
        }
    
    def test_pytorch_model(self, model_path: str, model_name: str) -> dict:
        """
        测试 PyTorch 模型
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
        
        Returns:
            测试结果
        """
        print(f"\nTesting {model_name}...")
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            return {
                'name': model_name,
                'path': model_path,
                'status': 'NOT_FOUND',
            }
        
        try:
            # 加载模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = resnet18(num_classes=10)
            
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            
            # 获取模型信息
            info = self._get_model_info(model)
            
            # 推理
            input_tensor = self.input_tensor.to(device)
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            
            # 计时
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # 后处理
            probs = torch.softmax(output, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
            # Top-5
            probs_np = probs.cpu().numpy().flatten()
            top5_indices = np.argsort(probs_np)[-5:][::-1]
            top5 = [(CIFAR10_CLASSES[i], probs_np[i]) for i in top5_indices]
            
            result = {
                'name': model_name,
                'path': model_path,
                'status': 'SUCCESS',
                'device': device,
                'total_params': info['total_params'],
                'model_size_mb': info['model_size_mb'],
                'predicted_class': CIFAR10_CLASSES[pred_class.item()],
                'predicted_id': pred_class.item(),
                'confidence': confidence.item(),
                'inference_time_ms': inference_time,
                'top5': top5,
            }
            
            print(f"  Status: SUCCESS")
            print(f"  Predicted: {result['predicted_class']} (ID: {result['predicted_id']})")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
            print(f"  Model size: {result['model_size_mb']:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"  Error: {e}")
            return {
                'name': model_name,
                'path': model_path,
                'status': 'ERROR',
                'error': str(e),
            }
    
    def test_quantized_model(self, model_path: str, model_name: str) -> dict:
        """
        测试量化模型
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
        
        Returns:
            测试结果
        """
        print(f"\nTesting {model_name}...")
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            return {
                'name': model_name,
                'path': model_path,
                'status': 'NOT_FOUND',
            }
        
        try:
            # 加载量化模型
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 创建基础模型并加载权重
            model = resnet18(num_classes=10)
            model.eval()
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # 检查是否是量化后的权重
                is_quantized = any('_packed_params' in k or 'scale' in k for k in state_dict.keys())
                
                if is_quantized:
                    # 旧格式量化模型，跳过
                    print(f"  Warning: Old quantized format, skipping...")
                    return {
                        'name': model_name,
                        'path': model_path,
                        'status': 'SKIPPED',
                        'error': 'Old quantized format not supported',
                    }
                else:
                    model.load_state_dict(state_dict)
            else:
                model.load_state_dict(checkpoint)
            
            # 应用动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
            quantized_model.eval()
            
            # 获取模型信息
            info = self._get_model_info(quantized_model)
            
            # 推理
            input_tensor = self.input_tensor
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    _ = quantized_model(input_tensor)
            
            # 计时
            start_time = time.time()
            
            with torch.no_grad():
                output = quantized_model(input_tensor)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # 后处理
            probs = torch.softmax(output, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
            # Top-5
            probs_np = probs.numpy().flatten()
            top5_indices = np.argsort(probs_np)[-5:][::-1]
            top5 = [(CIFAR10_CLASSES[i], probs_np[i]) for i in top5_indices]
            
            result = {
                'name': model_name,
                'path': model_path,
                'status': 'SUCCESS',
                'device': 'cpu',
                'total_params': info['total_params'],
                'model_size_mb': info['model_size_mb'],
                'predicted_class': CIFAR10_CLASSES[pred_class.item()],
                'predicted_id': pred_class.item(),
                'confidence': confidence.item(),
                'inference_time_ms': inference_time,
                'top5': top5,
            }
            
            print(f"  Status: SUCCESS")
            print(f"  Predicted: {result['predicted_class']} (ID: {result['predicted_id']})")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
            print(f"  Model size: {result['model_size_mb']:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'name': model_name,
                'path': model_path,
                'status': 'ERROR',
                'error': str(e),
            }
    
    def test_onnx_model(self, onnx_path: str, model_name: str) -> dict:
        """
        测试 ONNX 模型
        
        Args:
            onnx_path: ONNX 模型路径
            model_name: 模型名称
        
        Returns:
            测试结果
        """
        print(f"\nTesting {model_name}...")
        
        if not os.path.exists(onnx_path):
            print(f"  Model not found: {onnx_path}")
            return {
                'name': model_name,
                'path': onnx_path,
                'status': 'NOT_FOUND',
            }
        
        try:
            import onnxruntime as ort
            
            # 创建推理会话
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 获取输入输出信息
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # 获取模型大小
            model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            
            # 推理
            input_tensor = self.input_tensor.numpy()
            
            # 预热
            for _ in range(10):
                _ = session.run([output_name], {input_name: input_tensor})
            
            # 计时
            start_time = time.time()
            output = session.run([output_name], {input_name: input_tensor})[0]
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # 后处理
            output = output.squeeze()
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / exp_output.sum()
            
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            # Top-5
            top5_indices = np.argsort(probs)[-5:][::-1]
            top5 = [(CIFAR10_CLASSES[i], probs[i]) for i in top5_indices]
            
            result = {
                'name': model_name,
                'path': onnx_path,
                'status': 'SUCCESS',
                'device': session.get_providers()[0],
                'model_size_mb': model_size_mb,
                'predicted_class': CIFAR10_CLASSES[pred_class],
                'predicted_id': int(pred_class),
                'confidence': float(confidence),
                'inference_time_ms': inference_time,
                'top5': top5,
            }
            
            print(f"  Status: SUCCESS")
            print(f"  Predicted: {result['predicted_class']} (ID: {result['predicted_id']})")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
            print(f"  Model size: {result['model_size_mb']:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"  Error: {e}")
            return {
                'name': model_name,
                'path': onnx_path,
                'status': 'ERROR',
                'error': str(e),
            }
    
    def run_all_tests(self) -> list:
        """运行所有测试"""
        print("=" * 60)
        print("Running All Model Tests")
        print("=" * 60)
        print(f"Test image: {self.test_image}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = []
        
        # 1. 测试原始模型
        result = self.test_pytorch_model(
            'weights/best_model.pth',
            'Best Model (Original)'
        )
        results.append(result)
        
        # 2. 测试剪枝模型
        result = self.test_pytorch_model(
            'weights/pruned_model.pth',
            'Pruned Model'
        )
        results.append(result)
        
        # 3. 测试蒸馏模型
        result = self.test_pytorch_model(
            'weights/distilled_model.pth',
            'Distilled Model'
        )
        results.append(result)
        
        # 4. 测试量化模型
        result = self.test_quantized_model(
            'weights/quantized_model_dynamic.pth',
            'Quantized Model (Dynamic)'
        )
        results.append(result)
        
        # 5. 测试 ONNX 模型
        result = self.test_onnx_model(
            'onnx_models/resnet18.onnx',
            'ONNX Model'
        )
        results.append(result)
        
        self.results = results
        return results
    
    def save_results(self, output_file: str = None):
        """保存结果到文件"""
        if output_file is None:
            output_file = self.output_file
        
        if not output_file:
            print("No output file specified")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Model Test Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Test image: {self.test_image}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            for result in self.results:
                f.write("-" * 60 + "\n")
                f.write(f"Model: {result['name']}\n")
                f.write(f"Path: {result['path']}\n")
                f.write(f"Status: {result['status']}\n")
                
                if result['status'] == 'SUCCESS':
                    f.write(f"Device: {result['device']}\n")
                    total_params = result.get('total_params', 'N/A')
                    if isinstance(total_params, int):
                        f.write(f"Total params: {total_params:,}\n")
                    else:
                        f.write(f"Total params: {total_params}\n")
                    f.write(f"Model size: {result['model_size_mb']:.2f} MB\n")
                    f.write(f"Predicted class: {result['predicted_class']} (ID: {result['predicted_id']})\n")
                    f.write(f"Confidence: {result['confidence']:.4f}\n")
                    f.write(f"Inference time: {result['inference_time_ms']:.2f} ms\n")
                    f.write("Top-5 predictions:\n")
                    for i, (cls, prob) in enumerate(result['top5']):
                        f.write(f"  {i+1}. {cls}: {prob:.4f}\n")
                elif result['status'] == 'ERROR':
                    f.write(f"Error: {result['error']}\n")
                elif result['status'] == 'NOT_FOUND':
                    f.write("Model file not found\n")
                
                f.write("\n")
            
            # 汇总表格
            f.write("=" * 60 + "\n")
            f.write("Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Model':<25} {'Status':<10} {'Predicted':<15} {'Conf':<8} {'Time(ms)':<10} {'Size(MB)':<10}\n")
            f.write("-" * 80 + "\n")
            
            for result in self.results:
                if result['status'] == 'SUCCESS':
                    f.write(f"{result['name']:<25} {result['status']:<10} {result['predicted_class']:<15} {result['confidence']:<8.4f} {result['inference_time_ms']:<10.2f} {result['model_size_mb']:<10.2f}\n")
                else:
                    f.write(f"{result['name']:<25} {result['status']:<10}\n")
        
        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Test All Models')
    parser.add_argument('--image', type=str, default='test.jpg', help='Test image path')
    parser.add_argument('--output', type=str, default='test_results.txt', help='Output file path')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    tester = ModelTester(args.image, args.output)
    tester.run_all_tests()
    tester.save_results()


if __name__ == '__main__':
    main()