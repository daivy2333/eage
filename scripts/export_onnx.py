#!/usr/bin/env python
"""
ONNX 导出脚本
将 PyTorch 模型导出为 ONNX 格式
"""
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs import get_config
from models import resnet18
from export import export_to_onnx, get_onnx_model_info, verify_onnx_inference


def main():
    parser = argparse.ArgumentParser(description='Export ResNet-18 to ONNX')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output-path', type=str, default='onnx_models/resnet18.onnx', help='Output path')
    parser.add_argument('--opset-version', type=int, default=14, help='ONNX opset version')
    parser.add_argument('--dynamic-batch', action='store_true', help='Enable dynamic batch')
    parser.add_argument('--optimize', action='store_true', help='Optimize ONNX model')
    parser.add_argument('--verify', action='store_true', help='Verify ONNX inference')
    args = parser.parse_args()
    
    config = get_config()
    
    print("=" * 60)
    print("Exporting ResNet-18 to ONNX")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Opset version: {args.opset_version}")
    print(f"Dynamic batch: {args.dynamic_batch}")
    print(f"Optimize: {args.optimize}")
    print("=" * 60)
    
    # 加载模型
    print("\nLoading model...")
    model = resnet18(num_classes=config.data.num_classes)
    
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 导出 ONNX
    print("\nExporting to ONNX...")
    exported_path = export_to_onnx(
        model=model,
        output_path=args.output_path,
        input_size=(1, 3, 32, 32),
        opset_version=args.opset_version,
        dynamic_batch=args.dynamic_batch,
        optimize=args.optimize,
    )
    
    # 获取模型信息
    print("\nONNX Model Info:")
    info = get_onnx_model_info(exported_path)
    if info:
        print(f"  Inputs: {info['inputs']}")
        print(f"  Outputs: {info['outputs']}")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
        print(f"  Opset version: {info['opset_version']}")
    
    # 验证推理
    if args.verify:
        print("\nVerifying ONNX inference...")
        test_input = torch.randn(1, 3, 32, 32)
        verify_onnx_inference(exported_path, model, test_input)
    
    print(f"\nONNX model exported to {exported_path}")


if __name__ == '__main__':
    main()