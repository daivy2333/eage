#!/usr/bin/env python
"""
量化脚本
对模型进行 INT8 量化
"""
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs import get_config
from models import resnet18
from data import get_calibration_dataloader
from quantization import PostTrainingQuantizer, compare_model_size


def main():
    parser = argparse.ArgumentParser(description='Quantize ResNet-18 model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--qmethod', type=str, default='static', choices=['static', 'dynamic'], help='Quantization method')
    parser.add_argument('--qconfig', type=str, default='fbgemm', choices=['fbgemm', 'qnnpack'], help='Quantization config')
    parser.add_argument('--calibration-samples', type=int, default=1000, help='Number of calibration samples')
    parser.add_argument('--output-path', type=str, default='weights/quantized_model.pth', help='Output path')
    args = parser.parse_args()
    
    config = get_config()
    
    print("=" * 60)
    print("Quantizing ResNet-18 Model")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Quantization method: {args.qmethod}")
    print(f"Quantization config: {args.qconfig}")
    print(f"Calibration samples: {args.calibration_samples}")
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
    
    # 获取校准数据加载器
    print("\nPreparing calibration data...")
    calib_loader = get_calibration_dataloader(
        num_samples=args.calibration_samples
    )
    
    # 量化
    print(f"\nPerforming {args.qmethod} quantization...")
    
    if args.qmethod == 'static':
        quantizer = PostTrainingQuantizer(
            model=model,
            calibration_loader=calib_loader,
            qconfig=args.qconfig,
            device='cpu',
        )
        quantized_model = quantizer.quantize(num_calibration_batches=args.calibration_samples // 32)
    else:
        from quantization import DynamicQuantizer
        quantizer = DynamicQuantizer(model=model, qconfig=args.qconfig)
        quantized_model = quantizer.quantize()
    
    # 比较模型大小
    print("\nComparing model sizes...")
    size_info = compare_model_size(model, quantized_model)
    print(f"Original model size: {size_info['original_size_mb']:.2f} MB")
    print(f"Quantized model size: {size_info['quantized_size_mb']:.2f} MB")
    print(f"Compression ratio: {size_info['compression_ratio']:.2f}x")
    print(f"Size reduction: {size_info['size_reduction']:.1f}%")
    
    # 保存量化模型
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), args.output_path)
    
    print(f"\nQuantized model saved to {args.output_path}")


if __name__ == '__main__':
    main()