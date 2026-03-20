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
    parser.add_argument('--output-path', type=str, default='weights/quantized_model.pt', help='Output path')
    parser.add_argument('--save-format', type=str, default='torchscript', choices=['torchscript', 'legacy'], 
                        help='Save format: torchscript (recommended) or legacy')
    args = parser.parse_args()
    
    config = get_config()
    
    print("=" * 60)
    print("Quantizing ResNet-18 Model")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Quantization method: {args.qmethod}")
    print(f"Quantization config: {args.qconfig}")
    print(f"Calibration samples: {args.calibration_samples}")
    print(f"Save format: {args.save_format}")
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
    
    # 设置量化引擎（确保与 qconfig 匹配）
    torch.backends.quantized.engine = args.qconfig
    
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
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    
    if args.save_format == 'torchscript':
        # 使用 TorchScript 保存（推荐方式）
        print("\nSaving quantized model using TorchScript...")
        
        # 创建示例输入用于 trace
        example_input = torch.randn(1, 3, config.data.image_size, config.data.image_size)
        
        try:
            # 使用 torch.jit.trace 保存量化模型
            traced_model = torch.jit.trace(quantized_model, example_input)
            traced_model.save(args.output_path)
            print(f"TorchScript model saved to {args.output_path}")
            
            # 验证保存的模型可以正确加载
            loaded_model = torch.jit.load(args.output_path)
            with torch.no_grad():
                test_output = loaded_model(example_input)
            print(f"Model verification passed. Output shape: {test_output.shape}")
            
        except Exception as e:
            print(f"Warning: TorchScript trace failed: {e}")
            print("Falling back to legacy format...")
            
            # 回退到旧格式
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'quantized': True,
                'qmethod': args.qmethod,
                'qconfig': args.qconfig,
                'num_classes': config.data.num_classes,
            }, args.output_path)
            print(f"Model saved in legacy format to {args.output_path}")
    else:
        # 旧格式保存（保存原始权重，便于后续加载）
        print("\nSaving quantized model in legacy format...")
        
        # 对于动态量化，保存原始模型的权重（因为量化后的权重结构不同）
        # 这样加载时可以先加载原始权重，再应用动态量化
        torch.save({
            'model_state_dict': model.state_dict(),  # 保存原始模型权重
            'quantized': True,
            'qmethod': args.qmethod,
            'qconfig': args.qconfig,
            'num_classes': config.data.num_classes,
        }, args.output_path)
        print(f"Model saved to {args.output_path}")
        print("Note: Legacy format saves original weights. The model will be re-quantized on load.")
    
    print("\n" + "=" * 60)
    print("Quantization completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()