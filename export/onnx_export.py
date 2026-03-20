"""
ONNX 导出模块
支持导出 PyTorch 模型到 ONNX 格式
"""
import os
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import numpy as np

from configs import get_config


class ONNXExporter:
    """
    ONNX 模型导出器
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 32, 32),
        opset_version: int = 14,
        dynamic_batch: bool = True,
    ):
        """
        Args:
            model: 待导出的模型
            input_size: 输入尺寸 (B, C, H, W)
            opset_version: ONNX opset 版本
            dynamic_batch: 是否使用动态 batch
        """
        self.model = model
        self.input_size = input_size
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch
    
    def export(
        self,
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        optimize: bool = True,
    ) -> str:
        """
        导出模型到 ONNX
        
        Args:
            output_path: 输出路径
            input_names: 输入名称
            output_names: 输出名称
            optimize: 是否优化 ONNX 模型
        
        Returns:
            导出的 ONNX 文件路径
        """
        self.model.eval()
        
        # 创建输入张量
        dummy_input = torch.randn(self.input_size)
        
        # 设置动态 batch
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        
        # 默认输入输出名称
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        # 导出
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        
        print(f"Model exported to {output_path}")
        
        # 验证导出的模型
        self._validate_onnx_model(output_path)
        
        # 优化 ONNX 模型
        if optimize:
            output_path = self._optimize_onnx_model(output_path)
        
        return output_path
    
    def _validate_onnx_model(self, model_path: str) -> bool:
        """
        验证 ONNX 模型
        
        Args:
            model_path: 模型路径
        
        Returns:
            是否验证通过
        """
        try:
            import onnx
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            print("ONNX model validation passed!")
            return True
        except ImportError:
            print("Warning: onnx not installed, skipping validation")
            return False
        except Exception as e:
            print(f"ONNX model validation failed: {e}")
            return False
    
    def _optimize_onnx_model(self, model_path: str) -> str:
        """
        优化 ONNX 模型
        
        Args:
            model_path: 模型路径
        
        Returns:
            优化后的模型路径
        """
        try:
            import onnx
            from onnxsim import simplify
            
            # 加载模型
            model = onnx.load(model_path)
            
            # 简化模型
            model_simp, check = simplify(model)
            
            if check:
                # 保存优化后的模型
                optimized_path = model_path.replace('.onnx', '_optimized.onnx')
                onnx.save(model_simp, optimized_path)
                print(f"ONNX model optimized and saved to {optimized_path}")
                return optimized_path
            else:
                print("Warning: ONNX simplification check failed, using original model")
                return model_path
                
        except ImportError:
            print("Warning: onnx-simplifier not installed, skipping optimization")
            return model_path
        except Exception as e:
            print(f"Warning: ONNX optimization failed: {e}")
            return model_path


class ONNXQuantizedExporter:
    """
    量化模型 ONNX 导出器
    """
    
    def __init__(
        self,
        quantized_model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 32, 32),
    ):
        """
        Args:
            quantized_model: 量化后的模型
            input_size: 输入尺寸
        """
        self.quantized_model = quantized_model
        self.input_size = input_size
    
    def export(self, output_path: str) -> str:
        """
        导出量化模型到 ONNX
        
        Args:
            output_path: 输出路径
        
        Returns:
            导出的 ONNX 文件路径
        """
        self.quantized_model.eval()
        
        # 创建输入张量
        dummy_input = torch.randn(self.input_size)
        
        # 导出量化模型
        torch.onnx.export(
            self.quantized_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        print(f"Quantized model exported to {output_path}")
        
        return output_path


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    opset_version: int = 14,
    dynamic_batch: bool = True,
    optimize: bool = True,
) -> str:
    """
    导出模型到 ONNX 的便捷函数
    
    Args:
        model: 模型
        output_path: 输出路径
        input_size: 输入尺寸
        opset_version: ONNX opset 版本
        dynamic_batch: 是否使用动态 batch
        optimize: 是否优化
    
    Returns:
        导出的 ONNX 文件路径
    """
    exporter = ONNXExporter(
        model=model,
        input_size=input_size,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
    )
    
    return exporter.export(output_path, optimize=optimize)


def export_all_models(
    original_model: nn.Module,
    pruned_model: Optional[nn.Module] = None,
    quantized_model: Optional[nn.Module] = None,
    output_dir: str = 'onnx_models',
    model_name: str = 'resnet18',
) -> Dict[str, str]:
    """
    导出所有模型到 ONNX
    
    Args:
        original_model: 原始模型
        pruned_model: 剪枝模型
        quantized_model: 量化模型
        output_dir: 输出目录
        model_name: 模型名称
    
    Returns:
        各模型 ONNX 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # 导出原始模型
    original_path = os.path.join(output_dir, f'{model_name}_original.onnx')
    paths['original'] = export_to_onnx(original_model, original_path)
    
    # 导出剪枝模型
    if pruned_model is not None:
        pruned_path = os.path.join(output_dir, f'{model_name}_pruned.onnx')
        paths['pruned'] = export_to_onnx(pruned_model, pruned_path)
    
    # 导出量化模型
    if quantized_model is not None:
        quantized_path = os.path.join(output_dir, f'{model_name}_quantized.onnx')
        exporter = ONNXQuantizedExporter(quantized_model)
        paths['quantized'] = exporter.export(quantized_path)
    
    return paths


def verify_onnx_inference(
    onnx_path: str,
    pytorch_model: nn.Module,
    test_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    验证 ONNX 模型推理结果与 PyTorch 模型一致
    
    Args:
        onnx_path: ONNX 模型路径
        pytorch_model: PyTorch 模型
        test_input: 测试输入
        rtol: 相对误差容限
        atol: 绝对误差容限
    
    Returns:
        是否验证通过
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed, skipping verification")
        return False
    
    # PyTorch 推理
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX 推理
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None,
        {'input': test_input.numpy()}
    )[0]
    
    # 比较结果
    if np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
        print("ONNX inference verification passed!")
        return True
    else:
        max_diff = np.abs(pytorch_output - onnx_output).max()
        print(f"ONNX inference verification failed! Max difference: {max_diff}")
        return False


def get_onnx_model_info(onnx_path: str) -> Dict[str, Any]:
    """
    获取 ONNX 模型信息
    
    Args:
        onnx_path: ONNX 模型路径
    
    Returns:
        模型信息
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        # 获取输入信息
        inputs = []
        for inp in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in inp.type.tensor_type.shape.dim]
            inputs.append({
                'name': inp.name,
                'shape': shape,
            })
        
        # 获取输出信息
        outputs = []
        for out in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in out.type.tensor_type.shape.dim]
            outputs.append({
                'name': out.name,
                'shape': shape,
            })
        
        # 获取模型大小
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'model_size_mb': model_size,
            'opset_version': model.opset_import[0].version,
        }
        
    except ImportError:
        print("Warning: onnx not installed")
        return {}


if __name__ == "__main__":
    # 测试 ONNX 导出
    print("Testing ONNX export...")
    
    from models import resnet18
    
    # 创建模型
    model = resnet18(num_classes=10)
    model.eval()
    
    # 创建输出目录
    os.makedirs('onnx_models', exist_ok=True)
    
    # 导出模型
    output_path = 'onnx_models/resnet18_cifar10.onnx'
    exporter = ONNXExporter(
        model=model,
        input_size=(1, 3, 32, 32),
        opset_version=14,
        dynamic_batch=True,
    )
    
    exported_path = exporter.export(output_path)
    
    # 验证推理
    test_input = torch.randn(1, 3, 32, 32)
    verify_onnx_inference(exported_path, model, test_input)
    
    # 获取模型信息
    info = get_onnx_model_info(exported_path)
    print(f"\nONNX Model Info:")
    print(f"  Inputs: {info['inputs']}")
    print(f"  Outputs: {info['outputs']}")
    print(f"  Model Size: {info['model_size_mb']:.2f} MB")
    print(f"  Opset Version: {info['opset_version']}")
    
    print("\nONNX export test passed!")