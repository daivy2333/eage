"""
INT8 量化模块
实现 Post-training Quantization (PTQ)
"""
import os
import copy
from typing import Optional, Dict, Any, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import get_config


class QuantWrapper(nn.Module):
    """
    量化包装器
    为模型添加 QuantStub 和 DeQuantStub
    """
    def __init__(self, model: nn.Module):
        super(QuantWrapper, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class QuantizationConfig:
    """量化配置"""
    
    def __init__(
        self,
        qconfig: str = 'fbgemm',  # fbgemm (x86) 或 qnnpack (ARM)
        qmethod: str = 'static',  # static 或 dynamic
        calibration_method: str = 'minmax',  # minmax, histogram, entropy
    ):
        self.qconfig = qconfig
        self.qmethod = qmethod
        self.calibration_method = calibration_method


class PostTrainingQuantizer:
    """
    Post-training Quantization (PTQ) 量化器
    """
    
    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Optional[DataLoader] = None,
        qconfig: str = 'fbgemm',
        device: str = 'cpu',
    ):
        """
        Args:
            model: 待量化模型
            calibration_loader: 校准数据加载器
            qconfig: 量化配置 ('fbgemm' for x86, 'qnnpack' for ARM)
            device: 设备
        """
        self.model = model
        self.calibration_loader = calibration_loader
        self.qconfig_str = qconfig
        self.device = device
        
        # 量化配置
        self.qconfig = torch.quantization.get_default_qconfig(qconfig)
        
        # 量化后的模型
        self.quantized_model = None
    
    def prepare_model(self) -> nn.Module:
        """
        准备模型进行量化
        添加量化观察器
        
        Returns:
            准备好的模型
        """
        # 创建模型副本
        model = copy.deepcopy(self.model)
        model.eval()
        
        # 融合模块 (Conv + BN + ReLU)
        model = self._fuse_modules(model)
        
        # 使用 QuantWrapper 包装模型
        # 这会添加 QuantStub 和 DeQuantStub 来处理输入/输出的量化/反量化
        model = QuantWrapper(model)
        
        # 设置量化配置
        model.qconfig = self.qconfig
        
        # 准备量化
        torch.quantization.prepare(model, inplace=True)
        
        return model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        融合模块 (Conv + BN + ReLU)
        针对 ResNet 结构进行优化，处理非 Sequential 的层结构
        
        Args:
            model: 模型
        
        Returns:
            融合后的模型
        """
        modules_to_fuse = []
        
        # 1. 检查模型是否有 conv1, bn1, relu 层（ResNet 主模型结构）
        if hasattr(model, 'conv1') and hasattr(model, 'bn1'):
            if hasattr(model, 'relu'):
                # 融合 conv1 + bn1 + relu
                modules_to_fuse.append(('conv1', 'bn1', 'relu'))
            else:
                modules_to_fuse.append(('conv1', 'bn1'))
        
        # 2. 遍历所有模块，处理 BasicBlock 中的层
        for name, module in model.named_modules():
            # 跳过主模型本身
            if name == '':
                continue
            
            # 检查是否是 BasicBlock 或类似的残差块结构
            if hasattr(module, 'conv1') and hasattr(module, 'bn1'):
                # BasicBlock: conv1 + bn1
                modules_to_fuse.append((f'{name}.conv1', f'{name}.bn1'))
            
            if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
                # BasicBlock: conv2 + bn2
                modules_to_fuse.append((f'{name}.conv2', f'{name}.bn2'))
            
            # 处理 Sequential 结构（保持向后兼容）
            if isinstance(module, nn.Sequential):
                if len(module) >= 2:
                    if isinstance(module[0], nn.Conv2d):
                        if isinstance(module[1], nn.BatchNorm2d):
                            if len(module) >= 3 and isinstance(module[2], nn.ReLU):
                                modules_to_fuse.append((f'{name}.0', f'{name}.1', f'{name}.2'))
                            else:
                                modules_to_fuse.append((f'{name}.0', f'{name}.1'))
        
        # 融合模块
        if modules_to_fuse:
            try:
                model = torch.quantization.fuse_modules(model, modules_to_fuse)
                print(f"Fused modules: {modules_to_fuse}")
            except Exception as e:
                print(f"Warning: Failed to fuse some modules: {e}")
        
        return model
    
    def calibrate(self, model: nn.Module, num_batches: int = 100):
        """
        校准模型
        运行校准数据以确定量化参数
        
        Args:
            model: 准备好的模型
            num_batches: 校准批数
        """
        model.eval()
        
        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(self.calibration_loader, desc='Calibrating')):
                if i >= num_batches:
                    break
                images = images.to(self.device)
                model(images)
    
    def quantize(self, num_calibration_batches: int = 100) -> nn.Module:
        """
        执行量化
        
        Args:
            num_calibration_batches: 校准批数
        
        Returns:
            量化后的模型
        """
        print(f"Starting post-training quantization with {self.qconfig_str} backend...")
        
        # 准备模型
        model = self.prepare_model()
        
        # 校准
        if self.calibration_loader is not None:
            self.calibrate(model, num_calibration_batches)
        
        # 转换为量化模型
        torch.quantization.convert(model, inplace=True)
        
        self.quantized_model = model
        
        print("Quantization completed!")
        
        return model
    
    def save_quantized_model(self, path: str):
        """
        保存量化模型
        
        Args:
            path: 保存路径
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantize() first.")
        
        torch.save(self.quantized_model.state_dict(), path)
        print(f"Quantized model saved to {path}")
    
    def load_quantized_model(self, path: str) -> nn.Module:
        """
        加载量化模型
        
        Args:
            path: 模型路径
        
        Returns:
            加载的模型
        """
        self.quantized_model.load_state_dict(torch.load(path, weights_only=False))
        return self.quantized_model


class DynamicQuantizer:
    """
    动态量化器
    权重预先量化，激活在推理时动态量化
    """
    
    def __init__(
        self,
        model: nn.Module,
        qconfig: str = 'fbgemm',
    ):
        """
        Args:
            model: 待量化模型
            qconfig: 量化配置
        """
        self.model = model
        self.qconfig = qconfig
        self.quantized_model = None
    
    def quantize(self) -> nn.Module:
        """
        执行动态量化
        
        Returns:
            量化后的模型
        """
        print(f"Starting dynamic quantization with {self.qconfig} backend...")
        
        # 动态量化
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        print("Dynamic quantization completed!")
        
        return self.quantized_model


class QuantizationAwareTrainer:
    """
    量化感知训练 (QAT)
    在训练过程中模拟量化效果
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        qconfig: str = 'fbgemm',
        device: str = 'cuda',
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            qconfig: 量化配置
            device: 设备
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.qconfig = qconfig
        self.device = device
    
    def prepare_model(self) -> nn.Module:
        """
        准备模型进行量化感知训练
        
        Returns:
            准备好的模型
        """
        model = copy.deepcopy(self.model)
        
        # 融合模块
        model = self._fuse_modules(model)
        
        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.qconfig)
        
        # 准备量化感知训练
        torch.quantization.prepare_qat(model, inplace=True)
        
        return model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        融合模块 (Conv + BN + ReLU)
        针对 ResNet 结构进行优化，处理非 Sequential 的层结构
        
        Args:
            model: 模型
        
        Returns:
            融合后的模型
        """
        modules_to_fuse = []
        
        # 1. 检查模型是否有 conv1, bn1, relu 层（ResNet 主模型结构）
        if hasattr(model, 'conv1') and hasattr(model, 'bn1'):
            if hasattr(model, 'relu'):
                # 融合 conv1 + bn1 + relu
                modules_to_fuse.append(('conv1', 'bn1', 'relu'))
            else:
                modules_to_fuse.append(('conv1', 'bn1'))
        
        # 2. 遍历所有模块，处理 BasicBlock 中的层
        for name, module in model.named_modules():
            # 跳过主模型本身
            if name == '':
                continue
            
            # 检查是否是 BasicBlock 或类似的残差块结构
            if hasattr(module, 'conv1') and hasattr(module, 'bn1'):
                # BasicBlock: conv1 + bn1
                modules_to_fuse.append((f'{name}.conv1', f'{name}.bn1'))
            
            if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
                # BasicBlock: conv2 + bn2
                modules_to_fuse.append((f'{name}.conv2', f'{name}.bn2'))
            
            # 处理 Sequential 结构（保持向后兼容）
            if isinstance(module, nn.Sequential):
                if len(module) >= 2:
                    if isinstance(module[0], nn.Conv2d):
                        if isinstance(module[1], nn.BatchNorm2d):
                            if len(module) >= 3 and isinstance(module[2], nn.ReLU):
                                modules_to_fuse.append((f'{name}.0', f'{name}.1', f'{name}.2'))
                            else:
                                modules_to_fuse.append((f'{name}.0', f'{name}.1'))
        
        # 融合模块
        if modules_to_fuse:
            try:
                model = torch.quantization.fuse_modules(model, modules_to_fuse)
                print(f"Fused modules: {modules_to_fuse}")
            except Exception as e:
                print(f"Warning: Failed to fuse some modules: {e}")
        
        return model
    
    def train(self, epochs: int = 10) -> nn.Module:
        """
        执行量化感知训练
        
        Args:
            epochs: 训练轮数
        
        Returns:
            训练后的模型
        """
        model = self.prepare_model()
        model.to(self.device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            
            for images, labels in tqdm(self.train_loader, desc=f'QAT Epoch {epoch + 1}'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            print(f"Epoch {epoch + 1}: Test Accuracy = {acc:.2f}%")
        
        # 转换为量化模型
        model.cpu()
        torch.quantization.convert(model, inplace=True)
        
        return model


def quantize_model_static(
    model: nn.Module,
    calibration_loader: DataLoader,
    qconfig: str = 'fbgemm',
    num_calibration_batches: int = 100,
) -> nn.Module:
    """
    静态量化模型的便捷函数
    
    Args:
        model: 模型
        calibration_loader: 校准数据加载器
        qconfig: 量化配置
        num_calibration_batches: 校准批数
    
    Returns:
        量化后的模型
    """
    quantizer = PostTrainingQuantizer(
        model=model,
        calibration_loader=calibration_loader,
        qconfig=qconfig,
    )
    
    return quantizer.quantize(num_calibration_batches)


def quantize_model_dynamic(
    model: nn.Module,
    qconfig: str = 'fbgemm',
) -> nn.Module:
    """
    动态量化模型的便捷函数
    
    Args:
        model: 模型
        qconfig: 量化配置
    
    Returns:
        量化后的模型
    """
    quantizer = DynamicQuantizer(model=model, qconfig=qconfig)
    return quantizer.quantize()


def compare_model_size(original_model: nn.Module, quantized_model: nn.Module) -> Dict[str, float]:
    """
    比较模型大小
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
    
    Returns:
        大小比较信息
    """
    def get_model_size(model: nn.Module) -> float:
        """获取模型大小 (MB)"""
        torch.save(model.state_dict(), 'temp_model.pth')
        size = os.path.getsize('temp_model.pth') / (1024 * 1024)
        os.remove('temp_model.pth')
        return size
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': original_size / quantized_size,
        'size_reduction': (1 - quantized_size / original_size) * 100,
    }


if __name__ == "__main__":
    # 测试量化
    print("Testing quantization...")
    
    from models import resnet18
    from data import get_calibration_dataloader
    
    # 创建模型
    model = resnet18(num_classes=10)
    model.eval()
    
    # 获取校准数据加载器
    calib_loader = get_calibration_dataloader(num_samples=100)
    
    # 测试静态量化
    print("\nTesting static quantization...")
    quantizer = PostTrainingQuantizer(
        model=model,
        calibration_loader=calib_loader,
        qconfig='fbgemm',
    )
    
    quantized_model = quantizer.quantize(num_calibration_batches=10)
    
    # 比较模型大小
    size_info = compare_model_size(model, quantized_model)
    print(f"\nModel size comparison:")
    print(f"  Original: {size_info['original_size_mb']:.2f} MB")
    print(f"  Quantized: {size_info['quantized_size_mb']:.2f} MB")
    print(f"  Compression ratio: {size_info['compression_ratio']:.2f}x")
    print(f"  Size reduction: {size_info['size_reduction']:.1f}%")
    
    print("\nQuantization test passed!")