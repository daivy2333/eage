"""
ResNet-18 模型定义
适配 CIFAR-10 数据集 (32x32 图像)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type, Union
from collections import OrderedDict


class BasicBlock(nn.Module):
    """
    ResNet 基础块
    用于 ResNet-18 和 ResNet-34
    """
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
            downsample: 下采样层
            norm_layer: 归一化层
            dropout: Dropout 比例
        """
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        
        # 下采样
        self.downsample = downsample
        self.stride = stride
        
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNet(nn.Module):
    """
    ResNet 模型
    支持 ResNet-18 和 ResNet-34
    """
    
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        norm_layer: Optional[nn.Module] = None,
        dropout: float = 0.0,
        width_mult: float = 1.0,
    ):
        """
        Args:
            block: 残差块类型
            layers: 每个阶段的块数量
            num_classes: 类别数量
            zero_init_residual: 是否将残差分支初始化为零
            norm_layer: 归一化层
            dropout: Dropout 比例
            width_mult: 宽度乘数（用于剪枝）
        """
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.dropout = dropout
        self.width_mult = width_mult
        
        # 初始通道数
        self.in_channels = int(64 * width_mult)
        
        # 初始卷积层 (适配 CIFAR-10 的 32x32 图像)
        # 使用 3x3 卷积代替 7x7，步长为 1
        self.conv1 = nn.Conv2d(
            3, self.in_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 不使用 max pooling (CIFAR-10 图像太小)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(block, int(64 * width_mult), layers[0])
        self.layer2 = self._make_layer(block, int(128 * width_mult), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * width_mult), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * width_mult), layers[3], stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(int(512 * width_mult) * block.expansion, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 零初始化残差分支
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        构建残差层
        
        Args:
            block: 残差块类型
            out_channels: 输出通道数
            blocks: 块数量
            stride: 步长
        
        Returns:
            残差层
        """
        norm_layer = self._norm_layer
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(
            self.in_channels, out_channels,
            stride=stride, downsample=downsample,
            norm_layer=norm_layer, dropout=self.dropout
        ))
        
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels, out_channels,
                norm_layer=norm_layer, dropout=self.dropout
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征向量（不经过分类头）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_channel_config(self) -> List[int]:
        """获取各层的通道配置（用于剪枝）"""
        config = []
        
        # conv1
        config.append(self.conv1.out_channels)
        
        # layer1-4
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                config.append(block.conv1.out_channels)
                config.append(block.conv2.out_channels)
        
        return config


def resnet18(
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs
) -> ResNet:
    """
    构建 ResNet-18 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        **kwargs: 其他参数
    
    Returns:
        ResNet-18 模型
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    
    if pretrained:
        # 加载预训练权重
        # 注意：CIFAR-10 的 ResNet-18 与 ImageNet 的结构不同
        # 这里不自动加载预训练权重
        pass
    
    return model


def resnet34(
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs
) -> ResNet:
    """
    构建 ResNet-34 模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        **kwargs: 其他参数
    
    Returns:
        ResNet-34 模型
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    
    return model


class ResNetWithFeatures(ResNet):
    """
    带有中间特征输出的 ResNet
    用于知识蒸馏
    """
    
    def forward_with_features(self, x: torch.Tensor) -> tuple:
        """
        前向传播并返回中间特征
        
        Args:
            x: 输入张量
        
        Returns:
            (logits, features)
        """
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x
        
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features['avgpool'] = x
        
        logits = self.fc(x)
        
        return logits, features


def resnet18_with_features(
    num_classes: int = 10,
    **kwargs
) -> ResNetWithFeatures:
    """
    构建带有中间特征输出的 ResNet-18 模型
    
    Args:
        num_classes: 类别数量
        **kwargs: 其他参数
    
    Returns:
        ResNet-18 模型
    """
    model = ResNetWithFeatures(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    # 测试模型
    print("Testing ResNet-18 model...")
    
    # 创建模型
    model = resnet18(num_classes=10)
    
    # 打印模型结构
    print(model)
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试特征提取
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    print("Model test passed!")