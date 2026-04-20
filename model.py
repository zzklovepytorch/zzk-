# ==============================
# model.py —— CNN 模型定义
# ==============================
# 本文件提供两种模型可供选择：
#   1. SimpleCNN  ：自己从头搭建的简单卷积神经网络
#   2. ResNet18   ：经典的深度残差网络

# 卷积神经网络（CNN）的基本思路：
#   输入图片 → 卷积层（提取特征）→ 池化层（缩小尺寸）→ 全连接层（做分类）

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 6   # Intel Image Classification 共有 6 个场景类别


# =====================================================
# 模型一：自定义简单 CNN
# =====================================================
class SimpleCNN(nn.Module):
    """
    自定义简单卷积神经网络，适用于 Intel Image Classification（6 类场景）。

    网络结构（输入：3×150×150）：
        卷积块1: Conv(3→32)   → BN → ReLU → MaxPool  → 输出 32×75×75
        卷积块2: Conv(32→64)  → BN → ReLU → MaxPool  → 输出 64×37×37
        卷积块3: Conv(64→128) → BN → ReLU → MaxPool  → 输出 128×18×18
        卷积块4: Conv(128→256)→ BN → ReLU → MaxPool  → 输出 256×9×9
        AdaptiveAvgPool                               → 输出 256×4×4
        展平 → 256*4*4 = 4096
        全连接1: 4096 → 512 → ReLU → Dropout(0.5)
        全连接2: 512  → 6  （6 个类别的得分）
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        # ---- 卷积块 1 ----
        self.conv_block1 = nn.Sequential(
            # Conv2d(输入通道, 输出通道, 卷积核大小, padding=1 使输出尺寸不变)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # 批归一化：让每层输入的分布更稳定，加速训练
            nn.ReLU(inplace=True),# ReLU 激活：把负值置为0，给网络引入非线性
            nn.MaxPool2d(2)       # 最大池化：把特征图尺寸缩小一半 150→75
        )

        # ---- 卷积块 2 ----
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)       # 75→37（池化时向下取整）
        )

        # ---- 卷积块 3 ----
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)       # 37→18
        )

        # ---- 卷积块 4 ----
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)       # 18→9
        )

        # AdaptiveAvgPool：自适应平均池化，把任意尺寸的特征图统一成 4×4
        # 这样即使输入图片尺寸稍有变化，后面的全连接层尺寸也不会报错
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ---- 全连接分类器 ----
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 把 256×4×4 展平成一维向量（4096维）
            nn.Linear(256 * 4 * 4, 512),        # 全连接：4096维 → 512维
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                    # Dropout：训练时随机"关掉"50%神经元，防止过拟合
            nn.Linear(512, num_classes)         # 输出层：512维 → 6维（每类一个得分）
        )

    def forward(self, x):
        """
        前向传播：定义数据如何流过网络。
        x 的形状：(batch_size, 3, 150, 150)
        """
        x = self.conv_block1(x)    # → (batch, 32, 75, 75)
        x = self.conv_block2(x)    # → (batch, 64, 37, 37)
        x = self.conv_block3(x)    # → (batch, 128, 18, 18)
        x = self.conv_block4(x)    # → (batch, 256, 9, 9)
        x = self.adaptive_pool(x)  # → (batch, 256, 4, 4)
        x = self.classifier(x)     # → (batch, 6)
        return x


# =====================================================
# 模型二：ResNet18（迁移学习版）
# =====================================================
def get_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    """
    使用预训练的 ResNet18 模型进行迁移学习。

    参数:
        num_classes : 分类数量（Intel Image 为 6）
        pretrained  : 是否使用 ImageNet 预训练权重
    """
    if pretrained:
        # weights=DEFAULT 表示加载 ImageNet 上最优的预训练权重
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print("已加载 ResNet18 的 ImageNet 预训练权重")
    else:
        model = models.resnet18(weights=None)
        print("使用随机初始化的 ResNet18（未预训练）")

    # 替换最后的全连接层
    # ResNet18 原本的输出是 1000 类（ImageNet），我们改成 6 类
    in_features = model.fc.in_features   # 获取原全连接层的输入维度（512）
    model.fc = nn.Linear(in_features, num_classes)

    return model


# =====================================================
# 统一的模型获取接口
# =====================================================
def get_model(model_name='resnet18', pretrained=True):
    """
    根据名称返回对应模型。

    参数:
        model_name : 'simplecnn' 或 'resnet18'
        pretrained : 是否使用预训练权重（只对 resnet18 有效）
    """

    if model_name.lower() == 'simplecnn':
        model = SimpleCNN(num_classes=NUM_CLASSES)
        print("使用模型：SimpleCNN（自定义 CNN）")
    elif model_name.lower() == 'resnet18':
        model = get_resnet18(num_classes=NUM_CLASSES, pretrained=pretrained)
        print("使用模型：ResNet18")
    else:
        raise ValueError(f"未知模型名称: {model_name}，请选择 'simplecnn' 或 'resnet18'")

    # 统计并打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}    可训练参数量: {trainable_params:,}")

    return model


    # 单独运行此文件时，验证模型能否正常工作
if __name__ == '__main__':
    print("=" * 40)
    print("测试 SimpleCNN")
    print("=" * 40)
    model1 = get_model('simplecnn')
    # 用随机数据模拟一个 batch（4张 150×150 的彩色图片）
    dummy = torch.randn(4, 3, 150, 150)
    out = model1(dummy)
    print(f"输入形状: {dummy.shape}  →  输出形状: {out.shape}")   # 应为 (4, 6)

    print("\n" + "=" * 40)
    print("测试 ResNet18")
    print("=" * 40)
    model2 = get_model('resnet18', pretrained=False)
    out2 = model2(dummy)
    print(f"输入形状: {dummy.shape}  →  输出形状: {out2.shape}")  # 应为 (4, 6)
