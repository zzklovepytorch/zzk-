# ==============================
# dataset.py —— 数据加载与预处理
# ==============================

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# ---- 6 个场景类别（顺序与文件夹排列一致）----
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_NAMES_CN = ['建筑', '森林', '冰川', '山脉', '海洋', '街道']   # 中文对照，方便显示

# Intel Image 数据集的均值和标准差（根据数据集统计得到）
MEAN = (0.4302, 0.4575, 0.4539)
STD  = (0.2608, 0.2555, 0.2744)


def get_data_loaders(data_dir='./data', batch_size=32, val_split=0.1):
    """
    加载 Intel Image Classification 数据集，返回训练、验证、测试集的 DataLoader。

    参数:
        data_dir   : 数据集根目录（里面有 seg_train 和 seg_test 两个文件夹）
        batch_size : 每批图片数量
        val_split  : 从训练集中划分出多少比例作为验证集（默认 10%）

    返回:
        train_loader : 训练集加载器
        val_loader   : 验证集加载器
        test_loader  : 测试集加载器
    """

    # --- 训练集的数据增强 transform ---
    # 数据增强：通过随机变换让模型见到更多样的图片，提升泛化能力
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),             # 统一缩放到 150×150
        transforms.RandomHorizontalFlip(),          # 随机水平翻转（50% 概率）
        transforms.RandomVerticalFlip(p=0.1),       # 轻微垂直翻转（10% 概率）
        transforms.RandomRotation(15),              # 随机旋转 ±15 度
        transforms.ColorJitter(                     # 随机调整亮度、对比度、饱和度
            brightness=0.3, contrast=0.3, saturation=0.2
        ),
        transforms.ToTensor(),                      # 把 PIL 图片转成 Tensor，像素值 [0,255]→[0,1]
        transforms.Normalize(mean=MEAN, std=STD)    # 标准化：让每个通道的数值更均匀，加速收敛
    ])

    # --- 验证/测试集不做随机变换（保证评估结果稳定）---
    eval_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # ---- 构建训练集和测试集路径 ----
    # ImageFolder 会自动把每个子文件夹名当作类别标签
    train_dir = os.path.join(data_dir, 'seg_train', 'seg_train')
    test_dir  = os.path.join(data_dir, 'seg_test',  'seg_test')

    # 检查数据集目录是否存在，不存在则给出提示
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"找不到训练集目录：{train_dir}\n"
            "请先从 Kaggle 下载 Intel Image Classification 数据集，\n"
            "下载地址：https://www.kaggle.com/datasets/puneet6060/intel-image-classification\n"
            "下载后解压，把 seg_train 和 seg_test 文件夹放到 data/ 目录下。"
        )

    # 加载原始训练集（先不做增强，后面再覆盖 transform）
    full_train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir, transform=train_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_dir, transform=eval_transform
    )

    # ---- 从训练集中划分出验证集 ----
    # 例如 val_split=0.1 表示把 10% 的训练数据拿去做验证
    total = len(full_train_dataset)
    val_size   = int(total * val_split)
    train_size = total - val_size

    # random_split 随机分割数据集（注意：分割后 transform 以原始数据集为准）
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 验证集的 transform 应该和测试集一样（不做随机变换）
    # 这里通过包装数据集来替换 transform
    val_dataset.dataset = torchvision.datasets.ImageFolder(
        root=train_dir, transform=eval_transform
    )

    # ---- 创建 DataLoader ----
    # DataLoader 负责按 batch_size 分批送入模型，并支持多线程加速
    # 判断是否使用 GPU（pin_memory 只有在 GPU 时才有用）
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin
    )

    # 打印数据集信息
    print(f"训练集大小: {train_size} 张图片")
    print(f"验证集大小: {val_size} 张图片")
    print(f"测试集大小: {len(test_dataset)} 张图片")
    print(f"每个 batch 大小: {batch_size}")
    print(f"类别: {full_train_dataset.classes}")

    return train_loader, val_loader, test_loader
