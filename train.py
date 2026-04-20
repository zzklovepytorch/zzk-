# ==============================
# train.py —— 训练主程序
# ==============================
# 【使用说明】
#   1. 先下载 Intel Image Classification 数据集（见 README.md）
#   2. 直接在 PyCharm 中运行本文件，或在终端执行：python train.py
#   3. 训练完成后，模型权重保存到 models/best_model.pth
#              Loss/Accuracy 曲线保存到 results/training_curves.png

import os
# 解决 Anaconda 环境下 OpenMP 库冲突的问题（多个库都自带了 libiomp5md.dll）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt


# 设置中文字体，让图表标签显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

from dataset import get_data_loaders
from model import get_model


# ==================================================
# ★ 超参数配置（可以根据需求修改这里的数值）
# ==================================================
MODEL_NAME   = 'resnet18'    # 选择模型：'resnet18'（推荐）或 'simplecnn'
PRETRAINED   = True          # 是否使用 ImageNet 预训练权重（True 效果更好）
EPOCHS       = 20           # 训练轮数
BATCH_SIZE   = 32            # 每批图片数量（显存不够时改小，如 16）
LEARNING_RATE = 0.001        # 初始学习率
DATA_DIR     = './data'      # 数据集根目录
MODEL_SAVE_PATH = './models/best_model.pth'   # 模型保存路径
RESULTS_DIR  = './results'   # 结果保存目录


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个 epoch（完整遍历一次训练集）。
    返回：(平均损失, 准确率%)
    """
    model.train()   # 切换到"训练模式"（启用 Dropout、BatchNorm 的训练行为）
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):   #类似于dataloader
        # 把数据搬到 GPU 或 CPU
        images, labels = images.to(device), labels.to(device)

        # ===== 核心训练五步=====
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # ==================================

        # 累积统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)      # 取概率最大的类别索引
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 每 50 个 batch 打印一次进度
        if (batch_idx + 1) % 50 == 0:
            print(f"    [{batch_idx + 1:3d}/{len(loader)}] "
                  f"Loss: {total_loss / (batch_idx + 1):.4f}  "
                  f"Acc: {100. * correct / total:.1f}%")

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    在验证集或测试集上评估模型。
    返回：(平均损失, 准确率%)
    """
    model.eval()    # 切换到"评估模式"（关闭 Dropout，BatchNorm 用全局统计值）
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():   # 评估时不需要梯度，节省内存
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def plot_training_curves(history, save_path):
    """
    绘制并保存 Loss 和 Accuracy 的训练曲线。

    参数:
        history   : 字典，包含 'train_loss', 'val_loss', 'train_acc', 'val_acc' 列表
        save_path : 图片保存路径
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('训练过程曲线', fontsize=16, fontweight='bold')

    # ---- 左图：Loss 曲线 ----
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='训练集 Loss')
    ax1.plot(epochs, history['val_loss'],   'r-s', markersize=4, label='验证集 Loss')
    ax1.set_title('损失（Loss）变化曲线')
    ax1.set_xlabel('Epoch（训练轮数）')
    ax1.set_ylabel('Loss（损失值）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)   # 添加半透明网格，方便读数

    # ---- 右图：Accuracy 曲线 ----
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='训练集准确率')
    ax2.plot(epochs, history['val_acc'],   'r-s', markersize=4, label='验证集准确率')
    ax2.set_title('准确率（Accuracy）变化曲线')
    ax2.set_xlabel('Epoch（训练轮数）')
    ax2.set_ylabel('准确率（%）')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 训练曲线已保存到: {save_path}")
    plt.close()   # 关闭图形，释放内存


def main():
    # ---- 1. 准备目录 ----
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 2. 选择运算设备 ----
    # CUDA：NVIDIA GPU，速度比 CPU 快约 10~50 倍
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*50}")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}")

    # ---- 3. 加载数据集 ----
    print("\n正在加载数据集...")
    train_loader, val_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)

    # ---- 4. 创建模型 ----
    print("\n正在初始化模型...")
    model = get_model(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device)   # 把模型参数搬到CPU

    # ---- 5. 损失函数 & 优化器 ----
    # CrossEntropyLoss：多分类问题的标准损失函数
    #   它内部会自动计算 softmax，所以模型输出不需要手动加 softmax
    criterion = nn.CrossEntropyLoss()

    # 根据不同模型选择不同学习率策略
    if MODEL_NAME == 'resnet18' and PRETRAINED:
        # 迁移学习时：新加的分类头用大学习率，预训练层用小学习率
        # 这样可以在保留已学特征的同时，快速适应新任务
        optimizer = optim.Adam([
            {'params': model.fc.parameters(),  'lr': LEARNING_RATE},       # 新分类层
            {'params': [p for name, p in model.named_parameters()
                        if 'fc' not in name],  'lr': LEARNING_RATE * 0.1}  # 预训练层（小学习率）
        ])
    else:
        # 从头训练：所有层使用同一学习率
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率调度器：每 7 个 epoch 把学习率乘以 0.5
    # 效果：训练前期快速下降，后期精细调整
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # ---- 6. 开始训练 ----
    print(f"\n{'='*50}")
    print(f"开始训练！模型: {MODEL_NAME}  共 {EPOCHS} 轮")
    print(f"{'='*50}")

    best_val_acc = 0.0
    # 记录每轮的指标，用于最后绘图
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n【第 {epoch:2d}/{EPOCHS} 轮】  学习率: {current_lr:.6f}")

        # 训练一轮
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 在验证集上评估
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        # 更新学习率
        scheduler.step()

        # 记录历史数据
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印本轮结果
        print(f"  训练集 → Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  验证集 → Loss: {val_loss:.4f}  | Acc: {val_acc:.1f}%")

        # 如果验证集准确率创了新高，保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ 新最佳验证准确率！模型已保存到 {MODEL_SAVE_PATH}")

    # ---- 7. 训练结束，用测试集做最终评估 ----
    print(f"\n{'='*50}")
    print("训练完成！正在用测试集做最终评估...")
    # 加载训练过程中保存的最佳模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n最终结果：")
    print(f"  最佳验证准确率: {best_val_acc:.1f}%")
    print(f"  测试集准确率:   {test_acc:.1f}%")
    print(f"{'='*50}")

    # ---- 8. 绘制训练曲线 ----
    curve_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plot_training_curves(history, curve_path)

    print("\n全部完成！接下来可以运行 visualize.py 生成混淆矩阵 📊")


# Python 惯例：只有直接运行此文件时才执行 main()
if __name__ == '__main__':
    main()
