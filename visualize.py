# ==============================
# visualize.py —— 可视化分析
# ==============================
# 本文件提供以下可视化功能：
#   1. 混淆矩阵（Confusion Matrix）：直观展示模型把哪类图片搞混了
#   2. 错误样本分析（Bad Case）：找出分类错误的图片，分析原因
#
# 【使用前提】必须先运行 train.py 训练好模型（models/best_model.pth 要存在）
# 【运行方式】在终端执行：python visualize.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report

matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from dataset import get_data_loaders, CLASS_NAMES, CLASS_NAMES_CN
from model import get_model

# ---- 配置 ----
DATA_DIR        = './data'
MODEL_PATH      = './models/best_model.pth'
RESULTS_DIR     = './results'
BATCH_SIZE      = 32
MODEL_NAME      = 'resnet18'   # 与 train.py 中的 MODEL_NAME 保持一致


def get_all_predictions(model, loader, device):
    """
    用模型对整个数据集做预测，收集所有真实标签和预测标签。

    返回:
        all_labels   : 真实标签列表
        all_preds    : 预测标签列表
        all_probs    : 各类别预测概率列表（用于 Bad Case 分析）
        all_images   : 原始图像 Tensor 列表（用于展示错误样本）
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []
    all_images = []

    with torch.no_grad():
        for images, labels in loader:
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            # softmax 把原始得分（logits）转换成概率（所有类别概率之和=1）
            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(images.numpy())   # 保存原始（未标准化）图像用于展示

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), np.array(all_images))


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    绘制并保存混淆矩阵。

    混淆矩阵怎么看：
        - 行 = 真实类别，列 = 预测类别
        - 对角线上的数 = 预测正确的数量（越大越好）
        - 非对角线的数 = 预测错误的数量（越小越好）
        - 某行中哪列的值最大（非对角线），就说明该类最容易被误判为那类

    参数:
        y_true    : 真实标签数组
        y_pred    : 预测标签数组
        save_path : 图片保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 归一化：把每行除以该行总数，得到比例（更直观）
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('混淆矩阵分析 (Confusion Matrix)', fontsize=16, fontweight='bold')

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_normalized],
        ['原始数量', '归一化比例（每行求和=1）'],
        ['d', '.2f']
    ):
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)

        # 设置坐标轴标签
        ax.set_xticks(range(len(CLASS_NAMES_CN)))
        ax.set_yticks(range(len(CLASS_NAMES_CN)))
        ax.set_xticklabels(CLASS_NAMES_CN, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(CLASS_NAMES_CN, fontsize=11)
        ax.set_xlabel('预测类别', fontsize=12)
        ax.set_ylabel('真实类别', fontsize=12)
        ax.set_title(title, fontsize=13)

        # 在每个格子中写上数值
        thresh = data.max() / 2.0   # 根据背景深浅选择文字颜色
        for i in range(len(CLASS_NAMES_CN)):
            for j in range(len(CLASS_NAMES_CN)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha='center', va='center', fontsize=10,
                        color='white' if data[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 混淆矩阵已保存到: {save_path}")
    plt.close()

    # 打印各类别准确率
    print("\n各类别详细指标：")
    print(classification_report(y_true, y_pred,
                                 target_names=CLASS_NAMES_CN,
                                 digits=3))

    # 找出最容易混淆的两个类别
    cm_copy = cm_normalized.copy()
    np.fill_diagonal(cm_copy, 0)   # 把对角线（正确预测）置0，只看错误
    max_idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    true_cls  = CLASS_NAMES_CN[max_idx[0]]
    pred_cls  = CLASS_NAMES_CN[max_idx[1]]
    error_rate = cm_copy[max_idx] * 100
    print(f"\n📌 最容易混淆的两类：")
    print(f"   真实类别「{true_cls}」中有 {error_rate:.1f}% 的图片被错误预测为「{pred_cls}」")


def plot_bad_cases(images, y_true, y_pred, all_probs, save_path, n_show=12):
    """
    展示若干张分类错误的图片（Bad Case 分析）。

    参数:
        images    : 图像数组（来自 DataLoader 的标准化 Tensor，需反标准化）
        y_true    : 真实标签
        y_pred    : 预测标签
        all_probs : 各类别概率
        save_path : 保存路径
        n_show    : 展示的错误样本数量
    """
    # 找出所有预测错误的样本索引
    wrong_indices = np.where(y_true != y_pred)[0]
    print(f"\n共有 {len(wrong_indices)} 张图片被分类错误")

    if len(wrong_indices) == 0:
        print("没有错误样本，模型表现完美！")
        return

    # 最多展示 n_show 个
    show_idx = wrong_indices[:min(n_show, len(wrong_indices))]

    # 图像反标准化（把 Tensor 还原成可视化的 RGB 图像）
    mean = np.array([0.4302, 0.4575, 0.4539])
    std  = np.array([0.2608, 0.2555, 0.2744])

    n_cols = 4
    n_rows = (len(show_idx) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.5, n_rows * 3.5))
    fig.suptitle(f'错误样本分析（共展示 {len(show_idx)} 张）', fontsize=14, fontweight='bold')

    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, idx in enumerate(show_idx):
        # 反标准化：img = img * std + mean
        img = images[idx].transpose(1, 2, 0)   # (C,H,W) → (H,W,C)
        img = img * std + mean
        img = np.clip(img, 0, 1)               # 裁剪到 [0,1] 防止溢出

        true_name = CLASS_NAMES_CN[y_true[idx]]
        pred_name = CLASS_NAMES_CN[y_pred[idx]]
        confidence = all_probs[idx][y_pred[idx]] * 100   # 错误预测的置信度

        axes[i].imshow(img)
        axes[i].set_title(
            f"真实: {true_name}\n预测: {pred_name} ({confidence:.0f}%)",
            fontsize=9,
            color='red'
        )
        axes[i].axis('off')

    # 隐藏多余的子图
    for j in range(len(show_idx), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 错误样本分析图已保存到: {save_path}")
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 1. 加载设备和模型 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("\n正在加载模型...")
    model = get_model(MODEL_NAME, pretrained=False)   # 只加载结构，权重从文件读取
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到模型文件: {MODEL_PATH}\n请先运行 train.py 完成训练！"
        )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print("模型加载成功！")

    # ---- 2. 加载测试集 ----
    print("\n正在加载测试集...")
    _, _, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)

    # ---- 3. 获取所有预测结果 ----
    print("\n正在对测试集做预测（可能需要1-2分钟）...")
    y_true, y_pred, all_probs, all_images = get_all_predictions(model, test_loader, device)

    overall_acc = (y_true == y_pred).mean() * 100
    print(f"测试集总体准确率: {overall_acc:.2f}%")

    # ---- 4. 绘制混淆矩阵 ----
    print("\n正在绘制混淆矩阵...")
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path)

    # ---- 5. 错误样本分析 ----
    print("\n正在分析错误样本（Bad Cases）...")
    bad_path = os.path.join(RESULTS_DIR, 'bad_cases.png')
    plot_bad_cases(all_images, y_true, y_pred, all_probs, bad_path)

    print(f"\n{'='*50}")
    print("可视化分析完成！生成的文件：")
    print(f"  - {cm_path}")
    print(f"  - {bad_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
