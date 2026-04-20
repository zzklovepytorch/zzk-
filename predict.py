# ==============================
# predict.py —— 单张图片预测（含 Top-3）
# ==============================
# 支持的图片格式：jpg、png、bmp、webp 等常见格式

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torchvision import transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from model import get_model
from dataset import CLASS_NAMES, CLASS_NAMES_CN, MEAN, STD

# ========== 配置（修改这里）==========
MODEL_PATH = './models/best_model.pth'   # 训练好的模型路径
MODEL_NAME = 'resnet18'                  # 与 train.py 中的 MODEL_NAME 保持一致
IMAGE_PATH = 'D:/archive(数据集)/seg_test/seg_test/buildings/20898.jpg'   # ← 把你要预测的图片路径填在这里，例如：'./my_photo.jpg'
RESULTS_DIR = './results'
# ======================================


def predict_top3(image_path, model_path=MODEL_PATH):
    """
    对单张图片做预测，返回概率最高的 Top-3 类别。
    参数:image_path : 图片文件路径;model_path : 模型权重路径
    返回:
        top3_results : 列表，每项为 (类别中文名, 概率%) 的元组，按概率从高到低排列
        all_probs    : 全部 6 个类别的概率列表（%）
    """
    # 检查文件
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型文件: {model_path}\n请先运行 train.py 训练模型！"
        )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = get_model(MODEL_NAME, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 图片预处理（与训练/测试集保持一致）
    transform = transforms.Compose([
        transforms.Resize((150, 150)),          # 缩放到 150×150
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # 读取图片并预处理
    image = Image.open(image_path).convert('RGB')
    # unsqueeze(0)：加一个 batch 维度，因为模型期望输入是 (batch, C, H, W)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        # softmax：把原始分数转为概率（所有值 ∈ [0,1] 且加和为1）
        probs = torch.softmax(outputs, dim=1)[0]

    all_probs = [p.item() * 100 for p in probs]   # 转为百分比

    # 取 Top-3（按概率从高到低排列）
    top3_indices = probs.topk(3).indices.tolist()
    top3_results = [
        (CLASS_NAMES_CN[i], all_probs[i])
        for i in top3_indices
    ]

    return top3_results, all_probs


def visualize_prediction(image_path):
    """
    可视化预测结果：左图显示原图，右图显示所有类别的概率条形图。
    """
    top3_results, all_probs = predict_top3(image_path)

    # 打印 Top-3 结果
    print("\n" + "=" * 40)
    print("Top-3 预测结果：")
    for rank, (class_name, prob) in enumerate(top3_results, 1):
        bar = "█" * int(prob / 5)   # 用 █ 画一个简单的进度条
        print(f"  第{rank}名: {class_name:4s}  {prob:5.1f}%  {bar}")
    print("=" * 40)

    # 绘制可视化图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Intel Image 场景分类预测结果', fontsize=15, fontweight='bold')

    # 左：原始图片
    image = Image.open(image_path).convert('RGB')
    ax1.imshow(image)
    top1_name, top1_prob = top3_results[0]
    ax1.set_title(f'预测：{top1_name}\n置信度：{top1_prob:.1f}%', fontsize=13)
    ax1.axis('off')

    # 右：概率条形图（按概率从大到小排列）
    sorted_idx = sorted(range(6), key=lambda i: all_probs[i], reverse=True)
    sorted_names = [CLASS_NAMES_CN[i] for i in sorted_idx]
    sorted_probs = [all_probs[i] for i in sorted_idx]

    # Top-1 用红色，其余用蓝色
    colors = ['#FF6B6B' if i == 0 else '#74B9FF' for i in range(6)]

    bars = ax2.barh(sorted_names[::-1], sorted_probs[::-1], color=colors[::-1])
    ax2.set_xlabel('预测概率 (%)', fontsize=11)
    ax2.set_title('各类别预测概率', fontsize=13)
    ax2.set_xlim(0, 105)

    # 在条形右侧显示数值
    for bar, prob in zip(bars, sorted_probs[::-1]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{prob:.1f}%', va='center', fontsize=9)

    # 标注 Top-3 范围
    top3_label = mpatches.Patch(color='#FF6B6B', label='Top-1 预测')
    ax2.legend(handles=[top3_label], loc='lower right')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'prediction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测结果图已保存到: {save_path}")
    plt.close()

    return top3_results


# 引入 matplotlib.patches（用于图例）
import matplotlib.patches as mpatches

if __name__ == '__main__':
    if not IMAGE_PATH:
        print("=" * 50)
        print("请先设置 IMAGE_PATH 变量！")
        print("打开 predict.py，把第 26 行改成你的图片路径，例如：")
        print("  IMAGE_PATH = 'C:/Users/xxx/Pictures/mountain.jpg'")
        print("=" * 50)
    else:
        top3 = visualize_prediction(IMAGE_PATH)
        print(f"\n最终预测：{top3[0][0]}（置信度 {top3[0][1]:.1f}%）")
