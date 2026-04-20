# 图像场景分类 —— Intel Image Classification (PyTorch)

## 项目简介

本项目基于 **Intel Image Classification** 数据集，使用 **PyTorch** 构建卷积神经网络，实现对 6 类自然场景的智能识别。

| 类别（英文）| 类别（中文）| 图片示例 |
|------------|------------|---------|
| buildings  | 建筑 | 城市建筑、房屋 |
| forest     | 森林 | 密林、树木 |
| glacier    | 冰川 | 冰雪山地 |
| mountain   | 山脉 | 山峰、山谷 |
| sea        | 海洋 | 海面、海岸 |
| street     | 街道 | 道路、街景 |

---

## 环境配置



```bash
pip install -r requirements.txt
```


## 数据集准备

### 下载方式

**Kaggle 官方下载：**


搜索 **Intel Image Classification** 或直接访问：
   ```
   https://www.kaggle.com/datasets/puneet6060/intel-image-classification
   ```


### 解压后的目录结构

将下载的压缩包解压，把文件夹放到项目的 `data/` 目录下，最终结构如下：

```
image_classification/
└── data/
    ├── seg_train/
    │   └── seg_train/
    │       ├── buildings/   (约 2191 张)
    │       ├── forest/      (约 2271 张)
    │       ├── glacier/     (约 2404 张)
    │       ├── mountain/    (约 2512 张)
    │       ├── sea/         (约 2274 张)
    │       └── street/      (约 2382 张)
    └── seg_test/
        └── seg_test/
            ├── buildings/   (约 437 张)
            ├── forest/      (约 474 张)
            └── ...
```

> ⚠️ **注意**：目录名必须是 `seg_train/seg_train` 和 `seg_test/seg_test`（有两层同名目录），这是 Kaggle 数据集的原始结构，不要改动。

---

## 快速开始

### 第一步：训练模型

在 PyCharm 中打开 `train.py`，**直接点击运行按钮**，或在终端执行：

```bash
python train.py
```

**训练过程输出示例：**
```
使用设备: cpu

正在加载数据集...
训练集大小: 12288 张图片
验证集大小: 1346 张图片
测试集大小: 3000 张图片

开始训练！模型: resnet18  共 20 轮
==================================================
【第  1/20 轮】  学习率: 0.001000
    [50/384] Loss: 1.2345 | Acc: 52.3%
  ...
  训练集 → Loss: 0.4231 | Acc: 85.6%
  验证集 → Loss: 0.3891 | Acc: 87.2%
  ✓ 新最佳验证准确率！模型已保存到 ./models/best_model.pth
```

训练完成后会自动生成：
- `models/best_model.pth` — 最佳模型权重
- `results/training_curves.png` — Loss/Accuracy 变化曲线

**预计准确率：**
| 设备 | 模型 | 准确率 | 用时 |
|------|------|--------|------|
| GPU  | ResNet18（预训练）| ~92%+ | ~10 分钟 |
| CPU  | ResNet18（预训练）| ~92%+ | ~2 小时 |
| GPU  | SimpleCNN         | ~80%+ | ~15 分钟 |

---

### 第二步：可视化分析（混淆矩阵）

```bash
python visualize.py
```

生成文件：
- `results/confusion_matrix.png` — 混淆矩阵，看模型把哪两类搞混了
- `results/bad_cases.png` — 错误样本展示（Bad Case 分析）

---

### 第三步：预测自己的图片

1. 打开 `predict.py`
2. 修改第 26 行：
   ```python
   IMAGE_PATH = 'C:/Users/你的用户名/Pictures/mountain.jpg'
   ```
3. 运行，即可看到 **Top-3 预测结果**

---

## 项目文件说明

```
image_classification/
│
├── dataset.py      # 数据加载与预处理（读取数据集、数据增强）
├── model.py        # 模型定义（SimpleCNN + ResNet18 两种选择）
├── train.py        # 训练主程序（★ 从这里开始）
├── visualize.py    # 可视化分析（混淆矩阵、Bad Case 分析）
├── predict.py      # 单张图片预测（Top-3 输出）
├── requirements.txt# 依赖包列表
│
├── data/           # 数据集（需手动下载）
├── models/         # 训练好的模型权重
└── results/        # 训练曲线、混淆矩阵等输出图片
```

---

## 网络结构图

### 自定义 SimpleCNN

```
输入图片 (3×150×150)
    ↓
卷积块1: Conv(3→32)   → BN → ReLU → MaxPool → (32×75×75)
    ↓
卷积块2: Conv(32→64)  → BN → ReLU → MaxPool → (64×37×37)
    ↓
卷积块3: Conv(64→128) → BN → ReLU → MaxPool → (128×18×18)
    ↓
卷积块4: Conv(128→256)→ BN → ReLU → MaxPool → (256×9×9)
    ↓
AdaptiveAvgPool                            → (256×4×4)
    ↓
全连接: Linear(4096→512) → ReLU → Dropout(0.5)
    ↓
输出层: Linear(512→6)   → 6个类别得分
    ↓
预测结果（概率最高的类别）
```

### ResNet18（迁移学习）

```
预训练 ResNet18 主干（在 ImageNet 上训练的特征提取器）
    ↓
自适应平均池化 → 512维特征向量
    ↓
全连接层: Linear(512→6)  ← 只替换这一层
    ↓
预测结果
```

---

## 超参数说明

| 参数 | 默认值 | 含义 | 如何调整 |
|------|--------|------|---------|
| `EPOCHS` | 20 | 训练轮数 | CPU慢时先改5试试，追求准确率可以30+ |
| `BATCH_SIZE` | 32 | 每批图片数 | 显存不足改16，显存充足可以64 |
| `LEARNING_RATE` | 0.001 | 初始学习率 | 太大会不收敛，太小会很慢 |
| `MODEL_NAME` | resnet18 | 模型选择 | 改成 `simplecnn` 理解原理 |
| `PRETRAINED` | True | 是否用预训练权重 | 强烈建议True，效果好很多 |


## 主要依赖

```
torch>=1.12.0
torchvision>=0.13.0
Pillow>=9.0.0
matplotlib>=3.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
```
