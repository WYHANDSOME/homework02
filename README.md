# CIFAR-10 图像分类项目（PyTorch 实现）

## 📌 项目简介

本项目使用 **PyTorch** 框架构建了一个简单的卷积神经网络（CNN）模型，对 CIFAR-10 图像进行分类。  
CIFAR-10 是一个包含 10 个类别、共计 60000 张 32×32 彩色图片的经典计算机视觉数据集。

本项目包含完整的数据加载、模型训练、测试评估及图像可视化功能，并支持 **GPU 加速**。

---

## 📦 使用的技术

- Python 3.x  
- PyTorch  
- Torchvision  
- Matplotlib  
- NumPy

---

## 🧠 模型结构

```python
输入：3 x 32 x 32 彩色图像

Conv2D(3 → 16) + ReLU + MaxPool2D(2×2)  
Conv2D(16 → 8) + ReLU + MaxPool2D(2×2)  
Flatten → Linear(8×8×8 → 32) + ReLU  
Linear(32 → 10) → 输出10类
````

---

## 🔧 数据预处理

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.RandomGrayscale(),          # 随机灰度变换
    transforms.ToTensor(),                 # 转为张量
    transforms.Normalize((0.5, 0.5, 0.5),  # 归一化
                         (0.5, 0.5, 0.5))
])
```

---

## ⚙️ 训练参数

* 优化器：SGD（学习率=0.001，动量=0.9）
* 损失函数：CrossEntropyLoss
* 批大小：128
* 训练轮数：20
* 使用 GPU（如可用）

---

## 🚀 运行方式

1. 安装依赖：

   ```bash
   pip install torch torchvision matplotlib
   ```

2. 运行主程序：

   ```bash
   python main.py
   ```

---

## 📈 运行结果摘要

* 使用设备：**cuda**
* 每轮训练耗时约：18 秒
* **整体测试集准确率：56%**

### 📊 各类别准确率：

| 类别    | 准确率 |
| ----- | --- |
| plane | 63% |
| car   | 67% |
| bird  | 39% |
| cat   | 38% |
| deer  | 49% |
| dog   | 43% |
| frog  | 71% |
| horse | 63% |
| ship  | 73% |
| truck | 58% |

---

## 🖼️ 示例预测输出

```text
GroundTruth:   cat  ship  ship plane  
Predicted:     cat  ship plane  ship
```

---

## 🔄 转换为 Jupyter Notebook

你可以使用以下命令将 Python 脚本转换为 `.ipynb` 格式，方便在 Jupyter 中运行：

```bash
pip install jupytext
jupytext main.py --to notebook
```

---

## 📁 数据集说明

* 数据集：CIFAR-10（10类图像，自动下载）
* 下载路径：`./cifar10/`

---

## ✅ 完成情况

* [x] 加载与增强训练数据
* [x] 构建卷积神经网络模型
* [x] 模型训练与测试
* [x] 可视化输出与预测结果
* [x] 支持 GPU 加速

---



