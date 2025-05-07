CIFAR-10 图像分类项目（PyTorch 实现）
项目简介
本项目使用 PyTorch 框架构建一个简单的卷积神经网络（CNN）模型，对 CIFAR-10 数据集进行图像分类。CIFAR-10 是一个包含 10 类 32x32 彩色图像的数据集，常用于计算机视觉入门实验。

模型结构为两层卷积 + 两层全连接，使用交叉熵损失和 SGD 优化器进行训练，并在测试集上评估模型的整体与逐类准确率。

使用的技术
Python 3

PyTorch

Torchvision（数据加载与图像增强）

Matplotlib（用于可视化图像）

GPU 加速（如果可用）

模型结构
python
复制
编辑
输入：3 x 32 x 32 彩色图像

Conv2D(3 → 16) + ReLU + MaxPool2D(2×2)
Conv2D(16 → 8) + ReLU + MaxPool2D(2×2)
Flatten → Linear(8×8×8 → 32) + ReLU
Linear(32 → 10) → 输出为10类
数据增强与预处理
python
复制
编辑
transforms.Compose([
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.RandomGrayscale(),          # 随机灰度变换
    transforms.ToTensor(),                 # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5),  # 归一化
                         (0.5, 0.5, 0.5))
])
训练设置
学习率：0.001

优化器：SGD + Momentum（0.9）

损失函数：交叉熵

批大小（Batch size）：128

训练轮数（Epochs）：20

数据加载线程数（num_workers）：2

运行结果（摘要）
使用设备：GPU（cuda）

总共训练 20 个 epoch，每个 epoch 耗时约 18 秒

模型整体准确率：56%

分类准确率如下：

类别	准确率
plane	63%
car	67%
bird	39%
cat	38%
deer	49%
dog	43%
frog	71%
horse	63%
ship	73%
truck	58%

示例预测
text
复制
编辑
GroundTruth:   cat  ship  ship plane
Predicted:     cat  ship plane  ship
如何运行
安装依赖：

bash
复制
编辑
pip install torch torchvision matplotlib
运行主程序：

bash
复制
编辑
python main.py
（你也可以将该代码转为 .ipynb 以便交互式运行，参考 转换为 notebook）

转换为 Notebook
你可以使用以下命令将 .py 转为 .ipynb：

bash
复制
编辑
pip install jupytext
jupytext main.py --to notebook

