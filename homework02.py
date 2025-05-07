import torch                             # PyTorch 主库
import torchvision                       # 图像处理库，包含数据集和模型
import torchvision.transforms as transforms  # 图像预处理模块
from torch.autograd import Variable      # 旧式变量封装（现在不常用，但可以兼容）
import torch.nn as nn                    # 神经网络模块
import torch.nn.functional as F          # 常用的函数接口（如激活函数等）
import torch.optim as optim              # 优化器模块
import matplotlib.pyplot as plt          # 画图模块
import numpy as np                       # 数组与数学处理
import time                              # 计时模块

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 输入3通道图像，输出16通道，卷积核3×3
        self.act1 = nn.ReLU()                                     # 激活函数ReLU
        self.pool1 = nn.MaxPool2d(2)                              # 2x2 最大池化

        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)   # 第二层卷积，输出通道数减小为8
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(8 * 8 * 8, 32)                       # 展平后全连接层，输入维度=通道数×H×W
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)                              # 最后一层，输出为10类

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))               # 第一层卷积+激活+池化
        out = self.pool2(self.act2(self.conv2(out)))             # 第二层卷积+激活+池化
        out = out.view(-1, 8 * 8 * 8)                             # 展平为一维向量（每张图大小为[8,8,8]）
        out = self.act3(self.fc1(out))                           # 第一全连接层+激活
        out = self.fc2(out)                                      # 输出层
        return out

def imshow(img):
    img = img * 0.5 + 0.5                           # 反归一化
    npimg = img.numpy()                            # 转为 numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))     # 通道换位（CHW → HWC）
    plt.show()

def train(net, trainloader, criterion, optimizer):
    for epoch in range(20):                      # 共训练20轮
        timestart = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # 遍历每个批次
            inputs, labels = data                # 拆分图像与标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()                # 梯度清零
            outputs = net(inputs)                # 前向传播
            loss = criterion(outputs, labels)    # 计算损失
            loss.backward()                      # 反向传播
            optimizer.step()                     # 参数更新
            running_loss += loss.item()          # 累计loss
            
            if i % 500 == 499:
                print('[%d ,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        print('epoch %d cost %3f sec' % (epoch + 1, time.time()-timestart))
    print('Finished Training')

def test(net, testloader, classes):
    # 显示一些测试图片
    dataiter = iter(testloader)                        # 取出测试集第一批图像
    images, labels = dataiter.__next__()
    images = images.to(device)  # 将测试图像移动到GPU
    imshow(torchvision.utils.make_grid(images.cpu()))        # 拼接图像并显示（需要移回CPU才能显示）
    print('GroundTruth:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)                    # 模型预测
    _, predicted = torch.max(outputs.data, 1)          # 取每张图预测结果最大概率类别
    print('Predicted:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # 计算整体准确率
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # 计算每个类别的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()     # 压缩成一维，方便索引
        for i in range(4):                      # 因为 batch_size=4
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),     
        transforms.RandomGrayscale(),          
        transforms.ToTensor(),                 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    # 初始化数据加载器
    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 初始化网络
    net = Net()
    net = net.to(device)  # 将模型移动到GPU
    criterion = nn.CrossEntropyLoss()                                 # 分类问题常用的损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   # 随机梯度下降 + 动量

    # 训练网络
    train(net, trainloader, criterion, optimizer)
    
    # 测试网络
    test(net, testloader, classes)
