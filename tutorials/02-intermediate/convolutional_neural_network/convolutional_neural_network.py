import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# 卷积神经网络

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    # 网络层定义
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # 第一层卷积层：输入通道1（MNIST灰度图），输出通道16，卷积核5x5，步长1，填充2
        # 卷积后尺寸保持不变：(28-5+2*2)/1 + 1 = 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),  # 批标准化：加速训练，提高稳定性
            nn.ReLU(),           # 激活函数：引入非线性
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层：尺寸减半为14x14 无参数的下采样操作
        )
        # 第二层卷积层：输入通道16，输出通道32，卷积核5x5，步长1，填充2
        # 卷积后尺寸保持不变：(14-5+2*2)/1 + 1 = 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),  # 批标准化
            nn.ReLU(),           # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层：尺寸减半为7x7
        )
        # 全连接层：输入尺寸7x7x32，输出类别数
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
    # 前向传播定义
    def forward(self, x):
        # 前向传播路径
        out = self.layer1(x)    # 输入x经过第一层卷积层，输出尺寸：[batch_size, 16, 14, 14]
        out = self.layer2(out)  # 输出经过第二层卷积层，输出尺寸：[batch_size, 32, 7, 7]
        out = out.reshape(out.size(0), -1)  # 展平：[batch_size, 32*7*7]
        out = self.fc(out)      # 全连接层分类，输出尺寸：[batch_size, num_classes]
        return out


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 5
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    # 损失函数：衡量模型预测与真实标签的差异程度
    # CrossEntropyLoss适用于分类任务，内部整合了softmax和负对数似然
    # 公式：loss = -sum(y_true * log(y_pred))，其中y_true是one-hot编码的真实标签
    criterion = nn.CrossEntropyLoss()
    
    # 优化器：根据损失函数的梯度来更新模型参数，最小化损失
    # Adam是一种常用的自适应学习率优化算法，结合了Momentum和RMSProp的优点
    # 参数说明：
    # - model.parameters(): 需要优化的模型参数集合
    # - lr: 学习率，控制参数更新的步长大小
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)  # 每个epoch的总步数 = 训练数据集大小 / 批大小
    for epoch in range(num_epochs):  # 训练num_epochs轮
        for i, (images, labels) in enumerate(train_loader):  # 遍历每个批次
            images = images.to(device)  # 将图像数据移至指定设备
            labels = labels.to(device)  # 将标签移至指定设备

            # Forward pass（前向传播）：模型对输入图像进行预测
            outputs = model(images)  # outputs形状：[batch_size, num_classes]
            loss = criterion(outputs, labels)  # 计算损失：模型预测与真实标签的差异

            # Backward and optimize（反向传播与参数优化）
            optimizer.zero_grad()  # 清除之前的梯度，避免梯度累积
            loss.backward()  # 反向传播：计算所有可训练参数的梯度
            optimizer.step()  # 优化器更新参数：根据梯度调整模型权重

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')