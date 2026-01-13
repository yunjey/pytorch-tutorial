# ---------------------------------------------------------------------------- #
# ResNet模型实现：基于论文https://arxiv.org/pdf/1512.03385.pdf                   #
# 采用CIFAR-10数据集的模型架构（见论文4.2节）                                     #
# 部分代码参考PyTorch官方实现：                                                #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

# 导入必要的库
import torch                            # PyTorch核心库
import torch.nn as nn                   # 神经网络模块
import torchvision                      # 计算机视觉库，提供数据集和模型
import torchvision.transforms as transforms  # 图像预处理工具
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练

# 设备配置：优先使用GPU，否则使用CPU，M系列芯片可使用MPS
if torch.backends.mps.is_available():
    device = torch.device('mps')  # Apple Silicon M系列芯片加速
elif torch.cuda.is_available():
    device = torch.device('cuda')  # NVIDIA GPU加速
else:
    device = torch.device('cpu')   # CPU训练
print(f"使用设备: {device}")


# 3x3卷积层封装函数
# ResNet大量使用3x3卷积，此函数简化代码复用
def conv3x3(in_channels, out_channels, stride=1):
    # 定义3x3卷积：
    # - in_channels: 输入通道数
    # - out_channels: 输出通道数  
    # - kernel_size: 卷积核大小
    # - stride: 步长，默认1
    # - padding: 填充，设置为1保持尺寸不变
    # - bias: 不使用偏置，因为后续会接批标准化层
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# 残差块(Residual Block)定义
# ResNet的核心组件，通过残差连接解决深度网络训练问题
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # 第一个3x3卷积层：可能改变通道数和尺寸
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批标准化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，inplace=True节省内存
        # 第二个3x3卷积层：保持通道数和尺寸不变
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批标准化层
        # 下采样模块：当输入输出通道数或尺寸不匹配时使用
        self.downsample = downsample

    def forward(self, x):
        # 保存输入作为残差连接
        residual = x
        
        # 主路径：两次卷积+激活
        out = self.conv1(x)      # 第一次卷积
        out = self.bn1(out)      # 批标准化
        out = self.relu(out)     # 激活函数
        
        out = self.conv2(out)    # 第二次卷积
        out = self.bn2(out)      # 批标准化
        
        # 残差路径：如果需要下采样则调整尺寸和通道数
        if self.downsample:
            residual = self.downsample(x)
        
        # 残差连接：主路径输出 + 残差路径输出
        out += residual
        out = self.relu(out)     # 最终激活
        
        return out


# ResNet主网络定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        # 初始输入通道数
        self.in_channels = 16
        
        # 第一层：3x3卷积 + 批标准化 + ReLU
        # CIFAR-10输入为3通道，输出16通道
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 构建残差层：
        # layer1: 16通道，layers[0]个残差块，步长1
        # layer2: 32通道，layers[1]个残差块，步长2（尺寸减半）
        # layer3: 64通道，layers[2]个残差块，步长2（尺寸减半）
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        
        # 全局平均池化：将64x8x8特征图转为64x1x1
        self.avg_pool = nn.AvgPool2d(8)
        
        # 全连接层：将64维特征映射到10个类别
        self.fc = nn.Linear(64, num_classes)

    # 创建残差层的辅助函数
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # 当步长不为1或输入输出通道数不匹配时，需要下采样
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        # 添加第一个残差块（可能包含下采样）
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels  # 更新输入通道数
        
        # 添加剩余的残差块（不需要下采样）
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入x: [batch_size, 3, 32, 32]
        
        # 初始卷积层
        out = self.conv(x)      # [batch_size, 16, 32, 32]
        out = self.bn(out)      # 批标准化
        out = self.relu(out)    # 激活
        
        # 残差层1：保持尺寸不变
        out = self.layer1(out)  # [batch_size, 16, 32, 32]
        
        # 残差层2：尺寸减半
        out = self.layer2(out)  # [batch_size, 32, 16, 16]
        
        # 残差层3：尺寸减半
        out = self.layer3(out)  # [batch_size, 64, 8, 8]
        
        # 全局平均池化
        out = self.avg_pool(out)  # [batch_size, 64, 1, 1]
        
        # 展平特征
        out = out.view(out.size(0), -1)  # [batch_size, 64]
        
        # 全连接层分类
        out = self.fc(out)  # [batch_size, 10]
        
        return out
# 学习率更新函数
def update_lr(optimizer, lr):
    # 遍历优化器中的所有参数组，更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    # 主函数入口

    # 超参数设置 - 针对Mac优化
    num_epochs = 20         # 减少训练轮数（原80轮，可根据需要调整）
    batch_size = 32         # 减小批大小适配Mac内存（原100）
    learning_rate = 0.001   # 初始学习率

    # 图像预处理模块：数据增强提高模型泛化能力
    transform = transforms.Compose([
        transforms.Pad(4),               # 填充4个像素，32x32→40x40
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32),       # 随机裁剪回32x32
        transforms.ToTensor()            # 转换为Tensor格式
    ])

    # CIFAR-10数据集加载
    # 训练集：应用数据增强
    train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                 train=True,       # 训练集
                                                 transform=transform,  # 应用数据增强
                                                 download=True)    # 自动下载

    # 测试集：仅转换为Tensor，不应用数据增强
    test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                train=False,      # 测试集
                                                transform=transforms.ToTensor())

    # 数据加载器 - 优化版本
    # 训练数据加载器：多线程加载+内存锁定
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,  # 批大小
                                               shuffle=True,          # 打乱数据
                                               num_workers=4,         # 多线程加载（根据CPU核心数调整）
                                               pin_memory=True,       # 内存锁定加速数据传输
                                               persistent_workers=True)  # 保持线程存活

    # 测试数据加载器：多线程加载+内存锁定
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True)

    # 初始化ResNet模型
    # 使用ResidualBlock作为基本单元，[2, 2, 2]表示每个残差层包含2个残差块
    # 即创建一个ResNet-18模型（计算方式：2*(2+2+2)+2=14层？不，正确计算是：
    # 初始卷积层(1) + 3个残差层(每个2个块，每个块2层) + 全连接层(1) = 1+3*2*2+1=14层）
    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    # 损失函数和优化器
    # 交叉熵损失：适用于分类问题，内部包含softmax
    criterion = nn.CrossEntropyLoss()
    
    # Adam优化器：自适应学习率优化器，收敛速度快
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 混合精度训练配置
    scaler = GradScaler(enabled=(device.type == 'cuda' or device.type == 'mps'))

    # 训练模型
    total_step = len(train_loader)  # 每个epoch的总步数
    curr_lr = learning_rate         # 当前学习率
    
    # 训练模型
    total_step = len(train_loader)  # 每个epoch的总步数
    curr_lr = learning_rate         # 当前学习率
    
    for epoch in range(num_epochs):  # 遍历每个epoch
        model.train()  # 确保模型处于训练模式
        for i, (images, labels) in enumerate(train_loader):  # 遍历每个batch
            # 将数据移至指定设备
            images = images.to(device, non_blocking=True)  # 非阻塞传输
            labels = labels.to(device, non_blocking=True)  # 非阻塞传输

            # 前向传播：模型预测（混合精度）
            with autocast(enabled=(device.type == 'cuda' or device.type == 'mps')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播和优化（混合精度）
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清除
            scaler.scale(loss).backward()          # 缩放损失并反向传播
            scaler.step(optimizer)                 # 更新参数
            scaler.update()                        # 更新缩放器

            # 每100步打印一次损失
            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # 每20个epoch衰减一次学习率
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3  # 学习率除以3
            update_lr(optimizer, curr_lr)  # 更新优化器的学习率
    # 测试模型
    model.eval()  # 设置模型为评估模式
    
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        correct = 0  # 正确预测数
        total = 0    # 总样本数
        
        for images, labels in test_loader:  # 遍历测试集
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)  # 模型预测
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测数

        # 计算并打印准确率
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # 保存模型参数到文件
    torch.save(model.state_dict(), 'resnet.ckpt')