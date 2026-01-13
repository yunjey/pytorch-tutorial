import ssl

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 初始化一些示例数据
        self.data = torch.randn(100, 3)  # 100个样本，每个样本3个特征
        self.labels = torch.randint(0, 2, (100,))  # 100个标签，0或1

    def __getitem__(self, index):
        # 返回数据对 (特征, 标签)
        return self.data[index], self.labels[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.data)


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189)
# （CNN）的基本原理 是什么？


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

"""
神经网络中的延伸
在神经网络中，损失函数 ( L ) 是关于所有模型参数（权重 ( w_1, w_2, ... ) 和偏置 ( b_1, b_2, ... )）的高维多元函数。因此：

参数数量 = 偏导数数量
每个参数的偏导数 ( \frac{\partial L}{\partial \theta} ) 代表损失函数在该参数方向上的变化率
所有偏导数组成的向量称为梯度，用于指导参数更新方向
"""
if __name__ == '__main__':
    """
    偏导数的大小：变化率的强度
    偏导数的绝对值大小代表了函数在该参数方向上的变化率强度：

    偏导数绝对值越大 → 函数在该方向上变化越快 → 该参数对损失函数的影响越敏感
    偏导数绝对值越小 → 函数在该方向上变化越慢 → 该参数对损失函数的影响较不敏感
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    # Create tensors.
    x = torch.tensor(1., requires_grad=True)
    print(f"x Python 类型: {type(x)}")
    print(f"tensor 类名: {x.__class__.__name__}")
    print(f"tensor 模块: {x.__class__.__module__}")
    print(f"tensor 是否是 torch.Tensor: {isinstance(x, torch.Tensor)}")
    print("在 PyTorch 中，当你创建一个张量并设置 requires_grad=True 时，PyTorch 会自动构建一个计算图"
          "计算图记录了所有对该张量的操作，以便后续计算梯度"
          "y.backward() 会从 y 开始，沿着计算图反向传播，计算 y 相对于所有具有 requires_grad=True 的输入张量（这里是 x、w、b）的偏导数")
    w = torch.tensor(2., requires_grad=True)
    b = torch.tensor(3., requires_grad=True)

    # Build a computational graph.
    y = w * x + b  # y = 2 * x + 3
    print("""你不能直接打印 y.grad，因为：
    只有叶子张量（Leaf Tensor）才能保存梯度
    叶子张量是指直接创建的张量（不是通过其他张量计算得到的）
    在这个例子中，x、w、b 是叶子张量，而 y 是通过计算得到的中间张量
    PyTorch 默认只保存叶子张量的梯度，以节省内存空间""")
    # Compute gradients.
    # y.retain_grad()  # 保存 y 的梯度
    # 执行反向传播
    y.backward()

    # Print out the gradients.
    # 用于执行反向传播，计算梯度：
    print(x.grad)  # x.grad = 2
    print(w.grad)  # w.grad = 1
    print(b.grad)  # b.grad = 1

    # ================================================================== #
    #                    2. Basic autograd example 2                     #
    # ================================================================== #

    # Create tensors of shape (10, 3) and (10, 2).
    """
    torch.randn(10, 3): 创建一个形状为 (10, 3) 的张量（矩阵），其中包含从标准正态分布中随机采样的值

    10 表示批次大小（batch size），即一次处理10个样本
    3 表示每个样本有3个特征
    torch.randn(10, 2): 创建一个形状为 (10, 2) 的张量，作为模型的目标输出

    10 同样是批次大小，与输入数据对应
    2 表示每个样本期望输出2个值
    """
    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    print('x: ', x)
    print('y: ', y)
    # Build a fully connected layer.
    # 构建全连接层
    """
    第一个参数 3 是输入特征的维度
    第二个参数 2 是输出特征的维度
    这个线性层会自动初始化权重和偏置参数
    """
    linear = nn.Linear(3, 2)
    print('w: ', linear.weight)
    print('b: ', linear.bias)

    # Build loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

    # Forward pass.
    pred = linear(x)

    # Compute loss.
    loss = criterion(pred, y)
    print('loss: ', loss.item())

    # Backward pass.
    loss.backward()

    # Print out the gradients.
    print('dL/dw: ', linear.weight.grad)
    print('dL/db: ', linear.bias.grad)

    # 1-step gradient descent.
    optimizer.step()

    # You can also perform gradient descent at the low level.
    # linear.weight.data.sub_(0.01 * linear.weight.grad.data)
    # linear.bias.data.sub_(0.01 * linear.bias.grad.data)

    # Print out the loss after 1-step gradient descent.
    pred = linear(x)
    loss = criterion(pred, y)
    print('loss after 1 step optimization: ', loss.item())
    """
    这段代码的意义
    这是深度学习中构建神经网络的基础步骤：
    准备输入数据和目标数据
    定义模型结构（这里是一个简单的线性层）
    查看和理解模型参数
    在实际应用中，这段代码之后通常会添加：
    损失函数定义（如 MSE、交叉熵等）
    优化器选择（如 SGD、Adam 等）
    前向传播、损失计算、反向传播和参数更新的循环
    这段代码展示了PyTorch构建神经网络的核心概念，是理解更复杂模型的基础。
    """

    # ================================================================== #
    #                     3. Loading data from numpy                     #
    # ================================================================== #

    # Create a numpy array.
    x = np.array([[1, 2], [3, 4]])

    # Convert the numpy array to a torch tensor.
    y = torch.from_numpy(x)

    # Convert the torch tensor to a numpy array.
    z = y.numpy()
    # x==z: [[ True  True]
    print(f"x==z: {x == z}")

    # ================================================================== #
    #                         4. Input pipeline                           #
    # ================================================================== #

    # Download and construct CIFAR-10 dataset.
    print("""
    # CIFAR - 10
    # 是一个经典的图像分类数据集，包含：
    # 60000 张 32x32 彩色图像
    # 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
    # 每个类别有 6000 张图像
    # 训练集：50000 张
    # 测试集：10000 张
    # 数据预处理：
    # 图像被归一化到 [0, 1] 范围
    # 每个通道的均值和标准差分别为 [0.5, 0.5, 0.5] 和 [0.5, 0.5, 0.5]
    """)
    train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    # Fetch one data pair (read data from disk).
    image, label = train_dataset[0]
    print(f"image.size(): {image.size()}")
    print(f"label: {label}")

    # Data loader (this provides queues and threads in a very simple way).
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_loader)

    # Mini-batch images and labels.
    images, labels = data_iter.__next__()
    print(f"images.size(): {images.size()}")
    print(f"labels: {labels}")
    # Actual usage of the data loader is as below.
    for images, labels in train_loader:
        # Training code should be written here.
        pass

    # ================================================================== #
    #                5. Input pipeline for custom dataset                 #
    # ================================================================== #

    # You should build your custom dataset as below.

    # You can then use the prebuilt data loader.
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=64,
                                               shuffle=True)

    # ================================================================== #
    #                        6. Pretrained model                         #
    # ================================================================== #
    """
    torchvision.models：PyTorch视觉库(torchvision)中的模型模块，包含了多种经典的计算机视觉预定义模型（如ResNet、VGG、AlexNet等）
    resnet18：ResNet（Residual Network，残差网络）模型家族中的一个变体，指具有18层网络结构的ResNet模型
    pretrained=True：关键参数，指定加载在ImageNet数据集上预训练好的模型权重
    """
    """预训练模型是指：
       已经在大型数据集（这里是ImageNet，包含1400万张图像，1000
       个类别）上训练完成的模型
       模型权重已经学习到了通用的图像特征（如边缘、纹理、形状等高级视觉特征）
       可以直接用于推理，或作为迁移学习的起点"""
    # Download and load the pretrained ResNet-18.
    resnet = torchvision.models.resnet18(pretrained=True)
    # 模型微调（Fine - tuning）的经典实现
    # If you want to finetune only the top layer of the model, set as below.
    """
    作用：将ResNet-18模型中所有原有参数的requires_grad设置为False
    效果：在反向传播时，这些参数不会计算梯度，也就不会被更新
    原理：预训练模型的底层（卷积层）已经学习到了通用的视觉特征（如边缘、纹理、形状），这些特征对大多数图像任务都有效，无需重新学习
    """
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace the top layer for finetuning.
    """
    作用：将ResNet-18模型的全连接层（fc层）替换为一个新的全连接层，输出维度为100
    效果：模型的输出层从原来的1000个类别（ImageNet数据集的类别数）减少到100个类别
    原理：全连接层是模型的最后一层，负责将特征映射到类别空间。通过替换fc层，我们可以将模型用于不同的分类任务（如CIFAR - 10）
    """
    resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

    # Forward pass.
    images = torch.randn(64, 3, 224, 224)
    outputs = resnet(images)
    print(outputs.size())  # (64, 100)

    # ================================================================== #
    #                      7. Save and load the model                    #
    # ================================================================== #
    """
    torch.save(resnet, 'model.ckpt')：保存完整模型对象
    保存的是整个模型的Python对象，包括：
    模型的网络结构（如ResNet-18的卷积层、池化层、全连接层等）
    模型的所有参数（权重和偏置）
    模型的优化器状态（如果模型包含的话）
    其他与模型相关的Python对象（如类定义、导入依赖等）
    本质：使用Python的pickle序列化机制保存整个对象
    """
    # Save and load the entire model.
    torch.save(resnet, 'model.ckpt')
    model = torch.load('model.ckpt', weights_only=False)
    """
    只保存模型的参数状态字典（State Dictionary）：
    以字典形式保存所有可学习参数的名称和值
    不包含模型的网络结构信息
    不包含任何Python类定义或依赖
    本质：只保存模型的"权重"，不保存"骨架"
    """
    # Save and load only the model parameters (recommended).
    torch.save(resnet.state_dict(), 'params.ckpt')
    resnet.load_state_dict(torch.load('params.ckpt'))
