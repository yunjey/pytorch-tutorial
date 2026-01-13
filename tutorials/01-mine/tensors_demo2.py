import torch
import numpy as np
import torch  # for all things PyTorch
import torch.nn as nn  # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
import torch
import math

"""
    Tensors operations （张量运算）
"""


def tensors_demo():
    z = torch.zeros(5, 3)
    print(z)
    # 发现这些零是32位浮点数，这是PyTorch的默认类型。
    print(z.dtype)
    i = torch.ones((5, 3), dtype=torch.int16)
    print(i)
    print(i.dtype)
    torch.manual_seed(1729)
    r1 = torch.rand(2, 2)
    print('A random tensor:')
    print(r1)

    r2 = torch.rand(2, 2)
    print('\nA different random tensor:')
    print(r2)  # new values

    torch.manual_seed(1729)
    r3 = torch.rand(2, 2)
    print('\nShould match r1:')
    print(r3)  # repeats values of r1 because of re-seed
    ones = torch.ones(2, 3)
    print(ones)

    twos = torch.ones(2, 3) * 2  # every element is multiplied by 2
    print(twos)

    threes = ones + twos  # addition allowed because shapes are similar
    print(threes)  # tensors are added element-wise
    print(threes.shape)  # this has the same dimensions as input tensors

    r1 = torch.rand(2, 3)
    r2 = torch.rand(3, 2)
    # uncomment this line to get a runtime error
    # r3 = r1 + r2
    r = (torch.rand(2, 2) - 0.5) * 2  # values between -1 and 1
    print('A random matrix, r:')
    print(r)

    # Common mathematical operations are supported:
    print('\nAbsolute value of r:')
    print(torch.abs(r))

    # ...as are trigonometric functions:
    print('\nInverse sine of r:')
    """
    求每个元素的反正弦值
    """
    print(torch.asin(r))

    # ...and linear algebra operations like determinant and singular value decomposition
    print('\nDeterminant of r:')
    """
    计算 方阵 的 行列式（determinant）。行列式是一个标量，描述矩阵的缩放因子和方向变化。
    r = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
    print(torch.det(r))
    =1×4−2×3=−2
    """
    print(torch.det(r))
    print('\nSingular value decomposition of r:')
    print(torch.svd(r))

    # ...and statistical and aggregate operations:
    print('\nAverage and standard deviation of r:')
    # 同时计算张量的 标准差（standard deviation） 和 均值（mean）。
    print(torch.std_mean(r))
    print('\nMaximum value of r:')
    print(torch.max(r))


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    # 前向计算
    """
    前向计算
    就是让输入数据通过神经网络，得到模型的预测结果的过程。你可以把它想象成：
    输入数据（比如一张图片、一段文字）从网络的第一层开始，依次经过每一层的运算（卷积、矩阵乘法、激活函数等）。
    每一层都会对上一层的输出做处理，最终在网络的最后一层得到预测值（比如分类任务的类别概率、回归任务的数值）。
    作用是什么？
    得到预测结果：比如输入一张猫的图片，前向计算会输出模型认为这是 “猫” 的概率。
    计算损失：将预测结果与真实标签（比如 “猫”）比较，用损失函数（Loss Function）衡量模型预测的误差。

    反向传播（Backward Pass）
    反向传播就是根据损失函数，从网络的最后一层往回计算每一层参数的梯度的过程。梯度表示参数变化对损失的影响程度，优化器（如 SGD、Adam）会用这些梯度来更新参数，让损失更小。
    你可以把它想象成：
    从损失值开始，沿着网络的每一层反向计算参数的梯度（使用链式法则）。
    梯度会告诉我们：每个参数应该调整多少，才能让模型预测更准确。
    作用是什么？
    计算梯度：得到每个参数（权重 w、偏置 b）的梯度。更新参数：优化器根据梯度调整参数，让模型的预测结果越来越接近真实值。
    
    前向计算：输入数据 → 模型 → 预测值 → 损失。
    反向传播：损失 → 计算梯度 → 更新参数。
    """
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def pytorch_model_demo():
    net = LeNet()
    print(net)
    input = torch.rand(1, 1, 32, 32)  # stand-in for a 32x32 black & white image
    print('\nImage batch shape:')
    print(input.shape)

    output = net(input)  # we don't call forward() directly
    print('\nRaw output:')
    print(output)
    print(output.shape)


def tensors_demo1():
    x = torch.empty(3, 4)
    print(type(x))
    print(x)
   #随机张量与种子
    torch.manual_seed(1729)
    random1 = torch.rand(2, 3)
    print(random1)

    random2 = torch.rand(2, 3)
    print(random2)

    torch.manual_seed(1729)
    random3 = torch.rand(2, 3)
    print(random3)

    random4 = torch.rand(2, 3)
    print(random4)
    # In Brief: Tensor Broadcasting
    # 广播是一种机制，它允许在进行逐元素运算时，对不同形状的张量自动进行 “虚拟扩展”，使它们的形状兼容，从而可以进行运算。
    """
    广播的核心思想是：
    不需要实际复制数据，而是在逻辑上扩展张量的维度。
    它遵循一套规则，自动对齐不同形状的张量。
    广播的规则
    广播有两个核心规则：
    规则 1：维度对齐
    从 最后一个维度 开始向前比较两个张量的维度：
    如果两个维度的大小 相同 → 兼容。
    如果其中一个维度的大小是 1 → 兼容（会被扩展）。
    如果两个维度的大小 不同且都不是 1 → 不兼容，会报错。
    规则 2：维度扩展
    对于大小为 1 的维度，沿着该维度复制数据（在逻辑上），直到与另一个张量的对应维度大小一致。
    """
    rand = torch.rand(2, 4)
    doubled = rand * (torch.ones(1, 4) * 2)

    print(f"rand: {rand}")
    print(f"doubled: {doubled}")

if __name__ == '__main__':
    tensors_demo1()
