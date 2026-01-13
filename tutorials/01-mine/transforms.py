import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


"""
数据并不总是以训练机器学习算法所需的最终处理形式出现。我们使用变换对数据进行一些处理，使其适合训练。
所有TorchVision数据集都有两个参数——transform用于修改特征，target_transform用于修改标签，
这两个参数接收包含转换逻辑的可调用对象。torchvision.transforms模块提供了几种常用的现成转换方法。
"""
if __name__ == '__main__':
    # ToTensor将PIL图像或NumPy ndarray转换为FloatTensor</b2，并将图像的像素强度值缩放到[0.，1.]范围内。
    ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )
    target_transform = Lambda(lambda y: torch.zeros(
        10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))