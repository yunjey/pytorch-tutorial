


# 构建神经网络
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__':
    model = NeuralNetwork().to(device)
    print(model)
    # To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. Do not call model.forward() directly!
    # 要使用该模型，我们需向其传入输入数据。这会执行模型的forward以及一些后台操作。请勿直接调用model.forward()！
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    print(logits)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    input_image = torch.rand(3, 28, 28)
    print(input_image.size())
    # 我们初始化nn.Flatten层，将每个2D的28x28图像转换为一个包含784个像素值的连续数组（保持dim=0处的小批量维度）。
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    # 线性层是一个模块，它使用其存储的权重和偏置对输入进行线性变换。
    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3, 28, 28)
    logits = seq_modules(input_image)
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")