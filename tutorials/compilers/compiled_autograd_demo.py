import torch
import warnings
from torchvision.models import densenet121
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


"""
Python 解释器调用 Dynamo，因为此调用被装饰了 @torch.compile。
Dynamo 拦截 Python 字节码，模拟其执行并将操作记录到图中。
AOTDispatcher 禁用钩子，并调用自动梯度引擎来计算 model.linear.weight 和 
model.linear.bias 的梯度，并将操作记录到图中。使用 torch.autograd.Function，AOTDispatcher 重写了 train 的前向和反向传播实现。
Inductor 生成一个对应于 AOTDispatcher 前向和反向传播优化实现的函数。
Dynamo 设置优化后的函数，以便 Python 解释器接下来进行评估。
Python 解释器执行优化后的函数，该函数执行 loss = model(x).sum()。
Python 解释器执行 loss.backward()，调用自动梯度引擎，该引擎会路由到已编译的自动梯度引擎，因为我们将 torch._dynamo.config.compiled_autograd = True 设置为 True。
已编译的自动梯度计算 model.linear.weight 和 model.linear.bias 的梯度，并将操作记录到图中，包括它遇到的任何钩子。
在此过程中，它将记录 AOTDispatcher 之前重写的反向传播。然后，已编译的自动梯度生成一个新函数，该函数对应于 loss.backward() 
的完全跟踪实现，并以推理模式使用 torch.compile 执行它。
相同的步骤将递归应用于已编译的自动梯度图，但这次 AOTDispatcher 将不需要划分图。


"""


def train_demo():
    model = Model()
    x = torch.randn(10)
    torch._dynamo.config.compiled_autograd = True
    @torch.compile
    def train(model, x):
        loss = model(x).sum()
        loss.backward()
    train(model, x)

def train_demo1():
    model = Model()
    x = torch.randn(10)
    torch._dynamo.config.compiled_autograd = True
    @torch.compile
    def train(model, x):
        model = torch.compile(model)
        loss = model(x).sum()
        torch._dynamo.config.compiled_autograd = True
        torch.compile(lambda: loss.backward(), fullgraph=True)()
    train(model, x)
# TORCH_LOGS="compiled_autograd" CUDA_VISIBLE_DEVICES=1 python compiled_autograd_demo.py
# TORCH_LOGS="compiled_autograd_verbose" CUDA_VISIBLE_DEVICES=1 python compiled_autograd_demo.py
if __name__ == '__main__':
    train_demo1()
