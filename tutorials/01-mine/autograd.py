# 微分是把整体拆成无限小的局部，求「瞬间变化率」；积分是把无限小的局部拼回整体，求「累积总量」，二者是互逆运算，就像 “拆积木” 和 “搭积木” 的关系。
# 通俗说：比如汽车行驶，微分就是求某一秒的瞬时速度（不是平均速度）；比如山坡，微分就是求某一点的坡度（斜率）。
"""
定积分：求 “面积 / 总量”（核心应用）
对函数\(y=f(x)\)，在区间\([a,b]\)上的定积分\(\int_{a}^{b}f(x)dx\)，本质是：把区间\([a,b]\)拆成无数个微小区间，每个区间对应一个微小矩形（高 = f (x)，宽 = dx），把所有微小矩形的面积加起来，就是定积分的结果。
不定积分：微分的 “逆运算”
通俗说：知道 “每一点的斜率”，反推 “原来的曲线”；知道 “瞬时速度”，反推 “位移函数”。✅ 例子：已知微分（导数）\(y'=2x\)，不定积分就是 \(y=x^2 + C\)（C 是任意常数），因为\(x^2\)、\(x^2+1\)、\(x^2+100\)的导数都是 2x。

"""

# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import torch


def demo1():
    """
        Consider the simplest one-layer neural network,
        with input x, parameters w and b, and some loss function. It can be defined in PyTorch in the following manner:
        """
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")
    # Computing Gradients
    loss.backward()
    """
    我们只能获取计算图中叶节点的grad属性，这些叶节点的requires_grad属性被设置为True。对于图中的所有其他节点，梯度将不可用。

    """
    print(w.grad)
    print(b.grad)


def sin_demo():
    a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
    print(a)
    b = torch.sin(a)
    plt.plot(a.detach(), b.detach())
    print(b)
BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def sin_demo1():
    some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
    ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

    model = TinyModel()
    print(model.layer2.weight[0][0:10])  # just a small slice
    print(model.layer2.weight.grad)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    print(loss)
    loss.backward()
    print(model.layer2.weight[0][0:10])
    print(model.layer2.weight.grad[0][0:10])
    optimizer.step()
    print(model.layer2.weight[0][0:10])
    print(model.layer2.weight.grad[0][0:10])



if __name__ == '__main__':
    sin_demo1()
