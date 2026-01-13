## 1. **神经网络的本质：数据处理的 “流水线”**

在 PyTorch 中，**神经网络**本质上是一个 **由可训练参数（权重和偏置）组成的计算图**，它的作用是：

- 接收一个输入（通常是张量，比如图片的像素矩阵、文本的向量表示）
- 通过一系列 **线性变换**（矩阵乘法）和 **非线性激活**（比如 ReLU、Sigmoid）
- 输出一个预测结果（比如分类任务的类别概率、回归任务的数值）

你可以把它想象成一个 **“智能函数”**：

\(y = f(x; W, b)\)

其中：

- x 是输入（张量）
- W 和 b 是网络的 **可训练参数**（张量）
- f 是网络的计算逻辑（由多个层组成）
- y 是输出（张量）

------

## 2. **PyTorch 神经网络的核心组成**

在 PyTorch 中，神经网络通常由以下几个关键部分构成：

### （1）`torch.nn.Module`：网络的 “容器”

- 所有神经网络都必须继承自 `torch.nn.Module` 类

- 它是一个

   

  参数化的容器

  ，可以包含：

  - 网络的层（如 `nn.Linear`、`nn.Conv2d`、`nn.ReLU` 等）
  - 可训练的参数（`nn.Parameter`）
  - 自定义的计算逻辑

例如：

```python
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 定义层（包含可训练参数）
        self.fc1 = nn.Linear(10, 20)  # 输入10维，输出20维
        self.relu = nn.ReLU()         # 非线性激活
        self.fc2 = nn.Linear(20, 2)   # 输出2类

    def forward(self, x):
        # 定义数据流动的路径（前向传播）
        x = self.fc1(x)  # 线性变换：x @ W1 + b1
        x = self.relu(x) # 非线性激活
        x = self.fc2(x)  # 线性变换：x @ W2 + b2
        return x
```

------

### （2）**层（Layer）：网络的 “基本单元”**

层是神经网络的核心组件，每个层都是一个 **参数化的函数**，负责对输入张量进行特定的变换。常见的层包括：

| 层类型               | 作用                           | 数学表达（简化）                         |
| -------------------- | ------------------------------ | ---------------------------------------- |
| `nn.Linear(in, out)` | 线性变换（全连接层）           | \(y = xW^T + b\)                         |
| `nn.Conv2d(in, out)` | 二维卷积（提取空间特征）       | \(y = \text{Conv}(x, W) + b\)            |
| `nn.ReLU()`          | 非线性激活（增加模型表达能力） | \(y = \max(0, x)\)                       |
| `nn.Softmax(dim)`    | 归一化输出为概率分布           | \(y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}\) |

这些层的本质都是 **对张量的运算**，而层中的 `weight` 和 `bias` 是 **可训练的张量**（`nn.Parameter` 类型），会在训练过程中通过梯度下降更新。

------

### （3）**前向传播（forward）：数据的 “流动路径”**

- `forward` 方法定义了 **数据如何在网络中流动**
- 输入张量 `x` 依次经过各层的变换，最终得到输出张量
- 这个过程就是 **计算图的构建过程**（PyTorch 会自动记录运算，用于反向传播）

例如：

```python
net = MyNet()
x = torch.randn(3, 10)  # 3个样本，每个样本10维特征
y = net(x)              # 前向传播：x → fc1 → relu → fc2 → y
print(y.shape)          # 输出: torch.Size([3, 2])
```

------

### （4）**参数（Parameter）：网络的 “可训练变量”**

- 网络的参数（权重 W 和偏置 b）是 `nn.Parameter` 类型的张量
- 它们会被自动注册到网络的 `parameters()` 或 `named_parameters()` 方法中
- 在训练时，优化器（如 `torch.optim.SGD`）会根据梯度更新这些参数

查看网络参数：

```python
for name, param in net.named_parameters():
    print(name, param.shape)
```

输出：

```plaintext
fc1.weight torch.Size([20, 10])
fc1.bias   torch.Size([20])
fc2.weight torch.Size([2, 20])
fc2.bias   torch.Size([2])
```

------

## 3. **神经网络的 “学习” 过程**

神经网络之所以能 “智能”，是因为它的参数可以通过 **数据驱动的方式更新**（即训练过程），核心步骤如下：

### （1）定义损失函数（衡量预测与真实值的差距）

```python
loss_fn = nn.CrossEntropyLoss()  # 分类任务常用损失
```

### （2）定义优化器（负责更新参数）

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 随机梯度下降
```

### （3）训练循环（前向传播 → 计算损失 → 反向传播 → 更新参数）

```python
for epoch in range(100):
    # 前向传播
    y_pred = net(x)
    
    # 计算损失
    loss = loss_fn(y_pred, y_true)  # y_true是真实标签
    
    # 反向传播（计算梯度）
    optimizer.zero_grad()  # 清空上一轮梯度
    loss.backward()        # 自动计算参数的梯度
    
    # 更新参数（梯度下降）
    optimizer.step()
```

通过这个过程，网络的参数会逐渐调整，使得预测结果越来越接近真实值。

------

## 4. **PyTorch 神经网络的特点**

- **模块化**：可以像搭积木一样组合不同的层
- **自动微分**：`backward()` 方法自动计算梯度，无需手动推导
- **灵活性**：`forward` 方法可以写任意复杂的逻辑（如循环、条件判断）
- **GPU 加速**：只需调用 `.to("cuda")` 即可在 GPU 上运行“智能函数”：\(y = f(x; W, b)\)其中：x 是输入（张量）W 和 b 是网络的 可训练参数（张量）f 是网络的计算逻辑（由多个层组成）y 是输出（张量）