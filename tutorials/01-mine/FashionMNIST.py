import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 设备配置：优先使用GPU，否则使用CPU，M系列芯片可使用MPS
# if torch.backends.mps.is_available():
#     device = torch.device('mps')  # Apple Silicon M系列芯片加速
# elif torch.cuda.is_available():
#     device = torch.device('cuda')  # NVIDIA GPU加速
# else:
#     device = torch.device('cpu')   # CPU训练
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 在单个训练循环中，模型会对训练数据集（以批次形式输入）进行预测，并通过反向传播预测误差来调整模型的参数。
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 根据测试数据集检查模型的性能，以确保它在学习。
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

"""
下载模型数据
创建模型
优化模型参数
保存模型
加载模型
预测数据
"""
if __name__ == '__main__':
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    # 它是一个包含图像和对应标签的数据集对象。
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    # Creating Models
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    # Define model

    model = NeuralNetwork().to(device)
    print(f"model: {model}")
    # To train a model, we need a loss function and an optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval() # 设置模型为评估模式
    x, y = test_data[0][0], test_data[0][1]
    print(f"x: {x} ,y: {y}")
    with torch.no_grad():
        x = x.to(device)
        # logits 指的是 模型最后一层（通常是线性层 nn.Linear）的原始输出，这些输出 没有经过归一化，因此它们不是概率值。
        # Softmax 函数转换为概率分布：
        pred = model(x) # 模型预测，输出10个类别的logits
        # probs = torch.softmax(pred, dim=1)
        # print(probs)

        print(f"pred: {pred}")
        # 在分类任务中，我们通常只需要找到最大logits对应的类别即可，不需要转换为概率：
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')