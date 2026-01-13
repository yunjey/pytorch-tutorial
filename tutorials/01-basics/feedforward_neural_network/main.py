import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import ssl

# 前馈神经网络
"""
前馈神经网络是网络结构，反向传播是训练这个网络的核心算法。
神经网络的「骨架」—— 定义了「神经元如何分层、层与层如何连接、每层神经元数量、用什么激活函数」的整体框架，
决定了信息在网络中如何传递，是模型能拟合数据的基础，和 “反向传播（训练算法）” 是 “骨架” 和 “打磨骨架的方法” 的关系。

"""
ssl._create_default_https_context = ssl._create_unverified_context


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层→隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 隐藏层→输出层

    def forward(self, x):
        out = self.fc1(x) # 线性变换
        out = self.relu(out) # 非线性激活
        out = self.fc2(out) # 线性变换
        return out

"""
FNN与之前学习的模型对比
模型	        结构复杂度	    学习能力	             应用场景
线性回归	    简单（单层）	    只能学习线性关系	    简单回归任务
逻辑回归	    简单（单层+激活）	只能学习线性可分的分类	简单分类任务
前馈神经网络	复杂（多层+激活）	可学习复杂非线性关系	复杂分类/回归任务

"""
"""
前馈神经网络的作用
FNN的核心作用是学习输入与输出之间的复杂映射关系，主要用于两类任务：

分类任务：将输入数据分为不同类别（如代码中的MNIST数字分类）
回归任务：预测连续数值（如房价预测、股票价格预测）
其强大之处在于：通过多层结构和非线性激活，能够拟合几乎任何复杂的函数关系（这是神经网络的"万能近似定理"）。
"""
if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
