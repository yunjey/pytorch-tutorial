import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 线性回归

if __name__ == '__main__':
    # Hyper-parameters
    input_size = 1
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Toy dataset
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    # Linear regression model
    # 创建一个线性回归模型（也称为全连接层或仿射变换层）
    # 在PyTorch中，nn.Linear 会自动初始化权重 w 和偏置 b
    model = nn.Linear(input_size, output_size)

    # Loss and optimizer
    # 定义损失函数，用于衡量模型预测值与真实值之间的差异
    # 这里使用均方误差损失（Mean Squared Error Loss）
    criterion = nn.MSELoss()
    # 创建优化器，用于更新模型的参数（权重 w 和偏置 b）
    # 学习率的作用：学习率过大可能导致模型训练不稳定，过小则训练速度太慢
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(x_train)
        targets = torch.from_numpy(y_train)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        # 反向传播计算梯度：
        # 计算损失函数关于模型参数的梯度
        loss.backward()
        # 更新模型参数：
        # 使用优化器根据计算得到的梯度更新模型参数
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    """
    模型训练时确实使用了 x_train（输入）和 y_train（目标），但训练完成后：

    我们需要验证模型在训练数据上的拟合效果
    通过对相同的 x_train 进行预测，得到 predicted（模型输出）
    然后将 predicted 与真实的 y_train 对比绘图，直观展示模型学习的线性关系
    """
    # .detach()  # 从计算图中分离张量
    # Plot the graph
    predicted = model(
        torch.from_numpy(x_train)).detach().numpy()  # 将PyTorch张量转换回NumPy数组 因为matplotlib绘图库需要NumPy数组格式 方便后续的可视化操作
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.show()

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
