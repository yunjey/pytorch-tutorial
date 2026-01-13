import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import ssl
# 逻辑回归
ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == '__main__':
    # Hyper-parameters
    input_size = 28 * 28    # 784
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset (images and labels)
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Logistic regression model
    # 逻辑回归（多分类）：输出类别数量的得分
    # 逻辑回归：输出层维度 = 类别数量（这里10个数字）
    # 线性回归：输出层维度 = 1（预测单一连续值）

    model = nn.Linear(input_size, num_classes)

    # Loss and optimizer
    # nn.CrossEntropyLoss() computes softmax internally
    # 逻辑回归：使用交叉熵损失（自动包含softmax）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, input_size)
            # 逻辑回归：CrossEntropyLoss内部已包含softmax激活
            # Forward pass
            outputs = model(images)  # 输出是原始得分（logits）
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    """
    线性回归和逻辑回归的核心区别在于：

    任务目标不同：回归 vs 分类
    输出处理不同：直接输出 vs 激活函数映射
    损失函数不同：MSE vs 交叉熵
    """
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    """
    一、核心区别对比表
    对比维度	均方误差损失 (MSE)	交叉熵损失 (Cross-Entropy)
    数学定义	$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$	二分类：$CE = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log\hat{y}i + (1-y_i)\log(1-\hat{y}i)]$
    多分类：$CE = -\frac{1}{n}\sum{i=1}^{n}\sum{c=1}^{C}y_{ic}\log\hat{y}_{ic}$
    适用任务	回归任务（预测连续值，如房价、温度）	分类任务（预测离散类别，如图片分类、文本分类）
    输出范围假设	预测值$\hat{y}$可任意实数（$-\infty, +\infty$）	预测值需转换为概率分布（$(0, 1)$区间）
    激活函数配合	通常不需要特定激活函数（线性输出）
    或配合sigmoid/tanh（约束输出范围）	二分类：配合sigmoid激活
    多分类：配合softmax激活（PyTorch中CrossEntropyLoss内部自动计算）
    梯度特性	梯度与预测偏差$(y-\hat{y})$成正比
    预测远离真实值时梯度大，易不稳定	梯度与概率分布的差异相关
    训练更稳定，尤其适合分类任务
    
    场景	      推荐损失函数	代码示例
    预测连续值（如房价、温度）	MSE	criterion = nn.MSELoss()
    二分类（如垃圾邮件检测）	Binary Cross-Entropy	criterion = nn.BCELoss()
    多分类（如MNIST数字识别）	Cross-Entropy	criterion = nn.CrossEntropyLoss()
    简单来说：回归用MSE，分类用CrossEntropy，这是深度学习中的"黄金法则"之一！
    """