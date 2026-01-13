# 导入PyTorch核心库
import torch
import torch.nn as nn
# 导入TorchVision用于数据加载和预处理
import torchvision
import torchvision.transforms as transforms

# 设备配置：自动选择GPU（如果可用）否则使用CPU
# MPS是Apple Silicon GPU支持，这里也可以加上对MPS的支持
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

"""
RNN（循环神经网络）的用途与应用场景
RNN（Recurrent Neural Network，循环神经网络）是一类专门用于处理序列数据的深度学习模型，
其核心特点是能够记忆之前的信息并用于当前决策，这使得它在各种需要处理时序依赖关系的任务中表现出色。
RNN的核心特性
RNN通过在网络中引入循环连接，使模型能够：

处理任意长度的序列数据
捕捉序列中的时间依赖关系
保留序列的上下文信息

RNN的主要应用场景
1. 自然语言处理（NLP）
文本分类：情感分析、垃圾邮件检测、新闻分类
语言建模：预测下一个词的概率分布
机器翻译：将一种语言翻译成另一种语言
命名实体识别：识别文本中的人名、地名、组织名等
文本生成：自动生成文章、诗歌、对话等
2. 时间序列预测
股票价格预测：基于历史价格预测未来走势
天气预报：基于气象数据预测未来天气
电力负荷预测：预测未来电力需求
销售预测：预测产品未来销量
3. 语音处理
语音识别：将语音转换为文本
语音合成：将文本转换为语音
说话人识别：识别说话人的身份
4. 图像与视频分析
图像描述生成：为图像生成文字描述
视频分析：行为识别、动作检测
手写体识别：如代码示例中的MNIST数字分类
"""
# 定义RNN模型类，继承自nn.Module
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        RNN模型初始化函数
        :param input_size: 输入特征维度 (MNIST图像的每行像素数：28)
        :param hidden_size: 隐藏层维度 (LSTM单元的隐藏状态大小：128)
        :param num_layers: LSTM层数 (2层)
        :param num_classes: 分类数量 (MNIST有10个数字类别)
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_layers = num_layers    # LSTM层数
        
        # 定义LSTM层：
        # - input_size: 输入特征维度
        # - hidden_size: 隐藏层维度
        # - num_layers: LSTM层数
        # - batch_first=True: 输入输出形状为(batch_size, seq_length, feature_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层：将LSTM输出映射到分类结果
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量，形状为(batch_size, sequence_length, input_size)
        :return: 输出张量，形状为(batch_size, num_classes)
        """
        # 初始化LSTM的隐藏状态h0和细胞状态c0
        # 形状：(num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 前向传播LSTM
        # out: LSTM的输出，形状为(batch_size, seq_length, hidden_size)
        # _: 包含最终隐藏状态和细胞状态的元组（这里未使用）
        out, _ = self.lstm(x, (h0, c0))

        # 解码最后一个时间步的隐藏状态用于分类
        # out[:, -1, :] 表示取所有样本的最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    # 超参数设置
    sequence_length = 28     # 序列长度 (MNIST图像的行数：28)
    input_size = 28          # 输入特征维度 (MNIST图像的列数：28)
    hidden_size = 128        # 隐藏层维度
    num_layers = 2           # LSTM层数
    num_classes = 10         # 分类数量 (0-9数字)
    batch_size = 100         # 批次大小
    num_epochs = 2           # 训练轮数
    learning_rate = 0.01     # 学习率

    # MNIST数据集加载
    # 训练集
    train_dataset = torchvision.datasets.MNIST(
        root='../../data/',      # 数据集保存路径
        train=True,              # 训练集
        transform=transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
        download=True            # 自动下载（如果本地没有）
    )

    # 测试集
    test_dataset = torchvision.datasets.MNIST(
        root='../../data/',
        train=False,             # 测试集
        transform=transforms.ToTensor()
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True  # 训练时打乱数据
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False  # 测试时不打乱数据
    )

    # 实例化RNN模型（many-to-one架构：多个时间步输入，一个输出）
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

    # 损失函数和优化器
    # 交叉熵损失：适用于多分类任务
    criterion = nn.CrossEntropyLoss()
    # Adam优化器：自适应学习率优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    total_step = len(train_loader)  # 每个epoch的总步数
    for epoch in range(num_epochs):  # 遍历每个epoch
        for i, (images, labels) in enumerate(train_loader):  # 遍历每个批次
            # 将图像重塑为序列数据：
            # MNIST图像原始形状：(batch_size, 1, 28, 28)
            # 重塑后形状：(batch_size, sequence_length=28, input_size=28)
            # 即把28x28的图像看作28个时间步，每个时间步输入28个像素
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)  # 标签移至设备

            # 前向传播
            outputs = model(images)  # 模型预测
            loss = criterion(outputs, labels)  # 计算损失

            # 反向传播和优化
            optimizer.zero_grad()  # 清除梯度
            loss.backward()       # 反向传播计算梯度
            optimizer.step()      # 更新参数

            # 打印训练信息
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 测试模型
    model.eval()  # 设置模型为评估模式（关闭dropout等）
    with torch.no_grad():  # 关闭梯度计算，节省内存和计算
        correct = 0  # 正确预测数
        total = 0    # 总样本数
        for images, labels in test_loader:  # 遍历测试集
            # 重塑图像并移至设备
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)  # 模型预测
            
            # 获取预测结果：torch.max返回最大值和索引，索引即为预测类别
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测数

        # 打印测试准确率
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # 保存模型权重
    torch.save(model.state_dict(), 'model.ckpt')
    print("Model weights saved to 'model.ckpt'")