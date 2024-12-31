import numpy as np # 引入numpy

"""
pytorch
1. 张量
2. 自动微分
3. 线形和非线形，非线形主要用激活函数来处理
"""



"""
numpy
"""
"""1. 创建随机数组"""
random_array = np.random.randn(2, 2)  # 2行2列的随机数组

"""2.向量点积"""
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

"""3. 矩阵的乘法"""
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = np.dot(A, B)  # 矩阵乘法
# 结果：
# [[1*5 + 2*7, 1*6 + 2*8],
#  [3*5 + 4*7, 3*6 + 4*8]]
# 即：
# [[19, 22],
#  [43, 50]]

"""4.在神经网络中的应用"""
# 在神经网络中，dot 方法用于计算前向传播中的线性变换。例如：

# 输入数据
x = np.random.randn(64, 1000)  # 64个样本，每个样本1000维

# 权重矩阵
w1 = np.random.randn(1000, 100)  # 输入层到隐藏层的权重

# 计算隐藏层输出
h = np.dot(x, w1)  # 矩阵乘法，得到隐藏层输出

"""
GRU
nn.GRU 是 PyTorch 中的一个类，用于实现 Gated Recurrent Unit (GRU)，即门控循环单元。
GRU 是一种 循环神经网络 (RNN) 的变体，专门用于处理 序列数据（如时间序列、文本、语音等）。

它通过引入门控机制来解决传统 RNN 的 梯度消失问题，
同时比 LSTM（长短期记忆网络）更简单，计算效率更高。
"""

import torch
import torch.nn as nn

"""
这段代码实现了一个基于GRU的文本分类器，主要包含以下部分：

1. 模型结构：
   - Embedding层：将输入的单词索引转换为稠密的词向量表示
   - GRU层：处理序列数据，捕捉文本的上下文信息
   - 全连接层：将GRU的输出映射到分类结果

2. 参数说明：
   - vocab_size: 词汇表大小，决定Embedding层的输入维度
   - embed_dim: 词向量的维度
   - hidden_size: GRU隐藏层的维度
   - num_classes: 分类的类别数

3. 前向传播流程：
   - 输入形状为(batch_size, seq_len)的单词索引
   - 通过Embedding层转换为(batch_size, seq_len, embed_dim)的词向量
   - GRU层处理序列，输出最后一个时间步的隐藏状态
   - 全连接层将隐藏状态映射到分类结果

4. 特点：
   - 使用GRU处理序列数据，能够捕捉文本的上下文信息
   - 相比LSTM，GRU结构更简单，计算效率更高
   - 适用于文本分类、情感分析等任务

5. 使用示例：
   - 创建模型实例，指定词汇表大小、词向量维度等参数
   - 输入形状为(batch_size, seq_len)的单词索引
   - 输出形状为(batch_size, num_classes)的分类结果

这个模型可以用于各种文本分类任务，如情感分析、主题分类等。
"""

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        # nn本身就有embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, seq_len)
        x = self.embedding(x)  # 嵌入层: (batch_size, seq_len, embed_dim)
        _, hn = self.gru(x)    # GRU 输出: hn 的形状为 (num_layers, batch_size, hidden_size)
        hn = hn[-1]            # 取最后一层的隐藏状态: (batch_size, hidden_size)
        output = self.fc(hn)   # 全连接层: (batch_size, num_classes)
        return output

# 定义模型
model = TextClassifier(vocab_size=1000, embed_dim=128, hidden_size=256, num_classes=10)

# 输入数据 (batch_size=32, seq_len=20)
input_data = torch.randint(0, 1000, (32, 20))

# 前向传播
output = model(input_data)
print(output.shape)  # 输出形状: (batch_size=32, num_classes=10)


