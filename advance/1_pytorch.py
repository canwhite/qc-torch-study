""" 
PyTorch的核心是两个主要特征：
一个n维张量，类似于numpy，但可以在GPU上运行
搭建和训练神经网络时的自动微分/求导机制
"""

"""=============================part1: numpy========================"""

import numpy as np 

# 定义参数
N, D_in, H, D_out = 64, 1000, 100, 10  # N是批量大小; D_in是输入维度; H是隐藏层维度; D_out是输出维度

# 创建随机输入和输出数据
x = np.random.randn(N, D_in)  # 生成随机输入数据x，形状为(N, D_in)
y = np.random.randn(N, D_out)  # 生成随机输出数据y，形状为(N, D_out)

# 权重在神经网络中起着至关重要的作用：
# 1. 权重决定了输入数据如何被转换和传递
# 2. 在卷积层中，权重就是卷积核（filter），它决定了如何提取特征
# 3. 在全连接层中，权重决定了输入特征如何组合成输出
# 4. 通过训练不断调整权重，使网络能够学习到数据中的模式

# 设置卷积层时需要考虑：
# 1. 输入通道数：取决于输入数据的特征维度
# 2. 输出通道数：决定提取多少种特征
# 3. 卷积核大小：决定感受野的大小
# 4. 步长（stride）：决定卷积核移动的步幅
# 5. 填充（padding）：决定是否在输入边缘补零

# 例如：
# nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
# 表示一个卷积层，输入3通道，输出16通道，使用3x3卷积核，步长为1，填充为1

# 设置全连接层时需要考虑：
# 1. 输入维度：取决于输入数据的特征维度
# 2. 输出维度：决定输出多少个特征
w1 = np.random.randn(D_in, H)  # 随机初始化权重矩阵w1，形状为(D_in, H)
w2 = np.random.randn(H, D_out)  # 随机初始化权重矩阵w2，形状为(H, D_out)



""" 
先做线形变换：

1. 输入数据 x 和权重 w1 的维度
x 是一个形状为 (N, D_in) 的矩阵，其中 N 是样本的数量，D_in 是每个样本的输入特征数。

w1 是一个形状为 (D_in, H) 的矩阵，其中 H 是隐藏层的神经元数量。

2. 矩阵乘法的操作
矩阵乘法 x.dot(w1) 的结果是一个形状为 (N, H) 的矩阵，其中每个元素是输入数据 x 与权重 w1 的线性组合。

--------
通过矩阵乘法 x.dot(w1)，我们得到了隐藏层的激活值 h。
这个矩阵 h 的每一行对应一个样本，每一列对应隐藏层的一个神经元。
h 表示的是输入数据 x 经过权重 w1 线性变换后的结果。

再做非线形的变换：
在得到 h 之后，通常会应用一个非线性激活函数（如 ReLU、sigmoid 等）来引入非线性特性，然后再进行下一步的计算（如计算输出层的值）。


那线形变换之后，为什么要进行非线性变换呢？  

线性变换具有局限性，一般线性变换之后还是线性，这有局限，

为了克服线性变换的局限性，神经网络在每一层的线性变换之后引入非线性激活函数

"""


learning_rate = 1e-6
for t in range(500):
    #向前传递
    h = x.dot(w1) #先做线形
    h_relu = np.maximum(h, 0) #再做非线形
    y_pred = h_relu.dot(w2) #再做线形

    #计算损失
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反向传播，计算w1和w2对loss的梯度
    # 这里进行反向传播计算梯度，用于更新权重参数

    # 1. 首先计算预测值 y_pred 对损失 loss 的梯度
    #由于 loss = sum((y_pred - y)^2)，所以 grad_y_pred = 2*(y_pred - y)
    grad_y_pred = 2.0 * (y_pred - y)

    # 2. 然后计算 w2 的梯度 grad_w2
    # 由于 y_pred = h_relu.dot(w2)，所以 grad_w2 = h_relu.T.dot(grad_y_pred) 
    grad_w2 = h_relu.T.dot(grad_y_pred)

    # 3. 接着计算 h_relu 的梯度 grad_h_relu
    # 由于 y_pred = h_relu.dot(w2)，所以 grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h_relu = grad_y_pred.dot(w2.T)

    # 4. 然后计算 h 的梯度 grad_h
    # 由于 h_relu = max(h, 0)，所以当 h < 0 时梯度为 0，否则等于 grad_h_relu
    grad_h = grad_h_relu.copy()

    # 5. 最后计算 w1 的梯度 grad_w1
    #   由于 h = x.dot(w1)，所以 grad_w1 = x.T.dot(grad_h)
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)


    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    





"""=============================part2: tensors========================"""
import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device（“cuda：0”）＃取消注释以在GPU上运行

# N是批量大小; D_in是输入维度;
# H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

#创建随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 前向传递：计算预测y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算和打印损失
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop计算w1和w2相对于损耗的梯度
    """这是原本的求导过程，很麻烦是不是，"""
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2



"""=============================part3: backward自动求导，对上边tensor的更新========================"""
"""
这听起来很复杂，在实践中使用起来非常简单。 
如果我们想计算某些的tensor的梯度，
我们只需要在建立这个tensor时加入这么一句：
requires_grad=True。
这个tensor上的任何PyTorch的操作都将构造一个计算图，从而允许我们稍后在图中执行反向传播。
如果这个tensor x的requires_grad=True，那么反向传播之后x.grad将会是另一个张量，其为x关于某个标量值的梯度。

--------------
有时可能希望防止PyTorch在requires_grad=True的张量执行某些操作时构建计算图；
例如，在训练神经网络时，我们通常不希望通过权重更新步骤进行反向传播。
在这种情况下，我们可以使用torch.no_grad()上下文管理器来防止构造计算图。

"""

import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device（“cuda：0”）＃取消注释以在GPU上运行

# N是批量大小; D_in是输入维度;
# H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机Tensors以保持输入和输出。
# 设置requires_grad = False表示我们不需要计算渐变
# 在向后传球期间对于这些Tensors。
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 为权重创建随机Tensors。
# 设置requires_grad = True表示我们想要计算渐变
# 在向后传球期间尊重这些张贴。
"""requires_grad=True """
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：使用tensors上的操作计算预测值y; 
      # 由于w1和w2有requires_grad=True，涉及这些张量的操作将让PyTorch构建计算图，
    # 从而允许自动计算梯度。由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 使用Tensors上的操作计算和打印丢失。
    # loss是一个形状为()的张量
    # loss.item() 得到这个张量对应的python数值
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用autograd计算反向传播。这个调用将计算loss对所有requires_grad=True的tensor的梯度。
    # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()

    # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，
    # 所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 反向传播后手动将梯度设置为零
        w1.grad.zero_()
        w2.grad.zero_()


"""=============================part3: 自定义backward========================"""
import torch

class MyReLU(torch.autograd.Function):
    """
    我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，
    并完成张量的正向和反向传播。
    """
    @staticmethod
    def forward(ctx, x):
        """
        在正向传播中，我们接收到一个上下文对象和一个包含输入的张量；
        我们必须返回一个包含输出的张量，
        并且我们可以使用上下文对象来缓存对象，以便在反向传播中使用。
        """
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传播中，我们接收到上下文对象和一个张量，
        其包含了相对于正向传播过程中产生的输出的损失的梯度。
        我们可以从上下文对象中检索缓存的数据，
        并且必须计算并返回与正向传播的输入相关的损失的梯度。
        """
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x


device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# N是批大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# 产生随机权重的张量
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 正向传播：使用张量上的操作来计算输出值y；
    # 我们通过调用 MyReLU.apply 函数来使用自定义的ReLU
    y_pred = MyReLU.apply(x.mm(w1)).mm(w2)

    # 计算并输出loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用autograd计算反向传播过程。
    loss.backward()

    with torch.no_grad():
        # 用梯度下降更新权重
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 在反向传播之后手动清零梯度
        w1.grad.zero_()
        w2.grad.zero_()

"""=============================part4: nn初使用  ========================"""
# -*- coding: utf-8 -*-
import torch

# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

#创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用nn包将我们的模型定义为一系列的层。
# nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出。
# 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量。
# 在构造模型之后，我们使用.to()方法将其移动到所需的设备。
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# nn包还包含常用的损失函数的定义；
# 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数。
# 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值;
# 这是为了与前面我们手工计算损失的例子保持一致，
# 但是在实践中，通过设置reduction='elementwise_mean'来使用均方误差作为损失更为常见。
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # 前向传播：通过向模型传入x计算预测的y。
    # 模块对象重载了__call__运算符，所以可以像函数那样调用它们。
    # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量。
    y_pred = model(x)

     # 计算并打印损失。
     # 传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量。
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 反向传播之前清零梯度
    model.zero_grad()

    # 反向传播：计算模型的损失对所有可学习参数的导数（梯度）。
    # 在内部，每个模块的参数存储在requires_grad=True的张量中，
    # 因此这个调用将计算模型中所有可学习参数的梯度。
    loss.backward()

    # 使用梯度下降更新权重。
    # 每个参数都是张量，所以我们可以像我们以前那样可以得到它的数值和梯度
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

"""=============================part5: nn再进化  ========================"""

import torch

# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用nn包定义模型和损失函数
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
loss_fn = torch.nn.MSELoss(reduction='sum')

# 使用optim包定义优化器（Optimizer）。Optimizer将会为我们更新模型的权重。
# 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法。
# Adam构造函数的第一个参数告诉优化器应该更新哪些张量。
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):

    # 前向传播：通过像模型输入x计算预测的y
    y_pred = model(x)

    # 计算并打印loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
    optimizer.zero_grad()

    # 反向传播：根据模型的参数计算loss的梯度
    loss.backward()

    # 调用Optimizer的step函数使它所有参数更新
    optimizer.step()


"""=============================part6: 自定义nn模块儿  ========================"""
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们实例化了两个nn.Linear模块，并将它们作为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在前向传播的函数中，我们接收一个输入的张量，也必须返回一个输出张量。
        我们可以使用构造函数中定义的模块以及张量上的任意的（可微分的）操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N是批大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 通过实例化上面定义的类来构建我们的模型。
model = TwoLayerNet(D_in, H, D_out)

# 构造损失函数和优化器。
# SGD构造函数中对model.parameters()的调用，
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数。
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # 前向传播：通过向模型传递x计算预测值y
    y_pred = model(x)

    #计算并输出loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 清零梯度，反向传播，更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


"""=============================part7: 究极体  ========================"""
import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用。
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        对于模型的前向传播，我们随机选择0、1、2、3，
        并重用了多次计算隐藏层的middle_linear模块。
        由于每个前向传播构建一个动态计算图，
        我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
        在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
        这是Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 实例化上面定义的类来构造我们的模型
model = DynamicNet(D_in, H, D_out)

# 构造我们的损失函数（loss function）和优化器（Optimizer）。
# 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用了momentum方法。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):

    # 前向传播：通过向模型传入x计算预测的y。
    y_pred = model(x)

    # 计算并打印损失
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # 清零梯度，反向传播，更新权重 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()