""" 
神经网络可以通过 torch.nn 包来构建。
现在对于自动梯度(autograd)有一些了解，神经网络是基于自动梯度 (autograd)来定义一些模型。
一个 nn.Module 包括层和一个方法 forward(input) 它会返回输出(output)。
简单的前馈神经网络，它接收输入，让输入一个接着一个的通过一些层，最后给出输出.


一个典型的神经网络训练过程包括以下几点：

1.定义一个包含可训练参数的神经网络

2.迭代整个输入

3.通过神经网络处理输入

4.计算损失(loss)

5.反向传播梯度到神经网络的参数

6.更新网络的参数，典型的用一个简单的更新方法：
weight = weight - learning_rate *gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# 定义神经网络
class Net(nn.Module):

    # 初始化
    # 卷积层和全连接层是神经网络中的两种主要层类型。
    # 卷积层主要用于处理图像数据，它通过卷积核在输入数据上滑动，提取局部特征。
    # 全连接层则将前一层的所有神经元与当前层的所有神经元连接，用于整合前面层提取的特征，并输出最终的结果。
    # 在卷积层中，卷积核的大小、步幅和填充方式都会影响输出的特征图的大小。
    # 全连接层通常位于网络的末端，用于将前面层提取的特征映射到输出类别或数值。

    # 输入通道数通常由输入数据的特征维度决定，
    # 例如对于灰度图像，输入通道数为1，对于彩色图像，输入通道数为3。
    # 输出通道数决定了卷积层提取的特征数量，通常根据网络设计和任务需求来设置。
    # 卷积核的大小影响特征提取的局部感受野，通常选择3x3或5x5等常见尺寸


    # 全连接层的输入大小由前一层的输出大小决定，输出大小通常根据下一层的输入需求或最终的输出类别数来设置。
    # 例如，第一个全连接层的输入大小为16*5*5，因为第二个卷积层的输出通道数为16，且经过两次2x2的最大池化后，特征图大小变为5x5。
    # 第二个全连接层的输入大小为120，输出大小为84，这是根据网络设计和任务需求来设置的。
    # 第三个全连接层的输入大小为84，输出大小为10，因为最终的输出类别数为10。


    def __init__(self):
        super(Net, self).__init__()
        # 卷积层和全连接层的主要区别：
        # 1. 连接方式：
        #    - 卷积层：局部连接，每个神经元只与输入数据的局部区域相连
        #    - 全连接层：全局连接，每个神经元与前一层的所有神经元相连
        
        # 2. 参数共享：
        #    - 卷积层：使用共享的卷积核，大大减少参数量
        #    - 全连接层：每个连接都有独立的权重参数
        
        # 3. 空间信息：
        #    - 卷积层：保留输入数据的空间结构信息
        #    - 全连接层：丢失空间信息，将输入展平为一维向量
        
        # 4. 适用场景：
        #    - 卷积层：适合处理图像、视频等具有空间结构的数据
        #    - 全连接层：适合处理特征已经提取好的数据，通常用于网络末端
        # 定义第一个卷积层，输入通道为1，输出通道为6，卷积核大小为5x5

        # 卷积层的主要作用是提取输入数据的局部特征
        # 通过卷积核在输入数据上滑动，可以检测到边缘、纹理等局部特征
        # 卷积层的参数共享机制使得它能够高效地提取特征，同时减少参数量
        
        # 全连接层的主要作用是将前面层提取的特征进行整合和映射
        # 它通过将前一层的所有神经元与当前层的所有神经元连接
        # 可以将局部特征组合成更高级的全局特征
        # 通常用于将提取的特征映射到最终的输出类别或数值
        
        # 卷积层和全连接层在神经网络中通常配合使用：
        # 卷积层负责提取局部特征，全连接层负责整合全局特征
        # 这种组合使得神经网络能够同时捕捉局部和全局信息
        # 在图像处理任务中，通常先使用卷积层提取特征，再使用全连接层进行分类


        self.conv1 = nn.Conv2d(1, 6, 5)
        # 定义第二个卷积层，输入通道为6，输出通道为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, 5)


        # 定义第一个全连接层，输入大小为16*5*5，输出大小为120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二个全连接层，输入大小为120，输出大小为84
        self.fc2 = nn.Linear(120, 84)
        # 定义第三个全连接层，输入大小为84，输出大小为10
        self.fc3 = nn.Linear(84, 10)


    #前馈函数
    # ReLU激活是一种非线性激活函数，它将所有负值变为0，正值保持不变。
    # 公式为：ReLU(x) = max(0, x)
    # 最大池化是一种下采样操作，、
    # 它通过在输入特征图上滑动一个固定大小的窗口，并取窗口内最大值来减少特征图的尺寸。
    # 最大池化有助于保留显著特征，同时减少计算量和参数数量。

    # forward过程是神经网络的前向传播过程，主要用于预测
    # 它接收输入数据，经过网络各层的计算，最终输出预测结果
    # 这个过程不涉及参数更新，只是根据当前网络参数进行计算
    # 在训练时，forward的输出会与真实标签计算损失，用于反向传播
    # 在推理时，forward的输出就是模型的预测结果

    def forward(self, x):
        # 对第一个卷积层的输出进行ReLU激活，然后进行2x2的最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 对第二个卷积层的输出进行ReLU激活，然后进行2x2的最大池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)


        # 将张量展平为一维，以便输入全连接层
        x = x.view(-1, self.num_flat_features(x))
        # 对第一个全连接层的输出进行ReLU激活
        x = F.relu(self.fc1(x))
        # 对第二个全连接层的输出进行ReLU激活
        x = F.relu(self.fc2(x))
        # 输出第三个全连接层的输出
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        # 获取张量的所有维度，除了批次维度
        size = x.size()[1:]
        # 计算所有维度的乘积，得到展平后的特征数量
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# 创建神经网络实例
net = Net()
# 打印神经网络结构
print(net)


#定义了一个前馈函数，然后反向传播函数被自动通过 autograd 定义了。你可以使用任何张量操作在前馈函数上。
#一个模型可训练的参数可以通过调用 net.parameters() 返回：

########################################################################
# You just have to define the ``forward`` function, and the ``backward``
# function (where gradients are computed) is automatically defined for you
# using ``autograd``.
# You can use any of the Tensor operations in the ``forward`` function.
#
# The learnable parameters of a model are returned by ``net.parameters()``

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

########################################################################
# Let try a random 32x32 input
# Note: Expected input size to this net(LeNet) is 32x32. To use this net on
# MNIST dataset, please resize the images from the dataset to 32x32.

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

########################################################################
# Zero the gradient buffers of all parameters and backprops with random
# gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

########################################################################
# .. note::
#
#     ``torch.nn`` only supports mini-batches. The entire ``torch.nn``
#     package only supports inputs that are a mini-batch of samples, and not
#     a single sample.
#
#     For example, ``nn.Conv2d`` will take in a 4D Tensor of
#     ``nSamples x nChannels x Height x Width``.
#
#     If you have a single sample, just use ``input.unsqueeze(0)`` to add
#     a fake batch dimension.
#
# Before proceeding further, let's recap all the classes you’ve seen so far.
#
# **Recap:**
#   -  ``torch.Tensor`` - A *multi-dimensional array* with support for autograd
#      operations like ``backward()``. Also *holds the gradient* w.r.t. the
#      tensor.
#   -  ``nn.Module`` - Neural network module. *Convenient way of
#      encapsulating parameters*, with helpers for moving them to GPU,
#      exporting, loading, etc.
#   -  ``nn.Parameter`` - A kind of Tensor, that is *automatically
#      registered as a parameter when assigned as an attribute to a*
#      ``Module``.
#   -  ``autograd.Function`` - Implements *forward and backward definitions
#      of an autograd operation*. Every ``Tensor`` operation, creates at
#      least a single ``Function`` node, that connects to functions that
#      created a ``Tensor`` and *encodes its history*.
#
# **At this point, we covered:**
#   -  Defining a neural network
#   -  Processing inputs and calling backward
#
# **Still Left:**
#   -  Computing the loss
#   -  Updating the weights of the network
#
# Loss Function
# -------------
# A loss function takes the (output, target) pair of inputs, and computes a
# value that estimates how far away the output is from the target.
#
# There are several different
# `loss functions <https://pytorch.org/docs/nn.html#loss-functions>`_ under the
# nn package .
# A simple loss is: ``nn.MSELoss`` which computes the mean-squared error
# between the input and the target.
#
# For example:

output = net(input)
target = torch.randn(10)  # a dummy target, for example
print("target",target)
target = target.view(1, -1)  # make it the same shape as output
#得到一个标准执行方法，
# MSE Loss（均方误差损失）是一种常用的损失函数，用于衡量模型预测值与真实值之间的差异。
# 它计算的是预测值与真实值之间差异的平方的平均值。
# 公式为：MSE = (1/n) * Σ(y_pred - y_true)^2
# 其中，n是样本数量，y_pred是模型的预测值，y_true是真实值。
# 在PyTorch中，可以使用`nn.MSELoss()`来创建一个均方误差损失函数。

criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

########################################################################
# Now, if you follow ``loss`` in the backward direction, using its
# ``.grad_fn`` attribute, you will see a graph of computations that looks
# like this:
#
# ::
#
#     input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#           -> view -> linear -> relu -> linear -> relu -> linear
#           -> MSELoss
#           -> loss
#
# So, when we call ``loss.backward()``, the whole graph is differentiated
# w.r.t. the loss, and all Tensors in the graph that has ``requires_grad=True``
# will have their ``.grad`` Tensor accumulated with the gradient.
#
# For illustration, let us follow a few steps backward:

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

########################################################################
# Backprop
# --------
# To backpropagate the error all we have to do is to ``loss.backward()``.
# You need to clear the existing gradients though, else gradients will be
# accumulated to existing gradients.
#
#
# Now we shall call ``loss.backward()``, and have a look at conv1's bias
# gradients before and after the backward.


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

########################################################################
# Now, we have seen how to use loss functions.
#
# **Read Later:**
#
#   The neural network package contains various modules and loss functions
#   that form the building blocks of deep neural networks. A full list with
#   documentation is `here <https://pytorch.org/docs/nn>`_.
#
# **The only thing left to learn is:**
#
#   - Updating the weights of the network
#
# Update the weights
# ------------------
# The simplest update rule used in practice is the Stochastic Gradient
# Descent (SGD):
#
#      ``weight = weight - learning_rate * gradient``
#
# We can implement this using simple python code:
#
# .. code:: python
#
#     learning_rate = 0.01
#     for f in net.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#
# However, as you use neural networks, you want to use various different
# update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we built a small package: ``torch.optim`` that
# implements all these methods. Using it is very simple:

# a easy way to use the optimizer
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


###############################################################
# .. Note::
#
#       Observe how gradient buffers had to be manually set to zero using
#       ``optimizer.zero_grad()``. This is because gradients are accumulated
#       as explained in `Backprop`_ section.


