"""
pre：先给vscode安装一个插件： Python Image Preview

通常来说，当你处理图像，文本，语音或者视频数据时，你可以使用标准 python 包将数据加载成 numpy 数组格式，然后将这个数组转换成 torch.*Tensor
对于图像，可以用 Pillow，OpenCV
对于语音，可以用 scipy，librosa
对于文本，可以直接用 Python 或 Cython 基础数据加载模块，或者用 NLTK 和 SpaCy


特别是对于视觉，我们已经创建了一个叫做 totchvision 的包，
该包含有支持加载类似Imagenet，CIFAR10，MNIST 等公共数据集的数据
加载模块 torchvision.datasets 和支持加载图像数据数据转换模块 torch.utils.data.DataLoader。

task: 训练一个图像分类器:
我们将按次序的做如下几步：


使用torchvision加载并且归一化CIFAR10的训练和测试数据集
定义一个卷积神经网络
定义一个损失函数
在训练样本数据上训练网络
在测试样本数据上测试网络

"""

# torchvision 数据集的输出是范围在[0,1]之间的 PILImage，
# 我们将他们转换成归一化范围为[-1,1]之间的张量 Tensors。
# 导入必要的库

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 定义数据预处理步骤
# transforms.Compose 将多个数据预处理步骤组合在一起
transform = transforms.Compose(
    [transforms.ToTensor(),  # 将PIL图像转换为张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化张量，使其值在[-1, 1]之间

# 加载CIFAR10训练数据集
# root='./data' 指定数据集的存储路径
# train=True 表示加载训练数据集
# download=True 如果数据集不存在，则下载数据集
# transform=transform 对数据集应用上述定义的预处理步骤
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# 创建训练数据加载器
# trainset 是加载的训练数据集
# batch_size=4 每次加载4个样本
# shuffle=True 在每个epoch开始时打乱数据
# num_workers=2 使用2个子进程加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# 加载CIFAR10测试数据集
# train=False 表示加载测试数据集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# 创建测试数据加载器
# testset 是加载的测试数据集
# batch_size=4 每次加载4个样本
# shuffle=False 不打乱数据
# num_workers=2 使用2个子进程加载数据
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# 定义CIFAR10数据集的类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# mac上是MPS
print(torch.backends.mps.is_available())  # 输出 True 表示支持

# 定义一个继承自 nn.Module 的神经网络类 Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #创建第一个卷积层conv1，它有3个输入通道（对应于RGB图像），6个输出通道，以及一个5x5的卷积核。
        self.conv1 = nn.Conv2d(3, 6, 5)

        """ 什么时候需要池化层，什么时候不需要呢"""
        # 池化层通常在卷积层之后使用，用于减少特征图的空间维度（高度和宽度），从而减少计算量和参数数量。
        # 池化层通过在输入特征图上滑动一个固定大小的窗口，并取窗口内最大值或平均值来减少特征图的尺寸。
        # 最大池化有助于保留显著特征，同时减少计算量和参数数量。
        # 如果特征图的空间维度已经很小，或者你希望保留更多的空间信息，可以选择不使用池化层。
        # 例如，在处理非常小的图像或需要保留更多空间信息的任务中，可以跳过池化层。
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #创建第一个全连接层fc1，它将第二个卷积层的输出（经过池化后）展平成一个一维向量，大小为16 * 5 * 5，并映射到120个神经元。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 创建第二个全连接层fc2，它将第一个全连接层的输出映射到84个神经元。
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        
    def forward(self, x):
        #输入x首先通过卷积层conv1，然后应用ReLU激活函数，最后通过池化层pool。这个过程用于提取输入数据的特征。
        x = self.pool(F.relu(self.conv1(x)))
        #同上
        x = self.pool(F.relu(self.conv2(x)))
        #这行代码将经过两次卷积和池化的特征图展平成一维向量，以便输入到全连接层。-1表示自动计算该维度的大小，以保持数据总量不变。
        x = x.view(-1, 16 * 5 * 5)
        # 展平后的数据通过第一个全连接层fc1，并应用ReLU激活函数。
        """为什么要用ReLU激活函数"""
        # ReLU（Rectified Linear Unit）函数是一种常用的激活函数，它在神经网络中起到了非线性变换的作用。
        # ReLU函数的形式是 f(x) = max(0, x)，即对于输入x，如果x大于0，则输出x；如果x小于或等于0，则输出0。
        # 使用ReLU函数的原因主要有以下几点：
        # 1. 非线性：ReLU函数引入了非线性特性，使得神经网络可以学习并表示复杂的非线性关系。
        # 2. 计算简单：ReLU函数的计算非常简单，只需要一个比较操作和一个取最大值操作，这使得它在计算上非常高效。
        # 3. 避免梯度消失：在深层神经网络中，使用Sigmoid或Tanh等激活函数容易导致梯度消失问题，而ReLU函数在x>0时梯度为1，可以有效缓解这一问题。
        # 4. 稀疏激活性：ReLU函数在x<=0时输出0，这使得神经网络中的某些神经元在某些情况下不会被激活，从而增加了网络的稀疏性，减少了计算量。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        """为什么最后一个不用ReLU函数"""
        # 在神经网络的最后一层，通常不使用ReLU激活函数，原因如下：
        # 1. 输出范围：ReLU函数的输出范围是[0, +∞)，而许多任务的输出范围是有限的。例如，在分类任务中，输出通常是概率分布，范围在[0, 1]之间。
        # 2. 损失函数：在分类任务中，通常使用交叉熵损失函数，它要求输出层的激活函数是softmax函数，以确保输出是有效的概率分布。
        # 3. 回归任务：在回归任务中，输出通常是连续的实数值，而不是非负值。使用ReLU函数会限制输出的范围，导致模型无法输出负值。
        # 4. 避免非线性：在最后一层，通常希望输出是线性的，以便直接反映模型的预测结果，而不是经过非线性变换后的结果。
        x = self.fc3(x)
        return x

net = Net()

"""
显示一些图片
"""

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



"""
定义一个损失函数和优化器 让我们使用分类交叉熵Cross-Entropy 作损失函数，动量SGD做优化器。
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # momentum=0.9 表示使用动量优化算法


if __name__ == '__main__':

    """显示一些图片"""
    # get some random training images

    sample_images = []
    sample_labels = []
    # trainloader[:1] 表示从trainloader中取出第一个批次的数据。
    for i, data in enumerate(trainloader, 0):
        if i == 0:
            inputs, labels = data
            sample_images = inputs
            sample_labels = labels

    # imshow(torchvision.utils.make_grid(sample_images))
    # print labels
    # classes 是从数据集中获取的类别标签列表。通常在加载数据集时，数据集会提供一个包含所有类别名称的列表。
    # 例如，在 CIFAR-10 数据集中，classes 可能包含 ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']。
    print(sample_images,sample_labels)
    # 这些占位符后面的% classes[sample_labels[j]] for j in range(4))
    print(' '.join('%5s' % classes[sample_labels[j]] for j in range(4)))
   


    """
    开始训练，这里是两轮
    """
    for epoch in range(2): 

        running_loss = 0.0
        # 这里的后一个0的意思是
        # 在enumerate函数中，第二个参数0表示从0开始计数。
        # 这意味着在训练循环中，i将从0开始，而不是从1开始。
        # 这对于计算损失值的平均值和打印训练进度非常有用。
        for i, data in enumerate(trainloader, 0):

            # 获取输入信息
            inputs, labels = data

            # 先将梯度归零
            optimizer.zero_grad()

            # forward + backward + optimize
            # 先进行前向传播（forward），计算输出和损失函数。
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 然后进行反向传播（backward），计算梯度。
            loss.backward()
            # 最后，使用优化器更新模型参数。
            optimizer.step()

            # `running_loss` 是一个累加器，用于累加每个小批次的损失值。
            running_loss += loss.item()
            # `if i % 2000 == 1999:` 这个条件判断语句确保每2000个小批次打印一次平均损失值。
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                # `running_loss = 0.0` 这行代码将累加器重置为0，以便开始计算下一个2000个小批次的平均损失值。
                running_loss = 0.0

    print('Finished Training')


    """
    part1: 对于sample images预测一下
    """

    outputs = net(sample_images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                 for j in range(4)))

    """
    part2: 在整个数据集上来一遍呀
    """
    correct = 0
    total = 0
    #验证的时候不需要保留梯度缓存
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次数据
        for data in testloader:
            # 获取图像和对应的标签
            images, labels = data
            # 将图像输入到网络中进行前向传播，得到输出
            outputs = net(images)
            # 获取输出中概率最大的类别作为预测结果
            # 参数的1表示在维度1上进行最大值操作，即在每个样本的10个类别中选择概率最大的那个类别。
            _, predicted = torch.max(outputs.data, 1)
            # size(0) 返回的是当前批次中图像的数量，即 batch_size。
            # 因为 labels 是一个一维张量，size(0) 返回的是这个张量的第一个维度的大小，也就是图像的数量。
            # 累加当前批次中图像的总数
            total += labels.size(0)
            # 计算当前批次中预测正确的图像数量，并累加到correct中
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
