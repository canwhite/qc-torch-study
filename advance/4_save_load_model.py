"""
当保存和加载模型时，需要熟悉三个核心功能：

torch.save：将序列化对象保存到磁盘。此函数使用Python的pickle模块进行序列化。使用此函数可以保存如模型、tensor、字典等各种对象。
torch.load：使用pickle的unpickling功能将pickle对象文件反序列化到内存。此功能还可以有助于设备加载数据。
torch.nn.Module.load_state_dict：使用反序列化函数 state_dict 来加载模型的参数字典。
"""


"""part 1 : 什么是状态字典？state_dict 是什么？"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path

# 在 PyTorch 中最常见的模型保存使用‘.pt’或者是‘.pth’作为模型文件扩展名。
save_path = Path(__file__).parent / "model_save" / "model.pth"
save_model_path = Path(__file__).parent / "model_save" / "model_total.pth"
checkpoint_path = Path(__file__).parent / "model_save" / "checkpoint.pth"
# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # MaxPool2d 是最大池化层，用于对输入特征图进行下采样
        # 它通过在输入特征图上滑动一个固定大小的窗口，取窗口内的最大值作为输出
        # 主要作用：
        # 1. 降低特征图的空间维度，减少计算量
        # 2. 增强特征的平移不变性
        # 3. 保留最显著的特征，抑制噪声
        # 参数说明：
        # kernel_size: 池化窗口大小
        # stride: 滑动步长，默认为kernel_size
        # padding: 填充大小
        # dilation: 窗口元素之间的间距
        # return_indices: 是否返回最大值的位置索引
        # ceil_mode: 当剩余元素不足时是否使用ceil模式
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化模型
model = TheModelClass()
# 打印模型的参数
print("Model's parameters:")
for param in model.parameters():
    print(param)


# 初始化优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 这里拿到loss和epoch
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 输入数据
        inputs, labels = data
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 优化器更新参数
        optimizer.step()



# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])






"""part2 : 保存和加载模型"""

# 保存
# os.makedirs(save_path, exist_ok=True)  
torch.save(model.state_dict(), save_path)


# 加载
# model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(save_path))
# 在运行推理之前，务必调用model.eval()设置 dropout 和 batch normalization 层为评估模式。
# 如果不这么做，可能导致模型推断结果不一致。
model.eval()

# 保存和加载整个模型
torch.save(model, save_model_path)
# 模型类必须在此之前被定义
model = torch.load(save_model_path)
model.eval()


""" part3: 保存和加载 Checkpoint 用于推理/继续训练"""

""" 
# 保存
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)


# 加载
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()

"""




"""part4: 在一个文件中保存多个模型 """

""" 
# 保存
torch.save({
            'epoch': epoch,
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            'loss': loss,
            }, save_path) 

# 加载
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(save_path)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

"""



"""part5 : 使用不同模型参数下的热启动模式"""

"""

torch.save(modelA.state_dict(), PATH)

modelB = TheModelBClass(*args, **kwargs)
#在迁移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见的情况。 
#都可以通过在load_state_dict()函数中将strict参数设置为 False 来忽略非匹配键的函数。
#如果要将参数从一个层加载到另一个层，但是某些键不匹配，主要修改正在加载的 state_dict 中的参数键的名称以匹配要在加载到模型中的键即可。s
modelB.load_state_dict(torch.load(PATH), strict=False)


"""



""" part6: 通过设备保存/加载模型 """
"""

#保存在CPU上, 加载在CPU上
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))


# 保存在CPU上, 加载在GPU上
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# 确保在你提供给模型的任何输入张量上调用input = input.to(device)


"""