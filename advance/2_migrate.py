"""
实际中，基本没有人会从零开始（随机初始化）训练一个完整的卷积网络，
因为相对于网络，很难得到一个足够大的数据集[网络很深, 需要足够大数据集]。
通常的做法是在一个很大的数据集上进行预训练得到卷积网络ConvNet, 
然后将这个ConvNet的参数作为目标任务的初始化参数或者固定这些参数。
"""
# 这一行代码的作用是启用matplotlib的交互模式（interactive mode）
# 在交互模式下，plt.plot()等绘图函数会立即显示图像，而不需要调用plt.show()
# 这对于在Jupyter Notebook等交互式环境中实时查看绘图结果非常有用

# from __future__ import print_function, division

"""part 1: 导入相关的包"""

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


plt.ion()   # interactive mode

"""part2: 数据预处理"""

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#


"""
# 这里我们使用了PyTorch中预训练的ResNet-18模型
# ResNet-18是一个在ImageNet数据集上预训练好的卷积神经网络
# 它包含18层深度，适合用于迁移学习任务
# 我们通过models.resnet18(pretrained=True)加载预训练模型
# pretrained=True表示加载在ImageNet上预训练好的权重
# 这样我们就可以利用在大规模数据集上学到的特征，而不需要从头训练
# 对于我们的二分类任务（蚂蚁和蜜蜂），我们需要修改最后的全连接层
# 将输出类别从1000（ImageNet的类别数）改为2
"""

model_ft = models.resnet18(pretrained=True)


"""
# 这里我们获取了ResNet-18模型最后一个全连接层的输入特征数
# 这个值表示全连接层输入向量的维度
# 对于ResNet-18来说，这个值通常是512
# 我们需要知道这个值，因为我们要修改最后的全连接层
# 新的全连接层需要保持相同的输入维度，但输出维度要改为2（因为我们只有两个类别：蚂蚁和蜜蜂）
# 这样我们就可以将预训练模型适配到我们的二分类任务上
"""
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs

"""
# 这里我们使用学习率调度器StepLR来动态调整学习率
# StepLR会在每经过step_size个epoch后，将学习率乘以gamma
# 这里设置为每7个epoch将学习率乘以0.1，即学习率衰减为原来的1/10
# 这种学习率衰减策略有助于模型在训练后期更精细地调整参数
# 初始学习率较大可以快速收敛，后期学习率较小可以更精确地找到最优解
# 这种策略在深度学习训练中非常常见，可以有效提高模型性能
"""
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

visualize_model(model_ft)


######################################################################
# ConvNet as fixed feature extractor
# ----------------------------------
#
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad == False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
#



"""
在这里需要冻结除最后一层之外的所有网络。
通过设置requires_grad == Falsebackward()来冻结参数，
这样在反向传播backward()的时候他们的梯度就不会被计算。

这段代码展示了如何使用预训练的ResNet18模型作为固定特征提取器进行迁移学习。

1. 首先加载预训练的ResNet18模型：
   - model_conv = torchvision.models.resnet18(pretrained=True)
   - 这会加载在ImageNet上预训练好的ResNet18模型及其权重

2. 冻结所有层的参数：
   - for param in model_conv.parameters():
       param.requires_grad = False
   - 通过设置requires_grad=False，冻结了除最后一层外的所有层的参数
   - 这意味着在训练过程中，这些层的权重不会被更新

3. 替换最后一层全连接层：
   - num_ftrs = model_conv.fc.in_features
   - model_conv.fc = nn.Linear(num_ftrs, 2)
   - 获取原模型最后一层的输入特征数
   - 用一个新的全连接层替换原模型的最后一层，输出维度改为2（对应蚂蚁和蜜蜂两个类别）

4. 将模型移动到指定设备（GPU或CPU）：
   - model_conv = model_conv.to(device)

5. 定义损失函数：
   - criterion = nn.CrossEntropyLoss()
   - 使用交叉熵损失函数，适用于分类任务

6. 仅优化最后一层的参数：
   - optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
   - 优化器只对最后一层的参数进行优化，因为其他层的参数已被冻结

7. 设置学习率调度器：
   - exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
   - 每7个epoch将学习率乘以0.1，实现学习率衰减

这种固定特征提取器的方法适用于：
- 数据集较小
- 只需要微调最后一层
- 计算资源有限的情况
"""




model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

######################################################################
#

visualize_model(model_conv)

plt.ioff()
plt.show()








