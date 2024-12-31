import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# 这里是一个三目运算吗
# result = "Yes" if condition else "No"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
接着这些方法会递归地遍历所有模块，并将它们的参数和缓冲器转换为CUDA张量。

model.to(device)

记住你也必须在每一个步骤向GPU发送输入和目标：

inputs, labels = inputs.to(device), labels.to(device)

通过使用 DataParallel 让你的模型并行运行，你可以很容易的在多 GPU 上运行你的操作。
model = nn.DataParallel(model)

"""

input_size = 5
output_size = 2

batch_size = 30
data_size = 100


"""
生成一些玩具数据
"""
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# shuffle=True 表示在每个 epoch 开始时，数据加载器会对数据进行随机打乱。
# 这样可以确保每个 epoch 中的数据顺序不同，有助于模型更好地学习数据的分布。
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),batch_size=batch_size, shuffle=True)

"""搞个简单的模型"""
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


"""
这是整个教程的核心。
首先我们需要一个模型的实例，然后验证我们是否有多个 GPU。
如果我们有多个 GPU，
我们可以用 nn.DataParallel来包裹我们的模型。
然后我们使用 model.to(device) 把模型放到多 GPU 中。
"""

model = Model(input_size, output_size)



device = torch.device("mps" if torch.mps.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)


if torch.mps.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

"""运行模型： 现在我们可以看到输入和输出张量的大小了。"""
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())

"""
据并行自动拆分了你的数据并且将任务单发送到多个 GPU 上。
当每一个模型都完成自己的任务之后，DataParallel 收集并且合并这些结果，然后再返回给你。
"""



