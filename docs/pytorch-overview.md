# PyTorch 核心概念与组成

## 一、PyTorch 是什么

PyTorch 是一个开源的深度学习框架，由 Facebook (Meta) 开发，以 **动态计算图** 为核心特性，广泛用于学术研究和工业部署。

---

## 二、核心组成

### 1. `torch` — 张量计算核心

**定义**：张量 (Tensor) 是 PyTorch 的基本数据结构，类似于 NumPy 的 ndarray，但支持 GPU 加速。

```python
import torch

# 创建张量
x = torch.tensor([1.0, 2.0, 3.0])          # 从数据创建
y = torch.zeros(3, 4)                       # 全零张量 (3行4列)
z = torch.randn(2, 3)                       # 标准正态分布随机张量
w = torch.arange(0, 10, step=2)             # 等差序列 [0, 2, 4, 6, 8]

# 张量属性
print(x.shape)    # torch.Size([3])
print(x.dtype)    # torch.float32
print(x.device)   # cpu

# GPU 加速（如可用）
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(x_gpu.device)  # cuda:0
```

---

### 2. `torch.autograd` — 自动微分引擎

**定义**：自动计算梯度，是神经网络反向传播的基础。

```python
# requires_grad=True 开启梯度追踪
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 前向传播
y = x ** 2 + 2 * x + 1

# 反向传播
y.sum().backward()

# 查看梯度
print(x.grad)  # tensor([6., 8.])  即 dy/dx = 2x + 2
```

---

### 3. `torch.nn` — 神经网络模块

**定义**：预定义的层和容器，简化网络构建。

```python
import torch.nn as nn

# 方式一：Sequential 顺序容器
model = nn.Sequential(
    nn.Linear(10, 64),    # 全连接层：输入10 → 输出64
    nn.ReLU(),            # 激活函数
    nn.Linear(64, 1)      # 输出层
)

# 方式二：自定义 Module
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
print(model)
```

---

### 4. `torch.optim` — 优化算法

**定义**：内置 SGD、Adam、RMSProp 等优化器。

```python
import torch.optim as optim

# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()     # 梯度清零
    loss = compute_loss()
    loss.backward()           # 反向传播
    optimizer.step()          # 更新参数
```

---

### 5. `torch.utils.data` — 数据加载

**定义**：DataLoader 和 Dataset 工具，高效批量加载数据。

```python
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(torch.randn(1000, 10))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch.shape)  # torch.Size([32, 10])
```

---

### 6. `torch.jit` — 模型部署

**定义**：TorchScript 将 PyTorch 模型导出为可部署格式（无需 Python 环境）。

```python
# 脚本化
scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')

# 加载
loaded_model = torch.jit.load('model.pt')
```

---

## 三、关键概念

| 概念 | 说明 |
|------|------|
| **计算图** | 记录操作序列的有向图，用于自动求导。PyTorch 使用动态图（运行时构建） |
| **前向传播** | 数据从输入层流经网络到输出层的过程 |
| **反向传播** | 基于损失函数，通过链式法则计算梯度，更新网络参数 |
| **GPU 加速** | `.to('cuda')` 将张量/模型迁移到 GPU 加速计算 |
| **模型状态dict** | `model.state_dict()` 保存模型参数，用于checkpoint |
| **.eval() / train()** | 切换模型为评估/训练模式（影响 dropout、batchnorm 行为） |

---

## 四、完整训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 数据准备
X = torch.randn(1000, 20)
y = (X.sum(dim=1) > 0).float().unsqueeze(1)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. 模型定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = Net()

# 3. 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练循环
model.train()
for epoch in range(10):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 5. 评估
model.eval()
with torch.no_grad():
    accuracy = (model(X).round() == y).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
```

---

## 五、补充模块速览

| 模块 | 用途 |
|------|------|
| `torch.nn.functional` | 激活函数、损失函数等函数式 API |
| `torch.nn.init` | 参数初始化方法 |
| `torch.save / torch.load` | 模型持久化 |
| `torch.distributed` | 多 GPU / 多机分布式训练 |
| `torch.cuda` | CUDA 内存管理 |
