import torch
import torch.nn as nn
import torch.optim as optim
""" 
前向传播：生成预测结果。
反向传播：计算梯度，更新模型参数，优化模型性能。
"""

# 定义神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(1, 1)  # 全连接层

    def forward(self, x):
        return self.fc(x)

# 创建模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 输入数据
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
# 想要的输出
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 训练模型
for epoch in range(100):
    # 前向传播，得到的是预测值
    outputs = model(x)
    # 计算损失，实际值和预测值之间的差距
    loss = criterion(outputs, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    # 打印损失
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# 测试模型
test_x = torch.tensor([[5.0]])
predicted = model(test_x)
print(f"Predicted: {predicted.item()}")