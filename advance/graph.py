import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
"""
激活函数在神经网络中至关重要，因为它们引入非线性，使网络能够捕捉复杂的关系。
没有激活函数，网络仅能进行线性变换，无法处理非线性问题。
非线性使得神经网络能够模拟各种复杂函数，适用于多种任务，如分类和回归。
此外，激活函数如ReLU有助于解决梯度消失问题，提高训练效率，而softmax则在多类分类中提供概率输出。
总之，激活函数增强了模型的表达能力和训练稳定性。

以下是几种常见激活函数的图示描述：

Sigmoid函数：

形状：S形曲线。

范围：输出在0到1之间。

特点：将任何实数映射到0和1之间，适合二分类问题。



ReLU函数：

形状：对于正输入是直线，负输入为零。

范围：输出在0到正无穷之间。

特点：计算简单，帮助缓解梯度消失问题，但存在“死神经元”现象。



Tanh函数：

形状：S形曲线，类似于Sigmoid但中心在零点。

范围：输出在-1到1之间。

特点：零中心化，适合某些优化算法，但同样面临梯度消失问题。



Softmax函数：

形状：对于向量输入，输出概率分布。

范围：输出值在0到1之间，且总和为1。

特点：常用于多分类问题的输出层。



Leaky ReLU函数：

形状：正输入为直线，负输入为小斜率（如0.01x）。

范围：输出在负无穷到正无穷之间。

特点：缓解ReLU的“死神经元”问题。




ELU函数：

形状：正输入为直线，负输入为指数曲线。

范围：输出在负值和正无穷之间。

特点：加速学习，缓解梯度消失问题。




Swish函数：

形状：平滑，非单调，自适应激活。

范围：输出在负无穷到正无穷之间。

特点：复杂模式学习能力强，自 gating机制。



"""



# 定义一组输入值
x = np.linspace(-10, 10, 100)  # 从 -10 到 10 的 100 个点

# 计算 Softmax
x_tensor = torch.tensor(x, dtype=torch.float32)
softmax_values = F.softmax(x_tensor, dim=-1).numpy()

# 绘制 Softmax 函数
plt.figure(figsize=(8, 6))
plt.plot(x, softmax_values, label="Softmax", color="blue", linewidth=2)
plt.title("Softmax Function", fontsize=16)
plt.xlabel("Input Values (x)", fontsize=14)
plt.ylabel("Softmax Output", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

