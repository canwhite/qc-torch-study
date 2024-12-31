"""
一些基础知识
简单来说，张量是多维数组的泛化，可以看作是标量、向量和矩阵的扩展
张量：
0维标量，一维向量，二维矩阵
3维 = 时间序列
4维 = 图像
5维 = 视频
"""

import torch


"""创建四行三列的矩阵"""
x = torch.rand(4, 3) 
print(x)

"""全0矩阵"""
x = torch.zeros(4, 3, dtype=torch.long)
print(x)


"""用数据构建张量"""
x = torch.tensor([5.5, 3])
print(x)

"""从个张量创建一个张量"""
x = torch.randn_like(x, dtype=torch.float)    
# override dtype!
print(x)     

"""获取它的维度信息"""
print(x.size())


"""加法"""
# 解释一下这里是怎么加的，是行和行的和吗
# 在PyTorch中，张量的加法是按元素进行的。也就是说，两个张量的对应元素相加。
# 例如，如果x是形状为(4, 3)的张量，y是形状为(1, 2)的张量，那么x + y会先将y广播到与x相同的形状，然后再进行元素相加。
# 广播机制会自动调整y的形状，使其与x兼容，然后再进行加法操作。
# 因此，加法的结果是x和y的对应元素相加的结果。

y = torch.rand(1, 2)
print(x + y)
print(torch.add(x, y))


"""提供一个输出tensor作为参数"""
result = torch.empty(1, 2)
torch.add(x, y, out=result)
print(result)

"""加法: in-place"""
y.add_(x)
print(y)


"""选取列"""
x = torch.rand(4, 3) 
print(x)
# 这一句 `print(x[:, 1])` 的意思是打印张量 `x` 的第二列（索引从0开始）。
# 在PyTorch中，`x[:, 1]` 表示选择张量 `x` 的所有行，但只选择第二列。
print(x[:, 1])


"""改变大小"""

# randn 和 rand 的区别是：
# randn 生成的是服从标准正态分布（均值为0，标准差为1）的随机数。
# rand 生成的是在 [0, 1) 区间内均匀分布的随机数。

x = torch.randn(4, 4)
# 这一句 `y = x.view(16)` 的意思是将张量 `x` 的形状从 (4, 4) 改变为 (16,)。
# `view` 方法用于改变张量的形状，但不会改变张量的数据。

y = x.view(16)
# 下一句 `z = x.view(-1, 8)` 的意思是将张量 `x` 的形状从 (4, 4) 改变为 (2, 8)。
# `-1` 表示该维度的大小由其他维度自动推断出来。
# 自动推导的维度是通过其他维度的大小来推断的。
# 在这个例子中，`x` 的形状是 (4, 4)，我们希望将其形状改变为 (?, 8)，其中 `?` 是需要推断的维度。
# 由于 `x` 的总元素数是 16，而我们将第二个维度固定为 8，因此第一个维度的大小应该是 16 / 8 = 2。
# 所以，`x.view(-1, 8)` 的结果是形状为 (2, 8) 的张量。
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

""" tensor ，使用 .item() 来获得这个 value 。"""
x = torch.randn(1)
print(x)
print(x.item())








