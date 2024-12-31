""" 
自动微分（Automatic Differentiation，简称 AD）的主要目的是高效、精确地计算函数的导数
1. 计算复杂函数的导数
2. 加速优化过程
3. 支持高阶导数
4. 提高代码的可维护性
tensor 和 gradients 来举一些例子。

torch.Tensor 是包的核心类。如果将其属性
 .requires_grad 设置为 True，则会开始跟踪针对 tensor 的所有操作。
 完成计算后，您可以调用 .backward() 来自动计算所有梯度。
 该张量的梯度将累积到 .grad 属性中。

要停止 tensor 历史记录的跟踪，您可以调用 .detach()，
它将其与计算历史记录分离，并防止将来的计算被跟踪。

要停止跟踪历史记录（和使用内存），您还可以将代码块使用 with torch.no_grad(): 包装起来
因为评估阶段我们不需要梯度。

每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用，
（如果用户自己创建张量，则grad_fn 是 None ）。

如果你想计算导数，你可以调用 Tensor.backward()。
如果 Tensor 是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，
则需要指定一个gradient 参数来指定张量的形状。

"""

import torch



"""sample-1"""
# 创建一个张量并启用梯度计算
# 这里我们创建了一个值为3.0的张量，并启用了梯度计算。
# 这意味着PyTorch会跟踪所有对x的操作，以便在需要时计算梯度。
# 结果为8的原因是：
# 我们定义了一个计算图 y = x^2 + 2x + 1。
# 使用链式法则，我们可以计算y对x的导数：dy/dx = 2x + 2。
# 当x = 3.0时，dy/dx = 2*3 + 2 = 8。
# 因此，当我们调用y.backward()时，PyTorch会计算并存储x的梯度，结果为8。
x = torch.tensor(3.0, requires_grad=True)

# 定义计算图
y = x ** 2 + 2 * x + 1

# 计算梯度
y.backward()

# 查看梯度
print(x.grad)  # 输出 8.0


""" sample-2 """
# 解释一下这部分的原理，最终是如何计算出来的呢
# 在这个例子中，我们创建了一个2x2的张量x，并启用了梯度计算。
# 然后我们对x进行了一系列操作：
# 1. y = x + 2
# 2. z = y * y * 3
# 3. out = z.mean()
# 最后，我们调用out.backward()来计算梯度。
# 
# 计算过程如下：
# 1. y = x + 2，所以y的每个元素都是x的对应元素加2。
# 2. z = y * y * 3，所以z的每个元素都是y的对应元素的平方乘以3。
# 3. out = z.mean()，所以out是z的所有元素的平均值。

# 当我们调用out.backward()时，PyTorch会计算out对x的梯度。
# 由于out是z的平均值，所以out对z的梯度是1/4（因为z有4个元素）。
# 然后，out对y的梯度是2 * y * 3（因为z = y * y * 3）。
# 最后，out对x的梯度是2 * y * 3 * 1（因为y = x + 2）。
# 
# 因此，x.grad的值是2 * (x + 2) * 3 * 1/4。
# 由于x是全1的张量，所以x + 2 = 3，2 * 3 * 3 = 18，18 * 1/4 = 4.5。
# 所以，x.grad的值是4.5，对应于输出中的tensor([[4.5000, 4.5000], [4.5000, 4.5000]])。


x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)


z = y * y * 3
out = z.mean()

print(z, out)

out.backward()


print(x.grad)


""" sample-3 , 雅可比向量机积 """

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
#默认是False，调一次是True
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

