# Autograd 自动求导
import torch

# 创建一个张量并设置requires_grad=True用来追踪其计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 对这个张量做一次运算：
y = x + 2
print(y)
# y是计算的结果，所以它有grad_fn属性。
print(y.grad_fn)

# 对y进行更多操作
z = y * y * 3
out = z.mean()

print(z, out)

#  原地改变了现有张量的 requires_grad 标志。如果没有指定的话，默认输入的这个标志是 False。
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


# 现在开始进行反向传播，因为 out 是一个标量，因此 out.backward() 和 out.backward(torch.tensor(1.)) 等价。
out.backward()

# 输出导数 d(out)/dx
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
print(y)
while y.data.norm() < 1000:
    y = y * 2

print(y)
# 在这种情况下，y 不再是标量。torch.autograd 不能直接计算完整的雅可比矩阵，但是如果我们只想要雅可比向量积，只需将这个向量作为参数传给 backward：
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# 也可以通过将代码块包装在 with torch.no_grad(): 中，来阻止autograd跟踪设置了 .requires_grad=True 的张量的历史记录。
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)








