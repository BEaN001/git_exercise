import torch
import numpy as np


numpy_tensor = np.random.rand(10, 20)
print(numpy_tensor.shape, numpy_tensor)


pytorch_tensor1 = torch.Tensor(numpy_tensor)
pytorch_tensor2 = torch.from_numpy(numpy_tensor)

print(type(pytorch_tensor1), type(pytorch_tensor2))

# 如果 pytorch tensor 在 cpu 上
numpy_array1 = pytorch_tensor1.numpy()
print(type(numpy_array1))
# 如果 pytorch tensor 在 gpu 上
numpy_array2 = pytorch_tensor1.cpu().numpy()
print(type(numpy_array2))


# 我们可以使用以下两种方式将 Tensor 放到 GPU 上
# 第一种方式是定义 cuda 数据类型
dtype = torch.cuda.FloatTensor # 定义默认 GPU 的 数据类型
gpu_tensor = torch.randn(10, 20).type(dtype)

# 第二种方式更简单，推荐使用
gpu_tensor1 = torch.randn(10, 20).cuda(0) # 将 tensor 放到第一个 GPU 上
gpu_tensor2 = torch.randn(10, 20).cuda(1) # 将 tensor 放到第二个 GPU 上



