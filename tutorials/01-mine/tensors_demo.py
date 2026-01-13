import torch
import numpy as np


if __name__ == '__main__':
    # Initializing a Tensor
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(f"x_np.numpy() == np_array: {x_np.numpy() == np_array}")
    # The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")
    x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    # Operations on Tensors
    # 超过1200种张量运算，包括算术运算、线性代数运算、矩阵操作（转置、索引、切片）、采样等
    # 默认情况下，张量在CPU上创建。我们需要使用.to方法（在检查加速器可用性之后）将张量显式移动到加速器。请记住，跨设备复制大型张量在时间和内存方面可能成本很高！
    # We move our tensor to the current accelerator if available
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    # tensor = tensor.to(device)
    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:, 1] = 0
    print(tensor)
    # Joining tensors  拼接张量
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(f"t1: \n {t1} \n")
    # Arithmetic operations 算术运算
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # ``tensor.T`` returns the transpose of a tensor
    print(f"tensor: \n {tensor} \n")
    print(f"tensor.T: \n {tensor.T} \n")
    y1 = tensor @ tensor.T #矩阵乘法
    y2 = tensor.matmul(tensor.T) #矩阵乘法

    y3 = torch.rand_like(y1)
    print(f"y1: \n {y1} \n, \ny2: \n {y2} \n, \ny3: \n {y3}")
    torch.matmul(tensor, tensor.T, out=y3)

    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    print(f"agg tensor: \n {tensor} \n")
    # tensor.sum() 会对张量的所有元素进行求和，返回一个标量（只有一个值的张量）。
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))
