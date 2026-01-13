import torch


def stride_demo():
    """
    连续和非连续布局
    1.按照张量的逻辑形状（比如 2x3、3x2），以C 风格（行优先） 遍历每一个元素时，访问的内存地址是否是连续递增的。
    为什么内存布局重要？
    性能影响：连续布局的张量访问内存时，CPU/GPU 的缓存命中率更高，运算速度更快；非连续张量可能因为内存跳跃访问导致性能下降。
    操作限制：部分 PyTorch 操作（如view、resize_）仅支持连续张量，非连续张量会抛出RuntimeError。
    内存效率：像转置这样的操作通过修改布局而非复制数据，能节省大量内存。
    """
    # 从0到12 分成三行四列
    x = torch.arange(12).view(3, 4)
    print(f"view: {x}")
    print(f"shape: {x.shape}")
    # 在指定维度上从一个元素调到下一个元素所需的距离 dim 是两个数字，为了访问下一行需要往前移动4步，下一列应该向前移动1步
    print(f"stride: {x.stride()}")
    print(f"is_contiguous: {x.is_contiguous()}")
    print("\n***********************************\n")
    y = x.t()

    # 这里没有真正的转置，只是改变了视图的 stride “如何从内存中读取数据”
    print(f"v_t is_contiguous: {y.is_contiguous()}")
    print(f"v_t: {y}")
    print(f"v_t shape: {y.shape}")
    print(f"v_t stride: {y.stride()}")
    print("\n***********************************\n")
    z = y.contiguous()
    print(f"v_t is_contiguous: {z.is_contiguous()}")
    print(f"v_t: {z}")
    print(f"v_t shape: {z.shape}")
    print(f"v_t stride: {z.stride()}")


def storage_demo():
    print(f"PyTorch版本: {torch.__version__}")
    print(
        f"操作系统: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32)
    storage_x = x.storage()
    # 1. storage永远是一维的，不管Tensor是几维
    print(f"storage:\n{storage_x}\n")
    print("storage的类型：", type(storage_x))  # torch.storage._TypedStorage
    print("storage的长度：", len(storage_x))  # 6（元素总数）
    print("storage_x的id：", id(storage_x))

    # 2. 转置后的Tensor共享同一个storage（物理内存没复制）
    y = x.t()
    storage_y = y.storage()
    print("x的数据指针：", x.data_ptr())
    print("y的数据指针：", y.data_ptr())
    print("x和y的数据指针是否相同：", x.data_ptr() == y.data_ptr())

    print("storage_y的id：", id(storage_y))


def shared_storage_demo():
    """
    验证多个tensor共享storage时，修改一个会影响其他所有tensor
    """
    print("=" * 60)
    print("验证多个tensor共享storage的行为")
    print("=" * 60)

    # 创建原始tensor
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)

    # 创建多个共享storage的tensor
    y = x.t()  # 转置
    z = x.view(-1)  # 展平成一维
    w = x[1:, 2:]  # 切片
    v = x.reshape(2, 6)  # reshape（如果可能，会共享storage）

    print("\n【初始状态】")
    print(f"x (原始) =\n{x}")
    print(f"y (转置) =\n{y}")
    print(f"z (展平) = {z}")
    print(f"w (切片) =\n{w}")
    print(f"v (reshape) =\n{v}")

    # 验证它们是否共享存储（通过data_ptr比较）
    print("\n【内存地址验证】")
    print(f"x.data_ptr() = {x.data_ptr()}")
    print(f"y.data_ptr() = {y.data_ptr()}")
    print(f"z.data_ptr() = {z.data_ptr()}")
    print(f"w.data_ptr() = {w.data_ptr()} (切片会有偏移)")
    print(f"v.data_ptr() = {v.data_ptr()}")
    print(f"\nx和y是否共享底层内存: {x.data_ptr() == y.data_ptr()}")
    print(f"x和z是否共享底层内存: {x.data_ptr() == z.data_ptr()}")
    print(f"x和v是否共享底层内存: {x.data_ptr() == v.data_ptr()}")

    # 修改x的一个元素
    print("\n【修改x[0, 0] = 999】")
    x[0, 0] = 999

    print(f"x =\n{x}")
    print(f"y =\n{y}")  # y[0, 0]应该也变成999
    print(f"z = {z}")  # z[0]应该也变成999
    print(f"w =\n{w}")  # w不受影响，因为它是切片[1:, 2:]
    print(f"v =\n{v}")  # v[0, 0]应该也变成999

    # 修改y的一个元素
    print("\n【修改y[1, 1] = 888】")
    y[1, 1] = 888

    print(f"x =\n{x}")  # x[1, 1]应该也变成888
    print(f"y =\n{y}")
    print(f"z = {z}")  # z[5]应该也变成888
    print(f"v =\n{v}")  # v相应位置也会变化

    # 修改z的一个元素
    print("\n【修改z[10] = 777】")
    z[10] = 777

    print(f"x =\n{x}")  # x[2, 2]应该也变成777
    print(f"y =\n{y}")
    print(f"z = {z}")
    print(f"w =\n{w}")  # w[1, 0]应该也变成777（因为w是x[1:, 2:]）

    # 对比：使用clone()创建真正的副本
    print("\n【对比：使用clone()创建独立副本】")
    x_copy = x.clone()
    print(f"x_copy.data_ptr() = {x_copy.data_ptr()}")
    print(f"x_copy与x是否共享内存: {x_copy.data_ptr() == x.data_ptr()}")

    x[0, 1] = 666
    print("\n修改x[0, 1] = 666后：")
    print(f"x =\n{x}")
    print(f"x_copy =\n{x_copy}  (不受影响)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # shared_storage_demo()
    x = torch.randn(2, 3, 4)
    print(f"x: {x}")
    print(f"x shape: {x.shape}")
    print(f"x stride: {x.stride()}")
    # import json
    # print(json.dumps(x.tolist(), indent=4))
