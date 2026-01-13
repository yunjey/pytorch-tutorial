import torch

def foo1(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y


# TORCH_COMPILE_DEBUG=1 python xx.py
if __name__ == '__main__':
    x1 = torch.randint(256, (1, 8), dtype=torch.uint8)
    x2 = torch.randint(256, (8390, 8), dtype=torch.uint8)
    compiled_foo1 = torch.compile(foo1)
    result = compiled_foo1(x1, x2)
