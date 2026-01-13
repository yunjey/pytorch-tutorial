import torch

# def foo(x, y):
#     a = torch.sin(x)
#     b = torch.cos(y)
#     return a + b
#
#
# opt_foo1 = torch.compile(foo)
# print(opt_foo1(torch.randn(3, 3), torch.randn(3, 3)))
#
#
# @torch.compile
# def opt_foo2(x, y):
#     a = torch.sin(x)
#     b = torch.cos(y)
#     return a + b
#
#
# print(opt_foo2(torch.randn(3, 3), torch.randn(3, 3)))
#
# def inner(x):
#     return torch.sin(x)
#
#
# @torch.compile
# def outer(x, y):
#     a = inner(x)
#     b = torch.cos(y)
#     return a + b
#
#
# print(outer(torch.randn(3, 3), torch.randn(3, 3)))
#

# t = torch.randn(10, 100)
#
#
# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin = torch.nn.Linear(3, 3)
#
#     def forward(self, x):
#         return torch.nn.functional.relu(self.lin(x))
#
#
# mod1 = MyModule()
# mod1.compile()
# print(mod1(torch.randn(3, 3)))
#
# mod2 = MyModule()
# mod2 = torch.compile(mod2)
# print(mod2(torch.randn(3, 3)))


# Demonstrating Speedups 展示加速效果

def foo3(x):
    y = x + 1
    z = torch.nn.functional.relu(y)
    u = z * 2
    return u


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


opt_foo3 = torch.compile(foo3)
inp = torch.randn(4096, 4096).cuda()


def first_run():
    torch._logging.set_logs(graph_code=True)
    """
    请注意，torch.compile 似乎比即时执行要花费长得多的时间才能完成。这是因为 torch.compile 在最初几次执行时需要额外时间来编译模型。
    torch.compile 会尽可能重用已编译的代码，因此如果我们多运行几次优化后的模型，应该会看到与即时执行相比有显著的性能提升。
    """
    print("compile:", timed(lambda: opt_foo3(inp))[1])
    print("eager:", timed(lambda: foo3(inp))[1])


"""
eager time 0: 0.027955583572387695
eager time 1: 0.0004986880123615265
eager time 2: 0.00045683199167251585
eager time 3: 0.00045158401131629945
eager time 4: 0.00045363199710845946
eager time 5: 0.00045363199710845946
eager time 6: 0.0004556800127029419
eager time 7: 0.0004505600035190582
eager time 8: 0.00045043200254440307
eager time 9: 0.0004546560049057007
~~~~~~~~~~
compile time 0: 0.434231201171875
compile time 1: 0.00026624000072479246
compile time 2: 0.00023552000522613525
compile time 3: 0.0002234240025281906
compile time 4: 0.00021913599967956544
compile time 5: 0.00022220799326896668
compile time 6: 0.0002181120067834854
compile time 7: 0.0002242559939622879
compile time 8: 0.0002181120067834854
compile time 9: 0.00022118400037288665
~~~~~~~~~~
(eval) eager median: 0.0004541440010070801, compile median: 0.00022281599789857864, speedup: 2.038201948200314x
"""


def many_runs():
    # turn off logging for now to prevent spam
    torch._logging.set_logs(graph_code=False)
    eager_times = []
    for i in range(10):
        _, eager_time = timed(lambda: foo3(inp))
        eager_times.append(eager_time)
        print(f"eager time {i}: {eager_time}")
    print("~" * 10)

    compile_times = []
    for i in range(10):
        _, compile_time = timed(lambda: opt_foo3(inp))
        compile_times.append(compile_time)
        print(f"compile time {i}: {compile_time}")
    print("~" * 10)

    import numpy as np

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert speedup > 1
    print(
        f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
    )
    print("~" * 10)


def bar1(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def bar(a, b):
    x = a / (torch.abs(a) + 1)
    b = torch.where(b.sum() < 0, -b, b)
    return x * b


# Graph Breaks 图中断
"""“图中断”这一术语源于torch.compile尝试捕获并优化PyTorch操作图这一事实。当遇到不支持的Python代码时，这个图就必须被“中断”。
图中断会导致优化机会的损失，这可能仍然不尽如人意，但总比出现无声的错误或硬崩溃要好。"""



@torch.compile(fullgraph=True)
def bar_fixed(a, b):
    x = a / (torch.abs(a) + 1)

    def true_branch(y):
        return y * -1

    def false_branch(y):
        # NOTE: torch.cond doesn't allow aliased outputs
        return y.clone()

    x = torch.cond(b.sum() < 0, true_branch, false_branch, (b,))
    return x * b


def graph_breaks_fixed_demo():
    torch._logging.set_logs(graph_code=True)
    inp1 = torch.ones(10)
    inp2 = torch.ones(10)
    fixed = bar_fixed(inp1, inp2)
    fixed1 = bar_fixed(inp1, -inp2)
    print(f"fixed: {fixed}")
    print(f"fixed1: {fixed1}")


def graph_breaks_demo():
    torch._logging.set_logs(graph_code=True)
    opt_bar = torch.compile(bar)
    inp1 = torch.ones(10)
    inp2 = torch.ones(10)
    opt_bar(inp1, inp2)
    opt_bar(inp1, -inp2)


if __name__ == '__main__':
    graph_breaks_fixed_demo()
