import torch


@torch.compile
def my_big_model(x):
    return torch.relu(x)


# fail_on_recompile 防止重新编译
def fail_on_recompile():
    # first compilation
    my_big_model(torch.randn(3))

    with torch.compiler.set_stance("fail_on_recompile"):
        my_big_model(torch.randn(3))  # no recompilation - OK
        try:
            # 这里 shape 改变了，会触发 recompilation
            my_big_model(torch.randn(4))  # recompilation - error
        except Exception as e:
            print(e)


@torch.compile
def my_huge_model(x):
    if torch.compiler.is_compiling():
        return x + 1
    else:
        return x - 1


"""
报错过于 disruptive，我们可以改用 "eager_on_recompile"，它将导致 torch.compile 回退到立即执行模式而不是报错。
如果预计重新编译不会频繁发生，但一旦需要，我们宁愿承担立即执行的成本而不是重新编译的成本，那么这可能很有用。
"""


def eager_on_recompile():
    # first compilation
    print(my_huge_model(torch.zeros(3)))  # 1
    with torch.compiler.set_stance("eager_on_recompile"):
        print(my_huge_model(torch.zeros(3)))  # 1
        print(my_huge_model(torch.zeros(4)))  # -1
        print(my_huge_model(torch.zeros(3)))  # 1


# 衡量性能提升
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


@torch.compile
def my_gigantic_model(x, y):
    x = x @ y
    x = x @ y
    x = x @ y
    return x


"""
eager: 0.0004822399914264679
compiled: 0.00010444799810647964
"""


def force_eager_demo():
    inps = torch.randn(5, 5), torch.randn(5, 5)
    with torch.compiler.set_stance("force_eager"):
        print("eager:", timed(lambda: my_gigantic_model(*inps))[1])
    # warmups
    for _ in range(3):
        my_gigantic_model(*inps)
    print("compiled:", timed(lambda: my_gigantic_model(*inps))[1])


@torch.compile
def my_humongous_model(x):
    return torch.sin(x, x)

def fast_find_error():
    try:
        # sin() takes 1 positional argument but 2 were given
        with torch.compiler.set_stance("force_eager"):
            print(my_humongous_model(torch.randn(3)))
        # this call to the compiled model won't run
        print(my_humongous_model(torch.randn(3)))
    except Exception as e:
        print(e)

if __name__ == '__main__':
    fast_find_error()

