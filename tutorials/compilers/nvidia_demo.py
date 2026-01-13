# NOTE: a modern NVIDIA GPU (H100, A100, or V100) is recommended for this tutorial in
# order to reproduce the speedup numbers shown below and documented elsewhere.

import torch
import warnings
from torchvision.models import densenet121
import numpy as np

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )


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


# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to().cuda(),
        torch.randint(1000, (b,)).cuda(),
    )


N_ITERS = 10


def init_model():
    return densenet121().cuda()


model = init_model()

# Note that we generally recommend directly compiling a torch.nn.Module by calling
# its .compile() method.
model_opt = init_model()
model_opt.compile(mode="reduce-overhead")


def first_demo():
    inp = generate_data(16)[0]
    with torch.no_grad():
        print("eager:", timed(lambda: model(inp))[1])
        print("compile:", timed(lambda: model_opt(inp))[1])


"""
(eval) eager median: 0.01525604772567749, compile median: 0.003931119918823242, speedup: 3.8808400762916184x
eager eval time 0: 0.2951763916015625
eager eval time 1: 0.01678335952758789
eager eval time 2: 0.015734944343566894
eager eval time 3: 0.015243231773376465
eager eval time 4: 0.015268863677978516
eager eval time 5: 0.01522979164123535
eager eval time 6: 0.015177727699279785
eager eval time 7: 0.015617024421691895
eager eval time 8: 0.015202367782592773
eager eval time 9: 0.015126527786254883
~~~~~~~~~~
compile eval time 0: 5.565470703125
compile eval time 1: 0.24912281799316408 第二次还是慢了，尽管比第一次运行快得多。这是因为 "reduce-overhead" 模式会为 CUDA 图运行几次预热迭代。
compile eval time 2: 0.00450867223739624
compile eval time 3: 0.004577280044555664
compile eval time 4: 0.003706687927246094
compile eval time 5: 0.0037672960758209227
compile eval time 6: 0.003935231924057007
compile eval time 7: 0.003768320083618164
compile eval time 8: 0.003927007913589477
compile eval time 9: 0.0038635520935058594
"""


def predict_many_demo():
    eager_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, eager_time = timed(lambda: model(inp))
        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")

    print("~" * 10)

    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, compile_time = timed(lambda: model_opt(inp))
        compile_times.append(compile_time)
        print(f"compile eval time {i}: {compile_time}")
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


opt = torch.optim.Adam(model.parameters())


def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

"""
eager train time 0: 0.6821947631835937
eager train time 1: 0.0516577262878418
eager train time 2: 0.048728256225585936
eager train time 3: 0.047841407775878905
eager train time 4: 0.04823257446289062
eager train time 5: 0.048595008850097654
eager train time 6: 0.057622528076171874
eager train time 7: 0.05626262283325195
eager train time 8: 0.057923583984375
eager train time 9: 0.058123264312744144
~~~~~~~~~~

compile train time 0: 141.419421875
compile train time 1: 8.4247080078125
compile train time 2: 0.018790399551391602
compile train time 3: 0.010836992263793945
compile train time 4: 0.010805248260498047
compile train time 5: 0.010437631607055664
compile train time 6: 0.010218496322631837
compile train time 7: 0.012146688461303711
compile train time 8: 0.012992511749267579
compile train time 9: 0.012563455581665038
~~~~~~~~~~
(train) eager median: 0.05396017456054687, compile median: 0.012355072021484375, speedup: 4.3674512351457695x
"""
def train_many_demo():
    eager_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)
        _, eager_time = timed(lambda: train(model, inp))
        eager_times.append(eager_time)
        print(f"eager train time {i}: {eager_time}")
    print("~" * 10)
    # Note that because we are compiling a regular Python function, we do not
    # call any .compile() method.
    train_opt = torch.compile(train, mode="reduce-overhead")

    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)
        _, compile_time = timed(lambda: train_opt(model, inp))
        compile_times.append(compile_time)
        print(f"compile train time {i}: {compile_time}")
    print("~" * 10)

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert speedup > 1
    print(
        f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
    )
    print("~" * 10)


if __name__ == '__main__':
    train_many_demo()
