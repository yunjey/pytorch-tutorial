import torch
import torch.utils.benchmark as benchmark
from torch.utils.benchmark import Language
from torch.utils.benchmark import Timer
import pickle
import re
from torch.utils.benchmark import CallgrindStats, FunctionCounts


def cudaclu_timer():
    cpp_timer = Timer(
        "x * y;",
        """
            auto x = torch::ones({128});
            auto y = torch::ones({128});
        """,
        language=Language.CPP,
    )

    print(cpp_timer.blocked_autorange(min_run_time=1))


# 使用 Callgrind 进行 A/B 测试
"""
指令计数最有用之处在于它们允许对计算进行精细比较，这在分析性能时至关重要。
为了实际演示这一点，让我们将两个大小为 128 的 Tensor 相乘与一个 {128} x {1} 的乘法进行比较，后者将广播第二个 Tensor。
"""


def call_grind_timer():
    broadcasting_stats = Timer(
        "x * y;",
        """
            auto x = torch::ones({128});
            auto y = torch::ones({1});
        """,
        language=Language.CPP,
    ).collect_callgrind().as_standardized().stats(inclusive=False)
    # Let's round trip `broadcasting_stats` just to show that we can.
    broadcasting_stats = pickle.loads(pickle.dumps(broadcasting_stats))

    cpp_timer = Timer(
        "x * y;",
        """
            auto x = torch::ones({128});
            auto y = torch::ones({128});
        """,
        language=Language.CPP,
    )

    print(cpp_timer.blocked_autorange(min_run_time=1))
    stats: CallgrindStats = cpp_timer.collect_callgrind()
    inclusive_stats = stats.as_standardized().stats(inclusive=False)
    print(inclusive_stats[:10])
    # And now to diff the two tasks:
    delta = broadcasting_stats - inclusive_stats

    def extract_fn_name(fn: str):
        """Trim everything except the function name."""
        fn = ":".join(fn.split(":")[1:])
        return re.sub(r"\(.+\)", "(...)", fn)

    # We use `.transform` to make the diff readable:
    print(delta.transform(extract_fn_name))


if __name__ == '__main__':
    cudaclu_timer()
