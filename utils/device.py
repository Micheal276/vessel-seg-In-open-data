"""设备选择模块：全项目唯一的"用哪块硬件算"的入口。

设计理由
--------
1. 集中封装：训练、推理、测试脚本都只调用 ``get_device()``，不在各处手写
   ``torch.device("mps")``。换机器（Mac -> 有 NVIDIA 的服务器）时不需要改任何业务代码。
2. 优先级 mps > cuda > cpu：本项目主要在 Apple Silicon Mac 上开发，MPS 是 Apple GPU
   的 PyTorch 后端；如果在 Linux 服务器上运行则自动用 CUDA；都没有时退回 CPU，保证代码
   在任何环境下都能跑通（只是慢）。
3. PYTORCH_ENABLE_MPS_FALLBACK=1：MPS 后端还有少量算子没有实现，开启后这些算子会自动
   回退到 CPU 执行，而不是直接报错。该变量必须在 ``import torch`` 之前设置才保证生效，
   所以本模块在文件最顶部、导入 torch 之前用 ``os.environ.setdefault`` 设置它
   （setdefault：如果用户已经在命令行显式设置了，就尊重用户的值）。
4. 启动时打印实际设备：避免"以为在用 GPU，其实在用 CPU"这类静默的性能问题。
5. 统一 float32：MPS 不支持 float64，项目中所有张量都使用 float32（本模块的自检也用 float32）。

注意：MPS 上的计算结果不保证逐位（bit-exact）可复现，即使固定了随机种子，
两次运行的指标也可能在小数点后几位有差异。详见 utils/seed.py 与 README。

用法::

    python -m utils.device     # 打印设备并做一次 conv 前向自检
"""

import os

# 必须在 import torch 之前设置，否则 MPS 回退开关可能不生效
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (故意放在设置环境变量之后)


def get_device(verbose: bool = True) -> torch.device:
    """按 mps > cuda > cpu 的优先级返回可用设备，并（默认）打印选择结果。"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"[device] 使用设备: {device}")
    return device


def _self_check() -> None:
    """自检：在选中的设备上跑一次 float32 的 Conv2d 前向，确认后端真的可用。"""
    from torch import nn

    device = get_device()
    print(f"[device] torch {torch.__version__}, "
          f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")

    # 形状模拟本项目输入：batch=2，单通道（绿通道），64x64 patch
    conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1).to(device)
    x = torch.randn(2, 1, 64, 64, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = conv(x)

    print(f"[device] conv 前向完成: 输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}, "
          f"dtype={y.dtype}, device={y.device}")


if __name__ == "__main__":
    _self_check()
