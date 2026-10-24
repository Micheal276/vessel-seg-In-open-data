"""随机种子模块：让实验尽可能可复现。

设计理由
--------
1. 深度学习流程里有三个互相独立的随机数来源，必须同时固定：
   - Python 标准库 ``random``（例如打乱文件列表、选择验证集）
   - ``numpy``（例如随机采样 patch 的坐标、数据增强参数）
   - ``torch``（例如网络权重初始化、Dropout、torch 内部的随机操作）
   只固定其中一个，其余来源仍会让每次运行结果不同。
2. torch 在不同后端各有自己的随机数生成器：CPU 由 ``torch.manual_seed`` 覆盖；
   CUDA 需要 ``torch.cuda.manual_seed_all``；MPS 需要 ``torch.mps.manual_seed``。
   这里按"可用才设置"的方式处理，保证同一份代码在 Mac / 服务器 / 纯 CPU 上都能运行。
3. 关于可复现性的边界（面试常被问到）：
   - 在 CPU 上，固定种子后结果通常可以逐位复现。
   - 在 MPS（Apple GPU）上，部分算子的并行归约顺序不固定，浮点加法不满足结合律，
     因此**即使种子相同，结果也不保证逐位一致**，只能保证统计意义上接近。
     PyTorch 目前也没有为 MPS 提供类似 cudnn.deterministic 的确定性开关。
   - 在 CUDA 上可以额外开启 cudnn.deterministic 以换取确定性（代价是速度变慢），
     本函数在 CUDA 可用时顺带设置。
   所以本项目报告指标时以"同一设置下的结果"为准，并在 README 注明 MPS 不可逐位复现。
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """同时固定 random / numpy / torch（以及可用时的 cuda / mps）的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU 生成器（新版 torch 也会顺带设置已初始化的加速器）

    # 影响 Python 哈希随机化；只对之后启动的子进程（如 DataLoader worker）生效
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
