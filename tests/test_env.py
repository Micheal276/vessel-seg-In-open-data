"""环境冒烟测试：确认设备选择与随机种子模块可用。

设计理由：环境问题（装错 torch、MPS 不可用）应在写模型之前就暴露出来，
用 pytest 固化下来，以后每次改环境都能一键复查。
"""

import torch

from utils.device import get_device
from utils.seed import set_seed


def test_get_device_returns_usable_device():
    device = get_device(verbose=False)
    x = torch.ones(2, 2, dtype=torch.float32, device=device)
    assert (x * 2).sum().item() == 8.0


def test_set_seed_repeatable_on_cpu():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)
