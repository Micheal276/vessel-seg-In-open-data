"""Early stopping：监控验证集 Dice，连续 patience 个 epoch 没有足够提升就停止，并保留最佳权重。

设计理由（面试讲解要点）
------------------------
1. 监控什么：验证集（val split，4 张整图滑窗推理）上的逐图平均 Dice，mode="max"。
   不监控训练 loss——训练 loss 几乎总在下降，看不出过拟合；也不监控验证 loss——
   我们最终关心的是 Dice，而 BCE+Dice 的 loss 与 0.5 阈值下的 Dice 并不完全同步。
   测试集绝不参与，否则就是"在测试集上调参"。

2. min_delta：提升必须超过 min_delta 才算"进步"，避免指标在噪声范围内的小波动不断重置计数器。

3. patience：允许连续 patience 个 epoch 没进步。验证指标会抖动，太小会过早停止；
   太大浪费时间。停止时返回的不是最后一个 epoch，而是验证 Dice 最高那个 epoch 的权重。

4. 最佳权重拷贝到 CPU 内存（并可选同时写盘）。
   state_dict 里的张量和模型共享存储，必须 clone，否则之后训练会把"最佳权重"一起改掉；
   放到 CPU 不占 GPU（MPS）显存，也方便直接 torch.save。

5. 与模型、数据集都无关：只接收一个标量指标和一个 nn.Module。
"""

import math
from pathlib import Path

import torch


class EarlyStopping:
    """参数
    ----
    patience  : 连续多少个 epoch 无提升后停止
    min_delta : 认为"有提升"所需的最小变化量
    mode      : "max"（指标越大越好，如 Dice）或 "min"（越小越好，如 loss）
    ckpt_path : 若给出，每次刷新最佳时把权重写到该路径
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "max", ckpt_path=None):
        if mode not in ("max", "min"):
            raise ValueError(f"mode 必须是 'max' 或 'min'，实际 {mode!r}")
        if patience < 1:
            raise ValueError(f"patience 必须 >= 1，实际 {patience}")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None

        self.best_value = -math.inf if mode == "max" else math.inf
        self.best_epoch = None
        self.best_state = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def is_improvement(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def step(self, value: float, model: torch.nn.Module | None = None, epoch: int | None = None,
             extra: dict | None = None) -> bool:
        """喂入本 epoch 的验证指标；返回本次是否刷新了最佳。extra 会一并写进 checkpoint。"""
        improved = self.is_improvement(value)
        if improved:
            self.best_value = float(value)
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            if model is not None:
                self.best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                if self.ckpt_path is not None:
                    self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": self.best_state, "epoch": epoch, "best_value": self.best_value,
                                **(extra or {})}, self.ckpt_path)
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return improved

    def restore_best(self, model: torch.nn.Module) -> None:
        """把最佳权重加载回模型（若从未记录过最佳则不做任何事）。"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
