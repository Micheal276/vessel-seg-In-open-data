"""手写训练循环：AdamW + 余弦学习率（带 warmup）+ 梯度裁剪 + 每 epoch 整图验证 + early stopping。

设计理由（面试讲解要点）
------------------------
1. 一个训练 step 的标准五步：前向 → 算损失 → ``zero_grad`` → ``backward`` → ``optimizer.step``。
   在 backward 和 step 之间做 **梯度裁剪**（clip_grad_norm_）：把所有参数梯度的整体 L2 范数限制在
   max_norm 以内，防止某个 batch（比如几乎全是血管或全是背景的 patch）产生的异常大梯度把权重"踢飞"。

2. AdamW。
   Adam 自适应地为每个参数调整步长，对学习率不太敏感、收敛快；AdamW 把权重衰减从梯度里"解耦"出来，
   直接对权重做衰减，正则化效果比 Adam + L2 更符合预期（Loshchilov & Hutter, 2019）。

3. 余弦学习率 + 线性 warmup，按 step 更新（手写公式，不用 lr_scheduler 封装，便于讲解）：

       warmup 阶段（step < W）:  lr = base_lr · (step + 1) / W
       之后:                     lr = min_lr + ½(base_lr − min_lr)(1 + cos(π · progress))

   warmup：训练刚开始权重是随机的、BN 统计也不准，先用小学习率"热身"避免一开始就发散；
   余弦退火：前期大学习率快速下降，后期学习率平滑趋近 min_lr，在最优点附近精细收敛。
   周期按最大 epoch 数计算；early stopping 提前停时学习率尚未降到最低，这是可以接受的。

4. 每个 epoch 在 val split 上做"真正的"评估。
   不是在随机 patch 上算个 loss，而是对每张验证整图做滑窗推理（eval/infer.py），
   用 eval/metrics.py 在 FOV 内算 Dice / AUC 等——与最终测试完全相同的口径。
   early stopping 监控的就是这个验证 Dice（逐图平均，阈值 0.5）。

5. 日志落盘：每个 epoch 追加到 history.json / history.csv，并重画 loss / Dice 曲线 png，
   训练中途也能随时查看进度；最佳权重由 EarlyStopping 写到 best.pth。
   曲线上标注的"最佳 epoch"取 EarlyStopping.best_epoch（即真正保存的权重），而不是 val Dice 的 argmax——
   两者在 min_delta > 0 时可能不同（某 epoch 只高出不到 min_delta，不算进步、也不保存）。

6. 与数据集无关：只接收 DataLoader、验证样本列表和 config 字典，阶段二 OCTA 直接复用。
"""

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from eval.infer import predict_samples
from eval.metrics import METRIC_NAMES, MetricAccumulator, binarize
from train.early_stopping import EarlyStopping
from train.losses import BCEDiceLoss
from utils.io import save_json


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 0, min_lr: float = 0.0) -> float:
    """第 step 步（从 0 开始）的学习率：线性 warmup 后余弦退火到 min_lr。"""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def evaluate_probs(probs: list, samples: list, threshold: float = 0.5) -> dict:
    """给定概率图与样本，FOV 内计算 5 个指标的 summary（逐图 + 整体）。"""
    acc = MetricAccumulator()
    for prob, s in zip(probs, samples):
        acc.update(binarize(prob, threshold), s["label"], s["fov"], score=prob, name=s["name"])
    return acc.summary()


class Trainer:
    """参数
    ----
    model        : nn.Module（输出 1 通道 logits）
    train_loader : 产出 {"image","label","fov"} batch 的 DataLoader
    val_samples  : prepare_samples 产出的验证整图列表
    train_cfg    : config 的 ``train`` 小节
    infer_cfg    : config 的 ``infer`` 小节（滑窗参数）
    device       : torch.device
    run_dir      : 结果目录（history / 曲线 / best.pth）
    ckpt_extra   : 写进 checkpoint 的额外信息（如 unet 结构参数、config），便于测试脚本重建模型
    """

    def __init__(self, model, train_loader, val_samples, train_cfg: dict, infer_cfg: dict,
                 device: torch.device, run_dir, ckpt_extra: dict | None = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_samples = val_samples
        self.cfg = train_cfg
        self.infer_cfg = infer_cfg
        self.device = device
        self.run_dir = Path(run_dir)
        self.ckpt_extra = ckpt_extra or {}

        opt_cfg = train_cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]),
                                           weight_decay=float(opt_cfg.get("weight_decay", 1e-4)))
        self.criterion = BCEDiceLoss.from_config(train_cfg.get("loss", {}))
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(patience=es_cfg.get("patience", 20),
                                            min_delta=es_cfg.get("min_delta", 0.0), mode="max",
                                            ckpt_path=self.run_dir / "best.pth")
        self.history = []

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int, global_step: int, total_steps: int) -> tuple:
        opt_cfg = self.cfg["optimizer"]
        base_lr, min_lr = float(opt_cfg["lr"]), float(opt_cfg.get("min_lr", 0.0))
        warmup_steps = int(opt_cfg.get("warmup_epochs", 0)) * len(self.train_loader)
        clip = self.cfg.get("grad_clip", None)

        self.model.train()
        sums = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}
        n_batches = 0
        bar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False, mininterval=2.0)
        for batch in bar:
            lr = cosine_lr(global_step, total_steps, base_lr, warmup_steps, min_lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            image = batch["image"].to(self.device, dtype=torch.float32)
            label = batch["label"].to(self.device, dtype=torch.float32)
            fov = batch["fov"].to(self.device, dtype=torch.float32)

            logits = self.model(image)
            losses = self.criterion(logits, label, fov)

            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(clip))
            self.optimizer.step()

            for k in sums:
                sums[k] += float(losses[k].detach())
            n_batches += 1
            global_step += 1
            bar.set_postfix(loss=f"{sums['loss'] / n_batches:.4f}", lr=f"{lr:.2e}", refresh=False)

        return {k: v / max(n_batches, 1) for k, v in sums.items()}, global_step, lr

    def validate(self) -> dict:
        probs = predict_samples(self.model, self.val_samples, self.infer_cfg, self.device)
        return evaluate_probs(probs, self.val_samples, float(self.cfg.get("val_threshold", 0.5)))

    # ------------------------------------------------------------------
    def fit(self, epochs: int) -> dict:
        total_steps = epochs * len(self.train_loader)
        global_step = 0
        t_start = time.time()
        print(f"[trainer] epochs={epochs}, steps/epoch={len(self.train_loader)}, 设备={self.device}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_stats, global_step, lr = self.train_one_epoch(epoch, global_step, total_steps)
            val = self.validate()
            val_dice = val["mean"]["dice"]
            improved = self.early_stopping.step(val_dice, self.model, epoch,
                                                extra={**self.ckpt_extra, "val_mean": val["mean"]})

            row = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_stats.items()},
                   **{f"val_{k}": val["mean"][k] for k in METRIC_NAMES},
                   "val_global_dice": val["global"]["dice"],
                   "epoch_time_s": time.time() - t0, "elapsed_s": time.time() - t_start,
                   "best": improved}
            self.history.append(row)
            print(f"[trainer] epoch {epoch:3d} | lr {lr:.2e} | loss {train_stats['loss']:.4f} "
                  f"(bce {train_stats['bce']:.4f}, dice_loss {train_stats['dice_loss']:.4f}) | "
                  f"val dice {val_dice:.4f} auc {val['mean']['auc']:.4f} | "
                  f"{row['epoch_time_s']:.1f}s{'  *best*' if improved else ''}", flush=True)
            self.save_logs()

            if self.early_stopping.should_stop:
                print(f"[trainer] early stopping：连续 {self.early_stopping.patience} 个 epoch 验证 Dice 无提升，"
                      f"在 epoch {epoch} 停止", flush=True)
                break

        self.early_stopping.restore_best(self.model)
        result = {"best_epoch": self.early_stopping.best_epoch,
                  "best_val_dice": self.early_stopping.best_value,
                  "epochs_run": len(self.history), "max_epochs": epochs,
                  "stopped_early": self.early_stopping.should_stop,
                  "wall_time_s": time.time() - t_start,
                  "checkpoint": str(self.run_dir / "best.pth")}
        save_json(result, self.run_dir / "train_summary.json")
        print(f"[trainer] 完成：最佳 epoch {result['best_epoch']}，val Dice {result['best_val_dice']:.4f}，"
              f"共 {result['epochs_run']} epoch，用时 {result['wall_time_s'] / 60:.1f} 分钟", flush=True)
        return result

    # ------------------------------------------------------------------
    def save_logs(self) -> None:
        save_json(self.history, self.run_dir / "history.json")
        with open(self.run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)
        plot_history(self.history, self.run_dir / "curves.png", best_epoch=self.early_stopping.best_epoch)


def resolve_best_epoch(history: list, best_epoch: int | None = None) -> int | None:
    """确定曲线上要标注的"最佳 epoch"——必须与 EarlyStopping 实际保存的权重一致。

    优先级：显式传入的 best_epoch（来自 EarlyStopping.best_epoch / train_summary.json）
    → history 中最后一个 ``best=True`` 的行（EarlyStopping.step 返回 True 的 epoch）
    → 旧日志没有 ``best`` 字段时才退回 nanargmax(val_dice)。
    注意不能直接用 nanargmax：early stopping 要求提升超过 min_delta 才算"进步"，
    某个 epoch 的 val Dice 可能比保存的最佳略高（差值 < min_delta）却没有被保存。
    """
    if best_epoch is not None:
        return int(best_epoch)
    flagged = [r["epoch"] for r in history if r.get("best")]
    if flagged:
        return int(flagged[-1])
    if any("best" in r for r in history):
        return None                       # 有 best 字段但从未刷新（如全 NaN）
    dice = np.array([r["val_dice"] for r in history], dtype=np.float64)
    return int(history[int(np.nanargmax(dice))]["epoch"]) if np.isfinite(dice).any() else None


def plot_history(history: list, path, best_epoch: int | None = None) -> None:
    """左：训练 loss（总 / BCE / Dice loss）；右：验证 Dice 与 AUC，标出 EarlyStopping 保存的最佳 epoch。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("train_loss", "total"), ("train_bce", "BCE"), ("train_dice_loss", "Dice loss")):
        axes[0].plot(ep, [r[key] for r in history], label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [r["val_dice"] for r in history], label="val Dice (mean, t=0.5)")
    axes[1].plot(ep, [r["val_auc"] for r in history], label="val AUC (mean)")
    best = resolve_best_epoch(history, best_epoch)
    if best is not None:
        axes[1].axvline(best, color="gray", ls="--", lw=1, label=f"best epoch {best} (early stopping)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val metric")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
