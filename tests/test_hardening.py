"""阶段一加固的单元测试：曲线最佳 epoch 与早停一致；checkpoint 记录预处理配置并在测试时核对。

设计理由：这两处都是"不报错、只误导"的问题——曲线上标的最佳 epoch 与实际保存的权重不是同一个，
或测试时换了预处理却浑然不觉。用合成数据把正确行为固化下来；全部在 CPU 上跑，几秒内完成。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.test import check_preprocess_consistency, checkpoint_preprocess
from train.early_stopping import EarlyStopping
from train.patch_dataset import RandomPatchDataset
from train.trainer import Trainer, plot_history, resolve_best_epoch
from models.unet import UNet

PRE = {"green_channel": True, "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid": [8, 8]},
       "fill_outside_fov": {"enabled": False}, "denoise": {"method": "none"}}
CFG = {"data": {"name": "drive"}, "unet": {"preprocess": PRE}}


# ---------------------------------------------------------------------------
# 曲线上的最佳 epoch
# ---------------------------------------------------------------------------

def _history_from_early_stopping(dices, min_delta):
    es = EarlyStopping(patience=100, min_delta=min_delta)
    rows = []
    for ep, d in enumerate(dices, start=1):
        improved = es.step(d, model=None, epoch=ep)
        rows.append({"epoch": ep, "val_dice": d, "val_auc": 0.9, "train_loss": 1.0, "train_bce": 0.5,
                     "train_dice_loss": 0.5, "best": improved})
    return rows, es


def test_best_epoch_respects_min_delta_not_argmax():
    # epoch 3 比 epoch 2 只高 0.0002 < min_delta，不算进步：早停保存的是 epoch 2，argmax 却是 epoch 3
    rows, es = _history_from_early_stopping([0.70, 0.80, 0.8002, 0.79], min_delta=0.0005)
    assert int(np.argmax([r["val_dice"] for r in rows])) + 1 == 3
    assert es.best_epoch == 2
    assert resolve_best_epoch(rows) == 2                    # 来自 best 标记
    assert resolve_best_epoch(rows, best_epoch=es.best_epoch) == 2
    assert resolve_best_epoch(rows, best_epoch=7) == 7       # 显式传入优先


def test_best_epoch_legacy_history_without_flag_and_all_nan():
    legacy = [{"epoch": 1, "val_dice": 0.5}, {"epoch": 2, "val_dice": 0.6}, {"epoch": 3, "val_dice": 0.55}]
    assert resolve_best_epoch(legacy) == 2
    nan_rows = [{"epoch": 1, "val_dice": float("nan"), "best": False}]
    assert resolve_best_epoch(nan_rows) is None


def test_plot_history_writes_png(tmp_path):
    rows, es = _history_from_early_stopping([0.70, 0.80, 0.8002, 0.79], min_delta=0.0005)
    out = tmp_path / "curves.png"
    plot_history(rows, out, best_epoch=es.best_epoch)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# checkpoint 中的预处理配置
# ---------------------------------------------------------------------------

def test_consistency_new_checkpoint_matches():
    ckpt = {"preprocess": dict(PRE), "data_name": "drive"}
    assert checkpoint_preprocess(ckpt) == (PRE, "drive")
    assert check_preprocess_consistency(ckpt, CFG) == []


def test_consistency_detects_mismatch_and_dataset_change():
    other = {**PRE, "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid": [8, 8]}}
    warns = check_preprocess_consistency({"preprocess": other, "data_name": "rose"}, CFG)
    assert len(warns) == 2
    assert any("不一致" in w for w in warns) and any("rose" in w for w in warns)


def test_consistency_legacy_checkpoint_with_config_only():
    ckpt = {"model": {}, "config": {"data": {"name": "drive"}, "unet": {"preprocess": PRE}}}
    assert checkpoint_preprocess(ckpt) == (PRE, "drive")
    assert check_preprocess_consistency(ckpt, CFG) == []


def test_consistency_checkpoint_without_any_record_warns_not_crash():
    warns = check_preprocess_consistency({"model": {}, "epoch": 3}, CFG)
    assert len(warns) == 1 and "未记录" in warns[0]


def test_trainer_checkpoint_contains_preprocess(tmp_path):
    """端到端：Trainer 把 ckpt_extra（scripts/train.py 传入 preprocess / data_name）写进 best.pth。"""
    rng = np.random.default_rng(0)
    samples = []
    for i in range(2):
        label = (rng.random((32, 32)) > 0.8).astype(np.uint8)
        samples.append({"image": rng.random((32, 32)).astype(np.float32), "label": label,
                        "fov": np.ones((32, 32), np.uint8), "name": str(i)})
    loader = DataLoader(RandomPatchDataset(samples, 16, 4, seed=0), batch_size=4)
    train_cfg = {"optimizer": {"lr": 1e-3}, "early_stopping": {"patience": 5, "min_delta": 0.0}}
    trainer = Trainer(UNet(in_channels=1, base_channels=4, depth=2), loader, samples, train_cfg,
                      {"patch_size": 16, "stride": 8, "batch_size": 8}, torch.device("cpu"), tmp_path,
                      ckpt_extra={"preprocess": PRE, "data_name": "drive"})
    result = trainer.fit(2)
    ckpt = torch.load(tmp_path / "best.pth", map_location="cpu", weights_only=False)
    assert ckpt["preprocess"] == PRE and ckpt["data_name"] == "drive"
    assert check_preprocess_consistency(ckpt, CFG) == []
    assert (tmp_path / "curves.png").exists()
    assert resolve_best_epoch(trainer.history) == result["best_epoch"]
