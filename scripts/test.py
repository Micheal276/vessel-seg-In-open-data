"""U-Net 测试入口：加载最佳权重 → test split 整图滑窗推理 → FOV 内指标 → 保存表格、概率图、二值图。

设计理由
--------
1. 主报告阈值固定为 0.5（sigmoid 输出的自然分界），不依赖任何数据。
2. 附加报告"val 上搜索的最佳阈值"：阈值只在 val split 上按逐图平均 Dice 选，再原样用到 test，
   test 从不参与任何选择（否则就是在测试集上调参）。
3. 与数据集无关：数据集按 config 的 ``data.name`` 从注册表构造，OCTA 复用本脚本只换 --config。
4. 预处理一致性检查：网络只见过训练时那套预处理（绿通道 / CLAHE 参数…）的输入，测试时若 config 改了
   预处理，模型照样能跑出结果、只是指标悄悄变差。因此加载 checkpoint 后对比其中记录的预处理配置
   （新 checkpoint 的 ``preprocess`` 键；旧 checkpoint 退回 ``config.unet.preprocess``）与当前 config，
   不一致或无记录时打印醒目警告（不中断：有时就是想故意换预处理做实验）。

用法::

    python -m scripts.test --config configs/drive.yaml --ckpt results/unet_drive_train/<时间戳>/best.pth

输出 ``results/unet_<experiment>/<时间戳>/``：metrics.json / metrics.md / metrics.csv、
probs/<name>.npy（float32 概率图）、pred/<name>.png（阈值 0.5 二值图，FOV 外置 0）。
"""

import argparse
import csv

import numpy as np
import torch
from PIL import Image

from utils.device import get_device  # 先导入：在 import torch 前设置 MPS 回退环境变量
from datasets import build_dataset
from eval.infer import predict_samples
from eval.metrics import METRIC_NAMES, format_summary
from models.unet import UNet
from train.patch_dataset import prepare_samples
from train.trainer import evaluate_probs
from utils.io import load_config, make_run_dir, resolve_path, save_json
from utils.seed import set_seed


def search_threshold(probs: list, samples: list, grid) -> tuple:
    """在 grid=[start, stop(含), step] 上找使逐图平均 Dice 最大的阈值，返回 (t, dice)。"""
    start, stop, step = grid
    candidates = np.round(np.arange(start, stop + step / 2, step), 4)
    best_t, best_d = 0.5, -1.0
    for t in candidates:
        d = evaluate_probs(probs, samples, float(t))["mean"]["dice"]
        if d > best_d:
            best_t, best_d = float(t), d
    return best_t, best_d


def checkpoint_preprocess(ckpt: dict) -> tuple:
    """从 checkpoint 取训练时的 (预处理配置, 数据集名)；没有记录的项返回 None。

    新 checkpoint 直接有 ``preprocess`` / ``data_name``；更早的 checkpoint 只要存了完整 ``config``
    也能从中恢复；两者都没有时返回 (None, None)。
    """
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    pre = ckpt.get("preprocess")
    if pre is None:
        pre = cfg.get("unet", {}).get("preprocess")
    name = ckpt.get("data_name")
    if name is None and isinstance(cfg.get("data"), dict):
        name = cfg["data"].get("name", "drive")
    return pre, name


def check_preprocess_consistency(ckpt: dict, cfg: dict) -> list:
    """比较 checkpoint 记录的预处理 / 数据集名与当前 config，返回警告字符串列表（空列表 = 一致）。"""
    pre_ckpt, name_ckpt = checkpoint_preprocess(ckpt)
    pre_now = cfg["unet"]["preprocess"]
    name_now = cfg["data"].get("name", "drive")
    warnings = []
    if pre_ckpt is None:
        warnings.append("checkpoint 未记录训练时的预处理配置，无法核对；请确认当前 config 的 unet.preprocess 与训练时一致")
    elif pre_ckpt != pre_now:
        warnings.append(f"预处理配置与训练时不一致！训练时: {pre_ckpt}；当前 config: {pre_now}")
    if name_ckpt is not None and name_ckpt != name_now:
        warnings.append(f"数据集与训练时不同：训练时 data.name={name_ckpt!r}，当前 {name_now!r}（跨数据集测试请确认是有意为之）")
    return warnings


def table_rows(results: dict) -> list:
    rows = []
    for (split, label), s in results.items():
        rows.append([split, label] + [f"{s['mean'][k]:.4f} ± {s['std'][k]:.4f}" for k in METRIC_NAMES]
                    + [f"{s['global']['dice']:.4f}", f"{s['global']['auc']:.4f}"])
    return rows


HEADER = ["Split", "Threshold", "Dice", "Sensitivity", "Specificity", "Accuracy", "AUC", "Global Dice", "Global AUC"]


def main(config_path: str, ckpt_path: str):
    """测试并保存结果，返回结果目录（Path）。"""
    cfg = load_config(config_path)
    set_seed(int(cfg["train"]["seed"]))
    device = get_device()
    test_cfg = cfg.get("test", {})
    thr = float(test_cfg.get("threshold", 0.5))

    ckpt = torch.load(resolve_path(ckpt_path), map_location="cpu", weights_only=False)
    model = UNet.from_config(ckpt.get("unet_cfg", cfg["unet"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[test] 权重: {ckpt_path}（epoch {ckpt.get('epoch')}, 训练时 val Dice {ckpt.get('best_value')}）")
    pre_warnings = check_preprocess_consistency(ckpt, cfg)
    for w in pre_warnings:
        print(f"[test] !!! 警告：{w}", flush=True)

    pre_cfg = cfg["unet"]["preprocess"]
    samples = {s: prepare_samples(build_dataset(cfg, s), pre_cfg) for s in ("val", "test")}
    probs = {s: predict_samples(model, samples[s], cfg["infer"], device) for s in ("val", "test")}

    best_t, best_val_dice = search_threshold(probs["val"], samples["val"], test_cfg.get("threshold_grid", [0.05, 0.95, 0.01]))
    print(f"[test] val 上最佳阈值 t={best_t:.2f}（val 逐图平均 Dice {best_val_dice:.4f}）")

    results = {}
    for split in ("test", "val"):
        results[(split, f"{thr:g}")] = evaluate_probs(probs[split], samples[split], thr)
        results[(split, f"best-val t={best_t:.2f}")] = evaluate_probs(probs[split], samples[split], best_t)

    run_dir = make_run_dir(f"unet_{cfg['experiment']}")
    for (split, label), s in results.items():
        print(f"\n[test] {split} | threshold {label}")
        print(format_summary(s))

    md = [f"# U-Net 指标（{cfg['experiment']}，FOV 内）", "",
          f"权重：`{ckpt_path}`（epoch {ckpt.get('epoch')}）。指标为逐图 mean ± std（总体标准差）；"
          "Global 为所有图像素汇总后的整体指标。",
          f"best-val 阈值只在 val split 上按逐图平均 Dice 搜索（t={best_t:.2f}），test 未参与选择。", "",
          "| " + " | ".join(HEADER) + " |", "| " + " | ".join(["---"] * len(HEADER)) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in table_rows(results)]
    md += ["", "## test 逐图（阈值 %g）" % thr, "", "| name | " + " | ".join(METRIC_NAMES) + " |",
           "| --- |" + " --- |" * len(METRIC_NAMES)]
    for m in results[("test", f"{thr:g}")]["per_image"]:
        md.append(f"| {m['name']} | " + " | ".join(f"{m[k]:.4f}" for k in METRIC_NAMES) + " |")
    (run_dir / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "threshold"] + [f"{k}_mean" for k in METRIC_NAMES]
                        + [f"{k}_std" for k in METRIC_NAMES] + [f"global_{k}" for k in METRIC_NAMES])
        for (split, label), s in results.items():
            writer.writerow([split, label] + [s["mean"][k] for k in METRIC_NAMES] + [s["std"][k] for k in METRIC_NAMES]
                            + [s["global"][k] for k in METRIC_NAMES])

    save_json({"checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt.get("epoch"), "threshold": thr,
               "preprocess_warnings": pre_warnings,
               "best_val_threshold": best_t, "best_val_threshold_dice": best_val_dice,
               "results": {f"{split}|{label}": s for (split, label), s in results.items()},
               "splits": {s: [x["name"] for x in samples[s]] for s in samples}}, run_dir / "metrics.json")
    save_json(cfg, run_dir / "config.json")

    (run_dir / "probs").mkdir(exist_ok=True)
    (run_dir / "pred").mkdir(exist_ok=True)
    for s, prob in zip(samples["test"], probs["test"]):
        np.save(run_dir / "probs" / f"{s['name']}.npy", prob.astype(np.float32))
        pred = (prob > thr).astype(np.uint8) * s["fov"]
        Image.fromarray(pred * 255).save(run_dir / "pred" / f"{s['name']}.png")

    print(f"\n[test] 结果目录: {run_dir}")
    print((run_dir / "metrics.md").read_text(encoding="utf-8"))
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估手写 U-Net（test split）")
    parser.add_argument("--config", default="configs/drive.yaml")
    parser.add_argument("--ckpt", required=True, help="best.pth 路径")
    args = parser.parse_args()
    main(args.config, args.ckpt)
