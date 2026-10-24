"""任务 1.2 经典 baseline 入口：{Frangi, 匹配滤波} × {Otsu, 训练集全局阈值}，输出指标表与可视化。

用法::

    python -m classical.run_baseline --config configs/drive.yaml                 # 默认 train,val,test
    python -m classical.run_baseline --config configs/drive.yaml --splits train,val  # 调参/检查时完全不碰 test

``--splits`` 指定要加载和评估的 split：不在列表里的 split 根本不会被读取，也不会计算/打印其指标
（test 的对比图与响应图只有在 test 被选中时才生成）。全局阈值搜索所用的 split（config 中
``threshold.search_split``，即 train）必须包含在内。参数调优脚本见 classical/tune_baseline.py。

设计理由（面试讲解要点）
------------------------
1. 流程：读数据 → 预处理（classical/preprocess.py）→ 两种滤波得到 FOV 内归一化到 [0,1] 的连续响应
   （classical/filters.py）→ 两种阈值化得到二值图 → 用 eval/metrics.py 计算指标。
   指标模块与数据集、方法无关，这里只是调用者；AUC 用连续响应，Dice 等用二值图（计划书要求）。

2. 两种阈值化，都报告：
   - Otsu：逐图、无监督。在该图 FOV 内响应的直方图上找"类间方差最大"的阈值。
     不需要任何标签，但它假设直方图是双峰的；血管只占 FOV 约 12%，响应分布偏斜时 Otsu 容易偏。
   - 全局阈值：有监督。在 **train split** 上枚举候选阈值（config 里的 grid），选使
     **train 上逐图平均 Dice** 最大的那个，然后固定地用到 val / test。
     选"逐图平均 Dice"而不是"整体像素 Dice"，是为了和指标表的主口径（逐图 mean）一致。
     只用 train 不用 val：val 保留为"没参与任何选择"的检查集，test 更是从未参与调参或选阈值。

3. 输出全部写进 ``results/classical_<experiment>/<时间戳>/``（每次运行新目录，旧结果不覆盖）：
   - ``metrics.md`` / ``metrics.csv`` / ``metrics.json``：指标表（json 含逐图结果、阈值、阈值搜索曲线）；
   - ``config.json``：本次运行使用的完整配置快照，结果可追溯；
   - ``maps/<split>/``：test 图的连续响应 png、二值 png，以及归一化响应 ``.npy``（float32，
     供任务 1.5 画 ROC、做三方法对比；results/ 不入 git）；
   - ``figures/compare_<name>.png``：原图 / 绿通道 / 预处理后 / 金标准 / 两种滤波的响应与二值图。

4. 训练集 / 测试集严格分离：所有参数在 config（如 configs/drive.yaml）里，只在 train 上调过；本脚本里 test
   只被"读出来算指标"，不参与任何选择。
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # 无界面后端：只存图不弹窗
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

from classical.filters import enhance
from classical.preprocess import preprocess
from datasets import build_dataset
from eval.metrics import (METRIC_NAMES, MetricAccumulator, binarize, confusion_counts,
                          format_summary, metrics_from_counts)
from utils.io import load_config, make_run_dir, save_json

METHODS = ("frangi", "matched_filter")
METHOD_LABELS = {"frangi": "Frangi", "matched_filter": "Matched filter"}
THRESHOLDS = ("otsu", "global")
THRESHOLD_LABELS = {"otsu": "Otsu (per image)", "global": "Global (searched on train)"}
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------------

def compute_records(cfg: dict, split: str) -> list:
    """对一个 split 的每张图做预处理和两种滤波，返回记录列表（全部在内存里，20 张图约几百 MB 以内）。"""
    ds = build_dataset(cfg, split)   # 按 config 的 data.name 经注册表选适配类，本脚本不认识具体数据集
    records = []
    for i in range(len(ds)):
        sample = ds[i]
        stages = preprocess(sample["image"], sample["fov"], cfg["preprocess"])
        resp = {m: enhance(stages["output"], sample["fov"], m, cfg["classical"]) for m in METHODS}
        records.append({"name": sample["name"], "image": sample["image"], "label": sample["label"],
                        "fov": sample["fov"], "stages": stages, "resp": resp})
    return records


def otsu_threshold(resp: np.ndarray, fov: np.ndarray) -> float:
    """只用 FOV 内的响应值计算 Otsu 阈值（FOV 外的大片 0 会把直方图拉偏）。"""
    return float(threshold_otsu(resp[fov.astype(bool)]))


def search_global_threshold(records: list, method: str, grid) -> tuple:
    """在给定记录（train split）上枚举阈值，返回 (最佳阈值, 搜索曲线)。

    准则：逐图 Dice 的平均值最大。曲线为 [(阈值, 平均 Dice), ...]，存进 json 便于复查。
    """
    start, stop, step = grid
    candidates = np.round(np.arange(start, stop + step / 2, step), 6)
    curve = []
    for t in candidates:
        dices = [metrics_from_counts(confusion_counts(binarize(r["resp"][method], t), r["label"], r["fov"]))["dice"]
                 for r in records]
        curve.append((float(t), float(np.mean(dices))))
    best_t, _ = max(curve, key=lambda tc: tc[1])
    return best_t, curve


def evaluate(records: list, method: str, thr_mode: str, global_t: float) -> tuple:
    """用指定阈值方式评估一个 split，返回 (summary, 每图阈值 dict, 二值图 dict)。"""
    acc = MetricAccumulator()
    thresholds, binaries = {}, {}
    for r in records:
        score = r["resp"][method]
        t = otsu_threshold(score, r["fov"]) if thr_mode == "otsu" else global_t
        pred = binarize(score, t) * r["fov"]   # FOV 外置 0（指标本就只在 FOV 内算，这里是为了存图干净）
        acc.update(pred, r["label"], r["fov"], score=score, name=r["name"])
        thresholds[r["name"]] = t
        binaries[r["name"]] = pred
    return acc.summary(), thresholds, binaries


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _fmt(summary: dict, k: str) -> str:
    return f"{summary['mean'][k]:.4f} ± {summary['std'][k]:.4f}"


def markdown_table(results: dict, split: str, thr_values: dict) -> str:
    """一个 split 的 Markdown 指标表：逐图 mean ± std，外加整体像素级 Dice / AUC。"""
    head = ("| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for m in METHODS:
        for t in THRESHOLDS:
            s = results[split][m][t]
            tlabel = THRESHOLD_LABELS[t] + (f", t={thr_values[m]:.2f}" if t == "global" else "")
            rows.append(f"| {METHOD_LABELS[m]} | {tlabel} | " + " | ".join(_fmt(s, k) for k in METRIC_NAMES)
                        + f" | {s['global']['dice']:.4f} | {s['global']['auc']:.4f} |")
    return head + "\n" + "\n".join(rows)


def write_csv(results: dict, path) -> None:
    """每行一个 (split, method, threshold)；每个指标给 mean / std / global 三列。"""
    fields = ["split", "method", "threshold"] + [f"{k}_{agg}" for k in METRIC_NAMES for agg in ("mean", "std", "global")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split in results:
            for m in METHODS:
                for t in THRESHOLDS:
                    s = results[split][m][t]
                    row = {"split": split, "method": m, "threshold": t}
                    for k in METRIC_NAMES:
                        row[f"{k}_mean"] = f"{s['mean'][k]:.6f}"
                        row[f"{k}_std"] = f"{s['std'][k]:.6f}"
                        row[f"{k}_global"] = f"{s['global'][k]:.6f}"
                    w.writerow(row)


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def save_maps(records: list, binaries: dict, out_dir) -> None:
    """保存连续响应（png + npy）与二值图（png）。binaries[(method, thr)][name] -> 二值图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        for m in METHODS:
            Image.fromarray(to_u8(r["resp"][m])).save(out_dir / f"{r['name']}_{m}_response.png")
            np.save(out_dir / f"{r['name']}_{m}_response.npy", r["resp"][m].astype(np.float32))
            for t in THRESHOLDS:
                Image.fromarray(binaries[(m, t)][r["name"]] * 255).save(out_dir / f"{r['name']}_{m}_{t}.png")


def _gray_title(image: np.ndarray, pre_cfg: dict) -> str:
    """灰度阶段的面板标题：彩色图且开启 green_channel 时为绿通道，否则为灰度。"""
    if image.ndim == 3:
        return "Green channel" if pre_cfg.get("green_channel", True) else "Gray (channel mean)"
    return "Gray (single channel)"


def _preprocess_title(pre_cfg: dict) -> str:
    """预处理输出的面板标题：按 config 中实际开启的步骤拼接。"""
    steps = []
    if pre_cfg.get("clahe", {}).get("enabled", False):
        steps.append("CLAHE")
    if pre_cfg.get("fill_outside_fov", {}).get("enabled", False):
        steps.append("FOV fill")
    if pre_cfg.get("denoise", {}).get("method", "none") != "none":
        steps.append("denoise")
    return " + ".join(steps) if steps else "Preprocessed (none)"


def save_comparison(r: dict, binaries: dict, dices: dict, path, pre_cfg: dict | None = None,
                    title: str = "") -> None:
    """2×4 对比图：原图 / 绿通道 / 预处理后 / 金标准；Frangi 响应 / Frangi 二值 / MF 响应 / MF 二值。

    二值图用"全局阈值"版本（train 上选出的阈值），标题里注明该图 Dice。
    面板标题由图像通道数与预处理 config 生成（不写死"RGB / 绿通道"），单通道 OCTA 也能正确标注。
    """
    panels = [
        (r["image"], "Original (RGB)" if r["image"].ndim == 3 else "Original"),
        (r["stages"]["gray"], _gray_title(r["image"], pre_cfg or {})),
        (r["stages"]["output"], _preprocess_title(pre_cfg or {})), (r["label"], "Ground truth"),
        (r["resp"]["frangi"], "Frangi response"),
        (binaries[("frangi", "global")][r["name"]], f"Frangi binary (global), Dice={dices['frangi']:.3f}"),
        (r["resp"]["matched_filter"], "MF response"),
        (binaries[("matched_filter", "global")][r["name"]], f"MF binary (global), Dice={dices['matched_filter']:.3f}"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{title} image {r['name']} (test split)".strip(), fontsize=12)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_splits(text: str) -> tuple:
    """解析 "--splits train,val" 为 ("train", "val")；按 SPLITS 的固定顺序返回，未知名字报错。"""
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = sorted(set(names) - set(SPLITS))
    if not names or unknown:
        raise ValueError(f"--splits 只能取 {SPLITS} 的非空子集，实际 {text!r}")
    return tuple(s for s in SPLITS if s in names)


def main(config_path: str, splits: tuple = SPLITS):
    """运行经典 baseline，返回结果目录（Path）。"""
    cfg = load_config(config_path)
    ccfg = cfg["classical"]
    thr_cfg = ccfg["threshold"]
    if thr_cfg["search_split"] not in splits:
        raise ValueError(f"全局阈值在 {thr_cfg['search_split']!r} 上搜索，--splits 必须包含它，实际 {splits}")
    run_dir = make_run_dir(f"classical_{cfg['experiment']}")
    print(f"[baseline] 结果目录: {run_dir}；评估的 split: {', '.join(splits)}")

    records = {}
    for split in splits:
        records[split] = compute_records(cfg, split)
        print(f"[baseline] {split}: {len(records[split])} 张图已完成预处理与滤波")

    # 全局阈值：只在 search_split（train）上搜索
    search_split = thr_cfg["search_split"]
    if thr_cfg["search_criterion"] != "mean_dice":
        raise ValueError(f"暂只支持 search_criterion=mean_dice，实际 {thr_cfg['search_criterion']!r}")
    global_t, curves = {}, {}
    for m in METHODS:
        global_t[m], curves[m] = search_global_threshold(records[search_split], m, thr_cfg["grid"])
        print(f"[baseline] {METHOD_LABELS[m]}: {search_split} 上最佳全局阈值 t={global_t[m]:.2f} "
              f"(平均 Dice={dict(curves[m])[global_t[m]]:.4f})")

    results, per_thresholds, all_binaries = {}, {}, {}
    for split in splits:
        results[split], per_thresholds[split], all_binaries[split] = {}, {}, {}
        for m in METHODS:
            results[split][m], per_thresholds[split][m] = {}, {}
            for t in THRESHOLDS:
                summary, thrs, bins = evaluate(records[split], m, t, global_t[m])
                results[split][m][t] = summary
                per_thresholds[split][m][t] = thrs
                all_binaries[split][(m, t)] = bins

    for split in [s for s in ("val", "test") if s in splits]:
        for m in METHODS:
            for t in THRESHOLDS:
                print(f"\n[baseline] {split} | {METHOD_LABELS[m]} | {THRESHOLD_LABELS[t]}")
                print(format_summary(results[split][m][t]))

    # 指标表
    md = [f"# 经典 baseline 指标（{cfg['experiment'].upper()}，FOV 内）", "",
          "指标为逐图 mean ± std（总体标准差）；Global 为所有图像素汇总后的整体指标。",
          f"全局阈值在 {search_split} split 上以逐图平均 Dice 最大为准则搜索，"
          f"Frangi t={global_t['frangi']:.2f}，匹配滤波 t={global_t['matched_filter']:.2f}。", ""]
    for split in [s for s in ("test", "val", "train") if s in splits]:
        md += [f"## {split} split（{len(records[split])} 张）", "", markdown_table(results, split, global_t), ""]
    (run_dir / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    write_csv(results, run_dir / "metrics.csv")
    save_json({"global_thresholds": global_t, "threshold_search_split": search_split,
               "threshold_search_criterion": thr_cfg["search_criterion"],
               "threshold_search_curves": curves, "per_image_thresholds": per_thresholds,
               "results": results, "evaluated_splits": list(splits),
               "splits": {s: [r["name"] for r in records[s]] for s in splits}},
              run_dir / "metrics.json")
    save_json(cfg, run_dir / "config.json")

    # 响应图 / 二值图 / npy
    for split in [s for s in ccfg.get("save_splits", ["test"]) if s in splits]:
        save_maps(records[split], all_binaries[split], run_dir / "maps" / split)

    # 对比图（test 前 n 张；未选 test 时跳过）
    if "test" in splits:
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        # 对比图里的二值图用全局阈值版本，所以取 "global" 阈值方式下的逐图 Dice
        per_image = {m: {p["name"]: p["dice"] for p in results["test"][m]["global"]["per_image"]} for m in METHODS}
        for r in records["test"][: ccfg.get("n_figure_images", 2)]:
            dices = {m: per_image[m][r["name"]] for m in METHODS}
            save_comparison(r, all_binaries["test"], dices, fig_dir / f"compare_{r['name']}.png",
                            cfg["preprocess"], cfg["experiment"].upper())

    print(f"\n[baseline] 指标表: {run_dir / 'metrics.md'}")
    print((run_dir / "metrics.md").read_text(encoding="utf-8"))
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="经典 baseline（数据集由 config 的 data.name 决定）：Frangi / 匹配滤波 × Otsu / 全局阈值")
    parser.add_argument("--config", default="configs/drive.yaml")
    parser.add_argument("--splits", default=",".join(SPLITS),
                        help="逗号分隔，要加载并评估的 split（默认 train,val,test）；调参时用 train 或 train,val")
    args = parser.parse_args()
    main(args.config, parse_splits(args.splits))
