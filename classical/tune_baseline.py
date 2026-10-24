"""经典 baseline 的参数网格搜索：只用 train split，可复现、可审计。

用法::

    python -m classical.tune_baseline --config configs/drive.yaml

为什么调参绝对不能看 test（面试讲解要点）
------------------------------------------
1. test 的作用是估计"方法在没见过的数据上的表现"。只要某个参数是因为 test 分数更高才被选中，
   test 就变成了训练数据的一部分，报告出来的数字会系统性偏高（选择偏差 / 数据泄漏），
   而且偏高多少无从得知。经典方法虽然没有"训练"，但 Frangi 的 gamma、匹配滤波的 σ / L、
   预处理的填充宽度等都是"用数据选出来的超参数"，同样会过拟合到用来选它们的那几张图上。
2. 本项目 test 只有 4 张图，噪声很大：在几十个组合里挑 test 最高的那个，几乎一定挑到"恰好适合
   这 4 张图"的组合。所以所有选择（参数、全局阈值）都只在 train（12 张）上做；
   val 保留为没参与任何选择的检查集，test 只在最终 run_baseline 里读出来算一次指标。
3. "只在 train 上调过"必须能从代码验证，而不是一句口头保证：本脚本只通过 ``_load_split`` 读数据，
   它对 train 以外的 split 直接抛异常；同时经 ``datasets.build_dataset`` 构造 val / test 数据集对象、
   只取其样本名单（构造时只列文件名，不读图像），确认加载的图像与 val/test 名单不相交。
   这一检查只依赖基类契约（``.names``），不含任何数据集特有的文件名逻辑，换成 OCTA 同样成立。
   test 的图像在本脚本里从未被读取。

搜索内容
--------
- 预处理：FOV 外填充的 ``erode_px``（FOV 最外圈重新填充的宽度）× 高斯去噪 σ；CLAHE 固定为 config 值；
- Frangi：``gamma``（"auto" 表示 skimage 默认 None）；其余（sigmas、beta）固定为 config 值；
- 匹配滤波：高斯横截面 ``sigma`` × 模板长度 ``length``；方向数固定 12。
两种方法共享同一组预处理网格，分别独立选最优（方法之间没有耦合）。

目标函数与 run_baseline 完全一致：在 train 上枚举 config 中的阈值网格（0.01–0.99，步长 0.01），
取使"逐图 Dice 平均值"最大的全局阈值，该最大值即为该参数组合的得分。为了快，这里用排序 +
二分查找一次算出所有阈值下的 TP/FP/FN（与 ``binarize`` 的"严格大于"语义相同），
结束时再用 run_baseline.search_global_threshold 对最优组合和当前 config 组合复核，两者必须一致。
另外顺带记录逐图 Otsu 的平均 Dice（仅供参考，不参与选择）。

输出 ``results/classical_tune/<时间戳>/``：
- ``grid_results.csv``：每行一个 (方法, 参数组合)，含 train 全局阈值、train 平均 Dice、Otsu 平均 Dice、排名；
- ``grid_results.json``：网格定义、全部结果、各方法最优组合、当前 config 组合的得分与排名、复核结果；
- ``config.json``：运行时的配置快照。
"""

import argparse
import copy
import csv
import itertools
import time

import numpy as np

from classical.filters import enhance
from classical.preprocess import preprocess
from classical.run_baseline import METHODS, METHOD_LABELS, otsu_threshold, search_global_threshold
from datasets import build_dataset
from eval.metrics import binarize, confusion_counts, metrics_from_counts
from utils.io import load_config, make_run_dir, save_json

TUNE_SPLIT = "train"   # 唯一允许读取的 split

# 网格（保持适中：CPU 上约几分钟）。"auto" = skimage 默认 gamma=None。
GRID = {
    "preprocess": {
        "erode_px": [0, 2, 3, 5, 8],
        "denoise_sigma": [1.0, 1.5, 2.0],
    },
    "frangi": {
        "gamma": ["auto", 0.003, 0.006, 0.01, 0.02],
    },
    "matched_filter": {
        "sigma": [1.0, 1.5, 2.0, 2.5],
        "length": [7, 9, 11, 13],
    },
}


# ---------------------------------------------------------------------------
# 数据：只允许 train
# ---------------------------------------------------------------------------

def _load_split(cfg: dict, split: str) -> list:
    """唯一的数据入口。split 不是 train 时直接报错，保证 val/test 图像不会被读取。"""
    if split != TUNE_SPLIT:
        raise RuntimeError(f"tune_baseline 只允许读取 {TUNE_SPLIT!r} split，拒绝读取 {split!r}")
    ds = build_dataset(cfg, split)
    # 名单级检查（与具体数据集无关）：再构造 val / test 两个数据集对象，只取它们的样本名单
    # （构造时只列文件名、不读图像），确认 train 与 val/test 不相交。val/test 的图像从未被读取。
    held_out = set(build_dataset(cfg, "val").names) | set(build_dataset(cfg, "test").names)
    expected = list(ds.names)
    if set(expected) & held_out:
        raise RuntimeError(f"train 名单与 val/test 相交: {sorted(set(expected) & held_out)}")
    samples = [ds[i] for i in range(len(ds))]
    names = [s["name"] for s in samples]
    if names != expected:
        raise RuntimeError(f"加载的图像 {names} 与 train 名单 {expected} 不符")
    return samples


# ---------------------------------------------------------------------------
# 快速目标函数（与 run_baseline.search_global_threshold 等价）
# ---------------------------------------------------------------------------

def threshold_candidates(grid) -> np.ndarray:
    """与 run_baseline.search_global_threshold 相同的候选阈值。"""
    start, stop, step = grid
    return np.round(np.arange(start, stop + step / 2, step), 6)


def dice_curve(resp: np.ndarray, label: np.ndarray, fov: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """一张图在所有候选阈值下的 Dice（FOV 内，预测为 resp > t）。

    把 FOV 内血管 / 背景像素的响应分别排序，"大于 t 的个数" = n - searchsorted(..., side="right")，
    一次得到所有阈值的 TP、FP，FN = 血管数 - TP。
    """
    f = fov.astype(bool)
    s = resp[f].astype(np.float64)
    y = label[f].astype(bool)
    pos = np.sort(s[y])
    neg = np.sort(s[~y])
    tp = len(pos) - np.searchsorted(pos, cands, side="right")
    fp = len(neg) - np.searchsorted(neg, cands, side="right")
    fn = len(pos) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2.0 * tp / np.maximum(den, 1), 1.0)


def score_responses(resps: list, samples: list, cands: np.ndarray) -> dict:
    """一个参数组合在 train 上的得分：最佳全局阈值、对应平均 Dice、逐图 Otsu 平均 Dice。"""
    curves = np.stack([dice_curve(r, s["label"], s["fov"], cands) for r, s in zip(resps, samples)])
    mean_curve = curves.mean(axis=0)
    best = int(np.argmax(mean_curve))   # 并列时取第一个（最小阈值），与 max() 的行为一致
    otsu = [metrics_from_counts(confusion_counts(binarize(r, otsu_threshold(r, s["fov"])),
                                                 s["label"], s["fov"]))["dice"]
            for r, s in zip(resps, samples)]
    return {"global_t": float(cands[best]), "train_mean_dice": float(mean_curve[best]),
            "train_otsu_mean_dice": float(np.mean(otsu))}


# ---------------------------------------------------------------------------
# 配置组合
# ---------------------------------------------------------------------------

def make_cfg(base: dict, pre: dict, method: str, params: dict) -> dict:
    """在 config 副本上覆盖本次组合的参数。"""
    cfg = copy.deepcopy(base)
    cfg["preprocess"]["fill_outside_fov"]["erode_px"] = pre["erode_px"]
    cfg["preprocess"]["denoise"]["method"] = "gaussian"
    cfg["preprocess"]["denoise"]["sigma"] = pre["denoise_sigma"]
    if method == "frangi":
        cfg["classical"]["frangi"]["gamma"] = None if params["gamma"] == "auto" else params["gamma"]
    else:
        cfg["classical"]["matched_filter"]["sigma"] = params["sigma"]
        cfg["classical"]["matched_filter"]["length"] = params["length"]
    return cfg


def current_params(cfg: dict) -> dict:
    """当前 config 对应的组合（用于在网格结果里定位它的排名）。"""
    pre = {"erode_px": cfg["preprocess"]["fill_outside_fov"]["erode_px"],
           "denoise_sigma": cfg["preprocess"]["denoise"]["sigma"]}
    g = cfg["classical"]["frangi"]["gamma"]
    return {"frangi": {**pre, "gamma": "auto" if g is None else g},
            "matched_filter": {**pre, "sigma": cfg["classical"]["matched_filter"]["sigma"],
                               "length": cfg["classical"]["matched_filter"]["length"]}}


def _combos(method: str) -> list:
    keys = list(GRID[method])
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[method][k] for k in keys))]


def _same(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in a)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    base = load_config(config_path)
    if base["classical"]["threshold"]["search_criterion"] != "mean_dice":
        raise ValueError("目标函数与 run_baseline 一致，仅支持 search_criterion=mean_dice")
    cands = threshold_candidates(base["classical"]["threshold"]["grid"])
    run_dir = make_run_dir("classical_tune")
    print(f"[tune] 结果目录: {run_dir}")

    samples = _load_split(base, TUNE_SPLIT)
    print(f"[tune] 只读取 {TUNE_SPLIT} split：{len(samples)} 张 {[s['name'] for s in samples]}")

    pre_combos = _combos("preprocess")
    rows = []
    t0 = time.time()
    for pi, pre in enumerate(pre_combos):
        for method in METHODS:
            for params in _combos(method):
                cfg = make_cfg(base, pre, method, params)
                resps = []
                for s in samples:
                    out = preprocess(s["image"], s["fov"], cfg["preprocess"])["output"]
                    resps.append(enhance(out, s["fov"], method, cfg["classical"]))
                rows.append({"method": method, **pre, **params, **score_responses(resps, samples, cands)})
        print(f"[tune] 预处理组合 {pi + 1}/{len(pre_combos)} {pre} 完成（累计 {time.time() - t0:.0f} s）")

    # 排名、最优、当前 config
    cur = current_params(base)
    summary = {}
    for method in METHODS:
        mrows = sorted([r for r in rows if r["method"] == method], key=lambda r: -r["train_mean_dice"])
        for rank, r in enumerate(mrows, 1):
            r["rank"] = rank
        cur_row = next((r for r in mrows if _same(cur[method], r)), None)
        summary[method] = {"best": mrows[0], "current_config": cur[method], "current_row": cur_row,
                           "n_combos": len(mrows)}

    # 复核：用 run_baseline 的原始阈值搜索重算最优组合与当前组合，必须与快速实现一致
    checks = []
    for method in METHODS:
        for tag, r in (("best", summary[method]["best"]), ("current", summary[method]["current_row"])):
            if r is None:
                continue
            keys = list(GRID["preprocess"]) + list(GRID[method])
            cfg = make_cfg(base, {k: r[k] for k in GRID["preprocess"]}, method, {k: r[k] for k in GRID[method]})
            recs = []
            for s in samples:
                out = preprocess(s["image"], s["fov"], cfg["preprocess"])["output"]
                recs.append({"label": s["label"], "fov": s["fov"],
                             "resp": {method: enhance(out, s["fov"], method, cfg["classical"])}})
            t_ref, curve = search_global_threshold(recs, method, base["classical"]["threshold"]["grid"])
            d_ref = dict(curve)[t_ref]
            ok = abs(t_ref - r["global_t"]) < 1e-9 and abs(d_ref - r["train_mean_dice"]) < 1e-9
            checks.append({"method": method, "which": tag, "params": {k: r[k] for k in keys},
                           "ref_t": t_ref, "ref_dice": d_ref, "fast_t": r["global_t"],
                           "fast_dice": r["train_mean_dice"], "match": ok})
            if not ok:
                raise RuntimeError(f"快速目标函数与 run_baseline 不一致: {checks[-1]}")

    # 保存
    fields = ["method", "rank", *GRID["preprocess"], "gamma", "sigma", "length",
              "global_t", "train_mean_dice", "train_otsu_mean_dice"]
    with open(run_dir / "grid_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["method"], r["rank"])):
            w.writerow({k: r.get(k, "") for k in fields})
    save_json({"split_used": TUNE_SPLIT, "train_names": [s["name"] for s in samples],
               "objective": "train 逐图平均 Dice（全局阈值在 train 上按 config 网格搜索，与 run_baseline 相同）",
               "threshold_grid": base["classical"]["threshold"]["grid"], "grid": GRID,
               "fixed": {"clahe": base["preprocess"]["clahe"],
                         "frangi_sigmas": base["classical"]["frangi"]["sigmas"],
                         "frangi_beta": base["classical"]["frangi"]["beta"],
                         "mf_n_angles": base["classical"]["matched_filter"]["n_angles"]},
               "summary": summary, "consistency_checks": checks, "results": rows,
               "elapsed_s": time.time() - t0},
              run_dir / "grid_results.json")
    save_json(base, run_dir / "config.json")

    print(f"\n[tune] 共 {len(rows)} 个组合，耗时 {time.time() - t0:.0f} s；复核 {len(checks)} 项全部一致")
    for method in METHODS:
        b, c = summary[method]["best"], summary[method]["current_row"]
        keys = list(GRID["preprocess"]) + list(GRID[method])
        print(f"\n[tune] {METHOD_LABELS[method]}（{summary[method]['n_combos']} 个组合）")
        print(f"  最优: {{{', '.join(f'{k}={b[k]}' for k in keys)}}}  train 平均 Dice={b['train_mean_dice']:.4f} "
              f"(t={b['global_t']:.2f}; Otsu {b['train_otsu_mean_dice']:.4f})")
        if c is None:
            print(f"  当前 config {summary[method]['current_config']} 不在网格内")
        else:
            print(f"  当前 config: {{{', '.join(f'{k}={c[k]}' for k in keys)}}}  train 平均 Dice={c['train_mean_dice']:.4f} "
                  f"(t={c['global_t']:.2f}; Otsu {c['train_otsu_mean_dice']:.4f})，排名 {c['rank']}")
    print(f"\n[tune] 网格结果: {run_dir / 'grid_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="经典 baseline 参数网格搜索（只用 train split）")
    parser.add_argument("--config", default="configs/drive.yaml")
    main(parser.parse_args().config)
