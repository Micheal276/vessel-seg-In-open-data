"""任务 1.5 对比报告：在同一 split（默认 test）上把 Frangi、匹配滤波、U-Net 放进同一张指标表。

设计理由（面试讲解要点）
------------------------
1. 不重新跑方法，只"重新算指标"。
   经典 baseline（classical/run_baseline.py）与 U-Net 测试（scripts/test.py）已经把连续分数图存成 float32 .npy：
   经典方法是 FOV 内归一化后的响应图（maps/<split>/<name>_<method>_response.npy），U-Net 是 sigmoid 概率图
   （probs/<name>.npy）。本模块读这些图 + 数据集自带的 label / FOV，统一用 eval/metrics.py 重新计算全部指标。
   这样三种方法的数字出自同一段代码、同一份 label/FOV，表格口径必然一致；
   同时把重算结果与两个来源 run 的 metrics.json 逐项比对（mean / std / global / 逐图，容差默认 1e-6），
   不一致直接报错——这能抓住"拿错了 run 目录""label 读法变了""阈值不是那个阈值"之类的静默错误。

2. 每种方法用哪个阈值（阈值是方法的一部分，必须事先定好、不能看 test）：
   - 主表（primary）：经典方法用 **train split 上搜索的全局阈值**（Frangi 与匹配滤波各一个，取自经典 run 的
     metrics.json ``global_thresholds``）；U-Net 用 **0.5**（sigmoid 的自然分界，scripts/test.py 的主报告阈值）。
     三者都是"在训练数据上确定、对 test 原样使用"的固定阈值，可比性最好。
     不用 Otsu 当主行：Otsu 逐图自适应，对匹配滤波明显偏保守（Sens 低），会让对比偏向"阈值选择"而非"方法本身"。
   - 副表（secondary）：经典方法的逐图 Otsu 阈值（在 FOV 内的响应直方图上重新计算，复用
     classical/run_baseline.py::otsu_threshold），以及 U-Net 在 val 上搜索的阈值（来源 run 已给出），作为参考。
   AUC 与阈值无关，同一方法在两张表里的 AUC 相同。

3. 指标口径与 eval/metrics.py 相同：只在 FOV 内；逐图 mean ± std（总体标准差）+ 整体像素级 Global Dice / AUC。

4. 与数据集无关：label / FOV 通过 ``datasets.build_dataset(cfg, split)`` 按 config 的 ``data.name`` 获取；
   两个来源目录由 config 的 ``compare`` 小节或命令行给出。阶段二 OCTA 只需换 config 和目录（FOV 全 1 也成立）。
   眼底图特有的分析（视盘）由 config 开关控制，本模块不含任何数据集特有的常量或文件名逻辑。
   本模块不会重算经典响应或重跑 U-Net 推理：需要新结果时先运行 classical.run_baseline / scripts.test。

用法::

    python -m eval.evaluate --config configs/drive.yaml \\
        --classical-dir results/classical_drive/<时间戳> --unet-dir results/unet_drive/<时间戳>

输出 ``results/compare_<experiment>/<时间戳>/``：metrics.md / metrics.csv / metrics.json，以及 eval/visualize.py
生成的 roc.png、dice_boxplot.png、example_<name>.png、zoom_<name>.png。
可选"补充分析"（config ``compare.supplementary``，缺省 false）：细 / 粗血管召回率；视盘区域假阳性另由
``compare.supplementary_optic_disc``（缺省 false）单独开关——视盘是眼底图特有结构，OCTA 不应开启。
两者都是启发式区域，仅用于解释主表差异；比值分母为 0 时记为 NaN / n/a，不会崩溃。
局部放大图同样只在 ``supplementary_optic_disc`` 为 true 时裁视盘区域，否则只裁细血管最密区域。
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from classical.run_baseline import otsu_threshold
from datasets import build_dataset
from eval.metrics import METRIC_NAMES, MetricAccumulator, binarize
from utils.io import REPO_ROOT, load_config, load_json, make_run_dir, resolve_path, save_json

CLASSICAL_METHODS = {"frangi": "Frangi", "matched_filter": "Matched filter"}
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# 读取来源 run
# ---------------------------------------------------------------------------

def load_labels(cfg: dict, split: str) -> list:
    """按 config 的数据集注册名读取 split 的 {image, label, fov, name}（image 只用于可视化）。"""
    ds = build_dataset(cfg, split)
    return [ds[i] for i in range(len(ds))]


def load_classical(classical_dir, split: str, names: list) -> dict:
    """读经典 run：每种方法每张图的归一化响应图、train 上搜索的全局阈值、来源指标。"""
    classical_dir = resolve_path(classical_dir)
    src = load_json(classical_dir / "metrics.json")
    if src["splits"][split] != names:
        raise ValueError(f"经典 run 的 {split} 划分 {src['splits'][split]} 与当前数据集 {names} 不一致")
    scores = {m: {n: np.load(classical_dir / "maps" / split / f"{n}_{m}_response.npy").astype(np.float32)
                  for n in names} for m in CLASSICAL_METHODS}
    return {"dir": classical_dir, "scores": scores, "global_thresholds": src["global_thresholds"],
            "search_split": src.get("threshold_search_split"), "source": src["results"][split],
            "per_image_thresholds": src.get("per_image_thresholds", {}).get(split, {})}


def load_unet(unet_dir, split: str, names: list) -> dict:
    """读 U-Net 测试 run：概率图、主阈值（0.5）、val 上搜索的阈值、来源指标。"""
    unet_dir = resolve_path(unet_dir)
    src = load_json(unet_dir / "metrics.json")
    if src["splits"][split] != names:
        raise ValueError(f"U-Net run 的 {split} 划分 {src['splits'][split]} 与当前数据集 {names} 不一致")
    thr, best_t = float(src["threshold"]), float(src["best_val_threshold"])
    probs = {n: np.load(unet_dir / "probs" / f"{n}.npy").astype(np.float32) for n in names}
    return {"dir": unet_dir, "scores": probs, "threshold": thr, "best_val_threshold": best_t,
            "checkpoint": src.get("checkpoint"),
            "source": {"main": src["results"][f"{split}|{thr:g}"],
                       "best_val": src["results"][f"{split}|best-val t={best_t:.2f}"]}}


# ---------------------------------------------------------------------------
# 计算与核对
# ---------------------------------------------------------------------------

def evaluate_row(samples: list, scores: dict, threshold) -> tuple:
    """threshold 为浮点数（固定阈值）或 "otsu"（逐图在 FOV 内算 Otsu）。返回 (summary, 二值图, 每图阈值)。"""
    acc = MetricAccumulator()
    binaries, thresholds = {}, {}
    for s in samples:
        score = scores[s["name"]]
        t = otsu_threshold(score, s["fov"]) if threshold == "otsu" else float(threshold)
        pred = binarize(score, t) * s["fov"]
        acc.update(pred, s["label"], s["fov"], score=score, name=s["name"])
        binaries[s["name"]], thresholds[s["name"]] = pred, t
    return acc.summary(), binaries, thresholds


def max_abs_diff(summary: dict, source: dict) -> float:
    """重算结果与来源 summary 的最大绝对差（mean / std / global 的 5 个指标 + 逐图 5 个指标；NaN 与 NaN 视为相等）。"""
    pairs = []
    for agg in ("mean", "std", "global"):
        pairs += [(summary[agg][k], source[agg][k]) for k in METRIC_NAMES]
    src_by_name = {m["name"]: m for m in source["per_image"]}
    for m in summary["per_image"]:
        pairs += [(m[k], src_by_name[m["name"]][k]) for k in METRIC_NAMES]
    diffs = [0.0 if (np.isnan(a) and np.isnan(b)) else abs(float(a) - float(b)) for a, b in pairs]
    return float(max(diffs))


def build_rows(samples: list, classical: dict, unet: dict) -> list:
    """组装所有行：primary（主表 3 行）+ secondary（Otsu 与 U-Net best-val 阈值）。"""
    specs = []
    for m, label in CLASSICAL_METHODS.items():
        t = float(classical["global_thresholds"][m])
        specs.append(dict(key=f"{m}_global", method=label, table="primary", threshold=t,
                          threshold_label=f"Global t={t:.2f} (searched on {classical['search_split']})",
                          scores=classical["scores"][m], source=classical["source"][m]["global"]))
    specs.append(dict(key="unet_main", method="U-Net", table="primary", threshold=unet["threshold"],
                      threshold_label=f"t={unet['threshold']:g}", scores=unet["scores"],
                      source=unet["source"]["main"]))
    for m, label in CLASSICAL_METHODS.items():
        specs.append(dict(key=f"{m}_otsu", method=label, table="secondary", threshold="otsu",
                          threshold_label="Otsu (per image)", scores=classical["scores"][m],
                          source=classical["source"][m]["otsu"]))
    bt = unet["best_val_threshold"]
    specs.append(dict(key="unet_best_val", method="U-Net", table="secondary", threshold=bt,
                      threshold_label=f"t={bt:.2f} (searched on val)", scores=unet["scores"],
                      source=unet["source"]["best_val"]))

    rows = []
    for spec in specs:
        summary, binaries, thresholds = evaluate_row(samples, spec["scores"], spec["threshold"])
        rows.append({**spec, "summary": summary, "binaries": binaries, "thresholds": thresholds,
                     "max_abs_diff": max_abs_diff(summary, spec["source"])})
    return rows


def verify(rows: list, tol: float = TOLERANCE) -> None:
    bad = [(r["key"], r["max_abs_diff"]) for r in rows if not r["max_abs_diff"] <= tol]
    if bad:
        raise AssertionError(f"重算指标与来源 run 的 metrics.json 不一致（容差 {tol:g}）：{bad}")


def _ratio(num: int, den: int) -> float:
    """安全除法：分母为 0 时返回 NaN（报告中显示为 n/a），而不是抛 ZeroDivisionError。"""
    return float(num) / den if den > 0 else float("nan")


def supplementary_analysis(rows: list, samples: list, thin_radius: float = 1.5, disc_radius: int = 45,
                           optic_disc: bool = True) -> dict:
    """补充分析（主表 3 行，像素跨图汇总）：细 / 粗血管召回率，以及（可选）视盘区域的假阳性。

    - 细血管：金标准中局部半径 ≤ thin_radius（宽约 ≤ 3 像素）的像素，其余血管为粗血管；
    - 视盘（optic_disc=True 时）：eval/visualize.py::find_optic_disc 的启发式中心、半径 disc_radius 的圆
      （与 FOV 取交）。这是眼底图特有的结构，OCTA 等数据应关闭（config ``compare.supplementary_optic_disc``）。
    这些区域是启发式定义，只用来解释主表差异，不作为正式指标。
    所有比值在分母为 0 时记为 NaN（例如全部是 1 像素宽的细血管时，粗血管像素数为 0）。
    """
    from eval.visualize import disc_mask, find_optic_disc, thin_vessel_mask

    regions = {}
    for s in samples:
        lab, fov = s["label"].astype(bool), s["fov"].astype(bool)
        thin = thin_vessel_mask(s["label"], thin_radius) & fov
        g = {"thin": thin, "thick": lab & fov & ~thin, "fov": fov, "label": lab}
        if optic_disc:
            g["disc"] = disc_mask(lab.shape, find_optic_disc(s["image"], s["fov"]), disc_radius) & fov
        regions[s["name"]] = g
    out = {"thin_radius": thin_radius, "optic_disc": bool(optic_disc),
           "disc_radius": disc_radius if optic_disc else None, "methods": {}}
    for r in rows:
        if r["table"] != "primary":
            continue
        c = {"thin_tp": 0, "thin_n": 0, "thick_tp": 0, "thick_n": 0, "fp": 0}
        if optic_disc:
            c["fp_disc"] = 0
        for name, g in regions.items():
            p = r["binaries"][name].astype(bool)
            c["thin_tp"] += int(np.count_nonzero(p & g["thin"]))
            c["thin_n"] += int(np.count_nonzero(g["thin"]))
            c["thick_tp"] += int(np.count_nonzero(p & g["thick"]))
            c["thick_n"] += int(np.count_nonzero(g["thick"]))
            fp = p & ~g["label"] & g["fov"]
            c["fp"] += int(np.count_nonzero(fp))
            if optic_disc:
                c["fp_disc"] += int(np.count_nonzero(fp & g["disc"]))
        c["sens_thin"] = _ratio(c["thin_tp"], c["thin_n"])
        c["sens_thick"] = _ratio(c["thick_tp"], c["thick_n"])
        if optic_disc:
            c["fp_disc_frac"] = _ratio(c["fp_disc"], c["fp"])
        out["methods"][r["method"]] = c
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

HEADER = ["Method", "Threshold", "Dice", "Sensitivity", "Specificity", "Accuracy", "AUC", "Global Dice", "Global AUC"]


def _rel(path) -> str:
    """仓库内路径显示为相对仓库根的形式（README / 报告里可直接引用）。"""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _num(x: float, digits: int = 4) -> str:
    """数值格式化；NaN（分母为 0 的比值）显示为 n/a。"""
    return "n/a" if x is None or np.isnan(x) else f"{x:.{digits}f}"


def _md_table(rows: list) -> list:
    out = ["| " + " | ".join(HEADER) + " |", "| " + " | ".join(["---"] * len(HEADER)) + " |"]
    for r in rows:
        s = r["summary"]
        cells = [r["method"], r["threshold_label"]] + [f"{s['mean'][k]:.4f} ± {s['std'][k]:.4f}" for k in METRIC_NAMES]
        cells += [f"{s['global']['dice']:.4f}", f"{s['global']['auc']:.4f}"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def write_markdown(rows: list, path, split: str, n: int, classical: dict, unet: dict, tol: float,
                   extra: dict | None = None) -> None:
    primary = [r for r in rows if r["table"] == "primary"]
    secondary = [r for r in rows if r["table"] == "secondary"]
    names = [m["name"] for m in primary[0]["summary"]["per_image"]]
    md = [f"# 方法对比（{split} split，{n} 张，FOV 内）", "",
          "指标为逐图 mean ± std（总体标准差）；Global 为所有图 FOV 内像素汇总后的整体指标。",
          f"主表阈值：经典方法用在 {classical['search_split']} split 上按逐图平均 Dice 搜索的全局阈值，"
          f"U-Net 用 {unet['threshold']:g}；均未看 {split}。AUC 与阈值无关。", "",
          "## 主表", ""] + _md_table(primary) + [
          "", "## 副表：Otsu（逐图）与 U-Net val 搜索阈值", ""] + _md_table(secondary) + [
          "", "## 逐图 Dice（主表阈值）", "",
          "| Method | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for r in primary:
        md.append(f"| {r['method']} | " + " | ".join(f"{m['dice']:.4f}" for m in r["summary"]["per_image"]) + " |")
    if extra:
        disc_on = extra.get("optic_disc", True)
        note = f"细血管 = 金标准局部半径 ≤ {extra['thin_radius']} 像素（宽约 ≤ 3 像素）"
        if disc_on:
            note += f"；视盘 = 启发式中心、半径 {extra['disc_radius']} 像素的圆（与 FOV 取交）"
        md += ["", "## 补充分析（启发式区域，像素跨图汇总，主表阈值）", "",
               note + "。仅用于解释差异，不是正式指标。", ""]
        if disc_on:
            md += ["| Method | Sens（细血管） | Sens（粗血管） | FP 总数 | 视盘圆内 FP | 视盘内 FP 占比 |",
                   "| --- | --- | --- | --- | --- | --- |"]
        else:
            md += ["| Method | Sens（细血管） | Sens（粗血管） | FP 总数 |", "| --- | --- | --- | --- |"]
        for m, c in extra["methods"].items():
            line = f"| {m} | {_num(c['sens_thin'])} | {_num(c['sens_thick'])} | {c['fp']} |"
            if disc_on:
                line += f" {c['fp_disc']} | {_num(c['fp_disc_frac'], 3)} |"
            md.append(line)
        n_thin = next(iter(extra["methods"].values()))
        md.append(f"\n细血管像素 {n_thin['thin_n']}，粗血管像素 {n_thin['thick_n']}。")
    md += ["", "## 来源与核对", "",
           f"- 经典方法：`{_rel(classical['dir'])}`（连续响应 maps/{split}/*.npy）",
           f"- U-Net：`{_rel(unet['dir'])}`（probs/*.npy；权重 `{unet['checkpoint']}`）",
           f"- 所有指标由 eval/metrics.py 从上述 .npy 与数据集 label/FOV 重新计算，"
           f"并与来源 metrics.json 逐项（mean/std/global/逐图）比对，容差 {tol:g}："]
    md += [f"  - {r['key']}：最大绝对差 {r['max_abs_diff']:.2e}" for r in rows]
    Path(path).write_text("\n".join(md) + "\n", encoding="utf-8")


def write_csv(rows: list, path) -> None:
    fields = ["table", "key", "method", "threshold"] + [f"{k}_{agg}" for k in METRIC_NAMES
                                                        for agg in ("mean", "std", "global")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            s = r["summary"]
            row = {"table": r["table"], "key": r["key"], "method": r["method"], "threshold": r["threshold_label"]}
            for k in METRIC_NAMES:
                for agg in ("mean", "std", "global"):
                    row[f"{k}_{agg}"] = f"{s[agg][k]:.6f}"
            w.writerow(row)


def write_json(rows: list, path, split: str, classical: dict, unet: dict, tol: float, cfg: dict,
               extra: dict | None = None, figures: dict | None = None) -> None:
    save_json({"split": split, "tolerance": tol, "supplementary": extra, "figures": figures, "config_experiment": cfg.get("experiment"),
               "classical_dir": classical["dir"], "unet_dir": unet["dir"], "unet_checkpoint": unet["checkpoint"],
               "rows": [{k: r[k] for k in ("key", "method", "table", "threshold_label", "thresholds",
                                            "summary", "max_abs_diff")} for r in rows]}, path)


# ---------------------------------------------------------------------------

def main(config_path: str, classical_dir=None, unet_dir=None, split=None, figures: bool = True) -> Path:
    cfg = load_config(config_path)
    ccfg = cfg.get("compare", {})
    classical_dir = classical_dir or ccfg.get("classical_dir")
    unet_dir = unet_dir or ccfg.get("unet_dir")
    split = split or ccfg.get("split", "test")
    tol = float(ccfg.get("tolerance", TOLERANCE))
    if not classical_dir or not unet_dir:
        raise ValueError("需要经典 run 目录与 U-Net run 目录（--classical-dir / --unet-dir 或 config 的 compare 小节）")

    samples = load_labels(cfg, split)
    names = [s["name"] for s in samples]
    classical = load_classical(classical_dir, split, names)
    unet = load_unet(unet_dir, split, names)
    rows = build_rows(samples, classical, unet)
    for r in rows:
        print(f"[evaluate] {r['key']:<22} Dice {r['summary']['mean']['dice']:.4f}  "
              f"AUC {r['summary']['mean']['auc']:.4f}  | 与来源最大差 {r['max_abs_diff']:.2e}")
    verify(rows, tol)

    # 补充分析是眼底图导向的启发式（细/粗血管分界、视盘），默认关闭：新数据集的 config 不写就不会跑；
    # 视盘部分再单独开关（OCTA 没有视盘）。configs/drive.yaml 两者都显式设为 true。
    extra = None
    if ccfg.get("supplementary", False):
        extra = supplementary_analysis(rows, samples, optic_disc=bool(ccfg.get("supplementary_optic_disc", False)))
    optic_disc_fig = bool(ccfg.get("supplementary_optic_disc", False))
    run_dir = make_run_dir(f"compare_{cfg['experiment']}")
    write_markdown(rows, run_dir / "metrics.md", split, len(samples), classical, unet, tol, extra)
    write_csv(rows, run_dir / "metrics.csv")
    figs = None
    if figures:
        from eval import visualize
        figs = visualize.make_all(rows, samples, run_dir, optic_disc=optic_disc_fig)
    write_json(rows, run_dir / "metrics.json", split, classical, unet, tol, cfg, extra, figs)
    print(f"[evaluate] 结果目录: {run_dir}")
    print((run_dir / "metrics.md").read_text(encoding="utf-8"))
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="经典方法与 U-Net 的对比表与图")
    parser.add_argument("--config", default="configs/drive.yaml")
    parser.add_argument("--classical-dir", default=None, help="classical.run_baseline 的结果目录")
    parser.add_argument("--unet-dir", default=None, help="scripts.test 的结果目录")
    parser.add_argument("--split", default=None, help="默认取 config compare.split（test）")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    main(args.config, args.classical_dir, args.unet_dir, args.split, not args.no_figures)
