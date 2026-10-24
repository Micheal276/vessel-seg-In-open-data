"""任务 1.5 对比报告（eval/evaluate.py、eval/visualize.py）的单元测试。

设计理由：对比表的价值在于"三种方法口径一致"，所以重点测重算与核对逻辑（阈值规则、与来源的差值、
不一致时报错），以及错误图配色、细血管 / 视盘等辅助区域在可预知的合成图上行为正确。
"""

import numpy as np
import pytest

from eval.evaluate import build_rows, evaluate_row, max_abs_diff, verify
from eval.visualize import (error_map, find_optic_disc, make_all, thin_vessel_mask, vessel_radius_map,
                            OUTSIDE_FOV)


def _samples(n=2, h=48, w=48, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        label = np.zeros((h, w), np.uint8)
        label[10:12, 5:40] = 1
        label[20:30, 15:18] = 1
        fov = np.ones((h, w), np.uint8)
        fov[:, :3] = 0
        out.append({"name": str(i), "label": label, "fov": fov,
                    "image": rng.random((h, w, 3)).astype(np.float32)})
    return out


def _scores(samples, noise, seed):
    rng = np.random.default_rng(seed)
    return {s["name"]: np.clip(s["label"] * 0.7 + rng.random(s["label"].shape) * noise, 0, 1).astype(np.float32)
            for s in samples}


def test_error_map_colors():
    pred = np.array([[1, 1, 0, 0]], np.uint8)
    label = np.array([[1, 0, 1, 0]], np.uint8)
    fov = np.array([[1, 1, 1, 0]], np.uint8)
    em = error_map(pred, label, fov)
    assert tuple(em[0, 0]) == (1, 1, 1)              # TP 白
    assert tuple(em[0, 1]) == (1, 0, 0)              # FP 红
    assert em[0, 2, 2] == 1 and em[0, 2, 0] == 0     # FN 蓝
    assert np.allclose(em[0, 3], OUTSIDE_FOV)        # FOV 外暗灰


def test_evaluate_row_fixed_and_otsu():
    samples = _samples()
    scores = _scores(samples, 0.2, 1)
    s_fixed, bins, thr = evaluate_row(samples, scores, 0.5)
    assert all(t == 0.5 for t in thr.values())
    assert s_fixed["mean"]["dice"] == pytest.approx(1.0)       # 血管 ≥0.7、背景 ≤0.2
    assert all(b[:, :3].sum() == 0 for b in bins.values())     # FOV 外置 0
    s_otsu, _, thr_o = evaluate_row(samples, scores, "otsu")
    from classical.run_baseline import otsu_threshold
    assert all(thr_o[x["name"]] == otsu_threshold(scores[x["name"]], x["fov"]) for x in samples)
    assert s_otsu["mean"]["dice"] > 0.85


def test_build_rows_and_verify_against_source():
    samples = _samples()
    names = [s["name"] for s in samples]
    cl_scores = {"frangi": _scores(samples, 0.8, 2), "matched_filter": _scores(samples, 0.9, 3)}
    un_scores = _scores(samples, 1.0, 4)     # 血管/背景分数有重叠，阈值 0.5 与 0.6 结果不同
    src_cl = {m: {"global": evaluate_row(samples, cl_scores[m], 0.4)[0],
                  "otsu": evaluate_row(samples, cl_scores[m], "otsu")[0]} for m in cl_scores}
    classical = {"scores": cl_scores, "global_thresholds": {"frangi": 0.4, "matched_filter": 0.4},
                 "search_split": "train", "source": src_cl}
    unet = {"scores": un_scores, "threshold": 0.5, "best_val_threshold": 0.3,
            "source": {"main": evaluate_row(samples, un_scores, 0.5)[0],
                       "best_val": evaluate_row(samples, un_scores, 0.3)[0]}}
    rows = build_rows(samples, classical, unet)
    assert [r["key"] for r in rows if r["table"] == "primary"] == ["frangi_global", "matched_filter_global", "unet_main"]
    assert len(rows) == 6
    verify(rows)                                               # 全部一致，不报错
    assert all(r["max_abs_diff"] == 0.0 for r in rows)
    assert [m["name"] for m in rows[0]["summary"]["per_image"]] == names

    # 来源用了另一个阈值 → 核对失败
    unet["source"]["main"] = evaluate_row(samples, un_scores, 0.6)[0]
    rows_bad = build_rows(samples, classical, unet)
    with pytest.raises(AssertionError):
        verify(rows_bad)


def test_max_abs_diff_nan_equal():
    s = {"mean": {k: float("nan") for k in ("dice", "sensitivity", "specificity", "accuracy", "auc")}}
    s["std"], s["global"] = dict(s["mean"]), dict(s["mean"])
    s["per_image"] = [{"name": "a", **s["mean"]}]
    assert max_abs_diff(s, s) == 0.0


def test_vessel_radius_thin_vs_thick():
    label = np.zeros((60, 60), np.uint8)
    label[10, 5:55] = 1           # 宽 1 像素
    label[30:39, 5:55] = 1        # 宽 9 像素
    r = vessel_radius_map(label)
    assert r[10, 30] <= 1.0
    assert r[34, 30] >= 4.0
    thin = thin_vessel_mask(label)
    assert thin[10, 5:55].all()
    assert not thin[30:39, 20:40].any()


def test_find_optic_disc_on_bright_blob():
    img = np.full((120, 120, 3), 0.3, np.float32)
    rr, cc = np.ogrid[:120, :120]
    img[(rr - 50) ** 2 + (cc - 80) ** 2 <= 15 ** 2] = 0.95
    r, c = find_optic_disc(img, np.ones((120, 120), np.uint8), sigma=5)
    assert abs(r - 50) <= 2 and abs(c - 80) <= 2


def test_make_all_writes_figures(tmp_path):
    samples = _samples(h=160, w=160)
    for s in samples:   # 加个"视盘"让裁剪逻辑有东西可找
        s["image"][60:80, 100:120] = 1.0
    scores = _scores(samples, 0.6, 5)
    rows = []
    for method in ("Frangi", "Matched filter", "U-Net"):
        summary, bins, _ = evaluate_row(samples, scores, 0.5)
        rows.append({"method": method, "table": "primary", "threshold_label": "t=0.5 (test)",
                     "scores": scores, "binaries": bins, "summary": summary})
    out = make_all(rows, samples, tmp_path)
    for p in [out["roc"], out["dice_boxplot"], *out["examples"].values(), *out["zoom"].values()]:
        assert p.exists() and p.stat().st_size > 0
