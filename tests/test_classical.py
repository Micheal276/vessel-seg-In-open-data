"""经典方法（预处理 + Frangi + 匹配滤波 + 阈值搜索）的单元测试。

设计理由：经典方法最常见的错误是"极性反了"（检测亮线而不是暗线）、模板不是零均值、
FOV 外的值泄漏进归一化。这些错误不会报错，只会让指标悄悄变差，所以用合成图把它们固化下来：
一条已知位置的暗线，两种滤波在线上的响应都必须明显高于背景。
"""

import numpy as np
import pytest

from classical.filters import (enhance, frangi_response, matched_filter_bank, matched_filter_kernel,
                               matched_filter_response, normalize_in_fov)
from classical.preprocess import apply_clahe, denoise, fill_outside_fov, preprocess, to_single_channel
from classical.run_baseline import otsu_threshold, search_global_threshold


def _dark_line_image(size=64, vertical=True, half_width=2):
    """亮背景（0.7）+ 一条高斯横截面的暗线（深度 0.4、σ=1.5）+ 少量噪声，返回 (image, line_mask)。

    用高斯横截面而不是理想阶跃，更接近真实血管；加 σ=0.01 的噪声是因为完美直线上沿线方向的
    Hessian 特征值 λ1 恰好为 0，skimage 的 Frangi 靠"λ1/λ2 很大"来压掉错误极性的响应，
    λ1=0 时这个机制失效（真实图像不会出现精确的 0）。
    line_mask 为距中心线 ≤ half_width 像素的区域。
    """
    c = size // 2
    d = np.abs(np.arange(size) - c).astype(np.float32)
    profile = 0.7 - 0.4 * np.exp(-(d ** 2) / (2 * 1.5 ** 2))
    img = np.tile(profile, (size, 1)) if vertical else np.tile(profile[:, None], (1, size))
    img = img + np.random.default_rng(0).normal(0, 0.01, img.shape)
    mask1d = d <= half_width
    mask = np.tile(mask1d, (size, 1)) if vertical else np.tile(mask1d[:, None], (1, size))
    return img.astype(np.float32), mask


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------

def test_to_single_channel():
    rgb = np.random.default_rng(0).random((8, 9, 3)).astype(np.float32)
    g = to_single_channel(rgb, green_channel=True)
    assert g.shape == (8, 9) and g.dtype == np.float32
    np.testing.assert_array_equal(g, rgb[:, :, 1])
    gray = np.random.default_rng(1).random((8, 9)).astype(np.float32)
    np.testing.assert_array_equal(to_single_channel(gray, green_channel=True), gray)  # 单通道原样返回（OCTA）
    assert to_single_channel(rgb, green_channel=False).shape == (8, 9)


def test_preprocess_dtype_and_range():
    rng = np.random.default_rng(0)
    rgb = rng.random((60, 50, 3)).astype(np.float32)
    fov = np.zeros((60, 50), np.uint8)
    fov[5:55, 5:45] = 1
    cfg = {"green_channel": True, "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid": [4, 4]},
           "fill_outside_fov": {"enabled": True, "iterations": 10, "erode_px": 2},
           "denoise": {"method": "gaussian", "sigma": 1.0}}
    stages = preprocess(rgb, fov, cfg)
    for key in ("gray", "clahe", "denoised", "output"):
        x = stages[key]
        assert x.shape == (60, 50) and x.dtype == np.float32, key
        assert 0.0 <= x.min() and x.max() <= 1.0, key

    # 关掉绿通道开关、输入单通道（OCTA 场景）也能跑
    octa_cfg = dict(cfg, green_channel=False, clahe={"enabled": False})
    assert preprocess(rgb[:, :, 0], fov, octa_cfg)["output"].shape == (60, 50)


def test_clahe_and_denoise_float32():
    x = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert apply_clahe(x, 2.0, (4, 4)).dtype == np.float32
    for method in ("gaussian", "median", "none"):
        y = denoise(x, method, sigma=1.0, ksize=3)
        assert y.dtype == np.float32 and y.shape == x.shape
    with pytest.raises(ValueError):
        denoise(x, "bilateral")


def test_fill_outside_fov_keeps_inside_and_removes_step():
    img = np.zeros((40, 40), np.float32)
    fov = np.zeros((40, 40), np.uint8)
    fov[10:30, 10:30] = 1
    img[fov == 1] = 0.6
    out = fill_outside_fov(img, fov, iterations=5)
    np.testing.assert_array_equal(out[fov == 1], img[fov == 1])   # FOV 内不变
    assert np.allclose(out[5:35, 5:35], 0.6)                      # 外侧 5 像素环带被填成内侧灰度


# ---------------------------------------------------------------------------
# 匹配滤波核
# ---------------------------------------------------------------------------

def test_matched_kernel_zero_mean_and_12_orientations():
    bank = matched_filter_bank(sigma=1.5, length=9, n_angles=12)
    assert len(bank) == 12
    for k in bank:
        assert k.dtype == np.float32
        assert abs(float(k.sum())) < 1e-4          # 支撑域内零均值 => 总和为 0
        assert k.min() < 0 < k.max()                # 中心负、两侧正
    # 方向两两不同，且 0° 与 90° 互为转置（对称的正方形网格）
    assert not np.allclose(bank[0], bank[1])
    np.testing.assert_allclose(matched_filter_kernel(1.5, 9, 90.0), bank[0].T, atol=1e-5)


def test_matched_kernel_center_is_minimum_along_cross_section():
    k = matched_filter_kernel(sigma=2.0, length=9, angle_deg=0.0)   # 0°：血管沿竖直方向
    c = k.shape[0] // 2
    row = k[c]
    assert np.argmin(row) == c                     # 横截面中心最"负"


# ---------------------------------------------------------------------------
# 两种滤波在合成暗线上的响应
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vertical", [True, False])
def test_filters_respond_on_dark_line(vertical):
    img, line = _dark_line_image(vertical=vertical)
    inner = np.zeros_like(line)
    inner[8:-8, 8:-8] = True                       # 避开图像边界
    background = inner & ~line
    # 背景取离线较远的像素（距离 > 6），避开两侧的过渡带
    c = img.shape[0] // 2
    far = np.ones_like(line)
    if vertical:
        far[:, c - 6:c + 7] = False
    else:
        far[c - 6:c + 7, :] = False
    background &= far

    for resp in (frangi_response(img, [1.0, 2.0], beta=1.0, gamma=0.006),
                 matched_filter_response(img, sigma=1.5, length=9)):
        assert resp.dtype == np.float32
        assert resp[line & inner].mean() > resp[background].mean() + 1e-3


def test_frangi_polarity_matters():
    """black_ridges=False 时（检测亮线），暗线上的响应应明显低于 black_ridges=True。"""
    img, line = _dark_line_image(half_width=1)
    dark = frangi_response(img, [1.0, 2.0], gamma=0.006, black_ridges=True)
    bright = frangi_response(img, [1.0, 2.0], gamma=0.006, black_ridges=False)
    assert dark[line].mean() > 2 * bright[line].mean()
    assert dark[line].mean() > 5 * dark[~line].mean()


# ---------------------------------------------------------------------------
# 归一化与阈值
# ---------------------------------------------------------------------------

def test_normalize_in_fov():
    resp = np.random.default_rng(0).normal(size=(30, 30)).astype(np.float32)
    fov = np.zeros((30, 30), np.uint8)
    fov[5:25, 5:25] = 1
    resp[0, 0] = 100.0                             # FOV 外极大值不应影响归一化
    out = normalize_in_fov(resp, fov)
    assert out.dtype == np.float32
    assert np.isclose(out[fov == 1].min(), 0.0) and np.isclose(out[fov == 1].max(), 1.0)
    assert np.all(out[fov == 0] == 0.0)
    out_e = normalize_in_fov(resp, fov, erode_px=2)
    assert 0.0 <= out_e.min() and out_e.max() <= 1.0


def test_enhance_and_thresholds_on_synthetic_line():
    img, line = _dark_line_image()
    fov = np.ones_like(line, dtype=np.uint8)
    label = line.astype(np.uint8)
    cfg = {"frangi": {"sigmas": [1.0, 2.0], "beta": 1.0, "gamma": 0.006},
           "matched_filter": {"sigma": 1.5, "length": 9, "n_angles": 12},
           "normalize": {"fov_erode_px": 0}}
    for method in ("frangi", "matched_filter"):
        resp = enhance(img, fov, method, cfg)
        assert 0.0 <= resp.min() and resp.max() <= 1.0
        t = otsu_threshold(resp, fov)
        assert 0.0 < t < 1.0
        rec = [{"resp": {method: resp}, "label": label, "fov": fov}]
        best_t, curve = search_global_threshold(rec, method, [0.05, 0.95, 0.05])
        assert len(curve) == 19
        assert max(d for _, d in curve) == dict(curve)[best_t]
        assert dict(curve)[best_t] > 0.8           # 合成单线上，最佳阈值应能较好地分出这条线


# ---------------------------------------------------------------------------
# run_baseline --splits 与 tune_baseline
# ---------------------------------------------------------------------------

def test_parse_splits():
    from classical.run_baseline import parse_splits
    assert parse_splits("train") == ("train",)
    assert parse_splits("test, train") == ("train", "test")   # 固定顺序
    with pytest.raises(ValueError):
        parse_splits("train,foo")
    with pytest.raises(ValueError):
        parse_splits("")


def test_tune_refuses_non_train_split():
    """tune_baseline 的数据入口对 val/test 直接抛异常（在读取任何文件之前）。"""
    from classical.tune_baseline import _load_split
    for split in ("val", "test"):
        with pytest.raises(RuntimeError):
            _load_split({}, split)


def test_fast_dice_curve_matches_search_global_threshold():
    """tune_baseline 的排序 + 二分实现必须与 run_baseline 的逐阈值实现给出相同的最佳阈值和 Dice。"""
    from classical.tune_baseline import dice_curve, threshold_candidates
    rng = np.random.default_rng(0)
    grid = [0.01, 0.99, 0.01]
    cands = threshold_candidates(grid)
    records = []
    for _ in range(3):
        label = (rng.random((40, 50)) < 0.15).astype(np.uint8)
        resp = np.clip(0.35 * label + rng.random((40, 50)) * 0.7, 0, 1).astype(np.float32)
        resp[0, :5] = 0.5   # 恰好落在候选阈值上的值：检验"严格大于"语义
        fov = np.ones_like(label)
        fov[:, :4] = 0
        records.append({"label": label, "fov": fov, "resp": {"m": resp}})
    t_ref, curve = search_global_threshold(records, "m", grid)
    mean_curve = np.mean([dice_curve(r["resp"]["m"], r["label"], r["fov"], cands) for r in records], axis=0)
    np.testing.assert_allclose(mean_curve, [d for _, d in curve], rtol=0, atol=1e-12)
    assert float(cands[int(np.argmax(mean_curve))]) == t_ref
