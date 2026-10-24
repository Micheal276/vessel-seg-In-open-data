"""经典血管增强滤波器：Frangi（Hessian 管状度）与高斯匹配滤波，以及 FOV 内归一化。

设计理由（面试讲解要点）
------------------------
1. Frangi 滤波（skimage.filters.frangi）。
   在多个尺度 σ 上计算图像的 Hessian 矩阵（二阶导，先用高斯 σ 平滑），其两个特征值
   |λ1| ≤ |λ2| 描述局部二阶结构：
     - 管状（血管）：沿血管方向曲率小 → |λ1| ≈ 0；横截面方向曲率大 → |λ2| 大；
     - 斑点：|λ1| ≈ |λ2| 都大；背景：两者都小。
   管状度 = exp(-R_B²/2β²) · (1 - exp(-S²/2γ²))，其中 R_B = |λ1|/|λ2|（区分线和斑点），
   S = sqrt(λ1²+λ2²)（区分结构和平坦背景）。各尺度取最大值，σ 对应不同血管粗细。
   极性：绿通道里血管比背景**暗**，暗血管横截面是"谷"，λ2 > 0，所以必须 ``black_ridges=True``；
   极性选反了响应会落在血管之间的亮背景上，Dice 会极低——这是计划书提醒"检查极性"的原因。

2. 匹配滤波（Chaudhuri et al. 1989，手写）。
   观察：血管横截面灰度近似一条倒置的高斯曲线，且在一小段长度 L 内近似直线。
   于是构造模板：横向（垂直于血管）是 -exp(-x²/2σ²)，纵向（沿血管）在长度 L 内保持不变，
   只在 |x| ≤ 3σ、|y| ≤ L/2 的邻域内有定义。
   - **减去均值（零均值）**：让模板在平坦背景上响应为 0，只对"中间暗、两侧亮"的结构有正响应，
     同时对整体亮度偏移不敏感。
   - **12 个方向**：血管方向未知，把模板每 15° 旋转一次（0°…165°，180° 与 0° 等价），
     对每个方向卷积后逐像素取最大响应，相当于"找最匹配的方向"。
   用 ``cv2.filter2D`` 做卷积（实际是相关；模板关于中心点对称，相关 = 卷积）。

3. FOV 内 min-max 归一化到 [0,1]。
   AUC 只关心排序，任何单调变换都不改变 AUC；但阈值化（Otsu、全局阈值）需要不同图像的响应在
   同一尺度上，所以每张图在 FOV 内做 min-max 归一化。FOV 外置 0（不参与任何指标）。
   可选：归一化时用腐蚀几个像素的 FOV 统计 min/max，避免 FOV 边缘残留的极端值把整张图的动态范围
   压扁；腐蚀只影响 min/max 的取值范围，结果会裁剪到 [0,1]，指标仍在原始 FOV 上计算。
"""

import numpy as np
import cv2
from skimage.filters import frangi


# ---------------------------------------------------------------------------
# Frangi
# ---------------------------------------------------------------------------

def frangi_response(gray: np.ndarray, sigmas, alpha: float = 0.5, beta: float = 0.5,
                    gamma=None, black_ridges: bool = True) -> np.ndarray:
    """多尺度 Frangi 管状度响应（非负，越大越像血管），返回 float32。

    sigmas : 尺度列表（像素），对应要检测的血管半径范围
    alpha, beta, gamma : skimage 的 Frangi 参数（2D 下 alpha 不起作用）；gamma=None 时 skimage 自动取
    black_ridges : True 检测暗血管（眼底绿通道），False 检测亮血管（如 OCTA）
    """
    resp = frangi(np.asarray(gray, dtype=np.float32), sigmas=list(sigmas), alpha=alpha, beta=beta,
                  gamma=gamma, black_ridges=black_ridges, mode="reflect")
    return resp.astype(np.float32)


# ---------------------------------------------------------------------------
# 匹配滤波
# ---------------------------------------------------------------------------

def matched_filter_kernel(sigma: float, length: float, angle_deg: float) -> np.ndarray:
    """构造一个方向的零均值高斯匹配核（float32）。

    以核中心为原点，(u, v) 为旋转后的坐标：u 垂直于血管（横截面），v 沿血管。
    角度 0° 时血管沿竖直方向（图像行方向）。
    支撑域 N = {|u| ≤ 3σ, |v| ≤ L/2}；N 内取 -exp(-u²/2σ²) 再减去 N 内均值，N 外为 0。
    """
    half = int(np.ceil(np.hypot(3.0 * sigma, length / 2.0)))  # 能装下任意方向旋转后的支撑域
    coords = np.arange(-half, half + 1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)                            # x: 列方向, y: 行方向
    theta = np.deg2rad(np.float32(angle_deg))
    u = x * np.cos(theta) + y * np.sin(theta)                     # 横截面坐标
    v = -x * np.sin(theta) + y * np.cos(theta)                    # 沿血管坐标

    support = (np.abs(u) <= 3.0 * sigma) & (np.abs(v) <= length / 2.0)
    kernel = np.zeros_like(x, dtype=np.float32)
    kernel[support] = -np.exp(-(u[support] ** 2) / (2.0 * sigma ** 2))
    kernel[support] -= kernel[support].mean()                     # 支撑域内零均值
    return kernel.astype(np.float32)


def matched_filter_bank(sigma: float, length: float, n_angles: int = 12) -> list:
    """n_angles 个方向的匹配核，角度均匀取在 [0°, 180°)，12 个方向即每 15° 一个。"""
    return [matched_filter_kernel(sigma, length, 180.0 * k / n_angles) for k in range(n_angles)]


def matched_filter_response(gray: np.ndarray, sigma: float, length: float, n_angles: int = 12) -> np.ndarray:
    """对所有方向卷积，逐像素取最大响应，返回 float32（暗血管处为正）。"""
    gray = np.asarray(gray, dtype=np.float32)
    responses = [cv2.filter2D(gray, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
                 for k in matched_filter_bank(sigma, length, n_angles)]
    return np.max(np.stack(responses, axis=0), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def normalize_in_fov(resp: np.ndarray, fov: np.ndarray, erode_px: int = 0) -> np.ndarray:
    """在 FOV 内把响应 min-max 归一化到 [0,1]，FOV 外置 0，返回 float32。

    erode_px > 0 时，min/max 只在腐蚀 erode_px 像素后的 FOV 内统计（避开边缘残留伪影），
    归一化后裁剪到 [0,1]；输出仍覆盖完整的原始 FOV。
    """
    resp = np.asarray(resp, dtype=np.float32)
    fov = fov.astype(np.uint8)
    stat_mask = fov
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        stat_mask = cv2.erode(fov, k)
    vals = resp[stat_mask.astype(bool)]
    if vals.size == 0:
        raise ValueError("FOV（腐蚀后）为空，无法归一化")
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        out = np.zeros_like(resp)
    else:
        out = np.clip((resp - lo) / (hi - lo), 0.0, 1.0)
    out[~fov.astype(bool)] = 0.0
    return out.astype(np.float32)


def enhance(gray: np.ndarray, fov: np.ndarray, method: str, cfg: dict) -> np.ndarray:
    """按 config 计算某种滤波器的归一化响应。method: "frangi" / "matched_filter"。

    cfg 是 config 的 ``classical`` 小节，包含 frangi / matched_filter / normalize 子项。
    """
    if method == "frangi":
        c = cfg["frangi"]
        resp = frangi_response(gray, c["sigmas"], alpha=c.get("alpha", 0.5), beta=c.get("beta", 0.5),
                               gamma=c.get("gamma"), black_ridges=c.get("black_ridges", True))
    elif method == "matched_filter":
        c = cfg["matched_filter"]
        # 模板假设血管比背景暗；亮血管（如 OCTA）先把图像取反，保持"血管处响应为正"
        src = gray if c.get("dark_vessels", True) else -np.asarray(gray, dtype=np.float32)
        resp = matched_filter_response(src, c["sigma"], c["length"], c.get("n_angles", 12))
    else:
        raise ValueError(f"未知滤波方法 {method!r}")
    return normalize_in_fov(resp, fov, cfg.get("normalize", {}).get("fov_erode_px", 0))
