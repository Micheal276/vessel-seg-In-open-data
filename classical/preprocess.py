"""经典方法的预处理：通道选择 → CLAHE → 去噪（外加可选的"FOV 外填充"）。

设计理由（面试讲解要点）
------------------------
1. 为什么用绿通道。
   眼底照片里血红蛋白对绿光（~540 nm）吸收最强，血管在绿通道里最暗、与背景对比度最高；
   红通道接近饱和（眼底整体偏红），蓝通道又暗又噪。所以几乎所有经典视网膜血管方法都只用绿通道。
   OCTA 本身就是单通道灰度图，没有"选通道"这一步，因此用 config 开关 ``green_channel`` 控制：
   阶段二只需在 yaml 里把它关掉，代码不变。

2. CLAHE（限制对比度的自适应直方图均衡）在做什么。
   把图像切成 tile_grid 个小块，每块单独做直方图均衡，再双线性插值拼接，
   从而校正眼底图"中间亮、边缘暗"的照明不均，增强局部血管对比度。
   "限制对比度"（clip_limit）是把直方图中超过上限的部分截掉、平均分给其它灰度级，
   防止在平坦区域把噪声也放大。clip_limit 越大增强越猛、噪声也越明显。
   OpenCV 的 CLAHE 只接受 uint8 / uint16，这里量化到 uint8 处理后再转回 float32 [0,1]——
   眼底原图本来就是 8 位，量化不损失信息。

3. 去噪：高斯或中值，类型和参数都在 config 里。
   Frangi 和匹配滤波都是二阶/模板类算子，对高频噪声敏感，先轻度平滑能减少背景里的假响应。
   高斯平滑简单、线性、可解释；中值对椒盐/散斑类噪声更稳、且保边。DRIVE 噪声不重，默认用小 σ 高斯；
   OCTA 散斑噪声更强，阶段二可在 config 里换成中值滤波，代码不变。

4. FOV 外填充（可选，config 开关 ``fill_outside_fov``）。
   眼底图 FOV 外是一片纯黑。FOV 边界是一个"亮→暗"的强阶跃，Frangi（Hessian）和匹配滤波（模板）
   都会在这条边界上产生一圈很强的假响应，进而拉坏 min-max 归一化和阈值。
   做法（Soares et al. 2006 的思路）：从 FOV 往外逐圈"长"像素，每个新像素取它 3×3 邻域内
   已知像素的平均值，迭代若干圈。这样 FOV 边界外侧变成了与内侧相近的灰度，阶跃消失。
   DRIVE 的 FOV 掩膜边缘内侧还有一圈 1–3 像素的暗边（掩膜比实际亮区略大），它同样会被当成"暗血管"，
   所以可选地先把 FOV 腐蚀几个像素再填充，让这圈暗边也被内侧灰度替换。
   只用于计算滤波响应；所有指标仍然只在原始 FOV 内计算。

5. 纯函数 + float32。
   每个函数都是 numpy 进、numpy 出，不读写文件、不依赖全局状态，便于单元测试和在 U-Net 输入里复用
   （任务 1.4 的输入同样是"绿通道 + CLAHE"）。全项目统一 float32。
"""

import cv2
import numpy as np


def to_single_channel(image: np.ndarray, green_channel: bool = True) -> np.ndarray:
    """把输入图像变成 H×W float32 单通道。

    - 输入已经是 H×W（如 OCTA）：原样返回（转 float32）。
    - 输入 H×W×3 且 green_channel=True：取绿通道（RGB 顺序下索引 1）。
    - 输入 H×W×3 且 green_channel=False：按 ITU-R BT.601 权重转灰度（彩色图又不想用绿通道时的退路）。
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image 应为 H×W 或 H×W×3，实际形状 {image.shape}")
    if green_channel:
        return np.ascontiguousarray(image[:, :, 1])
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (image @ weights).astype(np.float32)


def apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid) -> np.ndarray:
    """对 [0,1] float32 单通道图做 CLAHE，返回 [0,1] float32。"""
    if gray.ndim != 2:
        raise ValueError(f"CLAHE 需要单通道 H×W，实际形状 {gray.shape}")
    u8 = np.clip(np.rint(gray * 255.0), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(t) for t in tile_grid))
    return clahe.apply(u8).astype(np.float32) / np.float32(255.0)


def denoise(gray: np.ndarray, method: str = "gaussian", sigma: float = 1.0, ksize: int = 3) -> np.ndarray:
    """去噪。method: "gaussian"（用 sigma，核大小由 OpenCV 按 σ 自动取）/ "median"（用 ksize）/ "none"。"""
    gray = np.asarray(gray, dtype=np.float32)
    if method == "none":
        return gray
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma)).astype(np.float32)
    if method == "median":
        # OpenCV 的 float32 中值滤波只支持 ksize 3 或 5；更大的核需要 uint8，这里不做隐式量化
        if ksize not in (3, 5):
            raise ValueError(f"float32 中值滤波只支持 ksize=3 或 5，实际 {ksize}")
        return cv2.medianBlur(gray, int(ksize)).astype(np.float32)
    raise ValueError(f"未知去噪方法 {method!r}，可选 gaussian / median / none")


def fill_outside_fov(gray: np.ndarray, fov: np.ndarray, iterations: int = 20, erode_px: int = 0) -> np.ndarray:
    """把 FOV 外的像素用"逐圈向外取邻域均值"的方式填充，消除 FOV 边界的强阶跃。

    每次迭代：对当前已知区域做 3×3 均值（只平均已知像素），把紧贴已知区域的一圈未知像素填上，
    已知区域向外扩一圈。迭代 iterations 次后，已知区域外 iterations 像素宽的环带被填充；
    只要大于滤波核半径即可（更外面的像素不会影响 FOV 内的响应）。

    erode_px > 0 时，先把 FOV 腐蚀 erode_px 像素作为"已知区域"，即 FOV 最外圈的 erode_px 个像素
    也被重新填充。原因：DRIVE 的 FOV 掩膜比实际亮区略大，掩膜边缘内侧有 1–3 像素宽的暗边，
    它与内侧亮区形成一条"暗带"，会被暗血管检测器当成血管，在二值图上出现一整圈假阳性。
    已知区域内的像素保持不变。
    """
    gray = np.asarray(gray, dtype=np.float32)
    fov_u8 = fov.astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        fov_u8 = cv2.erode(fov_u8, k)
    known = fov_u8.astype(bool)
    out = np.where(known, gray, np.float32(0.0)).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    for _ in range(int(iterations)):
        num = cv2.filter2D(out * known, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        den = cv2.filter2D(known.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        ring = (~known) & (den > 0)
        if not ring.any():
            break
        out[ring] = num[ring] / den[ring]
        known |= ring
    return out


def preprocess(image: np.ndarray, fov: np.ndarray, cfg: dict) -> dict:
    """按 config 的 ``preprocess`` 小节依次执行各步，返回每一步的中间结果（便于可视化）。

    cfg 形如::

        green_channel: true
        clahe: {enabled: true, clip_limit: 2.0, tile_grid: [8, 8]}
        denoise: {method: gaussian, sigma: 1.0, ksize: 3}
        fill_outside_fov: {enabled: true, iterations: 20, erode_px: 3}

    执行顺序：通道选择 → CLAHE → （可选）FOV 外填充 → 去噪。
    返回 {"gray", "clahe", "denoised", "output"}，均为 H×W float32；"output" 是送入滤波器的最终图
    （目前与 "denoised" 相同，单独留一个键是为了以后插入新步骤时调用方不用改）。
    """
    stages = {}
    x = to_single_channel(image, cfg.get("green_channel", True))
    stages["gray"] = x

    clahe_cfg = cfg.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        x = apply_clahe(x, clahe_cfg["clip_limit"], clahe_cfg["tile_grid"])
    stages["clahe"] = x

    # 先填充 FOV 外再去噪：否则高斯/中值会把 FOV 外的纯黑"抹"进 FOV 边缘一两个像素
    fill_cfg = cfg.get("fill_outside_fov", {})
    if fill_cfg.get("enabled", False):
        x = fill_outside_fov(x, fov, fill_cfg.get("iterations", 20), fill_cfg.get("erode_px", 0))

    dn = cfg.get("denoise", {"method": "none"})
    x = denoise(x, dn.get("method", "none"), sigma=dn.get("sigma", 1.0), ksize=dn.get("ksize", 3))
    stages["denoised"] = x
    stages["output"] = x
    return stages
