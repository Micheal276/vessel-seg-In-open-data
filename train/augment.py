"""数据增强：随机翻转、90° 旋转、小角度旋转、亮度/对比度扰动（手写，基于 numpy + OpenCV）。

设计理由（面试讲解要点）
------------------------
1. 为什么要增强。
   DRIVE 训练图只有 12 张，U-Net 很容易记住训练图。血管分割有天然的对称性：
   血管在图里朝哪个方向走、整张图左右翻转，都不改变"它是不是血管"。
   用这些变换造出"新样本"，相当于告诉网络"方向和亮度不重要，形状才重要"。

2. 几何变换必须对 image / label / fov 同步。
   同一组随机参数（是否翻转、转几个 90°、转多少度）作用在三者上，否则标签和图像错位，
   网络学到的是噪声。实现上先抽一次参数，再分别作用到三个数组上（见 ``augment_sample``）。
   - 翻转与 90° 旋转是像素的精确重排，不插值，三者天然对齐、label 保持二值。
   - 小角度旋转需要插值：image 用双线性（平滑），label / fov 用 **最近邻**——
     双线性会在血管边缘产生 0.5 之类的中间值，破坏 {0,1} 二值约定；最近邻只会"搬运"已有的值。
   - 旋转后角落会露出原图外的区域，三者统一用镜像（BORDER_REFLECT_101）填充，
     保证填进来的图像内容和标签仍然一一对应（若 image 镜像而 label 补 0，就会出现"有血管图像、无血管标签"的错误样本）。

3. 亮度/对比度扰动只作用于 image。
   模拟不同相机、曝光造成的整体明暗差异：x' = (x - mean)·c + mean + b，再裁剪到 [0,1]。
   光度变换不改变血管位置，所以 label / fov 不变。

4. 所有随机数来自调用者传入的 ``np.random.Generator``。
   不用全局随机状态，便于单元测试复现、也便于以后多进程 DataLoader 各自持有独立生成器。

5. 与数据集无关：只认 H×W（或 H×W×C）的 numpy 数组，参数全部来自 config 的 ``train.augment`` 小节。
"""

import cv2
import numpy as np

DEFAULT_AUG = {
    "enabled": True,
    "hflip": 0.5,              # 水平翻转概率
    "vflip": 0.5,              # 垂直翻转概率
    "rot90": True,             # 随机旋转 0/90/180/270°
    "rotate_prob": 0.5,        # 小角度旋转概率
    "rotate_deg": 15.0,        # 小角度旋转范围 [-deg, deg]
    "brightness": 0.1,         # 亮度偏移 b ~ U(-0.1, 0.1)
    "contrast": 0.1,           # 对比度系数 c ~ U(0.9, 1.1)
}


def rotate_small(arr: np.ndarray, angle_deg: float, nearest: bool) -> np.ndarray:
    """绕图像中心旋转 angle_deg 度（逆时针为正），尺寸不变，边界镜像填充。"""
    h, w = arr.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), angle_deg, 1.0)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = cv2.warpAffine(arr, mat, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT_101)
    return out.astype(arr.dtype, copy=False)


def adjust_brightness_contrast(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """x' = (x - mean)·contrast + mean + brightness，裁剪到 [0,1]，保持 float32。"""
    mean = np.float32(image.mean())
    out = (image - mean) * np.float32(contrast) + mean + np.float32(brightness)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def sample_params(rng: np.random.Generator, cfg: dict) -> dict:
    """先抽好一组随机参数，再同步作用到 image / label / fov 上。"""
    cfg = {**DEFAULT_AUG, **(cfg or {})}
    return {
        "hflip": bool(rng.random() < cfg["hflip"]),
        "vflip": bool(rng.random() < cfg["vflip"]),
        "k90": int(rng.integers(0, 4)) if cfg["rot90"] else 0,
        "angle": float(rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"]))
        if (cfg["rotate_deg"] > 0 and rng.random() < cfg["rotate_prob"]) else 0.0,
        "brightness": float(rng.uniform(-cfg["brightness"], cfg["brightness"])),
        "contrast": float(rng.uniform(1.0 - cfg["contrast"], 1.0 + cfg["contrast"])),
    }


def apply_geometric(arr: np.ndarray, params: dict, nearest: bool) -> np.ndarray:
    """按 params 对单个数组做几何变换（翻转 → 90° 旋转 → 小角度旋转）。"""
    if params["hflip"]:
        arr = arr[:, ::-1]
    if params["vflip"]:
        arr = arr[::-1, :]
    if params["k90"]:
        arr = np.rot90(arr, params["k90"], axes=(0, 1))
    arr = np.ascontiguousarray(arr)  # 翻转/rot90 返回的是视图（负步长），OpenCV 需要连续内存
    if params["angle"] != 0.0:
        arr = rotate_small(arr, params["angle"], nearest=nearest)
    return arr


def augment_sample(image: np.ndarray, label: np.ndarray, fov: np.ndarray,
                   rng: np.random.Generator, cfg: dict | None = None):
    """对一个 patch 做随机增强，返回 (image, label, fov)。cfg 为 None 或 enabled=False 时原样返回。"""
    cfg = {**DEFAULT_AUG, **(cfg or {})}
    if not cfg["enabled"]:
        return image, label, fov
    params = sample_params(rng, cfg)
    image = apply_geometric(image, params, nearest=False)
    label = apply_geometric(label, params, nearest=True)
    fov = apply_geometric(fov, params, nearest=True)
    image = adjust_brightness_contrast(image, params["brightness"], params["contrast"])
    return image, label, fov
