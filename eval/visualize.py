"""任务 1.5 可视化：ROC 曲线、逐图 Dice 箱线图、典型样例对比与错误图，以及辅助区域（视盘 / 细血管）的提取。

设计理由（面试讲解要点）
------------------------
1. ROC（pooled）：把所有测试图 FOV 内的像素拼在一起画一条曲线，曲线下面积就是 eval/metrics.py 的
   Global AUC（同一组像素、同一个 sklearn 实现），图例里的 AUC 直接取该值，保证"图与表一致"。
   ROC 不依赖阈值，横轴 1−Specificity、纵轴 Sensitivity；血管只占 FOV 约 12%，所以真正有区分度的
   部分在左侧低 FPR 区域，另外画一个 FPR ≤ 0.2 的放大子图。

2. 逐图 Dice 箱线图：测试只有 4 张图，箱线图的四分位数不太有统计意义，因此把每张图的点叠加上去并标注图名，
   读者能直接看到"是哪张图拉低了平均"。

3. 错误图配色：TP 白、FP 红、FN 蓝、TN 黑，FOV 外用暗灰（与 FOV 内的黑色 TN 区分开，表示"不参与评估"）。
   这是分割论文里最常见的配色，红/蓝分别对应"多分"和"漏分"，一眼能看出方法的偏向。

4. 辅助区域（只用于可视化裁剪与补充分析，不参与主指标）：
   - 视盘：眼底图上最亮的大块区域。取灰度图在 FOV 内做大尺度高斯平滑后的最大值位置（启发式；
     对 OCTA 等无视盘的数据没有意义，调用方可不用）。
   - 细血管：金标准骨架上距离变换给出局部半径 r，把每个血管像素归到最近的骨架点，r ≤ 阈值的算"细血管"。
     这样能分别统计细 / 粗血管的召回率，量化"U-Net 是否更擅长细血管"。

5. 只依赖 numpy / matplotlib / scipy / skimage / sklearn，与数据集无关：输入是 eval/evaluate.py 组装好的行
   （每行含方法名、连续分数图、二值图、summary）与样本列表。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.metrics import roc_curve

COLORS = {"Frangi": "#1f77b4", "Matched filter": "#ff7f0e", "U-Net": "#2ca02c"}
OUTSIDE_FOV = np.array([0.22, 0.22, 0.22], dtype=np.float32)


# ---------------------------------------------------------------------------
# 辅助：显示用图像、错误图、区域
# ---------------------------------------------------------------------------

def to_display(image: np.ndarray) -> np.ndarray:
    """float32 [0,1] 的 H×W 或 H×W×3 图转成可 imshow 的 RGB。"""
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img[..., :3]


def gray_of(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    return img if img.ndim == 2 else img[..., :3].mean(axis=2)


def error_map(pred: np.ndarray, label: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """TP 白 / FP 红 / FN 蓝 / TN 黑；FOV 外暗灰。返回 float32 H×W×3。"""
    p, t, f = pred.astype(bool), label.astype(bool), fov.astype(bool)
    out = np.zeros(p.shape + (3,), dtype=np.float32)
    out[p & t] = (1.0, 1.0, 1.0)
    out[p & ~t] = (1.0, 0.0, 0.0)
    out[~p & t] = (0.0, 0.35, 1.0)
    out[~f] = OUTSIDE_FOV
    return out


def find_optic_disc(image: np.ndarray, fov: np.ndarray, sigma: float = 15.0) -> tuple:
    """视盘中心 (row, col) 的启发式估计：FOV（腐蚀 20 像素）内大尺度平滑灰度的最大值位置。"""
    g = ndi.gaussian_filter(gray_of(image), sigma)
    inner = ndi.binary_erosion(fov.astype(bool), iterations=20)
    g = np.where(inner, g, -np.inf)
    r, c = np.unravel_index(int(np.argmax(g)), g.shape)
    return int(r), int(c)


def disc_mask(shape, center: tuple, radius: int) -> np.ndarray:
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2


def vessel_radius_map(label: np.ndarray) -> np.ndarray:
    """每个血管像素的局部半径：最近骨架点处的距离变换值；背景为 0。"""
    lab = label.astype(bool)
    dist = ndi.distance_transform_edt(lab)
    skel = skeletonize(lab)
    _, (ir, ic) = ndi.distance_transform_edt(~skel, return_indices=True)
    radius = dist[ir, ic].astype(np.float32)
    radius[~lab] = 0.0
    return radius


def thin_vessel_mask(label: np.ndarray, max_radius: float = 1.5) -> np.ndarray:
    """局部半径 ≤ max_radius（即宽度约 ≤ 3 像素）的血管像素。"""
    return (vessel_radius_map(label) <= max_radius) & label.astype(bool)


def crop_box(center: tuple, size: int, shape) -> tuple:
    """以 center 为中心、边长 size 的裁剪框（贴边时平移进图内）；图比 size 小时退化为整图。"""
    size = int(min(size, shape[0], shape[1]))
    r0 = int(np.clip(center[0] - size // 2, 0, shape[0] - size))
    c0 = int(np.clip(center[1] - size // 2, 0, shape[1] - size))
    return slice(r0, r0 + size), slice(c0, c0 + size)


def thin_vessel_center(label: np.ndarray, fov: np.ndarray, avoid: tuple | None, size: int,
                       avoid_radius: int) -> tuple:
    """细血管像素密度最高的 size×size 窗口中心（avoid 不为 None 时避开该点附近，且窗口尽量在 FOV 内）。

    图很小、FOV 腐蚀后为空时退回整个 FOV（再不行就整图），保证总能返回一个位置。
    """
    size = int(min(size, label.shape[0], label.shape[1]))
    dens = ndi.uniform_filter(thin_vessel_mask(label).astype(np.float32), size)
    valid = ndi.binary_erosion(fov.astype(bool), iterations=max(size // 2, 1))
    if avoid is not None:
        valid &= ~disc_mask(label.shape, avoid, avoid_radius + size // 2)
    if not valid.any():
        valid = fov.astype(bool) if fov.any() else np.ones(label.shape, bool)
    dens = np.where(valid, dens, -1.0)
    r, c = np.unravel_index(int(np.argmax(dens)), dens.shape)
    return int(r), int(c)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------

def plot_roc(rows: list, samples: list, path) -> None:
    """FOV 内像素跨图汇总的 ROC；图例 AUC 取各行 summary 的 Global AUC。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for r in rows:
        y = np.concatenate([s["label"][s["fov"].astype(bool)] for s in samples]).astype(np.uint8)
        sc = np.concatenate([r["scores"][s["name"]][s["fov"].astype(bool)] for s in samples])
        fpr, tpr, _ = roc_curve(y, sc)
        lab = f"{r['method']} (AUC {r['summary']['global']['auc']:.4f})"
        for ax in axes:
            ax.plot(fpr, tpr, color=COLORS.get(r["method"]), lw=1.6, label=lab)
    axes[0].plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    axes[0].set_title("ROC (pooled in-FOV pixels, test)")
    axes[1].set_xlim(0, 0.2)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_title("zoom: FPR ≤ 0.2")
    for ax in axes:
        ax.set_xlabel("False positive rate (1 − Specificity)")
        ax.set_ylabel("True positive rate (Sensitivity)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_dice_box(rows: list, path) -> None:
    """逐图 Dice 箱线图 + 每张图的点（标注图名）。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    data = [[m["dice"] for m in r["summary"]["per_image"]] for r in rows]
    ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": "black"})
    offsets = np.linspace(-0.12, 0.12, len(rows[0]["summary"]["per_image"]))
    for i, r in enumerate(rows, start=1):
        for off, m in zip(offsets, r["summary"]["per_image"]):
            ax.scatter(i + off, m["dice"], color=COLORS.get(r["method"]), s=30, zorder=3, edgecolor="k", lw=0.5)
            ax.annotate(m["name"], (i + off, m["dice"]), xytext=(5, 0), textcoords="offset points",
                        fontsize=7, va="center")
    ticks = [f"{r['method']}\n" + r["threshold_label"].replace(" (", "\n(") for r in rows]
    ax.set_xticks(range(1, len(rows) + 1), ticks, fontsize=8)
    ax.set_ylabel("Dice (per image, in FOV)")
    ax.set_title(f"Per-image Dice on test (n={len(data[0])})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_example(sample: dict, rows: list, path) -> None:
    """2×3：原图 / 金标准 / Frangi / 匹配滤波 / U-Net / U-Net 错误图。rows 顺序为 Frangi, MF, U-Net。"""
    name, label, fov = sample["name"], sample["label"], sample["fov"]
    unet = rows[-1]
    panels = [(to_display(sample["image"]), "Original"), (label, "Ground truth")]
    for r in rows:
        d = next(m["dice"] for m in r["summary"]["per_image"] if m["name"] == name)
        panels.append((r["binaries"][name], f"{r['method']} ({r['threshold_label'].split(' (')[0]}) Dice {d:.3f}"))
    panels.append((error_map(unet["binaries"][name], label, fov), "U-Net errors: TP white, FP red, FN blue"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Test image {name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_zoom(sample: dict, rows: list, path, size: int = 128, disc_radius: int = 45,
              optic_disc: bool = True) -> dict:
    """局部放大（视盘区域 / 细血管最密区域）× 原图、金标准、三种方法的错误图。返回裁剪框信息。

    optic_disc=False（无视盘的数据，如 OCTA）时只画细血管最密区域一行。
    """
    name, label, fov = sample["name"], sample["label"], sample["fov"]
    disc = find_optic_disc(sample["image"], fov) if optic_disc else None
    thin = thin_vessel_center(label, fov, disc, size, disc_radius)
    regions = [("thin vessels", crop_box(thin, size, label.shape))]
    if optic_disc:
        regions.insert(0, ("optic disc", crop_box(disc, size, label.shape)))
    cols = [("Original", lambda: to_display(sample["image"])), ("Ground truth", lambda: to_display(label))]
    cols += [(f"{r['method']} errors", (lambda r=r: error_map(r["binaries"][name], label, fov))) for r in rows]
    full = [fn() for _, fn in cols]
    fig, axes = plt.subplots(len(regions), len(cols), figsize=(2.6 * len(cols), 2.8 * len(regions)), squeeze=False)
    for i, (rname, (rs, cs)) in enumerate(regions):
        for j, ((cname, _), img) in enumerate(zip(cols, full)):
            ax = axes[i, j]
            ax.imshow(img[rs, cs], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(cname, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{rname}\nrows {rs.start}-{rs.stop}, cols {cs.start}-{cs.stop}", fontsize=8)
    fig.suptitle(f"Test image {name}: zoomed error maps (TP white, FP red, FN blue)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return {"optic_disc_center": disc, "thin_center": thin, "size": size}


def make_all(rows: list, samples: list, run_dir, optic_disc: bool = True) -> dict:
    """生成全部图；只用主表 3 行（Frangi / 匹配滤波 / U-Net）。返回生成的文件列表与裁剪信息。

    optic_disc 控制局部放大图是否裁视盘区域（眼底图特有；由 config compare.supplementary_optic_disc 决定）。
    """
    run_dir = Path(run_dir)
    primary = [r for r in rows if r["table"] == "primary"]
    out = {"roc": run_dir / "roc.png", "dice_boxplot": run_dir / "dice_boxplot.png", "examples": {}, "zoom": {}}
    plot_roc(primary, samples, out["roc"])
    plot_dice_box(primary, out["dice_boxplot"])
    crops = {}
    for s in samples:
        out["examples"][s["name"]] = run_dir / f"example_{s['name']}.png"
        plot_example(s, primary, out["examples"][s["name"]])
        out["zoom"][s["name"]] = run_dir / f"zoom_{s['name']}.png"
        crops[s["name"]] = plot_zoom(s, primary, out["zoom"][s["name"]], optic_disc=optic_disc)
    out["crops"] = crops
    return out
