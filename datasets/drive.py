"""DRIVE 数据集适配器。

数据格式（已实际检查用户手中的数据）
------------------------------------
- ``training/images/XX_training.tif``：RGB 彩色眼底图，uint8，584 行 × 565 列（H×W），XX = 21..40。
  TIFF 为 LZW 压缩；tifffile 解 LZW 需要额外的 imagecodecs 包，而 Pillow 原生支持，
  所以统一用 Pillow 读取，不新增依赖。
- ``training/1st_manual/XX_manual1.gif``：第一位专家的手工血管标注，灰度，取值 {0, 255}。
- ``training/mask/XX_training_mask.gif``：FOV（视野）掩膜，灰度，取值 {0, 255}。
- ``test/`` 只有 images 和 mask，**没有标签**，因此不参与指标计算。

偏离计划书的划分（用户决定）
----------------------------
由于测试集没有标签，train / val / test 全部从 20 张带标签训练图中按 config 里的种子固定划分
（默认 12 / 4 / 4）。划分逻辑在 ``datasets/base.py::split_names``，本文件只负责"把文件读对"。

二值化阈值
----------
标签和 FOV 实际只有 0 与 255 两个值，用 ``> 127`` 二值化：对纯 {0,255} 数据与 ``> 0`` 等价，
但若将来遇到经过插值/有损压缩的版本（边缘出现中间灰度），取中间阈值更稳妥。
实测 20 张训练图中落在 FOV 外的血管像素占比均 < 0.3%（只有视野边缘少量像素），
说明标签与 FOV 是对齐的；后续指标只在 FOV 内计算。

用法::

    python -m datasets.drive          # 打印划分信息，并随机抽 3 张保存 label 叠加图
"""

from pathlib import Path

import numpy as np
from PIL import Image

from datasets.base import VesselDataset
from utils.io import resolve_path


def _read_binary_gif(path: Path) -> np.ndarray:
    """读取 {0,255} 的 gif 掩膜，返回 uint8 的 {0,1} 数组。"""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


class DRIVEDataset(VesselDataset):
    """DRIVE 眼底血管数据集（仅使用带标签的 training/ 目录）。

    参数
    ----
    root  : DRIVE 根目录（含 training/），相对路径按仓库根目录解析
    split : "train" / "val" / "test"
    seed, sizes : 划分种子和各 split 大小；一般直接用 ``from_config`` 从 yaml 里取
    """

    def __init__(self, root, split: str, seed: int, sizes: dict):
        self.root = resolve_path(root)
        self.train_dir = self.root / "training"
        if not self.train_dir.is_dir():
            raise FileNotFoundError(f"找不到 DRIVE 训练目录: {self.train_dir}")
        super().__init__(split, seed, sizes)

    @classmethod
    def from_config(cls, cfg: dict, split: str) -> "DRIVEDataset":
        """从 configs/drive.yaml 读出的 dict 构造数据集。"""
        split_cfg = cfg["data"]["split"]
        return cls(cfg["data"]["root"], split, split_cfg["seed"], split_cfg["sizes"])

    def all_names(self) -> list:
        # 以 images 目录为准，名字取文件名前缀 "21_training.tif" -> "21"
        names = [p.name.split("_")[0] for p in (self.train_dir / "images").glob("*_training.tif")]
        return sorted(names)

    def _paths(self, name: str) -> dict:
        return {
            "image": self.train_dir / "images" / f"{name}_training.tif",
            "label": self.train_dir / "1st_manual" / f"{name}_manual1.gif",
            "fov": self.train_dir / "mask" / f"{name}_training_mask.gif",
        }

    def load_sample(self, name: str) -> dict:
        paths = self._paths(name)
        rgb = np.array(Image.open(paths["image"]).convert("RGB"))       # uint8, H×W×3
        image = rgb.astype(np.float32) / np.float32(255.0)              # float32, [0,1]
        return {
            "image": image,
            "label": _read_binary_gif(paths["label"]),
            "fov": _read_binary_gif(paths["fov"]),
            "name": name,
        }


def make_overlay(image: np.ndarray, label: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """生成可视化：血管以半透明绿色叠加在原图上，FOV 边界画成黄色细线。返回 uint8 RGB。"""
    import cv2

    out = (image * 255).astype(np.uint8).copy()
    vessel = label.astype(bool)
    green = np.array([0, 255, 0], dtype=np.float32)
    out[vessel] = (0.4 * out[vessel] + 0.6 * green).astype(np.uint8)

    contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color=(255, 255, 0), thickness=1)
    return out


def _check(config_path: str = "configs/drive.yaml", n_vis: int = 3) -> None:
    """任务 1.1 验收脚本：打印各 split 数量、尺寸、名字，随机抽样保存叠加图。"""
    from utils.io import load_config, make_run_dir, save_json

    cfg = load_config(config_path)
    split_cfg = cfg["data"]["split"]
    print(f"[drive] root={cfg['data']['root']}  seed={split_cfg['seed']}  sizes={split_cfg['sizes']}")

    datasets = {s: DRIVEDataset.from_config(cfg, s) for s in ("train", "val", "test")}
    summary = {}
    for split, ds in datasets.items():
        shapes = set()
        outside = []
        for i in range(len(ds)):
            sample = ds[i]
            shapes.add(sample["image"].shape)
            lab, fov = sample["label"], sample["fov"]
            outside.append(float((lab & (1 - fov)).sum() / max(lab.sum(), 1)))
        h, w, c = next(iter(shapes))
        print(f"[drive] {split:5s}: n={len(ds):2d}  image H×W×C={h}×{w}×{c}"
              f"{'' if len(shapes) == 1 else '  (尺寸不一致! ' + str(shapes) + ')'}"
              f"  FOV 外血管像素占比最大={max(outside):.4%}")
        print(f"         names={ds.names}")
        summary[split] = {"n": len(ds), "shapes": sorted(shapes), "names": ds.names,
                          "max_label_outside_fov_ratio": max(outside)}

    # 用 config 里的种子派生一个独立的 rng 抽样，保证每次抽到同样 3 张
    all_names = sorted(n for ds in datasets.values() for n in ds.names)
    rng = np.random.default_rng(split_cfg["seed"])
    picked = sorted(rng.choice(all_names, size=n_vis, replace=False).tolist())
    name_to_split = {n: s for s, ds in datasets.items() for n in ds.names}

    run_dir = make_run_dir("drive_check")
    for name in picked:
        ds = datasets[name_to_split[name]]
        sample = ds[ds.names.index(name)]
        overlay = make_overlay(sample["image"], sample["label"], sample["fov"])
        out_path = run_dir / f"overlay_{name}_{name_to_split[name]}.png"
        Image.fromarray(overlay).save(out_path)
        print(f"[drive] 保存叠加图: {out_path}")

    summary["visualized"] = picked
    save_json(summary, run_dir / "summary.json")
    print(f"[drive] 摘要: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    _check()
