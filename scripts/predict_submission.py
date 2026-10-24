"""官方测试集提交：对无标签的 DRIVE 官方测试图（01_test.tif … 20_test.tif）生成 grand-challenge 提交包。

提交格式（grand-challenge DRIVE）
---------------------------------
一个 zip，根目录下恰好是 1.png, 2.png, …, 20.png（无前导零、无子文件夹），
01_test.tif → 1.png …… 20_test.tif → 20.png；每张是与原图同尺寸（584×565）的二值血管图。
评分为逐图在 FOV 掩膜内的 Dice 再取平均。

设计理由（面试讲解要点）
------------------------
1. 不走带标签的数据集接口。``datasets/`` 里的 Dataset 约定每个样本都有 label，官方测试集没有标签，
   硬塞进去只能伪造一个假 label。这里直接复用两块与标签无关的函数：
   ``classical.preprocess.preprocess``（预处理）和 ``eval.infer.sliding_window_predict``（滑窗推理），
   读图方式（PIL 转 RGB、除以 255）与 ``datasets/drive.py`` 一致，预处理后同样裁到 [0,1]
   （与 ``train.patch_dataset.prepare_samples`` 一致）——保证网络看到的输入分布和训练时相同。

2. 预处理配置取自 checkpoint，而不是当前 config。网络只认训练时那套预处理（绿通道 / CLAHE 参数），
   checkpoint 里记录的才是"真相"；当前 config 只用于滑窗参数（``infer`` 小节）。
   两者不一致时打印警告（复用 ``scripts.test.check_preprocess_consistency``）。

3. 阈值不硬编码。默认从 ``submission.threshold_from`` 指向的 metrics.json 读 ``best_val_threshold``
   （scripts.test 只在 val split 上按逐图平均 Dice 选出的阈值），--threshold 可显式覆盖；
   文件或字段缺失时退回 0.5 并醒目提示。实际用到的阈值和来源写进 manifest.json。

4. 二值化与 FOV。用 ``eval.metrics.binarize``（prob > t，严格大于，与报告指标时同一规则），
   再把 FOV 外置 0：评分只看 FOV 内，但 FOV 外留下的假阳性会让提交图看起来不干净，也可能被别的
   评估脚本误计。保存为 8 位单通道 PNG，取值 {0,255}（{0,1} 的 PNG 肉眼看全黑，评估端也可能按 >127 判前景）。

5. 文件名映射单独成函数（``submission_name``）：取文件名开头的数字转 int 去掉前导零，
   "01_test.tif" → "1.png"。重复编号直接报错，防止两张图互相覆盖后包里少一张却没人发现。

6. 可验证性。同一脚本也能跑带标签的训练图（--ids 选子集，掩膜按 "<主名>_mask.gif" 自动匹配），
   这样可以用写出的 PNG 重新算 Dice，与 scripts.test 报告的数值对照，证明"PNG 这一步"没有引入错误。

用法::

    python -m scripts.predict_submission --config configs/drive.yaml \\
        --ckpt results/unet_drive_train/20261013-161717/best.pth \\
        --images-dir data/DRIVE/test/images --mask-dir data/DRIVE/test/mask

输出（默认 ``results/submission_<experiment>/<时间戳>/``，--out 可指定目录）：
png/<k>.png、probs/<k>.npy（float32 概率图）、manifest.json、drive_submission.zip。
"""

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.device import get_device  # 先导入：在 import torch 前设置 MPS 回退环境变量
from classical.preprocess import preprocess
from eval.infer import sliding_window_predict, to_chw_tensor
from eval.metrics import binarize
from models.unet import UNet
from scripts.test import check_preprocess_consistency, checkpoint_preprocess
from utils.io import load_config, load_json, make_run_dir, resolve_path, save_json

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
ZIP_NAME = "drive_submission.zip"


def image_number(filename: str) -> int:
    """取文件名开头的数字（去前导零）："01_test.tif" → 1，"25_training.tif" → 25。没有数字开头则报错。"""
    m = re.match(r"(\d+)", Path(filename).name)
    if m is None:
        raise ValueError(f"文件名 {filename!r} 不以数字开头，无法映射为提交文件名")
    return int(m.group(1))


def submission_name(filename: str) -> str:
    """原图文件名 → 提交 PNG 文件名："01_test.tif" → "1.png"。"""
    return f"{image_number(filename)}.png"


def find_mask(image_path: Path, mask_dir: Path) -> Path:
    """按 DRIVE 命名找 FOV 掩膜："01_test.tif" → "01_test_mask.gif"，"25_training.tif" → "25_training_mask.gif"。"""
    candidates = sorted(mask_dir.glob(f"{image_path.stem}_mask.*"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"{image_path.name} 在 {mask_dir} 下应恰好有 1 个掩膜 '{image_path.stem}_mask.*'，"
                                f"实际找到 {[c.name for c in candidates]}")
    return candidates[0]


def list_images(images_dir: Path, ids=None) -> list:
    """列出目录下的图像，按编号排序；ids 非空时只保留这些编号。编号重复时报错。"""
    paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")]
    if ids:
        wanted = {int(i) for i in ids}
        paths = [p for p in paths if image_number(p.name) in wanted]
        missing = wanted - {image_number(p.name) for p in paths}
        if missing:
            raise FileNotFoundError(f"{images_dir} 下找不到编号 {sorted(missing)} 的图像")
    paths.sort(key=lambda p: image_number(p.name))
    numbers = [image_number(p.name) for p in paths]
    dup = sorted({n for n in numbers if numbers.count(n) > 1})
    if dup:
        raise ValueError(f"编号重复 {dup}：多张图会映射到同一个 PNG 名")
    if not paths:
        raise FileNotFoundError(f"{images_dir} 下没有图像")
    return paths


def read_fov(path: Path) -> np.ndarray:
    """读 FOV 掩膜为 {0,1} uint8（与 datasets/drive.py 的 >127 二值化一致）。"""
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def to_submission_png(prob: np.ndarray, fov: np.ndarray, threshold: float) -> np.ndarray:
    """概率图 → 提交用 uint8 二值图：prob > t 为 255，FOV 外一律 0。"""
    pred = binarize(prob, threshold) * (fov > 0)
    return (pred * 255).astype(np.uint8)


def resolve_threshold(cli_threshold, threshold_json) -> tuple:
    """返回 (阈值, 来源说明)。优先级：命令行 > metrics.json 的 best_val_threshold > 0.5。"""
    if cli_threshold is not None:
        return float(cli_threshold), "命令行 --threshold"
    if threshold_json:
        path = resolve_path(threshold_json)
        if path.exists():
            t = load_json(path).get("best_val_threshold")
            if t is not None:
                return float(t), f"{threshold_json} 的 best_val_threshold（val split 上选出）"
            print(f"[submit] !!! {threshold_json} 中没有 best_val_threshold，退回 0.5", flush=True)
        else:
            print(f"[submit] !!! 找不到 {threshold_json}，退回 0.5", flush=True)
    else:
        print("[submit] !!! 未指定阈值来源，退回 0.5", flush=True)
    return 0.5, "缺省 0.5（未找到 val 上选出的阈值）"


def write_zip(png_dir: Path, names: list, zip_path: Path) -> Path:
    """把 names 中的 PNG 平铺写入 zip 根目录（不带任何子路径）。"""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            zf.write(png_dir / n, arcname=n)
    return zip_path


def main(config_path: str, ckpt_path: str, images_dir: str, mask_dir: str,
         threshold=None, out=None, ids=None, threshold_json=None) -> Path:
    """生成提交包，返回结果目录（Path）。"""
    cfg = load_config(config_path)
    device = get_device()
    sub_cfg = cfg.get("submission", {})
    if threshold_json is None:
        threshold_json = sub_cfg.get("threshold_from")
    t, t_source = resolve_threshold(threshold, threshold_json)
    print(f"[submit] 阈值 t={t:g}（来源：{t_source}）")

    ckpt = torch.load(resolve_path(ckpt_path), map_location="cpu", weights_only=False)
    model = UNet.from_config(ckpt.get("unet_cfg", cfg["unet"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for w in check_preprocess_consistency(ckpt, cfg):
        print(f"[submit] !!! 警告：{w}", flush=True)
    pre_cfg, _ = checkpoint_preprocess(ckpt)
    if pre_cfg is None:
        pre_cfg = cfg["unet"]["preprocess"]
        print("[submit] !!! checkpoint 无预处理记录，改用当前 config 的 unet.preprocess", flush=True)
    infer_cfg = cfg["infer"]

    run_dir = resolve_path(out) if out else make_run_dir(f"submission_{cfg.get('experiment', 'drive')}")
    png_dir, prob_dir = run_dir / "png", run_dir / "probs"
    png_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    images_dir, mask_dir = resolve_path(images_dir), resolve_path(mask_dir)
    entries = []
    for img_path in list_images(images_dir, ids):
        mask_path = find_mask(img_path, mask_dir)
        rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / np.float32(255.0)
        fov = read_fov(mask_path)
        if fov.shape != rgb.shape[:2]:
            raise ValueError(f"{img_path.name} 尺寸 {rgb.shape[:2]} 与掩膜 {fov.shape} 不一致")
        x = np.clip(preprocess(rgb, fov, pre_cfg)["output"], 0.0, 1.0).astype(np.float32)
        prob = sliding_window_predict(model, to_chw_tensor(x), int(infer_cfg["patch_size"]), int(infer_cfg["stride"]),
                                      device, int(infer_cfg.get("batch_size", 16)))
        png = to_submission_png(prob, fov, t)
        name = submission_name(img_path.name)
        Image.fromarray(png, mode="L").save(png_dir / name)
        np.save(prob_dir / f"{Path(name).stem}.npy", prob)
        frac = float((png[fov > 0] > 0).mean())
        entries.append({"source": img_path.name, "mask": mask_path.name, "png": name,
                        "height": int(png.shape[0]), "width": int(png.shape[1]),
                        "vessel_fraction_in_fov": frac})
        print(f"[submit] {img_path.name} -> {name}  FOV 内血管占比 {frac:.4f}")

    zip_path = write_zip(png_dir, [e["png"] for e in entries], run_dir / ZIP_NAME)
    save_json({"checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt.get("epoch"),
               "config": str(config_path), "preprocess": pre_cfg, "infer": infer_cfg,
               "threshold": t, "threshold_source": t_source,
               "images_dir": str(images_dir), "mask_dir": str(mask_dir),
               "zip": ZIP_NAME, "images": entries}, run_dir / "manifest.json")
    print(f"[submit] 共 {len(entries)} 张；zip: {zip_path}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description="生成 DRIVE 官方测试集提交包（1.png..20.png 的 zip）")
    p.add_argument("--config", default="configs/drive.yaml")
    p.add_argument("--ckpt", required=True, help="best.pth 路径")
    p.add_argument("--images-dir", default="data/DRIVE/test/images")
    p.add_argument("--mask-dir", default="data/DRIVE/test/mask")
    p.add_argument("--threshold", type=float, default=None,
                   help="二值化阈值；缺省时读 config 的 submission.threshold_from 里的 best_val_threshold")
    p.add_argument("--threshold-json", default=None, help="覆盖 submission.threshold_from")
    p.add_argument("--out", default=None, help="结果目录；缺省 results/submission_<experiment>/<时间戳>/")
    p.add_argument("--ids", nargs="*", type=int, default=None, help="只处理这些编号的图（用于在带标签图上核对）")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(a.config, a.ckpt, a.images_dir, a.mask_dir, threshold=a.threshold, out=a.out, ids=a.ids,
         threshold_json=a.threshold_json)
