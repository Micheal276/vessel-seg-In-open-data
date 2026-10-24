"""官方测试集提交脚本（scripts/predict_submission.py）的测试。

设计理由
--------
提交格式错了不会有任何报错，只会在排行榜上得到一个莫名其妙的低分，因此把格式约定逐条写成断言：
- 文件名映射："01_test.tif" → "1.png"（去前导零），编号重复报错；
- 掩膜匹配："<主名>_mask.*"；
- PNG：8 位单通道 L、取值 {0,255}、与输入同尺寸、FOV 外全 0；
- zip：根目录下恰好是这些 PNG，没有子目录或多余文件；
- 阈值来源：命令行 > metrics.json 的 best_val_threshold > 缺省 0.5。
端到端部分用一个随机初始化的极小 U-Net 和临时目录里的假图，不读真实数据、不写 results/。
"""

import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from models.unet import UNet
from scripts.predict_submission import (find_mask, list_images, main, resolve_threshold, submission_name,
                                        to_submission_png)


def test_submission_name_strips_leading_zeros():
    assert submission_name("01_test.tif") == "1.png"
    assert submission_name("09_test.tif") == "9.png"
    assert submission_name("20_test.tif") == "20.png"
    assert submission_name("25_training.tif") == "25.png"
    with pytest.raises(ValueError):
        submission_name("test_01.tif")


def test_official_names_cover_1_to_20():
    names = [submission_name(f"{i:02d}_test.tif") for i in range(1, 21)]
    assert names == [f"{i}.png" for i in range(1, 21)]


def test_find_mask_and_duplicate_numbers(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "mask").mkdir()
    for name in ("01_test.tif", "1_test.tif"):
        Image.new("RGB", (4, 4)).save(tmp_path / "img" / name)
    Image.new("L", (4, 4)).save(tmp_path / "mask" / "01_test_mask.gif")
    assert find_mask(tmp_path / "img" / "01_test.tif", tmp_path / "mask").name == "01_test_mask.gif"
    with pytest.raises(FileNotFoundError):
        find_mask(tmp_path / "img" / "1_test.tif", tmp_path / "mask")
    with pytest.raises(ValueError):          # 01 与 1 都会映射成 1.png
        list_images(tmp_path / "img")


def test_to_submission_png_binary_and_zero_outside_fov():
    prob = np.array([[0.2, 0.38, 0.39], [0.9, 0.9, 0.0]], dtype=np.float32)
    fov = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.uint8)
    png = to_submission_png(prob, fov, 0.38)
    assert png.dtype == np.uint8
    # 严格大于阈值才是前景；FOV 外（第 2 行第 2 列）即使概率 0.9 也为 0
    assert png.tolist() == [[0, 0, 255], [255, 0, 0]]


def test_resolve_threshold_priority(tmp_path):
    js = tmp_path / "metrics.json"
    js.write_text(json.dumps({"best_val_threshold": 0.38}), encoding="utf-8")
    assert resolve_threshold(0.6, str(js))[0] == 0.6
    assert resolve_threshold(None, str(js))[0] == 0.38
    assert resolve_threshold(None, str(tmp_path / "missing.json"))[0] == 0.5
    js.write_text(json.dumps({}), encoding="utf-8")
    assert resolve_threshold(None, str(js))[0] == 0.5


def test_end_to_end_zip_and_png_format(tmp_path):
    h, w = 40, 36
    rng = np.random.default_rng(0)
    img_dir, mask_dir = tmp_path / "images", tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()
    yy, xx = np.mgrid[:h, :w]
    fov = (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (min(h, w) / 2 - 2) ** 2).astype(np.uint8) * 255
    for i in (1, 2, 3):
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(img_dir / f"{i:02d}_test.tif")
        Image.fromarray(fov).save(mask_dir / f"{i:02d}_test_mask.gif")

    unet_cfg = {"in_channels": 1, "out_channels": 1, "base_channels": 4, "depth": 2}
    torch.manual_seed(0)
    pre = {"green_channel": True, "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid": [2, 2]},
           "fill_outside_fov": {"enabled": False}, "denoise": {"method": "none"}}
    ckpt = tmp_path / "best.pth"
    torch.save({"model": UNet.from_config(unet_cfg).state_dict(), "unet_cfg": unet_cfg, "preprocess": pre,
                "data_name": "drive", "epoch": 1}, ckpt)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps({"experiment": "drive", "data": {"name": "drive"},
                               "unet": {**unet_cfg, "preprocess": pre},
                               "infer": {"patch_size": 16, "stride": 8, "batch_size": 8}}), encoding="utf-8")

    run_dir = main(str(cfg), str(ckpt), str(img_dir), str(mask_dir), threshold=0.5, out=str(tmp_path / "run"))

    with zipfile.ZipFile(run_dir / "drive_submission.zip") as zf:
        assert sorted(zf.namelist()) == ["1.png", "2.png", "3.png"]
        for n in zf.namelist():
            im = Image.open(zf.open(n))
            a = np.array(im)
            assert im.mode == "L" and a.shape == (h, w)
            assert set(np.unique(a).tolist()) <= {0, 255}
            assert (a[fov == 0] == 0).all()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["threshold"] == 0.5
    assert [e["source"] for e in manifest["images"]] == ["01_test.tif", "02_test.tif", "03_test.tif"]
    assert [e["png"] for e in manifest["images"]] == ["1.png", "2.png", "3.png"]
    assert (run_dir / "probs" / "1.npy").exists()
