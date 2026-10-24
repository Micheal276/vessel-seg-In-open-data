"""可迁移性测试：证明阶段二接入新数据集"只需一个 Dataset 适配器 + 一个 config"。

设计理由
--------
阶段二（OCTA）的约束是：只允许新增 Dataset 适配类和 config，models / train / eval / classical / scripts
一行都不改。这里在**测试内部**定义一个假的单通道数据集（H×W 图像、全 1 FOV、1 像素宽血管），
用 monkeypatch 临时登记进 ``datasets.DATASETS``（不进入真实注册表），再写一个临时 config，
然后直接调用各脚本的 ``main``：
- eval.evaluate：全 1 FOV + 只有细血管（粗血管像素数为 0）时补充分析不崩溃，比值报告 NaN / n/a；
  config 不写 ``compare.supplementary`` 时默认不跑补充分析；
- 端到端冒烟：classical.run_baseline → scripts.train（1 个 epoch 的小 U-Net）→ scripts.test → eval.evaluate。
所有结果目录都重定向到 pytest 的临时目录，不污染 results/。
"""

import copy
import itertools
import math

import numpy as np
import pytest
import yaml

import datasets
from datasets.base import VesselDataset
from eval.evaluate import evaluate_row, supplementary_analysis
from utils.io import load_config, save_json

H, W = 128, 144
NAMES = [f"s{i}" for i in range(6)]
SIZES = {"train": 2, "val": 2, "test": 2}


def _synthetic(name: str) -> dict:
    """确定性生成一张单通道合成图：暗背景 + 若干条 1 像素宽的亮血管 + 噪声。"""
    rng = np.random.default_rng(int(name[1:]) + 100)
    label = np.zeros((H, W), np.uint8)
    for r in rng.choice(np.arange(12, H - 12, 6), 3, replace=False):   # 横线（行间隔 ≥ 6，不会并成粗血管）
        label[r, 8:W - 8] = 1
    for c in rng.choice(np.arange(12, W - 12, 6), 2, replace=False):   # 竖线
        label[8:H - 8, c] = 1
    diag = np.arange(10, min(H, W) - 10)
    label[diag, diag] = 1
    image = 0.25 + 0.45 * label + rng.normal(0.0, 0.04, (H, W))
    return {"image": np.clip(image, 0.0, 1.0).astype(np.float32), "label": label,
            "fov": np.ones((H, W), np.uint8), "name": name}


class FakeOCTADataset(VesselDataset):
    """测试专用的假适配器：单通道、无 FOV（全 1）、1 像素宽血管。只存在于本测试文件中。"""

    @classmethod
    def from_config(cls, cfg: dict, split: str) -> "FakeOCTADataset":
        sc = cfg["data"]["split"]
        return cls(split, sc["seed"], sc["sizes"])

    def all_names(self) -> list:
        return list(NAMES)

    def load_sample(self, name: str) -> dict:
        return _synthetic(name)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """登记假数据集，并把所有脚本的结果目录重定向到 tmp_path。"""
    monkeypatch.setitem(datasets.DATASETS, "fake", FakeOCTADataset)
    counter = itertools.count()

    def fake_run_dir(exp_name, results_root="results", timestamp=None):
        d = tmp_path / "results" / exp_name / f"run{next(counter)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    import classical.run_baseline
    import eval.evaluate
    import scripts.test
    import scripts.train
    for mod in (classical.run_baseline, eval.evaluate, scripts.train, scripts.test):
        monkeypatch.setattr(mod, "make_run_dir", fake_run_dir)
    return tmp_path


def _fake_config() -> dict:
    """在 configs/drive.yaml 基础上只改"数据 / 预处理开关 / patch 大小 / 规模"——与阶段二允许改的范围一致。"""
    cfg = copy.deepcopy(load_config("configs/drive.yaml"))
    cfg["experiment"] = "fake"
    cfg["data"] = {"name": "fake", "root": "unused", "split": {"seed": 0, "sizes": dict(SIZES)}}
    cfg["preprocess"]["green_channel"] = False
    cfg["preprocess"]["clahe"]["tile_grid"] = [4, 4]
    cfg["classical"]["frangi"]["black_ridges"] = False       # 血管为亮
    cfg["classical"]["matched_filter"]["dark_vessels"] = False
    cfg["classical"]["n_figure_images"] = 1
    cfg["unet"]["preprocess"]["green_channel"] = False
    cfg["unet"]["preprocess"]["clahe"]["tile_grid"] = [4, 4]
    cfg["unet"].update({"base_channels": 8, "depth": 2})
    cfg["train"].update({"patch_size": 32, "patches_per_image": 8, "batch_size": 8, "epochs": 1})
    cfg["train"]["optimizer"]["warmup_epochs"] = 0
    cfg["infer"] = {"patch_size": 32, "stride": 16, "batch_size": 16}
    cfg["test"]["threshold_grid"] = [0.1, 0.9, 0.1]
    cfg["compare"] = {"split": "test", "tolerance": 1.0e-6}   # 不写 supplementary：默认关闭
    return cfg


def _write_cfg(cfg: dict, path) -> str:
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


def _write_fake_sources(tmp_path, samples: list) -> tuple:
    """不跑方法，直接按 evaluate 读取的格式写出"经典 run"与"U-Net run"（分数图 + 来源 metrics.json）。"""
    rng = np.random.default_rng(0)
    names = [s["name"] for s in samples]
    scores = {key: {s["name"]: np.clip(s["label"] * 0.6 + rng.random((H, W)) * 0.6, 0, 1).astype(np.float32)
                    for s in samples} for key in ("frangi", "matched_filter", "unet")}
    cdir, udir = tmp_path / "src_classical", tmp_path / "src_unet"
    (cdir / "maps" / "test").mkdir(parents=True)
    (udir / "probs").mkdir(parents=True)
    for m in ("frangi", "matched_filter"):
        for n in names:
            np.save(cdir / "maps" / "test" / f"{n}_{m}_response.npy", scores[m][n])
    for n in names:
        np.save(udir / "probs" / f"{n}.npy", scores["unet"][n])
    gt = {"frangi": 0.5, "matched_filter": 0.55}
    save_json({"global_thresholds": gt, "threshold_search_split": "train", "splits": {"test": names},
               "results": {"test": {m: {"global": evaluate_row(samples, scores[m], gt[m])[0],
                                        "otsu": evaluate_row(samples, scores[m], "otsu")[0]} for m in gt}}},
              cdir / "metrics.json")
    save_json({"threshold": 0.5, "best_val_threshold": 0.4, "checkpoint": "fake.pth", "splits": {"test": names},
               "results": {"test|0.5": evaluate_row(samples, scores["unet"], 0.5)[0],
                           "test|best-val t=0.40": evaluate_row(samples, scores["unet"], 0.4)[0]}},
              udir / "metrics.json")
    return cdir, udir


# ---------------------------------------------------------------------------

def test_fake_dataset_contract(fake_env):
    """假适配器经注册表构造，返回符合基类契约的样本（check_sample 在 __getitem__ 里执行）。"""
    cfg = _fake_config()
    ds = {s: datasets.build_dataset(cfg, s) for s in ("train", "val", "test")}
    assert sum(len(d) for d in ds.values()) == len(NAMES)
    s = ds["test"][0]
    assert s["image"].shape == (H, W) and s["fov"].all()
    assert "fake" not in datasets.DATASETS or datasets.DATASETS["fake"] is FakeOCTADataset


def test_supplementary_no_thick_vessels_no_crash():
    """全部是 1 像素宽血管 → 粗血管像素数为 0：比值记为 NaN，不抛 ZeroDivisionError。"""
    samples = [_synthetic(n) for n in NAMES[:2]]
    rows = [{"table": "primary", "method": "X", "binaries": {s["name"]: s["label"] for s in samples}}]
    for disc in (True, False):
        out = supplementary_analysis(rows, samples, optic_disc=disc)
        c = out["methods"]["X"]
        assert c["thick_n"] == 0 and math.isnan(c["sens_thick"])
        assert c["sens_thin"] == pytest.approx(1.0)
        assert c["fp"] == 0
        if disc:
            assert math.isnan(c["fp_disc_frac"])      # FP 为 0 → 视盘内 FP 占比 n/a
        else:
            assert "fp_disc" not in c


@pytest.mark.parametrize("supp", [None, (True, False), (True, True)])
def test_evaluate_end_to_end_on_fake_dataset(fake_env, supp):
    """evaluate.main 在假数据集上端到端运行：默认不跑补充分析；开启时 NaN 显示为 n/a，不崩溃。"""
    from eval.evaluate import main as evaluate_main
    from utils.io import load_json

    cfg = _fake_config()
    if supp is not None:
        cfg["compare"]["supplementary"], cfg["compare"]["supplementary_optic_disc"] = supp
    samples = [datasets.build_dataset(cfg, "test")[i] for i in range(SIZES["test"])]
    cdir, udir = _write_fake_sources(fake_env, samples)
    path = _write_cfg(cfg, fake_env / "fake.yaml")
    run_dir = evaluate_main(path, classical_dir=str(cdir), unet_dir=str(udir), figures=True)

    md = (run_dir / "metrics.md").read_text(encoding="utf-8")
    js = load_json(run_dir / "metrics.json")
    assert (run_dir / "roc.png").exists() and (run_dir / "metrics.csv").exists()
    assert all((run_dir / f"zoom_{s['name']}.png").exists() for s in samples)
    if supp is None:
        assert "补充分析" not in md and js["supplementary"] is None
    else:
        assert "补充分析" in md and "n/a" in md
        assert ("视盘" in md) == supp[1]


def test_end_to_end_smoke_classical_unet_evaluate(fake_env):
    """阶段二演练：只有"适配器 + config"，同一套 classical / scripts / eval 入口全部跑通并输出指标表。"""
    from classical.run_baseline import main as baseline_main
    from eval.evaluate import main as evaluate_main
    from scripts.test import main as test_main
    from scripts.train import main as train_main

    cfg = _fake_config()
    path = _write_cfg(cfg, fake_env / "fake.yaml")

    cdir = baseline_main(path)
    assert (cdir / "metrics.md").exists()
    assert (cdir / "maps" / "test" / f"{datasets.build_dataset(cfg, 'test').names[0]}_frangi_response.npy").exists()

    summary = train_main(path)
    assert summary["checkpoint"].endswith("best.pth")

    udir = test_main(path, summary["checkpoint"])
    assert (udir / "metrics.md").exists() and (udir / "probs").is_dir()

    run_dir = evaluate_main(path, classical_dir=str(cdir), unet_dir=str(udir), figures=False)
    md = (run_dir / "metrics.md").read_text(encoding="utf-8")
    assert "## 主表" in md and "U-Net" in md and "Frangi" in md
