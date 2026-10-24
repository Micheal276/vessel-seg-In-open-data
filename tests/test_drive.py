"""DRIVE 数据集与划分逻辑的单元测试。

设计理由：数据划分一旦出错（split 重叠 = 测试集泄漏进训练），后面所有指标都失去意义，
而且很难从指标上看出来。所以用测试把"互不相交、大小正确、同种子同结果"固化下来。
需要真实数据的测试在 data/DRIVE 不存在时自动跳过（data/ 不入 git）。
"""

import numpy as np
import pytest

from datasets.base import check_sample, split_names
from datasets.drive import DRIVEDataset
from utils.io import load_config, resolve_path

CFG = load_config("configs/drive.yaml")
SIZES = CFG["data"]["split"]["sizes"]
SEED = CFG["data"]["split"]["seed"]
HAS_DATA = (resolve_path(CFG["data"]["root"]) / "training").is_dir()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="data/DRIVE 不存在")

NAMES = [str(i) for i in range(21, 41)]


def test_split_names_disjoint_and_sizes():
    splits = split_names(NAMES, SEED, SIZES)
    assert {s: len(v) for s, v in splits.items()} == {"train": 12, "val": 4, "test": 4}
    sets = [set(splits[s]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set().union(*sets) == set(NAMES)


def test_split_names_deterministic_and_order_independent():
    a = split_names(NAMES, SEED, SIZES)
    b = split_names(list(reversed(NAMES)), SEED, SIZES)   # 输入顺序不同
    np.random.seed(999)                                    # 全局随机状态不应影响划分
    c = split_names(NAMES, SEED, SIZES)
    assert a == b == c
    assert split_names(NAMES, SEED + 1, SIZES) != a        # 换种子应换划分


def test_split_names_rejects_oversize():
    with pytest.raises(ValueError):
        split_names(NAMES[:10], SEED, SIZES)


@needs_data
def test_drive_splits_match_config():
    ds = {s: DRIVEDataset.from_config(CFG, s) for s in ("train", "val", "test")}
    assert [len(ds[s]) for s in ("train", "val", "test")] == [12, 4, 4]
    assert ds["train"].names == split_names(NAMES, SEED, SIZES)["train"]
    all_names = ds["train"].names + ds["val"].names + ds["test"].names
    assert len(set(all_names)) == 20


@needs_data
def test_drive_sample_contract():
    ds = DRIVEDataset.from_config(CFG, "val")
    sample = ds[0]
    assert set(sample) == {"image", "label", "fov", "name"}
    assert sample["name"] == ds.names[0]

    image, label, fov = sample["image"], sample["label"], sample["fov"]
    assert image.shape == (584, 565, 3) and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    for arr in (label, fov):
        assert arr.shape == (584, 565) and arr.dtype == np.uint8
        assert set(np.unique(arr).tolist()) == {0, 1}

    # 血管像素应几乎全部在 FOV 内（实测 < 0.3%）
    outside = (label & (1 - fov)).sum() / label.sum()
    assert outside < 0.01


def test_invalid_split_raises():
    if not HAS_DATA:
        pytest.skip("data/DRIVE 不存在")
    with pytest.raises(ValueError):
        DRIVEDataset.from_config(CFG, "validation")


def test_split_names_rejects_undersize_unless_allowed():
    small = {"train": 10, "val": 4, "test": 4}             # 总和 18 < 20
    with pytest.raises(ValueError):
        split_names(NAMES, SEED, small)
    splits = split_names(NAMES, SEED, small, allow_unused=True)
    assert [len(splits[s]) for s in ("train", "val", "test")] == [10, 4, 4]


def test_split_names_rejects_bad_sizes():
    with pytest.raises(ValueError):
        split_names(NAMES, SEED, {"train": 16, "val": 4})          # 缺 test
    with pytest.raises(ValueError):
        split_names(NAMES, SEED, {"train": 17, "val": 4, "test": -1})


def _good_sample():
    return {"image": np.zeros((4, 5, 3), np.float32), "label": np.zeros((4, 5), np.uint8),
            "fov": np.ones((4, 5), np.uint8), "name": "x"}


def test_check_sample_raises_value_error():
    check_sample(_good_sample())                              # 合法样本不报错
    bad_cases = [
        lambda s: s.pop("fov"),
        lambda s: s.__setitem__("image", s["image"].astype(np.float64)),
        lambda s: s.__setitem__("image", s["image"] + 2.0),
        lambda s: s.__setitem__("label", s["label"].astype(np.int32)),
        lambda s: s.__setitem__("label", np.zeros((3, 5), np.uint8)),
        lambda s: s.__setitem__("fov", s["fov"] * 255),
    ]
    for mutate in bad_cases:
        s = _good_sample()
        mutate(s)
        with pytest.raises(ValueError):
            check_sample(s)
