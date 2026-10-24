"""数据集注册表：config 里写 ``data.name: drive``，代码按名字找到对应的适配类。

设计理由：训练 / 测试脚本只调用 ``build_dataset(cfg, split)``，不出现 ``DRIVEDataset`` 这样的具体类名。
阶段二接入 OCTA 时，只需新增 ``datasets/rose.py`` 并在下面的 ``DATASETS`` 里登记一行，
scripts/、train/、eval/、models/ 都不用改。每个适配类都实现 ``from_config(cfg, split)``。
"""

from datasets.drive import DRIVEDataset

DATASETS = {
    "drive": DRIVEDataset,
}


def build_dataset(cfg: dict, split: str):
    """按 cfg["data"]["name"] 构造对应 split 的数据集（未写 name 时默认 drive，兼容旧 config）。"""
    name = cfg["data"].get("name", "drive")
    if name not in DATASETS:
        raise ValueError(f"未注册的数据集 {name!r}；已注册: {sorted(DATASETS)}")
    return DATASETS[name].from_config(cfg, split)
