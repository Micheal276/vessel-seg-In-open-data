"""U-Net 训练入口：读 config → 构建数据集（按注册名）→ 预处理 → patch 采样 → 训练 + early stopping。

设计理由：脚本只负责"组装"，所有逻辑在 datasets/ train/ models/ eval/ 里；数据集由
config 的 ``data.name`` 经注册表选择，因此 DRIVE 与阶段二的 OCTA 共用这一个脚本，只换 --config。

用法::

    python -m scripts.train --config configs/drive.yaml                 # 完整训练（train 训练，val 早停）
    python -m scripts.train --config configs/drive.yaml --overfit 2     # 过拟合测试：train 前 2 张，无增强，
                                                                        # 并在这 2 张上验证
    python -m scripts.train --config configs/drive.yaml --epochs 5      # 覆盖最大 epoch 数（冒烟测试）

结果：``results/unet_<experiment>_train/<时间戳>/``（过拟合测试为 ``..._overfit``），含
config.json、history.json/csv、curves.png、train_summary.json、best.pth（不入 git）。

best.pth 除权重外还记录 ``unet_cfg``（重建模型）、``preprocess``（unet.preprocess 小节）、``data_name``
与完整 ``config``：scripts/test.py 用它们检查测试时的预处理是否与训练时一致
（预处理不一致是"不报错、只让指标变差"的典型错误）。
"""

import argparse

import torch
from torch.utils.data import DataLoader

from utils.device import get_device  # 先导入：在 import torch 前设置 MPS 回退环境变量
from datasets import build_dataset
from models.unet import UNet, count_parameters
from train.patch_dataset import RandomPatchDataset, prepare_samples
from train.trainer import Trainer
from utils.io import load_config, make_run_dir, save_json
from utils.seed import set_seed


def main(config_path: str, overfit: int = 0, epochs: int | None = None) -> dict:
    cfg = load_config(config_path)
    tcfg = cfg["train"]
    set_seed(int(tcfg["seed"]))
    device = get_device()

    pre_cfg = cfg["unet"]["preprocess"]
    train_samples = prepare_samples(build_dataset(cfg, "train"), pre_cfg)
    augment_cfg = tcfg.get("augment")
    patches_per_image = int(tcfg["patches_per_image"])
    max_epochs = int(tcfg["epochs"])
    early_cfg = dict(tcfg.get("early_stopping", {}))

    if overfit > 0:
        # 过拟合测试：少量图、不增强、在同一批图上"验证"，检验模型+损失+训练循环能否把训练集拟合好
        train_samples = train_samples[:overfit]
        val_samples = train_samples
        augment_cfg = None
        patches_per_image = int(tcfg["overfit"]["patches_per_image"])
        max_epochs = int(tcfg["overfit"]["epochs"])
        early_cfg["patience"] = max_epochs + 1  # 过拟合测试不早停
        run_name = f"unet_{cfg['experiment']}_overfit"
    else:
        val_samples = prepare_samples(build_dataset(cfg, "val"), pre_cfg)
        run_name = f"unet_{cfg['experiment']}_train"
    if epochs is not None:
        max_epochs = int(epochs)

    run_dir = make_run_dir(run_name)
    save_json({"config": cfg, "overfit": overfit, "max_epochs": max_epochs, "patches_per_image": patches_per_image,
               "train_names": [s["name"] for s in train_samples], "val_names": [s["name"] for s in val_samples]},
              run_dir / "config.json")
    print(f"[train] 结果目录: {run_dir}")
    print(f"[train] train: {[s['name'] for s in train_samples]}  val: {[s['name'] for s in val_samples]}")

    patch_ds = RandomPatchDataset(train_samples, tcfg["patch_size"], patches_per_image,
                                  fov_bias=tcfg.get("fov_bias", 0.8), augment_cfg=augment_cfg,
                                  seed=int(tcfg["seed"]))
    loader = DataLoader(patch_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=True,
                        num_workers=0, generator=torch.Generator().manual_seed(int(tcfg["seed"])))

    model = UNet.from_config(cfg["unet"])
    print(f"[train] UNet 参数量: {count_parameters(model):,}")

    tcfg_run = {**tcfg, "early_stopping": early_cfg}
    trainer = Trainer(model, loader, val_samples, tcfg_run, cfg["infer"], device, run_dir,
                      ckpt_extra={"unet_cfg": {k: v for k, v in cfg["unet"].items() if k != "preprocess"},
                                  # 预处理与数据集名单独存一份：测试时据此检查"推理预处理 == 训练预处理"
                                  "preprocess": pre_cfg, "data_name": cfg["data"].get("name", "drive"),
                                  "config": cfg})
    return trainer.fit(max_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练手写 U-Net")
    parser.add_argument("--config", default="configs/drive.yaml")
    parser.add_argument("--overfit", type=int, default=0, help="过拟合测试：只用 train 前 N 张图（0 = 关闭）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 config 里的最大 epoch 数")
    args = parser.parse_args()
    main(args.config, args.overfit, args.epochs)
