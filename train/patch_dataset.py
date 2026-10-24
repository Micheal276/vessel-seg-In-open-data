"""U-Net 的数据层：整图预处理（缓存）+ 随机 patch 采样的 torch Dataset。

设计理由（面试讲解要点）
------------------------
1. 为什么用 patch 训练。
   训练图只有 12 张，整图训练每个 epoch 只有 12 个样本、batch 很小、BN 统计不稳。
   每张图随机裁 N 个 64×64 / 96×96 的 patch：样本数放大几百倍，每个 batch 都包含
   不同图、不同位置的血管，梯度更平稳；patch 小也省显存，Mac 上可以开较大的 batch。
   U-Net 是全卷积网络，训练在 patch 上、推理时在整图上滑窗，权重完全通用。

2. "每个 epoch 重新随机采样"。
   ``__getitem__`` 每次被调用都现抽一个随机位置 + 随机增强，所以每个 epoch 看到的 patch 都不同，
   相当于无限的数据流；``__len__`` = 图像数 × patches_per_image 只决定"一个 epoch 有多少步"。

3. 偏向 FOV 采样（fov_bias）。
   眼底图约 30% 的面积在 FOV 外，是纯黑背景，裁到那里的 patch 几乎没有信息。
   以概率 fov_bias 把 patch 中心取在一个随机的 FOV 像素上，其余时候在全图均匀取，
   既集中学习有效区域，也让网络见过 FOV 边界（推理时整图会经过那里）。OCTA 的 fov 全 1，此项无影响。

4. 预处理只做一次并缓存。
   绿通道 + CLAHE（复用 classical/preprocess.py，开关来自 config 的 ``unet.preprocess``）
   在构造时对每张整图做一次，之后只是裁剪，数据加载几乎不花时间——Mac 上不开多进程 worker 也不会成为瓶颈。
   CLAHE 必须在整图上做（它依赖 tile 级直方图），不能在 patch 上做，否则训练与推理的输入分布不一致。

5. 与数据集无关。
   输入是任意 ``VesselDataset``（DRIVE / ROSE / OCTA-500），通过 ``prepare_samples`` 统一成
   ``{image: H×W float32 [0,1], label, fov, name}``；patch 大小、数量、增强参数都来自 config。
   随机数来自自己持有的 ``np.random.Generator``（由 config 种子派生），不依赖全局状态。
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from classical.preprocess import preprocess
from eval.infer import to_chw_tensor
from train.augment import augment_sample


def prepare_samples(dataset, pre_cfg: dict) -> list:
    """对数据集里每张整图做预处理，返回 [{image: H×W float32, label, fov, name}, ...]。

    pre_cfg 与经典方法的 ``preprocess`` 小节格式相同（green_channel / clahe / denoise / fill_outside_fov）。
    """
    samples = []
    for i in range(len(dataset)):
        s = dataset[i]
        img = preprocess(s["image"], s["fov"], pre_cfg)["output"]
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        samples.append({"image": img, "label": s["label"], "fov": s["fov"], "name": s["name"]})
    return samples


class RandomPatchDataset(Dataset):
    """从预处理后的整图里随机裁 patch。

    参数
    ----
    samples           : prepare_samples 的输出
    patch_size        : patch 边长（建议 16 的倍数）
    patches_per_image : 每个 epoch 每张图采样多少个 patch
    fov_bias          : patch 中心落在 FOV 内的概率（其余时候全图均匀）
    augment_cfg       : train/augment.py 的参数；None 表示不增强
    seed              : 本 Dataset 随机数生成器的种子
    """

    def __init__(self, samples: list, patch_size: int, patches_per_image: int,
                 fov_bias: float = 0.8, augment_cfg: dict | None = None, seed: int = 0):
        self.samples = samples
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.fov_bias = float(fov_bias)
        self.augment_cfg = augment_cfg
        self.rng = np.random.default_rng(seed)

        for s in samples:
            h, w = s["label"].shape
            if h < self.patch_size or w < self.patch_size:
                raise ValueError(f"图像 {s['name']} 尺寸 {h}×{w} 小于 patch_size={self.patch_size}")
        # 预先存好每张图 FOV 像素的坐标，偏向 FOV 采样时直接随机取一个
        self.fov_coords = [np.argwhere(s["fov"] > 0) for s in samples]

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _sample_corner(self, idx_img: int):
        """返回 patch 左上角 (top, left)。"""
        h, w = self.samples[idx_img]["label"].shape
        p = self.patch_size
        coords = self.fov_coords[idx_img]
        if len(coords) > 0 and self.rng.random() < self.fov_bias:
            cy, cx = coords[self.rng.integers(len(coords))]
            top, left = int(cy) - p // 2, int(cx) - p // 2
        else:
            top, left = int(self.rng.integers(0, h - p + 1)), int(self.rng.integers(0, w - p + 1))
        # 中心靠近边界时把 patch 推回图内（不做填充，保证 patch 全是真实像素）
        return min(max(top, 0), h - p), min(max(left, 0), w - p)

    def __getitem__(self, index: int) -> dict:
        idx_img = index // self.patches_per_image
        s = self.samples[idx_img]
        top, left = self._sample_corner(idx_img)
        sl = (slice(top, top + self.patch_size), slice(left, left + self.patch_size))
        image, label, fov = s["image"][sl], s["label"][sl], s["fov"][sl]
        if self.augment_cfg is not None:
            image, label, fov = augment_sample(image, label, fov, self.rng, self.augment_cfg)
        return {
            "image": to_chw_tensor(image),
            "label": torch.from_numpy(np.ascontiguousarray(label, dtype=np.float32))[None],
            "fov": torch.from_numpy(np.ascontiguousarray(fov, dtype=np.float32))[None],
        }
