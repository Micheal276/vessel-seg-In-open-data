"""滑窗推理：把整图切成重叠的窗口逐块预测，再重叠平均拼回整图。验证（trainer）和测试（scripts/test.py）共用。

设计理由（面试讲解要点）
------------------------
1. 为什么滑窗而不是整图一次前向。
   网络是在 96×96 这类小 patch 上训练的，用同样大小的窗口推理，输入的"视野大小"与训练一致；
   窗口小也让显存占用可控，任意大的图（OCTA-500 的大图）都能处理。

2. 为什么要重叠 + 平均。
   卷积网络在窗口边缘只能看到"一半的邻域"（另一半是 padding），预测最不可靠；
   若窗口不重叠，拼接处会出现明显的接缝。stride < patch_size 让每个像素被多个窗口覆盖，
   对所有覆盖它的窗口的 **概率**（sigmoid 之后）求平均：边缘处的不可靠预测被窗口中心的可靠预测稀释，
   接缝消失，同时也相当于一种轻量的测试时集成。
   在概率空间而不是 logits 空间平均：概率有界、平均值仍可解释为概率，不会被个别极端 logits 主导。

3. 边界处理：先把整图镜像填充，使窗口能恰好覆盖（H、W 至少为窗口大小，且 (H - patch) 能被 stride 整除），
   拼完后再裁回原尺寸。镜像而不是补 0，避免在图像边界制造假的暗边。

4. 计数图归一化：每个像素累加了 k 个窗口的预测，除以覆盖次数 k 得到平均值；
   填充保证每个原图像素至少被一个窗口覆盖（k ≥ 1）。

5. 与数据集、模型结构都无关：输入 C×H×W 张量和任意"输入 N×C×p×p → 输出 N×1×p×p"的模型。
"""

import numpy as np
import torch
import torch.nn.functional as F


def to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    """H×W 或 H×W×C 的 float32 数组 -> C×H×W 的 float32 张量（训练 patch 与推理整图共用）。"""
    if image.ndim == 2:
        image = image[None]
    else:
        image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))


def _pad_amount(size: int, patch: int, stride: int) -> int:
    """使 size + pad >= patch 且 (size + pad - patch) % stride == 0 的最小 pad。"""
    if size <= patch:
        return patch - size
    return (-(size - patch)) % stride


def window_starts(size: int, patch: int, stride: int) -> list:
    """填充后的一维窗口起点列表（最后一个窗口恰好贴住末端）。"""
    return list(range(0, size - patch + 1, stride))


@torch.no_grad()
def sliding_window_predict(model: torch.nn.Module, image: torch.Tensor, patch_size: int, stride: int,
                           device: torch.device, batch_size: int = 16, apply_sigmoid: bool = True) -> np.ndarray:
    """对单张图做滑窗推理，返回 H×W float32 的概率图（apply_sigmoid=False 时为平均 logits）。

    image : C×H×W float32 张量（CPU 上即可，内部按 batch 搬到 device）
    """
    if stride <= 0 or stride > patch_size:
        raise ValueError(f"stride 应在 (0, patch_size] 内，实际 stride={stride}, patch_size={patch_size}")
    was_training = model.training
    model.eval()

    _, h, w = image.shape
    pad_h, pad_w = _pad_amount(h, patch_size, stride), _pad_amount(w, patch_size, stride)
    x = image[None]
    if pad_h or pad_w:
        mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    x = x[0]
    hp, wp = h + pad_h, w + pad_w

    corners = [(t, l) for t in window_starts(hp, patch_size, stride) for l in window_starts(wp, patch_size, stride)]
    prob_sum = torch.zeros((hp, wp), dtype=torch.float32)
    count = torch.zeros((hp, wp), dtype=torch.float32)

    for i in range(0, len(corners), batch_size):
        batch_corners = corners[i:i + batch_size]
        batch = torch.stack([x[:, t:t + patch_size, l:l + patch_size] for t, l in batch_corners])
        out = model(batch.to(device=device, dtype=torch.float32))
        if apply_sigmoid:
            out = torch.sigmoid(out)
        out = out[:, 0].float().cpu()
        for (t, l), pred in zip(batch_corners, out):
            prob_sum[t:t + patch_size, l:l + patch_size] += pred
            count[t:t + patch_size, l:l + patch_size] += 1.0

    if was_training:
        model.train()
    result = (prob_sum / count)[:h, :w]
    return result.numpy().astype(np.float32)


def predict_samples(model: torch.nn.Module, samples: list, infer_cfg: dict, device: torch.device) -> list:
    """对 prepare_samples 产出的每张整图做滑窗推理，返回与 samples 对应的概率图列表。

    infer_cfg: {"patch_size", "stride", "batch_size"}（config 的 ``infer`` 小节）
    """
    return [sliding_window_predict(model, to_chw_tensor(s["image"]), int(infer_cfg["patch_size"]),
                                   int(infer_cfg["stride"]), device, int(infer_cfg.get("batch_size", 16)))
            for s in samples]
