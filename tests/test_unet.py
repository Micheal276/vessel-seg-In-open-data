"""U-Net 相关模块（模型、增强、损失、滑窗推理、early stopping、patch 采样）的单元测试。

设计理由：这些模块的错误大多"不报错、只让指标变差"——增强时 label 与图像错位、Dice 公式写错、
滑窗拼接错位、early stopping 保存了错误的权重。用可预知结果的合成输入把正确行为固化下来。
测试都在 CPU 上跑，结果确定、速度快。
"""

import math

import numpy as np
import pytest
import torch
from torch import nn

from eval.infer import sliding_window_predict, to_chw_tensor, window_starts
from models.unet import UNet, count_parameters
from train.augment import adjust_brightness_contrast, apply_geometric, augment_sample
from train.early_stopping import EarlyStopping
from train.losses import BCEDiceLoss, soft_dice_loss
from train.patch_dataset import RandomPatchDataset
from train.trainer import cosine_lr

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h,w", [(64, 64), (96, 96), (61, 77), (37, 50), (17, 16)])
def test_unet_forward_shape_any_size(h, w):
    model = UNet(in_channels=1, base_channels=8, depth=4).eval()
    with torch.no_grad():
        y = model(torch.rand(2, 1, h, w))
    assert y.shape == (2, 1, h, w)
    assert y.dtype == torch.float32


def test_unet_in_channels_and_param_count():
    model = UNet(in_channels=3, base_channels=32, depth=4).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 3, 32, 32)).shape == (1, 1, 32, 32)
    # 32 起步、4 次下采样的 U-Net 参数量约 776 万（与 docstring 一致）
    assert 7.0e6 < count_parameters(model) < 8.5e6


# ---------------------------------------------------------------------------
# 增强
# ---------------------------------------------------------------------------

def _pattern(h=40, w=48):
    """非对称的合成图：label 是 image>0.5 的位置，任何错位都会破坏这个对应关系。"""
    rng = np.random.default_rng(1)
    image = rng.random((h, w)).astype(np.float32)
    label = (image > 0.5).astype(np.uint8)
    fov = np.ones((h, w), np.uint8)
    fov[:, :5] = 0
    return image, label, fov


def test_geometric_keeps_alignment_exact():
    image, label, fov = _pattern()
    for hflip in (False, True):
        for vflip in (False, True):
            for k in range(4):
                p = {"hflip": hflip, "vflip": vflip, "k90": k, "angle": 0.0}
                img2, lab2 = apply_geometric(image, p, False), apply_geometric(label, p, True)
                np.testing.assert_array_equal(lab2, (img2 > 0.5).astype(np.uint8))


def test_augment_rotation_keeps_label_binary_and_aligned():
    """小角度旋转：label 用最近邻保持二值；用一条粗血管验证旋转后图像与标签仍重合。"""
    h = w = 64
    image = np.full((h, w), 0.8, np.float32)
    image[:, 28:36] = 0.2                     # 暗竖条 = 血管
    label = (image < 0.5).astype(np.uint8)
    fov = np.ones((h, w), np.uint8)
    cfg = {"hflip": 0.5, "vflip": 0.5, "rot90": True, "rotate_prob": 1.0, "rotate_deg": 15.0,
           "brightness": 0.0, "contrast": 0.0}
    rng = np.random.default_rng(0)
    for _ in range(10):
        img2, lab2, fov2 = augment_sample(image, label, fov, rng, cfg)
        assert img2.dtype == np.float32 and lab2.dtype == np.uint8
        assert set(np.unique(lab2)) <= {0, 1} and set(np.unique(fov2)) <= {0, 1}
        # 远离插值边缘的像素：图像暗处必须是血管，亮处必须是背景
        assert np.all(lab2[img2 < 0.3] == 1)
        assert np.all(lab2[img2 > 0.7] == 0)


def test_brightness_contrast_only_image_and_clipped():
    image, label, fov = _pattern()
    out = adjust_brightness_contrast(image, 0.5, 1.5)
    assert out.dtype == np.float32 and out.min() >= 0.0 and out.max() <= 1.0
    cfg = {"hflip": 0.0, "vflip": 0.0, "rot90": False, "rotate_prob": 0.0, "brightness": 0.2, "contrast": 0.2}
    img2, lab2, fov2 = augment_sample(image, label, fov, np.random.default_rng(0), cfg)
    np.testing.assert_array_equal(lab2, label)
    np.testing.assert_array_equal(fov2, fov)


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------

def test_dice_loss_perfect_prediction_near_zero():
    target = (torch.rand(2, 1, 32, 32) > 0.8).float()
    logits = (target * 2 - 1) * 50.0                # 完美且极其自信的预测
    assert soft_dice_loss(logits, target).item() < 1e-4


def test_dice_loss_worst_prediction_near_one():
    target = (torch.rand(2, 1, 32, 32) > 0.8).float()
    logits = (1 - target * 2) * 50.0                # 完全相反
    assert soft_dice_loss(logits, target, smooth=1e-6).item() > 0.999


def test_dice_loss_fov_mask_ignores_outside():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :4] = 1
    fov = torch.zeros(1, 1, 8, 8)
    fov[..., :4, :] = 1
    logits = (target * 2 - 1) * 50.0
    logits[..., 6:, 6:] = 50.0                      # FOV 外的假阳性
    assert soft_dice_loss(logits, target, fov).item() < 1e-4
    assert soft_dice_loss(logits, target, None).item() > 0.05


def test_bce_dice_combination_weights():
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    logits = torch.randn(1, 1, 16, 16)
    out = BCEDiceLoss(bce_weight=2.0, dice_weight=0.5, dice_in_fov=False)(logits, target)
    expected = 2.0 * nn.functional.binary_cross_entropy_with_logits(logits, target) + 0.5 * soft_dice_loss(logits, target)
    assert torch.allclose(out["loss"], expected)


# ---------------------------------------------------------------------------
# 滑窗推理
# ---------------------------------------------------------------------------

class Identity(nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("h,w,patch,stride", [(50, 63, 16, 8), (64, 64, 32, 32), (20, 70, 32, 12), (10, 10, 16, 4)])
def test_sliding_window_reconstructs_identity(h, w, patch, stride):
    image = torch.rand(1, h, w)
    out = sliding_window_predict(Identity(), image, patch, stride, CPU, batch_size=5, apply_sigmoid=False)
    assert out.shape == (h, w)
    np.testing.assert_allclose(out, image[0].numpy(), atol=1e-6)


def test_sliding_window_sigmoid_and_restores_train_mode():
    model = Identity().train()
    image = torch.randn(1, 30, 30)
    out = sliding_window_predict(model, image, 16, 8, CPU)
    np.testing.assert_allclose(out, torch.sigmoid(image[0]).numpy(), atol=1e-6)
    assert model.training


def test_window_starts_cover_end():
    assert window_starts(64, 32, 16) == [0, 16, 32]


def test_to_chw_tensor():
    assert to_chw_tensor(np.zeros((5, 6), np.float32)).shape == (1, 5, 6)
    assert to_chw_tensor(np.zeros((5, 6, 3), np.float32)).shape == (3, 5, 6)


# ---------------------------------------------------------------------------
# early stopping
# ---------------------------------------------------------------------------

def test_early_stopping_logic_and_best_weights(tmp_path):
    model = nn.Linear(2, 1)
    es = EarlyStopping(patience=2, min_delta=0.01, mode="max", ckpt_path=tmp_path / "best.pth")
    values = [0.50, 0.60, 0.605, 0.62, 0.61, 0.615]
    improved = []
    for epoch, v in enumerate(values, start=1):
        with torch.no_grad():
            model.weight.fill_(v)                     # 权重 = 当前 epoch 的指标，便于检查保存的是哪个
        improved.append(es.step(v, model, epoch))
        if es.should_stop:
            break
    # 0.605 比 0.60 只高 0.005 < min_delta，不算提升；0.62 提升；之后连续 2 次无提升 -> 停
    assert improved == [True, True, False, True, False, False]
    assert es.should_stop and es.best_epoch == 4 and math.isclose(es.best_value, 0.62)
    with torch.no_grad():
        model.weight.fill_(0.0)
    es.restore_best(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.62))
    saved = torch.load(tmp_path / "best.pth", weights_only=False)
    assert saved["epoch"] == 4


def test_early_stopping_min_mode_and_nan():
    es = EarlyStopping(patience=1, mode="min")
    assert es.step(1.0) and not es.step(float("nan")) and es.should_stop


# ---------------------------------------------------------------------------
# patch 采样与学习率
# ---------------------------------------------------------------------------

def test_random_patch_dataset_shapes_and_fov_bias():
    h, w = 80, 90
    fov = np.zeros((h, w), np.uint8)
    fov[20:60, 30:70] = 1
    sample = {"image": np.random.default_rng(0).random((h, w)).astype(np.float32),
              "label": fov.copy(), "fov": fov, "name": "x"}
    ds = RandomPatchDataset([sample, sample], patch_size=16, patches_per_image=50, fov_bias=1.0,
                            augment_cfg=None, seed=0)
    assert len(ds) == 100
    item = ds[3]
    assert item["image"].shape == (1, 16, 16) and item["label"].shape == (1, 16, 16)
    # fov_bias=1 时 patch 中心必在 FOV 内 -> patch 中心像素的 fov 为 1
    assert all(ds[i]["fov"][0, 8, 8] == 1 for i in range(50))


def test_cosine_lr_schedule():
    assert math.isclose(cosine_lr(0, 100, 1.0, warmup_steps=10), 0.1)
    assert math.isclose(cosine_lr(9, 100, 1.0, warmup_steps=10), 1.0)
    assert math.isclose(cosine_lr(10, 100, 1.0, warmup_steps=10), 1.0)
    assert math.isclose(cosine_lr(100, 100, 1.0, warmup_steps=10, min_lr=0.01), 0.01)
    assert cosine_lr(55, 100, 1.0, warmup_steps=10) == pytest.approx(0.5)
