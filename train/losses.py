"""损失函数：BCEWithLogits + soft Dice 的加权组合（手写 Dice）。

设计理由（面试讲解要点）
------------------------
1. 为什么不只用 BCE。
   DRIVE 的 FOV 内血管像素只占约 12%，是典型的类别不平衡。BCE 对每个像素一视同仁，
   网络把"不确定的地方都猜背景"就能拿到很低的 BCE，结果细血管被大量漏检（Sensitivity 低）。

2. Dice loss 在做什么。
   Dice = 2|P∩G| / (|P| + |G|) 衡量预测与金标准的重叠，本身就是我们的评价指标。
   把二值预测换成 sigmoid 概率 p（"soft"），就得到可导的 soft Dice：

       soft_dice = (2·Σ p·g + s) / (Σ p + Σ g + s),   loss = 1 - soft_dice

   它只关心前景的重叠比例，与背景像素有多少无关，天然抵抗类别不平衡。
   s 是平滑项：防止分母为 0（patch 里没有血管时），也让梯度更平稳。
   这里在整个 batch 上求和后再算一个 Dice（而不是逐 patch 算再平均）：
   很多 patch 里血管极少甚至没有，逐 patch 的 Dice 会剧烈抖动。

3. 为什么组合而不是只用 Dice。
   纯 Dice loss 的梯度在训练初期不稳定、概率容易被推到 0/1 两端（校准差，AUC 受影响）；
   BCE 提供稳定的逐像素梯度。两者加权：loss = w_bce·BCE + w_dice·DiceLoss，权重在 config 里配。

4. 只在 FOV 内算（可选）。
   FOV 外是纯黑背景，网络很容易学会，计入损失只会稀释前景比例；指标也只在 FOV 内算，
   所以可以用 FOV 掩膜把 FOV 外像素排除在 Dice（以及可选的 BCE）之外。
   OCTA 没有 FOV 时 fov 全 1，结果等价于不加掩膜，代码不用改。

5. 全部 float32，不依赖具体数据集。
"""

import torch
import torch.nn.functional as F
from torch import nn


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                   smooth: float = 1.0) -> torch.Tensor:
    """1 - soft Dice（整个 batch 汇总计算）。logits/target/mask 形状相同 (N,1,H,W)。"""
    prob = torch.sigmoid(logits)
    target = target.float()
    if mask is not None:
        mask = mask.float()
        prob = prob * mask
        target = target * mask
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor | None = None) -> torch.Tensor:
    """BCEWithLogits；给 mask 时只对 mask==1 的像素求平均。"""
    per_pixel = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    if mask is None:
        return per_pixel.mean()
    mask = mask.float()
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)


class BCEDiceLoss(nn.Module):
    """loss = bce_weight · BCE + dice_weight · (1 - softDice)。

    参数
    ----
    bce_weight, dice_weight : 两项权重
    dice_in_fov : Dice 只在 FOV 内计算
    bce_in_fov  : BCE 只在 FOV 内计算
    smooth      : Dice 平滑项
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 dice_in_fov: bool = True, bce_in_fov: bool = False, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_in_fov = dice_in_fov
        self.bce_in_fov = bce_in_fov
        self.smooth = float(smooth)

    @classmethod
    def from_config(cls, loss_cfg: dict) -> "BCEDiceLoss":
        return cls(**(loss_cfg or {}))

    def forward(self, logits, target, fov=None) -> dict:
        """返回 {"loss", "bce", "dice_loss"}，后两项 detach 后用于记录。"""
        bce = masked_bce_with_logits(logits, target, fov if self.bce_in_fov else None)
        dice = soft_dice_loss(logits, target, fov if self.dice_in_fov else None, self.smooth)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return {"loss": loss, "bce": bce.detach(), "dice_loss": dice.detach()}
