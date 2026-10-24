"""手写 U-Net（Ronneberger et al., 2015 的结构思路，按本项目需求精简）。

设计理由（面试讲解要点）
------------------------
1. 编码器-解码器 + 跳跃连接。
   编码器每层 (Conv3×3-BN-ReLU)×2 后 MaxPool 下采样一次，通道翻倍：感受野逐层变大，
   学到"这里是不是一根血管"的上下文语义，但空间分辨率越来越低。
   解码器用 ConvTranspose2d（可学习的 2× 上采样）逐层恢复分辨率，并把编码器同尺度的特征图
   沿通道拼接（skip concat）进来——细血管只有 1–2 像素宽，下采样后就丢了，跳跃连接把
   高分辨率的边缘/细节直接送回解码器，这是 U-Net 适合细长结构分割的关键。

2. 4 次下采样、基础通道 32（可配置）。
   通道 32→64→128→256→512（瓶颈）。原论文从 64 起步，参数约 3100 万；DRIVE 训练图只有 12 张，
   32 起步参数约 780 万，已足够、且在 Mac MPS 上训练更快、更不容易过拟合。
   4 次下采样后 96×96 的 patch 变成 6×6，瓶颈处的感受野覆盖整个 patch。

3. 用 padding=1 的卷积（"same" 卷积），输出与输入同尺寸。
   原论文用无 padding 卷积、输出比输入小、需要裁剪 skip 特征，拼图逻辑复杂；
   same 卷积让输出与输入逐像素对齐，滑窗推理拼回整图时最省心。

4. BatchNorm。
   让每层输入分布稳定，可用较大学习率、收敛快；卷积后接 BN 时卷积的 bias 是多余的
   （BN 会减掉均值），所以 bias=False。推理时 model.eval() 使用训练期累计的滑动均值/方差。

5. 输入尺寸不是 16 的倍数怎么办（例如整图 584×565）。
   4 次 2× 下采样要求 H、W 能被 2^4=16 整除，否则 MaxPool 向下取整后上采样回来的尺寸对不上，
   skip concat 会报错。这里在 forward 开头把输入右/下 **镜像填充** 到 16 的倍数，输出时再裁回原尺寸：
   镜像填充比补 0 更不容易在边界产生假边缘。这样网络对任意尺寸输入都能工作，
   滑窗推理也不必强制窗口大小（但训练 patch 仍建议取 16 的倍数，避免无谓填充）。

6. 输出 1 通道 logits（不加 sigmoid）。
   二分类分割只需一个通道；sigmoid 交给损失函数（BCEWithLogitsLoss 内部用 log-sum-exp，
   数值上比"先 sigmoid 再 BCE"稳定）和推理代码去做。

7. 与数据集无关：in_channels 可配置（眼底绿通道 / OCTA 灰度都是 1），阶段二不改本文件。
"""

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Sequential):
    """(Conv3×3 → BN → ReLU) × 2，尺寸不变，通道 in_ch → out_ch。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    """解码器的一层：ConvTranspose2d 2× 上采样（通道减半）→ 与 skip 拼接 → DoubleConv。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)  # 拼接后通道 = 上采样通道 + skip 通道

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net：depth 次下采样，基础通道 base_channels，输出 out_channels 个 logits 通道。

    参数
    ----
    in_channels   : 输入通道数（预处理后的单通道图为 1）
    out_channels  : 输出通道数（二值分割为 1）
    base_channels : 第一层通道数，之后每下采样一次翻倍
    depth         : 下采样次数（计划书要求 4）
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chs = [base_channels * 2 ** i for i in range(depth + 1)]  # 例：[32, 64, 128, 256, 512]

        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(DoubleConv(chs[i], chs[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.ups = nn.ModuleList(Up(chs[i + 1], chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    @classmethod
    def from_config(cls, unet_cfg: dict) -> "UNet":
        return cls(in_channels=unet_cfg.get("in_channels", 1),
                   out_channels=unet_cfg.get("out_channels", 1),
                   base_channels=unet_cfg.get("base_channels", 32),
                   depth=unet_cfg.get("depth", 4))

    def _pad_to_multiple(self, x: torch.Tensor):
        """右/下填充到 2^depth 的倍数；返回 (填充后的张量, 原始 H, 原始 W)。"""
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            # reflect 要求填充量 < 对应边长；极小输入时退回 replicate
            mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self._pad_to_multiple(x)

        x = self.inc(x)
        skips = [x]
        for down in self.downs:
            x = down(self.pool(x))
            skips.append(x)

        skips.pop()  # 最后一个是瓶颈本身，不作为 skip
        for up in self.ups:
            x = up(x, skips.pop())

        return self.head(x)[..., :h, :w]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
