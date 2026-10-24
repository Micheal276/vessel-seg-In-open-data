# 方法对比（test split，4 张，FOV 内）

指标为逐图 mean ± std（总体标准差）；Global 为所有图 FOV 内像素汇总后的整体指标。
主表阈值：经典方法用在 train split 上按逐图平均 Dice 搜索的全局阈值，U-Net 用 0.5；均未看 test。AUC 与阈值无关。

## 主表

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Global t=0.45 (searched on train) | 0.7018 ± 0.0562 | 0.6665 ± 0.1009 | 0.9726 ± 0.0132 | 0.9342 ± 0.0117 | 0.9150 ± 0.0264 | 0.7039 | 0.9089 |
| Matched filter | Global t=0.21 (searched on train) | 0.6994 ± 0.0368 | 0.6748 ± 0.1295 | 0.9693 ± 0.0218 | 0.9327 ± 0.0103 | 0.9357 ± 0.0190 | 0.7026 | 0.9146 |
| U-Net | t=0.5 | 0.8100 ± 0.0205 | 0.8006 ± 0.0738 | 0.9783 ± 0.0084 | 0.9558 ± 0.0066 | 0.9783 ± 0.0063 | 0.8109 | 0.9762 |

## 副表：Otsu（逐图）与 U-Net val 搜索阈值

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7027 ± 0.0544 | 0.6869 ± 0.0901 | 0.9676 ± 0.0161 | 0.9322 ± 0.0117 | 0.9150 ± 0.0264 | 0.7042 | 0.9089 |
| Matched filter | Otsu (per image) | 0.6279 ± 0.0728 | 0.4856 ± 0.0911 | 0.9938 ± 0.0024 | 0.9322 ± 0.0176 | 0.9357 ± 0.0190 | 0.6283 | 0.9146 |
| U-Net | t=0.38 (searched on val) | 0.8105 ± 0.0175 | 0.8218 ± 0.0677 | 0.9742 ± 0.0090 | 0.9549 ± 0.0054 | 0.9783 ± 0.0063 | 0.8119 | 0.9762 |

## 逐图 Dice（主表阈值）

| Method | 25 | 28 | 31 | 40 |
| --- | --- | --- | --- | --- |
| Frangi | 0.6322 | 0.7377 | 0.6641 | 0.7732 |
| Matched filter | 0.6446 | 0.7421 | 0.6892 | 0.7216 |
| U-Net | 0.7797 | 0.8376 | 0.8096 | 0.8131 |

## 补充分析（启发式区域，像素跨图汇总，主表阈值）

细血管 = 金标准局部半径 ≤ 1.5 像素（宽约 ≤ 3 像素）；视盘 = 启发式中心、半径 45 像素的圆（与 FOV 取交）。仅用于解释差异，不是正式指标。

| Method | Sens（细血管） | Sens（粗血管） | FP 总数 | 视盘圆内 FP | 视盘内 FP 占比 |
| --- | --- | --- | --- | --- | --- |
| Frangi | 0.5087 | 0.7140 | 22171 | 2256 | 0.102 |
| Matched filter | 0.4692 | 0.7457 | 24720 | 3537 | 0.143 |
| U-Net | 0.6136 | 0.8653 | 17466 | 936 | 0.054 |

细血管像素 31901，粗血管像素 76869。

## 来源与核对

- 经典方法：`results/classical_drive/20261013-165209`（连续响应 maps/test/*.npy）
- U-Net：`results/unet_drive/20261013-164103`（probs/*.npy；权重 `results/unet_drive_train/20261013-161717/best.pth`）
- 所有指标由 eval/metrics.py 从上述 .npy 与数据集 label/FOV 重新计算，并与来源 metrics.json 逐项（mean/std/global/逐图）比对，容差 1e-06：
  - frangi_global：最大绝对差 0.00e+00
  - matched_filter_global：最大绝对差 0.00e+00
  - unet_main：最大绝对差 0.00e+00
  - frangi_otsu：最大绝对差 0.00e+00
  - matched_filter_otsu：最大绝对差 0.00e+00
  - unet_best_val：最大绝对差 0.00e+00
