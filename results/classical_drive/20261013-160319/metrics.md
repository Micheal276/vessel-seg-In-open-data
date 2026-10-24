# 经典 baseline 指标（DRIVE，FOV 内）

指标为逐图 mean ± std（总体标准差）；Global 为所有图像素汇总后的整体指标。
全局阈值在 train split 上以逐图平均 Dice 最大为准则搜索，Frangi t=0.45，匹配滤波 t=0.27。

## test split（4 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.6717 ± 0.0557 | 0.6776 ± 0.0930 | 0.9581 ± 0.0161 | 0.9228 ± 0.0125 | 0.9014 ± 0.0266 | 0.6732 | 0.8952 |
| Frangi | Global (searched on train), t=0.45 | 0.6703 ± 0.0573 | 0.6645 ± 0.1000 | 0.9612 ± 0.0141 | 0.9240 ± 0.0125 | 0.9014 ± 0.0266 | 0.6723 | 0.8952 |
| Matched filter | Otsu (per image) | 0.5710 ± 0.0926 | 0.4517 ± 0.1087 | 0.9859 ± 0.0019 | 0.9210 ± 0.0200 | 0.9290 ± 0.0206 | 0.5731 | 0.9033 |
| Matched filter | Global (searched on train), t=0.27 | 0.6198 ± 0.0932 | 0.8177 ± 0.1054 | 0.8909 ± 0.0601 | 0.8790 ± 0.0390 | 0.9290 ± 0.0206 | 0.6131 | 0.9033 |

## val split（4 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7120 ± 0.0230 | 0.6986 ± 0.0521 | 0.9598 ± 0.0096 | 0.9240 ± 0.0081 | 0.9103 ± 0.0152 | 0.7130 | 0.9091 |
| Frangi | Global (searched on train), t=0.45 | 0.7121 ± 0.0232 | 0.6882 ± 0.0532 | 0.9625 ± 0.0099 | 0.9251 ± 0.0085 | 0.9103 ± 0.0152 | 0.7133 | 0.9091 |
| Matched filter | Otsu (per image) | 0.6205 ± 0.0483 | 0.4884 ± 0.0584 | 0.9875 ± 0.0015 | 0.9189 ± 0.0179 | 0.9359 ± 0.0103 | 0.6170 | 0.9144 |
| Matched filter | Global (searched on train), t=0.27 | 0.6947 ± 0.0682 | 0.7674 ± 0.0767 | 0.9292 ± 0.0524 | 0.9056 ± 0.0361 | 0.9359 ± 0.0103 | 0.6858 | 0.9144 |

## train split（12 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.6803 ± 0.0761 | 0.7045 ± 0.0443 | 0.9451 ± 0.0414 | 0.9152 ± 0.0336 | 0.9061 ± 0.0166 | 0.6729 | 0.9048 |
| Frangi | Global (searched on train), t=0.45 | 0.6809 ± 0.0728 | 0.6929 ± 0.0475 | 0.9494 ± 0.0364 | 0.9175 ± 0.0297 | 0.9061 ± 0.0166 | 0.6755 | 0.9048 |
| Matched filter | Otsu (per image) | 0.6028 ± 0.0501 | 0.4866 ± 0.0555 | 0.9824 ± 0.0110 | 0.9211 ± 0.0119 | 0.9305 ± 0.0182 | 0.6049 | 0.8911 |
| Matched filter | Global (searched on train), t=0.27 | 0.6567 ± 0.0644 | 0.6560 ± 0.1389 | 0.9542 ± 0.0331 | 0.9163 ± 0.0209 | 0.9305 ± 0.0182 | 0.6580 | 0.8911 |
