# 经典 baseline 指标（DRIVE，FOV 内）

指标为逐图 mean ± std（总体标准差）；Global 为所有图像素汇总后的整体指标。
全局阈值在 train split 上以逐图平均 Dice 最大为准则搜索，Frangi t=0.45，匹配滤波 t=0.21。

## train split（12 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7059 ± 0.0816 | 0.7115 ± 0.0439 | 0.9535 ± 0.0428 | 0.9234 ± 0.0349 | 0.9183 ± 0.0177 | 0.6970 | 0.9170 |
| Frangi | Global (searched on train), t=0.45 | 0.7071 ± 0.0764 | 0.6952 ± 0.0486 | 0.9592 ± 0.0369 | 0.9265 ± 0.0300 | 0.9183 ± 0.0177 | 0.7008 | 0.9170 |
| Matched filter | Otsu (per image) | 0.6577 ± 0.0356 | 0.5353 ± 0.0407 | 0.9875 ± 0.0143 | 0.9310 ± 0.0119 | 0.9379 ± 0.0178 | 0.6564 | 0.9184 |
| Matched filter | Global (searched on train), t=0.21 | 0.7030 ± 0.0626 | 0.6801 ± 0.0858 | 0.9649 ± 0.0265 | 0.9286 ± 0.0200 | 0.9379 ± 0.0178 | 0.7002 | 0.9184 |
