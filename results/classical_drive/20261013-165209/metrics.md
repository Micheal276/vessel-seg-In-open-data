# 经典 baseline 指标（DRIVE，FOV 内）

指标为逐图 mean ± std（总体标准差）；Global 为所有图像素汇总后的整体指标。
全局阈值在 train split 上以逐图平均 Dice 最大为准则搜索，Frangi t=0.45，匹配滤波 t=0.21。

## test split（4 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7027 ± 0.0544 | 0.6869 ± 0.0901 | 0.9676 ± 0.0161 | 0.9322 ± 0.0117 | 0.9150 ± 0.0264 | 0.7042 | 0.9089 |
| Frangi | Global (searched on train), t=0.45 | 0.7018 ± 0.0562 | 0.6665 ± 0.1009 | 0.9726 ± 0.0132 | 0.9342 ± 0.0117 | 0.9150 ± 0.0264 | 0.7039 | 0.9089 |
| Matched filter | Otsu (per image) | 0.6279 ± 0.0728 | 0.4856 ± 0.0911 | 0.9938 ± 0.0024 | 0.9322 ± 0.0176 | 0.9357 ± 0.0190 | 0.6283 | 0.9146 |
| Matched filter | Global (searched on train), t=0.21 | 0.6994 ± 0.0368 | 0.6748 ± 0.1295 | 0.9693 ± 0.0218 | 0.9327 ± 0.0103 | 0.9357 ± 0.0190 | 0.7026 | 0.9146 |

## val split（4 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7342 ± 0.0288 | 0.7032 ± 0.0536 | 0.9682 ± 0.0098 | 0.9318 ± 0.0056 | 0.9221 ± 0.0156 | 0.7362 | 0.9207 |
| Frangi | Global (searched on train), t=0.45 | 0.7345 ± 0.0289 | 0.6902 ± 0.0538 | 0.9716 ± 0.0089 | 0.9331 ± 0.0057 | 0.9221 ± 0.0156 | 0.7366 | 0.9207 |
| Matched filter | Otsu (per image) | 0.6526 ± 0.0268 | 0.5063 ± 0.0391 | 0.9937 ± 0.0030 | 0.9271 ± 0.0114 | 0.9410 ± 0.0120 | 0.6516 | 0.9175 |
| Matched filter | Global (searched on train), t=0.21 | 0.7213 ± 0.0387 | 0.7497 ± 0.0818 | 0.9500 ± 0.0338 | 0.9211 ± 0.0182 | 0.9410 ± 0.0120 | 0.7182 | 0.9175 |

## train split（12 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7059 ± 0.0816 | 0.7115 ± 0.0439 | 0.9535 ± 0.0428 | 0.9234 ± 0.0349 | 0.9183 ± 0.0177 | 0.6970 | 0.9170 |
| Frangi | Global (searched on train), t=0.45 | 0.7071 ± 0.0764 | 0.6952 ± 0.0486 | 0.9592 ± 0.0369 | 0.9265 ± 0.0300 | 0.9183 ± 0.0177 | 0.7008 | 0.9170 |
| Matched filter | Otsu (per image) | 0.6577 ± 0.0356 | 0.5353 ± 0.0407 | 0.9875 ± 0.0143 | 0.9310 ± 0.0119 | 0.9379 ± 0.0178 | 0.6564 | 0.9184 |
| Matched filter | Global (searched on train), t=0.21 | 0.7030 ± 0.0626 | 0.6801 ± 0.0858 | 0.9649 ± 0.0265 | 0.9286 ± 0.0200 | 0.9379 ± 0.0178 | 0.7002 | 0.9184 |
