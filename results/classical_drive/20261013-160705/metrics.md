# 经典 baseline 指标（DRIVE，FOV 内）

指标为逐图 mean ± std（总体标准差）；Global 为所有图像素汇总后的整体指标。
全局阈值在 train split 上以逐图平均 Dice 最大为准则搜索，Frangi t=0.45，匹配滤波 t=0.28。

## test split（4 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7027 ± 0.0544 | 0.6869 ± 0.0901 | 0.9676 ± 0.0161 | 0.9322 ± 0.0117 | 0.9150 ± 0.0264 | 0.7042 | 0.9089 |
| Frangi | Global (searched on train), t=0.45 | 0.7018 ± 0.0562 | 0.6665 ± 0.1009 | 0.9726 ± 0.0132 | 0.9342 ± 0.0117 | 0.9150 ± 0.0264 | 0.7039 | 0.9089 |
| Matched filter | Otsu (per image) | 0.6474 ± 0.0647 | 0.5133 ± 0.0878 | 0.9925 ± 0.0029 | 0.9339 ± 0.0170 | 0.9374 ± 0.0192 | 0.6456 | 0.9119 |
| Matched filter | Global (searched on train), t=0.28 | 0.6734 ± 0.0623 | 0.7780 ± 0.1169 | 0.9329 ± 0.0406 | 0.9115 ± 0.0193 | 0.9374 ± 0.0192 | 0.6730 | 0.9119 |

## val split（4 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7342 ± 0.0288 | 0.7032 ± 0.0536 | 0.9682 ± 0.0098 | 0.9318 ± 0.0056 | 0.9221 ± 0.0156 | 0.7362 | 0.9207 |
| Frangi | Global (searched on train), t=0.45 | 0.7345 ± 0.0289 | 0.6902 ± 0.0538 | 0.9716 ± 0.0089 | 0.9331 ± 0.0057 | 0.9221 ± 0.0156 | 0.7366 | 0.9207 |
| Matched filter | Otsu (per image) | 0.6647 ± 0.0273 | 0.5219 ± 0.0418 | 0.9932 ± 0.0030 | 0.9288 ± 0.0107 | 0.9428 ± 0.0103 | 0.6643 | 0.9288 |
| Matched filter | Global (searched on train), t=0.28 | 0.7385 ± 0.0462 | 0.7533 ± 0.0484 | 0.9553 ± 0.0298 | 0.9271 ± 0.0188 | 0.9428 ± 0.0103 | 0.7364 | 0.9288 |

## train split（12 张）

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC | Global Dice | Global AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frangi | Otsu (per image) | 0.7059 ± 0.0816 | 0.7115 ± 0.0439 | 0.9535 ± 0.0428 | 0.9234 ± 0.0349 | 0.9183 ± 0.0177 | 0.6970 | 0.9170 |
| Frangi | Global (searched on train), t=0.45 | 0.7071 ± 0.0764 | 0.6952 ± 0.0486 | 0.9592 ± 0.0369 | 0.9265 ± 0.0300 | 0.9183 ± 0.0177 | 0.7008 | 0.9170 |
| Matched filter | Otsu (per image) | 0.6633 ± 0.0379 | 0.5519 ± 0.0475 | 0.9844 ± 0.0204 | 0.9303 ± 0.0146 | 0.9384 ± 0.0186 | 0.6609 | 0.8897 |
| Matched filter | Global (searched on train), t=0.28 | 0.6697 ± 0.0874 | 0.6661 ± 0.1542 | 0.9555 ± 0.0507 | 0.9178 ± 0.0357 | 0.9384 ± 0.0186 | 0.6635 | 0.8897 |
