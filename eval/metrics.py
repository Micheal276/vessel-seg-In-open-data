"""分割指标模块：Dice / Sensitivity / Specificity / Accuracy（手写）+ AUC（sklearn）。

设计理由（面试讲解要点）
------------------------
1. 一切从混淆矩阵出发。
   四个阈值类指标都只是 TP / FP / TN / FN 四个计数的不同组合：

       Dice（= F1） = 2TP / (2TP + FP + FN)
       Sensitivity  = TP / (TP + FN)        # 血管被找出来的比例（召回率）
       Specificity  = TN / (TN + FP)        # 背景被正确判为背景的比例
       Accuracy     = (TP + TN) / (TP + FP + TN + FN)

   所以先写 ``confusion_counts``，再由计数算指标（``metrics_from_counts``）。
   这样"逐图平均"和"整体像素级"两种汇总方式能共用同一套公式，只是输入的计数不同。

2. 只在 FOV（视野）内计算。
   眼底图 FOV 外是一大片黑色背景，模型很容易全判为背景。若把它们算进去，
   TN 会被大量"白送"的像素撑大，Specificity 和 Accuracy 虚高（DRIVE 上 FOV 外约占 30% 像素）。
   因此所有函数都接受 ``fov`` 掩膜，只统计 FOV 内像素；``fov=None`` 表示整幅图都有效
   （例如 OCTA 没有 FOV 时）。

3. 计数用整数而不是浮点。
   全项目张量统一 float32，但 float32 只有 24 位尾数，累加到 ~1600 万（2^24）以后
   "+1" 会被舍掉；整体像素级统计要把多张图的计数相加，用 Python int 计数最稳妥，
   只在最后一步做一次除法得到比值。

4. 二值化是调用者的事。
   阈值类指标要求输入已经是 {0,1} 的二值预测：经典方法用 Otsu / 搜索阈值，
   U-Net 用 sigmoid > 0.5，阈值的选择本身就是方法的一部分，不应该藏在指标模块里。
   为了方便提供 ``binarize(score, threshold)``，但 ``confusion_counts`` 收到非二值输入会直接报错，
   防止"把概率图误当二值图"这种静默错误。

5. AUC 用连续分数。
   AUC 衡量的是"排序能力"：随机取一个血管像素和一个背景像素，前者分数更高的概率。
   它与阈值无关，所以必须喂连续响应（Frangi 响应、匹配滤波响应、U-Net 概率），
   而不是二值图。用 ``sklearn.metrics.roc_auc_score`` 计算（计划书允许，且实现经过充分验证）。

6. 分母为 0 的约定（明确写出来，避免各处行为不一致）：
   - Dice：若 2TP+FP+FN = 0，即金标准和预测在有效区域内都没有前景，二者完全一致，记为 1.0。
   - Sensitivity：若 TP+FN = 0（金标准没有前景），"找回了多少血管"无定义，记为 NaN。
   - Specificity：若 TN+FP = 0（金标准没有背景），无定义，记为 NaN。
   - Accuracy：若有效像素数为 0（FOV 为空），记为 NaN。
   - AUC：若有效区域内金标准只有一类（全前景或全背景），ROC 无定义，记为 NaN
     （sklearn 在这种情况下会抛异常，这里显式判断后返回 NaN）。
   NaN 而不是 0：0 会被当成"很差"拉低平均值，而 NaN 表示"这张图不提供该指标的信息"。
   汇总时逐图平均使用 nanmean（跳过 NaN），并报告每个指标实际参与平均的图像数。

7. 两种汇总方式都报告（计划书要求）。
   - 逐图平均（mean ± std）：每张图权重相同，反映"典型一张图"上的表现，也能看出图间波动。
   - 整体像素级（global）：把所有图的 TP/FP/TN/FN 相加再算指标，AUC 用所有图 FOV 内像素拼接后计算；
     每个像素权重相同，大 FOV / 血管多的图影响更大。
   两者一般不相等（比值的平均 ≠ 总和的比值），论文里两种口径都有，所以都给出。

8. 与数据集无关。
   本模块只认 numpy 数组（pred / label / fov / score），不知道 DRIVE 或 OCTA 的存在。
   经典方法、U-Net、阶段二的 OCTA 都直接调用，迁移时无需修改本文件。

用法::

    from eval.metrics import compute_metrics, MetricAccumulator, binarize

    acc = MetricAccumulator()
    for sample, prob in zip(dataset, probs):
        acc.update(binarize(prob, 0.5), sample["label"], sample["fov"], score=prob, name=sample["name"])
    report = acc.summary()   # {"per_image": [...], "mean": {...}, "std": {...}, "n_valid": {...}, "global": {...}}
"""

import numpy as np
from sklearn.metrics import roc_auc_score

# 5 个指标的固定顺序，打印 / 存 CSV 时都用它
METRIC_NAMES = ("dice", "sensitivity", "specificity", "accuracy", "auc")


# ---------------------------------------------------------------------------
# 输入检查
# ---------------------------------------------------------------------------

def _as_binary(arr, what: str) -> np.ndarray:
    """把 {0,1} 整数或 bool 数组转成 bool；含其它值则报错（提醒调用者先二值化）。"""
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr
    # 用"是否等于 0 或 1"判断，兼容 uint8 / int / 恰好是 0.0、1.0 的浮点数组
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{what} 必须是二值 {{0,1}} 数组；若是连续分数，请先用 binarize() 或自选阈值二值化")
    return arr.astype(bool)


def _valid_mask(shape, fov) -> np.ndarray:
    """返回有效像素的 bool 掩膜：fov=None 时整幅图有效。"""
    if fov is None:
        return np.ones(shape, dtype=bool)
    fov = _as_binary(fov, "fov")
    if fov.shape != tuple(shape):
        raise ValueError(f"fov 形状 {fov.shape} 与 label 形状 {tuple(shape)} 不一致")
    return fov


def _check_shape(a: np.ndarray, b: np.ndarray, what: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{what} 形状 {a.shape} 与 label 形状 {b.shape} 不一致")


def binarize(score, threshold: float) -> np.ndarray:
    """连续分数 -> {0,1} uint8 预测：score > threshold 为前景（严格大于）。"""
    return (np.asarray(score) > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# 混淆矩阵与阈值类指标
# ---------------------------------------------------------------------------

def confusion_counts(pred, label, fov=None) -> dict:
    """统计 FOV 内的 TP / FP / TN / FN，返回 Python int 的 dict。

    pred, label : 同形状的二值数组（bool 或 {0,1}）
    fov         : 同形状的二值掩膜，只统计 fov==1 的像素；None 表示全部像素
    """
    label = _as_binary(label, "label")
    pred = _as_binary(pred, "pred")
    _check_shape(pred, label, "pred")
    valid = _valid_mask(label.shape, fov)

    p = pred[valid]
    t = label[valid]
    # count_nonzero 返回 Python int，不存在 float32 累加精度问题
    return {
        "tp": int(np.count_nonzero(p & t)),
        "fp": int(np.count_nonzero(p & ~t)),
        "tn": int(np.count_nonzero(~p & ~t)),
        "fn": int(np.count_nonzero(~p & t)),
    }


def _ratio(num: int, den: int) -> float:
    """整数相除；分母为 0 返回 NaN（"无定义"）。"""
    return float(num) / float(den) if den > 0 else float("nan")


def metrics_from_counts(counts: dict) -> dict:
    """由 TP/FP/TN/FN 计算 Dice / Sensitivity / Specificity / Accuracy。

    分母为 0 的约定见模块 docstring 第 6 条：Dice 在双方都无前景时为 1.0，其余为 NaN。
    """
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    dice_den = 2 * tp + fp + fn
    return {
        "dice": 1.0 if dice_den == 0 else 2.0 * tp / dice_den,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
    }


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def _masked_scores(score, label, fov):
    """取出 FOV 内的 (label, score) 一维数组，供 AUC 使用。"""
    label = _as_binary(label, "label")
    score = np.asarray(score, dtype=np.float32)
    _check_shape(score, label, "score")
    valid = _valid_mask(label.shape, fov)
    return label[valid], score[valid]


def _auc_1d(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """一维 AUC；金标准只有一类时返回 NaN（sklearn 此时会抛 ValueError）。"""
    n_pos = int(np.count_nonzero(y_true))
    if n_pos == 0 or n_pos == y_true.size:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auc_score(score, label, fov=None) -> float:
    """FOV 内连续分数 score 相对二值 label 的 ROC AUC。"""
    y_true, y_score = _masked_scores(score, label, fov)
    return _auc_1d(y_true, y_score)


# ---------------------------------------------------------------------------
# 单图 + 多图汇总
# ---------------------------------------------------------------------------

def compute_metrics(pred, label, fov=None, score=None) -> dict:
    """单张图的全部 5 个指标（外加 4 个计数，便于排查）。

    score 为连续分数图（概率 / 滤波响应）；不提供时 AUC 记为 NaN，
    因为用二值图算 AUC 没有意义（只有一个工作点）。
    """
    counts = confusion_counts(pred, label, fov)
    result = metrics_from_counts(counts)
    result["auc"] = float("nan") if score is None else auc_score(score, label, fov)
    result.update(counts)
    return result


class MetricAccumulator:
    """逐张图喂入，最后同时给出"逐图平均"和"整体像素级"两种结果。

    - 逐图：每次 ``update`` 计算一次 ``compute_metrics`` 并保存。
    - 整体：累加整数计数；若提供 score，同时保存 FOV 内的 (label, score) 一维数组，
      最后拼接计算全局 AUC。DRIVE 每张图 FOV 内约 23 万像素，float32 存储几张到几十张图都没有内存压力。
      只要有一张图没给 score，全局 AUC 就记为 NaN（否则全局 AUC 只覆盖部分图像，口径不一致）。
    """

    def __init__(self):
        self.per_image = []
        self.total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self._labels = []
        self._scores = []
        self._all_have_score = True

    def update(self, pred, label, fov=None, score=None, name=None) -> dict:
        """加入一张图，返回该图的指标 dict。"""
        m = compute_metrics(pred, label, fov, score)
        m["name"] = name if name is not None else str(len(self.per_image))
        self.per_image.append(m)
        for k in self.total:
            self.total[k] += m[k]
        if score is None:
            self._all_have_score = False
        else:
            y_true, y_score = _masked_scores(score, label, fov)
            self._labels.append(y_true)
            self._scores.append(y_score)
        return m

    def summary(self) -> dict:
        """返回 {"per_image", "mean", "std", "n_valid", "global"}。

        mean / std 对每个指标用 nanmean / nanstd（总体标准差，ddof=0），跳过无定义的图；
        n_valid 记录每个指标实际参与平均的图像数。
        """
        if not self.per_image:
            raise ValueError("MetricAccumulator 为空：请先 update() 至少一张图")

        mean, std, n_valid = {}, {}, {}
        for k in METRIC_NAMES:
            vals = np.array([m[k] for m in self.per_image], dtype=np.float64)
            ok = ~np.isnan(vals)
            n_valid[k] = int(ok.sum())
            # 这里用 float64 只是对少量标量求均值，与张量的 float32 约定无关
            mean[k] = float(vals[ok].mean()) if ok.any() else float("nan")
            std[k] = float(vals[ok].std()) if ok.any() else float("nan")

        global_m = metrics_from_counts(self.total)
        if self._all_have_score:
            global_m["auc"] = _auc_1d(np.concatenate(self._labels), np.concatenate(self._scores))
        else:
            global_m["auc"] = float("nan")
        global_m.update(self.total)

        return {"per_image": self.per_image, "mean": mean, "std": std,
                "n_valid": n_valid, "global": global_m}


def format_summary(summary: dict) -> str:
    """把 summary 排成一张小表（逐图 mean±std 与整体像素级），用于终端打印。"""
    lines = [f"{'metric':<12} {'mean ± std (per-image)':>24} {'global (pixel)':>16}"]
    for k in METRIC_NAMES:
        lines.append(f"{k:<12} {summary['mean'][k]:>13.4f} ± {summary['std'][k]:<8.4f} "
                     f"{summary['global'][k]:>16.4f}")
    return "\n".join(lines)
