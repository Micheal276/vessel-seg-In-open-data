"""指标模块单元测试：用手工构造的小数组逐条验证公式。

设计理由：指标是所有实验结论的"尺子"，尺子错了，所有对比都没意义。
这里每个用例的期望值都在注释里手算给出，不依赖被测代码本身推导，
覆盖：各公式、FOV 外像素被忽略、自身比较全为 1、预测取反、分母为 0、逐图平均 vs 整体像素级。
"""

import math

import numpy as np
import pytest

from eval.metrics import (
    MetricAccumulator,
    auc_score,
    binarize,
    compute_metrics,
    confusion_counts,
    metrics_from_counts,
)

# ---------------------------------------------------------------------------
# 基础例子 A（2×4，全图有效）
#   label = 1 1 0 0      pred = 1 0 1 0
#           1 0 0 0             1 0 0 0
#   逐像素：(0,0) TP  (0,1) FN  (0,2) FP  (0,3) TN
#           (1,0) TP  (1,1) TN  (1,2) TN  (1,3) TN
#   => TP=2, FP=1, TN=4, FN=1
#   Dice = 2*2/(2*2+1+1) = 4/6 = 2/3
#   Sens = 2/(2+1) = 2/3
#   Spec = 4/(4+1) = 4/5
#   Acc  = (2+4)/8 = 3/4
# ---------------------------------------------------------------------------
LABEL_A = np.array([[1, 1, 0, 0],
                    [1, 0, 0, 0]], dtype=np.uint8)
PRED_A = np.array([[1, 0, 1, 0],
                   [1, 0, 0, 0]], dtype=np.uint8)


def test_confusion_counts_by_hand():
    assert confusion_counts(PRED_A, LABEL_A) == {"tp": 2, "fp": 1, "tn": 4, "fn": 1}


def test_formulas_by_hand():
    m = metrics_from_counts({"tp": 2, "fp": 1, "tn": 4, "fn": 1})
    assert m["dice"] == pytest.approx(2 / 3)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["accuracy"] == pytest.approx(3 / 4)


def test_counts_are_python_int():
    c = confusion_counts(PRED_A, LABEL_A)
    assert all(type(v) is int for v in c.values())


def test_bool_input_same_as_uint8():
    assert confusion_counts(PRED_A.astype(bool), LABEL_A.astype(bool)) == confusion_counts(PRED_A, LABEL_A)


# ---------------------------------------------------------------------------
# FOV：例子 A 加上 FOV
#   fov = 1 1 1 0
#         1 1 0 0
#   FOV 内像素：(0,0) TP (0,1) FN (0,2) FP (1,0) TP (1,1) TN
#   => TP=2, FP=1, TN=1, FN=1（FOV 外的 3 个 TN 被排除）
#   Dice = 4/6 = 2/3, Sens = 2/3, Spec = 1/2, Acc = 3/5
# ---------------------------------------------------------------------------
FOV_A = np.array([[1, 1, 1, 0],
                  [1, 1, 0, 0]], dtype=np.uint8)


def test_fov_restricts_counts():
    assert confusion_counts(PRED_A, LABEL_A, FOV_A) == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
    m = compute_metrics(PRED_A, LABEL_A, FOV_A)
    assert m["dice"] == pytest.approx(2 / 3)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_pixels_outside_fov_are_ignored():
    """把 FOV 外的 pred / label / score 改成任意值，所有指标（含 AUC）都不变。"""
    score = np.array([[0.9, 0.2, 0.6, 0.0],
                      [0.8, 0.1, 0.0, 0.0]], dtype=np.float32)
    base = compute_metrics(PRED_A, LABEL_A, FOV_A, score=score)

    outside = FOV_A == 0
    pred2, label2, score2 = PRED_A.copy(), LABEL_A.copy(), score.copy()
    pred2[outside] = 1            # 外面全判前景
    label2[outside] = 1           # 外面全标成血管
    score2[outside] = -100.0      # 外面给极端低分（若被计入会严重破坏 AUC）
    changed = compute_metrics(pred2, label2, FOV_A, score=score2)

    for k in ("dice", "sensitivity", "specificity", "accuracy", "auc", "tp", "fp", "tn", "fn"):
        assert changed[k] == pytest.approx(base[k]), k


# ---------------------------------------------------------------------------
# AUC
#   label = 0 0 1 1, score = 0.1 0.4 0.35 0.8
#   正负样本对共 2×2=4 对：
#     0.35 vs 0.1 胜；0.35 vs 0.4 负；0.8 vs 0.1 胜；0.8 vs 0.4 胜  => 3/4 = 0.75
#   全部同分 => 每对都算半胜 => 0.5
# ---------------------------------------------------------------------------
def test_auc_by_hand():
    label = np.array([0, 0, 1, 1], dtype=np.uint8)
    assert auc_score(np.array([0.1, 0.4, 0.35, 0.8]), label) == pytest.approx(0.75)
    assert auc_score(np.full(4, 0.5), label) == pytest.approx(0.5)


def test_auc_respects_fov():
    # 前 4 个像素同上例（AUC=0.75）；后 2 个在 FOV 外，分数故意给反（若计入会改变 AUC）
    label = np.array([0, 0, 1, 1, 1, 0], dtype=np.uint8)
    score = np.array([0.1, 0.4, 0.35, 0.8, 0.0, 1.0])
    fov = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
    assert auc_score(score, label, fov) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# label 与自身比较：TP=前景数，TN=背景数，FP=FN=0 => 四个比值均为 1；分数完美排序 => AUC=1
# ---------------------------------------------------------------------------
def test_label_vs_itself_all_ones():
    rng = np.random.default_rng(0)
    label = (rng.random((20, 20)) < 0.2).astype(np.uint8)
    fov = np.zeros((20, 20), dtype=np.uint8)
    fov[2:18, 2:18] = 1
    m = compute_metrics(label, label, fov, score=label.astype(np.float32))
    for k in ("dice", "sensitivity", "specificity", "accuracy", "auc"):
        assert m[k] == 1.0, k
    assert m["fp"] == 0 and m["fn"] == 0


# ---------------------------------------------------------------------------
# 预测取反：pred = 1 - label => TP=TN=0 => Dice=Sens=Spec=Acc=0；
# 分数 = 1 - label，每个正样本分数都低于每个负样本 => AUC=0
# ---------------------------------------------------------------------------
def test_inverted_prediction_all_zero():
    m = compute_metrics(1 - LABEL_A, LABEL_A, score=(1 - LABEL_A).astype(np.float32))
    assert (m["tp"], m["tn"]) == (0, 0)
    for k in ("dice", "sensitivity", "specificity", "accuracy", "auc"):
        assert m[k] == 0.0, k


# ---------------------------------------------------------------------------
# 分母为 0 的约定（见 eval/metrics.py docstring 第 6 条）
# ---------------------------------------------------------------------------
def test_zero_denominator_no_foreground_anywhere():
    # label、pred 都全 0（4 像素）：TP=FP=FN=0, TN=4
    #   Dice 分母 0 -> 约定 1.0；Sens 分母 TP+FN=0 -> NaN；Spec = 4/4 = 1；Acc = 4/4 = 1
    #   label 只有一类 -> AUC NaN
    z = np.zeros((2, 2), dtype=np.uint8)
    m = compute_metrics(z, z, score=np.zeros((2, 2), dtype=np.float32))
    assert m["dice"] == 1.0
    assert math.isnan(m["sensitivity"])
    assert m["specificity"] == 1.0
    assert m["accuracy"] == 1.0
    assert math.isnan(m["auc"])


def test_zero_denominator_all_foreground():
    # label、pred 都全 1：TP=4, 其余 0 -> Dice=1, Sens=1, Spec 分母 0 -> NaN, Acc=1, AUC NaN
    o = np.ones((2, 2), dtype=np.uint8)
    m = compute_metrics(o, o, score=np.ones((2, 2), dtype=np.float32))
    assert (m["dice"], m["sensitivity"], m["accuracy"]) == (1.0, 1.0, 1.0)
    assert math.isnan(m["specificity"])
    assert math.isnan(m["auc"])


def test_zero_denominator_false_positive_on_empty_label():
    # label 全 0，pred 有 1 个前景：TP=0, FP=1, TN=3, FN=0
    #   Dice = 0/(0+1+0) = 0（不是 1：预测有误报）；Sens NaN；Spec = 3/4；Acc = 3/4
    label = np.zeros((2, 2), dtype=np.uint8)
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    m = compute_metrics(pred, label)
    assert m["dice"] == 0.0
    assert math.isnan(m["sensitivity"])
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["accuracy"] == pytest.approx(3 / 4)


def test_empty_fov():
    # FOV 全 0：四个计数都是 0 -> Dice 约定 1.0，Sens/Spec/Acc 都 NaN，AUC NaN，AUC 不报错
    fov = np.zeros_like(LABEL_A)
    m = compute_metrics(PRED_A, LABEL_A, fov, score=PRED_A.astype(np.float32))
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (0, 0, 0, 0)
    assert m["dice"] == 1.0
    for k in ("sensitivity", "specificity", "accuracy", "auc"):
        assert math.isnan(m[k]), k


# ---------------------------------------------------------------------------
# 输入检查与 binarize
# ---------------------------------------------------------------------------
def test_non_binary_pred_raises():
    with pytest.raises(ValueError):
        confusion_counts(np.array([0.2, 0.7]), np.array([0, 1]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_counts(np.zeros(3, dtype=np.uint8), np.zeros(4, dtype=np.uint8))


def test_binarize_strictly_greater():
    # 0.5 不大于 0.5 -> 0
    assert binarize(np.array([0.2, 0.5, 0.51]), 0.5).tolist() == [0, 0, 1]


# ---------------------------------------------------------------------------
# 汇总：逐图平均 vs 整体像素级
#   图 1（1×4）：label 1 1 0 0, pred 1 1 0 0, score 0.9 0.8 0.1 0.2
#       TP=2 FP=0 TN=2 FN=0 -> Dice 1, Sens 1, Spec 1, Acc 1, AUC 1
#   图 2（1×8）：label 1 1 1 1 1 1 0 0, pred 全 0, score 0.3×6, 0.6, 0.7
#       TP=0 FP=0 TN=2 FN=6 -> Dice 0, Sens 0, Spec 1, Acc 2/8=0.25,
#       AUC：正样本都是 0.3，负样本 0.6/0.7，全输 -> 0
#   逐图平均：Dice 0.5, Sens 0.5, Spec 1, Acc 0.625, AUC 0.5；Dice 总体 std = 0.5
#   整体计数：TP=2 FP=0 TN=4 FN=6
#       Dice = 4/(4+0+6) = 0.4；Sens = 2/8 = 0.25；Spec = 4/4 = 1；Acc = 6/12 = 0.5
#   整体 AUC：正样本 {0.9,0.8,0.3×6}，负样本 {0.1,0.2,0.6,0.7}，共 8×4=32 对
#       0.9、0.8 各胜 4 对 = 8；每个 0.3 胜 0.1、0.2 两对，6×2 = 12；合计 20/32 = 0.625
# ---------------------------------------------------------------------------
def _two_image_accumulator():
    acc = MetricAccumulator()
    acc.update(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]),
               score=np.array([0.9, 0.8, 0.1, 0.2]), name="img1")
    acc.update(np.zeros(8, dtype=np.uint8), np.array([1, 1, 1, 1, 1, 1, 0, 0]),
               score=np.array([0.3] * 6 + [0.6, 0.7]), name="img2")
    return acc


def test_aggregator_mean_vs_global():
    s = _two_image_accumulator().summary()
    assert [m["name"] for m in s["per_image"]] == ["img1", "img2"]

    assert s["mean"]["dice"] == pytest.approx(0.5)
    assert s["mean"]["sensitivity"] == pytest.approx(0.5)
    assert s["mean"]["specificity"] == pytest.approx(1.0)
    assert s["mean"]["accuracy"] == pytest.approx(0.625)
    assert s["mean"]["auc"] == pytest.approx(0.5)
    assert s["std"]["dice"] == pytest.approx(0.5)

    g = s["global"]
    assert (g["tp"], g["fp"], g["tn"], g["fn"]) == (2, 0, 4, 6)
    assert g["dice"] == pytest.approx(0.4)
    assert g["sensitivity"] == pytest.approx(0.25)
    assert g["specificity"] == pytest.approx(1.0)
    assert g["accuracy"] == pytest.approx(0.5)
    assert g["auc"] == pytest.approx(0.625)

    # 两种口径确实不同
    assert s["mean"]["dice"] != pytest.approx(g["dice"])


def test_aggregator_skips_nan_in_mean():
    # 图 1：label、pred 都全 0 -> Sens NaN（不参与平均）
    # 图 2：label 1 1 0 0, pred 1 0 0 0 -> TP=1 FN=1 -> Sens 0.5
    # => Sens 均值 0.5，n_valid = 1；Spec 两张图都有定义，n_valid = 2
    acc = MetricAccumulator()
    acc.update(np.zeros(4, dtype=np.uint8), np.zeros(4, dtype=np.uint8))
    acc.update(np.array([1, 0, 0, 0]), np.array([1, 1, 0, 0]))
    s = acc.summary()
    assert s["mean"]["sensitivity"] == pytest.approx(0.5)
    assert s["n_valid"]["sensitivity"] == 1
    assert s["n_valid"]["specificity"] == 2


def test_aggregator_global_auc_nan_if_any_score_missing():
    acc = _two_image_accumulator()
    acc.update(np.array([1, 0]), np.array([1, 0]))   # 第三张没给 score
    s = acc.summary()
    assert math.isnan(s["global"]["auc"])
    assert s["n_valid"]["auc"] == 2                  # 逐图平均仍用有 score 的两张


def test_empty_accumulator_raises():
    with pytest.raises(ValueError):
        MetricAccumulator().summary()
