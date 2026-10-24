# Progress Report — Retinal Vessel Segmentation (Phase 1: DRIVE)

Date: 2026-10-13 · Status: **Phase 1 complete; paused at Explanation checkpoint 1 (manual)**

All numbers below come from generated files under `results/` (paths given), and each task was
implemented by one sub-agent and independently re-checked by another.

## 1. What was built

| Step | What | Commit | Verified |
|---|---|---|---|
| §1 + §2 | uv + Python 3.11.17 env, torch 2.14.1 on **MPS**; `utils/device.py`, `utils/seed.py`; repo skeleton | `e16f436` | ✅ |
| 1.1 | DRIVE dataset (`datasets/base.py`, `datasets/drive.py`), deterministic split, config + IO utils | `7c33bca`, `4ef9b67` | ✅ |
| 1.3 | Metrics (Dice, Sens, Spec, Acc hand-written; AUC via sklearn), FOV-only, per-image mean + global | `f72c111` | ✅ (matches sklearn to 0.0) |
| 1.2 | Classical baseline: green → CLAHE → Gaussian; Frangi + hand-written matched filter (12 orientations); Otsu + train-searched threshold; train-only tuning script | `b833c0d`, `64b98b7` | ✅ |
| 1.4 | Hand-written U-Net (4 levels, base 32), patch sampling, augmentation, BCE+Dice loss, AdamW + cosine, early stopping, sliding-window inference | `617bae7`, `8d803c5` | ✅ (test re-run reproduced exactly) |
| 1.5 | Comparison table, ROC, Dice boxplot, error maps; README phase-1 results | `e81b4dc` | ✅ (independent recompute, diff < 1e-6) |
| Closeout | Removed reuse blockers for phase 2 (registry-only data loading, fundus-only analysis switchable, portability smoke test) | `e030855` | ✅ (DRIVE results byte-identical) |

Test suite: **91 passed**.

## 2. Deviation from the plan (user-approved)

The available DRIVE release has **no labels for the official test set**. The 20 labelled
training images were split with seed 2026:

- train (12): 21, 22, 23, 27, 29, 30, 32, 33, 34, 35, 36, 39
- val (4): 24, 26, 37, 38
- test (4): 25, 28, 31, 40

Consequence: test scores rest on 4 images and are **not comparable to published DRIVE results**.

## 3. Results (test split, metrics inside FOV)

Source: `results/compare_drive/20261013-170110/metrics.md`

| Method | Threshold | Dice | Sensitivity | Specificity | Accuracy | AUC |
|---|---|---|---|---|---|---|
| Frangi | global t=0.45 (searched on train) | 0.7018 ± 0.0562 | 0.6665 | 0.9726 | 0.9342 | 0.9150 |
| Matched filter | global t=0.21 (searched on train) | 0.6994 ± 0.0368 | 0.6748 | 0.9693 | 0.9327 | 0.9357 |
| **U-Net** | t=0.5 | **0.8100 ± 0.0205** | 0.8006 | 0.9783 | 0.9558 | 0.9783 |

Per-image Dice (25 / 28 / 31 / 40):
Frangi 0.632 / 0.738 / 0.664 / 0.773 · MF 0.645 / 0.742 / 0.689 / 0.722 · U-Net 0.780 / 0.838 / 0.810 / 0.813

U-Net training: early stop at epoch 69, best epoch 44 (val Dice 0.8308), 1374.8 s (~23 min) on MPS.
Plan targets met: classical Dice in 0.6–0.75 ✅, U-Net test Dice ≥ 0.78 ✅.

Figures: `docs/figures/` (roc, dice_boxplot, example_25, example_28, zoom_25, zoom_40).

## 4. Brief explanation

- **Classical methods** enhance tube-like dark structures. Frangi uses Hessian eigenvalues
  (one large, one small → a line); the matched filter correlates with a Gaussian vessel
  cross-section in 12 directions. Both reach Dice ≈ 0.70 but confuse the bright optic-disc rim
  and background texture with vessels, and miss thin vessels.
- **U-Net** learns these cues from data. It wins on every metric and every test image, mainly
  by recovering more thin vessels (thin-vessel recall 0.61 vs 0.51 / 0.47) and far fewer
  optic-disc false positives (936 vs 2256 / 3537 pixels). Its remaining errors are missed
  capillary tips and slightly over-wide thick vessels.
- **Why metrics only in the FOV**: outside the circular field of view everything is black and
  trivially "correct background", which would inflate Accuracy/Specificity.

## 5. Issues found and fixed along the way

- Frangi's default `gamma` made Otsu thresholding collapse (Dice ≈ 0.3) → fixed `gamma=0.006`.
- A dark rim inside the FOV produced a ring of false vessels → FOV eroded 5 px and refilled
  before filtering (metrics still on the original FOV).
- Classical tuning was originally ad hoc → replaced by a committed, train-only grid search
  (`classical/tune_baseline.py`); matched filter updated to σ=1.0, L=13.
- Training curve marked the wrong best epoch (52 vs saved 44) → fixed.
- Code that would have needed edits for OCTA (hard-coded DRIVE loading, fundus-only analysis
  crashing with no thick vessels) → fixed before phase 2 freezes `models/ train/ eval/`.

## 6. Known caveats

- 4-image test split → large variance (± ~0.02–0.06 Dice).
- Matched-filter tuned values sit at the grid edge; its val Dice dropped 0.739 → 0.721.
- MPS results are not guaranteed bit-reproducible (this run did reproduce exactly).
- Phase-2 limits: single-channel input only, CLAHE quantises to uint8, images must be ≥ patch
  size, val split must be non-empty, ROSE's official split must be handled inside its adapter.

## 7. Next steps (waiting on user)

1. **Explanation checkpoint 1** (manual): green channel, CLAHE, Frangi/Hessian, FOV-only
   metrics, patch training + sliding window, Dice loss, early stopping.
2. **Decide**: should Dice be NaN (instead of 1.0) for empty label + empty prediction / empty
   FOV? Last chance before `eval/metrics.py` is frozen.
3. **Task 2.1** (manual): obtain ROSE (preferred) or OCTA-500; then the agent lists its
   structure, adds `datasets/rose.py` + `configs/rose.yaml`, and runs Tasks 2.2–2.3.
