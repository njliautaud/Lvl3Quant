# K=6 Megacap-Tech Meta-Classifier v1 — Findings

**Run date**: 2026-06-09
**Script**: `strategy/macro_picker/k6_meta_classifier_v1.py`
**MLflow experiment**: `k6_meta_classifier_v1` (id 201472772842653664) — 1 parent + 23 fold runs
**Outputs**: `output/macro_picker/k6_meta_classifier_v1/`
**Backlog item**: P2-2 (HC #589 A2 deliverable)
**Verdict**: **NEGATIVE — DO NOT DEPLOY**

---

## 1. Setup

- **Universe**: META, AVGO, TSLA, MSFT, AAPL, NVDA (K=6 megacap-tech rotation)
- **Baseline book**: `output/macro_picker/megacap_tech_extended_v1_20260609_171903/book_K6_mom60.parquet` (1505 trading days, 2020-01-02 -> 2025-12-31)
- **Features (18, all lagged 1d)**: SPY z-scores (20/60/200d), SPY RV20, VIX level + chg 5/10/20d, VIX term (VIX/VIX3M), DXY level + 20d mom, T10Y2Y + 30d chg (TNX-FVX proxy used; FRED timed out), K6 basket RV20 and 20d ret, sector breadth median/dispersion/positive-fraction across 9 sector ETFs.
- **Walk-forward**: SLIDING 24m train / 6m OOT / 3m step (HC #0) over 2018-2025 -> 23 folds.
- **Model**: LightGBM binary classifier (objective=binary, num_leaves=15, max_depth=4, lr=0.03, scale_pos_weight balanced per fold, early-stopping on last 10% of train).
- **Gate**: trade when P(K=6 day-t return > 0) >= 0.45, else hold cash (set daily_ret = 0).

### Script fix applied (minimum needed)

The dispatched code called `_load_prices()` from `megacap_tech_rotation` which only loads UNIVERSE + SPY/QQQ — the 9 sector ETFs (XLB/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY) were silently absent from the frame, leaving `sector_*` features all-NaN, which then caused `dropna(subset=feat_cols)` to drop EVERY row -> "no fold produced predictions" error.
Fix: added a fallback in `build_feature_matrix` to read sector ETFs directly from `prices_v2.parquet` when `_load_prices()` returns nothing for them. Zero architecture change.

### Other note

FRED `T10Y2Y` endpoint returned HTTP 504 -> code's fallback (`^TNX - ^FVX` 5Y proxy) was used. T10Y2Y feature importance is 2.5% gain share, so this fallback is unlikely to be material to the verdict.

---

## 2. Headline metrics — ungated vs gated

| Metric | Ungated K=6 baseline | Meta-gated (thr 0.45) | Delta |
|--------|----------------------|------------------------|-------|
| Sharpe | **2.34** | 1.43 | **-0.91** |
| Sortino | **2.85** | 1.37 | **-1.48** |
| Calmar | **5.36** | 1.11 | **-4.25** |
| CAGR | **105.3%** | 39.1% | **-66.2pp** |
| MaxDD | **-19.6%** | -35.2% | **-15.6pp WORSE** |
| Profit Factor | **1.68** | 1.49 | -0.19 |
| Win rate | **36.7%** | 22.5% | -14.2pp |
| Active days | 985 | 590 (59.9%) | -395 |
| Tail-DD pass (<=25%) | Yes | Yes | (just barely; quarter ref unchanged) |
| HC #428 R1 PASS | No | No | (regime imbalance in both) |

Both fail HC #428 R1 because the K=6 strategy itself is regime-asymmetric (green-day Sharpe ~10, red-day Sharpe ~-7). The gate does NOT close the regime gap — it shrinks both sides proportionally.

---

## 3. Per-regime Sharpe (SPY close-to-close classification)

| Regime | Days | Ungated Sharpe | Gated Sharpe | Ungated mean (bps) | Gated mean (bps) |
|--------|------|----------------|---------------|---------------------|-------------------|
| Green (SPY > +0.1%) | 749 | +9.86 | +7.03 | +124.7 | +70.8 |
| Red (SPY < -0.1%) | 607 | -7.38 | -6.12 | -81.3 | -55.1 |
| Flat | 149 | +1.85 | +2.17 | +14.5 | +14.0 |

Interpretation: the gate trims ~43% of green-day expectation, ~32% of red-day expectation, and leaves flat days roughly unchanged. It is NOT preferentially skipping the bad days — it is skipping too many good ones. Regime gap (|green - red| / max(|green|,|red|)) is essentially identical between ungated (1.75) and gated (1.87, slightly worse).

---

## 4. Gate-threshold sweep

| Threshold | Sharpe | Sortino | CAGR% | MaxDD% | Calmar | PF | WR% | Active days |
|-----------|--------|---------|-------|--------|--------|-----|------|-------------|
| BASELINE  | **2.34** | **2.85** | **105.3** | **-19.6** | **5.36** | **1.68** | **36.7** | 985 |
| 0.30 | 1.73 | 1.86 | 58.5 | -25.0 | 2.34 | 1.54 | 27.6 | 734 |
| 0.35 | 1.46 | 1.52 | 44.6 | -39.7 | 1.12 | 1.45 | 26.2 | 705 |
| 0.40 | 1.44 | 1.45 | 41.7 | -36.8 | 1.13 | 1.46 | 24.7 | 658 |
| **0.45 (spec)** | 1.43 | 1.37 | 39.1 | -35.2 | 1.11 | 1.49 | 22.5 | 590 |
| 0.50 | 1.13 | 0.99 | 25.9 | -48.5 | 0.53 | 1.41 | 17.8 | 477 |
| 0.55 | 0.81 | 0.57 | 13.9 | -28.5 | 0.49 | 1.37 | 11.3 | 300 |

Every threshold tried underperforms the ungated baseline on Sharpe, Sortino, Calmar, CAGR, and MaxDD. There is no threshold that makes this gate accretive.

---

## 5. Fold diagnostics (selected)

- AUC > 0.7 on 7 of 23 folds (2021Q4, 2022 all year, 2023Q3, 2025Q1, 2025Q2) — model CAN find signal in stressed regimes.
- AUC < 0.5 (worse than random) on 3 folds (2020Q2 0.26, 2021Q2 0.47, 2024Q1 0.56-ish).
- Early folds 0-3 (OOT 2020) trained on **pos_rate 0-5%** because the K=6 book begins 2020-01-02 -> for the 2018-2019 training spans there is essentially no label signal. These folds dominate the worst-AUC tail. Even with scale_pos_weight rebalancing the model is learning "is the strategy active" not "is it profitable."
- Top features by gain share: SPY z_ma60 (24.3%), SPY z_ma200 (18.2%), VIX level (14.0%), K6 basket 20d return (11.6%). The model is leaning on trend + volatility regime — sensible, but evidently overfit to the windows where late-2021/2022 selloffs had clean signatures.

---

## 6. Why the gate fails — structural diagnosis

1. **The K=6 baseline already has a high hit rate on its active days.** The book is 985 active days out of 1505; of those active days the strategy is positive on the majority. The classifier learning `sign(daily_ret_t)` over ALL days (including the ~520 inactive days where daily_ret == 0) sees pos_rate of 0.32 -> 0.46 across folds. Predicting P(pos) < 0.45 captures the inactive days correctly but also kills profitable active days.
2. **The target is too noisy at day-1 horizon.** Megacap day-to-day returns at this universe size are ~80% noise, ~20% structure. No macro feature predicts day-1 sign with AUC > 0.6 on average.
3. **The gate is symmetric — it cannot exploit the known K=6 asymmetry.** Red days are -81 bps mean ungated; green days are +125 bps. A useful gate would skip more red-day exposure than green-day exposure. This gate cuts both by similar ratios (and accidentally cuts more green than red in absolute terms), so net effect is purely return-suppressing.
4. **Compounding penalty.** Skipping ~40% of active days slashes CAGR by more than half because the geometric mean cost of sitting out winners outweighs the benefit of sitting out losers when WR is below 50%.

---

## 7. What to try instead

The verdict is NEGATIVE for `meta_gate at P<0.45` as specified. Possible next iterations (none authorized — listing for the backlog):

1. **Regression target instead of sign**. Train LGBM regressor on next-day return; gate when E[ret] < -0.001 (skip likely losers). Asymmetric gate matches the strategy's asymmetry.
2. **Regime-conditioned threshold**. Use a higher threshold (0.55) only on red SPY days and no gate on green days — exploit the known regime asymmetry directly.
3. **Multi-day horizon target**. Predict 5d forward return sign; current K=6 is a holding strategy, not daily-trade, so day-1 sign is too noisy.
4. **Drop bad early folds**. Restrict WF to folds where pos_rate_train >= 0.15 (fold 5+). Early folds inject garbage signal.
5. **Switch model family**. LGBM may be overfitting to the few late-2021/2022 regime shifts. Try a logistic regression on a 6-feature subset (SPY z200, VIX level, VIX term, sector breadth, k6 RV20, k6 20d ret) for a cleaner inductive bias.

None of (1)-(5) are launched. P2-2 is closed as NEGATIVE; a follow-up backlog item can be opened if any of these are deemed worth the compute.

---

## 8. Outputs

- `output/macro_picker/k6_meta_classifier_v1/feature_matrix.parquet` (1953 days x 18 feats)
- `output/macro_picker/k6_meta_classifier_v1/oot_predictions.parquet` (1505 OOT day predictions across 23 folds, dedup-first by date)
- `output/macro_picker/k6_meta_classifier_v1/gated_book.parquet` (K=6 book with p_pos / meta_gate / daily_ret_orig columns)
- `output/macro_picker/k6_meta_classifier_v1/feature_importance.csv`
- `output/macro_picker/k6_meta_classifier_v1/report.json`
- MLflow: parent run `83174635522449a1a545aec60f3f44f4` + 23 fold runs in experiment 201472772842653664

## 9. Live K=6 paper-trading state

UNTOUCHED. This was a research-only run; nothing was deployed.
