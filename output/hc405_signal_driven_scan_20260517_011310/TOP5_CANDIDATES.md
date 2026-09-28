# HC #405 — TOP 5 SIGNAL-DRIVEN CANDIDATES

Generated 2026-05-17 01:15:10 ET

## Question

Of the deploy-eligible Optuna configs, which (if any) generate their P&L from REAL signal alpha (gross MFE within the hold window) rather than from the structural passive_+K entry credit?

## Methodology

- Replayed each candidate through canonical `full_market_replay` on 15-day extended OOT (`output/v3_3_extended_oot_20260514/extended_oot_predictions.npz`).
- Canonical commission = 0.376 RT ticks (HC #392). Per-trial spread used (not constant).
- For each filled trade, gross MFE = max favorable in-position log-ret across all available horizons <= hold_seconds, in ticks (same logic as `hc404_decomposition_study.py`).
- 'Signal-driven' = gross_MFE_mean_tk >= 1.0 AND pct_fills_MFE_ge_1tk high.

## Trial 278 reference

- Sharpe: 13.48 | n_fills: 195 | gross MFE mean: 0.408 tk | % fills with MFE>=1tk: 44.1%
- Entry edge: +2.0 tk (passive_+2 credit)
- HC #404 finding: only 0.41 tk of the +1.99 tk/fill comes from signal alpha; the rest is the structural credit.

## Top 5 candidates (ranked by signal-driven proportion + mean MFE)

### 1. `trial_001202_sharpe_34.898`

- **side**: long | **horizon**: 1s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 2.846s | **cancel_evals**: 66 | **conf_pctile**: 0.04319 | **min_pred_strength**: 0.2377
- **spread_ticks**: 0.7473 | **ToD**: 14:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **12.36** | tk/fill: **+2.921** | n_fills: **32** | day_conc: **0.471** | PF: 11.83 | WR: 78.1%
- HC #344 strict (day_conc<=0.20): FAIL

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **1.250 tk** | (median): 1.000 tk
- Gross MAE within hold (mean): 1.250 tk
- % of fills with gross MFE >= 1 tk: **78.1%**
- % of fills with gross MFE >= 2 tk: 34.4%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 38.5%

**Would replace trial 278?** NO

---

### 2. `trial_001194_sharpe_24.564`

- **side**: long | **horizon**: 10s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.817s | **cancel_evals**: 41 | **conf_pctile**: 0.05568 | **min_pred_strength**: 0.0243
- **spread_ticks**: 1.4577 | **ToD**: 14:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **10.78** | tk/fill: **+2.954** | n_fills: **47** | day_conc: **0.560** | PF: 16.41 | WR: 83.0%
- HC #344 strict (day_conc<=0.20): FAIL

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **1.064 tk** | (median): 1.000 tk
- Gross MAE within hold (mean): 1.064 tk
- % of fills with gross MFE >= 1 tk: **63.8%**
- % of fills with gross MFE >= 2 tk: 19.1%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 34.7%

**Would replace trial 278?** NO

---

### 3. `trial_001308_sharpe_22.014`

- **side**: long | **horizon**: 5s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 3.110s | **cancel_evals**: 56 | **conf_pctile**: 0.03850 | **min_pred_strength**: 0.6708
- **spread_ticks**: 0.6976 | **ToD**: 14:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **13.38** | tk/fill: **+2.555** | n_fills: **36** | day_conc: **0.279** | PF: 8.50 | WR: 83.3%
- HC #344 strict (day_conc<=0.20): FAIL

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **1.056 tk** | (median): 1.000 tk
- Gross MAE within hold (mean): 1.056 tk
- % of fills with gross MFE >= 1 tk: **63.9%**
- % of fills with gross MFE >= 2 tk: 25.0%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 34.5%

**Would replace trial 278?** NO

---

### 4. `trial_000708_sharpe_22.267`

- **side**: long | **horizon**: 5s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.066s | **cancel_evals**: 62 | **conf_pctile**: 0.04599 | **min_pred_strength**: 0.1671
- **spread_ticks**: 1.2064 | **ToD**: 14:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **13.92** | tk/fill: **+2.413** | n_fills: **38** | day_conc: **0.471** | PF: 13.22 | WR: 89.5%
- HC #344 strict (day_conc<=0.20): FAIL

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **1.039 tk** | (median): 1.000 tk
- Gross MAE within hold (mean): 1.039 tk
- % of fills with gross MFE >= 1 tk: **68.4%**
- % of fills with gross MFE >= 2 tk: 26.3%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 34.2%

**Would replace trial 278?** NO

---

### 5. `trial_000656_sharpe_24.777`

- **side**: long | **horizon**: 5s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.233s | **cancel_evals**: 55 | **conf_pctile**: 0.06332 | **min_pred_strength**: 0.6000
- **spread_ticks**: 1.1410 | **ToD**: 14:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **9.59** | tk/fill: **+2.327** | n_fills: **69** | day_conc: **0.483** | PF: 10.51 | WR: 81.2%
- HC #344 strict (day_conc<=0.20): FAIL

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **0.964 tk** | (median): 1.000 tk
- Gross MAE within hold (mean): 0.964 tk
- % of fills with gross MFE >= 1 tk: **50.7%**
- % of fills with gross MFE >= 2 tk: 21.7%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 32.5%

**Would replace trial 278?** NO

---
