# HC #405 — TOP 5 SIGNAL-DRIVEN CANDIDATES

Generated 2026-05-17 01:09:05 ET

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

### 1. `trial_000226_sharpe_24.331`

- **side**: short | **horizon**: 1s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.427s | **cancel_evals**: 75 | **conf_pctile**: 0.06770 | **min_pred_strength**: 0.4031
- **spread_ticks**: 0.6294 | **ToD**: 13:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **10.04** | tk/fill: **+2.012** | n_fills: **308** | day_conc: **0.170** | PF: 5.64 | WR: 80.8%
- HC #344 strict (day_conc<=0.20): PASS

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **0.500 tk** | (median): 0.500 tk
- Gross MAE within hold (mean): 0.500 tk
- % of fills with gross MFE >= 1 tk: **45.8%**
- % of fills with gross MFE >= 2 tk: 19.5%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 20.0%

**Would replace trial 278?** NO

---

### 2. `trial_000514_sharpe_24.482`

- **side**: short | **horizon**: 10s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.689s | **cancel_evals**: 68 | **conf_pctile**: 0.06012 | **min_pred_strength**: 0.0123
- **spread_ticks**: 0.7882 | **ToD**: 12:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **11.29** | tk/fill: **+2.048** | n_fills: **420** | day_conc: **0.190** | PF: 6.08 | WR: 81.2%
- HC #344 strict (day_conc<=0.20): PASS

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **0.455 tk** | (median): 0.500 tk
- Gross MAE within hold (mean): 0.455 tk
- % of fills with gross MFE >= 1 tk: **45.0%**
- % of fills with gross MFE >= 2 tk: 15.0%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 18.5%

**Would replace trial 278?** NO

---

### 3. `trial_000613_sharpe_24.463`

- **side**: short | **horizon**: 10s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.142s | **cancel_evals**: 46 | **conf_pctile**: 0.09932 | **min_pred_strength**: 0.5405
- **spread_ticks**: 0.6297 | **ToD**: 12:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **10.64** | tk/fill: **+1.891** | n_fills: **451** | day_conc: **0.196** | PF: 5.77 | WR: 80.9%
- HC #344 strict (day_conc<=0.20): PASS

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **0.453 tk** | (median): 0.500 tk
- Gross MAE within hold (mean): 0.453 tk
- % of fills with gross MFE >= 1 tk: **43.0%**
- % of fills with gross MFE >= 2 tk: 13.5%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 18.5%

**Would replace trial 278?** NO

---

### 4. `trial_000278_sharpe_27.220`

- **side**: short | **horizon**: 30s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.477s | **cancel_evals**: 79 | **conf_pctile**: 0.04354 | **min_pred_strength**: 0.0676
- **spread_ticks**: 0.7697 | **ToD**: 13:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **13.48** | tk/fill: **+1.991** | n_fills: **195** | day_conc: **0.186** | PF: 9.43 | WR: 81.5%
- HC #344 strict (day_conc<=0.20): PASS

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **0.408 tk** | (median): 0.500 tk
- Gross MAE within hold (mean): 0.408 tk
- % of fills with gross MFE >= 1 tk: **44.1%**
- % of fills with gross MFE >= 2 tk: 9.7%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 16.9%

**Would replace trial 278?** NO

---

### 5. `trial_000486_sharpe_25.311`

- **side**: short | **horizon**: 5s | **order_type**: `passive_at_touch_plus_2`
- **hold_seconds**: 1.197s | **cancel_evals**: 54 | **conf_pctile**: 0.07323 | **min_pred_strength**: 0.5474
- **spread_ticks**: 0.7865 | **ToD**: 13:00 - 15:00 ET

**Headline (15d OOT @ canonical 0.376 commission):**

- Sharpe: **10.88** | tk/fill: **+2.147** | n_fills: **285** | day_conc: **0.196** | PF: 6.23 | WR: 82.8%
- HC #344 strict (day_conc<=0.20): PASS

**Signal-alpha measurement:**

- Gross MFE within hold (mean): **0.386 tk** | (median): 0.500 tk
- Gross MAE within hold (mean): 0.386 tk
- % of fills with gross MFE >= 1 tk: **43.5%**
- % of fills with gross MFE >= 2 tk: 11.2%
- Entry edge: +2.00 tk (passive credit)
- Signal alpha share (MFE / (MFE + edge)): 16.2%

**Would replace trial 278?** NO

---
