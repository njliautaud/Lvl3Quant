# CROSS-MODEL VERDICT (HC #358g / HC #350)
Models compared: v2, v3_2

## 1. Cumulative-edge final PnL (Lorenz-like)
- v2: n_filled=31, cum_ticks_final=47.34
- v3_2: n_filled=291, cum_ticks_final=340.58

## 2. FOSD pairwise tests
- v2_vs_v3_2: **no_dominance (curves cross)** | mean_a=1.527 mean_b=1.170

## 3. Edge-decay AUC (P99 / passive)
- v2: total_auc=-1.197 | best: head=5s side=long auc=4.587
- v3_2: total_auc=7.329 | best: head=1s side=long auc=4.468

## 4. Head-importance (Δ-Sharpe ablation, top 3 per model)
- v2: meta-MLP AUC=0.5778
   - pred_log_ret_5s: ΔSharpe=307.997
   - pred_log_ret_1s: ΔSharpe=292.423
   - pred_log_ret_10s: ΔSharpe=-413.866
- v3_2: meta-MLP AUC=0.5793
   - pred_pred_realized_vol_30s_ticks: ΔSharpe=533.177
   - pred_log_ret_1s: ΔSharpe=349.936
   - pred_pred_mfe_30s_ticks: ΔSharpe=272.268

## 5. Realized PnL under each model's optimal strategy
### v2
- Sharpe: 4700.17 | Sortino: 6770.97 | PF: 9.05 | WR: 83.87%
- fills/attempts: 31/1629 (1.9%)
- PnL: 47.34 ticks  ≈  $592
- Max DD: 2.38 ticks | adverse-sel: -2.516 ticks
- Config: `--cnn-threshold 0.0687 --cnn-horizon 1s --side-bias short --cancel-eval-window 75 --passive-offset-ticks 1 --max-hold-seconds 1.0`
### v3_2
- Sharpe: 5376.67 | Sortino: 5936.95 | PF: 14.96 | WR: 91.41%
- fills/attempts: 291/5801 (5.0%)
- PnL: 340.58 ticks  ≈  $4257
- Max DD: 7.13 ticks | adverse-sel: -1.610 ticks
- Config: `--cnn-threshold 0.0231 --cnn-horizon 10s --side-bias short --cancel-eval-window 37 --passive-offset-ticks 1 --max-hold-seconds 1.0`

---

## STRONGEST MODEL FOR EXECUTION: **v3_2**
because v3_2 produces the highest realized $ PnL (4257) under its optimal strategy with positive Sharpe=5376.67.
