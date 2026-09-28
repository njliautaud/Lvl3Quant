# HC #429 R4 — Best Model At Confidence (CONDITIONAL MFE/MAE leaderboard)

_Generated: 2026-05-19 ~13:30 ET (Session continuation)_
_Source: `output/hc429_native_mfe_matrix_combined.csv`_
_Cost: passive_at_touch + commission = 0.376 tk_
_Confidence ranking: |prediction| → top N% selection; side from sign_

## TOP 10 cells by net/fill @ top 0.5% confidence (PRE-FIFO, pre-HC428 gates)

| Rank | Model  | Horizon | Side  | MFE_top05 | MAE_top05 | **net_top05** | WR%   | n_top05 |
|:---:|--------|--------|-------|----------:|----------:|--------------:|------:|--------:|
|  1  | v3.4.2 | 1s     | long  | 1.381     | 0.291     | **+1.005**    | 68.2  | 1206    |
|  2  | v3.4.2 | 5s     | long  | 1.246     | 1.166     | **+0.870**    | 58.2  | 1203    |
|  3  | v3.4.2 | 1s     | short | 1.061     | 0.293     | **+0.685**    | 64.6  | 1206    |
|  4  | v3.4.2 | 10s    | long  | 1.067     | 1.937     | **+0.691**    | 54.7  | 1201    |
|  5  | v3.3   | 5s     | short | 1.003     | 0.887     | **+0.627**    | 55.5  | 1203    |
|  6  | v2     | 10s    | short | 0.999     | 1.665     | **+0.623**    | 61.2  | 9065    |
|  7  | v2     | 1s     | short | 0.956     | 0.569     | **+0.580**    | 71.9  | 8008    |
|  8  | v3.3   | 1s     | short | 0.942     | 0.374     | **+0.566**    | 60.2  | 1206    |
|  9  | v3.4.2 | 5s     | short | 0.880     | 0.886     | **+0.504**    | 58.4  | 1203    |
| 10  | v3.3   | 5s     | long  | 0.748     | 0.702     | **+0.372**    | 52.0  | 1203    |

**Net = MFE − MAE − 0.376** (passive-at-touch cost). Sub-MAE-symmetric net means tail capture must dominate WR for positive expectancy.

## OBSERVATIONS

1. **v3.4.2 dominates the conditional-edge pre-FIFO ranking** across 1s/5s/10s on the **LONG** side (top three slots). 
2. **WR > 60% only on 1s horizons** for all models. By 10s, WR collapses to ~55% even at top 0.5%.
3. **Max conditional MFE in the entire matrix = 1.381 ticks** (v3.4.2 / 1s / long). 
   **The Optuna sweep default TP = 8 is ~6× the max conditional MFE observed.** HC #429 R1 confirmed: TP grid must be re-anchored to `[1, 2, 3]` ticks, NOT `[2, 8]`.
4. **v2 (previous validated champion via HC #413) is no longer the conditional-MFE leader** — v3.4.2 dominates 4 of top 4 cells. But: v2 is the only model with full FIFO + HC408 + HC415 + day-conc validation on 36-day OOT. The v3.4.2 leaders need FIFO + HC428 R1/R2 pass before they can replace v2 in production.
5. **v3.3 / 30s / short has NEGATIVE conditional MFE at top0.5% (0.077)** — this horizon/side is a model failure mode; the model's high-confidence shorts at 30s don't move further down on average. Should be excluded from sweep search space.

## HC #429 — REVISED CHAMPION RANKING

| Stage | Champion | Evidence |
|------|----------|----------|
| **Confirmed (full FIFO + HC408/415 pass)** | v2 / 1s / short / top0.5% | net +0.274 tk · Sharpe√N 12.77 · Sortino 707 · PF 2.92 · WR 85% · n=639 on 25/26 days (HC #413 verdict) |
| **Pre-FIFO leader (conditional MFE only)** | **v3.4.2 / 1s / long / top0.5%** | MFE 1.38 · MAE 0.29 · net **+1.005** · WR 68% · n=1206 |
| **Pre-FIFO #2** | v3.4.2 / 5s / long / top0.5% | net +0.870 · WR 58% |
| **In training** | v3.4.3 (Neptune PID 1187376) | No predictions yet |

## NEXT GATE — what unlocks a champion change

Before v3.4.2 / 1s / long / top0.5% can replace v2 as production champion, it must:

1. **Pass HC #408 + HC #415** (full FIFO market replay, n_fills ≥ 50, day_conc ≤ 0.20, CI95lo > 0, per_day_pass_rate ≥ 0.80, n_days_with_fills ≥ 10).
2. **Pass HC #428 R1** (regime-agnostic: ≥40 days, stratified Sharpe green/red, |ΔSharpe|/max ≤ 0.50).
3. **Pass HC #428 R2** (TP ≤ p90 cond_MFE = ~3 ticks, hold ≤ 1.5h = 1.5s, cancel ≤ h = 1s).
4. **Re-export v3.4.2 OOT NPZ with `oot_dates`** (current NPZ is single-day-collapsed; day_conc not measurable from this matrix alone).

## REQUIRED FOLLOW-UP ACTIONS

- [ ] Build `v342_fifo_validate_1s_long_top05.py` → run on 36-day OOT, write HC413-style verdict
- [ ] Re-export v3.4.2 NPZ with `oot_dates` (or use `data/v342_oot_npz/v342_5d.npz` if it has dates)
- [ ] Compute v3.4.3 conditional MFE matrix immediately after v5 fold 0 completes (Neptune PID 1187376)
- [ ] Re-anchor next Optuna sweep TP grid: `TP ∈ {1, 2, 3}` ticks (was {2, 4, 6, 8})
- [ ] Re-anchor next Optuna sweep SL grid: `SL ∈ {0.5, 1, 1.5}` ticks at 1s; `{1, 2}` at 5s; `{1.5, 2.5}` at 10s

## GAPS (from `hc429_gaps.md`)

- v3.4.2 NPZ has no `oot_dates` field → day-concentration not measurable until re-export
- 1s/5s/10s MAE = negative-only signed proxy (LOWER BOUND on true intra-window adverse, same as v2-native baseline — directly comparable)
