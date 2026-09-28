# HC #413 — TP/SL Scalping Backtester (Phase 1)

End-to-end backtester that takes a CNN-Mamba prediction NPZ + an
MFE/MAE-at-confidence config CSV and produces realized P&L with
risk-adjusted metrics (Sharpe, Sortino, PF, WR), HC #344 day_conc, and
HC #408 honesty-gate compliance per (cell × confidence-tier × horizon × side).

## Files

| File | Purpose |
|------|---------|
| `backtester.py`  | Main entry point. Loads NPZ + MFE config, iterates cells, writes CSV + verdict.md. |
| `tp_sl_rules.py` | TP1/TP2/SL exit rules per HC #413 rule 3 (TP1=0.5·MFE, TP2=1.0·MFE, SL=min(\|MAE\|, 1.5·MFE)). |
| `fill_sim.py`    | Canonical FIFO market replay (HC #397B). Reuses `_load_fifo_labels` from `scripts/v3_3_research/full_market_replay.py`. Applies the same queue-position deflator (0.5) used in the v3.3 PPO eval pipeline. |
| `metrics.py`     | Sharpe√N, Sortino√N, PF, WR, day_conc (HC #344), CI_low_95 via 2000-rep percentile bootstrap (HC #408). |
| `smoke_test.sh`  | Wires it all together with the v3.3 NPZ + hc411 MFE matrix. |
| `verdict.md`     | (Generated) Promotion summary. |

## Canonical cost constants (CLAUDE.md COST CONSTANTS)

- ES_TICK_VALUE        = $12.50
- ES_RT_COMMISSION     = $4.70 (= 0.376 ticks)
- Passive limit total  ~ 0.376 ticks
- Market order total   ~ 1.376 ticks (commission + 1.0-tick spread crossing)

## Usage

```bash
# v3.3 smoke (default)
bash smoke_test.sh

# Swap to v3.4.2 60d NPZ when it lands (~22:00 ET 5/18) — no code change:
NPZ=/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2/fold_00_predictions.npz \
MODEL_FILTER=v3.4.2 \
bash smoke_test.sh
```

## CLI

```
python backtester.py \
  --npz <prediction NPZ> \
  --mfe-config <hc411 mfe_at_confidence_matrix.csv> \
  --output-dir <dir> \
  [--confidence-tier {all,top05,top1,top5,top10}]  default all \
  [--horizon {all,1s,5s,10s,30s}]                  default all \
  [--side {both,long,short}]                        default both \
  [--model {all,v3.3,v3.4.2,...}]                   default all (filters MFE rows) \
  [--order-type {passive_at_touch,market}]          default passive_at_touch \
  [--cancel-eval-window <int>]                      default 40 (10 s @ 250 ms stride) \
  [--labels-dir <dir>]                              default data/processed/mbo_events_smart_v3_fifo_labels \
  [--seed <int>]                                    default 42
```

## Output CSV schema (`scalping_backtest_results.csv`)

| column | meaning |
|---|---|
| cell_id | `<model>_<horizon>_<side>_<conf_tier>` |
| model, horizon, side, conf_tier | factor levels |
| n_fills | # filled entries (after FIFO entry gate) |
| n_tp1_hits / n_tp2_hits / n_sl_hits / n_time_stops | exit reason counts |
| gross_mfe_per_fill | mean pre-cost gross P&L per fill, ticks |
| realized_net_per_fill | mean net P&L per fill, ticks (HC #69 primary) |
| realized_net_per_fill_dollars | × $12.50 |
| sharpe_sqrtN | mean/std × √N (per-trade Sharpe — HC #397A style) |
| sortino_sqrtN | mean/downside-std × √N |
| pf | profit factor (gross win / gross loss) |
| wr | win rate, % |
| day_conc | max-day \|P&L\| / total \|P&L\| (HC #344) |
| ci_low_95_net | 2.5%ile of bootstrap mean (HC #408) |
| ci_low_95_net_dollars | × $12.50 |
| pass_hc344 | day_conc ≤ 0.20 AND n_fills ≥ 30 |
| pass_hc408_honesty | n_fills ≥ 50 AND CI_low_95 > 0 AND day_conc ≤ 0.20 |
| mfe_source, mae_source | original config values (informational) |
| tp1, tp2, sl | resolved thresholds in ticks |
| entry_cost_ticks | 0.376 (passive) or 1.376 (market) |
| order_type | as passed |

## What each module does

### `tp_sl_rules.py`
Computes TP1, TP2, SL thresholds from the (mfe, mae) cell. Resolves each
fill's exit by walking realized horizon checkpoints (1s → 5s → 10s →
30s), checking SL-first then TP2 then TP1. Unresolved fills take a
time-stop at the latest finite horizon's in-position value.

### `fill_sim.py`
Decides whether an entry order fills. For `passive_at_touch`, gates on
the FIFO label's `_filled` flag + a deflator (0.5 base, 0.125 for
`max_hold` exits) consistent with the v3.3 PPO canonical-replay eval. For
`market`, always fills. Applies canonical commissions per CLAUDE.md.

### `metrics.py`
Pure-function metric layer. All ticks-per-trade in / scalar out.

### `backtester.py`
Glue: loads NPZ, loads FIFO labels for matching OOT dates, iterates cells
from the MFE config CSV, gates by per-cell confidence percentile,
runs fill-sim, resolves TP/SL exits, writes CSV + verdict.

## v3.3 → v3.4.2 swap

Drop-in: only the `--npz` and `--model` arguments change. The NPZ schema is
identical (multi-head `pred_log_ret_{1s,5s,10s,30s}` + matching targets).
FIFO labels for v3.4.2 OOT dates are expected at the same `--labels-dir`.

## HC compliance map

- **HC #69**: risk-adjusted (Sharpe√N, Sortino√N, PF, WR) reported as primary.
- **HC #344**: day_conc reported; pass_hc344 column.
- **HC #393**: `seed=42` everywhere; outputs idempotent.
- **HC #397B**: canonical FIFO replay only — no midpoint shortcuts.
- **HC #408**: pass_hc408_honesty = n_fills≥50 AND CI_low_95>0 AND day_conc≤0.20.
- **HC #413** rule 3: TP1=0.5·MFE, TP2=1.0·MFE, SL=min(\|MAE\|, 1.5·MFE).
- **CLAUDE.md COST CONSTANTS**: passive=0.376, market=1.376 (NOT 2.0 / 1.24 /
  other defaults).
