# HC #437 Bug 2 — Bracket-Mode Canonical Run Command

## Phase A verification

**Implementation source**: `alpha_discovery/deep_models/fifo_market_replay.py`
- Function `_resolve_hc413_bracket()` lines 295-353
- Enabled in `FIFOReplayEngine.simulate()` via `order_management='hc413_bracket'`
- Validated: SL/TP2/TP1 are evaluated ONLY at horizon checkpoints (1s, 5s, 10s) from realized per-signal labels (`labels_by_h`), NOT intra-event. Priority per checkpoint: SL first, then TP2, then TP1 (HC #413 conservative ordering). Time-stop = realized return at latest finite checkpoint.

**Runner script**: `scripts/v3_4_research/hc437_v2_baseline_runner_bracket.py`
- Pools v2 NPZ predictions globally, selects top-N% conf side-aligned signals
- Passes per-signal `labels_by_h` from NPZ `labels[:, [0,1,2]]` (1s/5s/10s)
- Mirrors `hc432_v2_baseline_runner.py` interface

## Canonical run command (full 47-day v2 1s short top0.5 sanity baseline)

```
cd /home/jupiter/Lvl3Quant && \
  nice -n 15 python scripts/v3_4_research/hc437_v2_baseline_runner_bracket.py \
    --horizon 1 \
    --side short \
    --conf-band top0.5 \
    --tp1-ticks 0.4782 \
    --tp2-ticks 0.9564 \
    --sl-bracket-ticks 0.5686 \
    --hold-s 10.0 \
    --cancel-s 10.0 \
    --order-type passive_at_touch \
    --workers 10 \
    --config-name v2_short_1s_top0.5_bracket_HC437_47day
```

## 3-day reference (already produced)

`output/hc437_harness_debug/v2_short_1s_top0.5_bracket_HC437_3day_bracket_fifo_summary.json`
- n_fills=218, mean_net=+0.240 tk, PF=2.15, WR=77.5%, Sh√N=5.59
- exit_counts: tp2=169, sl=48, time_stop=1

## HC #437 R1 PASS gate

- net_tk_per_fill in [+0.224, +0.324] (HC #413 ref +0.274 ± 0.05)
- Sh√N ≥ 10
- PF ≥ 2.5
- WR ≥ 82%
