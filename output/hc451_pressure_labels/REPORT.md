# HC #451 R5 — Pressure Label Cache Report

- Dates processed: **248**  (range 20250714 .. 20260429)
- Total events (incl. NaN-edge): **2,485,641,693**
- Cache directory: `/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_pressure_labels`
- Neutrality band: 0.5 (ticks, same unit as source labels)

## Overall persistence rates

Format: `+1 (agree) / 0 (neutral) / -1 (flip)` as percent of all events.

| Horizon pair | +1 / 0 / -1 (%) |
|---|---|
| persistence_1s_10s | 34.20 / 49.09 / 16.71 |
| persistence_1s_30s | 32.25 / 46.10 / 21.65 |
| persistence_5s_30s | 47.40 / 30.40 / 22.20 |

## Confidence-stratified persistence (by |labels_1s|)

Thresholds derived from a uniform sample of valid |labels_1s|.

| Band | |labels_1s| >= (ticks) | n events | persistence_1s_10s (+1/0/-1) | persistence_1s_30s (+1/0/-1) | persistence_5s_30s (+1/0/-1) |
|---|---|---|---|---|---|
| top10pct | 2.500 | 322,474,185 | 68.83 /  8.89 / 22.28 | 61.22 /  6.50 / 32.28 | 60.15 / 15.18 / 24.67 |
| top5pct | 3.000 | 247,748,230 | 70.48 /  7.66 / 21.86 | 62.23 /  5.82 / 31.94 | 61.85 / 13.03 / 25.12 |
| top1pct | 6.500 | 24,797,797 | 78.85 /  4.37 / 16.78 | 68.97 /  3.68 / 27.35 | 69.95 /  6.84 / 23.21 |
| top0p5pct | 9.000 | 9,581,658 | 81.75 /  3.38 / 14.87 | 71.67 /  3.05 / 25.28 | 72.94 /  5.17 / 21.89 |

## Pressure score distribution

- Sample size: 4,827,795 (uniform sub-sample)
- Mean: +0.0052
- p10 / p25 / p50 / p75 / p90 / p99: -0.905 / -0.762 / +0.000 / +0.762 / +0.905 / +0.964
- |pressure_score| >= 0.9: 38.72%  (~all four horizons agree)
- |pressure_score| <= 0.1: 13.30%  (mixed / neutral)

```
  [-1.00,-0.90)  #################################################   19.11%  (n=922,562)
  [-0.90,-0.81)                                                       0.00%  (n=0)
  [-0.81,-0.71)  ################################                    12.43%  (n=599,948)
  [-0.71,-0.62)                                                       0.00%  (n=0)
  [-0.62,-0.52)                                                       0.00%  (n=0)
  [-0.52,-0.43)  #############################                       11.53%  (n=556,868)
  [-0.43,-0.33)                                                       0.00%  (n=0)
  [-0.33,-0.24)                                                       0.00%  (n=0)
  [-0.24,-0.14)                                                       0.00%  (n=0)
  [-0.14,-0.05)                                                       0.00%  (n=0)
  [-0.05,+0.05)  ##################################                  13.30%  (n=641,970)
  [+0.05,+0.14)                                                       0.00%  (n=0)
  [+0.14,+0.24)                                                       0.00%  (n=0)
  [+0.24,+0.33)                                                       0.00%  (n=0)
  [+0.33,+0.43)                                                       0.00%  (n=0)
  [+0.43,+0.52)  #############################                       11.52%  (n=556,231)
  [+0.52,+0.62)                                                       0.00%  (n=0)
  [+0.62,+0.71)                                                       0.00%  (n=0)
  [+0.71,+0.81)  ################################                    12.50%  (n=603,425)
  [+0.81,+0.90)                                                       0.00%  (n=0)
  [+0.90,+1.00)  ##################################################  19.61%  (n=946,791)
```

## Interpretation

For the **top 1%** of events by |labels_1s| (|labels_1s| >= 6.50 ticks), the 1s sign **agrees** with the 10s sign 78.8% of the time, **flips** 16.8%, and lands in the neutrality band 4.4%. At the 30s horizon the agreement rate is 69.0% with a flip rate of 27.3% (neutral 3.7%). For comparison, across **all events** (most of which are noise around the bid-ask), 1s-to-10s agreement is only 34.2% (flip 16.7%). This means the high-confidence 1s signal does carry forward — the directional edge observed at 1s is materially more likely to still be in the same direction at 10s, and even at 30s, than would be true of an average event. However, **the flip rate is non-trivial** (17% at 10s, 27% at 30s for the top 1%), which means the alpha is real but decays and reverses for a meaningful minority of trades — consistent with the HC #428 decay-window evidence and the existing belief that holds beyond ~10s start picking up noise rather than signal.

## Notes

- MFE/MAE within 10s is a 3-point approximation (1s, 5s, 10s sample of the path).
  For a true tick-resolution MFE/MAE, run `alpha_discovery/evaluation/mfe_mae_path_analysis.py --dense`.
- Neutrality band of 0.5 is in the source label unit (ticks), not bps. The
  HC brief used 'bps' loosely; the source NPZ labels per the codebase header are ticks.
- Pressure-score distribution is a 5M uniform sub-sample, not the full corpus,
  to keep this report's memory bounded.
