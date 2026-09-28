# TOD x Velocity — FIFO Market-Replay Validation (HC #74)

Generated: 2026-05-22 13:33:37 EDT
Source cells: /home/jupiter/Lvl3Quant/output/tod_velocity_stratification_v1/winning_cells.txt
OOT dates: 8 (DBN+pred, skipped ['20260308', '20260315'])

## HEADLINE
**ALL 6 LABEL-WINNERS DIED IN FIFO** under canonical (passive-passive 0.376 t commission only) cost.
With realistic market-exit cost (+1.0 t spread), 0/6 survive.

## Per-cell verdict

| side | h | conf | tod | vd | n_sig | n_fill | fill% | label | pas_net | mkt_net | gap_lbl_vs_pas | verdict |
|------|---|------|-----|----|------|-------|------|------|------|------|------|------|
| long | 10s | top5pc | mid_am | 8 | 162 | 149 | 92.0% | +0.686 | +0.298 | -0.702 | -0.387 | REJECT |
| short | 10s | top5pc | open | 9 | 919 | 749 | 81.5% | +0.614 | +0.578 | -0.422 | -0.036 | REJECT |
| short | 30s | top5pc | mid_am | 5 | 534 | 492 | 92.1% | -0.361 | -0.162 | -1.162 | +0.199 | REJECT |
| short | 5s | top1pc | open | 4 | 29 | 25 | 86.2% | +0.176 | +0.104 | -0.896 | -0.072 | REJECT |
| short | 10s | top1pc | late_pm | 2 | 4 | 3 | 75.0% | -0.501 | -0.709 | -1.709 | -0.208 | REJECT |
| long | 5s | top1pc | midday | 5 | 22 | 15 | 68.2% | +0.192 | -0.076 | -1.076 | -0.268 | REJECT |

## Key number: label-vs-FIFO gap (passive-passive)

This is the cost of label-fill optimism. Cells with gap > 1 tick = severe adverse selection.

- long 10s top5pc mid_am vd8: label +0.686t  ->  FIFO +0.298t  (gap -0.387t)
- short 10s top5pc open vd9: label +0.614t  ->  FIFO +0.578t  (gap -0.036t)
- short 30s top5pc mid_am vd5: label -0.361t  ->  FIFO -0.162t  (gap +0.199t)
- short 5s top1pc open vd4: label +0.176t  ->  FIFO +0.104t  (gap -0.072t)
- short 10s top1pc late_pm vd2: label -0.501t  ->  FIFO -0.709t  (gap -0.208t)
- long 5s top1pc midday vd5: label +0.192t  ->  FIFO -0.076t  (gap -0.268t)

## Top cell — long 10s top5pc mid_am vd8

- Label: net +0.686t Sh 1.91
- FIFO passive-passive: net +0.298t Sh 0.70  PF 1.17  fill 92.0%
- FIFO market-exit: net -0.702t Sh -1.64
- Verdict (passive): REJECT  reasons=[imb>=0.5]

## Structural interpretation

- No cell shows >1 tick label-vs-FIFO gap. Adverse selection within tolerance.
- Average fill rate across cells: 82.5%.

## Recommended next move

- DO NOT DEPLOY any tod_velocity_stratification_v1 cell. Labels lied.
- Stop label-based gate testing entirely; require FIFO validation up-front for every candidate.
- Consider chase orders (allow reprices) or wider TP buckets — re-spec, re-test, then re-FIFO.

## Notes

- Cost (a) passive-passive: pnl_net = pnl_gross - 0.376 t RT commission (entry+exit both passive limit, exit at mid by engine).
- Cost (b) market-exit:   pnl_net_b = pnl_net - 1.0 t spread crossing (exit by market at max_hold).
- TP/SL set wide (1000 t) so all fills exit by max_hold or EOD, per HC #428 R2.
- Engine: alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine — UNMODIFIED.
