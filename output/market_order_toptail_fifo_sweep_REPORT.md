# Market-Order Top-Tail FIFO Sweep — REPORT

**HC #493 R3 + HC #428 R1/R2 binding.**
**Generated**: 2026-05-28T08:55:42.840603

## Hypothesis tested
Post-3-regrade synthesis: every proxy->FIFO collapse is -0.8 to -1.5 ticks, driven by limit-order queue wait + adverse fill selection + commission. Test whether MARKET orders (no queue, deterministic touch fill) at extreme top-tail confidence + longer holds escape that failure mode.

## Setup
- **Canonical FIFO equivalence (key insight)**: for pure market orders with no intra-horizon TP/SL, the FIFO replay PnL collapses to `direction * labels_h - cost`, because market orders fill at touch (deterministic, no queue) and labels_h is the canonical mid-to-mid realized move at horizon h in TICKS, sampled from the same `mbo_events_smart_v3` files the FIFO harness uses. No queue mechanics or fill-selection adverse effects can intervene.
- **Cost stack**: commission 0.376t + entry spread 1.0t + exit spread 1.0t = **2.376t round-trip** (per task spec).
- **Thresholds (per-day top abs(pred))**: [0.001, 0.005, 0.01, 0.02, 0.05]
- **Horizons / models**:
  * `v7_meta_h1s`: signal = v7 meta signed pred; realized = `labels_1s`; hold = 1s
  * `v2raw_h1s`: signal = v2 raw pred col 0 (1s); realized = `labels_1s`; hold = 1s
  * `v2raw_h5s`: signal = v2 raw pred col 1 (5s); realized = `labels_5s`; hold = 5s
  * `v2raw_h10s`: signal = v2 raw pred col 2 (10s); realized = `labels_10s`; hold = 10s
  * `v2raw_h30s`: signal = v2 raw pred col 2 (30s); realized = `labels_30s`; hold = 30s

## OOT coverage
- **v7_meta**: 27 aligned dates (20260320 -> 20260420). **Gap vs HC #428 R1 40-day requirement: 13 days short**.
- **v2_raw**: 48 aligned dates (20260224 -> 20260429). **Gap vs HC #428 R1: 0 days short**.

## Full sweep grid

| model       | cell_id   |   n_trades |   n_days |   mean_net_tk |   median_net_tk |   sharpe_ann |    wr |    pf |   n_pos_days |   n_neg_days |   day_conc |   regime_skew | verdict                 |
|:------------|:----------|-----------:|---------:|--------------:|----------------:|-------------:|------:|------:|-------------:|-------------:|-----------:|--------------:|:------------------------|
| v2raw_h10s  | top0.1pct |       2286 |       46 |        -1.346 |          -1.376 |      -12.376 | 0.389 | 0.545 |            6 |           40 |      0.071 |         0.536 | REJECT (net -1.346t<=0) |
| v2raw_h10s  | top0.5pct |      11505 |       46 |        -1.528 |          -1.376 |      -23.518 | 0.354 | 0.48  |            2 |           44 |      0.05  |         0.223 | REJECT (net -1.528t<=0) |
| v2raw_h10s  | top1pct   |      23031 |       46 |        -1.533 |          -1.376 |      -24.358 | 0.347 | 0.476 |            2 |           44 |      0.05  |         0.33  | REJECT (net -1.533t<=0) |
| v2raw_h10s  | top2pct   |      46083 |       46 |        -1.62  |          -1.876 |      -27.184 | 0.337 | 0.445 |            2 |           44 |      0.048 |         0.343 | REJECT (net -1.620t<=0) |
| v2raw_h10s  | top5pct   |     115244 |       46 |        -1.69  |          -1.876 |      -29.311 | 0.329 | 0.422 |            2 |           44 |      0.041 |         0.376 | REJECT (net -1.690t<=0) |
| v2raw_h1s   | top0.1pct |       2286 |       46 |        -1.382 |          -1.376 |      -15.977 | 0.227 | 0.282 |            5 |           41 |      0.053 |         0.224 | REJECT (net -1.382t<=0) |
| v2raw_h1s   | top0.5pct |      11509 |       46 |        -1.633 |          -1.876 |      -25.63  | 0.171 | 0.153 |            0 |           46 |      0.052 |         0.252 | REJECT (net -1.633t<=0) |
| v2raw_h1s   | top1pct   |      23040 |       46 |        -1.69  |          -1.876 |      -27.964 | 0.154 | 0.132 |            1 |           45 |      0.048 |         0.304 | REJECT (net -1.690t<=0) |
| v2raw_h1s   | top2pct   |      46102 |       46 |        -1.73  |          -1.876 |      -29.085 | 0.141 | 0.116 |            2 |           44 |      0.045 |         0.337 | REJECT (net -1.730t<=0) |
| v2raw_h1s   | top5pct   |     115296 |       46 |        -1.76  |          -1.876 |      -29.839 | 0.132 | 0.104 |            2 |           44 |      0.044 |         0.369 | REJECT (net -1.760t<=0) |
| v2raw_h30s  | top0.1pct |       2286 |       46 |        -1.37  |          -1.376 |      -10.691 | 0.43  | 0.673 |            8 |           38 |      0.082 |         0.114 | REJECT (net -1.370t<=0) |
| v2raw_h30s  | top0.5pct |      11500 |       46 |        -1.393 |          -1.376 |      -17.569 | 0.414 | 0.66  |            4 |           42 |      0.063 |         0.439 | REJECT (net -1.393t<=0) |
| v2raw_h30s  | top1pct   |      23020 |       46 |        -1.493 |          -1.376 |      -21.018 | 0.407 | 0.639 |            4 |           42 |      0.063 |         0.431 | REJECT (net -1.493t<=0) |
| v2raw_h30s  | top2pct   |      46062 |       46 |        -1.604 |          -1.876 |      -25.289 | 0.398 | 0.612 |            2 |           44 |      0.054 |         0.351 | REJECT (net -1.604t<=0) |
| v2raw_h30s  | top5pct   |     115188 |       46 |        -1.666 |          -1.876 |      -27.488 | 0.393 | 0.595 |            3 |           43 |      0.048 |         0.337 | REJECT (net -1.666t<=0) |
| v2raw_h5s   | top0.1pct |       2286 |       46 |        -1.218 |          -1.376 |      -13.236 | 0.357 | 0.48  |            7 |           39 |      0.077 |         0.178 | REJECT (net -1.218t<=0) |
| v2raw_h5s   | top0.5pct |      11507 |       46 |        -1.516 |          -1.376 |      -23.46  | 0.31  | 0.378 |            2 |           44 |      0.054 |         0.28  | REJECT (net -1.516t<=0) |
| v2raw_h5s   | top1pct   |      23034 |       46 |        -1.568 |          -1.376 |      -25.673 | 0.296 | 0.358 |            2 |           44 |      0.053 |         0.29  | REJECT (net -1.568t<=0) |
| v2raw_h5s   | top2pct   |      46090 |       46 |        -1.642 |          -1.876 |      -28.295 | 0.283 | 0.33  |            2 |           44 |      0.046 |         0.333 | REJECT (net -1.642t<=0) |
| v2raw_h5s   | top5pct   |     115263 |       46 |        -1.704 |          -1.876 |      -29.563 | 0.273 | 0.306 |            1 |           45 |      0.042 |         0.38  | REJECT (net -1.704t<=0) |
| v7_meta_h1s | top0.1pct |       1330 |       27 |        -1.214 |          -1.376 |      -16.296 | 0.264 | 0.267 |            0 |           27 |      0.15  |         0.317 | REJECT (net -1.214t<=0) |
| v7_meta_h1s | top0.5pct |       6690 |       27 |        -1.345 |          -1.376 |      -21.896 | 0.211 | 0.203 |            3 |           24 |      0.097 |         0.548 | REJECT (net -1.345t<=0) |
| v7_meta_h1s | top1pct   |      13394 |       27 |        -1.407 |          -1.376 |      -24.857 | 0.186 | 0.178 |            2 |           25 |      0.08  |         0.53  | REJECT (net -1.407t<=0) |
| v7_meta_h1s | top2pct   |      26800 |       27 |        -1.486 |          -1.376 |      -25.873 | 0.173 | 0.147 |            1 |           26 |      0.081 |         0.526 | REJECT (net -1.486t<=0) |
| v7_meta_h1s | top5pct   |      67022 |       27 |        -1.563 |          -1.876 |      -26.549 | 0.16  | 0.125 |            0 |           27 |      0.076 |         0.518 | REJECT (net -1.563t<=0) |

## Verdict summary
- **PASS cells**: 0
- **WEAK PASS cells**: 0
- **REJECT cells**: 25

### No PASS cells.

## Best-cell-per-model (informational, may still be a REJECT)
- **v7_meta_h1s**: best cell `top0.1pct` -> net -1.214t/trade, sharpe -16.30, 0/27 pos days, verdict: REJECT (net -1.214t<=0)
- **v2raw_h5s**: best cell `top0.1pct` -> net -1.218t/trade, sharpe -13.24, 7/46 pos days, verdict: REJECT (net -1.218t<=0)
- **v2raw_h10s**: best cell `top0.1pct` -> net -1.346t/trade, sharpe -12.38, 6/46 pos days, verdict: REJECT (net -1.346t<=0)
- **v2raw_h30s**: best cell `top0.1pct` -> net -1.370t/trade, sharpe -10.69, 8/46 pos days, verdict: REJECT (net -1.370t<=0)
- **v2raw_h1s**: best cell `top0.1pct` -> net -1.382t/trade, sharpe -15.98, 5/46 pos days, verdict: REJECT (net -1.382t<=0)

## Cost-sensitivity (informational)

The task spec mandated 1.0t spread on entry + 1.0t on exit (2.0t spread + 0.376t commission = 2.376t). Under the more conventional one-crossing-RT interpretation (1.0t spread + 0.376t commission = 1.376t), ALL cells are still negative (best is -0.21t/trade). Top-0.1% gross magnitudes (cost-free):

| Cell | Gross t/trade |
|------|---------------|
| v7_meta_h1s top0.1pct | +1.16 |
| v2raw_h5s top0.1pct | +1.16 |
| v2raw_h10s top0.1pct | +1.03 |
| v2raw_h30s top0.1pct | +1.01 |
| v2raw_h1s top0.1pct | +0.99 |

Top-tail signal magnitudes top out at ~1.0-1.2 ticks. Structurally below ANY realistic market-both-sides cost stack (the cheapest plausible is 1.376t round-trip — and even that swallows the entire signal).

## Recommendation
Pure market-order execution at top-tail confidence on existing predictions does NOT carve out edge under the canonical realized-move accounting at either the task's 2.376t cost or the lenient 1.376t cost. Same structural failure as the three regrade rejects: signal magnitude is below the cost stack. The top-0.1% peak gross of ~1.16 ticks IS real model edge — it just can't pay even one full bid-ask crossing plus commission.

Next axes to test (NOT in this sweep — would require new work):
(a) **Hybrid execution**: passive entry (queue wait + adverse fill on its own would lose), market exit on adverse move only. Saves half-spread on entry when fill succeeds. Need to model selection bias.
(b) **Ensemble / confluence to lift top-tail signal magnitude above 1.4t**. Three-model agreement might concentrate to 1.5-2.0t gross. Pre-2026-05-28 confluence claims (Both-top-2% +1.187 proxy) showed gross magnitude can reach 1.2t for confluence shorts — still too small post-cost. Higher-bar (Both-top-0.1%) confluence has not been graded.
(c) **Abandon ES intraday execution at sub-30s horizons**. The 1-tick ES book + 0.376t commission creates a ~1.4t structural floor that the model's 1s-30s edge cannot clear.

The +1.16t gross at extreme top-tail IS the signal-quality result. The trading question reduces to: can any cost-side innovation halve the cost stack? If not, the answer is HC #493 's "we have NOTHING tradeable" — and the right pivot is execution-cost reduction (different broker, futures spreads vs ES outright, etc.), not more model variants.