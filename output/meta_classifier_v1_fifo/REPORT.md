# Meta-Classifier v1 — FIFO Market-Replay Validation (HC #74)

Tests label assumption (passive limit @ 0.376 t commission) against realized FIFO fills. Exit = market at horizon (+1 t spread crossing).

## short_10s_thr55 — VERDICT: REJECT
_reject reasons: pdays<11/16_
- Signaled: 4,248 | Filled: 3,078 | Fill rate: 72.46%
- Realized net t/trade: +5.030 (label assumed +3.463, delta +1.567)
- Sharpe pooled: 4.54 (label assumed 7.70)  | WR realized: 0.666
- Profitable days: 8/16 (gate >= 11) | days w/ any fill: 15/16
- Regime imb: nan (gate <= 0.5) | day_conc: 0.574 (gate <= 0.7)

## long_1s_thr55 — VERDICT: REJECT
_reject reasons: net_t<=0.1;Sharpe<=0.3;pdays<11/16_
- Signaled: 26,190 | Filled: 15,309 | Fill rate: 58.45%
- Realized net t/trade: -1.245 (label assumed +0.264, delta -1.509)
- Sharpe pooled: -13.93 (label assumed 17.36)  | WR realized: 0.251
- Profitable days: 1/16 (gate >= 11) | days w/ any fill: 14/16
- Regime imb: nan (gate <= 0.5) | day_conc: 0.200 (gate <= 0.7)

## Notes
- Cancel window = horizon h; max_hold = 1.5 h; TP/SL set wide so all fills exit at max_hold (market) per HC #428 R2.
- Net per trade = pnl_ticks - 0.376 (entry commission, by engine) - 1.0 (market-exit spread, post-correction).
- Per-fold OOT mapping mirrors meta_classifier_v1_walkforward (sliding window, HC #0).
