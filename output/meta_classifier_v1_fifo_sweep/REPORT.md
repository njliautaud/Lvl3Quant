# Meta-Classifier v1 — FIFO Market-Replay Validation (HC #74)

Tests label assumption (passive limit @ 0.376 t commission) against realized FIFO fills. Exit = market at horizon (+1 t spread crossing).

## long_1s_thr50 — VERDICT: REJECT
_reject reasons: net_t<=0.1;Sharpe<=0.3;pdays<11/16_
- Signaled: 107,972 | Filled: 60,468 | Fill rate: 56.00%
- Realized net t/trade: -1.294 (label assumed +0.113, delta -1.407)
- Sharpe pooled: -45.25 (label assumed 13.26)  | WR realized: 0.228
- Profitable days: 0/16 (gate >= 11) | days w/ any fill: 16/16
- Regime imb: nan (gate <= 0.5) | day_conc: 0.169 (gate <= 0.7)

## short_1s_thr55 — VERDICT: REJECT
_reject reasons: net_t<=0.1;Sharpe<=0.3;pdays<11/16_
- Signaled: 25,295 | Filled: 14,920 | Fill rate: 58.98%
- Realized net t/trade: -1.443 (label assumed +0.111, delta -1.554)
- Sharpe pooled: -48.57 (label assumed -0.74)  | WR realized: 0.240
- Profitable days: 0/16 (gate >= 11) | days w/ any fill: 16/16
- Regime imb: nan (gate <= 0.5) | day_conc: 0.228 (gate <= 0.7)

## short_5s_thr55 — VERDICT: REJECT
_reject reasons: net_t<=0.1;pdays<11/16_
- Signaled: 13,264 | Filled: 9,619 | Fill rate: 72.52%
- Realized net t/trade: -0.339 (label assumed +0.329, delta -0.668)
- Sharpe pooled: 2.49 (label assumed 8.97)  | WR realized: 0.426
- Profitable days: 6/16 (gate >= 11) | days w/ any fill: 14/16
- Regime imb: nan (gate <= 0.5) | day_conc: 0.259 (gate <= 0.7)

## Notes
- Cancel window = horizon h; max_hold = 1.5 h; TP/SL set wide so all fills exit at max_hold (market) per HC #428 R2.
- Net per trade = pnl_ticks - 0.376 (entry commission, by engine) - 1.0 (market-exit spread, post-correction).
- Per-fold OOT mapping mirrors meta_classifier_v1_walkforward (sliding window, HC #0).
