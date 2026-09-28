# Cost-Model Audit — 2026-05-12

**Canonical (per CLAUDE.md):** TICK=$12.50; COMMISSION_RT=$4.70 = 0.376 ticks. Passive limit total cost = 0.376 ticks (commission only, NO spread). Market/IOC total = 1.376 ticks (commission + 1 tick spread crossing).
Note: `backtest/` subdir & `paper_trading*.py`/`fifo_replay*.py` requested paths partly do not exist. Closest matches were `live_trading_linux/paper_trading_*.py`, `execution/fifo_*.py`, `alpha_discovery/deep_models/fifo_market_replay.py`, and `alpha_discovery/execution/fifo_*.py`. Audit broadened to anything referencing cost/commission/slippage/spread.

## 1. Summary table

| File | Line | Variable | Current | Correct | Severity |
|---|---|---|---|---|---|
| alpha_discovery/run_combined_strategy.py | 64 | COMMISSION_RT | 2.50 | 4.70 | CRITICAL |
| alpha_discovery/conditional_sim.py | 81 | COMMISSION_RT | 3.10 | 4.70 | CRITICAL |
| alpha_discovery/run_continuation.py | 62 | COMMISSION_TICKS | 3.00/12.50=0.24 | 4.70/12.50=0.376 | CRITICAL |
| alpha_discovery/run_hold_optimizer.py | 67 | COMMISSION_TICKS | 0.2 ($2.50) | 0.376 ($4.70) | CRITICAL |
| alpha_discovery/execution_engine.py | 74 | TOTAL_COST_TICKS_MARKET | 1.24 | 1.376 | HIGH |
| alpha_discovery/execution_engine.py | 75 | TOTAL_COST_TICKS_LIMIT | 0.74 | 0.376 | CRITICAL |
| alpha_discovery/run_multibar_alpha_scan.py | 93 | COST_MARKET_TOTAL | 1.24 | 1.376 | HIGH |
| alpha_discovery/slow_decay_backtest.py | 46 | TOTAL_COST_TICKS | 1.24 (SPREAD+COMM) | 1.376 mkt / 0.376 lim | HIGH |
| alpha_discovery/run_ask_orders_oos_holdout.py | 59-61 | SPREAD_COST + TOTAL_COST | $15.50 / 1.24t | $4.70 / 0.376t (limit) | HIGH |
| alpha_discovery/backtest_5min_signal.py | 164 | comm_ticks default | 0.24 | 0.376 | HIGH |
| alpha_discovery/advanced_strategy_test.py | 148/201/257 | comm_ticks default | 0.24 | 0.376 | HIGH |
| alpha_discovery/strategy3_flip_test.py | 86/142 | comm_ticks default | 0.24 | 0.376 | HIGH |
| alpha_discovery/run_hybrid_execution_sim.py | 76 | COMMISSION_TICKS comment | "0.24" but value=0.376 | label fix only | LOW |
| alpha_discovery/run_honest_backtest.py | 75 | COMMISSION_TICKS comment | "0.24" but value=0.376 | label fix only | LOW |
| alpha_discovery/continuation_reversal.py | 71 | comment "0.24t" | value=0.376 ok | label fix only | LOW |
| alpha_discovery/run_production_abc.py | 74 | comment "0.24" | value=0.376 ok | label fix only | LOW |
| alpha_discovery/continuous_signal_sim.py | 63 | comment "0.24" | value=0.376 ok | label fix only | LOW |
| alpha_discovery/exp_002_rl_reward_oos.py | 65 | comment "0.24t" | value=0.376 ok | label fix only | LOW |
| alpha_discovery/novel_targets_mbo_sweep.py | 59 | comment "0.24" | value=0.376 ok | label fix only | LOW |
| alpha_discovery/vol_magnitude_gated.py | 80 | comment "~0.24" | value=0.376 ok | label fix only | LOW |
| scripts/combined_strategy_backtest.py | 29-30 | SLIPPAGE_TICKS=1, COMM_PER_SIDE=0.50 | slip=0; comm=2.35/side | CRITICAL |
| live_trading/main.py | 60 | market_slippage_ticks default | 0.5 | 0.0 (HC #231(A)) | HIGH |
| live_trading/fill_simulator.py | 93 | market_slippage_ticks | configurable, default ext | 0.0 | HIGH |
| live_trading/trade_journal.py | 861 | slippage_ticks=0.1*i | test fixture | OK (test) | NONE |

Files audited and CLEAN (match canonical, no spread cost double-count):
constants.py, live_trading/rl_execution_agent.py, live_trading/performance_report.py, alpha_discovery/execution/fifo_rl_env.py (explicit HC #127 comment), alpha_discovery/execution/fifo_time_exit_backtest.py, alpha_discovery/deep_models/fifo_market_replay.py, alpha_discovery/mm_alpha_sim.py, execution/fifo_vol_conditioned_validation.py, execution/fifo_rules_optimizer.py, execution/fifo_parameter_sweep.py, execution/midday_optimization_v6.py, execution/multi_model_execution_sweep.py (HC #231(A)-compliant), execution/vol_conditioned_exec_params.py, scripts/validate_via_fifo_replay.py, scripts/validate_supervised_exec_v3_h15.py, scripts/extract_fifo_labels_for_lgbm.py, alpha_discovery/eval/test_metric_panel.py, all paper_trading_*.py (use COMMISSION_PER_SIDE constant, ok), alpha_discovery/max_profit_analysis.py (HC #290(C)).

## 2. Detailed findings

**Group A — wrong commission $ value (CRITICAL):**
- `run_combined_strategy.py:64` hardcodes $2.50 RT — under-states commission by 47%, inflates net P&L.
- `conditional_sim.py:81` hardcodes $3.10 RT — under-states by 34%.
- `run_continuation.py:62` and `run_hold_optimizer.py:67` use $2.50/$3.00 → 0.2 / 0.24 ticks.

**Group B — flagged WRONG patterns in CLAUDE.md (HIGH/CRITICAL):**
- `execution_engine.py:74` literally produces the canonical wrong value `TOTAL_COST_TICKS = 1.24`. Line 75's `TOTAL_COST_TICKS_LIMIT = 0.74` adds a fictitious 0.5-tick exit half-spread to limit orders; per HC #231(A) passive fills incur commission ONLY.
- `slow_decay_backtest.py:46` same 1.24 pattern.
- `run_multibar_alpha_scan.py:93` same 1.24 pattern; line 96 even has negative `COST_LIM_LIM_NET = -0.76` (treats limit edge as free profit).
- `run_ask_orders_oos_holdout.py:59-60` adds $12.50 SPREAD_COST on top of commission → $15.50 RT = 1.24t.
- `*advanced_strategy_test.py / strategy3_flip_test / backtest_5min_signal*` default `comm_ticks=0.24` (legacy $3.00 assumption).
- `scripts/combined_strategy_backtest.py:30` matches the CLAUDE.md flagged pattern `spread/slippage` separately summed (`SLIPPAGE_TICKS=1` per side + `$0.50` commission/side) — both wrong direction (slip overstated, comm understated).

**Group C — value correct, only comment is stale (LOW):**
Multiple files write `COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)` then a trailing comment "0.24 ticks". The computation `COMMISSION_RT / TICK_VALUE = 0.376` is correct; only the inline comment is misleading. P&L math is unaffected.

**Group D — live-stack slippage assumption:**
`live_trading/main.py:60` defaults `market_slippage_ticks=0.5`. Per HC #231(A) and the canonical model, ES book is 1 tick wide; passive fills incur 0 slippage and market/IOC pays exactly 1 tick spread crossing accounted in the 1.376 figure, not an additional 0.5. This double-counts slippage in the paper-trader fill sim if `market_slippage_ticks` is not overridden to 0 in config.

## 3. Proposed fix (DIFF — NOT APPLIED)

For each Group A/B file:
- Replace hardcoded `COMMISSION_RT = 2.50|3.00|3.10` with `COMMISSION_RT = 4.70`.
- Replace `comm_ticks=0.24` defaults with `comm_ticks=0.376`.
- In `execution_engine.py` and `slow_decay_backtest.py` and `run_multibar_alpha_scan.py`: set `TOTAL_COST_TICKS_MARKET = 1.376` and `TOTAL_COST_TICKS_LIMIT = COMMISSION_TICKS` (i.e. 0.376, no extra half-spread).
- `run_ask_orders_oos_holdout.py`: drop `SPREAD_COST = 12.50`; use commission-only for passive paths.
- `scripts/combined_strategy_backtest.py`: drop `SLIPPAGE_TICKS`; use `COMMISSION_PER_SIDE=2.35` (=$4.70/2).
- `live_trading/main.py:60`: default `market_slippage_ticks=0.0`. Verify configs that override this.
- Group C: simply update inline comments to "0.376 ticks" (cosmetic).
- Best long-term fix: have every file import from `constants.py` (already canonical) rather than redefining locals.

## 4. Risk assessment — tainted past results

**HIGH-RISK reports/MLflow artifacts (results likely overstate edge):**
- Anything produced by `run_combined_strategy.py`, `conditional_sim.py`, `run_continuation.py`, `run_hold_optimizer.py`: under-charged commission by 20-47%, P&L overstated.
- Anything from `execution_engine.py`, `slow_decay_backtest.py`, `run_multibar_alpha_scan.py`, `run_ask_orders_oos_holdout.py`: over-charged limit-order costs (extra 0.5t exit spread) → may have REJECTED viable passive strategies. Conversely, market-order results understated cost by ~0.14 ticks (1.24 vs 1.376).
- `backtest_5min_signal`, `advanced_strategy_test`, `strategy3_flip_test`: defaults of 0.24t commission → P&L overstated by ~0.14t per round-trip.
- `scripts/combined_strategy_backtest`: 2t round-trip slippage + $1/RT commission → P&L massively understated; any "rejected" cards from this script deserve re-screen.

**LOW-RISK (label-only):** Group C files. Numeric outputs are valid; only docs are stale.

**Live trading risk:** `live_trading/main.py` default 0.5-tick market slippage will make paper-trade reports look 0.5t worse per market fill than canonical, biasing the agent against market orders. Verify production config explicitly overrides to 0.

Recommend: re-run any execution-comparison study that selected limit-vs-market based on `execution_engine.py` / `run_multibar_alpha_scan.py` / `run_ask_orders_oos_holdout.py`, since the limit-side was systematically penalized.
