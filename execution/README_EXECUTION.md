# Execution Strategy Testing Framework

Market replay-based execution strategy testing for NQ futures trading.

## Overview

This framework orchestrates the Rust MBO fill simulator (`fill_sim_cli`) across
multiple execution strategies, model predictions, and signal thresholds to find
optimal execution parameters.

**Key principle:** All fills are simulated via event-by-event FIFO queue replay
on raw MBO (market-by-order) data. No theoretical fills -- every trade must
survive realistic queue position, adverse selection, and spread costs.

## Architecture

```
Model Predictions (.npz)
        |
        v
[Prediction Loader] -- converts per-event/window preds to bar-indexed z-scored signal
        |
        v
[Strategy Definitions] -- entry mode, signal threshold, exit rules, latency
        |
        v
[Rust fill_sim_cli] -- event-by-event FIFO queue sim on raw MBO data
        |
        v
[Aggregation & Analysis] -- per-strategy metrics, confidence tiers, MFE/MAE
        |
        v
[Comparison Tables] -- sorted by Sortino, grouped by strategy type
```

## Quick Start

```bash
# Run all strategies with CNN model (champion, IC_10s=0.132)
python execution_strategy_tester.py

# Run with Mamba multi-horizon predictions
python execution_strategy_tester.py --model mamba --multi-horizon

# Run specific strategy groups
python execution_strategy_tester.py --groups entry_modes,exit_strategies

# Quick test on 3 days
python execution_strategy_tester.py --max-days 3 --groups entry_modes

# Dry run: list all strategies without executing
python execution_strategy_tester.py --dry-run

# Run all models
python execution_strategy_tester.py --model all --workers 12
```

## Strategy Groups

### A. Entry Modes (`entry_modes`)
Compare fill rate vs cost tradeoff:
- **Passive limit at BBO** -- best price, lowest fill rate
- **Mid-price limit** -- between bid/ask, moderate fill rate
- **Chase entry** -- start passive, reprice to follow BBO
- **Chase + force cross** -- chase then market order if still unfilled
- **Market entry** -- guaranteed fill, pays full spread (2 ticks = $10)

### B. Confidence Gates (`confidence_gates`)
Vary signal z-score threshold from 1.0 to 5.0:
- Lower threshold = more trades, lower avg quality
- Higher threshold = fewer trades, higher conviction
- Top 1% signals (z>3.5+) historically show best per-trade economics

### C. Exit Strategies (`exit_strategies`)
- **Time-based exit** -- hold for 1s, 5s, 10s, 30s, 60s
- **Signal-flip exit** -- exit when model signal reverses
- **Conviction exit** -- delayed signal-flip (require N consecutive bars)
- **TP/SL** -- take-profit and stop-loss in ticks
- **Trailing stop** -- dynamic stop that follows favorable price movement
- **Ratchet stop** -- MFE-informed adaptive stop (locks in more as MFE grows)
- **Combined** -- signal-flip + TP/SL for bounded risk

### D. Latency Sensitivity (`latency`)
Test impact of order submission delay: 0ms, 5ms, 10ms, 20ms, 50ms.
Critical for understanding co-location requirements.

### E. Time-of-Day (`time_of_day`)
- **Prime hours** -- 10:30 AM - 2:30 PM ET (proven Sharpe lift)
- **Open** -- first 30 minutes (high volatility)
- **Close** -- last hour (mean reversion regime)

### F. Ultra-Selective (`ultra_selective`)
Monday deployment candidates: z>3.5 to z>5.0 thresholds.
Few trades per day but highest per-trade edge.

### G. Passive-then-Aggressive (`passive_then_aggressive`)
Start with passive limit, convert to market after timeout.
Balances fill rate with execution cost.

## Metrics Tracked

For each strategy:
- **Fill rate** -- % of signals that get filled
- **Avg fill time** -- milliseconds from signal to fill
- **Win rate** -- % of trades profitable
- **Avg trade P&L** -- in dollars and ticks
- **Avg winner / Avg loser** -- reward/risk ratio
- **Gross P&L** -- before costs
- **Net P&L** -- after commission ($4.12/RT) + slippage
- **Sortino ratio** -- risk-adjusted return (downside only)
- **Sharpe ratio** -- daily P&L based
- **Profit factor** -- gross profit / gross loss
- **Max drawdown** -- largest peak-to-trough
- **Trades per day** -- capacity/frequency
- **MFE/MAE** -- max favorable/adverse excursion per trade
- **Confidence tiers** -- performance at Top 1/5/10/25/50% signals

## Models

### CNN1D (Champion)
- **IC_10s = 0.132** (concat across 10+ folds)
- Per-event directional signal, 500-event window, stride 250
- Predictions: raw regression values, z-scored for fill sim

### Mamba (Multi-Horizon)
- **IC_10s = 0.07** on March data
- Predictions at 1s, 5s, 10s horizons
- Multi-horizon mode: use 1s for timing, 5s for direction, 10s for conviction
- Composite signal: sign(5s) * |1s| * (1 + |10s|), only when all horizons agree

### LGBM (Directional)
- **DA = 68.8% at Top 1%** signals
- Binary directional predictions with confidence scores

## NQ Futures Constants

- Tick size: 0.25 points ($5 per tick)
- Point value: $20 per point
- Commission: $4.12 round-trip per contract
- Market order cost: ~2 ticks ($10) for crossing spread
- RTH: 9:30 AM - 4:00 PM ET (6.5 hours, 234,000 100ms bars)

## Output

Results are saved to `execution/results/`:
- `exec_strategy_results_{model}_{timestamp}.json` -- full results
- `exec_strategy_test_{timestamp}.log` -- detailed log
- `sim_{timestamp}/` -- per-day per-strategy raw sim output

## Dependencies

- Python 3.8+ with numpy
- Rust fill_sim_cli binary: `rust_cache_builder/target/release/fill_sim_cli`
- Raw MBO data: `data/raw/mbo/glbx-mdp3-*.mbo.dbn.zst`
- Event data: `data/processed/mbo_events_smart_v3/` (for timestamps)
- Model predictions: `output/{model}/fold_*_oot_predictions.npz`
