# Tier2 Balanced Scalp - Paper Trading Engine

Paper-only forward test of the Tier2 Balanced Scalp wheel ruleset.
No real broker orders. SPY single-name (free yfinance data).

## What it implements

Rules extracted verbatim from `wheel_strategy_v1/strategy/tiers.py::balanced_tier()`:

| Parameter         | Value      |
|-------------------|------------|
| Underlying        | SPY        |
| Short put delta   | -0.22      |
| DTE band          | 30-45 (target 37) |
| Profit take       | 50% of max (SCALP variant) |
| Roll trigger      | DTE <= 10 if tested/breached |
| VIX max gate      | 32         |
| Leverage          | 1.0x       |
| Starting cash     | $100,000   |
| Max collateral/trade | 5% of equity |

Decision cycle: every 5 minutes during daemon mode. Quote polls obey a 60s
minimum spacing per task constraint. Per cycle the engine:
1. Marks each open short put to BS price using stored entry IV and current spot
2. Closes at 50% of max profit (scalp exit)
3. Rolls if DTE <= 10 and spot within 0.5% of strike (tested)
4. Expires OTM puts for full premium credit at DTE=0
5. Opens a new short put if no SPY position is open, VIX gate passes, and the
   chain yields a valid -0.22 delta candidate in the 30-45 DTE band

## Data source

- Spot quotes:   `yfinance.Ticker("SPY").fast_info.last_price`
- Option chain:  `yfinance.Ticker("SPY").option_chain(expiration)`
- VIX:           `yfinance.Ticker("^VIX").fast_info.last_price`

Pre-market caveat: yfinance often returns IV=0 or IV<5% for OTM strikes
before US market open. The engine floors IV at `VIX/100 * 1.1` so the BS
delta solver continues to work. During RTH the contract-level IV is
honored directly.

## Smoke test (passed 2026-06-09 08:49 ET, pre-open)

Engine started from zero state, fetched SPY ($739.22) and VIX (18.0),
loaded chain across 30-45 DTE band, picked SPY 708P exp 2026-07-17
(38 DTE, BS delta -0.22, premium $6.23/share, $623 credit, 1 contract),
wrote state, logged equity/realized/unrealized to MLflow, exited cleanly.

## Files

- Engine:        `live_trading_linux/wheel_paper_engine.py`
- pm2 config:    `live_trading_linux/wheel_paper_ecosystem.config.js`
- State JSON:    `live_trading_linux/wheel_paper_state/state.json`
- Trade JSONL:   `live_trading_linux/wheel_paper_state/trades.jsonl`
- Equity CSV:    `live_trading_linux/wheel_paper_state/equity.csv`
- Log:           `live_trading_linux/logs/wheel_paper_engine.log`

## MLflow

- Tracking URI:  `http://localhost:5000`
- Experiment:    `paper-wheel-tier2-scalp`
- Metrics:       `equity`, `realized_pnl`, `unrealized_pnl`, `open_positions`,
                 `spot`, `vix`  (stepped by trade count)

## EOD report

Posts to the QCC alerts table (`/home/jupiter/teleclaude-main/data/qcc.db`)
with `source='wheel_paper_engine'`, severity `info`. The existing
`alert-router` pm2 process forwards these to Discord via the standard
alert pipeline. EOD trigger fires automatically once per UTC date after
21:00 UTC (5pm ET, post-close). Manual `--eod` flag also available.

## Start / stop / inspect

```bash
# START the daemon (registered, currently stopped)
pm2 start wheel-paper-engine

# STOP cleanly
pm2 stop wheel-paper-engine

# Live log tail
pm2 logs wheel-paper-engine

# One-shot smoke test (no daemon)
cd /home/jupiter/Lvl3Quant && python3 -m live_trading_linux.wheel_paper_engine --smoke

# Force an EOD report right now
cd /home/jupiter/Lvl3Quant && python3 -m live_trading_linux.wheel_paper_engine --eod

# Inspect state
cat /home/jupiter/Lvl3Quant/live_trading_linux/wheel_paper_state/state.json | jq .

# Inspect trade log
tail -20 /home/jupiter/Lvl3Quant/live_trading_linux/wheel_paper_state/trades.jsonl
```

## Kill switch

`pm2 stop wheel-paper-engine` halts the daemon. State persists. Restart
resumes from saved positions.

For a hard reset (forget all paper positions):
```bash
pm2 stop wheel-paper-engine
rm /home/jupiter/Lvl3Quant/live_trading_linux/wheel_paper_state/state.json
pm2 start wheel-paper-engine
```

## Known limitations

1. SINGLE-NAME (SPY) only. The full backtest used ~329 names with
   fundamentals + IV-rank gating. To extend, a real multi-name option
   chain feed (Polygon Options paid tier, Tradier, or similar) needs
   to be wired in. Free yfinance is fine for SPY but stale/incomplete
   for the broader universe.
2. IV is the contract-level yfinance IV (or VIX-floored fallback).
   No skew model. The backtest engine has a skew module
   (`strategy/iv_skew.py`); could be ported in if needed.
3. Paper fills happen at chain mid. No slippage model. The backtest
   uses a 2.5% spread-fraction slippage model on entry and exit.
4. NAAIM and fundamentals gates are not enforced in the paper engine
   (SPY-only mode skips them).
5. No assignment simulation. The Tier2 SCALP variant closes at 50% of
   max profit or rolls at DTE 10, so assignment is rare. If a put
   expires in-the-money it currently still "expires" with the modeled
   premium - acceptable for the scalp variant but would need a true
   assignment leg for the full-wheel variant.
