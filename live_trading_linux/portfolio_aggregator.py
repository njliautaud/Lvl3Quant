#!/usr/bin/env python3
"""
Portfolio Aggregator — Multi-Strategy R1 Tracker
=================================================

Reads NAV from each R1-passing paper engine and computes portfolio-level
metrics including:
  - Weighted NAV (configurable per-strategy weights)
  - Daily returns and running Sharpe/Sortino
  - Green/Red regime classification (SPY close-to-close)
  - R1 regime gap check (live, rolling)
  - Cross-strategy correlation (once we have enough data)

Runs as a cron job at 16:30 ET (market close) on weekdays.
Outputs daily snapshot to output/portfolio_agg/.

Author: Claude (2026-07-10)
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "portfolio_agg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy Configuration ──────────────────────────────────────────────
# Weights from honest_portfolio_optimizer (2026-07-13): V5 68%, IC 14%, ETF 18%
# BPS GA REMOVED — fails permutation test (p=0.085)
# All strategies here PASS R1 AND permutation test in backtest
STRATEGIES = {
    "wheel_v5": {
        "weight": 0.68,
        "nav_sources": [
            ROOT / "live_trading_linux" / "wheel_v5_state" / "nav_history.json",
        ],
        "state_file": ROOT / "live_trading_linux" / "wheel_v5_state" / "state.json",
        "r1_gap_backtest": 0.21,  # Combined hedge gap
    },
    "ic_condors": {
        "weight": 0.14,
        "nav_sources": [
            ROOT / "live_trading_linux" / "wheel_ic_state" / "nav_history.json",
        ],
        "state_file": ROOT / "live_trading_linux" / "wheel_ic_state" / "state.json",
        "r1_gap_backtest": 0.145,  # Combined hedge gap (permutation-corrected Sharpe 2.05)
    },
    "etf_rotation_v3": {
        "weight": 0.18,
        "nav_sources": [
            ROOT / "output" / "nav_snapshots" / "nav_timeseries.csv",  # From nav_snapshot.py
            ROOT / "live_trading_linux" / "etf_rotation_v3_state" / "nav_history.json",
        ],
        "state_file": ROOT / "live_trading_linux" / "etf_rotation_v3_state" / "state.json",
        "r1_gap_backtest": 0.007,  # Best regime gap of all strategies
    },
}

STARTING_CAPITAL = 100_000.0
RISK_FREE = 0.04
TRADING_DAYS = 252
RF_DAILY = RISK_FREE / TRADING_DAYS

# R1 regime gate threshold
R1_GAP_THRESHOLD = 0.50


def load_nav_from_json(path):
    """Load NAV history from nav_history.json."""
    if not path.exists():
        return pd.Series(dtype=float)
    try:
        data = json.loads(path.read_text())
        if not data:
            return pd.Series(dtype=float)
        dates = [pd.Timestamp(d["date"]).normalize() for d in data]
        navs = [d["nav"] for d in data]
        s = pd.Series(navs, index=dates)
        # Take last value per day
        return s.groupby(s.index).last()
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return pd.Series(dtype=float)


def load_nav_from_csv(path, strategy_col=None):
    """Load NAV from CSV (used by nav_snapshot.py format)."""
    if not path.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(path)
        if strategy_col and strategy_col in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")[strategy_col].dropna()
        elif "nav" in df.columns:
            df["date"] = pd.to_datetime(df.get("date", df.get("timestamp")))
            return df.set_index("date")["nav"].dropna()
    except Exception as e:
        print(f"  Error loading {path}: {e}")
    return pd.Series(dtype=float)


def get_spy_daily_returns(lookback_days=120):
    """Fetch SPY daily returns for regime classification."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", period=f"{lookback_days}d", progress=False)
        if spy.empty:
            return pd.Series(dtype=float)
        close = spy["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.pct_change().dropna()
    except Exception as e:
        print(f"  SPY fetch error: {e}")
        return pd.Series(dtype=float)


def classify_regime(spy_returns, sigma_threshold=0.005):
    """Classify each day as green/red/flat based on SPY close-to-close.
    Uses +/- 0.5sigma threshold (annualized ~8% vol → daily ~0.5%).
    """
    regimes = {}
    for date, ret in spy_returns.items():
        if ret > sigma_threshold:
            regimes[date] = "green"
        elif ret < -sigma_threshold:
            regimes[date] = "red"
        else:
            regimes[date] = "flat"
    return regimes


def compute_sharpe(returns, annualize=True):
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return np.nan
    excess = returns - RF_DAILY
    if annualize:
        return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
    return excess.mean() / excess.std()


def compute_sortino(returns, annualize=True):
    """Annualized Sortino ratio."""
    if len(returns) < 2:
        return np.nan
    excess = returns - RF_DAILY
    downside = returns[returns < 0]
    if len(downside) < 1:
        return np.inf
    downside_std = downside.std()
    if downside_std == 0:
        return np.inf
    if annualize:
        return excess.mean() / downside_std * np.sqrt(TRADING_DAYS)
    return excess.mean() / downside_std


def compute_r1_gap(returns, regimes):
    """Compute R1 regime gap: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)."""
    green_rets = returns[[d for d in returns.index if regimes.get(d) == "green"]]
    red_rets = returns[[d for d in returns.index if regimes.get(d) == "red"]]

    if len(green_rets) < 5 or len(red_rets) < 5:
        return np.nan, np.nan, np.nan

    sharpe_green = compute_sharpe(green_rets)
    sharpe_red = compute_sharpe(red_rets)

    denom = max(abs(sharpe_green), abs(sharpe_red))
    if denom == 0:
        return 0.0, sharpe_green, sharpe_red

    gap = abs(sharpe_green - sharpe_red) / denom
    return gap, sharpe_green, sharpe_red


def load_strategy_current_nav(name, config):
    """Load current NAV from state file."""
    state_file = config.get("state_file")
    if state_file and Path(state_file).exists():
        try:
            state = json.loads(Path(state_file).read_text())
            # Different formats: some have 'nav', some need to compute from cash+positions
            nav = state.get("nav")
            if nav is None:
                nav = state.get("portfolio_value")
            if nav is None:
                nav = state.get("cash", 0)
                # Add margin held for CSP engines
                for pos in state.get("positions", []):
                    nav += pos.get("margin_held", 0)
            return float(nav) if nav else None
        except Exception as e:
            print(f"  {name} state error: {e}")
    return None


def run_aggregation():
    """Main aggregation: compute portfolio NAV, returns, regime metrics."""
    print(f"\n{'='*60}")
    print(f"PORTFOLIO AGGREGATOR — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    # Load NAV histories for each strategy
    all_navs = {}
    current_navs = {}

    for name, config in STRATEGIES.items():
        print(f"\nLoading {name} (weight: {config['weight']:.0%})...")
        nav_series = pd.Series(dtype=float)

        for source in config["nav_sources"]:
            if str(source).endswith(".json"):
                ns = load_nav_from_json(source)
            elif str(source).endswith(".csv"):
                ns = load_nav_from_csv(source, strategy_col=name)
            else:
                continue
            if len(ns) > len(nav_series):
                nav_series = ns

        if len(nav_series) > 0:
            all_navs[name] = nav_series
            print(f"  NAV history: {len(nav_series)} obs, "
                  f"${nav_series.iloc[0]:,.0f} -> ${nav_series.iloc[-1]:,.0f}")

        # Current NAV from state
        current_nav = load_strategy_current_nav(name, config)
        if current_nav is not None:
            current_navs[name] = current_nav
            print(f"  Current NAV: ${current_nav:,.0f}")

    # Compute individual strategy returns
    print(f"\n--- INDIVIDUAL STRATEGY RETURNS ---")
    strategy_returns = {}
    for name, nav in all_navs.items():
        ret = nav.pct_change().dropna()
        if len(ret) > 0:
            strategy_returns[name] = ret
            total_ret = (1 + ret).prod() - 1
            print(f"  {name:25s}: {len(ret)} daily returns, total={total_ret:+.2%}")

    # Compute weighted portfolio NAV
    print(f"\n--- PORTFOLIO-LEVEL METRICS ---")
    if len(strategy_returns) < 2:
        print("  Not enough strategies with return data for portfolio analysis.")
        print(f"  Have {len(strategy_returns)} strategies, need at least 2.")
    else:
        # Align returns on common dates
        ret_df = pd.DataFrame(strategy_returns)
        common = ret_df.dropna()

        if len(common) < 2:
            print(f"  Only {len(common)} common dates across strategies. Need more data.")
        else:
            # Weighted portfolio return
            weights = np.array([STRATEGIES[col]["weight"] for col in common.columns])
            weights = weights / weights.sum()  # Normalize in case some strategies missing

            portfolio_returns = (common * weights).sum(axis=1)
            portfolio_nav = STARTING_CAPITAL * (1 + portfolio_returns).cumprod()

            sharpe = compute_sharpe(portfolio_returns)
            sortino = compute_sortino(portfolio_returns)
            total_ret = (1 + portfolio_returns).prod() - 1
            max_dd = ((1 + portfolio_returns).cumprod() /
                      (1 + portfolio_returns).cumprod().cummax() - 1).min()

            print(f"  Dates: {common.index[0].date()} to {common.index[-1].date()} "
                  f"({len(common)} days)")
            print(f"  Weights: {dict(zip(common.columns, weights.round(3)))}")
            print(f"  Total return: {total_ret:+.2%}")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Sortino: {sortino:.2f}")
            print(f"  MaxDD: {max_dd:.1%}")
            print(f"  Final NAV: ${portfolio_nav.iloc[-1]:,.0f}")

            # Regime analysis
            spy_returns = get_spy_daily_returns()
            if len(spy_returns) > 0:
                regimes = classify_regime(spy_returns)
                gap, sg, sr = compute_r1_gap(portfolio_returns, regimes)

                n_green = sum(1 for d in common.index if regimes.get(d) == "green")
                n_red = sum(1 for d in common.index if regimes.get(d) == "red")
                n_flat = sum(1 for d in common.index if regimes.get(d) == "flat")

                print(f"\n  REGIME ANALYSIS:")
                print(f"    Green days: {n_green}, Red days: {n_red}, Flat days: {n_flat}")
                if not np.isnan(gap):
                    print(f"    Sharpe (green): {sg:.2f}")
                    print(f"    Sharpe (red):   {sr:.2f}")
                    print(f"    R1 gap:         {gap:.3f} {'PASS' if gap <= R1_GAP_THRESHOLD else 'FAIL'}")
                else:
                    print(f"    R1 gap: insufficient data (need 5+ green and 5+ red days)")

            # Cross-strategy correlation
            print(f"\n  CORRELATION MATRIX:")
            corr = common.corr()
            for col in corr.columns:
                row = "    " + f"{col:20s}"
                for col2 in corr.columns:
                    row += f"  {corr.loc[col, col2]:+.3f}"
                print(row)

    # Save snapshot
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "strategies": {},
        "portfolio": {},
    }

    for name in STRATEGIES:
        snapshot["strategies"][name] = {
            "weight": STRATEGIES[name]["weight"],
            "current_nav": current_navs.get(name),
            "r1_gap_backtest": STRATEGIES[name]["r1_gap_backtest"],
            "n_daily_obs": len(strategy_returns.get(name, [])),
        }

    if len(strategy_returns) >= 2:
        ret_df = pd.DataFrame(strategy_returns)
        common = ret_df.dropna()
        if len(common) >= 2:
            weights = np.array([STRATEGIES[col]["weight"] for col in common.columns])
            weights = weights / weights.sum()
            portfolio_returns = (common * weights).sum(axis=1)
            snapshot["portfolio"] = {
                "n_days": len(common),
                "total_return": float((1 + portfolio_returns).prod() - 1),
                "sharpe": float(compute_sharpe(portfolio_returns)),
                "sortino": float(compute_sortino(portfolio_returns)),
                "max_dd": float(((1 + portfolio_returns).cumprod() /
                                (1 + portfolio_returns).cumprod().cummax() - 1).min()),
            }

    # Save snapshot
    snapshot_file = OUTPUT_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(snapshot_file, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"\nSnapshot saved: {snapshot_file.name}")

    # Also save latest
    latest_file = OUTPUT_DIR / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    run_aggregation()
