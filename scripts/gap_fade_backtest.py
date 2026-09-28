#!/usr/bin/env python3
"""
Gap Fade Backtest — Fading large overnight gaps on growth stocks.
Academic research: gaps tend to fill (mean-revert intraday).

6 variants tested with 5-gate validation framework.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "SNOW",
    "DDOG", "NET", "CRWD", "SHOP", "SQ", "COIN", "MARA", "SOFI", "PLTR", "RBLX",
    "HOOD", "ARM", "SMCI", "MU", "AVGO", "NFLX", "UBER", "ABNB", "RIVN", "SNAP",
]

OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% slippage for stocks
MAX_CONCURRENT = 3
PERM_ITERATIONS = 1000

# Options constants for Variant F
OPTION_COMMISSION = 0.65
OPTION_SPREAD_PCT = 0.05
OPTION_BS_HAIRCUT = 0.30

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/gap_fade_results.json"


# ── DATA DOWNLOAD ──────────────────────────────────────────────────────
def download_data():
    """Download daily OHLCV for universe + SPY + VIX."""
    tickers = UNIVERSE + ["SPY", "^VIX"]
    # Download with extra buffer for 200-SMA calculation
    start = "2021-01-01"
    end = OOT_END

    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            # Flatten multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} rows")
            else:
                print(f"  {ticker}: SKIPPED (only {len(df)} rows)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    return data


# ── GAP DETECTION ──────────────────────────────────────────────────────
def compute_gaps(data):
    """Compute overnight gap percentage for each stock."""
    gaps = {}
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker].copy()
        df["prev_close"] = df["Close"].shift(1)
        df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
        gaps[ticker] = df
    return gaps


def get_regime_data(data):
    """Compute SPY 200-SMA regime and VIX level."""
    spy = data.get("SPY")
    vix = data.get("^VIX")

    regime = pd.DataFrame(index=spy.index)
    regime["spy_close"] = spy["Close"]
    regime["spy_sma200"] = spy["Close"].rolling(200).mean()
    regime["bull"] = regime["spy_close"] > regime["spy_sma200"]

    if vix is not None:
        regime["vix"] = vix["Close"].reindex(regime.index, method="ffill")
    else:
        regime["vix"] = 20.0  # default

    return regime


# ── BACKTESTER ─────────────────────────────────────────────────────────
def run_variant(gaps, regime, variant, initial_capital=INITIAL_CAPITAL):
    """
    Run a single variant backtest.
    Returns list of trades and equity curve.
    """
    # Collect all gap-down signals across stocks
    signals = []
    for ticker, df in gaps.items():
        oot_mask = (df.index >= OOT_START) & (df.index <= OOT_END)
        df_oot = df[oot_mask]
        for i in range(len(df_oot)):
            row = df_oot.iloc[i]
            date = df_oot.index[i]
            gap = row["gap_pct"]

            if pd.isna(gap):
                continue

            # Determine gap threshold per variant
            if variant in ("A", "C", "D", "E", "F"):
                threshold = -0.02
            elif variant == "B":
                threshold = -0.03
            else:
                continue

            if gap < threshold:  # gap DOWN
                signals.append({
                    "date": date,
                    "ticker": ticker,
                    "gap_pct": gap,
                    "open_price": row["Open"],
                    "close_price": row["Close"],
                })

    if not signals:
        return [], pd.Series(dtype=float)

    signals_df = pd.DataFrame(signals)
    signals_df = signals_df.sort_values(["date", "gap_pct"])  # most negative gap first

    # Apply filters per variant
    trades = []
    all_dates = sorted(signals_df["date"].unique())

    for date in all_dates:
        day_signals = signals_df[signals_df["date"] == date].copy()

        # Regime filter
        if date not in regime.index:
            continue

        regime_row = regime.loc[date] if date in regime.index else None
        if regime_row is None:
            continue

        # Variant D: VIX < 25 filter
        if variant == "D":
            if isinstance(regime_row, pd.DataFrame):
                vix_val = regime_row["vix"].iloc[0]
            else:
                vix_val = regime_row["vix"]
            if pd.isna(vix_val) or vix_val >= 25:
                continue

        # Variant E: SPY above 200-SMA filter
        if variant == "E":
            if isinstance(regime_row, pd.DataFrame):
                bull_val = regime_row["bull"].iloc[0]
            else:
                bull_val = regime_row["bull"]
            if not bull_val:
                continue

        # Variant C: top gap-down only
        if variant == "C":
            day_signals = day_signals.head(1)
        else:
            day_signals = day_signals.head(MAX_CONCURRENT)

        for _, sig in day_signals.iterrows():
            hold_days = 1 if variant != "B" else 3

            # For multi-day holds, we need the exit price
            ticker_data = gaps.get(sig["ticker"])
            if ticker_data is None:
                continue

            entry_date = sig["date"]
            entry_price = sig["open_price"]

            # Find exit date
            future_dates = ticker_data.index[ticker_data.index > entry_date]
            if variant == "A" or variant in ("C", "D", "E", "F"):
                # Same-day exit: buy at open, sell at close
                exit_price = sig["close_price"]
                exit_date = entry_date
            elif variant == "B":
                # Hold 3 days
                if len(future_dates) >= 3:
                    exit_date = future_dates[2]
                    exit_price = ticker_data.loc[exit_date, "Close"]
                elif len(future_dates) > 0:
                    exit_date = future_dates[-1]
                    exit_price = ticker_data.loc[exit_date, "Close"]
                else:
                    continue

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            # Compute return
            raw_ret = (exit_price - entry_price) / entry_price
            # Apply slippage (entry + exit)
            net_ret = raw_ret - 2 * SLIPPAGE_PCT

            # Variant F: options pricing
            if variant == "F":
                net_ret = _option_return(entry_price, exit_price, sig["gap_pct"])

            is_bull = regime_row["bull"] if not isinstance(regime_row, pd.DataFrame) else regime_row["bull"].iloc[0]

            trades.append({
                "date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                "ticker": sig["ticker"],
                "gap_pct": float(sig["gap_pct"]),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "raw_return": float(raw_ret),
                "net_return": float(net_ret),
                "regime": "bull" if is_bull else "bear",
            })

    if not trades:
        return [], pd.Series(dtype=float)

    # Build equity curve
    trades_df = pd.DataFrame(trades)
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    trades_df = trades_df.sort_values("date")

    # Simple: allocate equal weight per position, max MAX_CONCURRENT per day
    equity = initial_capital
    equity_curve = [{"date": str(trades_df["date"].iloc[0].date()), "equity": equity}]

    # Group by entry date
    for date, group in trades_df.groupby("date"):
        n_positions = len(group)
        allocation_per = equity / max(n_positions, 1)
        day_pnl = 0
        for _, t in group.iterrows():
            if variant == "F":
                # For options, net_return is already dollar P&L per contract
                day_pnl += t["net_return"]
            else:
                day_pnl += allocation_per * t["net_return"]
        equity += day_pnl
        equity_curve.append({"date": str(date.date()), "equity": equity})

    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")

    return trades, eq_df


def _option_return(entry_price, exit_price, gap_pct):
    """
    Simulate weekly call option return on gap-down stock.
    0.4-delta ATM call, 1-week expiry, BS pricing with 30% haircut.
    """
    # Simplified BS pricing for ATM call
    # ATM call with ~0.4 delta, 1 week to expiry
    S = entry_price
    K = S  # ATM
    T = 5 / 252  # 1 week
    r = 0.05
    sigma = max(abs(gap_pct) * 5, 0.30)  # implied vol estimate (min 30%)

    # BS call price (simplified)
    from math import log, sqrt, exp
    from scipy.stats import norm as norm_dist

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * norm_dist.cdf(d1) - K * exp(-r * T) * norm_dist.cdf(d2)

    # Apply 30% BS haircut (market premium over theoretical)
    call_price_market = call_price * (1 + OPTION_BS_HAIRCUT)

    # Apply 5% bid-ask spread
    entry_call_price = call_price_market * (1 + OPTION_SPREAD_PCT / 2)

    # Exit: price at close
    S_exit = exit_price
    T_exit = 4 / 252  # 4 days left
    d1_exit = (log(S_exit / K) + (r + 0.5 * sigma ** 2) * T_exit) / (sigma * sqrt(T_exit))
    d2_exit = d1_exit - sigma * sqrt(T_exit)
    call_price_exit = S_exit * norm_dist.cdf(d1_exit) - K * exp(-r * T_exit) * norm_dist.cdf(d2_exit)

    # Sell at bid (haircut on exit too)
    exit_call_price = call_price_exit * (1 - OPTION_SPREAD_PCT / 2)

    # P&L per contract (100 shares)
    pnl_per_contract = (exit_call_price - entry_call_price) * 100 - OPTION_COMMISSION

    return float(pnl_per_contract)


# ── VALIDATION ─────────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve, variant_name):
    """Compute performance metrics and 5-gate validation."""
    if not trades or len(trades) < 2:
        return {
            "variant": variant_name,
            "n_trades": len(trades),
            "gates_passed": 0,
            "gates_total": 5,
            "PASS": False,
            "reason": "Insufficient trades",
        }

    trades_df = pd.DataFrame(trades)
    returns = trades_df["net_return"].values

    n_trades = len(returns)
    total_return = float(np.prod(1 + returns) - 1) if variant_name != "F" else float(np.sum(returns))
    mean_ret = float(np.mean(returns))
    win_rate = float(np.sum(returns > 0) / n_trades)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    profit_factor = float(np.sum(winners) / abs(np.sum(losers))) if len(losers) > 0 and np.sum(losers) != 0 else float("inf")

    # Sharpe (annualized, assuming ~252 trading days)
    if np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(min(n_trades, 252)))
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(min(n_trades, 252)))
    else:
        sortino = float("inf") if mean_ret > 0 else 0.0

    # Max drawdown from equity curve
    if equity_curve is not None and len(equity_curve) > 0:
        eq_vals = equity_curve["equity"].values
        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / peak
        max_dd = float(np.min(dd))
    else:
        max_dd = 0.0

    # Regime analysis
    bull_trades = trades_df[trades_df["regime"] == "bull"]["net_return"]
    bear_trades = trades_df[trades_df["regime"] == "bear"]["net_return"]

    bull_sharpe = 0.0
    bear_sharpe = 0.0
    if len(bull_trades) > 1 and np.std(bull_trades) > 0:
        bull_sharpe = float(np.mean(bull_trades) / np.std(bull_trades) * np.sqrt(min(len(bull_trades), 252)))
    if len(bear_trades) > 1 and np.std(bear_trades) > 0:
        bear_sharpe = float(np.mean(bear_trades) / np.std(bear_trades) * np.sqrt(min(len(bear_trades), 252)))

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)

    # ── PERMUTATION TEST ──
    print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...")
    observed_mean = np.mean(returns)
    perm_means = np.zeros(PERM_ITERATIONS)
    rng = np.random.RandomState(42)
    for i in range(PERM_ITERATIONS):
        perm_means[i] = np.mean(rng.choice(returns, size=n_trades, replace=True) * rng.choice([-1, 1], size=n_trades))

    perm_p = float(np.mean(perm_means >= observed_mean))

    # ── 5-GATE VALIDATION ──
    gates = {
        "sharpe_gt_0.5": sharpe > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "max_dd_gt_neg50": max_dd > -0.50,
        "min_20_trades": n_trades >= 20,
    }
    gates_passed = sum(gates.values())

    result = {
        "variant": variant_name,
        "n_trades": n_trades,
        "total_return_pct": round(total_return * 100, 2) if variant_name != "F" else None,
        "total_return_dollars": round(total_return, 2) if variant_name == "F" else None,
        "mean_return_pct": round(mean_ret * 100, 4) if variant_name != "F" else round(mean_ret, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3) if sortino != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd * 100, 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_trades": int(len(bull_trades)),
        "bear_trades": int(len(bear_trades)),
        "perm_p_value": round(perm_p, 4),
        "gates": {k: bool(v) for k, v in gates.items()},
        "gates_passed": gates_passed,
        "gates_total": 5,
        "PASS": gates_passed == 5,
        "final_equity": round(float(equity_curve["equity"].iloc[-1]), 2) if equity_curve is not None and len(equity_curve) > 0 else INITIAL_CAPITAL,
    }

    return result


# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("GAP FADE BACKTEST — Growth Stock Overnight Gap Mean Reversion")
    print("=" * 70)

    # Download data
    data = download_data()
    if "SPY" not in data:
        print("FATAL: Could not download SPY data")
        return

    # Compute gaps and regime
    gaps = compute_gaps(data)
    regime = get_regime_data(data)

    print(f"\nStocks with data: {len(gaps)}")
    print(f"Regime data: {len(regime)} days")
    print(f"OOT period: {OOT_START} to {OOT_END}")
    print(f"Initial capital: ${INITIAL_CAPITAL}")

    # Run all variants
    variants = {
        "A": "Gap down >2%, buy open sell close (intraday)",
        "B": "Gap down >3%, buy open hold 3 days",
        "C": "TOP gap-down stock only, hold 1 day",
        "D": "Gap down >2% + VIX<25 filter, hold 1 day",
        "E": "Gap down >2% + SPY>200SMA (bull only), hold 1 day",
        "F": "Weekly calls on gap-down stocks (BS priced)",
    }

    results = {}
    all_results = []

    for var_code, var_desc in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {var_code}: {var_desc}")
        print(f"{'─' * 60}")

        trades, eq_curve = run_variant(gaps, regime, var_code)
        print(f"  Trades: {len(trades)}")

        metrics = compute_metrics(trades, eq_curve, f"{var_code}: {var_desc}")

        # Print summary
        print(f"  Sharpe: {metrics.get('sharpe', 'N/A')}")
        print(f"  Sortino: {metrics.get('sortino', 'N/A')}")
        print(f"  Win Rate: {metrics.get('win_rate', 'N/A')}")
        print(f"  PF: {metrics.get('profit_factor', 'N/A')}")
        print(f"  Max DD: {metrics.get('max_drawdown_pct', 'N/A')}%")
        print(f"  Perm p: {metrics.get('perm_p_value', 'N/A')}")
        print(f"  Gates: {metrics.get('gates_passed', 0)}/{metrics.get('gates_total', 5)} — {'PASS' if metrics.get('PASS') else 'FAIL'}")

        if metrics.get("gates"):
            for gate, passed in metrics["gates"].items():
                status = "PASS" if passed else "FAIL"
                print(f"    {status}: {gate}")

        results[var_code] = metrics
        all_results.append(metrics)

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Variant':<8} {'Trades':>7} {'Sharpe':>8} {'Sortino':>8} {'WR':>7} {'PF':>7} {'MaxDD%':>8} {'Gates':>7} {'Result':>7}")
    print("-" * 70)
    for var_code in variants:
        m = results[var_code]
        print(f"{var_code:<8} {m.get('n_trades', 0):>7} {m.get('sharpe', 0):>8} {str(m.get('sortino', 0)):>8} "
              f"{m.get('win_rate', 0):>7.1%} {str(m.get('profit_factor', 0)):>7} {m.get('max_drawdown_pct', 0):>7.1f}% "
              f"{m.get('gates_passed', 0)}/{m.get('gates_total', 5):>1}   {'PASS' if m.get('PASS') else 'FAIL':>5}")

    # Save results
    output = {
        "strategy": "gap_fade",
        "description": "Fading large overnight gaps on growth stocks",
        "universe_size": len(UNIVERSE),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks_with_data": list(gaps.keys()),
        "variants": results,
        "best_variant": max(results.keys(), key=lambda k: results[k].get("sharpe", -999)),
        "any_pass": any(r.get("PASS", False) for r in results.values()),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Best variant by Sharpe: {output['best_variant']}")
    print(f"Any variant passed 5-gate? {'YES' if output['any_pass'] else 'NO'}")


if __name__ == "__main__":
    main()
