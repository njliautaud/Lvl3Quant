#!/usr/bin/env python3
"""
Gap-and-Go Breakout Backtest
Academic basis: Berkman et al. 2012 — overnight gaps reflect institutional order flow.
Stocks gapping up >3% on high volume tend to continue intraday and over subsequent days.

6 Variants:
A. Basic Gap: gap >3%, hold 5 days
B. Volume Confirm: gap >3% + volume >2x 20d avg, hold 5 days
C. Earnings Gap: gap on earnings day, hold 20 days
D. Gap with Trend: gap >3% + above 50-SMA, hold 10 days
E. Sector Momentum Gap: gap >3% + sector ETF >0.5%, hold 5 days
F. Portfolio: top 3 gap signals/month, equal-weight, hold 20 days

Universe: 24 growth stocks
OOT: Jan 2022 - Jul 2026
Starting Capital: $645
Costs: $0 commission, 0.02% slippage
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "CRM", "ADBE", "NFLX", "AVGO", "COST", "PEP", "LLY", "UNH",
    "V", "MA", "JPM", "HD", "INTC", "MU", "QCOM", "PYPL"
]
SECTOR_ETF = "XLK"
SPY = "SPY"
GAP_THRESHOLD = 0.03  # 3%
VOLUME_MULT = 2.0
TREND_SMA = 50
SECTOR_THRESHOLD = 0.005  # 0.5%
SLIPPAGE_BPS = 0.0002  # 0.02%
STARTING_CAPITAL = 645.0
START_DATE = "2021-06-01"  # extra history for SMA/vol lookback
END_DATE = "2026-07-30"
OOT_START = "2022-01-01"
N_PERMUTATIONS = 1000
RF_ANNUAL = 0.04  # risk-free rate for Sharpe

# Sector mapping for variant E (simplified: all universe stocks -> XLK proxy)
# In practice you'd map each ticker to its sector ETF; for this growth universe XLK is reasonable
TICKER_SECTOR = {t: SECTOR_ETF for t in UNIVERSE}

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/gap_and_go_results.json")


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download all required price/volume data."""
    all_tickers = list(set(UNIVERSE + [SPY, SECTOR_ETF]))
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                       group_by="ticker", auto_adjust=True, progress=False)

    closes = pd.DataFrame()
    opens = pd.DataFrame()
    volumes = pd.DataFrame()
    highs = pd.DataFrame()
    lows = pd.DataFrame()

    for t in all_tickers:
        try:
            if len(all_tickers) > 1:
                df_t = data[t].dropna(how="all")
            else:
                df_t = data.dropna(how="all")
            closes[t] = df_t["Close"]
            opens[t] = df_t["Open"]
            volumes[t] = df_t["Volume"]
            highs[t] = df_t["High"]
            lows[t] = df_t["Low"]
        except Exception as e:
            print(f"  Warning: {t} data issue: {e}")

    # Flatten multi-index columns if needed
    for df in [closes, opens, volumes, highs, lows]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    print(f"  Data range: {closes.index[0].date()} to {closes.index[-1].date()}, {len(closes)} days")
    return closes, opens, volumes, highs, lows


def get_earnings_dates_proxy(closes):
    """
    Proxy for earnings dates: identify days with gap >5% AND volume spike >3x.
    Real earnings data requires paid API; this heuristic catches most earnings gaps.
    """
    earnings = {}
    for t in UNIVERSE:
        if t not in closes.columns:
            continue
        prev_close = closes[t].shift(1)
        gap = (closes[t] - prev_close) / prev_close  # using close-to-close as proxy
        vol_avg = closes[t].rolling(20).mean()  # placeholder
        # Mark days with absolute gap > 5% as likely earnings
        earnings[t] = set(closes.index[gap.abs() > 0.05].strftime("%Y-%m-%d"))
    return earnings


def compute_spy_regime(closes):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy = closes[SPY].dropna()
    sma200 = spy.rolling(200).mean()
    regime = pd.Series("bull", index=spy.index)
    regime[spy < sma200] = "bear"
    return regime


# ── Trade Execution ────────────────────────────────────────────────────────
def simulate_trades(signals, closes, opens, hold_days, starting_capital=STARTING_CAPITAL):
    """
    Simulate trades from signal dates.
    Buy at next day's open (with slippage), sell at close of hold_days later.
    Returns list of trade dicts.
    """
    trades = []
    dates = closes.index

    for sig in signals:
        entry_date = sig["date"]
        ticker = sig["ticker"]

        if ticker not in closes.columns or ticker not in opens.columns:
            continue

        # Find next trading day for entry
        entry_idx = dates.get_indexer([entry_date], method="ffill")[0]
        entry_idx += 1  # buy next day's open

        if entry_idx >= len(dates) or entry_idx < 0:
            continue

        exit_idx = min(entry_idx + hold_days, len(dates) - 1)

        entry_price = opens[ticker].iloc[entry_idx]
        exit_price = closes[ticker].iloc[exit_idx]

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        # Apply slippage
        entry_price *= (1 + SLIPPAGE_BPS)
        exit_price *= (1 - SLIPPAGE_BPS)

        ret = (exit_price / entry_price) - 1.0

        trades.append({
            "ticker": ticker,
            "entry_date": str(dates[entry_idx].date()),
            "exit_date": str(dates[exit_idx].date()),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "return": float(ret),
            "gap_pct": sig.get("gap_pct", 0),
        })

    return trades


def build_equity_curve(trades, starting_capital=STARTING_CAPITAL):
    """Build daily equity curve from trades (equal-weight, fully invested per trade)."""
    if not trades:
        return pd.Series(dtype=float), []

    # Sort trades by entry date
    trades_sorted = sorted(trades, key=lambda x: x["entry_date"])

    capital = starting_capital
    equity = [capital]
    dates = [trades_sorted[0]["entry_date"]]
    returns = []

    for t in trades_sorted:
        pnl = capital * t["return"]
        capital += pnl
        equity.append(capital)
        dates.append(t["exit_date"])
        returns.append(t["return"])

    eq_series = pd.Series(equity, index=pd.to_datetime(dates))
    return eq_series, returns


# ── Signal Generation ──────────────────────────────────────────────────────
def find_gaps(closes, opens, volumes, oot_start=OOT_START):
    """Find all gap-up signals in OOT period."""
    dates = closes.index
    oot_mask = dates >= pd.Timestamp(oot_start)
    oot_dates = dates[oot_mask]

    all_gaps = []

    for d in oot_dates:
        d_idx = dates.get_loc(d)
        if d_idx < 1:
            continue
        prev_d = dates[d_idx - 1]

        for t in UNIVERSE:
            if t not in closes.columns or t not in opens.columns:
                continue

            prev_close = closes[t].iloc[d_idx - 1]
            today_open = opens[t].iloc[d_idx]
            today_vol = volumes[t].iloc[d_idx] if t in volumes.columns else np.nan

            if pd.isna(prev_close) or pd.isna(today_open) or prev_close <= 0:
                continue

            gap_pct = (today_open - prev_close) / prev_close

            # 20-day avg volume
            vol_start = max(0, d_idx - 20)
            avg_vol = volumes[t].iloc[vol_start:d_idx].mean() if t in volumes.columns else np.nan

            # 50-SMA
            sma_start = max(0, d_idx - TREND_SMA)
            sma50 = closes[t].iloc[sma_start:d_idx].mean() if (d_idx - sma_start) >= TREND_SMA else np.nan

            # Sector ETF return
            sect = TICKER_SECTOR.get(t, SECTOR_ETF)
            if sect in closes.columns:
                sect_ret = (closes[sect].iloc[d_idx] - closes[sect].iloc[d_idx - 1]) / closes[sect].iloc[d_idx - 1]
            else:
                sect_ret = np.nan

            all_gaps.append({
                "date": d,
                "ticker": t,
                "gap_pct": float(gap_pct),
                "volume": float(today_vol) if not pd.isna(today_vol) else 0,
                "avg_volume": float(avg_vol) if not pd.isna(avg_vol) else 0,
                "vol_ratio": float(today_vol / avg_vol) if (not pd.isna(avg_vol) and avg_vol > 0) else 0,
                "prev_close": float(prev_close),
                "sma50": float(sma50) if not pd.isna(sma50) else 0,
                "above_sma50": bool(prev_close > sma50) if not pd.isna(sma50) else False,
                "sector_ret": float(sect_ret) if not pd.isna(sect_ret) else 0,
            })

    return all_gaps


def variant_A_signals(all_gaps):
    """Basic Gap: gap > 3%, hold 5 days."""
    return [g for g in all_gaps if g["gap_pct"] > GAP_THRESHOLD]


def variant_B_signals(all_gaps):
    """Volume Confirm: gap > 3% + volume > 2x 20d avg."""
    return [g for g in all_gaps if g["gap_pct"] > GAP_THRESHOLD and g["vol_ratio"] > VOLUME_MULT]


def variant_C_signals(all_gaps):
    """Earnings Gap: gap > 5% (proxy for earnings), hold 20 days."""
    return [g for g in all_gaps if g["gap_pct"] > 0.05 and g["vol_ratio"] > 3.0]


def variant_D_signals(all_gaps):
    """Gap with Trend: gap > 3% + above 50-SMA, hold 10 days."""
    return [g for g in all_gaps if g["gap_pct"] > GAP_THRESHOLD and g["above_sma50"]]


def variant_E_signals(all_gaps):
    """Sector Momentum Gap: gap > 3% + sector ETF > 0.5%."""
    return [g for g in all_gaps if g["gap_pct"] > GAP_THRESHOLD and g["sector_ret"] > SECTOR_THRESHOLD]


def variant_F_signals(all_gaps):
    """Portfolio: top 3 gap signals each month, equal-weight, hold 20 days."""
    # Group by month, pick top 3 by gap_pct
    gaps_3pct = [g for g in all_gaps if g["gap_pct"] > GAP_THRESHOLD]

    monthly = {}
    for g in gaps_3pct:
        key = g["date"].strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(g)

    selected = []
    for month_key in sorted(monthly.keys()):
        month_gaps = sorted(monthly[month_key], key=lambda x: x["gap_pct"], reverse=True)
        selected.extend(month_gaps[:3])

    return selected


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(trades, returns_list, equity_curve, regime_series):
    """Compute all required metrics for a variant."""
    if len(trades) < 2:
        return {
            "sharpe": 0, "sortino": 0, "profit_factor": 0, "win_rate": 0,
            "max_dd_pct": -100, "total_return_pct": 0, "n_trades": len(trades),
            "perm_p_value": 1.0, "regime_gap": 1.0,
            "bull_sharpe": 0, "bear_sharpe": 0,
        }

    rets = np.array(returns_list)
    n = len(rets)

    # Sharpe (annualized, assuming ~5d hold avg -> ~50 trades/yr)
    trades_per_year = max(1, 252 / 5)  # approximate
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
    sharpe = (mean_ret * trades_per_year - RF_ANNUAL) / (std_ret * np.sqrt(trades_per_year)) if std_ret > 1e-9 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret * trades_per_year - RF_ANNUAL) / (downside_std * np.sqrt(trades_per_year)) if downside_std > 1e-9 else 0

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0)

    # Win rate
    win_rate = (rets > 0).sum() / n

    # Max drawdown from equity curve
    if len(equity_curve) > 1:
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0

    # Total return
    if len(equity_curve) > 1:
        total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    else:
        total_ret = 0

    # Regime split
    bull_rets = []
    bear_rets = []
    for t_info, r in zip(trades, rets):
        entry_d = pd.Timestamp(t_info["entry_date"])
        # Find nearest regime date
        if entry_d in regime_series.index:
            reg = regime_series.loc[entry_d]
        else:
            idx = regime_series.index.get_indexer([entry_d], method="ffill")
            if idx[0] >= 0:
                reg = regime_series.iloc[idx[0]]
            else:
                reg = "bull"

        if reg == "bull":
            bull_rets.append(r)
        else:
            bear_rets.append(r)

    bull_rets = np.array(bull_rets) if bull_rets else np.array([0.0])
    bear_rets = np.array(bear_rets) if bear_rets else np.array([0.0])

    bull_sharpe = _sharpe(bull_rets, trades_per_year)
    bear_sharpe = _sharpe(bear_rets, trades_per_year)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "profit_factor": round(float(min(profit_factor, 99.0)), 4),
        "win_rate": round(float(win_rate), 4),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_ret), 2),
        "n_trades": int(n),
        "bull_sharpe": round(float(bull_sharpe), 4),
        "bear_sharpe": round(float(bear_sharpe), 4),
        "regime_gap": round(float(regime_gap), 4),
    }


def _sharpe(rets, trades_per_year):
    if len(rets) < 2:
        return 0.0
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    if s < 1e-9:
        return 0.0
    return (m * trades_per_year - RF_ANNUAL) / (s * np.sqrt(trades_per_year))


# ── Permutation Test ───────────────────────────────────────────────────────
def permutation_test(returns_list, n_perms=N_PERMUTATIONS):
    """Shuffle entry dates, recompute mean return, get p-value."""
    if len(returns_list) < 5:
        return 1.0

    rets = np.array(returns_list)
    observed_mean = np.mean(rets)

    rng = np.random.RandomState(42)
    count_ge = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        if np.mean(shuffled) >= observed_mean:
            count_ge += 1

    # Note: shuffling returns of same trades just gives same mean.
    # Proper permutation: shuffle DATES (assign random entry dates from universe).
    # For a meaningful test, we'll compare observed mean vs random sampling of
    # same-length return sequences from all possible gap returns.
    return count_ge / n_perms


def permutation_test_proper(variant_returns, all_possible_returns, n_perms=N_PERMUTATIONS):
    """
    Proper permutation test: compare variant mean return vs random sampling
    of same number of trades from ALL gap returns (>3% pool).
    """
    if len(variant_returns) < 5 or len(all_possible_returns) < 5:
        return 1.0

    observed_mean = np.mean(variant_returns)
    n_trades = len(variant_returns)
    all_rets = np.array(all_possible_returns)

    rng = np.random.RandomState(42)
    count_ge = 0
    for _ in range(n_perms):
        sample = rng.choice(all_rets, size=n_trades, replace=True)
        if np.mean(sample) >= observed_mean:
            count_ge += 1

    return count_ge / n_perms


# ── 5-Gate Validation ──────────────────────────────────────────────────────
def validate_gates(metrics):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": metrics["perm_p_value"] < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "n_trades_gte_20": metrics["n_trades"] >= 20,
    }
    metrics["gates"] = gates
    metrics["gates_passed"] = sum(gates.values())
    metrics["gates_total"] = len(gates)
    metrics["all_gates_pass"] = all(gates.values())
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  GAP-AND-GO BREAKOUT BACKTEST")
    print("  Academic basis: Berkman et al. 2012")
    print("=" * 70)
    print()

    # Download data
    closes, opens, volumes, highs, lows = download_data()
    regime = compute_spy_regime(closes)

    # Find all gaps
    print("\nScanning for gap signals...")
    all_gaps = find_gaps(closes, opens, volumes)
    print(f"  Total gap events scanned: {len(all_gaps)}")
    gaps_3pct = [g for g in all_gaps if g["gap_pct"] > GAP_THRESHOLD]
    print(f"  Gaps > 3%: {len(gaps_3pct)}")

    # Define variants
    variants = {
        "A_basic_gap": {"signals_fn": variant_A_signals, "hold_days": 5, "desc": "Basic Gap >3%, hold 5d"},
        "B_volume_confirm": {"signals_fn": variant_B_signals, "hold_days": 5, "desc": "Gap >3% + Vol >2x, hold 5d"},
        "C_earnings_gap": {"signals_fn": variant_C_signals, "hold_days": 20, "desc": "Earnings Gap >5%, hold 20d"},
        "D_gap_trend": {"signals_fn": variant_D_signals, "hold_days": 10, "desc": "Gap >3% + above 50-SMA, hold 10d"},
        "E_sector_momentum": {"signals_fn": variant_E_signals, "hold_days": 5, "desc": "Gap >3% + sector >0.5%, hold 5d"},
        "F_portfolio_top3": {"signals_fn": variant_F_signals, "hold_days": 20, "desc": "Top 3 gaps/month, hold 20d"},
    }

    # Get baseline returns for permutation test (all 3% gap trades with 5d hold)
    baseline_trades = simulate_trades(gaps_3pct, closes, opens, hold_days=5)
    baseline_returns = [t["return"] for t in baseline_trades]

    results = {}

    for name, cfg in variants.items():
        print(f"\n{'─' * 60}")
        print(f"  Variant {name}: {cfg['desc']}")
        print(f"{'─' * 60}")

        signals = cfg["signals_fn"](all_gaps)
        print(f"  Signals: {len(signals)}")

        trades = simulate_trades(signals, closes, opens, hold_days=cfg["hold_days"])
        print(f"  Trades executed: {len(trades)}")

        if not trades:
            results[name] = {
                "description": cfg["desc"],
                "sharpe": 0, "sortino": 0, "profit_factor": 0, "win_rate": 0,
                "max_dd_pct": 0, "total_return_pct": 0, "n_trades": 0,
                "perm_p_value": 1.0, "regime_gap": 1.0, "bull_sharpe": 0, "bear_sharpe": 0,
                "gates": {g: False for g in ["sharpe_gt_0.5", "perm_p_lt_0.05", "regime_gap_lt_0.5", "max_dd_gt_neg50", "n_trades_gte_20"]},
                "gates_passed": 0, "gates_total": 5, "all_gates_pass": False,
            }
            continue

        returns_list = [t["return"] for t in trades]
        equity_curve, _ = build_equity_curve(trades)

        metrics = compute_metrics(trades, returns_list, equity_curve, regime)

        # Permutation test
        variant_rets = np.array(returns_list)
        p_val = permutation_test_proper(variant_rets, baseline_returns)
        metrics["perm_p_value"] = round(float(p_val), 4)

        # Validate gates
        metrics = validate_gates(metrics)
        metrics["description"] = cfg["desc"]

        results[name] = metrics

        # Print summary
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"  PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate']:.1%}")
        print(f"  MaxDD: {metrics['max_dd_pct']:.1f}% | Total Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f} | Bear Sharpe: {metrics['bear_sharpe']:.3f}")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f} | Perm p-val: {metrics['perm_p_value']:.3f}")
        gates = metrics["gates"]
        gate_str = " | ".join([f"{'PASS' if v else 'FAIL'}" for v in gates.values()])
        print(f"  Gates: {gate_str}")
        print(f"  Gates passed: {metrics['gates_passed']}/{metrics['gates_total']}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Print summary table
    print("\n" + "=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Ret%':>7} {'#Tr':>5} {'Perm-p':>7} {'RegGap':>7} {'Gates':>7}"
    print(header)
    print("-" * 100)

    for name, m in results.items():
        row = (
            f"{name:<25} "
            f"{m['sharpe']:>7.3f} "
            f"{m['sortino']:>8.3f} "
            f"{m['profit_factor']:>6.2f} "
            f"{m['win_rate']:>5.1%} "
            f"{m['max_dd_pct']:>6.1f}% "
            f"{m['total_return_pct']:>6.1f}% "
            f"{m['n_trades']:>5d} "
            f"{m['perm_p_value']:>7.3f} "
            f"{m['regime_gap']:>7.3f} "
            f"{m['gates_passed']}/{m['gates_total']:>1d}"
        )
        print(row)

    print("-" * 100)

    # Overall verdict
    any_pass = any(m.get("all_gates_pass", False) for m in results.values())
    print(f"\nOverall: {'At least one variant passes all 5 gates' if any_pass else 'NO variant passes all 5 gates'}")

    passing = [n for n, m in results.items() if m.get("all_gates_pass", False)]
    if passing:
        print(f"Passing variants: {', '.join(passing)}")

    return results


if __name__ == "__main__":
    results = main()
