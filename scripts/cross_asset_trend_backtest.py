#!/usr/bin/env python3
"""
Cross-Asset Trend Following Backtest (AQR / Managed Futures Style)
===================================================================
Academic basis: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"

Universe: SPY, QQQ, EFA, TLT, GLD, DBC (or GSG fallback)
6 variants (A-F) + 60/40 benchmark + QQQ buy-and-hold.

OOT: Jan 2022 – Jul 2026, starting capital $645.
Cost: $0 commission, 0.02% slippage per trade.
5-gate validation applied to each variant.
"""

import json, warnings, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────
UNIVERSE = ["SPY", "QQQ", "EFA", "TLT", "GLD"]  # DBC added dynamically
BENCH_60_40 = {"SPY": 0.6, "AGG": 0.4}
START = "2020-01-01"       # need 12-month lookback before OOT
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
INITIAL_CAPITAL = 645.0
SLIPPAGE_BPS = 2           # 0.02% per trade
PERM_SHUFFLES = 1000

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/cross_asset_trend_results.json")

# ── DATA DOWNLOAD ───────────────────────────────────────────────────────
print("=" * 70)
print("CROSS-ASSET TREND FOLLOWING BACKTEST")
print("Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum")
print("=" * 70)

# Try DBC first, fall back to GSG
commodity_ticker = "DBC"
all_tickers = UNIVERSE + [commodity_ticker, "AGG"]

print(f"\n[1/5] Downloading {len(all_tickers)} tickers (trying DBC for commodities)...")
sys.stdout.flush()

raw = yf.download(all_tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"].copy()
else:
    prices = raw[["Close"]].copy()
    prices.columns = all_tickers

# Check if DBC has data; if not, try GSG
if commodity_ticker not in prices.columns or prices[commodity_ticker].dropna().empty:
    print("  DBC unavailable, trying GSG...")
    commodity_ticker = "GSG"
    all_tickers = UNIVERSE + [commodity_ticker, "AGG"]
    raw = yf.download(all_tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

UNIVERSE_FINAL = UNIVERSE + [commodity_ticker]
print(f"  Using commodity ETF: {commodity_ticker}")
print(f"  Universe: {UNIVERSE_FINAL}")

prices = prices.ffill().dropna(how="all")
prices.index = pd.to_datetime(prices.index).tz_localize(None)

# SPY regime for regime gap analysis
spy_200 = prices["SPY"].rolling(200).mean()
regime = (prices["SPY"] > spy_200).astype(int)  # 1=bull, 0=bear

oot_mask = prices.index >= OOT_START
oot_dates = prices.index[oot_mask]

# Monthly rebalance dates (first trading day of each month in OOT)
monthly_dates = prices.loc[oot_mask].resample("MS").first().index
rebal_dates = []
for md in monthly_dates:
    candidates = prices.index[(prices.index >= md) & (prices.index < md + pd.DateOffset(days=10))]
    if len(candidates) > 0:
        rebal_dates.append(candidates[0])
rebal_dates = pd.DatetimeIndex(rebal_dates)

print(f"  Data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')} ({len(prices)} days)")
print(f"  OOT: {oot_dates[0].strftime('%Y-%m-%d')} to {oot_dates[-1].strftime('%Y-%m-%d')} ({len(oot_dates)} days)")
print(f"  Rebalance dates: {len(rebal_dates)}")
sys.stdout.flush()


# ── HELPERS ─────────────────────────────────────────────────────────────
def month_return_series(equity_curve):
    monthly = equity_curve.resample("ME").last()
    return monthly.pct_change().dropna()


def calc_metrics(equity_curve, trades_count):
    ec = equity_curve.dropna()
    if len(ec) < 30:
        return {}
    total_ret = ec.iloc[-1] / ec.iloc[0] - 1
    years = (ec.index[-1] - ec.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    daily_ret = ec.pct_change().dropna()
    sr = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    down = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 0 else 0

    running_max = ec.cummax()
    dd = (ec - running_max) / running_max
    max_dd = dd.min()

    mr = month_return_series(ec)
    wins = (mr > 0).sum()
    losses = (mr <= 0).sum()
    wr = wins / max(wins + losses, 1)
    pf = mr[mr > 0].sum() / abs(mr[mr < 0].sum()) if (mr < 0).any() else float("inf")

    # Regime analysis
    daily_ret_df = daily_ret.to_frame("ret")
    daily_ret_df["regime"] = regime.reindex(daily_ret_df.index).ffill()
    bull = daily_ret_df[daily_ret_df["regime"] == 1]["ret"]
    bear = daily_ret_df[daily_ret_df["regime"] == 0]["ret"]
    sr_bull = bull.mean() / bull.std() * np.sqrt(252) if len(bull) > 30 and bull.std() > 0 else 0
    sr_bear = bear.mean() / bear.std() * np.sqrt(252) if len(bear) > 30 and bear.std() > 0 else 0
    regime_gap = abs(sr_bull - sr_bear) / max(abs(sr_bull), abs(sr_bear), 1e-9)

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sr, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(min(pf, 99.0), 2),
        "win_rate_monthly": round(wr * 100, 1),
        "trades": trades_count,
        "final_equity": round(ec.iloc[-1], 2),
        "sharpe_bull": round(sr_bull, 3),
        "sharpe_bear": round(sr_bear, 3),
        "regime_gap": round(regime_gap, 3),
    }


def permutation_test_trend(variant_params, n_perms=PERM_SHUFFLES):
    """
    Permutation test: randomize signal timing by shuffling monthly
    trend decisions across assets. Tests whether the trend signal's
    TIMING adds value vs random in/out decisions with same avg exposure.
    Returns p-value.
    """
    tickers = UNIVERSE_FINAL

    # Compute actual monthly asset returns during OOT
    monthly_asset_rets = {}
    for t in tickers:
        t_rets = prices[t].loc[oot_dates].pct_change().fillna(0)
        # Group by rebalance periods
        monthly_asset_rets[t] = []
        for j in range(len(rebal_dates)):
            start = rebal_dates[j]
            end = rebal_dates[j + 1] if j + 1 < len(rebal_dates) else oot_dates[-1]
            mask = (t_rets.index >= start) & (t_rets.index <= end)
            period_ret = (1 + t_rets[mask]).prod() - 1
            monthly_asset_rets[t].append(period_ret)

    n_periods = len(monthly_asset_rets[tickers[0]])

    # Compute actual trend signals per period
    lookback = variant_params["lookback_days"]
    sma_confirm = variant_params.get("use_sma_confirm", False)
    top_n = variant_params.get("top_n", None)
    use_vp = variant_params.get("use_vol_parity", False)

    actual_signals = []  # list of dicts: {ticker: weight} per period
    for j, date in enumerate(rebal_dates):
        long_assets = []
        for t in tickers:
            sig = trend_signal_return(prices, t, date, lookback)
            if sma_confirm:
                sig = sig and trend_signal_sma(prices, t, date, 50)
            if sig:
                long_assets.append(t)

        if top_n is not None and len(long_assets) > top_n:
            mom = momentum_rank(prices, long_assets, date, lookback)
            long_assets = sorted(long_assets, key=lambda x: mom.get(x, -np.inf), reverse=True)[:top_n]

        weights = {t: 0.0 for t in tickers}
        if len(long_assets) > 0:
            if use_vp:
                vols = {}
                for t in long_assets:
                    v = realized_vol(prices, t, date, 60)
                    if pd.isna(v) or v < 1e-6:
                        v = 0.15
                    vols[t] = v
                inv_vols = {t: 1.0 / v for t, v in vols.items()}
                total_inv = sum(inv_vols.values())
                for t in long_assets:
                    weights[t] = inv_vols[t] / total_inv
            else:
                w = 1.0 / len(long_assets)
                for t in long_assets:
                    weights[t] = w
        actual_signals.append(weights)

    # Compute observed strategy return
    def portfolio_return(signals_list):
        total = 1.0
        for j in range(min(n_periods, len(signals_list))):
            period_ret = sum(signals_list[j].get(t, 0) * monthly_asset_rets[t][j]
                            for t in tickers)
            total *= (1 + period_ret)
        return total - 1

    observed_ret = portfolio_return(actual_signals)

    # Permutation: shuffle which periods get which signal configuration
    count_ge = 0
    for _ in range(n_perms):
        perm_idx = np.random.permutation(len(actual_signals))
        shuffled_signals = [actual_signals[i] for i in perm_idx]
        perm_ret = portfolio_return(shuffled_signals)
        if perm_ret >= observed_ret:
            count_ge += 1

    return count_ge / n_perms


def five_gate_validation(metrics, pval):
    """5-gate validation. Returns dict of gate results."""
    gates = {
        "sharpe_gt_0.5": metrics.get("sharpe", 0) > 0.5,
        "perm_p_lt_0.05": pval < 0.05,
        "regime_gap_lt_0.5": metrics.get("regime_gap", 1) < 0.5,
        "maxdd_gt_neg50": metrics.get("max_drawdown_pct", -100) > -50,
        "trades_gte_20": metrics.get("trades", 0) >= 20,
    }
    gates["pass_all"] = all(gates.values())
    return gates


def apply_slippage(old_weights, new_weights, tickers):
    """Calculate slippage cost from weight changes."""
    total_turnover = 0
    for t in tickers:
        old_w = old_weights.get(t, 0)
        new_w = new_weights.get(t, 0)
        total_turnover += abs(new_w - old_w)
    # Each unit of turnover costs SLIPPAGE_BPS on the traded notional
    cost_frac = total_turnover * SLIPPAGE_BPS / 10000
    return cost_frac


# ── TREND SIGNAL FUNCTIONS ──────────────────────────────────────────────
def trend_signal_return(prices_df, ticker, date, lookback_days):
    """Return-based trend signal: positive if N-day return > 0."""
    loc = prices_df.index.get_loc(date)
    if loc < lookback_days:
        return False
    past = prices_df[ticker].iloc[loc - lookback_days]
    current = prices_df[ticker].iloc[loc]
    if pd.isna(past) or pd.isna(current) or past == 0:
        return False
    return (current / past - 1) > 0


def trend_signal_sma(prices_df, ticker, date, sma_days=50):
    """SMA confirmation: price above N-day SMA."""
    loc = prices_df.index.get_loc(date)
    if loc < sma_days:
        return False
    sma = prices_df[ticker].iloc[loc - sma_days + 1: loc + 1].mean()
    current = prices_df[ticker].iloc[loc]
    return current > sma


def realized_vol(prices_df, ticker, date, window=60):
    """60-day realized volatility (annualized)."""
    loc = prices_df.index.get_loc(date)
    if loc < window:
        return np.nan
    rets = prices_df[ticker].iloc[loc - window: loc].pct_change().dropna()
    if len(rets) < window - 5:
        return np.nan
    return rets.std() * np.sqrt(252)


def momentum_rank(prices_df, tickers, date, lookback_days=252):
    """Rank tickers by momentum (return over lookback). Higher = better."""
    loc = prices_df.index.get_loc(date)
    if loc < lookback_days:
        return {}
    mom = {}
    for t in tickers:
        past = prices_df[t].iloc[loc - lookback_days]
        current = prices_df[t].iloc[loc]
        if pd.isna(past) or pd.isna(current) or past == 0:
            mom[t] = -np.inf
        else:
            mom[t] = current / past - 1
    return mom


# ── BACKTEST ENGINE ─────────────────────────────────────────────────────
def run_variant(variant_name, lookback_days, use_vol_parity=False,
                top_n=None, use_sma_confirm=False):
    """Run a single variant backtest."""
    tickers = UNIVERSE_FINAL
    equity = INITIAL_CAPITAL
    equity_series = {}
    current_weights = {t: 0.0 for t in tickers}
    trades = 0

    for i, date in enumerate(oot_dates):
        if date in rebal_dates:
            # Determine which assets are LONG
            long_assets = []
            for t in tickers:
                signal = trend_signal_return(prices, t, date, lookback_days)
                if use_sma_confirm:
                    signal = signal and trend_signal_sma(prices, t, date, 50)
                if signal:
                    long_assets.append(t)

            # Top-N filtering by momentum rank
            if top_n is not None and len(long_assets) > top_n:
                mom = momentum_rank(prices, long_assets, date, lookback_days)
                long_assets = sorted(long_assets, key=lambda x: mom.get(x, -np.inf), reverse=True)[:top_n]

            # Compute weights
            new_weights = {t: 0.0 for t in tickers}
            if len(long_assets) > 0:
                if use_vol_parity:
                    # Inverse volatility weighting
                    vols = {}
                    for t in long_assets:
                        v = realized_vol(prices, t, date, 60)
                        if pd.isna(v) or v < 1e-6:
                            v = 0.15  # default 15% vol
                        vols[t] = v
                    inv_vols = {t: 1.0 / v for t, v in vols.items()}
                    total_inv = sum(inv_vols.values())
                    for t in long_assets:
                        new_weights[t] = inv_vols[t] / total_inv
                else:
                    # Equal weight
                    w = 1.0 / len(long_assets)
                    for t in long_assets:
                        new_weights[t] = w

            # Count trades (weight changes)
            for t in tickers:
                if abs(new_weights[t] - current_weights[t]) > 0.01:
                    trades += 1

            # Apply slippage
            cost = apply_slippage(current_weights, new_weights, tickers)
            equity *= (1 - cost)
            current_weights = new_weights

        # Daily return
        if i > 0:
            prev_date = oot_dates[i - 1]
            daily_ret = 0.0
            for t in tickers:
                if current_weights[t] > 0:
                    p_now = prices[t].loc[date] if date in prices.index else np.nan
                    p_prev = prices[t].loc[prev_date] if prev_date in prices.index else np.nan
                    if not pd.isna(p_now) and not pd.isna(p_prev) and p_prev > 0:
                        daily_ret += current_weights[t] * (p_now / p_prev - 1)
            equity *= (1 + daily_ret)

        equity_series[date] = equity

    return pd.Series(equity_series), trades


# ── RUN ALL VARIANTS ────────────────────────────────────────────────────
print("\n[2/5] Running 6 variants...")
sys.stdout.flush()

variants = {
    "A_12mo_equal": {"lookback_days": 252, "use_vol_parity": False, "top_n": None, "use_sma_confirm": False},
    "B_6mo_equal": {"lookback_days": 126, "use_vol_parity": False, "top_n": None, "use_sma_confirm": False},
    "C_3mo_equal": {"lookback_days": 63, "use_vol_parity": False, "top_n": None, "use_sma_confirm": False},
    "D_12mo_volparity": {"lookback_days": 252, "use_vol_parity": True, "top_n": None, "use_sma_confirm": False},
    "E_12mo_top3": {"lookback_days": 252, "use_vol_parity": False, "top_n": 3, "use_sma_confirm": False},
    "F_12mo_sma50": {"lookback_days": 252, "use_vol_parity": False, "top_n": None, "use_sma_confirm": True},
}

results = {}
equity_curves = {}

for name, params in variants.items():
    ec, trades = run_variant(name, **params)
    equity_curves[name] = ec
    metrics = calc_metrics(ec, trades)
    results[name] = metrics
    status = "✓" if metrics.get("sharpe", 0) > 0.5 else "✗"
    print(f"  {status} {name}: Sharpe={metrics.get('sharpe', 'N/A')}, "
          f"Sortino={metrics.get('sortino', 'N/A')}, "
          f"MaxDD={metrics.get('max_drawdown_pct', 'N/A')}%, "
          f"Final=${metrics.get('final_equity', 'N/A')}, "
          f"Trades={trades}, "
          f"RegimeGap={metrics.get('regime_gap', 'N/A')}")
    sys.stdout.flush()


# ── BENCHMARKS ──────────────────────────────────────────────────────────
print("\n[3/5] Running benchmarks...")
sys.stdout.flush()

# QQQ Buy-and-Hold
qqq_oot = prices["QQQ"].loc[oot_dates]
qqq_ec = INITIAL_CAPITAL * qqq_oot / qqq_oot.iloc[0]
qqq_metrics = calc_metrics(qqq_ec, 1)
results["BENCH_QQQ_BH"] = qqq_metrics
print(f"  QQQ B&H: Sharpe={qqq_metrics.get('sharpe', 'N/A')}, Final=${qqq_metrics.get('final_equity', 'N/A')}")

# SPY Buy-and-Hold
spy_oot = prices["SPY"].loc[oot_dates]
spy_ec = INITIAL_CAPITAL * spy_oot / spy_oot.iloc[0]
spy_metrics = calc_metrics(spy_ec, 1)
results["BENCH_SPY_BH"] = spy_metrics
print(f"  SPY B&H: Sharpe={spy_metrics.get('sharpe', 'N/A')}, Final=${spy_metrics.get('final_equity', 'N/A')}")

# 60/40 (SPY/AGG)
if "AGG" in prices.columns:
    spy_ret = prices["SPY"].loc[oot_dates].pct_change().fillna(0)
    agg_ret = prices["AGG"].loc[oot_dates].pct_change().fillna(0)
    port_ret = 0.6 * spy_ret + 0.4 * agg_ret
    ec_6040 = INITIAL_CAPITAL * (1 + port_ret).cumprod()
    metrics_6040 = calc_metrics(ec_6040, 1)
    results["BENCH_60_40"] = metrics_6040
    print(f"  60/40: Sharpe={metrics_6040.get('sharpe', 'N/A')}, Final=${metrics_6040.get('final_equity', 'N/A')}")


# ── PERMUTATION TESTS ──────────────────────────────────────────────────
print("\n[4/5] Running permutation tests (1000 shuffles each)...")
sys.stdout.flush()

for name in variants:
    ec = equity_curves[name]
    pval = permutation_test_trend(variants[name])
    results[name]["perm_pval"] = round(pval, 4)

    # 5-gate validation
    gates = five_gate_validation(results[name], pval)
    results[name]["gates"] = gates
    gate_str = "PASS" if gates["pass_all"] else "FAIL"
    failed = [k for k, v in gates.items() if not v and k != "pass_all"]
    fail_detail = f" (failed: {', '.join(failed)})" if failed else ""
    print(f"  {name}: p={pval:.4f} → {gate_str}{fail_detail}")
    sys.stdout.flush()


# ── CROSS-ASSET ROTATION ANALYSIS ──────────────────────────────────────
print("\n[5/5] Cross-asset rotation analysis...")
sys.stdout.flush()

# For each variant, show how often each asset class was held
for name in ["A_12mo_equal", "D_12mo_volparity"]:
    params = variants[name]
    lookback = params["lookback_days"]
    sma_confirm = params["use_sma_confirm"]

    asset_months_long = {t: 0 for t in UNIVERSE_FINAL}
    total_months = 0

    for date in rebal_dates:
        total_months += 1
        for t in UNIVERSE_FINAL:
            signal = trend_signal_return(prices, t, date, lookback)
            if sma_confirm:
                signal = signal and trend_signal_sma(prices, t, date, 50)
            if signal:
                asset_months_long[t] += 1

    print(f"\n  {name} — asset exposure (months long / {total_months} total):")
    for t in UNIVERSE_FINAL:
        pct = asset_months_long[t] / total_months * 100
        bar = "█" * int(pct / 5)
        print(f"    {t:5s}: {asset_months_long[t]:2d}/{total_months} ({pct:5.1f}%) {bar}")


# ── SUMMARY TABLE ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — ALL VARIANTS + BENCHMARKS")
print("=" * 70)

header = f"{'Variant':<22s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR%':>7s} {'MaxDD%':>7s} {'PF':>6s} {'WR%':>5s} {'Trades':>6s} {'RGap':>5s} {'Final$':>8s} {'5Gate':>6s}"
print(header)
print("-" * len(header))

for name in list(variants.keys()) + ["BENCH_QQQ_BH", "BENCH_SPY_BH", "BENCH_60_40"]:
    m = results.get(name, {})
    if not m:
        continue
    gate_str = ""
    if "gates" in m:
        gate_str = "PASS" if m["gates"]["pass_all"] else "FAIL"
    print(f"{name:<22s} {m.get('sharpe', 0):>7.3f} {m.get('sortino', 0):>8.3f} "
          f"{m.get('cagr_pct', 0):>7.2f} {m.get('max_drawdown_pct', 0):>7.2f} "
          f"{m.get('profit_factor', 0):>6.2f} {m.get('win_rate_monthly', 0):>5.1f} "
          f"{m.get('trades', 0):>6d} {m.get('regime_gap', 0):>5.3f} "
          f"{m.get('final_equity', 0):>8.2f} {gate_str:>6s}")

# ── SAVE RESULTS ────────────────────────────────────────────────────────
output = {
    "strategy": "Cross-Asset Trend Following (AQR/Managed Futures Style)",
    "paper": "Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum",
    "universe": UNIVERSE_FINAL,
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "slippage_bps": SLIPPAGE_BPS,
    "run_timestamp": datetime.now().isoformat(),
    "variants": {},
    "benchmarks": {},
}

for name in variants:
    output["variants"][name] = results[name]

for name in ["BENCH_QQQ_BH", "BENCH_SPY_BH", "BENCH_60_40"]:
    if name in results:
        output["benchmarks"][name] = results[name]

# Find best variant
passing = {k: v for k, v in output["variants"].items()
           if v.get("gates", {}).get("pass_all", False)}
if passing:
    best = max(passing, key=lambda k: passing[k]["sharpe"])
    output["best_variant"] = best
    output["best_sharpe"] = passing[best]["sharpe"]
    output["recommendation"] = f"{best} passes all 5 gates with Sharpe {passing[best]['sharpe']}"
else:
    # Best by Sharpe even if failing
    best = max(output["variants"], key=lambda k: output["variants"][k].get("sharpe", 0))
    output["best_variant"] = best
    output["best_sharpe"] = output["variants"][best]["sharpe"]
    failed_gates = [k for k, v in output["variants"][best].get("gates", {}).items()
                    if not v and k != "pass_all"]
    output["recommendation"] = f"No variant passes all 5 gates. Best: {best} (Sharpe {output['variants'][best]['sharpe']}), failed: {', '.join(failed_gates)}"

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_PATH}")
print(f"\nRECOMMENDATION: {output['recommendation']}")
print("=" * 70)
