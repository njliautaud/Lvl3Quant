#!/usr/bin/env python3
"""
Institutional Flow Proxy Backtest
6 variants testing institutional flow detection via public volume data on sector ETFs.
Walk-forward OOT: Jan 2022 to Jul 2026. Initial capital: $645. Commission: $0, 0.02% slippage.
"""

import json
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
START_DATE = "2021-01-01"  # extra lookback for 200-SMA and volume averages
OOT_START = "2022-01-01"
END_DATE = dt.date.today().isoformat()
N_PERMUTATIONS = 1000
RANDOM_SEED = 42
SLIPPAGE_PCT = 0.0002  # 0.02%
MAX_POSITIONS = 3

ETF_UNIVERSE = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
SECTOR_ETFS = [t for t in ETF_UNIVERSE if t not in ("SPY", "QQQ")]

# 5 validation gates
GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.50,
    "max_dd_floor": -0.50,
    "min_trades": 20,
}

# ── DATA DOWNLOAD ──────────────────────────────────────────────────────────
print("Downloading price data...")
raw = yf.download(ETF_UNIVERSE, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Handle multi-level columns from yf.download
if isinstance(raw.columns, pd.MultiIndex):
    price = raw["Close"].copy()
    volume = raw["Volume"].copy()
else:
    # Single ticker fallback
    price = raw[["Close"]].copy()
    volume = raw[["Volume"]].copy()

price = price.ffill()
volume = volume.ffill().fillna(0)

spy = price["SPY"]
spy_sma200 = spy.rolling(200).mean()
regime = (spy > spy_sma200).astype(int)  # 1=bull, 0=bear

print(f"Data loaded: {price.index[0].date()} to {price.index[-1].date()}, {len(price)} days")


# ── HELPERS ─────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Compute RSI for a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def apply_slippage(ret, n_trades_in_basket=1):
    """Apply round-trip slippage (entry + exit)."""
    return ret - 2 * SLIPPAGE_PCT


def compute_metrics(equity_curve, trade_returns, regime_series, trade_dates):
    """Compute Sharpe, Sortino, PF, WR, permutation p-value, regime gap, MDD, trade count."""
    daily_returns = equity_curve.pct_change().dropna()
    n_trades = len(trade_returns)

    # Sharpe (annualised)
    if daily_returns.std() == 0 or len(daily_returns) < 10:
        sharpe = 0.0
    else:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    # Sortino (annualised)
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float(daily_returns.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = 0.0

    # Profit factor
    gains = trade_returns[trade_returns > 0].sum() if n_trades > 0 else 0
    losses = abs(trade_returns[trade_returns < 0].sum()) if n_trades > 0 else 0
    pf = float(gains / losses) if losses > 0 else (99.0 if gains > 0 else 0.0)

    # Win rate
    wr = float((trade_returns > 0).sum() / n_trades) if n_trades > 0 else 0.0

    # Max drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_dd = float(dd.min())

    # Regime-split Sharpe
    bull_sharpe = 0.0
    bear_sharpe = 0.0
    regime_gap = 1.0
    if n_trades > 0 and len(trade_dates) == n_trades:
        regime_at_trade = regime.reindex(trade_dates, method="ffill")
        bull_mask = regime_at_trade.values == 1
        bear_mask = ~bull_mask

        bull_rets = trade_returns[bull_mask] if bull_mask.sum() > 0 else np.array([0.0])
        bear_rets = trade_returns[bear_mask] if bear_mask.sum() > 0 else np.array([0.0])

        if len(bull_rets) > 1 and bull_rets.std() > 0:
            bull_sharpe = float(bull_rets.mean() / bull_rets.std() * np.sqrt(252))
        if len(bear_rets) > 1 and bear_rets.std() > 0:
            bear_sharpe = float(bear_rets.mean() / bear_rets.std() * np.sqrt(252))

        denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)
        regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    # Permutation test
    if n_trades >= 5:
        observed_mean = float(trade_returns.mean())
        rng = np.random.RandomState(RANDOM_SEED)
        count_ge = 0
        for _ in range(N_PERMUTATIONS):
            flips = rng.choice([-1, 1], size=n_trades)
            perm_mean = float((trade_returns * flips).mean())
            if perm_mean >= observed_mean:
                count_ge += 1
        perm_p = count_ge / N_PERMUTATIONS
    else:
        perm_p = 1.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "perm_p": round(perm_p, 4),
        "regime_gap": round(regime_gap, 3),
        "sharpe_bull": round(bull_sharpe, 3),
        "sharpe_bear": round(bear_sharpe, 3),
        "max_dd": round(max_dd, 4),
        "n_trades": n_trades,
        "final_equity": round(float(equity_curve.iloc[-1]), 2),
    }


def check_gates(m):
    passed = []
    if m["sharpe"] >= GATES["sharpe_min"]:
        passed.append("sharpe")
    if m["perm_p"] <= GATES["perm_p_max"]:
        passed.append("perm_p")
    if m["regime_gap"] <= GATES["regime_gap_max"]:
        passed.append("regime_gap")
    if m["max_dd"] >= GATES["max_dd_floor"]:
        passed.append("max_dd")
    if m["n_trades"] >= GATES["min_trades"]:
        passed.append("min_trades")
    return passed


# Pre-compute volume ratios for all ETFs
print("Pre-computing volume indicators...")
vol_ratio_20 = {}  # volume / 20-day avg volume
dollar_volume = {}
dollar_vol_avg20 = {}
rsi_14 = {}
sma_20 = {}

for tk in ETF_UNIVERSE:
    if tk not in volume.columns:
        continue
    avg20 = volume[tk].rolling(20).mean()
    vol_ratio_20[tk] = volume[tk] / avg20.replace(0, np.nan)
    dv = price[tk] * volume[tk]
    dollar_volume[tk] = dv
    dollar_vol_avg20[tk] = dv.rolling(20).mean()
    rsi_14[tk] = compute_rsi(price[tk], 14)
    sma_20[tk] = price[tk].rolling(20).mean()


# ── VARIANT A: VOLUME BREADTH ─────────────────────────────────────────────
def run_variant_a():
    """
    Count how many of 13 sector ETFs have volume >1.5x avg.
    If >8/13 have high vol -> buy the top 3 by vol ratio. Hold 5d.
    """
    print("\n[A] Volume Breadth...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []
    positions = []  # list of (exit_date_idx, return_pct)

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]

        # Check how many ETFs have vol > 1.5x 20d avg
        high_vol_tickers = []
        for tk in ETF_UNIVERSE:
            if tk not in vol_ratio_20:
                continue
            vr = vol_ratio_20[tk]
            if date in vr.index and not np.isnan(vr.loc[date]) and vr.loc[date] > 1.5:
                high_vol_tickers.append((tk, vr.loc[date]))

        if len(high_vol_tickers) >= 8:
            # Sort by volume ratio descending, pick top 3
            high_vol_tickers.sort(key=lambda x: -x[1])
            top = high_vol_tickers[:MAX_POSITIONS]
            hold_days = 5
            per_pos = equity / len(top)
            total_ret = 0.0

            loc_date = price.index.get_loc(date)
            for tk, _ in top:
                p = price[tk]
                entry_price = p.iloc[loc_date]
                exit_loc = min(loc_date + hold_days, len(p) - 1)
                exit_price = p.iloc[exit_loc]
                ret = (exit_price / entry_price) - 1
                ret = apply_slippage(ret)
                total_ret += ret / len(top)

            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += hold_days
        else:
            i += 1

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    print(f"    Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}, "
          f"Final: ${metrics['final_equity']}")
    return metrics


# ── VARIANT B: ROTATION DETECTION ─────────────────────────────────────────
def run_variant_b():
    """
    If XLK volume > 2x avg BUT XLP volume < 1x avg -> rotation from defensive to offensive.
    Buy XLK, hold 10d. And vice versa (XLP high, XLK low -> buy XLP).
    """
    print("\n[B] Rotation Detection...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    offensive = ["XLK", "XLY", "XLC"]
    defensive = ["XLP", "XLU", "XLV"]

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        loc_date = price.index.get_loc(date)
        hold_days = 10
        signal = None

        # Check offensive vs defensive volume
        off_high = 0
        def_low = 0
        off_low = 0
        def_high = 0

        for tk in offensive:
            if tk in vol_ratio_20 and date in vol_ratio_20[tk].index:
                vr = vol_ratio_20[tk].loc[date]
                if not np.isnan(vr):
                    if vr > 2.0:
                        off_high += 1
                    if vr < 1.0:
                        off_low += 1

        for tk in defensive:
            if tk in vol_ratio_20 and date in vol_ratio_20[tk].index:
                vr = vol_ratio_20[tk].loc[date]
                if not np.isnan(vr):
                    if vr < 1.0:
                        def_low += 1
                    if vr > 2.0:
                        def_high += 1

        buy_tickers = []
        if off_high >= 2 and def_low >= 2:
            # Rotation into offensive
            buy_tickers = offensive
        elif def_high >= 2 and off_low >= 2:
            # Rotation into defensive
            buy_tickers = defensive

        if buy_tickers:
            total_ret = 0.0
            n = min(len(buy_tickers), MAX_POSITIONS)
            for tk in buy_tickers[:n]:
                p = price[tk]
                entry_price = p.iloc[loc_date]
                exit_loc = min(loc_date + hold_days, len(p) - 1)
                exit_price = p.iloc[exit_loc]
                ret = (exit_price / entry_price) - 1
                ret = apply_slippage(ret)
                total_ret += ret / n

            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += hold_days
        else:
            i += 1

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    print(f"    Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}, "
          f"Final: ${metrics['final_equity']}")
    return metrics


# ── VARIANT C: DOLLAR VOLUME SURGE ────────────────────────────────────────
def run_variant_c():
    """
    Track dollar volume (price x volume). Buy sector ETFs where
    dollar volume > 2x 20-day avg. Hold 5d.
    """
    print("\n[C] Dollar Volume Surge...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        loc_date = price.index.get_loc(date)
        hold_days = 5

        # Find ETFs with dollar volume surge
        surges = []
        for tk in SECTOR_ETFS:
            if tk not in dollar_volume or tk not in dollar_vol_avg20:
                continue
            dv = dollar_volume[tk]
            dv_avg = dollar_vol_avg20[tk]
            if date not in dv.index:
                continue
            cur = dv.loc[date]
            avg = dv_avg.loc[date]
            if not np.isnan(cur) and not np.isnan(avg) and avg > 0:
                ratio = cur / avg
                if ratio > 2.0:
                    surges.append((tk, ratio))

        if surges:
            surges.sort(key=lambda x: -x[1])
            top = surges[:MAX_POSITIONS]
            total_ret = 0.0
            n = len(top)
            for tk, _ in top:
                p = price[tk]
                entry_price = p.iloc[loc_date]
                exit_loc = min(loc_date + hold_days, len(p) - 1)
                exit_price = p.iloc[exit_loc]
                ret = (exit_price / entry_price) - 1
                ret = apply_slippage(ret)
                total_ret += ret / n

            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += hold_days
        else:
            i += 1

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    print(f"    Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}, "
          f"Final: ${metrics['final_equity']}")
    return metrics


# ── VARIANT D: VOLUME + MOMENTUM CONFLUENCE ───────────────────────────────
def run_variant_d():
    """
    Volume > 1.5x avg AND price > 20-SMA AND RSI > 50. Buy, hold 10d.
    """
    print("\n[D] Volume + Momentum Confluence...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        loc_date = price.index.get_loc(date)
        hold_days = 10

        candidates = []
        for tk in SECTOR_ETFS:
            if tk not in vol_ratio_20 or tk not in rsi_14 or tk not in sma_20:
                continue
            if date not in vol_ratio_20[tk].index:
                continue

            vr = vol_ratio_20[tk].loc[date]
            rsi = rsi_14[tk].loc[date] if date in rsi_14[tk].index else np.nan
            sma = sma_20[tk].loc[date] if date in sma_20[tk].index else np.nan
            cur_price = price[tk].loc[date] if date in price[tk].index else np.nan

            if np.isnan(vr) or np.isnan(rsi) or np.isnan(sma) or np.isnan(cur_price):
                continue

            if vr > 1.5 and cur_price > sma and rsi > 50:
                candidates.append((tk, vr * (rsi / 50)))  # composite score

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            top = candidates[:MAX_POSITIONS]
            total_ret = 0.0
            n = len(top)
            for tk, _ in top:
                p = price[tk]
                entry_price = p.iloc[loc_date]
                exit_loc = min(loc_date + hold_days, len(p) - 1)
                exit_price = p.iloc[exit_loc]
                ret = (exit_price / entry_price) - 1
                ret = apply_slippage(ret)
                total_ret += ret / n

            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += hold_days
        else:
            i += 1

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    print(f"    Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}, "
          f"Final: ${metrics['final_equity']}")
    return metrics


# ── VARIANT E: ACCUMULATION/DISTRIBUTION ──────────────────────────────────
def run_variant_e():
    """
    Track 5-day cumulative volume on up-days vs down-days (A/D proxy).
    Buy when A/D ratio > 2. Hold 10d.
    """
    print("\n[E] Accumulation/Distribution Proxy...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    # Pre-compute daily returns and up/down classification
    daily_ret = price.pct_change()

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        loc_date = price.index.get_loc(date)
        hold_days = 10

        if loc_date < 5:
            equity_curve.loc[date] = equity
            i += 1
            continue

        candidates = []
        for tk in SECTOR_ETFS:
            if tk not in volume.columns or tk not in daily_ret.columns:
                continue
            v = volume[tk]
            dr = daily_ret[tk]

            # 5-day window
            window_v = v.iloc[loc_date - 5:loc_date]
            window_r = dr.iloc[loc_date - 5:loc_date]

            up_vol = window_v[window_r > 0].sum()
            down_vol = window_v[window_r < 0].sum()

            if down_vol > 0:
                ad_ratio = up_vol / down_vol
            else:
                ad_ratio = 99.0 if up_vol > 0 else 1.0

            if ad_ratio > 2.0:
                candidates.append((tk, ad_ratio))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            top = candidates[:MAX_POSITIONS]
            total_ret = 0.0
            n = len(top)
            for tk, _ in top:
                p = price[tk]
                entry_price = p.iloc[loc_date]
                exit_loc = min(loc_date + hold_days, len(p) - 1)
                exit_price = p.iloc[exit_loc]
                ret = (exit_price / entry_price) - 1
                ret = apply_slippage(ret)
                total_ret += ret / n

            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += hold_days
        else:
            i += 1

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    print(f"    Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}, "
          f"Final: ${metrics['final_equity']}")
    return metrics


# ── VARIANT F: ADVERSARIAL RANDOM ─────────────────────────────────────────
def run_variant_f():
    """
    Random sector ETF selections, same sizing and hold periods as best variant.
    Uses 5-day hold, max 3 positions (matching variant A/C).
    """
    print("\n[F] Adversarial Random...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []
    rng = np.random.RandomState(RANDOM_SEED + 99)

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        loc_date = price.index.get_loc(date)
        hold_days = 5

        # Random chance of entering (roughly match trade frequency of real variants)
        if rng.random() < 0.15:  # ~15% chance per day
            n_picks = rng.randint(1, MAX_POSITIONS + 1)
            picks = rng.choice(SECTOR_ETFS, size=min(n_picks, len(SECTOR_ETFS)), replace=False)
            total_ret = 0.0
            n = len(picks)
            for tk in picks:
                p = price[tk]
                entry_price = p.iloc[loc_date]
                exit_loc = min(loc_date + hold_days, len(p) - 1)
                exit_price = p.iloc[exit_loc]
                ret = (exit_price / entry_price) - 1
                ret = apply_slippage(ret)
                total_ret += ret / n

            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += hold_days
        else:
            i += 1

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    print(f"    Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}, "
          f"Final: ${metrics['final_equity']}")
    return metrics


# ── RUN ALL ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INSTITUTIONAL FLOW PROXY BACKTEST")
print("=" * 70)

results = {}
variant_funcs = {
    "A_volume_breadth": run_variant_a,
    "B_rotation_detection": run_variant_b,
    "C_dollar_volume_surge": run_variant_c,
    "D_volume_momentum_confluence": run_variant_d,
    "E_accumulation_distribution": run_variant_e,
    "F_adversarial_random": run_variant_f,
}

for name, func in variant_funcs.items():
    results[name] = func()

# ── SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Variant':<32} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MDD':>8} {'Trades':>7} {'S_Bull':>7} {'S_Bear':>7} {'RGap':>6} {'Perm_p':>7} {'Gates':>6}")
print("-" * 120)

for name, m in results.items():
    gates_str = f"{len(m['gates_passed'])}/5"
    print(f"{name:<32} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} "
          f"{m['win_rate']:>5.1%} {m['max_dd']:>8.4f} {m['n_trades']:>7d} "
          f"{m['sharpe_bull']:>7.3f} {m['sharpe_bear']:>7.3f} {m['regime_gap']:>6.3f} "
          f"{m['perm_p']:>7.4f} {gates_str:>6}")

# Check which passed all 5 gates
print("\n" + "=" * 70)
print("GATE ANALYSIS")
print("=" * 70)
for name, m in results.items():
    passed = m["gates_passed"]
    status = "PASS" if len(passed) == 5 else "FAIL"
    failed = [g for g in ["sharpe", "perm_p", "regime_gap", "max_dd", "min_trades"] if g not in passed]
    failed_str = f" (failed: {', '.join(failed)})" if failed else ""
    print(f"  {name:<32} {status} {len(passed)}/5{failed_str}")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/institutional_flow_proxy_results.json")
output = {
    "backtest": "institutional_flow_proxy",
    "run_date": dt.datetime.now().isoformat(),
    "config": {
        "oot_start": OOT_START,
        "end_date": END_DATE,
        "initial_capital": INITIAL_CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "max_positions": MAX_POSITIONS,
        "n_permutations": N_PERMUTATIONS,
        "universe": ETF_UNIVERSE,
    },
    "gates": GATES,
    "variants": results,
}
output_path.write_text(json.dumps(output, indent=2, default=str))
print(f"\nResults saved to {output_path}")
print("Done.")
