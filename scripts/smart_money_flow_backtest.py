#!/usr/bin/env python3
"""
Smart Money Flow Detection Backtest
6 variants testing institutional/behavioral patterns in equities.
Walk-forward OOT: Jan 2022 to present. Initial capital: $645. Commission: $0 (Robinhood).
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
START_DATE = "2021-06-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
END_DATE = dt.date.today().isoformat()
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "SHOP", "SQ", "COIN", "SNOW", "DDOG", "NET",
    "RBLX", "PLTR", "UBER", "LYFT",
]

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLC", "XLI", "XLB", "XLRE", "XLU", "XLP"]

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
all_tickers = list(set(STOCK_UNIVERSE + SECTOR_ETFS + ["SPY", "QQQ"]))
raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Handle multi-level columns from yf.download
price = raw["Close"].copy()
volume = raw["Volume"].copy()

# Forward-fill then drop remaining NaN rows
price = price.ffill()
volume = volume.ffill().fillna(0)

spy = price["SPY"]
spy_sma200 = spy.rolling(200).mean()
regime = (spy > spy_sma200).astype(int)  # 1=bull, 0=bear

qqq = price["QQQ"]

print(f"Data loaded: {price.index[0].date()} to {price.index[-1].date()}, {len(price)} days")


# ── HELPERS ─────────────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trade_returns, regime_series, trade_dates):
    """Compute Sharpe, permutation p-value, regime gap, MDD, trade count."""
    daily_returns = equity_curve.pct_change().dropna()
    n_trades = len(trade_returns)

    # Sharpe (annualised)
    if daily_returns.std() == 0 or len(daily_returns) < 10:
        sharpe = 0.0
    else:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    # Max drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_dd = float(dd.min())

    # Regime gap
    if len(trade_returns) > 0 and len(trade_dates) == len(trade_returns):
        bull_mask = np.array([regime.reindex(trade_dates).iloc[i] == 1 if i < len(trade_dates) else True
                              for i in range(len(trade_dates))])
        bull_rets = trade_returns[bull_mask] if bull_mask.sum() > 0 else np.array([0.0])
        bear_rets = trade_returns[~bull_mask] if (~bull_mask).sum() > 0 else np.array([0.0])

        bull_sharpe = float(bull_rets.mean() / bull_rets.std() * np.sqrt(252)) if bull_rets.std() > 0 and len(bull_rets) > 1 else 0.0
        bear_sharpe = float(bear_rets.mean() / bear_rets.std() * np.sqrt(252)) if bear_rets.std() > 0 and len(bear_rets) > 1 else 0.0
        denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)
        regime_gap = abs(bull_sharpe - bear_sharpe) / denom
    else:
        regime_gap = 1.0

    # Permutation test
    if n_trades >= 5:
        observed_mean = float(trade_returns.mean())
        rng = np.random.RandomState(RANDOM_SEED)
        count_ge = 0
        for _ in range(N_PERMUTATIONS):
            perm = trade_returns.copy()
            rng.shuffle(perm)
            # Shuffle entry dates - this randomises the timing
            rand_idx = rng.choice(len(trade_returns), size=len(trade_returns), replace=True)
            perm_mean = float(perm.mean())  # mean is invariant to shuffle, so use sign flip
            flips = rng.choice([-1, 1], size=len(trade_returns))
            perm_mean = float((trade_returns * flips).mean())
            if perm_mean >= observed_mean:
                count_ge += 1
        perm_p = count_ge / N_PERMUTATIONS
    else:
        perm_p = 1.0

    return {
        "sharpe": round(sharpe, 3),
        "perm_p": round(perm_p, 4),
        "regime_gap": round(regime_gap, 3),
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


# ── VARIANT A: INSTITUTIONAL ACCUMULATION ───────────────────────────────────
def run_variant_a():
    print("\n[A] Institutional Accumulation Pattern...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        scores = []
        for tk in STOCK_UNIVERSE:
            if tk not in price.columns:
                continue
            p = price[tk]
            v = volume[tk]
            loc = p.index.get_loc(date)
            if loc < 20:
                continue
            ret_20d = (p.iloc[loc] / p.iloc[loc - 20]) - 1
            ret_5d = (p.iloc[loc] / p.iloc[loc - 5]) - 1
            avg_vol_20 = v.iloc[loc - 20:loc].mean()
            avg_vol_5 = v.iloc[loc - 5:loc].mean()

            # Positive 20d return, negative 5d return, low volume pullback
            if ret_20d > 0 and ret_5d < 0 and avg_vol_5 < avg_vol_20:
                score = ret_20d * (1 - avg_vol_5 / max(avg_vol_20, 1))
                scores.append((tk, score, loc))

        scores.sort(key=lambda x: -x[1])
        top3 = scores[:3]

        if top3:
            hold_days = 10
            per_stock = equity / len(top3)
            total_ret = 0
            for tk, _, loc in top3:
                p = price[tk]
                exit_loc = min(loc + hold_days, len(p) - 1)
                ret = (p.iloc[exit_loc] / p.iloc[loc]) - 1
                total_ret += ret * (per_stock / equity)
            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += hold_days  # skip hold period
        else:
            i += 1

        equity_curve.loc[date] = equity

    # Fill forward equity curve for all OOT dates
    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns)
    td = pd.DatetimeIndex(trade_dates_list)
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    return metrics


# ── VARIANT B: EXHAUSTION GAP REVERSAL ──────────────────────────────────────
def run_variant_b():
    print("\n[B] Exhaustion Gap Reversal...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    # Pre-compute open prices
    opens = raw["Open"].ffill()

    for i, date in enumerate(oot_dates):
        if i < 1:
            equity_curve.loc[date] = equity
            continue

        prev_date = oot_dates[i - 1]
        signals = []
        for tk in STOCK_UNIVERSE:
            if tk not in price.columns or tk not in opens.columns:
                continue
            p = price[tk]
            o = opens[tk]
            v = volume[tk]
            loc = p.index.get_loc(date)
            if loc < 20:
                continue

            prev_close = p.iloc[loc - 1]
            today_open = o.iloc[loc]
            today_close = p.iloc[loc]
            avg_vol = v.iloc[loc - 20:loc].mean()
            today_vol = v.iloc[loc]

            gap_pct = (today_open / prev_close) - 1
            # Gap down >3%, high volume, closes above open
            if gap_pct < -0.03 and today_vol > 2 * avg_vol and today_close > today_open:
                signals.append((tk, loc))

        for tk, loc in signals:
            p = price[tk]
            # Buy next day, hold 5 days
            entry_loc = min(loc + 1, len(p) - 1)
            exit_loc = min(entry_loc + 5, len(p) - 1)
            ret = (p.iloc[exit_loc] / p.iloc[entry_loc]) - 1
            equity *= (1 + ret)
            trade_returns.append(ret)
            trade_dates_list.append(date)

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    return metrics


# ── VARIANT C: SMART MONEY DIVERGENCE ───────────────────────────────────────
def run_variant_c():
    print("\n[C] Smart Money Divergence...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    # Collect all signals first, then process date-by-date
    all_signals = []  # (entry_date, ticker, entry_loc, exit_loc)

    for tk in STOCK_UNIVERSE:
        if tk not in price.columns:
            continue
        p = price[tk]
        v = volume[tk]
        sma5 = p.rolling(5).mean()
        cooldown_until = -1  # per-ticker cooldown

        for i, date in enumerate(oot_dates):
            loc = p.index.get_loc(date)
            if loc < 25 or loc <= cooldown_until:
                continue

            # New 20d low
            low_20d = p.iloc[loc - 20:loc].min()
            if p.iloc[loc] > low_20d:
                continue

            # Volume decreasing (current 5d avg < prior 5d avg)
            vol_recent = v.iloc[loc - 5:loc].mean()
            vol_prior = v.iloc[loc - 10:loc - 5].mean()
            if vol_recent >= vol_prior:
                continue

            # Wait for close above 5-SMA
            for j in range(1, 6):
                check_loc = loc + j
                if check_loc >= len(p):
                    break
                if p.iloc[check_loc] > sma5.iloc[check_loc]:
                    exit_loc = min(check_loc + 10, len(p) - 1)
                    all_signals.append((p.index[check_loc], tk, check_loc, exit_loc))
                    cooldown_until = exit_loc  # don't re-enter same stock during hold
                    break

    # Group signals by entry date, allocate capital equally across concurrent trades
    from collections import defaultdict
    by_date = defaultdict(list)
    for entry_date, tk, entry_loc, exit_loc in all_signals:
        by_date[entry_date].append((tk, entry_loc, exit_loc))

    sorted_dates = sorted(by_date.keys())
    for entry_date in sorted_dates:
        trades = by_date[entry_date]
        n = len(trades)
        per_stock = equity / max(n, 1)
        total_ret = 0
        for tk, entry_loc, exit_loc in trades:
            p = price[tk]
            ret = (p.iloc[exit_loc] / p.iloc[entry_loc]) - 1
            total_ret += ret / n  # equal-weight
        equity *= (1 + total_ret)
        trade_returns.append(total_ret)
        trade_dates_list.append(entry_date)

    # Build equity curve
    if trade_dates_list:
        equity_curve = pd.Series(INITIAL_CAPITAL, index=oot_dates)
        sorted_trades = sorted(zip(trade_dates_list, trade_returns), key=lambda x: x[0])
        cum_equity = INITIAL_CAPITAL
        trade_idx = 0
        for date in oot_dates:
            while trade_idx < len(sorted_trades) and sorted_trades[trade_idx][0] <= date:
                cum_equity *= (1 + sorted_trades[trade_idx][1])
                trade_idx += 1
            equity_curve.loc[date] = cum_equity
    else:
        equity_curve = pd.Series(INITIAL_CAPITAL, index=oot_dates)

    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    return metrics


# ── VARIANT D: FOLLOW THE FLOW ──────────────────────────────────────────────
def run_variant_d():
    print("\n[D] Follow the Flow (High-Conviction Entries)...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        signals = []
        for tk in STOCK_UNIVERSE:
            if tk not in price.columns:
                continue
            p = price[tk]
            v = volume[tk]
            loc = p.index.get_loc(date)
            if loc < 252:
                continue

            daily_ret = (p.iloc[loc] / p.iloc[loc - 1]) - 1
            avg_vol = v.iloc[loc - 20:loc].mean()
            today_vol = v.iloc[loc]
            high_52w = p.iloc[loc - 252:loc].max()
            dist_from_high = p.iloc[loc] / high_52w

            # Up >2%, >2x volume, within 5% of 52w high
            if daily_ret > 0.02 and today_vol > 2 * avg_vol and dist_from_high > 0.95:
                signals.append((tk, loc, daily_ret))

        if signals:
            per_stock = equity / len(signals)
            total_ret = 0
            for tk, loc, _ in signals:
                p = price[tk]
                exit_loc = min(loc + 15, len(p) - 1)
                ret = (p.iloc[exit_loc] / p.iloc[loc]) - 1
                total_ret += ret * (per_stock / equity)
            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)
            i += 15
        else:
            i += 1

        equity_curve.loc[date] = equity

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    return metrics


# ── VARIANT E: SECTOR ROTATION VIA FLOW ─────────────────────────────────────
def run_variant_e():
    print("\n[E] Sector Rotation via Flow...")
    oot_dates = price.index[price.index >= OOT_START]
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    # Monthly rebalance dates
    rebal_dates = []
    current_month = None
    for d in oot_dates:
        ym = (d.year, d.month)
        if ym != current_month:
            rebal_dates.append(d)
            current_month = ym

    for ri, rdate in enumerate(rebal_dates):
        loc_r = price.index.get_loc(rdate)
        if loc_r < 20:
            equity_curve.loc[rdate] = equity
            continue

        scores = []
        for etf in SECTOR_ETFS:
            if etf not in price.columns:
                continue
            p = price[etf]
            v = volume[etf]
            ret_20d = (p.iloc[loc_r] / p.iloc[loc_r - 20]) - 1
            avg_vol = v.iloc[loc_r - 20:loc_r].mean()
            cur_vol = v.iloc[loc_r]
            vol_ratio = cur_vol / max(avg_vol, 1)
            pv_score = ret_20d * vol_ratio
            scores.append((etf, pv_score))

        scores.sort(key=lambda x: -x[1])
        top2 = scores[:2]

        # Hold ~20 trading days (until next rebalance)
        if ri + 1 < len(rebal_dates):
            exit_date = rebal_dates[ri + 1]
        else:
            exit_date = oot_dates[-1]

        if top2:
            per_etf = equity / len(top2)
            total_ret = 0
            for etf, _ in top2:
                p = price[etf]
                entry_p = p.loc[rdate]
                exit_p = p.loc[exit_date] if exit_date in p.index else p.iloc[p.index.get_loc(exit_date, method='ffill')]
                ret = (exit_p / entry_p) - 1
                total_ret += ret * (per_etf / equity)
            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(rdate)

        equity_curve.loc[rdate] = equity

    # Fill curve
    full_curve = pd.Series(dtype=float)
    for d in oot_dates:
        if d in equity_curve.index:
            full_curve.loc[d] = equity_curve.loc[d]
    full_curve = full_curve.reindex(oot_dates).ffill().bfill()

    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(full_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    return metrics


# ── VARIANT F: ADVERSARIAL BASELINE (RANDOM) ───────────────────────────────
def run_variant_f():
    print("\n[F] Adversarial Baseline (Random)...")
    oot_dates = price.index[price.index >= OOT_START]
    rng = np.random.RandomState(RANDOM_SEED)
    equity = INITIAL_CAPITAL
    equity_curve = pd.Series(dtype=float)
    trade_returns = []
    trade_dates_list = []

    # ~same frequency as variant A: trade every ~10 days
    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]
        # Random number of stocks (1-3)
        n_stocks = rng.randint(1, 4)
        picks = rng.choice(STOCK_UNIVERSE, size=min(n_stocks, len(STOCK_UNIVERSE)), replace=False)
        hold = rng.randint(5, 16)  # random hold 5-15 days

        per_stock = equity / len(picks)
        total_ret = 0
        valid = 0
        for tk in picks:
            if tk not in price.columns:
                continue
            p = price[tk]
            loc = p.index.get_loc(date)
            exit_loc = min(loc + hold, len(p) - 1)
            ret = (p.iloc[exit_loc] / p.iloc[loc]) - 1
            total_ret += ret / len(picks)
            valid += 1

        if valid > 0:
            equity *= (1 + total_ret)
            trade_returns.append(total_ret)
            trade_dates_list.append(date)

        equity_curve.loc[date] = equity
        i += hold

    equity_curve = equity_curve.reindex(oot_dates).ffill().bfill()
    tr = np.array(trade_returns) if trade_returns else np.array([0.0])
    td = pd.DatetimeIndex(trade_dates_list) if trade_dates_list else pd.DatetimeIndex([oot_dates[0]])
    metrics = compute_metrics(equity_curve, tr, regime, td)
    metrics["gates_passed"] = check_gates(metrics)
    return metrics


# ── BENCHMARK: QQQ BUY-AND-HOLD ────────────────────────────────────────────
def run_benchmark():
    print("\n[Benchmark] QQQ Buy-and-Hold...")
    oot_dates = price.index[price.index >= OOT_START]
    qqq_oot = qqq.reindex(oot_dates)
    equity_curve = INITIAL_CAPITAL * (qqq_oot / qqq_oot.iloc[0])
    daily_rets = equity_curve.pct_change().dropna()
    sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_dd = float(dd.min())
    return {
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "final_equity": round(float(equity_curve.iloc[-1]), 2),
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    results["A_institutional_accumulation"] = run_variant_a()
    results["B_exhaustion_gap_reversal"] = run_variant_b()
    results["C_smart_money_divergence"] = run_variant_c()
    results["D_follow_the_flow"] = run_variant_d()
    results["E_sector_rotation_flow"] = run_variant_e()
    results["F_adversarial_random"] = run_variant_f()
    results["benchmark_qqq"] = run_benchmark()

    # Summary
    print("\n" + "=" * 90)
    print(f"{'SMART MONEY FLOW BACKTEST RESULTS':^90}")
    print(f"{'OOT: ' + OOT_START + ' to ' + END_DATE + ' | Initial: $' + str(INITIAL_CAPITAL):^90}")
    print("=" * 90)
    print(f"{'Variant':<30} {'Sharpe':>7} {'Perm-p':>8} {'RGap':>7} {'MDD':>8} {'Trades':>7} {'Final$':>9} {'Gates':>8}")
    print("-" * 90)

    for name, m in results.items():
        if name == "benchmark_qqq":
            print(f"{'QQQ Buy-Hold (benchmark)':<30} {m['sharpe']:>7.3f} {'n/a':>8} {'n/a':>7} {m['max_dd']:>8.4f} {'n/a':>7} {m['final_equity']:>9.2f} {'n/a':>8}")
        else:
            gp = m.get("gates_passed", [])
            print(f"{name:<30} {m['sharpe']:>7.3f} {m['perm_p']:>8.4f} {m['regime_gap']:>7.3f} {m['max_dd']:>8.4f} {m['n_trades']:>7d} {m['final_equity']:>9.2f} {len(gp):<2}/5")

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/smart_money_flow_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
