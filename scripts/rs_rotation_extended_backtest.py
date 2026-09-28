#!/usr/bin/env python3
"""
Relative Strength Rotation with Mean Reversion Twist — Extended Backtest
========================================================================
6 variants across defensive, equity, and mixed ETF pools.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2021-01-01"  # extra lookback for indicators
N_PERM = 1000
SEED = 42

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/rs_rotation_extended_results.json")

# All tickers we need
ALL_TICKERS = ["GLD", "TLT", "UUP", "SHY", "QQQ", "SPY", "IWM", "XLK"]


# ─── Data Download ───────────────────────────────────────────────────────────
def download_data():
    print("Downloading price data...")
    data = yf.download(ALL_TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns: (Price, Ticker)
    close = data["Close"]
    # Forward fill then back fill small gaps
    close = close.ffill().bfill()
    print(f"  Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ─── Indicator Calculations ──────────────────────────────────────────────────
def calc_returns(close, window=20):
    return close.pct_change(window)

def calc_volatility(close, window=20):
    daily_ret = close.pct_change()
    return daily_ret.rolling(window).std() * np.sqrt(252)

def calc_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_vol_adj_rs(close, window=20):
    """Vol-adjusted relative strength: return / volatility."""
    ret = calc_returns(close, window)
    vol = calc_volatility(close, window)
    vol = vol.replace(0, np.nan)
    return ret / vol

def calc_sma(series, window=200):
    return series.rolling(window).mean()


# ─── Backtesting Engine ─────────────────────────────────────────────────────
def run_rotation_backtest(close, pool, signal_func, rebal_freq="W", name="Strategy"):
    """
    Generic rotation backtest.
    signal_func(close, pool, date, current_holding, spy_close) -> ticker to hold
    Returns DataFrame of daily returns.
    """
    oot_mask = close.index >= OOT_START
    dates = close.index[oot_mask]

    holdings = []
    current_holding = None
    rebal_dates = set()

    # Determine weekly rebalance dates (every Friday or last trading day of week)
    weekly = close.loc[oot_mask].resample("W-FRI").last().index
    for wd in weekly:
        # Find the closest actual trading day
        mask = close.index <= wd
        if mask.any():
            rebal_dates.add(close.index[mask][-1])

    daily_returns = []
    trade_count = 0
    trade_log = []

    for i, date in enumerate(dates):
        if date in rebal_dates or current_holding is None:
            new_holding = signal_func(close, pool, date, current_holding)
            if new_holding != current_holding:
                trade_count += 1
                trade_log.append({"date": str(date.date()), "from": current_holding, "to": new_holding})
                current_holding = new_holding

        # Daily return of current holding
        if current_holding and i > 0:
            prev_date = dates[i - 1]
            if current_holding in close.columns:
                p0 = close.loc[prev_date, current_holding]
                p1 = close.loc[date, current_holding]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret = (p1 / p0) - 1
                    # Apply slippage on trade days
                    if date in rebal_dates and trade_log and trade_log[-1]["date"] == str(date.date()):
                        ret -= SLIPPAGE_PCT
                    daily_returns.append({"date": date, "return": ret, "holding": current_holding})
                else:
                    daily_returns.append({"date": date, "return": 0.0, "holding": current_holding})
            else:
                daily_returns.append({"date": date, "return": 0.0, "holding": current_holding})
        else:
            daily_returns.append({"date": date, "return": 0.0, "holding": current_holding})

    df = pd.DataFrame(daily_returns)
    if len(df) == 0:
        return df, trade_count, trade_log
    df.set_index("date", inplace=True)
    return df, trade_count, trade_log


# ─── Signal Functions for Each Variant ───────────────────────────────────────
def make_vol_adj_rs_signal(pool):
    """Variant A/C: Pure vol-adjusted RS rotation. Hold top 1."""
    def signal(close, pool_tickers, date, current_holding):
        scores = {}
        for t in pool_tickers:
            if t not in close.columns:
                continue
            hist = close.loc[:date, t]
            if len(hist) < 25:
                continue
            ret20 = (hist.iloc[-1] / hist.iloc[-20]) - 1 if len(hist) >= 20 else 0
            vol20 = hist.pct_change().iloc[-20:].std() * np.sqrt(252) if len(hist) >= 20 else 1
            if vol20 > 0:
                scores[t] = ret20 / vol20
            else:
                scores[t] = 0
        if not scores:
            return current_holding or pool_tickers[0]
        return max(scores, key=scores.get)
    return signal


def make_rs_rsi_exit_signal(pool):
    """Variant B: Vol-adj RS but exit if RSI(14) > 70 (overbought), switch to #2."""
    def signal(close, pool_tickers, date, current_holding):
        scores = {}
        rsi_vals = {}
        for t in pool_tickers:
            if t not in close.columns:
                continue
            hist = close.loc[:date, t]
            if len(hist) < 25:
                continue
            ret20 = (hist.iloc[-1] / hist.iloc[-20]) - 1 if len(hist) >= 20 else 0
            vol20 = hist.pct_change().iloc[-20:].std() * np.sqrt(252) if len(hist) >= 20 else 1
            if vol20 > 0:
                scores[t] = ret20 / vol20
            else:
                scores[t] = 0
            # Calculate RSI
            delta = hist.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            if len(gain) > 0 and pd.notna(gain.iloc[-1]) and pd.notna(loss.iloc[-1]):
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
                rsi_vals[t] = 100 - (100 / (1 + rs))
            else:
                rsi_vals[t] = 50

        if not scores:
            return current_holding or pool_tickers[0]

        ranked = sorted(scores, key=scores.get, reverse=True)
        top = ranked[0]

        # If current holding is overbought and not top-ranked, or if top is overbought
        if current_holding and current_holding in rsi_vals:
            if rsi_vals[current_holding] > 70 and len(ranked) > 1:
                # Switch to next best that isn't overbought
                for candidate in ranked:
                    if candidate != current_holding and rsi_vals.get(candidate, 50) <= 70:
                        return candidate
                # All overbought, stick with top
                return top

        return top
    return signal


def make_regime_overlay_signal(pool):
    """Variant D: Mixed pool but exclude QQQ in bear regime (SPY < 200-SMA)."""
    def signal(close, pool_tickers, date, current_holding):
        # Check regime
        spy_hist = close.loc[:date, "SPY"]
        if len(spy_hist) >= 200:
            sma200 = spy_hist.iloc[-200:].mean()
            is_bear = spy_hist.iloc[-1] < sma200
        else:
            is_bear = False

        active_pool = [t for t in pool_tickers if not (is_bear and t == "QQQ")]
        if not active_pool:
            active_pool = pool_tickers

        scores = {}
        for t in active_pool:
            if t not in close.columns:
                continue
            hist = close.loc[:date, t]
            if len(hist) < 25:
                continue
            ret20 = (hist.iloc[-1] / hist.iloc[-20]) - 1 if len(hist) >= 20 else 0
            vol20 = hist.pct_change().iloc[-20:].std() * np.sqrt(252) if len(hist) >= 20 else 1
            if vol20 > 0:
                scores[t] = ret20 / vol20
            else:
                scores[t] = 0
        if not scores:
            return current_holding or active_pool[0]
        return max(scores, key=scores.get)
    return signal


def make_dual_momentum_signal(pool):
    """Variant E: Hold top RS asset only if absolute return > 0, else SHY."""
    def signal(close, pool_tickers, date, current_holding):
        scores = {}
        abs_returns = {}
        for t in pool_tickers:
            if t not in close.columns:
                continue
            hist = close.loc[:date, t]
            if len(hist) < 25:
                continue
            ret20 = (hist.iloc[-1] / hist.iloc[-20]) - 1 if len(hist) >= 20 else 0
            vol20 = hist.pct_change().iloc[-20:].std() * np.sqrt(252) if len(hist) >= 20 else 1
            if vol20 > 0:
                scores[t] = ret20 / vol20
            else:
                scores[t] = 0
            abs_returns[t] = ret20

        if not scores:
            return "SHY"

        top = max(scores, key=scores.get)
        # Only hold if absolute return is positive
        if abs_returns.get(top, 0) > 0:
            return top
        else:
            return "SHY"
    return signal


def make_composite_score_signal(pool):
    """Variant F: 40% vol-adj RS + 30% 5d momentum + 30% inverse vol. Rank, hold top 1."""
    def signal(close, pool_tickers, date, current_holding):
        metrics = {}
        for t in pool_tickers:
            if t not in close.columns:
                continue
            hist = close.loc[:date, t]
            if len(hist) < 25:
                continue
            ret20 = (hist.iloc[-1] / hist.iloc[-20]) - 1 if len(hist) >= 20 else 0
            vol20 = hist.pct_change().iloc[-20:].std() * np.sqrt(252) if len(hist) >= 20 else 1
            ret5 = (hist.iloc[-1] / hist.iloc[-5]) - 1 if len(hist) >= 5 else 0
            va_rs = ret20 / vol20 if vol20 > 0 else 0
            inv_vol = 1.0 / vol20 if vol20 > 0 else 0
            metrics[t] = {"va_rs": va_rs, "mom5": ret5, "inv_vol": inv_vol}

        if not metrics:
            return current_holding or pool_tickers[0]

        # Rank each metric (higher = better rank = higher number)
        tickers = list(metrics.keys())
        n = len(tickers)

        def rank_metric(key):
            vals = [(t, metrics[t][key]) for t in tickers]
            vals.sort(key=lambda x: x[1])
            ranks = {}
            for i, (t, _) in enumerate(vals):
                ranks[t] = i / max(n - 1, 1)  # normalize to [0,1]
            return ranks

        rs_ranks = rank_metric("va_rs")
        mom_ranks = rank_metric("mom5")
        vol_ranks = rank_metric("inv_vol")

        composite = {}
        for t in tickers:
            composite[t] = 0.4 * rs_ranks[t] + 0.3 * mom_ranks[t] + 0.3 * vol_ranks[t]

        return max(composite, key=composite.get)
    return signal


# ─── Performance Metrics ─────────────────────────────────────────────────────
def calc_metrics(df, trade_count, name=""):
    if df.empty or "return" not in df.columns or len(df) < 10:
        return {
            "name": name, "sharpe": 0, "sortino": 0, "total_return": 0,
            "ann_return": 0, "max_dd": -1, "profit_factor": 0,
            "win_rate": 0, "n_trades": trade_count, "calmar": 0,
        }

    rets = df["return"].values
    n_days = len(rets)
    n_years = n_days / 252

    total_ret = np.prod(1 + rets) - 1
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    # Sharpe
    mu = np.mean(rets) * 252
    sigma = np.std(rets, ddof=1) * np.sqrt(252)
    sharpe = mu / sigma if sigma > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    down_std = np.std(downside, ddof=1) * np.sqrt(252) if len(downside) > 1 else sigma
    sortino = mu / down_std if down_std > 0 else 0

    # Max drawdown
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Profit factor (sum of positive returns / |sum of negative returns|)
    pos = rets[rets > 0].sum()
    neg = abs(rets[rets < 0].sum())
    pf = pos / neg if neg > 0 else (999 if pos > 0 else 0)

    # Win rate (% of positive return days)
    wr = (rets > 0).sum() / len(rets)

    # Calmar
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    # Final equity
    final_equity = ACCOUNT_SIZE * (1 + total_ret)

    return {
        "name": name,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return": round(total_ret * 100, 2),
        "ann_return": round(ann_ret * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr * 100, 1),
        "n_trades": trade_count,
        "calmar": round(calmar, 3),
        "final_equity": round(final_equity, 2),
    }


# ─── Regime Analysis ─────────────────────────────────────────────────────────
def regime_analysis(df, close):
    """Split returns by bull/bear regime. Bull = SPY > 200-SMA."""
    if df.empty:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 1.0}

    spy = close["SPY"]
    sma200 = spy.rolling(200).mean()

    bull_rets = []
    bear_rets = []
    for date, row in df.iterrows():
        if date in sma200.index and pd.notna(sma200.loc[date]):
            if spy.loc[date] > sma200.loc[date]:
                bull_rets.append(row["return"])
            else:
                bear_rets.append(row["return"])

    def sharpe_from_list(r):
        if len(r) < 10:
            return 0
        arr = np.array(r)
        mu = np.mean(arr) * 252
        sig = np.std(arr, ddof=1) * np.sqrt(252)
        return mu / sig if sig > 0 else 0

    bull_sharpe = sharpe_from_list(bull_rets)
    bear_sharpe = sharpe_from_list(bear_rets)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_days": len(bull_rets),
        "bear_days": len(bear_rets),
    }


# ─── Permutation Test ────────────────────────────────────────────────────────
def permutation_test(close, pool, signal_func, actual_sharpe, n_perm=N_PERM):
    """
    Shuffle which asset gets selected each rebalance week.
    Count how often random selection beats actual Sharpe.
    """
    rng = np.random.RandomState(SEED)
    oot_mask = close.index >= OOT_START
    dates = close.index[oot_mask]

    # Get rebalance dates
    weekly = close.loc[oot_mask].resample("W-FRI").last().index
    rebal_dates = set()
    for wd in weekly:
        mask = close.index <= wd
        if mask.any():
            rebal_dates.add(close.index[mask][-1])

    rebal_list = sorted(rebal_dates)

    count_better = 0
    perm_sharpes = []

    for _ in range(n_perm):
        # Random assignment at each rebalance
        assignments = {d: rng.choice(pool) for d in rebal_list}

        daily_rets = []
        current = None
        for i, date in enumerate(dates):
            if date in rebal_dates:
                current = assignments[date]
            if current and i > 0:
                prev = dates[i - 1]
                p0 = close.loc[prev, current] if current in close.columns else np.nan
                p1 = close.loc[date, current] if current in close.columns else np.nan
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    daily_rets.append((p1 / p0) - 1)
                else:
                    daily_rets.append(0)
            else:
                daily_rets.append(0)

        arr = np.array(daily_rets)
        if len(arr) > 10:
            mu = np.mean(arr) * 252
            sig = np.std(arr, ddof=1) * np.sqrt(252)
            perm_s = mu / sig if sig > 0 else 0
        else:
            perm_s = 0
        perm_sharpes.append(perm_s)
        if perm_s >= actual_sharpe:
            count_better += 1

    p_value = (count_better + 1) / (n_perm + 1)  # +1 for continuity correction
    return round(p_value, 4), perm_sharpes


# ─── 5-Gate Validation ───────────────────────────────────────────────────────
def validate_5gate(metrics, regime, perm_p):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "maxdd_gt_neg50": metrics["max_dd"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    close = download_data()

    # Define variants
    variants = [
        {
            "name": "A) Defensive RS (GLD/TLT/UUP/SHY)",
            "pool": ["GLD", "TLT", "UUP", "SHY"],
            "signal_func": make_vol_adj_rs_signal(["GLD", "TLT", "UUP", "SHY"]),
        },
        {
            "name": "B) Defensive RS + RSI Exit",
            "pool": ["GLD", "TLT", "UUP", "SHY"],
            "signal_func": make_rs_rsi_exit_signal(["GLD", "TLT", "UUP", "SHY"]),
        },
        {
            "name": "C) Mixed Pool Binary (QQQ/GLD/TLT/UUP)",
            "pool": ["QQQ", "GLD", "TLT", "UUP"],
            "signal_func": make_vol_adj_rs_signal(["QQQ", "GLD", "TLT", "UUP"]),
        },
        {
            "name": "D) Mixed Pool + Regime Overlay",
            "pool": ["QQQ", "GLD", "TLT", "UUP"],
            "signal_func": make_regime_overlay_signal(["QQQ", "GLD", "TLT", "UUP"]),
        },
        {
            "name": "E) Dual Momentum (abs return filter)",
            "pool": ["QQQ", "GLD", "TLT", "UUP"],
            "signal_func": make_dual_momentum_signal(["QQQ", "GLD", "TLT", "UUP"]),
        },
        {
            "name": "F) Composite Score (RS+Mom+InvVol)",
            "pool": ["QQQ", "GLD", "TLT", "UUP"],
            "signal_func": make_composite_score_signal(["QQQ", "GLD", "TLT", "UUP"]),
        },
    ]

    all_results = []

    print("\n" + "=" * 80)
    print("RELATIVE STRENGTH ROTATION — EXTENDED BACKTEST")
    print(f"OOT Period: {OOT_START} to {OOT_END} | Account: ${ACCOUNT_SIZE}")
    print("=" * 80)

    for v in variants:
        print(f"\n{'─' * 60}")
        print(f"Running: {v['name']}")
        print(f"{'─' * 60}")

        df, n_trades, trade_log = run_rotation_backtest(
            close, v["pool"], v["signal_func"], name=v["name"]
        )

        metrics = calc_metrics(df, n_trades, v["name"])
        regime = regime_analysis(df, close)

        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"  Total Return: {metrics['total_return']:.1f}% | Ann Return: {metrics['ann_return']:.1f}%")
        print(f"  MaxDD: {metrics['max_dd']:.1f}% | PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate']:.1f}%")
        print(f"  Trades: {n_trades} | Final Equity: ${metrics['final_equity']:.2f}")
        print(f"  Bull Sharpe: {regime['bull_sharpe']:.3f} | Bear Sharpe: {regime['bear_sharpe']:.3f} | Gap: {regime['regime_gap']:.3f}")

        # Permutation test
        print(f"  Running {N_PERM} permutations...")
        perm_p, perm_sharpes = permutation_test(close, v["pool"], v["signal_func"], metrics["sharpe"])
        print(f"  Perm p-value: {perm_p:.4f}")

        # 5-gate validation
        gates = validate_5gate(metrics, regime, perm_p)

        # Holding distribution
        if not df.empty and "holding" in df.columns:
            holding_pct = df["holding"].value_counts(normalize=True) * 100
            holding_str = ", ".join([f"{t}: {p:.1f}%" for t, p in holding_pct.items()])
        else:
            holding_str = "N/A"

        result = {
            "variant": v["name"],
            "pool": v["pool"],
            "metrics": metrics,
            "regime": regime,
            "perm_p_value": perm_p,
            "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
            "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
            "gates": gates,
            "holding_distribution": holding_str,
            "n_rebalances": n_trades,
        }
        all_results.append(result)

        # Print gate results
        gate_str = " | ".join([
            f"{'PASS' if v else 'FAIL'}: {k}" for k, v in gates.items() if k != "all_pass"
        ])
        status = "ALL GATES PASS" if gates["all_pass"] else "FAILED"
        print(f"  5-Gate: {status}")
        print(f"    {gate_str}")
        print(f"  Holdings: {holding_str}")

    # ─── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Variant':<42} {'Sharpe':>7} {'Sortino':>8} {'Ret%':>7} {'MaxDD%':>7} {'PF':>6} {'Perm-p':>7} {'Gates':>6}"
    print(header)
    print("-" * len(header))

    passed = []
    for r in all_results:
        m = r["metrics"]
        status = "PASS" if r["gates"]["all_pass"] else "FAIL"
        if r["gates"]["all_pass"]:
            passed.append(r)
        print(f"{r['variant']:<42} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['total_return']:>6.1f}% {m['max_dd']:>6.1f}% {m['profit_factor']:>6.2f} {r['perm_p_value']:>7.4f} {status:>6}")

    print(f"\n{len(passed)}/{len(all_results)} variants passed all 5 gates.")

    if passed:
        best = max(passed, key=lambda x: x["metrics"]["sharpe"])
        print(f"\nBest validated variant: {best['variant']}")
        print(f"  Sharpe {best['metrics']['sharpe']:.3f}, Sortino {best['metrics']['sortino']:.3f}, "
              f"Return {best['metrics']['total_return']:.1f}%, MaxDD {best['metrics']['max_dd']:.1f}%")

    # ─── Save Results ─────────────────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "account_size": ACCOUNT_SIZE,
        "slippage_pct": SLIPPAGE_PCT,
        "n_permutations": N_PERM,
        "variants": all_results,
        "n_passed": len(passed),
        "best_variant": passed[0]["variant"] if passed else None,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
