#!/usr/bin/env python3
"""
Cross-Sectional Momentum Backtest v1
======================================
Buy recent winners, sell (avoid) recent losers — ranked across a universe.
Classic Jegadeesh & Titman (1993) anomaly.

Configs tested:
  1. Top 5 by 12m return (skip last month), hold 1 month
  2. Top 5 by 6m return (skip last month), hold 1 month
  3. Top 5 by 3m return (skip last month), hold 1 month
  4. Top 10 by 12m return (skip last month), hold 1 month
  5. Top 5 by 12m return, hold 3 months (quarterly rebal)
  6. L/S: long top 5 / short bottom 5 by 12m return, hold 1 month
  7. Top 5 by 6m return, hold 1 month, ONLY when SPY > 200d MA
  8. Top 3 by 12m return (skip last month), hold 1 month (concentrated)

Quality gates (inline):
  - Permutation test (1000 shuffles, p < 0.05)
  - R1 regime test (green/red Sharpe gap < 0.50)
  - Sub-period consistency (first-half & second-half both positive)
  - Outlier robustness (remove top/bottom 5%, still profitable)
"""

import sys, json, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "xsmom_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# UNIVERSE & PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','V','MA',
    'UNH','JNJ','PG','HD','KO','PEP','COST','MCD','AVGO','LLY',
    'ABBV','MRK','CVX','XOM','WMT','BAC','GS','MS','CRM','NFLX',
]

START_DATE = "2014-01-01"
END_DATE   = "2026-07-01"

N_PERMUTATIONS   = 1000
RISK_FREE_RATE   = 0.04
REGIME_GAP_LIMIT = 0.50

CONFIGS = [
    {"name": "12m_top5_1m",       "lookback_months": 12, "skip_last": True,  "top_n": 5,  "hold_months": 1, "long_short": False, "trend_filter": False},
    {"name": "6m_top5_1m",        "lookback_months": 6,  "skip_last": True,  "top_n": 5,  "hold_months": 1, "long_short": False, "trend_filter": False},
    {"name": "3m_top5_1m",        "lookback_months": 3,  "skip_last": True,  "top_n": 5,  "hold_months": 1, "long_short": False, "trend_filter": False},
    {"name": "12m_top10_1m",      "lookback_months": 12, "skip_last": True,  "top_n": 10, "hold_months": 1, "long_short": False, "trend_filter": False},
    {"name": "12m_top5_3m",       "lookback_months": 12, "skip_last": True,  "top_n": 5,  "hold_months": 3, "long_short": False, "trend_filter": False},
    {"name": "12m_LS5_1m",        "lookback_months": 12, "skip_last": True,  "top_n": 5,  "hold_months": 1, "long_short": True,  "trend_filter": False},
    {"name": "6m_top5_1m_trend",  "lookback_months": 6,  "skip_last": True,  "top_n": 5,  "hold_months": 1, "long_short": False, "trend_filter": True},
    {"name": "12m_top3_1m_conc",  "lookback_months": 12, "skip_last": True,  "top_n": 3,  "hold_months": 1, "long_short": False, "trend_filter": False},
]


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    """Download adjusted close prices for all tickers + SPY via yfinance."""
    import yfinance as yf

    all_tickers = TICKERS + ["SPY"]
    print(f"Downloading data for {len(all_tickers)} tickers from {START_DATE} to {END_DATE}...")

    data = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns: (Price, Ticker)
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    # Drop tickers with too much missing data (>20%)
    missing_pct = close.isnull().mean()
    valid = missing_pct[missing_pct < 0.20].index.tolist()
    close = close[valid].ffill().dropna()

    print(f"  Loaded {len(valid)} tickers, {len(close)} trading days ({close.index[0].date()} to {close.index[-1].date()})")
    return close


# ═════════════════════════════════════════════════════════════════════════════
# REBALANCE DATE GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def get_rebalance_dates(close_index, hold_months=1):
    """Get month-end rebalance dates from the close price index."""
    # Use last trading day of each month
    monthly = pd.Series(close_index, index=close_index).resample("ME").last()
    dates = [pd.Timestamp(d) for d in monthly.values]

    if hold_months == 1:
        return dates
    else:
        # For quarterly (hold_months=3), take every 3rd month-end
        return dates[::hold_months]


# ═════════════════════════════════════════════════════════════════════════════
# MOMENTUM RANKING
# ═════════════════════════════════════════════════════════════════════════════

def compute_momentum_signal(close, date, lookback_months, skip_last):
    """
    Compute trailing return for each stock as of `date`.
    If skip_last=True, skip the most recent month (T-lookback to T-1m).
    Returns a Series of returns indexed by ticker.
    """
    stock_cols = [c for c in close.columns if c != "SPY"]

    # Find the row index for `date`
    mask = close.index <= date
    if mask.sum() == 0:
        return None

    end_idx = close.index[mask][-1]

    if skip_last:
        # End of signal window = 1 month before rebalance
        end_signal_mask = close.index <= (end_idx - pd.DateOffset(months=1))
        if end_signal_mask.sum() == 0:
            return None
        end_signal = close.index[end_signal_mask][-1]
    else:
        end_signal = end_idx

    # Start of signal window = lookback_months before end_signal
    start_signal_mask = close.index <= (end_signal - pd.DateOffset(months=lookback_months))
    if start_signal_mask.sum() == 0:
        return None
    start_signal = close.index[start_signal_mask][-1]

    # Compute return
    p_start = close.loc[start_signal, stock_cols]
    p_end   = close.loc[end_signal, stock_cols]

    ret = (p_end / p_start) - 1.0

    # Drop NaN
    ret = ret.dropna()
    return ret


def check_trend_filter(close, date):
    """Check if SPY is above its 200-day MA on `date`."""
    if "SPY" not in close.columns:
        return True  # No filter if SPY data missing

    mask = close.index <= date
    spy = close.loc[mask, "SPY"]
    if len(spy) < 200:
        return True  # Not enough data, allow

    ma200 = spy.iloc[-200:].mean()
    return spy.iloc[-1] > ma200


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(close, config):
    """
    Run one config. Returns period_returns, period_meta, and per_period_stock_rets
    (the latter used for fast permutation testing).
    """
    rebal_dates = get_rebalance_dates(close.index, config["hold_months"])

    period_returns = []
    period_meta = []
    # For permutation: store (available_tickers, individual_rets_dict) per valid period
    period_stock_data = []

    for i in range(len(rebal_dates) - 1):
        entry_date = rebal_dates[i]
        exit_date  = rebal_dates[i + 1]

        # Trend filter
        if config["trend_filter"] and not check_trend_filter(close, entry_date):
            continue

        # Compute momentum signal
        mom = compute_momentum_signal(close, entry_date, config["lookback_months"], config["skip_last"])
        if mom is None or len(mom) < config["top_n"] * 2:
            continue

        available = mom.index.tolist()

        # Compute ALL individual stock returns for this period (for permutation reuse)
        entry_prices = close.loc[entry_date, available]
        exit_prices  = close.loc[exit_date, available]
        all_rets = ((exit_prices / entry_prices) - 1.0).to_dict()

        # Rank and select
        ranked = mom.sort_values(ascending=False)
        top_n = ranked.head(config["top_n"]).index.tolist()

        long_ret = np.mean([all_rets[t] for t in top_n])

        if config["long_short"]:
            bottom_n = ranked.tail(config["top_n"]).index.tolist()
            short_ret = -np.mean([all_rets[t] for t in bottom_n])
            period_ret = (long_ret + short_ret) / 2.0
        else:
            period_ret = long_ret

        # SPY prior-month return for regime classification
        spy_mask_end = close.index <= entry_date
        spy_mask_start = close.index <= (entry_date - pd.DateOffset(months=1))
        if spy_mask_start.sum() > 0 and "SPY" in close.columns:
            spy_start = close.loc[close.index[spy_mask_start][-1], "SPY"]
            spy_end   = close.loc[close.index[spy_mask_end][-1], "SPY"]
            spy_prior_ret = (spy_end / spy_start) - 1.0
        else:
            spy_prior_ret = 0.0

        period_returns.append(period_ret)
        period_meta.append({
            "date": str(pd.Timestamp(entry_date).date()),
            "spy_prior_ret": spy_prior_ret,
            "period_ret": period_ret,
            "n_stocks": config["top_n"],
        })
        period_stock_data.append((available, all_rets))

    return period_returns, period_meta, period_stock_data


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(period_returns, hold_months=1):
    """Compute Sharpe, Sortino, WR, PF, MaxDD, CAGR from period returns."""
    rets = np.array(period_returns)
    n = len(rets)
    if n < 3:
        return None

    periods_per_year = 12 / hold_months
    mean_ret = rets.mean()
    std_ret  = rets.std(ddof=1)

    # Sharpe (annualized)
    excess = mean_ret - RISK_FREE_RATE / periods_per_year
    sharpe = (excess / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    # Sortino (annualized)
    downside = rets[rets < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else std_ret
    sortino = (excess / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    # Win rate
    win_rate = (rets > 0).sum() / n

    # Profit factor
    gross_profit = rets[rets > 0].sum() if (rets > 0).any() else 0.0
    gross_loss   = abs(rets[rets < 0].sum()) if (rets < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (from cumulative return curve)
    cum = (1 + rets).cumprod()
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd = drawdowns.min() * 100  # Negative percentage

    # CAGR
    total_return = cum[-1]
    n_years = n / periods_per_year
    cagr = (total_return ** (1 / n_years) - 1) * 100 if n_years > 0 and total_return > 0 else 0.0

    return {
        "n_periods": n,
        "n_trades_total": n,  # Each period = 1 portfolio trade
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "mean_return_pct": round(mean_ret * 100, 4),
        "max_dd_pct": round(max_dd, 2),
        "cagr_pct": round(cagr, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY GATES
# ═════════════════════════════════════════════════════════════════════════════

def permutation_test(config, real_mean, period_stock_data, n_perms=1000):
    """
    Test if momentum selection beats random selection (p < 0.05).
    Uses precomputed per-period stock returns for speed.
    """
    print(f"    Running {n_perms} permutation tests (vectorized)...")
    top_n = config["top_n"]
    is_ls = config["long_short"]

    random_means = np.zeros(n_perms)

    for seed in range(n_perms):
        rng = np.random.RandomState(seed)
        perm_rets = []

        for available, all_rets in period_stock_data:
            picks = list(rng.choice(available, size=min(top_n, len(available)), replace=False))
            long_ret = np.mean([all_rets[t] for t in picks])

            if is_ls:
                remaining = [t for t in available if t not in picks]
                shorts = list(rng.choice(remaining, size=min(top_n, len(remaining)), replace=False))
                short_ret = -np.mean([all_rets[t] for t in shorts])
                perm_rets.append((long_ret + short_ret) / 2.0)
            else:
                perm_rets.append(long_ret)

        if perm_rets:
            random_means[seed] = np.mean(perm_rets)

    p_value = (random_means >= real_mean).sum() / n_perms

    return {
        "pass": p_value < 0.05,
        "p_value": round(float(p_value), 4),
        "real_mean": round(real_mean, 6),
        "random_mean_avg": round(float(random_means.mean()), 6),
        "random_mean_std": round(float(random_means.std()), 6),
    }


def regime_test(period_meta):
    """R1: Sharpe in green-period entries vs red-period entries."""
    green_rets = [m["period_ret"] for m in period_meta if m["spy_prior_ret"] > 0]
    red_rets   = [m["period_ret"] for m in period_meta if m["spy_prior_ret"] <= 0]

    if len(green_rets) < 5 or len(red_rets) < 5:
        return {"pass": False, "reason": "insufficient_regime_data",
                "n_green": len(green_rets), "n_red": len(red_rets)}

    green_arr = np.array(green_rets)
    red_arr   = np.array(red_rets)

    def _sharpe(r):
        if r.std() == 0:
            return 0.0
        return (r.mean() / r.std()) * np.sqrt(12)

    sharpe_green = _sharpe(green_arr)
    sharpe_red   = _sharpe(red_arr)

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0.0

    return {
        "pass": gap <= REGIME_GAP_LIMIT,
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "gap_ratio": round(gap, 3),
        "n_green": len(green_rets),
        "n_red": len(red_rets),
    }


def subperiod_test(period_returns):
    """First-half and second-half both positive mean return."""
    n = len(period_returns)
    mid = n // 2
    first_half  = np.array(period_returns[:mid])
    second_half = np.array(period_returns[mid:])

    fh_mean = first_half.mean()
    sh_mean = second_half.mean()

    return {
        "pass": fh_mean > 0 and sh_mean > 0,
        "first_half_mean_pct": round(fh_mean * 100, 4),
        "second_half_mean_pct": round(sh_mean * 100, 4),
        "n_first": len(first_half),
        "n_second": len(second_half),
    }


def outlier_robustness_test(period_returns):
    """Remove top/bottom 5% of returns, must still be profitable."""
    rets = np.array(sorted(period_returns))
    n = len(rets)
    trim = max(1, int(n * 0.05))
    trimmed = rets[trim:-trim]

    if len(trimmed) < 5:
        return {"pass": False, "reason": "insufficient_data_after_trim"}

    trimmed_mean = trimmed.mean()

    return {
        "pass": trimmed_mean > 0,
        "trimmed_mean_pct": round(trimmed_mean * 100, 4),
        "n_removed": trim * 2,
        "n_remaining": len(trimmed),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CROSS-SECTIONAL MOMENTUM BACKTEST v1")
    print("=" * 70)

    close = load_data()

    results = []

    for cfg in CONFIGS:
        print(f"\n{'─' * 60}")
        print(f"Config: {cfg['name']}")
        print(f"  Lookback={cfg['lookback_months']}m, Top-N={cfg['top_n']}, Hold={cfg['hold_months']}m, "
              f"L/S={cfg['long_short']}, TrendFilter={cfg['trend_filter']}")
        print(f"{'─' * 60}")

        period_returns, period_meta, period_stock_data = run_backtest(close, cfg)

        if len(period_returns) < 10:
            print(f"  SKIP: only {len(period_returns)} periods")
            continue

        metrics = compute_metrics(period_returns, cfg["hold_months"])
        if metrics is None:
            print(f"  SKIP: insufficient data for metrics")
            continue

        print(f"  Periods: {metrics['n_periods']}  |  Sharpe: {metrics['sharpe']:.3f}  |  "
              f"Sortino: {metrics['sortino']:.3f}  |  WR: {metrics['win_rate']:.1%}")
        print(f"  Mean Ret: {metrics['mean_return_pct']:.3f}%  |  PF: {metrics['profit_factor']:.2f}  |  "
              f"MaxDD: {metrics['max_dd_pct']:.1f}%  |  CAGR: {metrics['cagr_pct']:.1f}%")

        # Quality gates
        print(f"  Running quality gates...")

        mean_ret = np.mean(period_returns)

        perm = permutation_test(cfg, mean_ret, period_stock_data, N_PERMUTATIONS)
        print(f"    Permutation: {'PASS' if perm['pass'] else 'FAIL'} (p={perm['p_value']:.4f})")

        regime = regime_test(period_meta)
        print(f"    Regime R1:   {'PASS' if regime['pass'] else 'FAIL'} "
              f"(green={regime.get('sharpe_green','N/A')}, red={regime.get('sharpe_red','N/A')}, "
              f"gap={regime.get('gap_ratio','N/A')})")

        subperiod = subperiod_test(period_returns)
        print(f"    Sub-period:  {'PASS' if subperiod['pass'] else 'FAIL'} "
              f"(1H={subperiod['first_half_mean_pct']:.3f}%, 2H={subperiod['second_half_mean_pct']:.3f}%)")

        outlier = outlier_robustness_test(period_returns)
        print(f"    Outlier:     {'PASS' if outlier['pass'] else 'FAIL'} "
              f"(trimmed_mean={outlier.get('trimmed_mean_pct','N/A')}%)")

        all_pass = perm["pass"] and regime["pass"] and subperiod["pass"] and outlier["pass"]
        print(f"  ══> ALL GATES: {'PASS' if all_pass else 'FAIL'}")

        result = {
            "config_name": cfg["name"],
            **metrics,
            "quality_gates": {
                "permutation_test": perm,
                "regime_test": regime,
                "subperiod_consistency": subperiod,
                "outlier_robustness": outlier,
                "all_pass": all_pass,
            },
        }
        results.append(result)

    # Save report
    report_path = OUTPUT / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"REPORT SAVED: {report_path}")
    print(f"{'=' * 70}")

    # Summary table
    print(f"\n{'Config':<25} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} "
          f"{'MeanRet%':>9} {'CAGR%':>7} {'MaxDD%':>7} {'Gates':>6}")
    print("─" * 85)
    for r in results:
        gates = "PASS" if r["quality_gates"]["all_pass"] else "FAIL"
        print(f"{r['config_name']:<25} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
              f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
              f"{r['mean_return_pct']:>8.3f}% {r['cagr_pct']:>6.1f}% "
              f"{r['max_dd_pct']:>6.1f}% {gates:>6}")


if __name__ == "__main__":
    main()
