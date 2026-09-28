#!/usr/bin/env python3
"""
Correlation Breakdown Backtest
Tests whether sector correlation breakdowns predict profitable dip-buying opportunities.

Hypothesis: When sector ETF correlations spike or break down, individual sectors that
diverge from the pack are more likely to mean-revert.

6 Signal Variants:
  A. High Dispersion Dip-Buy
  B. Correlation Breakdown
  C. Dispersion Mean-Reversion
  D. Relative Strength Reversal
  E. Correlation Spike + Dip
  F. Combined: High Dispersion + Bounce Confirmation
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
SECTOR_ETFS = ["XLK", "XLP", "XLC", "XLY", "XLF", "XLI", "XLE", "XLU", "XLB", "XLRE", "XLV"]
BENCHMARK = "SPY"
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
COST_RT_PCT = 0.10 / 100  # 0.10% round-trip
HOLD_DAYS = 5
TP_PCT = 0.03  # +3%
SL_PCT = -0.05  # -5%
N_PERMUTATIONS = 1000
RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/scripts/growth_research/results")


def download_data():
    """Download daily data for sector ETFs + SPY."""
    print(f"Downloading data from {START_DATE} to {END_DATE}...")
    data = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    close = data["Close"][ALL_TICKERS].dropna()
    print(f"  Got {len(close)} trading days from {close.index[0].date()} to {close.index[-1].date()}")
    return close


def compute_rsi(series, period=14):
    """Compute RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_features(close):
    """Compute all features needed for signal generation."""
    sectors = close[SECTOR_ETFS]
    spy = close[BENCHMARK]

    # Returns
    ret_5d = sectors.pct_change(5)
    ret_1d = sectors.pct_change(1)
    spy_ret_1d = spy.pct_change(1)

    # RSI(14) for each sector
    rsi = pd.DataFrame({t: compute_rsi(close[t], 14) for t in SECTOR_ETFS}, index=close.index)

    # Cross-sectional dispersion: std of 5-day returns across sectors
    dispersion = ret_5d.std(axis=1)
    dispersion_80pct = dispersion.rolling(252).quantile(0.80)
    dispersion_60d_avg = dispersion.rolling(60).mean()

    # Rolling 20-day correlation of each sector vs SPY
    corr_vs_spy = pd.DataFrame(index=close.index, columns=SECTOR_ETFS, dtype=float)
    for t in SECTOR_ETFS:
        corr_vs_spy[t] = ret_1d[t].rolling(20).corr(spy_ret_1d)

    # Average pairwise correlation across all sectors (using 20d rolling)
    # Approximate: average of each sector's corr vs SPY
    avg_pairwise_corr = corr_vs_spy.mean(axis=1)

    # Z-score of 5-day return vs cross-sectional mean
    cs_mean = ret_5d.mean(axis=1)
    cs_std = ret_5d.std(axis=1)
    z_scores = ret_5d.sub(cs_mean, axis=0).div(cs_std, axis=0)

    # 10-day low for bounce confirmation
    low_10d = sectors.rolling(10).min()

    # SPY open vs close for regime classification
    spy_open = yf.download(BENCHMARK, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)["Open"]

    return {
        "sectors": sectors,
        "spy": spy,
        "ret_5d": ret_5d,
        "ret_1d": ret_1d,
        "spy_ret_1d": spy_ret_1d,
        "rsi": rsi,
        "dispersion": dispersion,
        "dispersion_80pct": dispersion_80pct,
        "dispersion_60d_avg": dispersion_60d_avg,
        "corr_vs_spy": corr_vs_spy,
        "avg_pairwise_corr": avg_pairwise_corr,
        "z_scores": z_scores,
        "low_10d": low_10d,
        "spy_open": spy_open,
    }


def generate_signals_A(feat, close):
    """High Dispersion Dip-Buy: dispersion > 80th pct AND lowest-RSI sector has RSI < 35."""
    trades = []
    for i in range(100, len(close) - HOLD_DAYS - 1):
        dt = close.index[i]
        disp = feat["dispersion"].iloc[i]
        disp_thresh = feat["dispersion_80pct"].iloc[i]
        if pd.isna(disp) or pd.isna(disp_thresh) or disp <= disp_thresh:
            continue
        rsi_row = feat["rsi"].iloc[i]
        valid = rsi_row.dropna()
        if len(valid) == 0:
            continue
        lowest_rsi_sector = valid.idxmin()
        if valid[lowest_rsi_sector] < 35:
            trades.append((dt, lowest_rsi_sector, i))
    return trades


def generate_signals_B(feat, close):
    """Correlation Breakdown: sector corr vs SPY < 0.5 AND RSI < 35."""
    trades = []
    for i in range(100, len(close) - HOLD_DAYS - 1):
        dt = close.index[i]
        corr_row = feat["corr_vs_spy"].iloc[i]
        rsi_row = feat["rsi"].iloc[i]
        for t in SECTOR_ETFS:
            c = corr_row[t]
            r = rsi_row[t]
            if pd.notna(c) and pd.notna(r) and c < 0.5 and r < 35:
                trades.append((dt, t, i))
    return trades


def generate_signals_C(feat, close):
    """Dispersion Mean-Reversion: dispersion > 1.5x 60d avg AND sector in bottom 2 AND RSI < 40."""
    trades = []
    for i in range(100, len(close) - HOLD_DAYS - 1):
        dt = close.index[i]
        disp = feat["dispersion"].iloc[i]
        disp_avg = feat["dispersion_60d_avg"].iloc[i]
        if pd.isna(disp) or pd.isna(disp_avg) or disp_avg == 0 or disp <= 1.5 * disp_avg:
            continue
        ret_row = feat["ret_5d"].iloc[i].dropna()
        if len(ret_row) < 3:
            continue
        bottom2 = ret_row.nsmallest(2).index.tolist()
        rsi_row = feat["rsi"].iloc[i]
        for t in bottom2:
            if pd.notna(rsi_row[t]) and rsi_row[t] < 40:
                trades.append((dt, t, i))
    return trades


def generate_signals_D(feat, close):
    """Relative Strength Reversal: z-score of 5d return < -2.0."""
    trades = []
    for i in range(100, len(close) - HOLD_DAYS - 1):
        dt = close.index[i]
        z_row = feat["z_scores"].iloc[i]
        for t in SECTOR_ETFS:
            z = z_row[t]
            if pd.notna(z) and z < -2.0:
                trades.append((dt, t, i))
    return trades


def generate_signals_E(feat, close):
    """Correlation Spike + Dip: avg pairwise corr > 0.85 AND sector drops more than SPY."""
    trades = []
    for i in range(100, len(close) - HOLD_DAYS - 1):
        dt = close.index[i]
        avg_corr = feat["avg_pairwise_corr"].iloc[i]
        if pd.isna(avg_corr) or avg_corr <= 0.85:
            continue
        spy_ret = feat["spy_ret_1d"].iloc[i]
        if pd.isna(spy_ret):
            continue
        worst_sector = None
        worst_diff = 0
        for t in SECTOR_ETFS:
            sec_ret = feat["ret_1d"][t].iloc[i]
            if pd.notna(sec_ret):
                diff = sec_ret - spy_ret
                if diff < worst_diff:
                    worst_diff = diff
                    worst_sector = t
        if worst_sector is not None and worst_diff < 0:
            trades.append((dt, worst_sector, i))
    return trades


def generate_signals_F(feat, close):
    """Combined: Signal A conditions + bounce confirmation (yesterday near 10d low, today > yesterday)."""
    trades = []
    for i in range(100, len(close) - HOLD_DAYS - 1):
        dt = close.index[i]
        disp = feat["dispersion"].iloc[i]
        disp_thresh = feat["dispersion_80pct"].iloc[i]
        if pd.isna(disp) or pd.isna(disp_thresh) or disp <= disp_thresh:
            continue
        rsi_row = feat["rsi"].iloc[i]
        valid = rsi_row.dropna()
        if len(valid) == 0:
            continue
        lowest_rsi_sector = valid.idxmin()
        if valid[lowest_rsi_sector] >= 35:
            continue
        t = lowest_rsi_sector
        # Bounce confirmation: yesterday near 10d low, today closes higher than yesterday
        if i < 1:
            continue
        yesterday_close = close[t].iloc[i - 1]
        today_close = close[t].iloc[i]
        low_10d = feat["low_10d"][t].iloc[i - 1]
        if pd.isna(yesterday_close) or pd.isna(today_close) or pd.isna(low_10d):
            continue
        # "near 10d low" = within 1% of 10d low
        if yesterday_close <= low_10d * 1.01 and today_close > yesterday_close:
            trades.append((dt, t, i))
    return trades


def simulate_trades(trades, close):
    """Simulate trades with 5-day hold, +3% TP, -5% SL, 0.10% RT cost."""
    results = []
    for dt, ticker, idx in trades:
        entry_price = close[ticker].iloc[idx]
        if pd.isna(entry_price) or entry_price == 0:
            continue

        exit_price = None
        exit_day = None
        for d in range(1, HOLD_DAYS + 1):
            if idx + d >= len(close):
                break
            day_close = close[ticker].iloc[idx + d]
            if pd.isna(day_close):
                continue
            pct = (day_close - entry_price) / entry_price
            if pct >= TP_PCT:
                exit_price = entry_price * (1 + TP_PCT)
                exit_day = d
                break
            elif pct <= SL_PCT:
                exit_price = entry_price * (1 + SL_PCT)
                exit_day = d
                break

        if exit_price is None:
            # Exit at end of hold period
            last_idx = min(idx + HOLD_DAYS, len(close) - 1)
            exit_price = close[ticker].iloc[last_idx]
            exit_day = last_idx - idx

        if pd.isna(exit_price):
            continue

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - COST_RT_PCT

        # Regime: green if SPY close > SPY open on entry day
        spy_close = close[BENCHMARK].iloc[idx]
        spy_open_val = None
        try:
            spy_open_val = close[BENCHMARK].iloc[idx]  # placeholder
        except:
            pass

        results.append({
            "entry_date": dt,
            "ticker": ticker,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "hold_days": exit_day,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "idx": idx,
        })
    return results


def classify_regime(results, spy_open_series, spy_close_series):
    """Classify each trade's entry day as green or red based on SPY."""
    for r in results:
        dt = r["entry_date"]
        try:
            if dt in spy_open_series.index and dt in spy_close_series.index:
                o = spy_open_series.loc[dt]
                c = spy_close_series.loc[dt]
                if isinstance(o, pd.Series):
                    o = o.iloc[0]
                if isinstance(c, pd.Series):
                    c = c.iloc[0]
                r["regime"] = "green" if c > o else "red"
            else:
                r["regime"] = "unknown"
        except:
            r["regime"] = "unknown"
    return results


def compute_metrics(results, label):
    """Compute all required metrics for a set of trade results."""
    if len(results) == 0:
        return {
            "variant": label,
            "total_trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "sharpe": 0,
            "sortino": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "sharpe_green": 0,
            "sharpe_red": 0,
            "regime_gap": 0,
            "regime_gap_pass": False,
            "p_value": 1.0,
            "day_concentration": 0,
            "day_conc_pass": False,
            "overall_pass": False,
        }

    rets = np.array([r["net_return"] for r in results])
    n = len(rets)
    wins = np.sum(rets > 0)
    wr = wins / n

    avg_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-9

    # Annualized Sharpe (assume ~50 trades/year scaling, or use sqrt(252/avg_hold))
    avg_hold = np.mean([r["hold_days"] for r in results])
    trades_per_year = 252 / max(avg_hold, 1)
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-9 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-9 else 0

    # Profit Factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max Drawdown (on cumulative equity curve)
    equity = np.cumsum(rets)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

    # Regime-stratified Sharpe
    green_rets = np.array([r["net_return"] for r in results if r.get("regime") == "green"])
    red_rets = np.array([r["net_return"] for r in results if r.get("regime") == "red"])

    def _sharpe(arr):
        if len(arr) < 2:
            return 0
        s = np.std(arr, ddof=1)
        if s < 1e-9:
            return 0
        return (np.mean(arr) / s) * np.sqrt(trades_per_year)

    sharpe_green = _sharpe(green_rets)
    sharpe_red = _sharpe(red_rets)
    denom = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / denom if denom > 0 else 0
    regime_pass = regime_gap < 0.50

    # Permutation test
    observed_mean = avg_ret
    count_better = 0
    for _ in range(N_PERMUTATIONS):
        perm_rets = np.random.permutation(rets)
        # Shuffle entry dates to break signal-return link
        perm_mean = np.mean(perm_rets)
        if perm_mean >= observed_mean:
            count_better += 1
    # Actually we need to shuffle returns against dates to test signal timing
    # Simple approach: shuffle returns and compare mean (tests if ordering matters)
    # More correct: compare observed mean to distribution of random-entry means
    # For simplicity, use the standard permutation: is the mean return significantly > 0?
    count_better = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = rets * np.random.choice([-1, 1], size=n)
        if np.mean(shuffled) >= observed_mean:
            count_better += 1
    p_value = count_better / N_PERMUTATIONS

    # Day concentration
    date_counts = pd.Series([r["entry_date"] for r in results]).value_counts()
    day_conc = date_counts.max() / n if n > 0 else 0
    day_conc_pass = day_conc < 0.70

    overall_pass = regime_pass and day_conc_pass and p_value < 0.05 and sharpe > 0.5 and pf > 1.0

    return {
        "variant": label,
        "total_trades": n,
        "win_rate": round(wr, 4),
        "avg_return": round(avg_ret * 100, 4),  # in %
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "max_drawdown": round(max_dd * 100, 3),  # in %
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 3),
        "regime_gap_pass": regime_pass,
        "green_trades": len(green_rets),
        "red_trades": len(red_rets),
        "p_value": round(p_value, 4),
        "day_concentration": round(day_conc, 4),
        "day_conc_pass": day_conc_pass,
        "overall_pass": overall_pass,
    }


def main():
    np.random.seed(42)

    # Download data
    close = download_data()

    # Download SPY OHLC for regime classification
    print("Downloading SPY OHLC for regime classification...")
    spy_ohlc = yf.download(BENCHMARK, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    spy_open = spy_ohlc["Open"]
    spy_close_price = spy_ohlc["Close"]
    if isinstance(spy_open, pd.DataFrame):
        spy_open = spy_open.iloc[:, 0]
    if isinstance(spy_close_price, pd.DataFrame):
        spy_close_price = spy_close_price.iloc[:, 0]

    # Compute features
    print("Computing features...")
    feat = compute_features(close)

    # Signal generators
    signal_generators = {
        "A_HighDispersionDipBuy": generate_signals_A,
        "B_CorrelationBreakdown": generate_signals_B,
        "C_DispersionMeanRev": generate_signals_C,
        "D_RelStrengthReversal": generate_signals_D,
        "E_CorrSpikeDip": generate_signals_E,
        "F_CombinedBounce": generate_signals_F,
    }

    all_metrics = {}

    for label, gen_func in signal_generators.items():
        print(f"\n{'='*60}")
        print(f"Signal Variant: {label}")
        print(f"{'='*60}")

        # Generate signals
        trades = gen_func(feat, close)
        print(f"  Raw signals: {len(trades)}")

        # Simulate
        results = simulate_trades(trades, close)
        print(f"  Executed trades: {len(results)}")

        # Classify regime
        results = classify_regime(results, spy_open, spy_close_price)

        # Compute metrics
        metrics = compute_metrics(results, label)
        all_metrics[label] = metrics

        # Print
        print(f"  Total Trades:    {metrics['total_trades']}")
        print(f"  Win Rate:        {metrics['win_rate']:.1%}")
        print(f"  Avg Return:      {metrics['avg_return']:.3f}%")
        print(f"  Sharpe:          {metrics['sharpe']:.3f}")
        print(f"  Sortino:         {metrics['sortino']:.3f}")
        print(f"  Profit Factor:   {metrics['profit_factor']:.3f}")
        print(f"  Max Drawdown:    {metrics['max_drawdown']:.3f}%")
        print(f"  Sharpe (green):  {metrics['sharpe_green']:.3f}  ({metrics.get('green_trades', 0)} trades)")
        print(f"  Sharpe (red):    {metrics['sharpe_red']:.3f}  ({metrics.get('red_trades', 0)} trades)")
        print(f"  Regime Gap:      {metrics['regime_gap']:.3f}  {'PASS' if metrics['regime_gap_pass'] else 'FAIL'}")
        print(f"  P-value:         {metrics['p_value']:.4f}  {'PASS' if metrics['p_value'] < 0.05 else 'FAIL'}")
        print(f"  Day Conc:        {metrics['day_concentration']:.4f}  {'PASS' if metrics['day_conc_pass'] else 'FAIL'}")
        print(f"  OVERALL:         {'*** PASS ***' if metrics['overall_pass'] else 'FAIL'}")

    # Summary table
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    header = f"{'Variant':<28} {'Trades':>6} {'WR':>6} {'AvgRet%':>8} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD%':>8} {'RegGap':>7} {'P-val':>6} {'DayConc':>8} {'Result':>8}"
    print(header)
    print("-" * 100)

    for label, m in all_metrics.items():
        regime_str = f"{m['regime_gap']:.2f}"
        pval_str = f"{m['p_value']:.3f}"
        conc_str = f"{m['day_concentration']:.3f}"
        result = "PASS" if m["overall_pass"] else "FAIL"
        gates = []
        if not m["regime_gap_pass"]:
            gates.append("REG")
        if not m["day_conc_pass"]:
            gates.append("CONC")
        if m["p_value"] >= 0.05:
            gates.append("PVAL")
        if m["sharpe"] <= 0.5:
            gates.append("SR")
        if m["profit_factor"] <= 1.0:
            gates.append("PF")
        fail_reason = ",".join(gates) if gates else ""
        result_str = result + (f"({fail_reason})" if fail_reason else "")

        print(f"{label:<28} {m['total_trades']:>6} {m['win_rate']:>5.1%} {m['avg_return']:>8.3f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['max_drawdown']:>8.3f} {regime_str:>7} {pval_str:>6} {conc_str:>8} {result_str:>8}")

    print(f"\nGates: Sharpe>0.5, PF>1.0, P-value<0.05, RegimeGap<0.50, DayConc<0.70")
    print(f"Cost: {COST_RT_PCT*100:.2f}% RT | Exit: {HOLD_DAYS}d hold / +{TP_PCT*100:.0f}% TP / {SL_PCT*100:.0f}% SL")
    print(f"Data: {START_DATE} to {END_DATE} | Permutations: {N_PERMUTATIONS}")

    # Save results
    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "cost_rt_pct": COST_RT_PCT * 100,
            "hold_days": HOLD_DAYS,
            "tp_pct": TP_PCT * 100,
            "sl_pct": SL_PCT * 100,
            "n_permutations": N_PERMUTATIONS,
            "tickers": ALL_TICKERS,
        },
        "variants": all_metrics,
    }

    results_path = RESULTS_DIR / "correlation_breakdown_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
