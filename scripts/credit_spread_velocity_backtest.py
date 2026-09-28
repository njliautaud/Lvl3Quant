#!/usr/bin/env python3
"""
Credit Spread Velocity Backtest
================================
Signal: Rate of change of HYG vs TLT returns as leading indicator for sector ETF moves.
Uses SLIDING window (252d lookback, 1d step) per HC #0.
5-gate validation framework per HC #428.
"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLE", "XLU", "XLP", "XLF", "XLK", "XLC", "XLV", "XLY", "XLRE", "XLB", "XLI"]
RATE_SENSITIVE = ["XLF", "XLRE", "XLY"]
SIGNAL_TICKERS = ["HYG", "TLT", "SPY"]
ALL_TICKERS = SIGNAL_TICKERS + SECTOR_ETFS

LOOKBACK = 252  # sliding window
CSV3_WINDOW = 3   # 3-day credit spread velocity
CSV10_WINDOW = 10  # 10-day credit spread velocity
FORWARD_HORIZONS = [1, 3, 5]
HOLD_PERIOD = 5
COST_RT_PCT = 0.001  # 0.10% round-trip cost
N_QUINTILES = 5
N_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
DAY_CONC_CAP = 0.70

# 5-gate thresholds
GATE_SHARPE = 0.5
GATE_PVALUE = 0.05
GATE_REGIME_GAP = 0.50
GATE_MIN_TRADES = 50
GATE_MAX_DD = 0.40

OUTPUT_PATH = "/home/jupiter/Lvl3Quant/output/credit_spread_velocity_results.json"


def download_data(years=5):
    """Download daily data for all tickers."""
    end = datetime.now()
    start = end.replace(year=end.year - years)
    print(f"Downloading {years}y data for {len(ALL_TICKERS)} tickers...")

    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    # Drop any tickers with insufficient data
    min_rows = LOOKBACK + CSV10_WINDOW + max(FORWARD_HORIZONS) + 50
    valid = close.dropna(axis=1, thresh=min_rows)
    missing = set(ALL_TICKERS) - set(valid.columns)
    if missing:
        print(f"  WARNING: Insufficient data for {missing}")

    print(f"  Got {len(valid)} days, {len(valid.columns)} tickers")
    return valid


def compute_credit_spread_velocity(close_df):
    """Compute HYG - TLT daily return, then rolling sums."""
    hyg_ret = close_df["HYG"].pct_change()
    tlt_ret = close_df["TLT"].pct_change()

    # Credit spread velocity = HYG return - TLT return
    # When HYG outperforms TLT, credit spreads are tightening (risk-on)
    # When TLT outperforms HYG, credit spreads are widening (risk-off)
    csv_daily = hyg_ret - tlt_ret
    csv_3d = csv_daily.rolling(CSV3_WINDOW).sum()
    csv_10d = csv_daily.rolling(CSV10_WINDOW).sum()

    return csv_daily, csv_3d, csv_10d


def compute_forward_returns(close_df, horizons):
    """Compute forward returns for each sector ETF at each horizon."""
    fwd = {}
    for h in horizons:
        fwd[h] = close_df[SECTOR_ETFS].pct_change(h).shift(-h)
    return fwd


def sliding_quintile_backtest(csv_signal, fwd_returns, spy_close):
    """
    SLIDING window quintile sort backtest.
    For each day t, use the past LOOKBACK days to define quintile breakpoints,
    then classify day t's signal and measure forward return.
    """
    results = []
    dates = csv_signal.index

    for i in range(LOOKBACK + 1, len(dates)):
        t = dates[i]
        sig_val = csv_signal.iloc[i]

        if pd.isna(sig_val):
            continue

        # Sliding window: use past LOOKBACK days for quintile breakpoints
        window = csv_signal.iloc[max(0, i - LOOKBACK):i].dropna()
        if len(window) < 50:
            continue

        # Compute quintile breakpoints from the sliding window
        breakpoints = np.percentile(window, [20, 40, 60, 80])

        # Assign quintile (1=bottom/most bearish, 5=top/most bullish)
        if sig_val <= breakpoints[0]:
            q = 1
        elif sig_val <= breakpoints[1]:
            q = 2
        elif sig_val <= breakpoints[2]:
            q = 3
        elif sig_val <= breakpoints[3]:
            q = 4
        else:
            q = 5

        # SPY regime (green = close > prior close)
        spy_today = spy_close.get(t, np.nan)
        if i > 0:
            spy_prev = spy_close.get(dates[i-1], np.nan)
        else:
            spy_prev = np.nan

        if pd.notna(spy_today) and pd.notna(spy_prev):
            regime = "green" if spy_today > spy_prev else "red"
        else:
            regime = "unknown"

        results.append({
            "date": t,
            "signal": sig_val,
            "quintile": q,
            "regime": regime,
        })

    return pd.DataFrame(results).set_index("date")


def permutation_test(signal_quintiles, fwd_returns_series, n_perms=N_PERMUTATIONS):
    """
    Permutation test: shuffle signal-quintile assignments, compute long-short spread.
    Returns p-value.
    """
    # Align
    aligned = pd.DataFrame({
        "quintile": signal_quintiles,
        "fwd_ret": fwd_returns_series
    }).dropna()

    if len(aligned) < GATE_MIN_TRADES:
        return 1.0, 0.0

    # Observed spread: Q5 mean - Q1 mean
    q5 = aligned[aligned["quintile"] == 5]["fwd_ret"]
    q1 = aligned[aligned["quintile"] == 1]["fwd_ret"]

    if len(q5) < 10 or len(q1) < 10:
        return 1.0, 0.0

    observed_spread = q5.mean() - q1.mean()

    # Permutation
    rng = np.random.default_rng(42)
    count_extreme = 0
    for _ in range(n_perms):
        shuffled_q = rng.permutation(aligned["quintile"].values)
        shuf_q5 = aligned["fwd_ret"].values[shuffled_q == 5]
        shuf_q1 = aligned["fwd_ret"].values[shuffled_q == 1]
        if len(shuf_q5) > 0 and len(shuf_q1) > 0:
            perm_spread = shuf_q5.mean() - shuf_q1.mean()
            if abs(perm_spread) >= abs(observed_spread):
                count_extreme += 1

    p_value = count_extreme / n_perms
    return p_value, observed_spread


def compute_sharpe(returns_series):
    """Annualized Sharpe ratio from daily returns."""
    if len(returns_series) < 10 or returns_series.std() == 0:
        return 0.0
    return (returns_series.mean() / returns_series.std()) * np.sqrt(252)


def compute_max_drawdown(cumulative_returns):
    """Max drawdown from cumulative return series."""
    peak = cumulative_returns.expanding().max()
    dd = (cumulative_returns - peak) / peak
    return abs(dd.min()) if len(dd) > 0 else 0.0


def compute_day_concentration(daily_pnl):
    """Max single-day P&L as fraction of total absolute P&L."""
    total = daily_pnl.abs().sum()
    if total == 0:
        return 1.0
    return daily_pnl.abs().max() / total


def run_sector_analysis(sector, quintile_df, fwd_returns_dict, horizon=5):
    """Run full analysis for one sector at one horizon."""
    fwd = fwd_returns_dict[horizon]

    if sector not in fwd.columns:
        return None

    fwd_sector = fwd[sector]

    # Align quintile assignments with forward returns
    aligned = quintile_df[["quintile", "regime"]].copy()
    aligned["fwd_ret"] = fwd_sector
    aligned = aligned.dropna()

    if len(aligned) < GATE_MIN_TRADES:
        return None

    # ── Quintile analysis ──
    quintile_stats = {}
    for q in range(1, N_QUINTILES + 1):
        qdata = aligned[aligned["quintile"] == q]["fwd_ret"]
        quintile_stats[q] = {
            "mean_ret": float(qdata.mean()) if len(qdata) > 0 else 0.0,
            "count": int(len(qdata)),
            "std": float(qdata.std()) if len(qdata) > 1 else 0.0,
        }

    # ── Long-short strategy: long Q5, short Q1 ──
    q5_days = aligned[aligned["quintile"] == 5].index
    q1_days = aligned[aligned["quintile"] == 1].index

    # Build daily P&L series
    daily_pnl = pd.Series(0.0, index=aligned.index, dtype=float)
    # Long on Q5 signal days
    for d in q5_days:
        r = aligned.loc[d, "fwd_ret"]
        daily_pnl.loc[d] += r - COST_RT_PCT  # long, pay cost
    # Short on Q1 signal days
    for d in q1_days:
        r = aligned.loc[d, "fwd_ret"]
        daily_pnl.loc[d] += -r - COST_RT_PCT  # short, pay cost

    # Only keep days with actual trades
    trade_days = daily_pnl[daily_pnl != 0.0]
    n_trades = len(trade_days)

    if n_trades < GATE_MIN_TRADES:
        return None

    # ── Performance metrics ──
    sharpe = compute_sharpe(trade_days)
    cum_ret = (1 + trade_days).cumprod()
    max_dd = compute_max_drawdown(cum_ret)
    day_conc = compute_day_concentration(trade_days)
    total_ret = float(cum_ret.iloc[-1] - 1) if len(cum_ret) > 0 else 0.0
    win_rate = float((trade_days > 0).mean())

    # ── Permutation test ──
    p_value, observed_spread = permutation_test(
        aligned["quintile"], aligned["fwd_ret"]
    )

    # ── Regime stratification ──
    green_days = trade_days[aligned.loc[trade_days.index, "regime"] == "green"]
    red_days = trade_days[aligned.loc[trade_days.index, "regime"] == "red"]

    sharpe_green = compute_sharpe(green_days) if len(green_days) > 10 else 0.0
    sharpe_red = compute_sharpe(red_days) if len(red_days) > 10 else 0.0

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0.0

    # ── 5-gate validation ──
    gates = {
        "G1_sharpe": sharpe > GATE_SHARPE,
        "G2_pvalue": p_value < GATE_PVALUE,
        "G3_regime_gap": regime_gap < GATE_REGIME_GAP,
        "G4_trade_count": n_trades > GATE_MIN_TRADES,
        "G5_max_dd": max_dd < GATE_MAX_DD,
    }
    all_pass = all(gates.values())

    return {
        "sector": sector,
        "horizon": horizon,
        "n_trades": n_trades,
        "quintile_returns": quintile_stats,
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret * 100, 2),
        "win_rate": round(win_rate, 3),
        "max_drawdown": round(max_dd, 3),
        "day_concentration": round(day_conc, 4),
        "p_value": round(p_value, 4),
        "observed_spread_bps": round(observed_spread * 10000, 1),
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 3),
        "gates": gates,
        "overall": "PASS" if all_pass else "FAIL",
    }


def main():
    print("=" * 80)
    print("CREDIT SPREAD VELOCITY BACKTEST")
    print("Signal: HYG-TLT return differential (3d rolling sum)")
    print("Method: Sliding window quintile sort, 252d lookback")
    print("=" * 80)

    # ── Download data ──
    close = download_data(years=5)

    # Check we have required tickers
    for t in ["HYG", "TLT", "SPY"]:
        if t not in close.columns:
            print(f"FATAL: Missing {t} data. Aborting.")
            return

    # ── Compute signals ──
    print("\nComputing credit spread velocity...")
    csv_daily, csv_3d, csv_10d = compute_credit_spread_velocity(close)

    # ── Compute forward returns ──
    print("Computing forward returns...")
    fwd_returns = compute_forward_returns(close, FORWARD_HORIZONS)

    # ── Sliding window quintile assignment ──
    print("Running sliding window quintile backtest (this may take a moment)...")
    quintile_df = sliding_quintile_backtest(csv_3d, fwd_returns, close["SPY"])
    print(f"  Generated {len(quintile_df)} signal days")

    # ── Per-sector analysis ──
    print("\n" + "=" * 80)
    print("PER-SECTOR RESULTS (5-day horizon, 3d credit spread velocity)")
    print("=" * 80)

    all_results = {}
    pass_count = 0

    for sector in SECTOR_ETFS:
        if sector not in close.columns:
            print(f"\n{sector}: SKIPPED (no data)")
            continue

        result = run_sector_analysis(sector, quintile_df, fwd_returns, horizon=5)

        if result is None:
            print(f"\n{sector}: SKIPPED (insufficient trades)")
            continue

        all_results[sector] = result

        is_sensitive = sector in RATE_SENSITIVE
        tag = " [RATE-SENSITIVE]" if is_sensitive else ""

        print(f"\n{'─' * 60}")
        print(f"{sector}{tag}")
        print(f"{'─' * 60}")
        print(f"  Trades: {result['n_trades']}  |  Sharpe: {result['sharpe']:.3f}  |  "
              f"WR: {result['win_rate']:.1%}  |  Return: {result['total_return_pct']:.1f}%")
        print(f"  MaxDD: {result['max_drawdown']:.1%}  |  DayConc: {result['day_concentration']:.3f}")
        print(f"  p-value: {result['p_value']:.4f}  |  Q5-Q1 spread: {result['observed_spread_bps']:.1f} bps")
        print(f"  Regime: green={result['sharpe_green']:.3f}  red={result['sharpe_red']:.3f}  gap={result['regime_gap']:.3f}")

        # Quintile breakdown
        qr = result["quintile_returns"]
        q_line = "  Quintiles (mean ret bps): "
        for q in range(1, 6):
            q_line += f"Q{q}={qr[q]['mean_ret']*10000:+.1f}  "
        print(q_line)

        # Gates
        gates = result["gates"]
        gate_str = "  Gates: "
        for g, v in gates.items():
            gate_str += f"{g}={'PASS' if v else 'FAIL'}  "
        print(gate_str)
        print(f"  >>> OVERALL: {result['overall']}")

        if result["overall"] == "PASS":
            pass_count += 1

    # ── Also test 1d and 3d horizons for rate-sensitive sectors ──
    print("\n" + "=" * 80)
    print("RATE-SENSITIVE SECTORS — MULTI-HORIZON ANALYSIS")
    print("=" * 80)

    multi_horizon_results = {}
    for sector in RATE_SENSITIVE:
        if sector not in close.columns:
            continue
        multi_horizon_results[sector] = {}
        print(f"\n{sector}:")
        for h in FORWARD_HORIZONS:
            result = run_sector_analysis(sector, quintile_df, fwd_returns, horizon=h)
            if result:
                multi_horizon_results[sector][h] = result
                print(f"  {h}d: Sharpe={result['sharpe']:.3f}  spread={result['observed_spread_bps']:.1f}bps  "
                      f"p={result['p_value']:.4f}  {result['overall']}")
            else:
                print(f"  {h}d: insufficient data")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not all_results:
        print("No sectors had sufficient data for analysis.")
        return

    # Sort by Sharpe
    sorted_sectors = sorted(all_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    print(f"\nTotal sectors analyzed: {len(all_results)}")
    print(f"Sectors passing all 5 gates: {pass_count}")

    print("\nRanked by Sharpe (5d horizon):")
    print(f"  {'Sector':<8} {'Sharpe':>8} {'p-val':>8} {'RegGap':>8} {'Spread':>10} {'Result':>8}")
    print(f"  {'─'*54}")
    for sector, r in sorted_sectors:
        sensitive = "*" if sector in RATE_SENSITIVE else " "
        print(f"  {sector:<7}{sensitive} {r['sharpe']:>8.3f} {r['p_value']:>8.4f} "
              f"{r['regime_gap']:>8.3f} {r['observed_spread_bps']:>8.1f}bps {r['overall']:>8}")

    print("\n  * = rate-sensitive sector")

    # Key findings
    sig_sectors = [s for s, r in all_results.items() if r["p_value"] < 0.05]
    print(f"\nStatistically significant (p<0.05): {sig_sectors if sig_sectors else 'NONE'}")

    pass_sectors = [s for s, r in all_results.items() if r["overall"] == "PASS"]
    print(f"Full 5-gate PASS: {pass_sectors if pass_sectors else 'NONE'}")

    # Hypothesis check
    print("\n--- HYPOTHESIS CHECK ---")
    print("H: Bottom quintile (fast credit widening) -> XLF, XLRE, XLY underperform")
    for s in RATE_SENSITIVE:
        if s in all_results:
            qr = all_results[s]["quintile_returns"]
            q1_ret = qr[1]["mean_ret"] * 10000
            q5_ret = qr[5]["mean_ret"] * 10000
            direction = "CONFIRMED" if q1_ret < q5_ret else "REJECTED"
            print(f"  {s}: Q1={q1_ret:+.1f}bps Q5={q5_ret:+.1f}bps -> {direction}")

    # ── Save results ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "lookback": LOOKBACK,
            "csv_window": CSV3_WINDOW,
            "hold_period": HOLD_PERIOD,
            "cost_rt_pct": COST_RT_PCT,
            "n_permutations": N_PERMUTATIONS,
            "data_years": 5,
            "method": "sliding_window_quintile_sort",
        },
        "per_sector_5d": {k: v for k, v in all_results.items()},
        "multi_horizon_rate_sensitive": {
            sector: {str(h): r for h, r in horizons.items()}
            for sector, horizons in multi_horizon_results.items()
        },
        "summary": {
            "total_analyzed": len(all_results),
            "pass_5gate": pass_sectors,
            "significant_p05": sig_sectors,
            "pass_count": pass_count,
        }
    }

    # Deep-convert numpy types and integer keys for JSON serialization
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    output = sanitize(output)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
