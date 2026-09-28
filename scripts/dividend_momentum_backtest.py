#!/usr/bin/env python3
"""
Dividend Aristocrat Momentum Rotation Backtest
===============================================
Academic basis: Dividend growth premium + momentum rotation.
Universe: ~40 well-known dividend aristocrats / high-yield large-caps.
Variants A-E with 5-gate validation.

Author: Claude Opus 4.6
Date: 2026-07-29
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

UNIVERSE = [
    "JNJ", "PG", "KO", "PEP", "MCD", "WMT", "HD", "ABBV", "ABT", "MMM",
    "CAT", "EMR", "SHW", "GPC", "ITW", "ADP", "CTAS", "BDX", "CLX", "HRL",
    "T", "VZ", "XOM", "CVX", "IBM", "TGT", "LOW", "SYY", "ADM", "AFL",
    "ED", "NEE", "DUK", "SO", "O", "SCHW", "BLK", "APD", "LIN", "ECL",
]

BENCHMARK = "SPY"
START_DATE = "2020-07-01"  # extra history for lookbacks
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
INITIAL_CAPITAL = 10_000.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade (one-way)
COMMISSION = 0.0  # $0 commission
RISK_FREE_RATE = 0.04  # ~4% for Sharpe/Sortino (T-bill proxy)
SMA_PERIOD = 200  # SPY 200-SMA for regime hedge

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/dividend_momentum_results.json")

# Variant definitions
VARIANTS = {
    "A": {"top_n": 5, "mom_months": 3, "label": "Top 5, 3-mo momentum"},
    "B": {"top_n": 5, "mom_months": 6, "label": "Top 5, 6-mo momentum"},
    "C": {"top_n": 3, "mom_months": 3, "label": "Top 3, 3-mo momentum (concentrated)"},
    "D": {"top_n": 5, "mom_months": 1, "label": "Top 5, 1-mo momentum (fast rotation)"},
    "E": {"top_n": 5, "mom_months": 3, "label": "Top 5, 3-mo mom + relative strength vs SPY"},
}


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download adjusted close prices for universe + SPY."""
    tickers = UNIVERSE + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Drop tickers with too much missing data (>20%)
    missing_pct = prices.isnull().mean()
    valid = missing_pct[missing_pct < 0.2].index.tolist()
    dropped = [t for t in tickers if t not in valid]
    if dropped:
        print(f"  Dropped (>20% missing): {dropped}")
    prices = prices[valid].ffill().bfill()

    print(f"  Got {len(valid)-1} stocks + SPY, {len(prices)} trading days")
    return prices


# ─── Momentum Calculation ────────────────────────────────────────────────────

def calc_momentum(prices, months, tickers):
    """Calculate N-month total return momentum for each stock."""
    lookback = months * 21  # ~21 trading days per month
    mom = prices[tickers].pct_change(lookback)
    return mom


def calc_relative_strength_vs_spy(prices, months, tickers):
    """Calculate relative strength vs SPY (stock return - SPY return)."""
    lookback = months * 21
    stock_ret = prices[tickers].pct_change(lookback)
    spy_ret = prices[BENCHMARK].pct_change(lookback)
    return stock_ret.sub(spy_ret, axis=0)


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(prices, variant_key):
    """Run a single variant backtest. Returns daily equity curve + stats."""
    cfg = VARIANTS[variant_key]
    top_n = cfg["top_n"]
    mom_months = cfg["mom_months"]
    is_relative = (variant_key == "E")

    stock_tickers = [t for t in UNIVERSE if t in prices.columns]
    spy = prices[BENCHMARK]
    spy_sma200 = spy.rolling(SMA_PERIOD).mean()

    # Get rebalance dates (first trading day of each month in OOT period)
    oot_prices = prices.loc[OOT_START:OOT_END]
    monthly_groups = oot_prices.groupby([oot_prices.index.year, oot_prices.index.month])
    rebalance_dates = [group.index[0] for _, group in monthly_groups]

    # Calculate momentum series (need full history for lookback)
    if is_relative:
        mom = calc_relative_strength_vs_spy(prices, mom_months, stock_tickers)
        # Additional filter: only buy if outperforming SPY (relative strength > 0)
        mom_rank = mom.copy()
    else:
        mom = calc_momentum(prices, mom_months, stock_tickers)
        mom_rank = mom.copy()

    # Daily equity tracking
    oot_dates = oot_prices.index
    equity = pd.Series(index=oot_dates, dtype=float)
    equity.iloc[0] = INITIAL_CAPITAL

    holdings = {}  # ticker -> shares
    cash = INITIAL_CAPITAL
    prev_holdings = {}
    trade_count = 0

    for i, date in enumerate(oot_dates):
        # Check if rebalance day
        if date in rebalance_dates:
            # Get momentum ranking on this date
            today_mom = mom_rank.loc[:date].iloc[-1].dropna()

            if is_relative:
                # Only consider stocks outperforming SPY
                today_mom = today_mom[today_mom > 0]

            if len(today_mom) == 0:
                # No valid stocks, go to cash
                selected = []
            else:
                selected = today_mom.nlargest(top_n).index.tolist()

            # Regime hedge: half-size when SPY < 200-SMA
            regime_scale = 1.0
            if not pd.isna(spy_sma200.loc[:date].iloc[-1]):
                if spy.loc[date] < spy_sma200.loc[:date].iloc[-1]:
                    regime_scale = 0.5

            # Calculate current portfolio value
            port_value = cash
            for ticker, shares in holdings.items():
                port_value += shares * prices.loc[date, ticker]

            # Sell all current holdings
            for ticker, shares in holdings.items():
                sell_price = prices.loc[date, ticker] * (1 - SLIPPAGE_PCT)
                cash += shares * sell_price
                if ticker not in prev_holdings or prev_holdings.get(ticker, 0) != shares:
                    trade_count += 1

            # Buy new holdings
            holdings = {}
            if len(selected) > 0:
                alloc_per_stock = (port_value * regime_scale) / len(selected)
                for ticker in selected:
                    buy_price = prices.loc[date, ticker] * (1 + SLIPPAGE_PCT)
                    shares = int(alloc_per_stock / buy_price)
                    if shares > 0:
                        holdings[ticker] = shares
                        cash -= shares * buy_price
                        trade_count += 1

            prev_holdings = holdings.copy()

        # Calculate daily equity
        port_value = cash
        for ticker, shares in holdings.items():
            port_value += shares * prices.loc[date, ticker]
        equity.iloc[i] = port_value

    return equity, trade_count


# ─── Performance Metrics ─────────────────────────────────────────────────────

def calc_metrics(equity, benchmark_equity, trade_count):
    """Calculate comprehensive performance metrics."""
    daily_ret = equity.pct_change().dropna()
    bench_ret = benchmark_equity.pct_change().dropna()

    # Align
    common = daily_ret.index.intersection(bench_ret.index)
    daily_ret = daily_ret.loc[common]
    bench_ret = bench_ret.loc[common]

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    bench_total = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1
    years = len(daily_ret) / 252

    # Annualized return
    ann_return = (1 + total_return) ** (1 / years) - 1
    bench_ann = (1 + bench_total) ** (1 / years) - 1

    # Volatility
    ann_vol = daily_ret.std() * np.sqrt(252)

    # Sharpe
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = (ann_return - RISK_FREE_RATE) / downside_vol

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Max drawdown duration
    in_dd = equity < peak
    dd_groups = (~in_dd).cumsum()
    dd_durations = in_dd.groupby(dd_groups).sum()
    max_dd_days = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate (monthly)
    monthly_ret = equity.resample("ME").last().pct_change().dropna()
    win_rate = (monthly_ret > 0).mean()

    # Profit factor (monthly)
    wins = monthly_ret[monthly_ret > 0].sum()
    losses = abs(monthly_ret[monthly_ret < 0].sum())
    profit_factor = wins / losses if losses > 0 else float("inf")

    # Beta and Alpha
    if len(daily_ret) > 30:
        beta, alpha_daily, _, _, _ = stats.linregress(bench_ret, daily_ret)
        alpha = alpha_daily * 252
    else:
        beta, alpha = 0, 0

    # Tracking error
    te = (daily_ret - bench_ret).std() * np.sqrt(252)
    info_ratio = (ann_return - bench_ann) / te if te > 0 else 0

    # Turnover estimate (trades per year)
    turnover_pa = trade_count / years if years > 0 else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": round(ann_return * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_dd_duration_days": max_dd_days,
        "calmar": round(calmar, 3),
        "win_rate_monthly_pct": round(win_rate * 100, 1),
        "profit_factor_monthly": round(profit_factor, 3),
        "beta": round(beta, 3),
        "alpha_ann_pct": round(alpha * 100, 2),
        "info_ratio": round(info_ratio, 3),
        "tracking_error_pct": round(te * 100, 2),
        "total_trades": trade_count,
        "turnover_pa": round(turnover_pa, 1),
        "final_equity": round(equity.iloc[-1], 2),
        "benchmark_total_return_pct": round(bench_total * 100, 2),
        "benchmark_ann_return_pct": round(bench_ann * 100, 2),
    }


# ─── 5-Gate Validation ──────────────────────────────────────────────────────

def five_gate_validation(equity, benchmark_equity, metrics):
    """
    5-gate validation:
    G1: Sharpe > 0.3 (risk-adjusted bar)
    G2: Max drawdown < 40%
    G3: Win rate > 45% monthly
    G4: Outperforms benchmark on total return
    G5: Calmar > 0.2 (return/risk efficiency)
    """
    gates = {}

    # G1: Sharpe ratio
    gates["G1_sharpe_gt_0.3"] = {
        "pass": metrics["sharpe"] > 0.3,
        "value": metrics["sharpe"],
        "threshold": 0.3,
    }

    # G2: Max drawdown
    gates["G2_max_dd_lt_40pct"] = {
        "pass": metrics["max_drawdown_pct"] > -40,
        "value": metrics["max_drawdown_pct"],
        "threshold": -40,
    }

    # G3: Win rate
    gates["G3_win_rate_gt_45pct"] = {
        "pass": metrics["win_rate_monthly_pct"] > 45,
        "value": metrics["win_rate_monthly_pct"],
        "threshold": 45,
    }

    # G4: Outperforms benchmark
    gates["G4_beats_benchmark"] = {
        "pass": metrics["total_return_pct"] > metrics["benchmark_total_return_pct"],
        "value": metrics["total_return_pct"],
        "threshold": metrics["benchmark_total_return_pct"],
    }

    # G5: Calmar ratio
    gates["G5_calmar_gt_0.2"] = {
        "pass": metrics["calmar"] > 0.2,
        "value": metrics["calmar"],
        "threshold": 0.2,
    }

    gates["gates_passed"] = sum(1 for g in gates.values() if isinstance(g, dict) and g.get("pass", False))
    gates["total_gates"] = 5
    gates["all_passed"] = gates["gates_passed"] == 5

    return gates


# ─── Yearly Breakdown ────────────────────────────────────────────────────────

def yearly_breakdown(equity, benchmark_equity):
    """Per-year return breakdown."""
    yearly = {}
    for year in sorted(equity.index.year.unique()):
        yr_eq = equity[equity.index.year == year]
        yr_bench = benchmark_equity[benchmark_equity.index.year == year]
        if len(yr_eq) < 2 or len(yr_bench) < 2:
            continue
        strat_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0] - 1) * 100
        bench_ret = (yr_bench.iloc[-1] / yr_bench.iloc[0] - 1) * 100
        yr_daily = yr_eq.pct_change().dropna()
        yr_vol = yr_daily.std() * np.sqrt(252) * 100 if len(yr_daily) > 1 else 0
        peak = yr_eq.cummax()
        yr_dd = ((yr_eq - peak) / peak).min() * 100
        yearly[str(year)] = {
            "strategy_return_pct": round(strat_ret, 2),
            "benchmark_return_pct": round(bench_ret, 2),
            "excess_pct": round(strat_ret - bench_ret, 2),
            "volatility_pct": round(yr_vol, 2),
            "max_dd_pct": round(yr_dd, 2),
        }
    return yearly


# ─── Regime Analysis ─────────────────────────────────────────────────────────

def regime_analysis(equity, spy_prices):
    """Split performance by regime: SPY above/below 200-SMA."""
    spy_sma = spy_prices.rolling(SMA_PERIOD).mean()
    daily_ret = equity.pct_change().dropna()

    # Align
    common = daily_ret.index.intersection(spy_sma.dropna().index)
    daily_ret = daily_ret.loc[common]
    spy_at = spy_prices.loc[common]
    sma_at = spy_sma.loc[common]

    bull = daily_ret[spy_at > sma_at]
    bear = daily_ret[spy_at <= sma_at]

    def _stats(rets, label):
        if len(rets) < 5:
            return {"regime": label, "days": len(rets), "note": "insufficient data"}
        ann_r = rets.mean() * 252
        ann_v = rets.std() * np.sqrt(252)
        sh = (ann_r - RISK_FREE_RATE) / ann_v if ann_v > 0 else 0
        return {
            "regime": label,
            "days": len(rets),
            "ann_return_pct": round(ann_r * 100, 2),
            "ann_vol_pct": round(ann_v * 100, 2),
            "sharpe": round(sh, 3),
            "win_rate_daily_pct": round((rets > 0).mean() * 100, 1),
        }

    return {
        "bull": _stats(bull, "SPY > 200-SMA"),
        "bear": _stats(bear, "SPY <= 200-SMA"),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DIVIDEND ARISTOCRAT MOMENTUM ROTATION BACKTEST")
    print("=" * 70)
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% | Commission: ${COMMISSION}")
    print(f"Regime Hedge: Half-size when SPY < 200-SMA")
    print()

    # Download data
    prices = download_data()
    spy_oot = prices[BENCHMARK].loc[OOT_START:OOT_END]

    # Normalize SPY to same starting capital for comparison
    benchmark_equity = spy_oot / spy_oot.iloc[0] * INITIAL_CAPITAL

    results = {
        "strategy": "Dividend Aristocrat Momentum Rotation",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "universe_size": len([t for t in UNIVERSE if t in prices.columns]),
        "slippage_pct": SLIPPAGE_PCT * 100,
        "commission": COMMISSION,
        "risk_free_rate": RISK_FREE_RATE,
        "regime_hedge": "Half-size when SPY < 200-SMA",
        "run_timestamp": dt.datetime.now().isoformat(),
        "variants": {},
    }

    best_sharpe = -999
    best_variant = None

    for vkey in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vkey]
        print(f"\n{'─'*60}")
        print(f"Variant {vkey}: {cfg['label']}")
        print(f"{'─'*60}")

        equity, trade_count = run_backtest(prices, vkey)
        metrics = calc_metrics(equity, benchmark_equity, trade_count)
        gates = five_gate_validation(equity, benchmark_equity, metrics)
        yearly = yearly_breakdown(equity, benchmark_equity)
        regimes = regime_analysis(equity, prices[BENCHMARK].loc[OOT_START:OOT_END])

        # Print summary
        print(f"  Total Return:  {metrics['total_return_pct']:+.2f}%  (SPY: {metrics['benchmark_total_return_pct']:+.2f}%)")
        print(f"  Ann Return:    {metrics['ann_return_pct']:+.2f}%  (SPY: {metrics['benchmark_ann_return_pct']:+.2f}%)")
        print(f"  Sharpe:        {metrics['sharpe']:.3f}")
        print(f"  Sortino:       {metrics['sortino']:.3f}")
        print(f"  Max DD:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar:        {metrics['calmar']:.3f}")
        print(f"  Win Rate (mo): {metrics['win_rate_monthly_pct']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor_monthly']:.3f}")
        print(f"  Alpha (ann):   {metrics['alpha_ann_pct']:+.2f}%")
        print(f"  Beta:          {metrics['beta']:.3f}")
        print(f"  Info Ratio:    {metrics['info_ratio']:.3f}")
        print(f"  Trades:        {metrics['total_trades']}")
        print(f"  Final Equity:  ${metrics['final_equity']:,.2f}")
        print(f"  Gates Passed:  {gates['gates_passed']}/5 {'PASS' if gates['all_passed'] else 'FAIL'}")

        for gname, gval in gates.items():
            if isinstance(gval, dict) and "pass" in gval:
                status = "PASS" if gval["pass"] else "FAIL"
                print(f"    {gname}: {status} (val={gval['value']}, thresh={gval['threshold']})")

        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_variant = vkey

        results["variants"][vkey] = {
            "config": cfg,
            "metrics": metrics,
            "five_gate_validation": gates,
            "yearly_breakdown": yearly,
            "regime_analysis": regimes,
        }

    # Summary
    results["best_variant"] = best_variant
    results["best_sharpe"] = best_sharpe

    print(f"\n{'='*70}")
    print(f"BEST VARIANT: {best_variant} ({VARIANTS[best_variant]['label']})")
    print(f"  Sharpe: {best_sharpe:.3f}")
    bm = results["variants"][best_variant]["metrics"]
    print(f"  Return: {bm['total_return_pct']:+.2f}% | Sortino: {bm['sortino']:.3f} | MaxDD: {bm['max_drawdown_pct']:.2f}%")
    bg = results["variants"][best_variant]["five_gate_validation"]
    print(f"  Gates: {bg['gates_passed']}/5 {'ALL PASSED' if bg['all_passed'] else 'SOME FAILED'}")
    print(f"{'='*70}")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
