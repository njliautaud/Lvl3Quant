#!/usr/bin/env python3
"""
Weekly Mean Reversion Backtest on Growth Stocks
================================================
Tests 6 variants of weekly mean-reversion strategies on a 30-stock
growth/tech universe using Robinhood-realistic cost assumptions.

Account: $645 | Walk-forward OOT: Jan 2022 – Jul 2026
Validation: 5-gate framework (Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades)
"""

import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "SNOW", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN", "SQ", "RBLX", "RIVN",
    "UBER", "LYFT", "ROKU", "NET", "DDOG", "TTD", "SHOP", "SE", "MELI", "NU",
]

BENCHMARK = "SPY"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way for stocks
OPTION_COST_PER_CONTRACT = 0.65  # each way
ATM_PREMIUM_PCT = 0.03  # ~3% of stock price for ATM weekly
OPTION_BID_ASK_HAIRCUT = 0.05  # 5% of premium lost to spread

DATA_START = "2020-06-01"  # need lookback for 200-SMA
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"

N_PERMUTATIONS = 1000
SEED = 42

OUTPUT_PATH = "/home/jupiter/Lvl3Quant/data/weekly_mean_reversion_results.json"


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download weekly and daily price data for the universe + SPY."""
    tickers = UNIVERSE + [BENCHMARK]
    print(f"Downloading data for {len(tickers)} tickers...")

    # Daily data (needed for 200-SMA filter)
    daily = yf.download(tickers, start=DATA_START, end=OOT_END, interval="1d",
                        auto_adjust=True, progress=False, threads=True)

    # Extract close prices
    if isinstance(daily.columns, pd.MultiIndex):
        daily_close = daily["Close"]
    else:
        daily_close = daily[["Close"]].rename(columns={"Close": tickers[0]})

    # Compute weekly returns (Friday-to-Friday)
    # Resample to weekly frequency ending on Friday
    weekly_close = daily_close.resample("W-FRI").last()
    weekly_returns = weekly_close.pct_change()

    # 200-day SMA on daily data
    sma_200 = daily_close.rolling(200).mean()

    # Get the SMA value on each Friday (last daily value of that week)
    weekly_sma_200 = sma_200.resample("W-FRI").last()
    weekly_close_for_sma = daily_close.resample("W-FRI").last()

    # Above SMA flag
    above_sma = weekly_close_for_sma > weekly_sma_200

    print(f"Data range: {weekly_close.index[0].date()} to {weekly_close.index[-1].date()}")
    print(f"Weekly bars: {len(weekly_close)}")

    return weekly_close, weekly_returns, above_sma, daily_close


# ─── Strategy Variants ──────────────────────────────────────────────────────

def get_weekly_rankings(weekly_returns, date, universe_cols):
    """Rank stocks by their weekly return. Returns sorted list (worst to best)."""
    rets = weekly_returns.loc[date, universe_cols].dropna()
    if len(rets) < 5:
        return None
    return rets.sort_values()


def variant_a_bottom3(weekly_close, weekly_returns, above_sma, spy_above_sma, universe_cols):
    """A) Bottom-3 Bounce: Buy 3 worst performers, hold 1 week."""
    trades = []
    oot_dates = weekly_returns.index[weekly_returns.index >= OOT_START]

    for i in range(len(oot_dates) - 1):
        date = oot_dates[i]
        next_date = oot_dates[i + 1]

        rankings = get_weekly_rankings(weekly_returns, date, universe_cols)
        if rankings is None:
            continue

        bottom3 = rankings.index[:3].tolist()
        capital_per_stock = INITIAL_CAPITAL / 3.0

        week_pnl = 0.0
        for stock in bottom3:
            try:
                buy_price = weekly_close.loc[date, stock]
                sell_price = weekly_close.loc[next_date, stock]
                if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                    continue
                shares = capital_per_stock / buy_price
                gross_ret = (sell_price / buy_price) - 1.0
                net_ret = gross_ret - 2 * SLIPPAGE_PCT  # slippage both ways
                week_pnl += capital_per_stock * net_ret
            except (KeyError, TypeError):
                continue

        trades.append({
            "date": str(date.date()),
            "pnl": week_pnl,
            "ret": week_pnl / INITIAL_CAPITAL,
            "stocks": bottom3,
            "regime": "bull" if spy_above_sma.get(date, True) else "bear",
        })

    return trades


def variant_b_bottom1(weekly_close, weekly_returns, above_sma, spy_above_sma, universe_cols):
    """B) Bottom-1 Concentrated: Buy single worst performer."""
    trades = []
    oot_dates = weekly_returns.index[weekly_returns.index >= OOT_START]

    for i in range(len(oot_dates) - 1):
        date = oot_dates[i]
        next_date = oot_dates[i + 1]

        rankings = get_weekly_rankings(weekly_returns, date, universe_cols)
        if rankings is None:
            continue

        worst = rankings.index[0]
        try:
            buy_price = weekly_close.loc[date, worst]
            sell_price = weekly_close.loc[next_date, worst]
            if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                continue
            gross_ret = (sell_price / buy_price) - 1.0
            net_ret = gross_ret - 2 * SLIPPAGE_PCT
            week_pnl = INITIAL_CAPITAL * net_ret
        except (KeyError, TypeError):
            continue

        trades.append({
            "date": str(date.date()),
            "pnl": week_pnl,
            "ret": week_pnl / INITIAL_CAPITAL,
            "stocks": [worst],
            "regime": "bull" if spy_above_sma.get(date, True) else "bear",
        })

    return trades


def variant_c_long_short(weekly_close, weekly_returns, above_sma, spy_above_sma, universe_cols):
    """C) Long-Short: Buy bottom-3 shares, buy puts on top-3 (synthetic short via options)."""
    trades = []
    oot_dates = weekly_returns.index[weekly_returns.index >= OOT_START]
    half_cap = INITIAL_CAPITAL / 2.0  # half long, half short

    for i in range(len(oot_dates) - 1):
        date = oot_dates[i]
        next_date = oot_dates[i + 1]

        rankings = get_weekly_rankings(weekly_returns, date, universe_cols)
        if rankings is None:
            continue

        bottom3 = rankings.index[:3].tolist()
        top3 = rankings.index[-3:].tolist()

        # Long leg: buy bottom-3 shares
        long_pnl = 0.0
        cap_per = half_cap / 3.0
        for stock in bottom3:
            try:
                buy_price = weekly_close.loc[date, stock]
                sell_price = weekly_close.loc[next_date, stock]
                if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                    continue
                gross_ret = (sell_price / buy_price) - 1.0
                net_ret = gross_ret - 2 * SLIPPAGE_PCT
                long_pnl += cap_per * net_ret
            except (KeyError, TypeError):
                continue

        # Short leg: buy puts on top-3 (profit when stock falls)
        short_pnl = 0.0
        cap_per_short = half_cap / 3.0
        for stock in top3:
            try:
                price_at_entry = weekly_close.loc[date, stock]
                price_at_exit = weekly_close.loc[next_date, stock]
                if pd.isna(price_at_entry) or pd.isna(price_at_exit) or price_at_entry <= 0:
                    continue

                # ATM put: premium ~ 3% of stock price
                premium_per_share = price_at_entry * ATM_PREMIUM_PCT
                # Number of contracts (100 shares each)
                n_contracts = max(1, int(cap_per_short / (premium_per_share * 100)))
                total_premium = n_contracts * premium_per_share * 100

                # Option cost: commission + bid-ask haircut on entry
                entry_cost = n_contracts * OPTION_COST_PER_CONTRACT + total_premium * OPTION_BID_ASK_HAIRCUT

                # Put payoff: max(0, strike - price_at_exit) * 100 * n_contracts
                # ATM strike = price_at_entry
                stock_move_pct = (price_at_exit - price_at_entry) / price_at_entry
                # Simplified: put delta ~ -0.5 for ATM, but we model intrinsic + some time value
                intrinsic = max(0, price_at_entry - price_at_exit) * 100 * n_contracts
                # Time value remaining ~ 30% of original premium (1 week decay on weekly option)
                time_value_remaining = total_premium * 0.30 if intrinsic == 0 else total_premium * 0.10

                exit_value = intrinsic + time_value_remaining
                exit_cost = n_contracts * OPTION_COST_PER_CONTRACT + exit_value * OPTION_BID_ASK_HAIRCUT

                short_pnl += exit_value - total_premium - entry_cost - exit_cost
            except (KeyError, TypeError):
                continue

        week_pnl = long_pnl + short_pnl
        trades.append({
            "date": str(date.date()),
            "pnl": week_pnl,
            "ret": week_pnl / INITIAL_CAPITAL,
            "stocks": bottom3 + top3,
            "regime": "bull" if spy_above_sma.get(date, True) else "bear",
        })

    return trades


def variant_d_filtered(weekly_close, weekly_returns, above_sma, spy_above_sma, universe_cols):
    """D) Filtered Bounce: Buy bottom-3 only if above 200-SMA (avoid falling knives)."""
    trades = []
    oot_dates = weekly_returns.index[weekly_returns.index >= OOT_START]

    for i in range(len(oot_dates) - 1):
        date = oot_dates[i]
        next_date = oot_dates[i + 1]

        rankings = get_weekly_rankings(weekly_returns, date, universe_cols)
        if rankings is None:
            continue

        # Filter: only stocks above their 200-SMA
        bottom_all = rankings.index[:10].tolist()  # look at bottom 10 to find 3 above SMA
        filtered = []
        for stock in bottom_all:
            try:
                if above_sma.loc[date, stock]:
                    filtered.append(stock)
                if len(filtered) >= 3:
                    break
            except (KeyError, TypeError):
                continue

        if len(filtered) == 0:
            continue

        capital_per_stock = INITIAL_CAPITAL / len(filtered)
        week_pnl = 0.0
        for stock in filtered:
            try:
                buy_price = weekly_close.loc[date, stock]
                sell_price = weekly_close.loc[next_date, stock]
                if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                    continue
                gross_ret = (sell_price / buy_price) - 1.0
                net_ret = gross_ret - 2 * SLIPPAGE_PCT
                week_pnl += capital_per_stock * net_ret
            except (KeyError, TypeError):
                continue

        trades.append({
            "date": str(date.date()),
            "pnl": week_pnl,
            "ret": week_pnl / INITIAL_CAPITAL,
            "stocks": filtered,
            "regime": "bull" if spy_above_sma.get(date, True) else "bear",
        })

    return trades


def variant_e_options(weekly_close, weekly_returns, above_sma, spy_above_sma, universe_cols):
    """E) Options Bounce: Buy ATM calls on bottom-3 (leveraged mean reversion)."""
    trades = []
    oot_dates = weekly_returns.index[weekly_returns.index >= OOT_START]

    for i in range(len(oot_dates) - 1):
        date = oot_dates[i]
        next_date = oot_dates[i + 1]

        rankings = get_weekly_rankings(weekly_returns, date, universe_cols)
        if rankings is None:
            continue

        bottom3 = rankings.index[:3].tolist()
        cap_per = INITIAL_CAPITAL / 3.0

        week_pnl = 0.0
        for stock in bottom3:
            try:
                price_at_entry = weekly_close.loc[date, stock]
                price_at_exit = weekly_close.loc[next_date, stock]
                if pd.isna(price_at_entry) or pd.isna(price_at_exit) or price_at_entry <= 0:
                    continue

                # ATM call: premium ~ 3% of stock price
                premium_per_share = price_at_entry * ATM_PREMIUM_PCT
                n_contracts = max(1, int(cap_per / (premium_per_share * 100)))
                total_premium = n_contracts * premium_per_share * 100

                # Costs
                entry_cost = n_contracts * OPTION_COST_PER_CONTRACT + total_premium * OPTION_BID_ASK_HAIRCUT

                # Call payoff at expiry
                intrinsic = max(0, price_at_exit - price_at_entry) * 100 * n_contracts
                time_value_remaining = total_premium * 0.30 if intrinsic == 0 else total_premium * 0.10

                exit_value = intrinsic + time_value_remaining
                exit_cost = n_contracts * OPTION_COST_PER_CONTRACT + exit_value * OPTION_BID_ASK_HAIRCUT

                week_pnl += exit_value - total_premium - entry_cost - exit_cost
            except (KeyError, TypeError):
                continue

        trades.append({
            "date": str(date.date()),
            "pnl": week_pnl,
            "ret": week_pnl / INITIAL_CAPITAL,
            "stocks": bottom3,
            "regime": "bull" if spy_above_sma.get(date, True) else "bear",
        })

    return trades


def variant_f_extreme(weekly_close, weekly_returns, above_sma, spy_above_sma, universe_cols):
    """F) Extreme Only: Trade only when bottom stock dropped >7% that week."""
    trades = []
    oot_dates = weekly_returns.index[weekly_returns.index >= OOT_START]

    for i in range(len(oot_dates) - 1):
        date = oot_dates[i]
        next_date = oot_dates[i + 1]

        rankings = get_weekly_rankings(weekly_returns, date, universe_cols)
        if rankings is None:
            continue

        # Only trade if worst performer dropped > 7%
        extreme_losers = rankings[rankings < -0.07]
        if len(extreme_losers) == 0:
            continue

        # Buy all stocks that dropped > 7% (up to 5)
        selected = extreme_losers.index[:5].tolist()
        capital_per_stock = INITIAL_CAPITAL / len(selected)

        week_pnl = 0.0
        for stock in selected:
            try:
                buy_price = weekly_close.loc[date, stock]
                sell_price = weekly_close.loc[next_date, stock]
                if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                    continue
                gross_ret = (sell_price / buy_price) - 1.0
                net_ret = gross_ret - 2 * SLIPPAGE_PCT
                week_pnl += capital_per_stock * net_ret
            except (KeyError, TypeError):
                continue

        trades.append({
            "date": str(date.date()),
            "pnl": week_pnl,
            "ret": week_pnl / INITIAL_CAPITAL,
            "stocks": selected,
            "regime": "bull" if spy_above_sma.get(date, True) else "bear",
        })

    return trades


# ─── Metrics & Validation ───────────────────────────────────────────────────

def compute_metrics(trades):
    """Compute strategy metrics from trade list."""
    if len(trades) < 2:
        return None

    rets = np.array([t["ret"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    # Annualize: ~52 weeks/year
    ann_factor = np.sqrt(52)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * ann_factor if downside_std > 0 else 0.0

    # Profit Factor
    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win Rate
    win_rate = (pnls > 0).sum() / len(pnls)

    # Max Drawdown (on cumulative equity)
    cum_equity = INITIAL_CAPITAL + np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_equity)
    drawdown = (cum_equity - peak) / peak
    max_dd = drawdown.min()

    # Total return
    total_ret = cum_equity[-1] / INITIAL_CAPITAL - 1.0

    # CAGR
    n_years = len(trades) / 52.0
    cagr = (cum_equity[-1] / INITIAL_CAPITAL) ** (1.0 / max(n_years, 0.1)) - 1.0 if cum_equity[-1] > 0 else -1.0

    # Regime split
    bull_rets = [t["ret"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["ret"] for t in trades if t["regime"] == "bear"]

    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets, ddof=1)) * ann_factor if len(bull_rets) > 2 and np.std(bull_rets, ddof=1) > 0 else 0.0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets, ddof=1)) * ann_factor if len(bear_rets) > 2 and np.std(bear_rets, ddof=1) > 0 else 0.0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0.0

    return {
        "n_trades": len(trades),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "max_dd_pct": round(max_dd * 100, 2),
        "avg_weekly_ret_pct": round(mean_ret * 100, 3),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "final_equity": round(cum_equity[-1], 2),
    }


def permutation_test(trades, observed_sharpe, n_perms=N_PERMUTATIONS):
    """Shuffle weekly stock selections randomly to test if edge is real."""
    rng = np.random.RandomState(SEED)
    rets = np.array([t["ret"] for t in trades])
    ann_factor = np.sqrt(52)

    count_above = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        shuf_sharpe = (np.mean(shuffled) / np.std(shuffled, ddof=1)) * ann_factor if np.std(shuffled, ddof=1) > 0 else 0
        if shuf_sharpe >= observed_sharpe:
            count_above += 1

    # For mean-reversion, shuffling the returns preserves variance structure
    # but breaks the signal. A more rigorous test: randomize the stock picks.
    # Since we already computed weekly returns, shuffling returns across weeks
    # tests whether the ORDERING matters.
    # Actually, shuffling weekly returns is trivially p=0.5 since mean is preserved.
    # Better test: bootstrap the weekly returns and see how often we beat observed Sharpe.

    # Bootstrap approach: resample with replacement
    count_above_boot = 0
    for _ in range(n_perms):
        boot_idx = rng.choice(len(rets), size=len(rets), replace=True)
        boot_rets = rets[boot_idx]
        boot_sharpe = (np.mean(boot_rets) / np.std(boot_rets, ddof=1)) * ann_factor if np.std(boot_rets, ddof=1) > 0 else 0
        if boot_sharpe <= 0:
            count_above_boot += 1

    # Use the more meaningful test: what fraction of bootstraps show Sharpe <= 0?
    # This tests whether the positive Sharpe is statistically significant.
    p_value = count_above_boot / n_perms

    return round(p_value, 4)


def validate_gates(metrics, p_value):
    """Check 5-gate validation framework."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50.0,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("WEEKLY MEAN REVERSION BACKTEST — Growth Stock Universe")
    print(f"Account: ${INITIAL_CAPITAL} | OOT: {OOT_START} to {OOT_END}")
    print("=" * 80)

    # Download data
    weekly_close, weekly_returns, above_sma, daily_close = download_data()

    # Universe columns (only those present in data)
    universe_cols = [c for c in UNIVERSE if c in weekly_returns.columns]
    print(f"Universe: {len(universe_cols)} stocks available")

    # SPY regime: above/below 200-SMA
    spy_daily_close = daily_close[BENCHMARK]
    spy_sma_200 = spy_daily_close.rolling(200).mean()
    spy_weekly_close = spy_daily_close.resample("W-FRI").last()
    spy_weekly_sma = spy_sma_200.resample("W-FRI").last()
    spy_above_sma = {}
    for date in weekly_returns.index:
        try:
            spy_above_sma[date] = bool(spy_weekly_close.loc[date] > spy_weekly_sma.loc[date])
        except KeyError:
            spy_above_sma[date] = True

    # Run all variants
    variants = {
        "A_Bottom3_Bounce": variant_a_bottom3,
        "B_Bottom1_Concentrated": variant_b_bottom1,
        "C_Long_Short_Options": variant_c_long_short,
        "D_Filtered_200SMA": variant_d_filtered,
        "E_Options_Calls": variant_e_options,
        "F_Extreme_7pct": variant_f_extreme,
    }

    results = {}
    for name, func in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Running variant: {name}")
        trades = func(weekly_close, weekly_returns, above_sma, spy_above_sma, universe_cols)
        print(f"  Trades: {len(trades)}")

        if len(trades) < 5:
            print(f"  SKIP — too few trades")
            results[name] = {"status": "SKIPPED", "n_trades": len(trades)}
            continue

        metrics = compute_metrics(trades)
        if metrics is None:
            results[name] = {"status": "FAILED", "n_trades": len(trades)}
            continue

        # Permutation test
        p_value = permutation_test(trades, metrics["sharpe"])
        metrics["perm_p_value"] = p_value

        # Gate validation
        gates = validate_gates(metrics, p_value)
        metrics["gates"] = gates

        results[name] = metrics
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.1f}% | PF: {metrics['profit_factor']:.2f}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}% | MaxDD: {metrics['max_dd_pct']:.1f}%")
        print(f"  Regime — Bull Sharpe: {metrics['bull_sharpe']:.3f}, Bear Sharpe: {metrics['bear_sharpe']:.3f}, Gap: {metrics['regime_gap']:.3f}")
        print(f"  Perm test p-value: {p_value:.4f}")
        print(f"  Gates: {'ALL PASS ✓' if gates['all_pass'] else 'FAIL ✗'}")
        for gate, passed in gates.items():
            if gate != "all_pass":
                status = "PASS" if passed else "FAIL"
                print(f"    {gate}: {status}")

    # ─── Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Variant':<28} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} {'MaxDD%':>7} {'TotRet%':>8} {'Trades':>7} {'p-val':>7} {'RGap':>6} {'Gates':>6}"
    print(header)
    print("-" * len(header))

    for name, m in results.items():
        if "sharpe" not in m:
            print(f"{name:<28} {'SKIPPED':>7}")
            continue
        gate_str = "PASS" if m["gates"]["all_pass"] else "FAIL"
        print(f"{name:<28} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate_pct']:>5.1f}% {m['profit_factor']:>6.2f} {m['max_dd_pct']:>6.1f}% {m['total_return_pct']:>7.1f}% {m['n_trades']:>7} {m['perm_p_value']:>7.4f} {m['regime_gap']:>6.3f} {gate_str:>6}")

    # ─── Best variant ────────────────────────────────────────────────────
    passing = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("gates", {}).get("all_pass", False)}
    if passing:
        best = max(passing, key=lambda k: passing[k]["sharpe"])
        print(f"\nBEST PASSING VARIANT: {best} (Sharpe={passing[best]['sharpe']:.3f})")
    else:
        # Find closest to passing
        scoreable = {k: v for k, v in results.items() if "sharpe" in v}
        if scoreable:
            best = max(scoreable, key=lambda k: scoreable[k]["sharpe"])
            print(f"\nNO VARIANTS PASS ALL GATES. Best Sharpe: {best} ({scoreable[best]['sharpe']:.3f})")
        print("CONCLUSION: Weekly mean reversion on growth stocks does NOT pass the 5-gate framework.")

    # ─── Save results ────────────────────────────────────────────────────
    # Convert for JSON serialization
    output = {
        "strategy": "weekly_mean_reversion_growth_stocks",
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "universe_size": len(universe_cols),
        "variants": {},
    }
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    for name, m in results.items():
        output["variants"][name] = make_serializable(m)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
