#!/usr/bin/env python3
"""
Day-of-Week and Monthly Seasonality Backtest
=============================================
Tests 6 calendar-anomaly variants on growth stocks and sector ETFs.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

Variants:
  A) Buy-Wednesday-Sell-Friday (QQQ)
  B) First-5-Days of Month (SPY)
  C) Sell-Monday Buy-Tuesday (QQQ)
  D) Month-End Growth Rally (top-3 growth stocks)
  E) Sector Monthly Rotation (XLK, XLE, XLF, SPY)
  F) Avoid-Options-Expiry (QQQ non-OpEx vs OpEx weeks)
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

# ── Constants ────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # extra history for 200-SMA
TICKERS = ["SPY", "QQQ", "NVDA", "TSLA", "AMD", "META", "AMZN", "AAPL", "MSFT", "XLK", "XLE", "XLF"]
GROWTH_TICKERS = ["NVDA", "TSLA", "AMD", "META", "AMZN", "AAPL", "MSFT"]
SECTOR_ETFS = ["XLK", "XLE", "XLF", "SPY"]
PERM_ITERS = 1000
SMA_PERIOD = 200

OUT_PATH = Path("/home/jupiter/Lvl3Quant/data/day_seasonality_results.json")


def download_data():
    """Download OHLC data for all tickers."""
    print("Downloading data via yfinance...")
    data = {}
    for t in TICKERS:
        df = yf.download(t, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[t] = df
    print(f"  Downloaded {len(data)} tickers, SPY rows: {len(data.get('SPY', []))}")
    return data


def compute_regime(spy_close):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma = spy_close.rolling(SMA_PERIOD).mean()
    regime = pd.Series("bull", index=spy_close.index)
    regime[spy_close < sma] = "bear"
    return regime


def apply_slippage(ret):
    """Apply round-trip slippage (buy + sell)."""
    return ret - 2 * SLIPPAGE_PCT


def calc_metrics(trade_rets, regime_labels=None):
    """Calculate strategy metrics from array of per-trade returns."""
    trade_rets = np.array(trade_rets, dtype=float)
    n = len(trade_rets)
    if n < 2:
        return {"n_trades": n, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "max_dd_pct": 0, "total_ret_pct": 0, "mean_ret_pct": 0,
                "regime_gap": 1.0, "bull_sharpe": 0, "bear_sharpe": 0}

    mean_r = np.mean(trade_rets)
    std_r = np.std(trade_rets, ddof=1)
    sharpe = (mean_r / std_r) * np.sqrt(252 / max(1, np.mean([1]))) if std_r > 0 else 0
    # Annualize: assume average ~1 trade per week -> ~52/year, adjust
    trades_per_year = n / max(1, (pd.Timestamp(OOT_END) - pd.Timestamp(OOT_START)).days / 365.25)
    sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0

    downside = trade_rets[trade_rets < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_r / down_std) * np.sqrt(trades_per_year) if down_std > 0 else 0

    gross_wins = np.sum(trade_rets[trade_rets > 0])
    gross_losses = np.abs(np.sum(trade_rets[trade_rets < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else 99.0
    wr = np.sum(trade_rets > 0) / n

    # Max drawdown on equity curve
    equity = CAPITAL * np.cumprod(1 + trade_rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd)

    total_ret = np.prod(1 + trade_rets) - 1

    # Regime gap
    bull_sharpe = 0.0
    bear_sharpe = 0.0
    regime_gap = 0.0
    if regime_labels is not None and len(regime_labels) == n:
        rl = np.array(regime_labels)
        bull_mask = rl == "bull"
        bear_mask = rl == "bear"
        bull_rets = trade_rets[bull_mask]
        bear_rets = trade_rets[bear_mask]
        if len(bull_rets) > 2 and np.std(bull_rets) > 0:
            bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets, ddof=1)) * np.sqrt(trades_per_year)
        if len(bear_rets) > 2 and np.std(bear_rets) > 0:
            bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets, ddof=1)) * np.sqrt(trades_per_year)
        denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
        regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    return {
        "n_trades": int(n),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 4),
        "max_dd_pct": round(float(max_dd) * 100, 2),
        "total_ret_pct": round(float(total_ret) * 100, 2),
        "mean_ret_pct": round(float(mean_r) * 100, 4),
        "regime_gap": round(float(regime_gap), 3),
        "bull_sharpe": round(float(bull_sharpe), 3),
        "bear_sharpe": round(float(bear_sharpe), 3),
    }


def permutation_test(trade_rets, n_iter=PERM_ITERS):
    """Shuffle trade assignment to get null distribution of mean return."""
    observed = np.mean(trade_rets)
    count = 0
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=(n_iter, len(trade_rets)))
    shuffled_means = np.mean(trade_rets * signs, axis=1)
    p_value = np.mean(shuffled_means >= observed)
    return float(p_value)


def gate_check(metrics, p_val):
    """5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_val < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ── Strategy Implementations ─────────────────────────────────────────────

def strategy_a_wed_fri(data, regime):
    """A) Buy QQQ at Wednesday open, sell at Friday close. Every week."""
    qqq = data["QQQ"].loc[OOT_START:OOT_END].copy()
    qqq["dow"] = qqq.index.dayofweek  # Mon=0 .. Fri=4

    trade_rets = []
    regime_labels = []

    # Group by ISO week
    qqq["year_week"] = qqq.index.isocalendar().year.astype(str) + "-" + qqq.index.isocalendar().week.astype(str).str.zfill(2)
    for _, grp in qqq.groupby("year_week"):
        wed = grp[grp["dow"] == 2]
        fri = grp[grp["dow"] == 4]
        if len(wed) == 0 or len(fri) == 0:
            continue
        buy_price = wed.iloc[0]["Open"]
        sell_price = fri.iloc[-1]["Close"]
        ret = apply_slippage((sell_price / buy_price) - 1)
        trade_rets.append(ret)
        regime_labels.append(regime.loc[wed.index[0]] if wed.index[0] in regime.index else "bull")

    return np.array(trade_rets), regime_labels


def strategy_b_first5(data, regime):
    """B) Buy SPY on last trading day of month, sell at close of 5th trading day next month."""
    spy = data["SPY"].loc[OOT_START:OOT_END].copy()
    spy["month"] = spy.index.to_period("M")

    trade_rets = []
    regime_labels = []

    months = spy["month"].unique()
    for i in range(len(months) - 1):
        curr_month = spy[spy["month"] == months[i]]
        next_month = spy[spy["month"] == months[i + 1]]
        if len(curr_month) == 0 or len(next_month) < 5:
            continue
        buy_price = curr_month.iloc[-1]["Close"]  # last day of month at close
        sell_idx = min(4, len(next_month) - 1)  # 5th trading day (0-indexed=4)
        sell_price = next_month.iloc[sell_idx]["Close"]
        ret = apply_slippage((sell_price / buy_price) - 1)
        trade_rets.append(ret)
        regime_labels.append(regime.loc[curr_month.index[-1]] if curr_month.index[-1] in regime.index else "bull")

    return np.array(trade_rets), regime_labels


def strategy_c_skip_monday(data, regime):
    """C) Sell Monday, buy QQQ Tuesday open, sell Friday close."""
    qqq = data["QQQ"].loc[OOT_START:OOT_END].copy()
    qqq["dow"] = qqq.index.dayofweek

    trade_rets = []
    regime_labels = []

    qqq["year_week"] = qqq.index.isocalendar().year.astype(str) + "-" + qqq.index.isocalendar().week.astype(str).str.zfill(2)
    for _, grp in qqq.groupby("year_week"):
        tue = grp[grp["dow"] == 1]  # Tuesday
        fri = grp[grp["dow"] == 4]  # Friday
        if len(tue) == 0 or len(fri) == 0:
            continue
        buy_price = tue.iloc[0]["Open"]
        sell_price = fri.iloc[-1]["Close"]
        ret = apply_slippage((sell_price / buy_price) - 1)
        trade_rets.append(ret)
        regime_labels.append(regime.loc[tue.index[0]] if tue.index[0] in regime.index else "bull")

    return np.array(trade_rets), regime_labels


def strategy_d_month_end_growth(data, regime):
    """D) Buy top-3 growth stocks (by prior month perf) on 20th, sell last trading day."""
    trade_rets = []
    regime_labels = []

    # Build monthly returns for ranking
    growth_monthly = {}
    for t in GROWTH_TICKERS:
        df = data[t].loc[OOT_START:OOT_END]
        growth_monthly[t] = df["Close"].resample("ME").last().pct_change()

    # Get trading dates for each stock
    all_dates = data["SPY"].loc[OOT_START:OOT_END].index

    months = pd.period_range(OOT_START, OOT_END, freq="M")
    for i in range(1, len(months)):
        prev_month = months[i - 1]
        curr_month = months[i]

        # Rank by prior month return
        prior_rets = {}
        for t in GROWTH_TICKERS:
            if prev_month in growth_monthly[t].index.to_period("M").values:
                mask = growth_monthly[t].index.to_period("M") == prev_month
                vals = growth_monthly[t][mask]
                if len(vals) > 0 and not np.isnan(vals.iloc[0]):
                    prior_rets[t] = vals.iloc[0]

        if len(prior_rets) < 3:
            continue

        top3 = sorted(prior_rets, key=prior_rets.get, reverse=True)[:3]

        # Find ~20th trading day and last trading day of current month
        curr_dates = all_dates[all_dates.to_period("M") == curr_month]
        if len(curr_dates) < 15:
            continue

        # Find the trading day on or after the 20th calendar day
        target_20th = pd.Timestamp(f"{curr_month.start_time.year}-{curr_month.start_time.month:02d}-20")
        buy_dates = curr_dates[curr_dates >= target_20th]
        if len(buy_dates) == 0:
            continue
        buy_date = buy_dates[0]
        sell_date = curr_dates[-1]

        if buy_date >= sell_date:
            continue

        # Equal-weight top 3
        portfolio_ret = 0.0
        valid = 0
        for t in top3:
            tdf = data[t].loc[OOT_START:OOT_END]
            if buy_date in tdf.index and sell_date in tdf.index:
                bp = tdf.loc[buy_date, "Open"]
                sp = tdf.loc[sell_date, "Close"]
                if bp > 0:
                    portfolio_ret += apply_slippage((sp / bp) - 1)
                    valid += 1

        if valid > 0:
            trade_rets.append(portfolio_ret / valid)
            regime_labels.append(regime.loc[buy_date] if buy_date in regime.index else "bull")

    return np.array(trade_rets), regime_labels


def strategy_e_sector_rotation(data, regime):
    """E) Monthly sector rotation: buy best prior-month sector ETF, hold entire month."""
    trade_rets = []
    regime_labels = []

    sector_monthly = {}
    for t in SECTOR_ETFS:
        df = data[t].loc[DATA_START:OOT_END]
        sector_monthly[t] = df["Close"].resample("ME").last().pct_change()

    months = pd.period_range(OOT_START, OOT_END, freq="M")
    all_dates = data["SPY"].loc[OOT_START:OOT_END].index

    for i in range(1, len(months)):
        prev_month = months[i - 1]
        curr_month = months[i]

        # Find best prior-month ETF
        prior_rets = {}
        for t in SECTOR_ETFS:
            mask = sector_monthly[t].index.to_period("M") == prev_month
            vals = sector_monthly[t][mask]
            if len(vals) > 0 and not np.isnan(vals.iloc[0]):
                prior_rets[t] = vals.iloc[0]

        if len(prior_rets) == 0:
            continue

        best = max(prior_rets, key=prior_rets.get)

        # Buy first day of month, sell last day
        curr_dates = all_dates[all_dates.to_period("M") == curr_month]
        if len(curr_dates) < 2:
            continue

        tdf = data[best].loc[OOT_START:OOT_END]
        buy_date = curr_dates[0]
        sell_date = curr_dates[-1]
        if buy_date in tdf.index and sell_date in tdf.index:
            bp = tdf.loc[buy_date, "Open"]
            sp = tdf.loc[sell_date, "Close"]
            if bp > 0:
                ret = apply_slippage((sp / bp) - 1)
                trade_rets.append(ret)
                regime_labels.append(regime.loc[buy_date] if buy_date in regime.index else "bull")

    return np.array(trade_rets), regime_labels


def get_opex_dates(start, end):
    """Get 3rd Friday of each month (options expiration)."""
    opex = []
    current = pd.Timestamp(start)
    while current <= pd.Timestamp(end):
        # 3rd Friday: first day of month, find first Friday, add 14 days
        first = current.replace(day=1)
        # dayofweek: Mon=0..Fri=4
        days_until_fri = (4 - first.dayofweek) % 7
        first_fri = first + pd.Timedelta(days=days_until_fri)
        third_fri = first_fri + pd.Timedelta(days=14)
        opex.append(third_fri)
        # Next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return opex


def strategy_f_avoid_opex(data, regime):
    """F) Buy QQQ on non-OpEx weeks. Compare returns OpEx vs non-OpEx."""
    qqq = data["QQQ"].loc[OOT_START:OOT_END].copy()
    qqq["dow"] = qqq.index.dayofweek

    opex_dates = get_opex_dates(OOT_START, OOT_END)

    # Mark OpEx week: 3rd Friday ± 2 trading days
    opex_windows = set()
    for opex in opex_dates:
        for delta in range(-2, 3):
            d = opex + pd.Timedelta(days=delta)
            opex_windows.add(d.date())

    qqq["is_opex"] = pd.Series(qqq.index.date, index=qqq.index).apply(lambda d: d in opex_windows)

    # Weekly trades
    qqq["year_week"] = qqq.index.isocalendar().year.astype(str) + "-" + qqq.index.isocalendar().week.astype(str).str.zfill(2)

    non_opex_rets = []
    opex_rets = []
    regime_labels = []

    for _, grp in qqq.groupby("year_week"):
        if len(grp) < 2:
            continue
        buy_price = grp.iloc[0]["Open"]
        sell_price = grp.iloc[-1]["Close"]
        ret = apply_slippage((sell_price / buy_price) - 1)

        is_opex_week = grp["is_opex"].any()
        if is_opex_week:
            opex_rets.append(ret)
        else:
            non_opex_rets.append(ret)
            regime_labels.append(regime.loc[grp.index[0]] if grp.index[0] in regime.index else "bull")

    # Strategy: only trade non-OpEx weeks
    return np.array(non_opex_rets), regime_labels, np.array(opex_rets)


def run_all():
    """Run all 6 strategy variants."""
    data = download_data()

    # Compute regime
    spy_close = data["SPY"]["Close"]
    regime = compute_regime(spy_close)

    results = {}

    # ── A) Buy-Wednesday-Sell-Friday ──
    print("\n[A] Buy-Wednesday-Sell-Friday (QQQ)...")
    rets_a, reg_a = strategy_a_wed_fri(data, regime)
    m_a = calc_metrics(rets_a, reg_a)
    p_a = permutation_test(rets_a) if len(rets_a) > 5 else 1.0
    gates_a = gate_check(m_a, p_a)
    results["A_Wed_Fri"] = {**m_a, "perm_p": round(p_a, 4), "gates": gates_a}
    print(f"  Trades: {m_a['n_trades']}, Sharpe: {m_a['sharpe']}, PF: {m_a['pf']}, WR: {m_a['wr']:.1%}, DD: {m_a['max_dd_pct']:.1f}%, p={p_a:.4f}")
    print(f"  Gates: {gates_a}")

    # ── B) First-5-Days of Month ──
    print("\n[B] First-5-Days of Month (SPY)...")
    rets_b, reg_b = strategy_b_first5(data, regime)
    m_b = calc_metrics(rets_b, reg_b)
    p_b = permutation_test(rets_b) if len(rets_b) > 5 else 1.0
    gates_b = gate_check(m_b, p_b)
    results["B_First5Days"] = {**m_b, "perm_p": round(p_b, 4), "gates": gates_b}
    print(f"  Trades: {m_b['n_trades']}, Sharpe: {m_b['sharpe']}, PF: {m_b['pf']}, WR: {m_b['wr']:.1%}, DD: {m_b['max_dd_pct']:.1f}%, p={p_b:.4f}")
    print(f"  Gates: {gates_b}")

    # ── C) Sell-Monday Buy-Tuesday ──
    print("\n[C] Sell-Monday Buy-Tuesday (QQQ)...")
    rets_c, reg_c = strategy_c_skip_monday(data, regime)
    m_c = calc_metrics(rets_c, reg_c)
    p_c = permutation_test(rets_c) if len(rets_c) > 5 else 1.0
    gates_c = gate_check(m_c, p_c)
    results["C_SkipMonday"] = {**m_c, "perm_p": round(p_c, 4), "gates": gates_c}
    print(f"  Trades: {m_c['n_trades']}, Sharpe: {m_c['sharpe']}, PF: {m_c['pf']}, WR: {m_c['wr']:.1%}, DD: {m_c['max_dd_pct']:.1f}%, p={p_c:.4f}")
    print(f"  Gates: {gates_c}")

    # ── D) Month-End Growth Rally ──
    print("\n[D] Month-End Growth Rally (Top-3 growth)...")
    rets_d, reg_d = strategy_d_month_end_growth(data, regime)
    m_d = calc_metrics(rets_d, reg_d)
    p_d = permutation_test(rets_d) if len(rets_d) > 5 else 1.0
    gates_d = gate_check(m_d, p_d)
    results["D_MonthEndGrowth"] = {**m_d, "perm_p": round(p_d, 4), "gates": gates_d}
    print(f"  Trades: {m_d['n_trades']}, Sharpe: {m_d['sharpe']}, PF: {m_d['pf']}, WR: {m_d['wr']:.1%}, DD: {m_d['max_dd_pct']:.1f}%, p={p_d:.4f}")
    print(f"  Gates: {gates_d}")

    # ── E) Sector Monthly Rotation ──
    print("\n[E] Sector Monthly Rotation (XLK/XLE/XLF/SPY)...")
    rets_e, reg_e = strategy_e_sector_rotation(data, regime)
    m_e = calc_metrics(rets_e, reg_e)
    p_e = permutation_test(rets_e) if len(rets_e) > 5 else 1.0
    gates_e = gate_check(m_e, p_e)
    results["E_SectorRotation"] = {**m_e, "perm_p": round(p_e, 4), "gates": gates_e}
    print(f"  Trades: {m_e['n_trades']}, Sharpe: {m_e['sharpe']}, PF: {m_e['pf']}, WR: {m_e['wr']:.1%}, DD: {m_e['max_dd_pct']:.1f}%, p={p_e:.4f}")
    print(f"  Gates: {gates_e}")

    # ── F) Avoid-Options-Expiry ──
    print("\n[F] Avoid-Options-Expiry (QQQ)...")
    rets_f, reg_f, opex_rets_f = strategy_f_avoid_opex(data, regime)
    m_f = calc_metrics(rets_f, reg_f)
    p_f = permutation_test(rets_f) if len(rets_f) > 5 else 1.0
    gates_f = gate_check(m_f, p_f)
    opex_mean = float(np.mean(opex_rets_f)) * 100 if len(opex_rets_f) > 0 else 0
    non_opex_mean = float(np.mean(rets_f)) * 100 if len(rets_f) > 0 else 0
    results["F_AvoidOpEx"] = {
        **m_f, "perm_p": round(p_f, 4), "gates": gates_f,
        "opex_week_mean_ret_pct": round(opex_mean, 4),
        "non_opex_week_mean_ret_pct": round(non_opex_mean, 4),
        "opex_week_count": len(opex_rets_f),
        "non_opex_week_count": len(rets_f),
    }
    print(f"  Non-OpEx trades: {m_f['n_trades']}, Sharpe: {m_f['sharpe']}, PF: {m_f['pf']}, WR: {m_f['wr']:.1%}")
    print(f"  OpEx mean ret: {opex_mean:.4f}%, Non-OpEx mean ret: {non_opex_mean:.4f}%")
    print(f"  Gates: {gates_f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY — 5-Gate Validation Results")
    print("=" * 70)
    passed = []
    for name, res in results.items():
        status = "PASS" if res["gates"]["all_pass"] else "FAIL"
        failed_gates = [g for g, v in res["gates"].items() if not v and g != "all_pass"]
        fg_str = f" (failed: {', '.join(failed_gates)})" if failed_gates else ""
        print(f"  {name:25s} → {status}{fg_str}  |  Sharpe={res['sharpe']:.2f}  PF={res['pf']:.2f}  WR={res['wr']:.1%}  DD={res['max_dd_pct']:.1f}%  p={res['perm_p']:.3f}")
        if res["gates"]["all_pass"]:
            passed.append(name)

    if passed:
        print(f"\n  ACTIONABLE strategies: {', '.join(passed)}")
    else:
        print("\n  No strategies passed all 5 gates. Calendar anomalies are weak/arbitraged away in this period.")

    # Add metadata
    output = {
        "metadata": {
            "run_date": dt.datetime.now().isoformat(),
            "oot_period": f"{OOT_START} to {OOT_END}",
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "permutation_iters": PERM_ITERS,
            "tickers": TICKERS,
            "regime_definition": "SPY > 200-SMA = bull, else bear",
        },
        "strategies": results,
        "passed_all_gates": passed,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUT_PATH}")

    return output


if __name__ == "__main__":
    run_all()
