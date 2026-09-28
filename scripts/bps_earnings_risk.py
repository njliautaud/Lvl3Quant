#!/usr/bin/env python3
"""
BPS Earnings Risk Study
========================
HC #664 R4: Quantify how earnings announcements affect BPS performance.

Hypothesis: Positions with earnings during holding period have higher breach
risk due to large gap moves, but may also collect more premium (higher IV).

Tests 3 approaches:
  1. Baseline (no filter)
  2. Skip trades if earnings within holding period
  3. ONLY trade around earnings (premium harvesting)

Uses:
  - Cached earnings_dates.parquet (yfinance-sourced)
  - BPS trade data from bps_assignment_risk study
  - 5% BA cost per HC conventions
  - VIX-scaled sizing + 2% CB + VIX cutoff 30 (optimal config)
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import timedelta

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

OUTPUT = ROOT / "output" / "bps_earnings_risk"
OUTPUT.mkdir(parents=True, exist_ok=True)

BA_COST_FRAC = 0.05


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_all():
    """Load trades, earnings dates, macro/VIX, and sector mapping."""
    trades = pd.read_parquet(ROOT / "output" / "bps_assignment_risk" / "trades_close_1dte.parquet")
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    # Earnings dates from cache
    try:
        earnings = pd.read_parquet(ROOT / "wheel_strategy_v1" / "data" / "cache" / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except Exception:
        earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    # Macro for VIX
    from higher_returns_study import load_data
    _, _, macro, fund, _, _ = load_data()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    sector_map = dict(zip(fund["ticker"], fund["sector"]))

    return trades, earnings, macro, sector_map


def apply_ba_cost(trades):
    """Apply 5% BA cost to P&L."""
    trades = trades.copy()
    open_cost = trades["net_credit"].abs() * BA_COST_FRAC * 2
    closed_early = trades["exit_type"].isin(["profit_take", "early_close_1DTE", "loss_stop"])
    close_cost = pd.Series(0.0, index=trades.index)
    close_cost[closed_early] = trades.loc[closed_early, "net_credit"].abs() * BA_COST_FRAC * 2
    trades["ba_cost"] = open_cost + close_cost
    trades["honest_pnl"] = trades["realized_pnl"] - trades["ba_cost"]
    return trades


# ═══════════════════════════════════════════════════════════════════
# Earnings Flagging
# ═══════════════════════════════════════════════════════════════════

def flag_earnings_trades(trades, earnings):
    """
    For each trade, flag whether an earnings announcement falls within
    the [open_date, close_date] holding period.

    Also flags trades where earnings is within 1 day AFTER close (gap risk
    from overnight move if we hold through expiry eve).
    """
    trades = trades.copy()
    trades["has_earnings"] = False
    trades["earnings_date_in_hold"] = pd.NaT
    trades["days_to_earnings"] = np.nan

    # Build per-ticker earnings lookup (sorted dates for binary search)
    earnings_by_ticker = {}
    for ticker in trades["ticker"].unique():
        tk_earn = earnings[earnings["ticker"] == ticker]["earnings_date"].sort_values().values
        if len(tk_earn) > 0:
            earnings_by_ticker[ticker] = tk_earn

    for idx, row in trades.iterrows():
        ticker = row["ticker"]
        if ticker not in earnings_by_ticker:
            continue

        earn_dates = earnings_by_ticker[ticker]
        open_dt = row["open_date"]
        close_dt = row["close_date"]

        # Check if any earnings date falls within [open - 1 day, close + 1 day]
        # The buffer captures overnight gap risk
        window_start = open_dt - pd.Timedelta(days=1)
        window_end = close_dt + pd.Timedelta(days=1)

        mask = (earn_dates >= window_start) & (earn_dates <= window_end)
        if mask.any():
            matching = earn_dates[mask]
            trades.at[idx, "has_earnings"] = True
            trades.at[idx, "earnings_date_in_hold"] = pd.Timestamp(matching[0])
            # Days from open to earnings
            trades.at[idx, "days_to_earnings"] = (pd.Timestamp(matching[0]) - open_dt).days

    n_with = trades["has_earnings"].sum()
    n_total = len(trades)
    print(f"\n  Earnings flagging: {n_with}/{n_total} trades ({n_with/n_total*100:.1f}%) "
          f"have earnings in holding period")

    # For tickers missing from earnings data, try quarterly approximation
    missing_tickers = set(trades["ticker"].unique()) - set(earnings_by_ticker.keys())
    if missing_tickers:
        print(f"  Missing earnings data for: {sorted(missing_tickers)}")
        print(f"  Using quarterly approximation for missing tickers...")
        trades = _approximate_quarterly_earnings(trades, missing_tickers)

    return trades


def _approximate_quarterly_earnings(trades, missing_tickers):
    """
    For tickers without earnings data, approximate using typical quarterly
    schedule (mid-Jan, mid-Apr, mid-Jul, mid-Oct).
    """
    approx_months = [1, 4, 7, 10]
    approx_day = 20  # Most large-caps report around 15-25th

    for ticker in missing_tickers:
        tk_trades = trades[trades["ticker"] == ticker]
        for idx in tk_trades.index:
            open_dt = trades.at[idx, "open_date"]
            close_dt = trades.at[idx, "close_date"]
            window_start = open_dt - pd.Timedelta(days=1)
            window_end = close_dt + pd.Timedelta(days=1)

            # Generate approximate earnings dates for the relevant year range
            for year in range(open_dt.year, close_dt.year + 1):
                for month in approx_months:
                    approx_date = pd.Timestamp(year=year, month=month, day=approx_day)
                    if window_start <= approx_date <= window_end:
                        trades.at[idx, "has_earnings"] = True
                        trades.at[idx, "earnings_date_in_hold"] = approx_date
                        trades.at[idx, "days_to_earnings"] = (approx_date - open_dt).days
                        break

    n_approx = trades.loc[trades["ticker"].isin(missing_tickers), "has_earnings"].sum()
    print(f"  Approximated {n_approx} earnings-period trades for {len(missing_tickers)} tickers")
    return trades


# ═══════════════════════════════════════════════════════════════════
# Performance Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_group(trades, label):
    """Compute key metrics for a group of trades."""
    if len(trades) == 0:
        return {"label": label, "n_trades": 0, "error": "no trades"}

    n = len(trades)

    # Breach rate
    breached = trades["status_at_close"].isin(["breached", "full_loss"])
    breach_count = breached.sum()
    breach_rate = breach_count / n * 100

    # P&L
    avg_pnl = trades["honest_pnl"].mean()
    median_pnl = trades["honest_pnl"].median()
    total_pnl = trades["honest_pnl"].sum()

    # Win rate
    wins = (trades["honest_pnl"] > 0).sum()
    win_rate = wins / n * 100

    # Worst loss
    worst_loss = trades["honest_pnl"].min()

    # Average premium collected (net credit)
    avg_credit = trades["net_credit"].mean()

    # Average distance to short strike at open
    avg_distance = trades["distance_to_short_pct"].mean()

    # Gap breach rate (overnight gaps specifically)
    gap_breaches = (trades["gap_breach"] == True).sum() if "gap_breach" in trades.columns else 0
    gap_breach_rate = gap_breaches / n * 100

    # Overnight move stats
    if "overnight_move_pct" in trades.columns:
        ovm = trades["overnight_move_pct"].dropna()
        avg_overnight = ovm.mean() if len(ovm) > 0 else np.nan
        worst_overnight = ovm.min() if len(ovm) > 0 else np.nan
    else:
        avg_overnight = np.nan
        worst_overnight = np.nan

    # P&L distribution
    pnl_p10 = trades["honest_pnl"].quantile(0.10)
    pnl_p25 = trades["honest_pnl"].quantile(0.25)
    pnl_p75 = trades["honest_pnl"].quantile(0.75)
    pnl_p90 = trades["honest_pnl"].quantile(0.90)

    # Profit factor (trade-level)
    trade_wins = trades.loc[trades["honest_pnl"] > 0, "honest_pnl"].sum()
    trade_losses = abs(trades.loc[trades["honest_pnl"] < 0, "honest_pnl"].sum())
    pf = trade_wins / trade_losses if trade_losses > 0 else float("inf")

    return {
        "label": label,
        "n_trades": n,
        "breach_rate_pct": round(breach_rate, 2),
        "breach_count": int(breach_count),
        "gap_breach_rate_pct": round(gap_breach_rate, 2),
        "avg_pnl": round(avg_pnl, 2),
        "median_pnl": round(median_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "worst_loss": round(worst_loss, 2),
        "avg_credit": round(avg_credit, 2),
        "avg_distance_to_short_pct": round(avg_distance, 2),
        "avg_overnight_move_pct": round(avg_overnight, 4) if not np.isnan(avg_overnight) else None,
        "worst_overnight_move_pct": round(worst_overnight, 4) if not np.isnan(worst_overnight) else None,
        "pnl_p10": round(pnl_p10, 2),
        "pnl_p25": round(pnl_p25, 2),
        "pnl_p75": round(pnl_p75, 2),
        "pnl_p90": round(pnl_p90, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Portfolio Simulation (with optimal config)
# ═══════════════════════════════════════════════════════════════════

def simulate_portfolio(trades, macro, sector_map, label="test",
                       starting_capital=100_000):
    """
    Run portfolio sim with optimal BPS config:
    VIX-scaled + 2% CB + VIX cutoff 30 + sector cap 25%.

    Returns metrics dict + equity curve.
    """
    trades = trades.copy().sort_values("open_date").reset_index(drop=True)
    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")

    vix_by_date = macro.set_index("date")["vix"].to_dict()
    trades["open_vix"] = trades["open_date"].map(vix_by_date)

    # VIX scaling
    def vix_scalar(vix):
        if pd.isna(vix) or vix <= 15.0:
            return 1.0
        return max(0.0, 1.0 - (vix - 15.0) / 30.0)

    trades["vix_scalar"] = trades["open_vix"].apply(vix_scalar)

    # VIX hard cutoff at 30
    mask = trades["open_vix"].fillna(0) <= 30
    n_vix_blocked = (~mask).sum()
    trades = trades[mask].copy()

    trades["scaled_pnl"] = trades["honest_pnl"] * trades["vix_scalar"]

    # Day-by-day simulation with sector cap 25% and 2% CB
    trades_by_open = defaultdict(list)
    for _, row in trades.iterrows():
        trades_by_open[row["open_date"]].append(row)

    equity = starting_capital
    daily_equity = []
    accepted_trades = {}
    frozen_until = None
    n_accepted = 0
    n_sector_blocked = 0
    n_cb_frozen = 0

    all_dates = sorted(set(trades["open_date"].unique()) | set(trades["close_date"].unique()))

    for dt in all_dates:
        day_pnl = 0.0
        keys_to_remove = []
        for key, tr in accepted_trades.items():
            if tr["close_date"] == dt:
                day_pnl += tr["scaled_pnl"]
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del accepted_trades[key]

        equity += day_pnl
        daily_equity.append({"date": dt, "equity": equity, "daily_pnl": day_pnl})

        # 2% CB check
        if len(daily_equity) >= 2:
            prev_eq = daily_equity[-2]["equity"]
            if prev_eq > 0:
                daily_ret = day_pnl / prev_eq
                if daily_ret < -0.02:
                    freeze_end = dt + pd.Timedelta(days=1)
                    if frozen_until is None or freeze_end > frozen_until:
                        frozen_until = freeze_end

        opening_today = trades_by_open.get(dt, [])
        if not opening_today:
            continue

        if frozen_until is not None and dt <= frozen_until:
            n_cb_frozen += len(opening_today)
            continue

        # Sector cap 25%
        sector_exposure = defaultdict(float)
        total_exposure = 0.0
        for key, t in accepted_trades.items():
            margin = t["spread_width"] * 100 * t["contracts"]
            sector_exposure[t["sector"]] += margin
            total_exposure += margin

        for tr in opening_today:
            sec = tr["sector"]
            margin_this = tr["spread_width"] * 100 * tr["contracts"]
            denom = max(total_exposure + margin_this, equity, 1.0)
            sector_pct = (sector_exposure.get(sec, 0) + margin_this) / denom

            if sector_pct > 0.25:
                n_sector_blocked += 1
                continue

            key = (tr["ticker"], tr["open_date"])
            accepted_trades[key] = tr
            sector_exposure[sec] = sector_exposure.get(sec, 0) + margin_this
            total_exposure += margin_this
            n_accepted += 1

    eq_df = pd.DataFrame(daily_equity)
    if eq_df.empty or len(eq_df) < 30:
        return {"label": label, "error": "insufficient data", "n_accepted": n_accepted}

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.sort_values("date").reset_index(drop=True)
    eq_df["ret"] = eq_df["equity"].pct_change()
    rets = eq_df["ret"].dropna()

    total_days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
    total_years = max(total_days / 365.25, 0.01)
    total_return = eq_df["equity"].iloc[-1] / starting_capital
    cagr = (total_return ** (1 / total_years)) - 1 if total_return > 0 else -1.0

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0

    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = float(eq_df["dd"].min())

    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")
    wr = float(len(rets[rets > 0]) / len(rets) * 100) if len(rets) > 0 else 0.0

    return {
        "label": label,
        "n_trades_accepted": n_accepted,
        "n_vix_blocked": n_vix_blocked,
        "n_sector_blocked": n_sector_blocked,
        "n_cb_frozen": n_cb_frozen,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "final_equity": round(eq_df["equity"].iloc[-1], 2),
        "equity_curve": eq_df[["date", "equity", "daily_pnl"]],
    }


# ═══════════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("BPS EARNINGS RISK STUDY")
    print("Do earnings announcements during holding period increase breach risk?")
    print("=" * 70)

    trades, earnings, macro, sector_map = load_all()
    trades = apply_ba_cost(trades)
    trades = flag_earnings_trades(trades, earnings)

    n_total = len(trades)
    n_with_earnings = trades["has_earnings"].sum()
    n_without = n_total - n_with_earnings
    print(f"\n{'='*70}")
    print(f"DATASET: {n_total} trades, {trades['ticker'].nunique()} tickers")
    print(f"  WITH earnings in hold:    {n_with_earnings} ({n_with_earnings/n_total*100:.1f}%)")
    print(f"  WITHOUT earnings in hold: {n_without} ({n_without/n_total*100:.1f}%)")

    # ── SECTION 1: Trade-Level Comparison ──
    print(f"\n{'='*70}")
    print("SECTION 1: TRADE-LEVEL METRICS — EARNINGS vs NO-EARNINGS")
    print("=" * 70)

    with_earn = trades[trades["has_earnings"]].copy()
    without_earn = trades[~trades["has_earnings"]].copy()

    m_all = analyze_group(trades, "All Trades (baseline)")
    m_with = analyze_group(with_earn, "WITH earnings")
    m_without = analyze_group(without_earn, "WITHOUT earnings")

    for label, m in [("ALL TRADES", m_all), ("WITH EARNINGS", m_with), ("WITHOUT EARNINGS", m_without)]:
        print(f"\n  {label} ({m['n_trades']} trades):")
        print(f"    Breach rate:     {m['breach_rate_pct']:>6.2f}%  ({m['breach_count']} breaches)")
        print(f"    Gap breach rate: {m['gap_breach_rate_pct']:>6.2f}%")
        print(f"    Win rate:        {m['win_rate_pct']:>6.1f}%")
        print(f"    Profit factor:   {m['profit_factor']:>6.2f}")
        print(f"    Avg P&L:        ${m['avg_pnl']:>8.2f}")
        print(f"    Median P&L:     ${m['median_pnl']:>8.2f}")
        print(f"    Worst loss:     ${m['worst_loss']:>8.2f}")
        print(f"    Avg credit:     ${m['avg_credit']:>8.2f}")
        print(f"    Avg distance:    {m['avg_distance_to_short_pct']:>6.2f}%")
        print(f"    P&L [p10/p25/p75/p90]: ${m['pnl_p10']:.0f} / ${m['pnl_p25']:.0f} / ${m['pnl_p75']:.0f} / ${m['pnl_p90']:.0f}")

    # ── SECTION 2: Earnings Premium Analysis ──
    print(f"\n{'='*70}")
    print("SECTION 2: EARNINGS PREMIUM — Do we collect more credit near earnings?")
    print("=" * 70)

    avg_credit_with = with_earn["net_credit"].mean()
    avg_credit_without = without_earn["net_credit"].mean()
    premium_ratio = avg_credit_with / avg_credit_without if avg_credit_without > 0 else np.nan

    print(f"  Avg net credit WITH earnings:    ${avg_credit_with:>8.2f}")
    print(f"  Avg net credit WITHOUT earnings: ${avg_credit_without:>8.2f}")
    print(f"  Premium ratio (with/without):     {premium_ratio:>8.2f}x")

    # Check if higher premium compensates for higher risk
    risk_adj_with = m_with["avg_pnl"] / abs(m_with["worst_loss"]) if m_with["worst_loss"] != 0 else 0
    risk_adj_without = m_without["avg_pnl"] / abs(m_without["worst_loss"]) if m_without["worst_loss"] != 0 else 0
    print(f"\n  Risk-adjusted return (avg P&L / |worst loss|):")
    print(f"    WITH earnings:    {risk_adj_with:>8.4f}")
    print(f"    WITHOUT earnings: {risk_adj_without:>8.4f}")

    # ── SECTION 3: Per-Ticker Earnings Impact ──
    print(f"\n{'='*70}")
    print("SECTION 3: PER-TICKER EARNINGS IMPACT (top 15 most affected)")
    print("=" * 70)

    ticker_stats = []
    for ticker in sorted(trades["ticker"].unique()):
        tk_all = trades[trades["ticker"] == ticker]
        tk_with = tk_all[tk_all["has_earnings"]]
        tk_without = tk_all[~tk_all["has_earnings"]]

        if len(tk_with) < 3:
            continue

        breach_with = tk_with["status_at_close"].isin(["breached", "full_loss"]).mean() * 100
        breach_without = tk_without["status_at_close"].isin(["breached", "full_loss"]).mean() * 100 if len(tk_without) > 0 else 0

        ticker_stats.append({
            "ticker": ticker,
            "n_earnings_trades": len(tk_with),
            "n_other_trades": len(tk_without),
            "breach_with_pct": round(breach_with, 1),
            "breach_without_pct": round(breach_without, 1),
            "breach_diff": round(breach_with - breach_without, 1),
            "avg_pnl_with": round(tk_with["honest_pnl"].mean(), 2),
            "avg_pnl_without": round(tk_without["honest_pnl"].mean(), 2) if len(tk_without) > 0 else 0,
            "avg_credit_with": round(tk_with["net_credit"].mean(), 2),
            "avg_credit_without": round(tk_without["net_credit"].mean(), 2) if len(tk_without) > 0 else 0,
        })

    ticker_df = pd.DataFrame(ticker_stats).sort_values("breach_diff", ascending=False)

    print(f"\n  {'Ticker':<8} {'N_Earn':>6} {'Breach_E':>9} {'Breach_O':>9} {'Diff':>6} "
          f"{'AvgPnL_E':>10} {'AvgPnL_O':>10} {'Credit_E':>10} {'Credit_O':>10}")
    print("  " + "-" * 95)
    for _, row in ticker_df.head(15).iterrows():
        print(f"  {row['ticker']:<8} {row['n_earnings_trades']:>6} {row['breach_with_pct']:>8.1f}% "
              f"{row['breach_without_pct']:>8.1f}% {row['breach_diff']:>+5.1f} "
              f"${row['avg_pnl_with']:>9.2f} ${row['avg_pnl_without']:>9.2f} "
              f"${row['avg_credit_with']:>9.2f} ${row['avg_credit_without']:>9.2f}")

    # ── SECTION 4: Portfolio-Level Simulation (3 approaches) ──
    print(f"\n{'='*70}")
    print("SECTION 4: PORTFOLIO SIMULATION — 3 APPROACHES")
    print("(Optimal config: VIX-scaled + 2% CB + VIX cutoff 30 + sector 25%)")
    print("=" * 70)

    # Approach 1: Baseline (all trades)
    print("\n  Running Approach 1: Baseline (no filter)...")
    sim_baseline = simulate_portfolio(trades, macro, sector_map, label="Baseline (no filter)")

    # Approach 2: Skip earnings trades
    print("  Running Approach 2: Skip trades with earnings...")
    sim_skip = simulate_portfolio(without_earn, macro, sector_map, label="Skip earnings")

    # Approach 3: Only trade around earnings
    print("  Running Approach 3: ONLY trade around earnings...")
    sim_earn_only = simulate_portfolio(with_earn, macro, sector_map, label="Earnings only")

    sims = [sim_baseline, sim_skip, sim_earn_only]
    print(f"\n  {'Approach':<25} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
          f"{'WR':>6} {'PF':>6} {'Trades':>7} {'Final$':>10}")
    print("  " + "-" * 100)
    for s in sims:
        if "error" in s:
            print(f"  {s['label']:<25} ERROR: {s['error']}")
            continue
        print(f"  {s['label']:<25} {s['cagr_pct']:>6.1f}% {s['sharpe']:>7.2f} {s['sortino']:>8.2f} "
              f"{s['max_dd_pct']:>6.1f}% {s['win_rate_pct']:>5.1f}% {s['profit_factor']:>5.2f} "
              f"{s['n_trades_accepted']:>7} ${s['final_equity']:>10,.0f}")

    # ── SECTION 5: Earnings Timing Analysis ──
    print(f"\n{'='*70}")
    print("SECTION 5: EARNINGS TIMING — When in the holding period do earnings hit?")
    print("=" * 70)

    earn_trades = trades[trades["has_earnings"]].copy()
    if len(earn_trades) > 0:
        # Days from open to earnings
        timing = earn_trades["days_to_earnings"].dropna()
        print(f"  Days from trade open to earnings announcement:")
        print(f"    Mean:   {timing.mean():>5.1f} days")
        print(f"    Median: {timing.median():>5.1f} days")
        print(f"    Min:    {timing.min():>5.1f} days")
        print(f"    Max:    {timing.max():>5.1f} days")

        # Breach rate by timing bucket
        earn_trades["timing_bucket"] = pd.cut(
            earn_trades["days_to_earnings"],
            bins=[-2, 0, 2, 4, 7, 14],
            labels=["Before open", "0-2 days", "2-4 days", "4-7 days", "7-14 days"],
        )
        print(f"\n  Breach rate by timing:")
        for bucket in earn_trades["timing_bucket"].cat.categories:
            subset = earn_trades[earn_trades["timing_bucket"] == bucket]
            if len(subset) > 0:
                br = subset["status_at_close"].isin(["breached", "full_loss"]).mean() * 100
                avg_pnl = subset["honest_pnl"].mean()
                print(f"    {bucket:<15}: {len(subset):>5} trades, breach={br:>5.1f}%, avg P&L=${avg_pnl:>8.2f}")

    # ── SECTION 6: Sector Breakdown ──
    print(f"\n{'='*70}")
    print("SECTION 6: EARNINGS RISK BY SECTOR")
    print("=" * 70)

    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")
    sector_stats = []
    for sector in sorted(trades["sector"].unique()):
        s_trades = trades[trades["sector"] == sector]
        s_with = s_trades[s_trades["has_earnings"]]
        s_without = s_trades[~s_trades["has_earnings"]]

        if len(s_with) < 5:
            continue

        br_with = s_with["status_at_close"].isin(["breached", "full_loss"]).mean() * 100
        br_without = s_without["status_at_close"].isin(["breached", "full_loss"]).mean() * 100 if len(s_without) > 0 else 0

        sector_stats.append({
            "sector": sector,
            "n_with": len(s_with),
            "n_without": len(s_without),
            "breach_with": round(br_with, 1),
            "breach_without": round(br_without, 1),
            "breach_diff": round(br_with - br_without, 1),
            "avg_pnl_with": round(s_with["honest_pnl"].mean(), 2),
            "avg_pnl_without": round(s_without["honest_pnl"].mean(), 2) if len(s_without) > 0 else 0,
        })

    if sector_stats:
        sector_df = pd.DataFrame(sector_stats).sort_values("breach_diff", ascending=False)
        print(f"\n  {'Sector':<25} {'N_Earn':>6} {'Breach_E':>9} {'Breach_O':>9} {'Diff':>6} "
              f"{'PnL_E':>9} {'PnL_O':>9}")
        print("  " + "-" * 80)
        for _, row in sector_df.iterrows():
            print(f"  {row['sector']:<25} {row['n_with']:>6} {row['breach_with']:>8.1f}% "
                  f"{row['breach_without']:>8.1f}% {row['breach_diff']:>+5.1f} "
                  f"${row['avg_pnl_with']:>8.2f} ${row['avg_pnl_without']:>8.2f}")

    # ── SECTION 7: Bottom-Line Verdict ──
    print(f"\n{'='*70}")
    print("SECTION 7: VERDICT — DOES THE EXTRA PREMIUM COMPENSATE FOR EXTRA RISK?")
    print("=" * 70)

    breach_delta = m_with["breach_rate_pct"] - m_without["breach_rate_pct"]
    pnl_delta = m_with["avg_pnl"] - m_without["avg_pnl"]
    wr_delta = m_with["win_rate_pct"] - m_without["win_rate_pct"]

    print(f"\n  Earnings effect on trades:")
    print(f"    Breach rate:  {m_without['breach_rate_pct']:.1f}% -> {m_with['breach_rate_pct']:.1f}% ({breach_delta:+.1f}pp)")
    print(f"    Win rate:     {m_without['win_rate_pct']:.1f}% -> {m_with['win_rate_pct']:.1f}% ({wr_delta:+.1f}pp)")
    print(f"    Avg P&L:     ${m_without['avg_pnl']:.2f} -> ${m_with['avg_pnl']:.2f} ({pnl_delta:+.2f})")
    print(f"    Avg credit:  ${m_without['avg_credit']:.2f} -> ${m_with['avg_credit']:.2f} ({premium_ratio:.2f}x)")

    # Portfolio-level verdict
    if "error" not in sim_skip and "error" not in sim_baseline:
        sharpe_delta = sim_skip["sharpe"] - sim_baseline["sharpe"]
        sortino_delta = sim_skip["sortino"] - sim_baseline["sortino"]
        dd_delta = sim_skip["max_dd_pct"] - sim_baseline["max_dd_pct"]

        print(f"\n  Portfolio impact of SKIPPING earnings trades:")
        print(f"    Sharpe:  {sim_baseline['sharpe']:.2f} -> {sim_skip['sharpe']:.2f} ({sharpe_delta:+.2f})")
        print(f"    Sortino: {sim_baseline['sortino']:.2f} -> {sim_skip['sortino']:.2f} ({sortino_delta:+.2f})")
        print(f"    Max DD:  {sim_baseline['max_dd_pct']:.1f}% -> {sim_skip['max_dd_pct']:.1f}% ({dd_delta:+.1f}pp)")

        if sharpe_delta > 0.05:
            verdict = "SKIP EARNINGS — Avoiding earnings trades improves risk-adjusted returns"
        elif sharpe_delta < -0.05:
            verdict = "KEEP EARNINGS — Premium compensates for risk, skipping hurts returns"
        else:
            verdict = "NEUTRAL — Earnings filter has minimal portfolio-level impact"

        print(f"\n  >>> VERDICT: {verdict}")
    else:
        print("\n  >>> Could not compute portfolio verdict (simulation error)")

    # ── Save Results ──
    results = {
        "generated": pd.Timestamp.now().isoformat(),
        "ba_cost_frac": BA_COST_FRAC,
        "n_total_trades": n_total,
        "n_with_earnings": int(n_with_earnings),
        "n_without_earnings": int(n_without),
        "pct_with_earnings": round(n_with_earnings / n_total * 100, 1),
        "trade_level": {
            "all": m_all,
            "with_earnings": m_with,
            "without_earnings": m_without,
        },
        "earnings_premium": {
            "avg_credit_with": round(avg_credit_with, 2),
            "avg_credit_without": round(avg_credit_without, 2),
            "premium_ratio": round(premium_ratio, 3),
        },
        "portfolio_sims": {
            "baseline": {k: v for k, v in sim_baseline.items() if k != "equity_curve"},
            "skip_earnings": {k: v for k, v in sim_skip.items() if k != "equity_curve"},
            "earnings_only": {k: v for k, v in sim_earn_only.items() if k != "equity_curve"},
        },
        "per_ticker": ticker_stats,
        "per_sector": sector_stats if sector_stats else [],
    }

    with open(OUTPUT / "earnings_risk_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save equity curves
    for name, sim in [("baseline", sim_baseline), ("skip_earnings", sim_skip), ("earnings_only", sim_earn_only)]:
        if "equity_curve" in sim:
            sim["equity_curve"].to_parquet(OUTPUT / f"eq_{name}.parquet")

    # Save flagged trades
    save_cols = ["open_date", "close_date", "ticker", "exit_type", "has_earnings",
                 "earnings_date_in_hold", "days_to_earnings", "honest_pnl", "net_credit",
                 "breach_depth", "gap_breach", "status_at_close", "distance_to_short_pct"]
    trades[save_cols].to_parquet(OUTPUT / "trades_with_earnings_flag.parquet")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to {OUTPUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
