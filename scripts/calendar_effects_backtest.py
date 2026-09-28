#!/usr/bin/env python3
"""
Calendar/Seasonal Effects Backtest on Quality Stocks
=====================================================
6 variants (A-F) with 5-gate validation.

Universe: 20 quality large-caps
Period:   2022-01-01 to 2026-07-31
Capital:  $669, max $200/trade, max 3 concurrent, 2 bps slippage
"""

import json
import calendar
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2022-01-01"
END = "2026-07-31"
DATA_START = "2020-06-01"  # extra history for 200-SMA warmup + prior-year returns
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 2 bps each way (4 bps round-trip)
N_PERMS = 1000
SMA_WINDOW = 200
OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/calendar_effects_results.json")

np.random.seed(42)


# ── DATA DOWNLOAD ────────────────────────────────────────────────────────
def download_data():
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading {len(tickers)} tickers …")
    prices = {}
    for t in tickers:
        try:
            df = yf.download(t, start=DATA_START, end=END, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                prices[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: ERROR - {e}")
    print(f"Loaded {len(prices)} tickers")
    return prices


# ── HELPERS ──────────────────────────────────────────────────────────────
def get_regime(date, spy_df):
    """Bull = SPY > 200-SMA, Bear otherwise."""
    if date in spy_df.index:
        return spy_df.loc[date, "regime"]
    prior = spy_df.index[spy_df.index <= date]
    if len(prior) > 0:
        return spy_df.loc[prior[-1], "regime"]
    return "unknown"


def position_shares(price, capital_avail):
    """Max whole shares within MAX_PER_TRADE and available capital."""
    budget = min(MAX_PER_TRADE, capital_avail)
    if price <= 0 or budget <= 0:
        return 0
    return max(int(budget // price), 0)


def apply_slippage_ret(raw_ret):
    """Deduct round-trip slippage from a per-trade return."""
    return raw_ret - 2 * SLIPPAGE_BPS / 10_000


def calc_metrics(returns):
    """Sharpe, Sortino, win rate, profit factor, max drawdown, total return."""
    n = len(returns)
    if n < 2:
        return dict(sharpe=0, sortino=0, win_rate=0, profit_factor=0,
                    max_drawdown=0, total_return=0, num_trades=n)

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / n
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (
        999.0 if len(wins) else 0)
    mean_r = returns.mean()
    std_r = returns.std(ddof=1) if n > 1 else 1e-9
    down_std = losses.std(ddof=1) if len(losses) > 1 else 1e-9

    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 1e-12 else 0
    sortino = (mean_r / down_std) * np.sqrt(252) if down_std > 1e-12 else 0

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0
    total_ret = float(cum[-1] - 1) if len(cum) else 0

    return dict(sharpe=round(sharpe, 4), sortino=round(sortino, 4),
                win_rate=round(wr, 4), profit_factor=round(pf, 4),
                max_drawdown=round(max_dd, 4), total_return=round(total_ret, 4),
                num_trades=n)


def permutation_test(trade_returns, entry_dates, exit_dates, prices, n_perms=N_PERMS):
    """
    Random-timing permutation test: for each permutation, pick N random
    (ticker, entry) pairs from the universe on random trading days, apply
    the same median hold period, compute Sharpe. p-value = fraction of
    random Sharpes >= observed Sharpe.
    """
    n = len(trade_returns)
    if n < 5:
        return 1.0
    obs_sharpe = calc_metrics(trade_returns)["sharpe"]

    # Compute median hold length in trading days
    hold_days = []
    for ed, xd in zip(entry_dates, exit_dates):
        hold_days.append(max((xd - ed).days, 1))
    median_hold = int(np.median(hold_days))

    # Build a pool of (ticker, all_trading_days) for random sampling
    ticker_days = {}
    for t in UNIVERSE:
        if t in prices:
            idx = prices[t].loc[START:END].index
            if len(idx) > median_hold + 5:
                ticker_days[t] = idx
    if not ticker_days:
        return 1.0
    tickers_list = list(ticker_days.keys())

    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_perms):
        perm_rets = []
        for _ in range(n):
            t = tickers_list[rng.integers(0, len(tickers_list))]
            idx = ticker_days[t]
            max_start = len(idx) - median_hold - 1
            if max_start <= 0:
                continue
            si = rng.integers(0, max_start)
            ei = si + median_hold
            ep = float(prices[t].loc[idx[si], "Close"])
            xp = float(prices[t].loc[idx[ei], "Close"])
            if ep > 0:
                perm_rets.append(apply_slippage_ret((xp / ep) - 1))
        if len(perm_rets) >= 5:
            s = calc_metrics(np.array(perm_rets))["sharpe"]
            if s >= obs_sharpe:
                count_ge += 1
    return round(count_ge / n_perms, 4)


def regime_split(trade_returns, entry_dates, spy_df):
    """Split returns into bull/bear arrays."""
    bull, bear = [], []
    for r, d in zip(trade_returns, entry_dates):
        reg = get_regime(d, spy_df)
        if reg == "bull":
            bull.append(r)
        else:
            bear.append(r)
    return np.array(bull), np.array(bear)


def five_gate(metrics, perm_p, regime_gap):
    g = {}
    g["sharpe_gt_0.5"] = bool(metrics["sharpe"] > 0.5)
    g["perm_p_lt_0.05"] = bool(perm_p < 0.05)
    g["regime_gap_lt_0.5"] = bool(regime_gap < 0.5)
    g["max_dd_gt_neg50"] = bool(metrics["max_drawdown"] > -0.50)
    g["min_20_trades"] = bool(metrics["num_trades"] >= 20)
    g["all_passed"] = all(g.values())
    return g


# ── PORTFOLIO-LEVEL BACKTEST ENGINE ──────────────────────────────────────
def simulate_portfolio(signals, prices):
    """
    signals: list of (ticker, entry_date, exit_date)
    Returns: (np.array of per-trade returns, list of entry_dates, list of exit_dates)
    Respects capital, max concurrent, position sizing.
    """
    signals = sorted(signals, key=lambda x: x[1])
    capital = CAPITAL
    active = []  # (ticker, entry_date, exit_date, shares, entry_price)
    trade_rets = []
    entry_dates = []
    exit_dates = []

    all_dates = sorted(set(
        [s[1] for s in signals] + [s[2] for s in signals]
    ))
    if not all_dates:
        return np.array([]), [], []

    sig_by_date = {}
    for s in signals:
        sig_by_date.setdefault(s[1], []).append(s)

    for day in pd.DatetimeIndex(sorted(set(all_dates))):
        # Close positions whose exit_date <= today
        still_active = []
        for (t, ed, xd, shares, ep) in active:
            if day >= xd:
                if t in prices and xd in prices[t].index:
                    exit_price = float(prices[t].loc[xd, "Close"])
                    raw_ret = (exit_price / ep) - 1
                    trade_rets.append(apply_slippage_ret(raw_ret))
                    entry_dates.append(ed)
                    exit_dates.append(xd)
                    capital += shares * exit_price
                else:
                    capital += shares * ep  # flat if no exit price
            else:
                still_active.append((t, ed, xd, shares, ep))
        active = still_active

        # Open new positions scheduled for today
        if day in sig_by_date:
            for (t, ed, xd) in sig_by_date[day]:
                if len(active) >= MAX_CONCURRENT:
                    break
                if t not in prices or ed not in prices[t].index:
                    continue
                entry_price = float(prices[t].loc[ed, "Close"])
                shares = position_shares(entry_price, capital)
                if shares <= 0:
                    continue
                capital -= shares * entry_price
                active.append((t, ed, xd, shares, entry_price))

    # Force-close any remaining
    for (t, ed, xd, shares, ep) in active:
        if t in prices and len(prices[t]) > 0:
            exit_price = float(prices[t].iloc[-1]["Close"])
            raw_ret = (exit_price / ep) - 1
            trade_rets.append(apply_slippage_ret(raw_ret))
            entry_dates.append(ed)
            exit_dates.append(prices[t].index[-1])

    return np.array(trade_rets), entry_dates, exit_dates


# ── SIGNAL GENERATORS ────────────────────────────────────────────────────

def gen_A_turn_of_month(prices, spy_idx):
    """Buy last 2 trading days of month, sell on 3rd trading day of next month."""
    signals = []
    td = spy_idx  # use SPY trading calendar
    grouped = {}
    for d in td:
        key = (d.year, d.month)
        grouped.setdefault(key, []).append(d)

    sorted_keys = sorted(grouped.keys())
    for i, key in enumerate(sorted_keys):
        days = sorted(grouped[key])
        if len(days) < 2:
            continue
        entry_day = days[-2]  # 2nd-to-last trading day

        # Find next month
        if i + 1 >= len(sorted_keys):
            continue
        next_key = sorted_keys[i + 1]
        next_days = sorted(grouped[next_key])
        if len(next_days) < 3:
            continue
        exit_day = next_days[2]  # 3rd trading day of next month

        for t in UNIVERSE:
            if t in prices and entry_day in prices[t].index and exit_day in prices[t].index:
                signals.append((t, entry_day, exit_day))
    return signals


def gen_B_monday_reversal(prices, spy_idx):
    """Buy Monday close if stock down >1% on Monday, sell Friday close."""
    signals = []
    for d in spy_idx:
        if d.dayofweek != 0:  # Monday
            continue
        # Find Friday of same week
        fri_target = d + pd.Timedelta(days=4)
        fri_candidates = spy_idx[(spy_idx >= d) & (spy_idx <= fri_target)]
        fri_candidates = fri_candidates[fri_candidates.dayofweek == 4]
        if len(fri_candidates) == 0:
            continue
        exit_day = fri_candidates[0]

        for t in UNIVERSE:
            if t not in prices or d not in prices[t].index or exit_day not in prices[t].index:
                continue
            row = prices[t].loc[d]
            open_p = float(row["Open"])
            close_p = float(row["Close"])
            if open_p > 0 and (close_p - open_p) / open_p < -0.01:
                signals.append((t, d, exit_day))
    return signals


def gen_D_january_effect(prices, spy_idx):
    """Buy bottom 5 performers from prior year in first 5 days of Jan, hold 20 days."""
    signals = []
    years = sorted(set(spy_idx.year))
    for yr in years:
        if yr <= 2022:
            continue  # need full prior year in data
        # Prior year returns
        perf = {}
        for t in UNIVERSE:
            if t not in prices:
                continue
            df = prices[t]
            prev = df.index[(df.index.year == yr - 1)]
            if len(prev) < 20:
                continue
            perf[t] = float(df.loc[prev[-1], "Close"] / df.loc[prev[0], "Close"]) - 1
        if len(perf) < 5:
            continue
        bottom5 = sorted(perf, key=perf.get)[:5]

        jan_days = spy_idx[(spy_idx.year == yr) & (spy_idx.month == 1)]
        if len(jan_days) < 5:
            continue
        entry_day = jan_days[0]
        future = spy_idx[spy_idx >= entry_day]
        if len(future) < 21:
            continue
        exit_day = future[20]  # hold 20 trading days

        for t in bottom5:
            if t in prices and entry_day in prices[t].index and exit_day in prices[t].index:
                signals.append((t, entry_day, exit_day))
    return signals


def gen_E_quarter_end_rebalance(prices, spy_idx):
    """Buy stocks that dropped >5% in last 2 weeks of quarter, first week of new quarter, hold 10 days."""
    signals = []
    quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]

    for yr in sorted(set(spy_idx.year)):
        for qm, qd in quarter_ends:
            try:
                qe = pd.Timestamp(yr, qm, qd)
            except ValueError:
                qe = pd.Timestamp(yr, qm, 28)
            two_wk_before = qe - pd.Timedelta(days=14)
            q_tail = spy_idx[(spy_idx >= two_wk_before) & (spy_idx <= qe)]
            if len(q_tail) < 2:
                continue

            # First week of new quarter
            if qm == 12:
                nq_start = pd.Timestamp(yr + 1, 1, 1)
            else:
                nq_start = pd.Timestamp(yr, qm + 1, 1)
            nq_first_wk = spy_idx[(spy_idx >= nq_start) &
                                   (spy_idx < nq_start + pd.Timedelta(days=7))]
            if len(nq_first_wk) == 0:
                continue
            entry_day = nq_first_wk[0]

            future = spy_idx[spy_idx >= entry_day]
            if len(future) < 11:
                continue
            exit_day = future[10]

            for t in UNIVERSE:
                if t not in prices:
                    continue
                df = prices[t]
                qt = df.index.intersection(q_tail)
                if len(qt) < 2:
                    continue
                ret_qt = float(df.loc[qt[-1], "Close"] / df.loc[qt[0], "Close"]) - 1
                if ret_qt < -0.05:
                    if entry_day in df.index and exit_day in df.index:
                        signals.append((t, entry_day, exit_day))
    return signals


def gen_F_holiday_effect(prices, spy_idx):
    """Buy 2 days before major US holidays, sell day after."""
    signals = []
    years = sorted(set(spy_idx.year))

    for yr in years:
        holidays = []
        # Memorial Day: last Monday of May
        cal = calendar.monthcalendar(yr, 5)
        last_mon = [w[0] for w in cal if w[0] != 0][-1]
        holidays.append(pd.Timestamp(yr, 5, last_mon))
        # July 4th
        holidays.append(pd.Timestamp(yr, 7, 4))
        # Labor Day: first Monday of September
        cal = calendar.monthcalendar(yr, 9)
        first_mon = [w[0] for w in cal if w[0] != 0][0]
        holidays.append(pd.Timestamp(yr, 9, first_mon))
        # Thanksgiving: 4th Thursday of November
        cal = calendar.monthcalendar(yr, 11)
        thursdays = [w[3] for w in cal if w[3] != 0]
        if len(thursdays) >= 4:
            holidays.append(pd.Timestamp(yr, 11, thursdays[3]))
        # Christmas
        holidays.append(pd.Timestamp(yr, 12, 25))

        for hol in holidays:
            before = spy_idx[spy_idx < hol]
            if len(before) < 2:
                continue
            entry_day = before[-2]

            after = spy_idx[spy_idx > hol]
            if len(after) == 0:
                continue
            exit_day = after[0]

            for t in UNIVERSE:
                if t in prices and entry_day in prices[t].index and exit_day in prices[t].index:
                    signals.append((t, entry_day, exit_day))
    return signals


# ── EVALUATE ONE VARIANT ─────────────────────────────────────────────────
def evaluate_variant(name, signals, prices, spy_df):
    print(f"\n{'='*60}")
    print(f"Variant: {name}")
    print(f"{'='*60}")
    print(f"  Raw signals: {len(signals)}")

    # Filter to OOT period
    oot_start = pd.Timestamp(START)
    oot_end = pd.Timestamp(END)
    signals = [(t, e, x) for (t, e, x) in signals
               if e >= oot_start and x <= oot_end]
    print(f"  Signals in OOT window: {len(signals)}")

    trade_rets, entry_dates, exit_dates = simulate_portfolio(signals, prices)
    print(f"  Executed trades: {len(trade_rets)}")

    metrics = calc_metrics(trade_rets)

    # Regime
    if len(trade_rets) > 0:
        bull_rets, bear_rets = regime_split(trade_rets, entry_dates, spy_df)
        bull_m = calc_metrics(bull_rets)
        bear_m = calc_metrics(bear_rets)
        bull_sharpe = bull_m["sharpe"]
        bear_sharpe = bear_m["sharpe"]
        denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
        regime_gap = round(abs(bull_sharpe - bear_sharpe) / denom, 4)
    else:
        bull_sharpe = bear_sharpe = 0.0
        regime_gap = 0.0

    # Permutation (random-timing: random ticker + random entry, same hold length)
    print(f"  Permutation test ({N_PERMS} shuffles) …")
    perm_p = permutation_test(trade_rets, entry_dates, exit_dates, prices, N_PERMS)

    gates = five_gate(metrics, perm_p, regime_gap)

    result = {
        **metrics,
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "regime_gap": regime_gap,
        "permutation_p": perm_p,
        "five_gate": gates,
    }

    print(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
          f"WR={metrics['win_rate']:.1%}  PF={metrics['profit_factor']:.2f}  "
          f"MaxDD={metrics['max_drawdown']:.1%}  Return={metrics['total_return']:.1%}")
    print(f"  Bull Sharpe={bull_sharpe:.3f}  Bear Sharpe={bear_sharpe:.3f}  "
          f"Regime Gap={regime_gap:.3f}")
    print(f"  Perm p={perm_p:.4f}")
    gate_str = "PASS" if gates["all_passed"] else "FAIL"
    fails = [k for k, v in gates.items() if k != "all_passed" and not v]
    print(f"  5-Gate: {gate_str}" + (f"  (failed: {', '.join(fails)})" if fails else ""))

    return result


# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    prices = download_data()

    spy = prices["SPY"].copy()
    spy["SMA200"] = spy["Close"].rolling(SMA_WINDOW).mean()
    spy["regime"] = np.where(spy["Close"] > spy["SMA200"], "bull", "bear")

    spy_idx = spy.loc[DATA_START:].index  # full index for signal generation

    # Generate signals for each variant
    variant_generators = {
        "A_turn_of_month": gen_A_turn_of_month,
        "B_monday_reversal": gen_B_monday_reversal,
        "D_january_effect": gen_D_january_effect,
        "E_quarter_end_rebalance": gen_E_quarter_end_rebalance,
        "F_holiday_effect": gen_F_holiday_effect,
    }

    results = {}
    for name, gen_fn in variant_generators.items():
        sigs = gen_fn(prices, spy_idx)
        results[name] = evaluate_variant(name, sigs, prices, spy)

    # Variant C — skipped with explanation
    results["C_pre_earnings_drift"] = {
        "skipped": True,
        "reason": (
            "yfinance earnings_dates only returns future/recent dates reliably. "
            "Historical earnings dates for 2022-2025 are incomplete or missing "
            "for most tickers. Approximating with fixed quarterly weeks (3-4 of "
            "Jan/Apr/Jul/Oct) introduces 1-3 week timing error per company, "
            "which would corrupt the backtest. A proper implementation requires "
            "a dedicated earnings calendar (Wall Street Horizon, SEC 8-K scrape, "
            "or a premium data vendor)."
        ),
        "sharpe": None, "sortino": None, "win_rate": None,
        "profit_factor": None, "max_drawdown": None, "total_return": None,
        "num_trades": 0, "bull_sharpe": None, "bear_sharpe": None,
        "regime_gap": None, "permutation_p": None,
        "five_gate": {"all_passed": False, "reason": "skipped"},
    }

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} " \
             f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'Ret%':>7} {'PermP':>6} " \
             f"{'RGap':>5} {'Gate':>5}"
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        if r.get("skipped"):
            print(f"  {name:<28} SKIPPED")
            continue
        gate_str = "PASS" if r["five_gate"]["all_passed"] else "FAIL"
        print(f"  {name:<28} {r['num_trades']:>6} {r['sharpe']:>7.3f} "
              f"{r['sortino']:>8.3f} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
              f"{r['max_drawdown']:>6.1%} {r['total_return']:>6.1%} "
              f"{r['permutation_p']:>6.4f} {r['regime_gap']:>5.3f} {gate_str:>5}")

    # Dollar P&L
    print(f"\nDollar P&L on ${CAPITAL:.0f} account:")
    for name, r in results.items():
        if r.get("skipped"):
            continue
        dollar = CAPITAL * r["total_return"]
        print(f"  {name:<28} ${dollar:>+8.2f}")

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "universe": UNIVERSE,
            "start": START, "end": END,
            "capital": CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "slippage_bps": SLIPPAGE_BPS,
            "permutation_iterations": N_PERMS,
            "sma_window": SMA_WINDOW,
            "run_date": datetime.now().isoformat(),
        },
        "variants": results,
        "summary": {
            "total_variants_tested": sum(1 for r in results.values() if not r.get("skipped")),
            "passed_all_gates": sum(1 for r in results.values()
                                    if not r.get("skipped") and r["five_gate"]["all_passed"]),
            "best_sharpe_variant": max(
                ((n, r["sharpe"]) for n, r in results.items() if not r.get("skipped")),
                key=lambda x: x[1], default=("none", 0)
            )[0],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
