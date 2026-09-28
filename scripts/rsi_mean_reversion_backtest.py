#!/usr/bin/env python3
"""
RSI Mean Reversion on Growth Stocks — 6 Variant Backtest
Academic basis: Jegadeesh (1990), Lehmann (1990) — short-term reversal effect.

Variants:
  A. RSI(2) Bounce
  B. RSI(5) Mean Reversion
  C. 3-Day Decline
  D. Bollinger Bounce
  E. Portfolio RSI (weekly rotation)
  F. VIX-Filtered RSI

Universe: 24 growth/mega-cap stocks
OOT: Jan 2022 – Jul 2026
Capital: $645, $0 commission (Robinhood), 0.02% slippage
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "CRM", "ADBE", "NFLX", "AVGO", "COST", "PEP", "LLY", "UNH",
    "V", "MA", "JPM", "HD", "INTC", "MU", "QCOM", "PYPL",
]
START = "2021-06-01"   # extra lookback for 200-SMA
END   = "2026-07-30"
OOT_START = "2022-01-01"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0


# ── Data Download ──────────────────────────────────────────────────────────
print("Downloading price data …")
raw = yf.download(UNIVERSE + ["SPY", "^VIX"], start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)

def get_close(ticker):
    """Extract close series for a ticker from multi-level download."""
    try:
        s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)

closes = {t: get_close(t) for t in UNIVERSE}
spy_close = get_close("SPY")
vix_close = get_close("^VIX")

print(f"  SPY rows: {len(spy_close)}, VIX rows: {len(vix_close)}")
print(f"  Stocks with data: {sum(1 for v in closes.values() if len(v) > 200)}/{len(UNIVERSE)}")


# ── Indicator Helpers ──────────────────────────────────────────────────────
def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def sma(series, period):
    return series.rolling(period).mean()

def bollinger(series, period=20, num_std=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    return mid - num_std * std, mid, mid + num_std * std


# ── Pre-compute indicators for all stocks ──────────────────────────────────
indicators = {}
for t in UNIVERSE:
    c = closes[t]
    if len(c) < 250:
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["sma200"] = sma(c, 200)
    ind["rsi2"]  = rsi(c, 2)
    ind["rsi5"]  = rsi(c, 5)
    ind["rsi14"] = rsi(c, 14)
    ind["ret1"]  = c.pct_change()
    ind["ret3"]  = c.pct_change(3)
    bb_low, bb_mid, bb_up = bollinger(c)
    ind["bb_low"] = bb_low
    ind["bb_mid"] = bb_mid
    ind["above_200sma"] = c > ind["sma200"]
    indicators[t] = ind.dropna(subset=["sma200"])

# SPY regime
spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(spy_close.index).fillna(False)  # True = Bull


# ── Trade Simulator ────────────────────────────────────────────────────────
class Trade:
    __slots__ = ["ticker", "entry_date", "entry_price", "exit_date",
                 "exit_price", "shares", "pnl", "regime"]

def simulate_trades(signals, max_hold, exit_fn=None):
    """
    signals: list of (date, ticker, entry_price)
    max_hold: max holding days
    exit_fn: optional function(ticker, entry_idx, ind_df) -> exit_idx
    Returns list of Trade objects.
    """
    trades = []
    for date, ticker, entry_price in signals:
        ind = indicators.get(ticker)
        if ind is None:
            continue
        if date not in ind.index:
            continue
        loc = ind.index.get_loc(date)
        # apply slippage on entry
        actual_entry = entry_price * (1 + SLIPPAGE_PCT)

        exit_idx = None
        if exit_fn is not None:
            exit_idx = exit_fn(ticker, loc, ind)
        if exit_idx is None or exit_idx > loc + max_hold:
            exit_idx = min(loc + max_hold, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        exit_price = ind["close"].iloc[exit_idx] * (1 - SLIPPAGE_PCT)
        t = Trade()
        t.ticker = ticker
        t.entry_date = str(ind.index[loc].date())
        t.entry_price = actual_entry
        t.exit_date = str(ind.index[exit_idx].date())
        t.exit_price = exit_price
        t.shares = 0  # filled later by portfolio sim
        t.pnl = 0
        # regime at entry
        entry_dt = ind.index[loc]
        t.regime = "Bull" if spy_regime.get(entry_dt, False) else "Bear"
        trades.append(t)
    return trades


def backtest_trades(trades, capital=CAPITAL):
    """
    Size each trade to use full available capital (single position at a time).
    Returns equity curve and stats.
    """
    if not trades:
        return None
    trades_sorted = sorted(trades, key=lambda t: t.entry_date)

    equity = capital
    equity_curve = [(OOT_START, capital)]
    current_exit = None
    realized = []

    for t in trades_sorted:
        if t.entry_date < OOT_START:
            continue
        # skip if we're in a position
        if current_exit is not None and t.entry_date < current_exit:
            continue
        shares = int(equity / t.entry_price)
        if shares < 1:
            continue
        t.shares = shares
        pnl = shares * (t.exit_price - t.entry_price)
        t.pnl = pnl
        equity += pnl
        current_exit = t.exit_date
        equity_curve.append((t.exit_date, equity))
        realized.append(t)

    return realized, equity_curve


def backtest_portfolio_trades(weekly_picks, capital=CAPITAL):
    """
    For variant E: weekly rotation, equal-weight up to 3 positions.
    weekly_picks: list of (week_start_date, [(ticker, entry_price), ...])
    """
    if not weekly_picks:
        return None
    equity = capital
    equity_curve = [(OOT_START, capital)]
    realized = []

    for week_date, picks in sorted(weekly_picks):
        if str(week_date) < OOT_START:
            continue
        if not picks:
            continue
        n = min(3, len(picks))
        alloc = equity / n
        week_pnl = 0
        for ticker, entry_price in picks[:n]:
            ind = indicators.get(ticker)
            if ind is None:
                continue
            if week_date not in ind.index:
                continue
            loc = ind.index.get_loc(week_date)
            exit_idx = min(loc + 5, len(ind) - 1)
            actual_entry = entry_price * (1 + SLIPPAGE_PCT)
            exit_price = ind["close"].iloc[exit_idx] * (1 - SLIPPAGE_PCT)
            shares = int(alloc / actual_entry)
            if shares < 1:
                continue
            pnl = shares * (exit_price - actual_entry)
            t = Trade()
            t.ticker = ticker
            t.entry_date = str(week_date)
            t.entry_price = actual_entry
            t.exit_date = str(ind.index[exit_idx].date())
            t.exit_price = exit_price
            t.shares = shares
            t.pnl = pnl
            entry_dt = ind.index[loc]
            t.regime = "Bull" if spy_regime.get(entry_dt, False) else "Bear"
            realized.append(t)
            week_pnl += pnl
        equity += week_pnl
        equity_curve.append((str(week_date), equity))

    return realized, equity_curve


# ── Signal Generation per Variant ──────────────────────────────────────────
def gen_signals_A():
    """RSI(2) Bounce: RSI(2)<10 AND above 200-SMA."""
    signals = []
    for t, ind in indicators.items():
        mask = (ind["rsi2"] < 10) & ind["above_200sma"]
        for dt in ind.index[mask]:
            signals.append((dt, t, ind.loc[dt, "close"]))
    return signals

def exit_A(ticker, loc, ind):
    """Exit when RSI(2) > 60."""
    for i in range(loc+1, min(loc+11, len(ind))):
        if ind["rsi2"].iloc[i] > 60:
            return i
    return None

def gen_signals_B():
    """RSI(5) Mean Reversion: RSI(5)<20 AND above 200-SMA."""
    signals = []
    for t, ind in indicators.items():
        mask = (ind["rsi5"] < 20) & ind["above_200sma"]
        for dt in ind.index[mask]:
            signals.append((dt, t, ind.loc[dt, "close"]))
    return signals

def exit_B(ticker, loc, ind):
    """Exit when RSI(5) > 50."""
    for i in range(loc+1, min(loc+11, len(ind))):
        if ind["rsi5"].iloc[i] > 50:
            return i
    return None

def gen_signals_C():
    """3-Day Decline: 3 consecutive down days, total >5%, above 200-SMA."""
    signals = []
    for t, ind in indicators.items():
        ret1 = ind["ret1"]
        for i in range(3, len(ind)):
            if (ret1.iloc[i] < 0 and ret1.iloc[i-1] < 0 and ret1.iloc[i-2] < 0
                    and ind["ret3"].iloc[i] < -0.05
                    and ind["above_200sma"].iloc[i]):
                signals.append((ind.index[i], t, ind["close"].iloc[i]))
    return signals

def gen_signals_D():
    """Bollinger Bounce: price <= lower BB AND RSI(14)<30 AND above 200-SMA."""
    signals = []
    for t, ind in indicators.items():
        mask = ((ind["close"] <= ind["bb_low"]) &
                (ind["rsi14"] < 30) &
                ind["above_200sma"])
        for dt in ind.index[mask]:
            signals.append((dt, t, ind.loc[dt, "close"]))
    return signals

def exit_D(ticker, loc, ind):
    """Exit when price >= bb_mid."""
    for i in range(loc+1, min(loc+11, len(ind))):
        if ind["close"].iloc[i] >= ind["bb_mid"].iloc[i]:
            return i
    return None

def gen_signals_E():
    """Portfolio RSI: weekly, pick top-3 lowest RSI(5) if <30 and above 200-SMA."""
    # Build weekly dates (Mondays)
    all_dates = sorted(set().union(*(ind.index for ind in indicators.values())))
    all_dates = [d for d in all_dates if str(d.date()) >= OOT_START]
    weeks = {}
    for d in all_dates:
        wk = d.isocalendar()[1]
        yr = d.year
        key = (yr, wk)
        if key not in weeks:
            weeks[key] = d  # first trading day of each week

    weekly_picks = []
    for key in sorted(weeks.keys()):
        day = weeks[key]
        candidates = []
        for t, ind in indicators.items():
            if day not in ind.index:
                continue
            r5 = ind.loc[day, "rsi5"]
            above = ind.loc[day, "above_200sma"]
            if isinstance(r5, pd.Series):
                r5 = r5.iloc[0]
            if isinstance(above, pd.Series):
                above = above.iloc[0]
            if r5 < 30 and above:
                candidates.append((r5, t, ind.loc[day, "close"]))
        candidates.sort()  # lowest RSI first
        picks = [(t, p if not isinstance(p, pd.Series) else p.iloc[0])
                 for _, t, p in candidates[:3]]
        if picks:
            weekly_picks.append((day, picks))

    return weekly_picks

def gen_signals_F():
    """VIX-Filtered RSI: same as A but only when VIX < 25."""
    signals = []
    for t, ind in indicators.items():
        mask = (ind["rsi2"] < 10) & ind["above_200sma"]
        for dt in ind.index[mask]:
            vix_val = vix_close.get(dt, None)
            if vix_val is not None and vix_val < 25:
                signals.append((dt, t, ind.loc[dt, "close"]))
    return signals


# ── Metrics Calculation ────────────────────────────────────────────────────
def calc_metrics(realized, equity_curve, label):
    if not realized:
        return {"variant": label, "status": "NO TRADES", "n_trades": 0}

    returns = [t.pnl / (t.shares * t.entry_price) for t in realized if t.shares > 0]
    if not returns:
        return {"variant": label, "status": "NO TRADES", "n_trades": 0}

    returns = np.array(returns)
    n = len(returns)
    wins = (returns > 0).sum()
    wr = wins / n

    # Annualize: assume ~5-day avg hold → ~50 trades/yr
    avg_hold = np.mean([
        (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
        for t in realized
    ])
    trades_per_year = max(1, 252 / max(avg_hold, 1))
    ann_factor = np.sqrt(trades_per_year)

    mean_r = returns.mean()
    std_r = returns.std() if returns.std() > 0 else 1e-9
    sharpe = (mean_r / std_r) * ann_factor

    downside = returns[returns < 0]
    down_std = downside.std() if len(downside) > 0 and downside.std() > 0 else 1e-9
    sortino = (mean_r / down_std) * ann_factor

    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    # Equity curve drawdown
    eq_vals = [e[1] for e in equity_curve]
    peak = eq_vals[0]
    max_dd = 0
    for v in eq_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    final_eq = eq_vals[-1]
    total_return = (final_eq - CAPITAL) / CAPITAL

    # Regime split
    bull_r = [t.pnl / (t.shares * t.entry_price) for t in realized if t.regime == "Bull" and t.shares > 0]
    bear_r = [t.pnl / (t.shares * t.entry_price) for t in realized if t.regime == "Bear" and t.shares > 0]

    def regime_sharpe(rets):
        if len(rets) < 2:
            return 0.0
        r = np.array(rets)
        s = r.std()
        if s < 1e-12:
            return 0.0
        return (r.mean() / s) * ann_factor

    sh_bull = regime_sharpe(bull_r)
    sh_bear = regime_sharpe(bear_r)
    denom = max(abs(sh_bull), abs(sh_bear), 1e-9)
    regime_gap = abs(sh_bull - sh_bear) / denom

    return {
        "variant": label,
        "n_trades": n,
        "win_rate": round(wr, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "final_equity": round(final_eq, 2),
        "avg_hold_days": round(avg_hold, 1),
        "sharpe_bull": round(sh_bull, 3),
        "sharpe_bear": round(sh_bear, 3),
        "regime_gap": round(regime_gap, 4),
        "bull_trades": len(bull_r),
        "bear_trades": len(bear_r),
    }


# ── Permutation Test ───────────────────────────────────────────────────────
def permutation_test(returns, n_iter=1000):
    if len(returns) < 5:
        return 1.0
    obs_mean = np.mean(returns)
    count = 0
    for _ in range(n_iter):
        perm = returns * np.random.choice([-1, 1], size=len(returns))
        if np.mean(perm) >= obs_mean:
            count += 1
    return count / n_iter


# ── 5-Gate Validation ──────────────────────────────────────────────────────
def validate_5gate(metrics, pval):
    gates = {}
    gates["sharpe_gt_0.5"] = metrics.get("sharpe", 0) > 0.5
    gates["perm_p_lt_0.05"] = pval < 0.05
    gates["regime_gap_lt_0.5"] = metrics.get("regime_gap", 1) < 0.5
    gates["maxdd_gt_neg50"] = metrics.get("max_dd", -1) > -0.50
    gates["min_20_trades"] = metrics.get("n_trades", 0) >= 20
    gates["pass_all"] = all(gates.values())
    return gates


# ── Run All Variants ───────────────────────────────────────────────────────
results = {}

print("\n=== Variant A: RSI(2) Bounce ===")
sig_a = gen_signals_A()
trades_a = simulate_trades(sig_a, max_hold=10, exit_fn=exit_A)
res_a = backtest_trades(trades_a)
if res_a:
    real_a, eq_a = res_a
    m_a = calc_metrics(real_a, eq_a, "A_RSI2_Bounce")
    rets_a = np.array([t.pnl/(t.shares*t.entry_price) for t in real_a if t.shares>0])
    p_a = permutation_test(rets_a)
    m_a["perm_pval"] = round(p_a, 4)
    m_a["gates"] = validate_5gate(m_a, p_a)
    results["A"] = m_a
    print(f"  Trades: {m_a['n_trades']}, Sharpe: {m_a['sharpe']}, WR: {m_a['win_rate']}, "
          f"MaxDD: {m_a['max_dd']}, Return: {m_a['total_return']:.1%}")
else:
    print("  No trades generated.")
    results["A"] = {"variant": "A_RSI2_Bounce", "status": "NO TRADES", "n_trades": 0}

print("\n=== Variant B: RSI(5) Mean Reversion ===")
sig_b = gen_signals_B()
trades_b = simulate_trades(sig_b, max_hold=10, exit_fn=exit_B)
res_b = backtest_trades(trades_b)
if res_b:
    real_b, eq_b = res_b
    m_b = calc_metrics(real_b, eq_b, "B_RSI5_MeanRev")
    rets_b = np.array([t.pnl/(t.shares*t.entry_price) for t in real_b if t.shares>0])
    p_b = permutation_test(rets_b)
    m_b["perm_pval"] = round(p_b, 4)
    m_b["gates"] = validate_5gate(m_b, p_b)
    results["B"] = m_b
    print(f"  Trades: {m_b['n_trades']}, Sharpe: {m_b['sharpe']}, WR: {m_b['win_rate']}, "
          f"MaxDD: {m_b['max_dd']}, Return: {m_b['total_return']:.1%}")
else:
    print("  No trades generated.")
    results["B"] = {"variant": "B_RSI5_MeanRev", "status": "NO TRADES", "n_trades": 0}

print("\n=== Variant C: 3-Day Decline ===")
sig_c = gen_signals_C()
trades_c = simulate_trades(sig_c, max_hold=5)
res_c = backtest_trades(trades_c)
if res_c:
    real_c, eq_c = res_c
    m_c = calc_metrics(real_c, eq_c, "C_3DayDecline")
    rets_c = np.array([t.pnl/(t.shares*t.entry_price) for t in real_c if t.shares>0])
    p_c = permutation_test(rets_c)
    m_c["perm_pval"] = round(p_c, 4)
    m_c["gates"] = validate_5gate(m_c, p_c)
    results["C"] = m_c
    print(f"  Trades: {m_c['n_trades']}, Sharpe: {m_c['sharpe']}, WR: {m_c['win_rate']}, "
          f"MaxDD: {m_c['max_dd']}, Return: {m_c['total_return']:.1%}")
else:
    print("  No trades generated.")
    results["C"] = {"variant": "C_3DayDecline", "status": "NO TRADES", "n_trades": 0}

print("\n=== Variant D: Bollinger Bounce ===")
sig_d = gen_signals_D()
trades_d = simulate_trades(sig_d, max_hold=10, exit_fn=exit_D)
res_d = backtest_trades(trades_d)
if res_d:
    real_d, eq_d = res_d
    m_d = calc_metrics(real_d, eq_d, "D_BollingerBounce")
    rets_d = np.array([t.pnl/(t.shares*t.entry_price) for t in real_d if t.shares>0])
    p_d = permutation_test(rets_d)
    m_d["perm_pval"] = round(p_d, 4)
    m_d["gates"] = validate_5gate(m_d, p_d)
    results["D"] = m_d
    print(f"  Trades: {m_d['n_trades']}, Sharpe: {m_d['sharpe']}, WR: {m_d['win_rate']}, "
          f"MaxDD: {m_d['max_dd']}, Return: {m_d['total_return']:.1%}")
else:
    print("  No trades generated.")
    results["D"] = {"variant": "D_BollingerBounce", "status": "NO TRADES", "n_trades": 0}

print("\n=== Variant E: Portfolio RSI (Weekly Rotation) ===")
weekly_e = gen_signals_E()
res_e = backtest_portfolio_trades(weekly_e)
if res_e:
    real_e, eq_e = res_e
    m_e = calc_metrics(real_e, eq_e, "E_PortfolioRSI")
    rets_e = np.array([t.pnl/(t.shares*t.entry_price) for t in real_e if t.shares>0])
    p_e = permutation_test(rets_e)
    m_e["perm_pval"] = round(p_e, 4)
    m_e["gates"] = validate_5gate(m_e, p_e)
    results["E"] = m_e
    print(f"  Trades: {m_e['n_trades']}, Sharpe: {m_e['sharpe']}, WR: {m_e['win_rate']}, "
          f"MaxDD: {m_e['max_dd']}, Return: {m_e['total_return']:.1%}")
else:
    print("  No trades generated.")
    results["E"] = {"variant": "E_PortfolioRSI", "status": "NO TRADES", "n_trades": 0}

print("\n=== Variant F: VIX-Filtered RSI ===")
sig_f = gen_signals_F()
trades_f = simulate_trades(sig_f, max_hold=10, exit_fn=exit_A)
res_f = backtest_trades(trades_f)
if res_f:
    real_f, eq_f = res_f
    m_f = calc_metrics(real_f, eq_f, "F_VIXFiltered_RSI")
    rets_f = np.array([t.pnl/(t.shares*t.entry_price) for t in real_f if t.shares>0])
    p_f = permutation_test(rets_f)
    m_f["perm_pval"] = round(p_f, 4)
    m_f["gates"] = validate_5gate(m_f, p_f)
    results["F"] = m_f
    print(f"  Trades: {m_f['n_trades']}, Sharpe: {m_f['sharpe']}, WR: {m_f['win_rate']}, "
          f"MaxDD: {m_f['max_dd']}, Return: {m_f['total_return']:.1%}")
else:
    print("  No trades generated.")
    results["F"] = {"variant": "F_VIXFiltered_RSI", "status": "NO TRADES", "n_trades": 0}


# ── Summary Table ──────────────────────────────────────────────────────────
print("\n" + "="*120)
print(f"{'Variant':<25} {'Trades':>7} {'WR':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>7} "
      f"{'MaxDD':>8} {'Return':>9} {'Final$':>9} {'PermP':>7} {'Gates':>7}")
print("-"*120)

for key in ["A", "B", "C", "D", "E", "F"]:
    m = results.get(key, {})
    if m.get("n_trades", 0) == 0:
        print(f"{m.get('variant','?'):<25} {'—':>7} {'—':>7} {'—':>8} {'—':>8} {'—':>7} "
              f"{'—':>8} {'—':>9} {'—':>9} {'—':>7} {'—':>7}")
        continue
    g = m.get("gates", {})
    passed = sum(1 for gk, gv in g.items() if gv and gk != "pass_all")
    gate_str = f"{passed}/5" + (" *" if g.get("pass_all") else "")
    print(f"{m['variant']:<25} {m['n_trades']:>7} {m['win_rate']:>7.1%} {m['sharpe']:>8.3f} "
          f"{m['sortino']:>8.3f} {m['profit_factor']:>7.2f} {m['max_dd']:>8.1%} "
          f"{m['total_return']:>9.1%} {m['final_equity']:>9.2f} {m['perm_pval']:>7.3f} {gate_str:>7}")

print("="*120)
print(f"\nStarting capital: ${CAPITAL:.0f} | OOT: {OOT_START} to {END}")
print(f"Costs: $0 commission (Robinhood), {SLIPPAGE_PCT:.2%} slippage")
print("Gate legend: Sharpe>0.5 | PermP<0.05 | RegimeGap<0.5 | MaxDD>-50% | Trades>=20")
print("  * = ALL 5 gates passed")

# Regime detail
print("\n--- Regime Breakdown ---")
print(f"{'Variant':<25} {'Bull#':>7} {'Bear#':>7} {'ShBull':>8} {'ShBear':>8} {'Gap':>7}")
print("-"*65)
for key in ["A", "B", "C", "D", "E", "F"]:
    m = results.get(key, {})
    if m.get("n_trades", 0) == 0:
        continue
    print(f"{m['variant']:<25} {m['bull_trades']:>7} {m['bear_trades']:>7} "
          f"{m['sharpe_bull']:>8.3f} {m['sharpe_bear']:>8.3f} {m['regime_gap']:>7.3f}")


# ── Save Results ───────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/rsi_mean_reversion_results.json")

# Convert gates booleans for JSON
for k, v in results.items():
    if "gates" in v:
        v["gates"] = {gk: bool(gv) for gk, gv in v["gates"].items()}

with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
