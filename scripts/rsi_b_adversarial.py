#!/usr/bin/env python3
"""
RSI(5) Mean Reversion (Variant B) — Adversarial Validation
6 adversarial tests + bonus decorrelation analysis.

Tests:
  1. Inverse Direction (buy RSI>80 overbought)
  2. Buy-and-Hold Comparison
  3. Random Timing Percentile (1000 perms)
  4. Remove Key Component (drop 200-SMA filter)
  5. Cost Sensitivity (slippage sweep)
  6. Sub-Period Stability (4 sub-periods)
  BONUS: SPY/QQQ correlation
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "CRM", "ADBE", "NFLX", "AVGO", "COST", "PEP", "LLY", "UNH",
    "V", "MA", "JPM", "HD", "INTC", "MU", "QCOM", "PYPL",
]
START = "2021-06-01"
END = "2026-07-30"
OOT_START = "2022-01-01"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(UNIVERSE + ["SPY", "QQQ"], start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in UNIVERSE}
spy_close = get_close("SPY")
qqq_close = get_close("QQQ")

print(f"  SPY rows: {len(spy_close)}, QQQ rows: {len(qqq_close)}")
print(f"  Stocks with data: {sum(1 for v in closes.values() if len(v) > 200)}/{len(UNIVERSE)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
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


# ── Pre-compute indicators ───────────────────────────────────────────────
indicators = {}
for t in UNIVERSE:
    c = closes[t]
    if len(c) < 250:
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["sma200"] = sma(c, 200)
    ind["rsi5"] = rsi(c, 5)
    ind["above_200sma"] = c > ind["sma200"]
    indicators[t] = ind.dropna(subset=["sma200"])

spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(spy_close.index).fillna(False)


# ── Core Backtest Engine ─────────────────────────────────────────────────
def exit_rsi5_gt50(ticker, loc, ind):
    """Exit when RSI(5) > 50 or max 10 days."""
    for i in range(loc + 1, min(loc + 11, len(ind))):
        if ind["rsi5"].iloc[i] > 50:
            return i
    return None


def exit_rsi5_lt50(ticker, loc, ind):
    """Exit for inverse: when RSI(5) < 50 or max 10 days."""
    for i in range(loc + 1, min(loc + 11, len(ind))):
        if ind["rsi5"].iloc[i] < 50:
            return i
    return None


def run_backtest(signals, exit_fn=exit_rsi5_gt50, capital=CAPITAL, slippage=SLIPPAGE_PCT):
    """Run backtest on signals. Returns (per-trade returns, equity_curve, trades_info)."""
    trades = []
    for date, ticker, entry_price in signals:
        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)
        actual_entry = entry_price * (1 + slippage)

        exit_idx = exit_fn(ticker, loc, ind) if exit_fn else None
        if exit_idx is None or exit_idx > loc + 10:
            exit_idx = min(loc + 10, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        exit_price = ind["close"].iloc[exit_idx] * (1 - slippage)
        hold_days = (ind.index[exit_idx] - ind.index[loc]).days
        trades.append({
            "ticker": ticker,
            "entry_date": str(ind.index[loc].date()),
            "exit_date": str(ind.index[exit_idx].date()),
            "entry_price": actual_entry,
            "exit_price": exit_price,
            "hold_days": hold_days,
            "regime": "Bull" if spy_regime.get(ind.index[loc], False) else "Bear",
        })

    if not trades:
        return np.array([]), [], []

    trades_sorted = sorted(trades, key=lambda t: t["entry_date"])
    equity = capital
    equity_curve = [(OOT_START, capital)]
    realized = []
    current_exit = None

    for t in trades_sorted:
        if t["entry_date"] < OOT_START:
            continue
        if current_exit is not None and t["entry_date"] < current_exit:
            continue
        shares = int(equity / t["entry_price"])
        if shares < 1:
            continue
        pnl = shares * (t["exit_price"] - t["entry_price"])
        ret = pnl / (shares * t["entry_price"])
        equity += pnl
        current_exit = t["exit_date"]
        equity_curve.append((t["exit_date"], equity))
        t["shares"] = shares
        t["pnl"] = pnl
        t["return"] = ret
        realized.append(t)

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


def calc_sharpe(returns, avg_hold_override=None):
    """Annualized Sharpe from per-trade returns."""
    if len(returns) < 2:
        return 0.0
    avg_hold = avg_hold_override if avg_hold_override else 8.0
    trades_per_year = max(1, 252 / max(avg_hold, 1))
    ann_factor = np.sqrt(trades_per_year)
    mean_r = returns.mean()
    std_r = returns.std()
    if std_r < 1e-12:
        return 0.0
    return float((mean_r / std_r) * ann_factor)


def calc_full_metrics(returns, equity_curve, realized):
    """Full metrics dict."""
    if len(returns) < 2:
        return {"n_trades": 0, "sharpe": 0.0}

    n = len(returns)
    wins = int((returns > 0).sum())
    wr = wins / n

    avg_hold = np.mean([t["hold_days"] for t in realized])
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
    total_return_pct = ((final_eq - CAPITAL) / CAPITAL) * 100

    return {
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "final_equity": round(float(final_eq), 2),
        "avg_hold_days": round(float(avg_hold), 1),
    }


# ── Signal Generators ─────────────────────────────────────────────────────
def gen_signals_B():
    """RSI(5) < 20 AND above 200-SMA (original strategy)."""
    signals = []
    for t, ind in indicators.items():
        mask = (ind["rsi5"] < 20) & ind["above_200sma"]
        for dt in ind.index[mask]:
            signals.append((dt, t, ind.loc[dt, "close"]))
    return signals


def gen_signals_inverse():
    """RSI(5) > 80 AND above 200-SMA (buy overbought)."""
    signals = []
    for t, ind in indicators.items():
        mask = (ind["rsi5"] > 80) & ind["above_200sma"]
        for dt in ind.index[mask]:
            signals.append((dt, t, ind.loc[dt, "close"]))
    return signals


def gen_signals_no_sma():
    """RSI(5) < 20 WITHOUT the 200-SMA filter."""
    signals = []
    for t, ind in indicators.items():
        mask = ind["rsi5"] < 20
        for dt in ind.index[mask]:
            signals.append((dt, t, ind.loc[dt, "close"]))
    return signals


# ══════════════════════════════════════════════════════════════════════════
# Run baseline first
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RUNNING BASELINE: RSI(5) Mean Reversion (Variant B)")
print("=" * 70)

signals_B = gen_signals_B()
returns_B, eq_B, trades_B = run_backtest(signals_B)
metrics_B = calc_full_metrics(returns_B, eq_B, trades_B)
sharpe_B = metrics_B["sharpe"]

print(f"  Trades: {metrics_B['n_trades']}, Sharpe: {sharpe_B:.3f}, "
      f"WR: {metrics_B['win_rate']:.1%}, Return: {metrics_B['total_return_pct']:.1f}%")

results = {
    "baseline": metrics_B,
    "tests": {},
    "bonus": {},
    "summary": {},
}

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Inverse Direction (buy RSI > 80)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: INVERSE DIRECTION — Buy RSI(5) > 80 overbought breakouts")
print("=" * 70)

signals_inv = gen_signals_inverse()
returns_inv, eq_inv, trades_inv = run_backtest(signals_inv, exit_fn=exit_rsi5_lt50)
metrics_inv = calc_full_metrics(returns_inv, eq_inv, trades_inv)
sharpe_inv = metrics_inv["sharpe"]

test1_pass = sharpe_inv < 0.0
print(f"  Inverse trades: {metrics_inv['n_trades']}, Sharpe: {sharpe_inv:.3f}")
print(f"  PASS condition: inverse Sharpe < 0 → {'PASS' if test1_pass else 'FAIL'}")

results["tests"]["1_inverse_direction"] = {
    "description": "Buy RSI(5)>80 overbought instead of RSI(5)<20 oversold",
    "inverse_sharpe": round(sharpe_inv, 3),
    "inverse_metrics": metrics_inv,
    "original_sharpe": round(sharpe_B, 3),
    "pass_condition": "inverse Sharpe < 0",
    "passed": test1_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Buy-and-Hold Comparison
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: BUY-AND-HOLD COMPARISON — Equal-weight 24 stocks")
print("=" * 70)

# Build daily returns for equal-weight portfolio of the 24 stocks
oot_mask = spy_close.index >= OOT_START
oot_dates = spy_close.index[oot_mask]

# Compute daily returns for each stock, then average
daily_rets_list = []
for t in UNIVERSE:
    c = closes[t]
    if len(c) < 100:
        continue
    dr = c.pct_change().reindex(oot_dates).fillna(0)
    daily_rets_list.append(dr)

bh_daily_returns = pd.concat(daily_rets_list, axis=1).mean(axis=1)
bh_cumulative = (1 + bh_daily_returns).cumprod()
bh_final_return = float(bh_cumulative.iloc[-1] - 1)

# Buy-and-hold Sharpe (daily returns annualized)
bh_mean = float(bh_daily_returns.mean())
bh_std = float(bh_daily_returns.std())
bh_sharpe = (bh_mean / bh_std) * np.sqrt(252) if bh_std > 1e-12 else 0.0

sharpe_advantage = sharpe_B - bh_sharpe
test2_pass = sharpe_advantage >= 0.3

print(f"  Buy-and-hold Sharpe: {bh_sharpe:.3f}, Return: {bh_final_return:.1%}")
print(f"  Strategy Sharpe: {sharpe_B:.3f}")
print(f"  Sharpe advantage: {sharpe_advantage:.3f}")
print(f"  PASS condition: advantage >= 0.3 → {'PASS' if test2_pass else 'FAIL'}")

results["tests"]["2_buy_and_hold"] = {
    "description": "Compare to equal-weight buy-and-hold of same 24 stocks",
    "bh_sharpe": round(bh_sharpe, 3),
    "bh_total_return_pct": round(bh_final_return * 100, 2),
    "strategy_sharpe": round(sharpe_B, 3),
    "sharpe_advantage": round(sharpe_advantage, 3),
    "pass_condition": "strategy beats buy-and-hold by >= 0.3 Sharpe",
    "passed": test2_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Random Timing Percentile (1000 perms)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: RANDOM TIMING PERCENTILE — 1000 permutations")
print("=" * 70)

N_PERM = 1000

# Pre-compute valid OOT dates per stock
oot_dates_per_stock = {}
for t, ind in indicators.items():
    valid = ind.index[ind.index >= OOT_START]
    if len(valid) > 15:
        oot_dates_per_stock[t] = valid[:-10]

# Extract trade specs from real signals (only OOT)
real_signal_specs = []
for date, ticker, entry_price in signals_B:
    if str(date.date()) >= OOT_START and ticker in oot_dates_per_stock:
        real_signal_specs.append(ticker)

perm_sharpes = np.zeros(N_PERM)
for i in range(N_PERM):
    if (i + 1) % 200 == 0:
        print(f"  Permutation {i+1}/{N_PERM} ...")

    shuffled_signals = []
    for ticker in real_signal_specs:
        valid_dates = oot_dates_per_stock[ticker]
        rand_idx = np.random.randint(0, len(valid_dates))
        rand_date = valid_dates[rand_idx]
        ind = indicators[ticker]
        entry_price = ind.loc[rand_date, "close"]
        if isinstance(entry_price, pd.Series):
            entry_price = entry_price.iloc[0]
        shuffled_signals.append((rand_date, ticker, entry_price))

    perm_returns, _, _ = run_backtest(shuffled_signals)
    perm_sharpes[i] = calc_sharpe(perm_returns)

real_percentile = float(np.mean(perm_sharpes < sharpe_B) * 100)
p_value = float(np.mean(perm_sharpes >= sharpe_B))
test3_pass = real_percentile >= 90.0

print(f"  Real Sharpe: {sharpe_B:.3f}")
print(f"  Perm median Sharpe: {np.median(perm_sharpes):.3f}")
print(f"  Perm 5th-95th: [{np.percentile(perm_sharpes, 5):.3f}, {np.percentile(perm_sharpes, 95):.3f}]")
print(f"  Real percentile: {real_percentile:.1f}th")
print(f"  p-value: {p_value:.4f}")
print(f"  PASS condition: >= 90th percentile → {'PASS' if test3_pass else 'FAIL'}")

results["tests"]["3_random_timing"] = {
    "description": "1000 random entry date shuffles, same stocks and trade count",
    "real_sharpe": round(sharpe_B, 3),
    "perm_median_sharpe": round(float(np.median(perm_sharpes)), 3),
    "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
    "perm_5th_pct": round(float(np.percentile(perm_sharpes, 5)), 3),
    "perm_95th_pct": round(float(np.percentile(perm_sharpes, 95)), 3),
    "real_percentile": round(real_percentile, 1),
    "p_value": round(p_value, 4),
    "pass_condition": "real strategy >= 90th percentile of random timing",
    "passed": test3_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Remove Key Component (200-SMA filter)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: REMOVE 200-SMA FILTER — RSI(5)<20 without trend filter")
print("=" * 70)

signals_nosma = gen_signals_no_sma()
returns_nosma, eq_nosma, trades_nosma = run_backtest(signals_nosma)
metrics_nosma = calc_full_metrics(returns_nosma, eq_nosma, trades_nosma)
sharpe_nosma = metrics_nosma["sharpe"]

sma_advantage = sharpe_B - sharpe_nosma
test4_pass = sma_advantage >= 0.3

print(f"  Without SMA filter: {metrics_nosma['n_trades']} trades, Sharpe: {sharpe_nosma:.3f}")
print(f"  With SMA filter: {metrics_B['n_trades']} trades, Sharpe: {sharpe_B:.3f}")
print(f"  SMA filter advantage: {sma_advantage:.3f}")
print(f"  PASS condition: SMA version >= 0.3 Sharpe better → {'PASS' if test4_pass else 'FAIL'}")

results["tests"]["4_remove_sma_filter"] = {
    "description": "Run RSI(5)<20 without requiring price > 200-SMA",
    "no_filter_sharpe": round(sharpe_nosma, 3),
    "no_filter_metrics": metrics_nosma,
    "with_filter_sharpe": round(sharpe_B, 3),
    "sma_filter_advantage": round(sma_advantage, 3),
    "pass_condition": "200-SMA version has >= 0.3 Sharpe advantage",
    "passed": test4_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Cost Sensitivity (slippage sweep)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: COST SENSITIVITY — Slippage sweep")
print("=" * 70)

slippage_levels = [0.0002, 0.0005, 0.0010, 0.0020]
cost_results = {}
for slip in slippage_levels:
    ret_s, eq_s, tr_s = run_backtest(signals_B, slippage=slip)
    m_s = calc_full_metrics(ret_s, eq_s, tr_s)
    cost_results[f"{slip:.4f}"] = {
        "slippage_pct": round(slip * 100, 2),
        "sharpe": m_s["sharpe"],
        "n_trades": m_s["n_trades"],
        "total_return_pct": m_s.get("total_return_pct", 0),
        "win_rate": m_s.get("win_rate", 0),
    }
    print(f"  Slippage {slip:.2%}: Sharpe {m_s['sharpe']:.3f}, "
          f"Return {m_s.get('total_return_pct', 0):.1f}%, Trades {m_s['n_trades']}")

sharpe_at_020 = cost_results["0.0020"]["sharpe"]
test5_pass = sharpe_at_020 > 0.5

print(f"  Sharpe at 0.20% slippage: {sharpe_at_020:.3f}")
print(f"  PASS condition: Sharpe > 0.5 at 0.20% slippage → {'PASS' if test5_pass else 'FAIL'}")

results["tests"]["5_cost_sensitivity"] = {
    "description": "Test at 0.02%, 0.05%, 0.10%, 0.20% slippage levels",
    "slippage_results": cost_results,
    "sharpe_at_020_pct": round(sharpe_at_020, 3),
    "pass_condition": "Sharpe > 0.5 at 0.20% slippage",
    "passed": test5_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: Sub-Period Stability
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 6: SUB-PERIOD STABILITY — 4 equal sub-periods")
print("=" * 70)

# Split OOT (Jan 2022 - Jul 2026) into 4 equal sub-periods
oot_start_dt = pd.Timestamp("2022-01-01")
oot_end_dt = pd.Timestamp("2026-07-30")
total_days = (oot_end_dt - oot_start_dt).days
period_days = total_days // 4

sub_periods = []
for i in range(4):
    sp_start = oot_start_dt + pd.Timedelta(days=i * period_days)
    sp_end = oot_start_dt + pd.Timedelta(days=(i + 1) * period_days) if i < 3 else oot_end_dt
    sub_periods.append((str(sp_start.date()), str(sp_end.date())))

sub_period_results = {}
positive_count = 0
for i, (sp_start, sp_end) in enumerate(sub_periods):
    # Filter trades by entry date within sub-period
    sp_trades = [t for t in trades_B if sp_start <= t["entry_date"] <= sp_end]
    if len(sp_trades) < 2:
        sp_sharpe = 0.0
        sp_n = len(sp_trades)
    else:
        sp_returns = np.array([t["return"] for t in sp_trades])
        sp_sharpe = calc_sharpe(sp_returns)
        sp_n = len(sp_trades)

    is_positive = sp_sharpe > 0
    if is_positive:
        positive_count += 1

    sub_period_results[f"period_{i+1}"] = {
        "start": sp_start,
        "end": sp_end,
        "n_trades": sp_n,
        "sharpe": round(sp_sharpe, 3),
        "positive": is_positive,
    }
    print(f"  Period {i+1} ({sp_start} to {sp_end}): "
          f"{sp_n} trades, Sharpe {sp_sharpe:.3f} {'(+)' if is_positive else '(-)'}")

test6_pass = positive_count >= 3
print(f"  Positive sub-periods: {positive_count}/4")
print(f"  PASS condition: >= 3 of 4 positive Sharpe → {'PASS' if test6_pass else 'FAIL'}")

results["tests"]["6_sub_period_stability"] = {
    "description": "Split OOT into 4 equal sub-periods, check Sharpe sign",
    "sub_periods": sub_period_results,
    "positive_count": positive_count,
    "total_periods": 4,
    "pass_condition": "at least 3 of 4 sub-periods have positive Sharpe",
    "passed": test6_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# BONUS: SPY/QQQ Correlation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BONUS: DECORRELATION — Strategy vs SPY/QQQ daily returns")
print("=" * 70)

# Build daily strategy returns (0 on non-trade days)
strat_daily = pd.Series(0.0, index=oot_dates)
for t in trades_B:
    entry_d = pd.Timestamp(t["entry_date"])
    exit_d = pd.Timestamp(t["exit_date"])
    if entry_d in strat_daily.index and exit_d in strat_daily.index:
        # Spread trade return evenly across hold period for daily correlation
        hold_mask = (strat_daily.index >= entry_d) & (strat_daily.index <= exit_d)
        n_days = hold_mask.sum()
        if n_days > 0:
            strat_daily.loc[hold_mask] += t["return"] / n_days

spy_daily = spy_close.pct_change().reindex(oot_dates).fillna(0)
qqq_daily = qqq_close.pct_change().reindex(oot_dates).fillna(0)

corr_spy = float(strat_daily.corr(spy_daily))
corr_qqq = float(strat_daily.corr(qqq_daily))

# Also compute correlation only on days the strategy had positions
active_days = strat_daily[strat_daily != 0].index
if len(active_days) > 10:
    corr_spy_active = float(strat_daily.loc[active_days].corr(spy_daily.loc[active_days]))
    corr_qqq_active = float(strat_daily.loc[active_days].corr(qqq_daily.loc[active_days]))
else:
    corr_spy_active = 0.0
    corr_qqq_active = 0.0

print(f"  Correlation with SPY (all days):    {corr_spy:.4f}")
print(f"  Correlation with QQQ (all days):    {corr_qqq:.4f}")
print(f"  Correlation with SPY (active days): {corr_spy_active:.4f}")
print(f"  Correlation with QQQ (active days): {corr_qqq_active:.4f}")
print(f"  Active trading days: {len(active_days)} / {len(oot_dates)} total")

results["bonus"]["decorrelation"] = {
    "corr_spy_all_days": round(corr_spy, 4),
    "corr_qqq_all_days": round(corr_qqq, 4),
    "corr_spy_active_days": round(corr_spy_active, 4),
    "corr_qqq_active_days": round(corr_qqq_active, 4),
    "active_trading_days": len(active_days),
    "total_oot_days": len(oot_dates),
}


# ══════════════════════════════════════════════════════════════════════════
# OVERALL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
test_results = {
    "1_inverse": test1_pass,
    "2_buy_hold": test2_pass,
    "3_random_timing": test3_pass,
    "4_remove_sma": test4_pass,
    "5_cost_sensitivity": test5_pass,
    "6_sub_period": test6_pass,
}
tests_passed = sum(1 for v in test_results.values() if v)
total_tests = len(test_results)

results["summary"] = {
    "tests_passed": tests_passed,
    "total_tests": total_tests,
    "score": f"{tests_passed}/{total_tests}",
    "individual_results": {k: bool(v) for k, v in test_results.items()},
    "overall_verdict": "STRONG" if tests_passed >= 5 else "ACCEPTABLE" if tests_passed >= 4 else "WEAK" if tests_passed >= 3 else "REJECT",
    "run_timestamp": datetime.now().isoformat(),
}

print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION SUMMARY — RSI(5) Mean Reversion (Variant B)")
print("=" * 70)
print(f"  Overall Score: {tests_passed}/{total_tests}")
print()
for name, passed in test_results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
print()
verdict = results["summary"]["overall_verdict"]
print(f"  VERDICT: {verdict}")
if tests_passed >= 5:
    print("  Strategy passes rigorous adversarial validation.")
elif tests_passed >= 4:
    print("  Strategy is acceptable but has minor weaknesses.")
elif tests_passed >= 3:
    print("  Strategy shows some edge but has notable weaknesses.")
else:
    print("  Strategy fails adversarial validation. Exercise caution.")

print(f"\n  BONUS — Decorrelation:")
print(f"    SPY corr (active days): {corr_spy_active:.4f}")
print(f"    QQQ corr (active days): {corr_qqq_active:.4f}")
print("=" * 70)


# ── Save Results ──────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/rsi_b_adversarial_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
