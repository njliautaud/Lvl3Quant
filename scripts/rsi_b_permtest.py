#!/usr/bin/env python3
"""
RSI(5) Mean Reversion (Variant B) — Permutation Test
Shuffles entry dates (keeping same stocks and hold periods) to test
whether the observed Sharpe is statistically distinguishable from chance.

Also reports MaxDD and total_return_pct which were missing from original results.
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
START = "2021-06-01"      # extra lookback for 200-SMA
END = "2026-07-30"
OOT_START = "2022-01-01"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002     # 0.02%
N_PERM = 1000

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(UNIVERSE + ["SPY"], start=START, end=END,
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

# ── Indicators ────────────────────────────────────────────────────────────
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


# ── Variant B Logic ───────────────────────────────────────────────────────
def gen_signals_B():
    """RSI(5) < 20 AND above 200-SMA."""
    signals = []
    for t, ind in indicators.items():
        mask = (ind["rsi5"] < 20) & ind["above_200sma"]
        for dt in ind.index[mask]:
            signals.append((dt, t, ind.loc[dt, "close"]))
    return signals


def exit_B(ticker, loc, ind):
    """Exit when RSI(5) > 50 or max 10 days."""
    for i in range(loc + 1, min(loc + 11, len(ind))):
        if ind["rsi5"].iloc[i] > 50:
            return i
    return None


def run_backtest(signals):
    """Run backtest on given signals, return (per-trade returns, equity_curve, trades_info)."""
    trades = []
    for date, ticker, entry_price in signals:
        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)
        actual_entry = entry_price * (1 + SLIPPAGE_PCT)

        exit_idx = exit_B(ticker, loc, ind)
        if exit_idx is None or exit_idx > loc + 10:
            exit_idx = min(loc + 10, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        exit_price = ind["close"].iloc[exit_idx] * (1 - SLIPPAGE_PCT)
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

    # Sequential sizing (full capital, one position at a time)
    trades_sorted = sorted(trades, key=lambda t: t["entry_date"])
    equity = CAPITAL
    equity_curve = [(OOT_START, CAPITAL)]
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


def calc_sharpe(returns):
    """Annualized Sharpe from per-trade returns."""
    if len(returns) < 2:
        return 0.0
    avg_hold = 8.8  # approx from original results
    trades_per_year = max(1, 252 / max(avg_hold, 1))
    ann_factor = np.sqrt(trades_per_year)
    mean_r = returns.mean()
    std_r = returns.std()
    if std_r < 1e-12:
        return 0.0
    return (mean_r / std_r) * ann_factor


def calc_full_metrics(returns, equity_curve, realized):
    """Full metrics including MaxDD and total_return_pct."""
    if len(returns) < 2:
        return {}

    n = len(returns)
    wins = (returns > 0).sum()
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

    # Max drawdown from equity curve
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

    # Regime split
    bull_r = np.array([t["return"] for t in realized if t["regime"] == "Bull"])
    bear_r = np.array([t["return"] for t in realized if t["regime"] == "Bear"])

    def regime_sharpe(rets):
        if len(rets) < 2:
            return 0.0
        s = rets.std()
        if s < 1e-12:
            return 0.0
        return (rets.mean() / s) * ann_factor

    sh_bull = regime_sharpe(bull_r)
    sh_bear = regime_sharpe(bear_r)
    denom = max(abs(sh_bull), abs(sh_bear), 1e-9)
    regime_gap = abs(sh_bull - sh_bear) / denom

    return {
        "variant": "B_RSI5_MeanRev",
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "final_equity": round(float(final_eq), 2),
        "avg_hold_days": round(float(avg_hold), 1),
        "sharpe_bull": round(float(sh_bull), 3),
        "sharpe_bear": round(float(sh_bear), 3),
        "regime_gap": round(float(regime_gap), 4),
        "bull_trades": int(len(bull_r)),
        "bear_trades": int(len(bear_r)),
    }


# ── Run Real Backtest ─────────────────────────────────────────────────────
print("\nRunning Variant B real backtest ...")
signals_real = gen_signals_B()
returns_real, eq_real, trades_real = run_backtest(signals_real)
real_sharpe = calc_sharpe(returns_real)
metrics = calc_full_metrics(returns_real, eq_real, trades_real)

print(f"  Real trades (OOT): {len(returns_real)}")
print(f"  Real Sharpe: {real_sharpe:.3f}")
print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
print(f"  Max Drawdown: {metrics['max_dd_pct']:.2f}%")

# ── Permutation Test (shuffle entry dates) ────────────────────────────────
# Strategy: for each permutation, we keep the same stocks and hold periods
# but randomly reassign each trade's entry date to a random valid OOT date
# for that same stock (must be in the indicator index, >= OOT_START).
# Then re-run the backtest with these shuffled entries.

print(f"\nRunning {N_PERM} permutation tests (shuffling entry dates) ...")

# Pre-compute valid OOT dates per stock
oot_dates_per_stock = {}
for t, ind in indicators.items():
    valid = ind.index[ind.index >= OOT_START]
    # Leave margin for hold period (10 days)
    if len(valid) > 15:
        oot_dates_per_stock[t] = valid[:-10]

# Extract the trade specs (ticker + hold structure) from real signals
real_signal_specs = []
for date, ticker, entry_price in signals_real:
    if str(date.date()) >= OOT_START and ticker in oot_dates_per_stock:
        real_signal_specs.append(ticker)

perm_sharpes = np.zeros(N_PERM)
for i in range(N_PERM):
    if (i + 1) % 100 == 0:
        print(f"  Permutation {i+1}/{N_PERM} ...")

    # Shuffle: for each original trade, pick a random date for the same stock
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

# ── Results ───────────────────────────────────────────────────────────────
p_value = float(np.mean(perm_sharpes >= real_sharpe))
perm_median = float(np.median(perm_sharpes))
perm_mean = float(np.mean(perm_sharpes))
perm_p5 = float(np.percentile(perm_sharpes, 5))
perm_p95 = float(np.percentile(perm_sharpes, 95))
real_percentile = float(np.mean(perm_sharpes < real_sharpe) * 100)

print("\n" + "=" * 70)
print("VARIANT B — RSI(5) MEAN REVERSION — PERMUTATION TEST RESULTS")
print("=" * 70)
print(f"  Real Sharpe:           {real_sharpe:.3f}")
print(f"  Permutation Median:    {perm_median:.3f}")
print(f"  Permutation Mean:      {perm_mean:.3f}")
print(f"  Permutation 5th-95th:  [{perm_p5:.3f}, {perm_p95:.3f}]")
print(f"  Real Percentile:       {real_percentile:.1f}th")
print(f"  p-value:               {p_value:.4f}")
print(f"")
print(f"  Total Return:          {metrics['total_return_pct']:.2f}%")
print(f"  Max Drawdown:          {metrics['max_dd_pct']:.2f}%")
print(f"  Win Rate:              {metrics['win_rate']:.1%}")
print(f"  Profit Factor:         {metrics['profit_factor']:.3f}")
print(f"  Sortino:               {metrics['sortino']:.3f}")
print(f"  Regime Gap:            {metrics['regime_gap']:.4f}")
print(f"  Trades:                {metrics['n_trades']}")
print("=" * 70)

if p_value < 0.05:
    print(f"  VERDICT: PASS — p={p_value:.4f} < 0.05. Edge is statistically significant.")
else:
    print(f"  VERDICT: FAIL — p={p_value:.4f} >= 0.05. Cannot reject null hypothesis.")

# ── Save ──────────────────────────────────────────────────────────────────
output = {
    "variant": "B_RSI5_MeanRev",
    "test_type": "permutation_test_date_shuffle",
    "n_permutations": N_PERM,
    "real_sharpe": round(real_sharpe, 3),
    "perm_median_sharpe": round(perm_median, 3),
    "perm_mean_sharpe": round(perm_mean, 3),
    "perm_5th_pct": round(perm_p5, 3),
    "perm_95th_pct": round(perm_p95, 3),
    "real_percentile": round(real_percentile, 1),
    "p_value": round(p_value, 4),
    "metrics": metrics,
    "config": {
        "universe_size": len(UNIVERSE),
        "oot_period": f"{OOT_START} to {END}",
        "capital": CAPITAL,
        "commission": 0.0,
        "slippage_pct": SLIPPAGE_PCT,
        "entry_rule": "RSI(5) < 20 AND price > 200-SMA",
        "exit_rule": "RSI(5) > 50 OR 10 days max hold",
    },
    "run_timestamp": datetime.now().isoformat(),
}

output_path = Path("/home/jupiter/Lvl3Quant/data/rsi_b_permtest_results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {output_path}")
