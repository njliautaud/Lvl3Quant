#!/usr/bin/env python3
"""
Volume Anomaly Backtest — Growth Stocks
----------------------------------------
Tests whether unusual volume spikes on growth stocks (excluding earnings windows)
predict short-term continuation or reversal.

6 Variants:
  A: Volume Spike + Up Day (continuation) — hold 5d
  B: Volume Spike + Down Day Reversal (contrarian) — hold 5d
  C: Volume Dry-Up Breakout — hold 10d
  D: Relative Volume + RSI Combo — hold until RSI>50 or 10d max
  E: Accumulation Score Flip — hold 10d
  F: Volume Climax Reversal (extreme panic) — hold 5d

Kill switch: VIX > 20 AND SPY < 50-SMA → skip signals.
Earnings exclusion: skip signals within 3 days before/after earnings.
"""

import json
import warnings
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "CRM",
    "NFLX", "SHOP", "SQ", "SNOW", "PLTR", "COIN", "MELI", "MDB", "DDOG", "TTD"
]
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2021-01-01"  # extra for 200-SMA lookback
STARTING_CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02%
PERM_ITERATIONS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/volume_anomaly_results.json")

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading price data for", len(UNIVERSE), "tickers + SPY + ^VIX...")
tickers_all = UNIVERSE + ["SPY", "^VIX"]
raw = yf.download(tickers_all, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
    open_ = raw["Open"]
    high = raw["High"]
    low = raw["Low"]
    volume = raw["Volume"]
else:
    raise ValueError("Expected MultiIndex columns from yf.download with multiple tickers")

# Rename ^VIX column
if "^VIX" in close.columns:
    close = close.rename(columns={"^VIX": "VIX"})
    open_ = open_.rename(columns={"^VIX": "VIX"})
    high = high.rename(columns={"^VIX": "VIX"})
    low = low.rename(columns={"^VIX": "VIX"})
    volume = volume.rename(columns={"^VIX": "VIX"})

print(f"  Data shape: {close.shape[0]} days x {close.shape[1]} tickers")
print(f"  Date range: {close.index[0].date()} to {close.index[-1].date()}")

# ── Fetch earnings dates ──────────────────────────────────────────────────
print("Fetching earnings dates (this may take a moment)...")
earnings_blackout = {}  # ticker -> set of blackout dates
for ticker in UNIVERSE:
    try:
        t = yf.Ticker(ticker)
        # Try to get earnings dates
        try:
            edates = t.get_earnings_dates(limit=50)
            if edates is not None and len(edates) > 0:
                edate_list = edates.index.normalize()
            else:
                edate_list = pd.DatetimeIndex([])
        except Exception:
            # Fallback: try calendar
            try:
                cal = t.calendar
                if cal is not None and hasattr(cal, 'index'):
                    edate_list = pd.DatetimeIndex([cal.iloc[0]] if len(cal) > 0 else [])
                else:
                    edate_list = pd.DatetimeIndex([])
            except Exception:
                edate_list = pd.DatetimeIndex([])

        blackout = set()
        for ed in edate_list:
            for delta in range(-3, 4):  # 3 days before and after
                blackout.add(ed + timedelta(days=delta))
        earnings_blackout[ticker] = blackout
        n_earnings = len(edate_list)
        if n_earnings > 0:
            print(f"  {ticker}: {n_earnings} earnings dates, {len(blackout)} blackout days")
    except Exception as e:
        earnings_blackout[ticker] = set()
        print(f"  {ticker}: earnings fetch failed ({e}), no blackout")

# ── Precompute indicators ──────────────────────────────────────────────────
vol_avg_20 = volume[UNIVERSE].rolling(20).mean()
sma_200 = close[UNIVERSE].rolling(200).mean()
spy_sma_50 = close["SPY"].rolling(50).mean()
spy_sma_200 = close["SPY"].rolling(200).mean()

# RSI(5) for each ticker
def compute_rsi(series, period=5):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

rsi_5 = pd.DataFrame({t: compute_rsi(close[t], 5) for t in UNIVERSE})

# Kill switch: VIX > 20 AND SPY < 50-SMA
vix_close = close["VIX"] if "VIX" in close.columns else pd.Series(15.0, index=close.index)
kill_switch = (vix_close > 20) & (close["SPY"] < spy_sma_50)

# Regime: Bull = SPY above 200-SMA, Bear = below
regime_bull = close["SPY"] > spy_sma_200

# OOT mask
oot_mask = close.index >= OOT_START

# Daily returns for each ticker
daily_ret = close[UNIVERSE].pct_change()


# ── Earnings blackout check ───────────────────────────────────────────────
def is_in_earnings_blackout(ticker, date):
    """Check if date falls within 3 days of an earnings date for ticker."""
    if ticker not in earnings_blackout:
        return False
    # Normalize date
    if hasattr(date, 'normalize'):
        d = date.normalize()
    else:
        d = pd.Timestamp(date).normalize()
    return d in earnings_blackout[ticker]


# ── Signal generation ──────────────────────────────────────────────────────

def gen_signals_A():
    """A: Volume > 2x 20d avg AND close > open (green candle). Buy at close, hold 5d."""
    signals = []
    for ticker in UNIVERSE:
        v = volume[ticker]
        va = vol_avg_20[ticker]
        c = close[ticker]
        o = open_[ticker]
        ks = kill_switch
        mask = (oot_mask & (v > 2.0 * va) & (c > o) & va.notna() & ~ks)
        dates = close.index[mask]
        for d in dates:
            if not is_in_earnings_blackout(ticker, d):
                signals.append({"ticker": ticker, "entry_date": d, "hold": 5})
    return signals


def gen_signals_B():
    """B: Volume > 2x 20d avg AND close < open (red candle) AND price > 200-SMA.
    Contrarian reversal. Buy at close, hold 5d."""
    signals = []
    for ticker in UNIVERSE:
        v = volume[ticker]
        va = vol_avg_20[ticker]
        c = close[ticker]
        o = open_[ticker]
        s200 = sma_200[ticker]
        ks = kill_switch
        mask = (oot_mask & (v > 2.0 * va) & (c < o) & (c > s200) & va.notna() & s200.notna() & ~ks)
        dates = close.index[mask]
        for d in dates:
            if not is_in_earnings_blackout(ticker, d):
                signals.append({"ticker": ticker, "entry_date": d, "hold": 5})
    return signals


def gen_signals_C():
    """C: Volume dry-up breakout. Volume < 0.5x avg for 3+ consecutive days,
    then volume > 1.5x avg. Buy at breakout close, hold 10d."""
    signals = []
    for ticker in UNIVERSE:
        v = volume[ticker]
        va = vol_avg_20[ticker]
        ks = kill_switch

        low_vol = (v < 0.5 * va).astype(int)
        # Count consecutive low-vol days
        consec_low = low_vol.copy()
        for i in range(1, len(consec_low)):
            if consec_low.iloc[i] == 1:
                consec_low.iloc[i] = consec_low.iloc[i-1] + 1
            else:
                consec_low.iloc[i] = 0

        # Breakout: prev day had 3+ consec low vol, today volume > 1.5x avg
        prev_consec = consec_low.shift(1)
        breakout = (prev_consec >= 3) & (v > 1.5 * va) & va.notna()
        mask = oot_mask & breakout & ~ks
        dates = close.index[mask]
        for d in dates:
            if not is_in_earnings_blackout(ticker, d):
                signals.append({"ticker": ticker, "entry_date": d, "hold": 10})
    return signals


def gen_signals_D():
    """D: Volume > 1.5x avg AND RSI(5) < 30 AND price > 200-SMA.
    Buy at close, hold until RSI(5) > 50 or 10 days max."""
    signals = []
    for ticker in UNIVERSE:
        v = volume[ticker]
        va = vol_avg_20[ticker]
        c = close[ticker]
        s200 = sma_200[ticker]
        rsi = rsi_5[ticker]
        ks = kill_switch

        mask = (oot_mask & (v > 1.5 * va) & (rsi < 30) & (c > s200)
                & va.notna() & s200.notna() & rsi.notna() & ~ks)
        dates = close.index[mask]
        for d in dates:
            if not is_in_earnings_blackout(ticker, d):
                # Dynamic exit: hold until RSI > 50 or 10 days
                loc = close.index.get_loc(d)
                hold = 10  # default max
                for offset in range(1, 11):
                    if loc + offset >= len(close.index):
                        break
                    future_rsi = rsi.iloc[loc + offset]
                    if not pd.isna(future_rsi) and future_rsi > 50:
                        hold = offset
                        break
                signals.append({"ticker": ticker, "entry_date": d, "hold": hold})
    return signals


def gen_signals_E():
    """E: Accumulation score. Over trailing 10 days, count up-volume days
    (close > prior close AND volume > avg) minus down-volume days.
    When score goes from negative to positive, buy. Hold 10d."""
    signals = []
    for ticker in UNIVERSE:
        v = volume[ticker]
        va = vol_avg_20[ticker]
        c = close[ticker]
        ks = kill_switch

        # Up-volume day: close > prior close AND volume > avg
        up_vol = ((c > c.shift(1)) & (v > va)).astype(int)
        down_vol = ((c < c.shift(1)) & (v > va)).astype(int)

        # Accumulation score: rolling 10-day sum
        score = up_vol.rolling(10).sum() - down_vol.rolling(10).sum()
        prev_score = score.shift(1)

        # Flip: prev negative, current positive
        flip = (prev_score < 0) & (score > 0) & score.notna() & prev_score.notna()
        mask = oot_mask & flip & ~ks
        dates = close.index[mask]
        for d in dates:
            if not is_in_earnings_blackout(ticker, d):
                signals.append({"ticker": ticker, "entry_date": d, "hold": 10})
    return signals


def gen_signals_F():
    """F: Volume > 3x avg AND price drops > 3% intraday AND price > 200-SMA.
    Extreme panic buy. Hold 5d."""
    signals = []
    for ticker in UNIVERSE:
        v = volume[ticker]
        va = vol_avg_20[ticker]
        c = close[ticker]
        o = open_[ticker]
        h = high[ticker]
        l = low[ticker]
        s200 = sma_200[ticker]
        ks = kill_switch

        # Intraday drop > 3%: (high - low) / high > 0.03 AND close < open
        intraday_drop = ((h - l) / h > 0.03) & (c < o)
        mask = (oot_mask & (v > 3.0 * va) & intraday_drop & (c > s200)
                & va.notna() & s200.notna() & ~ks)
        dates = close.index[mask]
        for d in dates:
            if not is_in_earnings_blackout(ticker, d):
                signals.append({"ticker": ticker, "entry_date": d, "hold": 5})
    return signals


# ── Backtest engine ─────────────────────────────────────────────────────────
def backtest(signals, label=""):
    """
    Cash account backtest: 1 position at a time, fractional shares.
    Entry at close of signal day (with slippage), exit at close after hold days.
    """
    empty = {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "maxdd": 0,
             "n_trades": 0, "sharpe_bull": 0, "sharpe_bear": 0, "regime_gap": 0,
             "total_return_pct": 0}

    if not signals:
        return empty

    # Build trade list
    trades = []
    for sig in signals:
        ticker = sig["ticker"]
        entry_date = sig["entry_date"]
        hold = sig["hold"]
        if entry_date not in close.index:
            continue
        loc = close.index.get_loc(entry_date)
        exit_loc = loc + hold
        if exit_loc >= len(close):
            continue
        entry_price = close[ticker].iloc[loc]
        exit_price = close[ticker].iloc[exit_loc]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            continue
        # Apply slippage
        entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)  # buy slightly higher
        exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)    # sell slightly lower
        ret = (exit_price_adj - entry_price_adj) / entry_price_adj
        is_bull = regime_bull.loc[entry_date] if entry_date in regime_bull.index else True
        trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": close.index[exit_loc],
            "ret": ret,
            "hold": hold,
            "bull": is_bull,
        })

    if not trades:
        return empty

    trades_df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

    # 1 position at a time: skip signals that overlap with active position
    accepted = []
    current_exit = pd.Timestamp("1900-01-01")
    for _, t in trades_df.iterrows():
        if t["entry_date"] >= current_exit:
            accepted.append(t)
            current_exit = t["exit_date"]

    if not accepted:
        return empty

    at_df = pd.DataFrame(accepted)
    rets = at_df["ret"].values
    n_trades = len(rets)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    wr = len(wins) / n_trades if n_trades > 0 else 0
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Equity curve: full capital into each trade
    capital = STARTING_CAPITAL
    eq = [capital]
    for r in rets:
        capital *= (1 + r)
        eq.append(capital)

    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    maxdd = dd.min()

    # Annualize based on average hold
    avg_hold = at_df["hold"].mean()
    trades_per_year = 252 / avg_hold if avg_hold > 0 else 50
    mean_ret = rets.mean()
    std_ret = rets.std() if rets.std() > 0 else 1e-9
    downside = rets[rets < 0]
    downside_std = downside.std() if len(downside) > 1 and downside.std() > 0 else 1e-9

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year)

    # Regime split
    bull_rets = at_df[at_df["bull"] == True]["ret"].values
    bear_rets = at_df[at_df["bull"] == False]["ret"].values

    def _sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return (r.mean() / r.std()) * np.sqrt(trades_per_year)

    sharpe_bull = _sharpe(bull_rets)
    sharpe_bear = _sharpe(bear_rets)
    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    total_return_pct = (eq[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "maxdd": round(maxdd * 100, 2),
        "n_trades": n_trades,
        "n_bull_trades": len(bull_rets),
        "n_bear_trades": len(bear_rets),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "total_return_pct": round(total_return_pct, 2),
        "final_capital": round(eq[-1], 2),
        "avg_hold_days": round(avg_hold, 1),
        "avg_return_pct": round(mean_ret * 100, 3),
        "_rets": rets,  # internal, not saved
    }


# ── Permutation test ───────────────────────────────────────────────────────
def permutation_test(real_sharpe, signals, n_iter=PERM_ITERATIONS):
    """Shuffle entry dates within same stock among OOT dates. Return p-value."""
    if real_sharpe <= 0:
        return 1.0
    oot_dates = close.index[oot_mask]
    count_better = 0
    for _ in range(n_iter):
        shuffled = []
        for s in signals:
            new_date = np.random.choice(oot_dates)
            shuffled.append({"ticker": s["ticker"], "entry_date": new_date, "hold": s["hold"]})
        res = backtest(shuffled)
        if res["sharpe"] >= real_sharpe:
            count_better += 1
    return round(count_better / n_iter, 4)


# ── 5-Gate check ────────────────────────────────────────────────────────────
def five_gate(metrics, perm_p):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.50": metrics["regime_gap"] < 0.50,
        "maxdd_gt_neg50": metrics["maxdd"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["pass_all"] = all(gates.values())
    return gates


# ── Run all variants ────────────────────────────────────────────────────────
variant_names = {
    "A": "Volume Spike + Up Day (continuation, hold 5d)",
    "B": "Volume Spike + Down Day Reversal (contrarian, hold 5d)",
    "C": "Volume Dry-Up Breakout (hold 10d)",
    "D": "Relative Volume + RSI Combo (hold until RSI>50 or 10d)",
    "E": "Accumulation Score Flip (hold 10d)",
    "F": "Volume Climax Reversal (extreme panic, hold 5d)",
}

print("\n" + "=" * 80)
print("VOLUME ANOMALY BACKTEST — GROWTH STOCKS")
print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL}")
print(f"Universe: {len(UNIVERSE)} growth stocks | 1 position at a time | Fractional shares")
print(f"Costs: $0 commission, {SLIPPAGE_PCT*100:.2f}% slippage | Kill switch: VIX>20 & SPY<50SMA")
print(f"Earnings exclusion: +/- 3 days around earnings dates")
print("=" * 80)

generators = {
    "A": gen_signals_A,
    "B": gen_signals_B,
    "C": gen_signals_C,
    "D": gen_signals_D,
    "E": gen_signals_E,
    "F": gen_signals_F,
}

all_signals = {}
all_results = {}

for name in ["A", "B", "C", "D", "E", "F"]:
    desc = variant_names[name]
    print(f"\n[{name}] {desc}...")
    sigs = generators[name]()
    res = backtest(sigs, name)
    all_signals[name] = sigs
    all_results[name] = res
    print(f"    Raw signals: {len(sigs)} | Accepted trades: {res['n_trades']} "
          f"| Sharpe: {res['sharpe']:.3f} | WR: {res['wr']:.1%} | Ret: {res['total_return_pct']:.1f}%")

# ── Permutation tests ──────────────────────────────────────────────────────
print("\n" + "-" * 80)
print("Running permutation tests (1000 iterations each)...")
perm_results = {}
for name in ["A", "B", "C", "D", "E", "F"]:
    res = all_results[name]
    if res["n_trades"] < 5:
        perm_results[name] = 1.0
        print(f"  [{name}] Too few trades ({res['n_trades']}), skipping (p=1.0)")
        continue
    print(f"  [{name}] Permuting... (real Sharpe={res['sharpe']:.3f})", end="", flush=True)
    p = permutation_test(res["sharpe"], all_signals[name])
    perm_results[name] = p
    print(f" → p={p:.4f}")

# ── 5-Gate results ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5-GATE VALIDATION RESULTS")
print("=" * 80)

final_results = {}
for name in ["A", "B", "C", "D", "E", "F"]:
    res = all_results[name]
    perm_p = perm_results[name]
    gates = five_gate(res, perm_p)

    # Clean for JSON (remove numpy arrays)
    res_clean = {k: v for k, v in res.items() if k != "_rets"}
    res_clean["perm_p"] = perm_p
    res_clean["gates"] = gates

    status = "✓ PASS" if gates["pass_all"] else "✗ FAIL"
    print(f"\n[{name}] {variant_names[name]} — {status}")
    print(f"    Sharpe={res['sharpe']:.3f}  Sortino={res['sortino']:.3f}  PF={res['pf']:.2f}  "
          f"WR={res['wr']:.1%}  MDD={res['maxdd']:.1f}%")
    print(f"    Trades={res['n_trades']} (Bull:{res['n_bull_trades']}, Bear:{res['n_bear_trades']})  "
          f"Avg hold={res['avg_hold_days']:.1f}d  Avg ret={res['avg_return_pct']:.3f}%")
    print(f"    Sharpe_bull={res['sharpe_bull']:.3f}  Sharpe_bear={res['sharpe_bear']:.3f}  "
          f"Regime_gap={res['regime_gap']:.3f}  Perm_p={perm_p:.4f}")
    print(f"    Return: {res['total_return_pct']:.1f}% (${STARTING_CAPITAL} → ${res['final_capital']:.2f})")

    gate_str = " | ".join(f"{k}={'Y' if v else 'N'}" for k, v in gates.items())
    print(f"    Gates: {gate_str}")

    final_results[name] = res_clean

# ── Save results ────────────────────────────────────────────────────────────
output = {
    "backtest": "volume_anomaly_growth_stocks",
    "run_timestamp": datetime.now().isoformat(),
    "oot_period": f"{OOT_START} to {OOT_END}",
    "universe": UNIVERSE,
    "starting_capital": STARTING_CAPITAL,
    "slippage_pct": SLIPPAGE_PCT,
    "perm_iterations": PERM_ITERATIONS,
    "kill_switch": "VIX > 20 AND SPY < 50-SMA",
    "earnings_exclusion": "+/- 3 days around earnings dates",
    "variant_descriptions": variant_names,
    "results": final_results,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

# ── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Var':<4} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR':>6} {'MDD%':>7} "
      f"{'Trd':>5} {'Ret%':>7} {'Perm_p':>7} {'5G':>5}")
print("-" * 80)
for name in ["A", "B", "C", "D", "E", "F"]:
    r = final_results[name]
    g = "PASS" if r["gates"]["pass_all"] else "FAIL"
    print(f"{name:<4} {r['sharpe']:>7.3f} {r['sortino']:>7.3f} {r['pf']:>6.2f} {r['wr']:>6.1%} "
          f"{r['maxdd']:>7.1f} {r['n_trades']:>5} {r['total_return_pct']:>7.1f} "
          f"{r['perm_p']:>7.4f} {g:>5}")

passed = [n for n in ["A", "B", "C", "D", "E", "F"] if final_results[n]["gates"]["pass_all"]]
print(f"\nVariants passing all 5 gates: {passed if passed else 'NONE'}")
print(f"\nResults saved to {RESULTS_PATH}")
print("Done.")
