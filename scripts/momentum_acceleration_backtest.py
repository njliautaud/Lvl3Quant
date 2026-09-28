#!/usr/bin/env python3
"""
Momentum Acceleration Backtest — 6 Variants
Walk-forward OOT: Jan 2022 to present, $645 initial capital, $0 commission (Robinhood)
Benchmark: QQQ buy-and-hold
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "SHOP", "SQ", "COIN", "SNOW", "DDOG", "NET",
    "RBLX", "PLTR", "UBER", "LYFT",
]
INITIAL_CAPITAL = 645.0
START_DATE = "2021-06-01"   # extra lookback for indicators
OOT_START = "2022-01-03"
PERM_ITERATIONS = 1000
GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.5,
    "max_dd_floor": -0.50,
    "min_trades": 20,
}

# ─── Data download ───────────────────────────────────────────────────────────
print("Downloading price data …")
tickers_to_dl = UNIVERSE + ["SPY", "QQQ"]
raw = yf.download(tickers_to_dl, start=START_DATE, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
else:
    close = raw[["Close"]].copy()
    volume = raw[["Volume"]].copy()

close = close.ffill().dropna(how="all")
volume = volume.ffill().fillna(0)

# Ensure columns are strings (not tuples)
close.columns = [str(c) for c in close.columns]
volume.columns = [str(c) for c in volume.columns]

# Flatten index if needed
if isinstance(close.index, pd.MultiIndex):
    close.index = close.index.get_level_values(0)
    volume.index = volume.index.get_level_values(0)

close.index = pd.to_datetime(close.index).tz_localize(None)
volume.index = pd.to_datetime(volume.index).tz_localize(None)

# SPY 200-SMA for regime
spy_sma200 = close["SPY"].rolling(200).mean()
regime = (close["SPY"] > spy_sma200).astype(int)  # 1=bull, 0=bear

oot_mask = close.index >= pd.Timestamp(OOT_START)
oot_dates = close.index[oot_mask]

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, "
      f"{len(oot_dates)} OOT trading days")


# ─── Helper: precompute indicators ───────────────────────────────────────────
def compute_indicators(close_df, volume_df):
    """Return dict of DataFrames with precomputed indicators per ticker."""
    ind = {}
    for tk in UNIVERSE:
        if tk not in close_df.columns:
            continue
        c = close_df[tk]
        v = volume_df[tk] if tk in volume_df.columns else pd.Series(0, index=c.index)
        d = pd.DataFrame(index=c.index)
        d["close"] = c
        d["vol"] = v
        d["ret5"] = c.pct_change(5)
        d["ret20"] = c.pct_change(20)
        d["vol_avg20"] = v.rolling(20).mean()
        # RSI 14
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi14"] = 100 - 100 / (1 + rs)
        d["rsi14_prev"] = d["rsi14"].shift(1)
        d["sma50"] = c.rolling(50).mean()
        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        d["macd_hist"] = macd_line - signal_line
        d["macd_hist_prev"] = d["macd_hist"].shift(1)
        d["macd_hist_prev2"] = d["macd_hist"].shift(2)
        # Bollinger
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        d["bb_upper"] = sma20 + 2 * std20
        d["close_prev"] = c.shift(1)
        # ATR 14
        high = c  # approx (daily close only)
        low = c
        tr = pd.concat([
            (c - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        # Better ATR approx using intraday range proxy: use close-to-close as TR
        d["atr14"] = tr.rolling(14).mean()
        d["atr14_prev"] = d["atr14"].shift(1)
        ind[tk] = d
    return ind

indicators = compute_indicators(close, volume)


# ─── Signal generators ───────────────────────────────────────────────────────
def signal_A(date, ind):
    """Price Acceleration: 5d ret > 20d ret, both positive. Return top-3."""
    scores = {}
    for tk, d in ind.items():
        if date not in d.index:
            continue
        row = d.loc[date]
        if pd.isna(row["ret5"]) or pd.isna(row["ret20"]):
            continue
        if row["ret5"] > 0 and row["ret20"] > 0 and row["ret5"] > row["ret20"]:
            scores[tk] = row["ret5"] - row["ret20"]  # acceleration magnitude
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:3]


def signal_B(date, ind):
    """Volume-Confirmed Acceleration: A + volume > 1.5x 20d avg."""
    scores = {}
    for tk, d in ind.items():
        if date not in d.index:
            continue
        row = d.loc[date]
        if pd.isna(row["ret5"]) or pd.isna(row["ret20"]) or pd.isna(row["vol_avg20"]):
            continue
        if (row["ret5"] > 0 and row["ret20"] > 0 and row["ret5"] > row["ret20"]
                and row["vol"] > 1.5 * row["vol_avg20"] and row["vol_avg20"] > 0):
            scores[tk] = row["ret5"] - row["ret20"]
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:3]


def signal_C(date, ind):
    """RSI Momentum Shift: RSI crosses above 50 from below AND price > 50-SMA."""
    hits = []
    for tk, d in ind.items():
        if date not in d.index:
            continue
        row = d.loc[date]
        if pd.isna(row["rsi14"]) or pd.isna(row["rsi14_prev"]) or pd.isna(row["sma50"]):
            continue
        if (row["rsi14_prev"] < 50 and row["rsi14"] >= 50
                and row["close"] > row["sma50"]):
            hits.append((tk, row["rsi14"]))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [h[0] for h in hits[:3]]


def signal_D(date, ind):
    """MACD Histogram Acceleration: 2nd consecutive positive bar, bigger than prior."""
    hits = []
    for tk, d in ind.items():
        if date not in d.index:
            continue
        row = d.loc[date]
        h = row["macd_hist"]
        hp = row["macd_hist_prev"]
        hpp = row["macd_hist_prev2"]
        if pd.isna(h) or pd.isna(hp) or pd.isna(hpp):
            continue
        if h > 0 and hp > 0 and h > hp:
            hits.append((tk, h - hp))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [h[0] for h in hits[:2]]


def signal_E(date, ind):
    """Breakout from Consolidation: price breaks above BB upper AND ATR expanding."""
    hits = []
    for tk, d in ind.items():
        if date not in d.index:
            continue
        row = d.loc[date]
        if pd.isna(row["bb_upper"]) or pd.isna(row["close_prev"]) or pd.isna(row["atr14"]) or pd.isna(row["atr14_prev"]):
            continue
        if (row["close"] > row["bb_upper"]
                and row["close_prev"] <= row["bb_upper"]
                and row["atr14"] > row["atr14_prev"]):
            hits.append((tk, row["close"] / row["bb_upper"]))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [h[0] for h in hits[:3]]


def signal_F(date, ind):
    """Multi-Signal Composite: require 2/3 of {A, C, D} to agree."""
    a_set = set(signal_A(date, ind))
    c_set = set(signal_C(date, ind))
    d_set = set(signal_D(date, ind))
    counts = {}
    for tk in a_set | c_set | d_set:
        cnt = int(tk in a_set) + int(tk in c_set) + int(tk in d_set)
        if cnt >= 2:
            counts[tk] = cnt
    ranked = sorted(counts, key=counts.get, reverse=True)
    return ranked[:3]


# ─── Backtest engine ─────────────────────────────────────────────────────────
def run_backtest(signal_func, hold_days, ind, close_df, oot_dates_arr, max_positions=None):
    """
    Simple long-only backtest with equal-weight allocation.
    Returns list of trade dicts and daily equity series.
    """
    trades = []
    positions = {}  # tk -> {entry_date, entry_price, exit_date_idx, shares}
    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    equity_curve = []

    date_list = list(close_df.index)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    for date in oot_dates_arr:
        idx = date_to_idx.get(date)
        if idx is None:
            continue

        # Check exits
        exited = []
        for tk, pos in list(positions.items()):
            if idx >= pos["exit_idx"]:
                exit_price = close_df[tk].iloc[min(pos["exit_idx"], len(date_list) - 1)]
                if pd.isna(exit_price):
                    exit_price = pos["entry_price"]
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                cash += exit_price * pos["shares"]
                trades.append({
                    "ticker": tk,
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(date_list[min(pos["exit_idx"], len(date_list)-1)].date()),
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "ret": (exit_price / pos["entry_price"]) - 1,
                    "regime": int(regime.get(pos["entry_date"], 1)),
                })
                exited.append(tk)
        for tk in exited:
            del positions[tk]

        # Generate signals
        signals = signal_func(date, ind)
        if max_positions is not None:
            signals = signals[:max_positions]

        # Enter new positions
        n_new = len(signals)
        if n_new > 0 and cash > 10:
            alloc_per = cash / n_new
            for tk in signals:
                if tk in positions:
                    continue
                price = close_df[tk].iloc[idx] if tk in close_df.columns else None
                if price is None or pd.isna(price) or price <= 0:
                    continue
                shares = int(alloc_per / price)
                if shares < 1:
                    continue
                cost = shares * price
                if cost > cash:
                    continue
                cash -= cost
                exit_idx = min(idx + hold_days, len(date_list) - 1)
                positions[tk] = {
                    "entry_date": date,
                    "entry_price": price,
                    "exit_idx": exit_idx,
                    "shares": shares,
                }

        # Mark-to-market
        pos_value = 0
        for tk, pos in positions.items():
            cur_price = close_df[tk].iloc[idx] if tk in close_df.columns else pos["entry_price"]
            if pd.isna(cur_price):
                cur_price = pos["entry_price"]
            pos_value += cur_price * pos["shares"]
        equity = cash + pos_value
        equity_curve.append({"date": str(date.date()), "equity": equity})

    # Force-close remaining
    last_idx = len(date_list) - 1
    for tk, pos in positions.items():
        exit_price = close_df[tk].iloc[last_idx]
        if pd.isna(exit_price):
            exit_price = pos["entry_price"]
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        cash += exit_price * pos["shares"]
        trades.append({
            "ticker": tk,
            "entry_date": str(pos["entry_date"].date()),
            "exit_date": str(date_list[last_idx].date()),
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "shares": pos["shares"],
            "pnl": pnl,
            "ret": (exit_price / pos["entry_price"]) - 1,
            "regime": int(regime.get(pos["entry_date"], 1)),
        })

    return trades, equity_curve


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve):
    if not trades or not equity_curve:
        return {
            "sharpe": 0, "sortino": 0, "perm_p": 1.0, "regime_gap": 1.0,
            "max_dd_pct": -1.0, "n_trades": 0, "final_equity": INITIAL_CAPITAL,
            "total_return_pct": 0, "win_rate": 0, "profit_factor": 0,
            "gates_passed": 0, "gate_details": {},
        }

    eq = pd.DataFrame(equity_curve)
    eq["equity"] = eq["equity"].astype(float)
    daily_ret = eq["equity"].pct_change().dropna()

    # Sharpe (annualized)
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1
    max_dd = dd.min()

    # Win rate, profit factor
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    win_rate = len(wins) / len(rets) if rets else 0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Regime-stratified Sharpe
    bull_rets = [t["ret"] for t in trades if t["regime"] == 1]
    bear_rets = [t["ret"] for t in trades if t["regime"] == 0]

    def _sharpe_from_rets(r):
        if len(r) < 2:
            return 0
        arr = np.array(r)
        if arr.std() == 0:
            return 0
        return (arr.mean() / arr.std()) * np.sqrt(252 / 20)  # approx annualize per-trade

    sharpe_bull = _sharpe_from_rets(bull_rets)
    sharpe_bear = _sharpe_from_rets(bear_rets)
    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    # Permutation test
    actual_mean = np.mean(rets)
    rng = np.random.RandomState(42)
    rets_arr = np.array(rets)
    count_better = 0
    for _ in range(PERM_ITERATIONS):
        shuffled = rets_arr.copy()
        rng.shuffle(shuffled)
        # Randomly flip signs to break temporal structure
        signs = rng.choice([-1, 1], size=len(shuffled))
        if np.mean(shuffled * signs) >= actual_mean:
            count_better += 1
    perm_p = count_better / PERM_ITERATIONS

    n_trades = len(trades)
    final_eq = equity_curve[-1]["equity"]
    total_ret = (final_eq / INITIAL_CAPITAL - 1) * 100

    # Gate checks
    gate_details = {
        "sharpe_gt_0.5": sharpe > GATES["sharpe_min"],
        "perm_p_lt_0.05": perm_p < GATES["perm_p_max"],
        "regime_gap_lt_0.5": regime_gap < GATES["regime_gap_max"],
        "max_dd_gt_neg50pct": max_dd > GATES["max_dd_floor"],
        "trades_gte_20": n_trades >= GATES["min_trades"],
    }
    gates_passed = sum(gate_details.values())

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "perm_p": round(perm_p, 4),
        "regime_gap": round(regime_gap, 3),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "n_trades": n_trades,
        "final_equity": round(final_eq, 2),
        "total_return_pct": round(total_ret, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(pf, 2),
        "gates_passed": gates_passed,
        "gate_details": gate_details,
    }


# ─── QQQ benchmark ───────────────────────────────────────────────────────────
def qqq_benchmark(close_df, oot_dates_arr):
    qqq = close_df["QQQ"]
    start_price = qqq.loc[oot_dates_arr[0]]
    shares = int(INITIAL_CAPITAL / start_price)
    remainder = INITIAL_CAPITAL - shares * start_price
    eq = []
    for d in oot_dates_arr:
        p = qqq.loc[d]
        if pd.isna(p):
            continue
        eq.append({"date": str(d.date()), "equity": shares * p + remainder})
    return compute_metrics([], eq) if not eq else {
        "final_equity": round(eq[-1]["equity"], 2),
        "total_return_pct": round((eq[-1]["equity"] / INITIAL_CAPITAL - 1) * 100, 2),
    }


# ─── Run all variants ────────────────────────────────────────────────────────
VARIANTS = {
    "A_price_acceleration": {"func": signal_A, "hold": 20, "max_pos": 3},
    "B_volume_confirmed":   {"func": signal_B, "hold": 20, "max_pos": 3},
    "C_rsi_momentum_shift": {"func": signal_C, "hold": 15, "max_pos": 3},
    "D_macd_hist_accel":    {"func": signal_D, "hold": 10, "max_pos": 2},
    "E_breakout_consol":    {"func": signal_E, "hold": 10, "max_pos": 3},
    "F_multi_composite":    {"func": signal_F, "hold": 15, "max_pos": 3},
}

results = {}
oot_dates_np = oot_dates.to_numpy()
# Convert to list of pd.Timestamp for consistent indexing
oot_dates_list = [pd.Timestamp(d) for d in oot_dates_np]

print(f"\nRunning {len(VARIANTS)} variants with {PERM_ITERATIONS} permutations each …\n")

for name, cfg in VARIANTS.items():
    print(f"  {name} …", end=" ", flush=True)
    trades, eq_curve = run_backtest(
        cfg["func"], cfg["hold"], indicators, close, oot_dates_list, cfg["max_pos"]
    )
    metrics = compute_metrics(trades, eq_curve)
    results[name] = metrics
    passed = metrics["gates_passed"]
    print(f"Sharpe={metrics['sharpe']:.2f}  MDD={metrics['max_dd_pct']:.1f}%  "
          f"Trades={metrics['n_trades']}  Gates={passed}/5  "
          f"Final=${metrics['final_equity']:.0f}")

# Benchmark
bench = qqq_benchmark(close, oot_dates_list)
results["benchmark_QQQ_buyhold"] = bench

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MOMENTUM ACCELERATION BACKTEST — SUMMARY")
print("=" * 80)
print(f"{'Variant':<30} {'Sharpe':>7} {'Perm-p':>7} {'RGap':>6} {'MDD%':>7} "
      f"{'#Tr':>5} {'WR%':>6} {'PF':>6} {'Final$':>8} {'Gates':>6}")
print("-" * 80)

for name, m in results.items():
    if name.startswith("benchmark"):
        print(f"{'QQQ Buy&Hold (bench)':<30} {'--':>7} {'--':>7} {'--':>6} {'--':>7} "
              f"{'--':>5} {'--':>6} {'--':>6} {m['final_equity']:>8.0f} {'--':>6}")
    else:
        print(f"{name:<30} {m['sharpe']:>7.2f} {m['perm_p']:>7.3f} {m['regime_gap']:>6.2f} "
              f"{m['max_dd_pct']:>7.1f} {m['n_trades']:>5} {m['win_rate']:>6.1f} "
              f"{m['profit_factor']:>6.2f} {m['final_equity']:>8.0f} {m['gates_passed']:>4}/5")

# Gate details
print("\nGATE DETAILS:")
for name, m in results.items():
    if name.startswith("benchmark"):
        continue
    gd = m["gate_details"]
    status = " | ".join(f"{'PASS' if v else 'FAIL'} {k}" for k, v in gd.items())
    print(f"  {name}: {status}")

# Save results
output_path = Path("/home/jupiter/Lvl3Quant/data/momentum_acceleration_results.json")
with open(output_path, "w") as f:
    json.dump({
        "run_date": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {close.index[-1].date()}",
        "initial_capital": INITIAL_CAPITAL,
        "universe": UNIVERSE,
        "perm_iterations": PERM_ITERATIONS,
        "gates": GATES,
        "variants": results,
    }, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
