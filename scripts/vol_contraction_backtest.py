#!/usr/bin/env python3
"""
Volatility Contraction Breakout (Reverse) on Quality Stocks
Tests whether LOW volatility during dips predicts stronger mean-reversion.

6 Variants: A-F (BBW Squeeze, ATR Contraction, Range Contraction,
            HV Decline, Inside Day, Keltner Squeeze)
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
HOLD_DAYS = 10
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
ATR_FAST = 5
ATR_SLOW = 20
DIP_PCT = 0.05  # 5% below 20-day high

# ── HELPERS ─────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_bb(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std * sd
    lower = mid - std * sd
    bbw = (upper - lower) / mid
    return upper, mid, lower, bbw


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_keltner(close: pd.Series, high: pd.Series, low: pd.Series,
                    ema_period: int = 20, atr_period: int = 10, atr_mult: float = 1.5):
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr = compute_atr(high, low, close, atr_period)
    kc_upper = ema + atr_mult * atr
    kc_lower = ema - atr_mult * atr
    return kc_upper, kc_lower


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ── SIGNAL GENERATORS (return boolean Series) ──────────────────────────

def signal_A(df: pd.DataFrame) -> pd.Series:
    """BBW Squeeze: BBW < 20d avg BBW AND >5% below 20d high AND RSI<40"""
    _, _, _, bbw = compute_bb(df["Close"], BB_PERIOD, BB_STD)
    bbw_avg = bbw.rolling(20).mean()
    high20 = df["Close"].rolling(20).max()
    rsi = compute_rsi(df["Close"], RSI_PERIOD)
    return (bbw < bbw_avg) & (df["Close"] < high20 * (1 - DIP_PCT)) & (rsi < 40)


def signal_B(df: pd.DataFrame) -> pd.Series:
    """ATR Contraction: 5d ATR < 0.7× 20d ATR AND dip AND RSI<40"""
    atr5 = compute_atr(df["High"], df["Low"], df["Close"], ATR_FAST)
    atr20 = compute_atr(df["High"], df["Low"], df["Close"], ATR_SLOW)
    high20 = df["Close"].rolling(20).max()
    rsi = compute_rsi(df["Close"], RSI_PERIOD)
    return (atr5 < 0.7 * atr20) & (df["Close"] < high20 * (1 - DIP_PCT)) & (rsi < 40)


def signal_C(df: pd.DataFrame) -> pd.Series:
    """Range Contraction: today range < 50% of 20d avg range AND dip AND RSI<40"""
    daily_range = df["High"] - df["Low"]
    avg_range = daily_range.rolling(20).mean()
    high20 = df["Close"].rolling(20).max()
    rsi = compute_rsi(df["Close"], RSI_PERIOD)
    return (daily_range < 0.5 * avg_range) & (df["Close"] < high20 * (1 - DIP_PCT)) & (rsi < 40)


def signal_D(df: pd.DataFrame) -> pd.Series:
    """HV Decline: 5d vol < 10d vol < 20d vol AND dip (no RSI gate)"""
    hv5 = realized_vol(df["Close"], 5)
    hv10 = realized_vol(df["Close"], 10)
    hv20 = realized_vol(df["Close"], 20)
    high20 = df["Close"].rolling(20).max()
    return (hv5 < hv10) & (hv10 < hv20) & (df["Close"] < high20 * (1 - DIP_PCT))


def signal_E(df: pd.DataFrame) -> pd.Series:
    """Inside Day: today high < yesterday high AND today low > yesterday low AND dip AND RSI<40"""
    inside = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
    high20 = df["Close"].rolling(20).max()
    rsi = compute_rsi(df["Close"], RSI_PERIOD)
    return inside & (df["Close"] < high20 * (1 - DIP_PCT)) & (rsi < 40)


def signal_F(df: pd.DataFrame) -> pd.Series:
    """Keltner Squeeze: BB inside KC AND dip AND RSI<40"""
    bb_upper, _, bb_lower, _ = compute_bb(df["Close"], BB_PERIOD, BB_STD)
    kc_upper, kc_lower = compute_keltner(df["Close"], df["High"], df["Low"])
    high20 = df["Close"].rolling(20).max()
    rsi = compute_rsi(df["Close"], RSI_PERIOD)
    return (bb_upper < kc_upper) & (bb_lower > kc_lower) & \
           (df["Close"] < high20 * (1 - DIP_PCT)) & (rsi < 40)


VARIANTS = {
    "A_BBW_Squeeze": signal_A,
    "B_ATR_Contraction": signal_B,
    "C_Range_Contraction": signal_C,
    "D_HV_Decline": signal_D,
    "E_Inside_Day": signal_E,
    "F_Keltner_Squeeze": signal_F,
}

# ── DOWNLOAD DATA ──────────────────────────────────────────────────────

print(f"Downloading {len(UNIVERSE)} tickers from {START} to {END}...")
raw = yf.download(UNIVERSE, start=START, end=END, group_by="ticker", auto_adjust=True, progress=False)
print(f"Downloaded {len(raw)} bars")

# Parse into per-ticker DataFrames
data = {}
for ticker in UNIVERSE:
    try:
        tdf = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(tdf) > 50:
            data[ticker] = tdf
    except Exception:
        pass
print(f"Usable tickers: {len(data)}")

# Get SPY for regime classification
spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
spy_ret = spy["Close"].pct_change()

# ── BACKTEST ENGINE ────────────────────────────────────────────────────

def run_backtest(variant_name: str, signal_fn) -> dict:
    """Run a single variant backtest with position management."""
    trades = []

    for ticker, df in data.items():
        sig = signal_fn(df)
        dates = df.index[sig]

        for entry_date in dates:
            idx = df.index.get_loc(entry_date)
            if idx + HOLD_DAYS >= len(df):
                continue

            entry_price = df["Close"].iloc[idx]
            exit_price = df["Close"].iloc[idx + HOLD_DAYS]

            # Slippage
            entry_cost = entry_price * (1 + SLIPPAGE_BPS / 10000)
            exit_rev = exit_price * (1 - SLIPPAGE_BPS / 10000)

            shares = min(MAX_PER_TRADE, CAPITAL / MAX_CONCURRENT) / entry_cost
            pnl = (exit_rev - entry_cost) * shares
            ret = (exit_rev / entry_cost) - 1

            trades.append({
                "ticker": ticker,
                "entry_date": str(df.index[idx].date()),
                "exit_date": str(df.index[idx + HOLD_DAYS].date()),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "pnl": round(float(pnl), 4),
                "return": round(float(ret), 6),
            })

    if not trades:
        return {"variant": variant_name, "num_trades": 0, "skipped": True}

    # Sort by entry date for concurrent position management
    trades.sort(key=lambda t: t["entry_date"])

    # Apply concurrent position limit
    filtered = []
    active_exits = []
    for t in trades:
        # Remove expired positions
        active_exits = [ex for ex in active_exits if ex > t["entry_date"]]
        if len(active_exits) < MAX_CONCURRENT:
            filtered.append(t)
            active_exits.append(t["exit_date"])

    trades = filtered
    n_trades = len(trades)
    if n_trades < 1:
        return {"variant": variant_name, "num_trades": 0, "skipped": True}

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    # Build equity curve (simple sequential)
    equity = CAPITAL
    eq_curve = [CAPITAL]
    for p in pnls:
        equity += p
        eq_curve.append(equity)
    eq_curve = np.array(eq_curve)

    # Drawdown
    peak = np.maximum.accumulate(eq_curve)
    dd = (eq_curve - peak) / peak
    max_dd = float(dd.min())

    # Stats
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns)) if np.std(returns) > 0 else 1e-9
    win_rate = float(np.mean(returns > 0))
    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))

    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 1e-9
    profit_factor = gross_profit / gross_loss

    # Annualized Sharpe (assume ~25 trades/year ≈ 10-day holds)
    trades_per_year = 252 / HOLD_DAYS
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)

    # Sortino
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year)

    # ── REGIME ANALYSIS ──
    green_rets, red_rets = [], []
    for t in trades:
        edate = t["entry_date"]
        # Find SPY return on entry date
        try:
            spy_r = float(spy_ret.loc[edate]) if edate in spy_ret.index else 0
        except:
            spy_r = 0
        if spy_r >= 0:
            green_rets.append(t["return"])
        else:
            red_rets.append(t["return"])

    green_sharpe = (np.mean(green_rets) / np.std(green_rets) * np.sqrt(trades_per_year)) \
        if len(green_rets) > 2 and np.std(green_rets) > 0 else 0
    red_sharpe = (np.mean(red_rets) / np.std(red_rets) * np.sqrt(trades_per_year)) \
        if len(red_rets) > 2 and np.std(red_rets) > 0 else 0

    max_abs = max(abs(green_sharpe), abs(red_sharpe), 1e-9)
    regime_gap = abs(green_sharpe - red_sharpe) / max_abs

    # ── PERMUTATION TEST ──
    observed_mean = mean_ret
    n_perms = 1000
    rng = np.random.default_rng(42)
    perm_means = np.array([rng.permutation(returns).mean() for _ in range(n_perms)])
    p_value = float(np.mean(perm_means >= observed_mean))

    # ── 5-GATE VALIDATION ──
    gate_sharpe = sharpe > 0.5
    gate_perm = p_value < 0.05
    gate_regime = regime_gap < 0.5
    gate_dd = max_dd > -0.50
    gate_trades = n_trades >= 20
    gates_passed = sum([gate_sharpe, gate_perm, gate_regime, gate_dd, gate_trades])
    all_pass = gates_passed == 5

    result = {
        "variant": variant_name,
        "num_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "mean_return_pct": round(mean_ret * 100, 3),
        "win_rate": round(win_rate, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "final_equity": round(float(eq_curve[-1]), 2),
        "regime": {
            "green_sharpe": round(float(green_sharpe), 3),
            "red_sharpe": round(float(red_sharpe), 3),
            "regime_gap": round(float(regime_gap), 3),
            "green_trades": len(green_rets),
            "red_trades": len(red_rets),
        },
        "permutation": {
            "p_value": round(p_value, 4),
            "observed_mean_ret": round(observed_mean * 100, 4),
        },
        "gates": {
            "sharpe_gt_0.5": gate_sharpe,
            "perm_p_lt_0.05": gate_perm,
            "regime_gap_lt_0.5": gate_regime,
            "max_dd_gt_neg50": gate_dd,
            "trades_gte_20": gate_trades,
            "passed": gates_passed,
            "all_pass": all_pass,
        },
    }
    return result


# ── RUN ALL VARIANTS ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("VOLATILITY CONTRACTION BREAKOUT (REVERSE) BACKTEST")
print("=" * 70)

results = {}
for name, fn in VARIANTS.items():
    print(f"\nRunning {name}...")
    res = run_backtest(name, fn)
    results[name] = res

    if res.get("skipped"):
        print(f"  SKIPPED — no trades generated")
        continue

    gates = res["gates"]
    tag = "PASS" if gates["all_pass"] else f"FAIL ({gates['passed']}/5)"
    print(f"  Trades: {res['num_trades']}  |  PnL: ${res['total_pnl']:.2f}  |  "
          f"WR: {res['win_rate']:.1%}  |  Sharpe: {res['sharpe']:.2f}  |  "
          f"Sortino: {res['sortino']:.2f}  |  PF: {res['profit_factor']:.2f}  |  "
          f"MaxDD: {res['max_drawdown_pct']:.1f}%  |  [{tag}]")
    print(f"  Regime: green_S={res['regime']['green_sharpe']:.2f} "
          f"red_S={res['regime']['red_sharpe']:.2f} gap={res['regime']['regime_gap']:.2f}  |  "
          f"Perm p={res['permutation']['p_value']:.4f}")

# ── SUMMARY ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Variant':<25} {'Trades':>6} {'PnL':>9} {'WR':>6} {'Sharpe':>7} "
      f"{'Sortino':>8} {'PF':>6} {'MaxDD':>7} {'Gates':>6}")
print("-" * 90)

passed_variants = []
for name, res in results.items():
    if res.get("skipped"):
        print(f"{name:<25} {'—':>6} {'—':>9} {'—':>6} {'—':>7} {'—':>8} {'—':>6} {'—':>7} {'—':>6}")
        continue
    g = res["gates"]
    tag = "5/5" if g["all_pass"] else f"{g['passed']}/5"
    print(f"{name:<25} {res['num_trades']:>6} {res['total_pnl']:>9.2f} "
          f"{res['win_rate']:>5.1%} {res['sharpe']:>7.2f} {res['sortino']:>8.2f} "
          f"{res['profit_factor']:>6.2f} {res['max_drawdown_pct']:>6.1f}% {tag:>6}")
    if g["all_pass"]:
        passed_variants.append(name)

print(f"\nVariants passing all 5 gates: {passed_variants if passed_variants else 'NONE'}")

# ── SAVE ───────────────────────────────────────────────────────────────

output = {
    "strategy": "Volatility Contraction Breakout (Reverse)",
    "universe": UNIVERSE,
    "period": f"{START} to {END}",
    "capital": CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_bps": SLIPPAGE_BPS,
    "hold_days": HOLD_DAYS,
    "run_timestamp": datetime.now().isoformat(),
    "variants": results,
    "passed_variants": passed_variants,
}

out_path = Path("/home/jupiter/Lvl3Quant/data/vol_contraction_results.json")
out_path.write_text(json.dumps(output, indent=2, default=str))
print(f"\nResults saved to {out_path}")
