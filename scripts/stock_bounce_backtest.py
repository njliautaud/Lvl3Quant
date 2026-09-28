#!/usr/bin/env python3
"""
Oversold Bounce Backtest — 6 Variants on 20 Mega-Cap Stocks
=============================================================
Universe: AAPL MSFT AMZN GOOGL META NVDA TSLA JPM V MA UNH HD PG JNJ XOM CVX BAC WMT COST DIS
OOT period: Jan 2022 – Jul 2026
Capital: $645, max 1 position at a time, fractional shares, $0 commission, 0.02% slippage.

Variants:
  A) RSI-14 Oversold Bounce (RSI<30 → buy, sell RSI>50 or 3d)
  B) 3-Day Drop Buy (>5% 3d drop → buy, sell after 5d)
  C) Bollinger Band Mean Reversion (close < lower BB → buy, sell at SMA or 10d)
  D) Sector-Relative Oversold (underperform sector ETF >3% over 5d → buy, sell 5d)
  E) Volume Exhaustion Buy (>3% drop on 2x avg vol → buy, sell 3d)
  F) Multi-Factor Bounce (RSI<35 + >3% 5d drop + vol>1.5x → buy, sell 5d)

Validation 5-gate: Sharpe>0.5, perm_p<0.05, regime_gap<0.5, MDD>-50%, trades>=20.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "V", "MA", "UNH", "HD", "PG", "JNJ",
    "XOM", "CVX", "BAC", "WMT", "COST", "DIS",
]

SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "AMZN": "XLY", "GOOGL": "XLK",
    "META": "XLK", "NVDA": "XLK", "TSLA": "XLY",
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF",
    "UNH": "XLV", "JNJ": "XLV", "PG": "XLP", "WMT": "XLP", "COST": "XLP",
    "HD": "XLY", "DIS": "XLY",
    "XOM": "XLE", "CVX": "XLE",
}

SECTOR_ETFS = list(set(SECTOR_MAP.values()))

CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
START_DATE = "2021-06-01"  # extra buffer for indicators
END_DATE = "2026-07-29"
OOT_START = "2022-01-01"

N_PERM = 500

# ── DATA DOWNLOAD ──────────────────────────────────────────────────────────────
print("Downloading price data...")
all_tickers = UNIVERSE + SECTOR_ETFS + ["SPY"]
all_tickers = sorted(set(all_tickers))

data_raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                        auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
close = data_raw["Close"].copy()
volume = data_raw["Volume"].copy()

# Forward fill small gaps, drop rows with too many NaNs
close = close.ffill().dropna(how="all")
volume = volume.ffill().fillna(0)

# Align
common_idx = close.index
volume = volume.reindex(common_idx)

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# ── INDICATORS ──────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

print("Computing indicators...")
rsi = pd.DataFrame(index=close.index)
for t in UNIVERSE:
    if t in close.columns:
        rsi[t] = compute_rsi(close[t])

sma20 = close[UNIVERSE].rolling(20).mean()
std20 = close[UNIVERSE].rolling(20).std()
bb_lower = sma20 - 2 * std20

vol_avg20 = volume[UNIVERSE].rolling(20).mean()

ret_1d = close.pct_change()
ret_3d = close.pct_change(3)
ret_5d = close.pct_change(5)

# SPY regime: SPY > 200-SMA = bull
spy_sma200 = close["SPY"].rolling(200).mean()
regime = (close["SPY"] > spy_sma200).astype(int)  # 1=bull, 0=bear

# ── BACKTEST ENGINE ────────────────────────────────────────────────────────────
def run_backtest(signal_func, exit_func, variant_name, randomize_stock=False, rng=None):
    """
    Generic backtest engine.
    signal_func(date_idx, date) -> list of (ticker, score) where lower score = more oversold
    exit_func(ticker, entry_idx, entry_price, current_idx) -> bool (True = exit)
    """
    oot_mask = close.index >= pd.Timestamp(OOT_START)
    oot_dates = close.index[oot_mask]

    trades = []
    equity_curve = []
    cash = CAPITAL
    position = None  # (ticker, shares, entry_price, entry_idx, entry_date)

    for i_abs in range(len(close)):
        date = close.index[i_abs]
        if date < pd.Timestamp(OOT_START):
            continue

        # Check exit first
        if position is not None:
            ticker, shares, entry_price, entry_idx, entry_date = position
            current_price = close[ticker].iloc[i_abs]
            if pd.isna(current_price):
                current_price = entry_price

            should_exit = exit_func(ticker, entry_idx, entry_price, i_abs)
            if should_exit:
                exit_price = current_price * (1 - SLIPPAGE_PCT)  # selling
                pnl = (exit_price - entry_price) * shares
                pnl_pct = (exit_price / entry_price) - 1
                cash += exit_price * shares
                days_held = i_abs - entry_idx

                trade_regime = "bull" if regime.iloc[entry_idx] == 1 else "bear"
                trades.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": round(shares, 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "days_held": days_held,
                    "regime": trade_regime,
                })
                position = None

        # Check entry (only if flat)
        if position is None:
            candidates = signal_func(i_abs, date)

            if randomize_stock and candidates and rng is not None:
                # For permutation test: pick random stock instead of most oversold
                valid_tickers = [t for t in UNIVERSE if not pd.isna(close[t].iloc[i_abs])]
                if valid_tickers:
                    rand_ticker = rng.choice(valid_tickers)
                    candidates = [(rand_ticker, 0.0)]
                else:
                    candidates = []

            if candidates:
                # Pick most oversold (lowest score)
                candidates.sort(key=lambda x: x[1])
                best_ticker = candidates[0][0]
                price = close[best_ticker].iloc[i_abs]
                if pd.notna(price) and price > 0:
                    buy_price = price * (1 + SLIPPAGE_PCT)  # buying
                    shares = cash / buy_price  # fractional
                    position = (best_ticker, shares, buy_price, i_abs, date)
                    cash = 0.0

        # Track equity
        if position is not None:
            ticker, shares, entry_price, entry_idx, entry_date = position
            current_price = close[ticker].iloc[i_abs]
            if pd.isna(current_price):
                current_price = entry_price
            equity = cash + current_price * shares
        else:
            equity = cash
        equity_curve.append(equity)

    # Force close if still in position at end
    if position is not None:
        ticker, shares, entry_price, entry_idx, entry_date = position
        current_price = close[ticker].iloc[-1]
        if pd.isna(current_price):
            current_price = entry_price
        exit_price = current_price * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - entry_price) * shares
        cash += exit_price * shares
        trades.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()),
            "exit_date": str(close.index[-1].date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "shares": round(shares, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round((exit_price / entry_price - 1) * 100, 3),
            "days_held": len(close) - 1 - entry_idx,
            "regime": "bull" if regime.iloc[entry_idx] == 1 else "bear",
        })
        position = None

    return trades, equity_curve


def compute_metrics(trades, equity_curve, variant_name):
    """Compute performance metrics + 5-gate validation."""
    if not trades:
        return {
            "variant": variant_name, "n_trades": 0, "total_pnl": 0,
            "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "mdd_pct": 0,
            "avg_pnl_pct": 0, "gates_passed": 0, "gate_details": {},
        }

    trade_df = pd.DataFrame(trades)
    n_trades = len(trade_df)
    total_pnl = trade_df["pnl"].sum()
    avg_pnl = trade_df["pnl"].mean()
    avg_pnl_pct = trade_df["pnl_pct"].mean()
    wr = (trade_df["pnl"] > 0).mean()

    gross_profit = trade_df.loc[trade_df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trade_df.loc[trade_df["pnl"] < 0, "pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Daily returns from equity curve
    eq = np.array(equity_curve)
    daily_ret = np.diff(eq) / eq[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]

    sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) > 0 else 0

    downside = daily_ret[daily_ret < 0]
    sortino = np.mean(daily_ret) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = dd.min() * 100

    # Regime analysis
    bull_trades = trade_df[trade_df["regime"] == "bull"]
    bear_trades = trade_df[trade_df["regime"] == "bear"]

    bull_sharpe = 0
    bear_sharpe = 0
    if len(bull_trades) > 1:
        bull_rets = bull_trades["pnl_pct"].values / 100
        bull_sharpe = np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252 / max(bull_trades["days_held"].mean(), 1)) if np.std(bull_rets) > 0 else 0
    if len(bear_trades) > 1:
        bear_rets = bear_trades["pnl_pct"].values / 100
        bear_sharpe = np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252 / max(bear_trades["days_held"].mean(), 1)) if np.std(bear_rets) > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)

    # 5-gate validation (permutation done separately)
    gate_sharpe = sharpe > 0.5
    gate_mdd = mdd > -50
    gate_trades = n_trades >= 20
    gate_regime = regime_gap < 0.5 if (len(bull_trades) > 2 and len(bear_trades) > 2) else True  # pass if insufficient data

    gates_passed = sum([gate_sharpe, gate_mdd, gate_trades, gate_regime])

    return {
        "variant": variant_name,
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((total_pnl / CAPITAL) * 100, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "avg_pnl_pct": round(avg_pnl_pct, 3),
        "win_rate": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd, 2),
        "avg_days_held": round(trade_df["days_held"].mean(), 1),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_avg_pnl_pct": round(bull_trades["pnl_pct"].mean(), 3) if len(bull_trades) > 0 else 0,
        "bear_avg_pnl_pct": round(bear_trades["pnl_pct"].mean(), 3) if len(bear_trades) > 0 else 0,
        "bull_sharpe_approx": round(bull_sharpe, 3),
        "bear_sharpe_approx": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "gates_passed_of_5": gates_passed,  # perm gate added later
        "gate_details": {
            "sharpe_gt_0.5": gate_sharpe,
            "mdd_gt_neg50": gate_mdd,
            "trades_gte_20": gate_trades,
            "regime_gap_lt_0.5": gate_regime,
            "perm_p_lt_0.05": None,  # filled later
        },
    }


# ── STRATEGY DEFINITIONS ──────────────────────────────────────────────────────

# A) RSI-14 Oversold Bounce
def signal_A(i, date):
    candidates = []
    for t in UNIVERSE:
        if t in rsi.columns and i < len(rsi) and pd.notna(rsi[t].iloc[i]):
            if rsi[t].iloc[i] < 30:
                candidates.append((t, rsi[t].iloc[i]))
    return candidates

def exit_A(ticker, entry_idx, entry_price, current_idx):
    days_held = current_idx - entry_idx
    if days_held >= 3:
        return True
    if ticker in rsi.columns and current_idx < len(rsi) and pd.notna(rsi[ticker].iloc[current_idx]):
        if rsi[ticker].iloc[current_idx] > 50:
            return True
    return False

# B) 3-Day Drop Buy
def signal_B(i, date):
    candidates = []
    for t in UNIVERSE:
        if t in ret_3d.columns and i < len(ret_3d) and pd.notna(ret_3d[t].iloc[i]):
            if ret_3d[t].iloc[i] < -0.05:
                candidates.append((t, ret_3d[t].iloc[i]))
    return candidates

def exit_B(ticker, entry_idx, entry_price, current_idx):
    return (current_idx - entry_idx) >= 5

# C) Bollinger Band Mean Reversion
def signal_C(i, date):
    candidates = []
    for t in UNIVERSE:
        if t in close.columns and t in bb_lower.columns:
            if i < len(close) and pd.notna(close[t].iloc[i]) and pd.notna(bb_lower[t].iloc[i]):
                if close[t].iloc[i] < bb_lower[t].iloc[i]:
                    # Score: how far below BB (more negative = more oversold)
                    dist = (close[t].iloc[i] - bb_lower[t].iloc[i]) / bb_lower[t].iloc[i]
                    candidates.append((t, dist))
    return candidates

def exit_C(ticker, entry_idx, entry_price, current_idx):
    days_held = current_idx - entry_idx
    if days_held >= 10:
        return True
    if ticker in close.columns and ticker in sma20.columns:
        if current_idx < len(close) and pd.notna(close[ticker].iloc[current_idx]) and pd.notna(sma20[ticker].iloc[current_idx]):
            if close[ticker].iloc[current_idx] >= sma20[ticker].iloc[current_idx]:
                return True
    return False

# D) Sector-Relative Oversold
def signal_D(i, date):
    candidates = []
    for t in UNIVERSE:
        sector_etf = SECTOR_MAP.get(t)
        if sector_etf and sector_etf in ret_5d.columns and t in ret_5d.columns:
            if i < len(ret_5d) and pd.notna(ret_5d[t].iloc[i]) and pd.notna(ret_5d[sector_etf].iloc[i]):
                relative_perf = ret_5d[t].iloc[i] - ret_5d[sector_etf].iloc[i]
                if relative_perf < -0.03:
                    candidates.append((t, relative_perf))
    return candidates

def exit_D(ticker, entry_idx, entry_price, current_idx):
    return (current_idx - entry_idx) >= 5

# E) Volume Exhaustion Buy
def signal_E(i, date):
    candidates = []
    for t in UNIVERSE:
        if (t in ret_1d.columns and t in volume.columns and t in vol_avg20.columns):
            if i < len(ret_1d) and pd.notna(ret_1d[t].iloc[i]) and pd.notna(volume[t].iloc[i]) and pd.notna(vol_avg20[t].iloc[i]):
                if ret_1d[t].iloc[i] < -0.03 and vol_avg20[t].iloc[i] > 0:
                    vol_ratio = volume[t].iloc[i] / vol_avg20[t].iloc[i]
                    if vol_ratio > 2.0:
                        candidates.append((t, ret_1d[t].iloc[i]))  # more negative = more oversold
    return candidates

def exit_E(ticker, entry_idx, entry_price, current_idx):
    return (current_idx - entry_idx) >= 3

# F) Multi-Factor Bounce
def signal_F(i, date):
    candidates = []
    for t in UNIVERSE:
        if (t in rsi.columns and t in ret_5d.columns and t in volume.columns and t in vol_avg20.columns):
            if i < len(rsi):
                rsi_val = rsi[t].iloc[i] if pd.notna(rsi[t].iloc[i]) else 100
                ret5_val = ret_5d[t].iloc[i] if pd.notna(ret_5d[t].iloc[i]) else 0
                vol_val = volume[t].iloc[i] if pd.notna(volume[t].iloc[i]) else 0
                vol_avg = vol_avg20[t].iloc[i] if pd.notna(vol_avg20[t].iloc[i]) else 1

                if rsi_val < 35 and ret5_val < -0.03 and vol_avg > 0 and (vol_val / vol_avg) > 1.5:
                    # Composite score: lower RSI + bigger drop = more oversold
                    score = rsi_val / 100 + ret5_val
                    candidates.append((t, score))
    return candidates

def exit_F(ticker, entry_idx, entry_price, current_idx):
    return (current_idx - entry_idx) >= 5


# ── RUN ALL VARIANTS ──────────────────────────────────────────────────────────
strategies = [
    ("A_RSI14_Oversold", signal_A, exit_A),
    ("B_3Day_Drop", signal_B, exit_B),
    ("C_Bollinger_MeanRev", signal_C, exit_C),
    ("D_Sector_Relative", signal_D, exit_D),
    ("E_Volume_Exhaustion", signal_E, exit_E),
    ("F_MultiFactor_Bounce", signal_F, exit_F),
]

all_results = {}

for name, sig_func, exit_func in strategies:
    print(f"\n{'='*60}")
    print(f"Running variant: {name}")
    print(f"{'='*60}")

    trades, equity = run_backtest(sig_func, exit_func, name)
    metrics = compute_metrics(trades, equity, name)

    # Permutation test: randomize stock selection
    print(f"  Running {N_PERM} permutations...")
    real_sharpe = metrics["sharpe"]
    perm_sharpes = []
    rng = np.random.default_rng(42)

    for p in range(N_PERM):
        p_trades, p_equity = run_backtest(sig_func, exit_func, name,
                                           randomize_stock=True, rng=rng)
        if p_equity:
            eq = np.array(p_equity)
            dr = np.diff(eq) / eq[:-1]
            dr = dr[np.isfinite(dr)]
            p_sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0
        else:
            p_sharpe = 0
        perm_sharpes.append(p_sharpe)

    perm_p = np.mean(np.array(perm_sharpes) >= real_sharpe) if perm_sharpes else 1.0
    metrics["perm_p_value"] = round(perm_p, 4)
    metrics["gate_details"]["perm_p_lt_0.05"] = perm_p < 0.05
    if perm_p < 0.05:
        metrics["gates_passed_of_5"] += 1

    # Print summary
    print(f"\n  Trades: {metrics['n_trades']}")
    print(f"  Total P&L: ${metrics['total_pnl']:.2f} ({metrics['total_return_pct']:.1f}%)")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Perm p-value: {metrics['perm_p_value']:.4f}")
    print(f"  Bull avg: {metrics['bull_avg_pnl_pct']:.3f}%  Bear avg: {metrics['bear_avg_pnl_pct']:.3f}%")
    print(f"  Regime gap: {metrics['regime_gap']:.3f}")
    print(f"  Gates passed: {metrics['gates_passed_of_5']}/5")

    # Store
    all_results[name] = {
        "metrics": metrics,
        "trades": trades,
        "equity_final": round(equity[-1], 2) if equity else CAPITAL,
    }

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("FINAL SUMMARY — OVERSOLD BOUNCE STRATEGIES")
print("="*80)
print(f"{'Variant':<25} {'Trades':>6} {'P&L':>10} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MDD':>7} {'Perm-p':>7} {'Gates':>6}")
print("-"*95)

for name in all_results:
    m = all_results[name]["metrics"]
    print(f"{m['variant']:<25} {m['n_trades']:>6} ${m['total_pnl']:>8.2f} {m['win_rate']:>5.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>6.1f}% {m['perm_p_value']:>7.4f} {m['gates_passed_of_5']:>4}/5")

# Best strategy
best = max(all_results.keys(), key=lambda k: all_results[k]["metrics"]["sharpe"])
best_m = all_results[best]["metrics"]
print(f"\nBest by Sharpe: {best} (Sharpe={best_m['sharpe']:.3f}, {best_m['gates_passed_of_5']}/5 gates)")

# ── SAVE RESULTS ───────────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/stock_bounce_results.json")

# Convert for JSON serialization
save_data = {
    "run_date": datetime.now().isoformat(),
    "config": {
        "capital": CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "oot_period": f"{OOT_START} to {END_DATE}",
        "universe": UNIVERSE,
        "n_permutations": N_PERM,
    },
    "summary": {},
    "variant_details": {},
}

for name, result in all_results.items():
    save_data["summary"][name] = result["metrics"]
    save_data["variant_details"][name] = {
        "trades": result["trades"],
        "equity_final": result["equity_final"],
    }

# Find best
save_data["recommendation"] = {
    "best_by_sharpe": best,
    "best_metrics": best_m,
    "passing_all_5_gates": [
        name for name in all_results
        if all_results[name]["metrics"]["gates_passed_of_5"] == 5
    ],
}

with open(output_path, "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
