#!/usr/bin/env python3
"""
Correlation Regime Allocation Backtest
======================================
Shifts between concentrated growth exposure and diversified safe havens
based on cross-asset correlation dynamics.

6 Variants: A-F
Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation per variant.

Author: Claude Opus 4.6
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")

def log(msg):
    print(msg)
    sys.stdout.flush()

# ── Configuration ──────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%

GROWTH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM"]
SAFE_HAVENS = ["GLD", "TLT", "UUP", "SHY"]
INDEX = ["SPY", "QQQ"]
ALL_TICKERS = list(set(GROWTH + SAFE_HAVENS + INDEX + ["^VIX"]))

OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
FETCH_START = "2020-06-01"

PERM_ITERATIONS = 1000
np.random.seed(42)


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    log("Downloading price data...")
    data = yf.download(ALL_TICKERS, start=FETCH_START, end=OOT_END, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    if "^VIX" in close.columns:
        close = close.rename(columns={"^VIX": "VIX"})

    close = close.ffill().bfill()
    log(f"  Data shape: {close.shape}, {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ── Signal Computation ─────────────────────────────────────────────────────
def compute_rolling_pairwise_correlation(close, tickers, window=20):
    returns = close[tickers].pct_change()
    pairs = list(combinations(tickers, 2))
    corr_matrix = pd.DataFrame(index=close.index)
    for i, (t1, t2) in enumerate(pairs):
        corr_matrix[i] = returns[t1].rolling(window).corr(returns[t2])
    return corr_matrix.mean(axis=1)


def compute_rsi(series, period=5):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_rolling_beta(asset_returns, market_returns, window=20):
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def classify_regime(close):
    spy = close["SPY"]
    sma200 = spy.rolling(200).mean()
    regime = pd.Series("bull", index=close.index)
    regime[spy < sma200] = "bear"
    return regime


# ── Backtester (optimized) ─────────────────────────────────────────────────
def run_backtest(close, signal_df, capital=CAPITAL):
    """
    Optimized backtester. signal_df has columns: date, ticker, action.
    Returns (trades_list, equity_df).
    """
    if len(signal_df) == 0:
        return [], pd.DataFrame()

    # Pre-group signals by date for O(1) lookup
    sig_grouped = {}
    for _, row in signal_df.iterrows():
        d = row["date"]
        if d not in sig_grouped:
            sig_grouped[d] = []
        sig_grouped[d].append((row["ticker"], row["action"]))

    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    oot_dates = close.index[oot_mask]

    cash = capital
    position = None  # (ticker, shares, entry_price, entry_date)
    trades = []
    eq_dates = []
    eq_vals = []

    for date in oot_dates:
        if date in sig_grouped:
            for ticker, action in sig_grouped[date]:
                if action == "sell" and position is not None and position[0] == ticker:
                    exit_price = close.at[date, ticker] * (1 - SLIPPAGE_PCT)
                    proceeds = position[1] * exit_price
                    pnl = proceeds - (position[1] * position[2])
                    trades.append({
                        "entry_date": position[3],
                        "exit_date": date,
                        "ticker": position[0],
                        "shares": position[1],
                        "entry_price": position[2],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return_pct": pnl / (position[1] * position[2]) * 100,
                        "hold_days": (date - position[3]).days,
                    })
                    cash += proceeds
                    position = None
                elif action == "buy" and position is None:
                    price = close.at[date, ticker] * (1 + SLIPPAGE_PCT)
                    shares = int(cash / price)
                    if shares > 0:
                        cash -= shares * price
                        position = (ticker, shares, price, date)

        if position is not None:
            eq_vals.append(cash + position[1] * close.at[date, position[0]])
        else:
            eq_vals.append(cash)
        eq_dates.append(date)

    # Close open position at end
    if position is not None:
        last_date = oot_dates[-1]
        exit_price = close.at[last_date, position[0]] * (1 - SLIPPAGE_PCT)
        proceeds = position[1] * exit_price
        pnl = proceeds - (position[1] * position[2])
        trades.append({
            "entry_date": position[3],
            "exit_date": last_date,
            "ticker": position[0],
            "shares": position[1],
            "entry_price": position[2],
            "exit_price": exit_price,
            "pnl": pnl,
            "return_pct": pnl / (position[1] * position[2]) * 100,
            "hold_days": (last_date - position[3]).days,
        })

    equity_df = pd.DataFrame({"equity": eq_vals}, index=eq_dates)
    return trades, equity_df


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(trades, equity_df, capital=CAPITAL):
    if len(trades) == 0:
        return {
            "total_return_pct": 0.0, "final_equity": capital,
            "sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0,
            "win_rate": 0.0, "num_trades": 0,
            "max_drawdown_pct": 0.0, "avg_hold_days": 0.0,
        }

    df = pd.DataFrame(trades)
    final_eq = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else capital
    total_ret = (final_eq - capital) / capital * 100

    if len(equity_df) > 1:
        daily_returns = equity_df["equity"].pct_change().dropna()
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        downside = daily_returns[daily_returns < 0].std()
        sortino = (mean_ret / downside * np.sqrt(252)) if downside > 0 else 0.0
        peak = equity_df["equity"].cummax()
        dd = (equity_df["equity"] - peak) / peak * 100
        max_dd = dd.min()
    else:
        sharpe = sortino = max_dd = 0.0

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)

    return {
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(final_eq, 2),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "profit_factor": round(float(pf), 4),
        "win_rate": round(len(wins) / len(df) * 100, 2),
        "num_trades": len(df),
        "max_drawdown_pct": round(float(max_dd), 2),
        "avg_hold_days": round(float(df["hold_days"].mean()), 1),
    }


def compute_regime_metrics(trades, regime_series):
    if len(trades) == 0:
        return {"bull_trades": 0, "bear_trades": 0, "bull_sharpe": 0.0, "bear_sharpe": 0.0, "regime_gap": 0.0}

    df = pd.DataFrame(trades)
    regimes = []
    for d in df["entry_date"]:
        if d in regime_series.index:
            regimes.append(regime_series.loc[d])
        else:
            regimes.append("unknown")
    df["regime"] = regimes

    def _sharpe(tdf):
        if len(tdf) < 2:
            return 0.0
        rets = tdf["return_pct"].values / 100
        m, s = rets.mean(), rets.std()
        return (m / s * np.sqrt(252)) if s > 0 else 0.0

    bull = df[df["regime"] == "bull"]
    bear = df[df["regime"] == "bear"]
    bs = _sharpe(bull)
    brs = _sharpe(bear)
    mx = max(abs(bs), abs(brs), 1e-9)

    return {
        "bull_trades": len(bull), "bear_trades": len(bear),
        "bull_sharpe": round(bs, 4), "bear_sharpe": round(brs, 4),
        "regime_gap": round(abs(bs - brs) / mx, 4),
    }


# ── Permutation Test (optimized) ──────────────────────────────────────────
def permutation_test(trades, signal_df, close, n_iter=PERM_ITERATIONS):
    """
    Shuffle signal dates by random 1-60 day offset.
    Uses vectorized date shifting for speed.
    """
    if len(trades) == 0:
        return {"perm_p_value": 1.0, "actual_mean_pnl": 0.0, "perm_mean_pnl": 0.0}

    actual_mean_pnl = float(np.mean([t["pnl"] for t in trades]))
    valid_dates_set = set(close.index)
    perm_means = np.zeros(n_iter)

    for i in range(n_iter):
        offset = np.random.randint(1, 61)
        shifted = signal_df.copy()
        shifted["date"] = shifted["date"] + pd.Timedelta(days=offset)
        shifted = shifted[shifted["date"].isin(valid_dates_set)]

        if len(shifted) == 0:
            continue

        perm_trades, _ = run_backtest(close, shifted)
        if len(perm_trades) > 0:
            perm_means[i] = np.mean([t["pnl"] for t in perm_trades])

    p_value = float(np.mean(perm_means >= actual_mean_pnl))

    return {
        "perm_p_value": round(p_value, 4),
        "actual_mean_pnl": round(actual_mean_pnl, 4),
        "perm_mean_pnl": round(float(np.mean(perm_means)), 4),
    }


# ── 5-Gate Validation ──────────────────────────────────────────────────────
def validate_gates(metrics, regime, perm):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm["perm_p_value"] < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50.0,
        "num_trades_gte_20": metrics["num_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Signal Generators ─────────────────────────────────────────────────────

def variant_a_signals(close, avg_corr):
    """A) Rolling Correlation Signal with hysteresis."""
    signals = []
    holding = None

    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]

    for date in oot_dates:
        c = avg_corr.get(date, np.nan)
        if pd.isna(c):
            continue

        if holding is None:
            signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
            holding = "growth"
        elif holding == "growth" and c > 0.80:
            signals.append({"date": date, "ticker": "QQQ", "action": "sell"})
            signals.append({"date": date, "ticker": "GLD", "action": "buy"})
            holding = "haven"
        elif holding == "haven" and c < 0.60:
            signals.append({"date": date, "ticker": "GLD", "action": "sell"})
            signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
            holding = "growth"

    return pd.DataFrame(signals)


def variant_b_signals(close, avg_corr):
    """B) Correlation + VIX Confirmation."""
    signals = []
    holding = None
    vix = close.get("VIX", pd.Series(15, index=close.index))

    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]

    for date in oot_dates:
        c = avg_corr.get(date, np.nan)
        if pd.isna(c):
            continue
        v = vix.get(date, 15)

        if holding is None:
            signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
            holding = "growth"
        elif holding == "growth" and c > 0.80 and v > 20:
            signals.append({"date": date, "ticker": "QQQ", "action": "sell"})
            signals.append({"date": date, "ticker": "GLD", "action": "buy"})
            holding = "haven"
        elif holding == "haven" and (c < 0.60 or v < 18):
            signals.append({"date": date, "ticker": "GLD", "action": "sell"})
            signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
            holding = "growth"

    return pd.DataFrame(signals)


def variant_c_signals(close, avg_corr):
    """C) Dispersion Trade."""
    top6 = GROWTH[:6]
    returns = close[top6].pct_change()

    signals = []
    holding = None
    holding_ticker = None

    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]

    for date in oot_dates:
        c = avg_corr.get(date, np.nan)
        if pd.isna(c):
            continue

        if holding is None:
            if c > 0.80:
                idx = close.index.get_loc(date)
                if idx < 20:
                    continue
                window_rets = returns.iloc[idx - 20:idx]
                avg_per_stock = {}
                for t in top6:
                    others = [o for o in top6 if o != t]
                    cs = [window_rets[t].corr(window_rets[o]) for o in others]
                    avg_per_stock[t] = np.nanmean(cs)
                best = min(avg_per_stock, key=avg_per_stock.get)
                signals.append({"date": date, "ticker": best, "action": "buy"})
                holding = "idio"
                holding_ticker = best
            elif c < 0.50:
                signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
                holding = "broad"
                holding_ticker = "QQQ"
        else:
            if holding == "idio" and c < 0.65:
                signals.append({"date": date, "ticker": holding_ticker, "action": "sell"})
                holding = None
                holding_ticker = None
            elif holding == "broad" and c > 0.70:
                signals.append({"date": date, "ticker": "QQQ", "action": "sell"})
                holding = None
                holding_ticker = None

    return pd.DataFrame(signals)


def variant_d_signals(close, avg_corr):
    """D) Correlation Mean Reversion — contrarian."""
    signals = []
    holding = None
    holding_ticker = None

    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]
    oot_dates_list = list(oot_dates)
    oot_set = set(oot_dates)

    # Pre-compute spike and complacency dates
    growth_buy_dates = set()
    haven_buy_dates = set()

    for date in close.index:
        c = avg_corr.get(date, np.nan)
        if pd.isna(c):
            continue
        if c > 0.85:
            future = close.index[close.index > date]
            if len(future) >= 5:
                growth_buy_dates.add(future[4])
        elif c < 0.30:
            future = close.index[close.index > date]
            if len(future) >= 3:
                haven_buy_dates.add(future[2])

    hold_period = 15
    entry_idx = 0

    for i, date in enumerate(oot_dates_list):
        if holding is None:
            if date in growth_buy_dates:
                signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
                holding = "growth"
                holding_ticker = "QQQ"
                entry_idx = i
            elif date in haven_buy_dates:
                signals.append({"date": date, "ticker": "GLD", "action": "buy"})
                holding = "haven"
                holding_ticker = "GLD"
                entry_idx = i
        else:
            if i - entry_idx >= hold_period:
                signals.append({"date": date, "ticker": holding_ticker, "action": "sell"})
                holding = None
                holding_ticker = None

    return pd.DataFrame(signals)


def variant_e_signals(close, avg_corr):
    """E) Dynamic Beta Targeting."""
    spy_returns = close["SPY"].pct_change()
    qqq_returns = close["QQQ"].pct_change()
    gld_returns = close["GLD"].pct_change()

    # Pre-compute betas
    qqq_beta = compute_rolling_beta(qqq_returns, spy_returns, 20)
    gld_beta = compute_rolling_beta(gld_returns, spy_returns, 20)

    signals = []
    holding = None

    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]
    rebalance_dates = oot_dates[::5]

    for date in rebalance_dates:
        idx = close.index.get_loc(date)
        if idx < 25:
            continue

        if holding == "haven":
            current_beta = gld_beta.iloc[idx] if idx < len(gld_beta) else 0.0
        else:
            current_beta = qqq_beta.iloc[idx] if idx < len(qqq_beta) else 1.0

        if pd.isna(current_beta):
            continue

        if holding is None:
            signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
            holding = "growth"
        elif holding == "growth" and current_beta > 1.2:
            signals.append({"date": date, "ticker": "QQQ", "action": "sell"})
            signals.append({"date": date, "ticker": "GLD", "action": "buy"})
            holding = "haven"
        elif holding == "haven" and current_beta < 0.6:
            signals.append({"date": date, "ticker": "GLD", "action": "sell"})
            signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
            holding = "growth"

    return pd.DataFrame(signals)


def variant_f_signals(close, avg_corr):
    """F) Correlation Regime + RSI Combo."""
    signals = []
    holding = None
    holding_ticker = None
    target_class = "growth"

    # Pre-compute RSI for all growth stocks
    rsi_dict = {}
    for t in GROWTH:
        rsi_dict[t] = compute_rsi(close[t], 5)

    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]

    for date in oot_dates:
        c = avg_corr.get(date, np.nan)
        if pd.isna(c):
            continue

        if c > 0.75:
            new_target = "haven"
        elif c < 0.55:
            new_target = "growth"
        else:
            new_target = target_class

        if new_target != target_class:
            if holding is not None:
                signals.append({"date": date, "ticker": holding_ticker, "action": "sell"})
                holding = None
                holding_ticker = None
            target_class = new_target

        if holding is None:
            if target_class == "haven":
                signals.append({"date": date, "ticker": "GLD", "action": "buy"})
                holding = "haven"
                holding_ticker = "GLD"
            else:
                best_ticker = None
                best_rsi = 100
                for t in GROWTH:
                    r = rsi_dict[t].get(date, 50)
                    if not pd.isna(r) and r < 30 and r < best_rsi:
                        best_rsi = r
                        best_ticker = t

                if best_ticker is not None:
                    signals.append({"date": date, "ticker": best_ticker, "action": "buy"})
                    holding = "growth"
                    holding_ticker = best_ticker
                else:
                    signals.append({"date": date, "ticker": "QQQ", "action": "buy"})
                    holding = "growth"
                    holding_ticker = "QQQ"

        # Exit growth on RSI > 70
        if holding == "growth" and holding_ticker in GROWTH:
            r = rsi_dict[holding_ticker].get(date, 50)
            if not pd.isna(r) and r > 70:
                signals.append({"date": date, "ticker": holding_ticker, "action": "sell"})
                holding = None
                holding_ticker = None

    return pd.DataFrame(signals)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    close = download_data()

    top6 = GROWTH[:6]
    log("Computing rolling pairwise correlation for top-6 growth stocks...")
    avg_corr = compute_rolling_pairwise_correlation(close, top6, window=20)

    regime = classify_regime(close)

    variant_configs = {
        "A_rolling_corr": {
            "fn": variant_a_signals,
            "description": "Rolling 20d pairwise correlation among top-6 growth. >0.80 -> GLD, <0.60 -> QQQ. Hysteresis prevents whipsawing.",
        },
        "B_corr_vix_confirm": {
            "fn": variant_b_signals,
            "description": "Same as A but require VIX>20 confirmation before safe haven rotation. Filters benign high-correlation periods.",
        },
        "C_dispersion_trade": {
            "fn": variant_c_signals,
            "description": "High corr (>0.80): buy most idiosyncratic growth stock. Low corr (<0.50): buy QQQ broad basket.",
        },
        "D_corr_mean_revert": {
            "fn": variant_d_signals,
            "description": "Contrarian: buy growth 5 days after corr spike >0.85 (snap-back). Rotate to GLD when corr <0.30 (complacency).",
        },
        "E_dynamic_beta": {
            "fn": variant_e_signals,
            "description": "Target portfolio beta 0.8-1.0 to SPY. Beta>1.2 -> trim to GLD. Beta<0.6 -> add QQQ. Weekly rebalance.",
        },
        "F_corr_rsi_combo": {
            "fn": variant_f_signals,
            "description": "Correlation regime for asset class (high->haven, low->growth). Within growth, use RSI(5)<30 for entry timing.",
        },
    }

    results = {
        "strategy": "Correlation Regime Allocation",
        "run_date": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "capital": CAPITAL,
        "variants": {},
    }

    for name, cfg in variant_configs.items():
        log(f"\n{'='*60}")
        log(f"Running Variant: {name}")
        log(f"{'='*60}")

        signal_df = cfg["fn"](close, avg_corr)
        if len(signal_df) == 0:
            log(f"  WARNING: No signals generated for {name}")
            results["variants"][name] = {
                "metrics": compute_metrics([], pd.DataFrame()),
                "regime": {"bull_trades": 0, "bear_trades": 0, "bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0},
                "permutation": {"perm_p_value": 1.0, "actual_mean_pnl": 0.0, "perm_mean_pnl": 0.0},
                "gates": {"sharpe_gt_0.5": False, "perm_p_lt_0.05": False, "regime_gap_lt_0.5": True,
                          "max_dd_gt_neg50": True, "num_trades_gte_20": False, "all_passed": False},
                "description": cfg["description"],
            }
            continue

        trades, equity_df = run_backtest(close, signal_df)
        log(f"  Trades: {len(trades)}")

        metrics = compute_metrics(trades, equity_df)
        log(f"  Return: {metrics['total_return_pct']:.2f}%, Sharpe: {metrics['sharpe']:.4f}, "
            f"Sortino: {metrics['sortino']:.4f}, PF: {metrics['profit_factor']:.2f}, "
            f"WR: {metrics['win_rate']:.1f}%, MaxDD: {metrics['max_drawdown_pct']:.2f}%")

        regime_metrics = compute_regime_metrics(trades, regime)
        log(f"  Bull: {regime_metrics['bull_trades']}, Bear: {regime_metrics['bear_trades']}, "
            f"Gap: {regime_metrics['regime_gap']:.4f}")

        log(f"  Permutation test ({PERM_ITERATIONS} iters)...")
        perm = permutation_test(trades, signal_df, close)
        log(f"  Perm p={perm['perm_p_value']:.4f}")

        gates = validate_gates(metrics, regime_metrics, perm)
        passed = sum(1 for k, v in gates.items() if k != "all_passed" and v)
        log(f"  Gates: {passed}/5 {'*** ALL PASSED ***' if gates['all_passed'] else ''}")

        results["variants"][name] = {
            "metrics": metrics,
            "regime": regime_metrics,
            "permutation": perm,
            "gates": gates,
            "description": cfg["description"],
        }

    # ── Summary ────────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    log(f"{'Variant':<25} {'Sharpe':>8} {'Return%':>9} {'PF':>6} {'WR%':>6} {'Trades':>7} {'MaxDD%':>8} {'Perm-p':>8} {'Gates':>6}")
    log("-" * 90)
    for name, v in results["variants"].items():
        m = v["metrics"]
        p = v["permutation"]
        g = v["gates"]
        passed = sum(1 for k, val in g.items() if k != "all_passed" and val)
        star = " ***" if g["all_passed"] else ""
        log(f"{name:<25} {m['sharpe']:>8.4f} {m['total_return_pct']:>8.2f}% {m['profit_factor']:>6.2f} "
            f"{m['win_rate']:>5.1f}% {m['num_trades']:>7d} {m['max_drawdown_pct']:>7.2f}% "
            f"{p['perm_p_value']:>8.4f} {passed}/5{star}")

    output_path = Path("/home/jupiter/Lvl3Quant/data/correlation_regime_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
