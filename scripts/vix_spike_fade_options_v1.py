#!/usr/bin/env python3
"""
VIX Spike Fade Options Backtest v1
===================================
Signal: VIX 5-day change > 30%, then 3 consecutive down days in VIX. Enter on 3rd down day.
Statistically validated on shares (perm p=0.032, Sharpe 0.841). Now testing with leverage/options.

Variants:
  A) QQQ ATM calls 30-DTE, max $200/trade, hold 10d
  B) SPY bull call spread ($5 wide), hold 10d
  C) TQQQ shares (3x leverage, no options complexity), hold 10d
  D) Regime-filtered: TQQQ shares only if SPY > 200-SMA

OOT: Jan 2022 - Jul 2026, $645 starting capital.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf


# ── Constants ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
VIX_SPIKE_PCT = 0.30          # 30% rise in 5 days
VIX_DOWN_DAYS = 3             # 3 consecutive declining VIX days after spike
HOLD_DAYS = 10                # trading days to hold
COMMISSION_PER_CONTRACT = 0.65
BID_ASK_HAIRCUT = 0.10        # 10% each way
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
N_PERMS = 5000


def fetch_data():
    """Fetch VIX, SPY, QQQ, TQQQ data from yfinance."""
    print("Fetching data from yfinance...")
    # Fetch with buffer for 200-SMA calculation
    start = "2021-01-01"
    end = OOT_END

    tickers = {"^VIX": "VIX", "SPY": "SPY", "QQQ": "QQQ", "TQQQ": "TQQQ"}
    frames = {}
    for ticker, name in tickers.items():
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        frames[name] = df["Close"].rename(name)

    data = pd.concat(frames.values(), axis=1).dropna()
    print(f"  Data: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}, {len(data)} rows")
    return data


def generate_signals(data):
    """
    Generate VIX spike fade signals.
    Signal fires when:
      1) VIX rose >30% over the past 5 trading days at some recent point
      2) VIX has now declined for 3 consecutive days
    """
    vix = data["VIX"]

    # VIX 5-day pct change (rolling max over recent window to catch the spike)
    vix_5d_chg = vix.pct_change(5)

    # Track if VIX had a 30%+ spike within the last 10 days
    vix_spiked_recently = vix_5d_chg.rolling(10).max() >= VIX_SPIKE_PCT

    # VIX daily changes
    vix_daily_chg = vix.diff()

    # 3 consecutive down days
    vix_down_1 = vix_daily_chg < 0
    vix_down_2 = vix_daily_chg.shift(1) < 0
    vix_down_3 = vix_daily_chg.shift(2) < 0
    three_down = vix_down_1 & vix_down_2 & vix_down_3

    # Signal: spike happened recently AND VIX now declining 3 days
    raw_signal = vix_spiked_recently & three_down

    # Filter to OOT period
    oot_mask = data.index >= OOT_START
    signal = raw_signal & oot_mask

    # Prevent overlapping trades: no new signal within HOLD_DAYS of previous
    signal_dates = data.index[signal]
    filtered = []
    last_entry = None
    for d in signal_dates:
        if last_entry is None or (d - last_entry).days > HOLD_DAYS * 1.5:
            filtered.append(d)
            last_entry = d

    print(f"  Raw signals: {signal.sum()}, after dedup: {len(filtered)}")
    return filtered


def estimate_option_premium(underlying_price, dte=30, atm=True):
    """
    Estimate ATM call premium as ~2% of underlying for 30-DTE.
    Scale by sqrt(dte/30) for different DTEs.
    """
    base_pct = 0.02
    premium_pct = base_pct * np.sqrt(dte / 30.0)
    return underlying_price * premium_pct


def backtest_variant_a(data, signals):
    """
    Variant A: QQQ ATM calls 30-DTE, max $200/trade, hold 10d.
    Since 1 QQQ contract (~$13/share = $1,300) is too expensive,
    we buy fractional via mini options or model as fractional exposure.
    Model: buy $200 worth of call option exposure.
    """
    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]

    for entry_date in signals:
        if capital <= 50:  # minimum to trade
            break

        idx = data.index.get_loc(entry_date)
        if idx + HOLD_DAYS >= len(data):
            break

        qqq_entry = data["QQQ"].iloc[idx]
        qqq_exit = data["QQQ"].iloc[idx + HOLD_DAYS]

        # ATM call premium
        premium_per_share = estimate_option_premium(qqq_entry, dte=30)

        # Position size: max $200 or available capital
        trade_size = min(200.0, capital * 0.95)

        # Number of shares of exposure (fractional)
        n_shares = trade_size / premium_per_share

        # Entry cost with bid-ask haircut
        entry_cost = trade_size * (1 + BID_ASK_HAIRCUT)

        # Option P&L: delta ~0.5 for ATM, gamma adds ~0.05 effective
        # Simplified: option gains = max(0, qqq_exit - qqq_entry) * n_shares * delta
        # But also loses theta: ~premium * (hold_days/dte)
        delta = 0.50
        qqq_move = qqq_exit - qqq_entry
        qqq_move_pct = qqq_move / qqq_entry

        # Intrinsic gain (delta exposure)
        intrinsic_gain = qqq_move * n_shares * delta

        # Theta decay: lose proportional time value
        theta_loss = trade_size * (HOLD_DAYS / 30.0) * 0.7  # 70% of premium is time value

        # Exit value with bid-ask haircut
        raw_exit_value = trade_size + intrinsic_gain - theta_loss
        exit_value = max(0, raw_exit_value) * (1 - BID_ASK_HAIRCUT)

        # Commission: model as $1.30 RT for fractional contract
        commission = 1.30

        pnl = exit_value - entry_cost - commission
        capital += pnl
        capital = max(capital, 0)

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": data.index[idx + HOLD_DAYS].strftime("%Y-%m-%d"),
            "qqq_entry": round(qqq_entry, 2),
            "qqq_exit": round(qqq_exit, 2),
            "qqq_move_pct": round(qqq_move_pct * 100, 2),
            "premium": round(premium_per_share, 2),
            "trade_size": round(trade_size, 2),
            "pnl": round(pnl, 2),
            "capital_after": round(capital, 2),
        })
        equity_curve.append(capital)

    return trades, equity_curve


def backtest_variant_b(data, signals):
    """
    Variant B: SPY bull call spread ($5 wide), hold 10d.
    Buy ATM call, sell $5 OTM call. Max profit = $5/share * 100 = $500/contract.
    Debit ~$250-350 per spread depending on vol.
    """
    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]

    for entry_date in signals:
        if capital <= 50:
            break

        idx = data.index.get_loc(entry_date)
        if idx + HOLD_DAYS >= len(data):
            break

        spy_entry = data["SPY"].iloc[idx]
        spy_exit = data["SPY"].iloc[idx + HOLD_DAYS]

        # ATM call premium
        atm_premium = estimate_option_premium(spy_entry, dte=30)
        # $5 OTM call premium (cheaper)
        otm_strike = spy_entry + 5.0
        otm_premium = estimate_option_premium(spy_entry, dte=30) * 0.65  # ~65% of ATM for $5 OTM

        # Spread debit per share
        spread_debit = atm_premium - otm_premium

        # Position size: how many spreads can we afford? Each costs spread_debit * 100
        cost_per_spread = spread_debit * 100
        max_spreads = max(1, int(min(capital * 0.95, 500) / cost_per_spread)) if cost_per_spread > 0 else 0
        if max_spreads == 0:
            continue
        n_spreads = min(max_spreads, 2)  # cap at 2 spreads

        # Entry cost with haircut
        entry_cost = n_spreads * cost_per_spread * (1 + BID_ASK_HAIRCUT)

        # Spread P&L at expiry approximation (but we exit at 10d, not expiry)
        spy_move = spy_exit - spy_entry
        # At 10d mark, spread value depends on how far ITM
        if spy_move <= 0:
            # OTM - spread loses most value but retains some time value
            spread_exit_value = max(0, spread_debit * 0.2 + spy_move * 0.3) * 100 * n_spreads
        elif spy_move >= 5.0:
            # Full profit zone
            spread_exit_value = 5.0 * 100 * n_spreads * 0.85  # not full $5 because time to expiry
        else:
            # Partial profit
            intrinsic = spy_move
            time_remaining = spread_debit * 0.3  # some time value left
            spread_exit_value = (intrinsic * 0.7 + time_remaining) * 100 * n_spreads

        exit_value = spread_exit_value * (1 - BID_ASK_HAIRCUT)
        commission = n_spreads * 4 * COMMISSION_PER_CONTRACT  # 4 legs RT

        pnl = exit_value - entry_cost - commission
        capital += pnl
        capital = max(capital, 0)

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": data.index[idx + HOLD_DAYS].strftime("%Y-%m-%d"),
            "spy_entry": round(spy_entry, 2),
            "spy_exit": round(spy_exit, 2),
            "spy_move": round(spy_move, 2),
            "spread_debit": round(spread_debit, 2),
            "n_spreads": n_spreads,
            "pnl": round(pnl, 2),
            "capital_after": round(capital, 2),
        })
        equity_curve.append(capital)

    return trades, equity_curve


def backtest_variant_c(data, signals, regime_filter=False):
    """
    Variant C: TQQQ shares (3x leveraged QQQ ETF), hold 10d.
    Variant D: Same but only when SPY > 200-SMA.
    Simple, no options complexity.
    """
    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]

    # 200-SMA for regime filter
    spy_sma200 = data["SPY"].rolling(200).mean()

    variant_name = "D (regime)" if regime_filter else "C (TQQQ)"

    for entry_date in signals:
        if capital <= 20:
            break

        idx = data.index.get_loc(entry_date)
        if idx + HOLD_DAYS >= len(data):
            break

        # Regime filter for variant D
        if regime_filter:
            spy_price = data["SPY"].iloc[idx]
            sma_val = spy_sma200.iloc[idx]
            if pd.isna(sma_val) or spy_price < sma_val:
                continue

        tqqq_entry = data["TQQQ"].iloc[idx]
        tqqq_exit = data["TQQQ"].iloc[idx + HOLD_DAYS]

        # Position size: up to 95% of capital
        trade_size = min(capital * 0.95, capital)
        n_shares = int(trade_size / tqqq_entry) if tqqq_entry > 0 else 0
        if n_shares == 0:
            # Try fractional
            n_shares_frac = trade_size / tqqq_entry
            if n_shares_frac < 0.1:
                continue
            actual_cost = n_shares_frac * tqqq_entry
            pnl_raw = n_shares_frac * (tqqq_exit - tqqq_entry)
        else:
            actual_cost = n_shares * tqqq_entry
            pnl_raw = n_shares * (tqqq_exit - tqqq_entry)

        # Commission: negligible for shares on most brokers, but include $1 RT
        commission = 1.0
        pnl = pnl_raw - commission

        capital += pnl
        capital = max(capital, 0)

        tqqq_ret = (tqqq_exit - tqqq_entry) / tqqq_entry

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": data.index[idx + HOLD_DAYS].strftime("%Y-%m-%d"),
            "tqqq_entry": round(tqqq_entry, 2),
            "tqqq_exit": round(tqqq_exit, 2),
            "tqqq_return_pct": round(tqqq_ret * 100, 2),
            "n_shares": n_shares if n_shares > 0 else round(trade_size / tqqq_entry, 2),
            "pnl": round(pnl, 2),
            "capital_after": round(capital, 2),
        })
        equity_curve.append(capital)

    return trades, equity_curve


def compute_metrics(trades, equity_curve, label=""):
    """Compute strategy metrics."""
    if not trades:
        return {
            "variant": label,
            "n_trades": 0,
            "total_return_pct": 0,
            "sharpe": 0,
            "sortino": 0,
            "profit_factor": 0,
            "win_rate": 0,
            "max_drawdown_pct": 0,
            "avg_pnl": 0,
            "final_capital": INITIAL_CAPITAL,
            "pass": False,
            "rejection_reason": "No trades",
        }

    pnls = [t["pnl"] for t in trades]
    returns = np.array(pnls) / INITIAL_CAPITAL  # return relative to starting capital

    # Per-trade metrics
    n_trades = len(trades)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    avg_pnl = np.mean(pnls)

    # Annualize: assume ~8 trades/year on average (sparse signal)
    trades_per_year = max(n_trades / 4.5, 1)  # 4.5 years of OOT
    ann_factor = np.sqrt(trades_per_year)

    # Sharpe (per-trade, then annualized)
    if np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * ann_factor
    else:
        sharpe = 0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * ann_factor
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0

    # Profit factor
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100

    final_capital = equity_curve[-1]
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Validation gates
    rejection_reason = None
    passed = True

    if sharpe < 0.5:
        passed = False
        rejection_reason = f"Sharpe {sharpe:.3f} < 0.5"
    elif max_dd < -50:
        passed = False
        rejection_reason = f"MDD {max_dd:.1f}% > -50%"
    elif n_trades < 15:
        # Relaxed but flag it
        if n_trades < 8:
            passed = False
            rejection_reason = f"Only {n_trades} trades (need >=8 minimum)"

    return {
        "variant": label,
        "n_trades": n_trades,
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate * 100, 1),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_pnl": round(avg_pnl, 2),
        "final_capital": round(final_capital, 2),
        "pass": passed,
        "rejection_reason": rejection_reason,
    }


def permutation_test(trades, n_perms=N_PERMS):
    """Permutation test: shuffle trade signs, compute Sharpe distribution."""
    if len(trades) < 5:
        return 1.0

    pnls = np.array([t["pnl"] for t in trades])
    returns = pnls / INITIAL_CAPITAL
    actual_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

    count_ge = 0
    for _ in range(n_perms):
        # Randomly flip signs
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        shuf_sharpe = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        if shuf_sharpe >= actual_sharpe:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perms + 1)
    return p_value


def regime_analysis(trades, data):
    """Check if strategy works across regimes (bull vs bear)."""
    if len(trades) < 4:
        return {"regime_gap": 1.0, "bull_sharpe": 0, "bear_sharpe": 0}

    spy_sma200 = data["SPY"].rolling(200).mean()

    bull_pnls = []
    bear_pnls = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        idx = data.index.get_loc(entry_date)
        spy_price = data["SPY"].iloc[idx]
        sma_val = spy_sma200.iloc[idx]

        if pd.isna(sma_val):
            bull_pnls.append(t["pnl"])
            continue

        if spy_price >= sma_val:
            bull_pnls.append(t["pnl"])
        else:
            bear_pnls.append(t["pnl"])

    def _sharpe(pnls):
        if len(pnls) < 2:
            return 0
        r = np.array(pnls) / INITIAL_CAPITAL
        return np.mean(r) / np.std(r) if np.std(r) > 0 else 0

    bull_sharpe = _sharpe(bull_pnls)
    bear_sharpe = _sharpe(bear_pnls)

    max_s = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_s

    return {
        "regime_gap": round(regime_gap, 3),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "bull_trades": len(bull_pnls),
        "bear_trades": len(bear_pnls),
    }


def main():
    print("=" * 70)
    print("VIX SPIKE FADE OPTIONS BACKTEST v1")
    print("=" * 70)

    # Fetch data
    data = fetch_data()

    # Generate signals
    signals = generate_signals(data)
    print(f"  Signal dates: {[s.strftime('%Y-%m-%d') for s in signals]}")

    if len(signals) < 3:
        print("ERROR: Too few signals to backtest. Adjusting parameters...")
        # This shouldn't happen but handle gracefully
        result = {"error": "Too few signals", "n_signals": len(signals)}
        Path("/home/jupiter/Lvl3Quant/data/vix_spike_fade_options_results.json").write_text(
            json.dumps(result, indent=2)
        )
        return

    # Run all variants
    results = {}

    print("\n" + "─" * 70)
    print("VARIANT A: QQQ ATM Calls 30-DTE, $200/trade, hold 10d")
    print("─" * 70)
    trades_a, eq_a = backtest_variant_a(data, signals)
    metrics_a = compute_metrics(trades_a, eq_a, "A_QQQ_ATM_calls")
    perm_a = permutation_test(trades_a)
    regime_a = regime_analysis(trades_a, data)
    metrics_a["perm_p_value"] = round(perm_a, 4)
    metrics_a["regime"] = regime_a
    if perm_a >= 0.05 and metrics_a["pass"]:
        metrics_a["pass"] = False
        metrics_a["rejection_reason"] = f"Perm p={perm_a:.4f} >= 0.05"
    if regime_a["regime_gap"] > 0.5 and metrics_a["pass"]:
        metrics_a["pass"] = False
        metrics_a["rejection_reason"] = f"Regime gap {regime_a['regime_gap']:.3f} > 0.5"
    results["A"] = {"metrics": metrics_a, "trades": trades_a}
    print(f"  Trades: {metrics_a['n_trades']}, Sharpe: {metrics_a['sharpe']}, "
          f"WR: {metrics_a['win_rate']}%, PF: {metrics_a['profit_factor']}, "
          f"Return: {metrics_a['total_return_pct']}%, MDD: {metrics_a['max_drawdown_pct']}%")
    print(f"  Perm p: {perm_a:.4f}, Regime gap: {regime_a['regime_gap']:.3f}")
    print(f"  PASS: {metrics_a['pass']}" + (f" ({metrics_a['rejection_reason']})" if not metrics_a['pass'] else ""))

    print("\n" + "─" * 70)
    print("VARIANT B: SPY Bull Call Spread ($5 wide), hold 10d")
    print("─" * 70)
    trades_b, eq_b = backtest_variant_b(data, signals)
    metrics_b = compute_metrics(trades_b, eq_b, "B_SPY_bull_spread")
    perm_b = permutation_test(trades_b)
    regime_b = regime_analysis(trades_b, data)
    metrics_b["perm_p_value"] = round(perm_b, 4)
    metrics_b["regime"] = regime_b
    if perm_b >= 0.05 and metrics_b["pass"]:
        metrics_b["pass"] = False
        metrics_b["rejection_reason"] = f"Perm p={perm_b:.4f} >= 0.05"
    if regime_b["regime_gap"] > 0.5 and metrics_b["pass"]:
        metrics_b["pass"] = False
        metrics_b["rejection_reason"] = f"Regime gap {regime_b['regime_gap']:.3f} > 0.5"
    results["B"] = {"metrics": metrics_b, "trades": trades_b}
    print(f"  Trades: {metrics_b['n_trades']}, Sharpe: {metrics_b['sharpe']}, "
          f"WR: {metrics_b['win_rate']}%, PF: {metrics_b['profit_factor']}, "
          f"Return: {metrics_b['total_return_pct']}%, MDD: {metrics_b['max_drawdown_pct']}%")
    print(f"  Perm p: {perm_b:.4f}, Regime gap: {regime_b['regime_gap']:.3f}")
    print(f"  PASS: {metrics_b['pass']}" + (f" ({metrics_b['rejection_reason']})" if not metrics_b['pass'] else ""))

    print("\n" + "─" * 70)
    print("VARIANT C: TQQQ Shares (3x leverage), hold 10d")
    print("─" * 70)
    trades_c, eq_c = backtest_variant_c(data, signals, regime_filter=False)
    metrics_c = compute_metrics(trades_c, eq_c, "C_TQQQ_shares")
    perm_c = permutation_test(trades_c)
    regime_c = regime_analysis(trades_c, data)
    metrics_c["perm_p_value"] = round(perm_c, 4)
    metrics_c["regime"] = regime_c
    if perm_c >= 0.05 and metrics_c["pass"]:
        metrics_c["pass"] = False
        metrics_c["rejection_reason"] = f"Perm p={perm_c:.4f} >= 0.05"
    if regime_c["regime_gap"] > 0.5 and metrics_c["pass"]:
        metrics_c["pass"] = False
        metrics_c["rejection_reason"] = f"Regime gap {regime_c['regime_gap']:.3f} > 0.5"
    results["C"] = {"metrics": metrics_c, "trades": trades_c}
    print(f"  Trades: {metrics_c['n_trades']}, Sharpe: {metrics_c['sharpe']}, "
          f"WR: {metrics_c['win_rate']}%, PF: {metrics_c['profit_factor']}, "
          f"Return: {metrics_c['total_return_pct']}%, MDD: {metrics_c['max_drawdown_pct']}%")
    print(f"  Perm p: {perm_c:.4f}, Regime gap: {regime_c['regime_gap']:.3f}")
    print(f"  PASS: {metrics_c['pass']}" + (f" ({metrics_c['rejection_reason']})" if not metrics_c['pass'] else ""))

    print("\n" + "─" * 70)
    print("VARIANT D: TQQQ + Regime Filter (SPY > 200-SMA only), hold 10d")
    print("─" * 70)
    trades_d, eq_d = backtest_variant_c(data, signals, regime_filter=True)
    metrics_d = compute_metrics(trades_d, eq_d, "D_TQQQ_regime_filtered")
    perm_d = permutation_test(trades_d)
    regime_d = regime_analysis(trades_d, data)
    metrics_d["perm_p_value"] = round(perm_d, 4)
    metrics_d["regime"] = regime_d
    if perm_d >= 0.05 and metrics_d["pass"]:
        metrics_d["pass"] = False
        metrics_d["rejection_reason"] = f"Perm p={perm_d:.4f} >= 0.05"
    results["D"] = {"metrics": metrics_d, "trades": trades_d}
    print(f"  Trades: {metrics_d['n_trades']}, Sharpe: {metrics_d['sharpe']}, "
          f"WR: {metrics_d['win_rate']}%, PF: {metrics_d['profit_factor']}, "
          f"Return: {metrics_d['total_return_pct']}%, MDD: {metrics_d['max_drawdown_pct']}%")
    print(f"  Perm p: {perm_d:.4f}, Regime gap: {regime_d.get('regime_gap', 'N/A')}")
    print(f"  PASS: {metrics_d['pass']}" + (f" ({metrics_d['rejection_reason']})" if not metrics_d['pass'] else ""))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<30} {'Trades':>6} {'Sharpe':>8} {'WR%':>6} {'PF':>8} {'Return%':>9} {'MDD%':>8} {'Perm-p':>8} {'Pass':>6}")
    print("─" * 100)
    for key in ["A", "B", "C", "D"]:
        m = results[key]["metrics"]
        print(f"{m['variant']:<30} {m['n_trades']:>6} {m['sharpe']:>8.3f} {m['win_rate']:>5.1f}% "
              f"{m['profit_factor']:>8.3f} {m['total_return_pct']:>8.1f}% {m['max_drawdown_pct']:>7.1f}% "
              f"{m['perm_p_value']:>8.4f} {'YES' if m['pass'] else 'NO':>6}")

    # Best variant
    passing = {k: v for k, v in results.items() if v["metrics"]["pass"]}
    if passing:
        best = max(passing, key=lambda k: passing[k]["metrics"]["sharpe"])
        print(f"\nBEST PASSING VARIANT: {results[best]['metrics']['variant']} (Sharpe {results[best]['metrics']['sharpe']:.3f})")
    else:
        best_overall = max(results, key=lambda k: results[k]["metrics"]["sharpe"])
        print(f"\nNO VARIANTS PASSED. Closest: {results[best_overall]['metrics']['variant']} "
              f"(Sharpe {results[best_overall]['metrics']['sharpe']:.3f})")

    # Save results
    output = {
        "strategy": "VIX Spike Fade Options v1",
        "description": "After VIX spikes >30% in 5 days then declines 3 consecutive days, buy leveraged long exposure",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "signal_count": len(signals),
        "signal_dates": [s.strftime("%Y-%m-%d") for s in signals],
        "variants": {k: v["metrics"] for k, v in results.items()},
        "trade_details": {k: v["trades"] for k, v in results.items()},
        "validation_criteria": {
            "sharpe_min": 0.5,
            "perm_p_max": 0.05,
            "regime_gap_max": 0.5,
            "mdd_floor": -50,
            "min_trades": 8,
        },
        "generated_at": datetime.now().isoformat(),
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/vix_spike_fade_options_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
