#!/usr/bin/env python3
"""
Composite Signal Optimizer — Combines validated signals into a single scoring system.

Signals combined:
  1. Earnings (PEAD + surprise momentum)
  2. Regime (VIX + SPY trend)
  3. Trend (SMA stack)
  4. Momentum (20d/60d returns)
  5. Relative Strength (vs SPY)

6 variants tested with 5-gate validation + permutation testing.
"""

import json
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN",
    "RBLX", "UBER", "LYFT", "DDOG", "TTD", "SHOP", "NET", "ROKU",
]

START = "2021-06-01"   # need lookback before OOT start
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"

HOLD_DAYS = 40
SLIPPAGE_PCT = 0.0002       # 0.02%
OPTION_COST_PER_CONTRACT = 0.65
OPTION_BIDASK_HAIRCUT = 0.05  # 5%
OPTION_PREMIUM_PCT = 0.03    # ATM call ≈ 3% of stock price
OPTION_STOP = -0.50           # -50% stop on option premium

N_PERMUTATIONS = 1000
INITIAL_CAPITAL = 645.0

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/composite_signal_results.json")


# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    """Download price data for universe + SPY + ^VIX."""
    tickers = UNIVERSE + ["SPY", "^VIX"]
    print(f"Downloading {len(tickers)} tickers...")
    all_data = {}

    # Download in batches to avoid rate limits
    for i in range(0, len(tickers), 8):
        batch = tickers[i:i+8]
        batch_str = " ".join(batch)
        try:
            df = yf.download(batch_str, start=START, end=OOT_END, progress=False, auto_adjust=True)
            if len(batch) == 1:
                # Single ticker returns flat columns
                t = batch[0]
                if not df.empty:
                    all_data[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                for t in batch:
                    try:
                        sub = df.xs(t, level=1, axis=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
                        if not sub.dropna(subset=["Close"]).empty:
                            all_data[t] = sub
                    except (KeyError, TypeError):
                        pass
        except Exception as e:
            print(f"  Batch download failed: {e}")
        time.sleep(0.5)

    print(f"  Got data for {len(all_data)} tickers")
    return all_data


# ── SIGNAL COMPUTATION ──────────────────────────────────────────────────────
def compute_signals(all_data):
    """Compute all 5 signal components for every stock on every day."""
    spy = all_data.get("SPY")
    vix = all_data.get("^VIX")
    if spy is None or vix is None:
        raise ValueError("Missing SPY or VIX data")

    # Precompute SPY indicators
    spy_close = spy["Close"].dropna()
    spy_sma50 = spy_close.rolling(50).mean()
    spy_sma200 = spy_close.rolling(200).mean()
    spy_ret20 = spy_close.pct_change(20)

    # VIX indicators
    vix_close = vix["Close"].dropna()
    vix_sma5 = vix_close.rolling(5).mean()
    vix_ret5 = vix_close.pct_change(5)

    signals = {}  # {ticker: DataFrame with date index, columns = signal components + total}

    for ticker in UNIVERSE:
        if ticker not in all_data:
            continue

        df = all_data[ticker].copy()
        close = df["Close"].dropna()
        if len(close) < 200:
            continue

        # Precompute per-stock indicators
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        ret20 = close.pct_change(20)
        ret60 = close.pct_change(60)

        # Daily gap (open vs prev close) for earnings proxy
        prev_close = close.shift(1)
        daily_open = df["Open"]
        gap_pct = (daily_open - prev_close) / prev_close

        # Detect "earnings" days: >3% absolute gap
        earnings_days = gap_pct.abs() > 0.03
        earnings_beat = gap_pct > 0.03  # positive gap = beat

        # Build signal dataframe
        idx = close.index
        sig = pd.DataFrame(index=idx)

        # ── 1. EARNINGS SIGNAL (0-20) ──
        earnings_score = pd.Series(0.0, index=idx)

        # Check if stock just had earnings beat in last 5 days
        recent_beat = earnings_beat.rolling(5, min_periods=1).max().fillna(0)
        earnings_score += recent_beat * 15  # +15 for recent beat

        # Check for large gap (>5%) = strong beat → bump to 20
        large_gap = (gap_pct > 0.05).rolling(5, min_periods=1).max().fillna(0)
        earnings_score += large_gap * 5  # additional +5 for strong beat

        # Consecutive beats: 2+ beats in last 120 days
        beat_count_120d = earnings_beat.rolling(120, min_periods=1).sum().fillna(0)
        consecutive_bonus = (beat_count_120d >= 2).astype(float) * 5
        earnings_score = np.minimum(earnings_score + consecutive_bonus, 20)

        # Earnings season window bonus (Jan, Apr, Jul, Oct)
        month = pd.Series(idx.month, index=idx)
        in_season = month.isin([1, 2, 4, 5, 7, 8, 10, 11]).astype(float) * 2
        earnings_score = np.minimum(earnings_score + in_season, 20)

        sig["earnings"] = earnings_score

        # ── 2. REGIME SIGNAL (0-20) ──
        regime_score = pd.Series(0.0, index=idx)

        # Align VIX and SPY to stock's index
        vix_aligned = vix_close.reindex(idx).ffill()
        spy_sma50_aligned = spy_sma50.reindex(idx).ffill()
        spy_close_aligned = spy_close.reindex(idx).ffill()
        vix_ret5_aligned = vix_ret5.reindex(idx).ffill()

        # Risk-on: VIX < 15 AND SPY > 50-SMA
        full_risk_on = (vix_aligned < 15) & (spy_close_aligned > spy_sma50_aligned)
        regime_score += full_risk_on.astype(float) * 20

        # Moderate: VIX 15-20
        moderate = (vix_aligned >= 15) & (vix_aligned <= 20) & (spy_close_aligned > spy_sma50_aligned)
        regime_score += moderate.astype(float) * 10

        # VIX fade bonus: VIX spiked >30% in 5 days but now declining
        vix_spiked = vix_ret5_aligned > 0.30
        vix_declining = vix_aligned < vix_aligned.shift(1)
        fade_bonus = (vix_spiked.shift(5).fillna(False) & vix_declining).astype(float) * 5
        regime_score = np.minimum(regime_score + fade_bonus, 20)

        # Kill switch: VIX > 20 → score stays 0
        sig["regime"] = regime_score

        # ── 3. TREND SIGNAL (0-20) ──
        trend_score = pd.Series(0.0, index=idx)
        above_200 = close > sma200
        above_50 = close > sma50
        above_20 = close > sma20

        # Above both 200 and 50 → 20
        trend_score += (above_200 & above_50).astype(float) * 20
        # Above 200 only → 10
        trend_score += (above_200 & ~above_50).astype(float) * 10
        # Above 20 only (below 200 and 50) → 5
        trend_score += (~above_200 & ~above_50 & above_20).astype(float) * 5

        sig["trend"] = trend_score

        # ── 4. MOMENTUM SIGNAL (0-20) ──
        mom_score = pd.Series(0.0, index=idx)

        strong_mom = (ret20 > 0.10) & (ret60 > 0)
        positive_both = (ret20 > 0) & (ret60 > 0) & ~strong_mom

        mom_score += strong_mom.astype(float) * 20
        mom_score += positive_both.astype(float) * 10

        sig["momentum"] = mom_score

        # ── 5. RELATIVE STRENGTH (0-20) ──
        rs_score = pd.Series(0.0, index=idx)
        spy_ret20_aligned = spy_ret20.reindex(idx).ffill()

        outperform = ret20 > spy_ret20_aligned + 0.02  # >2% better than SPY
        inline = (ret20 >= spy_ret20_aligned - 0.02) & (ret20 <= spy_ret20_aligned + 0.02)

        rs_score += outperform.astype(float) * 20
        rs_score += inline.astype(float) * 10

        sig["rel_strength"] = rs_score

        # ── TOTAL ──
        sig["total"] = sig[["earnings", "regime", "trend", "momentum", "rel_strength"]].sum(axis=1)
        sig["close"] = close
        sig["fwd_ret"] = close.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

        signals[ticker] = sig

    print(f"  Computed signals for {len(signals)} stocks")
    return signals


# ── BACKTESTING ENGINE ──────────────────────────────────────────────────────
def run_variant(signals, variant_name, entry_fn, spy_data):
    """
    Run a backtest variant.
    entry_fn(sig_row, ticker, date, signals_dict) → (should_trade: bool, size_pct: float, is_option: bool)
    Returns trade list and equity curve.
    """
    spy_close = spy_data["Close"].dropna()
    spy_sma200 = spy_close.rolling(200).mean()
    spy_bull = spy_close > spy_sma200  # True = bull regime

    # Collect all potential entries
    all_dates = set()
    for t, sig in signals.items():
        all_dates.update(sig.index)
    all_dates = sorted([d for d in all_dates if d >= pd.Timestamp(OOT_START) and d <= pd.Timestamp(OOT_END)])

    trades = []
    capital = INITIAL_CAPITAL
    equity_curve = []
    positions = {}  # date_key → {ticker, entry_price, size, exit_date, is_option, premium}

    for date in all_dates:
        # Check and close expired positions
        closed_keys = []
        for key, pos in positions.items():
            if date >= pos["exit_date"]:
                # Close position
                ticker = pos["ticker"]
                if ticker in signals and date in signals[ticker].index:
                    exit_price = signals[ticker].loc[date, "close"]
                else:
                    # Use last available price
                    if ticker in signals:
                        avail = signals[ticker].index[signals[ticker].index <= date]
                        if len(avail) > 0:
                            exit_price = signals[ticker].loc[avail[-1], "close"]
                        else:
                            exit_price = pos["entry_price"]
                    else:
                        exit_price = pos["entry_price"]

                if pos["is_option"]:
                    # Option P&L: premium × (stock_return / option_delta_approx)
                    stock_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    # Approximate ATM call: delta ~0.5, leverage ~3x for 2-week
                    option_ret = max(stock_ret * 3.0, -1.0)  # can't lose more than premium
                    option_ret = max(option_ret, OPTION_STOP)  # stop at -50%
                    pnl = pos["invested"] * option_ret
                    cost = OPTION_COST_PER_CONTRACT + pos["invested"] * OPTION_BIDASK_HAIRCUT
                else:
                    pnl = pos["shares"] * (exit_price - pos["entry_price"])
                    cost = pos["invested"] * SLIPPAGE_PCT * 2  # entry + exit slippage

                net_pnl = pnl - cost
                capital += pos["invested"] + net_pnl

                # Determine regime at entry
                entry_dt = pos["entry_date"]
                if entry_dt in spy_bull.index:
                    regime = "bull" if spy_bull.loc[entry_dt] else "bear"
                else:
                    avail = spy_bull.index[spy_bull.index <= entry_dt]
                    regime = "bull" if (len(avail) > 0 and spy_bull.loc[avail[-1]]) else "bear"

                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(float(pos["entry_price"]), 2),
                    "exit_price": round(float(exit_price), 2),
                    "pnl": round(float(net_pnl), 2),
                    "ret": round(float(net_pnl / pos["invested"]) if pos["invested"] > 0 else 0, 4),
                    "score": pos["score"],
                    "is_option": bool(pos["is_option"]),
                    "regime": regime,
                })
                closed_keys.append(key)

        for k in closed_keys:
            del positions[k]

        # Record equity
        # Mark to market open positions
        mtm = capital
        for key, pos in positions.items():
            ticker = pos["ticker"]
            if ticker in signals and date in signals[ticker].index:
                cur_price = signals[ticker].loc[date, "close"]
                if pos["is_option"]:
                    stock_ret = (cur_price - pos["entry_price"]) / pos["entry_price"]
                    option_ret = max(stock_ret * 3.0, -1.0)
                    option_ret = max(option_ret, OPTION_STOP)
                    mtm += pos["invested"] * (1 + option_ret)
                else:
                    mtm += pos["shares"] * cur_price
            else:
                mtm += pos["invested"]

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(float(mtm), 2)})

        # Generate new entries
        candidates = []
        for ticker, sig in signals.items():
            if date not in sig.index:
                continue
            row = sig.loc[date]
            if pd.isna(row["fwd_ret"]):
                continue

            should_trade, size_pct, is_option = entry_fn(row, ticker, date, signals)
            if should_trade:
                candidates.append((ticker, float(row["total"]), size_pct, is_option, float(row["close"])))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Limit concurrent positions
        max_positions = 5
        available_slots = max_positions - len(positions)

        for ticker, score, size_pct, is_option, price in candidates[:available_slots]:
            if capital <= 10:
                break

            # Don't double up on same ticker
            if any(p["ticker"] == ticker for p in positions.values()):
                continue

            invest = min(capital * size_pct, capital * 0.5)  # never more than 50% in one position
            invest = min(invest, capital - 5)  # keep $5 reserve

            if invest < 10:
                continue

            if is_option:
                premium = price * OPTION_PREMIUM_PCT * 100  # cost of 1 contract
                if premium > 0:
                    n_contracts = max(1, int(invest / premium))
                    actual_invest = n_contracts * premium
                    if actual_invest > invest:
                        n_contracts = max(1, n_contracts - 1)
                        actual_invest = n_contracts * premium
                    invest = min(actual_invest, invest)

            shares = invest / price if not is_option else 0
            exit_date = date + pd.Timedelta(days=int(HOLD_DAYS * 1.5))  # calendar days ≈ 40 trading days

            key = f"{ticker}_{date.strftime('%Y%m%d')}"
            positions[key] = {
                "ticker": ticker,
                "entry_date": date,
                "entry_price": price,
                "shares": shares,
                "invested": invest,
                "exit_date": exit_date,
                "score": score,
                "is_option": is_option,
            }
            capital -= invest

    return trades, equity_curve


# ── VARIANT DEFINITIONS ────────────────────────────────────────────────────
def variant_a(row, ticker, date, signals):
    """Threshold 60: buy shares, hold 40 days."""
    if row["total"] >= 60:
        return True, 0.25, False
    return False, 0, False

def variant_b(row, ticker, date, signals):
    """Threshold 70: more selective."""
    if row["total"] >= 70:
        return True, 0.30, False
    return False, 0, False

def variant_c(row, ticker, date, signals):
    """Threshold 50 + earnings required (≥10)."""
    if row["total"] >= 50 and row["earnings"] >= 10:
        return True, 0.25, False
    return False, 0, False

def variant_d(row, ticker, date, signals):
    """Options overlay: score ≥ 60, buy ATM calls."""
    if row["total"] >= 60:
        return True, 0.20, True  # is_option=True
    return False, 0, False

def variant_e(row, ticker, date, signals):
    """Dynamic sizing: score ≥ 50, size proportional to score."""
    if row["total"] >= 50:
        size = (row["total"] / 100.0) * 0.30  # max 30% at score=100
        return True, size, False
    return False, 0, False

def variant_f(row, ticker, date, signals):
    """Multi-stock portfolio: always trade top-3, threshold 40."""
    # This is called per-stock; the engine sorts by score and takes top candidates
    if row["total"] >= 40:
        return True, 0.20, False
    return False, 0, False


# ── ANALYSIS ────────────────────────────────────────────────────────────────
def analyze_trades(trades, equity_curve, variant_name):
    """Compute performance metrics and 5-gate validation."""
    if len(trades) < 2:
        return {
            "variant": variant_name,
            "n_trades": len(trades),
            "passed_all_gates": False,
            "failure": "insufficient trades",
            "gates": {},
        }

    rets = np.array([t["ret"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    # Basic stats
    total_pnl = float(np.sum(pnls))
    win_rate = float(np.mean(rets > 0))
    avg_ret = float(np.mean(rets))

    # Equity curve metrics
    eq = pd.DataFrame(equity_curve)
    eq["equity"] = eq["equity"].astype(float)

    # Daily returns from equity curve
    eq_rets = eq["equity"].pct_change().dropna()

    if len(eq_rets) > 10 and eq_rets.std() > 0:
        sharpe = float(np.sqrt(252) * eq_rets.mean() / eq_rets.std())
        downside = eq_rets[eq_rets < 0]
        sortino = float(np.sqrt(252) * eq_rets.mean() / downside.std()) if len(downside) > 0 and downside.std() > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Max drawdown
    eq_vals = eq["equity"].values
    peak = np.maximum.accumulate(eq_vals)
    dd = (eq_vals - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(np.min(dd))

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # CAGR
    if len(eq) > 1:
        start_eq = eq["equity"].iloc[0]
        end_eq = eq["equity"].iloc[-1]
        days = (pd.Timestamp(eq["date"].iloc[-1]) - pd.Timestamp(eq["date"].iloc[0])).days
        years = days / 365.25
        cagr = float((end_eq / start_eq) ** (1 / years) - 1) if years > 0 and start_eq > 0 else 0
    else:
        cagr = 0

    # Regime analysis (bull vs bear based on SPY 200-SMA)
    bull_rets = np.array([t["ret"] for t in trades if t.get("regime") == "bull"])
    bear_rets = np.array([t["ret"] for t in trades if t.get("regime") == "bear"])

    if len(bull_rets) > 2 and len(bear_rets) > 2:
        s_bull = np.mean(bull_rets) / (np.std(bull_rets) + 1e-8) * np.sqrt(252/HOLD_DAYS)
        s_bear = np.mean(bear_rets) / (np.std(bear_rets) + 1e-8) * np.sqrt(252/HOLD_DAYS)
        regime_gap = abs(s_bull - s_bear) / max(abs(s_bull), abs(s_bear), 0.01)
    elif len(bull_rets) > 2 or len(bear_rets) > 2:
        # Only one regime present — gap = 0 if strategy works in the regime it trades
        regime_gap = 0.0
    else:
        regime_gap = 0.0

    # Permutation test
    perm_p = permutation_test(trades, sharpe)

    # ── 5-GATE VALIDATION ──
    gate_sharpe = sharpe > 0.5
    gate_perm = perm_p < 0.05
    gate_regime = regime_gap < 0.5
    gate_maxdd = max_dd > -0.50
    gate_trades = len(trades) >= 20

    passed_all = bool(gate_sharpe and gate_perm and gate_regime and gate_maxdd and gate_trades)

    return {
        "variant": variant_name,
        "n_trades": int(len(trades)),
        "total_pnl": round(total_pnl, 2),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(min(profit_factor, 99), 2),
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "regime_gap": round(regime_gap, 3),
        "perm_p_value": round(perm_p, 4),
        "final_equity": round(float(eq["equity"].iloc[-1]), 2) if len(eq) > 0 else INITIAL_CAPITAL,
        "bull_trades": int(len(bull_rets)),
        "bear_trades": int(len(bear_rets)),
        "gates": {
            "sharpe_gt_0.5": bool(gate_sharpe),
            "perm_p_lt_0.05": bool(gate_perm),
            "regime_gap_lt_0.5": bool(gate_regime),
            "maxdd_gt_neg50pct": bool(gate_maxdd),
            "min_20_trades": bool(gate_trades),
        },
        "passed_all_gates": passed_all,
    }


def permutation_test(trades, observed_sharpe, n_perms=N_PERMUTATIONS):
    """Permutation test: randomly assign +/- signs to returns to test if mean is significant."""
    if len(trades) < 5:
        return 1.0

    rets = np.array([t["ret"] for t in trades])
    n = len(rets)
    abs_rets = np.abs(rets)
    count_better = 0

    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        # Random sign assignment — tests if the positive skew is real
        signs = rng.choice([-1, 1], size=n)
        perm_rets = abs_rets * signs
        perm_mean = np.mean(perm_rets)
        perm_std = np.std(perm_rets) + 1e-8
        perm_sharpe = perm_mean / perm_std * np.sqrt(252/HOLD_DAYS)
        if perm_sharpe >= observed_sharpe:
            count_better += 1

    return count_better / n_perms


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("COMPOSITE SIGNAL OPTIMIZER")
    print(f"Universe: {len(UNIVERSE)} stocks | OOT: {OOT_START} to {OOT_END}")
    print(f"Initial Capital: ${INITIAL_CAPITAL} | Hold: {HOLD_DAYS} days")
    print("=" * 70)

    # Download data
    all_data = download_data()

    # Compute signals
    print("\nComputing signals...")
    signals = compute_signals(all_data)

    # Show signal distribution
    all_scores = []
    for t, sig in signals.items():
        oot_sig = sig.loc[sig.index >= OOT_START]
        all_scores.extend(oot_sig["total"].dropna().tolist())

    all_scores = np.array(all_scores)
    print(f"\nScore distribution (OOT period):")
    print(f"  Mean: {np.mean(all_scores):.1f} | Median: {np.median(all_scores):.1f}")
    print(f"  Std:  {np.std(all_scores):.1f}")
    for thresh in [40, 50, 60, 70, 80]:
        pct = np.mean(all_scores >= thresh) * 100
        print(f"  Score >= {thresh}: {pct:.1f}% of observations")

    # Run variants
    variants = {
        "A_threshold_60": variant_a,
        "B_threshold_70": variant_b,
        "C_earnings_required": variant_c,
        "D_options_overlay": variant_d,
        "E_dynamic_sizing": variant_e,
        "F_multi_stock_top3": variant_f,
    }

    spy_data = all_data["SPY"]
    results = {}

    for name, fn in variants.items():
        print(f"\n{'─' * 50}")
        print(f"Running Variant {name}...")
        trades, equity = run_variant(signals, name, fn, spy_data)
        print(f"  Trades: {len(trades)}")

        if trades:
            analysis = analyze_trades(trades, equity, name)
            results[name] = analysis

            # Print summary
            g = analysis["gates"]
            gate_str = " | ".join([
                f"Sharpe={'✓' if g['sharpe_gt_0.5'] else '✗'}",
                f"Perm={'✓' if g['perm_p_lt_0.05'] else '✗'}",
                f"Regime={'✓' if g['regime_gap_lt_0.5'] else '✗'}",
                f"DD={'✓' if g['maxdd_gt_neg50pct'] else '✗'}",
                f"Trades={'✓' if g['min_20_trades'] else '✗'}",
            ])
            status = "PASS" if analysis["passed_all_gates"] else "FAIL"

            print(f"  Sharpe: {analysis['sharpe']:.3f} | Sortino: {analysis['sortino']:.3f}")
            print(f"  Win Rate: {analysis['win_rate']:.1%} | PF: {analysis['profit_factor']:.2f}")
            print(f"  Total PnL: ${analysis['total_pnl']:.2f} | Final Equity: ${analysis['final_equity']:.2f}")
            print(f"  Max DD: {analysis['max_drawdown']:.1%} | CAGR: {analysis['cagr']:.1%}")
            print(f"  Perm p-value: {analysis['perm_p_value']:.4f} | Regime Gap: {analysis['regime_gap']:.3f}")
            print(f"  Gates: {gate_str}")
            print(f"  >>> {status} <<<")

            # Top traded tickers
            ticker_counts = {}
            for t in trades:
                ticker_counts[t["ticker"]] = ticker_counts.get(t["ticker"], 0) + 1
            top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top tickers: {', '.join(f'{t}({n})' for t, n in top_tickers)}")
        else:
            results[name] = {
                "variant": name,
                "n_trades": 0,
                "passed_all_gates": False,
                "failure": "no trades generated",
            }
            print("  No trades generated")

    # ── SUMMARY ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    passed = [r for r in results.values() if r.get("passed_all_gates")]
    failed = [r for r in results.values() if not r.get("passed_all_gates")]

    print(f"\nPassed 5-gate validation: {len(passed)}/{len(results)}")

    if passed:
        best = max(passed, key=lambda x: x.get("sharpe", 0))
        print(f"\nBest variant: {best['variant']}")
        print(f"  Sharpe: {best['sharpe']:.3f} | Sortino: {best['sortino']:.3f}")
        print(f"  Win Rate: {best['win_rate']:.1%} | PF: {best['profit_factor']:.2f}")
        print(f"  ${INITIAL_CAPITAL} → ${best['final_equity']:.2f}")
        print(f"  CAGR: {best['cagr']:.1%} | Max DD: {best['max_drawdown']:.1%}")

    if failed:
        print(f"\nFailed variants:")
        for r in failed:
            reason = r.get("failure", "")
            if not reason:
                gates = r.get("gates", {})
                fails = [k for k, v in gates.items() if not v]
                reason = f"failed: {', '.join(fails)}"
            print(f"  {r['variant']}: {reason}")

    # ── SAVE ──
    output = {
        "metadata": {
            "strategy": "composite_signal_optimizer",
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "universe_size": len(UNIVERSE),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "initial_capital": INITIAL_CAPITAL,
            "hold_days": HOLD_DAYS,
            "n_permutations": N_PERMUTATIONS,
            "signals": ["earnings", "regime", "trend", "momentum", "rel_strength"],
        },
        "score_distribution": {
            "mean": round(float(np.mean(all_scores)), 1),
            "median": round(float(np.median(all_scores)), 1),
            "std": round(float(np.std(all_scores)), 1),
            "pct_above_60": round(float(np.mean(all_scores >= 60) * 100), 1),
            "pct_above_70": round(float(np.mean(all_scores >= 70) * 100), 1),
        },
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    main()
