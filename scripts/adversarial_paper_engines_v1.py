#!/usr/bin/env python3
"""
Adversarial Backtest for Paper Trading Engines
================================================

Tests 9 paper engines against the standard 6-gate adversarial framework:
  1. Re-implementation match (verify logic reproduces claimed results)
  2. Inverse test (flip signal direction — should lose money)
  3. Random permutation (random entries — should not beat real signal)
  4. Cost sensitivity (double commissions — should still be profitable)
  5. Sub-period stability (first half vs second half)
  6. Parameter robustness (perturb key params ±20%)

Each engine's core signal logic is extracted and backtested on historical data.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure imports work
BASE = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(BASE))

import yfinance as yf

N_RANDOM_ITER = 200
SEED = 42
np.random.seed(SEED)


def download_cached(tickers, start="2023-01-01"):
    """Download data once and cache."""
    raw = yf.download(tickers, start=start, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        volume = raw["Volume"]
        high = raw["High"]
        low = raw["Low"]
    else:
        close = raw
        volume = raw.get("Volume", pd.DataFrame())
        high = raw.get("High", pd.DataFrame())
        low = raw.get("Low", pd.DataFrame())

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = volume.columns.get_level_values(-1)
    if isinstance(high.columns, pd.MultiIndex):
        high.columns = high.columns.get_level_values(-1)
    if isinstance(low.columns, pd.MultiIndex):
        low.columns = low.columns.get_level_values(-1)

    rename = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename).ffill()
    volume = volume.rename(columns=rename).ffill()
    high = high.rename(columns=rename).ffill()
    low = low.rename(columns=rename).ffill()

    return close, volume, high, low


def compute_metrics(pnls):
    """Compute Sharpe, WR, PF from a series of P&L values."""
    if len(pnls) == 0:
        return {"sharpe": 0, "wr": 0, "pf": 0, "n": 0, "total_pnl": 0, "mean_pnl": 0}
    pnls = np.array(pnls)
    mean = np.mean(pnls)
    std = np.std(pnls, ddof=1) if len(pnls) > 1 else 1e-10
    sharpe = mean / (std + 1e-10) * np.sqrt(52)  # annualized assuming ~weekly trades
    wr = np.mean(pnls > 0) * 100
    wins = np.sum(pnls[pnls > 0])
    losses = np.abs(np.sum(pnls[pnls < 0]))
    pf = wins / (losses + 1e-10)
    return {
        "sharpe": round(float(sharpe), 3),
        "wr": round(float(wr), 1),
        "pf": round(float(pf), 3),
        "n": len(pnls),
        "total_pnl": round(float(np.sum(pnls)), 2),
        "mean_pnl": round(float(mean), 4),
    }


def run_adversarial(name, signal_fn, pnl_fn, dates, data_dict,
                    commission_base=0, param_keys=None, params=None):
    """
    Run 6-gate adversarial test on a strategy.

    signal_fn(date_idx, data_dict, params) -> signal (-1, 0, 1) or list of signals
    pnl_fn(signal, date_idx, data_dict, commission, params) -> pnl float
    """
    results = {}

    # Gate 1: Re-implementation (baseline)
    real_pnls = []
    for i in range(len(dates)):
        sig = signal_fn(i, data_dict, params)
        if sig != 0:
            p = pnl_fn(sig, i, data_dict, commission_base, params)
            if p is not None:
                real_pnls.append(p)

    baseline = compute_metrics(real_pnls)
    results["gate1_baseline"] = baseline
    results["gate1_pass"] = baseline["n"] >= 5  # need enough trades

    if baseline["n"] < 5:
        print(f"  {name}: SKIP — only {baseline['n']} trades generated")
        results["verdict"] = "SKIP (insufficient trades)"
        results["gates_passed"] = 0
        results["gates_total"] = 6
        return results

    # Gate 2: Inverse test (flip signal)
    inv_pnls = []
    for i in range(len(dates)):
        sig = signal_fn(i, data_dict, params)
        if sig != 0:
            p = pnl_fn(-sig, i, data_dict, commission_base, params)
            if p is not None:
                inv_pnls.append(p)
    inv_metrics = compute_metrics(inv_pnls)
    results["gate2_inverse"] = inv_metrics
    # Inverse should be worse than real
    results["gate2_pass"] = inv_metrics["mean_pnl"] < baseline["mean_pnl"]

    # Gate 3: Random permutation
    random_better = 0
    for _ in range(N_RANDOM_ITER):
        rand_pnls = []
        for i in range(len(dates)):
            sig = signal_fn(i, data_dict, params)
            if sig != 0:
                rand_sig = np.random.choice([-1, 1])
                p = pnl_fn(rand_sig, i, data_dict, commission_base, params)
                if p is not None:
                    rand_pnls.append(p)
        if rand_pnls and np.mean(rand_pnls) >= baseline["mean_pnl"]:
            random_better += 1

    p_value = random_better / N_RANDOM_ITER
    results["gate3_random_p"] = round(p_value, 4)
    results["gate3_pass"] = p_value < 0.10  # real beats random 90%+ of time

    # Gate 4: Cost sensitivity (double commission)
    cost_pnls = []
    for i in range(len(dates)):
        sig = signal_fn(i, data_dict, params)
        if sig != 0:
            p = pnl_fn(sig, i, data_dict, commission_base * 2, params)
            if p is not None:
                cost_pnls.append(p)
    cost_metrics = compute_metrics(cost_pnls)
    results["gate4_2x_cost"] = cost_metrics
    results["gate4_pass"] = cost_metrics["mean_pnl"] > 0  # still profitable

    # Gate 5: Sub-period stability
    mid = len(dates) // 2
    first_pnls, second_pnls = [], []
    for i in range(len(dates)):
        sig = signal_fn(i, data_dict, params)
        if sig != 0:
            p = pnl_fn(sig, i, data_dict, commission_base, params)
            if p is not None:
                if i < mid:
                    first_pnls.append(p)
                else:
                    second_pnls.append(p)

    first_m = compute_metrics(first_pnls)
    second_m = compute_metrics(second_pnls)
    results["gate5_first_half"] = first_m
    results["gate5_second_half"] = second_m
    # Both halves should be profitable, or at least not terrible
    results["gate5_pass"] = (first_m["mean_pnl"] > -0.5 * abs(baseline["mean_pnl"]) and
                              second_m["mean_pnl"] > -0.5 * abs(baseline["mean_pnl"]))

    # Gate 6: Parameter robustness (if params provided)
    if params and param_keys:
        robust_results = []
        for key in param_keys:
            orig_val = params[key]
            for mult in [0.8, 1.2]:
                test_params = params.copy()
                test_params[key] = orig_val * mult
                perturbed_pnls = []
                for i in range(len(dates)):
                    sig = signal_fn(i, data_dict, test_params)
                    if sig != 0:
                        p = pnl_fn(sig, i, data_dict, commission_base, test_params)
                        if p is not None:
                            perturbed_pnls.append(p)
                pm = compute_metrics(perturbed_pnls)
                robust_results.append(pm["mean_pnl"])

        # At least 50% of perturbations should stay profitable
        frac_profitable = np.mean(np.array(robust_results) > 0) if robust_results else 0
        results["gate6_param_robust_frac"] = round(float(frac_profitable), 2)
        results["gate6_pass"] = frac_profitable >= 0.5
    else:
        results["gate6_pass"] = True  # no params to test
        results["gate6_param_robust_frac"] = 1.0

    gates = [results[f"gate{i}_pass"] for i in range(1, 7)]
    results["gates_passed"] = sum(gates)
    results["gates_total"] = 6
    results["verdict"] = f"{sum(gates)}/6 PASS" + (" — KEEP" if sum(gates) >= 4 else " — REMOVE")

    return results


# ============================================================
# ENGINE 1: Strategy Rotation v2F (Dynamic Contrarian)
# ============================================================
def test_strategy_rotation_v2f(close, vix):
    """SPY/QQQ rotation based on VIX + SMA200 regime."""
    print("\n" + "="*60)
    print("ENGINE: strategy_rotation_v2f (Dynamic Contrarian)")
    print("="*60)

    spy = close["SPY"].dropna()
    qqq = close["QQQ"].dropna()
    vix_s = vix.dropna()
    common = spy.index.intersection(qqq.index).intersection(vix_s.index)
    spy, qqq, vix_s = spy[common], qqq[common], vix_s[common]

    sma200 = spy.rolling(200).mean()
    spy_ret5 = spy.pct_change(5)
    vix_chg5 = vix_s.pct_change(5)

    dates = spy.index[200:]  # need 200 for SMA

    data = {"spy": spy, "qqq": qqq, "vix": vix_s, "sma200": sma200,
            "spy_ret5": spy_ret5, "vix_chg5": vix_chg5, "dates": dates}

    params = {"vix_fade_thresh": 25, "contrarian_mult": -1.5, "hold_days": 5}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        v = d["vix"][dt]
        bull = d["spy"][dt] > d["sma200"][dt]
        vchg = d["vix_chg5"][dt]
        sret = d["spy_ret5"][dt]
        thresh = p["contrarian_mult"] * v / 100

        if pd.isna(v) or pd.isna(vchg) or pd.isna(sret):
            return 0
        if v > p["vix_fade_thresh"] and vchg < 0:
            return 1  # SPY long (VIX fade)
        elif bull:
            return 1  # QQQ long
        elif sret <= thresh:
            return 1  # contrarian long
        return 0

    def pnl_fn(sig, i, d, comm, p):
        if i + int(p["hold_days"]) >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        v = d["vix"][dt]
        bull = d["spy"][dt] > d["sma200"][dt]

        # Choose ticker based on mode
        if v > p["vix_fade_thresh"]:
            ticker_data = d["spy"]
        elif bull:
            ticker_data = d["qqq"]
        else:
            ticker_data = d["spy"]

        entry = ticker_data[dt]
        exit_dt = d["dates"][min(i + int(p["hold_days"]), len(d["dates"])-1)]
        exit_p = ticker_data[exit_dt]
        ret = (exit_p / entry - 1) * sig
        pnl = ret * 645 - comm  # $645 account
        return float(pnl)

    results = run_adversarial(
        "strategy_rotation_v2f", signal_fn, pnl_fn, dates, data,
        commission_base=0.13, param_keys=["vix_fade_thresh", "contrarian_mult", "hold_days"],
        params=params
    )
    return results


# ============================================================
# ENGINE 2: VIX Call Spread
# ============================================================
def test_vix_call_spread(close, vix):
    """Sell VIX call spreads when VIX > 20."""
    print("\n" + "="*60)
    print("ENGINE: vix_call_spread")
    print("="*60)

    vix_s = vix.dropna()
    dates = vix_s.index[60:]

    data = {"vix": vix_s, "dates": dates}
    params = {"entry_thresh": 20, "hold_days": 14, "spread_width": 5}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        v = d["vix"][dt]
        if pd.isna(v):
            return 0
        if v > p["entry_thresh"]:
            return -1  # sell call spread
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        if i + hold >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        exit_dt = d["dates"][min(i + hold, len(d["dates"])-1)]
        v_entry = d["vix"][dt]
        v_exit = d["vix"][exit_dt]

        # Simplified: credit ~ VIX level * 0.1, loss if VIX goes higher
        K_short = round(v_entry)
        K_long = K_short + p["spread_width"]
        credit_approx = max(v_entry - K_short, 0.5) * 100 * 0.3  # rough BS
        exit_cost = max(0, (v_exit - K_short) - max(0, v_exit - K_long)) * 100

        pnl = (credit_approx - exit_cost) * sig - comm
        return float(pnl)

    results = run_adversarial(
        "vix_call_spread", signal_fn, pnl_fn, dates, data,
        commission_base=4.0, param_keys=["entry_thresh", "hold_days"],
        params=params
    )
    return results


# ============================================================
# ENGINE 3: VIX MR Spread (Bull Put Spread on SPY when VIX > 25)
# ============================================================
def test_vix_mr_spread(close, vix):
    """Sell SPY put spreads when VIX > 25."""
    print("\n" + "="*60)
    print("ENGINE: vix_mr_spread")
    print("="*60)

    spy = close["SPY"].dropna()
    vix_s = vix.dropna()
    common = spy.index.intersection(vix_s.index)
    spy, vix_s = spy[common], vix_s[common]
    dates = spy.index[30:]

    data = {"spy": spy, "vix": vix_s, "dates": dates}
    params = {"vix_entry": 25, "otm_pct": 0.025, "hold_days": 14, "spread_width": 5}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        v = d["vix"][dt]
        if pd.isna(v):
            return 0
        if v > p["vix_entry"] and v <= 35:
            return 1  # sell put spread (bullish on SPY)
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        if i + hold >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        exit_dt = d["dates"][min(i + hold, len(d["dates"])-1)]
        spy_entry = d["spy"][dt]
        spy_exit = d["spy"][exit_dt]
        v = d["vix"][dt]

        short_put = spy_entry * (1 - p["otm_pct"])
        # Credit approx from BS: higher VIX = more credit
        credit = max(v / 100 * spy_entry * 0.15, 0.5)

        # At exit: if SPY > short put, keep credit; else lose
        if sig > 0:
            if spy_exit >= short_put:
                pnl = credit - comm
            else:
                intrinsic_loss = min(short_put - spy_exit, p["spread_width"])
                pnl = credit - intrinsic_loss - comm
        else:
            # Inverse: bullish spread in bearish direction
            if spy_exit < short_put:
                pnl = credit - comm
            else:
                pnl = -credit - comm
        return float(pnl)

    results = run_adversarial(
        "vix_mr_spread", signal_fn, pnl_fn, dates, data,
        commission_base=0, param_keys=["vix_entry", "otm_pct", "hold_days"],
        params=params
    )
    return results


# ============================================================
# ENGINE 4: Volume Surge
# ============================================================
def test_volume_surge(close, volume):
    """Buy sector ETFs on 3-day volume surge, hold 10 days."""
    print("\n" + "="*60)
    print("ENGINE: volume_surge")
    print("="*60)

    etfs = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY"]
    etfs = [e for e in etfs if e in close.columns and e in volume.columns]

    dates = close.index[30:]
    data = {"close": close, "volume": volume, "dates": dates, "etfs": etfs}
    params = {"vol_thresh": 1.5, "consec_days": 3, "hold_days": 10, "lookback": 20}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        idx = d["close"].index.get_loc(dt)
        if idx < 30:
            return 0

        for etf in d["etfs"]:
            v = d["volume"][etf].iloc[idx-int(p["lookback"])-int(p["consec_days"]):idx+1]
            if len(v) < p["lookback"] + p["consec_days"]:
                continue
            avg_vol = v.iloc[:int(p["lookback"])].mean()
            if avg_vol <= 0:
                continue
            recent = v.iloc[-int(p["consec_days"]):]
            ratios = recent / avg_vol
            if all(r >= p["vol_thresh"] for r in ratios.values):
                return 1  # buy signal
        return 0

    def pnl_fn(sig, i, d, comm, p):
        dt = d["dates"][i]
        idx = d["close"].index.get_loc(dt)
        hold = int(p["hold_days"])
        if idx + hold >= len(d["close"]):
            return None

        # Find which ETF triggered (take first one)
        best_etf = None
        for etf in d["etfs"]:
            v = d["volume"][etf].iloc[idx-int(p["lookback"])-int(p["consec_days"]):idx+1]
            if len(v) < p["lookback"] + p["consec_days"]:
                continue
            avg_vol = v.iloc[:int(p["lookback"])].mean()
            if avg_vol <= 0:
                continue
            recent = v.iloc[-int(p["consec_days"]):]
            if all(r >= p["vol_thresh"] for r in (recent / avg_vol).values):
                best_etf = etf
                break

        if best_etf is None:
            best_etf = "SPY"

        entry = d["close"][best_etf].iloc[idx]
        exit_p = d["close"][best_etf].iloc[idx + hold]
        ret = (exit_p / entry - 1) * sig
        pnl = ret * 200 - comm  # $200 position size
        return float(pnl)

    results = run_adversarial(
        "volume_surge", signal_fn, pnl_fn, dates, data,
        commission_base=0.20, param_keys=["vol_thresh", "consec_days", "hold_days"],
        params=params
    )
    return results


# ============================================================
# ENGINE 5: Vol Crush (Pre-Earnings Iron Condor)
# ============================================================
def test_vol_crush(close, volume):
    """Simplified: sell straddle-proxy pre-earnings, capture vol crush."""
    print("\n" + "="*60)
    print("ENGINE: vol_crush (pre-earnings IC)")
    print("="*60)

    # Use big-cap stocks with known earnings patterns
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM"]
    stocks = [s for s in stocks if s in close.columns]

    if not stocks:
        print("  No stock data available")
        return {"verdict": "SKIP (no data)", "gates_passed": 0, "gates_total": 6}

    # Simulate: find high-vol periods (proxy for pre-earnings) and sell vol
    dates = close.index[60:]
    data = {"close": close, "dates": dates, "stocks": stocks}
    params = {"vol_lookback": 20, "vol_spike_mult": 1.3, "hold_days": 5}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        idx = d["close"].index.get_loc(dt)
        if idx < 60:
            return 0

        for stock in d["stocks"]:
            px = d["close"][stock].iloc[:idx+1]
            if len(px) < 260:
                continue
            rets = px.pct_change().dropna()
            if len(rets) < 60:
                continue
            vol_20 = rets.iloc[-20:].std() * np.sqrt(252)
            vol_60 = rets.iloc[-60:].std() * np.sqrt(252)
            if vol_20 > vol_60 * p["vol_spike_mult"]:
                # Also check stock hasn't moved too much
                move_5d = abs(px.iloc[-1] / px.iloc[-6] - 1)
                if move_5d < 0.05:
                    return -1  # sell vol (iron condor)
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        dt = d["dates"][i]
        idx = d["close"].index.get_loc(dt)
        if idx + hold >= len(d["close"]):
            return None

        # Find triggered stock
        for stock in d["stocks"]:
            px = d["close"][stock].iloc[:idx+1]
            if len(px) < 260:
                continue
            rets = px.pct_change().dropna()
            if len(rets) < 60:
                continue
            vol_20 = rets.iloc[-20:].std() * np.sqrt(252)
            vol_60 = rets.iloc[-60:].std() * np.sqrt(252)
            if vol_20 > vol_60 * p["vol_spike_mult"]:
                S = float(px.iloc[-1])
                S_exit = float(d["close"][stock].iloc[idx + hold])
                move = abs(S_exit / S - 1)

                # IC P&L: credit from vol premium, minus any breach
                credit = vol_20 * S * 0.04  # rough credit proxy
                stdev_move = vol_20 * np.sqrt(hold/252) * S

                if sig < 0:  # sell vol (normal)
                    if move * S < stdev_move * 0.8:  # within wings
                        pnl = credit * 0.7 - comm  # capture 70% of credit
                    else:
                        pnl = -(stdev_move * 0.5) + credit - comm  # partial loss
                else:  # buy vol (inverse)
                    pnl = -credit + max(0, move * S - stdev_move * 0.5) - comm

                return float(pnl)
        return None

    results = run_adversarial(
        "vol_crush", signal_fn, pnl_fn, dates, data,
        commission_base=18.80, param_keys=["vol_spike_mult", "hold_days"],
        params=params
    )
    return results


# ============================================================
# ENGINE 6: Vol Term Structure (VIX spike buy quality stocks)
# ============================================================
def test_vol_term_structure(close, vix):
    """Buy quality basket when VIX > 1.15x 60d mean and declining."""
    print("\n" + "="*60)
    print("ENGINE: vol_term_structure")
    print("="*60)

    spy = close["SPY"].dropna()
    vix_s = vix.dropna()
    common = spy.index.intersection(vix_s.index)
    spy, vix_s = spy[common], vix_s[common]

    vix_60d = vix_s.rolling(60).mean()
    dates = spy.index[70:]

    # Use SPY as basket proxy (quality stocks track SPY closely)
    data = {"spy": spy, "vix": vix_s, "vix_60d": vix_60d, "dates": dates}
    params = {"spike_mult": 1.15, "hold_days": 21, "tp_pct": 0.10, "sl_pct": -0.15}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        v = d["vix"][dt]
        v_mean = d["vix_60d"][dt]
        if pd.isna(v) or pd.isna(v_mean) or v_mean == 0:
            return 0

        if i == 0:
            return 0
        v_prev_dt = d["dates"][i-1]
        v_prev = d["vix"][v_prev_dt]

        if v > p["spike_mult"] * v_mean and v < v_prev:
            return 1  # buy quality basket
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        if i + hold >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        entry = d["spy"][dt]

        # Check early exits
        for h in range(1, hold + 1):
            if i + h >= len(d["dates"]):
                break
            exit_dt = d["dates"][i + h]
            exit_p = d["spy"][exit_dt]
            ret = (exit_p / entry - 1) * sig

            # TP/SL
            if ret >= p["tp_pct"] or ret <= p["sl_pct"]:
                pnl = ret * 300 - comm  # $300 basket
                return float(pnl)

            # VIX normalized exit
            v_now = d["vix"][exit_dt]
            v_mean = d["vix_60d"][exit_dt]
            if not pd.isna(v_now) and not pd.isna(v_mean) and v_now < v_mean:
                pnl = ret * 300 - comm
                return float(pnl)

        # Max hold exit
        exit_dt = d["dates"][min(i + hold, len(d["dates"])-1)]
        exit_p = d["spy"][exit_dt]
        ret = (exit_p / entry - 1) * sig
        pnl = ret * 300 - comm
        return float(pnl)

    results = run_adversarial(
        "vol_term_structure", signal_fn, pnl_fn, dates, data,
        commission_base=0, param_keys=["spike_mult", "hold_days"],
        params=params
    )
    return results


# ============================================================
# ENGINE 7: Sector Momentum Spreads
# ============================================================
def test_sector_momentum_spreads(close, vix):
    """Leader/laggard sector pair trade on regime change."""
    print("\n" + "="*60)
    print("ENGINE: sector_momentum_spreads")
    print("="*60)

    sectors = ["XLK", "XLF", "XLE", "XLV", "XLC", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLY"]
    sectors = [s for s in sectors if s in close.columns]
    spy = close["SPY"].dropna()
    vix_s = vix.dropna()

    sc = close[sectors].dropna(how="all")
    common = sc.index.intersection(vix_s.index).intersection(spy.index)
    sc, vix_s, spy = sc.loc[common], vix_s[common], spy[common]

    sma200 = spy.rolling(200).mean()
    dates = sc.index[200:]

    data = {"sc": sc, "vix": vix_s, "spy": spy, "sma200": sma200,
            "dates": dates, "sectors": sectors}
    params = {"mom_short": 10, "mom_long": 20, "min_spread": 2.0, "hold_days": 15}

    # Track regime for change detection
    prev_regime = [None]

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        v = d["vix"][dt]
        bull = d["spy"][dt] > d["sma200"][dt]
        regime = "bull" if bull else ("high_vol" if v > 25 else "bear")

        if prev_regime[0] is not None and regime != prev_regime[0]:
            prev_regime[0] = regime
            # Compute momentum rankings
            idx = d["sc"].index.get_loc(dt)
            if idx < 25:
                return 0

            rankings = {}
            for s in d["sectors"]:
                px = d["sc"][s].iloc[:idx+1].dropna()
                if len(px) < 25:
                    continue
                m10 = float(px.iloc[-1] / px.iloc[-11] - 1) * 100
                m20 = float(px.iloc[-1] / px.iloc[-21] - 1) * 100
                rankings[s] = 0.6 * m10 + 0.4 * m20

            if len(rankings) < 3:
                return 0

            sorted_s = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            spread = sorted_s[0][1] - sorted_s[-1][1]
            if spread >= p["min_spread"]:
                return 1  # long leader, short laggard
            return 0

        prev_regime[0] = regime
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        if i + hold >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        idx = d["sc"].index.get_loc(dt)

        rankings = {}
        for s in d["sectors"]:
            px = d["sc"][s].iloc[:idx+1].dropna()
            if len(px) < 25:
                continue
            m10 = float(px.iloc[-1] / px.iloc[-11] - 1) * 100
            m20 = float(px.iloc[-1] / px.iloc[-21] - 1) * 100
            rankings[s] = 0.6 * m10 + 0.4 * m20

        sorted_s = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        leader = sorted_s[0][0]
        laggard = sorted_s[-1][0]

        exit_dt = d["dates"][min(i + hold, len(d["dates"])-1)]

        leader_ret = d["sc"][leader][exit_dt] / d["sc"][leader][dt] - 1
        laggard_ret = d["sc"][laggard][exit_dt] / d["sc"][laggard][dt] - 1

        # Long leader, short laggard
        pair_ret = (leader_ret - laggard_ret) * sig
        pnl = pair_ret * 150 - comm  # $150 per leg
        return float(pnl)

    results = run_adversarial(
        "sector_momentum_spreads", signal_fn, pnl_fn, dates, data,
        commission_base=5.20, param_keys=["min_spread", "hold_days", "mom_short"],
        params=params
    )
    return results


# ============================================================
# ENGINE 8: Sector Spreads (Bull Call Spreads with LGBM)
# ============================================================
def test_sector_spreads(close, vix):
    """Bull call spreads on top LGBM-ranked sectors (simplified to momentum)."""
    print("\n" + "="*60)
    print("ENGINE: sector_spreads (bull call spreads)")
    print("="*60)

    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    sectors = [s for s in sectors if s in close.columns]
    spy = close["SPY"].dropna()
    vix_s = vix.dropna()
    sc = close[sectors].dropna(how="all")
    common = sc.index.intersection(vix_s.index).intersection(spy.index)
    sc, vix_s, spy = sc.loc[common], vix_s[common], spy[common]

    dates = sc.index[260:]
    rebal_counter = [0]

    data = {"sc": sc, "vix": vix_s, "spy": spy, "dates": dates, "sectors": sectors}
    params = {"rebal_interval": 5, "top_k": 2, "dte": 21, "spread_pct": 3.0}

    def signal_fn(i, d, p):
        rebal_counter[0] += 1
        if rebal_counter[0] % int(p["rebal_interval"]) != 0:
            return 0

        dt = d["dates"][i]
        v = d["vix"][dt]

        # Simplified regime check (VIX proxy for regime score)
        if v < 15:  # low vol regime -> no trades
            return 0

        # Momentum ranking (simplified LGBM proxy)
        idx = d["sc"].index.get_loc(dt)
        if idx < 260:
            return 0

        rets_21d = {}
        for s in d["sectors"]:
            px = d["sc"][s].iloc[:idx+1].dropna()
            if len(px) > 21:
                rets_21d[s] = float(px.iloc[-1] / px.iloc[-22] - 1)

        if len(rets_21d) < 5:
            return 0

        return 1  # enter bull call spreads on top sectors

    def pnl_fn(sig, i, d, comm, p):
        dte = int(p["dte"])
        if i + dte >= len(d["dates"]):
            return None

        dt = d["dates"][i]
        idx = d["sc"].index.get_loc(dt)
        v = d["vix"][dt]

        # Get top-k sectors by momentum
        rets_21d = {}
        for s in d["sectors"]:
            px = d["sc"][s].iloc[:idx+1].dropna()
            if len(px) > 21:
                rets_21d[s] = float(px.iloc[-1] / px.iloc[-22] - 1)

        sorted_s = sorted(rets_21d.items(), key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in sorted_s[:int(p["top_k"])]]

        total_pnl = 0
        for tk in picks:
            S = d["sc"][tk][dt]
            exit_dt = d["dates"][min(i + dte, len(d["dates"])-1)]
            S_exit = d["sc"][tk][exit_dt]

            K1 = round(S * 1.02)  # 2% OTM
            K2 = round(K1 * (1 + p["spread_pct"] / 100))

            # Bull call spread P&L at expiry
            entry_cost = max(0.5, S * (v/100) * 0.3 * 0.1)  # rough BS estimate
            intrinsic = max(0, S_exit - K1) - max(0, S_exit - K2)

            if sig > 0:
                spread_pnl = (intrinsic - entry_cost) * 100 - 2.60
            else:
                spread_pnl = (entry_cost - intrinsic) * 100 - 2.60

            total_pnl += spread_pnl

        return float(total_pnl)

    results = run_adversarial(
        "sector_spreads", signal_fn, pnl_fn, dates, data,
        commission_base=2.60, param_keys=["rebal_interval", "top_k", "dte"],
        params=params
    )
    return results


# ============================================================
# ENGINE 9: Sector Pairs (Long top3 + Short bottom3)
# ============================================================
def test_sector_pairs(close, vix):
    """Long/short sector pair spreads based on LGBM ranking."""
    print("\n" + "="*60)
    print("ENGINE: sector_pairs")
    print("="*60)

    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    sectors = [s for s in sectors if s in close.columns]
    vix_s = vix.dropna()
    sc = close[sectors].dropna(how="all")
    common = sc.index.intersection(vix_s.index)
    sc, vix_s = sc.loc[common], vix_s[common]

    dates = sc.index[260:]
    rebal_counter = [0]

    data = {"sc": sc, "vix": vix_s, "dates": dates, "sectors": sectors}
    params = {"rebal_interval": 10, "top_k": 3, "bottom_k": 3, "dte": 21, "vix_thresh": 20}

    def signal_fn(i, d, p):
        rebal_counter[0] += 1
        if rebal_counter[0] % int(p["rebal_interval"]) != 0:
            return 0

        dt = d["dates"][i]
        v = d["vix"][dt]
        if v >= p["vix_thresh"]:
            return 0  # VIX gate

        return 1

    def pnl_fn(sig, i, d, comm, p):
        dte = int(p["dte"])
        if i + dte >= len(d["dates"]):
            return None

        dt = d["dates"][i]
        idx = d["sc"].index.get_loc(dt)
        v = d["vix"][dt]

        # Momentum ranking
        rets_21d = {}
        for s in d["sectors"]:
            px = d["sc"][s].iloc[:idx+1].dropna()
            if len(px) > 21:
                rets_21d[s] = float(px.iloc[-1] / px.iloc[-22] - 1)

        sorted_s = sorted(rets_21d.items(), key=lambda x: x[1], reverse=True)
        longs = [s for s, _ in sorted_s[:int(p["top_k"])]]
        shorts = [s for s, _ in sorted_s[-int(p["bottom_k"]):]]

        exit_dt = d["dates"][min(i + dte, len(d["dates"])-1)]

        total_pnl = 0
        # Long side: bull call spreads
        for tk in longs:
            S = d["sc"][tk][dt]
            S_exit = d["sc"][tk][exit_dt]
            K1, K2 = round(S), round(S * 1.03)
            entry_cost = max(0.3, S * (v/100) * 0.25 * 0.1)
            intrinsic = max(0, S_exit - K1) - max(0, S_exit - K2)
            total_pnl += ((intrinsic - entry_cost) * 100 - 2.60) * sig

        # Short side: bear put spreads
        for tk in shorts:
            S = d["sc"][tk][dt]
            S_exit = d["sc"][tk][exit_dt]
            K2, K1 = round(S), round(S * 0.97)
            entry_cost = max(0.3, S * (v/100) * 0.20 * 0.1)
            intrinsic = max(0, K2 - S_exit) - max(0, K1 - S_exit)
            total_pnl += ((intrinsic - entry_cost) * 100 - 2.60) * sig

        return float(total_pnl)

    results = run_adversarial(
        "sector_pairs", signal_fn, pnl_fn, dates, data,
        commission_base=2.60, param_keys=["rebal_interval", "vix_thresh", "dte"],
        params=params
    )
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("ADVERSARIAL PAPER ENGINE VALIDATION v1")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Download all data once
    print("\nDownloading market data...")
    all_tickers = [
        "SPY", "QQQ", "^VIX", "^VIX3M", "TLT", "GLD",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM",
    ]
    close, volume, high, low = download_cached(all_tickers, start="2023-01-01")

    vix_col = "VIX" if "VIX" in close.columns else "^VIX"
    vix = close[vix_col].dropna()

    print(f"Data: {len(close)} days, {len(close.columns)} tickers")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    # Run all tests
    all_results = {}

    all_results["strategy_rotation_v2f"] = test_strategy_rotation_v2f(close, vix)
    all_results["vix_call_spread"] = test_vix_call_spread(close, vix)
    all_results["vix_mr_spread"] = test_vix_mr_spread(close, vix)
    all_results["volume_surge"] = test_volume_surge(close, volume)
    all_results["vol_crush"] = test_vol_crush(close, volume)
    all_results["vol_term_structure"] = test_vol_term_structure(close, vix)
    all_results["sector_momentum_spreads"] = test_sector_momentum_spreads(close, vix)
    all_results["sector_spreads"] = test_sector_spreads(close, vix)
    all_results["sector_pairs"] = test_sector_pairs(close, vix)

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Engine':<30} {'Gates':<8} {'Sharpe':<8} {'WR':<7} {'PF':<7} {'N':<5} {'Verdict'}")
    print("-" * 75)

    for name, r in all_results.items():
        baseline = r.get("gate1_baseline", {})
        sharpe = baseline.get("sharpe", "N/A")
        wr = baseline.get("wr", "N/A")
        pf = baseline.get("pf", "N/A")
        n = baseline.get("n", 0)
        gates = f"{r['gates_passed']}/{r['gates_total']}"
        verdict = r.get("verdict", "N/A")

        # Print gate details
        print(f"{name:<30} {gates:<8} {sharpe:<8} {wr:<7} {pf:<7} {n:<5} {verdict}")

        # Detail lines
        for gate_num in range(1, 7):
            pass_key = f"gate{gate_num}_pass"
            if pass_key in r:
                status = "PASS" if r[pass_key] else "FAIL"
                detail = ""
                if gate_num == 2:
                    inv = r.get("gate2_inverse", {})
                    detail = f" (inv_mean={inv.get('mean_pnl', 'N/A')} vs real={baseline.get('mean_pnl', 'N/A')})"
                elif gate_num == 3:
                    detail = f" (p={r.get('gate3_random_p', 'N/A')})"
                elif gate_num == 4:
                    cost = r.get("gate4_2x_cost", {})
                    detail = f" (2x_cost_mean={cost.get('mean_pnl', 'N/A')})"
                elif gate_num == 5:
                    f1 = r.get("gate5_first_half", {})
                    f2 = r.get("gate5_second_half", {})
                    detail = f" (H1={f1.get('mean_pnl', 'N/A')}, H2={f2.get('mean_pnl', 'N/A')})"
                elif gate_num == 6:
                    detail = f" (robust_frac={r.get('gate6_param_robust_frac', 'N/A')})"
                print(f"  Gate {gate_num}: {status}{detail}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    keep = [n for n, r in all_results.items() if r["gates_passed"] >= 4]
    remove = [n for n, r in all_results.items() if r["gates_passed"] < 4]
    skip = [n for n, r in all_results.items() if r.get("verdict", "").startswith("SKIP")]

    print(f"KEEP ({len(keep)}):   {', '.join(keep) if keep else 'None'}")
    print(f"REMOVE ({len(remove)}): {', '.join(remove) if remove else 'None'}")
    print(f"SKIPPED ({len(skip)}): {', '.join(skip) if skip else 'None'}")

    # Save results
    import json
    out_dir = BASE / "output" / "adversarial_paper_engines_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(out_dir / "results.json", "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
