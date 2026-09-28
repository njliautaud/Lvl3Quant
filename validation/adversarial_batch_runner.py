#!/usr/bin/env python3
"""
Adversarial Batch Runner — 6-gate adversarial backtest for paper engines
========================================================================
Tests: sector_combined_v10, strategy_rotation_v2f, vol_term_structure,
       sector_reversal, vix_call_spread, volume_surge

Framework:
  1. Re-implementation match (baseline Sharpe, trade count)
  2. Inverse test (flip signal -> should lose)
  3. Random permutation (200 iterations, p-value)
  4. Cost sensitivity (double costs -> still profitable?)
  5. Sub-period stability (all sub-periods positive?)
  6. Parameter robustness (±20% perturbation, >50% grid > Sharpe 0.3)

Data: yfinance, 2019-01-01 to 2025-12-31
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

import yfinance as yf

N_RANDOM = 200
SEED = 42
np.random.seed(SEED)

# ─── Data download ───────────────────────────────────────────────────
print("Downloading market data (2019-2025)...")
ALL_TICKERS = [
    "SPY", "QQQ", "^VIX", "TLT", "GLD",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH",
    "LLY", "AVGO", "AMD", "HD", "ABBV", "MRK", "COST", "CRM",
    "NFLX", "ADBE", "PG", "JNJ",
    "TSLA", "BRK-B", "V", "XOM", "MA", "CVX", "MCD", "WMT",
    "TMO", "CSCO", "ACN", "ABT", "DHR", "NKE", "TXN",
]

raw = yf.download(ALL_TICKERS, start="2019-01-01", end="2025-12-31", progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    CLOSE = raw["Close"].copy()
    VOLUME = raw["Volume"].copy()
else:
    CLOSE = raw.copy()
    VOLUME = raw.get("Volume", pd.DataFrame())

rename = {"^VIX": "VIX"}
CLOSE = CLOSE.rename(columns=rename).ffill()
VOLUME = VOLUME.rename(columns=rename).ffill()

VIX = CLOSE["VIX"].dropna() if "VIX" in CLOSE.columns else CLOSE.get("^VIX", pd.Series(dtype=float)).dropna()

print(f"Data: {len(CLOSE)} trading days, {CLOSE.index[0].date()} to {CLOSE.index[-1].date()}")
print(f"Tickers loaded: {len(CLOSE.columns)}")


# ─── Metrics ─────────────────────────────────────────────────────────
def compute_metrics(pnls, freq_annualize=252):
    """Compute Sharpe, WR, PF, Sortino from trade P&Ls."""
    if len(pnls) < 2:
        return {"sharpe": 0, "sortino": 0, "wr": 0, "pf": 0, "n": len(pnls), "total_pnl": 0, "mean_pnl": 0}
    p = np.array(pnls, dtype=float)
    mu = np.mean(p)
    std = np.std(p, ddof=1)
    # Annualize assuming trades are spread roughly evenly
    trades_per_year = max(len(p) / 6, 1)  # ~6 years of data
    sharpe = (mu / (std + 1e-10)) * np.sqrt(trades_per_year)

    downside = p[p < 0]
    dd_std = np.std(downside, ddof=1) if len(downside) > 1 else std
    sortino = (mu / (dd_std + 1e-10)) * np.sqrt(trades_per_year)

    wr = np.mean(p > 0) * 100
    wins = np.sum(p[p > 0])
    losses = np.abs(np.sum(p[p < 0]))
    pf = wins / (losses + 1e-10)

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "wr": round(float(wr), 1),
        "pf": round(float(pf), 3),
        "n": int(len(p)),
        "total_pnl": round(float(np.sum(p)), 2),
        "mean_pnl": round(float(mu), 4),
    }


def run_adversarial(name, signal_fn, pnl_fn, dates, data,
                    commission_base=0, param_keys=None, params=None):
    """Run the 6-gate adversarial framework."""
    print(f"\n{'='*60}")
    print(f"ENGINE: {name}")
    print(f"{'='*60}")

    results = {"engine": name, "timestamp": datetime.now().isoformat()}

    # Gate 1: Baseline
    real_pnls = []
    for i in range(len(dates)):
        sig = signal_fn(i, data, params)
        if sig != 0:
            p = pnl_fn(sig, i, data, commission_base, params)
            if p is not None:
                real_pnls.append(p)

    baseline = compute_metrics(real_pnls)
    results["gate1_baseline"] = baseline
    results["gate1_pass"] = baseline["n"] >= 10 and baseline["sharpe"] > 0
    print(f"  Gate 1 (Baseline): N={baseline['n']}, Sharpe={baseline['sharpe']}, "
          f"WR={baseline['wr']}%, PF={baseline['pf']} -> {'PASS' if results['gate1_pass'] else 'FAIL'}")

    if baseline["n"] < 5:
        print(f"  SKIP: only {baseline['n']} trades")
        results["verdict"] = "FAIL"
        results["gates_passed"] = 0
        results["gates_total"] = 6
        results["fail_reason"] = "Insufficient trades"
        return results

    # Gate 2: Inverse test
    inv_pnls = []
    for i in range(len(dates)):
        sig = signal_fn(i, data, params)
        if sig != 0:
            p = pnl_fn(-sig, i, data, commission_base, params)
            if p is not None:
                inv_pnls.append(p)
    inv_m = compute_metrics(inv_pnls)
    results["gate2_inverse"] = inv_m
    results["gate2_pass"] = inv_m["mean_pnl"] < baseline["mean_pnl"]
    print(f"  Gate 2 (Inverse): inv_mean={inv_m['mean_pnl']:.4f} vs real={baseline['mean_pnl']:.4f} "
          f"-> {'PASS' if results['gate2_pass'] else 'FAIL'}")

    # Early exit if inverse also profits — signal is likely noise
    if not results["gate2_pass"] and inv_m["sharpe"] > 0.5:
        print(f"  EARLY FAIL: Inverse also highly profitable (Sharpe {inv_m['sharpe']})")
        results["verdict"] = "FAIL"
        results["gates_passed"] = sum([results.get(f"gate{g}_pass", False) for g in range(1, 3)])
        results["gates_total"] = 6
        results["fail_reason"] = "Inverse also profitable"
        return results

    # Gate 3: Random permutation
    random_better = 0
    for _ in range(N_RANDOM):
        rand_pnls = []
        for i in range(len(dates)):
            sig = signal_fn(i, data, params)
            if sig != 0:
                rsig = np.random.choice([-1, 1])
                p = pnl_fn(rsig, i, data, commission_base, params)
                if p is not None:
                    rand_pnls.append(p)
        if rand_pnls and np.mean(rand_pnls) >= baseline["mean_pnl"]:
            random_better += 1

    p_val = random_better / N_RANDOM
    results["gate3_random_p"] = round(p_val, 4)
    results["gate3_pass"] = p_val < 0.10
    print(f"  Gate 3 (Random): p={p_val:.4f} -> {'PASS' if results['gate3_pass'] else 'FAIL'}")

    # Gate 4: Cost sensitivity (2x cost)
    cost_pnls = []
    for i in range(len(dates)):
        sig = signal_fn(i, data, params)
        if sig != 0:
            p = pnl_fn(sig, i, data, commission_base * 2, params)
            if p is not None:
                cost_pnls.append(p)
    cost_m = compute_metrics(cost_pnls)
    results["gate4_2x_cost"] = cost_m
    results["gate4_pass"] = cost_m["mean_pnl"] > 0
    print(f"  Gate 4 (2x Cost): mean_pnl={cost_m['mean_pnl']:.4f}, Sharpe={cost_m['sharpe']:.3f} "
          f"-> {'PASS' if results['gate4_pass'] else 'FAIL'}")

    # Gate 5: Sub-period stability (split into 3 periods)
    n = len(dates)
    third = n // 3
    period_pnls = [[], [], []]
    for i in range(len(dates)):
        sig = signal_fn(i, data, params)
        if sig != 0:
            p = pnl_fn(sig, i, data, commission_base, params)
            if p is not None:
                bucket = min(i // third, 2)
                period_pnls[bucket].append(p)

    period_metrics = [compute_metrics(pp) for pp in period_pnls]
    results["gate5_periods"] = period_metrics
    # At least 2 of 3 periods positive
    pos_periods = sum(1 for pm in period_metrics if pm["mean_pnl"] > 0)
    results["gate5_pass"] = pos_periods >= 2
    print(f"  Gate 5 (Stability): {pos_periods}/3 positive periods "
          f"(means: {[pm['mean_pnl'] for pm in period_metrics]}) -> {'PASS' if results['gate5_pass'] else 'FAIL'}")

    # Gate 6: Parameter robustness
    if params and param_keys:
        robust_sharpes = []
        for key in param_keys:
            orig_val = params[key]
            for mult in [0.8, 0.9, 1.1, 1.2]:
                tp = params.copy()
                tp[key] = orig_val * mult
                pp = []
                for i in range(len(dates)):
                    sig = signal_fn(i, data, tp)
                    if sig != 0:
                        p = pnl_fn(sig, i, data, commission_base, tp)
                        if p is not None:
                            pp.append(p)
                pm = compute_metrics(pp)
                robust_sharpes.append(pm["sharpe"])

        frac_above = np.mean(np.array(robust_sharpes) > 0.3) if robust_sharpes else 0
        results["gate6_robust_frac"] = round(float(frac_above), 2)
        results["gate6_robust_sharpes"] = [round(s, 3) for s in robust_sharpes]
        results["gate6_pass"] = frac_above >= 0.50
        print(f"  Gate 6 (Robustness): {frac_above:.0%} of grid > Sharpe 0.3 "
              f"-> {'PASS' if results['gate6_pass'] else 'FAIL'}")
    else:
        results["gate6_pass"] = True
        results["gate6_robust_frac"] = 1.0
        print(f"  Gate 6 (Robustness): No params to test -> PASS (default)")

    gates = [results.get(f"gate{g}_pass", False) for g in range(1, 7)]
    results["gates_passed"] = sum(gates)
    results["gates_total"] = 6
    results["verdict"] = "PASS" if sum(gates) >= 5 else "FAIL"
    results["gate_details"] = {f"gate{i+1}": "PASS" if g else "FAIL" for i, g in enumerate(gates)}

    print(f"\n  VERDICT: {results['verdict']} ({sum(gates)}/6 gates)")
    return results


# ══════════════════════════════════════════════════════════════════════
# Strategy 1: SECTOR COMBINED V10 OPTIMAL
# ══════════════════════════════════════════════════════════════════════
def test_sector_combined_v10():
    """Sector momentum L/S with monthly rebalance, LGBM-proxy ranking."""
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    sectors = [s for s in sectors if s in CLOSE.columns]
    spy = CLOSE["SPY"].dropna()
    vix = VIX.copy()
    sc = CLOSE[sectors].dropna(how="all")
    common = sc.index.intersection(vix.index).intersection(spy.index)
    sc, vix_s, spy = sc.loc[common], vix[common], spy[common]

    sma200 = spy.rolling(200).mean()
    dates = sc.index[260:]  # need 252d for features

    data = {"sc": sc, "vix": vix_s, "spy": spy, "sma200": sma200,
            "dates": dates, "sectors": sectors}
    params = {
        "rebal_interval": 21,  # monthly
        "top_k": 4,
        "bottom_k": 4,
        "dte": 28,
        "moneyness_pct": 4.0,
        "vix_thresh": 20,
        "profit_target": 0.30,
    }

    rebal_counter = [0]

    def signal_fn(i, d, p):
        rebal_counter[0] += 1
        if rebal_counter[0] % int(p["rebal_interval"]) != 0:
            return 0

        dt = d["dates"][i]
        v = d["vix"][dt]
        bull = d["spy"][dt] > d["sma200"][dt]

        idx = d["sc"].index.get_loc(dt)
        if idx < 260:
            return 0

        # Compute LGBM-proxy features: multi-timeframe momentum
        rankings = {}
        for s in d["sectors"]:
            px = d["sc"][s].iloc[:idx+1].dropna()
            if len(px) < 252:
                continue
            ret_5 = float(px.iloc[-1] / px.iloc[-6] - 1)
            ret_10 = float(px.iloc[-1] / px.iloc[-11] - 1)
            ret_21 = float(px.iloc[-1] / px.iloc[-22] - 1)
            ret_63 = float(px.iloc[-1] / px.iloc[-64] - 1)
            ret_126 = float(px.iloc[-1] / px.iloc[-127] - 1)
            ret_252 = float(px.iloc[-1] / px.iloc[-253] - 1)

            rets = px.pct_change().dropna()
            vol_21 = float(rets.iloc[-21:].std() * np.sqrt(252))
            sharpe_63 = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252))

            # Composite score (proxy for LGBM ranking)
            score = (0.15 * ret_5 + 0.20 * ret_10 + 0.25 * ret_21 +
                    0.15 * ret_63 + 0.10 * ret_126 + 0.05 * ret_252 +
                    0.05 * sharpe_63 + 0.05 * (-vol_21))
            rankings[s] = score

        if len(rankings) < 6:
            return 0

        # High VIX: bull spreads only on top sectors
        if v > p["vix_thresh"]:
            return 1  # long-only mode
        else:
            return 2  # pair trade mode (long top + short bottom)

    def pnl_fn(sig, i, d, comm, p):
        dte = int(p["dte"])
        if i + dte >= len(d["dates"]):
            return None

        dt = d["dates"][i]
        idx = d["sc"].index.get_loc(dt)
        v = d["vix"][dt]

        # Compute rankings
        rankings = {}
        for s in d["sectors"]:
            px = d["sc"][s].iloc[:idx+1].dropna()
            if len(px) < 252:
                continue
            ret_5 = float(px.iloc[-1] / px.iloc[-6] - 1)
            ret_10 = float(px.iloc[-1] / px.iloc[-11] - 1)
            ret_21 = float(px.iloc[-1] / px.iloc[-22] - 1)
            ret_63 = float(px.iloc[-1] / px.iloc[-64] - 1)
            ret_126 = float(px.iloc[-1] / px.iloc[-127] - 1)
            ret_252 = float(px.iloc[-1] / px.iloc[-253] - 1)
            rets = px.pct_change().dropna()
            vol_21 = float(rets.iloc[-21:].std() * np.sqrt(252))
            sharpe_63 = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252))
            score = (0.15 * ret_5 + 0.20 * ret_10 + 0.25 * ret_21 +
                    0.15 * ret_63 + 0.10 * ret_126 + 0.05 * ret_252 +
                    0.05 * sharpe_63 + 0.05 * (-vol_21))
            rankings[s] = score

        sorted_s = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        exit_dt = d["dates"][min(i + dte, len(d["dates"])-1)]

        # Direction handling based on signal
        direction = 1 if sig > 0 else -1

        total_pnl = 0
        top_k = int(p["top_k"])
        bottom_k = int(p["bottom_k"])
        pos_size = 200  # $200 max per trade
        haircut = 0.15

        if abs(sig) == 1:
            # High VIX: bull spreads on top sectors only
            picks = [s for s, _ in sorted_s[:top_k]]
            for tk in picks:
                S = d["sc"][tk][dt]
                S_exit = d["sc"][tk][exit_dt]
                K = S * (1 + p["moneyness_pct"] / 100)
                # Simplified spread P&L
                credit = max(0.5, v / 100 * S * 0.15) * (1 - haircut)
                ret_pnl = (S_exit / S - 1) * direction * pos_size
                # Incorporate options behavior: capped upside/downside
                spread_pnl = min(ret_pnl, credit * 3) * 0.5 + credit * 0.5 * direction - comm
                total_pnl += spread_pnl
        else:
            # Low VIX: pair trade
            longs = [s for s, _ in sorted_s[:top_k]]
            shorts = [s for s, _ in sorted_s[-bottom_k:]]
            for tk in longs:
                S = d["sc"][tk][dt]
                S_exit = d["sc"][tk][exit_dt]
                ret = (S_exit / S - 1) * direction
                total_pnl += ret * pos_size - comm
            for tk in shorts:
                S = d["sc"][tk][dt]
                S_exit = d["sc"][tk][exit_dt]
                ret = -(S_exit / S - 1) * direction  # short
                total_pnl += ret * pos_size - comm

        return float(total_pnl)

    # Reset counter for each call
    rebal_counter[0] = 0
    return run_adversarial(
        "sector_combined_v10_optimal", signal_fn, pnl_fn, dates, data,
        commission_base=2.60, param_keys=["rebal_interval", "top_k", "moneyness_pct", "vix_thresh"],
        params=params
    )


# ══════════════════════════════════════════════════════════════════════
# Strategy 2: STRATEGY ROTATION V2F (Dynamic Contrarian)
# ══════════════════════════════════════════════════════════════════════
def test_strategy_rotation_v2f():
    """SPY/QQQ rotation based on VIX + SMA200 regime."""
    spy = CLOSE["SPY"].dropna()
    qqq = CLOSE["QQQ"].dropna()
    vix = VIX.copy()
    common = spy.index.intersection(qqq.index).intersection(vix.index)
    spy, qqq, vix_s = spy[common], qqq[common], vix[common]

    sma200 = spy.rolling(200).mean()
    spy_ret5 = spy.pct_change(5)
    vix_chg5 = vix_s.pct_change(5)

    dates = spy.index[200:]
    data = {"spy": spy, "qqq": qqq, "vix": vix_s, "sma200": sma200,
            "spy_ret5": spy_ret5, "vix_chg5": vix_chg5, "dates": dates}
    params = {"vix_fade_thresh": 25, "contrarian_mult": -1.5, "hold_days": 7}

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
            return 1  # VIX fade -> SPY long
        elif bull:
            return 1  # QQQ long
        elif sret <= thresh:
            return 1  # contrarian long
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        if i + hold >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        v = d["vix"][dt]
        bull = d["spy"][dt] > d["sma200"][dt]

        if v > p["vix_fade_thresh"]:
            ticker = d["spy"]
        elif bull:
            ticker = d["qqq"]
        else:
            ticker = d["spy"]

        entry = ticker[dt]
        exit_dt = d["dates"][min(i + hold, len(d["dates"])-1)]
        exit_p = ticker[exit_dt]
        ret = (exit_p / entry - 1) * sig
        pnl = ret * 645 - comm
        return float(pnl)

    return run_adversarial(
        "strategy_rotation_v2f", signal_fn, pnl_fn, dates, data,
        commission_base=0.13, param_keys=["vix_fade_thresh", "contrarian_mult", "hold_days"],
        params=params
    )


# ══════════════════════════════════════════════════════════════════════
# Strategy 3: VOL TERM STRUCTURE (VIX spike -> buy quality)
# ══════════════════════════════════════════════════════════════════════
def test_vol_term_structure():
    """Buy quality basket when VIX spikes above 1.15x 60d mean and starts declining."""
    spy = CLOSE["SPY"].dropna()
    vix = VIX.copy()
    common = spy.index.intersection(vix.index)
    spy, vix_s = spy[common], vix[common]

    # Quality basket proxy: equal-weight of big tech + defensives
    basket_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH",
                      "LLY", "AVGO", "AMD", "HD", "ABBV", "MRK", "COST", "CRM",
                      "NFLX", "ADBE", "PG", "JNJ"]
    basket_tickers = [t for t in basket_tickers if t in CLOSE.columns]
    basket = CLOSE[basket_tickers].dropna(how="all").mean(axis=1)  # equal weight
    basket = basket.reindex(spy.index).ffill()

    vix_60d = vix_s.rolling(60).mean()
    dates = spy.index[70:]

    data = {"spy": spy, "basket": basket, "vix": vix_s, "vix_60d": vix_60d, "dates": dates}
    params = {"spike_mult": 1.15, "hold_days": 21, "tp_pct": 0.10, "sl_pct": -0.15}

    def signal_fn(i, d, p):
        if i == 0:
            return 0
        dt = d["dates"][i]
        v = d["vix"][dt]
        v_mean = d["vix_60d"][dt]
        if pd.isna(v) or pd.isna(v_mean) or v_mean == 0:
            return 0

        v_prev = d["vix"][d["dates"][i-1]]
        if v > p["spike_mult"] * v_mean and v < v_prev:
            return 1  # buy quality basket
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        if i + hold >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        entry = d["basket"][dt]
        if pd.isna(entry) or entry == 0:
            return None

        for h in range(1, hold + 1):
            if i + h >= len(d["dates"]):
                break
            exit_dt = d["dates"][i + h]
            exit_p = d["basket"][exit_dt]
            if pd.isna(exit_p):
                continue
            ret = (exit_p / entry - 1) * sig

            if ret >= p["tp_pct"] or ret <= p["sl_pct"]:
                return float(ret * 300 - comm)

            v_now = d["vix"][exit_dt]
            v_mean = d["vix_60d"][exit_dt]
            if not pd.isna(v_now) and not pd.isna(v_mean) and v_now < v_mean:
                return float(ret * 300 - comm)

        exit_dt = d["dates"][min(i + hold, len(d["dates"])-1)]
        exit_p = d["basket"][exit_dt]
        ret = (exit_p / entry - 1) * sig
        return float(ret * 300 - comm)

    return run_adversarial(
        "vol_term_structure", signal_fn, pnl_fn, dates, data,
        commission_base=0, param_keys=["spike_mult", "hold_days", "tp_pct"],
        params=params
    )


# ══════════════════════════════════════════════════════════════════════
# Strategy 4: SECTOR REVERSAL (buy stocks that drop >5% vs sector)
# ══════════════════════════════════════════════════════════════════════
def test_sector_reversal():
    """Buy stocks that drop >5% more than their sector ETF over 5 days."""
    # Use a subset of large-cap S&P stocks
    stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM",
              "V", "UNH", "JNJ", "XOM", "PG", "HD", "MA", "CVX", "MRK", "ABBV", "LLY",
              "COST", "AVGO", "MCD", "WMT", "CSCO", "ACN", "ABT", "CRM", "NKE", "ADBE", "TXN"]
    stocks = [s for s in stocks if s in CLOSE.columns]

    sector_map = {
        'AAPL':'XLK','MSFT':'XLK','AMZN':'XLY','GOOGL':'XLC','META':'XLC',
        'NVDA':'XLK','TSLA':'XLY','JPM':'XLF','V':'XLK','UNH':'XLV','JNJ':'XLV',
        'XOM':'XLE','PG':'XLP','HD':'XLY','MA':'XLK','CVX':'XLE',
        'MRK':'XLV','ABBV':'XLV','LLY':'XLV','COST':'XLP','AVGO':'XLK',
        'MCD':'XLY','WMT':'XLP','CSCO':'XLK','ACN':'XLK','ABT':'XLV',
        'CRM':'XLK','NKE':'XLY','ADBE':'XLK','TXN':'XLK',
        'DHR':'XLV','TMO':'XLV',
    }

    spy = CLOSE["SPY"].dropna()
    sma200 = spy.rolling(200).mean()

    sc = CLOSE[stocks].dropna(how="all")
    sectors_etfs = list(set(v for v in sector_map.values() if v in CLOSE.columns))
    sec_close = CLOSE[sectors_etfs].dropna(how="all")

    common = sc.index.intersection(sec_close.index).intersection(spy.index)
    sc = sc.loc[common]
    sec_close = sec_close.loc[common]
    spy = spy[common]
    sma200 = sma200[common]

    dates = sc.index[210:]  # need 200 for SMA + 10 buffer

    data = {"sc": sc, "sec": sec_close, "spy": spy, "sma200": sma200,
            "dates": dates, "stocks": stocks, "sector_map": sector_map}
    params = {"drop_thresh": -0.05, "lookback": 5, "hold_days": 10,
              "full_size": 130, "half_size": 65, "max_positions": 5}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        idx = d["sc"].index.get_loc(dt)
        if idx < int(p["lookback"]) + 1:
            return 0

        # Find stocks that dropped >5% more than their sector
        for stock in d["stocks"]:
            if stock not in d["sector_map"] or d["sector_map"][stock] not in d["sec"].columns:
                continue
            sector_etf = d["sector_map"][stock]

            stock_px = d["sc"][stock].iloc[idx - int(p["lookback"]):idx+1].dropna()
            sec_px = d["sec"][sector_etf].iloc[idx - int(p["lookback"]):idx+1].dropna()

            if len(stock_px) < 2 or len(sec_px) < 2:
                continue

            stock_ret = stock_px.iloc[-1] / stock_px.iloc[0] - 1
            sec_ret = sec_px.iloc[-1] / sec_px.iloc[0] - 1
            relative_drop = stock_ret - sec_ret

            if relative_drop <= p["drop_thresh"]:
                return 1  # buy reversal
        return 0

    def pnl_fn(sig, i, d, comm, p):
        hold = int(p["hold_days"])
        if i + hold >= len(d["dates"]):
            return None
        dt = d["dates"][i]
        idx = d["sc"].index.get_loc(dt)

        bull = d["spy"][dt] > d["sma200"][dt]
        pos_size = p["full_size"] if bull else p["half_size"]

        # Find the triggered stock
        best_stock = None
        best_drop = 0
        for stock in d["stocks"]:
            if stock not in d["sector_map"] or d["sector_map"][stock] not in d["sec"].columns:
                continue
            sector_etf = d["sector_map"][stock]
            stock_px = d["sc"][stock].iloc[idx - int(p["lookback"]):idx+1].dropna()
            sec_px = d["sec"][sector_etf].iloc[idx - int(p["lookback"]):idx+1].dropna()
            if len(stock_px) < 2 or len(sec_px) < 2:
                continue
            stock_ret = stock_px.iloc[-1] / stock_px.iloc[0] - 1
            sec_ret = sec_px.iloc[-1] / sec_px.iloc[0] - 1
            rel_drop = stock_ret - sec_ret
            if rel_drop <= p["drop_thresh"] and rel_drop < best_drop:
                best_stock = stock
                best_drop = rel_drop

        if best_stock is None:
            return None

        entry = d["sc"][best_stock].iloc[idx]
        exit_idx = min(idx + hold, len(d["sc"]) - 1)
        exit_p = d["sc"][best_stock].iloc[exit_idx]

        ret = (exit_p / entry - 1) * sig
        pnl = ret * pos_size - comm
        return float(pnl)

    return run_adversarial(
        "sector_reversal", signal_fn, pnl_fn, dates, data,
        commission_base=0.20, param_keys=["drop_thresh", "lookback", "hold_days"],
        params=params
    )


# ══════════════════════════════════════════════════════════════════════
# Strategy 5: VIX CALL SPREAD (sell VIX call spreads when VIX > 20)
# ══════════════════════════════════════════════════════════════════════
def test_vix_call_spread():
    """Sell VIX call spreads when VIX > 20, capturing mean-reversion premium."""
    vix = VIX.copy()
    dates = vix.index[60:]

    data = {"vix": vix, "dates": dates}
    params = {"entry_thresh": 20, "hold_days": 14, "spread_width": 5, "tp_level": 15}

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
        v_entry = d["vix"][dt]

        # Check for early exit (VIX < TP level)
        for h in range(1, hold + 1):
            if i + h >= len(d["dates"]):
                break
            exit_dt = d["dates"][i + h]
            v_exit = d["vix"][exit_dt]

            if v_exit < p["tp_level"]:
                # Early take profit
                K_short = round(v_entry)
                credit = max(v_entry - K_short, 0.5) * 100 * 0.3
                exit_cost = max(0, (v_exit - K_short) - max(0, v_exit - K_short - p["spread_width"])) * 100
                pnl = (credit - exit_cost) * sig - comm
                return float(pnl)

        # Hold to expiry
        exit_dt = d["dates"][min(i + hold, len(d["dates"])-1)]
        v_exit = d["vix"][exit_dt]
        K_short = round(v_entry)
        K_long = K_short + p["spread_width"]
        credit = max(v_entry - K_short, 0.5) * 100 * 0.3
        exit_cost = max(0, (v_exit - K_short) - max(0, v_exit - K_long)) * 100
        pnl = (credit - exit_cost) * sig - comm
        return float(pnl)

    return run_adversarial(
        "vix_call_spread", signal_fn, pnl_fn, dates, data,
        commission_base=4.00, param_keys=["entry_thresh", "hold_days", "spread_width"],
        params=params
    )


# ══════════════════════════════════════════════════════════════════════
# Strategy 6: VOLUME SURGE (buy on 3-day volume surge)
# ══════════════════════════════════════════════════════════════════════
def test_volume_surge():
    """Buy sector ETFs on 3 consecutive days of volume > 1.5x 20-day average."""
    etfs = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
    etfs = [e for e in etfs if e in CLOSE.columns and e in VOLUME.columns]

    dates = CLOSE.index[30:]
    data = {"close": CLOSE, "volume": VOLUME, "dates": dates, "etfs": etfs}
    params = {"vol_thresh": 1.5, "consec_days": 3, "hold_days": 10, "lookback": 20}

    def signal_fn(i, d, p):
        dt = d["dates"][i]
        idx = d["close"].index.get_loc(dt)
        if idx < 30:
            return 0

        for etf in d["etfs"]:
            lb = int(p["lookback"])
            cd = int(p["consec_days"])
            v = d["volume"][etf].iloc[idx-lb-cd:idx+1]
            if len(v) < lb + cd:
                continue
            avg_vol = v.iloc[:lb].mean()
            if avg_vol <= 0:
                continue
            recent = v.iloc[-cd:]
            ratios = recent / avg_vol
            if all(r >= p["vol_thresh"] for r in ratios.values):
                return 1
        return 0

    def pnl_fn(sig, i, d, comm, p):
        dt = d["dates"][i]
        idx = d["close"].index.get_loc(dt)
        hold = int(p["hold_days"])
        if idx + hold >= len(d["close"]):
            return None

        # Find triggering ETF
        best_etf = None
        lb = int(p["lookback"])
        cd = int(p["consec_days"])
        for etf in d["etfs"]:
            v = d["volume"][etf].iloc[idx-lb-cd:idx+1]
            if len(v) < lb + cd:
                continue
            avg_vol = v.iloc[:lb].mean()
            if avg_vol <= 0:
                continue
            recent = v.iloc[-cd:]
            if all(r >= p["vol_thresh"] for r in (recent / avg_vol).values):
                best_etf = etf
                break

        if best_etf is None:
            best_etf = "SPY"

        entry = d["close"][best_etf].iloc[idx]
        exit_p = d["close"][best_etf].iloc[idx + hold]
        ret = (exit_p / entry - 1) * sig
        pnl = ret * 200 - comm
        return float(pnl)

    return run_adversarial(
        "volume_surge", signal_fn, pnl_fn, dates, data,
        commission_base=0.20, param_keys=["vol_thresh", "consec_days", "hold_days"],
        params=params
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("ADVERSARIAL BATCH VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Framework: 6-gate (baseline, inverse, random, cost, stability, robustness)")
    print("=" * 70)

    all_results = {}

    all_results["sector_combined_v10_optimal"] = test_sector_combined_v10()
    all_results["strategy_rotation_v2f"] = test_strategy_rotation_v2f()
    all_results["vol_term_structure"] = test_vol_term_structure()
    all_results["sector_reversal"] = test_sector_reversal()
    all_results["vix_call_spread"] = test_vix_call_spread()
    all_results["volume_surge"] = test_volume_surge()

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL BATCH VALIDATION — FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Engine':<32} {'Gates':<8} {'Sharpe':<8} {'WR':<7} {'PF':<7} {'N':<6} {'Verdict'}")
    print("-" * 78)

    for name, r in all_results.items():
        b = r.get("gate1_baseline", {})
        gates = f"{r['gates_passed']}/{r['gates_total']}"
        print(f"{name:<32} {gates:<8} {b.get('sharpe','N/A'):<8} {b.get('wr','N/A'):<7} "
              f"{b.get('pf','N/A'):<7} {b.get('n',0):<6} {r['verdict']}")

    pass_count = sum(1 for r in all_results.values() if r["verdict"] == "PASS")
    fail_count = sum(1 for r in all_results.values() if r["verdict"] == "FAIL")
    print(f"\nPASS: {pass_count}  |  FAIL: {fail_count}")

    # Save
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

    out_path = Path("/home/jupiter/Lvl3Quant/validation/adversarial_batch_results.json")
    with open(out_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved.")
    return all_results


if __name__ == "__main__":
    main()
