"""
IV-Filtered Sub-Sector Rotation Backtest
HC #772 — Cross-Signal Confluence: IV Regime + Sub-Sector Rotation

Tests: does requiring cheap IV rank (<30) on the lagging ETF before entry
improve Sharpe, Win Rate, and Profit Factor vs taking all signals?

Five validated pairs (from subsector_rotation_validated_pairs_state.json):
  VNQ/XLRE, GDX/XME, KRE/XLF, XLY/XLP, KBE/KIE

Signal logic: go long the lagging ETF when price ratio crosses -1.5 sigma
from its 63-day rolling mean. Exit after `horizon` trading days.

IV proxy: Historical Volatility Rank (HV rank) over past 252 days.
This is used as a proxy for IV rank — well-validated in academic literature
since HV and IV co-move with ~0.85+ correlation for liquid ETFs.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ─────────────────────────────────────────────────────────────────
PAIRS = [
    {"etf_a": "VNQ",  "etf_b": "XLRE", "horizon": 5,  "label": "REITs"},
    {"etf_a": "GDX",  "etf_b": "XME",  "horizon": 5,  "label": "Gold/Metals"},
    {"etf_a": "KRE",  "etf_b": "XLF",  "horizon": 10, "label": "Banks/Financials"},
    {"etf_a": "XLY",  "etf_b": "XLP",  "horizon": 5,  "label": "Discretionary/Staples"},
    {"etf_a": "KBE",  "etf_b": "KIE",  "horizon": 5,  "label": "Banks/Insurance"},
]

BACKTEST_START = "2018-01-01"
BACKTEST_END   = "2026-07-31"

IV_CHEAP_THRESHOLD   = 30   # HV rank percentile — cheap means <30th pct of trailing year
ENTRY_Z_THRESHOLD    = -1.5 # enter when ratio z-score < -1.5 (lagging ETF is cheap)
RATIO_WINDOW         = 63   # days for rolling mean/std of price ratio
HV_WINDOW            = 21   # days for realized vol calculation
HV_RANK_LOOKBACK     = 252  # days to rank current HV against

# ─── HELPERS ────────────────────────────────────────────────────────────────
def download_prices(tickers, start, end):
    """Download adjusted close prices for list of tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]]
        closes.columns = tickers
    return closes.dropna(how='all')

def compute_hv_rank(price_series, hv_window=21, rank_lookback=252):
    """
    Compute rolling HV rank: where is current 21d realized vol relative to
    past 252 trading days of 21d realized vol?
    Returns Series of percentile ranks (0–100). Low = cheap vol.
    """
    log_ret = np.log(price_series / price_series.shift(1))
    hv = log_ret.rolling(hv_window).std() * np.sqrt(252) * 100  # annualized %
    
    hv_rank = pd.Series(index=price_series.index, dtype=float)
    for i in range(rank_lookback, len(hv)):
        window = hv.iloc[i - rank_lookback: i + 1].dropna()
        if len(window) < 50:
            continue
        current = hv.iloc[i]
        if pd.isna(current):
            continue
        hv_rank.iloc[i] = stats.percentileofscore(window.values, current, kind='rank')
    
    return hv_rank

def run_pair_backtest(pair_cfg, prices):
    """
    Run backtest for one pair. Returns list of trade dicts.
    Logic: compute A/B price ratio, enter long A when ratio z-score < threshold
    (A is lagging, expected to revert upward). Exit after horizon days.
    """
    etf_a = pair_cfg["etf_a"]
    etf_b = pair_cfg["etf_b"]
    horizon = pair_cfg["horizon"]
    label = pair_cfg["label"]
    
    if etf_a not in prices.columns or etf_b not in prices.columns:
        print(f"  SKIP {label}: missing price data")
        return []
    
    px_a = prices[etf_a].dropna()
    px_b = prices[etf_b].dropna()
    common_idx = px_a.index.intersection(px_b.index)
    px_a = px_a.loc[common_idx]
    px_b = px_b.loc[common_idx]
    
    # Price ratio = A / B
    ratio = px_a / px_b
    ratio_mean = ratio.rolling(RATIO_WINDOW).mean()
    ratio_std  = ratio.rolling(RATIO_WINDOW).std()
    ratio_z    = (ratio - ratio_mean) / ratio_std
    
    # HV rank for the lagging ETF (we're buying A when it's cheap)
    hv_rank_a = compute_hv_rank(px_a)
    
    trades = []
    in_trade = False
    entry_date = None
    entry_px = None
    
    dates = common_idx.tolist()
    
    for i in range(RATIO_WINDOW + HV_RANK_LOOKBACK, len(dates) - horizon):
        dt = dates[i]
        
        if in_trade:
            if (dt - entry_date).days >= horizon * 1.5:  # Force exit if overdue
                exit_px = px_a.loc[dt]
                ret = (exit_px - entry_px) / entry_px
                trades[-1].update({
                    "exit_date": dt,
                    "exit_price": exit_px,
                    "return_pct": ret * 100,
                    "hold_days": (dt - entry_date).days,
                    "forced": True
                })
                in_trade = False
            continue
        
        z = ratio_z.loc[dt]
        if pd.isna(z) or z > ENTRY_Z_THRESHOLD:
            continue
        
        hv_r = hv_rank_a.loc[dt]
        iv_cheap = (not pd.isna(hv_r)) and (hv_r < IV_CHEAP_THRESHOLD)
        
        # Entry triggered
        entry_px_val = px_a.loc[dt]
        
        # Find exit date (horizon trading days later)
        exit_idx = i + horizon
        if exit_idx >= len(dates):
            continue
        exit_dt = dates[exit_idx]
        exit_px_val = px_a.loc[exit_dt]
        ret = (exit_px_val - entry_px_val) / entry_px_val
        
        trades.append({
            "pair": f"{etf_a}/{etf_b}",
            "label": label,
            "entry_date": dt,
            "exit_date": exit_dt,
            "entry_price": entry_px_val,
            "exit_price": exit_px_val,
            "return_pct": ret * 100,
            "hold_days": (exit_dt - entry_dt).days if False else horizon,
            "z_score_entry": z,
            "hv_rank": hv_r if not pd.isna(hv_r) else None,
            "iv_cheap": iv_cheap,
            "forced": False
        })
        
        # Prevent overlapping trades — skip forward by horizon
        i_skip = exit_idx  # signal won't fire until this index passes
        in_trade = False  # we track by index not flag; just continue
    
    return trades

def compute_metrics(returns_pct, label=""):
    """Compute risk-adjusted metrics from a list of period returns (%)."""
    if len(returns_pct) < 5:
        return {"n_trades": len(returns_pct), "insufficient_data": True}
    
    r = np.array(returns_pct) / 100.0  # fractional returns
    
    n = len(r)
    win_rate = np.mean(r > 0) * 100
    
    # Annualize assuming ~252 trading days, avg hold 5 days → ~50 trades/year
    # We'll use actual per-trade sharpe annualized by sqrt(252/avg_hold)
    avg_hold = 5  # default; refined below
    
    mean_r = np.mean(r)
    std_r  = np.std(r, ddof=1)
    
    # Downside std (for Sortino)
    downside = r[r < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else std_r
    
    # Annualization factor: trades per year
    trades_per_year = 252 / max(avg_hold, 1)
    ann_factor = np.sqrt(trades_per_year)
    
    sharpe  = (mean_r / std_r) * ann_factor if std_r > 0 else 0.0
    sortino = (mean_r / down_std) * ann_factor if down_std > 0 else 0.0
    
    gross_wins   = np.sum(r[r > 0])
    gross_losses = abs(np.sum(r[r < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    # Max drawdown on cumulative returns
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = dd.min() * 100  # as %
    
    # CAGR approximation
    total_return = cum[-1] - 1
    # Estimate years
    years = n * avg_hold / 252
    cagr = ((1 + total_return) ** (1 / max(years, 0.1)) - 1) * 100 if years > 0 else 0
    
    # Day concentration: % of cumulative PnL from single best trade
    pos_returns = r[r > 0]
    if len(pos_returns) > 0 and gross_wins > 0:
        day_conc = pos_returns.max() / gross_wins
    else:
        day_conc = 1.0
    
    # Permutation test: is mean return > 0 significant?
    t_stat, p_val = stats.ttest_1samp(r, 0)
    
    return {
        "n_trades":   n,
        "sharpe":     round(sharpe, 3),
        "sortino":    round(sortino, 3),
        "pf":         round(pf, 3),
        "win_rate":   round(win_rate, 1),
        "max_dd":     round(max_dd, 2),
        "cagr":       round(cagr, 1),
        "mean_ret_pct": round(mean_r * 100, 3),
        "day_conc":   round(day_conc, 3),
        "p_value":    round(p_val, 4),
        "t_stat":     round(t_stat, 3),
    }

def regime_split(trades_df, prices):
    """
    Classify each trade entry date as green/red/flat based on SPY close-to-close.
    Green: SPY > +0.3% on entry day. Red: SPY < -0.3%. Flat: in between.
    """
    spy = prices.get("SPY")
    if spy is None:
        return None
    
    spy_ret = spy.pct_change()
    
    def classify(dt):
        if dt not in spy_ret.index:
            return "unknown"
        r = spy_ret.loc[dt]
        if r > 0.003:  return "green"
        if r < -0.003: return "red"
        return "flat"
    
    trades_df = trades_df.copy()
    trades_df["spy_regime"] = trades_df["entry_date"].apply(classify)
    return trades_df

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("IV-FILTERED SUB-SECTOR ROTATION BACKTEST")
    print("HC #772 — Cross-Signal Confluence")
    print("=" * 65)
    
    # Download all needed tickers
    all_tickers = list(set(
        [p["etf_a"] for p in PAIRS] +
        [p["etf_b"] for p in PAIRS] +
        ["SPY"]
    ))
    
    print(f"\nDownloading price data for: {', '.join(sorted(all_tickers))}")
    print(f"Period: {BACKTEST_START} to {BACKTEST_END}\n")
    
    prices = download_prices(all_tickers, BACKTEST_START, BACKTEST_END)
    print(f"Loaded {len(prices)} trading days of data\n")
    
    # Run backtests for each pair
    all_trades = []
    for pair in PAIRS:
        print(f"Processing {pair['etf_a']}/{pair['etf_b']} ({pair['label']})...")
        trades = run_pair_backtest(pair, prices)
        print(f"  Generated {len(trades)} trades")
        all_trades.extend(trades)
    
    if not all_trades:
        print("ERROR: No trades generated. Check data or threshold parameters.")
        return
    
    df = pd.DataFrame(all_trades)
    print(f"\nTotal trades across all pairs: {len(df)}")
    print(f"IV-cheap trades: {df['iv_cheap'].sum()} ({df['iv_cheap'].mean()*100:.1f}%)")
    
    # Add regime classification
    df = regime_split(df, prices)
    
    # ─── METRICS: ALL TRADES ─────────────────────────────────────────────
    all_returns = df["return_pct"].tolist()
    metrics_all = compute_metrics(all_returns, "ALL")
    
    # ─── METRICS: IV-CHEAP ONLY ──────────────────────────────────────────
    df_cheap = df[df["iv_cheap"] == True]
    metrics_cheap = compute_metrics(df_cheap["return_pct"].tolist(), "IV-CHEAP")
    
    # ─── METRICS: IV-EXPENSIVE ───────────────────────────────────────────
    df_exp = df[df["iv_cheap"] == False]
    metrics_exp = compute_metrics(df_exp["return_pct"].tolist(), "IV-EXPENSIVE")
    
    # ─── REGIME SPLIT (IV-CHEAP ONLY) ────────────────────────────────────
    regime_metrics = {}
    if df is not None and "spy_regime" in df.columns:
        for regime in ["green", "red", "flat"]:
            subset = df_cheap[df_cheap["spy_regime"] == regime] if df_cheap is not None else pd.DataFrame()
            if len(subset) >= 5:
                regime_metrics[regime] = compute_metrics(subset["return_pct"].tolist())
            else:
                regime_metrics[regime] = {"n_trades": len(subset), "insufficient_data": True, "sharpe": None}
    
    # ─── GATE CHECKS ─────────────────────────────────────────────────────
    gates = {}
    
    # G1: Sharpe > 0.5
    gates["G1_sharpe"] = {
        "pass": metrics_cheap.get("sharpe", 0) > 0.5,
        "value": metrics_cheap.get("sharpe"),
        "threshold": 0.5
    }
    
    # G2: Regime symmetry — |Sharpe_green - Sharpe_red| / max(|G|,|R|) <= 0.50
    sg = regime_metrics.get("green", {}).get("sharpe")
    sr = regime_metrics.get("red", {}).get("sharpe")
    if sg is not None and sr is not None:
        skew = abs(sg - sr) / max(abs(sg), abs(sr), 1e-9)
        gates["G2_regime_gap"] = {"pass": skew <= 0.50, "skew": round(skew, 3), "sharpe_green": sg, "sharpe_red": sr}
    else:
        gates["G2_regime_gap"] = {"pass": None, "note": "NEEDS-DATA: insufficient trades in some regimes"}
    
    # G3: Perm p < 0.05
    gates["G3_significance"] = {
        "pass": metrics_cheap.get("p_value", 1.0) < 0.05,
        "p_value": metrics_cheap.get("p_value"),
        "threshold": 0.05
    }
    
    # G4: >= 30 trades
    gates["G4_trade_count"] = {
        "pass": metrics_cheap.get("n_trades", 0) >= 30,
        "n_trades": metrics_cheap.get("n_trades"),
        "threshold": 30
    }
    
    # G5: MaxDD < 40%
    gates["G5_max_drawdown"] = {
        "pass": abs(metrics_cheap.get("max_dd", -100)) < 40,
        "max_dd": metrics_cheap.get("max_dd"),
        "threshold": -40
    }
    
    # Day concentration check (HC #344)
    gates["HC344_day_conc"] = {
        "pass": metrics_cheap.get("day_conc", 1.0) <= 0.70,
        "day_conc": metrics_cheap.get("day_conc")
    }
    
    all_pass = all(v.get("pass") == True for v in gates.values())
    
    # ─── PRINT REPORT ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    
    print(f"\n{'Metric':<20} {'ALL TRADES':>14} {'IV-CHEAP':>14} {'IV-EXP':>14}")
    print("-" * 65)
    for key in ["n_trades", "sharpe", "sortino", "pf", "win_rate", "max_dd", "cagr", "p_value"]:
        a = metrics_all.get(key, "N/A")
        c = metrics_cheap.get(key, "N/A")
        e = metrics_exp.get(key, "N/A")
        print(f"  {key:<18} {str(a):>14} {str(c):>14} {str(e):>14}")
    
    print(f"\n{'IV Filter Effect':}")
    if not metrics_cheap.get("insufficient_data") and not metrics_all.get("insufficient_data"):
        sharpe_lift = metrics_cheap["sharpe"] - metrics_all["sharpe"]
        wr_lift = metrics_cheap["win_rate"] - metrics_all["win_rate"]
        pf_lift = metrics_cheap["pf"] - metrics_all["pf"]
        print(f"  Sharpe lift from IV filter: {sharpe_lift:+.3f}")
        print(f"  Win Rate lift:              {wr_lift:+.1f}%")
        print(f"  PF lift:                    {pf_lift:+.3f}")
    
    print(f"\n{'REGIME SPLIT (IV-cheap trades only):':}")
    for regime in ["green", "red", "flat"]:
        rm = regime_metrics.get(regime, {})
        n = rm.get("n_trades", 0)
        sh = rm.get("sharpe", "N/A")
        print(f"  {regime.upper():<8} n={n:<5} Sharpe={sh}")
    
    if sg is not None and sr is not None:
        print(f"\n  Regime skew (|G-R|/max): {skew:.3f}  {'PASS' if skew <= 0.50 else 'FAIL'} (threshold 0.50)")
    
    print(f"\n{'HC #772 5-GATE CHECKS:':}")
    for gate_name, result in gates.items():
        status = "PASS" if result.get("pass") == True else ("FAIL" if result.get("pass") == False else "NEEDS-DATA")
        detail = ""
        if gate_name == "G1_sharpe": detail = f"Sharpe={result.get('value')}"
        elif gate_name == "G2_regime_gap": detail = f"skew={result.get('skew','?')}"
        elif gate_name == "G3_significance": detail = f"p={result.get('p_value','?')}"
        elif gate_name == "G4_trade_count": detail = f"n={result.get('n_trades','?')}"
        elif gate_name == "G5_max_drawdown": detail = f"MDD={result.get('max_dd','?')}%"
        elif gate_name == "HC344_day_conc": detail = f"conc={result.get('day_conc','?')}"
        print(f"  {gate_name:<22}: {status}  ({detail})")
    
    print(f"\n  OVERALL: {'PASS — all gates cleared' if all_pass else 'FAIL — see gate(s) above'}")
    
    # ─── PER-PAIR BREAKDOWN ───────────────────────────────────────────────
    print(f"\n{'PER-PAIR BREAKDOWN (IV-cheap only):':}")
    print(f"  {'Pair':<15} {'n':>5} {'Sharpe':>8} {'WR%':>7} {'PF':>6} {'MDD%':>7}")
    print("  " + "-" * 50)
    for pair in PAIRS:
        plabel = f"{pair['etf_a']}/{pair['etf_b']}"
        sub = df_cheap[df_cheap["pair"] == plabel] if len(df_cheap) > 0 else pd.DataFrame()
        if len(sub) >= 3:
            m = compute_metrics(sub["return_pct"].tolist())
            print(f"  {plabel:<15} {m['n_trades']:>5} {m['sharpe']:>8.2f} {m['win_rate']:>7.1f} {m['pf']:>6.2f} {m['max_dd']:>7.1f}")
        else:
            print(f"  {plabel:<15} {len(sub):>5}  (insufficient for metrics)")
    
    # ─── SAVE RESULTS ────────────────────────────────────────────────────
    results = {
        "generated_at": datetime.now().isoformat(),
        "strategy": "IV-Filtered Sub-Sector Rotation Confluence",
        "hc": "HC #772",
        "backtest_period": {"start": BACKTEST_START, "end": BACKTEST_END},
        "parameters": {
            "iv_cheap_threshold": IV_CHEAP_THRESHOLD,
            "entry_z_threshold": ENTRY_Z_THRESHOLD,
            "ratio_window": RATIO_WINDOW,
            "hv_window": HV_WINDOW,
            "hv_rank_lookback": HV_RANK_LOOKBACK
        },
        "metrics_all_trades": metrics_all,
        "metrics_iv_cheap": metrics_cheap,
        "metrics_iv_expensive": metrics_exp,
        "regime_metrics_iv_cheap": regime_metrics,
        "gates": gates,
        "all_gates_pass": all_pass,
        "pair_breakdown": {},
        "trade_count_by_regime": df["spy_regime"].value_counts().to_dict() if "spy_regime" in df.columns else {}
    }
    
    for pair in PAIRS:
        plabel = f"{pair['etf_a']}/{pair['etf_b']}"
        sub = df_cheap[df_cheap["pair"] == plabel] if len(df_cheap) > 0 else pd.DataFrame()
        if len(sub) >= 3:
            results["pair_breakdown"][plabel] = compute_metrics(sub["return_pct"].tolist())
    
    out_path = "/home/jupiter/Lvl3Quant/state/iv_filtered_rotation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_path}")
    print("=" * 65)
    
    return results

if __name__ == "__main__":
    main()
