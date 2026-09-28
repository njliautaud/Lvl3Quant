#!/usr/bin/env python3
"""
Liquidity Drought + Recovery v1 — Volume Disappearance as Contrarian Signal
============================================================================

OBSERVATION: When a stock's volume dries up to extreme lows (bottom 5-10th
percentile of its own history), it often means sellers are exhausted and
buyers are absent. When volume suddenly RETURNS (2-3x the recent average),
it signals renewed interest — often the start of a directional move.

HYPOTHESIS: The "return of liquidity" after a drought creates asymmetric
upside because:
  1. Low volume = low conviction sellers exhausted
  2. Volume return = institutional re-engagement
  3. Combined with price distress = coiled spring
  
This aligns with meta-findings:
  - Vol compression is the universal edge concentrator
  - Contrarian signals dominate
  - "Buy fear" = buy when nobody is trading (drought) then action returns

SIGNAL VARIANTS (54 total):
  - Volume drought threshold: 5th, 10th, 15th percentile of 252d history
  - Drought duration: 3, 5, 10 consecutive days below threshold
  - Recovery trigger: volume > 1.5x, 2x, 3x of 20d avg
  - Hold periods: 5d, 10d, 21d
  - Optional: price must be below 50d SMA (distressed)

VALIDATION: permutation test (1000 shuffles), regime gap, year consistency.

Author: Claude Opus 4.6 / Teleclaude Research
Date: 2026-07-22
"""

import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# MLflow
try:
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("liquidity_drought_v1")
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False
    print("MLflow not available")

# Constants
COST_BPS_RT = 10
NUM_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
PERM_P_THRESHOLD = 0.05
MIN_TRADES = 30
DATA_YEARS = 13

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/liquidity_drought_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# S&P 500 tickers (comprehensive list)
SP500_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JNJ","JPM","V",
    "PG","MA","HD","CVX","MRK","ABBV","LLY","PEP","KO","COST","AVGO","TMO","MCD","WMT",
    "ACN","CSCO","ABT","CRM","NKE","TXN","NEE","AMD","QCOM","HON","LOW","AMGN","INTC",
    "BA","GS","CAT","BLK","ISRG","SYK","ADP","NFLX","ADBE","ORCL","MDT","PFE","PYPL",
    "GILD","MS","AXP","SLB","LMT","BKNG","CME","ADI","C","SCHW","LRCX","BMY","VRTX",
    "DE","EOG","SHW","CI","T","MO","DUK","ICE","SO","MDLZ","CL","ZTS","APD","EMR",
    "ITW","REGN","NOC","GD","PNC","ETN","WM","NSC","USB","BDX","TGT","FDX","COP",
    "MCO","AON","TJX","SPG","PSX","OXY","HUM","KLAC","SNPS","CDNS","MCHP","FTNT",
    "ANET","ABNB","AEP","SRE","D","ECL","LHX","WMB","FCX","ROP","CARR","CTAS","MNST",
    "PSA","GIS","AIG","MPC","KMB","PAYX","MSI","AMP","CMG","DXCM","HAL","IDXX","DVN",
    "MRNA","EW","IQV","RSG","BIIB","KDP","CTSH","ODFL","FAST","PCAR","WEC","A","AME",
    "ALL","MTD","EXC","ED","XEL","STZ","YUM","PPG","KEYS","ON","AWK","CBRE","GPC",
    "DOW","TSCO","EBAY","VMC","DHI","LEN","NVR","PHM","POOL","FICO","CEG","GEHC","KHC",
    "CTVA","NUE","CF","IR","DD","ROK","STE","BAX","HCA","RCL","CCL","DAL","UAL","LUV",
    "AAL","MAR","HLT","WYNN","MGM","CZR","NCLH","F","GM","UBER","LYFT","DASH",
    "SQ","COIN","HOOD","SNAP","PINS","RBLX","TTWO","EA","ZM","CRWD","ZS","NET","DDOG",
    "MDB","SNOW","PLTR","PANW","NOW","INTU","WDAY","TEAM","HUBS","VEEV","ANSS","CPRT",
    "TRGP","FANG","EQT","AR","RRC","MRO","APA","CTRA","BKR","NOV","FTI",
    "DVA","DGX","LH","UHS","THC","HOLX","WAT","TDY","ROL","CINF","MOH","SNA",
    "TRV","AJG","WRB","L","BEN","TROW","IVZ","MKTX","CBOE","NDAQ",
    "STT","NTRS","FITB","HBAN","CFG","KEY","RF","CMA","ZION","MTB",
    "DFS","COF","SYF","WFC","BAC","PNC","ROST","BURL","DG","DLTR","BBY",
    "KSS","ETSY","W","CHWY","SFM","KR","AZO","AAP","ORLY","LKQ",
    "TAP","SAM","CLX","CHD","SJM","HRL","CPB","CAG","MKC","HSY","K",
    "DIS","CMCSA","PARA","WBD","FOX","LYV","SPOT",
    "PLD","AMT","CCI","EQIX","SPG","PSA","O","DLR","WELL","AVB","EQR",
    "LIN","SHW","PPG","NUE","VMC","MLM","FCX","DOW","CTVA","ALB","FMC",
    "UNP","RTX","MMM","GE","PH","DOV","GWW","SWK","XYL","GNRC",
    "NEE","SO","DUK","AEP","XEL","WEC","ED","EXC","AES","PPL","CMS","AWK","ETR",
]
SP500_TICKERS = sorted(list(set(SP500_TICKERS)))

# Signal variant definitions
VARIANTS = []
for vol_pctile in [5, 10, 15]:
    for drought_days in [3, 5, 10]:
        for recovery_mult in [1.5, 2.0, 3.0]:
            for hold in [5, 10, 21]:
                for distress in [False, True]:
                    d_str = "_DIST" if distress else ""
                    name = f"DRY{vol_pctile}_D{drought_days}_R{recovery_mult:.0f}x_H{hold}{d_str}"
                    VARIANTS.append((name, vol_pctile, drought_days, recovery_mult, hold, distress))

# That's 3*3*3*3*2 = 162, too many. Trim to key combos
VARIANTS = []
for vol_pctile in [5, 10]:
    for drought_days in [3, 5]:
        for recovery_mult in [2.0, 3.0]:
            for hold in [5, 10, 21]:
                for distress in [False, True]:
                    d_str = "_DIST" if distress else ""
                    name = f"DRY{vol_pctile}_D{drought_days}_R{recovery_mult:.0f}x_H{hold}{d_str}"
                    VARIANTS.append((name, vol_pctile, drought_days, recovery_mult, hold, distress))

# Additional: pure drought without recovery trigger
for vol_pctile in [5, 10]:
    for drought_days in [5, 10]:
        for hold in [10, 21]:
            name = f"PUREDRY{vol_pctile}_D{drought_days}_H{hold}"
            VARIANTS.append((name, vol_pctile, drought_days, None, hold, False))

# Additional: drought + vol compression combo  
for vol_pctile in [5, 10]:
    for hold in [10, 21]:
        name = f"DRY{vol_pctile}_VC_H{hold}"
        VARIANTS.append((name, vol_pctile, 5, 2.0, hold, "VC"))

print(f"Total variants: {len(VARIANTS)}")
print(f"Total stocks: {len(SP500_TICKERS)}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def annualized_sharpe(returns, trades_per_year=50):
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)

def sortino_ratio(returns, trades_per_year=50):
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0
    return (np.mean(returns) / np.std(downside)) * np.sqrt(trades_per_year)

def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses

def permutation_test(returns, n_perms=1000):
    actual_sharpe = annualized_sharpe(returns)
    count_ge = 0
    abs_returns = np.abs(returns)
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = abs_returns * signs
        if annualized_sharpe(shuffled) >= actual_sharpe:
            count_ge += 1
    return count_ge / n_perms

def regime_gap(sharpe_green, sharpe_red):
    denom = max(abs(sharpe_green), abs(sharpe_red))
    if denom == 0:
        return 0.0
    return abs(sharpe_green - sharpe_red) / denom

def compute_regime(spy_close):
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series("RED", index=spy_close.index)
    regime[spy_close > sma200] = "GREEN"
    regime[sma200.isna()] = np.nan
    return regime


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def download_data():
    import yfinance as yf
    end = dt.datetime.now()
    start = end - dt.timedelta(days=DATA_YEARS * 365)
    all_tickers = sorted(set(SP500_TICKERS + ["SPY"]))
    print(f"[INFO] Downloading {len(all_tickers)} tickers, {start.date()} to {end.date()}")
    t0 = time.time()
    stock_data = {}
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            data = yf.download(" ".join(batch), start=start, end=end,
                             group_by="ticker", progress=False, threads=True)
            if len(batch) == 1:
                if not data.empty:
                    stock_data[batch[0]] = data
            else:
                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].dropna(how="all")
                            if len(df) > 252:
                                stock_data[ticker] = df
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] Batch failed: {e}")
        if (i // batch_size + 1) % 5 == 0:
            print(f"[INFO] Downloaded {min(i+batch_size, len(all_tickers))}/{len(all_tickers)}")
    print(f"[INFO] Downloaded {len(stock_data)} tickers in {time.time()-t0:.0f}s")
    return stock_data


# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(stock_data):
    """Compute liquidity drought + recovery signals."""
    spy_close = stock_data.get("SPY")
    if spy_close is None:
        raise RuntimeError("No SPY data")
    spy_close = spy_close["Close"].squeeze()
    spy_regime = compute_regime(spy_close)

    all_signals = []
    processed = 0

    for ticker in SP500_TICKERS:
        if ticker not in stock_data:
            continue
        try:
            df = stock_data[ticker]
            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            if len(close) < 500:
                continue

            # Volume percentile (current vol ranked in 252d history)
            vol_avg_20d = volume.rolling(20).mean()
            def _vol_pctile(s):
                if len(s) < 252:
                    return np.nan
                current = s.iloc[-1]
                history = s.iloc[:-1]
                if np.isnan(current) or history.isna().all():
                    return np.nan
                return (history < current).sum() / len(history) * 100
            vol_pctile = volume.rolling(253).apply(_vol_pctile, raw=False)

            # Consecutive low-volume days counter
            # For each threshold, compute how many consecutive days volume has been below it
            is_low_5 = vol_pctile < 5
            is_low_10 = vol_pctile < 10
            is_low_15 = vol_pctile < 15

            def consecutive_true(series):
                """Count consecutive True values ending at each position."""
                result = pd.Series(0, index=series.index, dtype=int)
                count = 0
                for i in range(len(series)):
                    if series.iloc[i]:
                        count += 1
                    else:
                        count = 0
                    result.iloc[i] = count
                return result

            consec_low_5 = consecutive_true(is_low_5)
            consec_low_10 = consecutive_true(is_low_10)

            # Volume recovery: today's volume vs 20d average
            vol_ratio = volume / vol_avg_20d

            # Price vs 50d SMA (distress filter)
            sma50 = close.rolling(50).mean()
            below_sma50 = close < sma50

            # Vol compression (realized vol percentile)
            log_ret = np.log(close / close.shift(1))
            realized_vol = log_ret.rolling(20).std() * np.sqrt(252)
            def _rv_pctile(s):
                if len(s) < 252:
                    return np.nan
                current = s.iloc[-1]
                history = s.iloc[:-1]
                if np.isnan(current) or history.isna().all():
                    return np.nan
                return (history < current).sum() / len(history) * 100
            rv_pctile = realized_vol.rolling(253).apply(_rv_pctile, raw=False)

            # Forward returns
            fwd_5d = close.shift(-5) / close - 1
            fwd_10d = close.shift(-10) / close - 1
            fwd_21d = close.shift(-21) / close - 1

            sig = pd.DataFrame({
                "ticker": ticker,
                "close": close,
                "vol_pctile": vol_pctile,
                "consec_low_5": consec_low_5,
                "consec_low_10": consec_low_10,
                "vol_ratio": vol_ratio,
                "below_sma50": below_sma50,
                "rv_pctile": rv_pctile,
                "fwd_5d": fwd_5d,
                "fwd_10d": fwd_10d,
                "fwd_21d": fwd_21d,
            }, index=close.index)

            sig["regime"] = sig.index.map(spy_regime)
            sig["year"] = sig.index.year
            sig = sig.dropna(subset=["vol_pctile", "vol_ratio"])
            all_signals.append(sig)
            processed += 1
        except Exception:
            pass

    print(f"[INFO] Computed signals for {processed} stocks")
    if not all_signals:
        raise RuntimeError("No signals generated")
    return pd.concat(all_signals, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name, vol_pctile_thresh, drought_days, recovery_mult,
                     hold_days, distress_filter, signals_df):
    """Evaluate a single variant."""
    # Choose appropriate consecutive-low column
    if vol_pctile_thresh <= 5:
        consec_col = "consec_low_5"
    else:
        consec_col = "consec_low_10"

    # Drought condition: N consecutive days of low volume
    mask = signals_df[consec_col] >= drought_days

    # Recovery trigger (if specified)
    if recovery_mult is not None:
        mask &= signals_df["vol_ratio"] >= recovery_mult

    # Distress filter
    if distress_filter == True:
        mask &= signals_df["below_sma50"] == True
    elif distress_filter == "VC":
        mask &= signals_df["rv_pctile"] < 15

    trades = signals_df[mask].copy()
    fwd_col = f"fwd_{hold_days}d"
    if fwd_col not in trades.columns:
        return None
    trades = trades.dropna(subset=[fwd_col])

    if len(trades) < MIN_TRADES:
        return None

    cost_pct = COST_BPS_RT / 10000
    returns = trades[fwd_col].values - cost_pct
    n_trades = len(returns)

    sharpe = annualized_sharpe(returns)
    sort_r = sortino_ratio(returns)
    pf = profit_factor(returns)
    wr = (returns > 0).mean() * 100

    green_mask = trades["regime"] == "GREEN"
    red_mask = trades["regime"] == "RED"
    green_returns = returns[green_mask.values]
    red_returns = returns[red_mask.values]
    sharpe_green = annualized_sharpe(green_returns) if len(green_returns) >= 10 else 0
    sharpe_red = annualized_sharpe(red_returns) if len(red_returns) >= 10 else 0
    gap = regime_gap(sharpe_green, sharpe_red)

    trades["_ret"] = returns
    yearly = trades.groupby("year")["_ret"].mean()
    pct_years_prof = (yearly > 0).mean() * 100

    return {
        "name": name, "n_trades": n_trades,
        "sharpe": round(sharpe, 3), "sortino": round(sort_r, 3),
        "profit_factor": round(pf, 2), "win_rate": round(wr, 1),
        "mean_ret_pct": round(np.mean(returns)*100, 3),
        "sharpe_green": round(sharpe_green, 3), "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(gap, 3),
        "pct_years_profitable": round(pct_years_prof, 1),
        "n_years": len(yearly),
        "green_trades": int(green_mask.sum()), "red_trades": int(red_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 80)
    print(f"LIQUIDITY DROUGHT + RECOVERY v1 — {len(SP500_TICKERS)} stocks × {DATA_YEARS} years")
    print(f"Variants: {len(VARIANTS)}")
    print("=" * 80)

    run = None
    if HAS_MLFLOW:
        run = mlflow.start_run(run_name=f"liquidity_drought_v1_{dt.datetime.now().strftime('%Y%m%d_%H%M')}")

    stock_data = download_data()
    signals_df = compute_signals(stock_data)
    print(f"[INFO] {len(signals_df)} total signal-days")

    results = []
    for i, (name, vp, dd, rm, hold, dist) in enumerate(VARIANTS):
        result = evaluate_variant(name, vp, dd, rm, hold, dist, signals_df)
        if result:
            results.append(result)
            status = "PASS" if result["regime_gap"] <= REGIME_GAP_THRESHOLD and result["sharpe"] > 0 else "skip"
            print(f"  [{i+1}/{len(VARIANTS)}] {name}: Sharpe {result['sharpe']}, WR {result['win_rate']}%, "
                  f"RG {result['regime_gap']}, trades {result['n_trades']} [{status}]")
        else:
            print(f"  [{i+1}/{len(VARIANTS)}] {name}: insufficient trades")

    if not results:
        print("\n[ERROR] No variants had sufficient trades")
        if run:
            mlflow.log_param("status", "no_valid_variants")
            mlflow.end_run()
        return

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    candidates = [r for r in results if r["sharpe"] > 0 and r["regime_gap"] <= REGIME_GAP_THRESHOLD]
    print(f"\n{'='*80}")
    print(f"INITIAL SCREEN: {len(candidates)} of {len(results)} pass (Sharpe>0 + RG<=0.50)")

    top_candidates = sorted(candidates, key=lambda x: x["sharpe"], reverse=True)[:20]
    print(f"\nRunning {NUM_PERMUTATIONS}-shuffle permutation on top {len(top_candidates)}...")

    final_pass = []
    for r in top_candidates:
        name = r["name"]
        for vname, vp, dd, rm, hold, dist in VARIANTS:
            if vname == name:
                break

        if vp <= 5:
            consec_col = "consec_low_5"
        else:
            consec_col = "consec_low_10"

        mask = signals_df[consec_col] >= dd
        if rm is not None:
            mask &= signals_df["vol_ratio"] >= rm
        if dist == True:
            mask &= signals_df["below_sma50"] == True
        elif dist == "VC":
            mask &= signals_df["rv_pctile"] < 15

        trades = signals_df[mask].copy()
        fwd_col = f"fwd_{hold}d"
        trades = trades.dropna(subset=[fwd_col])
        returns = trades[fwd_col].values - COST_BPS_RT / 10000

        perm_p = permutation_test(returns, NUM_PERMUTATIONS)
        r["perm_p"] = round(perm_p, 3)
        g1 = perm_p < PERM_P_THRESHOLD
        g2 = r["regime_gap"] <= REGIME_GAP_THRESHOLD
        g3 = r["pct_years_profitable"] >= 60
        r["gates_passed"] = sum([g1, g2, g3])

        status = "PASS ALL" if all([g1, g2, g3]) else f"{r['gates_passed']}/3"
        print(f"  {name}: perm p={perm_p:.3f} {'P' if g1 else 'F'}, "
              f"RG={r['regime_gap']:.3f} {'P' if g2 else 'F'}, "
              f"years={r['pct_years_profitable']:.0f}% {'P' if g3 else 'F'} -> [{status}]")

        if all([g1, g2, g3]):
            final_pass.append(r)

    print(f"\n{'='*80}")
    print(f"FINAL: {len(final_pass)} of {len(VARIANTS)} PASS ALL 3 GATES")
    print("="*80)
    if final_pass:
        for r in sorted(final_pass, key=lambda x: x["sharpe"], reverse=True):
            print(f"  {r['name']}: Sharpe {r['sharpe']}, Sortino {r['sortino']}, "
                  f"WR {r['win_rate']}%, PF {r['profit_factor']}, trades {r['n_trades']}, "
                  f"RG {r['regime_gap']}, perm_p {r['perm_p']}")

    all_results = {"variants": results, "final_pass": final_pass,
                   "meta": {"total_variants": len(VARIANTS), "n_stocks": len(SP500_TICKERS),
                            "data_years": DATA_YEARS, "n_signal_days": len(signals_df),
                            "runtime_s": round(time.time()-t_start, 1)}}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if HAS_MLFLOW and run:
        mlflow.log_param("n_stocks", len(SP500_TICKERS))
        mlflow.log_param("n_variants", len(VARIANTS))
        mlflow.log_metric("n_pass_all_gates", len(final_pass))
        mlflow.log_metric("n_pass_initial", len(candidates))
        if final_pass:
            best = max(final_pass, key=lambda x: x["sharpe"])
            mlflow.log_metric("best_sharpe", best["sharpe"])
            mlflow.log_metric("best_regime_gap", best["regime_gap"])
            mlflow.log_param("best_variant", best["name"])
        mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
        mlflow.end_run()

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")

if __name__ == "__main__":
    main()
