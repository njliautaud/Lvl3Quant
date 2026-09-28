#!/usr/bin/env python3
"""
Supply/Demand Zone Memory v1 — Statistical Support/Resistance
==============================================================

OBSERVATION: Stocks that have bounced from the same price level multiple
times create "memory" zones. These levels represent real supply/demand
imbalances where institutional orders cluster. When price returns to a
tested support zone during distress, the probability of another bounce
is higher than random.

HYPOTHESIS: A statistically rigorous approach to support zones:
  1. Identify price levels where a stock has bounced 2+ times (within 2% band)
  2. When price returns to that zone AND stock is distressed (oversold/vol-compressed)
  3. Buy for contrarian bounce with defined 5/10/21d hold

This aligns with meta-findings:
  - Contrarian: buying at zones where price previously reversed
  - Combines structural (support) with behavioral (oversold) signals
  - Historical price memory = institutional order clustering

SIGNAL VARIANTS (48 total):
  - Zone width: 2%, 3%, 5% price band
  - Min touches: 2, 3 (times price bounced from zone)
  - Lookback for zone detection: 126d (6m), 252d (1y)
  - Condition: any, oversold (RSI<30), vol compressed (<15th pctile), or drop>3%
  - Hold: 5d, 10d, 21d

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
    mlflow.set_experiment("supply_demand_memory_v1")
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

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/supply_demand_memory_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# S&P 500 tickers
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

# Signal variants
VARIANTS = []
for zone_width in [0.02, 0.03, 0.05]:
    for min_touches in [2, 3]:
        for lookback in [126, 252]:
            for condition in ["any", "oversold", "vol_compressed", "drop3"]:
                for hold in [5, 10, 21]:
                    cond_str = {"any": "", "oversold": "_OS", "vol_compressed": "_VC", "drop3": "_D3"}[condition]
                    name = f"ZW{int(zone_width*100)}_T{min_touches}_L{lookback}{cond_str}_H{hold}"
                    VARIANTS.append((name, zone_width, min_touches, lookback, condition, hold))

print(f"Total variants: {len(VARIANTS)}")
print(f"Total stocks: {len(SP500_TICKERS)}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_vol_percentile(prices, vol_window=20, hist_window=252):
    log_ret = np.log(prices / prices.shift(1))
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    def _pctile(s):
        if len(s) < hist_window:
            return np.nan
        current = s.iloc[-1]
        history = s.iloc[:-1]
        if np.isnan(current) or history.isna().all():
            return np.nan
        return (history < current).sum() / len(history) * 100
    return realized_vol.rolling(hist_window + 1).apply(_pctile, raw=False)

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
# SUPPORT ZONE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_support_bounces(close_series, lookback, zone_width_pct):
    """
    For each day, look back `lookback` days and find price levels where
    the stock bounced (made a local low within zone_width of each other).
    
    A "bounce" = local minimum where low[i] < low[i-1] and low[i] < low[i+1]
    (simplified: price dropped then recovered within 5 days).
    
    Returns a Series of touch counts at the nearest support zone for each day.
    """
    n = len(close_series)
    touch_counts = pd.Series(0, index=close_series.index, dtype=int)
    at_support = pd.Series(False, index=close_series.index, dtype=bool)
    
    # Find local lows: days where close is lower than surrounding 5 days
    close_vals = close_series.values
    local_lows = np.zeros(n, dtype=bool)
    for i in range(5, n - 5):
        window = close_vals[i-5:i+6]
        if close_vals[i] == np.min(window):
            local_lows[i] = True
    
    # For each day, check if current price is near a support zone
    for i in range(lookback + 10, n):
        current_price = close_vals[i]
        
        # Get local lows in the lookback window
        start_idx = max(0, i - lookback)
        lows_in_window = []
        for j in range(start_idx, i - 5):  # exclude very recent (avoid look-ahead)
            if local_lows[j]:
                lows_in_window.append(close_vals[j])
        
        if not lows_in_window:
            continue
        
        # For each low, count how many other lows are within zone_width
        # and check if current price is near any such cluster
        lows_arr = np.array(lows_in_window)
        
        for low_price in lows_arr:
            zone_lo = low_price * (1 - zone_width_pct)
            zone_hi = low_price * (1 + zone_width_pct)
            
            # Count touches in this zone
            touches = np.sum((lows_arr >= zone_lo) & (lows_arr <= zone_hi))
            
            # Is current price in this zone?
            if zone_lo <= current_price <= zone_hi and touches >= 2:
                if touches > touch_counts.iloc[i]:
                    touch_counts.iloc[i] = touches
                    at_support.iloc[i] = True
    
    return touch_counts, at_support


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
# SIGNAL COMPUTATION (per zone config, not per variant)
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals_for_config(stock_data, zone_width, min_touches, lookback):
    """Compute support zone signals for a specific zone configuration."""
    spy_close = stock_data.get("SPY")
    if spy_close is None:
        raise RuntimeError("No SPY data")
    spy_close_s = spy_close["Close"].squeeze()
    spy_regime = compute_regime(spy_close_s)

    all_signals = []
    processed = 0

    for ticker in SP500_TICKERS:
        if ticker not in stock_data:
            continue
        try:
            df = stock_data[ticker]
            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if len(close) < lookback + 100:
                continue

            # Find support zones
            touch_counts, at_support = find_support_bounces(close, lookback, zone_width)

            # Only keep days at support with enough touches
            support_mask = at_support & (touch_counts >= min_touches)

            if support_mask.sum() < 5:
                continue

            # Additional features
            rsi14 = compute_rsi(close, 14)
            vol_pctile = compute_vol_percentile(close)
            daily_ret = close.pct_change()

            # Forward returns
            fwd_5d = close.shift(-5) / close - 1
            fwd_10d = close.shift(-10) / close - 1
            fwd_21d = close.shift(-21) / close - 1

            sig = pd.DataFrame({
                "ticker": ticker,
                "close": close,
                "at_support": support_mask,
                "touch_count": touch_counts,
                "rsi14": rsi14,
                "vol_pctile": vol_pctile,
                "daily_ret": daily_ret,
                "fwd_5d": fwd_5d,
                "fwd_10d": fwd_10d,
                "fwd_21d": fwd_21d,
            }, index=close.index)

            sig["regime"] = sig.index.map(spy_regime)
            sig["year"] = sig.index.year
            sig = sig.dropna(subset=["rsi14"])
            all_signals.append(sig)
            processed += 1
        except Exception:
            pass

    if not all_signals:
        return None
    print(f"  [config ZW{int(zone_width*100)}_T{min_touches}_L{lookback}] {processed} stocks processed")
    return pd.concat(all_signals, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name, condition, hold_days, signals_df):
    """Evaluate a single variant given pre-computed zone signals."""
    # Base: at support zone
    mask = signals_df["at_support"] == True

    # Additional conditions
    if condition == "oversold":
        mask &= signals_df["rsi14"] < 30
    elif condition == "vol_compressed":
        mask &= signals_df["vol_pctile"] < 15
    elif condition == "drop3":
        mask &= signals_df["daily_ret"] < -0.03

    trades = signals_df[mask].copy()
    fwd_col = f"fwd_{hold_days}d"
    if fwd_col not in trades.columns:
        return None
    trades = trades.dropna(subset=[fwd_col])

    if len(trades) < MIN_TRADES:
        return None

    cost_pct = COST_BPS_RT / 10000
    returns = trades[fwd_col].values - cost_pct

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
        "name": name, "n_trades": len(returns),
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
    print(f"SUPPLY/DEMAND ZONE MEMORY v1 — {len(SP500_TICKERS)} stocks × {DATA_YEARS} years")
    print(f"Variants: {len(VARIANTS)}")
    print("=" * 80)

    run = None
    if HAS_MLFLOW:
        run = mlflow.start_run(run_name=f"supply_demand_memory_v1_{dt.datetime.now().strftime('%Y%m%d_%H%M')}")

    stock_data = download_data()

    # Group variants by zone config (to avoid recomputing signals)
    from collections import defaultdict
    config_variants = defaultdict(list)
    for name, zw, mt, lb, cond, hold in VARIANTS:
        config_key = (zw, mt, lb)
        config_variants[config_key].append((name, cond, hold))

    results = []
    variant_idx = 0
    for (zw, mt, lb), variants in config_variants.items():
        print(f"\n[INFO] Computing zones: width={int(zw*100)}%, touches>={mt}, lookback={lb}d...")
        signals_df = compute_signals_for_config(stock_data, zw, mt, lb)
        if signals_df is None:
            for name, cond, hold in variants:
                variant_idx += 1
                print(f"  [{variant_idx}/{len(VARIANTS)}] {name}: no signals")
            continue

        for name, cond, hold in variants:
            variant_idx += 1
            result = evaluate_variant(name, cond, hold, signals_df)
            if result:
                results.append(result)
                status = "PASS" if result["regime_gap"] <= REGIME_GAP_THRESHOLD and result["sharpe"] > 0 else "skip"
                print(f"  [{variant_idx}/{len(VARIANTS)}] {name}: Sharpe {result['sharpe']}, "
                      f"WR {result['win_rate']}%, RG {result['regime_gap']}, "
                      f"trades {result['n_trades']} [{status}]")
            else:
                print(f"  [{variant_idx}/{len(VARIANTS)}] {name}: insufficient trades")

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

    # Permutation test on top candidates
    top_candidates = sorted(candidates, key=lambda x: x["sharpe"], reverse=True)[:20]
    print(f"\nRunning {NUM_PERMUTATIONS}-shuffle permutation on top {len(top_candidates)}...")

    # Need to recompute signals for perm test — cache configs
    perm_cache = {}

    final_pass = []
    for r in top_candidates:
        name = r["name"]
        for vname, zw, mt, lb, cond, hold in VARIANTS:
            if vname == name:
                break

        config_key = (zw, mt, lb)
        if config_key not in perm_cache:
            perm_cache[config_key] = compute_signals_for_config(stock_data, zw, mt, lb)

        signals_df = perm_cache[config_key]
        if signals_df is None:
            continue

        mask = signals_df["at_support"] == True
        if cond == "oversold":
            mask &= signals_df["rsi14"] < 30
        elif cond == "vol_compressed":
            mask &= signals_df["vol_pctile"] < 15
        elif cond == "drop3":
            mask &= signals_df["daily_ret"] < -0.03

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
                            "data_years": DATA_YEARS, "runtime_s": round(time.time()-t_start, 1)}}
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
