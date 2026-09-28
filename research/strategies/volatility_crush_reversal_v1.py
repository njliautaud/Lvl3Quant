#!/usr/bin/env python3
"""
Volatility Crush Reversal v1
==============================

HYPOTHESIS: After periods of extremely LOW realized volatility (ATR compression),
stocks tend to experience sharp moves. Instead of predicting direction, we look
for stocks that have ALREADY had the vol expansion (sharp move happened) AND are
now oversold. The crush-then-expand-then-oversold pattern creates high-probability
mean-reversion entries because:
  1. Vol compression created a coiled spring
  2. Spring released (sharp move occurred)
  3. Move overshot (now oversold) -> bounce probability elevated

SIGNAL VARIANTS (~80-100):
  - Vol compression detection: ATR(14d) < 10th, 15th, or 20th percentile of 252d ATR range
    (stock was recently in low-vol regime)
  - Compression recency: vol was compressed within last 5d, 10d, or 21d
  - Expansion trigger: stock then dropped 3%+ or 5%+ from the compressed state
  - Flow confirmation: MFI<20, RSI<30, or no filter
  - Hold periods: 5d, 10d, 21d

VALIDATION: permutation test (1000 shuffles), regime gap, year consistency.

Author: Claude Opus 4.6 / Teleclaude Research
Date: 2026-07-23
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
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# MLflow
try:
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("volatility_crush_reversal_v1")
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

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/volatility_crush_reversal_v1")
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
# 3 ATR pctiles * 3 recency windows * 2 drop thresholds * 3 flow filters * 3 hold = 162
# That's more than needed; trim by using 2 recency windows for the 15th pctile only
VARIANTS = []
for atr_pctile in [10, 15, 20]:
    for recency in [5, 10, 21]:
        for drop_thresh in [0.03, 0.05]:
            for flow_filter in ["none", "rsi_low", "mfi_low"]:
                for hold in [5, 10, 21]:
                    drop_str = f"D{int(drop_thresh*100)}"
                    flow_str = {"none": "NF", "rsi_low": "RSI", "mfi_low": "MFI"}[flow_filter]
                    name = f"ATR{atr_pctile}_R{recency}_{drop_str}_{flow_str}_H{hold}"
                    VARIANTS.append((name, atr_pctile, recency, drop_thresh, flow_filter, hold))

print(f"Total variants: {len(VARIANTS)}")
print(f"Total stocks: {len(SP500_TICKERS)}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(high, low, close, period=14):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index (volume-weighted RSI)."""
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    delta_tp = typical_price.diff()
    pos_mf = raw_mf.where(delta_tp > 0, 0.0)
    neg_mf = raw_mf.where(delta_tp < 0, 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


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
# VOLATILITY CRUSH DETECTION + SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_vol_crush_signals(stock_data):
    """
    For each stock, compute:
    - ATR(14d) and its rolling 252d percentile rank (to detect compression)
    - Whether ATR was below various percentile thresholds recently
    - Return from the compressed point to now (to detect expansion/drop)
    - RSI, MFI for oversold confirmation
    - Forward returns for evaluation
    """
    spy_df = stock_data.get("SPY")
    if spy_df is None:
        raise RuntimeError("No SPY data")
    spy_close = spy_df["Close"].squeeze()
    spy_regime = compute_regime(spy_close)

    all_signals = []
    processed = 0

    for ticker in SP500_TICKERS:
        if ticker not in stock_data:
            continue
        try:
            df = stock_data[ticker]
            close = df["Close"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]

            if len(close) < 400:  # Need 252d window + buffer
                continue

            # ATR(14d)
            atr14 = compute_atr(high, low, close, period=14)

            # Rolling 252d percentile rank of ATR (0-100)
            # This tells us where today's ATR sits relative to the last year
            atr_pctile = atr14.rolling(252).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) == 252 else np.nan,
                raw=False
            )

            # For each recency window, check if ATR was below threshold recently
            # We need: "was vol compressed within last N days?"
            # For each threshold (10th, 15th, 20th pctile), create a boolean
            # Then rolling max over recency windows tells us if it was True recently
            compressed_10 = (atr_pctile <= 10).astype(float)
            compressed_15 = (atr_pctile <= 15).astype(float)
            compressed_20 = (atr_pctile <= 20).astype(float)

            # Rolling max: was compressed within last N days?
            comp_10_5d = compressed_10.rolling(5, min_periods=1).max()
            comp_10_10d = compressed_10.rolling(10, min_periods=1).max()
            comp_10_21d = compressed_10.rolling(21, min_periods=1).max()

            comp_15_5d = compressed_15.rolling(5, min_periods=1).max()
            comp_15_10d = compressed_15.rolling(10, min_periods=1).max()
            comp_15_21d = compressed_15.rolling(21, min_periods=1).max()

            comp_20_5d = compressed_20.rolling(5, min_periods=1).max()
            comp_20_10d = compressed_20.rolling(10, min_periods=1).max()
            comp_20_21d = compressed_20.rolling(21, min_periods=1).max()

            # Expansion trigger: recent drop from the pre-compression price level
            # Use the max close over the last 21 days as the "pre-drop peak"
            rolling_max_21 = close.rolling(21).max()
            drawdown_from_peak = (close - rolling_max_21) / rolling_max_21

            # RSI and MFI for flow confirmation
            rsi14 = compute_rsi(close, 14)
            mfi14 = compute_mfi(high, low, close, volume, 14)

            # Forward returns
            fwd_5d = close.shift(-5) / close - 1
            fwd_10d = close.shift(-10) / close - 1
            fwd_21d = close.shift(-21) / close - 1

            sig = pd.DataFrame({
                "ticker": ticker,
                "close": close,
                "atr_pctile": atr_pctile,
                "drawdown_from_peak": drawdown_from_peak,
                "rsi14": rsi14,
                "mfi14": mfi14,
                # Compression flags for each (threshold, recency) combo
                "comp_10_5d": comp_10_5d,
                "comp_10_10d": comp_10_10d,
                "comp_10_21d": comp_10_21d,
                "comp_15_5d": comp_15_5d,
                "comp_15_10d": comp_15_10d,
                "comp_15_21d": comp_15_21d,
                "comp_20_5d": comp_20_5d,
                "comp_20_10d": comp_20_10d,
                "comp_20_21d": comp_20_21d,
                "fwd_5d": fwd_5d,
                "fwd_10d": fwd_10d,
                "fwd_21d": fwd_21d,
            }, index=close.index)

            sig["regime"] = sig.index.map(spy_regime)
            sig["year"] = sig.index.year
            sig = sig.dropna(subset=["atr_pctile", "rsi14"])
            all_signals.append(sig)
            processed += 1
        except Exception:
            pass

    if not all_signals:
        return None
    print(f"  [vol_crush] {processed} stocks processed")
    return pd.concat(all_signals, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name, atr_pctile, recency, drop_thresh, flow_filter, hold_days, signals_df):
    """Evaluate a single variant given pre-computed vol crush signals."""

    # Was vol compressed (ATR below threshold) within the recency window?
    comp_col = f"comp_{atr_pctile}_{recency}d"
    if comp_col not in signals_df.columns:
        return None
    mask = signals_df[comp_col] > 0  # Was compressed recently

    # Expansion trigger: stock dropped from peak by at least drop_thresh
    mask &= signals_df["drawdown_from_peak"] <= -drop_thresh

    # Flow confirmation filter (oversold confirmation)
    if flow_filter == "rsi_low":
        mask &= signals_df["rsi14"] < 30
    elif flow_filter == "mfi_low":
        mask &= signals_df["mfi14"] < 20
    # "none" = no additional filter

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
        "mean_ret_pct": round(np.mean(returns) * 100, 3),
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
    print(f"VOLATILITY CRUSH REVERSAL v1 — {len(SP500_TICKERS)} stocks x {DATA_YEARS} years")
    print(f"Variants: {len(VARIANTS)}")
    print("=" * 80)

    run = None
    if HAS_MLFLOW:
        run = mlflow.start_run(
            run_name=f"volatility_crush_reversal_v1_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
        )

    stock_data = download_data()

    print(f"\n[INFO] Computing volatility crush signals for all stocks...")
    signals_df = compute_vol_crush_signals(stock_data)
    if signals_df is None:
        print("[ERROR] No signals computed")
        if run:
            mlflow.log_param("status", "no_signals")
            mlflow.end_run()
        return

    results = []
    for idx, (name, atr_pctile, recency, drop_thresh, flow_filter, hold) in enumerate(VARIANTS, 1):
        result = evaluate_variant(name, atr_pctile, recency, drop_thresh, flow_filter, hold, signals_df)
        if result:
            results.append(result)
            status = (
                "PASS" if result["regime_gap"] <= REGIME_GAP_THRESHOLD
                and result["sharpe"] > 0
                else "skip"
            )
            print(
                f"  [{idx}/{len(VARIANTS)}] {name}: Sharpe {result['sharpe']}, "
                f"WR {result['win_rate']}%, RG {result['regime_gap']}, "
                f"trades {result['n_trades']} [{status}]"
            )
        else:
            print(f"  [{idx}/{len(VARIANTS)}] {name}: insufficient trades")

    if not results:
        print("\n[ERROR] No variants had sufficient trades")
        if run:
            mlflow.log_param("status", "no_valid_variants")
            mlflow.end_run()
        return

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    candidates = [
        r for r in results
        if r["sharpe"] > 0 and r["regime_gap"] <= REGIME_GAP_THRESHOLD
    ]
    print(f"\n{'='*80}")
    print(f"INITIAL SCREEN: {len(candidates)} of {len(results)} pass (Sharpe>0 + RG<=0.50)")

    # Permutation test on top candidates
    top_candidates = sorted(candidates, key=lambda x: x["sharpe"], reverse=True)[:20]
    print(f"\nRunning {NUM_PERMUTATIONS}-shuffle permutation on top {len(top_candidates)}...")

    final_pass = []
    for r in top_candidates:
        name = r["name"]
        # Find variant params
        for vname, atr_pctile, recency, drop_thresh, flow_filter, hold in VARIANTS:
            if vname == name:
                break

        # Reconstruct the trade filter
        comp_col = f"comp_{atr_pctile}_{recency}d"
        mask = signals_df[comp_col] > 0
        mask &= signals_df["drawdown_from_peak"] <= -drop_thresh

        if flow_filter == "rsi_low":
            mask &= signals_df["rsi14"] < 30
        elif flow_filter == "mfi_low":
            mask &= signals_df["mfi14"] < 20

        trades = signals_df[mask].copy()
        fwd_col = f"fwd_{hold}d"
        trades = trades.dropna(subset=[fwd_col])
        returns = trades[fwd_col].values - COST_BPS_RT / 10000

        perm_p = permutation_test(returns, NUM_PERMUTATIONS)
        r["perm_p"] = round(perm_p, 3)
        g1 = perm_p < PERM_P_THRESHOLD
        g2 = r["regime_gap"] <= REGIME_GAP_THRESHOLD
        g3 = r["pct_years_profitable"] >= 75  # >75% profitable years
        r["gates_passed"] = sum([g1, g2, g3])
        r["perm_pass"] = g1
        r["regime_pass"] = g2
        r["year_consistency_pass"] = g3

        status = "PASS ALL" if all([g1, g2, g3]) else f"{r['gates_passed']}/3"
        print(
            f"  {name}: perm p={perm_p:.3f} {'P' if g1 else 'F'}, "
            f"RG={r['regime_gap']:.3f} {'P' if g2 else 'F'}, "
            f"years={r['pct_years_profitable']:.0f}% {'P' if g3 else 'F'} -> [{status}]"
        )

        if all([g1, g2, g3]):
            final_pass.append(r)

    print(f"\n{'='*80}")
    print(f"FINAL: {len(final_pass)} of {len(VARIANTS)} PASS ALL 3 GATES")
    print("=" * 80)
    if final_pass:
        for r in sorted(final_pass, key=lambda x: x["sharpe"], reverse=True):
            print(
                f"  {r['name']}: Sharpe {r['sharpe']}, Sortino {r['sortino']}, "
                f"WR {r['win_rate']}%, PF {r['profit_factor']}, trades {r['n_trades']}, "
                f"RG {r['regime_gap']}, perm_p {r['perm_p']}"
            )

    # Save ALL results — including perm_p for tested candidates
    all_results = {
        "hypothesis": "Volatility crush reversal — after ATR compression, stocks that experience sharp drops (vol expansion + overshoot) have elevated mean-reversion probability",
        "variants": results,
        "top_candidates_tested": top_candidates,
        "final_pass": final_pass,
        "meta": {
            "total_variants": len(VARIANTS),
            "n_stocks": len(SP500_TICKERS),
            "data_years": DATA_YEARS,
            "cost_bps_rt": COST_BPS_RT,
            "num_permutations": NUM_PERMUTATIONS,
            "regime_gap_threshold": REGIME_GAP_THRESHOLD,
            "perm_p_threshold": PERM_P_THRESHOLD,
            "min_trades": MIN_TRADES,
            "runtime_s": round(time.time() - t_start, 1),
            "timestamp": dt.datetime.now().isoformat(),
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")

    if HAS_MLFLOW and run:
        mlflow.log_param("n_stocks", len(SP500_TICKERS))
        mlflow.log_param("n_variants", len(VARIANTS))
        mlflow.log_param("hypothesis", "volatility_crush_reversal")
        mlflow.log_metric("n_pass_all_gates", len(final_pass))
        mlflow.log_metric("n_pass_initial", len(candidates))
        mlflow.log_metric("n_total_results", len(results))
        if final_pass:
            best = max(final_pass, key=lambda x: x["sharpe"])
            mlflow.log_metric("best_sharpe", best["sharpe"])
            mlflow.log_metric("best_sortino", best["sortino"])
            mlflow.log_metric("best_regime_gap", best["regime_gap"])
            mlflow.log_metric("best_perm_p", best["perm_p"])
            mlflow.log_param("best_variant", best["name"])
        try:
            mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
        except Exception as e:
            print(f"[WARN] mlflow.log_artifact failed (cross-node permissions): {e}")
        mlflow.end_run()

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
