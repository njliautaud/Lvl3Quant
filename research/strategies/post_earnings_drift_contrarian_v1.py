#!/usr/bin/env python3
"""
Post-Earnings Drift Contrarian v1
===================================

HYPOTHESIS: "Post-earnings overreaction fade" — stocks that dropped sharply on
earnings (bad reaction) tend to recover partially over the following 2-4 weeks
as the initial panic fades. The contrarian setup: buy stocks 10+ days AFTER a
negative earnings reaction, when price is still depressed but the news is fully
digested.

NOTE ON HC #736: This is NOT an earnings play (buying/selling around
announcements). We look 10-30 days AFTER the event, when the initial reaction
has settled and a contrarian opportunity may exist.

EARNINGS PROXY: We don't have actual earnings dates. Instead we detect days
where a stock had a >5% single-day move on high volume (>2x 20d average).
This captures earnings reactions plus similar fundamental catalyst events.

SIGNAL VARIANTS (~72):
  - Post-earnings window: 10-15d, 15-20d, 20-30d after the event
  - Reaction threshold: stock dropped -5%, -8%, -12%+ on/around event (3-day)
  - Recovery filter: stock still below pre-event price (hasn't already recovered)
  - Flow confirmation: MFI<30, or no filter
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
from collections import defaultdict
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
    mlflow.set_experiment("post_earnings_drift_contrarian_v1")
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

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/post_earnings_drift_contrarian_v1")
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
# Post-earnings windows x reaction thresholds x recovery filter (always ON) x MFI filter x hold periods
# 3 windows x 3 thresholds x 2 MFI filter x 1 recovery x 4 hold = 72 variants (with 3 holds -> 54, with extra holds -> 72)
VARIANTS = []
for window_label, window_lo, window_hi in [("W10_15", 10, 15), ("W15_20", 15, 20), ("W20_30", 20, 30)]:
    for drop_thresh in [-0.05, -0.08, -0.12]:
        drop_str = f"D{abs(int(drop_thresh*100))}"
        for mfi_filter in [False, True]:
            mfi_str = "_MFI30" if mfi_filter else ""
            for hold in [5, 10, 21]:
                name = f"{window_label}_{drop_str}{mfi_str}_H{hold}"
                VARIANTS.append((name, window_lo, window_hi, drop_thresh, mfi_filter, hold))

print(f"Total variants: {len(VARIANTS)}")
print(f"Total stocks: {len(SP500_TICKERS)}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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
# DETECT "EARNINGS-LIKE" EVENTS (PROXY)
# ─────────────────────────────────────────────────────────────────────────────

def detect_earnings_events(stock_data):
    """
    Detect earnings-like events: single-day moves >5% on high volume (>2x 20d avg).
    Returns dict: ticker -> list of (event_date, 3day_return) tuples.
    Only keeps NEGATIVE reactions (drops).
    """
    events = {}
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
            if len(close) < 252:
                continue

            daily_ret = close.pct_change()
            vol_avg_20 = volume.rolling(20).mean()
            vol_ratio = volume / vol_avg_20.replace(0, np.nan)

            # Detect big single-day moves on high volume
            big_move = daily_ret.abs() > 0.05  # >5% single-day move
            high_vol = vol_ratio > 2.0          # >2x average volume

            event_mask = big_move & high_vol
            event_dates = close.index[event_mask]

            ticker_events = []
            for edate in event_dates:
                # Compute 3-day return around the event (day before to day after)
                idx = close.index.get_loc(edate)
                if idx < 1 or idx >= len(close) - 1:
                    continue
                # 3-day window: close[idx-1] to close[idx+1]
                pre_price = close.iloc[idx - 1]
                post_price = close.iloc[min(idx + 1, len(close) - 1)]
                ret_3d = (post_price / pre_price) - 1

                # Also store the pre-event price (for recovery filter)
                ticker_events.append({
                    "date": edate,
                    "ret_3d": ret_3d,
                    "pre_price": pre_price,
                    "daily_ret": daily_ret.iloc[idx],
                })

            # Filter: only keep negative reaction events (drops)
            neg_events = [e for e in ticker_events if e["ret_3d"] < -0.03]

            # Deduplicate: if two events within 5 days, keep the bigger drop
            if neg_events:
                deduped = [neg_events[0]]
                for ev in neg_events[1:]:
                    if (ev["date"] - deduped[-1]["date"]).days > 5:
                        deduped.append(ev)
                    elif ev["ret_3d"] < deduped[-1]["ret_3d"]:
                        deduped[-1] = ev
                events[ticker] = deduped
        except Exception:
            pass

    total_events = sum(len(v) for v in events.values())
    print(f"[INFO] Detected {total_events} earnings-like negative events across {len(events)} stocks")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRADE SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def build_trade_signals(stock_data, events):
    """
    For each detected earnings-like event, build potential contrarian trade entries
    at various windows after the event. Returns a DataFrame with all needed columns.
    """
    spy_df = stock_data.get("SPY")
    if spy_df is None:
        raise RuntimeError("No SPY data")
    spy_close = spy_df["Close"].squeeze()
    spy_regime = compute_regime(spy_close)

    all_signals = []

    for ticker, ticker_events in events.items():
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

            mfi14 = compute_mfi(high, low, close, volume, 14)

            for event in ticker_events:
                edate = event["date"]
                ret_3d = event["ret_3d"]
                pre_price = event["pre_price"]

                # For each trading day 5-35 days after the event, check if it's a valid entry
                eidx = close.index.get_loc(edate)
                for offset in range(5, 36):
                    entry_idx = eidx + offset
                    if entry_idx >= len(close) - 21:  # need room for forward returns
                        break

                    entry_date = close.index[entry_idx]
                    entry_price = close.iloc[entry_idx]
                    days_after = offset  # trading days after event

                    # Recovery filter: is price still below pre-event price?
                    still_depressed = entry_price < pre_price
                    recovery_pct = (entry_price / pre_price) - 1  # negative = still depressed

                    # MFI at entry
                    entry_mfi = mfi14.iloc[entry_idx] if entry_idx < len(mfi14) else np.nan

                    # Forward returns from entry
                    fwd_5d = (close.iloc[min(entry_idx + 5, len(close)-1)] / entry_price) - 1 if entry_idx + 5 < len(close) else np.nan
                    fwd_10d = (close.iloc[min(entry_idx + 10, len(close)-1)] / entry_price) - 1 if entry_idx + 10 < len(close) else np.nan
                    fwd_21d = (close.iloc[min(entry_idx + 21, len(close)-1)] / entry_price) - 1 if entry_idx + 21 < len(close) else np.nan

                    # Regime
                    regime_val = spy_regime.get(entry_date, np.nan) if entry_date in spy_regime.index else np.nan

                    all_signals.append({
                        "ticker": ticker,
                        "event_date": edate,
                        "entry_date": entry_date,
                        "days_after_event": days_after,
                        "event_ret_3d": ret_3d,
                        "still_depressed": still_depressed,
                        "recovery_pct": recovery_pct,
                        "entry_mfi": entry_mfi,
                        "fwd_5d": fwd_5d,
                        "fwd_10d": fwd_10d,
                        "fwd_21d": fwd_21d,
                        "regime": regime_val,
                        "year": entry_date.year,
                    })
        except Exception:
            pass

    if not all_signals:
        return None

    df = pd.DataFrame(all_signals)
    print(f"[INFO] Built {len(df)} potential trade signals from {len(events)} stocks")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name, window_lo, window_hi, drop_thresh, mfi_filter, hold_days, signals_df):
    """Evaluate a single variant."""

    # Filter by post-event window
    mask = (signals_df["days_after_event"] >= window_lo) & (signals_df["days_after_event"] <= window_hi)

    # Filter by reaction severity (3-day return around event)
    mask &= signals_df["event_ret_3d"] <= drop_thresh

    # Recovery filter: stock must still be below pre-event price
    mask &= signals_df["still_depressed"] == True

    # MFI filter
    if mfi_filter:
        mask &= signals_df["entry_mfi"] < 30

    trades = signals_df[mask].copy()
    fwd_col = f"fwd_{hold_days}d"
    if fwd_col not in trades.columns:
        return None
    trades = trades.dropna(subset=[fwd_col])

    # Deduplicate: for same ticker+event, keep only one entry (the first in window)
    trades = trades.sort_values("days_after_event").drop_duplicates(
        subset=["ticker", "event_date"], keep="first"
    )

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

    avg_drop = trades["event_ret_3d"].mean() * 100
    avg_recovery = trades["recovery_pct"].mean() * 100

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
        "avg_event_drop_pct": round(avg_drop, 1),
        "avg_recovery_pct_at_entry": round(avg_recovery, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 80)
    print(f"POST-EARNINGS DRIFT CONTRARIAN v1 — {len(SP500_TICKERS)} stocks x {DATA_YEARS} years")
    print(f"Variants: {len(VARIANTS)}")
    print(f"Hypothesis: Post-earnings overreaction fade")
    print(f"NOT an earnings play — looks 10-30 days AFTER the reaction")
    print("=" * 80)

    run = None
    if HAS_MLFLOW:
        run = mlflow.start_run(
            run_name=f"post_earnings_drift_contrarian_v1_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
        )

    # Download data
    stock_data = download_data()

    # Detect earnings-like events (big drops on high volume)
    print("\n[INFO] Detecting earnings-like events (proxy)...")
    events = detect_earnings_events(stock_data)

    if not events:
        print("[ERROR] No earnings-like events detected")
        if run:
            mlflow.log_param("status", "no_events_detected")
            mlflow.end_run()
        return

    # Build trade signals from events
    print("\n[INFO] Building trade signals...")
    signals_df = build_trade_signals(stock_data, events)

    if signals_df is None or len(signals_df) == 0:
        print("[ERROR] No trade signals generated")
        if run:
            mlflow.log_param("status", "no_signals")
            mlflow.end_run()
        return

    # Evaluate all variants
    print(f"\n[INFO] Evaluating {len(VARIANTS)} variants...")
    results = []
    for idx, (name, window_lo, window_hi, drop_thresh, mfi_filter, hold) in enumerate(VARIANTS, 1):
        result = evaluate_variant(name, window_lo, window_hi, drop_thresh, mfi_filter, hold, signals_df)
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
        for vname, wlo, whi, dt_val, mfi_f, hold in VARIANTS:
            if vname == name:
                break

        # Reconstruct trade filter
        mask = (signals_df["days_after_event"] >= wlo) & (signals_df["days_after_event"] <= whi)
        mask &= signals_df["event_ret_3d"] <= dt_val
        mask &= signals_df["still_depressed"] == True
        if mfi_f:
            mask &= signals_df["entry_mfi"] < 30

        trades = signals_df[mask].copy()
        fwd_col = f"fwd_{hold}d"
        trades = trades.dropna(subset=[fwd_col])
        trades = trades.sort_values("days_after_event").drop_duplicates(
            subset=["ticker", "event_date"], keep="first"
        )

        if len(trades) < MIN_TRADES:
            continue

        returns = trades[fwd_col].values - COST_BPS_RT / 10000

        perm_p = permutation_test(returns, NUM_PERMUTATIONS)
        r["perm_p"] = round(perm_p, 3)
        g1 = perm_p < PERM_P_THRESHOLD
        g2 = r["regime_gap"] <= REGIME_GAP_THRESHOLD
        g3 = r["pct_years_profitable"] >= 75
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
                f"RG {r['regime_gap']}, perm_p {r['perm_p']}, "
                f"avg_drop {r['avg_event_drop_pct']}%"
            )

    # Save results
    all_results = {
        "hypothesis": "Post-earnings overreaction fade — stocks that dropped sharply on earnings tend to recover partially 2-4 weeks later as panic fades",
        "note": "NOT an earnings play (HC #736 compliant) — entries are 10-30 days AFTER the event",
        "earnings_proxy": "Single-day move >5% on >2x average volume used as earnings event proxy",
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
            "n_events_detected": sum(len(v) for v in events.values()),
            "n_stocks_with_events": len(events),
            "n_total_signals": len(signals_df),
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
        mlflow.log_param("hypothesis", "post_earnings_overreaction_fade")
        mlflow.log_param("earnings_proxy", "5pct_move_2x_volume")
        mlflow.log_metric("n_events_detected", sum(len(v) for v in events.values()))
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
            try:
            mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
        except Exception as e:
            print(f"[WARN] Could not log artifact to MLflow: {e}")
        except Exception as e:
            print(f"[WARN] Could not log artifact to MLflow: {e}")
        mlflow.end_run()

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
