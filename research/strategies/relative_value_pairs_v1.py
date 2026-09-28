#!/usr/bin/env python3
"""
Relative Value Pairs v1 — Intra-Sector Mean Reversion
======================================================

HYPOTHESIS: "Intra-sector relative value mean reversion" — when two stocks in
the SAME sector diverge significantly in performance (one outperforms, one
underperforms), the spread tends to revert. Buy the laggard + short the leader
within sector = market-neutral pair. This should be inherently regime-agnostic
since the long/short cancels market beta.

SIGNAL VARIANTS (~60-80):
  - Sector grouping: GICS sector (11 sectors)
  - Spread measurement: 21d return difference, 63d return difference
  - Entry threshold: top/bottom 10th, 20th, 30th percentile of intra-sector
    return spread
  - MFI/flow confirmation: ON (laggard must have MFI<30) or OFF
  - Hold periods: 5d, 10d, 21d

IMPLEMENTATION: For each day, within each sector, rank stocks by recent
relative performance. Go LONG bottom performers, SHORT top performers.
Compute portfolio return as (long basket - short basket) / 2. Track daily P&L.

VALIDATION: permutation test (1000 shuffles), regime gap, year consistency.
Cost: 10 bps RT per leg = 20 bps total for the pair.

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
    mlflow.set_experiment("relative_value_pairs_v1")
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False
    print("MLflow not available")

# Constants
COST_BPS_RT_PAIR = 20  # 10 bps per leg x 2 legs
NUM_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
PERM_P_THRESHOLD = 0.05
MIN_TRADES = 30
DATA_YEARS = 13

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/relative_value_pairs_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GICS Sector -> Ticker mapping (S&P 500 representative)
SECTOR_MAP = {
    "Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","AMD","QCOM","INTC",
        "TXN","ADI","LRCX","KLAC","SNPS","CDNS","MCHP","FTNT","ANET","NOW",
        "INTU","WDAY","TEAM","HUBS","VEEV","ANSS","PANW","CRWD","ZS","NET",
        "DDOG","MDB","SNOW","PLTR","NFLX","ACN","CSCO","ON","KEYS","FICO",
    ],
    "Financials": [
        "JPM","V","MA","BRK-B","GS","BLK","AXP","MS","C","SCHW","PNC","CME",
        "ICE","MCO","AON","AIG","AMP","ALL","MTD","COF","DFS","SYF","WFC",
        "BAC","STT","NTRS","FITB","HBAN","CFG","KEY","RF","CMA","ZION","MTB",
        "BEN","TROW","IVZ","MKTX","CBOE","NDAQ","TRV","AJG","WRB","L",
    ],
    "Healthcare": [
        "UNH","JNJ","LLY","MRK","ABBV","TMO","ABT","PFE","AMGN","GILD",
        "MDT","ISRG","SYK","BMY","VRTX","REGN","BIIB","HUM","DXCM","IDXX",
        "EW","IQV","HOLX","WAT","TDY","DVA","DGX","LH","UHS","THC","MOH",
        "HCA","BAX","STE","A","BDX","ZTS","MRNA",
    ],
    "Consumer_Discretionary": [
        "AMZN","TSLA","HD","MCD","LOW","NKE","BKNG","TJX","CMG","ABNB",
        "ROST","BURL","DG","DLTR","BBY","KSS","ETSY","W","CHWY","SFM",
        "F","GM","UBER","LYFT","DASH","MAR","HLT","WYNN","MGM","CZR",
        "NCLH","RCL","CCL","DAL","UAL","LUV","AAL","DHI","LEN","NVR","PHM",
        "POOL",
    ],
    "Communication_Services": [
        "GOOGL","META","DIS","CMCSA","PARA","WBD","FOX","LYV","SPOT",
        "SNAP","PINS","RBLX","TTWO","EA","ZM","SQ","COIN","HOOD",
    ],
    "Industrials": [
        "CAT","HON","BA","LMT","GE","RTX","DE","GD","NOC","ETN","WM",
        "NSC","FDX","ADP","CTAS","PAYX","MSI","CARR","IR","PH","DOV",
        "GWW","SWK","XYL","GNRC","ROK","EMR","ITW","MMM","FAST","PCAR",
        "ODFL","CPRT","RSG","ROL","UNP",
    ],
    "Energy": [
        "CVX","COP","EOG","SLB","OXY","PSX","MPC","HAL","DVN","FANG",
        "EQT","AR","RRC","MRO","APA","CTRA","BKR","NOV","FTI","TRGP",
    ],
    "Consumer_Staples": [
        "PG","KO","PEP","COST","WMT","MDLZ","CL","MO","KDP","GIS",
        "KMB","CLX","CHD","SJM","HRL","CPB","CAG","MKC","HSY","K",
        "STZ","TAP","SAM","MNST","KR","SFM","AZO","AAP","ORLY","LKQ",
        "TGT","TSCO",
    ],
    "Utilities": [
        "NEE","SO","DUK","AEP","XEL","WEC","ED","EXC","AES","PPL",
        "CMS","AWK","ETR","SRE","D","CEG",
    ],
    "Materials": [
        "LIN","SHW","PPG","NUE","VMC","MLM","FCX","DOW","CTVA","ALB",
        "FMC","APD","DD","CF","ECL",
    ],
    "Real_Estate": [
        "PLD","AMT","CCI","EQIX","SPG","PSA","O","DLR","WELL","AVB",
        "EQR","XLRE",
    ],
}

# Flatten all tickers
ALL_TICKERS = sorted(set(t for tickers in SECTOR_MAP.values() for t in tickers))
# Reverse map: ticker -> sector
TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_MAP.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector

# Signal variants
# 2 lookbacks * 3 entry thresholds * 2 MFI * 3 hold = 36
# Actually we want more granularity. Let's do:
# 2 spread_lookback * 3 percentile_threshold * 2 mfi_filter * 3 hold = 36
# Plus "combined" lookback (average of 21d and 63d) = 54
# That's a bit under 60. Let's add a 10d lookback too -> 4 * 3 * 2 * 3 = 72
VARIANTS = []
for spread_lookback in [10, 21, 42, 63]:
    for pctile_thresh in [10, 20, 30]:
        for mfi_filter in [False, True]:
            for hold in [5, 10, 21]:
                mfi_str = "_MFI" if mfi_filter else ""
                name = f"RV_L{spread_lookback}_P{pctile_thresh}{mfi_str}_H{hold}"
                VARIANTS.append((name, spread_lookback, pctile_thresh, mfi_filter, hold))

print(f"Total variants: {len(VARIANTS)}")
print(f"Total stocks: {len(ALL_TICKERS)}")
print(f"Sectors: {len(SECTOR_MAP)}")


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
    all_tickers = sorted(set(ALL_TICKERS + ["SPY"]))
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
# RELATIVE VALUE SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_signals(stock_data, spread_lookback):
    """
    For each day and each sector, rank stocks by their lookback-period return.
    Compute intra-sector percentile rank. Bottom = laggard, Top = leader.
    Returns a DataFrame with columns: date, ticker, sector, sector_pctile_rank,
    mfi14, fwd_5d, fwd_10d, fwd_21d, regime, year.
    """
    spy_df = stock_data.get("SPY")
    if spy_df is None:
        raise RuntimeError("No SPY data")
    spy_close = spy_df["Close"].squeeze()
    spy_regime = compute_regime(spy_close)

    # Compute lookback returns for all stocks
    ret_dict = {}
    mfi_dict = {}
    fwd_dict = {5: {}, 10: {}, 21: {}}

    for ticker in ALL_TICKERS:
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

            if len(close) < spread_lookback + 200:
                continue

            # Lookback return
            lookback_ret = close / close.shift(spread_lookback) - 1
            ret_dict[ticker] = lookback_ret

            # MFI
            mfi_dict[ticker] = compute_mfi(high, low, close, volume, 14)

            # Forward returns
            for h in [5, 10, 21]:
                fwd_dict[h][ticker] = close.shift(-h) / close - 1
        except Exception:
            pass

    if not ret_dict:
        return None

    # Build return panel
    ret_panel = pd.DataFrame(ret_dict)
    mfi_panel = pd.DataFrame(mfi_dict)
    fwd_panels = {h: pd.DataFrame(fwd_dict[h]) for h in [5, 10, 21]}

    # For each sector, compute intra-sector percentile rank
    all_rows = []
    processed_sectors = 0

    for sector, sector_tickers in SECTOR_MAP.items():
        avail = [t for t in sector_tickers if t in ret_panel.columns]
        if len(avail) < 4:  # need at least 4 stocks for meaningful rank
            continue

        sector_rets = ret_panel[avail]
        # Rank within sector: 0 = worst performer (laggard), 1 = best (leader)
        sector_pctile = sector_rets.rank(axis=1, pct=True)

        for ticker in avail:
            try:
                pctile_series = sector_pctile[ticker]
                mfi_series = mfi_panel[ticker] if ticker in mfi_panel.columns else pd.Series(
                    np.nan, index=pctile_series.index
                )

                sig = pd.DataFrame({
                    "ticker": ticker,
                    "sector": sector,
                    "sector_pctile": pctile_series,
                    "mfi14": mfi_series,
                }, index=pctile_series.index)

                for h in [5, 10, 21]:
                    if ticker in fwd_panels[h].columns:
                        sig[f"fwd_{h}d"] = fwd_panels[h][ticker]

                sig["regime"] = sig.index.map(spy_regime)
                sig["year"] = sig.index.year
                sig = sig.dropna(subset=["sector_pctile"])
                all_rows.append(sig)
            except Exception:
                pass

        processed_sectors += 1

    if not all_rows:
        return None

    result = pd.concat(all_rows, axis=0)
    print(f"  [lookback={spread_lookback}d] {processed_sectors} sectors, "
          f"{len(ret_dict)} stocks, {len(result)} signal rows")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name, pctile_thresh, mfi_filter, hold_days, signals_df):
    """
    Evaluate a relative value pairs variant.

    For each date:
      - Within each sector, identify laggards (bottom pctile_thresh%) and
        leaders (top pctile_thresh%).
      - Go LONG laggards, SHORT leaders.
      - Portfolio daily return = mean(laggard_fwd) - mean(leader_fwd) / 2
        (divided by 2 because we're using equal capital long+short)
      - Apply MFI filter on laggard side if requested.
    """
    fwd_col = f"fwd_{hold_days}d"
    if fwd_col not in signals_df.columns:
        return None

    df = signals_df.dropna(subset=[fwd_col, "sector_pctile"]).copy()
    if len(df) == 0:
        return None

    low_thresh = pctile_thresh / 100.0   # e.g., 0.10 for bottom 10%
    high_thresh = 1.0 - low_thresh       # e.g., 0.90 for top 10%

    # Identify laggards and leaders
    laggard_mask = df["sector_pctile"] <= low_thresh
    leader_mask = df["sector_pctile"] >= high_thresh

    # Apply MFI filter on laggards if requested
    if mfi_filter:
        laggard_mask &= df["mfi14"] < 30

    laggards = df[laggard_mask].copy()
    leaders = df[leader_mask].copy()

    if len(laggards) < MIN_TRADES or len(leaders) < MIN_TRADES:
        return None

    # Group by date: compute average long return (laggards) and short return (leaders)
    # The "trade" is a daily cross-sectional portfolio
    laggard_daily = laggards.groupby(laggards.index)[fwd_col].mean()
    leader_daily = leaders.groupby(leaders.index)[fwd_col].mean()

    # Align dates
    common_dates = laggard_daily.index.intersection(leader_daily.index)
    if len(common_dates) < MIN_TRADES:
        return None

    laggard_daily = laggard_daily.loc[common_dates]
    leader_daily = leader_daily.loc[common_dates]

    # Pair return: (long laggard + short leader) / 2
    # Long laggard return = laggard_fwd
    # Short leader return = -leader_fwd
    # Net = (laggard_fwd - leader_fwd) / 2
    cost_pct = COST_BPS_RT_PAIR / 10000  # 20 bps for the pair
    pair_returns = ((laggard_daily.values - leader_daily.values) / 2) - cost_pct

    n_trades = len(pair_returns)
    if n_trades < MIN_TRADES:
        return None

    sharpe = annualized_sharpe(pair_returns)
    sort_r = sortino_ratio(pair_returns)
    pf = profit_factor(pair_returns)
    wr = (pair_returns > 0).mean() * 100
    mean_ret = np.mean(pair_returns) * 100

    # Regime analysis — use regime from the dates
    spy_df_regime = signals_df.groupby(signals_df.index)["regime"].first()
    date_regimes = spy_df_regime.reindex(common_dates)

    green_mask = (date_regimes == "GREEN").values
    red_mask = (date_regimes == "RED").values
    green_returns = pair_returns[green_mask]
    red_returns = pair_returns[red_mask]
    sharpe_green = annualized_sharpe(green_returns) if len(green_returns) >= 10 else 0
    sharpe_red = annualized_sharpe(red_returns) if len(red_returns) >= 10 else 0
    gap = regime_gap(sharpe_green, sharpe_red)

    # Year consistency
    year_series = pd.Series(common_dates).dt.year.values
    yearly_rets = pd.DataFrame({"year": year_series, "ret": pair_returns})
    yearly_avg = yearly_rets.groupby("year")["ret"].mean()
    pct_years_prof = (yearly_avg > 0).mean() * 100

    return {
        "name": name, "n_trades": n_trades,
        "sharpe": round(sharpe, 3), "sortino": round(sort_r, 3),
        "profit_factor": round(pf, 2), "win_rate": round(wr, 1),
        "mean_ret_pct": round(mean_ret, 4),
        "sharpe_green": round(sharpe_green, 3), "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(gap, 3),
        "pct_years_profitable": round(pct_years_prof, 1),
        "n_years": len(yearly_avg),
        "green_trades": int(green_mask.sum()), "red_trades": int(red_mask.sum()),
        "avg_laggards_per_day": round(len(laggards) / max(len(common_dates), 1), 1),
        "avg_leaders_per_day": round(len(leaders) / max(len(common_dates), 1), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 80)
    print(f"RELATIVE VALUE PAIRS v1 — Intra-Sector Mean Reversion")
    print(f"Stocks: {len(ALL_TICKERS)} | Sectors: {len(SECTOR_MAP)} | Data: {DATA_YEARS}y")
    print(f"Variants: {len(VARIANTS)} | Cost: {COST_BPS_RT_PAIR} bps RT (pair)")
    print("=" * 80)

    run = None
    if HAS_MLFLOW:
        run = mlflow.start_run(
            run_name=f"relative_value_pairs_v1_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
        )

    stock_data = download_data()

    # Group variants by spread_lookback (to avoid recomputing signals)
    config_variants = defaultdict(list)
    for name, spread_lb, pctile, mfi, hold in VARIANTS:
        config_variants[spread_lb].append((name, pctile, mfi, hold))

    results = []
    variant_idx = 0
    signal_cache = {}

    for spread_lb, variants in sorted(config_variants.items()):
        print(f"\n[INFO] Computing sector signals: lookback={spread_lb}d...")
        signals_df = compute_sector_signals(stock_data, spread_lb)
        signal_cache[spread_lb] = signals_df

        if signals_df is None:
            for name, pctile, mfi, hold in variants:
                variant_idx += 1
                print(f"  [{variant_idx}/{len(VARIANTS)}] {name}: no signals")
            continue

        for name, pctile, mfi, hold in variants:
            variant_idx += 1
            result = evaluate_variant(name, pctile, mfi, hold, signals_df)
            if result:
                results.append(result)
                status = (
                    "PASS" if result["regime_gap"] <= REGIME_GAP_THRESHOLD
                    and result["sharpe"] > 0
                    else "skip"
                )
                print(
                    f"  [{variant_idx}/{len(VARIANTS)}] {name}: Sharpe {result['sharpe']}, "
                    f"WR {result['win_rate']}%, RG {result['regime_gap']}, "
                    f"trades {result['n_trades']} [{status}]"
                )
            else:
                print(f"  [{variant_idx}/{len(VARIANTS)}] {name}: insufficient trades")

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
        for vname, spread_lb, pctile, mfi, hold in VARIANTS:
            if vname == name:
                break

        signals_df = signal_cache.get(spread_lb)
        if signals_df is None:
            continue

        # Reconstruct the pair returns for permutation test
        fwd_col = f"fwd_{hold}d"
        df = signals_df.dropna(subset=[fwd_col, "sector_pctile"]).copy()

        low_thresh = pctile / 100.0
        high_thresh = 1.0 - low_thresh

        laggard_mask = df["sector_pctile"] <= low_thresh
        leader_mask = df["sector_pctile"] >= high_thresh
        if mfi:
            laggard_mask &= df["mfi14"] < 30

        laggards = df[laggard_mask]
        leaders = df[leader_mask]

        laggard_daily = laggards.groupby(laggards.index)[fwd_col].mean()
        leader_daily = leaders.groupby(leaders.index)[fwd_col].mean()
        common_dates = laggard_daily.index.intersection(leader_daily.index)

        cost_pct = COST_BPS_RT_PAIR / 10000
        pair_returns = ((laggard_daily.loc[common_dates].values -
                        leader_daily.loc[common_dates].values) / 2) - cost_pct

        perm_p = permutation_test(pair_returns, NUM_PERMUTATIONS)
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
                f"RG {r['regime_gap']}, perm_p {r['perm_p']}"
            )

    # Save ALL results
    all_results = {
        "hypothesis": "Intra-sector relative value mean reversion — long laggards + short leaders within same sector cancels market beta, spread reverts",
        "variants": results,
        "top_candidates_tested": top_candidates,
        "final_pass": final_pass,
        "meta": {
            "total_variants": len(VARIANTS),
            "n_stocks": len(ALL_TICKERS),
            "n_sectors": len(SECTOR_MAP),
            "data_years": DATA_YEARS,
            "cost_bps_rt_pair": COST_BPS_RT_PAIR,
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
        mlflow.log_param("n_stocks", len(ALL_TICKERS))
        mlflow.log_param("n_variants", len(VARIANTS))
        mlflow.log_param("n_sectors", len(SECTOR_MAP))
        mlflow.log_param("hypothesis", "intra_sector_relative_value_mean_reversion")
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
        mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
        mlflow.end_run()

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
