#!/usr/bin/env python3
"""
Cross-Sectional Anomaly Scanner — GPU-Accelerated Observation-First Discovery
==============================================================================
Scans a large stock universe (S&P 500 / Russell 1000) for surprising cross-
sectional patterns. No backtesting, no parameter sweeps, no strategy
construction — pure data observation.

Designed for Neptune RTX 3090 but falls back to CPU gracefully.

Scans:
  1. Overnight vs Intraday Return Decomposition
  2. Return Autocorrelation Clustering
  3. Tail Dependence Asymmetry
  4. Intra-Week Seasonality
  5. Volatility-of-Volatility (Vol-of-Vol)
  6. Lead-Lag Network
  7. Skewness Pricing
  8. Liquidity Dry-Up Events

Usage:
    python3 cross_sectional_anomaly_scanner.py [--no-cache] [--years 5] [--universe sp500]

Author: Claude Opus 4.6 / Teleclaude Research
HC #735 — Observation-first, NOT brute-force.
"""

import argparse
import datetime as dt
import json
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        DEVICE = torch.device("cuda")
        print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("[CPU] CUDA not available, falling back to numpy")
except ImportError:
    print("[CPU] PyTorch not installed, using numpy only")
    torch = None
    DEVICE = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_PATH = Path("/home/nick/Lvl3Quant/data/sp500_cache.pkl")
OUTPUT_JSON = Path("/home/nick/Lvl3Quant/output/cross_sectional_observations.json")
SECTOR_MAP_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ---------------------------------------------------------------------------
# Universe fetching
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> tuple:
    """Fetch S&P 500 constituents + sector mapping from Wikipedia."""
    try:
        tables = pd.read_html(SECTOR_MAP_URL)
        df = tables[0]
        sym_col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
        sec_col = [c for c in df.columns if "gics" in c.lower() and "sector" in c.lower()][0]
        tickers = df[sym_col].str.replace(".", "-", regex=False).tolist()
        sector_map = dict(zip(tickers, df[sec_col].tolist()))
        print(f"[Universe] Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers, sector_map
    except Exception as e:
        print(f"[Universe] Wikipedia fetch failed ({e}), using fallback list")
        return _fallback_tickers()


def get_russell1000_tickers() -> tuple:
    """Russell 1000 approximation: S&P 500 + mid-caps from IWB holdings."""
    sp_tickers, sp_sectors = get_sp500_tickers()
    # Supplement with well-known mid-caps to approach 1000 stocks
    midcap_extra = [
        "ABNB", "AFRM", "AI", "APP", "BILL", "CFLT", "COIN", "CRWD",
        "DASH", "DDOG", "DUOL", "ESTC", "FIVE", "GLOB", "GTLB", "HUBS",
        "IOT", "LPLA", "MANH", "MKTX", "MDB", "NET", "OKTA", "PCOR",
        "PINS", "RBLX", "ROKU", "S", "SHOP", "SNAP", "SQ", "TEAM",
        "TOST", "TTD", "TWLO", "U", "VEEV", "W", "WDAY", "ZI", "ZS",
        "PATH", "CELH", "CAVA", "BIRK", "CORT", "DUOL", "ELF", "FND",
        "GFS", "HWM", "IBKR", "KNSL", "LULU", "MELI", "ONON", "PLTR",
        "SMCI", "SPOT", "TW", "UBER", "WING",
    ]
    extra = [t for t in midcap_extra if t not in sp_tickers]
    all_tickers = sp_tickers + extra
    sector_map = {**sp_sectors, **{t: "Unknown" for t in extra}}
    print(f"[Universe] Extended to {len(all_tickers)} tickers (SP500 + mid-cap supplement)")
    return all_tickers, sector_map


def _fallback_tickers():
    """Minimal fallback — top ~100 liquid names."""
    tickers = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
        "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "LLY",
        "MRK", "ABBV", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
        "ACN", "CRM", "ABT", "TMO", "DHR", "LIN", "NKE", "NEE", "PM",
        "TXN", "RTX", "UPS", "HON", "LOW", "UNP", "AMGN", "IBM", "QCOM",
        "GE", "CAT", "INTU", "BA", "AMAT", "SBUX", "GS", "BLK", "ADP",
        "MDLZ", "DE", "ISRG", "GILD", "ADI", "BKNG", "SYK", "VRTX",
        "REGN", "MMC", "LRCX", "CI", "ZTS", "CB", "SCHW", "MO", "TMUS",
        "SO", "DUK", "PLD", "CME", "BDX", "CL", "ICE", "EOG", "SLB",
        "PNC", "USB", "TFC", "WM", "EMR", "FDX", "APD", "MCK", "SRE",
        "PSA", "CCI", "SPG", "AEP", "D", "EXC", "XEL", "WEC", "ES",
    ]
    return tickers, {t: "Unknown" for t in tickers}


# ---------------------------------------------------------------------------
# Data download + caching
# ---------------------------------------------------------------------------

def download_data(tickers: list, sector_map: dict, years: int = 5,
                  use_cache: bool = True) -> dict:
    """Download OHLCV data for all tickers. Cache result."""
    if use_cache and CACHE_PATH.exists():
        age_hours = (time.time() - CACHE_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"[Cache] Loading from {CACHE_PATH} (age: {age_hours:.1f}h)")
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if len(cached.get("prices", {})) >= 400:
                print(f"[Cache] {len(cached['prices'])} stocks loaded")
                return cached
            else:
                print(f"[Cache] Only {len(cached.get('prices', {}))} stocks, re-downloading")

    import yfinance as yf

    end = dt.datetime.now()
    start = end - dt.timedelta(days=years * 365)

    prices = {}
    volumes = {}
    failed = []

    # Download in batches to avoid rate limits
    batch_size = 50
    for i in tqdm(range(0, len(tickers), batch_size), desc="Downloading batches"):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch, start=start, end=end,
                group_by="ticker", auto_adjust=False,
                progress=False, threads=True
            )
            if len(batch) == 1:
                t = batch[0]
                if len(data) > 100:
                    prices[t] = data[["Open", "High", "Low", "Close", "Adj Close"]].copy()
                    volumes[t] = data["Volume"].copy()
                else:
                    failed.append(t)
            else:
                for t in batch:
                    try:
                        df_t = data[t].dropna(how="all")
                        if len(df_t) > 100:
                            prices[t] = df_t[["Open", "High", "Low", "Close", "Adj Close"]].copy()
                            volumes[t] = df_t["Volume"].copy()
                        else:
                            failed.append(t)
                    except (KeyError, TypeError):
                        failed.append(t)
        except Exception as e:
            print(f"[Download] Batch {i // batch_size} failed: {e}")
            failed.extend(batch)

        # Brief pause between batches
        if i + batch_size < len(tickers):
            time.sleep(0.5)

    print(f"[Download] Got {len(prices)} stocks, {len(failed)} failed")

    result = {
        "prices": prices,
        "volumes": volumes,
        "sector_map": sector_map,
        "download_date": str(dt.date.today()),
        "years": years,
    }

    # Cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"[Cache] Saved to {CACHE_PATH}")

    return result


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def build_return_matrices(prices: dict) -> dict:
    """Build aligned return matrices from price dict."""
    # Build close-to-close returns
    close_frames = {}
    open_frames = {}
    for t, df in prices.items():
        close_frames[t] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        open_frames[t] = df["Open"]

    close_df = pd.DataFrame(close_frames)
    open_df = pd.DataFrame(open_frames)

    # Align indices
    common_idx = close_df.index.intersection(open_df.index)
    close_df = close_df.loc[common_idx]
    open_df = open_df.loc[common_idx]

    # Drop stocks with too many NaNs (>20%)
    valid_mask = close_df.notna().mean() > 0.80
    close_df = close_df.loc[:, valid_mask]
    open_df = open_df.loc[:, close_df.columns.intersection(open_df.columns)]

    # Forward-fill small gaps
    close_df = close_df.ffill(limit=5)
    open_df = open_df.ffill(limit=5)

    # Returns
    total_ret = close_df.pct_change()
    overnight_ret = (open_df / close_df.shift(1)) - 1  # close-to-open
    intraday_ret = (close_df / open_df) - 1  # open-to-close

    return {
        "close": close_df,
        "open": open_df,
        "total_ret": total_ret,
        "overnight_ret": overnight_ret,
        "intraday_ret": intraday_ret,
    }


def surprise_score(effect_size: float, p_value: float, sample_size: int) -> float:
    """Compute a surprise score: higher = more interesting finding."""
    # Combine effect size, significance, and sample size
    if p_value <= 0 or np.isnan(p_value):
        p_value = 1e-15
    significance = min(-np.log10(p_value), 15)  # cap at 15
    n_factor = np.log10(max(sample_size, 1))
    return round(abs(effect_size) * significance * n_factor, 3)


def gpu_corr_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    """Compute correlation matrix on GPU if available, else CPU."""
    mat = returns_df.values.copy()
    # Replace NaN with 0 for correlation computation
    nan_mask = np.isnan(mat)
    mat[nan_mask] = 0.0

    if USE_GPU and torch is not None:
        t = torch.tensor(mat, dtype=torch.float32, device=DEVICE)
        # Demean
        means = t.mean(dim=0, keepdim=True)
        t = t - means
        # Zero out where original was NaN
        mask_t = torch.tensor(~nan_mask, dtype=torch.float32, device=DEVICE)
        t = t * mask_t
        # Correlation
        norms = torch.sqrt((t ** 2).sum(dim=0, keepdim=True))
        norms = torch.clamp(norms, min=1e-8)
        t_normed = t / norms
        corr = (t_normed.T @ t_normed).cpu().numpy()
        np.fill_diagonal(corr, 1.0)
        return corr
    else:
        # CPU fallback
        return np.corrcoef(mat.T)


# ---------------------------------------------------------------------------
# SCAN 1: Overnight vs Intraday Return Decomposition
# ---------------------------------------------------------------------------

def scan_overnight_vs_intraday(ret_data: dict, sector_map: dict) -> dict:
    """Decompose returns into overnight and intraday components."""
    print("\n" + "=" * 70)
    print("SCAN 1: Overnight vs Intraday Return Decomposition")
    print("=" * 70)

    overnight = ret_data["overnight_ret"].dropna(how="all")
    intraday = ret_data["intraday_ret"].dropna(how="all")
    tickers = overnight.columns.tolist()

    # 1a. Which stocks have persistent overnight alpha?
    on_mean = overnight.mean()
    id_mean = intraday.mean()
    on_sharpe = overnight.mean() / overnight.std() * np.sqrt(252)
    id_sharpe = intraday.mean() / intraday.std() * np.sqrt(252)

    # Annualized overnight vs intraday contribution
    on_annual = on_mean * 252
    id_annual = id_mean * 252

    # Top 20 overnight alpha stocks
    top_overnight = on_sharpe.nlargest(20)
    bot_overnight = on_sharpe.nsmallest(20)

    # 1b. Is overnight return predictive of next-day intraday?
    pred_corrs = {}
    for t in tickers:
        on_t = overnight[t].dropna()
        id_t = intraday[t].shift(-1).loc[on_t.index].dropna()
        common = on_t.index.intersection(id_t.index)
        if len(common) > 100:
            c, p = stats.pearsonr(on_t.loc[common], id_t.loc[common])
            pred_corrs[t] = {"corr": c, "pval": p, "n": len(common)}

    pred_df = pd.DataFrame(pred_corrs).T
    # Stocks where overnight predicts intraday reversal
    reversal_stocks = pred_df[
        (pred_df["corr"] < -0.05) & (pred_df["pval"] < 0.05)
    ].sort_values("corr")

    # 1c. Cross-sectional: when many stocks gap up, does intraday reverse?
    on_cross_mean = overnight.mean(axis=1)  # average overnight return across all stocks
    id_cross_mean = intraday.mean(axis=1)
    common_idx = on_cross_mean.dropna().index.intersection(id_cross_mean.dropna().index)
    cross_corr, cross_pval = stats.pearsonr(
        on_cross_mean.loc[common_idx], id_cross_mean.loc[common_idx]
    )

    # Quintile analysis: sort days by overnight gap, look at intraday
    on_quintiles = pd.qcut(on_cross_mean.loc[common_idx], 5, labels=False, duplicates="drop")
    id_by_quintile = id_cross_mean.loc[common_idx].groupby(on_quintiles).mean()

    findings = {
        "name": "Overnight vs Intraday Return Decomposition",
        "universe_size": len(tickers),
        "observations": {
            "overnight_alpha": {
                "finding": "Stocks with highest overnight Sharpe (annualized)",
                "top_5": {t: round(v, 3) for t, v in top_overnight.head(5).items()},
                "bottom_5": {t: round(v, 3) for t, v in bot_overnight.head(5).items()},
                "mean_overnight_sharpe": round(on_sharpe.mean(), 4),
                "mean_intraday_sharpe": round(id_sharpe.mean(), 4),
                "pct_positive_overnight_sharpe": round((on_sharpe > 0).mean() * 100, 1),
            },
            "overnight_predicts_intraday": {
                "finding": f"Overnight gap predicts next-day intraday for {len(reversal_stocks)} stocks (p<0.05, negative corr)",
                "sample_reversal_stocks": {
                    t: round(row["corr"], 4)
                    for t, row in reversal_stocks.head(10).iterrows()
                },
                "mean_predictive_corr": round(pred_df["corr"].mean(), 4) if len(pred_df) > 0 else None,
                "n_stocks_analyzed": len(pred_df),
            },
            "cross_sectional_reversal": {
                "finding": f"When all stocks gap {'up' if cross_corr < 0 else 'in same direction'}, intraday tends to {'reverse' if cross_corr < 0 else 'continue'}",
                "correlation": round(cross_corr, 4),
                "p_value": round(cross_pval, 6),
                "sample_size": len(common_idx),
                "intraday_by_overnight_quintile": {
                    f"Q{int(k)+1}": round(v * 10000, 2)
                    for k, v in id_by_quintile.items()
                },
            },
        },
        "surprise_score": surprise_score(cross_corr, cross_pval, len(common_idx)),
    }

    # Print summary
    print(f"\n  Universe: {len(tickers)} stocks")
    print(f"  Mean overnight Sharpe: {on_sharpe.mean():.4f}")
    print(f"  Mean intraday Sharpe:  {id_sharpe.mean():.4f}")
    print(f"  {(on_sharpe > 0).mean()*100:.1f}% of stocks have positive overnight Sharpe")
    print(f"\n  Overnight->Intraday reversal: corr={cross_corr:.4f}, p={cross_pval:.6f}")
    print(f"  {len(reversal_stocks)} stocks show significant overnight-to-intraday reversal")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# SCAN 2: Return Autocorrelation Clustering
# ---------------------------------------------------------------------------

def scan_autocorrelation_clustering(ret_data: dict) -> dict:
    """Cluster stocks by their autocorrelation profiles."""
    print("\n" + "=" * 70)
    print("SCAN 2: Return Autocorrelation Clustering")
    print("=" * 70)

    total_ret = ret_data["total_ret"].dropna(how="all")
    tickers = total_ret.columns.tolist()

    # Compute lag-1 through lag-5 autocorrelation for each stock
    ac_profiles = {}
    for t in tqdm(tickers, desc="  Computing autocorrelations"):
        r = total_ret[t].dropna()
        if len(r) < 100:
            continue
        acs = []
        for lag in range(1, 6):
            ac = r.autocorr(lag=lag)
            acs.append(ac if not np.isnan(ac) else 0.0)
        ac_profiles[t] = acs

    ac_df = pd.DataFrame(ac_profiles, index=[f"lag_{i}" for i in range(1, 6)]).T
    n_stocks = len(ac_df)

    # Cluster using hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(ac_df.values, method="ward")
    n_clusters = 5
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    ac_df["cluster"] = labels

    cluster_profiles = {}
    cluster_sizes = {}
    for c in range(1, n_clusters + 1):
        mask = ac_df["cluster"] == c
        cluster_sizes[c] = int(mask.sum())
        profile = ac_df.loc[mask, [f"lag_{i}" for i in range(1, 6)]].mean()
        cluster_profiles[c] = {col: round(val, 4) for col, val in profile.items()}

    # Characterize clusters
    cluster_types = {}
    for c, prof in cluster_profiles.items():
        lag1 = prof["lag_1"]
        if lag1 < -0.05:
            cluster_types[c] = "Mean-Reverting"
        elif lag1 > 0.05:
            cluster_types[c] = "Trending"
        else:
            cluster_types[c] = "Random Walk"

    # Stability test: split data in half, re-cluster, measure membership overlap
    half = len(total_ret) // 2
    first_half = total_ret.iloc[:half]
    second_half = total_ret.iloc[half:]

    ac_first = {}
    ac_second = {}
    for t in ac_df.index:
        r1 = first_half[t].dropna()
        r2 = second_half[t].dropna()
        if len(r1) > 50 and len(r2) > 50:
            ac_first[t] = [r1.autocorr(lag=l) or 0 for l in range(1, 6)]
            ac_second[t] = [r2.autocorr(lag=l) or 0 for l in range(1, 6)]

    common_tickers = list(set(ac_first.keys()) & set(ac_second.keys()))
    if len(common_tickers) > 50:
        df1 = pd.DataFrame(ac_first, index=[f"lag_{i}" for i in range(1, 6)]).T.loc[common_tickers]
        df2 = pd.DataFrame(ac_second, index=[f"lag_{i}" for i in range(1, 6)]).T.loc[common_tickers]
        Z1 = linkage(df1.values, method="ward")
        Z2 = linkage(df2.values, method="ward")
        l1 = fcluster(Z1, t=n_clusters, criterion="maxclust")
        l2 = fcluster(Z2, t=n_clusters, criterion="maxclust")
        # Adjusted Rand Index for stability
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(l1, l2)
    else:
        ari = None

    # Mean return by cluster
    cluster_returns = {}
    for c in range(1, n_clusters + 1):
        members = ac_df[ac_df["cluster"] == c].index.tolist()
        if members:
            mean_ret = total_ret[members].mean().mean() * 252
            cluster_returns[c] = round(mean_ret * 100, 2)

    findings = {
        "name": "Return Autocorrelation Clustering",
        "universe_size": n_stocks,
        "observations": {
            "cluster_profiles": {
                str(c): {
                    "type": cluster_types.get(c, "Unknown"),
                    "size": cluster_sizes[c],
                    "autocorrelation_profile": cluster_profiles[c],
                    "annualized_return_pct": cluster_returns.get(c),
                }
                for c in range(1, n_clusters + 1)
            },
            "stability": {
                "finding": f"Cluster membership stability (Adjusted Rand Index): {ari:.3f}" if ari else "Insufficient data for stability test",
                "ari": round(ari, 4) if ari else None,
                "interpretation": (
                    "Very stable" if ari and ari > 0.5
                    else "Moderately stable" if ari and ari > 0.2
                    else "Unstable — clusters shift over time" if ari
                    else "N/A"
                ),
            },
            "mean_lag1_autocorrelation": round(ac_df["lag_1"].mean() if "lag_1" in ac_df.columns else 0, 4),
            "pct_mean_reverting": round((ac_df["lag_1"] < -0.03).mean() * 100, 1) if "lag_1" in ac_df.columns else None,
            "pct_trending": round((ac_df["lag_1"] > 0.03).mean() * 100, 1) if "lag_1" in ac_df.columns else None,
        },
        "surprise_score": surprise_score(
            ac_df["lag_1"].mean() if "lag_1" in ac_df.columns else 0,
            0.01 if ari and ari > 0.2 else 0.5,
            n_stocks
        ),
    }

    print(f"\n  Clustered {n_stocks} stocks into {n_clusters} groups")
    for c in range(1, n_clusters + 1):
        print(f"    Cluster {c} ({cluster_types.get(c, '?')}): {cluster_sizes[c]} stocks, "
              f"lag-1 AC={cluster_profiles[c]['lag_1']:.4f}, "
              f"ann. return={cluster_returns.get(c, 'N/A')}%")
    if ari is not None:
        print(f"\n  Stability (ARI): {ari:.3f} — {'stable' if ari > 0.3 else 'unstable'}")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# SCAN 3: Tail Dependence Asymmetry
# ---------------------------------------------------------------------------

def scan_tail_dependence(ret_data: dict, sector_map: dict) -> dict:
    """Measure whether stock pairs crash together more than they rally together."""
    print("\n" + "=" * 70)
    print("SCAN 3: Tail Dependence Asymmetry")
    print("=" * 70)

    total_ret = ret_data["total_ret"].dropna(how="all")
    # Use top 100 most liquid for computational tractability
    vol_available = total_ret.std().dropna().nlargest(200)
    tickers = vol_available.index.tolist()[:100]
    ret_sub = total_ret[tickers].dropna()

    n = len(tickers)
    threshold = 0.10  # 10th/90th percentile for tails

    # Compute tail dependence for each pair
    # Lower tail: P(Y < q_low | X < q_low)
    # Upper tail: P(Y > q_high | X > q_high)
    lower_tail_deps = []
    upper_tail_deps = []
    pair_asymmetries = []

    # Use GPU for rank transformation if available
    if USE_GPU and torch is not None:
        mat = torch.tensor(ret_sub.values, dtype=torch.float32, device=DEVICE)
        # Rank transform (approximate via sorting)
        n_obs = mat.shape[0]
        ranks = torch.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            sorted_idx = torch.argsort(col)
            r = torch.zeros(n_obs, device=DEVICE)
            r[sorted_idx] = torch.arange(n_obs, dtype=torch.float32, device=DEVICE) / n_obs
            ranks[:, j] = r

        low_thresh = threshold
        high_thresh = 1.0 - threshold
        low_mask = (ranks < low_thresh)  # (T, N)
        high_mask = (ranks > high_thresh)

        # For efficiency, compute a subset of pairs
        n_pairs_sample = min(500, n * (n - 1) // 2)
        rng = np.random.RandomState(42)
        pairs = []
        pair_set = set()
        while len(pairs) < n_pairs_sample:
            i, j = rng.randint(0, n, size=2)
            if i != j and (i, j) not in pair_set:
                pair_set.add((i, j))
                pairs.append((i, j))

        for i_idx, j_idx in tqdm(pairs, desc="  Computing tail dependence"):
            low_i = low_mask[:, i_idx]
            low_j = low_mask[:, j_idx]
            high_i = high_mask[:, i_idx]
            high_j = high_mask[:, j_idx]

            n_low_i = low_i.sum().item()
            n_high_i = high_i.sum().item()

            if n_low_i > 5 and n_high_i > 5:
                ltd = (low_i & low_j).sum().item() / n_low_i
                utd = (high_i & high_j).sum().item() / n_high_i
                lower_tail_deps.append(ltd)
                upper_tail_deps.append(utd)
                pair_asymmetries.append(ltd - utd)
    else:
        # CPU version with numpy ranks
        from scipy.stats import rankdata
        ranked = np.apply_along_axis(rankdata, 0, ret_sub.values) / len(ret_sub)

        n_pairs_sample = min(500, n * (n - 1) // 2)
        rng = np.random.RandomState(42)
        pairs = []
        pair_set = set()
        while len(pairs) < n_pairs_sample:
            i, j = rng.randint(0, n, size=2)
            if i != j and (i, j) not in pair_set:
                pair_set.add((i, j))
                pairs.append((i, j))

        for i_idx, j_idx in tqdm(pairs, desc="  Computing tail dependence"):
            low_i = ranked[:, i_idx] < threshold
            low_j = ranked[:, j_idx] < threshold
            high_i = ranked[:, i_idx] > (1 - threshold)
            high_j = ranked[:, j_idx] > (1 - threshold)

            n_low = low_i.sum()
            n_high = high_i.sum()

            if n_low > 5 and n_high > 5:
                ltd = (low_i & low_j).sum() / n_low
                utd = (high_i & high_j).sum() / n_high
                lower_tail_deps.append(float(ltd))
                upper_tail_deps.append(float(utd))
                pair_asymmetries.append(float(ltd - utd))

    ltd_arr = np.array(lower_tail_deps)
    utd_arr = np.array(upper_tail_deps)
    asym_arr = np.array(pair_asymmetries)

    # Statistical test: is asymmetry significantly different from zero?
    if len(asym_arr) > 10:
        t_stat, p_val = stats.ttest_1samp(asym_arr, 0)
    else:
        t_stat, p_val = 0, 1

    # Sector-level tail asymmetry
    sector_asym = defaultdict(list)
    for idx, (i_idx, j_idx) in enumerate(pairs[:len(asym_arr)]):
        t_i = tickers[i_idx]
        t_j = tickers[j_idx]
        s_i = sector_map.get(t_i, "Unknown")
        s_j = sector_map.get(t_j, "Unknown")
        if s_i == s_j and s_i != "Unknown":
            sector_asym[s_i].append(asym_arr[idx])

    sector_results = {}
    for sec, vals in sector_asym.items():
        if len(vals) >= 5:
            sector_results[sec] = {
                "mean_asymmetry": round(np.mean(vals), 4),
                "n_pairs": len(vals),
            }

    findings = {
        "name": "Tail Dependence Asymmetry",
        "universe_size": len(tickers),
        "n_pairs_analyzed": len(asym_arr),
        "observations": {
            "overall": {
                "finding": (
                    f"Stocks crash together MORE than they rally together"
                    if np.mean(asym_arr) > 0
                    else f"Stocks rally together MORE than they crash together"
                ),
                "mean_lower_tail_dep": round(float(ltd_arr.mean()), 4),
                "mean_upper_tail_dep": round(float(utd_arr.mean()), 4),
                "mean_asymmetry": round(float(asym_arr.mean()), 4),
                "t_statistic": round(float(t_stat), 3),
                "p_value": round(float(p_val), 6),
                "sample_size": len(asym_arr),
                "pct_pairs_crash_more": round(float((asym_arr > 0).mean() * 100), 1),
            },
            "by_sector": dict(sorted(
                sector_results.items(),
                key=lambda x: abs(x[1]["mean_asymmetry"]),
                reverse=True
            )[:8]),
        },
        "surprise_score": surprise_score(float(asym_arr.mean()), float(p_val), len(asym_arr)),
    }

    print(f"\n  Analyzed {len(asym_arr)} stock pairs")
    print(f"  Mean lower tail dependence: {ltd_arr.mean():.4f}")
    print(f"  Mean upper tail dependence: {utd_arr.mean():.4f}")
    print(f"  Asymmetry (lower - upper):  {asym_arr.mean():.4f} (t={t_stat:.2f}, p={p_val:.6f})")
    print(f"  {(asym_arr > 0).mean()*100:.1f}% of pairs crash together more than rally together")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# SCAN 4: Intra-Week Seasonality
# ---------------------------------------------------------------------------

def scan_intraweek_seasonality(ret_data: dict) -> dict:
    """Day-of-week return patterns per stock."""
    print("\n" + "=" * 70)
    print("SCAN 4: Intra-Week Seasonality")
    print("=" * 70)

    total_ret = ret_data["total_ret"].dropna(how="all")
    tickers = total_ret.columns.tolist()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Add day of week
    total_ret_copy = total_ret.copy()
    total_ret_copy["dow"] = total_ret_copy.index.dayofweek

    # Cross-sectional: average return by day of week
    cross_day = total_ret_copy.groupby("dow")[tickers].mean().mean(axis=1)
    cross_day_std = total_ret_copy.groupby("dow")[tickers].std().mean(axis=1)

    # Per-stock day-of-week analysis
    stock_best_day = {}
    stock_worst_day = {}
    significant_patterns = 0
    total_tested = 0

    for t in tickers:
        r = total_ret[[t]].copy().dropna()
        r["dow"] = r.index.dayofweek
        day_means = r.groupby("dow")[t].mean()
        day_counts = r.groupby("dow")[t].count()

        if len(day_means) < 5 or day_counts.min() < 20:
            continue

        total_tested += 1

        # ANOVA test: are day-of-week returns significantly different?
        groups = [r[r["dow"] == d][t].values for d in range(5) if len(r[r["dow"] == d]) > 10]
        if len(groups) == 5:
            f_stat, p_val = stats.f_oneway(*groups)
            if p_val < 0.05:
                significant_patterns += 1

        best = day_means.idxmax()
        worst = day_means.idxmin()
        stock_best_day[t] = day_names[best]
        stock_worst_day[t] = day_names[worst]

    # Count how often each day is "best" or "worst"
    best_counts = pd.Series(stock_best_day).value_counts()
    worst_counts = pd.Series(stock_worst_day).value_counts()

    # Monday effect test
    mon_rets = total_ret_copy[total_ret_copy["dow"] == 0][tickers].mean(axis=1)
    other_rets = total_ret_copy[total_ret_copy["dow"] != 0][tickers].mean(axis=1)
    mon_t, mon_p = stats.ttest_ind(mon_rets.dropna(), other_rets.dropna())

    # Persistence test: first half vs second half day-of-week rankings
    half = len(total_ret_copy) // 2
    first_cross = total_ret_copy.iloc[:half].groupby("dow")[tickers].mean().mean(axis=1)
    second_cross = total_ret_copy.iloc[half:].groupby("dow")[tickers].mean().mean(axis=1)
    rank_corr, rank_p = stats.spearmanr(first_cross.values, second_cross.values)

    findings = {
        "name": "Intra-Week Seasonality",
        "universe_size": len(tickers),
        "observations": {
            "cross_sectional_day_returns_bps": {
                day_names[d]: round(cross_day.get(d, 0) * 10000, 2)
                for d in range(5)
            },
            "monday_effect": {
                "finding": f"Monday returns are {'lower' if mon_rets.mean() < other_rets.mean() else 'higher'} than other days",
                "monday_mean_bps": round(mon_rets.mean() * 10000, 2),
                "other_days_mean_bps": round(other_rets.mean() * 10000, 2),
                "t_statistic": round(float(mon_t), 3),
                "p_value": round(float(mon_p), 6),
            },
            "per_stock_patterns": {
                "n_tested": total_tested,
                "n_significant_anova_p05": significant_patterns,
                "pct_significant": round(significant_patterns / max(total_tested, 1) * 100, 1),
                "most_common_best_day": best_counts.head(3).to_dict() if len(best_counts) > 0 else {},
                "most_common_worst_day": worst_counts.head(3).to_dict() if len(worst_counts) > 0 else {},
            },
            "persistence": {
                "finding": f"Day-of-week rankings are {'persistent' if rank_corr > 0.5 else 'NOT persistent'} across halves",
                "rank_correlation": round(float(rank_corr), 3),
                "p_value": round(float(rank_p), 4),
            },
        },
        "surprise_score": surprise_score(
            float(mon_rets.mean() - other_rets.mean()),
            float(mon_p),
            len(mon_rets)
        ),
    }

    print(f"\n  Cross-sectional day-of-week returns (bps):")
    for d in range(5):
        print(f"    {day_names[d]:12s}: {cross_day.get(d, 0)*10000:+.2f} bps")
    print(f"\n  Monday effect: t={mon_t:.3f}, p={mon_p:.6f}")
    print(f"  {significant_patterns}/{total_tested} stocks show significant DoW patterns (p<0.05)")
    print(f"  Persistence rank corr: {rank_corr:.3f}")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# SCAN 5: Volatility-of-Volatility (Vol-of-Vol)
# ---------------------------------------------------------------------------

def scan_vol_of_vol(ret_data: dict) -> dict:
    """Analyze vol-of-vol and its relationship to tail risk and drawdowns."""
    print("\n" + "=" * 70)
    print("SCAN 5: Volatility-of-Volatility (Vol-of-Vol)")
    print("=" * 70)

    total_ret = ret_data["total_ret"].dropna(how="all")
    tickers = total_ret.columns.tolist()

    # Compute rolling 21-day volatility for each stock
    rolling_vol = total_ret.rolling(21).std() * np.sqrt(252)

    # Vol-of-vol: std of rolling vol
    volvol = {}
    kurtosis_vals = {}
    max_dd = {}

    for t in tqdm(tickers, desc="  Computing vol-of-vol"):
        rv = rolling_vol[t].dropna()
        r = total_ret[t].dropna()
        if len(rv) < 100 or len(r) < 100:
            continue

        volvol[t] = rv.std()
        kurtosis_vals[t] = stats.kurtosis(r.values, fisher=True)

        # Max drawdown
        cum = (1 + r).cumprod()
        running_max = cum.cummax()
        dd = (cum / running_max - 1)
        max_dd[t] = dd.min()

    vv_series = pd.Series(volvol)
    kurt_series = pd.Series(kurtosis_vals)
    dd_series = pd.Series(max_dd)

    # Quintile analysis
    vv_quintile = pd.qcut(vv_series, 5, labels=False, duplicates="drop")
    tickers_by_q = {}
    for q in range(5):
        q_tickers = vv_quintile[vv_quintile == q].index.tolist()
        tickers_by_q[q] = q_tickers

    # Compare high vs low volvol quintiles
    quintile_stats = {}
    for q in range(5):
        q_tickers = tickers_by_q.get(q, [])
        if not q_tickers:
            continue
        q_kurt = kurt_series.loc[kurt_series.index.isin(q_tickers)].mean()
        q_dd = dd_series.loc[dd_series.index.isin(q_tickers)].mean()
        q_ret = total_ret[q_tickers].mean().mean() * 252
        quintile_stats[f"Q{q+1}"] = {
            "n_stocks": len(q_tickers),
            "mean_kurtosis": round(float(q_kurt), 3),
            "mean_max_drawdown_pct": round(float(q_dd * 100), 1),
            "annualized_return_pct": round(float(q_ret * 100), 2),
        }

    # Correlation: volvol vs kurtosis
    common = vv_series.index.intersection(kurt_series.index)
    vv_kurt_corr, vv_kurt_p = stats.pearsonr(vv_series.loc[common], kurt_series.loc[common])

    # Correlation: volvol vs max drawdown
    common2 = vv_series.index.intersection(dd_series.index)
    vv_dd_corr, vv_dd_p = stats.pearsonr(vv_series.loc[common2], dd_series.loc[common2])

    # Forward-looking test: high volvol predicts future drawdown?
    # Split into first 60% and last 40%
    split = int(len(total_ret) * 0.6)
    first_ret = total_ret.iloc[:split]
    second_ret = total_ret.iloc[split:]

    first_vol = first_ret.rolling(21).std().iloc[-63:].mean() * np.sqrt(252)  # last quarter of first half
    first_volvol = first_ret.rolling(21).std().std()

    second_dd = {}
    for t in tickers:
        r2 = second_ret[t].dropna()
        if len(r2) > 50:
            cum = (1 + r2).cumprod()
            second_dd[t] = (cum / cum.cummax() - 1).min()

    common3 = first_volvol.dropna().index.intersection(pd.Series(second_dd).index)
    if len(common3) > 50:
        pred_corr, pred_p = stats.pearsonr(
            first_volvol.loc[common3],
            pd.Series(second_dd).loc[common3]
        )
    else:
        pred_corr, pred_p = 0, 1

    findings = {
        "name": "Volatility-of-Volatility (Vol-of-Vol)",
        "universe_size": len(volvol),
        "observations": {
            "quintile_analysis": quintile_stats,
            "volvol_vs_kurtosis": {
                "finding": f"Vol-of-vol {'strongly' if abs(vv_kurt_corr) > 0.3 else 'weakly'} correlates with excess kurtosis (fat tails)",
                "correlation": round(float(vv_kurt_corr), 4),
                "p_value": round(float(vv_kurt_p), 6),
            },
            "volvol_vs_drawdown": {
                "finding": f"High vol-of-vol stocks have {'deeper' if vv_dd_corr < 0 else 'similar'} drawdowns",
                "correlation": round(float(vv_dd_corr), 4),
                "p_value": round(float(vv_dd_p), 6),
            },
            "predictive_power": {
                "finding": f"Past vol-of-vol {'predicts' if pred_p < 0.05 else 'does NOT predict'} future drawdowns",
                "correlation": round(float(pred_corr), 4),
                "p_value": round(float(pred_p), 6),
                "sample_size": len(common3),
            },
        },
        "surprise_score": surprise_score(float(vv_kurt_corr), float(vv_kurt_p), len(common)),
    }

    print(f"\n  {len(volvol)} stocks analyzed")
    print(f"\n  Quintile analysis (Q1=low volvol, Q5=high volvol):")
    for q, s in quintile_stats.items():
        print(f"    {q}: kurtosis={s['mean_kurtosis']:.2f}, max_dd={s['mean_max_drawdown_pct']:.1f}%, "
              f"ann_ret={s['annualized_return_pct']:.2f}%")
    print(f"\n  Vol-of-vol vs kurtosis: r={vv_kurt_corr:.4f}, p={vv_kurt_p:.6f}")
    print(f"  Vol-of-vol vs drawdown: r={vv_dd_corr:.4f}, p={vv_dd_p:.6f}")
    print(f"  Predictive (past volvol -> future DD): r={pred_corr:.4f}, p={pred_p:.6f}")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# SCAN 6: Lead-Lag Network
# ---------------------------------------------------------------------------

def scan_lead_lag_network(ret_data: dict, sector_map: dict) -> dict:
    """Build a lead-lag network from cross-correlations."""
    print("\n" + "=" * 70)
    print("SCAN 6: Lead-Lag Network")
    print("=" * 70)

    total_ret = ret_data["total_ret"].dropna(how="all")
    # Use top 100 stocks by data completeness
    completeness = total_ret.notna().mean().nlargest(100)
    tickers = completeness.index.tolist()
    ret_sub = total_ret[tickers].fillna(0)

    n = len(tickers)
    lead_scores = np.zeros(n)  # positive = leads others
    lag_scores = np.zeros(n)

    # For each pair, compute cross-correlation at lags 1-5
    # A stock "leads" if its lag-k return correlates with the other's current return
    n_pairs_sample = min(1000, n * (n - 1) // 2)
    rng = np.random.RandomState(42)
    pairs = []
    pair_set = set()
    while len(pairs) < n_pairs_sample:
        i, j = rng.randint(0, n, size=2)
        if i != j and (i, j) not in pair_set:
            pair_set.add((i, j))
            pairs.append((i, j))

    significant_leads = defaultdict(int)

    if USE_GPU and torch is not None:
        mat = torch.tensor(ret_sub.values, dtype=torch.float32, device=DEVICE)
        T = mat.shape[0]

        for i_idx, j_idx in tqdm(pairs, desc="  Computing lead-lag"):
            best_lag = 0
            best_corr = 0
            for lag in range(1, 6):
                # Does stock i at time t-lag predict stock j at time t?
                x = mat[:-lag, i_idx]
                y = mat[lag:, j_idx]
                # Pearson correlation
                xm = x - x.mean()
                ym = y - y.mean()
                num = (xm * ym).sum()
                den = torch.sqrt((xm ** 2).sum() * (ym ** 2).sum())
                if den > 1e-8:
                    c = (num / den).item()
                    if abs(c) > abs(best_corr):
                        best_corr = c
                        best_lag = lag

            if abs(best_corr) > 0.03:  # meaningful threshold
                if best_corr > 0:
                    lead_scores[i_idx] += best_corr
                    lag_scores[j_idx] += best_corr
                    significant_leads[tickers[i_idx]] += 1
    else:
        for i_idx, j_idx in tqdm(pairs, desc="  Computing lead-lag"):
            best_corr = 0
            for lag in range(1, 6):
                x = ret_sub.iloc[:-lag, i_idx].values
                y = ret_sub.iloc[lag:, j_idx].values
                if len(x) > 50:
                    c = np.corrcoef(x, y)[0, 1]
                    if not np.isnan(c) and abs(c) > abs(best_corr):
                        best_corr = c

            if abs(best_corr) > 0.03:
                if best_corr > 0:
                    lead_scores[i_idx] += best_corr
                    lag_scores[j_idx] += best_corr
                    significant_leads[tickers[i_idx]] += 1

    # Identify bellwethers (high lead score, low lag score)
    lead_series = pd.Series(lead_scores, index=tickers)
    lag_series = pd.Series(lag_scores, index=tickers)
    net_lead = lead_series - lag_series

    bellwethers = net_lead.nlargest(15)
    followers = net_lead.nsmallest(15)

    # Sector composition of bellwethers
    bellwether_sectors = defaultdict(int)
    for t in bellwethers.index:
        s = sector_map.get(t, "Unknown")
        bellwether_sectors[s] += 1

    # Stability test: first half vs second half
    half = len(ret_sub) // 2
    first_leads = defaultdict(float)
    second_leads = defaultdict(float)

    for i_idx, j_idx in pairs[:200]:
        for lag in [1, 2]:
            x1 = ret_sub.iloc[:half - lag, i_idx].values
            y1 = ret_sub.iloc[lag:half, j_idx].values
            x2 = ret_sub.iloc[half:-lag, i_idx].values
            y2 = ret_sub.iloc[half + lag:, j_idx].values

            if len(x1) > 30 and len(x2) > 30:
                c1 = np.corrcoef(x1, y1)[0, 1] if len(x1) == len(y1) else 0
                c2 = np.corrcoef(x2, y2)[0, 1] if len(x2) == len(y2) else 0
                if not np.isnan(c1):
                    first_leads[tickers[i_idx]] += c1
                if not np.isnan(c2):
                    second_leads[tickers[i_idx]] += c2

    # Rank correlation of lead scores across halves
    common_leads = list(set(first_leads.keys()) & set(second_leads.keys()))
    if len(common_leads) > 20:
        fl = [first_leads[t] for t in common_leads]
        sl = [second_leads[t] for t in common_leads]
        stability_corr, stability_p = stats.spearmanr(fl, sl)
    else:
        stability_corr, stability_p = 0, 1

    findings = {
        "name": "Lead-Lag Network",
        "universe_size": len(tickers),
        "n_pairs_analyzed": len(pairs),
        "observations": {
            "bellwethers": {
                "finding": "Stocks that consistently move before others",
                "top_10": {t: round(float(v), 4) for t, v in bellwethers.head(10).items()},
                "sector_composition": dict(bellwether_sectors),
            },
            "followers": {
                "finding": "Stocks that consistently move after others",
                "top_10": {t: round(float(v), 4) for t, v in followers.head(10).items()},
            },
            "network_density": {
                "n_significant_lead_relationships": sum(significant_leads.values()),
                "mean_lead_connections_per_stock": round(
                    sum(significant_leads.values()) / max(len(significant_leads), 1), 2
                ),
            },
            "stability": {
                "finding": f"Lead-lag relationships are {'stable' if stability_corr > 0.3 else 'NOT stable'} over time",
                "rank_correlation": round(float(stability_corr), 3),
                "p_value": round(float(stability_p), 4),
            },
        },
        "surprise_score": surprise_score(float(stability_corr), float(stability_p), len(common_leads)),
    }

    print(f"\n  Analyzed {len(pairs)} pairs across {len(tickers)} stocks")
    print(f"\n  Top bellwethers (lead others):")
    for t, v in bellwethers.head(5).items():
        print(f"    {t:6s}: net lead score = {v:.4f} ({sector_map.get(t, '?')})")
    print(f"\n  Top followers (lag others):")
    for t, v in followers.head(5).items():
        print(f"    {t:6s}: net lag score = {v:.4f} ({sector_map.get(t, '?')})")
    print(f"\n  Stability: rank corr = {stability_corr:.3f}, p = {stability_p:.4f}")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# SCAN 7: Skewness Pricing
# ---------------------------------------------------------------------------

def scan_skewness_pricing(ret_data: dict) -> dict:
    """Do negatively-skewed stocks earn higher returns? (Skewness premium)."""
    print("\n" + "=" * 70)
    print("SCAN 7: Skewness Pricing")
    print("=" * 70)

    total_ret = ret_data["total_ret"].dropna(how="all")
    tickers = total_ret.columns.tolist()

    # Compute realized skewness for each stock
    skew_vals = {}
    ann_rets = {}
    sharpe_vals = {}

    for t in tickers:
        r = total_ret[t].dropna()
        if len(r) < 200:
            continue
        skew_vals[t] = stats.skew(r.values)
        ann_rets[t] = r.mean() * 252
        sharpe_vals[t] = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

    skew_series = pd.Series(skew_vals)
    ret_series = pd.Series(ann_rets)
    sharpe_series = pd.Series(sharpe_vals)

    # Quintile sort by skewness
    skew_quintile = pd.qcut(skew_series, 5, labels=False, duplicates="drop")

    quintile_stats = {}
    for q in range(5):
        q_tickers = skew_quintile[skew_quintile == q].index.tolist()
        if not q_tickers:
            continue
        q_ret = ret_series.loc[q_tickers].mean()
        q_sharpe = sharpe_series.loc[q_tickers].mean()
        q_skew = skew_series.loc[q_tickers].mean()
        quintile_stats[f"Q{q+1}"] = {
            "label": (
                "Most negatively skewed" if q == 0
                else "Most positively skewed" if q == 4
                else f"Mid-skew {q}"
            ),
            "n_stocks": len(q_tickers),
            "mean_skewness": round(float(q_skew), 3),
            "annualized_return_pct": round(float(q_ret * 100), 2),
            "mean_sharpe": round(float(q_sharpe), 3),
        }

    # Long-short spread: Q1 (negative skew) minus Q5 (positive skew)
    if "Q1" in quintile_stats and "Q5" in quintile_stats:
        ls_spread = quintile_stats["Q1"]["annualized_return_pct"] - quintile_stats["Q5"]["annualized_return_pct"]
        ls_sharpe = quintile_stats["Q1"]["mean_sharpe"] - quintile_stats["Q5"]["mean_sharpe"]
    else:
        ls_spread = 0
        ls_sharpe = 0

    # Direct correlation: skewness vs returns
    common = skew_series.index.intersection(ret_series.index)
    skew_ret_corr, skew_ret_p = stats.pearsonr(skew_series.loc[common], ret_series.loc[common])

    # Persistence of skewness: does past skewness predict future skewness?
    split = int(len(total_ret) * 0.5)
    first_skew = {}
    second_skew = {}
    for t in tickers:
        r1 = total_ret[t].iloc[:split].dropna()
        r2 = total_ret[t].iloc[split:].dropna()
        if len(r1) > 100 and len(r2) > 100:
            first_skew[t] = stats.skew(r1.values)
            second_skew[t] = stats.skew(r2.values)

    common_skew = list(set(first_skew.keys()) & set(second_skew.keys()))
    if len(common_skew) > 50:
        fs = [first_skew[t] for t in common_skew]
        ss = [second_skew[t] for t in common_skew]
        persist_corr, persist_p = stats.pearsonr(fs, ss)
    else:
        persist_corr, persist_p = 0, 1

    findings = {
        "name": "Skewness Pricing",
        "universe_size": len(skew_vals),
        "observations": {
            "quintile_sort": quintile_stats,
            "long_short_spread": {
                "finding": (
                    f"Neg-skew minus Pos-skew spread: {ls_spread:+.2f}% annual, "
                    f"{'confirming' if ls_spread > 0 else 'rejecting'} skewness premium"
                ),
                "spread_pct": round(ls_spread, 2),
                "sharpe_diff": round(ls_sharpe, 3),
            },
            "skew_return_correlation": {
                "finding": (
                    f"Skewness {'negatively' if skew_ret_corr < 0 else 'positively'} correlates with returns"
                ),
                "correlation": round(float(skew_ret_corr), 4),
                "p_value": round(float(skew_ret_p), 6),
            },
            "skewness_persistence": {
                "finding": f"Realized skewness is {'persistent' if persist_corr > 0.3 else 'NOT persistent'} over time",
                "correlation": round(float(persist_corr), 4),
                "p_value": round(float(persist_p), 6),
                "sample_size": len(common_skew),
            },
        },
        "surprise_score": surprise_score(float(skew_ret_corr), float(skew_ret_p), len(common)),
    }

    print(f"\n  {len(skew_vals)} stocks analyzed")
    print(f"\n  Quintile sort (Q1=most negative skew, Q5=most positive):")
    for q, s in quintile_stats.items():
        print(f"    {q}: skew={s['mean_skewness']:.3f}, ret={s['annualized_return_pct']:+.2f}%, "
              f"Sharpe={s['mean_sharpe']:.3f}")
    print(f"\n  Long-short (neg minus pos skew): {ls_spread:+.2f}% annual")
    print(f"  Skew-return correlation: {skew_ret_corr:.4f}, p={skew_ret_p:.6f}")
    print(f"  Skewness persistence: {persist_corr:.4f}")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# SCAN 8: Liquidity Dry-Up Events
# ---------------------------------------------------------------------------

def scan_liquidity_dryup(ret_data: dict, volumes: dict) -> dict:
    """What happens when a stock's volume drops to historical lows?"""
    print("\n" + "=" * 70)
    print("SCAN 8: Liquidity Dry-Up Events")
    print("=" * 70)

    total_ret = ret_data["total_ret"].dropna(how="all")
    tickers = total_ret.columns.tolist()

    # Build volume DataFrame
    vol_frames = {}
    for t in tickers:
        if t in volumes and volumes[t] is not None:
            v = volumes[t]
            if isinstance(v, pd.Series) and len(v) > 100:
                vol_frames[t] = v

    vol_df = pd.DataFrame(vol_frames)
    common_tickers = list(set(tickers) & set(vol_df.columns))
    vol_df = vol_df[common_tickers]
    ret_sub = total_ret[common_tickers]

    # For each stock, compute rolling 63-day volume percentile
    dryup_events = []
    normal_events = []

    forward_windows = [1, 5, 10, 21]
    fwd_ret_dryup = {w: [] for w in forward_windows}
    fwd_ret_normal = {w: [] for w in forward_windows}
    fwd_vol_dryup = {w: [] for w in forward_windows}
    fwd_vol_normal = {w: [] for w in forward_windows}

    for t in tqdm(common_tickers, desc="  Analyzing liquidity dry-ups"):
        v = vol_df[t].dropna()
        r = ret_sub[t]

        if len(v) < 252:
            continue

        # Rolling 252-day volume percentile
        rolling_pctile = v.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) == 252 else np.nan,
            raw=False
        )

        # Identify dry-up days: volume below 20th percentile of own history
        dryup_mask = rolling_pctile < 20
        normal_mask = (rolling_pctile >= 40) & (rolling_pctile <= 60)

        dryup_dates = dryup_mask[dryup_mask].index
        normal_dates = normal_mask[normal_mask].index

        for w in forward_windows:
            for d in dryup_dates:
                loc = r.index.get_loc(d) if d in r.index else None
                if loc is not None and loc + w < len(r):
                    fwd = r.iloc[loc + 1:loc + 1 + w].sum()
                    fwd_vol = r.iloc[loc + 1:loc + 1 + w].std() * np.sqrt(252)
                    fwd_ret_dryup[w].append(fwd)
                    if not np.isnan(fwd_vol):
                        fwd_vol_dryup[w].append(fwd_vol)

            # Sample normal dates (subsample for speed)
            for d in normal_dates[::5]:
                loc = r.index.get_loc(d) if d in r.index else None
                if loc is not None and loc + w < len(r):
                    fwd = r.iloc[loc + 1:loc + 1 + w].sum()
                    fwd_vol = r.iloc[loc + 1:loc + 1 + w].std() * np.sqrt(252)
                    fwd_ret_normal[w].append(fwd)
                    if not np.isnan(fwd_vol):
                        fwd_vol_normal[w].append(fwd_vol)

    # Compare forward returns and volatility
    window_results = {}
    for w in forward_windows:
        dr = np.array(fwd_ret_dryup[w])
        nr = np.array(fwd_ret_normal[w])
        dv = np.array(fwd_vol_dryup[w])
        nv = np.array(fwd_vol_normal[w])

        if len(dr) > 30 and len(nr) > 30:
            t_ret, p_ret = stats.ttest_ind(dr, nr)
            t_vol, p_vol = stats.ttest_ind(dv, nv) if len(dv) > 30 and len(nv) > 30 else (0, 1)
            window_results[f"{w}d_forward"] = {
                "dryup_mean_ret_bps": round(float(dr.mean() * 10000), 2),
                "normal_mean_ret_bps": round(float(nr.mean() * 10000), 2),
                "ret_diff_bps": round(float((dr.mean() - nr.mean()) * 10000), 2),
                "ret_t_stat": round(float(t_ret), 3),
                "ret_p_value": round(float(p_ret), 6),
                "dryup_mean_vol": round(float(dv.mean()), 4) if len(dv) > 0 else None,
                "normal_mean_vol": round(float(nv.mean()), 4) if len(nv) > 0 else None,
                "vol_t_stat": round(float(t_vol), 3),
                "vol_p_value": round(float(p_vol), 6),
                "n_dryup_events": len(dr),
                "n_normal_events": len(nr),
            }

    # Is illiquidity predictive? Aggregate finding
    best_window = None
    best_p = 1.0
    for w, res in window_results.items():
        if res["ret_p_value"] < best_p:
            best_p = res["ret_p_value"]
            best_window = w

    findings = {
        "name": "Liquidity Dry-Up Events",
        "universe_size": len(common_tickers),
        "observations": {
            "forward_returns_by_window": window_results,
            "most_significant_window": best_window,
            "overall_finding": (
                f"Liquidity dry-ups {'DO' if best_p < 0.05 else 'do NOT'} predict abnormal forward returns"
                f" (most significant at {best_window}, p={best_p:.4f})"
            ),
        },
        "surprise_score": surprise_score(
            float(window_results.get("5d_forward", {}).get("ret_diff_bps", 0)) / 10000,
            best_p,
            sum(r.get("n_dryup_events", 0) for r in window_results.values())
        ),
    }

    print(f"\n  {len(common_tickers)} stocks analyzed")
    print(f"\n  Forward returns after liquidity dry-up vs normal volume:")
    for w, res in window_results.items():
        print(f"    {w}: dryup={res['dryup_mean_ret_bps']:+.1f}bps vs "
              f"normal={res['normal_mean_ret_bps']:+.1f}bps, "
              f"diff={res['ret_diff_bps']:+.1f}bps "
              f"(p={res['ret_p_value']:.4f}, n={res['n_dryup_events']})")
    print(f"\n  {findings['observations']['overall_finding']}")
    print(f"  Surprise score: {findings['surprise_score']}")

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-Sectional Anomaly Scanner")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    parser.add_argument("--universe", type=str, default="sp500",
                        choices=["sp500", "russell1000"], help="Stock universe")
    args = parser.parse_args()

    print("=" * 70)
    print("  CROSS-SECTIONAL ANOMALY SCANNER")
    print("  Observation-First Discovery (HC #735)")
    print(f"  GPU: {'YES — ' + torch.cuda.get_device_name(0) if USE_GPU else 'NO — CPU mode'}")
    print(f"  Universe: {args.universe.upper()}")
    print(f"  History: {args.years} years")
    print("=" * 70)

    # 1. Get universe
    if args.universe == "russell1000":
        tickers, sector_map = get_russell1000_tickers()
    else:
        tickers, sector_map = get_sp500_tickers()

    # 2. Download data
    data = download_data(tickers, sector_map, years=args.years, use_cache=not args.no_cache)
    prices = data["prices"]
    volumes = data["volumes"]
    sector_map = data["sector_map"]

    print(f"\n[Data] {len(prices)} stocks loaded, {args.years} years of history")

    # 3. Build return matrices
    print("[Data] Building return matrices...")
    ret_data = build_return_matrices(prices)
    n_stocks = len(ret_data["total_ret"].columns)
    n_days = len(ret_data["total_ret"])
    print(f"[Data] Return matrix: {n_stocks} stocks x {n_days} days")

    # 4. Run all scans
    all_findings = []
    scan_functions = [
        ("overnight_intraday", lambda: scan_overnight_vs_intraday(ret_data, sector_map)),
        ("autocorrelation_clustering", lambda: scan_autocorrelation_clustering(ret_data)),
        ("tail_dependence", lambda: scan_tail_dependence(ret_data, sector_map)),
        ("intraweek_seasonality", lambda: scan_intraweek_seasonality(ret_data)),
        ("vol_of_vol", lambda: scan_vol_of_vol(ret_data)),
        ("lead_lag_network", lambda: scan_lead_lag_network(ret_data, sector_map)),
        ("skewness_pricing", lambda: scan_skewness_pricing(ret_data)),
        ("liquidity_dryup", lambda: scan_liquidity_dryup(ret_data, volumes)),
    ]

    results = {}
    for scan_name, scan_fn in scan_functions:
        try:
            result = scan_fn()
            results[scan_name] = result
            all_findings.append(result)
        except Exception as e:
            print(f"\n  [ERROR] Scan '{scan_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            results[scan_name] = {"name": scan_name, "error": str(e)}

    # 5. Summary
    print("\n" + "=" * 70)
    print("  SUMMARY — RANKED BY SURPRISE SCORE")
    print("=" * 70)

    scored = [(f.get("name", "?"), f.get("surprise_score", 0)) for f in all_findings if "surprise_score" in f]
    scored.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, score) in enumerate(scored, 1):
        stars = "*" * min(int(score / 2) + 1, 5)
        print(f"  {rank}. [{stars:5s}] {name}: surprise={score:.1f}")

    # 6. Save results
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "scan_date": str(dt.datetime.now()),
        "universe": args.universe,
        "n_stocks": n_stocks,
        "n_days": n_days,
        "years": args.years,
        "gpu_used": USE_GPU,
        "ranked_findings": [
            {"rank": i + 1, "name": name, "surprise_score": score}
            for i, (name, score) in enumerate(scored)
        ],
        "detailed_results": results,
    }

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to {OUTPUT_JSON}")

    print("\n" + "=" * 70)
    print("  SCAN COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
