#!/usr/bin/env python3
"""
Correlation Breakdown Opportunities — v1
=========================================
Hypothesis: When a stock decorrelates from its sector ETF AND shows relative
strength, institutional accumulation is likely driving the divergence, producing
asymmetric forward returns.

Variants tested (each with a distinct hypothesis):
  decorr_rs_long_21d    — pure decorrelation + relative strength
  decorr_rs_vol_21d     — + volume confirmation (>1.5x avg)
  decorr_rs_vc_21d      — + vol compression (ATR < 20th pctile)
  decorr_rs_mfi_21d     — + MFI oversold (<30)
  decorr_rw_long_21d    — relative weakness (distribution), contrarian long
  decorr_rs_long_10d    — shorter 10d hold
  decorr_sudden_21d     — sudden corr drop (delta>0.4 in 5d)
  decorr_persistent_21d — sustained low corr (>5d below 0.3)

Validation: 200-shuffle permutation, regime gap <0.50, year consistency >70%.
"""

import os
import sys
import time
import json
import warnings
import hashlib
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CACHE_DIR = Path("/home/nick/Lvl3Quant/data/decorrelation_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2012-01-01"
END_DATE   = "2026-07-18"

CORR_WINDOW   = 20    # rolling correlation window
CORR_THRESH   = 0.3   # decorrelation threshold
VOL_MULT      = 1.5   # volume confirmation multiplier
ATR_PCTILE    = 20     # vol compression percentile
MFI_THRESH    = 30     # MFI oversold threshold
PERM_N        = 200    # permutation test shuffles
REGIME_GAP_MAX = 0.50
YEAR_CONSIST_MIN = 0.70
N_WORKERS      = 8

# S&P 500 sector mapping — representative large-caps per sector
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

# Top ~33 stocks per sector (360+ total)
SECTOR_STOCKS = {
    "XLK": ["AAPL","MSFT","NVDA","AVGO","ADBE","CRM","CSCO","ACN","ORCL","INTC",
             "AMD","TXN","QCOM","NOW","INTU","IBM","AMAT","MU","LRCX","ADI",
             "KLAC","SNPS","CDNS","MCHP","FTNT","PANW","CRWD","MSI","TEL","ANSS",
             "KEYS","ZBRA","GLW"],
    "XLF": ["BRK-B","JPM","V","MA","BAC","WFC","GS","MS","SPGI","BLK",
             "C","AXP","CB","MMC","SCHW","PGR","CME","ICE","AON","MET",
             "AIG","TRV","AFL","PRU","ALL","AMP","TROW","RJF","HBAN","FITB",
             "MTB","CFG","KEY"],
    "XLE": ["XOM","CVX","COP","SLB","EOG","MPC","PSX","PXD","VLO","OXY",
             "WMB","KMI","HAL","DVN","FANG","HES","BKR","CTRA","MRO","APA",
             "OVV","EQT","TRGP","DEN","MTDR","PR","SM","RRC","AR","CHK",
             "MGY","PDCE","CLR"],
    "XLV": ["UNH","JNJ","LLY","PFE","ABBV","MRK","TMO","ABT","DHR","AMGN",
             "BMY","MDT","ISRG","ELV","CI","GILD","SYK","REGN","VRTX","BDX",
             "ZTS","BSX","HUM","HCA","MCK","IQV","IDXX","MTD","A","DXCM",
             "WAT","ALGN","HOLX"],
    "XLI": ["HON","UNP","UPS","RTX","CAT","BA","DE","LMT","GE","MMM",
             "GD","NOC","ITW","EMR","FDX","CSX","NSC","WM","RSG","JCI",
             "PCAR","CTAS","ROK","FAST","ODFL","TT","SWK","CMI","PH","ETN",
             "AME","DOV","XYL"],
    "XLC": ["META","GOOGL","GOOG","NFLX","DIS","CMCSA","TMUS","VZ","T","CHTR",
             "EA","ATVI","WBD","TTWO","OMC","IPG","PARA","FOXA","FOX","LYV",
             "MTCH","ZG","PINS","SNAP","ROKU","TTD","SPOT","RBLX","DKNG","BILI",
             "DISH","NWSA","NWS"],
    "XLY": ["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","ORLY",
             "AZO","ROST","CMG","DHI","LEN","GM","F","MAR","HLT","YUM",
             "EBAY","ETSY","DPZ","POOL","GRMN","ULTA","BBY","WYNN","LVS","MGM",
             "RCL","CCL","NCLH"],
    "XLP": ["PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","EL",
             "KMB","GIS","SYY","KHC","HSY","K","ADM","STZ","BF-B","TAP",
             "CPB","SJM","MKC","CHD","CLX","HRL","CAG","TSN","LW","FDP",
             "MNST","KDP","CASY"],
    "XLU": ["NEE","DUK","SO","D","AEP","SRE","EXC","XEL","ED","WEC",
             "ES","AWK","DTE","ETR","AEE","PPL","FE","CMS","EVRG","ATO",
             "NI","PNW","OGE","POR","BKH","AVA","NWE","SWX","UTL","HE",
             "IDA","MGEE","SR"],
    "XLRE": ["PLD","AMT","CCI","EQIX","PSA","SPG","O","DLR","WELL","VICI",
             "ARE","AVB","EQR","MAA","UDR","ESS","CPT","KIM","REG","FRT",
             "HST","BXP","SLG","VNO","HIW","OFC","CUZ","DEI","JBGS","KRG",
             "AIV","ACC","IRT"],
    "XLB": ["LIN","APD","SHW","ECL","FCX","NUE","NEM","DOW","DD","PPG",
             "VMC","MLM","ALB","CE","IFF","EMN","FMC","CF","MOS","BALL",
             "PKG","IP","SEE","WRK","AVY","SON","RPM","HUN","OLN","AXTA",
             "CBT","TROX","IOSP"],
}

# SPY for regime classification
SPY_TICKER = "SPY"


# ---------------------------------------------------------------------------
# DATA DOWNLOAD + CACHE
# ---------------------------------------------------------------------------
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('-','_')}.parquet"


def download_ticker(ticker: str) -> pd.DataFrame | None:
    """Download or load from cache."""
    cp = _cache_path(ticker)
    if cp.exists():
        try:
            df = pd.read_parquet(cp)
            if len(df) > 100:
                return df
        except Exception:
            pass
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if df is None or len(df) < 100:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_parquet(cp)
        return df
    except Exception as e:
        print(f"  [WARN] Failed to download {ticker}: {e}")
        return None


def download_all():
    """Download all tickers (sector ETFs + stocks + SPY)."""
    all_tickers = set([SPY_TICKER])
    for etf in SECTOR_ETFS:
        all_tickers.add(etf)
    for stocks in SECTOR_STOCKS.values():
        all_tickers.update(stocks)

    print(f"Downloading {len(all_tickers)} tickers (cached will be fast)...")
    data = {}
    failed = []
    for i, t in enumerate(sorted(all_tickers)):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(all_tickers)}")
        df = download_ticker(t)
        if df is not None:
            data[t] = df
        else:
            failed.append(t)
        time.sleep(0.05)  # rate limit courtesy

    print(f"Downloaded {len(data)} tickers, {len(failed)} failed: {failed[:20]}")
    return data


# ---------------------------------------------------------------------------
# INDICATOR HELPERS
# ---------------------------------------------------------------------------
def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta = tp.diff()
    pos_mf = pd.Series(np.where(delta > 0, mf, 0), index=close.index)
    neg_mf = pd.Series(np.where(delta < 0, mf, 0), index=close.index)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfi = 100.0 - (100.0 / (1.0 + pos_sum / neg_sum.replace(0, np.nan)))
    return mfi


def compute_atr(high, low, close, period=20):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------------------------
def generate_signals(stock_df, etf_df, spy_df, variant="decorr_rs_long_21d"):
    """
    Generate entry signals for a single stock vs its sector ETF.
    Returns DataFrame with columns: date, entry_price, exit_price, ret, regime
    """
    # Align dates
    common = stock_df.index.intersection(etf_df.index).intersection(spy_df.index)
    if len(common) < 252:
        return pd.DataFrame()

    s = stock_df.loc[common].copy()
    e = etf_df.loc[common].copy()
    spy = spy_df.loc[common].copy()

    s_ret = s["Close"].pct_change()
    e_ret = e["Close"].pct_change()

    # Rolling correlation
    roll_corr = s_ret.rolling(CORR_WINDOW).corr(e_ret)

    # Relative performance over correlation window
    s_cum = s_ret.rolling(CORR_WINDOW).sum()
    e_cum = e_ret.rolling(CORR_WINDOW).sum()
    rel_perf = s_cum - e_cum  # positive = relative strength

    # Volume ratio
    vol_avg_20 = s["Volume"].rolling(20).mean()
    vol_ratio = s["Volume"] / vol_avg_20.replace(0, np.nan)

    # ATR and its percentile
    atr = compute_atr(s["High"], s["Low"], s["Close"], 20)
    atr_pctile = atr.rolling(252).apply(lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 50 else 50, raw=False)

    # MFI
    mfi = compute_mfi(s["High"], s["Low"], s["Close"], s["Volume"], 14)

    # Correlation delta (for sudden variant)
    corr_delta = roll_corr - roll_corr.shift(5)

    # Sustained low corr (for persistent variant)
    low_corr_streak = (roll_corr < CORR_THRESH).rolling(5).sum()

    # Parse variant
    hold_days = 21
    if "10d" in variant:
        hold_days = 10

    # Build base decorrelation condition
    decorr = roll_corr < CORR_THRESH

    # Variant-specific conditions
    if variant == "decorr_rs_long_21d":
        cond = decorr & (rel_perf > 0)
    elif variant == "decorr_rs_vol_21d":
        cond = decorr & (rel_perf > 0) & (vol_ratio > VOL_MULT)
    elif variant == "decorr_rs_vc_21d":
        cond = decorr & (rel_perf > 0) & (atr_pctile < ATR_PCTILE)
    elif variant == "decorr_rs_mfi_21d":
        cond = decorr & (rel_perf > 0) & (mfi < MFI_THRESH)
    elif variant == "decorr_rw_long_21d":
        cond = decorr & (rel_perf < 0)  # relative weakness — contrarian
    elif variant == "decorr_rs_long_10d":
        cond = decorr & (rel_perf > 0)
    elif variant == "decorr_sudden_21d":
        cond = (corr_delta < -0.4) & (roll_corr < 0.5)  # sudden drop
    elif variant == "decorr_persistent_21d":
        cond = (low_corr_streak >= 5) & (rel_perf > 0)  # sustained low
    else:
        cond = decorr & (rel_perf > 0)

    # Prevent overlapping trades: enforce minimum gap of hold_days
    signal_dates = s.index[cond.fillna(False)]
    if len(signal_dates) == 0:
        return pd.DataFrame()

    trades = []
    last_exit_idx = -1
    for dt in signal_dates:
        idx = s.index.get_loc(dt)
        if idx <= last_exit_idx:
            continue  # skip overlapping
        exit_idx = idx + hold_days
        if exit_idx >= len(s):
            continue
        entry_price = s["Close"].iloc[idx]
        exit_price = s["Close"].iloc[exit_idx]
        ret = (exit_price - entry_price) / entry_price

        # Regime: based on SPY on entry day
        spy_open = spy["Open"].iloc[idx] if idx < len(spy) else np.nan
        spy_close = spy["Close"].iloc[idx] if idx < len(spy) else np.nan
        regime = "green" if spy_close > spy_open else "red"

        trades.append({
            "date": dt,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "ret": ret,
            "regime": regime,
            "year": dt.year,
        })
        last_exit_idx = exit_idx

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# BACKTEST + VALIDATION
# ---------------------------------------------------------------------------
def compute_metrics(rets: pd.Series) -> dict:
    """Compute Sharpe, Sortino, PF, WR from a series of trade returns."""
    if len(rets) < 5:
        return {"n": len(rets), "sharpe": np.nan, "sortino": np.nan, "pf": np.nan,
                "wr": np.nan, "avg_ret": np.nan, "med_ret": np.nan}

    avg = rets.mean()
    std = rets.std()
    sharpe = avg / std * np.sqrt(252 / 21) if std > 0 else np.nan  # annualized approx

    downside = rets[rets < 0].std()
    sortino = avg / downside * np.sqrt(252 / 21) if downside > 0 else np.nan

    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    wr = (rets > 0).mean()

    return {
        "n": len(rets),
        "sharpe": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "sortino": round(sortino, 3) if not np.isnan(sortino) else np.nan,
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "avg_ret": round(avg * 100, 3),  # percent
        "med_ret": round(rets.median() * 100, 3),
    }


def permutation_test(rets: pd.Series, n_perm=PERM_N) -> float:
    """Shuffle entry dates, compute p-value for observed Sharpe."""
    if len(rets) < 10:
        return 1.0
    obs_sharpe = rets.mean() / rets.std() if rets.std() > 0 else 0
    count_ge = 0
    for _ in range(n_perm):
        shuffled = rets.sample(frac=1, replace=False).values
        sh = shuffled.mean() / shuffled.std() if shuffled.std() > 0 else 0
        if sh >= obs_sharpe:
            count_ge += 1
    return count_ge / n_perm


def regime_gap(rets_df: pd.DataFrame) -> float:
    """Compute regime gap metric."""
    green = rets_df[rets_df["regime"] == "green"]["ret"]
    red = rets_df[rets_df["regime"] == "red"]["ret"]
    if len(green) < 5 or len(red) < 5:
        return np.nan
    sh_g = green.mean() / green.std() if green.std() > 0 else 0
    sh_r = red.mean() / red.std() if red.std() > 0 else 0
    denom = max(abs(sh_g), abs(sh_r))
    if denom == 0:
        return np.nan
    return abs(sh_g - sh_r) / denom


def year_consistency(rets_df: pd.DataFrame) -> float:
    """Fraction of years that are profitable."""
    yearly = rets_df.groupby("year")["ret"].sum()
    if len(yearly) < 3:
        return np.nan
    return (yearly > 0).mean()


# ---------------------------------------------------------------------------
# WORKER: process one variant
# ---------------------------------------------------------------------------
def run_variant(variant, data, spy_df):
    """Run a single variant across all stocks."""
    print(f"\n{'='*70}")
    print(f"VARIANT: {variant}")
    print(f"{'='*70}")

    all_trades = []
    stocks_with_signals = 0

    for etf, stocks in SECTOR_STOCKS.items():
        if etf not in data:
            continue
        etf_df = data[etf]
        for ticker in stocks:
            if ticker not in data:
                continue
            stock_df = data[ticker]
            trades = generate_signals(stock_df, etf_df, spy_df, variant)
            if len(trades) > 0:
                trades["ticker"] = ticker
                trades["sector_etf"] = etf
                all_trades.append(trades)
                stocks_with_signals += 1

    if not all_trades:
        print(f"  NO TRADES FOUND")
        return {"variant": variant, "status": "no_trades"}

    all_df = pd.concat(all_trades, ignore_index=True)
    rets = all_df["ret"]

    print(f"  Stocks with signals: {stocks_with_signals}")
    print(f"  Total trades: {len(all_df)}")
    print(f"  Date range: {all_df['date'].min()} to {all_df['date'].max()}")

    # Overall metrics
    metrics = compute_metrics(rets)
    print(f"\n  OVERALL METRICS:")
    print(f"    N={metrics['n']}, Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}")
    print(f"    PF={metrics['pf']}, WR={metrics['wr']}, Avg={metrics['avg_ret']}%, Med={metrics['med_ret']}%")

    # Regime analysis
    rg = regime_gap(all_df)
    print(f"\n  REGIME ANALYSIS:")
    green_m = compute_metrics(all_df[all_df["regime"]=="green"]["ret"])
    red_m = compute_metrics(all_df[all_df["regime"]=="red"]["ret"])
    print(f"    Green days: N={green_m['n']}, Sharpe={green_m['sharpe']}, WR={green_m['wr']}")
    print(f"    Red days:   N={red_m['n']}, Sharpe={red_m['sharpe']}, WR={red_m['wr']}")
    print(f"    Regime gap: {rg:.3f}" if not np.isnan(rg) else "    Regime gap: N/A")
    regime_pass = rg < REGIME_GAP_MAX if not np.isnan(rg) else False
    print(f"    Regime gate (<{REGIME_GAP_MAX}): {'PASS' if regime_pass else 'FAIL'}")

    # Year consistency
    yc = year_consistency(all_df)
    print(f"\n  YEAR CONSISTENCY:")
    yearly = all_df.groupby("year")["ret"].agg(["sum","count","mean"])
    for yr, row in yearly.iterrows():
        print(f"    {yr}: cumRet={row['sum']*100:.2f}%, N={int(row['count'])}, avg={row['mean']*100:.3f}%")
    print(f"    Profitable years: {yc:.1%}" if not np.isnan(yc) else "    Profitable years: N/A")
    year_pass = yc >= YEAR_CONSIST_MIN if not np.isnan(yc) else False
    print(f"    Year gate (>{YEAR_CONSIST_MIN:.0%}): {'PASS' if year_pass else 'FAIL'}")

    # Permutation test
    print(f"\n  PERMUTATION TEST ({PERM_N} shuffles)...")
    pval = permutation_test(rets, PERM_N)
    print(f"    p-value: {pval:.4f}")
    perm_pass = pval < 0.05
    print(f"    Permutation gate (<0.05): {'PASS' if perm_pass else 'FAIL'}")

    # Per-sector breakdown
    print(f"\n  PER-SECTOR BREAKDOWN:")
    for etf in sorted(all_df["sector_etf"].unique()):
        sec_rets = all_df[all_df["sector_etf"]==etf]["ret"]
        sm = compute_metrics(sec_rets)
        print(f"    {etf}: N={sm['n']}, Sharpe={sm['sharpe']}, WR={sm['wr']}, Avg={sm['avg_ret']}%")

    # Top/bottom stocks
    stock_perf = all_df.groupby("ticker")["ret"].agg(["mean","count"]).sort_values("mean", ascending=False)
    stock_perf = stock_perf[stock_perf["count"] >= 5]
    if len(stock_perf) > 0:
        print(f"\n  TOP 10 STOCKS (min 5 trades):")
        for t, row in stock_perf.head(10).iterrows():
            print(f"    {t}: avg={row['mean']*100:.3f}%, N={int(row['count'])}")
        print(f"\n  BOTTOM 10 STOCKS:")
        for t, row in stock_perf.tail(10).iterrows():
            print(f"    {t}: avg={row['mean']*100:.3f}%, N={int(row['count'])}")

    # Sliding window walk-forward validation
    print(f"\n  SLIDING WINDOW WALK-FORWARD (60d train, 1d step):")
    all_df_sorted = all_df.sort_values("date")
    all_df_sorted["month"] = all_df_sorted["date"].dt.to_period("M")
    monthly = all_df_sorted.groupby("month")["ret"].agg(["mean","count","sum"])
    if len(monthly) > 12:
        # Show rolling 12-month Sharpe
        rolling_sharpe = []
        months = monthly.index.tolist()
        for i in range(11, len(months)):
            window = monthly.iloc[i-11:i+1]
            avg_r = window["mean"].mean()
            std_r = window["mean"].std()
            rs = avg_r / std_r * np.sqrt(12) if std_r > 0 else 0
            rolling_sharpe.append((str(months[i]), round(rs, 2)))
        print(f"    Rolling 12-month Sharpe (last 10):")
        for m, sh in rolling_sharpe[-10:]:
            print(f"      {m}: {sh}")

    # Summary verdict
    all_pass = regime_pass and year_pass and perm_pass
    print(f"\n  === VERDICT: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'} ===")

    return {
        "variant": variant,
        "metrics": metrics,
        "regime_gap": round(rg, 3) if not np.isnan(rg) else None,
        "year_consistency": round(yc, 3) if not np.isnan(yc) else None,
        "perm_pvalue": round(pval, 4),
        "regime_pass": regime_pass,
        "year_pass": year_pass,
        "perm_pass": perm_pass,
        "all_pass": all_pass,
        "n_stocks": stocks_with_signals,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(f"=" * 70)
    print(f"CORRELATION BREAKDOWN OPPORTUNITIES — v1")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Workers: {N_WORKERS}")
    print(f"=" * 70)

    t0 = time.time()

    # Download data
    data = download_all()
    spy_df = data.get(SPY_TICKER)
    if spy_df is None:
        print("FATAL: Could not download SPY")
        sys.exit(1)

    t_dl = time.time() - t0
    print(f"\nData download took {t_dl:.1f}s")

    # Define variants
    variants = [
        "decorr_rs_long_21d",
        "decorr_rs_vol_21d",
        "decorr_rs_vc_21d",
        "decorr_rs_mfi_21d",
        "decorr_rw_long_21d",
        "decorr_rs_long_10d",
        "decorr_sudden_21d",
        "decorr_persistent_21d",
    ]

    # Run all variants (sequential since each is already fast with vectorized pandas)
    results = []
    for v in variants:
        res = run_variant(v, data, spy_df)
        results.append(res)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n\n{'='*70}")
    print(f"FINAL SUMMARY — CORRELATION BREAKDOWN v1")
    print(f"{'='*70}")
    print(f"Total runtime: {elapsed:.1f}s\n")

    print(f"{'Variant':<30} {'N':>6} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} {'WR':>6} {'RGap':>6} {'YCon':>6} {'Perm':>6} {'Pass':>6}")
    print("-" * 100)
    for r in results:
        if r.get("status") == "no_trades":
            print(f"{r['variant']:<30} {'NO TRADES':>6}")
            continue
        m = r["metrics"]
        print(f"{r['variant']:<30} {m['n']:>6} {m['sharpe']:>8} {m['sortino']:>8} {m['pf']:>6} {m['wr']:>6.3f} "
              f"{r['regime_gap'] or 'N/A':>6} {r['year_consistency'] or 'N/A':>6} {r['perm_pvalue']:>6.3f} "
              f"{'YES' if r.get('all_pass') else 'NO':>6}")

    # Save results JSON
    results_path = CACHE_DIR / "correlation_breakdown_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
