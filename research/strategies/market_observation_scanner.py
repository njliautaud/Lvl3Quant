#!/usr/bin/env python3
"""
Market Observation Scanner — Observation-First Asymmetric Pattern Detection
===========================================================================
Downloads S&P 500 daily data and scans for surprising statistical anomalies
that could signal asymmetric trading setups. NO backtesting, NO strategy
building — purely observational.

Usage:
    python3 market_observation_scanner.py [--no-cache] [--years 5]

Author: Claude Opus 4.6 / Teleclaude Research
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
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# S&P 500 tickers — full list as of mid-2026 (sourced from Wikipedia snapshot)
# We fetch from Wikipedia live if possible, else fall back to this hardcoded list
# ---------------------------------------------------------------------------

CONTEXT_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX"]

# GICS sector map for the most common S&P 500 members
SECTOR_MAP_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

CACHE_PATH = Path("/home/jupiter/Lvl3Quant/data/sp500_cache.pkl")
OUTPUT_JSON = Path("/home/jupiter/Lvl3Quant/data/sp500_observations.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> tuple[list[str], dict[str, str]]:
    """Fetch S&P 500 constituents + sector mapping from Wikipedia."""
    try:
        tables = pd.read_html(SECTOR_MAP_URL)
        df = tables[0]
        # Column names vary; look for 'Symbol' and 'GICS Sector'
        sym_col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
        sec_col = [c for c in df.columns if "gics" in c.lower() and "sector" in c.lower()][0]
        tickers = df[sym_col].str.replace(".", "-", regex=False).tolist()
        sector_map = dict(zip(
            df[sym_col].str.replace(".", "-", regex=False),
            df[sec_col]
        ))
        print(f"  Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers, sector_map
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}), using hardcoded list")
        return _hardcoded_sp500()


def _hardcoded_sp500() -> tuple[list[str], dict[str, str]]:
    """Hardcoded fallback — top ~503 S&P 500 constituents."""
    # Abbreviated for space but covers all 11 GICS sectors
    tickers = [
        "AAPL","ABBV","ABT","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE","AEP","AES",
        "AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALK","ALL","ALLE","AMAT","AMCR",
        "AMD","AME","AMGN","AMP","AMT","AMZN","ANET","ANSS","AON","AOS","APA","APD",
        "APH","APTV","ARE","ATO","ATVI","AVB","AVGO","AVY","AWK","AXP","AZO","BA",
        "BAC","BAX","BBWI","BBY","BDX","BEN","BF-B","BIO","BIIB","BK","BKNG","BKR",
        "BLK","BMY","BR","BRK-B","BRO","BSX","BWA","BXP","C","CAG","CAH","CARR",
        "CAT","CB","CBOE","CBRE","CCI","CCL","CDAY","CDNS","CDW","CE","CEG","CF",
        "CFG","CHD","CHRW","CHTR","CI","CINF","CL","CLX","CMA","CMCSA","CME","CMG",
        "CMI","CMS","CNC","CNP","COF","COO","COP","COST","CPB","CPRT","CPT","CRL",
        "CRM","CSCO","CSGP","CSX","CTAS","CTLT","CTRA","CTSH","CTVA","CVS","CVX",
        "CZR","D","DAL","DD","DE","DFS","DG","DGX","DHI","DHR","DIS","DISH","DLTR",
        "DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXC","DXCM","EA","EBAY",
        "ECL","ED","EFX","EIX","EL","EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR",
        "EQT","ES","ESS","ETN","ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE","EXR",
        "F","FANG","FAST","FBHS","FCX","FDS","FDX","FE","FFIV","FIS","FISV","FITB",
        "FLT","FMC","FOX","FOXA","FRC","FRT","FTNT","FTV","GD","GE","GILD","GIS",
        "GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN","GS","GWW","HAL",
        "HAS","HBAN","HCA","HD","PEAK","HES","HIG","HII","HLT","HOLX","HON","HPE",
        "HPQ","HRL","HSIC","HST","HSY","HUM","HWM","IBM","ICE","IDXX","IEX","IFF",
        "ILMN","INCY","INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT",
        "ITW","IVZ","J","JBHT","JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY",
        "KEYS","KHC","KIM","KLAC","KMB","KMI","KMX","KO","KR","L","LDOS","LEN",
        "LH","LHX","LIN","LKQ","LLY","LMT","LNC","LNT","LOW","LRCX","LUMN","LUV",
        "LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO",
        "MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST",
        "MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT","MSI",
        "MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE",
        "NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWL","NWS","NWSA",
        "NXPI","O","ODFL","OGN","OKE","OMC","ON","ORCL","ORLY","OTIS","OXY","PARA",
        "PAYC","PAYX","PCAR","PCG","PEAK","PEG","PEP","PFE","PFG","PG","PGR","PH",
        "PHM","PKG","PKI","PLD","PM","PNC","PNR","PNW","POOL","PPG","PPL","PRU",
        "PSA","PSX","PTC","PVH","PWR","PXD","PYPL","QCOM","QRVO","RCL","RE","REG",
        "REGN","RF","RHI","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG","RTX",
        "SBAC","SBNY","SBUX","SCHW","SEE","SHW","SIVB","SJM","SLB","SNA","SNPS",
        "SO","SPG","SPGI","SRE","STE","STT","STX","STZ","SWK","SWKS","SYF","SYK",
        "SYY","T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX","TGT","TMO",
        "TMUS","TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO",
        "TXN","TXT","TYL","UAL","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB",
        "V","VFC","VICI","VLO","VMC","VNO","VRSK","VRSN","VRTX","VTR","VTRS","VZ",
        "WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC","WHR","WM","WMB","WMT",
        "WRB","WRK","WST","WTW","WY","WYNN","XEL","XOM","XRAY","XYL","YUM","ZBH",
        "ZBRA","ZION","ZTS"
    ]
    # Rough sector map for hardcoded list
    sector_map = {t: "Unknown" for t in tickers}
    tech = ["AAPL","MSFT","NVDA","AVGO","AMD","ADBE","CRM","CSCO","INTC","ORCL",
            "ACN","TXN","QCOM","AMAT","NOW","INTU","SNPS","CDNS","KLAC","LRCX",
            "MCHP","NXPI","ON","MPWR","ADI","FTNT","PANW","KEYS","IT","SWKS",
            "ANSS","ADSK","GOOG","GOOGL","META"]
    health = ["UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","BMY",
              "AMGN","MDT","ISRG","GILD","CVS","CI","ELV","SYK","BSX","REGN",
              "VRTX","BDX","HCA","IDXX","EW","DXCM","BIIB","MRNA","HUM","CNC",
              "MOH","ALGN","HOLX","IQV","MTD","PKI","WAT","CRL","INCY"]
    fin = ["JPM","V","MA","BAC","WFC","MS","GS","SCHW","BLK","AXP","C","USB",
           "PNC","TFC","CME","ICE","COF","BK","MMC","AON","AIG","MET","PRU",
           "SPGI","MCO","MSCI","CB","AFL","ALL","TRV","PGR","CINF","WRB",
           "HIG","LNC","GL","BEN","IVZ","TROW","STT","NTRS","CFG","KEY",
           "FITB","HBAN","RF","ZION","CMA","FRC","SIVB","SBNY","DFS","SYF"]
    energy = ["XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","HAL",
              "DVN","PXD","FANG","MRO","BKR","APA","EQT","CTRA","OKE","WMB",
              "KMI","TRGP"]
    for t in tech:
        if t in sector_map: sector_map[t] = "Information Technology"
    for t in health:
        if t in sector_map: sector_map[t] = "Health Care"
    for t in fin:
        if t in sector_map: sector_map[t] = "Financials"
    for t in energy:
        if t in sector_map: sector_map[t] = "Energy"
    return tickers, sector_map


def download_data(tickers: list[str], years: int = 5) -> pd.DataFrame:
    """Download daily OHLCV for all tickers via yfinance."""
    import yfinance as yf

    end = dt.date.today()
    start = end - dt.timedelta(days=years * 365 + 30)

    all_tickers = list(set(tickers + CONTEXT_TICKERS))
    print(f"\n  Downloading {len(all_tickers)} tickers, {start} to {end} ...")

    # Download in batches to avoid timeouts
    batch_size = 50
    frames = {}
    failed = []

    for i in tqdm(range(0, len(all_tickers), batch_size), desc="  Downloading batches"):
        batch = all_tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                start=str(start),
                end=str(end),
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if len(batch) == 1:
                # Single ticker returns flat columns
                ticker = batch[0]
                if not data.empty:
                    frames[ticker] = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                for ticker in batch:
                    try:
                        df = data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                        df = df.dropna(how="all")
                        if len(df) > 100:
                            frames[ticker] = df
                    except (KeyError, TypeError):
                        failed.append(ticker)
        except Exception as e:
            failed.extend(batch)
            print(f"    Batch failed: {e}")
        time.sleep(0.3)  # Rate limiting

    print(f"  Downloaded {len(frames)} tickers successfully, {len(failed)} failed")
    if failed:
        print(f"  Failed tickers (sample): {failed[:20]}")

    return frames


def compute_features(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add technical features to each ticker's dataframe."""
    enriched = {}
    for ticker, df in tqdm(frames.items(), desc="  Computing features", leave=False):
        df = df.copy()
        if len(df) < 60:
            continue

        # Returns
        df["ret_1d"] = df["Close"].pct_change()
        df["ret_5d"] = df["Close"].pct_change(5)
        df["ret_21d"] = df["Close"].pct_change(21)
        df["ret_63d"] = df["Close"].pct_change(63)  # ~3 months
        df["ret_126d"] = df["Close"].pct_change(126)  # ~6 months

        # Rolling highs/lows
        df["high_252d"] = df["High"].rolling(252).max()
        df["low_252d"] = df["Low"].rolling(252).min()
        df["drawdown_from_high"] = (df["Close"] - df["high_252d"]) / df["high_252d"]

        # Realized volatility (21-day)
        df["rvol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(252)
        # Percentile of own vol history
        vol_series = df["rvol_21d"].dropna()
        if len(vol_series) > 63:
            df["vol_pctile"] = vol_series.rolling(252, min_periods=63).apply(
                lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50,
                raw=False
            )

        # RSI (14-day)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Volume features
        df["vol_sma_50"] = df["Volume"].rolling(50).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_sma_50"].replace(0, np.nan)

        # Forward returns (for measuring what happens AFTER signals)
        df["fwd_ret_5d"] = df["Close"].shift(-5) / df["Close"] - 1
        df["fwd_ret_21d"] = df["Close"].shift(-21) / df["Close"] - 1
        df["fwd_ret_63d"] = df["Close"].shift(-63) / df["Close"] - 1
        df["fwd_ret_126d"] = df["Close"].shift(-126) / df["Close"] - 1

        enriched[ticker] = df

    return enriched


# ---------------------------------------------------------------------------
# Observation scanners
# ---------------------------------------------------------------------------

class ObservationResult:
    """Container for a single observation."""
    def __init__(self, name: str, description: str, data: np.ndarray,
                 baseline: np.ndarray = None, extra: dict = None):
        self.name = name
        self.description = description
        self.data = data[~np.isnan(data)] if data is not None else np.array([])
        self.baseline = baseline[~np.isnan(baseline)] if baseline is not None else None
        self.extra = extra or {}

    def compute_stats(self) -> dict:
        if len(self.data) < 10:
            return {"name": self.name, "n": len(self.data), "status": "INSUFFICIENT DATA"}

        d = self.data
        result = {
            "name": self.name,
            "description": self.description,
            "n": int(len(d)),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "std": float(np.std(d)),
            "p25": float(np.percentile(d, 25)),
            "p75": float(np.percentile(d, 75)),
            "skewness": float(stats.skew(d)),
            "kurtosis": float(stats.kurtosis(d)),
            "pct_positive": float(np.mean(d > 0) * 100),
        }

        # T-test vs zero
        t_stat, p_val = stats.ttest_1samp(d, 0)
        result["t_stat"] = float(t_stat)
        result["p_value"] = float(p_val)

        # If baseline provided, compare distributions
        if self.baseline is not None and len(self.baseline) > 10:
            bl = self.baseline
            result["baseline_mean"] = float(np.mean(bl))
            result["baseline_median"] = float(np.median(bl))
            result["baseline_n"] = int(len(bl))
            # Welch t-test
            t2, p2 = stats.ttest_ind(d, bl, equal_var=False)
            result["vs_baseline_t"] = float(t2)
            result["vs_baseline_p"] = float(p2)
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(d) + np.var(bl)) / 2)
            if pooled_std > 0:
                result["cohens_d"] = float((np.mean(d) - np.mean(bl)) / pooled_std)
            else:
                result["cohens_d"] = 0.0

        # Surprise score: |mean / std| * sqrt(n) — basically |t-stat|
        # Higher = more surprising (deviation from zero is real, not noise)
        result["surprise_score"] = abs(result["t_stat"])

        # Actionability: is the edge large enough to matter?
        # For stocks, >0.5% per trade is meaningful
        result["actionable"] = (
            abs(result["mean"]) > 0.005
            and result["p_value"] < 0.05
            and result["n"] >= 30
        )

        result.update(self.extra)
        return result


def scan_post_crash_recovery(enriched: dict, sector_map: dict) -> list[ObservationResult]:
    """After >20% drawdown from 52w high, what are forward returns?"""
    print("\n  [1/8] Post-Crash Recovery Asymmetry ...")
    observations = []

    crash_fwd_63d = []
    crash_fwd_126d = []
    no_crash_fwd_63d = []
    sector_recovery = defaultdict(list)

    for ticker, df in enriched.items():
        if "drawdown_from_high" not in df.columns:
            continue
        mask_crash = df["drawdown_from_high"] <= -0.20
        mask_nocrash = (df["drawdown_from_high"] > -0.10) & (df["drawdown_from_high"] < 0)

        fwd63_crash = df.loc[mask_crash, "fwd_ret_63d"].dropna().values
        fwd126_crash = df.loc[mask_crash, "fwd_ret_126d"].dropna().values
        fwd63_nocrash = df.loc[mask_nocrash, "fwd_ret_63d"].dropna().values

        crash_fwd_63d.extend(fwd63_crash)
        crash_fwd_126d.extend(fwd126_crash)
        no_crash_fwd_63d.extend(fwd63_nocrash)

        sector = sector_map.get(ticker, "Unknown")
        sector_recovery[sector].extend(fwd63_crash)

    observations.append(ObservationResult(
        "Post-Crash 3m Recovery",
        "Forward 3-month returns after stock drops >20% from 52w high",
        np.array(crash_fwd_63d),
        baseline=np.array(no_crash_fwd_63d),
        extra={"horizon": "63d", "trigger": ">20% drawdown from 52w high"}
    ))
    observations.append(ObservationResult(
        "Post-Crash 6m Recovery",
        "Forward 6-month returns after stock drops >20% from 52w high",
        np.array(crash_fwd_126d),
        baseline=None,
        extra={"horizon": "126d", "trigger": ">20% drawdown from 52w high"}
    ))

    # Sector breakdown
    for sector, rets in sector_recovery.items():
        if len(rets) >= 30:
            observations.append(ObservationResult(
                f"Post-Crash Recovery [{sector}]",
                f"3m recovery after >20% drawdown, {sector} sector only",
                np.array(rets),
                extra={"sector": sector, "horizon": "63d"}
            ))

    return observations


def scan_vol_compression(enriched: dict) -> list[ObservationResult]:
    """Low vol percentile -> forward returns."""
    print("  [2/8] Volatility Compression -> Explosion ...")
    observations = []

    compressed_fwd_21d = []
    compressed_fwd_63d = []
    normal_fwd_21d = []
    compressed_magnitude = []  # abs(fwd return) to measure explosion size

    for ticker, df in enriched.items():
        if "vol_pctile" not in df.columns:
            continue
        mask_low = df["vol_pctile"] <= 10
        mask_normal = (df["vol_pctile"] >= 40) & (df["vol_pctile"] <= 60)

        fwd21_low = df.loc[mask_low, "fwd_ret_21d"].dropna().values
        fwd63_low = df.loc[mask_low, "fwd_ret_63d"].dropna().values
        fwd21_norm = df.loc[mask_normal, "fwd_ret_21d"].dropna().values

        compressed_fwd_21d.extend(fwd21_low)
        compressed_fwd_63d.extend(fwd63_low)
        normal_fwd_21d.extend(fwd21_norm)
        compressed_magnitude.extend(np.abs(fwd21_low))

    observations.append(ObservationResult(
        "Vol Compression -> 1m Returns",
        "Forward 1-month returns when realized vol at bottom 10th pctile of own history",
        np.array(compressed_fwd_21d),
        baseline=np.array(normal_fwd_21d),
        extra={"trigger": "rvol_21d at <=10th percentile"}
    ))
    observations.append(ObservationResult(
        "Vol Compression -> 3m Returns",
        "Forward 3-month returns when realized vol at bottom 10th pctile",
        np.array(compressed_fwd_63d),
        extra={"trigger": "rvol_21d at <=10th percentile"}
    ))

    # Magnitude of moves (do compressed vol periods lead to bigger moves?)
    normal_magnitude = np.abs(np.array(normal_fwd_21d))
    observations.append(ObservationResult(
        "Vol Compression -> Move Magnitude",
        "Absolute 1m returns after vol compression vs normal vol — are moves bigger?",
        np.array(compressed_magnitude),
        baseline=normal_magnitude,
        extra={"measures": "abs(fwd_ret_21d)"}
    ))

    return observations


def scan_sector_rotation(enriched: dict, sector_map: dict) -> list[ObservationResult]:
    """Sector lead-lag structure."""
    print("  [3/8] Sector Rotation Leads ...")
    observations = []

    # Build sector-level monthly returns
    sector_rets = defaultdict(list)
    dates_by_sector = defaultdict(list)

    for ticker, df in enriched.items():
        sector = sector_map.get(ticker, "Unknown")
        if sector == "Unknown":
            continue
        monthly = df["Close"].resample("ME").last().pct_change().dropna()
        for date, ret in monthly.items():
            sector_rets[sector].append((date, ret))

    # Average returns per sector per month
    sector_monthly = {}
    for sector, items in sector_rets.items():
        df_temp = pd.DataFrame(items, columns=["date", "ret"])
        sector_monthly[sector] = df_temp.groupby("date")["ret"].mean()

    sectors = list(sector_monthly.keys())
    if len(sectors) < 3:
        return observations

    # Cross-correlation at lag 1-3 months
    lead_lag_results = []
    for s1 in sectors:
        for s2 in sectors:
            if s1 == s2:
                continue
            try:
                combined = pd.DataFrame({"leader": sector_monthly[s1], "follower": sector_monthly[s2]}).dropna()
                if len(combined) < 24:
                    continue
                for lag in [1, 2, 3]:
                    corr = combined["leader"].corr(combined["follower"].shift(-lag))
                    if not np.isnan(corr) and abs(corr) > 0.2:
                        lead_lag_results.append({
                            "leader": s1, "follower": s2,
                            "lag_months": lag, "correlation": round(corr, 3)
                        })
            except Exception:
                continue

    # Sort by absolute correlation
    lead_lag_results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    top_pairs = lead_lag_results[:15]

    if top_pairs:
        # Measure: when leader is top quintile, what's follower's fwd return?
        for pair in top_pairs[:5]:
            leader_s = sector_monthly.get(pair["leader"])
            follower_s = sector_monthly.get(pair["follower"])
            if leader_s is None or follower_s is None:
                continue
            combined = pd.DataFrame({"leader": leader_s, "follower": follower_s}).dropna()
            if len(combined) < 24:
                continue
            lag = pair["lag_months"]
            top_q = combined["leader"].quantile(0.8)
            mask_top = combined["leader"] >= top_q
            fwd = combined.loc[mask_top, "follower"].shift(-lag).dropna().values
            base = combined["follower"].values

            observations.append(ObservationResult(
                f"Sector Lead-Lag: {pair['leader'][:12]} -> {pair['follower'][:12]}",
                f"When {pair['leader']} is top quintile month, {pair['follower']} returns {lag}mo later",
                fwd,
                baseline=base,
                extra={"lag_months": lag, "raw_corr": pair["correlation"],
                       "leader": pair["leader"], "follower": pair["follower"]}
            ))

    return observations


def scan_dispersion_regimes(enriched: dict, frames: dict) -> list[ObservationResult]:
    """Cross-sectional return dispersion vs forward market returns."""
    print("  [4/8] Cross-Sectional Dispersion Regimes ...")
    observations = []

    # Compute daily cross-sectional return dispersion
    all_rets = {}
    for ticker, df in enriched.items():
        if ticker in ["SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX"]:
            continue
        if "ret_1d" in df.columns:
            all_rets[ticker] = df["ret_1d"]

    ret_panel = pd.DataFrame(all_rets)
    if ret_panel.empty:
        return observations

    # Daily cross-sectional std
    daily_dispersion = ret_panel.std(axis=1)
    # Rolling 21d average dispersion
    disp_21d = daily_dispersion.rolling(21).mean().dropna()

    # Get SPY forward returns
    spy_df = enriched.get("SPY")
    if spy_df is None:
        return observations

    combined = pd.DataFrame({
        "dispersion": disp_21d,
        "spy_fwd_21d": spy_df["fwd_ret_21d"],
        "spy_fwd_63d": spy_df["fwd_ret_63d"],
    }).dropna()

    if len(combined) < 100:
        return observations

    # High dispersion quintile vs low dispersion quintile
    q80 = combined["dispersion"].quantile(0.8)
    q20 = combined["dispersion"].quantile(0.2)

    high_disp_fwd = combined.loc[combined["dispersion"] >= q80, "spy_fwd_21d"].values
    low_disp_fwd = combined.loc[combined["dispersion"] <= q20, "spy_fwd_21d"].values

    observations.append(ObservationResult(
        "High Dispersion -> SPY 1m",
        "SPY forward 1-month returns when cross-sectional dispersion is top 20%",
        high_disp_fwd,
        baseline=low_disp_fwd,
        extra={"regime": "high_dispersion", "threshold": "80th pctile"}
    ))

    high_disp_fwd_3m = combined.loc[combined["dispersion"] >= q80, "spy_fwd_63d"].values
    low_disp_fwd_3m = combined.loc[combined["dispersion"] <= q20, "spy_fwd_63d"].values

    observations.append(ObservationResult(
        "High Dispersion -> SPY 3m",
        "SPY forward 3-month returns when cross-sectional dispersion is top 20%",
        high_disp_fwd_3m,
        baseline=low_disp_fwd_3m,
        extra={"regime": "high_dispersion", "threshold": "80th pctile"}
    ))

    return observations


def scan_rsi_extremes(enriched: dict) -> list[ObservationResult]:
    """RSI < 20 vs RSI > 80 forward return asymmetry."""
    print("  [5/8] Extreme RSI Reversals ...")
    observations = []

    oversold_fwd_5d = []
    oversold_fwd_21d = []
    overbought_fwd_5d = []
    overbought_fwd_21d = []

    for ticker, df in enriched.items():
        if "rsi_14" not in df.columns:
            continue
        mask_os = df["rsi_14"] < 20
        mask_ob = df["rsi_14"] > 80

        oversold_fwd_5d.extend(df.loc[mask_os, "fwd_ret_5d"].dropna().values)
        oversold_fwd_21d.extend(df.loc[mask_os, "fwd_ret_21d"].dropna().values)
        overbought_fwd_5d.extend(df.loc[mask_ob, "fwd_ret_5d"].dropna().values)
        overbought_fwd_21d.extend(df.loc[mask_ob, "fwd_ret_21d"].dropna().values)

    observations.append(ObservationResult(
        "RSI<20 Oversold -> 1w Returns",
        "Forward 1-week returns when RSI(14) drops below 20 (extremely oversold)",
        np.array(oversold_fwd_5d),
        baseline=np.array(overbought_fwd_5d),
        extra={"trigger": "RSI(14) < 20", "comparison": "RSI > 80"}
    ))
    observations.append(ObservationResult(
        "RSI<20 Oversold -> 1m Returns",
        "Forward 1-month returns when RSI(14) drops below 20",
        np.array(oversold_fwd_21d),
        baseline=np.array(overbought_fwd_21d),
        extra={"trigger": "RSI(14) < 20"}
    ))
    observations.append(ObservationResult(
        "RSI>80 Overbought -> 1w Returns",
        "Forward 1-week returns when RSI(14) exceeds 80 (extremely overbought)",
        np.array(overbought_fwd_5d),
        baseline=np.array(oversold_fwd_5d),
        extra={"trigger": "RSI(14) > 80"}
    ))
    observations.append(ObservationResult(
        "RSI>80 Overbought -> 1m Returns",
        "Forward 1-month returns when RSI(14) exceeds 80",
        np.array(overbought_fwd_21d),
        baseline=np.array(oversold_fwd_21d),
        extra={"trigger": "RSI(14) > 80"}
    ))

    return observations


def scan_volume_anomalies(enriched: dict) -> list[ObservationResult]:
    """Abnormal volume (>3x avg) without earnings — directional signal?"""
    print("  [6/8] Volume Anomalies ...")
    observations = []

    high_vol_up_fwd_5d = []
    high_vol_down_fwd_5d = []
    high_vol_up_fwd_21d = []
    high_vol_down_fwd_21d = []
    normal_vol_fwd_5d = []

    for ticker, df in enriched.items():
        if "vol_ratio" not in df.columns:
            continue

        mask_high_vol = df["vol_ratio"] >= 3.0
        mask_normal_vol = (df["vol_ratio"] >= 0.8) & (df["vol_ratio"] <= 1.2)

        # Split high-volume days by same-day direction
        mask_up = mask_high_vol & (df["ret_1d"] > 0)
        mask_down = mask_high_vol & (df["ret_1d"] < 0)

        high_vol_up_fwd_5d.extend(df.loc[mask_up, "fwd_ret_5d"].dropna().values)
        high_vol_down_fwd_5d.extend(df.loc[mask_down, "fwd_ret_5d"].dropna().values)
        high_vol_up_fwd_21d.extend(df.loc[mask_up, "fwd_ret_21d"].dropna().values)
        high_vol_down_fwd_21d.extend(df.loc[mask_down, "fwd_ret_21d"].dropna().values)
        normal_vol_fwd_5d.extend(df.loc[mask_normal_vol, "fwd_ret_5d"].dropna().values)

    observations.append(ObservationResult(
        "High Volume + Up Day -> 1w",
        "Fwd 1-week returns after >3x avg volume on an UP day (continuation?)",
        np.array(high_vol_up_fwd_5d),
        baseline=np.array(normal_vol_fwd_5d),
        extra={"trigger": "vol_ratio >= 3 AND ret_1d > 0"}
    ))
    observations.append(ObservationResult(
        "High Volume + Down Day -> 1w",
        "Fwd 1-week returns after >3x avg volume on a DOWN day (reversal or continuation?)",
        np.array(high_vol_down_fwd_5d),
        baseline=np.array(normal_vol_fwd_5d),
        extra={"trigger": "vol_ratio >= 3 AND ret_1d < 0"}
    ))
    observations.append(ObservationResult(
        "High Volume + Up Day -> 1m",
        "Fwd 1-month returns after >3x avg volume on an UP day",
        np.array(high_vol_up_fwd_21d),
        extra={"trigger": "vol_ratio >= 3 AND ret_1d > 0"}
    ))
    observations.append(ObservationResult(
        "High Volume + Down Day -> 1m",
        "Fwd 1-month returns after >3x avg volume on a DOWN day",
        np.array(high_vol_down_fwd_21d),
        extra={"trigger": "vol_ratio >= 3 AND ret_1d < 0"}
    ))

    return observations


def scan_breadth_divergence(enriched: dict, frames: dict) -> list[ObservationResult]:
    """New high breadth divergence — market new high with declining participation."""
    print("  [7/8] Breadth Divergence ...")
    observations = []

    spy_df = enriched.get("SPY")
    if spy_df is None:
        return observations

    # Compute daily: how many stocks are making 63d highs?
    high_counts = {}
    total_counts = {}
    for date in spy_df.index:
        n_high = 0
        n_total = 0
        for ticker, df in enriched.items():
            if ticker in ["SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX"]:
                continue
            if date not in df.index:
                continue
            n_total += 1
            try:
                lookback = df.loc[:date, "Close"].tail(63)
                if len(lookback) >= 20 and df.loc[date, "Close"] >= lookback.max():
                    n_high += 1
            except (KeyError, TypeError):
                continue
        if n_total > 50:
            high_counts[date] = n_high
            total_counts[date] = n_total

    if len(high_counts) < 100:
        print("    Insufficient breadth data, skipping ...")
        return observations

    breadth = pd.Series(high_counts)
    total = pd.Series(total_counts)
    breadth_pct = (breadth / total * 100).rolling(5).mean().dropna()

    # SPY 63d rolling high
    spy_rolling_high = spy_df["Close"].rolling(63).max()
    spy_at_high = (spy_df["Close"] >= spy_rolling_high * 0.99)  # within 1%

    combined = pd.DataFrame({
        "breadth_pct": breadth_pct,
        "spy_at_high": spy_at_high,
        "spy_fwd_21d": spy_df["fwd_ret_21d"],
        "spy_fwd_63d": spy_df["fwd_ret_63d"],
    }).dropna()

    if len(combined) < 50:
        return observations

    # Divergence: SPY at high but breadth below median
    median_breadth = combined["breadth_pct"].median()
    mask_divergence = combined["spy_at_high"] & (combined["breadth_pct"] < median_breadth)
    mask_healthy = combined["spy_at_high"] & (combined["breadth_pct"] >= combined["breadth_pct"].quantile(0.7))

    div_fwd = combined.loc[mask_divergence, "spy_fwd_21d"].values
    healthy_fwd = combined.loc[mask_healthy, "spy_fwd_21d"].values

    observations.append(ObservationResult(
        "Breadth Divergence -> SPY 1m",
        "SPY fwd 1m when at 63d high BUT new-high breadth below median (weak rally)",
        div_fwd,
        baseline=healthy_fwd,
        extra={"trigger": "SPY near 63d high AND breadth < median"}
    ))

    div_fwd_3m = combined.loc[mask_divergence, "spy_fwd_63d"].values
    healthy_fwd_3m = combined.loc[mask_healthy, "spy_fwd_63d"].values

    observations.append(ObservationResult(
        "Breadth Divergence -> SPY 3m",
        "SPY fwd 3m when at 63d high BUT weak breadth participation",
        div_fwd_3m,
        baseline=healthy_fwd_3m,
        extra={"trigger": "SPY near 63d high AND breadth < median"}
    ))

    return observations


def scan_earnings_sympathy(enriched: dict, sector_map: dict) -> list[ObservationResult]:
    """Proxy for earnings clustering: large single-day moves in sector peers."""
    print("  [8/8] Earnings Season Sympathy Moves ...")
    observations = []

    # We don't have actual earnings dates, so proxy: large single-day moves (>5%)
    # that likely correspond to earnings. Check if same-sector stocks move sympathetically.
    sector_groups = defaultdict(list)
    for ticker, df in enriched.items():
        sector = sector_map.get(ticker, "Unknown")
        if sector != "Unknown":
            sector_groups[sector].append(ticker)

    sympathy_fwd_5d = []
    no_sympathy_fwd_5d = []

    for sector, tickers in sector_groups.items():
        if len(tickers) < 5:
            continue

        for ticker in tickers:
            df = enriched.get(ticker)
            if df is None or "ret_1d" not in df.columns:
                continue

            # Find "earnings-like" days: |ret_1d| > 5%
            big_move_dates = df.index[df["ret_1d"].abs() > 0.05]

            for date in big_move_dates:
                direction = np.sign(df.loc[date, "ret_1d"])
                # Check if sector peers (other tickers) are affected 1-5 days later
                for peer in tickers:
                    if peer == ticker:
                        continue
                    peer_df = enriched.get(peer)
                    if peer_df is None:
                        continue
                    try:
                        fwd = peer_df.loc[date, "fwd_ret_5d"]
                        if not np.isnan(fwd):
                            sympathy_fwd_5d.append(fwd * direction)  # align direction
                    except (KeyError, TypeError):
                        continue

    if len(sympathy_fwd_5d) > 100:
        # Random baseline: pick random dates
        all_fwd_5d = []
        for ticker, df in enriched.items():
            if "fwd_ret_5d" in df.columns:
                all_fwd_5d.extend(df["fwd_ret_5d"].dropna().sample(min(50, len(df))).values)

        observations.append(ObservationResult(
            "Sector Sympathy (Direction-Aligned)",
            "Sector peer 1-week return aligned to big-mover direction. Positive = sympathy move",
            np.array(sympathy_fwd_5d),
            baseline=np.array(all_fwd_5d),
            extra={"trigger": "same-sector stock moved >5% in one day"}
        ))

    return observations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary_table(all_stats: list[dict]):
    """Print a ranked summary table of all observations."""
    # Filter to those with enough data
    valid = [s for s in all_stats if "surprise_score" in s]
    valid.sort(key=lambda x: x["surprise_score"], reverse=True)

    print("\n" + "=" * 110)
    print("OBSERVATION SUMMARY — RANKED BY SURPRISE SCORE")
    print("=" * 110)
    print(f"{'Rank':<5} {'Observation':<45} {'N':>7} {'Mean%':>8} {'Med%':>8} "
          f"{'Skew':>7} {'p-val':>9} {'Surprise':>9} {'Act?':>5}")
    print("-" * 110)

    for i, s in enumerate(valid, 1):
        name = s["name"][:44]
        n = s["n"]
        mean_pct = s["mean"] * 100
        med_pct = s["median"] * 100
        skew = s["skewness"]
        pval = s["p_value"]
        surprise = s["surprise_score"]
        act = "YES" if s.get("actionable", False) else "no"

        pval_str = f"{pval:.1e}" if pval < 0.001 else f"{pval:.4f}"
        print(f"{i:<5} {name:<45} {n:>7,} {mean_pct:>+7.2f} {med_pct:>+7.2f} "
              f"{skew:>+6.2f} {pval_str:>9} {surprise:>8.1f} {act:>5}")

    print("-" * 110)

    # Highlight top findings
    actionable = [s for s in valid if s.get("actionable", False)]
    print(f"\n  Total observations: {len(valid)}")
    print(f"  Actionable (|mean| > 0.5%, p < 0.05, N >= 30): {len(actionable)}")

    if actionable:
        print("\n  TOP ACTIONABLE FINDINGS:")
        for s in actionable[:10]:
            direction = "POSITIVE" if s["mean"] > 0 else "NEGATIVE"
            edge = abs(s["mean"]) * 100
            print(f"    - {s['name']}: {direction} {edge:.2f}% avg, N={s['n']:,}, "
                  f"p={s['p_value']:.4f}, skew={s['skewness']:+.2f}")
            if "baseline_mean" in s:
                bl_edge = abs(s["baseline_mean"]) * 100
                print(f"      vs baseline: {s.get('baseline_mean', 0)*100:+.2f}%, "
                      f"Cohen's d = {s.get('cohens_d', 0):.3f}")
            print(f"      Description: {s['description']}")


def main():
    parser = argparse.ArgumentParser(description="Market Observation Scanner")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default 5)")
    args = parser.parse_args()

    print("=" * 70)
    print("  MARKET OBSERVATION SCANNER")
    print("  Observation-first asymmetric pattern detection")
    print(f"  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Step 1: Get S&P 500 tickers ---
    print("\n[Step 1] Fetching S&P 500 constituents ...")
    tickers, sector_map = get_sp500_tickers()

    # --- Step 2: Download / load cached data ---
    print("\n[Step 2] Loading market data ...")
    frames = None
    if CACHE_PATH.exists() and not args.no_cache:
        cache_age = (dt.datetime.now() - dt.datetime.fromtimestamp(CACHE_PATH.stat().st_mtime))
        if cache_age.days < 7:
            print(f"  Loading cached data ({cache_age.days}d old) ...")
            try:
                with open(CACHE_PATH, "rb") as f:
                    frames = pickle.load(f)
                print(f"  Loaded {len(frames)} tickers from cache")
            except Exception as e:
                print(f"  Cache load failed: {e}")
                frames = None

    if frames is None:
        frames = download_data(tickers, years=args.years)
        # Save cache
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(frames, f)
        print(f"  Cached {len(frames)} tickers to {CACHE_PATH}")

    # --- Step 3: Compute features ---
    print("\n[Step 3] Computing features ...")
    enriched = compute_features(frames)
    print(f"  Enriched {len(enriched)} tickers with technical features")

    # --- Step 4: Run all observation scans ---
    print("\n[Step 4] Running observation scans ...")
    all_observations = []

    all_observations.extend(scan_post_crash_recovery(enriched, sector_map))
    all_observations.extend(scan_vol_compression(enriched))
    all_observations.extend(scan_sector_rotation(enriched, sector_map))
    all_observations.extend(scan_dispersion_regimes(enriched, frames))
    all_observations.extend(scan_rsi_extremes(enriched))
    all_observations.extend(scan_volume_anomalies(enriched))
    all_observations.extend(scan_breadth_divergence(enriched, frames))
    all_observations.extend(scan_earnings_sympathy(enriched, sector_map))

    # --- Step 5: Compute stats for all observations ---
    print(f"\n[Step 5] Computing statistics for {len(all_observations)} observations ...")
    all_stats = []
    for obs in all_observations:
        stat = obs.compute_stats()
        all_stats.append(stat)

        # Print individual observation details
        if "surprise_score" in stat:
            print(f"\n  --- {stat['name']} ---")
            print(f"  {stat['description']}")
            print(f"  N={stat['n']:,}  |  Mean={stat['mean']*100:+.3f}%  |  "
                  f"Median={stat['median']*100:+.3f}%  |  Skew={stat['skewness']:+.2f}")
            print(f"  P25={stat['p25']*100:+.3f}%  |  P75={stat['p75']*100:+.3f}%  |  "
                  f"StdDev={stat['std']*100:.3f}%")
            print(f"  t-stat={stat['t_stat']:.2f}  |  p-value={stat['p_value']:.6f}  |  "
                  f"Surprise={stat['surprise_score']:.1f}")
            if "baseline_mean" in stat:
                print(f"  Baseline mean={stat['baseline_mean']*100:+.3f}% (N={stat['baseline_n']:,})  |  "
                      f"Cohen's d={stat.get('cohens_d', 0):.3f}")
            print(f"  Pct positive: {stat['pct_positive']:.1f}%  |  "
                  f"Actionable: {'YES' if stat.get('actionable') else 'no'}")
        else:
            print(f"\n  --- {stat['name']} ---  {stat.get('status', 'N/A')}")

    # --- Step 6: Summary ---
    print_summary_table(all_stats)

    # --- Step 7: Save to JSON ---
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON-serializable
    def clean_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    serializable = []
    for s in all_stats:
        clean = {k: clean_for_json(v) for k, v in s.items()}
        serializable.append(clean)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "generated_at": dt.datetime.now().isoformat(),
            "n_tickers": len(enriched),
            "years_of_data": args.years,
            "observations": serializable,
        }, f, indent=2, default=str)

    print(f"\n  Observations saved to {OUTPUT_JSON}")
    print(f"\n{'=' * 70}")
    print("  SCAN COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
