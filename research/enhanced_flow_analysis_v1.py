#!/usr/bin/env python3
"""
Enhanced Flow Analysis v1 — Comprehensive Macro Flow Ecosystem
==============================================================
Builds 5 macro flow signals from freely available yfinance data,
backtests them on SPY, and tests whether they improve contrarian signals.

Signals:
  1. CTA Positioning Proxy (MA crossover intensity)
  2. Cash-to-Equity Flow (risk appetite from volume ratios)
  3. Sector Rotation Flow (relative volume momentum by sector)
  4. Rate-Sensitive Flow (bond/credit positioning)
  5. Breadth Flow (advance-decline proxy from sector ETFs)

Author: Claude Opus 4.6 / Lvl3Quant
Date:   2026-07-23
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE = "2026-07-22"
FORWARD_WINDOWS = [5, 20]  # days

# CTA universe
CTA_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "EEM", "DBC"]
CTA_MA_WINDOWS = [20, 50, 100, 200]

# Cash vs equity
SAFETY_TICKERS = ["SHY", "BIL", "SGOV"]
EQUITY_TICKERS = ["SPY", "QQQ", "IWM"]

# Sector SPDR ETFs (all 11)
SECTOR_TICKERS = {
    "XLB": "Materials",
    "XLC": "Communication",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Healthcare",
    "XLY": "Discretionary",
}

# Rate-sensitive
RATE_TICKERS = ["TLT", "HYG", "LQD"]

# All tickers needed
ALL_TICKERS = sorted(set(
    CTA_TICKERS + SAFETY_TICKERS + EQUITY_TICKERS +
    list(SECTOR_TICKERS.keys()) + RATE_TICKERS + ["SPY"]
))

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/research/findings/enhanced_flow_v1_results.json")


# ─── Data Download ───────────────────────────────────────────────────────────
def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OHLCV data for all tickers."""
    print(f"Downloading {len(ALL_TICKERS)} tickers from {START_DATE} to {END_DATE}...")

    raw = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False, threads=True)

    # Extract close and volume
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()

    # Drop rows where SPY is missing (non-trading days)
    mask = close["SPY"].notna()
    close = close.loc[mask]
    volume = volume.loc[mask]

    print(f"  Got {len(close)} trading days, {close.shape[1]} tickers")
    missing = close.isna().sum()
    if missing.any():
        print(f"  Missing data: {dict(missing[missing > 0])}")

    # Forward-fill small gaps (weekends already excluded, this handles delistings/holidays)
    close = close.ffill(limit=5)
    volume = volume.ffill(limit=5).fillna(0)

    return close, volume


# ─── Signal 1: CTA Positioning Proxy ────────────────────────────────────────
def compute_cta_pressure(close: pd.DataFrame) -> pd.Series:
    """
    For each CTA-tracked asset, compute how far price is above/below
    each MA (20/50/100/200), normalize by ATR, then average across
    all assets and MAs to get a single CTA_pressure_score per day.

    Positive = price above MAs (trend-following CTAs likely long)
    Negative = price below MAs (CTAs likely short)
    """
    scores = []

    for ticker in CTA_TICKERS:
        if ticker not in close.columns:
            continue
        px = close[ticker].dropna()
        if len(px) < 200:
            continue

        for w in CTA_MA_WINDOWS:
            ma = px.rolling(w).mean()
            # Normalize distance by rolling 20d std (vol-adjusted)
            vol = px.pct_change().rolling(20).std() * px
            vol = vol.replace(0, np.nan)
            z = (px - ma) / vol
            z = z.clip(-5, 5)  # cap outliers
            scores.append(z.rename(f"{ticker}_ma{w}"))

    if not scores:
        raise ValueError("No CTA scores computed")

    df = pd.concat(scores, axis=1)
    cta_pressure = df.mean(axis=1)
    cta_pressure.name = "cta_pressure"
    return cta_pressure


# ─── Signal 2: Cash-to-Equity Flow ──────────────────────────────────────────
def compute_risk_appetite(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    Compare money-market/safety ETF dollar volume vs equity ETF dollar volume.
    High equity-to-safety ratio = risk-on. Low = risk-off.

    We use dollar volume (price * volume) for fair comparison.
    """
    safety_dv = pd.DataFrame()
    for t in SAFETY_TICKERS:
        if t in close.columns and t in volume.columns:
            safety_dv[t] = close[t] * volume[t]

    equity_dv = pd.DataFrame()
    for t in EQUITY_TICKERS:
        if t in close.columns and t in volume.columns:
            equity_dv[t] = close[t] * volume[t]

    safety_total = safety_dv.sum(axis=1).rolling(5).mean()
    equity_total = equity_dv.sum(axis=1).rolling(5).mean()

    # Ratio: equity / (equity + safety), then z-score
    ratio = equity_total / (equity_total + safety_total)
    ratio_z = (ratio - ratio.rolling(63).mean()) / ratio.rolling(63).std()
    ratio_z = ratio_z.clip(-3, 3)
    ratio_z.name = "risk_appetite"
    return ratio_z


# ─── Signal 3: Sector Rotation Flow ─────────────────────────────────────────
def compute_sector_flow(close: pd.DataFrame, volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    For each sector ETF, compute rolling 10d relative volume change vs SPY.
    Returns:
      - sector_flow_matrix: per-sector flow scores
      - sector_flow_breadth: how many sectors have positive flow (breadth)
    """
    spy_vol = volume["SPY"].rolling(10).mean()
    spy_vol_prev = volume["SPY"].rolling(10).mean().shift(10)
    spy_vol_chg = (spy_vol / spy_vol_prev - 1).replace([np.inf, -np.inf], 0)

    flows = {}
    for ticker, name in SECTOR_TICKERS.items():
        if ticker not in volume.columns:
            continue
        sec_vol = volume[ticker].rolling(10).mean()
        sec_vol_prev = volume[ticker].rolling(10).mean().shift(10)
        sec_vol_chg = (sec_vol / sec_vol_prev - 1).replace([np.inf, -np.inf], 0)

        # Relative volume change vs SPY
        rel_flow = sec_vol_chg - spy_vol_chg
        flows[name] = rel_flow

    sector_flow_matrix = pd.DataFrame(flows)

    # Composite: how dispersed is the flow? High dispersion = rotation happening
    sector_flow_dispersion = sector_flow_matrix.std(axis=1)
    sector_flow_dispersion.name = "sector_rotation_intensity"

    return sector_flow_matrix, sector_flow_dispersion


# ─── Signal 4: Rate-Sensitive Flow ──────────────────────────────────────────
def compute_fed_flow(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    Combine TLT momentum, HYG/LQD credit spread proxy, and bond volume
    to create a Fed positioning signal.

    - TLT rising + high volume = flight to safety / rate cut expectations
    - HYG/LQD ratio falling = credit stress
    """
    components = []

    # TLT momentum (20d)
    if "TLT" in close.columns:
        tlt_mom = close["TLT"].pct_change(20)
        tlt_z = (tlt_mom - tlt_mom.rolling(63).mean()) / tlt_mom.rolling(63).std()
        tlt_z = tlt_z.clip(-3, 3)
        components.append(tlt_z.rename("tlt_mom"))

    # HYG/LQD spread (credit quality indicator)
    if "HYG" in close.columns and "LQD" in close.columns:
        credit_ratio = close["HYG"] / close["LQD"]
        cr_mom = credit_ratio.pct_change(10)
        cr_z = (cr_mom - cr_mom.rolling(63).mean()) / cr_mom.rolling(63).std()
        cr_z = cr_z.clip(-3, 3)
        # Negative HYG/LQD momentum = credit stress = risk-off
        components.append(cr_z.rename("credit_momentum"))

    # TLT volume spike (unusual bond activity)
    if "TLT" in volume.columns:
        tlt_vol_ratio = volume["TLT"] / volume["TLT"].rolling(20).mean()
        tlt_vol_z = (tlt_vol_ratio - tlt_vol_ratio.rolling(63).mean()) / tlt_vol_ratio.rolling(63).std()
        tlt_vol_z = tlt_vol_z.clip(-3, 3)
        components.append(tlt_vol_z.rename("tlt_volume"))

    if not components:
        raise ValueError("No rate components computed")

    df = pd.concat(components, axis=1)
    # TLT momentum is inverted: rising TLT = rates falling = dovish
    # Credit momentum: positive = risk-on
    # Combine: positive = dovish/risk-on, negative = hawkish/risk-off
    fed_flow = df.mean(axis=1)
    fed_flow.name = "fed_flow"
    return fed_flow


# ─── Signal 5: Breadth Flow ─────────────────────────────────────────────────
def compute_breadth_flow(close: pd.DataFrame) -> pd.Series:
    """
    Advance-decline proxy: how many of 11 sectors are up on the day?
    Then compute rolling 10d breadth momentum.
    """
    sector_returns = pd.DataFrame()
    for ticker in SECTOR_TICKERS:
        if ticker in close.columns:
            sector_returns[ticker] = close[ticker].pct_change()

    # Daily breadth: fraction of sectors that are positive
    daily_breadth = (sector_returns > 0).sum(axis=1) / sector_returns.shape[1]

    # Rolling 10d breadth momentum (centered at 0.5)
    breadth_mom = daily_breadth.rolling(10).mean() - 0.5

    # Z-score for comparability
    breadth_z = (breadth_mom - breadth_mom.rolling(63).mean()) / breadth_mom.rolling(63).std()
    breadth_z = breadth_z.clip(-3, 3)
    breadth_z.name = "breadth_flow"
    return breadth_z


# ─── Forward Returns ────────────────────────────────────────────────────────
def compute_forward_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Compute SPY forward returns for various horizons."""
    fwd = pd.DataFrame(index=close.index)
    for w in FORWARD_WINDOWS:
        fwd[f"fwd_{w}d"] = close["SPY"].pct_change(w).shift(-w)
    return fwd


# ─── IC Computation ─────────────────────────────────────────────────────────
def compute_ic(signal: pd.Series, forward_ret: pd.Series) -> dict:
    """Compute rank IC (Spearman) between signal and forward returns."""
    aligned = pd.concat([signal, forward_ret], axis=1).dropna()
    if len(aligned) < 30:
        return {"ic": np.nan, "pval": np.nan, "n": len(aligned)}

    ic, pval = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return {"ic": round(float(ic), 4), "pval": round(float(pval), 6), "n": int(len(aligned))}


# ─── Backtest Engine ────────────────────────────────────────────────────────
def backtest_composite(signals: pd.DataFrame, spy_returns: pd.Series,
                       long_thresh: float = 0.5, short_thresh: float = -0.5) -> dict:
    """
    Simple long/short backtest on composite flow signal.
    Long SPY when composite > long_thresh, short when < short_thresh, else flat.
    """
    composite = signals.mean(axis=1)
    composite_z = (composite - composite.rolling(63).mean()) / composite.rolling(63).std()
    composite_z = composite_z.clip(-3, 3)

    # Position: +1 long, -1 short, 0 flat
    position = pd.Series(0.0, index=composite_z.index)
    position[composite_z > long_thresh] = 1.0
    position[composite_z < short_thresh] = -1.0

    # Lag position by 1 day (trade next day)
    position = position.shift(1)

    # Daily PnL
    daily_ret = spy_returns * position
    daily_ret = daily_ret.dropna()

    if len(daily_ret) < 252:
        return {"sharpe": np.nan, "cagr": np.nan, "maxdd": np.nan}

    # Metrics
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # CAGR
    cum = (1 + daily_ret).cumprod()
    years = len(daily_ret) / 252
    cagr = (cum.iloc[-1] ** (1 / years) - 1) if years > 0 else 0

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    maxdd = dd.min()

    # Sortino
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Win rate
    trading_days = daily_ret[daily_ret != 0]
    wr = (trading_days > 0).mean() if len(trading_days) > 0 else 0

    # Exposure
    exposure = (position.dropna() != 0).mean()

    # Long/short breakdown
    long_ret = daily_ret[position.shift(0) > 0]  # re-align
    short_ret = daily_ret[position.shift(0) < 0]

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "cagr": round(float(cagr * 100), 2),
        "maxdd": round(float(maxdd * 100), 2),
        "win_rate": round(float(wr * 100), 1),
        "exposure": round(float(exposure * 100), 1),
        "n_trading_days": int(len(trading_days)),
        "n_long_days": int(len(long_ret)),
        "n_short_days": int(len(short_ret)),
        "ann_return_pct": round(float(ann_ret * 100), 2),
        "ann_vol_pct": round(float(ann_vol * 100), 2),
    }


# ─── Contrarian Filter Test ─────────────────────────────────────────────────
def test_as_contrarian_filter(signals: pd.DataFrame, close: pd.DataFrame) -> dict:
    """
    Test whether macro flow signals improve contrarian entry signals.

    Simulate: oversold entries (SPY drops >1% in a day = contrarian buy signal).
    Compare forward returns conditioned on each macro flow signal state.
    """
    spy_ret = close["SPY"].pct_change()

    # Contrarian buy signal: SPY drops > 1% (oversold bounce candidate)
    entry_signal = spy_ret < -0.01

    # Forward returns after entries
    fwd_5d = close["SPY"].pct_change(5).shift(-5)
    fwd_20d = close["SPY"].pct_change(20).shift(-20)

    results = {}

    for sig_name in signals.columns:
        sig = signals[sig_name]
        sig_median = sig.median()

        # Split entries by signal state
        entries = entry_signal & sig.notna() & fwd_5d.notna()

        high_flow = entries & (sig > sig_median)
        low_flow = entries & (sig <= sig_median)

        n_high = high_flow.sum()
        n_low = low_flow.sum()

        if n_high < 10 or n_low < 10:
            results[sig_name] = {"status": "insufficient_data", "n_high": int(n_high), "n_low": int(n_low)}
            continue

        # 5d forward returns
        ret_high_5d = fwd_5d[high_flow].mean()
        ret_low_5d = fwd_5d[low_flow].mean()

        # 20d forward returns
        ret_high_20d = fwd_20d[high_flow & fwd_20d.notna()].mean()
        ret_low_20d = fwd_20d[low_flow & fwd_20d.notna()].mean()

        # T-test for significance
        t_stat_5d, p_val_5d = stats.ttest_ind(
            fwd_5d[high_flow].dropna(), fwd_5d[low_flow].dropna()
        )

        # Win rates
        wr_high_5d = (fwd_5d[high_flow] > 0).mean()
        wr_low_5d = (fwd_5d[low_flow] > 0).mean()

        results[sig_name] = {
            "n_high_flow_entries": int(n_high),
            "n_low_flow_entries": int(n_low),
            "5d_return": {
                "high_flow": round(float(ret_high_5d * 100), 3),
                "low_flow": round(float(ret_low_5d * 100), 3),
                "spread_bps": round(float((ret_high_5d - ret_low_5d) * 10000), 1),
                "t_stat": round(float(t_stat_5d), 2),
                "p_value": round(float(p_val_5d), 4),
                "wr_high": round(float(wr_high_5d * 100), 1),
                "wr_low": round(float(wr_low_5d * 100), 1),
            },
            "20d_return": {
                "high_flow": round(float(ret_high_20d * 100), 3) if not np.isnan(ret_high_20d) else None,
                "low_flow": round(float(ret_low_20d * 100), 3) if not np.isnan(ret_low_20d) else None,
            },
            "verdict": "IMPROVES" if (ret_high_5d > ret_low_5d and p_val_5d < 0.10) else
                       "MARGINAL" if (ret_high_5d > ret_low_5d) else "NO_HELP"
        }

    # Also test composite
    composite = signals.mean(axis=1)
    comp_median = composite.median()
    entries = entry_signal & composite.notna() & fwd_5d.notna()
    high_comp = entries & (composite > comp_median)
    low_comp = entries & (composite <= comp_median)

    if high_comp.sum() >= 10 and low_comp.sum() >= 10:
        ret_h = fwd_5d[high_comp].mean()
        ret_l = fwd_5d[low_comp].mean()
        t, p = stats.ttest_ind(fwd_5d[high_comp].dropna(), fwd_5d[low_comp].dropna())
        results["COMPOSITE"] = {
            "n_high": int(high_comp.sum()),
            "n_low": int(low_comp.sum()),
            "5d_return_high": round(float(ret_h * 100), 3),
            "5d_return_low": round(float(ret_l * 100), 3),
            "spread_bps": round(float((ret_h - ret_l) * 10000), 1),
            "t_stat": round(float(t), 2),
            "p_value": round(float(p), 4),
        }

    return results


# ─── Sector Flow Detail ─────────────────────────────────────────────────────
def summarize_sector_flows(sector_flow_matrix: pd.DataFrame) -> dict:
    """Summarize recent sector flow patterns."""
    recent = sector_flow_matrix.tail(20)

    avg_flow = recent.mean().sort_values(ascending=False)

    return {
        "top_inflows": {k: round(float(v * 100), 2) for k, v in avg_flow.head(3).items()},
        "top_outflows": {k: round(float(v * 100), 2) for k, v in avg_flow.tail(3).items()},
        "flow_dispersion_20d": round(float(recent.std().mean() * 100), 3),
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ENHANCED FLOW ANALYSIS v1 — Comprehensive Macro Flow Ecosystem")
    print("=" * 70)
    print()

    # 1. Download data
    close, volume = download_data()
    print()

    # 2. Compute all signals
    print("Computing flow signals...")

    cta = compute_cta_pressure(close)
    print(f"  [1] CTA Pressure: {cta.dropna().shape[0]} days, range [{cta.min():.2f}, {cta.max():.2f}]")

    risk_appetite = compute_risk_appetite(close, volume)
    print(f"  [2] Risk Appetite: {risk_appetite.dropna().shape[0]} days")

    sector_matrix, sector_dispersion = compute_sector_flow(close, volume)
    print(f"  [3] Sector Rotation: {sector_matrix.shape[1]} sectors, {sector_matrix.dropna().shape[0]} days")

    fed_flow = compute_fed_flow(close, volume)
    print(f"  [4] Fed Flow: {fed_flow.dropna().shape[0]} days")

    breadth = compute_breadth_flow(close)
    print(f"  [5] Breadth Flow: {breadth.dropna().shape[0]} days")

    # Combine into signal matrix
    signals = pd.DataFrame({
        "cta_pressure": cta,
        "risk_appetite": risk_appetite,
        "sector_rotation": sector_dispersion,
        "fed_flow": fed_flow,
        "breadth_flow": breadth,
    })
    signals = signals.dropna()
    print(f"\n  Combined signal matrix: {len(signals)} days with all signals")
    print()

    # 3. Compute forward returns
    fwd_returns = compute_forward_returns(close)

    # 4. Information Coefficient analysis
    print("=" * 70)
    print("SIGNAL PREDICTIVE POWER (IC = Rank Correlation with Forward Returns)")
    print("=" * 70)

    ic_results = {}
    for sig_name in signals.columns:
        ic_results[sig_name] = {}
        for fwd_name in fwd_returns.columns:
            ic = compute_ic(signals[sig_name], fwd_returns[fwd_name])
            ic_results[sig_name][fwd_name] = ic

    # Print IC table
    print(f"\n{'Signal':<22} {'IC(5d)':>8} {'p(5d)':>8} {'IC(20d)':>8} {'p(20d)':>8}")
    print("-" * 60)
    for sig_name, horizons in ic_results.items():
        ic5 = horizons.get("fwd_5d", {}).get("ic", np.nan)
        p5 = horizons.get("fwd_5d", {}).get("pval", np.nan)
        ic20 = horizons.get("fwd_20d", {}).get("ic", np.nan)
        p20 = horizons.get("fwd_20d", {}).get("pval", np.nan)
        star5 = "***" if p5 < 0.01 else "**" if p5 < 0.05 else "*" if p5 < 0.10 else ""
        star20 = "***" if p20 < 0.01 else "**" if p20 < 0.05 else "*" if p20 < 0.10 else ""
        print(f"  {sig_name:<20} {ic5:>7.4f}{star5:<3} {p5:>8.4f} {ic20:>7.4f}{star20:<3} {p20:>8.4f}")

    # Composite IC
    composite = signals.mean(axis=1)
    composite_ic = {}
    for fwd_name in fwd_returns.columns:
        composite_ic[fwd_name] = compute_ic(composite, fwd_returns[fwd_name])

    ic5c = composite_ic["fwd_5d"]["ic"]
    p5c = composite_ic["fwd_5d"]["pval"]
    ic20c = composite_ic["fwd_20d"]["ic"]
    p20c = composite_ic["fwd_20d"]["pval"]
    print("-" * 60)
    star5c = "***" if p5c < 0.01 else "**" if p5c < 0.05 else "*" if p5c < 0.10 else ""
    star20c = "***" if p20c < 0.01 else "**" if p20c < 0.05 else "*" if p20c < 0.10 else ""
    print(f"  {'COMPOSITE':<20} {ic5c:>7.4f}{star5c:<3} {p5c:>8.4f} {ic20c:>7.4f}{star20c:<3} {p20c:>8.4f}")
    print(f"\n  (* p<0.10, ** p<0.05, *** p<0.01)")

    # 5. Backtest composite signal
    print()
    print("=" * 70)
    print("COMPOSITE FLOW BACKTEST (Long SPY when bullish, Short when bearish)")
    print("=" * 70)

    spy_daily_ret = close["SPY"].pct_change()
    bt = backtest_composite(signals, spy_daily_ret)

    print(f"\n  Sharpe Ratio:     {bt['sharpe']:.3f}")
    print(f"  Sortino Ratio:    {bt['sortino']:.3f}")
    print(f"  CAGR:             {bt['cagr']:.2f}%")
    print(f"  Max Drawdown:     {bt['maxdd']:.2f}%")
    print(f"  Win Rate:         {bt['win_rate']:.1f}%")
    print(f"  Exposure:         {bt['exposure']:.1f}%")
    print(f"  Ann. Return:      {bt['ann_return_pct']:.2f}%")
    print(f"  Ann. Volatility:  {bt['ann_vol_pct']:.2f}%")
    print(f"  Long Days:        {bt['n_long_days']}")
    print(f"  Short Days:       {bt['n_short_days']}")

    # Buy-and-hold benchmark
    bnh_ret = spy_daily_ret.loc[signals.index[0]:].dropna()
    bnh_sharpe = (bnh_ret.mean() * 252) / (bnh_ret.std() * np.sqrt(252))
    bnh_cum = (1 + bnh_ret).cumprod()
    bnh_cagr = bnh_cum.iloc[-1] ** (1 / (len(bnh_ret)/252)) - 1
    print(f"\n  Buy-and-Hold SPY: Sharpe={bnh_sharpe:.3f}, CAGR={bnh_cagr*100:.2f}%")

    # 6. Contrarian filter test
    print()
    print("=" * 70)
    print("CONTRARIAN FILTER TEST (Do flow signals improve oversold-bounce trades?)")
    print("=" * 70)

    filter_results = test_as_contrarian_filter(signals, close)

    print(f"\n  Entry signal: SPY daily return < -1% (oversold bounce candidate)")
    print(f"\n  {'Flow Filter':<22} {'High Flow 5d':>12} {'Low Flow 5d':>12} {'Spread(bps)':>11} {'p-value':>8} {'Verdict':>10}")
    print("  " + "-" * 78)

    for sig_name, res in filter_results.items():
        if sig_name == "COMPOSITE" or "status" in res:
            continue
        r5 = res["5d_return"]
        print(f"  {sig_name:<22} {r5['high_flow']:>11.3f}% {r5['low_flow']:>11.3f}% {r5['spread_bps']:>10.1f} {r5['p_value']:>8.4f} {res['verdict']:>10}")

    if "COMPOSITE" in filter_results:
        cr = filter_results["COMPOSITE"]
        print("  " + "-" * 78)
        print(f"  {'COMPOSITE':<22} {cr['5d_return_high']:>11.3f}% {cr['5d_return_low']:>11.3f}% {cr['spread_bps']:>10.1f} {cr['p_value']:>8.4f}")

    # 7. Sector flow summary
    print()
    print("=" * 70)
    print("CURRENT SECTOR FLOWS (Last 20 Trading Days)")
    print("=" * 70)

    sector_summary = summarize_sector_flows(sector_matrix)
    print(f"\n  Top Inflows:  {sector_summary['top_inflows']}")
    print(f"  Top Outflows: {sector_summary['top_outflows']}")
    print(f"  Flow Dispersion: {sector_summary['flow_dispersion_20d']:.3f}%")

    # 8. Signal correlation matrix
    print()
    print("=" * 70)
    print("SIGNAL CORRELATION MATRIX")
    print("=" * 70)
    corr = signals.corr()
    print()
    print(corr.round(3).to_string())

    # 9. Save results
    results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "data_range": f"{START_DATE} to {END_DATE}",
            "n_trading_days": int(len(signals)),
            "script": "enhanced_flow_analysis_v1.py",
        },
        "information_coefficients": ic_results,
        "composite_ic": {k: v for k, v in composite_ic.items()},
        "backtest": bt,
        "benchmark": {
            "spy_buy_hold_sharpe": round(float(bnh_sharpe), 3),
            "spy_buy_hold_cagr": round(float(bnh_cagr * 100), 2),
        },
        "contrarian_filter_test": filter_results,
        "sector_flows": sector_summary,
        "signal_correlations": corr.round(4).to_dict(),
        "signal_stats": {
            col: {
                "mean": round(float(signals[col].mean()), 4),
                "std": round(float(signals[col].std()), 4),
                "min": round(float(signals[col].min()), 4),
                "max": round(float(signals[col].max()), 4),
                "skew": round(float(signals[col].skew()), 4),
            }
            for col in signals.columns
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")

    # 10. Summary verdict
    print()
    print("=" * 70)
    print("SUMMARY VERDICT")
    print("=" * 70)

    # Find best individual signal
    best_sig = max(ic_results.items(), key=lambda x: abs(x[1].get("fwd_5d", {}).get("ic", 0)))
    best_ic = best_sig[1]["fwd_5d"]["ic"]

    print(f"""
  Best individual signal: {best_sig[0]} (IC={best_ic:.4f} at 5d)
  Composite IC (5d):      {ic5c:.4f} (p={p5c:.4f})
  Composite IC (20d):     {ic20c:.4f} (p={p20c:.4f})

  Composite Backtest:     Sharpe={bt['sharpe']:.3f}, CAGR={bt['cagr']:.2f}%
  vs Buy-Hold SPY:        Sharpe={bnh_sharpe:.3f}, CAGR={bnh_cagr*100:.2f}%

  Contrarian filter value: {'YES' if any(r.get('verdict') == 'IMPROVES' for r in filter_results.values() if isinstance(r, dict) and 'verdict' in r) else 'MARGINAL/NO'}
  - Flow signals {'DO' if abs(ic5c) > 0.03 else 'may NOT'} have standalone predictive value for SPY
  - Flow signals {'DO' if any(r.get('verdict') in ('IMPROVES', 'MARGINAL') for r in filter_results.values() if isinstance(r, dict) and 'verdict' in r) else 'DO NOT'} help filter contrarian entries
""")

    print("DONE.")
    return results


if __name__ == "__main__":
    main()
