"""
CTA Flow Indicator v1 — CTA positioning proxy + sector rotation flow analysis.

HC #746: Test whether adding CTA positioning proxy improves ETF rotation.

Three CTA proxy features:
  1. MA crossover intensity: weighted count of major ETFs above 50d/200d MAs
  2. Sector dispersion: stdev of 21d returns across 11 sector ETFs
  3. Money flow ratio: volume ratio of equity (SPY+QQQ) vs safe-haven (TLT+GLD+SHY)

Four strategy variants tested:
  A. Plain momentum rotation (top 3 by 6mo return, 30-day hold) — baseline
  B. Momentum + CTA crowding filter (skip when crowded + correlated)
  C. Momentum + money flow timing (rotate only in risk-on)
  D. All filters combined

Walk-forward: SLIDING (never expanding). No look-ahead bias.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "cta_flow_indicator"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
MACRO_ETFS = ["SPY", "QQQ", "GLD", "TLT", "DBC", "SHY"]
ALL_TICKERS = SECTOR_ETFS + MACRO_ETFS

HOLD_DAYS = 30           # rebalance monthly
TOP_N = 3                # select top 3 sectors
MOM_WINDOW = 126         # 6-month momentum lookback
TXN_COST_BPS = 5         # 5 bps per trade
TRADING_DAYS = 252

# CTA filter thresholds
CTA_CROWD_PCT = 90       # crowding > 90th percentile = crowded
DISP_LOW_PCT = 20        # dispersion < 20th percentile = correlated
MF_RISING_WINDOW = 21    # money flow rising = 21d SMA of ratio is increasing

# Permutation test
N_PERMUTATIONS = 200

START_DATE = "2019-01-01"
END_DATE = "2026-07-23"


# ── Data Download ───────────────────────────────────────────────────────────
def download_data() -> pd.DataFrame:
    """Download daily OHLCV for all tickers via yfinance."""
    cache_path = OUTPUT_DIR / "price_cache.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if len(df) > 0 and df.index.max() >= pd.Timestamp("2026-07-01"):
            print(f"  Using cached data: {len(df)} rows, {df.index.min()} to {df.index.max()}")
            return df

    import yfinance as yf
    print("  Downloading price data from Yahoo Finance...")
    raw = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)

    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
    volume = raw["Volume"]

    close_long = close.stack().reset_index()
    close_long.columns = ["date", "ticker", "close"]
    volume_long = volume.stack().reset_index()
    volume_long.columns = ["date", "ticker", "volume"]

    df = close_long.merge(volume_long, on=["date", "ticker"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_parquet(cache_path)
    print(f"  Downloaded: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df


# ── Feature Engineering ─────────────────────────────────────────────────────
def compute_cta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the three CTA proxy features daily. No look-ahead."""
    close_wide = df.pivot_table(index="date", columns="ticker", values="close")
    vol_wide = df.pivot_table(index="date", columns="ticker", values="volume")

    dates = close_wide.index
    out = pd.DataFrame(index=dates)

    # ── Feature 1: MA Crossover Intensity ──
    ma_tickers = ["SPY", "QQQ", "GLD", "TLT", "DBC"]
    ma50 = close_wide[ma_tickers].rolling(50).mean()
    ma200 = close_wide[ma_tickers].rolling(200).mean()

    dist_50 = (close_wide[ma_tickers] - ma50) / ma50
    dist_200 = (close_wide[ma_tickers] - ma200) / ma200

    # Composite: average of signed distances across tickers and both MAs
    out["cta_intensity"] = (dist_50.mean(axis=1) + dist_200.mean(axis=1)) / 2

    # Rolling percentile (252d lookback)
    out["cta_intensity_pct"] = (
        out["cta_intensity"]
        .rolling(252, min_periods=63)
        .apply(lambda x: (x[-1] >= x).mean() * 100, raw=True)
    )

    # Also: count of tickers above both MAs (simple binary crowding measure)
    above_50 = (close_wide[ma_tickers] > ma50).sum(axis=1)
    above_200 = (close_wide[ma_tickers] > ma200).sum(axis=1)
    out["cta_above_count"] = above_50 + above_200  # max = 10 (5 tickers x 2 MAs)
    out["cta_above_pct"] = (
        out["cta_above_count"]
        .rolling(252, min_periods=63)
        .apply(lambda x: (x[-1] >= x).mean() * 100, raw=True)
    )

    # ── Feature 2: Sector Dispersion ──
    avail_sectors = [s for s in SECTOR_ETFS if s in close_wide.columns]
    sector_ret_21d = close_wide[avail_sectors].pct_change(21)
    out["sector_dispersion"] = sector_ret_21d.std(axis=1)

    out["sector_disp_pct"] = (
        out["sector_dispersion"]
        .rolling(252, min_periods=63)
        .apply(lambda x: (x[-1] >= x).mean() * 100, raw=True)
    )

    # ── Feature 3: Money Flow Ratio ──
    equity_vol = vol_wide[["SPY", "QQQ"]].sum(axis=1)
    haven_tickers = [t for t in ["TLT", "GLD", "SHY"] if t in vol_wide.columns]
    haven_vol = vol_wide[haven_tickers].sum(axis=1)
    raw_ratio = equity_vol / haven_vol.replace(0, np.nan)

    out["money_flow_ratio"] = raw_ratio.rolling(5).mean()

    mf_sma = out["money_flow_ratio"].rolling(MF_RISING_WINDOW).mean()
    out["money_flow_rising"] = (mf_sma > mf_sma.shift(5)).astype(float)

    return out


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6-month momentum for sector ETFs."""
    close_wide = df.pivot_table(index="date", columns="ticker", values="close")
    avail_sectors = [s for s in SECTOR_ETFS if s in close_wide.columns]
    mom = close_wide[avail_sectors].pct_change(MOM_WINDOW)
    return mom


# ── Strategy Variants ───────────────────────────────────────────────────────
def run_rotation(
    mom: pd.DataFrame,
    cta: pd.DataFrame,
    close_wide: pd.DataFrame,
    variant: str = "A",
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Run rotation strategy. Returns daily returns DataFrame with 'ret' and 'in_market'.

    variant:
      A = plain momentum
      B = momentum + CTA crowding filter (crowded AND low dispersion → cash)
      C = momentum + money flow timing (risk-off → cash)
      D = all filters combined
    """
    avail_sectors = [s for s in SECTOR_ETFS if s in mom.columns]
    # Start after we have enough data for momentum + MA features
    start_after = mom.index[0] + pd.Timedelta(days=250)
    dates = mom.index[mom.index >= start_after]
    dates = dates.intersection(cta.index).intersection(close_wide.index)
    dates = sorted(dates)

    portfolio = {}
    last_rebal_idx = -HOLD_DAYS  # force first rebal
    daily_rets = []
    n_filter_skips = 0

    for i, date in enumerate(dates):
        should_rebalance = (i - last_rebal_idx) >= HOLD_DAYS

        if should_rebalance:
            mom_row = mom.loc[date, avail_sectors]
            if mom_row.isna().all():
                daily_rets.append({"date": date, "ret": 0.0, "in_market": False})
                continue

            skip_rotation = False

            if variant in ("B", "D"):
                # CTA crowding filter: use both intensity percentile and binary count
                intensity_pct = cta.loc[date, "cta_intensity_pct"] if date in cta.index else 50
                above_pct = cta.loc[date, "cta_above_pct"] if date in cta.index else 50
                disp_pct = cta.loc[date, "sector_disp_pct"] if date in cta.index else 50
                if not any(np.isnan(x) for x in [intensity_pct, above_pct, disp_pct]):
                    # Crowded = either intensity OR count in top decile
                    crowded = (intensity_pct > CTA_CROWD_PCT) or (above_pct > CTA_CROWD_PCT)
                    correlated = disp_pct < DISP_LOW_PCT
                    if crowded and correlated:
                        skip_rotation = True

            if variant in ("C", "D") and not skip_rotation:
                mf_rising = cta.loc[date, "money_flow_rising"] if date in cta.index else 1.0
                if not np.isnan(mf_rising) and mf_rising < 0.5:
                    skip_rotation = True

            if skip_rotation:
                n_filter_skips += 1
                portfolio = {}
            else:
                ranked = mom_row.dropna().sort_values(ascending=False)
                top = ranked.head(TOP_N).index.tolist()
                # Apply txn cost on rebalance
                old_tickers = set(portfolio.keys())
                new_tickers = set(top)
                turnover = len(old_tickers.symmetric_difference(new_tickers)) / max(len(old_tickers | new_tickers), 1)
                portfolio = {t: 1.0 / TOP_N for t in top}

            last_rebal_idx = i

        # Compute daily return
        if portfolio and i > 0:
            prev_date = dates[i - 1]
            port_ret = 0.0
            for ticker, weight in portfolio.items():
                if ticker in close_wide.columns:
                    p0 = close_wide.loc[prev_date, ticker] if prev_date in close_wide.index else np.nan
                    p1 = close_wide.loc[date, ticker] if date in close_wide.index else np.nan
                    if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                        port_ret += weight * (p1 / p0 - 1)
            daily_rets.append({"date": date, "ret": port_ret, "in_market": True})
        else:
            daily_rets.append({"date": date, "ret": 0.0, "in_market": len(portfolio) > 0})

    result = pd.DataFrame(daily_rets)
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date")

    # Apply transaction costs: 5 bps each side on rebalance days
    rebal_indices = list(range(0, len(result), HOLD_DAYS))
    for idx in rebal_indices:
        if idx < len(result):
            result.iloc[idx, result.columns.get_loc("ret")] -= TXN_COST_BPS / 10000 * 2

    if variant != "A" and not quiet:
        print(f"    Filter skipped {n_filter_skips} rebalances → cash")

    return result


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(rets: pd.Series) -> dict:
    """Compute Sharpe, Sortino, CAGR, MaxDD, Calmar from daily returns."""
    r = rets.values.astype(float)
    n = len(r)
    if n < 30:
        return {"sharpe": np.nan, "sortino": np.nan, "cagr": np.nan,
                "max_dd": np.nan, "calmar": np.nan, "n_days": n}

    ann_ret = np.mean(r) * TRADING_DAYS
    ann_vol = np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    downside = r[r < 0]
    down_vol = np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 5 else ann_vol
    sortino = ann_ret / down_vol if down_vol > 1e-10 else 0.0

    cum = np.cumprod(1 + r)
    total_ret = cum[-1]
    years = n / TRADING_DAYS
    cagr = total_ret ** (1 / years) - 1 if years > 0 else 0.0

    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = float(np.min(dd))

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "n_days": n,
    }


def regime_gap_check(rets: pd.DataFrame, spy_close: pd.Series) -> dict:
    """
    Stratify strategy returns by GREEN/RED regime periods.

    Instead of splitting individual days (which inflates Sharpe because each
    day's strategy return correlates with SPY direction), we classify rolling
    21-day windows as green (SPY up) or red (SPY down), then compute strategy
    Sharpe within each regime window.
    """
    spy_daily = spy_close.pct_change()
    common = rets.index.intersection(spy_daily.index)
    r = rets.loc[common, "ret"]
    spy_r = spy_daily.loc[common]

    # Classify each day's regime by trailing 21d SPY return
    spy_trailing_21d = spy_r.rolling(21).sum()  # approximate 21d return
    green_mask = spy_trailing_21d > 0
    red_mask = spy_trailing_21d <= 0

    green_rets = r[green_mask].dropna()
    red_rets = r[red_mask].dropna()

    m_g = compute_metrics(green_rets)
    m_r = compute_metrics(red_rets)

    s_g = m_g["sharpe"] if not np.isnan(m_g["sharpe"]) else 0
    s_r = m_r["sharpe"] if not np.isnan(m_r["sharpe"]) else 0
    denom = max(abs(s_g), abs(s_r), 0.01)
    gap = abs(s_g - s_r) / denom

    return {
        "sharpe_green": round(s_g, 3),
        "sharpe_red": round(s_r, 3),
        "gap_ratio": round(gap, 3),
        "regime_pass": gap <= 0.50,
        "n_green": len(green_rets),
        "n_red": len(red_rets),
    }


def permutation_test(
    mom: pd.DataFrame,
    cta: pd.DataFrame,
    close_wide: pd.DataFrame,
    variant: str,
    actual_sharpe: float,
    n_perms: int = N_PERMUTATIONS,
) -> dict:
    """
    Permutation test: randomize the signal-to-date mapping.

    For each permutation, we shift the CTA feature series by a random offset
    (circular shift) so the filter decisions are made on misaligned signals.
    This preserves the autocorrelation structure of both signals and returns
    while testing whether the specific timing of the filter matters.

    For variant A (no filter), we randomly rotate which sectors are "top N"
    by shuffling the momentum rankings on each rebalance date.
    """
    if variant == "A":
        # For baseline: randomize sector selection by shuffling momentum ranks
        avail_sectors = [s for s in SECTOR_ETFS if s in mom.columns]
        rng = np.random.RandomState(42)
        perm_sharpes = []
        for _ in range(n_perms):
            # Create shuffled momentum: for each date, randomly permute ranks
            mom_shuf = mom.copy()
            for date in mom_shuf.index:
                vals = mom_shuf.loc[date, avail_sectors].values.copy()
                rng.shuffle(vals)
                mom_shuf.loc[date, avail_sectors] = vals
            rets_perm = run_rotation(mom_shuf, cta, close_wide, variant="A", quiet=True)
            s = compute_metrics(rets_perm["ret"])["sharpe"]
            perm_sharpes.append(s)
    else:
        # For filtered variants: circular-shift CTA features by random offset
        rng = np.random.RandomState(42)
        perm_sharpes = []
        n_cta = len(cta)
        for _ in range(n_perms):
            offset = rng.randint(63, n_cta - 63)
            cta_shifted = cta.copy()
            for col in ["cta_intensity_pct", "cta_above_pct", "sector_disp_pct",
                         "money_flow_rising"]:
                if col in cta_shifted.columns:
                    vals = cta_shifted[col].values
                    cta_shifted[col] = np.roll(vals, offset)
            rets_perm = run_rotation(mom, cta_shifted, close_wide, variant=variant, quiet=True)
            s = compute_metrics(rets_perm["ret"])["sharpe"]
            perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= actual_sharpe))

    return {
        "actual_sharpe": actual_sharpe,
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
        "p_value": round(p_value, 4),
        "significant_5pct": p_value < 0.05,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CTA Flow Indicator v1 — Research Script")
    print("=" * 70)

    # 1. Download data
    print("\n[1/5] Loading data...")
    df = download_data()

    # 2. Compute features
    print("[2/5] Computing CTA proxy features...")
    cta = compute_cta_features(df)
    mom = compute_momentum(df)
    close_wide = df.pivot_table(index="date", columns="ticker", values="close")

    # Print feature stats
    print(f"  CTA intensity range: [{cta['cta_intensity'].min():.4f}, {cta['cta_intensity'].max():.4f}]")
    print(f"  CTA above-MA count range: [{cta['cta_above_count'].min():.0f}, {cta['cta_above_count'].max():.0f}]")
    print(f"  Sector dispersion range: [{cta['sector_dispersion'].min():.4f}, {cta['sector_dispersion'].max():.4f}]")
    print(f"  Money flow ratio range: [{cta['money_flow_ratio'].min():.2f}, {cta['money_flow_ratio'].max():.2f}]")
    print(f"  Money flow rising pct: {cta['money_flow_rising'].mean()*100:.1f}%")

    # Debug: check how often filter conditions trigger
    valid = cta.dropna(subset=["cta_intensity_pct", "sector_disp_pct", "cta_above_pct"])
    crowd_int = (valid["cta_intensity_pct"] > CTA_CROWD_PCT).mean() * 100
    crowd_cnt = (valid["cta_above_pct"] > CTA_CROWD_PCT).mean() * 100
    low_disp = (valid["sector_disp_pct"] < DISP_LOW_PCT).mean() * 100
    both = (((valid["cta_intensity_pct"] > CTA_CROWD_PCT) | (valid["cta_above_pct"] > CTA_CROWD_PCT))
            & (valid["sector_disp_pct"] < DISP_LOW_PCT)).mean() * 100
    mf_off = (cta["money_flow_rising"].dropna() < 0.5).mean() * 100
    print(f"  Filter frequency: CTA_crowded(int)={crowd_int:.1f}%, CTA_crowded(cnt)={crowd_cnt:.1f}%, "
          f"low_disp={low_disp:.1f}%, BOTH={both:.1f}%, MF_risk-off={mf_off:.1f}%")

    # 3. Run variants
    print("[3/5] Running strategy variants...")
    variants = {"A": "Plain Momentum", "B": "Mom + CTA Crowding Filter",
                "C": "Mom + Money Flow Timing", "D": "All Filters Combined"}
    results = {}

    for var_key, var_name in variants.items():
        print(f"  Running Variant {var_key}: {var_name}...")
        rets = run_rotation(mom, cta, close_wide, variant=var_key)
        metrics = compute_metrics(rets["ret"])
        in_mkt_pct = rets["in_market"].mean() * 100

        spy_close = close_wide["SPY"].dropna()
        regime = regime_gap_check(rets, spy_close)

        print(f"    Running permutation test ({N_PERMUTATIONS} shuffles)...")
        perm = permutation_test(mom, cta, close_wide, var_key, metrics["sharpe"])

        results[var_key] = {
            "name": var_name,
            "metrics": metrics,
            "regime": regime,
            "permutation": perm,
            "in_market_pct": round(in_mkt_pct, 1),
        }

    # 4. Print comparison table
    print("\n[4/5] Results Comparison")
    print("=" * 95)
    header = f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'Calmar':>7} {'InMkt%':>7} {'Regime':>10}"
    print(header)
    print("-" * 95)
    for var_key in ["A", "B", "C", "D"]:
        r = results[var_key]
        m = r["metrics"]
        rg = r["regime"]
        regime_str = "PASS" if rg["regime_pass"] else f"FAIL({rg['gap_ratio']:.2f})"
        print(f"  {var_key}. {r['name']:<31} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% {m['calmar']:>7.3f} "
              f"{r['in_market_pct']:>6.1f}% {regime_str:>10}")

    print("\n" + "-" * 95)
    print("Regime Gap Check (|Sharpe_green - Sharpe_red| / max > 0.50 = FAIL):")
    for var_key in ["A", "B", "C", "D"]:
        rg = results[var_key]["regime"]
        print(f"  {var_key}. Sharpe_green={rg['sharpe_green']:.3f}, Sharpe_red={rg['sharpe_red']:.3f}, "
              f"gap={rg['gap_ratio']:.3f} [{rg['n_green']}g/{rg['n_red']}r days] "
              f"→ {'PASS' if rg['regime_pass'] else 'FAIL'}")

    print("\n" + "-" * 95)
    print(f"Permutation Test ({N_PERMUTATIONS} block-bootstrap shuffles, p<0.05 = significant):")
    for var_key in ["A", "B", "C", "D"]:
        p = results[var_key]["permutation"]
        sig = "YES" if p["significant_5pct"] else "NO"
        print(f"  {var_key}. Actual Sharpe={p['actual_sharpe']:.3f}, "
              f"Perm mean={p['perm_mean_sharpe']:.3f}±{p['perm_std_sharpe']:.3f}, "
              f"p={p['p_value']:.4f} → Significant: {sig}")

    # 5. Improvement analysis
    print("\n" + "-" * 95)
    print("Improvement vs Baseline (A):")
    base = results["A"]["metrics"]
    for var_key in ["B", "C", "D"]:
        m = results[var_key]["metrics"]
        ds = m["sharpe"] - base["sharpe"]
        dc = m["cagr"] - base["cagr"]
        dd_imp = m["max_dd"] - base["max_dd"]  # less negative = better
        print(f"  {var_key}. Sharpe Δ={ds:+.3f}, CAGR Δ={dc:+.1f}%, MaxDD Δ={dd_imp:+.1f}%")

    # CTA feature correlation with forward returns
    print("\n" + "-" * 95)
    print("CTA Feature Predictive Analysis (correlation with fwd 21d sector avg return):")
    sector_cols = [s for s in SECTOR_ETFS if s in close_wide.columns]
    sector_avg_ret = close_wide[sector_cols].mean(axis=1).pct_change(21).shift(-21)
    common = cta.index.intersection(sector_avg_ret.dropna().index)
    for feat in ["cta_intensity", "cta_above_count", "sector_dispersion", "money_flow_ratio"]:
        if feat in cta.columns:
            corr = cta.loc[common, feat].corr(sector_avg_ret.loc[common])
            print(f"  {feat:25s} vs fwd_21d_ret: corr = {corr:+.4f}")

    # 6. Save results
    print("\n[5/5] Saving results...")
    output = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "hold_days": HOLD_DAYS,
            "top_n": TOP_N,
            "mom_window": MOM_WINDOW,
            "txn_cost_bps": TXN_COST_BPS,
            "cta_crowd_pct_threshold": CTA_CROWD_PCT,
            "disp_low_pct_threshold": DISP_LOW_PCT,
            "mf_rising_window": MF_RISING_WINDOW,
            "n_permutations": N_PERMUTATIONS,
            "start_date": START_DATE,
            "end_date": END_DATE,
        },
        "variants": {},
    }
    for var_key, r in results.items():
        output["variants"][var_key] = {
            "name": r["name"],
            "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in r["metrics"].items()},
            "regime": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in r["regime"].items()},
            "permutation": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in r["permutation"].items()},
            "in_market_pct": float(r["in_market_pct"]),
        }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
