#!/usr/bin/env python3
"""
Sector PnL Decomposition v1 — Which sectors drive alpha?
=========================================================

Single run with v6 production parameters (K=2, weekly, 2% OTM, DTE=21).
Decomposes PnL by sector to answer:
  - Which sectors contribute most to the Sharpe 2.43 aggregate?
  - Would concentrating on top sectors improve risk-adjusted returns?
  - Which sectors drag performance?

Analyses:
  1. Per-sector metrics (PnL, WR, Sharpe contribution, selection frequency)
  2. Sector exclusion (remove each sector one at a time)
  3. Sector concentration (top-5 only, top-3 only)
  4. Temporal analysis (VIX regime, early vs late, LGBM selection momentum)

Base: moneyness_crossval_v1.py (production v4 infrastructure).
"""

import json
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Standardized tools ──
sys.path.insert(0, "/home/nick/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/nick/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "sector_pnl_decomp_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# v6 production parameters
TOP_K = 2
REBAL_DAYS = 5
DTE = 21
MONEYNESS_PCT = 2.0  # 2% OTM

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "sector_pnl_decomp_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable - results saved to disk only")


# Legacy 18 features
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

VALIDATED_CROSS_ASSET = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

V4_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)
    close = close.ffill()
    high = high.ffill()
    low = low.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    for t in ["SPY", "VIX"]:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")
    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


# ══════════════════════════════════════════════════════════════
# REGIME LOADING
# ══════════════════════════════════════════════════════════════

def load_regime_predictions():
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found at {REGIME_FILE}")
        fprint("  Will use VIX-based regime proxy instead")
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days "
           f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")
    return regime_series


def get_regime_score_at(regime_series, dt):
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (IDENTICAL TO PRODUCTION V4)
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]
    f["up_capture"] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0
    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df, sector_list):
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None
    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in VALIDATED_CROSS_ASSET}
    spy_ret = spy.pct_change().dropna()
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            f["sector_relative_vol_21d"] = float(sec_ret.iloc[-21:].std() / (spy_ret.iloc[-21:].std() + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0
    sector_cols = [c for c in sector_list if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion"] = 0.01
    else:
        f["cross_sector_dispersion"] = 0.01
    return f


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING (parameterized sector universe)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series,
                          dte, sector_universe, regime_mode="bull_bear"):
    import lightgbm as lgb
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(sector_universe)} sectors, "
           f"mode={regime_mode}, dte={dte}")
    records = []
    sector_cols = [c for c in sector_universe if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        rscore = get_regime_score_at(regime_series, dt)
        if regime_mode == "bull_bear":
            if rscore > REGIME_BULL_THRESHOLD:
                direction = "bull"
            elif rscore < REGIME_BEAR_THRESHOLD:
                direction = "bear"
            else:
                continue
        else:
            direction = "bull"

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]
            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue
            cross_asset = {}
            for col in feature_cols:
                if col in VALIDATED_CROSS_ASSET:
                    cross_asset = compute_cross_asset_features(tk, idx, close, sector_universe)
                    break
            fi = min(idx + dte, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "direction": direction}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols, label):
    import lightgbm as lgb
    if len(df) < 100:
        fprint(f"    {label}: Insufficient data ({len(df)} records)")
        return {}
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 50:
            continue
        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            direction = test_df["direction"].iloc[0] if "direction" in test_df.columns else "bull"
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "direction": direction,
            }
        except Exception:
            continue

    fprint(f"    {label}: {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (returns enriched trade records)
# ══════════════════════════════════════════════════════════════

def simulate_trades(rankings, close, high, low, regime_series, atr_dict,
                    top_k=TOP_K, dte=DTE, moneyness_pct=MONEYNESS_PCT,
                    sector_universe=None, skip_vix_25_30=True):
    """Simulate trades, returning enriched per-trade records."""
    if sector_universe is None:
        sector_universe = SECTORS

    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        if skip_vix_25_30 and 25.0 <= cv <= 30.0:
            continue

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        direction = ranking_data["direction"]

        # Filter scores to allowed sector universe
        scores = {k: v for k, v in scores.items() if k in sector_universe}
        if not scores:
            continue

        if direction == "bull":
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1])

        picks = [t for t, _ in ranked[:top_k]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        rscore = get_regime_score_at(regime_series, dt)

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= top_k:
                continue
            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + dte, len(close) - 1)
            if ei <= di:
                continue
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            K1 = round(S * (1 + moneyness_pct / 100), 2)
            K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                if direction == "bull":
                    entry_cost_ps, max_profit_ps = price_bull_call_spread(
                        S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv)
                else:
                    entry_cost_ps, max_profit_ps = price_bear_put_spread(
                        S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv)
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            Se = float(close[tk].iloc[ei])
            if direction == "bull":
                intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            else:
                intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

            pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # LGBM rank for this sector on this date
            lgbm_rank = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rank_pos = next((i for i, (t, _) in enumerate(lgbm_rank) if t == tk), -1) + 1

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "direction": direction,
                "vix": round(cv, 1),
                "regime_score": round(rscore, 3),
                "win": pnl > 0,
                "lgbm_rank": rank_pos,
                "lgbm_score": round(scores.get(tk, 0), 4),
                "entry_price": round(S, 2),
                "exit_price": round(Se, 2),
                "entry_cost": round(total_cost, 2),
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# ANALYSIS 1: PER-SECTOR METRICS
# ══════════════════════════════════════════════════════════════

def per_sector_analysis(trades, rankings):
    """Compute detailed per-sector metrics."""
    fprint(f"\n{'=' * 80}")
    fprint("ANALYSIS 1: PER-SECTOR PnL DECOMPOSITION")
    fprint(f"{'=' * 80}")

    sector_trades = defaultdict(list)
    for t in trades:
        sector_trades[t["ticker"]].append(t)

    # Count how often each sector appears in top-K selections across all dates
    selection_counts = defaultdict(int)
    total_dates = len(rankings)
    for dt, data in rankings.items():
        scores = data["scores"]
        direction = data["direction"]
        if direction == "bull":
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1])
        for tk, _ in ranked[:TOP_K]:
            selection_counts[tk] += 1

    results = {}
    fprint(f"\n{'Sector':<8} {'Trades':>6} {'Bull':>5} {'Bear':>5} {'WR':>6} {'TotalPnL':>10} "
           f"{'AvgPnL':>8} {'Sharpe':>7} {'Select%':>8} {'AvgRank':>8}")
    fprint("-" * 90)

    for tk in SECTORS:
        st = sector_trades.get(tk, [])
        if not st:
            fprint(f"  {tk:<8} {'—no trades—':>60}")
            results[tk] = {"n_trades": 0, "total_pnl": 0, "sharpe_contribution": 0}
            continue

        pnls = [t["pnl"] for t in st]
        bull_trades = [t for t in st if t["direction"] == "bull"]
        bear_trades = [t for t in st if t["direction"] == "bear"]
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        pnl_std = np.std(pnls) if len(pnls) > 1 else 1e-10
        sharpe_contrib = avg_pnl / (pnl_std + 1e-10) * np.sqrt(52)  # weekly
        select_pct = selection_counts.get(tk, 0) / max(total_dates, 1) * 100
        avg_rank = np.mean([t["lgbm_rank"] for t in st])

        results[tk] = {
            "n_trades": len(st),
            "n_bull": len(bull_trades),
            "n_bear": len(bear_trades),
            "win_rate": round(wins / len(st), 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "pnl_std": round(pnl_std, 2),
            "sharpe_contribution": round(sharpe_contrib, 3),
            "selection_pct": round(select_pct, 2),
            "avg_lgbm_rank": round(avg_rank, 2),
            "selection_count": selection_counts.get(tk, 0),
        }

        fprint(f"  {tk:<8} {len(st):>6} {len(bull_trades):>5} {len(bear_trades):>5} "
               f"{wins/len(st)*100:>5.1f}% ${total_pnl:>9.0f} ${avg_pnl:>7.1f} "
               f"{sharpe_contrib:>7.2f} {select_pct:>7.1f}% {avg_rank:>7.1f}")

    # PnL concentration
    total_pnl = sum(r["total_pnl"] for r in results.values())
    fprint(f"\n  Total PnL across all sectors: ${total_pnl:.0f}")
    sorted_sectors = sorted(results.items(), key=lambda x: x[1].get("total_pnl", 0), reverse=True)
    fprint(f"\n  PnL concentration (cumulative):")
    cum = 0
    for tk, r in sorted_sectors:
        if r["n_trades"] == 0:
            continue
        cum += r["total_pnl"]
        pct = cum / max(abs(total_pnl), 1) * 100
        fprint(f"    {tk}: ${r['total_pnl']:>8.0f} (cumulative: {pct:>6.1f}%)")

    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: SECTOR EXCLUSION
# ══════════════════════════════════════════════════════════════

def sector_exclusion_analysis(close, high, low, regime_series, atr_dict, feature_cols,
                              rebal_dates, full_sharpe):
    """Run strategy excluding each sector one at a time."""
    fprint(f"\n{'=' * 80}")
    fprint("ANALYSIS 2: SECTOR EXCLUSION (remove one at a time)")
    fprint(f"{'=' * 80}")
    fprint(f"  Full universe Sharpe: {full_sharpe:.3f}")
    fprint()

    results = {}
    for exclude_tk in SECTORS:
        reduced_universe = [s for s in SECTORS if s != exclude_tk]
        fprint(f"  Excluding {exclude_tk} ({len(reduced_universe)} sectors remaining)...")

        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols,
            regime_series, dte=DTE, sector_universe=reduced_universe,
        )
        rankings = walk_forward_lgbm_rank(records, feature_cols, f"excl_{exclude_tk}")

        if not rankings:
            results[exclude_tk] = {"sharpe": 0, "error": "no_rankings"}
            continue

        trades, final_eq = simulate_trades(
            rankings, close, high, low, regime_series, atr_dict,
            sector_universe=reduced_universe,
        )

        if trades and len(trades) >= 10:
            vr = validate_trades(trades, initial_capital=CAP, spy_prices=close["SPY"],
                                 strategy_name=f"excl_{exclude_tk}")
            delta = vr.sharpe - full_sharpe
            marker = " DRAG" if delta > 0.05 else (" CONTRIBUTOR" if delta < -0.05 else "")
            fprint(f"    Sharpe={vr.sharpe:.3f} (delta={delta:+.3f}){marker}, "
                   f"WR={vr.win_rate*100:.1f}%, {len(trades)} trades")
            results[exclude_tk] = {
                "sharpe": round(vr.sharpe, 4),
                "sharpe_delta": round(delta, 4),
                "win_rate": round(vr.win_rate, 4),
                "n_trades": len(trades),
                "final_equity": round(final_eq, 2),
                "sortino": round(vr.sortino, 4),
                "profit_factor": round(vr.profit_factor, 4),
            }
        else:
            results[exclude_tk] = {"sharpe": 0, "error": "insufficient_trades"}

    # Summary
    fprint(f"\n  EXCLUSION IMPACT RANKING (positive delta = sector was dragging):")
    sorted_excl = sorted(results.items(),
                         key=lambda x: x[1].get("sharpe_delta", 0), reverse=True)
    for tk, r in sorted_excl:
        if "error" in r:
            fprint(f"    {tk}: ERROR - {r['error']}")
        else:
            label = "DRAG" if r["sharpe_delta"] > 0.05 else ("KEY" if r["sharpe_delta"] < -0.05 else "NEUTRAL")
            fprint(f"    {tk}: delta={r['sharpe_delta']:+.3f} [{label}]")

    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: SECTOR CONCENTRATION
# ══════════════════════════════════════════════════════════════

def sector_concentration_analysis(close, high, low, regime_series, atr_dict,
                                  feature_cols, rebal_dates, sector_results, full_sharpe):
    """Test concentrated universes (top-5, top-3 by historical PnL)."""
    fprint(f"\n{'=' * 80}")
    fprint("ANALYSIS 3: SECTOR CONCENTRATION")
    fprint(f"{'=' * 80}")

    # Rank sectors by total PnL
    ranked = sorted(sector_results.items(),
                    key=lambda x: x[1].get("total_pnl", 0), reverse=True)
    ranked_sectors = [tk for tk, r in ranked if r.get("n_trades", 0) > 0]

    results = {}
    configs = [
        ("top_5", ranked_sectors[:5]),
        ("top_3", ranked_sectors[:3]),
        ("bottom_5", ranked_sectors[-5:] if len(ranked_sectors) >= 5 else ranked_sectors),
    ]

    for label, universe in configs:
        fprint(f"\n  {label.upper()}: {universe}")

        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols,
            regime_series, dte=DTE, sector_universe=universe,
        )
        rankings = walk_forward_lgbm_rank(records, feature_cols, label)

        if not rankings:
            results[label] = {"error": "no_rankings"}
            continue

        trades, final_eq = simulate_trades(
            rankings, close, high, low, regime_series, atr_dict,
            sector_universe=universe,
        )

        if trades and len(trades) >= 10:
            vr = validate_trades(trades, initial_capital=CAP, spy_prices=close["SPY"],
                                 strategy_name=label)
            delta = vr.sharpe - full_sharpe
            fprint(f"    Sharpe={vr.sharpe:.3f} (vs full: {delta:+.3f}), "
                   f"Sortino={vr.sortino:.3f}, WR={vr.win_rate*100:.1f}%, "
                   f"PF={vr.profit_factor:.2f}, {len(trades)} trades, "
                   f"${CAP:.0f}->${final_eq:.0f}")
            results[label] = {
                "universe": universe,
                "sharpe": round(vr.sharpe, 4),
                "sharpe_delta": round(delta, 4),
                "sortino": round(vr.sortino, 4),
                "win_rate": round(vr.win_rate, 4),
                "profit_factor": round(vr.profit_factor, 4),
                "n_trades": len(trades),
                "final_equity": round(final_eq, 2),
            }
        else:
            results[label] = {"universe": universe, "error": "insufficient_trades"}

    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 4: TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════

def temporal_analysis(trades, rankings):
    """Analyze sector performance across VIX regimes and time periods."""
    fprint(f"\n{'=' * 80}")
    fprint("ANALYSIS 4: TEMPORAL ANALYSIS")
    fprint(f"{'=' * 80}")

    results = {}

    # 4a: VIX regime breakdown per sector
    fprint(f"\n  4a. SECTOR PnL BY VIX REGIME")
    vix_bins = [(0, 15, "Low VIX (<15)"), (15, 20, "Mid VIX (15-20)"),
                (20, 25, "High VIX (20-25)"), (30, 100, "Crisis VIX (>30)")]

    regime_sector_pnl = {}
    for vix_lo, vix_hi, label in vix_bins:
        regime_trades = [t for t in trades if vix_lo <= t["vix"] < vix_hi]
        if not regime_trades:
            continue
        sector_pnl = defaultdict(list)
        for t in regime_trades:
            sector_pnl[t["ticker"]].append(t["pnl"])

        regime_sector_pnl[label] = {}
        fprint(f"\n    {label} ({len(regime_trades)} trades):")
        for tk in SECTORS:
            pnls = sector_pnl.get(tk, [])
            if pnls:
                avg = np.mean(pnls)
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                fprint(f"      {tk}: {len(pnls):>4} trades, WR={wr:>5.1f}%, avg=${avg:>6.1f}")
                regime_sector_pnl[label][tk] = {
                    "n_trades": len(pnls), "avg_pnl": round(avg, 2),
                    "win_rate": round(wr / 100, 4),
                }
    results["vix_regime_sector_pnl"] = regime_sector_pnl

    # 4b: Early vs late performance (split backtest in half)
    fprint(f"\n  4b. EARLY vs LATE PERIOD COMPARISON")
    dates = sorted(set(t["entry_date"] for t in trades))
    mid_idx = len(dates) // 2
    mid_date = dates[mid_idx]

    early_trades = [t for t in trades if t["entry_date"] <= mid_date]
    late_trades = [t for t in trades if t["entry_date"] > mid_date]

    period_results = {}
    for period_label, period_trades in [("Early", early_trades), ("Late", late_trades)]:
        fprint(f"\n    {period_label} period ({len(period_trades)} trades, "
               f"{period_trades[0]['entry_date'] if period_trades else '?'} to "
               f"{period_trades[-1]['entry_date'] if period_trades else '?'}):")
        sector_pnl = defaultdict(list)
        for t in period_trades:
            sector_pnl[t["ticker"]].append(t["pnl"])

        period_data = {}
        for tk in SECTORS:
            pnls = sector_pnl.get(tk, [])
            if pnls:
                avg = np.mean(pnls)
                total = sum(pnls)
                fprint(f"      {tk}: {len(pnls):>4} trades, total=${total:>7.0f}, avg=${avg:>6.1f}")
                period_data[tk] = {"n_trades": len(pnls), "total_pnl": round(total, 2),
                                   "avg_pnl": round(avg, 2)}
        period_results[period_label] = period_data
    results["early_vs_late"] = period_results

    # Check rank stability
    fprint(f"\n    Rank stability (do the same sectors dominate in both halves?):")
    early_ranked = sorted(period_results.get("Early", {}).items(),
                          key=lambda x: x[1].get("total_pnl", 0), reverse=True)
    late_ranked = sorted(period_results.get("Late", {}).items(),
                         key=lambda x: x[1].get("total_pnl", 0), reverse=True)
    early_top3 = [tk for tk, _ in early_ranked[:3]]
    late_top3 = [tk for tk, _ in late_ranked[:3]]
    overlap = set(early_top3) & set(late_top3)
    fprint(f"      Early top-3: {early_top3}")
    fprint(f"      Late top-3:  {late_top3}")
    fprint(f"      Overlap: {len(overlap)}/3 ({list(overlap) if overlap else 'none'})")
    results["rank_stability"] = {
        "early_top3": early_top3, "late_top3": late_top3,
        "overlap": list(overlap), "overlap_count": len(overlap),
    }

    # 4c: LGBM selection momentum
    fprint(f"\n  4c. LGBM SELECTION MOMENTUM (does LGBM keep picking the same sectors?)")
    date_selections = defaultdict(list)
    for dt in sorted(rankings.keys()):
        data = rankings[dt]
        scores = data["scores"]
        direction = data["direction"]
        if direction == "bull":
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1])
        top_picks = [tk for tk, _ in ranked[:TOP_K]]
        for tk in top_picks:
            date_selections[tk].append(str(dt.date()))

    # Compute streak lengths (consecutive selections)
    fprint(f"\n    Selection frequency and max consecutive streak:")
    streak_data = {}
    for tk in SECTORS:
        sel_dates = date_selections.get(tk, [])
        freq = len(sel_dates)
        # compute max streak from sorted dates
        if len(sel_dates) > 1:
            all_dates_sorted = sorted(rankings.keys())
            selected_set = set(sel_dates)
            streak = 0
            max_streak = 0
            for d in all_dates_sorted:
                if str(d.date()) in selected_set:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
        else:
            max_streak = len(sel_dates)

        fprint(f"      {tk}: selected {freq:>4} times, max streak: {max_streak}")
        streak_data[tk] = {"selection_count": freq, "max_streak": max_streak}
    results["selection_momentum"] = streak_data

    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"SECTOR PnL DECOMPOSITION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"v6 production: K={TOP_K}, rebal={REBAL_DAYS}d, moneyness={MONEYNESS_PCT}% OTM, DTE={DTE}")
    fprint(f"Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Universe: {len(SECTORS)} sectors: {SECTORS}")
    fprint()

    # 1. Data
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)
    feature_cols = V4_FEATURES

    # 2. Build rebalance dates
    rebal_freq = f"{REBAL_DAYS}B"
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(rebal_freq).last().dropna().values
    )
    fprint(f"\nRebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # 3. Full strategy run (v6 production)
    fprint(f"\n{'=' * 80}")
    fprint("FULL STRATEGY RUN (v6 production)")
    fprint(f"{'=' * 80}")

    records = build_feature_records(
        close, high, low, rebal_dates, feature_cols,
        regime_series, dte=DTE, sector_universe=SECTORS,
    )
    rankings = walk_forward_lgbm_rank(records, feature_cols, "full_v6")

    trades, final_eq = simulate_trades(
        rankings, close, high, low, regime_series, atr_dict,
    )

    fprint(f"\n  Full run: {len(trades)} trades, ${CAP:.0f} -> ${final_eq:.0f}")

    if not trades or len(trades) < 10:
        fprint("FATAL: Insufficient trades in full run. Aborting.")
        return

    full_result = validate_trades(trades, initial_capital=CAP, spy_prices=close["SPY"],
                                  strategy_name="full_v6")
    full_result.print_summary()
    full_sharpe = full_result.sharpe
    fprint(f"\n  Full Sharpe: {full_sharpe:.3f}, Sortino: {full_result.sortino:.3f}, "
           f"WR: {full_result.win_rate*100:.1f}%, PF: {full_result.profit_factor:.2f}")

    # 4. Per-sector analysis
    sector_results = per_sector_analysis(trades, rankings)

    # 5. Sector exclusion analysis
    exclusion_results = sector_exclusion_analysis(
        close, high, low, regime_series, atr_dict, feature_cols,
        rebal_dates, full_sharpe,
    )

    # 6. Sector concentration analysis
    concentration_results = sector_concentration_analysis(
        close, high, low, regime_series, atr_dict, feature_cols,
        rebal_dates, sector_results, full_sharpe,
    )

    # 7. Temporal analysis
    temporal_results = temporal_analysis(trades, rankings)

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 80}")
    fprint("FINAL SUMMARY & RECOMMENDATIONS")
    fprint(f"{'=' * 80}")

    # Top contributors
    sorted_sectors = sorted(sector_results.items(),
                            key=lambda x: x[1].get("total_pnl", 0), reverse=True)
    fprint(f"\n  PnL RANKING:")
    total_pnl = sum(r.get("total_pnl", 0) for _, r in sorted_sectors)
    cum = 0
    for i, (tk, r) in enumerate(sorted_sectors):
        if r.get("n_trades", 0) == 0:
            continue
        cum += r.get("total_pnl", 0)
        pct = r.get("total_pnl", 0) / max(abs(total_pnl), 1) * 100
        cum_pct = cum / max(abs(total_pnl), 1) * 100
        fprint(f"    {i+1}. {tk}: ${r.get('total_pnl', 0):>8.0f} ({pct:>5.1f}%) "
               f"[cumulative: {cum_pct:>5.1f}%] "
               f"Sharpe={r.get('sharpe_contribution', 0):.2f}")

    # Concentration verdict
    fprint(f"\n  CONCENTRATION VERDICT:")
    for label, r in concentration_results.items():
        if "error" in r:
            fprint(f"    {label}: ERROR")
        else:
            fprint(f"    {label} ({r.get('universe', '?')}): Sharpe={r['sharpe']:.3f} "
                   f"(delta={r['sharpe_delta']:+.3f})")

    # Drags
    drags = [(tk, r) for tk, r in exclusion_results.items()
             if r.get("sharpe_delta", 0) > 0.05]
    if drags:
        fprint(f"\n  SECTORS DRAGGING PERFORMANCE:")
        for tk, r in drags:
            fprint(f"    {tk}: removing it improves Sharpe by {r['sharpe_delta']:+.3f}")
    else:
        fprint(f"\n  No single sector is significantly dragging performance.")

    # Key contributors
    keys = [(tk, r) for tk, r in exclusion_results.items()
            if r.get("sharpe_delta", 0) < -0.05]
    if keys:
        fprint(f"\n  KEY CONTRIBUTING SECTORS (removing them hurts):")
        for tk, r in keys:
            fprint(f"    {tk}: removing it drops Sharpe by {r['sharpe_delta']:+.3f}")

    # Rank stability
    rs = temporal_results.get("rank_stability", {})
    overlap_n = rs.get("overlap_count", 0)
    fprint(f"\n  RANK STABILITY: {overlap_n}/3 top sectors overlap between early and late periods")
    if overlap_n >= 2:
        fprint(f"    => Stable sector leadership. Concentration may be viable.")
    else:
        fprint(f"    => Sector leadership rotates. Diversification is important.")

    # ── Save all results ──
    all_output = {
        "run_params": {
            "top_k": TOP_K, "rebal_days": REBAL_DAYS, "dte": DTE,
            "moneyness_pct": MONEYNESS_PCT, "capital": CAP,
            "sectors": SECTORS, "n_sectors": len(SECTORS),
        },
        "full_strategy": {
            "sharpe": round(full_sharpe, 4),
            "sortino": round(full_result.sortino, 4),
            "win_rate": round(full_result.win_rate, 4),
            "profit_factor": round(full_result.profit_factor, 4),
            "n_trades": len(trades),
            "final_equity": round(final_eq, 2),
        },
        "per_sector": sector_results,
        "exclusion": exclusion_results,
        "concentration": concentration_results,
        "temporal": temporal_results,
        "trades": trades,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # ── MLflow ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"sector_decomp_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    "top_k": TOP_K, "rebal_days": REBAL_DAYS, "dte": DTE,
                    "moneyness_pct": MONEYNESS_PCT, "capital": CAP,
                    "n_sectors": len(SECTORS), "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                })
                mlflow.log_metric("full_sharpe", full_sharpe)
                mlflow.log_metric("full_sortino", full_result.sortino)
                mlflow.log_metric("full_win_rate", full_result.win_rate)
                mlflow.log_metric("full_profit_factor", full_result.profit_factor)
                mlflow.log_metric("full_n_trades", len(trades))
                mlflow.log_metric("full_final_equity", final_eq)

                for tk, r in sector_results.items():
                    if r.get("n_trades", 0) > 0:
                        mlflow.log_metric(f"sector_{tk}_pnl", r.get("total_pnl", 0))
                        mlflow.log_metric(f"sector_{tk}_sharpe", r.get("sharpe_contribution", 0))
                        mlflow.log_metric(f"sector_{tk}_wr", r.get("win_rate", 0))
                        mlflow.log_metric(f"sector_{tk}_trades", r.get("n_trades", 0))

                for tk, r in exclusion_results.items():
                    if "sharpe_delta" in r:
                        mlflow.log_metric(f"excl_{tk}_sharpe_delta", r["sharpe_delta"])

                for label, r in concentration_results.items():
                    if "sharpe" in r:
                        mlflow.log_metric(f"conc_{label}_sharpe", r["sharpe"])

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
