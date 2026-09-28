#!/usr/bin/env python3
"""
Sector Pair Trades v1 — Long Best + Short Worst for Market Neutrality
=====================================================================

KEY QUESTION: Our long-only sector strategy fails the regime balance gate
(works better in bull markets). Does adding a short leg fix this?

Concept: Go long top-ranked sector via bull call spread AND short
bottom-ranked sector via bear put spread simultaneously. The pair
should be approximately market-neutral (long+short cancel beta).

6 Variants:
  1. Baseline (long-only): Production v4 — top 3 bull call, regime>0.4, DTE=21
  2. Pair: Top 1 long + Bottom 1 short (single pair)
  3. Pair: Top 2 long + Bottom 2 short (two pairs)
  4. Pair: Top 3 long + Bottom 3 short (three pairs)
  5. Long-only no regime filter: Top 3 bull call, all rebalances
  6. Pair no regime filter: Top 3 long + bottom 3 short, all rebalances

Honest pricing:
  - 15% entry haircut, no exit haircut (hold to expiry)
  - $2.60 commission/spread, $645 start, $200 max/trade
  - DTE=21, biweekly rebalance (10 trading days)
  - Walk-forward LGBM: 240d train (~12 biweekly periods), 10d step
  - BS pricing with IV = 1.2 * HV (from estimate_iv)
  - yfinance data 2009-2026

Analysis:
  - Sharpe, Sortino, CAGR, MaxDD, WR, PF for each variant
  - Regime analysis: Does pair trade have better regime balance?
  - Beta neutrality: correlation of pair returns with SPY
  - Long vs short leg PnL breakdown
  - 5-gate adversarial validation
  - MLflow logging

Concern: Bear put spreads have net negative PnL at expiry in honest
pricing (finding #69). The short leg may just lose money. But in a PAIR
context, the short leg's job is to hedge, not profit independently.
"""

import json
import sys
import time
import warnings
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
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
RESULTS_DIR = BASE / "research" / "findings"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "sector_pair_trades_v1_results.json"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
MAX_POS = 200.0

# Walk-forward LGBM
WF_TRAIN_PERIODS = 12  # ~240 days of biweekly data
WF_REBAL_FREQ = "2W-FRI"

# 21 production v4 features
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]
CROSS_ASSET_FEATURES = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]
ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES  # 21 total

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "sector_pair_trades_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers via yfinance."""
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

    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features."""
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


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Compute the 3 validated cross-asset features."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
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

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            f["sector_relative_vol_21d"] = float(sec_ret.iloc[-21:].std() / (spy_ret.iloc[-21:].std() + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

    # 3. Cross-sector dispersion
    sector_cols = [c for c in SECTORS if c in close_df.columns]
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
# ATR
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for all sectors."""
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
                atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols,
                          use_regime_filter=True):
    """Build feature + target records for LGBM walk-forward ranking.

    When use_regime_filter=True, only include dates with VIX>20.
    When False, include all dates.
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features, "
           f"regime_filter={use_regime_filter}")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # VIX regime filter
        if use_regime_filter and vix is not None:
            cv = float(vix.iloc[idx]) if idx < len(vix) else 20.0
            if cv < 20.0:
                continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features
            cross_asset = {}
            for col in feature_cols:
                if col in CROSS_ASSET_FEATURES:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            # Forward return target (DTE days forward)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols, variant_name):
    """Walk-forward LGBM: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
        return {}, None

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    all_importances = np.zeros(len(feature_cols))
    n_models = 0

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
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

            all_importances += m.feature_importances_
            n_models += 1
        except Exception:
            continue

    if n_models > 0:
        all_importances /= n_models
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": all_importances,
        }).sort_values("importance", ascending=False)
    else:
        imp_df = None

    fprint(f"    {variant_name}: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_variant(name, rankings, close, high, low, atr_dict,
                     top_k=3, bottom_k=0, use_regime_filter=True):
    """
    Simulate a variant with long (bull call) and/or short (bear put) legs.

    top_k: how many top-ranked sectors to go long (bull call spreads)
    bottom_k: how many bottom-ranked sectors to go short (bear put spreads)
    use_regime_filter: if True, skip rebalances where VIX < 20
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    long_pnl_total = 0.0
    short_pnl_total = 0.0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0

        # Regime filter
        if use_regime_filter and cv < 20.0:
            continue

        scores = rankings[dt]
        if not scores or len(scores) < max(top_k, bottom_k, 1):
            continue

        # Rank sectors
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        long_picks = [t for t, _ in ranked[:top_k]] if top_k > 0 else []
        short_picks = [t for t, _ in ranked[-bottom_k:]] if bottom_k > 0 else []

        # Avoid overlap (a sector can't be both long and short)
        short_picks = [t for t in short_picks if t not in long_picks]

        # Position sizing
        total_legs = len(long_picks) + len(short_picks)
        if total_legs == 0:
            continue
        max_per_trade = min(MAX_POS, equity / max(total_legs, 3))
        if max_per_trade < 30:
            continue

        # ── LONG LEG: bull call spreads on top-ranked ──
        for tk in long_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue

            S = float(close[tk].iloc[idx])
            di = idx
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue

            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            K1 = round(S, 2)  # ATM
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
            if total_cost <= 0 or total_cost > max_per_trade or total_cost > equity * 0.40:
                continue

            # Hold to expiry — intrinsic value only
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

            equity += pnl
            long_pnl_total += pnl

            # Regime classification using SPY during hold
            spy_entry = float(spy.iloc[di])
            spy_exit = float(spy.iloc[ei]) if ei < len(spy) else spy_entry
            regime = "bull" if spy_exit >= spy_entry else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(close.index[di].date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": regime,
                "leg": "long",
                "vix": round(cv, 1),
                "win": pnl > 0,
            })

        # ── SHORT LEG: bear put spreads on bottom-ranked ──
        for tk in short_picks:
            if tk not in close.columns or tk not in atr_dict:
                continue

            S = float(close[tk].iloc[idx])
            di = idx
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue

            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            # Bear put spread: buy ATM put (K2), sell OTM put (K1)
            K2 = round(S, 2)  # ATM (higher strike — long put)
            K1 = round(S * (1 - SPREAD_PCT / 100), 2)  # OTM (lower strike — short put)
            if K2 <= K1:
                continue

            try:
                entry_cost_ps, max_profit_ps = price_bear_put_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
            if total_cost <= 0 or total_cost > max_per_trade or total_cost > equity * 0.40:
                continue

            # Hold to expiry — intrinsic value only
            Se = float(close[tk].iloc[ei])
            # Bear put spread payoff: max(K2-Se,0) - max(K1-Se,0)
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
            pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

            equity += pnl
            short_pnl_total += pnl

            spy_entry = float(spy.iloc[di])
            spy_exit = float(spy.iloc[ei]) if ei < len(spy) else spy_entry
            regime = "bull" if spy_exit >= spy_entry else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(close.index[di].date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": regime,
                "leg": "short",
                "vix": round(cv, 1),
                "win": pnl > 0,
            })

    return trades, equity, long_pnl_total, short_pnl_total


# ══════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def compute_spy_correlation(trades, close):
    """Compute correlation of trade returns with SPY returns."""
    if not trades or len(trades) < 10:
        return 0.0

    spy = close["SPY"]
    trade_rets = []
    spy_rets = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])
        if entry in spy.index and exit_dt in spy.index:
            sr = float(spy.loc[exit_dt] / spy.loc[entry] - 1)
            trade_rets.append(t["pnl"])
            spy_rets.append(sr)

    if len(trade_rets) < 10:
        return 0.0

    corr = np.corrcoef(trade_rets, spy_rets)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def regime_breakdown(trades):
    """Compute regime-stratified metrics."""
    if not trades:
        return {}

    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    result = {}
    for label, subset in [("bull", bull_trades), ("bear", bear_trades)]:
        if not subset:
            result[label] = {"n": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0}
            continue
        n = len(subset)
        wins = sum(1 for t in subset if t["win"])
        total_pnl = sum(t["pnl"] for t in subset)
        result[label] = {
            "n": n,
            "wr": round(wins / n * 100, 1),
            "avg_pnl": round(total_pnl / n, 2),
            "total_pnl": round(total_pnl, 2),
        }

    # Regime balance metric
    bull_wr = result["bull"]["wr"]
    bear_wr = result["bear"]["wr"]
    if max(bull_wr, bear_wr) > 0:
        result["regime_imbalance"] = round(abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1), 3)
    else:
        result["regime_imbalance"] = 1.0

    return result


def leg_breakdown(trades):
    """Break down PnL by long vs short leg."""
    long_trades = [t for t in trades if t.get("leg") == "long"]
    short_trades = [t for t in trades if t.get("leg") == "short"]

    result = {}
    for label, subset in [("long", long_trades), ("short", short_trades)]:
        if not subset:
            result[label] = {"n": 0, "wr": 0, "total_pnl": 0, "avg_pnl": 0}
            continue
        n = len(subset)
        wins = sum(1 for t in subset if t["win"])
        total_pnl = sum(t["pnl"] for t in subset)
        result[label] = {
            "n": n,
            "wr": round(wins / n * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / n, 2),
        }
    return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"SECTOR PAIR TRADES v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only | No exit haircut")
    fprint(f"Walk-forward: {WF_TRAIN_PERIODS} train periods, biweekly rebalance")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. ATR
    atr_dict = compute_atr_series(high, low, close)

    # 3. Rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 4. Build LGBM rankings — TWO sets: regime-filtered and unfiltered
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS")
    fprint("=" * 80)

    fprint("\n--- Regime-filtered rankings (VIX>20 only) ---")
    records_regime = build_feature_records(
        close, high, low, rebal_dates, ALL_FEATURES, use_regime_filter=True
    )
    rankings_regime, imp_regime = walk_forward_lgbm_rank(
        records_regime, ALL_FEATURES, "regime_filtered"
    )

    fprint("\n--- Unfiltered rankings (all dates) ---")
    records_all = build_feature_records(
        close, high, low, rebal_dates, ALL_FEATURES, use_regime_filter=False
    )
    rankings_all, imp_all = walk_forward_lgbm_rank(
        records_all, ALL_FEATURES, "unfiltered"
    )

    # 5. Define 6 variants
    variants = [
        {
            "name": "V1_baseline_long_only",
            "desc": "Baseline: top 3 long, regime>0.4 (VIX>20), DTE=21",
            "rankings": rankings_regime,
            "top_k": 3, "bottom_k": 0,
            "use_regime_filter": True,
        },
        {
            "name": "V2_pair_1x1",
            "desc": "Pair: top 1 long + bottom 1 short, regime>0.4",
            "rankings": rankings_regime,
            "top_k": 1, "bottom_k": 1,
            "use_regime_filter": True,
        },
        {
            "name": "V3_pair_2x2",
            "desc": "Pair: top 2 long + bottom 2 short, regime>0.4",
            "rankings": rankings_regime,
            "top_k": 2, "bottom_k": 2,
            "use_regime_filter": True,
        },
        {
            "name": "V4_pair_3x3",
            "desc": "Pair: top 3 long + bottom 3 short, regime>0.4",
            "rankings": rankings_regime,
            "top_k": 3, "bottom_k": 3,
            "use_regime_filter": True,
        },
        {
            "name": "V5_long_no_regime",
            "desc": "Long-only top 3, NO regime filter (all dates)",
            "rankings": rankings_all,
            "top_k": 3, "bottom_k": 0,
            "use_regime_filter": False,
        },
        {
            "name": "V6_pair_no_regime",
            "desc": "Pair 3x3, NO regime filter (all dates)",
            "rankings": rankings_all,
            "top_k": 3, "bottom_k": 3,
            "use_regime_filter": False,
        },
    ]

    # 6. Simulate all variants
    fprint("\n" + "=" * 80)
    fprint("SIMULATING ALL VARIANTS")
    fprint("=" * 80)

    all_results = {}

    for v in variants:
        vname = v["name"]
        fprint(f"\n--- {vname}: {v['desc']} ---")

        if not v["rankings"]:
            fprint(f"  No rankings available, skipping")
            continue

        trades, final_eq, long_pnl, short_pnl = simulate_variant(
            vname, v["rankings"], close, high, low, atr_dict,
            top_k=v["top_k"], bottom_k=v["bottom_k"],
            use_regime_filter=v["use_regime_filter"],
        )

        fprint(f"  Trades: {len(trades)}, Final equity: ${final_eq:,.0f}")
        fprint(f"  Long leg PnL: ${long_pnl:,.0f}, Short leg PnL: ${short_pnl:,.0f}")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": v["desc"],
                "n_trades": len(trades) if trades else 0,
                "final_equity": round(final_eq, 2),
                "error": "insufficient_trades",
            }
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # SPY correlation (beta neutrality)
        spy_corr = compute_spy_correlation(trades, close)
        fprint(f"  SPY correlation: {spy_corr:.3f}")

        # Regime breakdown
        regime = regime_breakdown(trades)
        fprint(f"  Regime: Bull WR={regime.get('bull',{}).get('wr',0):.1f}% ({regime.get('bull',{}).get('n',0)} trades), "
               f"Bear WR={regime.get('bear',{}).get('wr',0):.1f}% ({regime.get('bear',{}).get('n',0)} trades)")
        fprint(f"  Regime imbalance: {regime.get('regime_imbalance', 1.0):.3f} (<0.50 = PASS)")

        # Leg breakdown (for pair variants)
        legs = leg_breakdown(trades)
        if legs.get("short", {}).get("n", 0) > 0:
            fprint(f"  Long leg: {legs['long']['n']} trades, WR={legs['long']['wr']:.1f}%, "
                   f"PnL=${legs['long']['total_pnl']:,.0f} (avg ${legs['long']['avg_pnl']:.1f})")
            fprint(f"  Short leg: {legs['short']['n']} trades, WR={legs['short']['wr']:.1f}%, "
                   f"PnL=${legs['short']['total_pnl']:,.0f} (avg ${legs['short']['avg_pnl']:.1f})")

        all_results[vname] = {
            "description": v["desc"],
            **result.to_dict(),
            "spy_correlation": round(spy_corr, 3),
            "regime_breakdown": regime,
            "leg_breakdown": legs,
            "long_total_pnl": round(long_pnl, 2),
            "short_total_pnl": round(short_pnl, 2),
        }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 100)
    fprint("SUMMARY COMPARISON TABLE")
    fprint("=" * 100)
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'SPYcorr':>8} {'RegImb':>7}")
    fprint("-" * 100)

    for vname in [v["name"] for v in variants]:
        r = all_results.get(vname)
        if not r or r.get("error"):
            fprint(f"  {vname:<25} — INSUFFICIENT DATA —")
            continue
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['spy_correlation']:>8.3f} "
               f"{r['regime_breakdown'].get('regime_imbalance', 1.0):>6.3f}")

    # ══════════════════════════════════════════════════════════════
    # KEY ANALYSIS
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "=" * 80)
    fprint("KEY ANALYSIS")
    fprint("=" * 80)

    # Q1: Does the pair trade improve regime balance?
    fprint("\n1. REGIME BALANCE:")
    for vname in [v["name"] for v in variants]:
        r = all_results.get(vname, {})
        if r.get("error"):
            continue
        rb = r.get("regime_breakdown", {})
        imb = rb.get("regime_imbalance", 1.0)
        passed = "PASS" if imb < 0.50 else "FAIL"
        fprint(f"   {vname:<25} imbalance={imb:.3f} [{passed}] "
               f"(bull WR={rb.get('bull',{}).get('wr',0):.1f}%, "
               f"bear WR={rb.get('bear',{}).get('wr',0):.1f}%)")

    # Q2: Beta neutrality
    fprint("\n2. BETA NEUTRALITY (SPY correlation):")
    for vname in [v["name"] for v in variants]:
        r = all_results.get(vname, {})
        if r.get("error"):
            continue
        sc = r.get("spy_correlation", 1.0)
        neutral = "NEUTRAL" if abs(sc) < 0.20 else ("LOW BETA" if abs(sc) < 0.40 else "HIGH BETA")
        fprint(f"   {vname:<25} SPY corr={sc:+.3f} [{neutral}]")

    # Q3: Does short leg add value?
    fprint("\n3. SHORT LEG VALUE:")
    for vname in [v["name"] for v in variants]:
        r = all_results.get(vname, {})
        if r.get("error"):
            continue
        legs = r.get("leg_breakdown", {})
        sl = legs.get("short", {})
        if sl.get("n", 0) > 0:
            verdict = "ADDS VALUE" if sl["total_pnl"] > 0 else "COSTS MONEY (hedge cost)"
            fprint(f"   {vname:<25} Short PnL=${sl['total_pnl']:>8,.0f} "
                   f"WR={sl['wr']:.1f}% [{verdict}]")

    # Q4: Does pair make regime filter unnecessary?
    fprint("\n4. IS REGIME FILTER NECESSARY?")
    v5 = all_results.get("V5_long_no_regime", {})
    v6 = all_results.get("V6_pair_no_regime", {})
    v1 = all_results.get("V1_baseline_long_only", {})
    v4 = all_results.get("V4_pair_3x3", {})

    if not v5.get("error") and not v1.get("error"):
        fprint(f"   Long-only with regime:    Sharpe={v1.get('sharpe',0):.2f}")
        fprint(f"   Long-only without regime: Sharpe={v5.get('sharpe',0):.2f}")
        if v5.get("sharpe", 0) >= v1.get("sharpe", 0) * 0.9:
            fprint(f"   --> Without regime filter is competitive (within 10%)")
        else:
            fprint(f"   --> Regime filter still needed for long-only")

    if not v6.get("error") and not v4.get("error"):
        fprint(f"   Pair with regime:    Sharpe={v4.get('sharpe',0):.2f}")
        fprint(f"   Pair without regime: Sharpe={v6.get('sharpe',0):.2f}")
        if v6.get("sharpe", 0) >= v4.get("sharpe", 0) * 0.9:
            fprint(f"   --> Pair neutrality makes regime filter less important!")
        else:
            fprint(f"   --> Regime filter still adds value even with pair")

    # Key question summary
    fprint("\n" + "=" * 80)
    fprint("KEY QUESTION: Does adding a short leg fix regime dependence?")
    fprint("=" * 80)

    baseline_imb = all_results.get("V1_baseline_long_only", {}).get(
        "regime_breakdown", {}).get("regime_imbalance", 1.0)
    pair_imb = all_results.get("V4_pair_3x3", {}).get(
        "regime_breakdown", {}).get("regime_imbalance", 1.0)
    pair_sharpe = all_results.get("V4_pair_3x3", {}).get("sharpe", 0)
    baseline_sharpe = all_results.get("V1_baseline_long_only", {}).get("sharpe", 0)

    fprint(f"  Baseline regime imbalance: {baseline_imb:.3f} (<0.50=PASS)")
    fprint(f"  Pair 3x3 regime imbalance: {pair_imb:.3f} (<0.50=PASS)")
    fprint(f"  Baseline Sharpe: {baseline_sharpe:.2f}")
    fprint(f"  Pair 3x3 Sharpe: {pair_sharpe:.2f}")

    if pair_imb < baseline_imb and pair_imb < 0.50:
        fprint(f"\n  YES — Pair trading improves regime balance!")
        if pair_sharpe >= baseline_sharpe * 0.8:
            fprint(f"  AND maintains acceptable Sharpe (within 20% of baseline)")
        else:
            fprint(f"  BUT Sharpe drops significantly ({baseline_sharpe:.2f} -> {pair_sharpe:.2f})")
    elif pair_imb < baseline_imb:
        fprint(f"\n  PARTIAL — Pair reduces imbalance but still fails gate")
    else:
        fprint(f"\n  NO — Pair trading does not improve regime balance")

    short_total = all_results.get("V4_pair_3x3", {}).get("short_total_pnl", 0)
    if short_total < 0:
        fprint(f"\n  CONCERN: Short leg loses ${abs(short_total):,.0f} total — confirms finding #69")
        fprint(f"  Bear put spreads at honest pricing are expensive hedges")
    elif short_total > 0:
        fprint(f"\n  POSITIVE: Short leg profits ${short_total:,.0f} — short momentum works!")

    # Feature importance
    fprint("\n" + "=" * 80)
    fprint("FEATURE IMPORTANCE (Top 10)")
    fprint("=" * 80)
    if imp_all is not None:
        for _, row in imp_all.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_all["importance"].max() * 30)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # ══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════
    output = {
        "timestamp": t0.isoformat(),
        "config": {
            "capital": CAP, "dte": DTE, "spread_pct": SPREAD_PCT,
            "haircut": DEFAULT_HAIRCUT, "commission": COMMISSION_RT_SPREAD,
            "wf_train_periods": WF_TRAIN_PERIODS, "rebal_freq": WF_REBAL_FREQ,
            "features": ALL_FEATURES,
        },
        "variants": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="sector_pair_trades_v1"):
                mlflow.log_params({
                    "capital": CAP, "dte": DTE, "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT, "commission": COMMISSION_RT_SPREAD,
                    "n_features": len(ALL_FEATURES),
                })
                for vname, r in all_results.items():
                    if r.get("error"):
                        continue
                    mlflow.log_metrics({
                        f"{vname}_sharpe": r.get("sharpe", 0),
                        f"{vname}_sortino": r.get("sortino", 0),
                        f"{vname}_wr": r.get("win_rate", 0),
                        f"{vname}_pf": r.get("profit_factor", 0),
                        f"{vname}_maxdd": r.get("max_dd", 0),
                        f"{vname}_gates": r.get("gates_passed", 0),
                        f"{vname}_spy_corr": r.get("spy_correlation", 0),
                    })
                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    fprint("DONE")


if __name__ == "__main__":
    main()
