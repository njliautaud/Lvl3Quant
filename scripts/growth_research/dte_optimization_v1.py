#!/usr/bin/env python3
"""
DTE Optimization v1 — Optimize Days-To-Expiry for Production v4 Sector Bull Call Spreads
==========================================================================================

Tests 8 DTE variants (7, 14, 21, 30, 45, 60, 90, adaptive) while holding all other
variables constant:
  - Same LGBM walk-forward model with 21 features (18 legacy + 3 cross-asset)
  - Same VIX>20 regime filter
  - Same 3% spread width, ATM entry
  - Same 15% entry haircut, hold-to-expiry (intrinsic value, no exit haircut)
  - Same biweekly rebalance (10 trading days)
  - Same position sizing ($200 max, $645 start)

Key hypothesis: DTE=14 may beat DTE=21 because biweekly rebalance aligns with
14-day expiry — the option expires exactly at next rebalance.

Analysis per variant:
  1. Sharpe, Sortino, CAGR, MaxDD, WR, PF, total trades, final equity
  2. Average spread cost as % of max payoff
  3. Theta drag — % of trades expiring worthless vs ITM
  4. Time-in-market efficiency
  5. Best DTE by VIX regime (20-25, 25-30, 30+)

5-gate adversarial validation on top 3 variants.
Logs to MLflow at http://localhost:5000.
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
    estimate_iv,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "dte_optimization_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = BASE / "research" / "findings" / "dte_optimization_v1_results.json"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "TLT"]
CAP = 645.0
SPREAD_PCT = 3.0
TOP_K = 3
MAX_POS = 200.0
VIX_ENTRY_MIN = 20.0

# DTE variants to test
DTE_VARIANTS = {
    "DTE_07": {"dte": 7, "label": "Weekly (7d)", "adaptive": False},
    "DTE_14": {"dte": 14, "label": "Biweekly (14d)", "adaptive": False},
    "DTE_21": {"dte": 21, "label": "Monthly (21d) — BASELINE", "adaptive": False},
    "DTE_30": {"dte": 30, "label": "Monthly+ (30d)", "adaptive": False},
    "DTE_45": {"dte": 45, "label": "6-week (45d)", "adaptive": False},
    "DTE_60": {"dte": 60, "label": "2-month (60d)", "adaptive": False},
    "DTE_90": {"dte": 90, "label": "Quarterly (90d)", "adaptive": False},
    "DTE_Adaptive": {"dte": None, "label": "Adaptive: 14+7*(VIX/20)", "adaptive": True},
}

# LGBM walk-forward
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
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "2W-FRI"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "dte_optimization_v1"

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

    for t in ["SPY", "VIX"]:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
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
        return {k: 0.0 for k in VALIDATED_CROSS_ASSET}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            sec_vol = sec_ret.iloc[-21:].std()
            spy_vol = spy_ret.iloc[-21:].std()
            f["sector_relative_vol_21d"] = float(sec_vol / (spy_vol + 1e-10))
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
# ATR COMPUTATION
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
# WALK-FORWARD LGBM RANKING (DTE-aware forward returns)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, dte_days):
    """
    Build feature + target records. Forward return target uses the specified DTE.
    VIX>20 regime filter applied.
    """
    import lightgbm as lgb

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # VIX > 20 regime filter
        if vix is not None and dt in vix.index:
            cv = float(vix.loc[dt])
        else:
            cv = 15.0  # default low — skip
        if cv < VIX_ENTRY_MIN:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features
            cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target using the DTE
            fi = min(idx + dte_days, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "vix": cv}
            records.append(rec)

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} rebal dates (VIX>20 filtered)")
    return df


def walk_forward_lgbm_rank(df, feature_cols, variant_name):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
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
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "vix": float(test_df["vix"].iloc[0]),
            }
        except Exception:
            continue

    fprint(f"    {variant_name}: {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# ADAPTIVE DTE FUNCTION
# ══════════════════════════════════════════════════════════════

def compute_adaptive_dte(vix_level):
    """Adaptive DTE: 14 + 7 * (VIX/20). Higher VIX = longer DTE."""
    raw = 14 + 7 * (vix_level / 20.0)
    return int(round(raw))


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_trades(name, rankings, close, atr_dict, dte_config):
    """
    Simulate bull call spreads from rankings for a specific DTE config.

    HONEST RULES:
      - Hold to expiry
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    fixed_dte = dte_config["dte"]
    is_adaptive = dte_config["adaptive"]

    equity = CAP
    trades = []

    # DTE-specific tracking
    total_spread_cost_pct = []  # entry cost as % of max payoff
    itm_count = 0
    otm_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]
        cv = ranking_data.get("vix", 20.0)

        if not scores:
            continue

        # Determine DTE for this trade
        if is_adaptive:
            trade_dte = compute_adaptive_dte(cv)
        else:
            trade_dte = fixed_dte

        # Pick top K sectors
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        # Position sizing
        max_pos = min(MAX_POS, equity / 3)
        if max_pos < 30:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
                continue

            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + trade_dte, len(close) - 1)
            if ei <= di:
                continue

            # ATR for pricing
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            # Price the bull call spread
            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=trade_dte, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            # Track spread cost as % of max payoff
            spread_width = K2 - K1
            cost_pct = entry_cost_ps / spread_width if spread_width > 0 else 1.0
            total_spread_cost_pct.append(cost_pct)

            # HOLD TO EXPIRY: compute intrinsic value at expiry
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)

            # Track ITM vs OTM
            if intrinsic > 0:
                itm_count += 1
            else:
                otm_count += 1

            exit_value_ps = intrinsic

            # PnL: exit value - entry cost - commission (no exit haircut at expiry)
            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # Regime classification for trade record
            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            spy_regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "vix": round(cv, 1),
                "dte_used": trade_dte,
                "entry_cost_ps": round(entry_cost_ps, 4),
                "spread_width": round(spread_width, 2),
                "intrinsic_at_expiry": round(intrinsic, 4),
                "win": pnl > 0,
            })

    # Compute DTE-specific analytics
    analytics = {
        "avg_spread_cost_pct": float(np.mean(total_spread_cost_pct)) if total_spread_cost_pct else 0.0,
        "median_spread_cost_pct": float(np.median(total_spread_cost_pct)) if total_spread_cost_pct else 0.0,
        "itm_at_expiry": itm_count,
        "otm_at_expiry": otm_count,
        "itm_rate": itm_count / (itm_count + otm_count) if (itm_count + otm_count) > 0 else 0.0,
        "theta_drag_rate": otm_count / (itm_count + otm_count) if (itm_count + otm_count) > 0 else 1.0,
    }

    return trades, equity, analytics


# ══════════════════════════════════════════════════════════════
# VIX REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze_by_vix_regime(trades):
    """Break down performance by VIX regime bands."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    regimes = {
        "VIX_20_25": df[(df["vix"] >= 20) & (df["vix"] < 25)],
        "VIX_25_30": df[(df["vix"] >= 25) & (df["vix"] < 30)],
        "VIX_30_plus": df[df["vix"] >= 30],
    }

    results = {}
    for regime_name, rdf in regimes.items():
        if len(rdf) < 3:
            results[regime_name] = {"n_trades": len(rdf), "insufficient": True}
            continue

        wins = rdf[rdf["pnl"] > 0]
        losses = rdf[rdf["pnl"] <= 0]
        total_pnl = rdf["pnl"].sum()
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-9

        results[regime_name] = {
            "n_trades": len(rdf),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(rdf["pnl"].mean(), 2),
            "win_rate": round(len(wins) / len(rdf), 3),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 1e-9 else 999.0,
            "avg_dte": round(rdf["dte_used"].mean(), 1) if "dte_used" in rdf.columns else 0,
        }

    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 90)
    fprint(f"DTE OPTIMIZATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 90)
    fprint(f"Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% ATM | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | VIX>20 regime filter | Biweekly rebalance | Top {TOP_K} sectors")
    fprint(f"Testing {len(DTE_VARIANTS)} DTE variants: {list(DTE_VARIANTS.keys())}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 3. Build rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # ── For each DTE variant, build LGBM rankings with DTE-matched forward returns ──
    all_results = {}
    all_trades = {}
    all_analytics = {}

    for var_name, var_config in DTE_VARIANTS.items():
        fprint("\n" + "=" * 90)
        fprint(f"VARIANT: {var_name} — {var_config['label']}")
        fprint("=" * 90)

        # For adaptive DTE, use DTE=21 for LGBM training (median expected DTE)
        # The actual DTE used in trading will vary per trade
        if var_config["adaptive"]:
            lgbm_dte = 21  # median of adaptive range
            fprint(f"  Adaptive DTE: using DTE=21 for LGBM forward returns (median of adaptive range)")
        else:
            lgbm_dte = var_config["dte"]
            fprint(f"  Fixed DTE: {lgbm_dte} days")

        # Build feature records with DTE-matched forward return
        fprint(f"  Building LGBM features with {lgbm_dte}d forward returns...")
        records = build_feature_records(
            close, high, low, rebal_dates, V4_FEATURES, lgbm_dte
        )

        if len(records) < 100:
            fprint(f"  SKIP: insufficient records ({len(records)})")
            continue

        # Walk-forward LGBM ranking
        rankings = walk_forward_lgbm_rank(records, V4_FEATURES, var_name)
        if not rankings:
            fprint(f"  SKIP: no rankings generated")
            continue

        # Simulate trades
        trades, final_eq, analytics = simulate_trades(
            var_name, rankings, close, atr_dict, var_config
        )

        if not trades or len(trades) < 10:
            fprint(f"  SKIP: insufficient trades ({len(trades) if trades else 0})")
            continue

        all_trades[var_name] = trades
        all_analytics[var_name] = analytics

        # Compute metrics via adversarial validator
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=var_name,
            n_perms=500,
        )

        # VIX regime analysis
        regime_analysis = analyze_by_vix_regime(trades)

        # Store results
        all_results[var_name] = {
            "label": var_config["label"],
            "dte": var_config["dte"],
            "adaptive": var_config["adaptive"],
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "cagr": result.cagr,
            "max_dd": result.max_dd,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "n_trades": result.n_trades,
            "final_equity": result.final_equity,
            "gates_passed": result.gates_passed,
            "gates_total": result.gates_total,
            "all_passed": result.all_passed,
            "avg_spread_cost_pct": analytics["avg_spread_cost_pct"],
            "itm_rate": analytics["itm_rate"],
            "theta_drag_rate": analytics["theta_drag_rate"],
            "regime_analysis": regime_analysis,
        }

        # Print result
        result.print_summary()
        fprint(f"\n  DTE-specific analytics:")
        fprint(f"    Avg spread cost as % of max payoff: {analytics['avg_spread_cost_pct']:.1%}")
        fprint(f"    Median spread cost %: {analytics['median_spread_cost_pct']:.1%}")
        fprint(f"    ITM at expiry: {analytics['itm_at_expiry']} ({analytics['itm_rate']:.1%})")
        fprint(f"    OTM (worthless) at expiry: {analytics['otm_at_expiry']} ({analytics['theta_drag_rate']:.1%})")
        fprint(f"\n  VIX regime breakdown:")
        for rname, rdata in regime_analysis.items():
            if rdata.get("insufficient"):
                fprint(f"    {rname}: {rdata['n_trades']} trades (insufficient)")
            else:
                fprint(f"    {rname}: {rdata['n_trades']} trades, WR={rdata['win_rate']:.1%}, "
                       f"PF={rdata['profit_factor']:.2f}, avg PnL=${rdata['avg_pnl']:.2f}")

    # ══════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════

    fprint("\n\n" + "=" * 120)
    fprint("SUMMARY: DTE OPTIMIZATION RESULTS")
    fprint("=" * 120)

    if not all_results:
        fprint("NO RESULTS — all variants had insufficient data")
        return

    # Sort by Sharpe
    sorted_variants = sorted(all_results.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    header = f"{'Variant':<16} {'DTE':>4} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} " \
             f"{'WR':>6} {'PF':>6} {'Trades':>7} {'Final$':>9} " \
             f"{'Cost%':>6} {'ITM%':>6} {'Gates':>6}"
    fprint(header)
    fprint("-" * 120)

    for var_name, r in sorted_variants:
        dte_str = "Adpt" if r["adaptive"] else str(r["dte"])
        baseline = " <<< BASELINE" if var_name == "DTE_21" else ""
        best = " *** BEST" if var_name == sorted_variants[0][0] else ""
        fprint(f"{var_name:<16} {dte_str:>4} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['cagr']*100:>6.1f}% {r['max_dd']*100:>6.1f}% "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['n_trades']:>7d} ${r['final_equity']:>8,.0f} "
               f"{r['avg_spread_cost_pct']*100:>5.1f}% {r['itm_rate']*100:>5.1f}% "
               f"{r['gates_passed']}/{r['gates_total']}{baseline}{best}")

    # ── Key insights ──
    fprint("\n" + "=" * 90)
    fprint("KEY INSIGHTS")
    fprint("=" * 90)

    baseline = all_results.get("DTE_21")
    best_name, best = sorted_variants[0]

    if baseline:
        fprint(f"\nBaseline (DTE=21): Sharpe={baseline['sharpe']:.2f}, "
               f"CAGR={baseline['cagr']*100:.1f}%, Final=${baseline['final_equity']:,.0f}")
        fprint(f"Best variant ({best_name}): Sharpe={best['sharpe']:.2f}, "
               f"CAGR={best['cagr']*100:.1f}%, Final=${best['final_equity']:,.0f}")

        if best_name != "DTE_21":
            sharpe_diff = best["sharpe"] - baseline["sharpe"]
            fprint(f"  Sharpe improvement: {sharpe_diff:+.2f} ({sharpe_diff/abs(baseline['sharpe'])*100:+.1f}%)")
        else:
            fprint(f"  Current DTE=21 is already optimal!")

    # DTE=14 hypothesis test
    dte14 = all_results.get("DTE_14")
    if dte14 and baseline:
        fprint(f"\nHypothesis test (DTE=14 vs DTE=21):")
        fprint(f"  DTE=14: Sharpe={dte14['sharpe']:.2f}, Cost%={dte14['avg_spread_cost_pct']*100:.1f}%, "
               f"ITM%={dte14['itm_rate']*100:.1f}%")
        fprint(f"  DTE=21: Sharpe={baseline['sharpe']:.2f}, Cost%={baseline['avg_spread_cost_pct']*100:.1f}%, "
               f"ITM%={baseline['itm_rate']*100:.1f}%")
        if dte14["sharpe"] > baseline["sharpe"]:
            fprint(f"  CONFIRMED: DTE=14 outperforms DTE=21 by Sharpe {dte14['sharpe']-baseline['sharpe']:+.2f}")
        else:
            fprint(f"  REJECTED: DTE=21 still better by Sharpe {baseline['sharpe']-dte14['sharpe']:+.2f}")

    # Cost efficiency analysis
    fprint(f"\nSpread cost efficiency (entry cost as % of max payoff):")
    for var_name, r in sorted_variants:
        dte_str = "Adaptive" if r["adaptive"] else f"DTE={r['dte']}"
        fprint(f"  {dte_str}: {r['avg_spread_cost_pct']*100:.1f}% "
               f"(shorter DTE = cheaper = more leverage but more theta risk)")

    # Theta drag analysis
    fprint(f"\nTheta drag (% of trades expiring worthless):")
    for var_name, r in sorted_variants:
        dte_str = "Adaptive" if r["adaptive"] else f"DTE={r['dte']}"
        fprint(f"  {dte_str}: {r['theta_drag_rate']*100:.1f}% expire OTM, "
               f"{r['itm_rate']*100:.1f}% expire ITM")

    # ══════════════════════════════════════════════════════════
    # 5-GATE ADVERSARIAL VALIDATION ON TOP 3
    # ══════════════════════════════════════════════════════════

    fprint("\n\n" + "=" * 90)
    fprint("5-GATE ADVERSARIAL VALIDATION — TOP 3 VARIANTS (full 2000 perms)")
    fprint("=" * 90)

    top3_names = [name for name, _ in sorted_variants[:3]]
    adversarial_results = {}

    for var_name in top3_names:
        if var_name not in all_trades:
            continue

        fprint(f"\n--- Full adversarial validation: {var_name} ---")
        result = validate_trades(
            all_trades[var_name],
            initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=f"{var_name}_FULL",
            n_perms=2000,
        )
        result.print_summary()

        adversarial_results[var_name] = {
            "gates_passed": result.gates_passed,
            "gates_total": result.gates_total,
            "all_passed": result.all_passed,
            "gates": [g.name + ": " + ("PASS" if g.passed else "FAIL") for g in result.gates],
        }

        # Update stored results with full adversarial
        if var_name in all_results:
            all_results[var_name]["adversarial_full"] = adversarial_results[var_name]

    # ══════════════════════════════════════════════════════════
    # BEST DTE BY VIX REGIME
    # ══════════════════════════════════════════════════════════

    fprint("\n\n" + "=" * 90)
    fprint("BEST DTE BY VIX REGIME")
    fprint("=" * 90)

    for regime_band in ["VIX_20_25", "VIX_25_30", "VIX_30_plus"]:
        fprint(f"\n  {regime_band}:")
        regime_sharpes = []
        for var_name, r in sorted_variants:
            ra = r.get("regime_analysis", {}).get(regime_band, {})
            if ra.get("insufficient"):
                continue
            if "profit_factor" in ra:
                dte_str = "Adaptive" if r["adaptive"] else f"DTE={r['dte']}"
                fprint(f"    {dte_str}: {ra['n_trades']} trades, WR={ra['win_rate']:.1%}, "
                       f"PF={ra['profit_factor']:.2f}, avg PnL=${ra['avg_pnl']:.2f}")
                regime_sharpes.append((var_name, ra["profit_factor"]))

        if regime_sharpes:
            best_regime = max(regime_sharpes, key=lambda x: x[1])
            dte_str = "Adaptive" if all_results[best_regime[0]]["adaptive"] else f"DTE={all_results[best_regime[0]]['dte']}"
            fprint(f"    >>> Best for {regime_band}: {dte_str} (PF={best_regime[1]:.2f})")

    # ══════════════════════════════════════════════════════════
    # LOG TO MLFLOW
    # ══════════════════════════════════════════════════════════

    if MLFLOW_OK:
        fprint("\n\nLogging to MLflow...")
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"dte_opt_{t0.strftime('%Y%m%d_%H%M%S')}"):
                # Log overall best
                mlflow.log_param("best_variant", best_name)
                mlflow.log_param("n_variants_tested", len(all_results))
                mlflow.log_param("capital", CAP)
                mlflow.log_param("spread_pct", SPREAD_PCT)
                mlflow.log_param("vix_min", VIX_ENTRY_MIN)

                # Log each variant's results
                for var_name, r in all_results.items():
                    prefix = var_name.lower()
                    mlflow.log_metric(f"{prefix}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", r["sortino"])
                    mlflow.log_metric(f"{prefix}_cagr", r["cagr"])
                    mlflow.log_metric(f"{prefix}_maxdd", r["max_dd"])
                    mlflow.log_metric(f"{prefix}_winrate", r["win_rate"])
                    mlflow.log_metric(f"{prefix}_pf", r["profit_factor"])
                    mlflow.log_metric(f"{prefix}_trades", r["n_trades"])
                    mlflow.log_metric(f"{prefix}_final_eq", r["final_equity"])
                    mlflow.log_metric(f"{prefix}_cost_pct", r["avg_spread_cost_pct"])
                    mlflow.log_metric(f"{prefix}_itm_rate", r["itm_rate"])
                    mlflow.log_metric(f"{prefix}_gates", r.get("gates_passed", 0))

            fprint("  MLflow logging complete")
        except Exception as e:
            fprint(f"  MLflow logging failed: {e}")

    # ══════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Clean results for JSON serialization
    clean_results = {}
    for var_name, r in all_results.items():
        clean_r = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                clean_r[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.bool_):
                clean_r[k] = bool(v)
            else:
                clean_r[k] = v
        clean_results[var_name] = clean_r

    output = {
        "experiment": "dte_optimization_v1",
        "run_date": t0.isoformat(),
        "config": {
            "capital": CAP,
            "spread_pct": SPREAD_PCT,
            "haircut": DEFAULT_HAIRCUT,
            "commission": COMMISSION_RT_SPREAD,
            "vix_entry_min": VIX_ENTRY_MIN,
            "rebalance_freq": WF_REBAL_FREQ,
            "top_k": TOP_K,
            "features": V4_FEATURES,
        },
        "results": clean_results,
        "ranking": [name for name, _ in sorted_variants],
        "best_variant": best_name,
        "hypothesis_dte14_vs_21": {
            "dte14_sharpe": dte14["sharpe"] if dte14 else None,
            "dte21_sharpe": baseline["sharpe"] if baseline else None,
            "dte14_wins": (dte14["sharpe"] > baseline["sharpe"]) if (dte14 and baseline) else None,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_FILE}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    fprint("=" * 90)
    fprint("DONE")


if __name__ == "__main__":
    main()
