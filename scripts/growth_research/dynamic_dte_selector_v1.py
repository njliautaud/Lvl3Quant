#!/usr/bin/env python3
"""
Dynamic DTE Selector v1 — Test VIX-Based DTE Switching for Sector Spreads
===========================================================================

Tests whether dynamically selecting DTE based on VIX level improves
sector bull/bear call/put spread performance vs fixed DTE=28 baseline.

6 Variants:
  A: Fixed DTE=28 baseline (V10 config)
  B: VIX-switched (DTE=14 when VIX>25, DTE=28 when VIX<=25)
  C: VIX-switched inverse (DTE=28 when VIX>25, DTE=14 when VIX<=25)
  D: VIX-gradual (DTE = 7 + round(21 * min(VIX/30, 1))) — shorter in high vol
  E: VIX-gradual inverse (DTE = 35 - round(21 * min(VIX/30, 1))) — longer in high vol
  F: Regime-optimal (VIX percentile rank — top quartile DTE=14, bottom three DTE=28)

All variants use:
  - LGBM ranking, 21 features (18 legacy + 3 cross-asset)
  - 500d sliding train window, biweekly (10d) rebalance
  - 11 sectors, top 4 bull + bottom 4 bear (8 positions)
  - 4% OTM, max($3, K*3%) width
  - BS pricing, 15% entry haircut, $2.60 commission RT
  - 30% profit target exit (daily intrinsic check)
  - Hold to expiry if target not hit
  - $645 starting capital, max $200 per trade

5-gate validation on every variant:
  1. Permutation test (300 shuffles, p < 0.05)
  2. Regime stability (VIX>20 vs VIX<=20, Sharpe gap < 0.50)
  3. Sub-period (first half vs second half, both Sharpe > 0.5)
  4. Outlier removal (trim 5% tails, Sharpe > 0.5)
  5. Yearly consistency (>= 60% of years profitable)

Output: results table sorted by Sharpe, JSON to disk, MLflow logging.
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
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Cross-platform detection ──
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

OUTPUT_DIR = BASE / "output" / "growth_research" / "dynamic_dte_selector_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "dynamic_dte_selector_v1"

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
# CONSTANTS
# ══════════════════════════════════════════════════════════════

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "TLT"]

CAP = 645.0
TOP_K = 4              # top 4 bull + bottom 4 bear = 8 positions
OTM_PCT = 0.04         # 4% OTM
PROFIT_TARGET = 0.30   # 30% profit target
DEFAULT_DTE = 28
HAIRCUT = 0.15
COMMISSION = 2.60      # $2.60 RT per spread
EARLY_EXIT_COMMISSION = 2.60
MAX_POS = 200.0
COST_WIDTH_MAX = 0.50

RISK_FREE_RATE = 0.045

WF_TRAIN_DAYS = 500    # 500 trading days sliding window
REBAL_INTERVAL = 10    # biweekly = 10 trading days

N_PERM = 300           # permutation test shuffles

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
assert len(ALL_FEATURES) == 21, f"Expected 21 features, got {len(ALL_FEATURES)}"


# ══════════════════════════════════════════════════════════════
# EMBEDDED BLACK-SCHOLES PRICING (self-contained)
# ══════════════════════════════════════════════════════════════

def bs_call(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def estimate_iv(atr, spot, vix=20.0, atr_period=14):
    """Estimate IV from ATR and VIX."""
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    sigma = realized_vol * iv_mult
    return max(sigma, 0.10)


def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0, haircut=HAIRCUT):
    """Price bull call spread. Returns (entry_cost, max_profit) per share."""
    if K2 <= K1:
        K2 = K1 + 0.50
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix)
    fair = bs_call(S, K1, T, sigma=sigma) - bs_call(S, K2, T, sigma=sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + haircut)
    max_profit = (K2 - K1) - entry_cost
    return float(entry_cost), float(max_profit)


def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0, haircut=HAIRCUT):
    """Price bear put spread. Returns (entry_cost, max_profit) per share."""
    if K2 <= K1:
        K2 = K1 + 0.50
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix)
    fair = bs_put(S, K2, T, sigma=sigma) - bs_put(S, K1, T, sigma=sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + haircut)
    max_profit = (K2 - K1) - entry_cost
    return float(entry_cost), float(max_profit)


# ══════════════════════════════════════════════════════════════
# DTE SELECTOR VARIANTS
# ══════════════════════════════════════════════════════════════

def dte_variant_A(vix_val, vix_history=None):
    """Fixed DTE=28 baseline."""
    return 28


def dte_variant_B(vix_val, vix_history=None):
    """VIX-switched: DTE=14 when VIX>25, DTE=28 when VIX<=25."""
    return 14 if vix_val > 25 else 28


def dte_variant_C(vix_val, vix_history=None):
    """VIX-switched inverse: DTE=28 when VIX>25, DTE=14 when VIX<=25."""
    return 28 if vix_val > 25 else 14


def dte_variant_D(vix_val, vix_history=None):
    """VIX-gradual: DTE = 7 + round(21 * min(VIX/30, 1)) — shorter in high vol."""
    return int(7 + round(21 * min(vix_val / 30.0, 1.0)))


def dte_variant_E(vix_val, vix_history=None):
    """VIX-gradual inverse: DTE = 35 - round(21 * min(VIX/30, 1)) — longer in high vol."""
    return int(35 - round(21 * min(vix_val / 30.0, 1.0)))


def dte_variant_F(vix_val, vix_history=None):
    """Regime-optimal: VIX percentile rank — top quartile DTE=14, bottom three DTE=28."""
    if vix_history is None or len(vix_history) < 63:
        return 28
    pct_rank = float((vix_history < vix_val).mean())
    return 14 if pct_rank >= 0.75 else 28


DTE_VARIANTS = {
    "A_Fixed28": {"func": dte_variant_A, "label": "Fixed DTE=28 (V10 baseline)"},
    "B_VIX_Switch": {"func": dte_variant_B, "label": "DTE=14 if VIX>25, else 28"},
    "C_VIX_Switch_Inv": {"func": dte_variant_C, "label": "DTE=28 if VIX>25, else 14"},
    "D_VIX_Gradual": {"func": dte_variant_D, "label": "DTE=7+21*min(VIX/30,1)"},
    "E_VIX_Gradual_Inv": {"func": dte_variant_E, "label": "DTE=35-21*min(VIX/30,1)"},
    "F_Regime_Optimal": {"func": dte_variant_F, "label": "VIX pctile: top quartile DTE=14"},
}


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download sector ETF + supporting ticker data via yfinance."""
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
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for each sector."""
    atr_dict = {}
    for tk in SECTORS:
        if tk not in high.columns or tk not in low.columns or tk not in close.columns:
            continue
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


def compute_legacy_features(px, spy_slice):
    """Compute 18 legacy momentum/quality features."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.15
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.15
    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0

    # MaxDD 63d
    eq63 = px.iloc[-63:]
    peak63 = eq63.cummax()
    dd63 = (eq63 / peak63 - 1)
    f["maxdd_63d"] = float(dd63.min())

    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

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
    """Compute 3 cross-asset features."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    # Beta
    if spy is not None and len(spy) > 63 and sector_px is not None and len(sector_px) > 63:
        spy_ret = spy.pct_change().dropna()
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

    # Relative vol
    if sector_px is not None and len(sector_px) > 21 and spy is not None and len(spy) > 21:
        sec_vol = sector_px.pct_change().iloc[-21:].std()
        spy_vol = spy.pct_change().iloc[-21:].std()
        f["sector_relative_vol_21d"] = float(sec_vol / (spy_vol + 1e-10))
    else:
        f["sector_relative_vol_21d"] = 1.0

    # Cross-sector dispersion
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
# LGBM WALK-FORWARD RANKING (500d sliding window)
# ══════════════════════════════════════════════════════════════

def get_biweekly_rebalance_dates(close_idx):
    """Get rebalance dates every 10 trading days."""
    dates = sorted(close_idx)
    rebal = []
    last = None
    for i, d in enumerate(dates):
        if last is None:
            if i >= 260:  # need 260 days for features
                rebal.append(d)
                last = i
        elif i - last >= REBAL_INTERVAL:
            rebal.append(d)
            last = i
    return pd.DatetimeIndex(rebal)


def build_feature_records(close, high, low, rebal_dates):
    """Build feature matrix for all rebalance dates. Forward return uses DTE=28 for ranking."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            legacy = compute_legacy_features(px, spy.iloc[:idx + 1])
            if not legacy:
                continue
            cross_asset = compute_cross_asset_features(tk, idx, close)
            # Forward return for label: use 28d horizon (fixed for ranking consistency)
            fi = min(idx + 28, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)
    fprint(f"  Feature records: {len(df)}, {len(df['date'].unique())} rebalance dates")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500-day (approx 24 rebal periods) sliding window."""
    import lightgbm as lgb

    if len(df) < 100:
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    # 500 trading days ~ 25 biweekly periods
    wf_periods = max(25, WF_TRAIN_DAYS // REBAL_INTERVAL)

    rankings = {}
    for i in range(wf_periods, len(dates)):
        train_dates = dates[max(0, i - wf_periods):i]
        test_date = dates[i]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 50:
            continue
        Xt = np.nan_to_num(train_df[ALL_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[ALL_FEATURES].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue

    fprint(f"  LGBM walk-forward: {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# STRIKE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct=OTM_PCT):
    """Compute strike prices: 4% OTM, adaptive width max($3, K*3%)."""
    if direction == "bull":
        K1 = round(S * (1.0 + otm_pct), 2)
        w = max(3.0, K1 * 0.03)
        K2 = round(K1 + w, 2)
    else:  # bear
        K2 = round(S * (1.0 - otm_pct), 2)
        w = max(3.0, K2 * 0.03)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE EXECUTION WITH PROFIT TARGET + DYNAMIC DTE
# ══════════════════════════════════════════════════════════════

def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity, dte):
    """Execute a single spread trade with profit target exit.

    Args:
        dte: DTE to use for this trade (varies by variant)
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    # Price the spread
    try:
        if direction == "bull":
            entry_cost_ps, max_profit_ps = price_bull_call_spread(S, K1, K2, dte, av, vix_val)
        else:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(S, K1, K2, dte, av, vix_val)
    except Exception:
        return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION
    if total_cost <= 0 or total_cost > MAX_POS or total_cost > equity * 0.40:
        return None

    # ── Profit target exit: walk daily ──
    exited_early = False
    exit_day_idx = ei
    exit_reason = "expiry"
    hold_days = dte

    if max_profit_ps > 0 and PROFIT_TARGET < 1.0:
        for check_idx in range(di + 1, ei + 1):
            if check_idx >= len(close):
                break
            S_now = float(close[tk].iloc[check_idx])

            # Compute intrinsic value of spread at this point
            if direction == "bull":
                intrinsic_now = max(S_now - K1, 0.0) - max(S_now - K2, 0.0)
            else:
                intrinsic_now = max(K2 - S_now, 0.0) - max(K1 - S_now, 0.0)

            unrealized_gain = intrinsic_now - entry_cost_ps
            if unrealized_gain >= PROFIT_TARGET * max_profit_ps:
                exited_early = True
                exit_day_idx = check_idx
                exit_reason = "profit_target_30pct"
                hold_days = check_idx - di
                break

    # Final P&L
    Se = float(close[tk].iloc[exit_day_idx])

    if exited_early:
        # Early exit: use intrinsic - entry cost (no haircut on exit, intrinsic-based)
        if direction == "bull":
            exit_intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            exit_intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (exit_intrinsic - entry_cost_ps) * 100 - COMMISSION - EARLY_EXIT_COMMISSION
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "K1": K1, "K2": K2,
        "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
        "max_profit_ps": round(max_profit_ps, 4),
        "exited_early": exited_early,
        "exit_reason": exit_reason,
        "hold_days": hold_days,
        "dte_used": dte,
    }


# ══════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════

def simulate_variant(variant_name, dte_func, rankings, close, atr_dict, rebal_dates):
    """Run full backtest for a DTE variant."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    vix_full = vix.dropna() if vix is not None else None

    equity = CAP
    trades = []
    dte_distribution = []

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue

        # Find closest ranking date
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        if not scores or len(scores) < 6:
            continue

        # Get VIX for DTE selection
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # VIX history for percentile-based variants
        di = close.index.get_loc(dt)
        vix_hist = vix_full.iloc[:di].values if vix_full is not None and di > 63 else None

        # Select DTE for this rebalance
        dte = dte_func(cv, vix_hist)
        dte = max(7, min(60, dte))  # clamp to sane range
        dte_distribution.append(dte)

        # Rank sectors
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

        n_positions = len(bull_picks) + len(bear_picks)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                max_pos = min(MAX_POS, equity * 0.40)
                if max_pos < 20:
                    continue

                result = execute_trade(tk, dt, direction, close, atr_dict, cv, equity, dte)
                if result is not None:
                    equity += result["pnl"]
                    di_entry = close.index.get_loc(dt)
                    ei_entry = min(di_entry + dte, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei_entry]) if ei_entry < len(spy) else sv
                    exit_idx = di_entry + result["hold_days"]
                    if exit_idx >= len(close):
                        exit_idx = ei_entry

                    trades.append({
                        **result,
                        "entry_date": str(dt.date()),
                        "exit_date": str(close.index[exit_idx].date()),
                        "ticker": tk,
                        "regime": "bull" if se >= sv else "bear",
                        "direction": direction,
                        "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                    })

    return trades, equity, dte_distribution


# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════

def compute_stats(trades, initial_capital=CAP):
    """Compute Sharpe, Sortino, PF, WR, MDD, total return."""
    if not trades or len(trades) < 5:
        return None

    pnls = np.array([t["pnl"] for t in trades])
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Win rate, profit factor
    wr = float(np.mean(pnls > 0))
    gross_win = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-10
    pf = gross_win / max(gross_loss, 1e-10)

    # Calendar month Sharpe
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()
    if len(monthly_pnl) > 2:
        sharpe = float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * np.sqrt(12))
    else:
        sharpe = 0.0

    # Sortino
    monthly_ret = monthly_pnl / initial_capital
    neg_ret = monthly_ret[monthly_ret < 0]
    if len(neg_ret) > 1:
        sortino = float(monthly_ret.mean() / (neg_ret.std() + 1e-10) * np.sqrt(12))
    else:
        sortino = sharpe * 1.5  # no negative months = very good

    total_return = float(equity[-1] / initial_capital - 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "max_dd": round(max_dd * 100, 2),
        "total_return_pct": round(total_return, 2),
        "final_equity": round(equity[-1], 2),
        "n_trades": len(trades),
        "avg_pnl": round(float(pnls.mean()), 2),
        "monthly_pnl_series": monthly_pnl,
    }


def compute_sharpe_from_trades(trades, initial_capital=CAP):
    """Quick Sharpe computation for validation gates."""
    if not trades or len(trades) < 3:
        return 0.0
    trade_df = pd.DataFrame(trades)
    trade_df["entry_month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("entry_month")["pnl"].sum()
    if len(monthly_pnl) < 2:
        return 0.0
    return float(monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * np.sqrt(12))


# ══════════════════════════════════════════════════════════════
# 5-GATE VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_variant(trades, variant_name, vix_series=None):
    """Run 5-gate validation. Returns dict with gate results."""
    gates = {}
    n = len(trades)
    if n < 10:
        return {"error": "Too few trades", "gates_passed": 0, "gates_total": 5}

    pnls = np.array([t["pnl"] for t in trades])
    actual_sharpe = compute_sharpe_from_trades(trades)

    # ── Gate 1: Permutation test (300 shuffles) ──
    rng = np.random.RandomState(42)
    perm_sharpes = []
    for _ in range(N_PERM):
        # Shuffle trade directions (flip PnL signs randomly)
        signs = rng.choice([-1, 1], size=n)
        shuffled = [dict(t) for t in trades]
        for j, s in enumerate(signs):
            shuffled[j]["pnl"] = trades[j]["pnl"] * s
        perm_sharpes.append(compute_sharpe_from_trades(shuffled))

    p_value = float(np.mean(np.array(perm_sharpes) >= actual_sharpe))
    gates["G1_permutation"] = {
        "passed": p_value < 0.05,
        "p_value": round(p_value, 4),
        "threshold": 0.05,
    }

    # ── Gate 2: Regime stability (VIX>20 vs VIX<=20) ──
    high_vix_trades = [t for t in trades if t.get("vix", 20) > 20]
    low_vix_trades = [t for t in trades if t.get("vix", 20) <= 20]
    sharpe_high = compute_sharpe_from_trades(high_vix_trades)
    sharpe_low = compute_sharpe_from_trades(low_vix_trades)
    gap = abs(sharpe_high - sharpe_low)
    gates["G2_regime_stability"] = {
        "passed": gap < 0.50 or len(high_vix_trades) < 10 or len(low_vix_trades) < 10,
        "sharpe_vix_high": round(sharpe_high, 3),
        "sharpe_vix_low": round(sharpe_low, 3),
        "gap": round(gap, 3),
        "threshold": 0.50,
    }

    # ── Gate 3: Sub-period stability ──
    half = n // 2
    first_half = trades[:half]
    second_half = trades[half:]
    sharpe_1h = compute_sharpe_from_trades(first_half)
    sharpe_2h = compute_sharpe_from_trades(second_half)
    gates["G3_subperiod"] = {
        "passed": sharpe_1h > 0.5 and sharpe_2h > 0.5,
        "sharpe_first_half": round(sharpe_1h, 3),
        "sharpe_second_half": round(sharpe_2h, 3),
        "threshold": 0.5,
    }

    # ── Gate 4: Outlier removal (trim 5% tails) ──
    sorted_pnls = np.sort(pnls)
    trim_n = max(1, int(n * 0.05))
    trimmed_trades = [t for t in trades if t["pnl"] >= sorted_pnls[trim_n] and t["pnl"] <= sorted_pnls[-trim_n - 1]]
    sharpe_trimmed = compute_sharpe_from_trades(trimmed_trades)
    gates["G4_outlier_removal"] = {
        "passed": sharpe_trimmed > 0.5,
        "sharpe_trimmed": round(sharpe_trimmed, 3),
        "trades_remaining": len(trimmed_trades),
        "threshold": 0.5,
    }

    # ── Gate 5: Yearly consistency (>=60% profitable) ──
    trade_df = pd.DataFrame(trades)
    trade_df["year"] = pd.to_datetime(trade_df["entry_date"]).dt.year
    yearly_pnl = trade_df.groupby("year")["pnl"].sum()
    pct_profitable = float((yearly_pnl > 0).mean()) if len(yearly_pnl) > 0 else 0
    gates["G5_yearly_consistency"] = {
        "passed": pct_profitable >= 0.60,
        "pct_profitable_years": round(pct_profitable, 3),
        "yearly_pnl": {str(y): round(v, 2) for y, v in yearly_pnl.items()},
        "threshold": 0.60,
    }

    gates_passed = sum(1 for g in gates.values() if g.get("passed", False))

    return {
        "gates": gates,
        "gates_passed": gates_passed,
        "gates_total": 5,
        "all_passed": gates_passed == 5,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("DYNAMIC DTE SELECTOR V1 — VIX-Based DTE Switching Research")
    fprint("=" * 70)

    # ── Download data ──
    fprint("\n[1/4] Downloading data...")
    close, high, low = download_data()

    # ── Compute ATR ──
    fprint("\n[2/4] Computing features...")
    atr_dict = compute_atr_series(high, low, close)

    # ── Build rebalance dates ──
    rebal_dates = get_biweekly_rebalance_dates(close.index)
    fprint(f"  Biweekly rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # ── Build features + LGBM rankings (shared across all variants) ──
    fprint("  Building feature records...")
    feat_df = build_feature_records(close, high, low, rebal_dates)

    fprint("  Training LGBM walk-forward model...")
    rankings = walk_forward_lgbm_rank(feat_df)

    if len(rankings) < 10:
        fprint("ERROR: Too few ranking dates. Aborting.")
        return

    # Filter rebalance dates to those with rankings
    valid_rebal = pd.DatetimeIndex([d for d in rebal_dates if any(rd <= d for rd in rankings.keys())])
    fprint(f"  Valid rebalance dates with rankings: {len(valid_rebal)}")

    # ── Run all variants ──
    fprint(f"\n[3/4] Running {len(DTE_VARIANTS)} DTE variants...")
    results = {}
    all_json = {}

    vix_series = close["VIX"] if "VIX" in close.columns else None

    for vname, vconfig in DTE_VARIANTS.items():
        fprint(f"\n  --- Variant {vname}: {vconfig['label']} ---")
        trades, final_eq, dte_dist = simulate_variant(
            vname, vconfig["func"], rankings, close, atr_dict, valid_rebal
        )
        fprint(f"    Trades: {len(trades)}, Final equity: ${final_eq:.2f}")

        if not trades or len(trades) < 10:
            fprint(f"    SKIP: too few trades")
            continue

        # DTE distribution stats
        if dte_dist:
            fprint(f"    DTE distribution: min={min(dte_dist)}, max={max(dte_dist)}, "
                   f"mean={np.mean(dte_dist):.1f}, median={np.median(dte_dist):.0f}")

        # Compute stats
        st = compute_stats(trades)
        if st is None:
            fprint(f"    SKIP: stats computation failed")
            continue

        fprint(f"    Sharpe={st['sharpe']:.3f}  Sortino={st['sortino']:.3f}  "
               f"PF={st['pf']:.3f}  WR={st['wr']:.4f}  MDD={st['max_dd']:.2f}%  "
               f"Return={st['total_return_pct']:.1f}%")

        # Validate
        fprint(f"    Running 5-gate validation...")
        val = validate_variant(trades, vname, vix_series)
        fprint(f"    Gates passed: {val['gates_passed']}/{val['gates_total']}")
        for gname, gres in val.get("gates", {}).items():
            status = "PASS" if gres.get("passed") else "FAIL"
            fprint(f"      [{status}] {gname}: {gres}")

        # Early exit stats
        early = sum(1 for t in trades if t.get("exited_early", False))
        avg_hold = np.mean([t.get("hold_days", 28) for t in trades])

        results[vname] = {
            "label": vconfig["label"],
            **{k: v for k, v in st.items() if k != "monthly_pnl_series"},
            "early_exit_rate": round(early / len(trades), 3),
            "avg_hold_days": round(avg_hold, 1),
            "dte_mean": round(np.mean(dte_dist), 1) if dte_dist else 28,
            "dte_std": round(np.std(dte_dist), 1) if dte_dist else 0,
            "gates_passed": val["gates_passed"],
            "gates_total": val["gates_total"],
            "all_gates_passed": val.get("all_passed", False),
            "validation": val,
        }

        all_json[vname] = {
            **results[vname],
            "trades_summary": {
                "total": len(trades),
                "bull": sum(1 for t in trades if t["direction"] == "bull"),
                "bear": sum(1 for t in trades if t["direction"] == "bear"),
                "early_exits": early,
            },
        }

    # ── Print results table ──
    fprint("\n" + "=" * 100)
    fprint("[4/4] RESULTS TABLE — Sorted by Sharpe (Calendar Month)")
    fprint("=" * 100)

    sorted_variants = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    header = f"{'Variant':<22} {'Sharpe':>8} {'Sortino':>8} {'PF':>7} {'WR':>7} {'MDD%':>7} {'Return%':>9} {'Trades':>7} {'Gates':>6} {'AvgDTE':>7}"
    fprint(header)
    fprint("-" * len(header))

    for vname, st in sorted_variants:
        gates_str = f"{st['gates_passed']}/{st['gates_total']}"
        fprint(f"{vname:<22} {st['sharpe']:>8.3f} {st['sortino']:>8.3f} {st['pf']:>7.3f} "
               f"{st['wr']:>7.4f} {st['max_dd']:>7.2f} {st['total_return_pct']:>9.1f} "
               f"{st['n_trades']:>7d} {gates_str:>6} {st['dte_mean']:>7.1f}")

    # ── Winner analysis ──
    if sorted_variants:
        winner_name, winner_stats = sorted_variants[0]
        fprint(f"\n{'='*70}")
        fprint(f"WINNER: {winner_name} — {results[winner_name]['label']}")
        fprint(f"  Sharpe: {winner_stats['sharpe']:.3f}")
        fprint(f"  Sortino: {winner_stats['sortino']:.3f}")
        fprint(f"  Profit Factor: {winner_stats['pf']:.3f}")
        fprint(f"  Win Rate: {winner_stats['wr']:.4f}")
        fprint(f"  Max Drawdown: {winner_stats['max_dd']:.2f}%")
        fprint(f"  Total Return: {winner_stats['total_return_pct']:.1f}%")
        fprint(f"  Avg Hold Days: {winner_stats['avg_hold_days']:.1f}")
        fprint(f"  Avg DTE: {winner_stats['dte_mean']:.1f} (std={winner_stats['dte_std']:.1f})")
        fprint(f"  Gates Passed: {winner_stats['gates_passed']}/{winner_stats['gates_total']}")
        fprint(f"  All Gates: {'YES' if winner_stats['all_gates_passed'] else 'NO'}")

        # Baseline comparison
        if "A_Fixed28" in results:
            baseline = results["A_Fixed28"]
            fprint(f"\n  vs Baseline (A_Fixed28):")
            fprint(f"    Sharpe delta: {winner_stats['sharpe'] - baseline['sharpe']:+.3f}")
            fprint(f"    Return delta: {winner_stats['total_return_pct'] - baseline['total_return_pct']:+.1f}%")
            fprint(f"    MDD delta: {winner_stats['max_dd'] - baseline['max_dd']:+.2f}%")

    # ── Save JSON ──
    json_path = OUTPUT_DIR / "dynamic_dte_selector_v1_results.json"
    with open(json_path, "w") as f:
        # Remove non-serializable objects
        clean_json = {}
        for k, v in all_json.items():
            clean = {kk: vv for kk, vv in v.items() if kk != "monthly_pnl_series"}
            # Clean nested validation dict
            if "validation" in clean:
                val = clean["validation"]
                if "gates" in val:
                    for gname, gval in val["gates"].items():
                        if "yearly_pnl" in gval:
                            gval["yearly_pnl"] = {str(k2): float(v2) for k2, v2 in gval["yearly_pnl"].items()}
            clean_json[k] = clean
        json.dump(clean_json, f, indent=2, default=str)
    fprint(f"\nResults saved to: {json_path}")

    # ── Log to MLflow ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"dte_selector_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Log best variant metrics
                if sorted_variants:
                    best_name, best_stats = sorted_variants[0]
                    mlflow.log_param("best_variant", best_name)
                    mlflow.log_param("n_variants", len(results))
                    mlflow.log_metric("best_sharpe", best_stats["sharpe"])
                    mlflow.log_metric("best_sortino", best_stats["sortino"])
                    mlflow.log_metric("best_pf", best_stats["pf"])
                    mlflow.log_metric("best_wr", best_stats["wr"])
                    mlflow.log_metric("best_mdd", best_stats["max_dd"])
                    mlflow.log_metric("best_total_return", best_stats["total_return_pct"])
                    mlflow.log_metric("best_gates_passed", best_stats["gates_passed"])

                    # Log all variant Sharpes
                    for vn, vs in results.items():
                        safe_name = vn.replace(" ", "_").lower()
                        mlflow.log_metric(f"sharpe_{safe_name}", vs["sharpe"])
                        mlflow.log_metric(f"gates_{safe_name}", vs["gates_passed"])

                    if "A_Fixed28" in results:
                        mlflow.log_metric("baseline_sharpe", results["A_Fixed28"]["sharpe"])
                        mlflow.log_metric("sharpe_improvement", best_stats["sharpe"] - results["A_Fixed28"]["sharpe"])

                mlflow.log_artifact(str(json_path))
            fprint(f"Logged to MLflow experiment: {EXPERIMENT_NAME}")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    fprint("Done.")


if __name__ == "__main__":
    main()
