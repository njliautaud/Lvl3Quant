#!/usr/bin/env python3
"""
Combined OTM Portfolio v1 — Does 2% OTM Improve the Combined Portfolio?
=========================================================================

Context:
  - V6 bull spreads with 2% OTM: Sharpe 2.44 vs ATM 2.04 (+19%)
  - Pair trades with 2% OTM: Sharpe 2.36 vs ATM 1.51 (+56%)
  - ATM combined portfolio (finding #119-126): Risk parity combo Sharpe 1.65, 100% invested

Question: Does 2% OTM improve the combined portfolio too?

4 Variants:
  A) ATM Combo (control): Bull spreads VIX>20 + pairs VIX<20, all ATM
  B) OTM 2% Combo: Same structure but ALL legs use 2% OTM
  C) OTM 2% Risk Parity: Same as B + risk parity overlay in low-VIX periods
  D) OTM 2% Weekly: Same as B but weekly rebalance for both bull and pair legs

Moneyness implementation:
  Bull calls: K1 = S * (1 + moneyness_pct/100), K2 = K1 * (1 + spread_width/100)
  Bear puts:  K1 = S * (1 - moneyness_pct/100), K2 = K1 * (1 - spread_width/100)

$645 starting capital, $200 max/trade, 21 DTE, 3% spread width, 15% entry haircut,
no exit haircut, $2.60 commission per spread, hold to expiry, LGBM with 21 features.
"""

import json
import sys
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "combined_otm_portfolio_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "combined_otm_portfolio_v1_results.json"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "TLT", "GLD", "^VIX"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
MAX_POS = 200.0

# Walk-forward: 500d train (~25 biweekly periods), 250d test
WF_TRAIN_DAYS = 500
WF_TEST_DAYS = 250
WF_TRAIN_PERIODS = 25
WF_REBAL_FREQ_BIWEEKLY = "2W-FRI"
WF_REBAL_FREQ_WEEKLY = "W-FRI"

# 21 production features
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
EXPERIMENT_NAME = "combined_otm_portfolio_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=2)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")

# ── GRU regime predictions (if available) ──
REGIME_PREDICTIONS = None
REGIME_DATES = None
try:
    regime_data = np.load(BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz", allow_pickle=True)
    REGIME_PREDICTIONS = regime_data.get("predictions", None)
    REGIME_DATES = regime_data.get("dates", None)
    if REGIME_PREDICTIONS is not None and REGIME_DATES is not None:
        fprint(f"GRU regime predictions loaded: {len(REGIME_PREDICTIONS)} dates")
    else:
        REGIME_PREDICTIONS = None
        REGIME_DATES = None
        fprint("GRU regime data incomplete, falling back to VIX>20 threshold")
except Exception as e:
    fprint(f"GRU regime data not loadable ({e}), using VIX>20 threshold")


def get_regime(dt, vix_value):
    """Return 'high_vol' or 'low_vol' using GRU predictions if available, else VIX threshold."""
    if REGIME_PREDICTIONS is not None and REGIME_DATES is not None:
        # Find closest date
        dt_str = str(pd.Timestamp(dt).date())
        dates_list = [str(d) for d in REGIME_DATES]
        if dt_str in dates_list:
            idx = dates_list.index(dt_str)
            pred = REGIME_PREDICTIONS[idx]
            # Assume prediction > 0.5 = high_vol regime
            return "high_vol" if pred > 0.5 else "low_vol"
    # Fallback: VIX > 20
    return "high_vol" if vix_value >= 20 else "low_vol"


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download sector ETFs, SPY, TLT, GLD, VIX from yfinance (2007-2026)."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start="2007-01-01", progress=False, auto_adjust=True)
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
    """Compute the 3 cross-asset features."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

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

def build_feature_records(close, high, low, rebal_dates, feature_cols):
    """Build feature + target records for LGBM walk-forward ranking."""
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = {}
            for col in feature_cols:
                if col in CROSS_ASSET_FEATURES:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

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


def walk_forward_lgbm_rank(df, feature_cols):
    """Walk-forward LGBM: sliding 500d train, 250d test window."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
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

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION ENGINE — WITH MONEYNESS PARAMETER
# ══════════════════════════════════════════════════════════════

def execute_bull_call_trades(picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                              moneyness_pct=0.0):
    """Execute bull call spreads on top-ranked sectors.

    moneyness_pct: 0 = ATM, 2 = 2% OTM (strikes shifted up by 2%).
    K1 = S * (1 + moneyness_pct / 100)
    K2 = K1 * (1 + spread_width / 100)
    """
    trades = []
    total_pnl = 0.0

    for tk in picks:
        if tk not in close.columns or tk not in atr_dict:
            continue

        S = float(close[tk].iloc[idx])
        ei = min(idx + DTE, len(close) - 1)
        if ei <= idx:
            continue

        if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
            av = float(atr_dict[tk].loc[dt])
        else:
            av = S * 0.015

        # Apply moneyness: shift strikes OTM
        K1 = round(S * (1 + moneyness_pct / 100), 2)
        K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
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

        # Hold to expiry -- intrinsic value only
        Se = float(close[tk].iloc[ei])
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

        total_pnl += pnl
        trades.append({
            "pnl": round(pnl, 2),
            "entry_date": str(close.index[idx].date()),
            "exit_date": str(close.index[ei].date()),
            "ticker": tk,
            "leg": "bull_call",
            "vix": round(cv, 1),
            "win": pnl > 0,
            "mode": "bull",
            "moneyness_pct": moneyness_pct,
            "K1": K1,
            "K2": K2,
        })

    return trades, total_pnl


def execute_bear_put_trades(picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                              moneyness_pct=0.0):
    """Execute bear put spreads on bottom-ranked sectors.

    moneyness_pct: 0 = ATM, 2 = 2% OTM (strikes shifted down by 2%).
    K1 = S * (1 - moneyness_pct / 100)          (higher strike, long put)
    K2 = K1 * (1 - spread_width / 100)           (lower strike, short put)
    """
    trades = []
    total_pnl = 0.0

    for tk in picks:
        if tk not in close.columns or tk not in atr_dict:
            continue

        S = float(close[tk].iloc[idx])
        ei = min(idx + DTE, len(close) - 1)
        if ei <= idx:
            continue

        if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
            av = float(atr_dict[tk].loc[dt])
        else:
            av = S * 0.015

        # Apply moneyness: shift strikes OTM (down for puts)
        K1 = round(S * (1 - moneyness_pct / 100), 2)  # Higher strike (long put)
        K2 = round(K1 * (1 - SPREAD_PCT / 100), 2)     # Lower strike (short put)
        if K1 <= K2:
            continue

        try:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(
                S=S, K1=K2, K2=K1, dte=DTE, atr=av, vix=cv
            )
        except Exception:
            continue

        total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
        if total_cost <= 0 or total_cost > max_per_trade or total_cost > equity * 0.40:
            continue

        # Hold to expiry -- intrinsic value only
        Se = float(close[tk].iloc[ei])
        intrinsic = max(K1 - Se, 0.0) - max(K2 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

        total_pnl += pnl
        trades.append({
            "pnl": round(pnl, 2),
            "entry_date": str(close.index[idx].date()),
            "exit_date": str(close.index[ei].date()),
            "ticker": tk,
            "leg": "bear_put",
            "vix": round(cv, 1),
            "win": pnl > 0,
            "mode": "pair",
            "moneyness_pct": moneyness_pct,
            "K1": K2,  # Lower strike
            "K2": K1,  # Higher strike
        })

    return trades, total_pnl


def execute_risk_parity_allocation(close, idx, dt, equity, alloc_pct=0.30):
    """Simple risk parity allocation: SPY/TLT/GLD 33/33/34. Returns PnL for the DTE period."""
    rp_tickers = {"SPY": 0.33, "TLT": 0.33, "GLD": 0.34}
    ei = min(idx + DTE, len(close) - 1)
    if ei <= idx:
        return 0.0

    alloc_dollars = equity * alloc_pct
    total_pnl = 0.0

    for tk, wt in rp_tickers.items():
        if tk not in close.columns:
            continue
        S_entry = float(close[tk].iloc[idx])
        S_exit = float(close[tk].iloc[ei])
        if S_entry <= 0:
            continue
        position_dollars = alloc_dollars * wt
        shares = position_dollars / S_entry
        pnl = shares * (S_exit - S_entry)
        total_pnl += pnl

    return total_pnl


# ══════════════════════════════════════════════════════════════
# VARIANT SIMULATORS
# ══════════════════════════════════════════════════════════════

def simulate_variant_A(rankings, close, atr_dict):
    """A_atm_combo (CONTROL): Bull spreads VIX>20 + pairs VIX<20, all ATM."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0

        scores = rankings[dt]
        if not scores or len(scores) < 6:
            total_periods += 1
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_per_trade = min(MAX_POS, equity / 3)
        if max_per_trade < 30:
            total_periods += 1
            continue

        total_periods += 1
        regime = get_regime(dt, cv)

        if regime == "high_vol":
            picks = [t for t, _ in ranked[:3]]
            new_trades, pnl = execute_bull_call_trades(
                picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=0.0)
            for t in new_trades:
                t["mode"] = "bull"
            all_trades.extend(new_trades)
            equity += pnl
            if new_trades:
                invested_periods += 1
        else:
            long_picks = [t for t, _ in ranked[:3]]
            short_picks = [t for t, _ in ranked[-3:]]
            short_picks = [t for t in short_picks if t not in long_picks]

            long_trades, long_pnl = execute_bull_call_trades(
                long_picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=0.0)
            for t in long_trades:
                t["mode"] = "pair_long"
            equity += long_pnl

            short_trades, short_pnl = execute_bear_put_trades(
                short_picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=0.0)
            for t in short_trades:
                t["mode"] = "pair_short"
            equity += short_pnl

            all_trades.extend(long_trades)
            all_trades.extend(short_trades)
            if long_trades or short_trades:
                invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    return all_trades, equity, pct_invested


def simulate_variant_B(rankings, close, atr_dict):
    """B_otm2_combo: Same structure as A but ALL legs use 2% OTM."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0

        scores = rankings[dt]
        if not scores or len(scores) < 6:
            total_periods += 1
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_per_trade = min(MAX_POS, equity / 3)
        if max_per_trade < 30:
            total_periods += 1
            continue

        total_periods += 1
        regime = get_regime(dt, cv)

        if regime == "high_vol":
            picks = [t for t, _ in ranked[:3]]
            new_trades, pnl = execute_bull_call_trades(
                picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=2.0)
            for t in new_trades:
                t["mode"] = "bull"
            all_trades.extend(new_trades)
            equity += pnl
            if new_trades:
                invested_periods += 1
        else:
            long_picks = [t for t, _ in ranked[:3]]
            short_picks = [t for t, _ in ranked[-3:]]
            short_picks = [t for t in short_picks if t not in long_picks]

            long_trades, long_pnl = execute_bull_call_trades(
                long_picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=2.0)
            for t in long_trades:
                t["mode"] = "pair_long"
            equity += long_pnl

            short_trades, short_pnl = execute_bear_put_trades(
                short_picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=2.0)
            for t in short_trades:
                t["mode"] = "pair_short"
            equity += short_pnl

            all_trades.extend(long_trades)
            all_trades.extend(short_trades)
            if long_trades or short_trades:
                invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    return all_trades, equity, pct_invested


def simulate_variant_C(rankings, close, atr_dict):
    """C_otm2_riskparity: Same as B + risk parity overlay in low-VIX periods."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0

        scores = rankings[dt]
        if not scores or len(scores) < 6:
            total_periods += 1
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_per_trade = min(MAX_POS, equity / 3)
        if max_per_trade < 30:
            total_periods += 1
            continue

        total_periods += 1
        regime = get_regime(dt, cv)

        if regime == "high_vol":
            picks = [t for t, _ in ranked[:3]]
            new_trades, pnl = execute_bull_call_trades(
                picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=2.0)
            for t in new_trades:
                t["mode"] = "bull"
            all_trades.extend(new_trades)
            equity += pnl
            if new_trades:
                invested_periods += 1
        else:
            # Pair trades with 2% OTM
            long_picks = [t for t, _ in ranked[:3]]
            short_picks = [t for t, _ in ranked[-3:]]
            short_picks = [t for t in short_picks if t not in long_picks]

            max_pair_trade = min(MAX_POS, equity / 6)  # Smaller size to leave room for RP

            long_trades, long_pnl = execute_bull_call_trades(
                long_picks, close, atr_dict, idx, dt, cv, max_pair_trade, equity,
                moneyness_pct=2.0)
            for t in long_trades:
                t["mode"] = "pair_long"
            equity += long_pnl

            short_trades, short_pnl = execute_bear_put_trades(
                short_picks, close, atr_dict, idx, dt, cv, max_pair_trade, equity,
                moneyness_pct=2.0)
            for t in short_trades:
                t["mode"] = "pair_short"
            equity += short_pnl

            all_trades.extend(long_trades)
            all_trades.extend(short_trades)

            # Risk parity overlay: 30% of equity into SPY/TLT/GLD
            rp_pnl = execute_risk_parity_allocation(close, idx, dt, equity, alloc_pct=0.30)
            if abs(rp_pnl) > 0:
                equity += rp_pnl
                all_trades.append({
                    "pnl": round(rp_pnl, 2),
                    "entry_date": str(close.index[idx].date()),
                    "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                    "ticker": "RP_BASKET",
                    "leg": "risk_parity",
                    "vix": round(cv, 1),
                    "win": rp_pnl > 0,
                    "mode": "risk_parity",
                    "moneyness_pct": 0.0,
                })

            if long_trades or short_trades:
                invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    return all_trades, equity, pct_invested


def simulate_variant_D(rankings_weekly, close, atr_dict):
    """D_otm2_weekly: Same as B but weekly rebalance for both bull and pair legs."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0

    for dt in sorted(rankings_weekly.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0

        scores = rankings_weekly[dt]
        if not scores or len(scores) < 6:
            total_periods += 1
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_per_trade = min(MAX_POS, equity / 3)
        if max_per_trade < 30:
            total_periods += 1
            continue

        total_periods += 1
        regime = get_regime(dt, cv)

        if regime == "high_vol":
            picks = [t for t, _ in ranked[:3]]
            new_trades, pnl = execute_bull_call_trades(
                picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=2.0)
            for t in new_trades:
                t["mode"] = "bull"
            all_trades.extend(new_trades)
            equity += pnl
            if new_trades:
                invested_periods += 1
        else:
            long_picks = [t for t, _ in ranked[:3]]
            short_picks = [t for t, _ in ranked[-3:]]
            short_picks = [t for t in short_picks if t not in long_picks]

            long_trades, long_pnl = execute_bull_call_trades(
                long_picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=2.0)
            for t in long_trades:
                t["mode"] = "pair_long"
            equity += long_pnl

            short_trades, short_pnl = execute_bear_put_trades(
                short_picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                moneyness_pct=2.0)
            for t in short_trades:
                t["mode"] = "pair_short"
            equity += short_pnl

            all_trades.extend(long_trades)
            all_trades.extend(short_trades)
            if long_trades or short_trades:
                invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    return all_trades, equity, pct_invested


# ══════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def compute_spy_beta(trades, close):
    """Compute SPY beta from trade returns."""
    if not trades or len(trades) < 10:
        return 0.0, 0.0

    spy = close["SPY"]
    trade_rets = []
    spy_rets = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])
        if entry in spy.index and exit_dt in spy.index:
            sr = float(spy.loc[exit_dt] / spy.loc[entry] - 1)
            trade_rets.append(t["pnl"] / max(CAP, 100))
            spy_rets.append(sr)

    if len(trade_rets) < 10:
        return 0.0, 0.0

    tr = np.array(trade_rets)
    sr = np.array(spy_rets)
    corr = np.corrcoef(tr, sr)[0, 1]
    cov = np.cov(tr, sr)
    beta = cov[0, 1] / (cov[1, 1] + 1e-10)

    return float(beta) if not np.isnan(beta) else 0.0, float(corr) if not np.isnan(corr) else 0.0


def compute_monthly_spy_correlation(trades, close):
    """Compute monthly return correlation with SPY."""
    if not trades or len(trades) < 20:
        return 0.0

    spy = close["SPY"]
    tdf = pd.DataFrame(trades)
    tdf["entry_dt"] = pd.to_datetime(tdf["entry_date"])
    tdf["month"] = tdf["entry_dt"].dt.to_period("M")

    monthly_pnl = tdf.groupby("month")["pnl"].sum()

    spy_monthly = spy.resample("ME").last().pct_change().dropna()

    common_months = []
    for mo in monthly_pnl.index:
        mo_end = mo.to_timestamp(how="E")
        closest = spy_monthly.index[spy_monthly.index <= mo_end]
        if len(closest) > 0:
            common_months.append((monthly_pnl[mo], float(spy_monthly.loc[closest[-1]])))

    if len(common_months) < 6:
        return 0.0

    trade_m, spy_m = zip(*common_months)
    corr = np.corrcoef(trade_m, spy_m)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def vix_regime_stratified_sharpe(trades, annualize_periods=26):
    """Compute Sharpe by VIX regime: <20, 20-30, >30."""
    if not trades:
        return {}

    regimes = {"vix_lt_20": [], "vix_20_30": [], "vix_gt_30": []}
    for t in trades:
        v = t.get("vix", 20)
        if v < 20:
            regimes["vix_lt_20"].append(t["pnl"])
        elif v <= 30:
            regimes["vix_20_30"].append(t["pnl"])
        else:
            regimes["vix_gt_30"].append(t["pnl"])

    result = {}
    for label, pnls in regimes.items():
        if len(pnls) < 5:
            result[label] = {"n": len(pnls), "sharpe": 0.0, "avg_pnl": 0.0}
            continue
        arr = np.array(pnls)
        avg = arr.mean()
        std = arr.std()
        sharpe = float(avg / (std + 1e-10) * np.sqrt(annualize_periods))
        result[label] = {
            "n": len(pnls),
            "sharpe": round(sharpe, 2),
            "avg_pnl": round(float(avg), 2),
            "total_pnl": round(float(arr.sum()), 2),
        }

    return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"COMBINED OTM PORTFOLIO v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only | No exit haircut")
    fprint(f"Walk-forward: {WF_TRAIN_PERIODS} train periods, biweekly + weekly rebalance")
    fprint(f"Question: Does 2% OTM improve the combined bull+pairs portfolio?")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. ATR
    atr_dict = compute_atr_series(high, low, close)

    # 3. Rebalance dates (biweekly for A/B/C, weekly for D)
    rebal_dates_bw = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ_BIWEEKLY).last().dropna().values
    )
    rebal_dates_wk = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ_WEEKLY).last().dropna().values
    )
    fprint(f"Biweekly rebalance dates: {len(rebal_dates_bw)} "
           f"({rebal_dates_bw[0].date()} to {rebal_dates_bw[-1].date()})")
    fprint(f"Weekly rebalance dates: {len(rebal_dates_wk)} "
           f"({rebal_dates_wk[0].date()} to {rebal_dates_wk[-1].date()})")

    spy_close = close["SPY"]

    # 4. Build LGBM rankings for biweekly
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (biweekly)")
    fprint("=" * 80)

    records_bw = build_feature_records(close, high, low, rebal_dates_bw, ALL_FEATURES)
    rankings_bw, imp_df = walk_forward_lgbm_rank(records_bw, ALL_FEATURES)

    if imp_df is not None:
        fprint("\nTop 5 features by importance:")
        for _, row in imp_df.head(5).iterrows():
            fprint(f"    {row['feature']}: {row['importance']:.1f}")

    if not rankings_bw:
        fprint("ERROR: No biweekly rankings generated. Exiting.")
        return

    # 5. Build LGBM rankings for weekly (for variant D)
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (weekly)")
    fprint("=" * 80)

    records_wk = build_feature_records(close, high, low, rebal_dates_wk, ALL_FEATURES)
    rankings_wk, _ = walk_forward_lgbm_rank(records_wk, ALL_FEATURES)

    if not rankings_wk:
        fprint("WARNING: No weekly rankings generated. Variant D may have no trades.")

    # 6. Define and run all variants
    fprint("\n" + "=" * 80)
    fprint("SIMULATING ALL VARIANTS")
    fprint("=" * 80)

    variant_configs = [
        ("A_atm_combo", "Bull VIX>20 + pairs VIX<20, ALL ATM (control)", simulate_variant_A, rankings_bw),
        ("B_otm2_combo", "Bull VIX>20 + pairs VIX<20, ALL 2% OTM", simulate_variant_B, rankings_bw),
        ("C_otm2_riskparity", "B + risk parity overlay low-VIX", simulate_variant_C, rankings_bw),
        ("D_otm2_weekly", "B but weekly rebalance", simulate_variant_D, rankings_wk),
    ]

    all_results = {}

    for name, desc, sim_func, rankings in variant_configs:
        fprint(f"\n--- {name}: {desc} ---")

        trades, final_eq, pct_invested = sim_func(rankings, close, atr_dict)

        fprint(f"  Trades: {len(trades)}, Final equity: ${final_eq:,.0f}, "
               f"% time invested: {pct_invested:.1f}%")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[name] = {
                "description": desc,
                "n_trades": len(trades) if trades else 0,
                "final_equity": round(final_eq, 2),
                "pct_invested": round(pct_invested, 1),
                "error": "insufficient_trades",
            }
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=name,
        )
        result.print_summary()

        # SPY beta and correlation
        beta, spy_corr = compute_spy_beta(trades, close)
        monthly_spy_corr = compute_monthly_spy_correlation(trades, close)
        fprint(f"  SPY beta: {beta:.3f}, trade correlation: {spy_corr:.3f}, "
               f"monthly correlation: {monthly_spy_corr:.3f}")

        # VIX regime stratified Sharpe
        # Weekly variant has ~52 periods/year vs 26 for biweekly
        ann_periods = 52 if name == "D_otm2_weekly" else 26
        regime_sharpe = vix_regime_stratified_sharpe(trades, annualize_periods=ann_periods)
        for label, data in regime_sharpe.items():
            fprint(f"    {label}: {data['n']} trades, Sharpe {data['sharpe']:.2f}, "
                   f"avg PnL ${data['avg_pnl']:.2f}")

        # Leg breakdown
        long_trades = [t for t in trades if t.get("leg") == "bull_call" or t.get("mode") in ("bull", "pair_long")]
        short_trades = [t for t in trades if t.get("leg") == "bear_put" or t.get("mode") == "pair_short"]
        rp_trades = [t for t in trades if t.get("mode") == "risk_parity"]

        long_pnl = sum(t["pnl"] for t in long_trades)
        short_pnl = sum(t["pnl"] for t in short_trades)
        rp_pnl = sum(t["pnl"] for t in rp_trades)

        fprint(f"  Long leg: {len(long_trades)} trades, PnL ${long_pnl:,.0f}")
        fprint(f"  Short leg: {len(short_trades)} trades, PnL ${short_pnl:,.0f}")
        if rp_trades:
            fprint(f"  Risk parity: {len(rp_trades)} trades, PnL ${rp_pnl:,.0f}")

        # Moneyness breakdown
        otm_trades = [t for t in trades if t.get("moneyness_pct", 0) > 0]
        atm_trades = [t for t in trades if t.get("moneyness_pct", 0) == 0 and t.get("mode") != "risk_parity"]
        if otm_trades:
            otm_wr = sum(1 for t in otm_trades if t["win"]) / len(otm_trades) * 100
            fprint(f"  OTM trades: {len(otm_trades)}, WR {otm_wr:.1f}%, avg PnL ${np.mean([t['pnl'] for t in otm_trades]):.2f}")
        if atm_trades:
            atm_wr = sum(1 for t in atm_trades if t["win"]) / len(atm_trades) * 100
            fprint(f"  ATM trades: {len(atm_trades)}, WR {atm_wr:.1f}%, avg PnL ${np.mean([t['pnl'] for t in atm_trades]):.2f}")

        total_return = (final_eq / CAP - 1) * 100

        rd = result.to_dict()
        rd.update({
            "description": desc,
            "pct_invested": round(pct_invested, 1),
            "spy_beta": round(beta, 3),
            "spy_corr": round(spy_corr, 3),
            "monthly_spy_corr": round(monthly_spy_corr, 3),
            "total_return_pct": round(total_return, 1),
            "regime_stratified_sharpe": regime_sharpe,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": round(long_pnl, 2),
            "short_pnl": round(short_pnl, 2),
        })
        if rp_trades:
            rd["rp_trades"] = len(rp_trades)
            rd["rp_pnl"] = round(rp_pnl, 2)

        all_results[name] = rd

    # ── Results summary table ──
    fprint("\n" + "=" * 80)
    fprint("RESULTS SUMMARY -- SORTED BY SHARPE")
    fprint("=" * 80)

    sortable = [(k, v) for k, v in all_results.items() if "error" not in v]
    sortable.sort(key=lambda x: x[1].get("sharpe", 0), reverse=True)

    fprint(f"\n{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>6} "
           f"{'MDD%':>7} {'CAGR%':>7} {'#Trades':>8} {'%Inv':>6} {'Beta':>6} {'Gates':>6}")
    fprint("-" * 105)

    for name, rd in sortable:
        fprint(f"{name:<25} {rd['sharpe']:>7.2f} {rd['sortino']:>8.2f} "
               f"{rd['profit_factor']:>6.2f} {rd['win_rate']*100:>6.1f} "
               f"{rd['max_dd']*100:>7.1f} {rd['cagr']*100:>7.1f} "
               f"{rd['n_trades']:>8} {rd['pct_invested']:>5.1f}% "
               f"{rd['spy_beta']:>6.3f} {rd['gates_passed']}/{rd['gates_total']}")

    errored = [(k, v) for k, v in all_results.items() if "error" in v]
    for name, rd in errored:
        fprint(f"{name:<25} {'--':>7} {'--':>8} {'--':>6} {'--':>6} "
               f"{'--':>7} {'--':>7} {rd['n_trades']:>8} {rd['pct_invested']:>5.1f}% "
               f"{'--':>6} {'--':>6}")

    # ── OTM vs ATM comparison ──
    fprint("\n" + "=" * 80)
    fprint("KEY COMPARISON: ATM vs 2% OTM")
    fprint("=" * 80)

    if "A_atm_combo" in all_results and "B_otm2_combo" in all_results:
        a = all_results["A_atm_combo"]
        b = all_results["B_otm2_combo"]
        if "error" not in a and "error" not in b:
            sharpe_diff = b["sharpe"] - a["sharpe"]
            sharpe_pct = (b["sharpe"] / a["sharpe"] - 1) * 100 if a["sharpe"] != 0 else 0
            fprint(f"  ATM combo (A): Sharpe {a['sharpe']:.2f}, Sortino {a['sortino']:.2f}, "
                   f"WR {a['win_rate']*100:.1f}%, PF {a['profit_factor']:.2f}")
            fprint(f"  OTM combo (B): Sharpe {b['sharpe']:.2f}, Sortino {b['sortino']:.2f}, "
                   f"WR {b['win_rate']*100:.1f}%, PF {b['profit_factor']:.2f}")
            fprint(f"  Sharpe improvement: {sharpe_diff:+.2f} ({sharpe_pct:+.1f}%)")
            fprint(f"  VERDICT: {'OTM IMPROVES combined portfolio' if sharpe_diff > 0.1 else 'OTM does NOT meaningfully improve' if sharpe_diff > -0.1 else 'OTM HURTS combined portfolio'}")

    # Best variant
    if sortable:
        best_name, best_rd = sortable[0]
        fprint(f"\n  BEST VARIANT: {best_name}")
        fprint(f"    Sharpe {best_rd['sharpe']:.2f}, Sortino {best_rd['sortino']:.2f}, "
               f"CAGR {best_rd['cagr']*100:.1f}%, MDD {best_rd['max_dd']*100:.1f}%, "
               f"%Inv {best_rd['pct_invested']:.1f}%, Gates {best_rd['gates_passed']}/{best_rd['gates_total']}")

    # ── Save JSON ──
    output = {
        "metadata": {
            "script": "combined_otm_portfolio_v1.py",
            "run_date": t0.strftime("%Y-%m-%d %H:%M:%S"),
            "capital": CAP,
            "dte": DTE,
            "spread_pct": SPREAD_PCT,
            "haircut": DEFAULT_HAIRCUT,
            "commission": COMMISSION_RT_SPREAD,
            "wf_train_periods": WF_TRAIN_PERIODS,
            "rebal_freq_biweekly": WF_REBAL_FREQ_BIWEEKLY,
            "rebal_freq_weekly": WF_REBAL_FREQ_WEEKLY,
            "data_range": f"{close.index[0].date()} to {close.index[-1].date()}",
            "n_sectors": len(SECTORS),
            "n_features": len(ALL_FEATURES),
            "moneyness_tested": [0.0, 2.0],
            "question": "Does 2% OTM improve the combined bull+pairs portfolio?",
        },
        "results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"otm_portfolio_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_variants": len(variant_configs),
                    "moneyness_values": "0,2",
                })

                for name, rd in all_results.items():
                    prefix = name.replace(" ", "_")
                    if "error" not in rd:
                        mlflow.log_metrics({
                            f"{prefix}_sharpe": rd.get("sharpe", 0),
                            f"{prefix}_sortino": rd.get("sortino", 0),
                            f"{prefix}_cagr": rd.get("cagr", 0),
                            f"{prefix}_maxdd": rd.get("max_dd", 0),
                            f"{prefix}_wr": rd.get("win_rate", 0),
                            f"{prefix}_pf": rd.get("profit_factor", 0),
                            f"{prefix}_pct_invested": rd.get("pct_invested", 0),
                            f"{prefix}_spy_beta": rd.get("spy_beta", 0),
                            f"{prefix}_gates": rd.get("gates_passed", 0),
                        })

                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
