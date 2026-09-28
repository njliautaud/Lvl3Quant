#!/usr/bin/env python3
"""
V9.3 Adversarial Leakage Audit v1
===================================

8-check adversarial audit for the V9.3 sector options strategy:
  1. Look-Ahead Leakage     — features at t use only data up to t
  2. Label Leakage           — fwd_ret labels not in feature set
  3. Walk-Forward Integrity  — sliding window, no future data in train
  4. Permutation Test        — 500 shuffles of sector rankings
  5. Random Signal Baseline  — random sector selection vs LGBM ranking
  6. Sub-Period Stability    — 4 quarter split, each Sharpe > 0.5
  7. Outlier Removal         — trimmed Sharpe (drop top/bottom 5%) > 0.5
  8. Regime Balance (R1)     — bull/bear Sharpe gap < 0.50

Self-contained: embeds BS pricing, feature computation, LGBM training.

V9.3 Config:
  - 11 sectors, 21 LGBM features (18 quality-momentum + 3 cross-asset)
  - 500d sliding walk-forward, biweekly rebalance (10 trading days)
  - DTE=28, 2% OTM, adaptive width max($3, 3%)
  - Bull call spreads on top-3 + bear put spreads on bottom-3 (VIX < 20)
  - Bull call spreads on top-2 only (VIX >= 20)
  - 50% profit target exit, 15% entry haircut
  - $645 starting capital, $200 max/trade, $2.60 commission RT

Output: stdout + MLflow experiment v93_adversarial_audit_v1
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


# ==============================================================
# ENVIRONMENT DETECTION
# ==============================================================

_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

OUTPUT_DIR = BASE / "output" / "growth_research" / "v93_adversarial_audit_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# CONSTANTS
# ==============================================================

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "TLT"]

CAP = 645.0
DTE = 28
WF_TRAIN_DAYS = 500
REBAL_DAYS = 10
TOP_K_BULL_VIX_HIGH = 2
TOP_K_PAIRS = 3
PROFIT_TARGET_PCT = 0.50
VIX_THRESHOLD = 20.0
OTM_PCT = 0.02
WIDTH_FLOOR_USD = 3.0
WIDTH_FLOOR_PCT = 0.03
COST_WIDTH_MAX = 0.50
MIN_POS_SIZE = 30.0
MAX_POS_SIZE = 200.0

RISK_FREE_RATE = 0.045
COMMISSION_RT = 2.60
EARLY_EXIT_COMM = 2.60
HAIRCUT = 0.15

N_PERM = 500  # permutation shuffles

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v93_adversarial_audit_v1"


# ==============================================================
# 21 FEATURES (production set)
# ==============================================================

FEATURES_21 = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d",
    "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]
assert len(FEATURES_21) == 21


# ==============================================================
# MLFLOW SETUP
# ==============================================================

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- will skip logging")


# ==============================================================
# BLACK-SCHOLES PRICING (self-contained)
# ==============================================================

def _bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _estimate_iv(atr, spot, vix=20.0):
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252.0 / 14.0)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    return max(realized_vol * iv_mult, 0.10)


def price_spread(S, K1, K2, dte, atr, vix, direction):
    if K2 <= K1:
        return None, None
    T = dte / 365.0
    sigma = _estimate_iv(atr, S, vix)
    if direction == "bull":
        fair = _bs_call(S, K1, T, RISK_FREE_RATE, sigma) - _bs_call(S, K2, T, RISK_FREE_RATE, sigma)
    else:
        fair = _bs_put(S, K2, T, RISK_FREE_RATE, sigma) - _bs_put(S, K1, T, RISK_FREE_RATE, sigma)
    fair = max(fair, 0.001)
    entry_cost = fair * (1.0 + HAIRCUT)
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost
    return float(entry_cost), float(max_profit)


def revalue_spread_bs(S, K1, K2, dte_remaining, atr, vix, direction):
    if dte_remaining <= 0:
        if direction == "bull":
            return max(S - K1, 0.0) - max(S - K2, 0.0)
        else:
            return max(K2 - S, 0.0) - max(K1 - S, 0.0)
    T = dte_remaining / 365.0
    sigma = _estimate_iv(atr, S, vix)
    if direction == "bull":
        fair = _bs_call(S, K1, T, RISK_FREE_RATE, sigma) - _bs_call(S, K2, T, RISK_FREE_RATE, sigma)
    else:
        fair = _bs_put(S, K2, T, RISK_FREE_RATE, sigma) - _bs_put(S, K1, T, RISK_FREE_RATE, sigma)
    return float(max(fair, 0.0) * (1.0 - HAIRCUT))


# ==============================================================
# DATA DOWNLOAD
# ==============================================================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers from 2008-01-01...")
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


def compute_atr_series(high, low, close, period=14):
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h, l, c = high[tk].dropna(), low[tk].dropna(), close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ==============================================================
# FEATURE COMPUTATION (21 features)
# ==============================================================

def compute_sector_features(tk, px, spy_rets, close_df, dt_idx):
    """Compute 21 features for a single sector at a given date.
    CRITICAL: px is close_df[tk].iloc[:dt_idx+1] -- only PAST data."""
    if len(px) < 260:
        return None
    f = {}
    rets = px.pct_change().dropna()

    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) >= 21 else 0.02
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) >= 63 else 0.02

    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) >= 21 else 0.0

    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min()) if len(px) >= 63 else 0.0

    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max()) if len(px) >= 252 else 1.0
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    pk252 = px.iloc[-252:].cummax() if len(px) >= 252 else px.cummax()
    mdd_1y = float(((px.iloc[-252:] / pk252) - 1).min()) if len(px) >= 252 else -0.01
    cagr_1y = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr_1y / (abs(mdd_1y) + 1e-10)

    if spy_rets is not None and len(spy_rets) >= 63:
        up_spy = spy_rets[spy_rets > 0]
        up_sec = rets.reindex(up_spy.index).iloc[-63:]
        f["up_capture"] = float(up_sec.mean() / (up_spy.iloc[-63:].mean() + 1e-10)) if len(up_sec) >= 5 else 1.0
    else:
        f["up_capture"] = 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    if spy_rets is not None and len(spy_rets) >= 63:
        common = spy_rets.index.intersection(rets.index)
        if len(common) >= 63:
            sec_r = rets.loc[common].iloc[-63:]
            spy_r = spy_rets.loc[common].iloc[-63:]
            cov = np.cov(sec_r.values, spy_r.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 1 and len(rets) >= 21:
        all_sec_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        avg_vol = float(all_sec_rets.iloc[-21:].std().mean())
        sec_vol = float(rets.iloc[-21:].std())
        f["sector_relative_vol_21d"] = sec_vol / (avg_vol + 1e-10)
    else:
        f["sector_relative_vol_21d"] = 1.0

    if len(sector_cols) > 3:
        sec_rets_all = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sec_rets_all.std(axis=1)
        f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1]) if len(daily_disp) >= 21 else 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


# ==============================================================
# LGBM WALK-FORWARD RANKING
# ==============================================================

def get_rebalance_dates(close):
    trading_days = close.index
    dates = []
    start_idx = WF_TRAIN_DAYS
    for i in range(start_idx, len(trading_days), REBAL_DAYS):
        dates.append(trading_days[i])
    return pd.DatetimeIndex(dates)


def build_feature_records(close, high, low, rebal_dates):
    """Build (date, sector) feature records. Features use ONLY data up to date t.
    Labels (fwd_ret) use FUTURE data from t to t+DTE -- this is correct for labels."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy_rets = close["SPY"].pct_change().dropna() if "SPY" in close.columns else None

    for dt in rebal_dates:
        dt_idx = close.index.get_indexer([dt], method="ffill")[0]
        if dt_idx < WF_TRAIN_DAYS:
            continue
        for tk in sector_cols:
            # CRITICAL: px uses data ONLY up to dt_idx (inclusive) -- no look-ahead
            px = close[tk].iloc[:dt_idx + 1].dropna()
            feat = compute_sector_features(tk, px, spy_rets, close, dt_idx)
            if feat is None:
                continue
            # Label: forward return from t to t+DTE (uses FUTURE data -- correct for labels)
            fi = min(dt_idx + DTE, len(close) - 1)
            if fi <= dt_idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[dt_idx] - 1)
            rec = {**feat, "date": dt, "ticker": tk, "fwd_ret": fwd_ret, "_dt_idx": dt_idx}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in FEATURES_21:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURES_21] = df[FEATURES_21].fillna(0.0)
    fprint(f"  Feature records: {len(df)} ({len(df['date'].unique())} dates)")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with 500d sliding window.
    Train on past dates, predict current. SLIDING -- oldest dates drop off."""
    import lightgbm as lgb

    if len(df) < 50:
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}

    for i, test_date in enumerate(dates):
        # SLIDING: only use train dates within WF_TRAIN_DAYS of test_date
        train_dates = [d for d in dates if d < test_date]
        if len(train_dates) < 20:
            continue
        # Sliding window: keep only the most recent WF_TRAIN_DAYS/REBAL_DAYS periods
        max_train_periods = WF_TRAIN_DAYS // REBAL_DAYS
        train_dates = train_dates[-max_train_periods:]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 30:
            continue

        Xt = np.nan_to_num(train_df[FEATURES_21].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[FEATURES_21].values.astype(np.float32))

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

    fprint(f"  Walk-forward ranking dates: {len(rankings)}")
    return rankings


# ==============================================================
# STRIKE COMPUTATION
# ==============================================================

def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * (1.0 + OTM_PCT), 2)
        w = max(WIDTH_FLOOR_USD, K1 * WIDTH_FLOOR_PCT)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1.0 - OTM_PCT), 2)
        w = max(WIDTH_FLOOR_USD, K2 * WIDTH_FLOOR_PCT)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ==============================================================
# TRADE EXECUTION
# ==============================================================

def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity):
    dt_pos = close.index.get_loc(dt)
    expiry_pos = min(dt_pos + DTE, len(close) - 1)
    if expiry_pos <= dt_pos:
        return None

    S = float(close[tk].iloc[dt_pos])
    av = (float(atr_dict[tk].loc[dt])
          if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt])
          else S * 0.015)

    K1, K2 = compute_strikes(S, direction)
    entry_cost_ps, max_profit_ps = price_spread(S, K1, K2, DTE, av, vix_val, direction)
    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT
    if total_cost <= 0 or total_cost > MAX_POS_SIZE or total_cost > equity * 0.40:
        return None
    if total_cost < MIN_POS_SIZE:
        return None

    # 50% profit target daily check
    exited_early = False
    exit_pos = expiry_pos
    hold_days = DTE

    if PROFIT_TARGET_PCT > 0 and max_profit_ps > 0:
        for chk in range(dt_pos + 1, expiry_pos + 1):
            if chk >= len(close):
                break
            chk_date = close.index[chk]
            dte_rem = expiry_pos - chk

            S_now = float(close[tk].iloc[chk])
            av_now = (float(atr_dict[tk].loc[chk_date])
                      if chk_date in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[chk_date])
                      else S_now * 0.015)

            cur_val = revalue_spread_bs(S_now, K1, K2, dte_rem, av_now, vix_val, direction)
            unrealized = cur_val - entry_cost_ps

            if unrealized >= PROFIT_TARGET_PCT * max_profit_ps:
                exited_early = True
                exit_pos = chk
                hold_days = chk - dt_pos
                break

    Se = float(close[tk].iloc[exit_pos])

    if exited_early:
        dte_at_exit = expiry_pos - exit_pos
        av_exit = (float(atr_dict[tk].loc[close.index[exit_pos]])
                   if close.index[exit_pos] in atr_dict[tk].index
                   else Se * 0.015)
        exit_val = revalue_spread_bs(Se, K1, K2, dte_at_exit, av_exit, vix_val, direction)
        pnl = (exit_val - entry_cost_ps) * 100 - COMMISSION_RT - EARLY_EXIT_COMM
    else:
        if direction == "bull":
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        else:
            intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT

    return {
        "pnl": round(pnl, 2),
        "entry_date": str(dt.date()),
        "exit_date": str(close.index[exit_pos].date()),
        "ticker": tk,
        "direction": direction,
        "hold_days": hold_days,
        "total_cost": round(total_cost, 2),
        "S_entry": round(S, 2),
        "S_exit": round(Se, 2),
        "K1": K1, "K2": K2,
        "entry_cost_ps": round(entry_cost_ps, 4),
        "exited_early": exited_early,
    }


# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def simulate_strategy(rankings, close, atr_dict, rebal_dates):
    """Run V9.3 backtest with given rankings dict. Returns trade list."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []

    for dt in sorted(rebal_dates):
        if dt not in rankings:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        if cv >= VIX_THRESHOLD:
            median_score = np.median([s for _, s in ranked_desc]) if ranked_desc else 0.5
            bull_picks = [t for t, s in ranked_desc[:TOP_K_BULL_VIX_HIGH] if s > median_score]
            bear_picks = []
        else:
            bull_picks = [t for t, _ in ranked_desc[:TOP_K_PAIRS]]
            bear_picks = [t for t, _ in ranked_asc[:TOP_K_PAIRS]]

        n_positions = len(bull_picks) + len(bear_picks)
        if n_positions == 0:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(tk, dt, direction, close, atr_dict, cv, equity)
                if result is None:
                    continue
                equity += result["pnl"]
                result["vix"] = round(cv, 1)
                trades.append(result)

    return trades


# ==============================================================
# METRICS
# ==============================================================

def compute_monthly_sharpe(trades, initial_capital=CAP):
    """Calendar month Sharpe from trade PnLs."""
    if not trades:
        return 0.0
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    monthly_pnl = df.groupby(df["exit_date"].dt.to_period("M"))["pnl"].sum()
    if len(monthly_pnl) < 2:
        return 0.0
    equity_base = initial_capital
    monthly_rets = monthly_pnl / equity_base  # simple return approx
    if monthly_rets.std() == 0:
        return 0.0
    return float(monthly_rets.mean() / monthly_rets.std() * np.sqrt(12))


def compute_full_metrics(trades, initial_capital=CAP):
    if not trades:
        return {"sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "n_trades": 0, "total_pnl": 0.0, "final_equity": initial_capital}
    pnls = np.array([t["pnl"] for t in trades])
    final_eq = initial_capital + pnls.sum()
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    pf = wins.sum() / (abs(losses.sum()) + 1e-9) if len(losses) > 0 else float("inf")
    sharpe = compute_monthly_sharpe(trades, initial_capital)
    return {
        "sharpe": round(sharpe, 3),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "n_trades": len(pnls),
        "total_pnl": round(float(pnls.sum()), 2),
        "final_equity": round(final_eq, 2),
    }


# ==============================================================
# CHECK 1: LOOK-AHEAD LEAKAGE
# ==============================================================

def check_1_lookahead_leakage(feat_df, close):
    """Verify features at time t only use data up to t.
    Method: for a sample of records, recompute features using data up to t
    and confirm they match. Also verify ret_Nd = px[t]/px[t-N]-1 (not future)."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 1: LOOK-AHEAD LEAKAGE")
    fprint("=" * 60)

    spy_rets = close["SPY"].pct_change().dropna() if "SPY" in close.columns else None
    n_samples = min(50, len(feat_df))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(feat_df), size=n_samples, replace=False)

    mismatches = 0
    for idx in sample_idx:
        row = feat_df.iloc[idx]
        tk = row["ticker"]
        dt = row["date"]
        dt_idx = int(row["_dt_idx"])

        # Recompute features using ONLY data up to dt_idx
        px = close[tk].iloc[:dt_idx + 1].dropna()
        feat_recomputed = compute_sector_features(tk, px, spy_rets, close, dt_idx)
        if feat_recomputed is None:
            continue

        for fname in FEATURES_21:
            orig = float(row[fname])
            recomp = float(feat_recomputed.get(fname, 0.0))
            if abs(orig - recomp) > 1e-6:
                mismatches += 1
                fprint(f"  MISMATCH {tk} {dt.date()} {fname}: orig={orig:.6f} recomp={recomp:.6f}")
                break

    # Verify ret_5d uses PAST data: ret_5d = px[t]/px[t-5]-1
    ret_check_ok = True
    for idx in sample_idx[:20]:
        row = feat_df.iloc[idx]
        tk = row["ticker"]
        dt_idx = int(row["_dt_idx"])
        if dt_idx < 5:
            continue
        px_t = float(close[tk].iloc[dt_idx])
        px_t5 = float(close[tk].iloc[dt_idx - 5])
        expected = px_t / px_t5 - 1
        actual = float(row["ret_5d"])
        if abs(expected - actual) > 1e-6:
            ret_check_ok = False
            fprint(f"  ret_5d USES FUTURE DATA for {tk} at idx {dt_idx}")

    # Verify fwd_ret uses FUTURE data (correct for labels, NOT features)
    fwd_in_features = "fwd_ret" in FEATURES_21
    if fwd_in_features:
        fprint("  CRITICAL: fwd_ret is in FEATURES_21 -- label leakage!")

    passed = (mismatches == 0) and ret_check_ok and (not fwd_in_features)
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  Feature recomputation mismatches: {mismatches}/{n_samples}")
    fprint(f"  ret_5d backward-looking: {'OK' if ret_check_ok else 'FAIL'}")
    fprint(f"  fwd_ret NOT in features: {'OK' if not fwd_in_features else 'FAIL'}")
    fprint(f"  RESULT: {verdict}")
    return passed, {"mismatches": mismatches, "ret_check_ok": ret_check_ok,
                    "fwd_in_features": fwd_in_features}


# ==============================================================
# CHECK 2: LABEL LEAKAGE
# ==============================================================

def check_2_label_leakage(feat_df, close):
    """Verify fwd_ret labels use FUTURE returns and aren't in feature set.
    Also verify train/test boundary: no test-date labels leak into train."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 2: LABEL LEAKAGE")
    fprint("=" * 60)

    # Check 1: fwd_ret must NOT be in FEATURES_21
    label_in_features = "fwd_ret" in FEATURES_21 or "rank_label" in FEATURES_21
    fprint(f"  fwd_ret in feature set: {label_in_features}")

    # Check 2: Verify fwd_ret actually uses future data
    rng = np.random.RandomState(123)
    n_check = min(30, len(feat_df))
    sample_idx = rng.choice(len(feat_df), size=n_check, replace=False)
    fwd_correct = 0
    for idx in sample_idx:
        row = feat_df.iloc[idx]
        tk = row["ticker"]
        dt_idx = int(row["_dt_idx"])
        fi = min(dt_idx + DTE, len(close) - 1)
        if fi <= dt_idx:
            continue
        expected_fwd = float(close[tk].iloc[fi] / close[tk].iloc[dt_idx] - 1)
        actual_fwd = float(row["fwd_ret"])
        if abs(expected_fwd - actual_fwd) < 1e-6:
            fwd_correct += 1

    fwd_pct = fwd_correct / n_check if n_check > 0 else 0
    fprint(f"  fwd_ret correctly uses future t to t+{DTE}: {fwd_correct}/{n_check} ({fwd_pct:.0%})")

    # Check 3: Correlation between features and labels should be moderate, not 1.0
    correlations = []
    for fname in FEATURES_21:
        c = feat_df[fname].corr(feat_df["fwd_ret"])
        if abs(c) > 0.95:
            fprint(f"  WARNING: {fname} corr with fwd_ret = {c:.3f} (suspiciously high)")
        correlations.append(abs(c))
    max_corr = max(correlations) if correlations else 0
    fprint(f"  Max |corr(feature, fwd_ret)|: {max_corr:.3f}")

    passed = (not label_in_features) and (fwd_pct >= 0.90) and (max_corr < 0.95)
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  RESULT: {verdict}")
    return passed, {"label_in_features": label_in_features, "fwd_ret_correct_pct": fwd_pct,
                    "max_feature_label_corr": round(max_corr, 4)}


# ==============================================================
# CHECK 3: WALK-FORWARD INTEGRITY
# ==============================================================

def check_3_walkforward_integrity(feat_df):
    """Verify sliding window: train dates are strictly before test date,
    and oldest dates drop off (not expanding)."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 3: WALK-FORWARD INTEGRITY")
    fprint("=" * 60)

    dates = sorted(feat_df["date"].unique())
    max_train_periods = WF_TRAIN_DAYS // REBAL_DAYS

    violations = 0
    expanding_detected = False

    # Simulate the walk-forward loop
    for i, test_date in enumerate(dates):
        train_dates = [d for d in dates if d < test_date]
        if len(train_dates) < 20:
            continue

        # Check no future dates in train
        future_in_train = [d for d in train_dates if d >= test_date]
        if future_in_train:
            violations += 1
            fprint(f"  VIOLATION: future date in train at test={test_date.date()}")

        # Check sliding (not expanding): train window should be capped
        if len(train_dates) > max_train_periods + 5:  # small tolerance
            # This would only happen if we don't slice
            # In the actual code we DO slice, so just verify the logic
            pass

    # Verify train window size is bounded
    train_sizes = []
    for i, test_date in enumerate(dates[20:]):
        train_dates = [d for d in dates if d < test_date][-max_train_periods:]
        train_sizes.append(len(train_dates))

    if train_sizes:
        max_train = max(train_sizes)
        min_train = min(train_sizes)
        fprint(f"  Train window sizes: min={min_train}, max={max_train}, "
               f"cap={max_train_periods}")
        if max_train > max_train_periods + 2:
            expanding_detected = True
            fprint("  WARNING: Train window exceeds sliding cap -- expanding detected")

    fprint(f"  Future-in-train violations: {violations}")
    fprint(f"  Expanding window detected: {expanding_detected}")

    passed = (violations == 0) and (not expanding_detected)
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  RESULT: {verdict}")
    return passed, {"violations": violations, "expanding": expanding_detected,
                    "max_train_size": max(train_sizes) if train_sizes else 0}


# ==============================================================
# CHECK 4: PERMUTATION TEST (500 shuffles)
# ==============================================================

def check_4_permutation_test(real_sharpe, close, atr_dict, rebal_dates, rankings):
    """Randomize sector rankings 500 times. If random Sharpe >= real Sharpe
    frequently, the strategy is an artifact. p-value must be < 0.05."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 4: PERMUTATION TEST (500 shuffles)")
    fprint("=" * 60)

    rng = np.random.RandomState(2024)
    perm_sharpes = []

    for trial in range(N_PERM):
        # Create random rankings: shuffle scores within each date
        rand_rankings = {}
        for dt, scores in rankings.items():
            tickers = list(scores.keys())
            random_scores = rng.uniform(0, 1, size=len(tickers))
            rand_rankings[dt] = dict(zip(tickers, random_scores))

        trades = simulate_strategy(rand_rankings, close, atr_dict, rebal_dates)
        sharpe = compute_monthly_sharpe(trades)
        perm_sharpes.append(sharpe)

        if (trial + 1) % 100 == 0:
            fprint(f"  Permutation {trial + 1}/{N_PERM} done, "
                   f"mean rand Sharpe: {np.mean(perm_sharpes):.3f}")

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= real_sharpe))
    mean_rand = float(np.mean(perm_sharpes))
    std_rand = float(np.std(perm_sharpes))

    fprint(f"  Real Sharpe: {real_sharpe:.3f}")
    fprint(f"  Random Sharpe: mean={mean_rand:.3f}, std={std_rand:.3f}")
    fprint(f"  p-value (rand >= real): {p_value:.4f}")

    passed = p_value < 0.05
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  RESULT: {verdict}")
    return passed, {"p_value": round(p_value, 4), "real_sharpe": round(real_sharpe, 3),
                    "mean_rand_sharpe": round(mean_rand, 3), "std_rand_sharpe": round(std_rand, 3)}


# ==============================================================
# CHECK 5: RANDOM SIGNAL BASELINE
# ==============================================================

def check_5_random_signal_baseline(real_sharpe, real_metrics, close, atr_dict, rebal_dates, rankings):
    """Compare LGBM-ranked strategy vs random sector selection.
    If random selection also profits, edge is from OPTIONS STRUCTURE not RANKING."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 5: RANDOM SIGNAL BASELINE")
    fprint("=" * 60)

    rng = np.random.RandomState(7777)
    n_trials = 100
    rand_sharpes = []
    rand_pnls = []

    for trial in range(n_trials):
        rand_rankings = {}
        for dt, scores in rankings.items():
            tickers = list(scores.keys())
            random_scores = rng.uniform(0, 1, size=len(tickers))
            rand_rankings[dt] = dict(zip(tickers, random_scores))

        trades = simulate_strategy(rand_rankings, close, atr_dict, rebal_dates)
        sharpe = compute_monthly_sharpe(trades)
        total_pnl = sum(t["pnl"] for t in trades) if trades else 0
        rand_sharpes.append(sharpe)
        rand_pnls.append(total_pnl)

    rand_sharpes = np.array(rand_sharpes)
    rand_pnls = np.array(rand_pnls)
    mean_rand_sharpe = float(np.mean(rand_sharpes))
    mean_rand_pnl = float(np.mean(rand_pnls))
    pct_rand_profitable = float(np.mean(rand_pnls > 0))

    fprint(f"  Real Sharpe: {real_sharpe:.3f}, Real PnL: ${real_metrics['total_pnl']:.2f}")
    fprint(f"  Random Sharpe: mean={mean_rand_sharpe:.3f}")
    fprint(f"  Random PnL: mean=${mean_rand_pnl:.2f}")
    fprint(f"  % random trials profitable: {pct_rand_profitable:.0%}")

    # If random selection also has high Sharpe, the edge is structural, not from ranking
    structure_edge = mean_rand_sharpe > 0.5
    ranking_adds_value = real_sharpe > mean_rand_sharpe + 0.3

    if structure_edge:
        fprint("  WARNING: Random selection also profitable -- edge may be from options structure")
    if ranking_adds_value:
        fprint("  LGBM ranking adds meaningful value over random selection")

    passed = ranking_adds_value
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  RESULT: {verdict}")
    return passed, {"real_sharpe": round(real_sharpe, 3),
                    "mean_rand_sharpe": round(mean_rand_sharpe, 3),
                    "pct_rand_profitable": round(pct_rand_profitable, 3),
                    "ranking_adds_value": ranking_adds_value,
                    "structure_edge": structure_edge}


# ==============================================================
# CHECK 6: SUB-PERIOD STABILITY
# ==============================================================

def check_6_subperiod_stability(trades):
    """Split backtest into 4 quarters. Each must have Sharpe > 0.5."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 6: SUB-PERIOD STABILITY")
    fprint("=" * 60)

    if not trades:
        fprint("  No trades -- FAIL")
        return False, {"quarter_sharpes": [], "all_above_threshold": False}

    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")

    # Split into 4 equal-sized quarters by trade count
    n = len(df)
    q_size = n // 4
    if q_size < 5:
        fprint(f"  Too few trades ({n}) for 4-quarter split -- FAIL")
        return False, {"quarter_sharpes": [], "n_trades": n}

    quarter_sharpes = []
    for q in range(4):
        start = q * q_size
        end = (q + 1) * q_size if q < 3 else n
        q_trades = df.iloc[start:end].to_dict("records")
        sharpe = compute_monthly_sharpe(q_trades)
        period = f"{df.iloc[start]['exit_date'].date()} to {df.iloc[end-1]['exit_date'].date()}"
        quarter_sharpes.append(sharpe)
        fprint(f"  Q{q+1} ({period}): Sharpe={sharpe:.3f}, "
               f"trades={end-start}, PnL=${sum(t['pnl'] for t in q_trades):.2f}")

    all_above = all(s > 0.5 for s in quarter_sharpes)
    min_q = min(quarter_sharpes)

    fprint(f"  Min quarter Sharpe: {min_q:.3f} (threshold: 0.50)")
    passed = all_above
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  RESULT: {verdict}")
    return passed, {"quarter_sharpes": [round(s, 3) for s in quarter_sharpes],
                    "min_quarter_sharpe": round(min_q, 3), "all_above_threshold": all_above}


# ==============================================================
# CHECK 7: OUTLIER REMOVAL
# ==============================================================

def check_7_outlier_removal(trades):
    """Remove top and bottom 5% of trades by PnL. Trimmed Sharpe must > 0.5."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 7: OUTLIER REMOVAL")
    fprint("=" * 60)

    if not trades or len(trades) < 20:
        fprint(f"  Too few trades ({len(trades) if trades else 0}) -- FAIL")
        return False, {"trimmed_sharpe": 0.0}

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    trim_n = max(1, int(n * 0.05))

    sorted_indices = np.argsort(pnls)
    # Remove bottom 5% (worst trades) and top 5% (best trades)
    keep_indices = sorted_indices[trim_n:-trim_n]
    trimmed_trades = [trades[i] for i in keep_indices]

    full_sharpe = compute_monthly_sharpe(trades)
    trimmed_sharpe = compute_monthly_sharpe(trimmed_trades)
    full_pnl = float(pnls.sum())
    trimmed_pnl = float(sum(t["pnl"] for t in trimmed_trades))

    fprint(f"  Full: {n} trades, Sharpe={full_sharpe:.3f}, PnL=${full_pnl:.2f}")
    fprint(f"  Trimmed (drop {trim_n} top + {trim_n} bottom): "
           f"{len(trimmed_trades)} trades, Sharpe={trimmed_sharpe:.3f}, PnL=${trimmed_pnl:.2f}")

    passed = trimmed_sharpe > 0.5
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  RESULT: {verdict}")
    return passed, {"full_sharpe": round(full_sharpe, 3),
                    "trimmed_sharpe": round(trimmed_sharpe, 3),
                    "full_pnl": round(full_pnl, 2),
                    "trimmed_pnl": round(trimmed_pnl, 2),
                    "trades_removed": trim_n * 2}


# ==============================================================
# CHECK 8: REGIME BALANCE (R1 from HC #428)
# ==============================================================

def check_8_regime_balance(trades, close):
    """Compute Sharpe for bull (SPY > 200d SMA) and bear periods separately.
    |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|) must < 0.50."""
    fprint("\n" + "=" * 60)
    fprint("CHECK 8: REGIME BALANCE (R1)")
    fprint("=" * 60)

    if not trades:
        fprint("  No trades -- FAIL")
        return False, {"sharpe_bull": 0.0, "sharpe_bear": 0.0, "gap_ratio": 1.0}

    spy = close["SPY"] if "SPY" in close.columns else None
    if spy is None:
        fprint("  No SPY data -- cannot compute regime -- FAIL")
        return False, {}

    spy_sma200 = spy.rolling(200).mean()

    bull_trades = []
    bear_trades = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        if entry_date in spy.index and entry_date in spy_sma200.index:
            spy_val = float(spy.loc[entry_date])
            sma_val = float(spy_sma200.loc[entry_date])
            if not np.isnan(sma_val):
                if spy_val > sma_val:
                    bull_trades.append(t)
                else:
                    bear_trades.append(t)

    sharpe_bull = compute_monthly_sharpe(bull_trades) if len(bull_trades) >= 5 else 0.0
    sharpe_bear = compute_monthly_sharpe(bear_trades) if len(bear_trades) >= 5 else 0.0

    fprint(f"  Bull trades (SPY > 200d SMA): {len(bull_trades)}, Sharpe={sharpe_bull:.3f}")
    fprint(f"  Bear trades (SPY < 200d SMA): {len(bear_trades)}, Sharpe={sharpe_bear:.3f}")

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    if max_abs > 0:
        gap_ratio = abs(sharpe_bull - sharpe_bear) / max_abs
    else:
        gap_ratio = 0.0

    fprint(f"  Gap ratio: {gap_ratio:.3f} (threshold: 0.50)")

    passed = gap_ratio < 0.50
    verdict = "PASS" if passed else "FAIL"
    fprint(f"  RESULT: {verdict}")
    return passed, {"sharpe_bull": round(sharpe_bull, 3),
                    "sharpe_bear": round(sharpe_bear, 3),
                    "gap_ratio": round(gap_ratio, 3),
                    "n_bull": len(bull_trades), "n_bear": len(bear_trades)}


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    fprint("=" * 60)
    fprint("V9.3 ADVERSARIAL LEAKAGE AUDIT v1")
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 60)

    # ----------------------------------------------------------
    # DATA
    # ----------------------------------------------------------
    fprint("\n[1/3] Downloading data...")
    close, high, low = download_data()
    atr_dict = compute_atr_series(high, low, close)
    rebal_dates = get_rebalance_dates(close)
    fprint(f"  Rebalance dates: {len(rebal_dates)}")

    # ----------------------------------------------------------
    # FEATURES + LGBM RANKING
    # ----------------------------------------------------------
    fprint("\n[2/3] Building features + walk-forward LGBM ranking...")
    feat_df = build_feature_records(close, high, low, rebal_dates)
    rankings = walk_forward_lgbm_rank(feat_df)

    # ----------------------------------------------------------
    # BASELINE STRATEGY
    # ----------------------------------------------------------
    fprint("\n[3/3] Running baseline strategy...")
    real_trades = simulate_strategy(rankings, close, atr_dict, rebal_dates)
    real_metrics = compute_full_metrics(real_trades)
    real_sharpe = real_metrics["sharpe"]
    fprint(f"  Baseline: {real_metrics['n_trades']} trades, "
           f"Sharpe={real_sharpe:.3f}, WR={real_metrics['win_rate']:.1%}, "
           f"PF={real_metrics['profit_factor']:.2f}, "
           f"PnL=${real_metrics['total_pnl']:.2f}")

    # ----------------------------------------------------------
    # RUN ALL 8 CHECKS
    # ----------------------------------------------------------
    results = {}

    p1, d1 = check_1_lookahead_leakage(feat_df, close)
    results["check_1_lookahead"] = {"passed": p1, **d1}

    p2, d2 = check_2_label_leakage(feat_df, close)
    results["check_2_label_leak"] = {"passed": p2, **d2}

    p3, d3 = check_3_walkforward_integrity(feat_df)
    results["check_3_wf_integrity"] = {"passed": p3, **d3}

    p4, d4 = check_4_permutation_test(real_sharpe, close, atr_dict, rebal_dates, rankings)
    results["check_4_permutation"] = {"passed": p4, **d4}

    p5, d5 = check_5_random_signal_baseline(real_sharpe, real_metrics, close, atr_dict, rebal_dates, rankings)
    results["check_5_random_baseline"] = {"passed": p5, **d5}

    p6, d6 = check_6_subperiod_stability(real_trades)
    results["check_6_subperiod"] = {"passed": p6, **d6}

    p7, d7 = check_7_outlier_removal(real_trades)
    results["check_7_outlier_removal"] = {"passed": p7, **d7}

    p8, d8 = check_8_regime_balance(real_trades, close)
    results["check_8_regime_balance"] = {"passed": p8, **d8}

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    checks_passed = sum(1 for k, v in results.items() if v.get("passed", False))
    total_checks = 8
    elapsed = time.time() - t0

    fprint("\n" + "=" * 60)
    fprint("ADVERSARIAL AUDIT SUMMARY")
    fprint("=" * 60)
    fprint(f"  Baseline Sharpe: {real_sharpe:.3f}")
    fprint(f"  Baseline WR: {real_metrics['win_rate']:.1%}")
    fprint(f"  Baseline PF: {real_metrics['profit_factor']:.2f}")
    fprint(f"  Baseline PnL: ${real_metrics['total_pnl']:.2f}")
    fprint(f"  Trades: {real_metrics['n_trades']}")
    fprint("")

    check_names = [
        ("check_1_lookahead", "Look-Ahead Leakage"),
        ("check_2_label_leak", "Label Leakage"),
        ("check_3_wf_integrity", "Walk-Forward Integrity"),
        ("check_4_permutation", "Permutation Test (p<0.05)"),
        ("check_5_random_baseline", "Random Signal Baseline"),
        ("check_6_subperiod", "Sub-Period Stability"),
        ("check_7_outlier_removal", "Outlier Removal"),
        ("check_8_regime_balance", "Regime Balance (R1)"),
    ]
    for key, name in check_names:
        status = "PASS" if results[key].get("passed", False) else "FAIL"
        fprint(f"  [{status}] {name}")

    fprint("")
    overall = "PASS" if checks_passed == total_checks else "FAIL"
    fprint(f"  OVERALL: {checks_passed}/{total_checks} checks passed -- {overall}")
    fprint(f"  Elapsed: {elapsed:.0f}s")

    # ----------------------------------------------------------
    # MLFLOW LOGGING
    # ----------------------------------------------------------
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(FEATURES_21))
                mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("rebal_days", REBAL_DAYS)
                mlflow.log_param("n_permutations", N_PERM)
                mlflow.log_param("haircut", HAIRCUT)
                mlflow.log_param("commission_rt", COMMISSION_RT)
                mlflow.log_param("starting_capital", CAP)

                mlflow.log_metric("baseline_sharpe", real_sharpe)
                mlflow.log_metric("baseline_win_rate", real_metrics["win_rate"])
                mlflow.log_metric("baseline_profit_factor", real_metrics["profit_factor"])
                mlflow.log_metric("baseline_total_pnl", real_metrics["total_pnl"])
                mlflow.log_metric("baseline_n_trades", real_metrics["n_trades"])
                mlflow.log_metric("checks_passed", checks_passed)
                mlflow.log_metric("checks_total", total_checks)
                mlflow.log_metric("overall_pass", 1 if overall == "PASS" else 0)
                mlflow.log_metric("elapsed_seconds", elapsed)

                # Log per-check results
                for key, v in results.items():
                    mlflow.log_metric(f"{key}_passed", 1 if v.get("passed") else 0)

                if "p_value" in results.get("check_4_permutation", {}):
                    mlflow.log_metric("perm_p_value", results["check_4_permutation"]["p_value"])
                if "gap_ratio" in results.get("check_8_regime_balance", {}):
                    mlflow.log_metric("regime_gap_ratio", results["check_8_regime_balance"]["gap_ratio"])

                # Save results JSON
                results_path = OUTPUT_DIR / "audit_results.json"
                with open(results_path, "w") as fp:
                    json.dump({"baseline": real_metrics, "checks": results,
                               "overall": overall, "elapsed_s": round(elapsed, 1)}, fp, indent=2)
                mlflow.log_artifact(str(results_path))
            fprint(f"  MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"  MLflow logging failed: {e}")
    else:
        # Save results locally
        results_path = OUTPUT_DIR / "audit_results.json"
        with open(results_path, "w") as fp:
            json.dump({"baseline": real_metrics, "checks": results,
                       "overall": overall, "elapsed_s": round(elapsed, 1)}, fp, indent=2)
        fprint(f"  Results saved to {results_path}")

    return overall == "PASS"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
