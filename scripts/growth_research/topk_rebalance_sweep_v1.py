#!/usr/bin/env python3
"""
Top-K × Rebalance Frequency Sweep v1
======================================

Sweeps two parameters from the production v4 honest test:
  Parameter 1: Top-K sectors to trade (1, 2, 3, 4, 5, 6)
  Parameter 2: Rebalance interval in trading days (5, 10, 15, 21, 42)

Total: 6 × 5 = 30 combinations

Core logic copied from production_v4_honest_test.py (which uses
research.tools.options_pricer and adversarial_validator, not available on
Neptune). All pricing and validation functions are embedded here.

Honest pricing: 15% entry haircut, no exit haircut, $2.60/spread commission,
$645 start, $200 max/trade, hold to expiry, intrinsic only.

Uses: Production v4c variant (best from prod v4 test) — regime>0.4/0.2 +
21 features (18 legacy + 3 cross-asset) + bull+bear combined + VIX 25-30 skip.
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════
# EMBEDDED OPTIONS PRICER (from research/tools/options_pricer.py)
# ══════════════════════════════════════════════════════════════

RISK_FREE_RATE = 0.045
DEFAULT_HAIRCUT = 0.15
COMMISSION_PER_LEG = 0.65
COMMISSION_RT_SPREAD = 2.60


def bs_call_price(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def estimate_iv(atr, spot, vix=20.0, atr_period=14):
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    sigma = realized_vol * iv_mult
    return max(sigma, 0.10)


def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0,
                           haircut=DEFAULT_HAIRCUT, r=RISK_FREE_RATE, sigma=None):
    if K2 <= K1:
        raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")
    T = dte / 365.0
    if sigma is None:
        sigma = estimate_iv(atr, S, vix)
    fair_value = bs_call_price(S, K1, T, r, sigma) - bs_call_price(S, K2, T, r, sigma)
    fair_value = max(fair_value, 0.001)
    entry_cost = fair_value * (1.0 + haircut)
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost
    return float(entry_cost), float(max_profit)


def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0,
                          haircut=DEFAULT_HAIRCUT, r=RISK_FREE_RATE, sigma=None):
    if K2 <= K1:
        raise ValueError(f"K2 ({K2}) must be > K1 ({K1})")
    T = dte / 365.0
    if sigma is None:
        sigma = estimate_iv(atr, S, vix)
    fair_value = bs_put_price(S, K2, T, r, sigma) - bs_put_price(S, K1, T, r, sigma)
    fair_value = max(fair_value, 0.001)
    entry_cost = fair_value * (1.0 + haircut)
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost
    return float(entry_cost), float(max_profit)


# ══════════════════════════════════════════════════════════════
# EMBEDDED ADVERSARIAL VALIDATOR (from research/tools/adversarial_validator.py)
# ══════════════════════════════════════════════════════════════

def compute_honest_sharpe(equity_series, annualization=12.0):
    if len(equity_series) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)
    monthly_equity = equity_series.resample("ME").last().dropna()
    if len(monthly_equity) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)
    monthly_returns = monthly_equity.pct_change().dropna()
    if len(monthly_returns) < 2 or monthly_returns.std() == 0:
        return 0.0, 0.0, monthly_returns
    mean_ret = monthly_returns.mean()
    std_ret = monthly_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(annualization)
    downside = monthly_returns[monthly_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (mean_ret / downside.std()) * np.sqrt(annualization)
    else:
        sortino = sharpe * 1.5
    return float(sharpe), float(sortino), monthly_returns


def compute_metrics(trades, initial_capital):
    if not trades:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": -1.0,
                "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0,
                "final_equity": initial_capital, "monthly_returns": pd.Series(dtype=float)}
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")
    pnls = df["pnl"].values
    equity_values = [initial_capital]
    for pnl in pnls:
        equity_values.append(equity_values[-1] + pnl)
    dates = [df["entry_date"].iloc[0] - pd.Timedelta(days=1)]
    dates.extend(df["exit_date"].tolist())
    equity_series = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
    equity_series = equity_series.groupby(equity_series.index).last()
    sharpe, sortino, monthly_rets = compute_honest_sharpe(equity_series)
    final_eq = equity_values[-1]
    total_days = (dates[-1] - dates[0]).days
    years = max(total_days / 365.25, 0.1)
    if final_eq > 0 and initial_capital > 0:
        cagr = (final_eq / initial_capital) ** (1 / years) - 1
    else:
        cagr = -1.0
    eq_arr = np.array(equity_values)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")
    return {"sharpe": sharpe, "sortino": sortino, "cagr": cagr, "max_dd": max_dd,
            "win_rate": win_rate, "profit_factor": profit_factor, "n_trades": len(pnls),
            "final_equity": final_eq, "monthly_returns": monthly_rets}


def gate_sign_flip(trades, initial_capital, real_sharpe, n_perms=2000):
    pnls = np.array([t["pnl"] for t in trades])
    dates = pd.to_datetime([t["exit_date"] for t in trades])
    beat_count = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnls))
        flipped_pnls = pnls * signs
        equity_vals = [initial_capital]
        for p in flipped_pnls:
            equity_vals.append(equity_vals[-1] + p)
        eq_series = pd.Series(equity_vals,
            index=pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)] + list(dates)))
        eq_series = eq_series.groupby(eq_series.index).last()
        perm_sharpe, _, _ = compute_honest_sharpe(eq_series)
        if perm_sharpe >= real_sharpe:
            beat_count += 1
    return beat_count / n_perms


def gate_regime_balance(trades, spy_prices):
    bull_pnls, bear_pnls = [], []
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    if spy_prices is not None and len(spy_prices) > 0:
        spy_prices = spy_prices.sort_index()
        for _, row in df.iterrows():
            ep = spy_prices[spy_prices.index <= row["entry_date"]]
            xp = spy_prices[spy_prices.index <= row["exit_date"]]
            if len(ep) == 0 or len(xp) == 0:
                continue
            if float(xp.iloc[-1]) >= float(ep.iloc[-1]):
                bull_pnls.append(row["pnl"])
            else:
                bear_pnls.append(row["pnl"])
    if len(bull_pnls) < 5 or len(bear_pnls) < 5:
        return True, 0.0
    bull_wr = np.mean([1 if p > 0 else 0 for p in bull_pnls])
    bear_wr = np.mean([1 if p > 0 else 0 for p in bear_pnls])
    wr_gap = abs(bull_wr - bear_wr)
    return wr_gap < 0.50, wr_gap


def gate_sub_period(trades):
    df = pd.DataFrame(trades).sort_values("exit_date", key=pd.to_datetime)
    mid = len(df) // 2
    h1 = df.iloc[:mid]["pnl"].sum()
    h2 = df.iloc[mid:]["pnl"].sum()
    return h1 > 0 and h2 > 0


def gate_outlier_removal(monthly_returns):
    if len(monthly_returns) < 3:
        return False
    best_idx = monthly_returns.idxmax()
    trimmed = monthly_returns.drop(best_idx)
    if len(trimmed) < 2 or trimmed.std() == 0:
        return False
    return trimmed.sum() > 0 and (trimmed.mean() / trimmed.std()) * np.sqrt(12) > 0


def gate_yearly_consistency(trades, threshold=0.60):
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year
    yearly = df.groupby("year")["pnl"].sum()
    if len(yearly) == 0:
        return False, 0.0
    pct = (yearly > 0).sum() / len(yearly)
    return pct >= threshold, pct


def run_5_gate(trades, initial_capital, spy_prices, n_perms=500):
    """Run 5-gate validation, return (gates_passed, gates_total, details dict)."""
    metrics = compute_metrics(trades, initial_capital)
    gates_passed = 0
    details = {}

    # Gate 1: sign-flip
    p_val = gate_sign_flip(trades, initial_capital, metrics["sharpe"], n_perms)
    g1 = p_val < 0.05
    gates_passed += int(g1)
    details["g1_signflip_pval"] = round(p_val, 4)

    # Gate 2: regime balance
    g2, wr_gap = gate_regime_balance(trades, spy_prices)
    gates_passed += int(g2)
    details["g2_regime_wr_gap"] = round(wr_gap, 4)

    # Gate 3: sub-period
    g3 = gate_sub_period(trades)
    gates_passed += int(g3)

    # Gate 4: outlier removal
    g4 = gate_outlier_removal(metrics["monthly_returns"])
    gates_passed += int(g4)

    # Gate 5: yearly consistency
    g5, yr_pct = gate_yearly_consistency(trades)
    gates_passed += int(g5)
    details["g5_yearly_pct"] = round(yr_pct, 4)

    return gates_passed, 5, details


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

BASE = Path("/home/nick/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "topk_rebalance_sweep_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2
WF_TRAIN_PERIODS = 12

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# Sweep parameters
TOP_K_VALUES = [1, 2, 3, 4, 5, 6]
REBAL_DAY_VALUES = [5, 10, 15, 21, 42]

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "topk_rebalance_sweep_v1"

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
# DATA (same as production v4)
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
            raise ValueError(f"Missing: {t}")
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


def load_regime_predictions():
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found, using VIX proxy")
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions: {len(regime_series)} days "
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
# FEATURES (same as production v4)
# ══════════════════════════════════════════════════════════════

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

V4_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET


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


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
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
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0
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
# WALK-FORWARD LGBM (parameterized rebalance frequency)
# ══════════════════════════════════════════════════════════════

def build_rebal_dates_by_days(close_index, rebal_days):
    """Build rebalance dates every N trading days."""
    trading_days = close_index.sort_values()
    rebal_dates = []
    for i in range(0, len(trading_days), rebal_days):
        rebal_dates.append(trading_days[i])
    return pd.DatetimeIndex(rebal_dates)


def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for all sectors on all rebal dates.
    Uses bull+bear combined mode (production v4c)."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        rscore = get_regime_score_at(regime_series, dt)
        if rscore > REGIME_BULL_THRESHOLD:
            direction = "bull"
        elif rscore < REGIME_BEAR_THRESHOLD:
            direction = "bear"
        else:
            continue  # gray zone

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]
            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue
            cross_asset = {}
            for col in feature_cols:
                if col in VALIDATED_CROSS_ASSET:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "direction": direction}
            records.append(rec)

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def walk_forward_lgbm_rank(df, feature_cols, name=""):
    """Walk-forward LGBM ranking."""
    import lightgbm as lgb

    if len(df) < 100:
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
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            direction = test_df["direction"].iloc[0] if "direction" in test_df.columns else "bull"
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "direction": direction,
            }
        except Exception:
            continue

    return rankings


# ══════════════════════════════════════════════════════════════
# ATR
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
# TRADE SIMULATION (parameterized top_k)
# ══════════════════════════════════════════════════════════════

def simulate_trades(rankings, close, high, low, regime_series, atr_dict,
                    top_k=3, skip_vix_25_30=True):
    """Simulate using production v4c rules with parameterized top_k."""
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

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= top_k:
                continue
            S = float(close[tk].loc[dt])
            di = close.index.get_loc(dt)
            ei = min(di + DTE, len(close) - 1)
            if ei <= di:
                continue
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                if direction == "bull":
                    entry_cost_ps, max_profit_ps = price_bull_call_spread(
                        S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv)
                else:
                    entry_cost_ps, max_profit_ps = price_bear_put_spread(
                        S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv)
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

            exit_value_ps = intrinsic
            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            spy_regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "direction": direction,
                "vix": round(cv, 1),
                "win": pnl > 0,
            })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"TOP-K x REBALANCE FREQUENCY SWEEP — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic only | Bull+Bear combined | VIX 25-30 skip")
    fprint(f"Top-K values: {TOP_K_VALUES}")
    fprint(f"Rebalance intervals (trading days): {REBAL_DAY_VALUES}")
    fprint(f"Total combinations: {len(TOP_K_VALUES) * len(REBAL_DAY_VALUES)}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime
    regime_series = load_regime_predictions()

    # 3. ATR
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # 4. Pre-build feature records for each rebalance frequency
    # The LGBM model needs to be retrained for each rebalance freq since
    # different rebal dates = different training windows
    fprint("\n" + "=" * 80)
    fprint("BUILDING FEATURE RECORDS FOR EACH REBALANCE FREQUENCY")
    fprint("=" * 80)

    records_by_rebal = {}
    rankings_by_rebal = {}

    for rebal_days in REBAL_DAY_VALUES:
        fprint(f"\n--- Rebalance every {rebal_days} trading days ---")
        rebal_dates = build_rebal_dates_by_days(close.index, rebal_days)
        fprint(f"  {len(rebal_dates)} rebalance dates")

        records = build_feature_records(close, high, low, rebal_dates,
                                        V4_FEATURES, regime_series)
        if len(records) < 100:
            fprint(f"  WARNING: Only {len(records)} records, may be insufficient")
            records_by_rebal[rebal_days] = records
            rankings_by_rebal[rebal_days] = {}
            continue

        fprint(f"  {len(records)} feature records built")

        rankings = walk_forward_lgbm_rank(records, V4_FEATURES,
                                          f"rebal_{rebal_days}d")
        fprint(f"  {len(rankings)} ranking dates from WF-LGBM")

        records_by_rebal[rebal_days] = records
        rankings_by_rebal[rebal_days] = rankings

    # 5. Run all combinations
    fprint("\n" + "=" * 80)
    fprint("RUNNING 30 PARAMETER COMBINATIONS")
    fprint("=" * 80)

    all_results = {}
    combo_num = 0

    for top_k, rebal_days in product(TOP_K_VALUES, REBAL_DAY_VALUES):
        combo_num += 1
        combo_name = f"K{top_k}_R{rebal_days}"
        fprint(f"\n[{combo_num}/30] {combo_name}: top_k={top_k}, rebal={rebal_days}d")

        rankings = rankings_by_rebal.get(rebal_days, {})
        if not rankings:
            fprint(f"  No rankings for rebal={rebal_days}d, skipping")
            all_results[combo_name] = {
                "top_k": top_k, "rebal_days": rebal_days,
                "error": "no_rankings"
            }
            continue

        trades, final_eq = simulate_trades(
            rankings, close, high, low, regime_series, atr_dict,
            top_k=top_k, skip_vix_25_30=True)

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            all_results[combo_name] = {
                "top_k": top_k, "rebal_days": rebal_days,
                "n_trades": len(trades) if trades else 0,
                "error": "too_few_trades"
            }
            continue

        metrics = compute_metrics(trades, CAP)

        fprint(f"  Trades: {metrics['n_trades']} | Sharpe: {metrics['sharpe']:.2f} | "
               f"Sortino: {metrics['sortino']:.2f} | WR: {metrics['win_rate']*100:.1f}% | "
               f"PF: {metrics['profit_factor']:.2f} | MaxDD: {metrics['max_dd']*100:.1f}% | "
               f"Final: ${metrics['final_equity']:,.0f}")

        result = {
            "top_k": top_k,
            "rebal_days": rebal_days,
            "n_trades": metrics["n_trades"],
            "sharpe": round(metrics["sharpe"], 3),
            "sortino": round(metrics["sortino"], 3),
            "cagr": round(metrics["cagr"], 4),
            "max_dd": round(metrics["max_dd"], 4),
            "win_rate": round(metrics["win_rate"], 4),
            "profit_factor": round(metrics["profit_factor"], 3),
            "final_equity": round(metrics["final_equity"], 2),
        }

        all_results[combo_name] = result

    # 6. Summary table
    fprint("\n" + "=" * 80)
    fprint("SWEEP RESULTS SUMMARY")
    fprint("=" * 80)
    fprint(f"{'Combo':<12} {'K':>2} {'R':>3} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} "
           f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'CAGR':>7} {'Final$':>9}")
    fprint("-" * 82)

    valid_results = []
    for combo_name in sorted(all_results.keys()):
        r = all_results[combo_name]
        if "error" in r:
            fprint(f"  {combo_name:<12} — {r.get('error', 'ERROR')} —")
            continue
        fprint(f"  {combo_name:<12} {r['top_k']:>2} {r['rebal_days']:>3} {r['n_trades']:>5} "
               f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['win_rate']*100:>5.1f}% "
               f"{r['profit_factor']:>5.2f} {r['max_dd']*100:>6.1f}% "
               f"{r['cagr']*100:>6.1f}% ${r['final_equity']:>8,.0f}")
        valid_results.append((combo_name, r))

    # 7. Top 5 by Sharpe — run 5-gate adversarial
    fprint("\n" + "=" * 80)
    fprint("TOP 5 BY SHARPE — 5-GATE ADVERSARIAL VALIDATION")
    fprint("=" * 80)

    top5 = sorted(valid_results, key=lambda x: x[1]["sharpe"], reverse=True)[:5]

    for rank, (combo_name, r) in enumerate(top5, 1):
        top_k = r["top_k"]
        rebal_days = r["rebal_days"]
        fprint(f"\n#{rank} {combo_name} (Sharpe={r['sharpe']:.2f})")

        rankings = rankings_by_rebal[rebal_days]
        trades, final_eq = simulate_trades(
            rankings, close, high, low, regime_series, atr_dict,
            top_k=top_k, skip_vix_25_30=True)

        gates_passed, gates_total, gate_details = run_5_gate(
            trades, CAP, spy_close, n_perms=500)

        fprint(f"  Gates: {gates_passed}/{gates_total} | "
               f"SignFlip p={gate_details.get('g1_signflip_pval', 'N/A')} | "
               f"RegimeGap={gate_details.get('g2_regime_wr_gap', 'N/A')} | "
               f"YearlyPct={gate_details.get('g5_yearly_pct', 'N/A')}")

        r["gates_passed"] = gates_passed
        r["gates_total"] = gates_total
        r["gate_details"] = gate_details

    # 8. Analysis by parameter dimension
    fprint("\n" + "=" * 80)
    fprint("ANALYSIS BY TOP-K (averaged across rebalance frequencies)")
    fprint("=" * 80)
    fprint(f"{'Top-K':>6} {'Avg Sharpe':>10} {'Avg Sortino':>11} {'Avg WR':>7} {'Avg Trades':>10}")
    fprint("-" * 50)

    for k in TOP_K_VALUES:
        k_results = [r for _, r in valid_results if r["top_k"] == k]
        if k_results:
            avg_sh = np.mean([r["sharpe"] for r in k_results])
            avg_so = np.mean([r["sortino"] for r in k_results])
            avg_wr = np.mean([r["win_rate"] for r in k_results])
            avg_tr = np.mean([r["n_trades"] for r in k_results])
            fprint(f"  {k:>5} {avg_sh:>10.2f} {avg_so:>11.2f} {avg_wr*100:>6.1f}% {avg_tr:>10.0f}")

    fprint("\n" + "=" * 80)
    fprint("ANALYSIS BY REBALANCE FREQUENCY (averaged across top-K)")
    fprint("=" * 80)
    fprint(f"{'Rebal':>6} {'Avg Sharpe':>10} {'Avg Sortino':>11} {'Avg WR':>7} {'Avg Trades':>10}")
    fprint("-" * 50)

    for rd in REBAL_DAY_VALUES:
        r_results = [r for _, r in valid_results if r["rebal_days"] == rd]
        if r_results:
            avg_sh = np.mean([r["sharpe"] for r in r_results])
            avg_so = np.mean([r["sortino"] for r in r_results])
            avg_wr = np.mean([r["win_rate"] for r in r_results])
            avg_tr = np.mean([r["n_trades"] for r in r_results])
            fprint(f"  {rd:>5}d {avg_sh:>10.2f} {avg_so:>11.2f} {avg_wr*100:>6.1f}% {avg_tr:>10.0f}")

    # 9. Interaction analysis
    fprint("\n" + "=" * 80)
    fprint("INTERACTION HEATMAP (Sharpe)")
    fprint("=" * 80)
    fprint(f"{'':>8}", end="")
    for rd in REBAL_DAY_VALUES:
        fprint(f"  R{rd:>3}d", end="")
    fprint()
    for k in TOP_K_VALUES:
        fprint(f"  K={k:<3}", end="")
        for rd in REBAL_DAY_VALUES:
            cname = f"K{k}_R{rd}"
            r = all_results.get(cname, {})
            sh = r.get("sharpe", 0)
            if "error" in r:
                fprint(f"  {'---':>5}", end="")
            else:
                fprint(f"  {sh:>5.2f}", end="")
        fprint()

    # 10. Commission impact analysis
    fprint("\n" + "=" * 80)
    fprint("COMMISSION IMPACT ANALYSIS")
    fprint("=" * 80)
    fprint(f"{'Combo':<12} {'Trades':>6} {'TotalComm':>10} {'Comm%ofCap':>11} {'CommPerTr':>10}")
    fprint("-" * 55)
    for combo_name, r in sorted(valid_results, key=lambda x: x[1]["n_trades"], reverse=True)[:10]:
        total_comm = r["n_trades"] * COMMISSION_RT_SPREAD
        comm_pct = total_comm / CAP * 100
        fprint(f"  {combo_name:<12} {r['n_trades']:>5} ${total_comm:>8.0f} {comm_pct:>10.1f}% "
               f"${COMMISSION_RT_SPREAD:>8.2f}")

    # 11. Key findings
    fprint("\n" + "=" * 80)
    fprint("KEY FINDINGS")
    fprint("=" * 80)

    if valid_results:
        best = max(valid_results, key=lambda x: x[1]["sharpe"])
        worst = min(valid_results, key=lambda x: x[1]["sharpe"])
        baseline = all_results.get("K3_R10", {})

        fprint(f"  BEST: {best[0]} — Sharpe {best[1]['sharpe']:.2f}, "
               f"Sortino {best[1]['sortino']:.2f}, WR {best[1]['win_rate']*100:.1f}%, "
               f"{best[1]['n_trades']} trades")
        fprint(f"  WORST: {worst[0]} — Sharpe {worst[1]['sharpe']:.2f}, "
               f"Sortino {worst[1]['sortino']:.2f}, WR {worst[1]['win_rate']*100:.1f}%, "
               f"{worst[1]['n_trades']} trades")
        if baseline and "error" not in baseline:
            fprint(f"  BASELINE (K3_R10): Sharpe {baseline['sharpe']:.2f}, "
                   f"Sortino {baseline['sortino']:.2f}, WR {baseline['win_rate']*100:.1f}%")
            fprint(f"  Best vs baseline: {best[1]['sharpe'] - baseline['sharpe']:+.2f} Sharpe")

        # Concentrated vs diversified
        k1_results = [r for _, r in valid_results if r["top_k"] == 1]
        k5_results = [r for _, r in valid_results if r["top_k"] == 5]
        if k1_results and k5_results:
            avg_k1 = np.mean([r["sharpe"] for r in k1_results])
            avg_k5 = np.mean([r["sharpe"] for r in k5_results])
            fprint(f"\n  Q1: Concentrated (K=1 avg Sharpe {avg_k1:.2f}) vs "
                   f"Diversified (K=5 avg Sharpe {avg_k5:.2f})")
            if avg_k1 > avg_k5:
                fprint(f"      CONCENTRATED WINS by {avg_k1 - avg_k5:.2f} Sharpe")
            else:
                fprint(f"      DIVERSIFIED WINS by {avg_k5 - avg_k1:.2f} Sharpe")

        # Weekly vs monthly
        r5_results = [r for _, r in valid_results if r["rebal_days"] == 5]
        r21_results = [r for _, r in valid_results if r["rebal_days"] == 21]
        if r5_results and r21_results:
            avg_r5 = np.mean([r["sharpe"] for r in r5_results])
            avg_r21 = np.mean([r["sharpe"] for r in r21_results])
            fprint(f"\n  Q2: Weekly (R=5d avg Sharpe {avg_r5:.2f}) vs "
                   f"Monthly (R=21d avg Sharpe {avg_r21:.2f})")
            if avg_r5 > avg_r21:
                fprint(f"      WEEKLY WINS by {avg_r5 - avg_r21:.2f} Sharpe")
            else:
                fprint(f"      MONTHLY WINS by {avg_r21 - avg_r5:.2f} Sharpe")

    # Save results
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"topk_rebal_sweep_{t0.strftime('%Y%m%d_%H%M')}"):
                for combo_name, r in all_results.items():
                    if "error" in r:
                        continue
                    mlflow.log_metric(f"{combo_name}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{combo_name}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{combo_name}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{combo_name}_pf", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{combo_name}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{combo_name}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{combo_name}_final_eq", r.get("final_equity", 0))
                    if "gates_passed" in r:
                        mlflow.log_metric(f"{combo_name}_gates", r["gates_passed"])

                mlflow.log_params({
                    "capital": CAP, "dte": DTE, "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT, "commission_rt": COMMISSION_RT_SPREAD,
                    "top_k_values": str(TOP_K_VALUES),
                    "rebal_day_values": str(REBAL_DAY_VALUES),
                    "n_combinations": len(TOP_K_VALUES) * len(REBAL_DAY_VALUES),
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "regime_bear_thresh": REGIME_BEAR_THRESHOLD,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "variant": "v4c_bullbear_vixskip",
                })
                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow logged to '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
