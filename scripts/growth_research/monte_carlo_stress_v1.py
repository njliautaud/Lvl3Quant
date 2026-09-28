#!/usr/bin/env python3
"""
Monte Carlo Stress Test v1 — Confidence Intervals for v6 Strategy
==================================================================

Runs the v6 production strategy (K=2, weekly, 2% OTM, DTE=21) ONCE to get
per-trade returns, then bootstraps 1,000 equity paths with block bootstrap
(5-trade blocks) to estimate confidence intervals on all key metrics.

Also runs 4 stress scenarios:
  1. Remove best 10% of trades (lucky trades removed)
  2. Double worst 10% of trade losses (fat tails)
  3. Random 2% slippage on 20% of trades (execution risk)
  4. 2x commission ($5.20 instead of $2.60)

Logs to MLflow experiment "monte_carlo_stress_v1".
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "monte_carlo_stress_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
MAX_POS = 200.0
SPREAD_PCT = 3.0
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# v6 production parameters
TOP_K = 2
REBAL_DAYS = 5
DTE = 21
MONEYNESS_PCT = 2.0  # 2% OTM

# Monte Carlo
N_BOOTSTRAP = 1000
BLOCK_SIZE = 5  # block bootstrap to preserve autocorrelation
RANDOM_SEED = 42

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "monte_carlo_stress_v1"

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


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD (identical to production)
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

    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    needed = ["SPY", "VIX"]
    for t in needed:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ══════════════════════════════════════════════════════════════
# REGIME LOADING (identical to production)
# ══════════════════════════════════════════════════════════════

def load_regime_predictions():
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found at {REGIME_FILE}")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days")
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
# FEATURE ENGINEERING (identical to production v4)
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

V4_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


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
# WALK-FORWARD LGBM RANKING (identical to production)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series, dte):
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")
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


def walk_forward_lgbm_rank(df, feature_cols):
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
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
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
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

    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION (identical to production)
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
# TRADE SIMULATION (v6 production: 2% OTM, hold to expiry)
# ══════════════════════════════════════════════════════════════

def simulate_trades(rankings, close, high, low, regime_series, atr_dict,
                    commission_override=None):
    """
    Simulate v6 strategy and return list of trade dicts with PnL.
    commission_override: if set, use this instead of COMMISSION_RT_SPREAD.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None
    commission = commission_override if commission_override is not None else COMMISSION_RT_SPREAD

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # VIX 25-30 sit-in-cash filter
        if 25.0 <= cv <= 30.0:
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

        picks = [t for t, _ in ranked[:TOP_K]]

        max_pos = min(MAX_POS, equity / 3)
        if max_pos < 30:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in close.columns or tk not in atr_dict or n_entered >= TOP_K:
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

            K1 = round(S * (1 + MONEYNESS_PCT / 100), 2)
            K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                if direction == "bull":
                    entry_cost_ps, max_profit_ps = price_bull_call_spread(
                        S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                    )
                else:
                    entry_cost_ps, max_profit_ps = price_bear_put_spread(
                        S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                    )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + commission

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            Se = float(close[tk].iloc[ei])

            if direction == "bull":
                intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            else:
                intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

            exit_value_ps = intrinsic
            pnl = (exit_value_ps - entry_cost_ps) * 100 - commission
            equity += pnl
            n_entered += 1

            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            spy_regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_cost": round(total_cost, 2),
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
# MONTE CARLO BOOTSTRAP ENGINE
# ══════════════════════════════════════════════════════════════

def compute_path_metrics(pnl_array, initial_capital=CAP):
    """Compute Sharpe, MaxDD, CAGR, final equity, WR from a PnL array."""
    equity = np.cumsum(pnl_array) + initial_capital
    final_eq = equity[-1]

    # Win rate
    wr = np.mean(pnl_array > 0) if len(pnl_array) > 0 else 0.0

    # Equity-based returns for Sharpe
    eq_series = np.concatenate([[initial_capital], equity])
    returns = np.diff(eq_series) / eq_series[:-1]

    # Sharpe (annualized assuming ~52 trades/year for weekly strategy)
    # We'll use actual trade count to estimate annualization
    n_trades = len(pnl_array)
    if n_trades > 1 and np.std(returns) > 1e-10:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(min(n_trades, 52))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # CAGR (assume ~1 trade per week, so n_trades/52 years)
    years = max(n_trades / 52.0, 0.5)
    if final_eq > 0 and initial_capital > 0:
        cagr = (final_eq / initial_capital) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    return {
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "cagr": float(cagr),
        "final_equity": float(final_eq),
        "win_rate": float(wr),
    }


def block_bootstrap(pnl_array, n_paths=1000, block_size=5, seed=42):
    """
    Block bootstrap: resample trade returns in blocks of `block_size`
    with replacement to preserve autocorrelation structure.
    Returns array of shape (n_paths, n_trades).
    """
    rng = np.random.RandomState(seed)
    n = len(pnl_array)
    n_blocks = int(np.ceil(n / block_size))

    # Create block start indices
    max_start = n - block_size
    if max_start < 0:
        max_start = 0

    paths = np.zeros((n_paths, n))
    for i in range(n_paths):
        # Sample block starts with replacement
        starts = rng.randint(0, max_start + 1, size=n_blocks)
        sampled = []
        for s in starts:
            sampled.extend(pnl_array[s:s + block_size].tolist())
        paths[i, :] = sampled[:n]  # trim to original length

    return paths


def run_monte_carlo(pnl_array, n_paths=N_BOOTSTRAP, block_size=BLOCK_SIZE):
    """Run Monte Carlo bootstrap and compute metrics for all paths."""
    fprint(f"\n{'=' * 80}")
    fprint(f"MONTE CARLO BOOTSTRAP: {n_paths} paths, block_size={block_size}")
    fprint(f"{'=' * 80}")
    fprint(f"  Input: {len(pnl_array)} trades, total PnL: ${sum(pnl_array):.2f}")

    paths = block_bootstrap(pnl_array, n_paths=n_paths, block_size=block_size)

    all_metrics = {
        "sharpe": [], "max_dd": [], "cagr": [],
        "final_equity": [], "win_rate": [],
    }

    for i in range(n_paths):
        m = compute_path_metrics(paths[i])
        for k in all_metrics:
            all_metrics[k].append(m[k])

    # Convert to numpy
    for k in all_metrics:
        all_metrics[k] = np.array(all_metrics[k])

    return all_metrics, paths


def report_monte_carlo(metrics, original_metrics):
    """Print Monte Carlo results with confidence intervals."""
    fprint(f"\n{'=' * 80}")
    fprint("MONTE CARLO RESULTS — CONFIDENCE INTERVALS")
    fprint(f"{'=' * 80}")

    fprint(f"\n  Original strategy: Sharpe={original_metrics['sharpe']:.3f}, "
           f"CAGR={original_metrics['cagr']*100:.1f}%, "
           f"MaxDD={original_metrics['max_dd']*100:.1f}%, "
           f"Final=${original_metrics['final_equity']:,.0f}")

    fprint(f"\n  {'Metric':<20} {'Median':>10} {'5th pct':>10} {'95th pct':>10} {'Original':>10}")
    fprint(f"  {'-' * 62}")

    for name, label, fmt in [
        ("sharpe", "Sharpe", ".3f"),
        ("cagr", "CAGR", ".1%"),
        ("max_dd", "Max Drawdown", ".1%"),
        ("final_equity", "Final Equity ($)", ",.0f"),
        ("win_rate", "Win Rate", ".1%"),
    ]:
        arr = metrics[name]
        med = np.median(arr)
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)
        orig = original_metrics[name]

        if fmt == ".1%":
            fprint(f"  {label:<20} {med*100:>9.1f}% {p5*100:>9.1f}% {p95*100:>9.1f}% {orig*100:>9.1f}%")
        elif fmt == ",.0f":
            fprint(f"  {label:<20} ${med:>9,.0f} ${p5:>9,.0f} ${p95:>9,.0f} ${orig:>9,.0f}")
        else:
            fprint(f"  {label:<20} {med:>10{fmt}} {p5:>10{fmt}} {p95:>10{fmt}} {orig:>10{fmt}}")

    # Sharpe threshold probabilities
    sharpes = metrics["sharpe"]
    fprint(f"\n  SHARPE THRESHOLD PROBABILITIES:")
    fprint(f"    P(Sharpe > 1.0) = {np.mean(sharpes > 1.0) * 100:.1f}%")
    fprint(f"    P(Sharpe > 1.5) = {np.mean(sharpes > 1.5) * 100:.1f}%")
    fprint(f"    P(Sharpe > 2.0) = {np.mean(sharpes > 2.0) * 100:.1f}%")

    # Tail risk
    max_dds = metrics["max_dd"]
    fprint(f"\n  TAIL RISK (Max Drawdown):")
    fprint(f"    P(MaxDD > -30%) = {np.mean(max_dds > -0.30) * 100:.1f}%  (good)")
    fprint(f"    P(MaxDD > -50%) = {np.mean(max_dds > -0.50) * 100:.1f}%  (good)")
    fprint(f"    P(MaxDD < -30%) = {np.mean(max_dds < -0.30) * 100:.1f}%  (tail risk)")
    fprint(f"    P(MaxDD < -50%) = {np.mean(max_dds < -0.50) * 100:.1f}%  (severe risk)")

    # Expected equity range
    final_eq = metrics["final_equity"]
    fprint(f"\n  EXPECTED FINAL EQUITY:")
    fprint(f"    5th percentile:  ${np.percentile(final_eq, 5):>10,.0f}")
    fprint(f"    25th percentile: ${np.percentile(final_eq, 25):>10,.0f}")
    fprint(f"    Median:          ${np.percentile(final_eq, 50):>10,.0f}")
    fprint(f"    75th percentile: ${np.percentile(final_eq, 75):>10,.0f}")
    fprint(f"    95th percentile: ${np.percentile(final_eq, 95):>10,.0f}")

    # Worst case path
    worst_idx = np.argmin(final_eq)
    worst_sharpe = sharpes[worst_idx]
    worst_dd = max_dds[worst_idx]
    worst_cagr = metrics["cagr"][worst_idx]
    fprint(f"\n  WORST-CASE PATH (lowest final equity):")
    fprint(f"    Final equity: ${final_eq[worst_idx]:,.0f}")
    fprint(f"    Sharpe: {worst_sharpe:.3f}")
    fprint(f"    CAGR: {worst_cagr*100:.1f}%")
    fprint(f"    MaxDD: {worst_dd*100:.1f}%")

    # 5th percentile path stats
    p5_idx = np.argsort(final_eq)[int(len(final_eq) * 0.05)]
    fprint(f"\n  5th PERCENTILE PATH (conservative estimate):")
    fprint(f"    Final equity: ${final_eq[p5_idx]:,.0f}")
    fprint(f"    Sharpe: {sharpes[p5_idx]:.3f}")
    fprint(f"    CAGR: {metrics['cagr'][p5_idx]*100:.1f}%")
    fprint(f"    MaxDD: {max_dds[p5_idx]*100:.1f}%")


# ══════════════════════════════════════════════════════════════
# STRESS TESTS
# ══════════════════════════════════════════════════════════════

def run_stress_tests(trades, rankings, close, high, low, regime_series, atr_dict):
    """Run 4 stress scenarios on the trade PnL vector."""
    pnl_array = np.array([t["pnl"] for t in trades])
    n = len(pnl_array)
    original = compute_path_metrics(pnl_array)

    fprint(f"\n{'=' * 80}")
    fprint("STRESS TESTS")
    fprint(f"{'=' * 80}")
    fprint(f"  Original: {n} trades, Sharpe={original['sharpe']:.3f}, "
           f"CAGR={original['cagr']*100:.1f}%, MaxDD={original['max_dd']*100:.1f}%")

    stress_results = {}

    # STRESS 1: Remove best 10% of trades
    fprint(f"\n  --- STRESS 1: Remove best 10% of trades ---")
    n_remove = max(1, int(n * 0.10))
    sorted_idx = np.argsort(pnl_array)[::-1]  # best first
    keep_idx = sorted_idx[n_remove:]  # remove best
    stress1_pnl = pnl_array[np.sort(keep_idx)]
    s1 = compute_path_metrics(stress1_pnl)
    fprint(f"    Removed {n_remove} best trades (max PnL removed: ${pnl_array[sorted_idx[0]]:.2f})")
    fprint(f"    Sharpe: {original['sharpe']:.3f} -> {s1['sharpe']:.3f}")
    fprint(f"    CAGR: {original['cagr']*100:.1f}% -> {s1['cagr']*100:.1f}%")
    fprint(f"    MaxDD: {original['max_dd']*100:.1f}% -> {s1['max_dd']*100:.1f}%")
    fprint(f"    Final: ${original['final_equity']:,.0f} -> ${s1['final_equity']:,.0f}")
    stress_results["remove_best_10pct"] = s1

    # STRESS 2: Double worst 10% of trade losses
    fprint(f"\n  --- STRESS 2: Double worst 10% of losses ---")
    stress2_pnl = pnl_array.copy()
    loss_idx = np.where(pnl_array < 0)[0]
    if len(loss_idx) > 0:
        loss_sorted = loss_idx[np.argsort(pnl_array[loss_idx])]  # worst first
        n_double = max(1, int(len(loss_sorted) * 0.10))
        worst_losses = loss_sorted[:n_double]
        stress2_pnl[worst_losses] *= 2.0
        s2 = compute_path_metrics(stress2_pnl)
        fprint(f"    Doubled {n_double} worst losses")
        fprint(f"    Worst loss doubled: ${pnl_array[worst_losses[0]]:.2f} -> ${stress2_pnl[worst_losses[0]]:.2f}")
        fprint(f"    Sharpe: {original['sharpe']:.3f} -> {s2['sharpe']:.3f}")
        fprint(f"    CAGR: {original['cagr']*100:.1f}% -> {s2['cagr']*100:.1f}%")
        fprint(f"    MaxDD: {original['max_dd']*100:.1f}% -> {s2['max_dd']*100:.1f}%")
        fprint(f"    Final: ${original['final_equity']:,.0f} -> ${s2['final_equity']:,.0f}")
    else:
        s2 = original.copy()
        fprint(f"    No losses to double!")
    stress_results["double_worst_10pct"] = s2

    # STRESS 3: Random 2% slippage on 20% of trades
    fprint(f"\n  --- STRESS 3: 2% slippage on 20% of trades ---")
    rng = np.random.RandomState(RANDOM_SEED)
    stress3_pnl = pnl_array.copy()
    entry_costs = np.array([t["entry_cost"] for t in trades])
    slip_mask = rng.random(n) < 0.20
    n_slipped = slip_mask.sum()
    slippage_amount = entry_costs * 0.02  # 2% of entry cost
    stress3_pnl[slip_mask] -= slippage_amount[slip_mask]
    s3 = compute_path_metrics(stress3_pnl)
    fprint(f"    Applied 2% slippage to {n_slipped}/{n} trades")
    fprint(f"    Total slippage cost: ${slippage_amount[slip_mask].sum():.2f}")
    fprint(f"    Sharpe: {original['sharpe']:.3f} -> {s3['sharpe']:.3f}")
    fprint(f"    CAGR: {original['cagr']*100:.1f}% -> {s3['cagr']*100:.1f}%")
    fprint(f"    MaxDD: {original['max_dd']*100:.1f}% -> {s3['max_dd']*100:.1f}%")
    fprint(f"    Final: ${original['final_equity']:,.0f} -> ${s3['final_equity']:,.0f}")
    stress_results["slippage_2pct"] = s3

    # STRESS 4: 2x commission ($5.20 instead of $2.60)
    fprint(f"\n  --- STRESS 4: 2x commission ($5.20 vs $2.60) ---")
    trades_2x, final_eq_2x = simulate_trades(
        rankings, close, high, low, regime_series, atr_dict,
        commission_override=5.20,
    )
    if trades_2x:
        s4_pnl = np.array([t["pnl"] for t in trades_2x])
        s4 = compute_path_metrics(s4_pnl)
        fprint(f"    Trades: {len(trades_2x)} (was {n})")
        fprint(f"    Sharpe: {original['sharpe']:.3f} -> {s4['sharpe']:.3f}")
        fprint(f"    CAGR: {original['cagr']*100:.1f}% -> {s4['cagr']*100:.1f}%")
        fprint(f"    MaxDD: {original['max_dd']*100:.1f}% -> {s4['max_dd']*100:.1f}%")
        fprint(f"    Final: ${original['final_equity']:,.0f} -> ${s4['final_equity']:,.0f}")
    else:
        s4 = {"sharpe": 0, "cagr": 0, "max_dd": -1, "final_equity": CAP, "win_rate": 0}
        fprint(f"    No trades with 2x commission!")
    stress_results["double_commission"] = s4

    # Summary table
    fprint(f"\n  {'STRESS SUMMARY':-^62}")
    fprint(f"  {'Scenario':<25} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Final$':>10}")
    fprint(f"  {'-' * 62}")
    fprint(f"  {'Original':<25} {original['sharpe']:>8.3f} {original['cagr']*100:>7.1f}% "
           f"{original['max_dd']*100:>7.1f}% ${original['final_equity']:>9,.0f}")
    for label, key in [
        ("Remove best 10%", "remove_best_10pct"),
        ("Double worst 10% loss", "double_worst_10pct"),
        ("2% slip on 20% trades", "slippage_2pct"),
        ("2x commission ($5.20)", "double_commission"),
    ]:
        s = stress_results[key]
        fprint(f"  {label:<25} {s['sharpe']:>8.3f} {s['cagr']*100:>7.1f}% "
               f"{s['max_dd']*100:>7.1f}% ${s['final_equity']:>9,.0f}")

    return stress_results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"MONTE CARLO STRESS TEST v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Strategy: v6 (K={TOP_K}, weekly={REBAL_DAYS}d, DTE={DTE}, "
           f"OTM={MONEYNESS_PCT}%, 21 features, GRU regime)")
    fprint(f"Capital: ${CAP:.0f} | Max pos: ${MAX_POS:.0f} | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Monte Carlo: {N_BOOTSTRAP} bootstrap paths, block_size={BLOCK_SIZE}")
    fprint()

    # ── PHASE 1: Run strategy once to get trade vector ──
    fprint("PHASE 1: Running v6 strategy to get per-trade returns...")

    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)
    feature_cols = V4_FEATURES

    # Build rebalance dates
    rebal_freq = f"{REBAL_DAYS}B"
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(rebal_freq).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)}")

    # Build features and LGBM rankings
    records = build_feature_records(close, high, low, rebal_dates, feature_cols,
                                    regime_series, dte=DTE)
    rankings = walk_forward_lgbm_rank(records, feature_cols)

    if not rankings:
        fprint("ERROR: No rankings produced. Cannot continue.")
        return

    # Simulate trades
    trades, final_eq = simulate_trades(rankings, close, high, low,
                                       regime_series, atr_dict)

    fprint(f"\nStrategy results: {len(trades)} trades, ${CAP:.0f} -> ${final_eq:,.0f}")

    if len(trades) < 20:
        fprint(f"ERROR: Only {len(trades)} trades — insufficient for Monte Carlo.")
        return

    # Validate with adversarial gates
    result = validate_trades(
        trades, initial_capital=CAP,
        spy_prices=close["SPY"],
        strategy_name="v6_production",
    )
    result.print_summary()

    pnl_array = np.array([t["pnl"] for t in trades])
    original_metrics = compute_path_metrics(pnl_array)

    # Trade statistics
    fprint(f"\n  TRADE STATISTICS:")
    fprint(f"    Total trades: {len(pnl_array)}")
    fprint(f"    Winners: {np.sum(pnl_array > 0)} ({np.mean(pnl_array > 0)*100:.1f}%)")
    fprint(f"    Losers: {np.sum(pnl_array < 0)} ({np.mean(pnl_array < 0)*100:.1f}%)")
    fprint(f"    Mean PnL: ${np.mean(pnl_array):.2f}")
    fprint(f"    Median PnL: ${np.median(pnl_array):.2f}")
    fprint(f"    Std PnL: ${np.std(pnl_array):.2f}")
    fprint(f"    Max win: ${np.max(pnl_array):.2f}")
    fprint(f"    Max loss: ${np.min(pnl_array):.2f}")
    fprint(f"    Skewness: {float(stats.skew(pnl_array)):.3f}")
    fprint(f"    Kurtosis: {float(stats.kurtosis(pnl_array)):.3f}")

    # ── PHASE 2: Monte Carlo Bootstrap ──
    fprint(f"\nPHASE 2: Monte Carlo Bootstrap ({N_BOOTSTRAP} paths)...")
    mc_metrics, mc_paths = run_monte_carlo(pnl_array)
    report_monte_carlo(mc_metrics, original_metrics)

    # ── PHASE 3: Stress Tests ──
    fprint(f"\nPHASE 3: Stress Tests...")
    stress_results = run_stress_tests(trades, rankings, close, high, low,
                                      regime_series, atr_dict)

    # ── SAVE RESULTS ──
    results = {
        "run_timestamp": t0.isoformat(),
        "strategy": {
            "name": "v6_production",
            "top_k": TOP_K,
            "rebal_days": REBAL_DAYS,
            "dte": DTE,
            "moneyness_pct": MONEYNESS_PCT,
            "n_features": len(V4_FEATURES),
            "capital": CAP,
            "max_pos": MAX_POS,
            "commission": COMMISSION_RT_SPREAD,
        },
        "original_metrics": {
            **original_metrics,
            "n_trades": len(trades),
            "gates_passed": result.gates_passed,
            "gates_total": result.gates_total,
            "sortino": result.sortino,
            "profit_factor": result.profit_factor,
        },
        "monte_carlo": {
            "n_paths": N_BOOTSTRAP,
            "block_size": BLOCK_SIZE,
            "sharpe": {
                "median": float(np.median(mc_metrics["sharpe"])),
                "p5": float(np.percentile(mc_metrics["sharpe"], 5)),
                "p95": float(np.percentile(mc_metrics["sharpe"], 95)),
                "p_gt_1": float(np.mean(mc_metrics["sharpe"] > 1.0)),
                "p_gt_1_5": float(np.mean(mc_metrics["sharpe"] > 1.5)),
                "p_gt_2": float(np.mean(mc_metrics["sharpe"] > 2.0)),
            },
            "max_dd": {
                "median": float(np.median(mc_metrics["max_dd"])),
                "p5": float(np.percentile(mc_metrics["max_dd"], 5)),
                "p95": float(np.percentile(mc_metrics["max_dd"], 95)),
                "p_gt_neg30": float(np.mean(mc_metrics["max_dd"] > -0.30)),
                "p_gt_neg50": float(np.mean(mc_metrics["max_dd"] > -0.50)),
            },
            "cagr": {
                "median": float(np.median(mc_metrics["cagr"])),
                "p5": float(np.percentile(mc_metrics["cagr"], 5)),
                "p95": float(np.percentile(mc_metrics["cagr"], 95)),
            },
            "final_equity": {
                "p5": float(np.percentile(mc_metrics["final_equity"], 5)),
                "p25": float(np.percentile(mc_metrics["final_equity"], 25)),
                "median": float(np.percentile(mc_metrics["final_equity"], 50)),
                "p75": float(np.percentile(mc_metrics["final_equity"], 75)),
                "p95": float(np.percentile(mc_metrics["final_equity"], 95)),
            },
            "win_rate": {
                "median": float(np.median(mc_metrics["win_rate"])),
                "p5": float(np.percentile(mc_metrics["win_rate"], 5)),
                "p95": float(np.percentile(mc_metrics["win_rate"], 95)),
            },
        },
        "stress_tests": {},
        "trades": trades,
    }

    # Convert stress results (numpy types to float)
    for k, v in stress_results.items():
        results["stress_tests"][k] = {kk: float(vv) for kk, vv in v.items()}

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save equity paths for later visualization
    np.savez_compressed(
        OUTPUT_DIR / "mc_paths.npz",
        paths=mc_paths,
        pnl_array=pnl_array,
    )
    fprint(f"Monte Carlo paths saved ({mc_paths.shape})")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"mc_stress_{t0.strftime('%Y%m%d_%H%M')}"):
                # Original metrics
                mlflow.log_metric("original_sharpe", original_metrics["sharpe"])
                mlflow.log_metric("original_cagr", original_metrics["cagr"])
                mlflow.log_metric("original_max_dd", original_metrics["max_dd"])
                mlflow.log_metric("original_final_equity", original_metrics["final_equity"])
                mlflow.log_metric("original_win_rate", original_metrics["win_rate"])
                mlflow.log_metric("original_n_trades", len(trades))
                mlflow.log_metric("original_sortino", result.sortino)
                mlflow.log_metric("original_profit_factor", result.profit_factor)
                mlflow.log_metric("gates_passed", result.gates_passed)

                # MC confidence intervals
                mlflow.log_metric("mc_sharpe_median", float(np.median(mc_metrics["sharpe"])))
                mlflow.log_metric("mc_sharpe_p5", float(np.percentile(mc_metrics["sharpe"], 5)))
                mlflow.log_metric("mc_sharpe_p95", float(np.percentile(mc_metrics["sharpe"], 95)))
                mlflow.log_metric("mc_p_sharpe_gt_1", float(np.mean(mc_metrics["sharpe"] > 1.0)))
                mlflow.log_metric("mc_p_sharpe_gt_1_5", float(np.mean(mc_metrics["sharpe"] > 1.5)))
                mlflow.log_metric("mc_p_sharpe_gt_2", float(np.mean(mc_metrics["sharpe"] > 2.0)))
                mlflow.log_metric("mc_maxdd_median", float(np.median(mc_metrics["max_dd"])))
                mlflow.log_metric("mc_final_eq_median", float(np.median(mc_metrics["final_equity"])))
                mlflow.log_metric("mc_final_eq_p5", float(np.percentile(mc_metrics["final_equity"], 5)))
                mlflow.log_metric("mc_final_eq_p95", float(np.percentile(mc_metrics["final_equity"], 95)))
                mlflow.log_metric("mc_cagr_median", float(np.median(mc_metrics["cagr"])))

                # Stress test results
                for scenario, s in stress_results.items():
                    mlflow.log_metric(f"stress_{scenario}_sharpe", s["sharpe"])
                    mlflow.log_metric(f"stress_{scenario}_cagr", s["cagr"])
                    mlflow.log_metric(f"stress_{scenario}_max_dd", s["max_dd"])

                # Parameters
                mlflow.log_params({
                    "strategy": "v6_production",
                    "top_k": TOP_K,
                    "rebal_days": REBAL_DAYS,
                    "dte": DTE,
                    "moneyness_pct": MONEYNESS_PCT,
                    "n_features": len(V4_FEATURES),
                    "capital": CAP,
                    "commission": COMMISSION_RT_SPREAD,
                    "n_bootstrap": N_BOOTSTRAP,
                    "block_size": BLOCK_SIZE,
                    "n_trades": len(trades),
                })

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
