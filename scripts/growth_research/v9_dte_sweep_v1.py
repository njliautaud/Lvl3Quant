#!/usr/bin/env python3
"""
V9 DTE Sweep v1 — Finding Optimal Holding Period with Real Pricing
===================================================================

Motivation:
  - V9 candidate at DTE=14 has chain-only Sharpe 3.50 (KB #236)
  - DTE=21 showed Sharpe 4.77 but only 33% chain coverage (KB #238)
  - Weekly options (7, 14 DTE) have 100% coverage
  - Monthly options (~28 DTE) should have ~95% coverage
  - Need to find the DTE sweet spot: enough time for move vs. theta cost

Variants tested:
  A: DTE=7  (weekly, fast theta decay but cheap entry)
  B: DTE=14 (current prod — baseline)
  C: DTE=21 (3-week — known poor coverage)
  D: DTE=28 (monthly — should have good coverage)
  E: DTE=35 (5-week — farther out, higher cost)

All variants use V9 core config:
  - Adaptive width max($3, 3%)
  - Cost/width filter < 50%
  - Real pricing where available
  - Pairs when VIX < 20, bull-only when VIX >= 20

Key analysis:
  1. Chain data coverage (% of trades with real pricing) at each DTE
  2. Full-period and chain-only Sharpe
  3. Cost/width distribution (do longer DTE = higher cost?)
  4. Year-by-year stability
  5. Adversarial validation (5/5 gates)

Output: output/growth_research/v9_dte_sweep_v1/
MLflow experiment: v9_dte_sweep_v1
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

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "v9_dte_sweep_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REBAL_FREQ = "W-FRI"
WF_TRAIN_PERIODS = 12

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03
MIN_BID = 0.05

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v9_dte_sweep_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable")

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(V6_FEATURES) == 17


# ══════════════════════════════════════════════════════════════
# CHAIN DATA
# ══════════════════════════════════════════════════════════════

def load_all_chains():
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
        fprint(f"  {tk}: {len(df):,} rows")
    return chains


def analyze_chain_dte_coverage(chains):
    """Analyze what DTE values are available in the chain data."""
    fprint(f"\n{'=' * 80}")
    fprint("CHAIN DATA DTE COVERAGE ANALYSIS")
    fprint(f"{'=' * 80}")

    dte_targets = [7, 14, 21, 28, 35]

    for target_dte in dte_targets:
        total_attempts = 0
        found_count = 0

        for tk, chain_df in chains.items():
            dates = sorted(chain_df["date"].unique())
            for dt in dates:
                total_attempts += 1
                day_data = chain_df[chain_df["date"] == dt]
                exps = day_data[["expiration", "dte"]].drop_duplicates()
                close_match = exps[(exps["dte"] - target_dte).abs() <= DTE_TOLERANCE]
                if len(close_match) > 0:
                    found_count += 1

        coverage = found_count / max(total_attempts, 1) * 100
        fprint(f"  DTE={target_dte:2d}: {found_count:,}/{total_attempts:,} "
               f"({coverage:.1f}% coverage within ±{DTE_TOLERANCE} days)")

    # Also show the DTE distribution in the actual data
    fprint(f"\n  DTE distribution across all chains:")
    all_dtes = []
    for tk, chain_df in chains.items():
        day_exps = chain_df.groupby("date")["dte"].unique()
        for dtes in day_exps:
            all_dtes.extend(dtes.tolist())

    all_dtes = np.array(all_dtes)
    for bucket_start, bucket_end in [(0, 10), (10, 17), (17, 25), (25, 32), (32, 40), (40, 60)]:
        count = np.sum((all_dtes >= bucket_start) & (all_dtes < bucket_end))
        fprint(f"    DTE {bucket_start}-{bucket_end}: {count:,} expiration slots")


def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
    if chain_df is None:
        return None
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[(chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
                          (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))]
        if nearby.empty:
            return None
        nearest_date = min(nearby["date"].unique(),
                          key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
        chain_day = chain_df[chain_df["date"] == nearest_date]

    exps = chain_day[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid_exps = exps[exps["dte_dist"] <= DTE_TOLERANCE]
    if valid_exps.empty:
        return None
    best_exp_row = valid_exps.loc[valid_exps["dte_dist"].idxmin()]
    actual_dte = int(best_exp_row["dte"])
    chain_exp = chain_day[chain_day["expiration"] == best_exp_row["expiration"]]

    opt_type = "c" if direction == "bull" else "p"
    near_target = K1 if direction == "bull" else K2
    far_target = K2 if direction == "bull" else K1

    near_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if near_opts.empty:
        return None
    near_opts["dist"] = (near_opts["strike"] - near_target).abs()
    near_leg = near_opts.sort_values("dist").iloc[0]
    if near_leg["dist"] / max(near_target, 1) > STRIKE_TOLERANCE:
        return None

    far_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    far_opts["dist"] = (far_opts["strike"] - far_target).abs()
    far_leg = far_opts.sort_values("dist").iloc[0]
    if far_leg["dist"] / max(far_target, 1) > STRIKE_TOLERANCE:
        return None

    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else \
        (float(near_leg["bid"]) + float(near_leg["ask"])) / 2
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else \
        (float(far_leg["bid"]) + float(far_leg["ask"])) / 2

    spread_cost_mid = abs(near_mid - far_mid)
    return {
        "found": True,
        "spread_cost_mid": spread_cost_mid,
        "near_strike": float(near_leg["strike"]),
        "far_strike": float(far_leg["strike"]),
        "actual_dte": actual_dte,
    }


# ══════════════════════════════════════════════════════════════
# DATA + FEATURES
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
    close = close.ffill(); high = high.ffill(); low = low.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


def load_regime_predictions():
    if not REGIME_FILE.exists():
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
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


def compute_legacy_features(px, spy_slice):
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0
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
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None
    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}
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


def compute_atr_series(high, low, close, period=14):
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna(); l = low[tk].dropna(); c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, regime_series, dte):
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue
        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            legacy = compute_legacy_features(px, spy.iloc[:idx + 1])
            if not legacy:
                continue
            cross_asset = compute_cross_asset_features(tk, idx, close)
            fi = min(idx + dte, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)
    df = pd.DataFrame(records)
    for c in V6_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[V6_FEATURES] = df[V6_FEATURES].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df):
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
        Xt = np.nan_to_num(train_df[V6_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[V6_FEATURES].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# STRIKES + EXECUTION
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct=0.02, width_value=3.0):
    """V9 adaptive width: max($width_value, strike * 3%)"""
    if direction == "bull":
        K1 = round(S * (1 + otm_pct), 2)
        pct_w = K1 * 0.03
        w = max(width_value, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * (1 - otm_pct), 2)
        pct_w = K2 * 0.03
        w = max(width_value, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, dte, max_pos):
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    used_real = False
    entry_cost_ps = None

    chain_df = chains.get(tk)
    if chain_df is not None:
        result = find_chain_spread_price(chain_df, dt, direction, K1, K2, dte)
        if result and result["found"]:
            entry_cost_ps = result["spread_cost_mid"]
            used_real = True
            if direction == "bull":
                K1 = result["near_strike"]
                K2 = result["far_strike"]
            else:
                K1 = result["far_strike"]
                K2 = result["near_strike"]

    if entry_cost_ps is None:
        try:
            if direction == "bull":
                entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=vix_val)
            else:
                entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=vix_val)
        except Exception:
            return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    Se = float(close[tk].iloc[ei])
    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "pnl": round(pnl, 2), "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2), "used_real_pricing": used_real,
        "K1": K1, "K2": K2, "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4), "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
    }


# ══════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_dte_variant(dte, rankings, close, atr_dict, chains, vix_threshold=20.0):
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0
    equity_curve = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        trade_mode = "pairs" if cv < vix_threshold else "bull_only"
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        max_pos = min(100, equity / 6) if trade_mode == "pairs" else min(200, equity / 3)
        if max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity,
                    chains, dte, max_pos
                )
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + dte, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0, "trade_mode": trade_mode,
                    })
                    equity_curve.append({"date": str(dt.date()), "equity": round(equity, 2)})

    return trades, equity, real_count, bs_count, equity_curve


def chain_only_analysis(trades):
    ct = [t for t in trades if t["entry_date"] >= "2019"]
    if not ct:
        return None
    pnls = [t["pnl"] for t in ct]
    real_ct = [t for t in ct if t["used_real_pricing"]]
    return {
        "trades": len(ct),
        "real_trades": len(real_ct),
        "pct_real": round(len(real_ct) / len(ct) * 100, 1),
        "wr": sum(1 for p in pnls if p > 0) / len(pnls),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52)),
        "total_pnl": sum(pnls),
        "avg_pnl": np.mean(pnls),
        "avg_cost_width": np.mean([t["cost_width_ratio"] for t in ct]),
    }


def yearly_analysis(trades):
    """Per-year breakdown."""
    years = sorted(set(t["entry_date"][:4] for t in trades))
    results = {}
    for yr in years:
        yt = [t for t in trades if t["entry_date"][:4] == yr]
        if not yt:
            continue
        pnls = [t["pnl"] for t in yt]
        results[yr] = {
            "trades": len(yt),
            "wr": sum(1 for p in pnls if p > 0) / len(pnls),
            "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52)),
            "total_pnl": sum(pnls),
            "pct_real": sum(1 for t in yt if t["used_real_pricing"]) / len(yt) * 100,
        }
    return results


def monte_carlo_ci(trades, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for Sharpe."""
    if len(trades) < 20:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        sh = float(np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(52))
        sharpes.append(sh)
    sharpes = np.array(sharpes)
    return {
        "mean": float(np.mean(sharpes)),
        "std": float(np.std(sharpes)),
        "ci_95_low": float(np.percentile(sharpes, 2.5)),
        "ci_95_high": float(np.percentile(sharpes, 97.5)),
        "p_positive": float(np.mean(sharpes > 0)),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

DTE_VALUES = [7, 14, 21, 28, 35]


def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V9 DTE SWEEP v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Testing DTE values: {DTE_VALUES}")
    fprint(f"V9 core config: adaptive max($3, 3%) width, cost/width < 50%, real pricing")
    fprint(f"Capital: ${CAP:.0f} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()
    analyze_chain_dte_coverage(chains)

    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values
    )
    spy_close = close["SPY"]

    # Build LGBM rankings for each DTE (different forward-return horizons)
    rankings_by_dte = {}
    for dte in DTE_VALUES:
        fprint(f"\n{'=' * 80}")
        fprint(f"BUILDING LGBM RANKINGS (DTE={dte})")
        fprint(f"{'=' * 80}")
        records = build_feature_records(close, high, low, rebal_dates, regime_series, dte=dte)
        rankings = walk_forward_lgbm_rank(records)
        rankings_by_dte[dte] = rankings

    # Simulate each DTE variant
    all_results = {}
    all_trades = {}

    for dte in DTE_VALUES:
        fprint(f"\n{'~' * 90}")
        fprint(f"DTE={dte} — V9 Core Config")
        fprint(f"{'~' * 90}")

        rankings = rankings_by_dte[dte]
        if not rankings:
            fprint(f"  No rankings for DTE={dte}, skipping")
            continue

        trades, final_eq, real_count, bs_count, eq_curve = simulate_dte_variant(
            dte, rankings, close, atr_dict, chains
        )
        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            continue

        all_trades[dte] = trades
        total = real_count + bs_count
        fprint(f"\n  Trades: {len(trades)} | Real: {real_count} ({real_count/total*100:.0f}%) | "
               f"Final: ${final_eq:,.0f}")

        # Adversarial validation
        vname = f"DTE_{dte}"
        result = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close, strategy_name=vname)
        result.print_summary()

        # Side breakdown
        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                real_pct = sum(1 for t in st if t["used_real_pricing"]) / len(st) * 100
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}, "
                       f"real {real_pct:.0f}%")

        # Cost/width distribution
        cwrs = [t["cost_width_ratio"] for t in trades]
        fprint(f"  Cost/width ratio: mean {np.mean(cwrs):.3f}, "
               f"p25={np.percentile(cwrs, 25):.3f}, p50={np.percentile(cwrs, 50):.3f}, "
               f"p75={np.percentile(cwrs, 75):.3f}")

        rd = result.to_dict()
        co = chain_only_analysis(trades)
        yearly = yearly_analysis(trades)
        mc = monte_carlo_ci(trades)

        all_results[dte] = {
            **rd, "chain_only_2019": co, "yearly": yearly,
            "monte_carlo": mc,
            "real_pct": round(real_count / max(total, 1) * 100, 1),
            "final_equity": round(final_eq, 2),
            "avg_cost_width": round(np.mean(cwrs), 4),
        }

    # ── COMPARISON ──
    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON — ALL DTE VALUES (FULL PERIOD)")
    fprint(f"{'=' * 140}")
    fprint(f"  {'DTE':>5} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'Gate':>5} {'Final$':>9} {'AvgCW':>7}")
    fprint(f"  {'-' * 75}")
    for dte in DTE_VALUES:
        r = all_results.get(dte)
        if not r:
            continue
        fprint(f"  {dte:>5} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f} {r['avg_cost_width']:>6.3f}")

    fprint(f"\n{'=' * 100}")
    fprint("CHAIN-ONLY 2019+ (REAL PRICING RELIABILITY)")
    fprint(f"{'=' * 100}")
    fprint(f"  {'DTE':>5} {'N':>5} {'Real':>5} {'%R':>5} {'Sharpe':>8} {'WR':>6} "
           f"{'PnL':>9} {'AvgCW':>7}")
    fprint(f"  {'-' * 60}")
    for dte in DTE_VALUES:
        r = all_results.get(dte)
        if not r or not r.get("chain_only_2019"):
            continue
        co = r["chain_only_2019"]
        fprint(f"  {dte:>5} {co['trades']:>5} {co['real_trades']:>5} {co['pct_real']:>4.0f}% "
               f"{co['sharpe']:>8.2f} {co['wr']*100:>5.1f}% "
               f"${co['total_pnl']:>8.0f} {co['avg_cost_width']:>6.3f}")

    # Year-by-year for each DTE
    fprint(f"\n{'=' * 100}")
    fprint("YEAR-BY-YEAR SHARPE")
    fprint(f"{'=' * 100}")
    all_years = sorted(set(
        yr for dte in DTE_VALUES if dte in all_results
        for yr in all_results[dte].get("yearly", {}).keys()
    ))
    header = f"  {'DTE':>5} " + " ".join(f"{yr:>8}" for yr in all_years)
    fprint(header)
    fprint(f"  {'-' * (8 + 9 * len(all_years))}")
    for dte in DTE_VALUES:
        r = all_results.get(dte)
        if not r:
            continue
        yearly = r.get("yearly", {})
        vals = []
        for yr in all_years:
            if yr in yearly:
                vals.append(f"{yearly[yr]['sharpe']:>8.2f}")
            else:
                vals.append(f"{'N/A':>8}")
        fprint(f"  {dte:>5} " + " ".join(vals))

    # Monte Carlo comparison
    fprint(f"\n{'=' * 100}")
    fprint("MONTE CARLO BOOTSTRAP (1000 resamples)")
    fprint(f"{'=' * 100}")
    fprint(f"  {'DTE':>5} {'Mean':>8} {'Std':>7} {'95% CI Low':>11} {'95% CI High':>12} {'P(>0)':>7}")
    fprint(f"  {'-' * 55}")
    for dte in DTE_VALUES:
        r = all_results.get(dte)
        if not r or not r.get("monte_carlo"):
            continue
        mc = r["monte_carlo"]
        fprint(f"  {dte:>5} {mc['mean']:>8.2f} {mc['std']:>7.2f} "
               f"{mc['ci_95_low']:>11.2f} {mc['ci_95_high']:>12.2f} "
               f"{mc['p_positive']*100:>6.1f}%")

    # ── VERDICT ──
    fprint(f"\n{'=' * 100}")
    fprint("VERDICT")
    fprint(f"{'=' * 100}")

    # Find best DTE by chain-only Sharpe (most reliable metric)
    best_dte = None
    best_chain_sharpe = -999
    for dte in DTE_VALUES:
        r = all_results.get(dte)
        if not r:
            continue
        co = r.get("chain_only_2019")
        if co and co["pct_real"] >= 20 and co["sharpe"] > best_chain_sharpe:
            gates = f"{r['gates_passed']}/{r['gates_total']}"
            if gates == "5/5":
                best_chain_sharpe = co["sharpe"]
                best_dte = dte

    if best_dte:
        best_r = all_results[best_dte]
        best_co = best_r["chain_only_2019"]
        prod_r = all_results.get(14, {})
        prod_co = prod_r.get("chain_only_2019", {})

        fprint(f"\n  Best DTE (5/5 gates, chain-only Sharpe, ≥20% real): DTE={best_dte}")
        fprint(f"  Chain-only Sharpe: {best_co['sharpe']:.2f} (real coverage: {best_co['pct_real']:.0f}%)")

        if best_dte != 14:
            fprint(f"\n  vs Current Production (DTE=14):")
            fprint(f"    DTE=14 chain Sharpe: {prod_co.get('sharpe', 0):.2f}, real: {prod_co.get('pct_real', 0):.0f}%")
            fprint(f"    DTE={best_dte} chain Sharpe: {best_co['sharpe']:.2f}, real: {best_co['pct_real']:.0f}%")
            improvement = (best_co['sharpe'] / max(prod_co.get('sharpe', 0.01), 0.01) - 1) * 100
            fprint(f"    Improvement: {improvement:+.1f}%")
            if improvement > 10:
                fprint(f"\n  ✅ DTE={best_dte} is a MEANINGFUL UPGRADE over DTE=14")
            else:
                fprint(f"\n  ⚠️ Marginal difference — stick with DTE=14 (proven)")
        else:
            fprint(f"\n  ✅ DTE=14 CONFIRMED as optimal (current production)")
    else:
        fprint("\n  ❌ No DTE variant passes all 5 gates with sufficient real pricing")

    # Coverage verdict
    fprint(f"\n  CHAIN COVERAGE RELIABILITY:")
    for dte in DTE_VALUES:
        r = all_results.get(dte)
        if not r or not r.get("chain_only_2019"):
            continue
        co = r["chain_only_2019"]
        reliability = "✅ RELIABLE" if co["pct_real"] >= 25 else "⚠️ LOW COVERAGE" if co["pct_real"] >= 10 else "❌ UNRELIABLE"
        fprint(f"    DTE={dte}: {co['pct_real']:.0f}% real-priced → {reliability}")

    # ── Save ──
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        serializable = {}
        for k, v in all_results.items():
            sv = {}
            for k2, v2 in v.items():
                if isinstance(v2, (np.floating, np.integer)):
                    sv[k2] = float(v2)
                elif isinstance(v2, dict):
                    sv[k2] = {k3: float(v3) if isinstance(v3, (np.floating, np.integer)) else v3
                              for k3, v3 in v2.items()}
                else:
                    sv[k2] = v2
            serializable[str(k)] = sv
        json.dump(serializable, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    # Save trades for each DTE
    for dte, trades in all_trades.items():
        trades_file = OUTPUT_DIR / f"trades_dte{dte}.json"
        with open(trades_file, "w") as f:
            json.dump(trades, f, indent=1, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"dte_sweep_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("dte_values", str(DTE_VALUES))
                mlflow.log_param("width_mode", "adaptive_max_3_3pct")
                mlflow.log_param("cost_width_max", COST_WIDTH_MAX)

                for dte, r in all_results.items():
                    mlflow.log_metric(f"dte{dte}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"dte{dte}_sortino", r["sortino"])
                    mlflow.log_metric(f"dte{dte}_win_rate", r["win_rate"])
                    mlflow.log_metric(f"dte{dte}_profit_factor", r["profit_factor"])
                    mlflow.log_metric(f"dte{dte}_max_dd", r["max_dd"])
                    mlflow.log_metric(f"dte{dte}_n_trades", r["n_trades"])
                    mlflow.log_metric(f"dte{dte}_gates_passed", r["gates_passed"])
                    mlflow.log_metric(f"dte{dte}_final_equity", r["final_equity"])
                    mlflow.log_metric(f"dte{dte}_real_pct", r["real_pct"])
                    co = r.get("chain_only_2019")
                    if co:
                        mlflow.log_metric(f"dte{dte}_chain_sharpe", co["sharpe"])
                        mlflow.log_metric(f"dte{dte}_chain_pct_real", co["pct_real"])
                    mc = r.get("monte_carlo")
                    if mc:
                        mlflow.log_metric(f"dte{dte}_mc_ci_low", mc["ci_95_low"])
                        mlflow.log_metric(f"dte{dte}_mc_ci_high", mc["ci_95_high"])

                if best_dte:
                    mlflow.log_metric("best_dte", best_dte)
                    mlflow.log_metric("best_chain_sharpe", best_chain_sharpe)

                mlflow.log_artifact(str(results_file))

            fprint(f"MLflow logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
