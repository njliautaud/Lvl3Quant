#!/usr/bin/env python3
"""
V9.1 Confidence-Weighted Position Sizing
=========================================

Currently V9.1 equal-sizes all K=3 positions. But LGBM scores have varying
confidence — the top-ranked sector might have score 0.95 while #2 has 0.60.

This tests whether position-sizing by confidence improves risk-adjusted returns.

Variants:
  A: Equal sizing (baseline V9.1)
  B: Proportional to LGBM score (normalize scores → weight allocations)
  C: Concentrated (K=1 with 2x position, K=2 with 1.5x, K=3 with 0.5x)
  D: Score-differential gate (only trade if top score > bottom by threshold)
  E: Dynamic K (trade K=1 if score spread < p25, K=3 if > p50, K=5 if > p75)
  F: Inverse-vol weighting (allocate more to lower-ATR sectors)

Output: output/growth_research/v91_confidence_sizing_v1/
MLflow experiment: v91_confidence_sizing_v1
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "v91_confidence_sizing_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 12
DTE = 28
REBAL_INTERVAL = 10

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v91_confidence_sizing_v1"

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
        "found": True, "spread_cost_mid": spread_cost_mid,
        "near_strike": float(near_leg["strike"]), "far_strike": float(far_leg["strike"]),
    }


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


def build_feature_records(close, high, low, rebal_dates, regime_series):
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
            fi = min(idx + DTE, len(close) - 1)
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


def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * 1.02, 2)
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * 0.98, 2)
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity, chains, max_pos):
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None
    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)
    used_real = False
    entry_cost_ps = None
    chain_df = chains.get(tk)
    if chain_df is not None:
        result = find_chain_spread_price(chain_df, dt, direction, K1, K2, DTE)
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
                entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
            else:
                entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
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


def simulate_variant(rankings, close, atr_dict, chains, sizing_mode="equal"):
    """Simulate with different position sizing strategies."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0
    score_spreads = []  # Track score spread for analysis

    sorted_dates = sorted(rankings.keys())
    n = REBAL_INTERVAL // 5
    rebal_dates = sorted_dates[::n]

    for dt in rebal_dates:
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        trade_mode = "pairs" if cv < 20.0 else "bull_only"
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        # Compute score spread (top - bottom)
        all_scores = [s for _, s in ranked_desc]
        score_spread = all_scores[0] - all_scores[-1] if len(all_scores) > 1 else 0

        # Determine K and weights based on sizing mode
        if sizing_mode == "equal":
            k = TOP_K
            bull_picks = [(t, 1.0) for t, _ in ranked_desc[:k]]
            bear_picks = [(t, 1.0) for t, _ in ranked_asc[:k]] if trade_mode == "pairs" else []

        elif sizing_mode == "proportional":
            k = TOP_K
            top_scores = [s for _, s in ranked_desc[:k]]
            total_s = sum(top_scores)
            if total_s > 0:
                bull_picks = [(t, s / total_s * k) for (t, s) in ranked_desc[:k]]
            else:
                bull_picks = [(t, 1.0) for t, _ in ranked_desc[:k]]
            if trade_mode == "pairs":
                bot_scores = [1 - s for _, s in ranked_asc[:k]]
                total_b = sum(bot_scores)
                if total_b > 0:
                    bear_picks = [(t, (1-s) / total_b * k) for (t, s) in ranked_asc[:k]]
                else:
                    bear_picks = [(t, 1.0) for t, _ in ranked_asc[:k]]
            else:
                bear_picks = []

        elif sizing_mode == "concentrated":
            k = TOP_K
            weights = [2.0, 1.5, 0.5]  # Top pick gets 2x, #2 gets 1.5x, #3 gets 0.5x
            bull_picks = [(t, weights[i]) for i, (t, _) in enumerate(ranked_desc[:k])]
            if trade_mode == "pairs":
                bear_picks = [(t, weights[i]) for i, (t, _) in enumerate(ranked_asc[:k])]
            else:
                bear_picks = []

        elif sizing_mode == "score_gate":
            # Only trade if score spread is above median
            score_spreads.append(score_spread)
            median_spread = np.median(score_spreads) if len(score_spreads) > 10 else 0
            if score_spread < median_spread:
                continue
            k = TOP_K
            bull_picks = [(t, 1.0) for t, _ in ranked_desc[:k]]
            bear_picks = [(t, 1.0) for t, _ in ranked_asc[:k]] if trade_mode == "pairs" else []

        elif sizing_mode == "dynamic_k":
            # Vary K based on score spread percentile
            score_spreads.append(score_spread)
            if len(score_spreads) > 10:
                pctile = sum(1 for s in score_spreads[:-1] if s < score_spread) / len(score_spreads[:-1])
            else:
                pctile = 0.5
            if pctile < 0.25:
                k = 1
            elif pctile < 0.50:
                k = 2
            elif pctile < 0.75:
                k = 3
            else:
                k = min(5, len(ranked_desc))
            bull_picks = [(t, 1.0) for t, _ in ranked_desc[:k]]
            bear_picks = [(t, 1.0) for t, _ in ranked_asc[:k]] if trade_mode == "pairs" else []

        elif sizing_mode == "inverse_vol":
            k = TOP_K
            bull_tickers = [t for t, _ in ranked_desc[:k]]
            bear_tickers = [t for t, _ in ranked_asc[:k]]
            # Weight inversely by ATR
            def get_inv_vol_weights(tickers):
                atrs = []
                for t in tickers:
                    if t in atr_dict and dt in atr_dict[t].index:
                        atrs.append(float(atr_dict[t].loc[dt]))
                    else:
                        atrs.append(1.0)
                inv = [1.0 / max(a, 0.01) for a in atrs]
                total = sum(inv)
                return [v / total * len(tickers) for v in inv]
            bull_weights = get_inv_vol_weights(bull_tickers)
            bull_picks = list(zip(bull_tickers, bull_weights))
            if trade_mode == "pairs":
                bear_weights = get_inv_vol_weights(bear_tickers)
                bear_picks = list(zip(bear_tickers, bear_weights))
            else:
                bear_picks = []
        else:
            raise ValueError(f"Unknown sizing mode: {sizing_mode}")

        # Execute trades with weights
        base_max_pos = min(100, equity / 6) if trade_mode == "pairs" else min(200, equity / 3)
        if base_max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk, weight in picks:
                adjusted_max_pos = base_max_pos * weight
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity, chains, adjusted_max_pos
                )
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0, "trade_mode": trade_mode,
                        "weight": round(weight, 2), "score_spread": round(score_spread, 4),
                    })

    return trades, equity, real_count, bs_count


def monte_carlo_ci(trades, n_bootstrap=1000, seed=42):
    if len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    rng = np.random.RandomState(seed)
    sharpes = [float(np.mean(rng.choice(pnls, len(pnls), True)) /
               (np.std(rng.choice(pnls, len(pnls), True)) + 1e-10) * np.sqrt(52))
               for _ in range(n_bootstrap)]
    return {
        "mean": float(np.mean(sharpes)),
        "ci_95_low": float(np.percentile(sharpes, 2.5)),
        "ci_95_high": float(np.percentile(sharpes, 97.5)),
    }


VARIANTS = {
    "A_equal": {"mode": "equal", "desc": "Equal sizing (baseline V9.1)"},
    "B_proportional": {"mode": "proportional", "desc": "Proportional to LGBM score"},
    "C_concentrated": {"mode": "concentrated", "desc": "Concentrated (2x/1.5x/0.5x by rank)"},
    "D_score_gate": {"mode": "score_gate", "desc": "Score-spread gate (skip low-spread dates)"},
    "E_dynamic_k": {"mode": "dynamic_k", "desc": "Dynamic K (1-5 by score spread)"},
    "F_inverse_vol": {"mode": "inverse_vol", "desc": "Inverse-vol weighting"},
}


def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V9.1 CONFIDENCE-WEIGHTED SIZING v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Testing position sizing strategies for V9.1")
    fprint(f"Config: DTE={DTE}, biweekly rebal, adaptive width, cost/width<{COST_WIDTH_MAX}")
    fprint(f"Capital: ${CAP:.0f}")
    fprint(f"\n{len(VARIANTS)} variants:")
    for vn, vc in VARIANTS.items():
        fprint(f"  {vn}: {vc['desc']}")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    # Build rankings (shared across all variants)
    fprint(f"\n{'=' * 80}")
    fprint(f"BUILDING LGBM RANKINGS (DTE={DTE})")
    fprint(f"{'=' * 80}")
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )
    records = build_feature_records(close, high, low, rebal_dates, regime_series)
    rankings = walk_forward_lgbm_rank(records)

    # Analyze score spread distribution
    spreads = []
    for dt, scores in rankings.items():
        vals = list(scores.values())
        if len(vals) > 1:
            spreads.append(max(vals) - min(vals))
    fprint(f"\nScore spread stats: mean={np.mean(spreads):.3f}, std={np.std(spreads):.3f}")
    fprint(f"  p25={np.percentile(spreads, 25):.3f}, p50={np.percentile(spreads, 50):.3f}, "
           f"p75={np.percentile(spreads, 75):.3f}")

    spy_close = close["SPY"]
    all_results = {}

    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 90}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 90}")

        trades, final_eq, real_count, bs_count = simulate_variant(
            rankings, close, atr_dict, chains, sizing_mode=vcfg["mode"]
        )

        if not trades or len(trades) < 5:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            continue

        total = real_count + bs_count
        fprint(f"\n  Trades: {len(trades)} | Real: {real_count} ({real_count/max(total,1)*100:.0f}%) | "
               f"Final: ${final_eq:,.0f}")

        result = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close, strategy_name=vname)
        result.print_summary()

        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

        real_t = [t for t in trades if t.get("used_real_pricing")]
        real_sharpe = None
        if real_t:
            rp = [t["pnl"] for t in real_t]
            real_sharpe = np.mean(rp) / (np.std(rp) + 1e-10) * np.sqrt(52)
            fprint(f"  REAL-ONLY: {len(real_t)} trades, Sharpe {real_sharpe:.2f}, "
                   f"WR {sum(1 for p in rp if p > 0)/len(rp):.1%}")

        # Weight analysis
        weights = [t.get("weight", 1.0) for t in trades]
        if any(w != 1.0 for w in weights):
            fprint(f"  Weight stats: mean={np.mean(weights):.2f}, std={np.std(weights):.2f}, "
                   f"min={min(weights):.2f}, max={max(weights):.2f}")
            # Win rate by weight quartile
            for label, lo, hi in [("Low weight", 0, np.percentile(weights, 25)),
                                   ("High weight", np.percentile(weights, 75), 100)]:
                wt = [t for t in trades if lo <= t.get("weight", 1.0) <= hi]
                if wt:
                    wwr = sum(1 for t in wt if t["win"]) / len(wt)
                    fprint(f"    {label}: {len(wt)} trades, WR {wwr:.1%}")

        pnls = [t["pnl"] for t in trades]
        rd = result.to_dict()
        mc = monte_carlo_ci(trades)
        all_results[vname] = {
            **rd, "final_equity": round(final_eq, 2),
            "monte_carlo": mc,
            "real_pct": round(real_count / max(total, 1) * 100, 1),
            "real_sharpe": round(real_sharpe, 2) if real_sharpe else None,
            "avg_pnl_per_trade": round(np.mean(pnls), 2),
        }

    # ── COMPARISON ──
    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON TABLE")
    fprint(f"{'=' * 140}")
    fprint(f"{'Variant':<30} {'Trades':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} {'WR':>6} "
           f"{'MDD':>8} {'Final$':>10} {'RealSh':>8} {'AvgPnL':>8}")
    fprint("-" * 140)

    baseline_sharpe = None
    for vn in VARIANTS:
        if vn not in all_results:
            continue
        r = all_results[vn]
        sharpe = r.get("sharpe", 0)
        if baseline_sharpe is None:
            baseline_sharpe = sharpe
        delta = ((sharpe / baseline_sharpe) - 1) * 100 if baseline_sharpe and baseline_sharpe != 0 else 0
        fprint(f"  {vn:<28} {r.get('n_trades', 0):>7} {sharpe:>8.2f} {r.get('sortino', 0):>8.2f} "
               f"{r.get('profit_factor', 0):>6.2f} {r.get('win_rate', 0):>5.1%} "
               f"{r.get('max_drawdown', 0):>7.1%} ${r['final_equity']:>9,.0f} "
               f"{r.get('real_sharpe', 'N/A'):>8} ${r['avg_pnl_per_trade']:>7.2f} "
               f"({'baseline' if delta == 0 else f'{delta:+.0f}%'})")

    # MC CI
    fprint(f"\n{'=' * 100}")
    fprint("MONTE CARLO 95% CI")
    fprint(f"{'=' * 100}")
    for vn in VARIANTS:
        if vn not in all_results or all_results[vn].get("monte_carlo") is None:
            continue
        mc = all_results[vn]["monte_carlo"]
        fprint(f"  {vn}: Sharpe [{mc['ci_95_low']:.2f}, {mc['ci_95_high']:.2f}], mean {mc['mean']:.2f}")

    # Conclusion
    fprint(f"\n{'=' * 100}")
    fprint("CONCLUSION")
    fprint(f"{'=' * 100}")
    best_vn = max(all_results, key=lambda k: all_results[k].get("sharpe", 0)) if all_results else None
    if best_vn:
        best = all_results[best_vn]
        b_sharpe = all_results.get("A_equal", {}).get("sharpe", 1)
        improvement = ((best.get("sharpe", 0) / b_sharpe) - 1) * 100 if b_sharpe else 0
        fprint(f"  BEST: {best_vn} — Sharpe {best.get('sharpe', 0):.2f}")
        if best_vn != "A_equal":
            fprint(f"  vs baseline: {improvement:+.1f}% Sharpe improvement")
            fprint(f"  Trade count: {all_results.get('A_equal', {}).get('n_trades', 0)} → {best.get('n_trades', 0)}")
        else:
            fprint(f"  Equal sizing IS optimal — position sizing adjustments don't help")

    # Save
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="confidence_sizing_v1"):
                mlflow.log_param("dte", DTE)
                mlflow.log_param("rebal_interval", REBAL_INTERVAL)
                mlflow.log_param("n_variants", len(VARIANTS))
                for vn, vr in all_results.items():
                    mlflow.log_metric(f"{vn}_sharpe", vr.get("sharpe", 0))
                    mlflow.log_metric(f"{vn}_trades", vr.get("n_trades", 0))
                    if vr.get("real_sharpe"):
                        mlflow.log_metric(f"{vn}_real_sharpe", vr["real_sharpe"])
                mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
            fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow error: {e}")


if __name__ == "__main__":
    main()
