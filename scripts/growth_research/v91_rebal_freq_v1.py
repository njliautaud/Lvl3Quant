#!/usr/bin/env python3
"""
V9.1 Rebalance Frequency Optimization
=======================================

Current V9.1: DTE=28 with weekly rebalance (every 5 trading days).
This means we open new positions every week but hold for 4 weeks.
This creates position overlap — up to 4 cohorts open simultaneously.

Hypothesis: Rebalancing every 2 or 4 weeks (matching DTE=28) should:
1. Reduce commission drag (fewer trades)
2. Reduce position overlap/sizing issues
3. Better match prediction horizon to trade horizon (MFE-within-horizon, HC #432)

Variants:
  A: Weekly rebal (current V9.1 baseline) — 5 trading days
  B: Biweekly rebal — 10 trading days
  C: Monthly rebal — 20 trading days (matches DTE=28)
  D: Every 3 weeks — 15 trading days
  E: Weekly rebal + DTE=14 (V9 baseline for comparison)

All use V9.1 config: adaptive max($3,3%), cost/width<50%, LGBM ranker, real pricing.

Output: output/growth_research/v91_rebal_freq_v1/
MLflow experiment: v91_rebal_freq_v1
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "v91_rebal_freq_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 12

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v91_rebal_freq_v1"

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
        "found": True,
        "spread_cost_mid": spread_cost_mid,
        "near_strike": float(near_leg["strike"]),
        "far_strike": float(far_leg["strike"]),
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


def simulate_variant(rankings, close, atr_dict, chains, dte, rebal_interval_days):
    """Simulate with specific rebalance frequency."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0

    # Filter rankings to match rebalance frequency
    sorted_dates = sorted(rankings.keys())
    if rebal_interval_days > 5:
        # Subsample: take every Nth rebalance date
        n = rebal_interval_days // 5  # weekly = 1, biweekly = 2, monthly = 4
        rebal_dates = sorted_dates[::n]
    else:
        rebal_dates = sorted_dates

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

    return trades, equity, real_count, bs_count


def monte_carlo_ci(trades, n_bootstrap=1000, seed=42):
    if len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    rng = np.random.RandomState(seed)
    sharpes = [float(np.mean(rng.choice(pnls, len(pnls), True)) /
               (np.std(rng.choice(pnls, len(pnls), True)) + 1e-10) * np.sqrt(52))
               for _ in range(n_bootstrap)]
    sharpes = np.array(sharpes)
    return {
        "mean": float(np.mean(sharpes)),
        "ci_95_low": float(np.percentile(sharpes, 2.5)),
        "ci_95_high": float(np.percentile(sharpes, 97.5)),
    }


VARIANTS = {
    "A_weekly_DTE28": {"dte": 28, "rebal_days": 5, "desc": "Weekly + DTE=28 (V9.1 baseline)"},
    "B_biweekly_DTE28": {"dte": 28, "rebal_days": 10, "desc": "Biweekly + DTE=28"},
    "C_monthly_DTE28": {"dte": 28, "rebal_days": 20, "desc": "Monthly + DTE=28 (aligned)"},
    "D_3week_DTE28": {"dte": 28, "rebal_days": 15, "desc": "3-weekly + DTE=28"},
    "E_weekly_DTE14": {"dte": 14, "rebal_days": 5, "desc": "Weekly + DTE=14 (V9 baseline)"},
}


def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V9.1 REBALANCE FREQUENCY TEST v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Testing rebalance frequency alignment with DTE=28")
    fprint(f"Capital: ${CAP:.0f} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"\n{len(VARIANTS)} variants:")
    for vn, vc in VARIANTS.items():
        fprint(f"  {vn}: {vc['desc']}")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    # Build rankings for DTE=14 and DTE=28
    spy_close = close["SPY"]
    rankings_cache = {}
    for dte in [14, 28]:
        fprint(f"\n{'=' * 80}")
        fprint(f"BUILDING LGBM RANKINGS (DTE={dte})")
        fprint(f"{'=' * 80}")
        rebal_dates = pd.DatetimeIndex(
            close.index.to_series().resample("W-FRI").last().dropna().values
        )
        records = build_feature_records(close, high, low, rebal_dates, regime_series, dte=dte)
        rankings = walk_forward_lgbm_rank(records)
        rankings_cache[dte] = rankings

    all_results = {}
    all_trades = {}

    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 90}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 90}")

        dte = vcfg["dte"]
        rankings = rankings_cache.get(dte, {})

        trades, final_eq, real_count, bs_count = simulate_variant(
            rankings, close, atr_dict, chains, dte, vcfg["rebal_days"]
        )

        if not trades or len(trades) < 5:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            continue

        all_trades[vname] = trades
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

        # Real-only analysis
        real_t = [t for t in trades if t.get("used_real_pricing")]
        if real_t:
            rp = [t["pnl"] for t in real_t]
            fprint(f"  REAL-ONLY: {len(real_t)} trades, Sharpe {np.mean(rp)/(np.std(rp)+1e-10)*np.sqrt(52):.2f}, "
                   f"WR {sum(1 for p in rp if p>0)/len(rp):.1%}")

        # Chain-only 2019+
        ct = [t for t in trades if t["entry_date"] >= "2019"]
        if ct:
            cp = [t["pnl"] for t in ct]
            fprint(f"  Chain 2019+: {len(ct)} trades, Sharpe {np.mean(cp)/(np.std(cp)+1e-10)*np.sqrt(52):.2f}, "
                   f"{sum(1 for t in ct if t['used_real_pricing'])/len(ct)*100:.0f}% real")

        # Commission analysis
        total_comm = len(trades) * COMMISSION_RT_SPREAD
        total_pnl = sum(t["pnl"] for t in trades)
        fprint(f"  Total commission: ${total_comm:,.0f} | Commission/PnL: {total_comm/max(total_pnl,1)*100:.1f}%")

        rd = result.to_dict()
        mc = monte_carlo_ci(trades)
        all_results[vname] = {
            **rd, "final_equity": round(final_eq, 2),
            "monte_carlo": mc,
            "real_pct": round(real_count / max(total, 1) * 100, 1),
            "total_commission": round(total_comm, 2),
            "comm_pnl_pct": round(total_comm / max(total_pnl, 1) * 100, 1),
        }

    # ── COMPARISON ──
    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON — ALL REBALANCE FREQUENCIES")
    fprint(f"{'=' * 140}")
    fprint(f"  {'Variant':<25} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'Gate':>5} {'Final$':>9} {'Comm%':>6}")
    fprint(f"  {'-' * 95}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r:
            continue
        fprint(f"  {vn:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f} {r['comm_pnl_pct']:>5.1f}%")

    # Monte Carlo
    fprint(f"\n{'=' * 100}")
    fprint("MONTE CARLO BOOTSTRAP")
    fprint(f"{'=' * 100}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r or not r.get("monte_carlo"):
            continue
        mc = r["monte_carlo"]
        fprint(f"  {vn:<25} Mean {mc['mean']:.2f}, 95% CI [{mc['ci_95_low']:.2f}, {mc['ci_95_high']:.2f}]")

    # ── VERDICT ──
    fprint(f"\n{'=' * 100}")
    fprint("VERDICT")
    fprint(f"{'=' * 100}")

    best_vn = None
    best_sh = -999
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r:
            continue
        if f"{r['gates_passed']}/{r['gates_total']}" == "5/5" and r["sharpe"] > best_sh:
            best_sh = r["sharpe"]
            best_vn = vn

    if best_vn:
        r = all_results[best_vn]
        fprint(f"  Best: {best_vn} (Sharpe {r['sharpe']:.2f}, {r['n_trades']} trades)")

    # Check if aligned rebal beats weekly
    weekly = all_results.get("A_weekly_DTE28")
    biweekly = all_results.get("B_biweekly_DTE28")
    monthly = all_results.get("C_monthly_DTE28")

    if weekly and biweekly:
        fprint(f"\n  Weekly vs Biweekly: Sharpe {weekly['sharpe']:.2f} vs {biweekly['sharpe']:.2f}")
    if weekly and monthly:
        fprint(f"  Weekly vs Monthly: Sharpe {weekly['sharpe']:.2f} vs {monthly['sharpe']:.2f}")
        fprint(f"  Commission drag: Weekly {weekly['comm_pnl_pct']:.1f}% vs Monthly {monthly['comm_pnl_pct']:.1f}%")

    # Save
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"rebal_freq_{t0.strftime('%Y%m%d_%H%M')}"):
                for vn, r in all_results.items():
                    mlflow.log_metric(f"{vn}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{vn}_wr", r["win_rate"])
                    mlflow.log_metric(f"{vn}_pf", r["profit_factor"])
                    mlflow.log_metric(f"{vn}_n_trades", r["n_trades"])
                    mlflow.log_metric(f"{vn}_comm_pct", r["comm_pnl_pct"])
                    mlflow.log_metric(f"{vn}_gates", r["gates_passed"])
                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
