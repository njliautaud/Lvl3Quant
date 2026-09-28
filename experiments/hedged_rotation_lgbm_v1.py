#!/usr/bin/env python3
"""
HEDGED ROTATION LGBM V1 — Walk-forward LGBM ranking for sector rotation hedging

Previous experiment (hedged_rotation_v1) showed:
  - Simple 21d momentum ranking is too weak for L/S and market-neutral
  - VIX timing (cash when VIX>=20) was best (Sharpe ~1.07, MDD -9.7%)
  - Production LGBM model achieves Sharpe 1.87-2.96 with better sector ranking

This experiment tests: does BETTER sector ranking (via walk-forward LGBM) make
hedged/timed approaches viable?

Walk-forward LGBM:
  - Train on 252d rolling window
  - Predict forward 21d sector return rank
  - Retrain monthly
  - Features: returns (5d/21d/63d), volatility, relative strength, correlation,
    beta, volume ratio, VIX level, VIX change

6 VARIANTS:
  A) LGBM L/S — Long top-2, short bottom-2 ranked sectors
  B) LGBM SPY-Hedged — Long top-3, short SPY hedge (beta-adjusted)
  C) LGBM + VIX Timing — Long top-3 when VIX<20, cash when VIX>=20
  D) LGBM + Dynamic Hedge — Long top-3, hedge ratio = min(VIX/20, 1.5) of SPY
  E) LGBM L/S + VIX Gate — L/S top-2/bottom-2 only when VIX<20, cash otherwise
  F) LGBM Conviction-Weighted — Long top-3 weighted by predicted return spread,
     short bottom-1, VIX<25 gate

5-GATE VALIDATION:
  G1: Sharpe > 0.5
  G2: Permutation test p < 0.05 (1000 shuffles)
  G3: Beat random selection baseline
  G4: Regime balance |S_green - S_red|/max < 0.50
  G5: Max drawdown > -50%

MLflow: localhost:5000, experiment "hedged_rotation_lgbm_v1"
"""
from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output/hedged_rotation_lgbm_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLP", "XLU", "XLRE", "XLB"]
BENCHMARK = "SPY"
ALL_TICKERS = SECTORS + [BENCHMARK]

INITIAL_CAPITAL = 645.0
COMMISSION = 0.0  # Robinhood: $0 for fractional shares

OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
TRAIN_WINDOW = 252  # 1 year rolling training window
FORWARD_PERIOD = 21  # 21 trading days forward return target
REBAL_FREQ = "ME"  # Monthly end rebalance

N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# 5-Gate thresholds
GATES = dict(sharpe=0.5, perm_p=0.05, mdd=-0.50, regime_gap=0.50)

# LGBM hyperparameters (conservative, avoid overfitting)
LGBM_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    verbose=-1,
    n_jobs=-1,
)

FEATURE_NAMES = [
    "ret_5d", "ret_21d", "ret_63d",
    "vol_21d",
    "rel_strength_spy",
    "corr_spy_63d",
    "beta_spy",
    "vol_ratio_20_60",
    "vix_level",
    "vix_change_21d",
]


# ── Data ────────────────────────────────────────────────────────────────────

def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download daily prices and volume for all tickers + VIX via yfinance."""
    import yfinance as yf

    tickers_dl = ALL_TICKERS + ["^VIX"]
    # Need data well before OOT for training window + feature lookback
    start = "2019-06-01"
    print(f"Downloading {len(tickers_dl)} tickers from yfinance...")
    raw = yf.download(tickers_dl, start=start, end=OOT_END, auto_adjust=True, progress=False)

    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    close = close.rename(columns={"^VIX": "VIX"})
    volume = volume.rename(columns={"^VIX": "VIX"})
    close = close.dropna(how="all").ffill()
    volume = volume.dropna(how="all").ffill().fillna(0)
    print(f"  Loaded {len(close)} trading days, cols: {sorted(close.columns.tolist())}")
    return close, volume


def build_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix: one row per (date, sector) with all features."""
    spy_ret = close[BENCHMARK].pct_change()
    vix = close["VIX"] if "VIX" in close.columns else pd.Series(20.0, index=close.index)

    rows = []
    for sector in SECTORS:
        sec_ret = close[sector].pct_change()
        sec_close = close[sector]
        sec_vol = volume[sector] if sector in volume.columns else pd.Series(0, index=close.index)

        # Returns
        ret_5d = sec_close.pct_change(5)
        ret_21d = sec_close.pct_change(21)
        ret_63d = sec_close.pct_change(63)

        # Volatility: 21d rolling std of daily returns (annualized)
        vol_21d = sec_ret.rolling(21).std() * np.sqrt(252)

        # Relative strength vs SPY (21d return sector - 21d return SPY)
        spy_ret_21d = close[BENCHMARK].pct_change(21)
        rel_strength = ret_21d - spy_ret_21d

        # 63d rolling correlation to SPY
        corr_63d = sec_ret.rolling(63).corr(spy_ret)

        # Rolling beta to SPY (63d)
        cov_63d = sec_ret.rolling(63).cov(spy_ret)
        var_spy_63d = spy_ret.rolling(63).var()
        beta = cov_63d / var_spy_63d.replace(0, np.nan)

        # Volume ratio (20d avg / 60d avg)
        vol_ma20 = sec_vol.rolling(20).mean()
        vol_ma60 = sec_vol.rolling(60).mean()
        vol_ratio = vol_ma20 / vol_ma60.replace(0, np.nan)

        # VIX features
        vix_level = vix
        vix_change_21d = vix.pct_change(21)

        # Forward 21d return (target)
        fwd_ret_21d = sec_close.pct_change(FORWARD_PERIOD).shift(-FORWARD_PERIOD)

        df = pd.DataFrame({
            "date": close.index,
            "sector": sector,
            "ret_5d": ret_5d.values,
            "ret_21d": ret_21d.values,
            "ret_63d": ret_63d.values,
            "vol_21d": vol_21d.values,
            "rel_strength_spy": rel_strength.values,
            "corr_spy_63d": corr_63d.values,
            "beta_spy": beta.values,
            "vol_ratio_20_60": vol_ratio.values,
            "vix_level": vix_level.values,
            "vix_change_21d": vix_change_21d.values,
            "fwd_ret_21d": fwd_ret_21d.values,
        })
        rows.append(df)

    features = pd.concat(rows, ignore_index=True)
    features = features.set_index("date")
    return features


def walk_forward_lgbm(features: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward LGBM: train on 252d window, predict at each monthly rebalance.

    Returns DataFrame with columns: [date, sector, predicted_return, rank]
    for each rebalance date in the OOT period.
    """
    # Get monthly rebalance dates in OOT period
    oot_dates = close.loc[OOT_START:OOT_END].index
    rebal_dates = close.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    # Also include first OOT date
    first_oot = oot_dates[0]
    rebal_dates = rebal_dates.union(pd.DatetimeIndex([first_oot])).sort_values()

    all_preds = []
    n_retrained = 0

    for rebal_dt in rebal_dates:
        # Training window: TRAIN_WINDOW trading days before rebal_dt
        train_end = rebal_dt - pd.Timedelta(days=1)
        all_dates_before = features.index[features.index <= train_end].unique()
        if len(all_dates_before) < TRAIN_WINDOW + 63:
            # Not enough data for training + feature lookback
            continue

        train_dates = all_dates_before[-TRAIN_WINDOW:]
        train_start = train_dates[0]

        # Training data: all sectors in the training window
        train_mask = (features.index >= train_start) & (features.index <= train_end)
        train_df = features[train_mask].copy()

        # Drop rows with NaN in features or target
        train_df = train_df.dropna(subset=FEATURE_NAMES + ["fwd_ret_21d"])

        if len(train_df) < 100:
            continue

        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df["fwd_ret_21d"].values

        # Train LGBM
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, y_train)
        n_retrained += 1

        # Predict for each sector at rebal_dt
        pred_mask = features.index == rebal_dt
        pred_df = features[pred_mask].copy()

        if len(pred_df) == 0:
            # Try nearest prior date
            nearest = features.index[features.index <= rebal_dt]
            if len(nearest) == 0:
                continue
            rebal_actual = nearest[-1]
            pred_mask = features.index == rebal_actual
            pred_df = features[pred_mask].copy()

        # Fill any NaN features with 0 for prediction
        pred_df[FEATURE_NAMES] = pred_df[FEATURE_NAMES].fillna(0)

        if len(pred_df) == 0:
            continue

        X_pred = pred_df[FEATURE_NAMES].values
        preds = model.predict(X_pred)

        for i, (idx, row) in enumerate(pred_df.iterrows()):
            all_preds.append({
                "date": rebal_dt,
                "sector": row["sector"],
                "predicted_return": preds[i],
            })

    pred_df = pd.DataFrame(all_preds)

    # Rank within each rebalance date (higher predicted return = higher rank)
    pred_df["rank"] = pred_df.groupby("date")["predicted_return"].rank(ascending=True)

    print(f"  Walk-forward LGBM: {n_retrained} retrains, {len(pred_df)} predictions")
    print(f"  Rebalance dates covered: {pred_df['date'].nunique()}")
    return pred_df


def classify_regime(spy_returns: pd.Series) -> pd.Series:
    """Green = positive day, Red = negative day."""
    regime = pd.Series("flat", index=spy_returns.index)
    regime[spy_returns > 0.001] = "green"
    regime[spy_returns < -0.001] = "red"
    return regime


# ── Portfolio simulation ──────────────────────────────────────────────────

def simulate_portfolio(weights_series: dict[str, pd.Series],
                       close: pd.DataFrame,
                       capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """Given a dict of {ticker: weight_series}, simulate daily returns.

    weights_series: dict mapping ticker -> pd.Series of target portfolio weight on rebal dates.
    Between rebal dates, positions drift with prices (buy-and-hold between rebals).

    Returns: pd.DataFrame with columns [value, daily_return].
    """
    dates = close.loc[OOT_START:OOT_END].index
    if len(dates) == 0:
        raise ValueError("No dates in OOT range")

    daily_ret = close.pct_change().loc[dates]

    # Determine rebalance dates (from weight series)
    all_rebal = set()
    for ws in weights_series.values():
        if len(ws) > 0:
            all_rebal.update(ws.index.tolist())
    all_rebal = sorted(all_rebal)

    portfolio_value = capital
    values = []
    current_weights = {}

    for dt in dates:
        if dt in all_rebal:
            new_w = {}
            for ticker, ws in weights_series.items():
                if dt in ws.index:
                    new_w[ticker] = ws.loc[dt]
                elif len(ws.loc[:dt]) > 0:
                    new_w[ticker] = ws.loc[:dt].iloc[-1]
                else:
                    new_w[ticker] = 0.0
            current_weights = new_w

        port_ret = 0.0
        for ticker, w in current_weights.items():
            if ticker in daily_ret.columns and not np.isnan(daily_ret.loc[dt, ticker]):
                port_ret += w * daily_ret.loc[dt, ticker]

        portfolio_value *= (1 + port_ret)
        values.append({"date": dt, "value": portfolio_value, "daily_return": port_ret})

    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


# ── Strategy definitions ────────────────────────────────────────────────────

def _get_ranked_sectors(lgbm_preds: pd.DataFrame, dt, n_top=None, n_bot=None):
    """Get top/bottom ranked sectors for a given date from LGBM predictions."""
    mask = lgbm_preds["date"] == dt
    if mask.sum() == 0:
        # Find nearest prior prediction date
        prior = lgbm_preds[lgbm_preds["date"] <= dt]["date"].unique()
        if len(prior) == 0:
            return [], [], {}
        nearest = prior[-1] if isinstance(prior[-1], pd.Timestamp) else pd.Timestamp(prior[-1])
        mask = lgbm_preds["date"] == nearest

    day_preds = lgbm_preds[mask].sort_values("rank", ascending=False)
    pred_dict = dict(zip(day_preds["sector"], day_preds["predicted_return"]))

    top = day_preds.head(n_top)["sector"].tolist() if n_top else []
    bot = day_preds.tail(n_bot)["sector"].tolist() if n_bot else []
    return top, bot, pred_dict


def strategy_A_long_short(lgbm_preds: pd.DataFrame, close: pd.DataFrame) -> dict:
    """LGBM L/S — Long top-2, short bottom-2 ranked sectors (equal weight)."""
    rebal_dates = lgbm_preds["date"].unique()
    weights = {t: pd.Series(dtype=float) for t in SECTORS}

    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        top2, bot2, _ = _get_ranked_sectors(lgbm_preds, dt, n_top=2, n_bot=2)
        for t in SECTORS:
            if t in top2:
                weights[t][dt] = 0.25
            elif t in bot2:
                weights[t][dt] = -0.25
            else:
                weights[t][dt] = 0.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS}


def strategy_B_spy_hedged(lgbm_preds: pd.DataFrame, close: pd.DataFrame) -> dict:
    """LGBM SPY-Hedged — Long top-3, short SPY hedge (beta-adjusted)."""
    rebal_dates = lgbm_preds["date"].unique()
    tickers = SECTORS + [BENCHMARK]
    weights = {t: pd.Series(dtype=float) for t in tickers}

    # Precompute rolling betas
    spy_ret = close[BENCHMARK].pct_change()

    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        top3, _, _ = _get_ranked_sectors(lgbm_preds, dt, n_top=3)

        # Compute portfolio beta
        port_beta = 0.0
        for t in SECTORS:
            if t in top3:
                weights[t][dt] = 1.0 / 3.0
                # Compute 63d beta
                sec_ret = close[t].pct_change()
                lookback = sec_ret.loc[:dt].tail(63)
                spy_lb = spy_ret.loc[:dt].tail(63)
                if len(lookback) >= 30:
                    cov = lookback.cov(spy_lb)
                    var = spy_lb.var()
                    beta = cov / var if var > 1e-9 else 1.0
                else:
                    beta = 1.0
                port_beta += beta / 3.0
            else:
                weights[t][dt] = 0.0

        # Beta-adjusted SPY hedge
        weights[BENCHMARK][dt] = -port_beta

    return {t: pd.Series(weights[t], dtype=float) for t in tickers}


def strategy_C_vix_timing(lgbm_preds: pd.DataFrame, close: pd.DataFrame) -> dict:
    """LGBM + VIX Timing — Long top-3 when VIX<20, cash when VIX>=20."""
    rebal_dates = lgbm_preds["date"].unique()
    weights = {t: pd.Series(dtype=float) for t in SECTORS}
    vix = close["VIX"] if "VIX" in close.columns else pd.Series(20.0, index=close.index)

    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        vix_val = vix.loc[:dt].iloc[-1] if len(vix.loc[:dt]) > 0 else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0

        top3, _, _ = _get_ranked_sectors(lgbm_preds, dt, n_top=3)

        for t in SECTORS:
            if vix_val < 20 and t in top3:
                weights[t][dt] = 1.0 / 3.0
            else:
                weights[t][dt] = 0.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS}


def strategy_D_dynamic_hedge(lgbm_preds: pd.DataFrame, close: pd.DataFrame) -> dict:
    """LGBM + Dynamic Hedge — Long top-3, hedge ratio = min(VIX/20, 1.5) of SPY."""
    rebal_dates = lgbm_preds["date"].unique()
    tickers = SECTORS + [BENCHMARK]
    weights = {t: pd.Series(dtype=float) for t in tickers}
    vix = close["VIX"] if "VIX" in close.columns else pd.Series(20.0, index=close.index)

    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        vix_val = vix.loc[:dt].iloc[-1] if len(vix.loc[:dt]) > 0 else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0

        top3, _, _ = _get_ranked_sectors(lgbm_preds, dt, n_top=3)

        for t in SECTORS:
            if t in top3:
                weights[t][dt] = 1.0 / 3.0
            else:
                weights[t][dt] = 0.0

        hedge_ratio = min(vix_val / 20.0, 1.5)
        weights[BENCHMARK][dt] = -hedge_ratio

    return {t: pd.Series(weights[t], dtype=float) for t in tickers}


def strategy_E_ls_vix_gate(lgbm_preds: pd.DataFrame, close: pd.DataFrame) -> dict:
    """LGBM L/S + VIX Gate — L/S top-2/bottom-2 only when VIX<20, cash otherwise."""
    rebal_dates = lgbm_preds["date"].unique()
    weights = {t: pd.Series(dtype=float) for t in SECTORS}
    vix = close["VIX"] if "VIX" in close.columns else pd.Series(20.0, index=close.index)

    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        vix_val = vix.loc[:dt].iloc[-1] if len(vix.loc[:dt]) > 0 else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0

        top2, bot2, _ = _get_ranked_sectors(lgbm_preds, dt, n_top=2, n_bot=2)

        for t in SECTORS:
            if vix_val < 20:
                if t in top2:
                    weights[t][dt] = 0.25
                elif t in bot2:
                    weights[t][dt] = -0.25
                else:
                    weights[t][dt] = 0.0
            else:
                weights[t][dt] = 0.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS}


def strategy_F_conviction_weighted(lgbm_preds: pd.DataFrame, close: pd.DataFrame) -> dict:
    """LGBM Conviction-Weighted — Long top-3 weighted by predicted return spread,
    short bottom-1, VIX<25 gate."""
    rebal_dates = lgbm_preds["date"].unique()
    tickers = SECTORS + [BENCHMARK]
    weights = {t: pd.Series(dtype=float) for t in SECTORS}
    vix = close["VIX"] if "VIX" in close.columns else pd.Series(20.0, index=close.index)

    for dt in rebal_dates:
        dt = pd.Timestamp(dt)
        vix_val = vix.loc[:dt].iloc[-1] if len(vix.loc[:dt]) > 0 else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0

        top3, bot1, pred_dict = _get_ranked_sectors(lgbm_preds, dt, n_top=3, n_bot=1)

        for t in SECTORS:
            if vix_val >= 25:
                weights[t][dt] = 0.0
            elif t in top3 and pred_dict:
                # Weight by predicted return spread (normalize top-3 predictions)
                top_preds = {s: pred_dict.get(s, 0) for s in top3}
                total_spread = sum(abs(v) for v in top_preds.values())
                if total_spread > 1e-9:
                    # Allocate 75% long proportional to conviction
                    w = 0.75 * abs(top_preds.get(t, 0)) / total_spread
                else:
                    w = 0.75 / 3.0
                weights[t][dt] = w
            elif t in bot1:
                weights[t][dt] = -0.25  # Short bottom-1 at 25%
            else:
                weights[t][dt] = 0.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS}


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pf: pd.DataFrame, regime: pd.Series) -> dict:
    """Compute risk-adjusted metrics for a portfolio DataFrame."""
    rets = pf["daily_return"].values
    vals = pf["value"].values

    # Sharpe (annualized)
    mu = np.mean(rets)
    sd = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
    sharpe = (mu / sd * np.sqrt(252)) if sd > 1e-9 else 0.0

    # Sortino (annualized)
    neg = rets[rets < 0]
    downside_sd = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-9
    sortino = (mu / downside_sd * np.sqrt(252)) if downside_sd > 1e-9 else 0.0

    # Profit factor
    gross_gains = np.sum(rets[rets > 0])
    gross_losses = np.abs(np.sum(rets[rets < 0]))
    pf_ratio = gross_gains / gross_losses if gross_losses > 1e-9 else float("inf")

    # Win rate
    wr = np.mean(rets > 0) if len(rets) > 0 else 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(vals)
    drawdown = (vals - cummax) / cummax
    mdd = float(np.min(drawdown))

    # CAGR
    n_years = len(rets) / 252
    total_ret = vals[-1] / vals[0] if vals[0] > 0 else 0
    cagr = (total_ret ** (1 / n_years) - 1) if n_years > 0 and total_ret > 0 else 0.0

    final_val = float(vals[-1])

    # Regime-stratified Sharpe
    aligned_regime = regime.reindex(pf.index).dropna()
    common = pf.index.intersection(aligned_regime.index)
    green_rets = rets[np.isin(pf.index, common[aligned_regime[common] == "green"])]
    red_rets = rets[np.isin(pf.index, common[aligned_regime[common] == "red"])]

    def _sharpe(r):
        if len(r) < 5:
            return 0.0
        m = np.mean(r)
        s = np.std(r, ddof=1)
        return (m / s * np.sqrt(252)) if s > 1e-9 else 0.0

    sharpe_green = _sharpe(green_rets)
    sharpe_red = _sharpe(red_rets)

    denom = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / denom if denom > 1e-9 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf_ratio, 3),
        "win_rate": round(wr, 4),
        "cagr": round(cagr, 4),
        "mdd": round(mdd, 4),
        "final_value": round(final_val, 2),
        "total_return_pct": round((final_val / INITIAL_CAPITAL - 1) * 100, 2),
        "n_days": len(rets),
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 4),
        "n_green_days": len(green_rets),
        "n_red_days": len(red_rets),
    }


# ── Permutation test ───────────────────────────────────────────────────────

def permutation_test(pf: pd.DataFrame, n_perms: int = N_PERMUTATIONS) -> float:
    """Shuffle daily returns, compute Sharpe distribution. Return p-value."""
    rng = np.random.RandomState(RANDOM_SEED)
    rets = pf["daily_return"].values.copy()
    real_sharpe = np.mean(rets) / (np.std(rets, ddof=1) + 1e-9) * np.sqrt(252)

    count_ge = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        s = np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-9) * np.sqrt(252)
        if s >= real_sharpe:
            count_ge += 1
    return count_ge / n_perms


def random_selection_sharpe(close: pd.DataFrame, n_runs: int = 200) -> float:
    """Baseline: random sector selection (same structure as top-3 long)."""
    rng = np.random.RandomState(RANDOM_SEED)
    sharpes = []
    dates = close.loc[OOT_START:OOT_END].index
    daily_ret = close[SECTORS].pct_change().loc[dates]
    rebal_dates_set = set(close.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index)

    for _ in range(n_runs):
        port_rets = []
        current_picks = rng.choice(SECTORS, 3, replace=False)

        for dt in dates:
            if dt in rebal_dates_set:
                current_picks = rng.choice(SECTORS, 3, replace=False)
            r = daily_ret.loc[dt, current_picks].mean()
            if not np.isnan(r):
                port_rets.append(r)

        if len(port_rets) > 10:
            arr = np.array(port_rets)
            s = np.mean(arr) / (np.std(arr, ddof=1) + 1e-9) * np.sqrt(252)
            sharpes.append(s)

    return float(np.median(sharpes))


# ── 5-Gate validation ───────────────────────────────────────────────────────

def five_gate_validation(metrics: dict, perm_p: float, random_sharpe: float) -> dict:
    """Apply 5-gate validation. Returns dict with pass/fail per gate."""
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > GATES["sharpe"],
        "G2_perm_p_lt_0.05": perm_p < GATES["perm_p"],
        "G3_beat_random": metrics["sharpe"] > random_sharpe,
        "G4_regime_balanced": metrics["regime_gap"] < GATES["regime_gap"],
        "G5_mdd_gt_neg50pct": metrics["mdd"] > GATES["mdd"],
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ── MLflow logging ──────────────────────────────────────────────────────────

def log_to_mlflow(variant: str, metrics: dict, gates: dict, perm_p: float,
                  feature_importance: dict | None = None):
    """Best-effort MLflow logging."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("hedged_rotation_lgbm_v1")
        with mlflow.start_run(run_name=f"variant_{variant}"):
            mlflow.log_param("variant", variant)
            mlflow.log_param("initial_capital", INITIAL_CAPITAL)
            mlflow.log_param("oot_start", OOT_START)
            mlflow.log_param("oot_end", OOT_END)
            mlflow.log_param("train_window", TRAIN_WINDOW)
            mlflow.log_param("forward_period", FORWARD_PERIOD)
            mlflow.log_param("lgbm_n_estimators", LGBM_PARAMS["n_estimators"])
            mlflow.log_param("lgbm_max_depth", LGBM_PARAMS["max_depth"])
            mlflow.log_param("lgbm_lr", LGBM_PARAMS["learning_rate"])
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_metric("perm_p_value", perm_p)
            for k, v in gates.items():
                mlflow.log_metric(f"gate_{k}", int(v))
            if feature_importance:
                for fname, imp in feature_importance.items():
                    mlflow.log_metric(f"fi_{fname}", imp)
            print(f"  [MLflow] Logged variant {variant}")
    except Exception as e:
        print(f"  [MLflow] Warning: {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 80)
    print("HEDGED ROTATION LGBM V1 — Walk-Forward LGBM Sector Ranking")
    print("=" * 80)

    # 1. Download data
    close, volume = download_data()

    # 2. Build features
    print("\nBuilding feature matrix...")
    features = build_features(close, volume)
    print(f"  Feature matrix: {len(features)} rows, features: {FEATURE_NAMES}")
    print(f"  Date range: {features.index.min()} to {features.index.max()}")

    # 3. Walk-forward LGBM training and prediction
    print("\nRunning walk-forward LGBM...")
    lgbm_preds = walk_forward_lgbm(features, close)

    if len(lgbm_preds) == 0:
        print("ERROR: No LGBM predictions generated. Check data availability.")
        return

    # Show sample predictions
    sample_date = lgbm_preds["date"].iloc[0]
    sample = lgbm_preds[lgbm_preds["date"] == sample_date].sort_values("rank", ascending=False)
    print(f"\n  Sample LGBM ranking for {sample_date}:")
    for _, row in sample.iterrows():
        print(f"    {row['sector']:>5}: pred_ret={row['predicted_return']:>8.4f}  rank={row['rank']:.0f}")

    # Get feature importance from last model (train one more for diagnostics)
    print("\n  Training final model for feature importance diagnostics...")
    train_end = pd.Timestamp(OOT_END) - pd.Timedelta(days=1)
    all_dates_before = features.index[features.index <= train_end].unique()
    train_dates = all_dates_before[-TRAIN_WINDOW:]
    train_mask = (features.index >= train_dates[0]) & (features.index <= train_end)
    train_df = features[train_mask].dropna(subset=FEATURE_NAMES + ["fwd_ret_21d"])
    diag_model = LGBMRegressor(**LGBM_PARAMS)
    diag_model.fit(train_df[FEATURE_NAMES].values, train_df["fwd_ret_21d"].values)
    fi = dict(zip(FEATURE_NAMES, diag_model.feature_importances_.tolist()))
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    print("  Feature importance (final model):")
    for fname, imp in fi_sorted:
        print(f"    {fname:<20s}: {imp:>6.0f}")

    # 4. Regime classification
    spy_ret = close[BENCHMARK].pct_change()
    regime = classify_regime(spy_ret)
    n_green = (regime == "green").sum()
    n_red = (regime == "red").sum()
    print(f"\nRegime: {n_green} green, {n_red} red, {(regime == 'flat').sum()} flat days")

    # 5. Random baseline
    print("\nComputing random selection baseline (200 runs)...")
    random_sharpe = random_selection_sharpe(close)
    print(f"  Random baseline Sharpe: {random_sharpe:.3f}")

    # 6. Run all strategies
    strategies = {
        "A_LGBM_LongShort": strategy_A_long_short,
        "B_LGBM_SPYHedged": strategy_B_spy_hedged,
        "C_LGBM_VIXTiming": strategy_C_vix_timing,
        "D_LGBM_DynHedge": strategy_D_dynamic_hedge,
        "E_LGBM_LS_VIXGate": strategy_E_ls_vix_gate,
        "F_LGBM_Conviction": strategy_F_conviction_weighted,
    }

    all_results = {}
    print("\n" + "=" * 80)
    print("RUNNING 6 STRATEGY VARIANTS")
    print("=" * 80)

    for name, strat_fn in strategies.items():
        print(f"\n{'─' * 60}")
        print(f"Strategy {name}")
        print(f"{'─' * 60}")

        weights = strat_fn(lgbm_preds, close)
        pf = simulate_portfolio(weights, close)
        metrics = compute_metrics(pf, regime)

        print("  Running permutation test (1000 shuffles)...")
        perm_p = permutation_test(pf)

        gates = five_gate_validation(metrics, perm_p, random_sharpe)

        print(f"  Sharpe:        {metrics['sharpe']:>8.3f}")
        print(f"  Sortino:       {metrics['sortino']:>8.3f}")
        print(f"  Profit Factor: {metrics['profit_factor']:>8.3f}")
        print(f"  Win Rate:      {metrics['win_rate']:>8.1%}")
        print(f"  CAGR:          {metrics['cagr']:>8.2%}")
        print(f"  MDD:           {metrics['mdd']:>8.2%}")
        print(f"  Final Value:   ${metrics['final_value']:>8.2f} (from ${INITIAL_CAPITAL})")
        print(f"  Total Return:  {metrics['total_return_pct']:>8.2f}%")
        print(f"  Sharpe (green): {metrics['sharpe_green']:>7.3f}  Sharpe (red): {metrics['sharpe_red']:>7.3f}  Gap: {metrics['regime_gap']:.4f}")
        print(f"  Perm p-value:  {perm_p:.4f}")
        print(f"  Random Sharpe: {random_sharpe:.3f}")
        print(f"\n  5-Gate Validation:")
        for g, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"    {g}: {status}")

        log_to_mlflow(name, metrics, gates, perm_p, fi)

        pf.to_csv(OUT_DIR / f"equity_{name}.csv")

        all_results[name] = {
            "metrics": metrics,
            "perm_p": round(perm_p, 4),
            "random_sharpe": round(random_sharpe, 3),
            "gates": gates,
        }

    # 7. Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    hdr = f"{'Variant':<24} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'CAGR':>7} {'MDD':>7} {'Final$':>8} {'RegGap':>7} {'PermP':>6} {'5-Gate':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in all_results.items():
        m = res["metrics"]
        g = res["gates"]
        verdict = "PASS" if g["all_pass"] else "FAIL"
        print(f"{name:<24} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>6.1%} {m['cagr']:>7.2%} {m['mdd']:>7.2%} "
              f"${m['final_value']:>7.2f} {m['regime_gap']:>7.4f} {res['perm_p']:>6.4f} {verdict:>7}")

    # 8. SPY buy-and-hold benchmark
    print(f"\n{'─' * 60}")
    print("BENCHMARK: SPY Buy-and-Hold")
    spy_oot = close[BENCHMARK].loc[OOT_START:OOT_END].dropna()
    spy_ret_oot = spy_oot.pct_change().dropna()
    spy_vals = INITIAL_CAPITAL * (1 + spy_ret_oot).cumprod()
    spy_sharpe = float(spy_ret_oot.mean() / spy_ret_oot.std() * np.sqrt(252))
    spy_final = float(spy_vals.iloc[-1])
    spy_mdd = float(((spy_vals - spy_vals.cummax()) / spy_vals.cummax()).min())
    print(f"  SPY Sharpe: {spy_sharpe:.3f}, Final: ${spy_final:.2f}, MDD: {spy_mdd:.2%}")

    # 9. Comparison vs v1 (momentum-based)
    print(f"\n{'─' * 60}")
    print("LGBM vs MOMENTUM COMPARISON")
    print("  (v1 used simple 21d momentum ranking; this uses walk-forward LGBM)")
    print("  v1 best: VIX Timing (D) Sharpe ~1.07, MDD -9.7%")
    best_lgbm = max(all_results.items(), key=lambda x: x[1]["metrics"]["sharpe"])
    print(f"  LGBM best: {best_lgbm[0]} Sharpe {best_lgbm[1]['metrics']['sharpe']:.3f}, MDD {best_lgbm[1]['metrics']['mdd']:.2%}")

    # 10. Save all results
    out_path = OUT_DIR / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "initial_capital": INITIAL_CAPITAL,
                "oot_start": OOT_START,
                "oot_end": OOT_END,
                "sectors": SECTORS,
                "train_window": TRAIN_WINDOW,
                "forward_period": FORWARD_PERIOD,
                "lgbm_params": {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"},
                "features": FEATURE_NAMES,
            },
            "spy_benchmark": {
                "sharpe": round(spy_sharpe, 3),
                "final_value": round(spy_final, 2),
                "mdd": round(spy_mdd, 4),
            },
            "random_baseline_sharpe": round(random_sharpe, 3),
            "feature_importance": fi,
            "variants": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Save LGBM predictions
    lgbm_preds.to_csv(OUT_DIR / "lgbm_predictions.csv", index=False)
    print(f"LGBM predictions saved to {OUT_DIR / 'lgbm_predictions.csv'}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # 11. Headline verdicts
    print("\n" + "=" * 80)
    print("HEADLINE VERDICTS")
    print("=" * 80)
    passing = [n for n, r in all_results.items() if r["gates"]["all_pass"]]
    failing = [n for n, r in all_results.items() if not r["gates"]["all_pass"]]
    if passing:
        print(f"  PASSED 5-gate: {', '.join(passing)}")
    else:
        print("  PASSED 5-gate: NONE")
    if failing:
        print(f"  FAILED 5-gate: {', '.join(failing)}")

    print(f"\n  Best risk-adjusted: {best_lgbm[0]} (Sharpe {best_lgbm[1]['metrics']['sharpe']:.3f})")
    print(f"  Does LGBM ranking improve hedged rotation?")
    if best_lgbm[1]["metrics"]["sharpe"] > 1.07:
        print(f"  YES — LGBM best Sharpe {best_lgbm[1]['metrics']['sharpe']:.3f} > momentum best 1.07")
    else:
        print(f"  UNCLEAR/NO — LGBM best Sharpe {best_lgbm[1]['metrics']['sharpe']:.3f} vs momentum best 1.07")


if __name__ == "__main__":
    main()
