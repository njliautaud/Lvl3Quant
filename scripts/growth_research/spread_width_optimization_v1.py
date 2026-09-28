#!/usr/bin/env python3
"""
Spread Width Optimization v1 — Production v4 Bull Call Spread Width Sweep
==========================================================================

Tests 8 spread width variants for the production v4 sector bull call spread strategy:
  1. 1% width — very narrow, cheap, high probability full payoff
  2. 2% width — narrow
  3. 3% width — current production standard
  4. 4% width — wider
  5. 5% width — matches VIX spike OTM strategy
  6. 7% width — wide, expensive, needs big move
  7. 10% width — very wide, most expensive
  8. Adaptive: width = 2% + 2% * (VIX/30)

Base strategy (production v4):
  - VIX > 20 regime filter (~33% of days)
  - LGBM walk-forward with 21 features (18 legacy + 3 cross-asset)
  - Top 3 sector ETFs from 11 sectors
  - Bull call spread: ATM long call, short at long_strike + width%
  - DTE=21, hold to expiry, biweekly rebalance
  - 15% entry haircut, no exit haircut, $2.60 commission, $645 start, $200 max/trade
  - BS pricing with IV = 1.2 * HV

Author: Claude (Opus 4.6), 2026-07-27
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


# ─── Platform-aware paths ───
if sys.platform == "win32":
    BASE = Path(r"C:\Users\claude\Lvl3Quant")
else:
    BASE = Path("/home/jupiter/Lvl3Quant")

OUTPUT_DIR = BASE / "research" / "findings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ───
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX"]
CAP = 645.0
DTE = 21
TOP_K = 3
MAX_POS = 200.0
COMMISSION_RT = 2.60
HAIRCUT = 0.15
RISK_FREE_RATE = 0.045
VIX_ENTRY_THRESHOLD = 20.0

# Walk-forward config
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "2W-FRI"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "spread_width_optimization_v1"
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

# Width variants
WIDTH_VARIANTS = {
    "1pct":      {"width_pct": 1.0,  "adaptive": False},
    "2pct":      {"width_pct": 2.0,  "adaptive": False},
    "3pct":      {"width_pct": 3.0,  "adaptive": False},
    "4pct":      {"width_pct": 4.0,  "adaptive": False},
    "5pct":      {"width_pct": 5.0,  "adaptive": False},
    "7pct":      {"width_pct": 7.0,  "adaptive": False},
    "10pct":     {"width_pct": 10.0, "adaptive": False},
    "adaptive":  {"width_pct": None, "adaptive": True},
}


# ══════════════════════════════════════════════════════════════
# BLACK-SCHOLES PRICING (inline to avoid import issues on Razer)
# ══════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def estimate_iv(atr, spot, vix=20.0, atr_period=14):
    if spot <= 0 or atr <= 0:
        return 0.25
    realized_vol = (atr / spot) * np.sqrt(252 / atr_period)
    iv_mult = 1.2 + 0.01 * max(vix - 20.0, 0.0)
    sigma = realized_vol * iv_mult
    return max(sigma, 0.10)


def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0):
    """Returns (entry_cost_per_share, max_profit_per_share)."""
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix)
    fair_value = bs_call_price(S, K1, T, RISK_FREE_RATE, sigma) - bs_call_price(S, K2, T, RISK_FREE_RATE, sigma)
    fair_value = max(fair_value, 0.001)
    entry_cost = fair_value * (1.0 + HAIRCUT)
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost
    return float(entry_cost), float(max_profit)


def compute_atr_val(high_s, low_s, close_s, period=14):
    """Compute most recent ATR from series."""
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_series


# ══════════════════════════════════════════════════════════════
# DATA
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

    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


# ══════════════════════════════════════════════════════════════
# FEATURES
# ══════════════════════════════════════════════════════════════

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

ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES


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
# WALK-FORWARD LGBM
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates):
    """Build feature + target records for VIX>20 dates only."""
    fprint(f"  Building feature records: {len(rebal_dates)} rebal dates...")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        cv = float(vix.iloc[idx]) if vix is not None else 15.0
        if cv < VIX_ENTRY_THRESHOLD:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = compute_cross_asset_features(tk, idx, close)

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk,
                   "fwd_ret": fwd_ret, "vix": cv}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} active dates (VIX>{VIX_ENTRY_THRESHOLD})")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
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

        Xt = np.nan_to_num(train_df[ALL_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[ALL_FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            vix_val = float(test_df["vix"].iloc[0]) if "vix" in test_df.columns else 20.0
            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
                "vix": vix_val,
            }
        except Exception:
            continue

    fprint(f"    {len(rankings)} ranking dates from walk-forward")
    return rankings


# ══════════════════════════════════════════════════════════════
# ATR SERIES
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
# TRADE SIMULATION (parameterized by width)
# ══════════════════════════════════════════════════════════════

def get_width_pct(variant_cfg, vix_val):
    """Return the spread width % for this variant given current VIX."""
    if variant_cfg["adaptive"]:
        return 2.0 + 2.0 * (vix_val / 30.0)
    return variant_cfg["width_pct"]


def simulate_trades_width(variant_name, variant_cfg, rankings, close, atr_dict):
    """
    Simulate bull call spreads with a specific width variant.

    Returns: trades list, final equity, and per-trade detail metrics.
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    total_commission = 0.0
    total_spread_costs = []
    total_contracts_per_trade = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]["scores"]
        if not scores:
            continue

        width_pct = get_width_pct(variant_cfg, cv)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        max_pos = min(MAX_POS, equity / 3)
        if max_pos < 30:
            continue

        for tk in picks:
            if tk not in close.columns or tk not in atr_dict:
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
            K2 = round(S * (1 + width_pct / 100), 2)
            if K2 <= K1:
                K2 = K1 + 0.50

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            cost_per_contract = entry_cost_ps * 100 + COMMISSION_RT
            if cost_per_contract <= 0:
                continue

            # How many contracts can we buy with max_pos?
            n_contracts = max(1, int(max_pos / cost_per_contract))
            total_cost = cost_per_contract * n_contracts

            if total_cost > equity * 0.40 or total_cost > max_pos * 1.1:
                n_contracts = max(1, int(min(equity * 0.40, max_pos) / cost_per_contract))
                total_cost = cost_per_contract * n_contracts

            if total_cost <= 0 or n_contracts < 1:
                continue

            # Expiry intrinsic
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)

            # PnL
            pnl = (intrinsic - entry_cost_ps) * 100 * n_contracts - COMMISSION_RT * n_contracts
            equity += pnl

            comm_this = COMMISSION_RT * n_contracts
            total_commission += comm_this

            # Track per-trade metrics
            spread_width_dollars = (K2 - K1) * 100  # per contract
            cost_pct_of_max = (entry_cost_ps * 100) / (spread_width_dollars) if spread_width_dollars > 0 else 1.0

            total_spread_costs.append(cost_pct_of_max)
            total_contracts_per_trade.append(n_contracts)

            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": "bull" if se >= sv else "bear",
                "vix": round(cv, 1),
                "width_pct": round(width_pct, 2),
                "n_contracts": n_contracts,
                "entry_cost_per_contract": round(entry_cost_ps * 100, 2),
                "spread_width_dollars": round(spread_width_dollars, 2),
                "cost_pct_of_max": round(cost_pct_of_max, 4),
                "commission": round(comm_this, 2),
                "win": pnl > 0,
            })

    return trades, equity, total_commission, total_spread_costs, total_contracts_per_trade


# ══════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades, initial_capital=CAP):
    """Compute Sharpe, Sortino, CAGR, MaxDD, WR, PF from trade list."""
    if not trades or len(trades) < 5:
        return {
            "sharpe": 0, "sortino": 0, "cagr": 0, "max_dd": -1,
            "win_rate": 0, "profit_factor": 0, "n_trades": len(trades) if trades else 0,
            "final_equity": initial_capital,
        }

    pnls = [t["pnl"] for t in trades]
    equity_curve = [initial_capital]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)
    equity_curve = np.array(equity_curve)

    # Equity-based returns
    eq_rets = np.diff(equity_curve) / equity_curve[:-1]
    eq_rets = eq_rets[np.isfinite(eq_rets)]

    if len(eq_rets) < 5:
        return {
            "sharpe": 0, "sortino": 0, "cagr": 0, "max_dd": -1,
            "win_rate": 0, "profit_factor": 0, "n_trades": len(trades),
            "final_equity": equity_curve[-1],
        }

    # Annualization: biweekly rebalance with TOP_K trades = ~26 periods/year * 3 trades
    trades_per_year = 26 * TOP_K
    ann_factor = np.sqrt(trades_per_year)

    sharpe = float(eq_rets.mean() / (eq_rets.std() + 1e-10) * ann_factor)
    neg_rets = eq_rets[eq_rets < 0]
    sortino = float(eq_rets.mean() / (neg_rets.std() + 1e-10) * ann_factor) if len(neg_rets) > 0 else sharpe * 2

    # Date-based CAGR
    first_date = pd.Timestamp(trades[0]["entry_date"])
    last_date = pd.Timestamp(trades[-1]["exit_date"])
    years = (last_date - first_date).days / 365.25
    if years > 0:
        cagr = float((equity_curve[-1] / initial_capital) ** (1 / years) - 1)
    else:
        cagr = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = float(drawdowns.min())

    # Win rate, profit factor
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "n_trades": len(trades),
        "final_equity": round(equity_curve[-1], 2),
    }


# ══════════════════════════════════════════════════════════════
# ADVERSARIAL VALIDATION (5 gates, inline)
# ══════════════════════════════════════════════════════════════

def adversarial_validate(trades, spy_prices, initial_capital=CAP, n_perms=2000):
    """Run 5-gate adversarial validation."""
    if not trades or len(trades) < 20:
        return {"gates_passed": 0, "gates_total": 5, "gates": {}, "error": "Too few trades"}

    pnls = np.array([t["pnl"] for t in trades])
    equity_curve = np.cumsum(pnls) + initial_capital
    eq_rets = np.diff(np.concatenate([[initial_capital], equity_curve])) / np.concatenate([[initial_capital], equity_curve[:-1]])

    gates = {}

    # Gate 1: Sign-flip permutation test
    real_sharpe = eq_rets.mean() / (eq_rets.std() + 1e-10)
    perm_sharpes = []
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnls))
        perm_pnls = pnls * signs
        perm_eq = np.cumsum(perm_pnls) + initial_capital
        perm_rets = np.diff(np.concatenate([[initial_capital], perm_eq])) / np.concatenate([[initial_capital], perm_eq[:-1]])
        perm_sharpes.append(perm_rets.mean() / (perm_rets.std() + 1e-10))
    p_val = (np.sum(np.array(perm_sharpes) >= real_sharpe) + 1) / (n_perms + 1)
    gates["sign_flip"] = {"passed": p_val < 0.05, "p_value": round(float(p_val), 4)}

    # Gate 2: Regime balance
    bull_pnls = [t["pnl"] for t in trades if t.get("regime") == "bull"]
    bear_pnls = [t["pnl"] for t in trades if t.get("regime") == "bear"]
    bull_mean = np.mean(bull_pnls) if bull_pnls else 0
    bear_mean = np.mean(bear_pnls) if bear_pnls else 0
    gates["regime_balance"] = {
        "passed": (len(bull_pnls) > 5 and len(bear_pnls) > 5 and
                   bull_mean > -abs(bear_mean) * 2),
        "bull_mean_pnl": round(float(bull_mean), 2),
        "bear_mean_pnl": round(float(bear_mean), 2),
        "bull_n": len(bull_pnls),
        "bear_n": len(bear_pnls),
    }

    # Gate 3: Sub-period stability
    mid = len(trades) // 2
    first_half = compute_metrics(trades[:mid], initial_capital)
    second_half = compute_metrics(trades[mid:], initial_capital)
    gates["sub_period"] = {
        "passed": first_half["sharpe"] > 0 and second_half["sharpe"] > 0,
        "first_half_sharpe": first_half["sharpe"],
        "second_half_sharpe": second_half["sharpe"],
    }

    # Gate 4: Outlier removal — remove best month, still profitable?
    trade_df = pd.DataFrame(trades)
    trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
    monthly = trade_df.groupby(trade_df["entry_date"].dt.to_period("M"))["pnl"].sum()
    if len(monthly) > 2:
        best_month = monthly.idxmax()
        without_best = monthly.drop(best_month)
        gates["outlier_removal"] = {
            "passed": float(without_best.sum()) > 0,
            "total_pnl_without_best_month": round(float(without_best.sum()), 2),
            "best_month_pnl": round(float(monthly.max()), 2),
        }
    else:
        gates["outlier_removal"] = {"passed": False, "error": "Too few months"}

    # Gate 5: Yearly consistency
    yearly = trade_df.groupby(trade_df["entry_date"].dt.year)["pnl"].sum()
    profitable_years = (yearly > 0).sum()
    total_years = len(yearly)
    gates["yearly_consistency"] = {
        "passed": profitable_years / total_years >= 0.60 if total_years > 0 else False,
        "profitable_years": int(profitable_years),
        "total_years": int(total_years),
        "pct": round(float(profitable_years / total_years), 2) if total_years > 0 else 0,
    }

    n_passed = sum(1 for g in gates.values() if g.get("passed", False))
    return {"gates_passed": n_passed, "gates_total": 5, "gates": gates}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"SPREAD WIDTH OPTIMIZATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Top-{TOP_K} sectors | VIX>{VIX_ENTRY_THRESHOLD}")
    fprint(f"Haircut: {HAIRCUT:.0%} entry only | Hold to expiry | Commission: ${COMMISSION_RT:.2f}")
    fprint(f"Features: {len(ALL_FEATURES)} (18 legacy + 3 cross-asset)")
    fprint(f"Width variants: {list(WIDTH_VARIANTS.keys())}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. ATR series
    atr_dict = compute_atr_series(high, low, close)

    # 3. Rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    # 4. Build features + walk-forward LGBM ranking (shared across all widths)
    fprint("\n--- WALK-FORWARD LGBM RANKING ---")
    records = build_feature_records(close, high, low, rebal_dates)
    rankings = walk_forward_lgbm_rank(records)

    if len(rankings) < 10:
        fprint("ERROR: Too few ranking dates. Check data.")
        return

    # 5. Run all width variants
    results = {}
    all_trades_by_variant = {}

    fprint("\n" + "=" * 80)
    fprint("RUNNING 8 WIDTH VARIANTS")
    fprint("=" * 80)

    for vname, vcfg in WIDTH_VARIANTS.items():
        fprint(f"\n--- {vname.upper()} ---")
        if vcfg["adaptive"]:
            fprint(f"  Width: adaptive = 2% + 2%*(VIX/30)")
        else:
            fprint(f"  Width: {vcfg['width_pct']:.1f}%")

        trades, final_eq, total_comm, cost_ratios, contracts_list = simulate_trades_width(
            vname, vcfg, rankings, close, atr_dict
        )

        metrics = compute_metrics(trades, CAP)
        fprint(f"  Trades: {metrics['n_trades']} | Equity: ${CAP:.0f}->${metrics['final_equity']:.0f}")
        fprint(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f} | "
               f"CAGR: {metrics['cagr']:.1%} | MaxDD: {metrics['max_dd']:.1%}")
        fprint(f"  WR: {metrics['win_rate']:.1%} | PF: {metrics['profit_factor']:.2f}")

        # Width-specific analysis
        if cost_ratios:
            avg_cost_pct = np.mean(cost_ratios)
            avg_contracts = np.mean(contracts_list)
            comm_drag = total_comm / max(sum(t["pnl"] for t in trades if t["pnl"] > 0), 1.0)
        else:
            avg_cost_pct = 0
            avg_contracts = 0
            comm_drag = 0

        fprint(f"  Avg cost as % of max payoff: {avg_cost_pct:.1%}")
        fprint(f"  Avg contracts/trade: {avg_contracts:.1f}")
        fprint(f"  Commission drag: {comm_drag:.1%} of gross profit")

        # Win rate by width bucket (for adaptive)
        if vcfg["adaptive"] and trades:
            widths = [t["width_pct"] for t in trades]
            narrow = [t for t in trades if t["width_pct"] < 3.5]
            wide = [t for t in trades if t["width_pct"] >= 3.5]
            fprint(f"  Adaptive width range: {min(widths):.1f}%-{max(widths):.1f}%")
            if narrow:
                fprint(f"    Narrow (<3.5%): WR={sum(1 for t in narrow if t['win'])/len(narrow):.1%}, "
                       f"avg PnL=${np.mean([t['pnl'] for t in narrow]):.2f}")
            if wide:
                fprint(f"    Wide (>=3.5%):  WR={sum(1 for t in wide if t['win'])/len(wide):.1%}, "
                       f"avg PnL=${np.mean([t['pnl'] for t in wide]):.2f}")

        results[vname] = {
            **metrics,
            "avg_cost_pct_of_max": round(float(avg_cost_pct), 4),
            "avg_contracts_per_trade": round(float(avg_contracts), 2),
            "commission_drag_pct": round(float(comm_drag), 4),
            "total_commission": round(float(total_comm), 2),
        }
        all_trades_by_variant[vname] = trades

    # 6. Summary table
    fprint("\n" + "=" * 80)
    fprint("SUMMARY TABLE — ALL WIDTH VARIANTS")
    fprint("=" * 80)
    header = f"{'Width':<12} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} {'Trades':>7} {'Final$':>8} {'CostPct':>8} {'AvgCntr':>8} {'CommDrag':>9}"
    fprint(header)
    fprint("-" * len(header))

    sorted_variants = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for vname, r in sorted_variants:
        fprint(f"{vname:<12} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['cagr']:>6.1%} "
               f"{r['max_dd']:>6.1%} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
               f"{r['n_trades']:>7d} {r['final_equity']:>8.0f} "
               f"{r['avg_cost_pct_of_max']:>7.1%} {r['avg_contracts_per_trade']:>8.1f} "
               f"{r['commission_drag_pct']:>8.1%}")

    # 7. Top 3 adversarial validation
    fprint("\n" + "=" * 80)
    fprint("5-GATE ADVERSARIAL VALIDATION — TOP 3 VARIANTS")
    fprint("=" * 80)

    top3 = [vname for vname, _ in sorted_variants[:3]]
    spy_prices = close["SPY"]

    for vname in top3:
        trades = all_trades_by_variant[vname]
        fprint(f"\n--- {vname.upper()} (Sharpe={results[vname]['sharpe']:.2f}) ---")
        adv = adversarial_validate(trades, spy_prices, CAP)
        results[vname]["adversarial"] = adv
        fprint(f"  Gates passed: {adv['gates_passed']}/{adv['gates_total']}")
        for gname, gdata in adv.get("gates", {}).items():
            status = "PASS" if gdata.get("passed") else "FAIL"
            detail_parts = [f"{k}={v}" for k, v in gdata.items() if k != "passed"]
            fprint(f"    [{status}] {gname}: {', '.join(detail_parts)}")

    # 8. Key dynamics analysis
    fprint("\n" + "=" * 80)
    fprint("KEY DYNAMICS ANALYSIS")
    fprint("=" * 80)

    # Width vs WR relationship
    fixed_variants = [(vname, results[vname]) for vname in ["1pct", "2pct", "3pct", "4pct", "5pct", "7pct", "10pct"]
                      if vname in results]
    if fixed_variants:
        fprint("\nWidth vs Win Rate (wider = lower WR but higher payoff?):")
        for vname, r in fixed_variants:
            avg_win = np.mean([t["pnl"] for t in all_trades_by_variant[vname] if t["pnl"] > 0]) if any(t["pnl"] > 0 for t in all_trades_by_variant[vname]) else 0
            avg_loss = np.mean([t["pnl"] for t in all_trades_by_variant[vname] if t["pnl"] <= 0]) if any(t["pnl"] <= 0 for t in all_trades_by_variant[vname]) else 0
            fprint(f"  {vname}: WR={r['win_rate']:.1%}, avg_win=${avg_win:.2f}, avg_loss=${avg_loss:.2f}, "
                   f"payoff_ratio={abs(avg_win/(avg_loss+1e-10)):.2f}")

    # Optimal width conclusion
    best = sorted_variants[0]
    fprint(f"\nBest overall: {best[0]} (Sharpe={best[1]['sharpe']:.2f})")
    baseline = results.get("3pct", {})
    if baseline:
        delta = best[1]["sharpe"] - baseline.get("sharpe", 0)
        fprint(f"vs 3% baseline: Sharpe delta = {delta:+.2f}")

    # 9. MLflow logging
    if MLFLOW_OK:
        fprint("\n--- MLflow Logging ---")
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            for vname, r in results.items():
                with mlflow.start_run(run_name=f"width_{vname}"):
                    mlflow.log_params({
                        "variant": vname,
                        "width_pct": WIDTH_VARIANTS[vname].get("width_pct", "adaptive"),
                        "adaptive": WIDTH_VARIANTS[vname]["adaptive"],
                        "dte": DTE, "top_k": TOP_K, "cap": CAP,
                        "haircut": HAIRCUT, "commission": COMMISSION_RT,
                        "vix_threshold": VIX_ENTRY_THRESHOLD,
                    })
                    mlflow.log_metrics({
                        "sharpe": r["sharpe"],
                        "sortino": r["sortino"],
                        "cagr": r["cagr"],
                        "max_dd": r["max_dd"],
                        "win_rate": r["win_rate"],
                        "profit_factor": r["profit_factor"],
                        "n_trades": r["n_trades"],
                        "final_equity": r["final_equity"],
                        "avg_cost_pct_of_max": r["avg_cost_pct_of_max"],
                        "avg_contracts_per_trade": r["avg_contracts_per_trade"],
                        "commission_drag_pct": r["commission_drag_pct"],
                    })
            fprint(f"  Logged {len(results)} runs to MLflow experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"  MLflow logging failed: {e}")

    # 10. Save results
    output = {
        "timestamp": t0.isoformat(),
        "config": {
            "cap": CAP, "dte": DTE, "top_k": TOP_K,
            "haircut": HAIRCUT, "commission_rt": COMMISSION_RT,
            "vix_threshold": VIX_ENTRY_THRESHOLD,
            "features": len(ALL_FEATURES),
            "wf_train_periods": WF_TRAIN_PERIODS,
        },
        "results": results,
        "ranking_sorted": [vname for vname, _ in sorted_variants],
        "elapsed_seconds": round((datetime.now() - t0).total_seconds(), 1),
    }

    out_file = OUTPUT_DIR / "spread_width_optimization_v1_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to: {out_file}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE.")


if __name__ == "__main__":
    main()
