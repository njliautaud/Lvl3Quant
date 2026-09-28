#!/usr/bin/env python3
"""
Moneyness Cross-Validation v1 — Strike Offset Sensitivity Analysis
====================================================================

Based on production_v4_honest_test.py (CANONICAL, Sharpe ~1.87 at ATM).

Tests 7 moneyness offsets to find optimal strike placement:
  A: 2% ITM  (offset = -2%)  -> buy call at S-2%, sell at S+1%
  B: 1% ITM  (offset = -1%)  -> buy call at S-1%, sell at S+2%
  C: ATM     (offset =  0%)  -> buy call at S,    sell at S+3%  (reproduces prod v4)
  D: 1% OTM  (offset = +1%)  -> buy call at S+1%, sell at S+4%
  E: 2% OTM  (offset = +2%)  -> buy call at S+2%, sell at S+5%
  F: 3% OTM  (offset = +3%)  -> buy call at S+3%, sell at S+6%
  G: 5% OTM  (offset = +5%)  -> buy call at S+5%, sell at S+8%

ALL logic is identical to production v4 (walk-forward LGBM, GRU regime filter,
hold-to-expiry pricing, 15% entry haircut, intrinsic-only at expiry, DTE=21,
$645 starting capital, biweekly rebalance). ONLY the strike offset changes.

Uses Variant A config from prod v4 (regime>0.4, 18 legacy features, bull only)
since that is the canonical ATM baseline producing Sharpe ~1.87.
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


# -- Standardized tools --
# Detect which node we're on and set paths accordingly
import os
_hostname = os.uname().nodename.lower()
if "neptune" in _hostname or "nick" in str(Path.home()):
    _BASE_STR = "/home/nick/Lvl3Quant"
else:
    _BASE_STR = "/home/jupiter/Lvl3Quant"
sys.path.insert(0, _BASE_STR)

try:
    from research.tools.options_pricer import (
        price_bull_call_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    from research.tools.adversarial_validator import validate_trades
except ImportError:
    fprint("WARNING: Could not import research.tools — defining inline")
    COMMISSION_RT_SPREAD = 2.60
    DEFAULT_HAIRCUT = 0.15

    def compute_atr(prices, window=14):
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        tr = high - low
        return tr.rolling(window).mean()

    def estimate_iv(atr, price, dte=21, mult=1.2):
        daily_vol = atr / price
        annual_vol = daily_vol * np.sqrt(252) * mult
        return annual_vol

    def price_bull_call_spread(S, K_long, K_short, iv, dte, r=0.05):
        from scipy.stats import norm
        T = dte / 365.0
        if T <= 0 or iv <= 0:
            return 0.0, 0.0
        d1_l = (np.log(S / K_long) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2_l = d1_l - iv * np.sqrt(T)
        c_long = S * norm.cdf(d1_l) - K_long * np.exp(-r * T) * norm.cdf(d2_l)
        d1_s = (np.log(S / K_short) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2_s = d1_s - iv * np.sqrt(T)
        c_short = S * norm.cdf(d1_s) - K_short * np.exp(-r * T) * norm.cdf(d2_s)
        debit = c_long - c_short
        max_profit = K_short - K_long - debit
        return max(debit, 0.001), max(max_profit, 0)

    def validate_trades(trades, n_perms=1000):
        """Inline 5-gate adversarial validation."""
        if not trades:
            return {"gates_passed": 0, "total_gates": 5, "details": {}}
        pnls = [t.get("pnl", 0) for t in trades]
        total = sum(pnls)
        # Gate 1: Sign-flip permutation
        n_beat = sum(1 for _ in range(n_perms) if sum(p * np.random.choice([-1, 1]) for p in pnls) >= total)
        g1 = n_beat / n_perms < 0.05
        # Gate 2: Regime balance
        bull_pnl = [t["pnl"] for t in trades if t.get("regime", "bull") == "bull"]
        bear_pnl = [t["pnl"] for t in trades if t.get("regime", "bull") == "bear"]
        bull_wr = np.mean([1 for p in bull_pnl if p > 0]) if bull_pnl else 0.5
        bear_wr = np.mean([1 for p in bear_pnl if p > 0]) if bear_pnl else 0.5
        g2 = abs(bull_wr - bear_wr) < 0.50
        # Gate 3: Sub-period
        mid = len(pnls) // 2
        g3 = sum(pnls[:mid]) > 0 and sum(pnls[mid:]) > 0
        # Gate 4: Outlier removal
        n_remove = max(1, len(pnls) // 20)
        sorted_pnls = sorted(pnls)
        g4 = sum(sorted_pnls[n_remove:-n_remove]) > 0 if len(sorted_pnls) > 2*n_remove else True
        # Gate 5: Yearly consistency
        years = {}
        for t in trades:
            y = t.get("date", "2020-01-01")[:4]
            years.setdefault(y, []).append(t["pnl"])
        profitable_years = sum(1 for y, ps in years.items() if sum(ps) > 0)
        g5 = profitable_years / max(len(years), 1) >= 0.60
        passed = sum([g1, g2, g3, g4, g5])
        return {
            "gates_passed": passed,
            "total_gates": 5,
            "details": {"sign_flip": g1, "regime_balance": g2, "sub_period": g3, "outlier": g4, "yearly": g5}
        }

# -- Config (IDENTICAL to production v4) --
BASE = Path(_BASE_STR)
OUTPUT_DIR = BASE / "output" / "growth_research" / "moneyness_xval_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4

# Regime predictions path — works on both Jupiter and Neptune
REGIME_FILE_CANDIDATES = [
    Path("/home/nick/Lvl3Quant/output/regime_detector_v1/regime_predictions_v1.npz"),
    BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz",
]

# LGBM walk-forward
WF_TRAIN_PERIODS = 12
WF_REBAL_FREQ = "2W-FRI"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "moneyness_xval_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")

# -- Moneyness variants to test --
MONEYNESS_VARIANTS = [
    ("A_ITM2",  -2.0, "2% ITM (offset=-2%)"),
    ("B_ITM1",  -1.0, "1% ITM (offset=-1%)"),
    ("C_ATM",    0.0, "ATM (offset=0%, baseline)"),
    ("D_OTM1",  +1.0, "1% OTM (offset=+1%)"),
    ("E_OTM2",  +2.0, "2% OTM (offset=+2%)"),
    ("F_OTM3",  +3.0, "3% OTM (offset=+3%)"),
    ("G_OTM5",  +5.0, "5% OTM (offset=+5%)"),
]


# ================================================================
# DATA DOWNLOAD (identical to production v4)
# ================================================================

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


# ================================================================
# REGIME LOADING (identical to production v4)
# ================================================================

def load_regime_predictions():
    """Load GRU regime predictions and build a date-indexed Series."""
    regime_file = None
    for candidate in REGIME_FILE_CANDIDATES:
        if candidate.exists():
            regime_file = candidate
            break

    if regime_file is None:
        fprint(f"WARNING: Regime file not found at any candidate path")
        fprint("  Will use VIX-based regime proxy instead")
        return None

    data = np.load(regime_file, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days "
           f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")
    fprint(f"  Mean score: {regime_series.mean():.3f}, "
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}")
    return regime_series


def get_regime_score_at(regime_series, dt):
    """Get regime score at a given date, with nearest-date fallback."""
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ================================================================
# FEATURE ENGINEERING (identical to production v4, legacy 18 only)
# ================================================================

LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]


def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
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


# ================================================================
# WALK-FORWARD LGBM RANKING (identical to production v4)
# ================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Bull-only mode (regime>0.4), identical to prod v4 Variant A.
    """
    import lightgbm as lgb

    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

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
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols, variant_name):
    """Walk-forward LGBM ranking: sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
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
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            rankings[test_date] = {
                "scores": dict(zip(test_df["ticker"], test_df["score"])),
            }

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

    fprint(f"    {variant_name}: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ================================================================
# ATR COMPUTATION (identical to production v4)
# ================================================================

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
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ================================================================
# TRADE SIMULATION — WITH MONEYNESS OFFSET
# ================================================================

def simulate_trades_with_moneyness(name, rankings, close, high, low, regime_series,
                                    atr_dict, moneyness_offset_pct):
    """
    Simulate bull call spreads from rankings with a moneyness offset.

    IDENTICAL to production v4 simulate_trades() except for strike selection:
      - K1 = S * (1 + moneyness_offset_pct / 100)
      - K2 = K1 * (1 + SPREAD_PCT / 100)   ... WAIT, that changes spread width.

    Actually, to keep spread WIDTH constant at 3% of S:
      - K1 = S * (1 + moneyness_offset_pct / 100)
      - K2 = K1 + S * SPREAD_PCT / 100
    This keeps the dollar width of the spread identical regardless of offset.

    HONEST RULES (same as prod v4):
      - Hold to expiry
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        ranking_data = rankings[dt]
        scores = ranking_data["scores"]

        if not scores:
            continue

        # Top K by LGBM score (highest predicted forward return) — bull only
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        # Position sizing: fixed, max $200 per trade or 1/3 of equity
        max_pos = min(200, equity / 3)
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

            # ATR for pricing
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                av = float(atr_dict[tk].loc[dt])
            else:
                av = S * 0.015

            # MONEYNESS OFFSET: shift strikes by offset percentage of S
            # K1 = long call strike, K2 = short call strike
            # Spread width stays constant at SPREAD_PCT% of S
            K1 = round(S * (1 + moneyness_offset_pct / 100), 2)
            K2 = round(K1 + S * SPREAD_PCT / 100, 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
                continue

            # HOLD TO EXPIRY: compute intrinsic value at expiry
            Se = float(close[tk].iloc[ei])

            # Bull call spread intrinsic at expiry
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)

            exit_value_ps = intrinsic

            # PnL: exit value - entry cost - commission (no exit haircut at expiry)
            pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
            equity += pnl
            n_entered += 1

            # Regime classification for trade record
            sv = float(spy.loc[dt]) if dt in spy.index else 0
            se = float(spy.iloc[ei]) if ei < len(spy) else sv
            spy_regime = "bull" if se >= sv else "bear"

            trades.append({
                "pnl": round(pnl, 2),
                "entry_date": str(dt.date()),
                "exit_date": str(close.index[ei].date()),
                "ticker": tk,
                "regime": spy_regime,
                "direction": "bull",
                "vix": round(cv, 1),
                "win": pnl > 0,
                "K1": K1,
                "K2": K2,
                "S_entry": round(S, 2),
                "S_exit": round(Se, 2),
                "moneyness_offset_pct": moneyness_offset_pct,
            })

    return trades, equity


# ================================================================
# RANDOM BASELINE (identical to production v4)
# ================================================================

def random_baseline_test(rankings, close, high, low, regime_series, atr_dict,
                         moneyness_offset_pct, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, data in rankings.items():
            rand_scores = {tk: np.random.random() for tk in data["scores"].keys()}
            rand_rankings[dt] = {"scores": rand_scores}

        trades, final_eq = simulate_trades_with_moneyness(
            f"Random_{trial}", rand_rankings, close, high, low,
            regime_series, atr_dict, moneyness_offset_pct=moneyness_offset_pct,
        )

        if trades and len(trades) >= 10:
            result = validate_trades(
                trades, initial_capital=CAP,
                spy_prices=close["SPY"],
                strategy_name=f"Random_{trial}",
                n_perms=500,
            )
            random_sharpes.append(result.sharpe)
            fprint(f"    Random trial {trial}: Sharpe {result.sharpe:.2f}, "
                   f"${CAP:.0f}->${final_eq:.0f}")
        else:
            random_sharpes.append(0.0)

    return random_sharpes


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"MONEYNESS CROSS-VALIDATION v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread width: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}")
    fprint(f"Using Variant A config (18 legacy features, bull only, regime>0.4)")
    fprint(f"Testing {len(MONEYNESS_VARIANTS)} moneyness offsets")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 5. Build feature records + LGBM rankings (ONCE — rankings are the same
    #    for all moneyness variants since we're only changing strikes, not signals)
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (shared across all moneyness variants)")
    fprint("=" * 80)

    records = build_feature_records(
        close, high, low, rebal_dates, LEGACY_FEATURES, regime_series,
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, LEGACY_FEATURES, "moneyness_xval")

    if not rankings:
        fprint("FATAL: No rankings produced. Cannot proceed.")
        return

    # 6. Run each moneyness variant
    all_results = {}

    for vname, offset_pct, desc in MONEYNESS_VARIANTS:
        fprint("\n" + "=" * 80)
        fprint(f"VARIANT {vname}: {desc}")
        fprint(f"  Strike logic: K1 = S * (1 + {offset_pct:.1f}%), "
               f"K2 = K1 + S * {SPREAD_PCT:.0f}%")
        fprint("=" * 80)

        trades, final_eq = simulate_trades_with_moneyness(
            vname, rankings, close, high, low, regime_series, atr_dict,
            moneyness_offset_pct=offset_pct,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc,
                "moneyness_offset_pct": offset_pct,
                "n_trades": len(trades) if trades else 0,
                "error": "insufficient trades",
            }
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, regime_series, atr_dict,
            moneyness_offset_pct=offset_pct,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        # Average trade stats
        avg_pnl = np.mean([t["pnl"] for t in trades])
        median_pnl = np.median([t["pnl"] for t in trades])
        avg_k1_moneyness = np.mean([(t["K1"] / t["S_entry"] - 1) * 100 for t in trades])

        all_results[vname] = {
            "description": desc,
            "moneyness_offset_pct": offset_pct,
            **result.to_dict(),
            "avg_pnl": round(avg_pnl, 2),
            "median_pnl": round(median_pnl, 2),
            "avg_actual_k1_moneyness_pct": round(avg_k1_moneyness, 2),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
        }

    # -- SUMMARY COMPARISON --
    fprint("\n" + "=" * 80)
    fprint("MONEYNESS CROSS-VALIDATION SUMMARY")
    fprint("=" * 80)
    fprint(f"{'Variant':<12} {'Offset':>7} {'Trades':>7} {'Sharpe':>7} {'Sortino':>8} "
           f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'CAGR':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 105)

    for vname, offset_pct, desc in MONEYNESS_VARIANTS:
        r = all_results.get(vname, {})
        if "error" in r:
            fprint(f"  {vname:<12} {offset_pct:>+6.0f}%   -- {r.get('error', 'NO DATA')} --")
            continue
        fprint(f"  {vname:<12} {offset_pct:>+6.0f}% {r['n_trades']:>6} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['cagr']*100:>6.1f}% "
               f"{r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # Best variant analysis
    fprint("\n" + "=" * 80)
    fprint("BEST VARIANT ANALYSIS")
    fprint("=" * 80)

    valid_results = {k: v for k, v in all_results.items() if "error" not in v and v.get("sharpe") is not None}
    if valid_results:
        best_sharpe_name = max(valid_results, key=lambda k: valid_results[k]["sharpe"])
        best = valid_results[best_sharpe_name]
        atm = valid_results.get("C_ATM", {})

        fprint(f"  Best by Sharpe: {best_sharpe_name} (Sharpe {best['sharpe']:.2f})")
        if atm:
            fprint(f"  ATM baseline:   C_ATM (Sharpe {atm['sharpe']:.2f})")
            delta = best['sharpe'] - atm['sharpe']
            if best_sharpe_name != "C_ATM":
                fprint(f"  Improvement:    {delta:+.2f} Sharpe ({delta/atm['sharpe']*100:+.1f}%)")
            else:
                fprint(f"  ATM IS the best -- no improvement from shifting moneyness")

        # Best by Sortino
        best_sortino_name = max(valid_results, key=lambda k: valid_results[k]["sortino"])
        fprint(f"  Best by Sortino: {best_sortino_name} (Sortino {valid_results[best_sortino_name]['sortino']:.2f})")

        # Best by profit factor
        best_pf_name = max(valid_results, key=lambda k: valid_results[k]["profit_factor"])
        fprint(f"  Best by PF: {best_pf_name} (PF {valid_results[best_pf_name]['profit_factor']:.2f})")

        # Sanity check: ATM should be ~1.87 Sharpe
        if atm:
            if abs(atm["sharpe"] - 1.87) > 0.3:
                fprint(f"\n  WARNING: ATM Sharpe is {atm['sharpe']:.2f}, expected ~1.87. "
                       f"Delta = {atm['sharpe'] - 1.87:+.2f}")
                fprint(f"  This may indicate a bug if the delta is large.")
            else:
                fprint(f"\n  SANITY CHECK PASSED: ATM Sharpe {atm['sharpe']:.2f} is within "
                       f"0.3 of expected 1.87")

    # Feature importance
    if imp_df is not None:
        fprint("\n" + "=" * 80)
        fprint("FEATURE IMPORTANCE (Top 10)")
        fprint("=" * 80)
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "moneyness_xval_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"moneyness_xval_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log each variant's metrics
                for vname, offset_pct, desc in MONEYNESS_VARIANTS:
                    r = all_results.get(vname, {})
                    if "error" in r:
                        continue
                    prefix = vname
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_cagr", r.get("cagr", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(LEGACY_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "rebal_freq": WF_REBAL_FREQ,
                    "moneyness_offsets_tested": str([v[1] for v in MONEYNESS_VARIANTS]),
                    "n_variants": len(MONEYNESS_VARIANTS),
                })

                mlflow.log_artifact(str(results_path))

                # Log best variant info
                if valid_results:
                    best_name = max(valid_results, key=lambda k: valid_results[k]["sharpe"])
                    mlflow.log_param("best_variant", best_name)
                    mlflow.log_metric("best_sharpe", valid_results[best_name]["sharpe"])

            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
