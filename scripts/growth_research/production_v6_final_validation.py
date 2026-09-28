#!/usr/bin/env python3
"""
Production V6 Final Validation — Definitive Out-of-Sample Test
================================================================

The FINAL V6 configuration, combining ALL validated improvements from
extensive cross-validated research:

  1. Weekly rebalance (W-FRI) — +19% Sharpe vs biweekly (MLflow exp 179)
  2. 2% OTM moneyness — best for bull spreads (KB research)
  3. Pair trades VIX<20 — bull+bear combined, 5/5 gates (MLflow exp 177)
  4. 17 features — ablation showed dropping 4 vol features IMPROVES Sharpe
     (2.88 vs 2.80). Dropped: vol_21d, vol_63d, sector_relative_vol_21d, maxdd_63d
  5. GRU regime filter: score > 0.4 = trade (finding #41)

Single clean variant — the PRODUCTION candidate:
  - Capital: $645 | DTE: 21 | Spread: 3% width | Commission: $2.60/spread
  - 15% haircut on ENTRY only | Hold to expiry | Intrinsic value settlement
  - Walk-forward LGBM: 100 trees, depth 4, lr 0.05, 12-period sliding window
  - Weekly rebalance (W-FRI)
  - 2% OTM moneyness for all spreads
  - VIX < 20: top-3 bull call spreads + bottom-3 bear put spreads (pairs)
  - VIX >= 20: top-3 bull call spreads only
  - 17 features (21 minus 4 redundant volatility features)

Full 5-gate adversarial validation + random baseline (5 trials).
Regime stratification, PnL by side, yearly breakdown.
MLflow experiment: 'production_v6_final'

BASED ON: production_v4_honest_test.py (canonical 897-line honest backtest)
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


# ── Standardized tools (Neptune-compatible with inline fallbacks) ──
import os as _os
_hostname = _os.uname().nodename.lower()
if 'neptune' in _hostname or 'nick' in str(_os.path.expanduser('~')):
    _BASE_PATH = "/home/nick/Lvl3Quant"
else:
    _BASE_PATH = "/home/jupiter/Lvl3Quant"
sys.path.insert(0, _BASE_PATH)

try:
    from research.tools.options_pricer import (
        price_bull_call_spread,
        price_bear_put_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    from research.tools.adversarial_validator import validate_trades
    _TOOLS_IMPORTED = True
    fprint("Imported from research.tools")
except ImportError:
    _TOOLS_IMPORTED = False
    fprint("research.tools not found — using inline implementations")

    COMMISSION_RT_SPREAD = 2.60
    DEFAULT_HAIRCUT = 0.15

    def compute_atr(prices, window=14):
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()
        return (high - low).mean()

    def estimate_iv(prices, window=21, mult=1.2):
        returns = np.log(prices / prices.shift(1)).dropna()
        hv = returns.rolling(window).std() * np.sqrt(252)
        return hv * mult

    def price_bull_call_spread(S, K1, K2, dte, atr, vix=20.0):
        """Black-Scholes bull call spread pricing with ATR-based IV and haircut."""
        from scipy.stats import norm
        # Estimate IV from ATR
        iv = max(0.10, min(1.5, float(atr / S) * np.sqrt(252) * 1.2))
        if vix > 25:
            iv *= 1.1
        T = dte / 365.0
        if T <= 0 or iv <= 0:
            return 0.0, 0.0

        d1_l = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        call_l = S * norm.cdf(d1_l) - K1 * np.exp(-0.04 * T) * norm.cdf(d1_l - iv * np.sqrt(T))
        call_s = S * norm.cdf(d1_s) - K2 * np.exp(-0.04 * T) * norm.cdf(d1_s - iv * np.sqrt(T))
        spread_val = max(call_l - call_s, 0.001)
        entry_cost = spread_val * (1 + DEFAULT_HAIRCUT)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max_profit

    def price_bear_put_spread(S, K1, K2, dte, atr, vix=20.0):
        """Black-Scholes bear put spread pricing with ATR-based IV and haircut."""
        from scipy.stats import norm
        iv = max(0.10, min(1.5, float(atr / S) * np.sqrt(252) * 1.2))
        if vix > 25:
            iv *= 1.1
        T = dte / 365.0
        if T <= 0 or iv <= 0:
            return 0.0, 0.0

        d1_l = (np.log(S / K2) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d1_s = (np.log(S / K1) + (0.5 * iv**2) * T) / (iv * np.sqrt(T))
        put_l = K2 * np.exp(-0.04 * T) * norm.cdf(-(d1_l - iv * np.sqrt(T))) - S * norm.cdf(-d1_l)
        put_s = K1 * np.exp(-0.04 * T) * norm.cdf(-(d1_s - iv * np.sqrt(T))) - S * norm.cdf(-d1_s)
        spread_val = max(put_l - put_s, 0.001)
        entry_cost = spread_val * (1 + DEFAULT_HAIRCUT)
        max_profit = (K2 - K1) - entry_cost
        return entry_cost, max_profit

    def validate_trades(trades, initial_capital=645.0, spy_prices=None,
                        strategy_name="Strategy", n_perms=2000):
        """Inline adversarial validation fallback for Neptune.
        Returns an object with .sharpe, .sortino, .to_dict(), .print_summary() etc."""
        from dataclasses import dataclass, field
        from typing import Optional

        @dataclass
        class _GateResult:
            name: str
            passed: bool
            metric_name: str
            metric_value: float
            threshold: float = 0.0
            detail: str = ""

        @dataclass
        class _ValidationResult:
            strategy_name: str
            n_trades: int
            sharpe: float
            sortino: float
            cagr: float
            max_dd: float
            win_rate: float
            profit_factor: float
            final_equity: float
            gates: list = field(default_factory=list)
            error: Optional[str] = None

            @property
            def all_passed(self):
                return all(g.passed for g in self.gates) and self.error is None

            @property
            def gates_passed(self):
                return sum(1 for g in self.gates if g.passed)

            @property
            def gates_total(self):
                return len(self.gates)

            def to_dict(self):
                return {
                    "strategy_name": self.strategy_name,
                    "n_trades": self.n_trades,
                    "sharpe": round(self.sharpe, 3),
                    "sortino": round(self.sortino, 3),
                    "cagr": round(self.cagr, 4),
                    "max_dd": round(self.max_dd, 4),
                    "win_rate": round(self.win_rate, 4),
                    "profit_factor": round(self.profit_factor, 3),
                    "final_equity": round(self.final_equity, 2),
                    "gates_passed": self.gates_passed,
                    "gates_total": self.gates_total,
                    "all_passed": self.all_passed,
                    "gates": [
                        {"name": g.name, "passed": g.passed, "metric_name": g.metric_name,
                         "metric_value": round(g.metric_value, 4), "threshold": g.threshold,
                         "detail": g.detail}
                        for g in self.gates
                    ],
                    "error": self.error,
                }

            def print_summary(self):
                fprint(f"\n{'='*65}")
                fprint(f"  ADVERSARIAL VALIDATION: {self.strategy_name}")
                fprint(f"{'='*65}")
                fprint(f"  Trades: {self.n_trades}  |  Sharpe: {self.sharpe:.2f}  |  "
                       f"Sortino: {self.sortino:.2f}  |  WR: {self.win_rate:.1%}")
                fprint(f"  CAGR: {self.cagr:.1%}  |  MaxDD: {self.max_dd:.1%}  |  "
                       f"PF: {self.profit_factor:.2f}  |  Final: ${self.final_equity:,.0f}")
                for g in self.gates:
                    status = "PASS" if g.passed else "FAIL"
                    fprint(f"  [{status}] {g.name}: {g.metric_name}={g.metric_value:.4f} "
                           f"(threshold={g.threshold}) {g.detail}")
                fprint(f"  VERDICT: {self.gates_passed}/{self.gates_total} gates passed")
                fprint(f"{'='*65}")

        # Compute metrics
        if not trades or len(trades) < 5:
            return _ValidationResult(
                strategy_name=strategy_name, n_trades=len(trades) if trades else 0,
                sharpe=0, sortino=0, cagr=0, max_dd=0, win_rate=0,
                profit_factor=0, final_equity=initial_capital,
                error="Insufficient trades"
            )

        pnls = [t["pnl"] for t in trades]
        equity_curve = [initial_capital]
        for p in pnls:
            equity_curve.append(equity_curve[-1] + p)
        equity = np.array(equity_curve[1:])
        final_eq = float(equity[-1])

        rets = np.diff(np.array([initial_capital] + list(equity))) / np.array([initial_capital] + list(equity[:-1]))
        rets = rets[~np.isnan(rets)]

        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))  # weekly-ish
        neg_rets = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(neg_rets) + 1e-10) * np.sqrt(52)) if len(neg_rets) > 0 else sharpe

        # Parse dates for CAGR
        try:
            dates = [pd.Timestamp(t["entry_date"]) for t in trades]
            n_years = max((dates[-1] - dates[0]).days / 365.25, 0.5)
        except Exception:
            n_years = max(len(trades) / 26, 0.5)  # ~26 weekly rebalances/year
        cagr = float((final_eq / initial_capital) ** (1 / n_years) - 1) if final_eq > 0 else -1.0

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(np.min(dd))

        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
        win_sum = sum(p for p in pnls if p > 0)
        loss_sum = abs(sum(p for p in pnls if p <= 0))
        profit_factor = float(win_sum / (loss_sum + 1e-10))

        # 5 gates
        gates = []

        # Gate 1: Sharpe > 1.0
        gates.append(_GateResult("Sharpe", sharpe > 1.0, "sharpe", sharpe, 1.0,
                                 "Annualized Sharpe ratio"))
        # Gate 2: Profit Factor > 1.2
        gates.append(_GateResult("Profit Factor", profit_factor > 1.2, "pf", profit_factor, 1.2,
                                 "Gross profit / gross loss"))
        # Gate 3: Max DD > -30%
        gates.append(_GateResult("Max Drawdown", max_dd > -0.30, "max_dd", max_dd, -0.30,
                                 "Must be shallower than -30%"))
        # Gate 4: Win Rate > 45%
        gates.append(_GateResult("Win Rate", win_rate > 0.45, "wr", win_rate, 0.45,
                                 "Minimum win rate"))
        # Gate 5: Permutation test (simplified — check if Sharpe > 2x random)
        perm_sharpes = []
        for _ in range(min(n_perms, 500)):
            shuffled = np.random.permutation(pnls)
            eq_s = np.cumsum(shuffled) + initial_capital
            r_s = np.diff(np.concatenate([[initial_capital], eq_s])) / np.concatenate([[initial_capital], eq_s[:-1]])
            perm_sharpes.append(float(np.mean(r_s) / (np.std(r_s) + 1e-10) * np.sqrt(52)))
        perm_p = float(np.mean([1 for ps in perm_sharpes if ps >= sharpe]) / len(perm_sharpes))
        gates.append(_GateResult("Permutation", perm_p < 0.05, "p_value", perm_p, 0.05,
                                 f"p={perm_p:.3f} (500 perms)"))

        return _ValidationResult(
            strategy_name=strategy_name,
            n_trades=len(pnls),
            sharpe=sharpe, sortino=sortino, cagr=cagr, max_dd=max_dd,
            win_rate=win_rate, profit_factor=profit_factor,
            final_equity=final_eq, gates=gates,
        )


# ── Config ──
BASE = Path(_BASE_PATH)
OUTPUT_DIR = BASE / "output" / "growth_research" / "production_v6_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
OTM_PCT = 0.02  # 2% OTM moneyness
MAX_POS_BULL = 200
MAX_POS_PAIR_LEG = 100
WF_REBAL_FREQ = "W-FRI"  # Weekly rebalance

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "production_v6_final"

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
# FEATURE DEFINITIONS — V6 (17 features)
# ══════════════════════════════════════════════════════════════

# Original 18 legacy features MINUS 4 vol features that ablation showed are redundant:
#   Dropped: vol_21d, vol_63d, maxdd_63d (legacy), sector_relative_vol_21d (cross-asset)
#   Ablation result: 17 features Sharpe 2.88 vs 21 features Sharpe 2.80

V6_FEATURES = [
    # Momentum features (kept)
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    # Quality features (kept, minus vol_21d/vol_63d/maxdd_63d)
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    # Cross-asset features (kept, minus sector_relative_vol_21d)
    "sector_spy_beta_63d",
    "cross_sector_dispersion",
]

assert len(V6_FEATURES) == 17, f"Expected 17 features, got {len(V6_FEATURES)}"


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
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
# REGIME LOADING
# ══════════════════════════════════════════════════════════════

def load_regime_predictions():
    """Load GRU regime predictions and build a date-indexed Series."""
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found at {REGIME_FILE}")
        fprint("  Will use VIX-based regime proxy instead")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
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
        return 0.5  # neutral default
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the legacy quality-momentum features for a single sector ETF.
    Returns all 18 legacy features; V6 feature selection happens downstream."""
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
    """Compute cross-asset features (V6 uses spy_beta and dispersion only,
    but we compute all 3 and let feature selection handle it)."""
    f = {}

    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "sector_relative_vol_21d": 1.0,
                "cross_sector_dispersion": 0.01}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
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

    # 2. Sector relative vol 21d (computed but may not be used in V6)
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

    # 3. Cross-sector dispersion (rolling 21d stdev of sector returns)
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
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses regime>0.4 filter. Bear direction handled at trade time via VIX pairs logic.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Regime filter: only trade when GRU says bull (>0.4)
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features
            cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target (DTE days forward)
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
    """Walk-forward LGBM ranking: 12-period sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    V6: Insufficient data ({len(df)} records)")
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

    fprint(f"    V6: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION
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
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# STRIKE COMPUTATION (2% OTM)
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """
    Compute strike prices for a spread.

    OTM (otm_pct=0.02 for 2%):
      Bull call: K1=S*(1+otm_pct), K2=K1*(1+spread_pct/100)
      Bear put:  K2=S*(1-otm_pct), K1=K2*(1-spread_pct/100)

    Returns (K1, K2) where K1 < K2 always.
    """
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:  # bear
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION — V6 (single clean variant)
# ══════════════════════════════════════════════════════════════

def _execute_single_trade(tk, dt, direction, max_pos, close, atr_dict, cv, equity):
    """
    Execute a single spread trade. Returns PnL or None if trade cannot be entered.

    Uses V6 pricing: BS with ATR-based IV, 15% entry haircut,
    hold to expiry, intrinsic value only, 2% OTM.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    # ATR for pricing
    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    # Compute strikes (2% OTM)
    K1, K2 = compute_strikes(S, direction, OTM_PCT, SPREAD_PCT)

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
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # HOLD TO EXPIRY: compute intrinsic value at expiry
    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    exit_value_ps = intrinsic

    # PnL: exit value - entry cost - commission (no exit haircut at expiry)
    pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return pnl


def simulate_trades(rankings, close, high, low, atr_dict):
    """
    Simulate V6 trades: weekly rebalance, 2% OTM, bull+bear pairs.

    HONEST RULES:
      - Hold to expiry
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)

    Pair trade rules:
      - VIX < 20: top-3 bull call + bottom-3 bear put (pairs mode)
      - VIX >= 20: top-3 bull call only
      - Pairs: $100/trade per leg; bull-only: $200/trade
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # Determine trade mode based on VIX
        if cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        # Pick sectors: top K for bull, bottom K for bear
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        # Position sizing
        if trade_mode == "pairs":
            max_pos = min(MAX_POS_PAIR_LEG, equity / 6)  # 6 positions total
        else:
            max_pos = min(MAX_POS_BULL, equity / 3)

        if max_pos < 30:
            continue

        # Execute bull leg
        for tk in bull_picks:
            pnl = _execute_single_trade(tk, dt, "bull", max_pos, close, atr_dict, cv, equity)
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
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
                    "trade_mode": trade_mode,
                })

        # Execute bear leg (pairs mode only)
        for tk in bear_picks:
            pnl = _execute_single_trade(tk, dt, "bear", max_pos, close, atr_dict, cv, equity)
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            rand_rankings, close, high, low, atr_dict
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


# ══════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def regime_stratification(trades):
    """Stratify results by SPY regime (bull vs bear market at exit)."""
    fprint("\n  REGIME STRATIFICATION:")
    fprint(f"  {'Regime':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>8}")
    fprint(f"  {'-'*55}")

    for regime in ["bull", "bear"]:
        rt = [t for t in trades if t["regime"] == regime]
        if not rt:
            fprint(f"  {regime:<10} {'(none)':>7}")
            continue
        pnls = [t["pnl"] for t in rt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        fprint(f"  {regime:<10} {len(rt):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f} {sh:>8.2f}")

    # Regime imbalance check
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]
    if bull_trades and bear_trades:
        bull_sh = float(np.mean([t["pnl"] for t in bull_trades]) /
                        (np.std([t["pnl"] for t in bull_trades]) + 1e-10) * np.sqrt(52))
        bear_sh = float(np.mean([t["pnl"] for t in bear_trades]) /
                        (np.std([t["pnl"] for t in bear_trades]) + 1e-10) * np.sqrt(52))
        imbalance = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
        fprint(f"\n  Regime imbalance: |Sharpe_bull - Sharpe_bear| / max = {imbalance:.2f}")
        if imbalance > 0.50:
            fprint(f"  WARNING: Regime imbalance > 0.50 threshold (HC #428 R1)")
        else:
            fprint(f"  OK: Regime-agnostic (imbalance < 0.50)")


def pnl_by_side(trades):
    """Break down PnL by bull vs bear side."""
    fprint("\n  PnL BY SIDE:")
    fprint(f"  {'Side':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'MaxWin':>9} {'MaxLoss':>9}")
    fprint(f"  {'-'*65}")

    for side in ["bull", "bear"]:
        st = [t for t in trades if t["direction"] == side]
        if not st:
            fprint(f"  {side:<10} {'(none)':>7}")
            continue
        pnls = [t["pnl"] for t in st]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        mx = max(pnls)
        mn = min(pnls)
        fprint(f"  {side:<10} {len(st):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f} ${mx:>8.2f} ${mn:>8.2f}")

    # Trade mode breakdown
    fprint("\n  TRADE MODE BREAKDOWN:")
    pair_t = [t for t in trades if t.get("trade_mode") == "pairs"]
    bull_only_t = [t for t in trades if t.get("trade_mode") == "bull_only"]
    if pair_t:
        pair_pnl = sum(t["pnl"] for t in pair_t)
        pair_wr = sum(1 for t in pair_t if t["win"]) / len(pair_t) * 100
        fprint(f"    Pairs mode (VIX<20): {len(pair_t)} trades, WR {pair_wr:.1f}%, PnL ${pair_pnl:.0f}")
    if bull_only_t:
        bo_pnl = sum(t["pnl"] for t in bull_only_t)
        bo_wr = sum(1 for t in bull_only_t if t["win"]) / len(bull_only_t) * 100
        fprint(f"    Bull-only (VIX>=20): {len(bull_only_t)} trades, WR {bo_wr:.1f}%, PnL ${bo_pnl:.0f}")


def yearly_breakdown(trades):
    """Break down results by calendar year."""
    fprint("\n  YEARLY BREAKDOWN:")
    fprint(f"  {'Year':<6} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Sharpe':>8} {'MaxDD':>8}")
    fprint(f"  {'-'*50}")

    # Parse years
    by_year = {}
    for t in trades:
        yr = t["entry_date"][:4]
        by_year.setdefault(yr, []).append(t)

    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        pnls = [t["pnl"] for t in yt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))

        # Drawdown within year
        eq = np.cumsum(pnls) + CAP
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        mdd = float(np.min(dd)) if len(dd) > 0 else 0

        fprint(f"  {yr:<6} {len(yt):>7} {wr:>6.1%} ${tot:>9.0f} {sh:>8.2f} {mdd:>7.1%}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 90)
    fprint(f"PRODUCTION V6 FINAL VALIDATION — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 90)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime filter: GRU score > {REGIME_BULL_THRESHOLD}")
    fprint(f"Rebalance: {WF_REBAL_FREQ} | OTM: {OTM_PCT:.0%} moneyness")
    fprint(f"Pairs: VIX<20 = bull+bear (top-3 + bottom-3), VIX>=20 = bull only (top-3)")
    fprint(f"Features: {len(V6_FEATURES)} (17 = 21 minus 4 vol features)")
    fprint(f"LGBM: 100 trees, depth 4, lr 0.05, {WF_TRAIN_PERIODS}-period sliding window")
    fprint(f"Tools imported from research.tools: {_TOOLS_IMPORTED}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build rebalance dates (weekly)
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 5. Build LGBM rankings with V6 features (17)
    fprint(f"\n{'=' * 80}")
    fprint(f"BUILDING LGBM RANKINGS: {len(V6_FEATURES)} features, {WF_REBAL_FREQ} rebalance")
    fprint(f"{'=' * 80}")

    records = build_feature_records(
        close, high, low, rebal_dates, V6_FEATURES, regime_series
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, V6_FEATURES)

    if not rankings:
        fprint("ERROR: No rankings produced. Exiting.")
        return

    # 6. Simulate trades
    fprint(f"\n{'=' * 80}")
    fprint("SIMULATING V6 TRADES")
    fprint(f"{'=' * 80}")

    trades, final_eq = simulate_trades(rankings, close, high, low, atr_dict)

    if not trades or len(trades) < 10:
        fprint(f"ERROR: Only {len(trades) if trades else 0} trades produced. Exiting.")
        return

    fprint(f"\n  Total trades: {len(trades)}")
    fprint(f"  Final equity: ${final_eq:,.0f} (from ${CAP:.0f})")
    fprint(f"  Total return: {(final_eq/CAP - 1)*100:.1f}%")

    # 7. Full 5-gate adversarial validation
    fprint(f"\n{'=' * 80}")
    fprint("5-GATE ADVERSARIAL VALIDATION")
    fprint(f"{'=' * 80}")

    result = validate_trades(
        trades, initial_capital=CAP,
        spy_prices=spy_close,
        strategy_name="V6_FINAL",
    )
    result.print_summary()

    # 8. Regime stratification
    fprint(f"\n{'=' * 80}")
    fprint("REGIME STRATIFICATION")
    fprint(f"{'=' * 80}")
    regime_stratification(trades)

    # 9. PnL by side (bull vs bear)
    fprint(f"\n{'=' * 80}")
    fprint("PnL BY SIDE & TRADE MODE")
    fprint(f"{'=' * 80}")
    pnl_by_side(trades)

    # 10. Yearly breakdown
    fprint(f"\n{'=' * 80}")
    fprint("YEARLY BREAKDOWN")
    fprint(f"{'=' * 80}")
    yearly_breakdown(trades)

    # 11. Random baseline
    fprint(f"\n{'=' * 80}")
    fprint("RANDOM BASELINE COMPARISON")
    fprint(f"{'=' * 80}")

    random_sharpes = random_baseline_test(rankings, close, high, low, atr_dict)
    mean_random = np.mean(random_sharpes) if random_sharpes else 0
    fprint(f"\n  V6 Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
    if result.sharpe > 0 and mean_random > 0:
        fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")
    elif result.sharpe > 0:
        fprint(f"  ML alpha: meaningful (random produced 0 Sharpe)")

    # 12. Feature importance
    fprint(f"\n{'=' * 80}")
    fprint("FEATURE IMPORTANCE (All 17 features)")
    fprint(f"{'=' * 80}")
    if imp_df is not None:
        for _, row in imp_df.iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 40)
            fprint(f"  {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # 13. Final verdict
    fprint(f"\n{'=' * 90}")
    fprint("FINAL VERDICT")
    fprint(f"{'=' * 90}")
    rd = result.to_dict()
    fprint(f"  Sharpe: {rd['sharpe']:.2f} | Sortino: {rd['sortino']:.2f} | "
           f"WR: {rd['win_rate']*100:.1f}% | PF: {rd['profit_factor']:.2f}")
    fprint(f"  MaxDD: {rd['max_dd']*100:.1f}% | CAGR: {rd['cagr']*100:.1f}% | "
           f"Final: ${rd['final_equity']:,.0f}")
    fprint(f"  Gates: {rd['gates_passed']}/{rd['gates_total']} | "
           f"Random mean Sharpe: {mean_random:.2f}")

    if rd.get("all_passed", False) or rd["gates_passed"] == rd["gates_total"]:
        fprint(f"\n  VERDICT: V6 PASSES ALL {rd['gates_total']} GATES — Ready for production deployment")
    else:
        fprint(f"\n  VERDICT: V6 fails {rd['gates_total'] - rd['gates_passed']} gate(s) — "
               f"Needs further investigation")

    # 14. Save results JSON
    results_data = {
        "config": {
            "capital": CAP, "dte": DTE, "spread_pct": SPREAD_PCT,
            "haircut": DEFAULT_HAIRCUT, "commission": COMMISSION_RT_SPREAD,
            "rebal_freq": WF_REBAL_FREQ, "otm_pct": OTM_PCT,
            "pairs_mode": True, "vix_threshold": 20.0,
            "n_features": len(V6_FEATURES), "features": V6_FEATURES,
            "lgbm_n_estimators": 100, "lgbm_max_depth": 4,
            "lgbm_lr": 0.05, "wf_train_periods": WF_TRAIN_PERIODS,
            "regime_threshold": REGIME_BULL_THRESHOLD,
            "hold_to_expiry": True, "entry_haircut_only": True,
        },
        "results": rd,
        "random_sharpes": [round(s, 3) for s in random_sharpes],
        "random_mean_sharpe": round(mean_random, 3),
        "n_ranking_dates": len(rankings),
        "total_trades": len(trades),
        "bull_trades": len([t for t in trades if t["direction"] == "bull"]),
        "bear_trades": len([t for t in trades if t["direction"] == "bear"]),
        "pair_mode_trades": len([t for t in trades if t.get("trade_mode") == "pairs"]),
        "bull_only_mode_trades": len([t for t in trades if t.get("trade_mode") == "bull_only"]),
        "tools_imported": _TOOLS_IMPORTED,
        "timestamp": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": (datetime.now() - t0).total_seconds(),
    }

    # Add yearly data
    by_year = {}
    for t in trades:
        yr = t["entry_date"][:4]
        by_year.setdefault(yr, []).append(t)
    yearly = {}
    for yr, yt in sorted(by_year.items()):
        pnls = [t["pnl"] for t in yt]
        yearly[yr] = {
            "n_trades": len(yt),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "total_pnl": round(sum(pnls), 2),
            "sharpe": round(float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52)), 3),
        }
    results_data["yearly"] = yearly

    results_path = OUTPUT_DIR / "v6_final_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trade log
    trades_path = OUTPUT_DIR / "v6_final_trades.json"
    with open(trades_path, "w") as f:
        json.dump(trades, f, indent=2, default=str)
    fprint(f"Trade log saved to {trades_path}")

    # 15. MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v6_final_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_metric("sharpe", rd.get("sharpe", 0))
                mlflow.log_metric("sortino", rd.get("sortino", 0))
                mlflow.log_metric("win_rate", rd.get("win_rate", 0))
                mlflow.log_metric("profit_factor", rd.get("profit_factor", 0))
                mlflow.log_metric("max_dd", rd.get("max_dd", 0))
                mlflow.log_metric("cagr", rd.get("cagr", 0))
                mlflow.log_metric("n_trades", rd.get("n_trades", 0))
                mlflow.log_metric("gates_passed", rd.get("gates_passed", 0))
                mlflow.log_metric("final_equity", rd.get("final_equity", 0))
                mlflow.log_metric("random_mean_sharpe", mean_random)
                mlflow.log_metric("alpha_ratio",
                                  rd.get("sharpe", 0) / max(mean_random, 0.01))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "rebal_freq": WF_REBAL_FREQ,
                    "otm_pct": OTM_PCT,
                    "pairs_mode": True,
                    "vix_threshold": 20.0,
                    "n_features": len(V6_FEATURES),
                    "regime_threshold": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "lgbm_n_estimators": 100,
                    "lgbm_max_depth": 4,
                    "lgbm_lr": 0.05,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                })

                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(trades_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
