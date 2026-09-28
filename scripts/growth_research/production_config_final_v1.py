#!/usr/bin/env python3
"""
Production Config Final Validation v1
=======================================

Definitive test of the SINGLE most impactful validated change from 18+ experiments:
3% OTM instead of 2% OTM at DTE=21.

3 Variants (minimal, focused):
  A_control:    DTE=21, OTM=2%, Spread=3%, K=3, WF=25, GRU>0.4, VIX<20 — CURRENT V6 PRODUCTION
  B_optimal:    DTE=21, OTM=3%, Spread=3%, K=3, WF=25, GRU>0.4, VIX<20 — ONLY CHANGE: 3% OTM
  C_aggressive: DTE=21, OTM=3%, Spread=3%, K=2, WF=15, GRU>0.4, VIX<20 — stack concentration + shorter WF

Fixed params: Capital $645, Haircut 15%, Comm $2.60, weekly (W-FRI),
hold to expiry, intrinsic only, 21 features.

Includes:
  - Full 5-gate adversarial validation
  - Random baseline comparison (5 trials each)
  - OOS rolling stability check (4 time periods)
  - MLflow logging
  - Results JSON
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


# ================================================================
# INLINE FALLBACKS for Neptune (missing research.tools)
# ================================================================

RISK_FREE_RATE = 0.045
DEFAULT_HAIRCUT = 0.15
COMMISSION_RT_SPREAD = 2.60

try:
    sys.path.insert(0, "/home/jupiter/Lvl3Quant")
    from research.tools.options_pricer import (
        price_bull_call_spread,
        price_bear_put_spread,
        estimate_iv,
        compute_atr,
        COMMISSION_RT_SPREAD,
        DEFAULT_HAIRCUT,
    )
    from research.tools.adversarial_validator import validate_trades
    fprint("Using research.tools (Jupiter path available)")
except ImportError:
    fprint("research.tools not found -- using inline fallbacks")

    from scipy.stats import norm

    def bs_call_price(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    def bs_put_price(S, K, T, r=RISK_FREE_RATE, sigma=0.25):
        if T <= 0 or sigma <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
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
        max_profit = (K2 - K1) - entry_cost
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
        max_profit = (K2 - K1) - entry_cost
        return float(entry_cost), float(max_profit)

    def compute_atr(high, low, close, period=14):
        return None

    # -- Inline adversarial validator --

    class _GateResult:
        def __init__(self, name, passed, metric_name, metric_value, threshold, detail=""):
            self.name = name
            self.passed = passed
            self.metric_name = metric_name
            self.metric_value = metric_value
            self.threshold = threshold
            self.detail = detail

        def __str__(self):
            status = "PASS" if self.passed else "FAIL"
            return f"  [{status}] {self.name}: {self.metric_name}={self.metric_value:.4f} (threshold: {self.threshold})"

    class _ValidationResult:
        def __init__(self, strategy_name, n_trades, sharpe, sortino, cagr, max_dd,
                     win_rate, profit_factor, final_equity, gates=None, error=None):
            self.strategy_name = strategy_name
            self.n_trades = n_trades
            self.sharpe = sharpe
            self.sortino = sortino
            self.cagr = cagr
            self.max_dd = max_dd
            self.win_rate = win_rate
            self.profit_factor = profit_factor
            self.final_equity = final_equity
            self.gates = gates or []
            self.error = error

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
            fprint("\n" + "=" * 65)
            fprint(f"  ADVERSARIAL VALIDATION: {self.strategy_name}")
            fprint("=" * 65)
            if self.error:
                fprint(f"  ERROR: {self.error}")
                fprint("=" * 65)
                return
            fprint(f"  Trades: {self.n_trades}  |  Sharpe: {self.sharpe:.2f}  |  "
                   f"Sortino: {self.sortino:.2f}  |  WR: {self.win_rate*100:.1f}%")
            fprint(f"  CAGR: {self.cagr*100:.1f}%  |  MaxDD: {self.max_dd*100:.1f}%  |  "
                   f"PF: {self.profit_factor:.2f}  |  Final: ${self.final_equity:,.0f}")
            fprint("-" * 65)
            for gate in self.gates:
                fprint(gate)
            fprint("-" * 65)
            verdict = "ALL GATES PASSED" if self.all_passed else f"FAILED ({self.gates_passed}/{self.gates_total} passed)"
            fprint(f"  VERDICT: {verdict}")
            fprint("=" * 65)

    def validate_trades(trades, initial_capital=645.0, spy_prices=None,
                        strategy_name="strategy", n_perms=2000):
        """Inline 5-gate adversarial validation."""
        if not trades or len(trades) < 5:
            return _ValidationResult(strategy_name, 0, 0, 0, 0, 0, 0, 0, initial_capital,
                                     error="Too few trades")

        pnls = np.array([t["pnl"] for t in trades])
        n = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        # Equity curve
        eq = [initial_capital]
        for p in pnls:
            eq.append(eq[-1] + p)
        eq = np.array(eq)
        final_eq = eq[-1]

        # Dates for equity series
        dates = pd.to_datetime([t["entry_date"] for t in trades])
        eq_series = pd.Series(eq[1:], index=dates)
        monthly_eq = eq_series.resample("ME").last().dropna()
        if len(monthly_eq) > 1:
            monthly_rets = monthly_eq.pct_change().dropna()
        else:
            monthly_rets = pd.Series(dtype=float)

        # Metrics
        if len(monthly_rets) > 1:
            sharpe = float(monthly_rets.mean() / (monthly_rets.std() + 1e-10) * np.sqrt(12))
            dr = monthly_rets[monthly_rets < 0]
            sortino = float(monthly_rets.mean() / (dr.std() + 1e-10) * np.sqrt(12)) if len(dr) > 0 else sharpe
        else:
            sharpe = 0.0
            sortino = 0.0

        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / (peak + 1e-10)
        max_dd = float(dd.min())

        win_rate = float(np.sum(pnls > 0) / n) if n > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / (gross_loss + 1e-10)

        years = max((dates.max() - dates.min()).days / 365.25, 0.5)
        cagr = (final_eq / initial_capital) ** (1.0 / years) - 1.0 if final_eq > 0 else -1.0

        gates = []

        # Gate 1: Sign-flip permutation
        perm_count = 0
        for _ in range(min(n_perms, 2000)):
            signs = np.random.choice([-1, 1], size=n)
            perm_total = float(np.sum(pnls * signs))
            if perm_total >= float(pnls.sum()):
                perm_count += 1
        perm_pval = perm_count / min(n_perms, 2000)
        gates.append(_GateResult("Sign-Flip Permutation", perm_pval < 0.05,
                                 "p_value", perm_pval, 0.05))

        # Gate 2: Regime balance
        if spy_prices is not None and "regime" in trades[0]:
            bull_pnl = sum(t["pnl"] for t in trades if t.get("regime") == "bull")
            bear_pnl = sum(t["pnl"] for t in trades if t.get("regime") == "bear")
            both_pos = bull_pnl > 0 and bear_pnl > 0
            regime_ratio = min(bull_pnl, bear_pnl) / (max(abs(bull_pnl), abs(bear_pnl)) + 1e-10)
            gates.append(_GateResult("Regime Balance", both_pos,
                                     "min_regime_ratio", regime_ratio, 0.0,
                                     f"Bull PnL: ${bull_pnl:.0f}, Bear PnL: ${bear_pnl:.0f}"))
        else:
            gates.append(_GateResult("Regime Balance", True, "skipped", 1.0, 0.0, "No SPY data"))

        # Gate 3: Sub-period stability
        mid = n // 2
        h1_pnl = float(pnls[:mid].sum())
        h2_pnl = float(pnls[mid:].sum())
        gates.append(_GateResult("Sub-Period Stability", h1_pnl > 0 and h2_pnl > 0,
                                 "min_half_pnl", min(h1_pnl, h2_pnl), 0.0,
                                 f"H1: ${h1_pnl:.0f}, H2: ${h2_pnl:.0f}"))

        # Gate 4: Outlier removal (remove best month)
        if len(monthly_rets) > 2:
            trimmed = monthly_rets.drop(monthly_rets.idxmax())
            trimmed_sharpe = float(trimmed.mean() / (trimmed.std() + 1e-10) * np.sqrt(12))
            gates.append(_GateResult("Outlier Removal", trimmed_sharpe > 0,
                                     "trimmed_sharpe", trimmed_sharpe, 0.0))
        else:
            gates.append(_GateResult("Outlier Removal", True, "skipped", 1.0, 0.0, "Too few months"))

        # Gate 5: Yearly consistency
        yearly = eq_series.resample("YE").last().pct_change().dropna()
        if len(yearly) > 0:
            pct_pos_years = float((yearly > 0).mean())
            gates.append(_GateResult("Yearly Consistency", pct_pos_years >= 0.5,
                                     "pct_positive_years", pct_pos_years, 0.5))
        else:
            gates.append(_GateResult("Yearly Consistency", True, "skipped", 1.0, 0.5))

        return _ValidationResult(
            strategy_name=strategy_name,
            n_trades=n,
            sharpe=sharpe,
            sortino=sortino,
            cagr=cagr,
            max_dd=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            final_equity=final_eq,
            gates=gates,
        )


# ================================================================
# CONFIG
# ================================================================

BASE = Path("/home/nick/Lvl3Quant") if Path("/home/nick").exists() else Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "production_config_final_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
DTE = 21

# V6 config (fixed across all variants)
REBAL_FREQ = "W-FRI"
USE_PAIRS = True
TOTAL_BULL_BUDGET = 200.0
TOTAL_PAIR_LEG_BUDGET = 100.0
GRU_THRESHOLD = 0.4
VIX_THRESHOLD = 20.0

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "production_config_final_v1"

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


# ================================================================
# PRODUCTION CONFIG VARIANTS
# ================================================================

VARIANTS = {
    "A_control": {
        "otm_pct": 0.02,
        "top_k": 3,
        "wf_periods": 25,
        "desc": "Current V6 production (DTE=21, OTM=2%, K=3, WF=25) — CONTROL",
    },
    "B_optimal": {
        "otm_pct": 0.03,
        "top_k": 3,
        "wf_periods": 25,
        "desc": "Optimal from research (DTE=21, OTM=3%, K=3, WF=25) — ONLY change is 3% OTM",
    },
    "C_aggressive": {
        "otm_pct": 0.03,
        "top_k": 2,
        "wf_periods": 15,
        "desc": "Aggressive optimal (DTE=21, OTM=3%, K=2, WF=15) — concentration + shorter WF",
    },
}


# ================================================================
# DATA DOWNLOAD
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
# REGIME LOADING
# ================================================================

def load_regime_predictions():
    """Load GRU regime predictions and build a date-indexed Series."""
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found at {REGIME_FILE}")
        fprint("  Will use VIX-based proxy: VIX < 25 = bull")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days "
           f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")

    fprint(f"  GRU score distribution: mean={regime_series.mean():.3f}, "
           f"median={regime_series.median():.3f}, "
           f"std={regime_series.std():.3f}")
    pct_above = (regime_series > GRU_THRESHOLD).mean() * 100
    fprint(f"  Days with GRU > {GRU_THRESHOLD}: {pct_above:.1f}%")

    return regime_series


def get_regime_score_at(regime_series, dt, vix_val=None):
    """Get regime score at a given date, with VIX fallback."""
    if regime_series is None:
        if vix_val is not None and vix_val < 25:
            return 0.6
        return 0.3
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ================================================================
# FEATURE ENGINEERING (17 + 3 cross-asset = 20 features + regime = 21)
# ================================================================

LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d",
]

VALIDATED_CROSS_ASSET = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

ALL_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET + ["regime_score"]


def compute_legacy_features(px, spy_slice):
    """Compute the legacy quality-momentum features for a single sector ETF."""
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
    else:
        f["trend_r2_63d"] = 0.0

    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Compute the 3 validated cross-asset features."""
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


# ================================================================
# WALK-FORWARD LGBM RANKING
# ================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for ALL sectors on ALL rebal dates (unfiltered)."""
    fprint(f"  Building feature records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        cv = float(vix.iloc[idx]) if vix is not None else None
        rscore = get_regime_score_at(regime_series, dt, vix_val=cv)

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

            rec = {**legacy, **cross_asset, "regime_score": rscore,
                   "date": dt, "ticker": tk, "fwd_ret": fwd_ret,
                   "vix": cv if cv is not None else 20.0}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols, wf_train_periods):
    """Walk-forward LGBM ranking — produces rankings for ALL dates."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
        return {}, None, {}, {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    date_vix = {}
    date_regime = {}
    all_importances = np.zeros(len(feature_cols))
    n_models = 0

    for i in range(wf_train_periods, len(dates)):
        train_dates = dates[max(0, i - wf_train_periods):i]
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

            if "vix" in test_df.columns:
                date_vix[test_date] = float(test_df["vix"].iloc[0])
            if "regime_score" in test_df.columns:
                date_regime[test_date] = float(test_df["regime_score"].iloc[0])

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

    fprint(f"    WF={wf_train_periods}: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df, date_vix, date_regime


# ================================================================
# ATR + STRIKE COMPUTATION
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


def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread."""
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ================================================================
# TRADE SIMULATION (with per-variant OTM, K, regime filtering)
# ================================================================

def simulate_trades(name, rankings, close, high, low, atr_dict,
                    date_vix, date_regime, regime_series,
                    otm_pct, top_k):
    """Simulate trades with variant-specific OTM% and K."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    skipped_gru = 0
    traded_dates = 0

    max_pos_bull = TOTAL_BULL_BUDGET / top_k
    max_pos_pair_leg = TOTAL_PAIR_LEG_BUDGET / top_k

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        # Get VIX for this date
        cv = date_vix.get(dt, None)
        if cv is None and vix is not None and dt in vix.index:
            cv = float(vix.loc[dt])
        if cv is None:
            cv = 20.0

        # GRU regime filter (fixed GRU>0.4 for all variants)
        rscore = date_regime.get(dt, None)
        if rscore is None:
            rscore = get_regime_score_at(regime_series, dt, vix_val=cv)
        if rscore <= GRU_THRESHOLD:
            skipped_gru += 1
            continue

        traded_dates += 1

        scores = rankings[dt]
        if not scores:
            continue

        # VIX threshold determines trade mode (fixed VIX<20 for all variants)
        if USE_PAIRS and cv < VIX_THRESHOLD:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:top_k]]
        bear_picks = [t for t, _ in ranked_asc[:top_k]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(max_pos_pair_leg, equity / 6)
        else:
            max_pos = min(max_pos_bull, equity / 3)

        if max_pos < 10:
            continue

        for tk in bull_picks:
            pnl = _execute_single_trade(tk, dt, "bull", max_pos, close, atr_dict, cv, equity, otm_pct)
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
                    "gru_score": round(date_regime.get(dt, 0.5), 3),
                })

        for tk in bear_picks:
            pnl = _execute_single_trade(tk, dt, "bear", max_pos, close, atr_dict, cv, equity, otm_pct)
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
                    "gru_score": round(date_regime.get(dt, 0.5), 3),
                })

    fprint(f"  {name}: {traded_dates} traded dates, {skipped_gru} skipped by GRU, "
           f"{len(trades)} trades, equity ${equity:.0f}")

    return trades, equity, traded_dates, skipped_gru


def _execute_single_trade(tk, dt, direction, max_pos, close, atr_dict, cv, equity, otm_pct):
    """Execute a single spread trade. Returns PnL or None."""
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    K1, K2 = compute_strikes(S, direction, otm_pct, SPREAD_PCT)

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

    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    exit_value_ps = intrinsic
    pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return pnl


# ================================================================
# RANDOM BASELINE
# ================================================================

def random_baseline_test(rankings, close, high, low, atr_dict,
                         date_vix, date_regime, regime_series,
                         otm_pct, top_k, n_trials=5):
    """Test if random sector selection produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials, K={top_k})...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq, _, _ = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict,
            date_vix, date_regime, regime_series,
            otm_pct, top_k,
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
# OOS ROLLING STABILITY CHECK (4 TIME PERIODS)
# ================================================================

def oos_rolling_stability(trades, variant_name):
    """Split OOS trades into 4 chronological periods and check consistency.

    This is the key robustness check: if the strategy works, it should work
    across different market environments, not just one lucky stretch.
    """
    if not trades or len(trades) < 20:
        fprint(f"  OOS Stability: Too few trades ({len(trades) if trades else 0}) for 4-period split")
        return None

    # Sort by entry date
    sorted_trades = sorted(trades, key=lambda t: t["entry_date"])
    n = len(sorted_trades)
    q = n // 4

    periods = {
        "Q1_earliest": sorted_trades[:q],
        "Q2_early_mid": sorted_trades[q:2*q],
        "Q3_late_mid": sorted_trades[2*q:3*q],
        "Q4_latest": sorted_trades[3*q:],
    }

    fprint(f"\n  OOS ROLLING STABILITY CHECK: {variant_name}")
    fprint(f"  {'Period':<14} {'Dates':>25} {'Trades':>7} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'PnL':>9} {'Verdict':>8}")
    fprint("  " + "-" * 95)

    period_results = {}
    n_profitable = 0

    for pname, ptrades in periods.items():
        if len(ptrades) < 5:
            fprint(f"  {pname:<14} {'Too few trades':>25}")
            period_results[pname] = {"error": "too_few_trades"}
            continue

        pnls = np.array([t["pnl"] for t in ptrades])
        total_pnl = float(pnls.sum())
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        wr = float(len(wins)) / len(pnls) * 100
        gp = float(wins.sum()) if len(wins) > 0 else 0.0
        gl = float(abs(losses.sum())) if len(losses) > 0 else 1e-10
        pf = gp / (gl + 1e-10)

        # Simple Sharpe proxy from trade PnLs
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe_proxy = float(pnls.mean() / pnls.std() * np.sqrt(52))  # annualized weekly
        else:
            sharpe_proxy = 0.0

        date_range = f"{ptrades[0]['entry_date']} to {ptrades[-1]['entry_date']}"
        profitable = total_pnl > 0
        if profitable:
            n_profitable += 1
        verdict = "PASS" if profitable else "FAIL"

        fprint(f"  {pname:<14} {date_range:>25} {len(ptrades):>7} {sharpe_proxy:>7.2f} "
               f"{wr:>5.1f}% {pf:>5.2f} ${total_pnl:>8,.0f} {verdict:>8}")

        period_results[pname] = {
            "n_trades": len(ptrades),
            "date_range": date_range,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(wr / 100, 4),
            "profit_factor": round(pf, 3),
            "sharpe_proxy": round(sharpe_proxy, 3),
            "profitable": profitable,
        }

    # Overall stability verdict
    stability_ratio = n_profitable / len(periods)
    fprint("  " + "-" * 95)
    if stability_ratio >= 0.75:
        stability_verdict = "STABLE"
        fprint(f"  STABILITY VERDICT: {stability_verdict} ({n_profitable}/4 periods profitable)")
    elif stability_ratio >= 0.50:
        stability_verdict = "MARGINAL"
        fprint(f"  STABILITY VERDICT: {stability_verdict} ({n_profitable}/4 periods profitable) — needs monitoring")
    else:
        stability_verdict = "UNSTABLE"
        fprint(f"  STABILITY VERDICT: {stability_verdict} ({n_profitable}/4 periods profitable) — REJECT")

    # Check for monotonic equity growth
    period_pnls = [period_results[p].get("total_pnl", 0) for p in periods if "total_pnl" in period_results.get(p, {})]
    if len(period_pnls) >= 3:
        # Is recent performance better than early? (positive trend = good)
        if period_pnls[-1] > period_pnls[0]:
            fprint(f"  TREND: Improving (latest period PnL > earliest) — healthy signal")
        else:
            fprint(f"  TREND: Declining (latest period PnL < earliest) — watch for decay")

    return {
        "periods": period_results,
        "n_profitable": n_profitable,
        "stability_ratio": round(stability_ratio, 2),
        "stability_verdict": stability_verdict,
    }


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = datetime.now()
    fprint("=" * 90)
    fprint(f"PRODUCTION CONFIG FINAL VALIDATION v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 90)
    fprint(f"DEFINITIVE TEST: Best parameters from 18+ experiments")
    fprint(f"FIXED: DTE={DTE} | Spread={SPREAD_PCT:.0f}% | GRU>{GRU_THRESHOLD} | VIX<{VIX_THRESHOLD:.0f}")
    fprint(f"Capital: ${CAP:.0f} | Bull budget: ${TOTAL_BULL_BUDGET:.0f} | "
           f"Pair leg budget: ${TOTAL_PAIR_LEG_BUDGET:.0f}")
    fprint(f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Features: {len(ALL_FEATURES)} (17 legacy + 3 cross-asset + 1 regime)")
    fprint(f"\n3 Variants (focused on the SINGLE most impactful change: OTM%):")
    for vname, cfg in VARIANTS.items():
        fprint(f"  {vname}: OTM={cfg['otm_pct']*100:.0f}%, K={cfg['top_k']}, "
               f"WF={cfg['wf_periods']} — {cfg['desc']}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    # 4. Build weekly rebalance dates
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values
    )
    fprint(f"Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    spy_close = close["SPY"]

    # 5. Build feature records ONCE
    fprint("\n" + "=" * 90)
    fprint(f"Building feature records (shared dataset, DTE={DTE})")
    fprint("=" * 90)

    records = build_feature_records(
        close, high, low, rebal_dates, ALL_FEATURES, regime_series,
    )

    if len(records) < 100:
        fprint("ERROR: Insufficient feature records. Cannot proceed.")
        return

    # 6. Train LGBM for each WF setting needed
    # WF=25 is shared by A and B, WF=15 is for C only
    fprint("\n" + "=" * 90)
    fprint("Training LGBM models (walk-forward)")
    fprint("=" * 90)

    wf_rankings = {}
    wf_importances = {}

    for wf in sorted(set(cfg["wf_periods"] for cfg in VARIANTS.values())):
        fprint(f"\n  Training LGBM with WF={wf}...")
        rankings, imp_df, date_vix, date_regime = walk_forward_lgbm_rank(
            records.copy(), ALL_FEATURES, wf,
        )
        wf_rankings[wf] = (rankings, date_vix, date_regime)
        wf_importances[wf] = imp_df

    # ================================================================
    # RUN EACH VARIANT
    # ================================================================
    all_results = {}

    for vname, cfg in VARIANTS.items():
        otm_pct = cfg["otm_pct"]
        top_k = cfg["top_k"]
        wf = cfg["wf_periods"]
        desc = cfg["desc"]

        rankings, date_vix, date_regime = wf_rankings[wf]

        fprint("\n" + "=" * 90)
        fprint(f"{vname}: {desc}")
        fprint(f"  OTM={otm_pct*100:.0f}%, K={top_k}, WF={wf}")
        fprint("=" * 90)

        if not rankings:
            fprint(f"  No rankings for WF={wf}")
            all_results[vname] = {"description": desc, "error": "no_rankings"}
            continue

        # Simulate trades
        trades, final_eq, traded_dates, skipped_gru = simulate_trades(
            vname, rankings, close, high, low, atr_dict,
            date_vix, date_regime, regime_series,
            otm_pct, top_k,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {
                "description": desc,
                "otm_pct": otm_pct,
                "top_k": top_k,
                "wf_periods": wf,
                "n_trades": len(trades) if trades else 0,
                "traded_dates": traded_dates,
                "skipped_gru": skipped_gru,
                "error": "too_few_trades",
            }
            continue

        fprint(f"  {len(trades)} trades, final equity: ${final_eq:.0f}")

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Direction breakdown
        bull_trades = [t for t in trades if t["direction"] == "bull"]
        bear_trades = [t for t in trades if t["direction"] == "bear"]
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # Mode breakdown
        pairs_trades = [t for t in trades if t["trade_mode"] == "pairs"]
        bull_only_trades = [t for t in trades if t["trade_mode"] == "bull_only"]
        pairs_pnl = sum(t["pnl"] for t in pairs_trades)
        bull_only_pnl = sum(t["pnl"] for t in bull_only_trades)
        fprint(f"  Mode breakdown:")
        fprint(f"    Pairs mode: {len(pairs_trades)} trades, PnL ${pairs_pnl:.0f}")
        fprint(f"    Bull-only:  {len(bull_only_trades)} trades, PnL ${bull_only_pnl:.0f}")

        # OOS Rolling Stability Check (4 time periods)
        stability = oos_rolling_stability(trades, vname)

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
            date_vix, date_regime, regime_series,
            otm_pct, top_k,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"\n  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            "otm_pct": otm_pct,
            "top_k": top_k,
            "wf_periods": wf,
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "traded_dates": traded_dates,
            "skipped_gru": skipped_gru,
            "n_bull_trades": len(bull_trades),
            "n_bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 3),
            "bear_wr": round(bear_wr, 3),
            "n_pairs_trades": len(pairs_trades),
            "n_bull_only_trades": len(bull_only_trades),
            "pairs_pnl": round(pairs_pnl, 2),
            "bull_only_pnl": round(bull_only_pnl, 2),
            "oos_stability": stability,
        }

    # ================================================================
    # SUMMARY COMPARISON
    # ================================================================
    fprint("\n" + "=" * 115)
    fprint("PRODUCTION CONFIG FINAL COMPARISON")
    fprint("=" * 115)
    fprint(f"{'Variant':<16} {'OTM':>4} {'K':>2} {'WF':>3} {'Dates':>5} {'Trades':>6} {'Sharpe':>7} "
           f"{'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Stab':>6} {'Final$':>8}")
    fprint("-" * 115)

    for vname in VARIANTS:
        r = all_results.get(vname, {})
        cfg = VARIANTS[vname]

        if "error" in r:
            fprint(f"  {vname:<16} {cfg['otm_pct']*100:.0f}% {cfg['top_k']:>2} {cfg['wf_periods']:>3} "
                   f"{r.get('traded_dates',0):>5} -- {r.get('error','NO DATA')} --")
            continue
        if "sharpe" not in r:
            fprint(f"  {vname:<16} -- NO DATA --")
            continue

        stab = r.get("oos_stability", {})
        stab_str = f"{stab.get('n_profitable', '?')}/4" if stab else "N/A"

        fprint(f"  {vname:<16} {cfg['otm_pct']*100:.0f}% {cfg['top_k']:>2} {cfg['wf_periods']:>3} "
               f"{r.get('traded_dates',0):>5} {r['n_trades']:>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"{stab_str:>6} ${r['final_equity']:>7,.0f}")

    # -- HEAD-TO-HEAD: A vs B (the key comparison) --
    fprint("\n" + "=" * 115)
    fprint("HEAD-TO-HEAD: A_control (OTM=2%) vs B_optimal (OTM=3%)")
    fprint("=" * 115)

    a = all_results.get("A_control", {})
    b = all_results.get("B_optimal", {})

    if "sharpe" in a and "sharpe" in b:
        metrics = [
            ("Sharpe", a["sharpe"], b["sharpe"]),
            ("Sortino", a["sortino"], b["sortino"]),
            ("Win Rate", a["win_rate"], b["win_rate"]),
            ("Profit Factor", a["profit_factor"], b["profit_factor"]),
            ("Max Drawdown", a["max_dd"], b["max_dd"]),
            ("CAGR", a["cagr"], b["cagr"]),
            ("Final Equity", a["final_equity"], b["final_equity"]),
            ("Gates Passed", a["gates_passed"], b["gates_passed"]),
            ("Trade Count", a["n_trades"], b["n_trades"]),
        ]

        fprint(f"\n  {'Metric':<16} {'A (2% OTM)':>12} {'B (3% OTM)':>12} {'Delta':>10} {'Winner':>8}")
        fprint("  " + "-" * 65)

        b_wins = 0
        a_wins = 0
        for name, av, bv in metrics:
            delta = bv - av
            if name == "Max Drawdown":
                # Less negative is better for drawdown
                winner = "B" if bv > av else "A" if av > bv else "TIE"
            elif name == "Trade Count":
                winner = "-"  # Not a quality metric
            else:
                winner = "B" if bv > av else "A" if av > bv else "TIE"

            if winner == "B":
                b_wins += 1
            elif winner == "A":
                a_wins += 1

            if name in ("Win Rate",):
                fprint(f"  {name:<16} {av*100:>11.1f}% {bv*100:>11.1f}% {delta*100:>9.1f}% {winner:>8}")
            elif name in ("Max Drawdown", "CAGR"):
                fprint(f"  {name:<16} {av*100:>11.1f}% {bv*100:>11.1f}% {delta*100:>9.1f}% {winner:>8}")
            elif name == "Final Equity":
                fprint(f"  {name:<16} ${av:>10,.0f} ${bv:>10,.0f} ${delta:>9,.0f} {winner:>8}")
            elif name in ("Trade Count", "Gates Passed"):
                fprint(f"  {name:<16} {av:>12.0f} {bv:>12.0f} {delta:>10.0f} {winner:>8}")
            else:
                fprint(f"  {name:<16} {av:>12.2f} {bv:>12.2f} {delta:>10.2f} {winner:>8}")

        fprint("  " + "-" * 65)
        overall = "B (3% OTM)" if b_wins > a_wins else "A (2% OTM)" if a_wins > b_wins else "TIE"
        fprint(f"  OVERALL WINNER: {overall} ({b_wins} vs {a_wins} metrics)")

        # Statistical significance
        if a["sharpe"] != 0:
            sharpe_pct_change = (b["sharpe"] - a["sharpe"]) / abs(a["sharpe"]) * 100
            fprint(f"  Sharpe change: {sharpe_pct_change:+.1f}%")
    else:
        fprint("  Cannot compare — one or both variants failed")

    # -- C_aggressive analysis --
    fprint("\n" + "=" * 115)
    fprint("AGGRESSIVE VARIANT ANALYSIS: C_aggressive (OTM=3%, K=2, WF=15)")
    fprint("=" * 115)

    c = all_results.get("C_aggressive", {})
    if "sharpe" in c and "sharpe" in b:
        fprint(f"  C vs B: Sharpe {c['sharpe']:.2f} vs {b['sharpe']:.2f} "
               f"(delta: {c['sharpe']-b['sharpe']:+.2f})")
        fprint(f"  C vs B: Sortino {c['sortino']:.2f} vs {b['sortino']:.2f}")
        fprint(f"  C trades: {c['n_trades']} vs B trades: {b['n_trades']}")
        fprint(f"  Concentration (K=2 vs K=3) effect: "
               f"{'Higher quality' if c['sharpe'] > b['sharpe'] else 'Diluted edge'}")
        fprint(f"  Shorter WF (15 vs 25) effect: "
               f"{'More responsive' if c['sharpe'] > b['sharpe'] else 'More noise'}")

        c_stab = c.get("oos_stability", {})
        if c_stab:
            fprint(f"  OOS Stability: {c_stab.get('stability_verdict', 'N/A')} "
                   f"({c_stab.get('n_profitable', '?')}/4 periods profitable)")
    elif "sharpe" in c:
        fprint(f"  C: Sharpe={c['sharpe']:.2f}, Sortino={c['sortino']:.2f}")
    else:
        fprint("  C_aggressive failed or insufficient trades")

    # -- KEY FINDINGS + RECOMMENDATION --
    fprint("\n" + "=" * 115)
    fprint("FINAL RECOMMENDATION")
    fprint("=" * 115)

    valid_results = {k: v for k, v in all_results.items() if "sharpe" in v}

    if valid_results:
        best_name = max(valid_results.keys(), key=lambda k: valid_results[k]["sharpe"])
        best = valid_results[best_name]
        control = valid_results.get("A_control", {})

        fprint(f"\n  BEST VARIANT: {best_name}")
        fprint(f"    Sharpe={best['sharpe']:.2f}, Sortino={best['sortino']:.2f}, "
               f"WR={best['win_rate']*100:.1f}%, PF={best['profit_factor']:.2f}")
        fprint(f"    Gates: {best['gates_passed']}/{best['gates_total']}, "
               f"Final: ${best['final_equity']:,.0f}")

        stab = best.get("oos_stability", {})
        if stab:
            fprint(f"    OOS Stability: {stab.get('stability_verdict', 'N/A')} "
                   f"({stab.get('n_profitable', '?')}/4 periods)")

        if control and "sharpe" in control:
            delta = best["sharpe"] - control["sharpe"]
            fprint(f"\n  vs CONTROL (A — current V6 production):")
            fprint(f"    Sharpe delta: {delta:+.2f}")
            fprint(f"    Control: Sharpe={control['sharpe']:.2f}, Final=${control['final_equity']:,.0f}")

        # Production deployment recommendation
        fprint(f"\n  DEPLOYMENT RECOMMENDATION:")
        if best_name == "A_control":
            fprint(f"  KEEP current V6 production config. No change needed.")
            fprint(f"  The 3% OTM change does NOT improve risk-adjusted returns.")
        elif best_name == "B_optimal":
            if best["gates_passed"] >= best["gates_total"] - 1 and stab and stab.get("n_profitable", 0) >= 3:
                fprint(f"  UPGRADE to B_optimal: Switch OTM from 2% to 3%.")
                fprint(f"  Single parameter change, validated across 4 OOS periods.")
                fprint(f"  All other V6 params remain unchanged.")
            else:
                gates_issue = best["gates_passed"] < best["gates_total"] - 1
                stab_issue = not stab or stab.get("n_profitable", 0) < 3
                fprint(f"  HOLD — B_optimal shows promise but has concerns:")
                if gates_issue:
                    fprint(f"    - Only {best['gates_passed']}/{best['gates_total']} adversarial gates")
                if stab_issue:
                    fprint(f"    - OOS stability insufficient ({stab.get('n_profitable', 0)}/4 periods)")
        else:  # C_aggressive
            fprint(f"  CAUTION — C_aggressive is best but involves multiple parameter changes.")
            fprint(f"  Consider phased rollout: first test B_optimal (3% OTM only),")
            fprint(f"  then if stable, test K=2 and WF=15 separately.")

        # Random baseline comparison
        fprint(f"\n  RANDOM BASELINE CHECK:")
        for vname in VARIANTS:
            r = valid_results.get(vname, {})
            if "sharpe" in r and "random_mean_sharpe" in r:
                alpha_ratio = r["sharpe"] / (r["random_mean_sharpe"] + 1e-10) if r["random_mean_sharpe"] != 0 else float('inf')
                fprint(f"    {vname}: ML Sharpe {r['sharpe']:.2f} vs Random {r['random_mean_sharpe']:.2f} "
                       f"(alpha ratio: {alpha_ratio:.1f}x)")

    # Save results
    results_path = OUTPUT_DIR / "production_config_final_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save feature importances
    for wf, imp_df in wf_importances.items():
        if imp_df is not None:
            imp_path = OUTPUT_DIR / f"feature_importances_wf{wf}.json"
            imp_data = imp_df.to_dict(orient="records")
            with open(imp_path, "w") as f:
                json.dump(imp_data, f, indent=2, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"prod_config_final_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    if "sharpe" not in r:
                        continue
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))
                    mlflow.log_metric(f"{prefix}_traded_dates", r.get("traded_dates", 0))
                    mlflow.log_metric(f"{prefix}_cagr", r.get("cagr", 0))

                    # OOS stability metrics
                    stab = r.get("oos_stability", {})
                    if stab:
                        mlflow.log_metric(f"{prefix}_oos_stability_ratio", stab.get("stability_ratio", 0))
                        mlflow.log_metric(f"{prefix}_oos_n_profitable", stab.get("n_profitable", 0))

                # Determine best
                valid = {k: v for k, v in all_results.items() if "sharpe" in v}
                if valid:
                    best_name = max(valid.keys(), key=lambda k: valid[k]["sharpe"])
                    mlflow.log_param("best_variant", best_name)
                    mlflow.log_metric("best_sharpe", valid[best_name]["sharpe"])

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "gru_threshold": GRU_THRESHOLD,
                    "vix_threshold": VIX_THRESHOLD,
                    "haircut": DEFAULT_HAIRCUT,
                    "rebal_freq": REBAL_FREQ,
                    "use_pairs": USE_PAIRS,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(ALL_FEATURES),
                    "n_variants": len(VARIANTS),
                    "total_bull_budget": TOTAL_BULL_BUDGET,
                    "total_pair_leg_budget": TOTAL_PAIR_LEG_BUDGET,
                    "study": "production_config_final_v1",
                    "variants": "A_control(2%OTM,K3,WF25) B_optimal(3%OTM,K3,WF25) C_aggressive(3%OTM,K2,WF15)",
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
