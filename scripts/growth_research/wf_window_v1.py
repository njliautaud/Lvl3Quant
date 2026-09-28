#!/usr/bin/env python3
"""
Walk-Forward Window Length Sensitivity v1
==========================================

Tests how many historical rebalance periods to use for LGBM training.
All variants use FIXED DTE=21, OTM=2%, Spread=3%, K=3 -- only WF window varies.

5 Variants:
  A_wf10: WF=10 (short window -- ~10 weeks training)
  B_wf15: WF=15 (~15 weeks)
  C_wf25: WF=25 (current baseline -- ~25 weeks = ~6 months)
  D_wf40: WF=40 (~40 weeks)
  E_wf60: WF=60 (~60 weeks = ~14 months)

Each variant trains its OWN LGBM with different n_train window sizes.
All other parameters fixed: DTE=21, Spread=3%, OTM=2%, K=3, weekly rebalance.

V6 structure:
  - Weekly rebalance (W-FRI)
  - 17+3 features (legacy 17 + 3 cross-asset)
  - Pairs mode (bull VIX>=20, bull+bear VIX<20)
  - Regime filter: GRU >0.4 for bull
  - Hold to expiry, intrinsic only, 15% entry haircut
  - $645 starting capital, $200/K per trade leg
  - Commission: $2.60/spread

5-gate adversarial validation for all variants.
Random baseline comparison (5 trials each).
MLflow experiment: wf_window_v1
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
        return None  # not used directly

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
OUTPUT_DIR = BASE / "output" / "growth_research" / "wf_window_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
DTE = 21
OTM_PCT = 0.02
TOP_K = 3               # FIXED for all variants
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# V6 config
REBAL_FREQ = "W-FRI"
USE_PAIRS = True
TOTAL_BULL_BUDGET = 200.0
TOTAL_PAIR_LEG_BUDGET = 100.0

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "wf_window_v1"

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
# WALK-FORWARD WINDOW VARIANTS
# ================================================================

VARIANTS = {
    "A_wf10": {"wf_periods": 10, "desc": "WF=10 (~10 weeks training)"},
    "B_wf15": {"wf_periods": 15, "desc": "WF=15 (~15 weeks training)"},
    "C_wf25": {"wf_periods": 25, "desc": "WF=25 (current V6 baseline -- ~6 months)"},
    "D_wf40": {"wf_periods": 40, "desc": "WF=40 (~40 weeks training)"},
    "E_wf60": {"wf_periods": 60, "desc": "WF=60 (~14 months training)"},
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

# 21 features total (17 legacy + 3 cross-asset + 1 regime score)
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
# WALK-FORWARD LGBM RANKING (parameterized window)
# ================================================================

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """Build feature + target records for all sectors on all rebal dates."""
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features, DTE={DTE}")

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
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

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
                   "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols, wf_train_periods):
    """Walk-forward LGBM ranking with configurable sliding train window."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
        return {}, None

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
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
    return rankings, imp_df


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
# TRADE SIMULATION (fixed K=3)
# ================================================================

def simulate_trades(name, rankings, close, high, low, atr_dict):
    """Simulate trades using V6 config with fixed TOP_K=3."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    max_pos_bull = TOTAL_BULL_BUDGET / TOP_K
    max_pos_pair_leg = TOTAL_PAIR_LEG_BUDGET / TOP_K

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        if USE_PAIRS and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(max_pos_pair_leg, equity / 6)
        else:
            max_pos = min(max_pos_bull, equity / 3)

        if max_pos < 10:
            continue

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


def _execute_single_trade(tk, dt, direction, max_pos, close, atr_dict, cv, equity):
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

def random_baseline_test(rankings, close, high, low, atr_dict, n_trials=5):
    """Test if random sector selection produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials, K={TOP_K})...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict,
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
    fprint(f"WALK-FORWARD WINDOW LENGTH SENSITIVITY v1 -- {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"V6 Config: Weekly rebalance, pairs (VIX<20)")
    fprint(f"FIXED: DTE={DTE} | OTM={OTM_PCT*100:.0f}% | Spread={SPREAD_PCT:.0f}% | K={TOP_K}")
    fprint(f"Capital: ${CAP:.0f} | Total bull budget: ${TOTAL_BULL_BUDGET:.0f} | "
           f"Total pair leg budget: ${TOTAL_PAIR_LEG_BUDGET:.0f}")
    fprint(f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}")
    fprint(f"Features: {len(ALL_FEATURES)} (17 legacy + 3 cross-asset + 1 regime)")
    fprint(f"\n5 Variants (varying WF training window only):")
    for vname, cfg in VARIANTS.items():
        fprint(f"  {vname}: {cfg['desc']}")
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

    # Build feature records ONCE (shared by all variants -- only LGBM window differs)
    fprint("\n" + "=" * 80)
    fprint(f"Building feature records (shared dataset, DTE={DTE})")
    fprint("=" * 80)

    records = build_feature_records(
        close, high, low, rebal_dates, ALL_FEATURES, regime_series,
    )

    if len(records) < 100:
        fprint("ERROR: Insufficient feature records. Cannot proceed.")
        return

    # ================================================================
    # TRAIN SEPARATE LGBM FOR EACH WF WINDOW + SIMULATE
    # ================================================================
    all_results = {}
    all_importances = {}

    for vname, cfg in VARIANTS.items():
        wf_periods = cfg["wf_periods"]
        desc = cfg["desc"]

        fprint("\n" + "=" * 80)
        fprint(f"{vname}: {desc}")
        fprint(f"  Training LGBM with sliding window of {wf_periods} periods")
        fprint("=" * 80)

        # Train LGBM with this specific window size
        rankings, imp_df = walk_forward_lgbm_rank(records.copy(), ALL_FEATURES, wf_periods)

        if not rankings:
            fprint(f"  No rankings produced for WF={wf_periods}")
            all_results[vname] = {"description": desc, "wf_periods": wf_periods,
                                  "n_trades": 0, "error": "no_rankings"}
            continue

        if imp_df is not None:
            all_importances[vname] = imp_df

        fprint(f"  Rankings produced for {len(rankings)} dates")

        # Simulate trades
        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict,
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[vname] = {"description": desc, "wf_periods": wf_periods,
                                  "n_trades": len(trades) if trades else 0,
                                  "error": "too_few_trades"}
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

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"\n  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": desc,
            "wf_periods": wf_periods,
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "n_bull_trades": len(bull_trades),
            "n_bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 3),
            "bear_wr": round(bear_wr, 3),
            "n_ranking_dates": len(rankings),
        }

    # ================================================================
    # SUMMARY COMPARISON
    # ================================================================
    fprint("\n" + "=" * 95)
    fprint("SUMMARY: WALK-FORWARD WINDOW LENGTH SENSITIVITY")
    fprint("=" * 95)
    fprint(f"{'Variant':<12} {'WF':>3} {'~Weeks':>6} {'Ranks':>5} {'Trades':>6} {'Sharpe':>7} "
           f"{'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8}")
    fprint("-" * 95)

    for vname in VARIANTS:
        r = all_results.get(vname, {})
        wf = VARIANTS[vname]["wf_periods"]
        if "error" in r:
            fprint(f"  {vname:<12} {wf:>3} {wf:>5}w  -- {r.get('error','NO DATA')} --")
            continue
        if "sharpe" not in r:
            fprint(f"  {vname:<12} {wf:>3} -- NO DATA --")
            continue
        fprint(f"  {vname:<12} {wf:>3} {wf:>5}w {r.get('n_ranking_dates',0):>5} "
               f"{r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f}")

    # -- BIAS-VARIANCE TRADEOFF ANALYSIS --
    fprint("\n" + "=" * 95)
    fprint("BIAS-VARIANCE TRADEOFF ANALYSIS")
    fprint("=" * 95)

    valid_results = {k: v for k, v in all_results.items() if "sharpe" in v}

    if valid_results:
        # Best variant
        overall_best = max(valid_results.keys(), key=lambda k: valid_results[k]["sharpe"])
        ob = valid_results[overall_best]
        baseline = valid_results.get("C_wf25", {})

        fprint(f"\n  BEST VARIANT: {overall_best}")
        fprint(f"    WF={ob['wf_periods']}, Sharpe={ob['sharpe']:.2f}, "
               f"Sortino={ob['sortino']:.2f}, Gates={ob['gates_passed']}/{ob['gates_total']}")

        if baseline and "sharpe" in baseline:
            delta_sharpe = ob["sharpe"] - baseline["sharpe"]
            pct_delta = (delta_sharpe / abs(baseline["sharpe"])) * 100 if baseline["sharpe"] != 0 else float('inf')
            fprint(f"\n  vs V6 Baseline (C_wf25, WF=25):")
            fprint(f"    Sharpe delta: {delta_sharpe:+.2f} ({pct_delta:+.1f}%)")
            fprint(f"    Baseline: Sharpe={baseline['sharpe']:.2f}, "
                   f"Sortino={baseline['sortino']:.2f}, "
                   f"Gates={baseline['gates_passed']}/{baseline['gates_total']}")

        # Sharpe trend across WF window
        fprint(f"\n  Sharpe by WF window:")
        ks = sorted(valid_results.values(), key=lambda v: v["wf_periods"])
        for v in ks:
            bar = "#" * int(max(v["sharpe"], 0) * 10)
            fprint(f"    WF={v['wf_periods']:>2}: Sharpe={v['sharpe']:>6.2f}  {bar}")

        # Number of ranking dates (longer window = fewer OOT dates)
        fprint(f"\n  Ranking dates by WF window (longer window = fewer tradeable dates):")
        for v in ks:
            fprint(f"    WF={v['wf_periods']:>2}: {v.get('n_ranking_dates', 0)} dates, "
                   f"{v['n_trades']} trades")

        # Max drawdown comparison
        fprint(f"\n  Max drawdown by WF window:")
        for v in ks:
            bar = "#" * int(abs(v["max_dd"]) * 200)
            fprint(f"    WF={v['wf_periods']:>2}: MaxDD={v['max_dd']*100:>6.1f}%  {bar}")

        # Win rate + PF comparison
        fprint(f"\n  Win rate & profit factor by WF window:")
        for v in ks:
            fprint(f"    WF={v['wf_periods']:>2}: WR={v['win_rate']*100:>5.1f}%, "
                   f"PF={v['profit_factor']:>5.2f}")

    # -- FEATURE IMPORTANCE COMPARISON --
    if all_importances:
        fprint("\n" + "=" * 95)
        fprint("FEATURE IMPORTANCE COMPARISON (top 5 per variant)")
        fprint("=" * 95)
        for vname, imp_df in all_importances.items():
            wf = VARIANTS[vname]["wf_periods"]
            top5 = imp_df.head(5)
            features_str = ", ".join(f"{r['feature']}({r['importance']:.0f})"
                                     for _, r in top5.iterrows())
            fprint(f"  WF={wf:>2}: {features_str}")

    # -- KEY FINDINGS --
    fprint("\n" + "=" * 95)
    fprint("KEY FINDINGS")
    fprint("=" * 95)

    if valid_results:
        sharpes = [(v["wf_periods"], v["sharpe"]) for v in valid_results.values()]
        sharpes.sort(key=lambda x: x[0])

        increasing = all(sharpes[i][1] <= sharpes[i+1][1] for i in range(len(sharpes)-1))
        decreasing = all(sharpes[i][1] >= sharpes[i+1][1] for i in range(len(sharpes)-1))

        if decreasing:
            fprint("  SHORTER WINDOWS WIN: Sharpe decreases as WF increases.")
            fprint("  More recent data is more predictive -- market regime shifts matter.")
            fprint("  Implication: shorten the training window for better adaptivity.")
        elif increasing:
            fprint("  LONGER WINDOWS WIN: Sharpe increases as WF increases.")
            fprint("  More training data helps -- signal is stable across regimes.")
            fprint("  Implication: use more history for more robust models.")
        else:
            best_wf = max(sharpes, key=lambda x: x[1])
            fprint(f"  NON-MONOTONIC: Optimal WF={best_wf[0]} (Sharpe={best_wf[1]:.2f}).")
            fprint("  There's a sweet spot in the bias-variance tradeoff.")
            fprint("  Too short = high variance (noisy models).")
            fprint("  Too long = high bias (stale patterns).")

        best_sharpe = max(valid_results.values(), key=lambda v: v["sharpe"])
        best_sortino = max(valid_results.values(), key=lambda v: v["sortino"])
        lowest_dd = max(valid_results.values(), key=lambda v: v["max_dd"])

        fprint(f"\n  Best Sharpe:  WF={best_sharpe['wf_periods']} ({best_sharpe['sharpe']:.2f})")
        fprint(f"  Best Sortino: WF={best_sortino['wf_periods']} ({best_sortino['sortino']:.2f})")
        fprint(f"  Lowest MaxDD: WF={lowest_dd['wf_periods']} ({lowest_dd['max_dd']*100:.1f}%)")

        # Recommendation
        fprint(f"\n  RECOMMENDATION:")
        if best_sharpe['wf_periods'] == 25:
            fprint(f"  Current V6 baseline (WF=25) is already optimal. No change needed.")
        else:
            fprint(f"  Consider switching to WF={best_sharpe['wf_periods']} "
                   f"(Sharpe {best_sharpe['sharpe']:.2f} vs baseline "
                   f"{valid_results.get('C_wf25', {}).get('sharpe', 0):.2f})")

    # Save results
    results_path = OUTPUT_DIR / "wf_window_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save feature importances per variant
    if all_importances:
        imp_path = OUTPUT_DIR / "feature_importances_by_window.json"
        imp_data = {}
        for vname, imp_df in all_importances.items():
            imp_data[vname] = imp_df.to_dict(orient="records")
        with open(imp_path, "w") as f:
            json.dump(imp_data, f, indent=2, default=str)
        fprint(f"Feature importances saved to {imp_path}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"wf_window_{t0.strftime('%Y%m%d_%H%M')}"):
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
                    mlflow.log_metric(f"{prefix}_wf_periods", r.get("wf_periods", 0))
                    mlflow.log_metric(f"{prefix}_n_ranking_dates", r.get("n_ranking_dates", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "otm_pct": OTM_PCT,
                    "spread_pct": SPREAD_PCT,
                    "top_k": TOP_K,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "rebal_freq": REBAL_FREQ,
                    "use_pairs": USE_PAIRS,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(ALL_FEATURES),
                    "n_variants": len(VARIANTS),
                    "total_bull_budget": TOTAL_BULL_BUDGET,
                    "total_pair_leg_budget": TOTAL_PAIR_LEG_BUDGET,
                    "study": "wf_window_sensitivity",
                })

                mlflow.log_artifact(str(results_path))
                if all_importances:
                    mlflow.log_artifact(str(imp_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
