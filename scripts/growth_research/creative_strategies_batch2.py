"""
Creative Strategies Batch 2 — Three New Strategy Ideas
=======================================================
HC #0  : Sliding walk-forward (no expanding window)
HC #428: Regime-agnostic validation (R1 — all OOT days, regime gap < 0.50)
HC #694: Commission-free (Robinhood ETFs)

STRATEGIES:
  1. Dual Momentum (Antonacci-style) — absolute + relative momentum, monthly rebal
  2. Yield Curve Regime Signal — 2yr-10yr proxy via IEF/SHY as equity allocation signal
  3. Volatility Regime Switching + Mean-Reversion Overlay — VIX as entry/exit signal

DCA: $500 initial + $100/wk.  Period: 2012-2026.

VALIDATION GATES (all must pass):
  - Permutation test p < 0.05 (500 shuffles)
  - R1 regime test: |Sharpe_green - Sharpe_red| / max < 0.50
  - Sub-period consistency: positive Sharpe in all 3 sub-periods
  - Outlier robustness: remove best 10 days, Sharpe still positive
"""

import sys, os, json, warnings
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.stats import percentileofscore

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/creative_strategies_batch2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 500.0
WEEKLY_DCA = 100.0
START = "2011-01-01"  # buffer for lookback
END = "2026-07-17"
EFFECTIVE_START = "2012-01-01"

N_PERMUTATIONS = 500
np.random.seed(42)

ts = lambda: datetime.now().strftime('%H:%M:%S')

# =============================================================================
# 1. DATA DOWNLOAD
# =============================================================================
print(f"[{ts()}] Downloading data...", flush=True)

TICKERS = ["SPY", "UPRO", "EFA", "AGG", "GLD", "TLT", "IEF", "SHY", "QQQ", "IWM", "^VIX"]

raw = {}
for t in TICKERS:
    try:
        df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  WARNING: {t} empty", flush=True)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        raw[t] = df["Close"].rename(t)
        print(f"  {t}: {len(df)} rows ({df.index[0].date()} - {df.index[-1].date()})", flush=True)
    except Exception as e:
        print(f"  ERROR {t}: {e}", flush=True)

prices = pd.DataFrame(raw).sort_index().ffill()
prices.index = pd.to_datetime(prices.index)
prices = prices.loc[EFFECTIVE_START:]

vix = prices["^VIX"].rename("VIX") if "^VIX" in prices.columns else None
spy_ret = prices["SPY"].pct_change()
spy_close_to_close = spy_ret  # for regime classification

print(f"[{ts()}] Price matrix: {prices.shape}, {prices.index[0].date()} to {prices.index[-1].date()}", flush=True)

# =============================================================================
# HELPER: DCA Portfolio Simulator
# =============================================================================
def simulate_dca(daily_returns: pd.Series, name: str) -> pd.DataFrame:
    """Simulate $500 initial + $100/wk DCA on a daily return series."""
    dr = daily_returns.dropna()
    dates = dr.index
    portfolio_value = INITIAL_CAPITAL
    shares = INITIAL_CAPITAL  # treat as units
    values = []
    contributions = INITIAL_CAPITAL
    last_dca_week = None

    for i, (date, ret) in enumerate(dr.items()):
        # Apply return
        portfolio_value *= (1 + ret)

        # Weekly DCA (every Monday or first trading day of week)
        week_num = date.isocalendar()[1]
        year = date.year
        week_key = (year, week_num)
        if week_key != last_dca_week and i > 0:
            portfolio_value += WEEKLY_DCA
            contributions += WEEKLY_DCA
            last_dca_week = week_key

        values.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "contributions": contributions,
        })

    result = pd.DataFrame(values).set_index("date")
    result["strategy"] = name
    return result


def compute_metrics(daily_returns: pd.Series) -> dict:
    """Compute risk-adjusted metrics from daily returns."""
    dr = daily_returns.dropna()
    if len(dr) < 20:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "cagr": 0, "max_dd": -1, "n_days": len(dr)}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    wr = (dr > 0).mean()

    cum = (1 + dr).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    years = len(dr) / 252
    total_ret = cum.iloc[-1] / cum.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "n_days": len(dr),
    }


# =============================================================================
# HELPER: Classify green/red days for R1 regime test
# =============================================================================
def classify_regime_days(spy_returns: pd.Series) -> pd.Series:
    """Classify each day as green (SPY up) or red (SPY down)."""
    return (spy_returns > 0).map({True: "green", False: "red"})


# =============================================================================
# STRATEGY 1: DUAL MOMENTUM (Antonacci-style)
# =============================================================================
def run_dual_momentum(prices_df: pd.DataFrame, lookback_months: int = 12,
                       use_upro: bool = False, vix_filter: bool = False,
                       vix_series: pd.Series = None) -> pd.Series:
    """
    Antonacci Dual Momentum:
    - If SPY 'lookback_months' return > 0 AND SPY > EFA → hold SPY (or UPRO)
    - If SPY return > 0 but EFA > SPY → hold EFA
    - If SPY return <= 0 → hold AGG
    Monthly rebalance.
    """
    lookback_days = lookback_months * 21

    spy = prices_df["SPY"]
    efa = prices_df["EFA"]
    agg = prices_df["AGG"]
    upro = prices_df["UPRO"] if "UPRO" in prices_df.columns else spy

    spy_ret_daily = spy.pct_change()
    efa_ret_daily = efa.pct_change()
    agg_ret_daily = agg.pct_change()
    upro_ret_daily = upro.pct_change()

    # Monthly rebalance dates
    monthly = spy.resample("ME").last().index

    position = "AGG"  # start defensive
    daily_returns = []

    for date in spy.index:
        if date < spy.index[lookback_days]:
            daily_returns.append((date, 0.0))
            continue

        # Rebalance check: first trading day after month-end
        if any(date > m and date <= m + pd.Timedelta(days=5) for m in monthly):
            # Calculate lookback returns
            lookback_start_idx = spy.index.get_loc(date) - lookback_days
            if lookback_start_idx < 0:
                lookback_start_idx = 0
            lookback_start = spy.index[lookback_start_idx]

            spy_momentum = spy.loc[date] / spy.loc[lookback_start] - 1
            efa_momentum = efa.loc[date] / efa.loc[lookback_start] - 1

            # VIX filter: if VIX > 30, force defensive
            if vix_filter and vix_series is not None and date in vix_series.index:
                if vix_series.loc[date] > 30:
                    position = "AGG"
                    r = agg_ret_daily.get(date, 0.0)
                    daily_returns.append((date, r if not np.isnan(r) else 0.0))
                    continue

            # Dual momentum logic
            if spy_momentum > 0:
                if spy_momentum >= efa_momentum:
                    position = "UPRO" if use_upro else "SPY"
                else:
                    position = "EFA"
            else:
                position = "AGG"

        # Apply daily return based on current position
        if position == "SPY":
            r = spy_ret_daily.get(date, 0.0)
        elif position == "UPRO":
            r = upro_ret_daily.get(date, 0.0)
        elif position == "EFA":
            r = efa_ret_daily.get(date, 0.0)
        else:
            r = agg_ret_daily.get(date, 0.0)

        daily_returns.append((date, r if not np.isnan(r) else 0.0))

    result = pd.DataFrame(daily_returns, columns=["date", "return"]).set_index("date")
    return result["return"]


# =============================================================================
# STRATEGY 2: YIELD CURVE REGIME SIGNAL
# =============================================================================
def run_yield_curve_regime(prices_df: pd.DataFrame) -> pd.Series:
    """
    Yield curve proxy: IEF (7-10yr) vs SHY (1-3yr).
    IEF/SHY ratio as proxy for curve shape.
    - Normal (ratio rising / high): UPRO
    - Flattening (ratio declining): SPY
    - Inverted proxy (ratio below threshold): GLD/TLT mix
    - Recovery (ratio rising from low): UPRO
    Monthly rebalance.
    """
    ief = prices_df["IEF"]
    shy = prices_df["SHY"]
    spy = prices_df["SPY"]
    upro = prices_df["UPRO"] if "UPRO" in prices_df.columns else spy
    gld = prices_df["GLD"]
    tlt = prices_df["TLT"]

    # Curve proxy: IEF/SHY ratio (higher = steeper curve = normal)
    curve_ratio = ief / shy
    # 63-day (3-month) moving average for trend
    curve_ma = curve_ratio.rolling(63).mean()
    # 252-day percentile rank
    curve_pctile = curve_ratio.rolling(252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
    )

    spy_ret = spy.pct_change()
    upro_ret = upro.pct_change()
    gld_ret = gld.pct_change()
    tlt_ret = tlt.pct_change()

    monthly = spy.resample("ME").last().index
    regime = "NORMAL"
    daily_returns = []

    for date in spy.index:
        if date < spy.index[252]:
            daily_returns.append((date, 0.0))
            continue

        # Monthly rebalance
        if any(date > m and date <= m + pd.Timedelta(days=5) for m in monthly):
            cr = curve_ratio.get(date, np.nan)
            cm = curve_ma.get(date, np.nan)
            cp = curve_pctile.get(date, np.nan)

            if np.isnan(cr) or np.isnan(cm) or np.isnan(cp):
                regime = "NORMAL"
            elif cp < 0.15:
                # Curve very flat/inverted — defensive
                if cr > cm:
                    regime = "RECOVERY"  # rising from low = aggressive re-entry
                else:
                    regime = "INVERTED"
            elif cp < 0.35 and cr < cm:
                regime = "FLATTENING"
            else:
                regime = "NORMAL"

        # Apply returns based on regime
        if regime == "NORMAL":
            r = upro_ret.get(date, 0.0)
        elif regime == "RECOVERY":
            r = upro_ret.get(date, 0.0)
        elif regime == "FLATTENING":
            r = spy_ret.get(date, 0.0)
        elif regime == "INVERTED":
            # 50% GLD + 50% TLT
            r_gld = gld_ret.get(date, 0.0)
            r_tlt = tlt_ret.get(date, 0.0)
            r_gld = 0.0 if np.isnan(r_gld) else r_gld
            r_tlt = 0.0 if np.isnan(r_tlt) else r_tlt
            r = 0.5 * r_gld + 0.5 * r_tlt
        else:
            r = spy_ret.get(date, 0.0)

        daily_returns.append((date, r if not np.isnan(r) else 0.0))

    result = pd.DataFrame(daily_returns, columns=["date", "return"]).set_index("date")
    return result["return"]


# =============================================================================
# STRATEGY 3: VOLATILITY REGIME SWITCHING + MEAN-REVERSION OVERLAY
# =============================================================================
def run_vol_mean_reversion(prices_df: pd.DataFrame, vix_series: pd.Series) -> pd.Series:
    """
    VIX mean-reversion strategy:
    - Low VIX (<15) and declining: hold UPRO (sell vol environment)
    - VIX spike but mean-reverting (>20 but declining from peak): aggressive UPRO entry
    - VIX high (>25) and RISING: defensive (GLD/TLT)
    - Otherwise: SPY
    Daily signal, weekly rebalance.
    """
    spy = prices_df["SPY"]
    upro = prices_df["UPRO"] if "UPRO" in prices_df.columns else spy
    gld = prices_df["GLD"]
    tlt = prices_df["TLT"]

    spy_ret = spy.pct_change()
    upro_ret = upro.pct_change()
    gld_ret = gld.pct_change()
    tlt_ret = tlt.pct_change()

    # VIX features
    vix_ma10 = vix_series.rolling(10).mean()
    vix_ma50 = vix_series.rolling(50).mean()
    vix_pctile = vix_series.rolling(252).apply(
        lambda x: percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
    )
    # Rolling peak for mean-reversion detection
    vix_rolling_max_20 = vix_series.rolling(20).max()

    regime = "SPY"
    daily_returns = []
    last_rebal_week = None

    for date in spy.index:
        if date < spy.index[252]:
            daily_returns.append((date, 0.0))
            continue

        # Weekly rebalance (every Friday)
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_rebal_week:
            last_rebal_week = week_key

            v = vix_series.get(date, np.nan)
            v_ma10 = vix_ma10.get(date, np.nan)
            v_ma50 = vix_ma50.get(date, np.nan)
            v_peak20 = vix_rolling_max_20.get(date, np.nan)

            if np.isnan(v) or np.isnan(v_ma10) or np.isnan(v_ma50):
                regime = "SPY"
            elif v < 15 and v < v_ma10:
                # Low VIX, declining — complacency, ride leverage
                regime = "UPRO_LOW_VOL"
            elif v > 20 and v < v_peak20 * 0.85 and v < v_ma10:
                # VIX spiked but is now mean-reverting (declined 15%+ from 20d peak)
                # This is the buy-the-dip signal
                regime = "UPRO_MEAN_REV"
            elif v > 25 and v > v_ma10:
                # VIX high and still rising — duck
                regime = "DEFENSIVE"
            elif v > 20 and v > v_ma10:
                # Moderately elevated and rising
                regime = "CAUTIOUS"
            else:
                regime = "SPY"

        # Apply returns
        if regime in ("UPRO_LOW_VOL", "UPRO_MEAN_REV"):
            r = upro_ret.get(date, 0.0)
        elif regime == "DEFENSIVE":
            r_gld = gld_ret.get(date, 0.0)
            r_tlt = tlt_ret.get(date, 0.0)
            r_gld = 0.0 if np.isnan(r_gld) else r_gld
            r_tlt = 0.0 if np.isnan(r_tlt) else r_tlt
            r = 0.5 * r_gld + 0.5 * r_tlt
        elif regime == "CAUTIOUS":
            # 50% SPY + 50% TLT
            r_spy = spy_ret.get(date, 0.0)
            r_tlt = tlt_ret.get(date, 0.0)
            r_spy = 0.0 if np.isnan(r_spy) else r_spy
            r_tlt = 0.0 if np.isnan(r_tlt) else r_tlt
            r = 0.5 * r_spy + 0.5 * r_tlt
        else:
            r = spy_ret.get(date, 0.0)

        daily_returns.append((date, r if not np.isnan(r) else 0.0))

    result = pd.DataFrame(daily_returns, columns=["date", "return"]).set_index("date")
    return result["return"]


# =============================================================================
# VALIDATION FRAMEWORK
# =============================================================================
def run_permutation_test(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                          n_perms: int = N_PERMUTATIONS) -> float:
    """
    Permutation test: compare strategy excess return vs benchmark.
    Shuffle the EXCESS returns (strategy - benchmark) to test if timing adds value.
    H0: strategy timing is random (excess returns are noise).
    """
    aligned = pd.DataFrame({
        "strat": strategy_returns,
        "bench": benchmark_returns
    }).dropna()

    excess = (aligned["strat"] - aligned["bench"]).values
    observed_mean = np.mean(excess)

    count_better = 0
    for _ in range(n_perms):
        # Randomly flip signs of excess returns (destroy timing signal)
        signs = np.random.choice([-1, 1], size=len(excess))
        perm_mean = np.mean(excess * signs)
        if perm_mean >= observed_mean:
            count_better += 1

    return (count_better + 1) / (n_perms + 1)


def run_regime_test(strategy_returns: pd.Series, spy_returns: pd.Series) -> dict:
    """
    R1 regime test: strategy EXCESS Sharpe on green vs red SPY periods.
    Uses rolling 21-day windows classified by SPY cumulative return to avoid
    single-day noise. Compares excess returns (strategy - SPY) in each regime.
    """
    aligned = pd.DataFrame({
        "strat": strategy_returns,
        "spy": spy_returns
    }).dropna()

    # Classify regime by rolling 21-day SPY return (green = up market, red = down)
    spy_rolling = aligned["spy"].rolling(21).sum()
    excess = aligned["strat"] - aligned["spy"]

    green_mask = spy_rolling > 0
    red_mask = spy_rolling <= 0

    green_excess = excess[green_mask].dropna()
    red_excess = excess[red_mask].dropna()

    sharpe_green = (green_excess.mean() / green_excess.std() * np.sqrt(252)
                    if len(green_excess) > 20 and green_excess.std() > 0 else 0)
    sharpe_red = (red_excess.mean() / red_excess.std() * np.sqrt(252)
                  if len(red_excess) > 20 and red_excess.std() > 0 else 0)

    gap = abs(sharpe_green - sharpe_red)
    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_ratio = gap / max_sharpe if max_sharpe > 0 else 0

    return {
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap_ratio": round(regime_ratio, 3),
        "pass": regime_ratio < 0.50,
    }


def run_subperiod_test(strategy_returns: pd.Series) -> dict:
    """Split into 3 equal sub-periods, check Sharpe positive in all."""
    dr = strategy_returns.dropna()
    n = len(dr)
    split1 = n // 3
    split2 = 2 * n // 3

    periods = [
        dr.iloc[:split1],
        dr.iloc[split1:split2],
        dr.iloc[split2:],
    ]

    sharpes = []
    for i, p in enumerate(periods):
        s = p.mean() / p.std() * np.sqrt(252) if len(p) > 10 and p.std() > 0 else 0
        sharpes.append(round(s, 3))

    return {
        "period_sharpes": sharpes,
        "all_positive": all(s > 0 for s in sharpes),
        "pass": all(s > 0 for s in sharpes),
    }


def run_outlier_robustness(strategy_returns: pd.Series, n_remove: int = 10) -> dict:
    """Remove best N days, check Sharpe still positive."""
    dr = strategy_returns.dropna().copy()
    # Remove top N return days
    top_n_idx = dr.nlargest(n_remove).index
    trimmed = dr.drop(top_n_idx)

    sharpe_full = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    sharpe_trimmed = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0

    return {
        "sharpe_full": round(sharpe_full, 3),
        "sharpe_trimmed": round(sharpe_trimmed, 3),
        "pass": sharpe_trimmed > 0,
    }


def validate_strategy(name: str, returns: pd.Series, spy_returns: pd.Series) -> dict:
    """Run all 4 validation gates."""
    print(f"\n{'='*60}", flush=True)
    print(f"  VALIDATING: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Metrics
    metrics = compute_metrics(returns)
    print(f"  Sharpe={metrics['sharpe']}  Sortino={metrics['sortino']}  "
          f"PF={metrics['pf']}  WR={metrics['wr']}  CAGR={metrics['cagr']}  "
          f"MaxDD={metrics['max_dd']}  Days={metrics['n_days']}", flush=True)

    # Gate 1: Permutation test
    print(f"  [Gate 1] Permutation test ({N_PERMUTATIONS} shuffles)...", flush=True)
    p_val = run_permutation_test(returns.dropna(), spy_returns)
    perm_pass = p_val < 0.05
    print(f"    p-value = {p_val:.4f}  {'PASS' if perm_pass else 'FAIL'}", flush=True)

    # Gate 2: Regime test
    print(f"  [Gate 2] R1 regime test...", flush=True)
    regime = run_regime_test(returns, spy_returns)
    print(f"    Sharpe_green={regime['sharpe_green']}  Sharpe_red={regime['sharpe_red']}  "
          f"Gap={regime['regime_gap_ratio']}  {'PASS' if regime['pass'] else 'FAIL'}", flush=True)

    # Gate 3: Sub-period consistency
    print(f"  [Gate 3] Sub-period consistency...", flush=True)
    subperiod = run_subperiod_test(returns)
    print(f"    Period Sharpes: {subperiod['period_sharpes']}  "
          f"{'PASS' if subperiod['pass'] else 'FAIL'}", flush=True)

    # Gate 4: Outlier robustness
    print(f"  [Gate 4] Outlier robustness (remove best 10 days)...", flush=True)
    outlier = run_outlier_robustness(returns)
    print(f"    Full={outlier['sharpe_full']}  Trimmed={outlier['sharpe_trimmed']}  "
          f"{'PASS' if outlier['pass'] else 'FAIL'}", flush=True)

    gates_passed = sum([perm_pass, regime["pass"], subperiod["pass"], outlier["pass"]])
    overall = gates_passed == 4

    print(f"\n  RESULT: {gates_passed}/4 gates passed — "
          f"{'VALIDATED' if overall else 'FAILED'}", flush=True)

    return {
        "name": name,
        "metrics": metrics,
        "permutation_p": round(p_val, 4),
        "permutation_pass": perm_pass,
        "regime_test": regime,
        "subperiod_test": subperiod,
        "outlier_test": outlier,
        "gates_passed": gates_passed,
        "validated": overall,
    }


# =============================================================================
# RUN ALL STRATEGIES
# =============================================================================

results = {}
all_returns = {}

# SPY benchmark
spy_daily = prices["SPY"].pct_change().loc[EFFECTIVE_START:]

# --- STRATEGY 1: DUAL MOMENTUM VARIANTS ---
print(f"\n[{ts()}] === STRATEGY 1: DUAL MOMENTUM ===", flush=True)

dm_variants = [
    ("DualMom_12m",         {"lookback_months": 12, "use_upro": False, "vix_filter": False}),
    ("DualMom_6m",          {"lookback_months": 6,  "use_upro": False, "vix_filter": False}),
    ("DualMom_3m",          {"lookback_months": 3,  "use_upro": False, "vix_filter": False}),
    ("DualMom_1m",          {"lookback_months": 1,  "use_upro": False, "vix_filter": False}),
    ("DualMom_12m_UPRO",    {"lookback_months": 12, "use_upro": True,  "vix_filter": False}),
    ("DualMom_12m_VIXfilt", {"lookback_months": 12, "use_upro": False, "vix_filter": True}),
    ("DualMom_6m_UPRO_VIX", {"lookback_months": 6,  "use_upro": True,  "vix_filter": True}),
]

for name, params in dm_variants:
    print(f"\n[{ts()}] Running {name}...", flush=True)
    ret = run_dual_momentum(prices, vix_series=vix, **params)
    ret = ret.loc[EFFECTIVE_START:]
    all_returns[name] = ret
    metrics = compute_metrics(ret)
    print(f"  {name}: Sharpe={metrics['sharpe']} Sortino={metrics['sortino']} "
          f"CAGR={metrics['cagr']} MaxDD={metrics['max_dd']}", flush=True)

# Pick best DM variant by Sharpe
best_dm_name = max(
    [(n, compute_metrics(r)["sharpe"]) for n, r in all_returns.items() if n.startswith("DualMom")],
    key=lambda x: x[1]
)[0]
print(f"\n[{ts()}] Best Dual Momentum variant: {best_dm_name}", flush=True)
results["dual_momentum"] = validate_strategy(best_dm_name, all_returns[best_dm_name], spy_daily)

# Also validate the classic 12m version regardless
if best_dm_name != "DualMom_12m":
    results["dual_momentum_classic"] = validate_strategy("DualMom_12m", all_returns["DualMom_12m"], spy_daily)


# --- STRATEGY 2: YIELD CURVE REGIME ---
print(f"\n[{ts()}] === STRATEGY 2: YIELD CURVE REGIME ===", flush=True)
yc_ret = run_yield_curve_regime(prices)
yc_ret = yc_ret.loc[EFFECTIVE_START:]
all_returns["YieldCurve"] = yc_ret
results["yield_curve"] = validate_strategy("YieldCurve_Regime", yc_ret, spy_daily)


# --- STRATEGY 3: VOL MEAN-REVERSION ---
print(f"\n[{ts()}] === STRATEGY 3: VOL MEAN-REVERSION ===", flush=True)
if vix is not None:
    vmr_ret = run_vol_mean_reversion(prices, vix)
    vmr_ret = vmr_ret.loc[EFFECTIVE_START:]
    all_returns["VolMeanRev"] = vmr_ret
    results["vol_mean_reversion"] = validate_strategy("Vol_MeanReversion", vmr_ret, spy_daily)
else:
    print("  SKIP: VIX data not available", flush=True)


# --- BENCHMARK: Buy & Hold SPY ---
all_returns["SPY_BH"] = spy_daily


# =============================================================================
# DCA PORTFOLIO COMPARISON
# =============================================================================
print(f"\n[{ts()}] === DCA PORTFOLIO SIMULATION ===", flush=True)

dca_results = {}
for name, ret in all_returns.items():
    dca = simulate_dca(ret, name)
    final_val = dca["portfolio_value"].iloc[-1]
    total_contrib = dca["contributions"].iloc[-1]
    gain = final_val - total_contrib
    gain_pct = gain / total_contrib * 100
    dca_results[name] = {
        "final_value": round(final_val, 2),
        "contributions": round(total_contrib, 2),
        "gain": round(gain, 2),
        "gain_pct": round(gain_pct, 1),
    }
    print(f"  {name}: ${final_val:,.0f} (contributed ${total_contrib:,.0f}, "
          f"gain ${gain:,.0f} = {gain_pct:.1f}%)", flush=True)


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print(f"\n\n{'='*80}", flush=True)
print(f"  CREATIVE STRATEGIES BATCH 2 — FINAL RESULTS", flush=True)
print(f"{'='*80}\n", flush=True)

summary_rows = []
for key, res in results.items():
    m = res["metrics"]
    summary_rows.append({
        "Strategy": res["name"],
        "Sharpe": m["sharpe"],
        "Sortino": m["sortino"],
        "CAGR": f"{m['cagr']*100:.1f}%",
        "MaxDD": f"{m['max_dd']*100:.1f}%",
        "PF": m["pf"],
        "WR": f"{m['wr']*100:.1f}%",
        "Perm_p": res["permutation_p"],
        "Regime": "PASS" if res["regime_test"]["pass"] else "FAIL",
        "SubPer": "PASS" if res["subperiod_test"]["pass"] else "FAIL",
        "Outlier": "PASS" if res["outlier_test"]["pass"] else "FAIL",
        "Gates": f"{res['gates_passed']}/4",
        "VERDICT": "VALIDATED" if res["validated"] else "FAILED",
    })

# Add SPY benchmark
spy_metrics = compute_metrics(spy_daily)
summary_rows.append({
    "Strategy": "SPY_BuyHold",
    "Sharpe": spy_metrics["sharpe"],
    "Sortino": spy_metrics["sortino"],
    "CAGR": f"{spy_metrics['cagr']*100:.1f}%",
    "MaxDD": f"{spy_metrics['max_dd']*100:.1f}%",
    "PF": spy_metrics["pf"],
    "WR": f"{spy_metrics['wr']*100:.1f}%",
    "Perm_p": "-",
    "Regime": "-",
    "SubPer": "-",
    "Outlier": "-",
    "Gates": "-",
    "VERDICT": "BENCHMARK",
})

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False), flush=True)

# All variant metrics
print(f"\n\n--- ALL DUAL MOMENTUM VARIANTS ---", flush=True)
for name in sorted(all_returns.keys()):
    if name.startswith("DualMom"):
        m = compute_metrics(all_returns[name])
        d = dca_results.get(name, {})
        print(f"  {name:25s}  Sharpe={m['sharpe']:6.3f}  CAGR={m['cagr']*100:5.1f}%  "
              f"MaxDD={m['max_dd']*100:6.1f}%  DCA=${d.get('final_value',0):>10,.0f}", flush=True)


# =============================================================================
# SAVE RESULTS
# =============================================================================
output = {
    "run_date": datetime.now().isoformat(),
    "period": f"{EFFECTIVE_START} to {END}",
    "dca": {"initial": INITIAL_CAPITAL, "weekly": WEEKLY_DCA},
    "strategies": {},
    "dca_results": dca_results,
    "spy_benchmark": spy_metrics,
}

for key, res in results.items():
    output["strategies"][key] = res

# JSON-safe
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_dict(v) for v in d]
    return make_serializable(d)

output = clean_dict(output)

results_path = OUTPUT_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n[{ts()}] Results saved to {results_path}", flush=True)


# =============================================================================
# CHARTS
# =============================================================================
print(f"\n[{ts()}] Generating charts...", flush=True)

# Chart 1: Equity curves (DCA)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top: All strategies cumulative returns (no DCA, pure return)
ax = axes[0]
for name in ["SPY_BH", best_dm_name, "YieldCurve", "VolMeanRev"]:
    if name in all_returns:
        cum = (1 + all_returns[name].dropna()).cumprod()
        label = name if name != "SPY_BH" else "SPY Buy&Hold"
        ax.plot(cum.index, cum.values, label=label, linewidth=1.5)
ax.set_title("Cumulative Returns (No DCA, $1 Start)")
ax.legend(fontsize=9)
ax.set_ylabel("Growth of $1")
ax.grid(True, alpha=0.3)

# Bottom: DCA portfolio values
ax = axes[1]
for name in ["SPY_BH", best_dm_name, "YieldCurve", "VolMeanRev"]:
    if name in all_returns:
        dca = simulate_dca(all_returns[name], name)
        label = name if name != "SPY_BH" else "SPY Buy&Hold"
        ax.plot(dca.index, dca["portfolio_value"], label=label, linewidth=1.5)
        # Also plot contributions as dashed line
if "SPY_BH" in all_returns:
    dca_spy = simulate_dca(all_returns["SPY_BH"], "SPY")
    ax.plot(dca_spy.index, dca_spy["contributions"], '--', color='gray',
            label="Total Contributed", linewidth=1, alpha=0.6)
ax.set_title("DCA Portfolio Value ($500 + $100/wk)")
ax.legend(fontsize=9)
ax.set_ylabel("Portfolio Value ($)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = OUTPUT_DIR / "equity_curves.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {chart_path}", flush=True)

# Chart 2: Validation summary heatmap
fig, ax = plt.subplots(figsize=(10, 4))
strat_names = [r["name"] for r in results.values()]
gate_names = ["Permutation\np<0.05", "Regime\nGap<0.50", "Sub-period\nAll Positive", "Outlier\nRobust"]
gate_matrix = []
for res in results.values():
    gate_matrix.append([
        1 if res["permutation_pass"] else 0,
        1 if res["regime_test"]["pass"] else 0,
        1 if res["subperiod_test"]["pass"] else 0,
        1 if res["outlier_test"]["pass"] else 0,
    ])

gate_arr = np.array(gate_matrix)
im = ax.imshow(gate_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(gate_names)))
ax.set_xticklabels(gate_names, fontsize=10)
ax.set_yticks(range(len(strat_names)))
ax.set_yticklabels(strat_names, fontsize=10)
for i in range(len(strat_names)):
    for j in range(len(gate_names)):
        ax.text(j, i, "PASS" if gate_arr[i, j] else "FAIL",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if gate_arr[i, j] == 0 else "black")
ax.set_title("Validation Gates — Creative Strategies Batch 2")
plt.tight_layout()
val_chart_path = OUTPUT_DIR / "validation_gates.png"
plt.savefig(val_chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {val_chart_path}", flush=True)


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n\n{'='*80}", flush=True)
print(f"  BATCH 2 COMPLETE", flush=True)
print(f"{'='*80}", flush=True)

validated = [r["name"] for r in results.values() if r["validated"]]
failed = [r["name"] for r in results.values() if not r["validated"]]

print(f"\n  VALIDATED ({len(validated)}): {', '.join(validated) if validated else 'NONE'}", flush=True)
print(f"  FAILED ({len(failed)}): {', '.join(failed) if failed else 'NONE'}", flush=True)

# Best strategy
if validated:
    best = max(validated, key=lambda n: next(r["metrics"]["sharpe"] for r in results.values() if r["name"] == n))
    best_m = next(r["metrics"] for r in results.values() if r["name"] == best)
    print(f"\n  BEST VALIDATED: {best}", flush=True)
    print(f"    Sharpe={best_m['sharpe']}  Sortino={best_m['sortino']}  "
          f"CAGR={best_m['cagr']*100:.1f}%  MaxDD={best_m['max_dd']*100:.1f}%", flush=True)

print(f"\n[{ts()}] Done.", flush=True)
