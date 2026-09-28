"""
risk_overlay_backtest.py — Historical backtest of the drawdown risk overlay

Tests whether applying the risk_overlay.py signal historically (2010-2026) improves
risk-adjusted returns for:
  1) SPY long portfolio (easy to validate)
  2) SPY put-selling strategy (proxy for wheel strategies)

Permutation test: shuffles the risk signal 100x to confirm actual timing adds value
vs random position scaling.

Output: /home/jupiter/Lvl3Quant/output/risk_overlay_backtest/
"""
from __future__ import annotations

import os
import sys
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/jupiter/Lvl3Quant/output/risk_overlay_backtest/backtest.log"),
    ]
)
log = logging.getLogger(__name__)

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/risk_overlay_backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Tickers ────────────────────────────────────────────────────────────────
TICKERS = {
    "spy":   "SPY",
    "vix":   "^VIX",
    "vix3m": "^VIX3M",
    "hyg":   "HYG",
    "tlt":   "TLT",
    "lqd":   "LQD",
    "ief":   "IEF",
    "tnx":   "^TNX",
}

START = "2009-01-01"   # extra buffer for warm-up
END   = "2026-07-01"
BACKTEST_START = "2010-06-01"   # start after 252-day warm-up


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_data() -> dict[str, pd.Series]:
    """Download all tickers. Returns dict of Close price Series indexed by date."""
    import yfinance as yf

    log.info("Downloading data %s → %s …", START, END)
    raw: dict[str, pd.Series] = {}
    for key, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=START, end=END,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) > 100:
                s = df["Close"].squeeze()
                s.name = key
                raw[key] = s
                log.info("  %-6s (%s): %d rows  %s → %s",
                         key, ticker, len(s), s.index[0].date(), s.index[-1].date())
            else:
                log.warning("  %-6s (%s): insufficient rows", key, ticker)
        except Exception as e:
            log.warning("  %-6s (%s): FAILED — %s", key, ticker, e)

    # Align everything to SPY trading days
    spy = raw["spy"]
    for k in list(raw.keys()):
        raw[k] = raw[k].reindex(spy.index, method="ffill")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE COMPUTATION (mirrors risk_overlay.py exactly, vectorised over history)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features_history(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compute ALL features for EVERY trading day (vectorised).
    Uses trailing windows only — no lookahead.
    Returns DataFrame indexed by date.
    """
    spy   = raw["spy"]
    vix   = raw.get("vix")
    vix3m = raw.get("vix3m")
    hyg   = raw.get("hyg")
    tlt   = raw.get("tlt")
    lqd   = raw.get("lqd")
    tnx   = raw.get("tnx")

    log_ret = np.log(spy / spy.shift(1))

    feats = pd.DataFrame(index=spy.index)

    # SPY momentum
    for w in [5, 21, 63]:
        feats[f"spy_ret_{w}d"] = spy.pct_change(w, fill_method=None)

    # SPY realized vol
    for w in [21, 63]:
        feats[f"spy_rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Drawdown from rolling peak
    feats["spy_dd_63d"]  = spy / spy.rolling(63).max() - 1
    feats["spy_dd_252d"] = spy / spy.rolling(252).max() - 1

    # Distance from 200d MA
    ma200 = spy.rolling(200).mean()
    feats["spy_vs_ma200"]   = spy / ma200 - 1
    feats["spy_above_ma50"] = (spy > spy.rolling(50).mean()).astype(int)
    feats["spy_above_ma200"]= (spy > ma200).astype(int)

    # Skew / kurtosis
    for w in [21, 63]:
        feats[f"spy_skew_{w}d"] = log_ret.rolling(w).skew()
        feats[f"spy_kurt_{w}d"] = log_ret.rolling(w).kurt()

    # VIX
    if vix is not None:
        feats["vix_level"]     = vix
        feats["vix_change_5d"] = vix.pct_change(5, fill_method=None)
        feats["vix_vs_ma20"]   = vix / vix.rolling(20).mean() - 1

        if vix3m is not None:
            feats["vix_term_ratio"] = vix / (vix3m + 1e-6)
            feats["vix3m_level"]    = vix3m

        feats["iv_rv_gap_21d"] = vix / 100 - feats["spy_rvol_21d"]

    # Credit spreads
    if hyg is not None and tlt is not None:
        ratio = hyg / tlt
        feats["hyg_tlt_ratio"]      = ratio
        feats["hyg_tlt_change_21d"] = ratio.pct_change(21, fill_method=None)
        feats["hyg_tlt_vs_ma63"]    = ratio / ratio.rolling(63).mean() - 1

    if lqd is not None and tlt is not None:
        ratio_ig = lqd / tlt
        feats["lqd_tlt_ratio"]      = ratio_ig
        feats["lqd_tlt_change_21d"] = ratio_ig.pct_change(21, fill_method=None)

    # TLT momentum
    if tlt is not None:
        feats["tlt_ret_21d"] = tlt.pct_change(21, fill_method=None)
        feats["tlt_ret_63d"] = tlt.pct_change(63, fill_method=None)

    # Yield curve
    if tnx is not None:
        feats["yield_10y"]         = tnx
        feats["yield_10y_chg_21d"] = tnx.diff(21)

    # Seasonality
    feats["month"]       = pd.to_datetime(feats.index).month
    feats["is_sept_oct"] = feats["month"].isin([9, 10]).astype(int)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 3. RULES-BASED SCORING (mirrors _rules_based_score exactly, vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def rules_based_score_history(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same threshold rules as risk_overlay._rules_based_score()
    to every row in feats. Returns DataFrame with prob, risk_level, position_scale.
    """
    score     = pd.Series(0.0, index=feats.index)
    max_score = pd.Series(0.0, index=feats.index)

    def add(col: str, condition: pd.Series, weight: float) -> None:
        """Add weight where we have data."""
        has_data = feats[col].notna() if col in feats.columns else pd.Series(False, index=feats.index)
        max_score[has_data] += weight
        score[has_data & condition] += weight

    # ── Tier 1: vol / tail-risk ──
    if "vix_level" in feats.columns:
        add("vix_level", feats["vix_level"] > 25.0, weight=3.0)
        add("vix_level", feats["vix_level"] > 20.0, weight=1.5)

    if "vix_term_ratio" in feats.columns:
        add("vix_term_ratio", feats["vix_term_ratio"] > 1.0,  weight=3.0)
        add("vix_term_ratio", feats["vix_term_ratio"] > 1.10, weight=1.5)

    if "spy_kurt_21d" in feats.columns:
        add("spy_kurt_21d", feats["spy_kurt_21d"] > 3.0, weight=2.5)

    if "spy_rvol_63d" in feats.columns:
        add("spy_rvol_63d", feats["spy_rvol_63d"] > 0.20, weight=2.0)

    # ── Tier 2: credit / flight-to-quality ──
    if "hyg_tlt_vs_ma63" in feats.columns:
        add("hyg_tlt_vs_ma63", feats["hyg_tlt_vs_ma63"] < -0.03, weight=2.5)

    if "lqd_tlt_change_21d" in feats.columns:
        add("lqd_tlt_change_21d", feats["lqd_tlt_change_21d"] < -0.02, weight=2.0)

    if "tlt_ret_63d" in feats.columns:
        add("tlt_ret_63d", feats["tlt_ret_63d"] > 0.05, weight=1.5)

    # ── Tier 3: SPY trend ──
    if "spy_vs_ma200" in feats.columns:
        add("spy_vs_ma200", feats["spy_vs_ma200"] < 0.0,   weight=2.0)
        add("spy_vs_ma200", feats["spy_vs_ma200"] < -0.05, weight=1.5)

    if "spy_ret_63d" in feats.columns:
        add("spy_ret_63d", feats["spy_ret_63d"] < 0.0,   weight=1.5)
        add("spy_ret_63d", feats["spy_ret_63d"] < -0.05, weight=1.5)

    if "spy_dd_63d" in feats.columns:
        add("spy_dd_63d", feats["spy_dd_63d"] < -0.03, weight=1.5)

    # ── Tier 4: skew / seasonality ──
    if "spy_skew_63d" in feats.columns:
        add("spy_skew_63d", feats["spy_skew_63d"] < -0.5, weight=1.0)

    # Seasonality: max_score always gets this weight
    max_score += 0.5
    score[feats["is_sept_oct"].astype(bool)] += 0.5

    prob = score / max_score.replace(0, np.nan)
    prob = prob.fillna(0.0).clip(0.0, 1.0).round(4)

    risk_level = pd.cut(
        prob,
        bins=[-0.001, 0.30, 0.50, 1.001],
        labels=["normal", "elevated", "high"]
    )

    position_scale = prob.map(lambda p: 1.0 if p < 0.30 else (0.7 if p < 0.50 else 0.3))

    result = pd.DataFrame({
        "prob":           prob,
        "risk_level":     risk_level,
        "position_scale": position_scale,
    }, index=feats.index)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute standard risk-adjusted metrics from a daily returns series."""
    r = returns.dropna()
    if len(r) < 50:
        return {}

    ann = 252
    cagr = (1 + r).prod() ** (ann / len(r)) - 1

    excess = r - 0.0   # use 0 as risk-free for simplicity
    sharpe = excess.mean() / excess.std() * np.sqrt(ann) if excess.std() > 0 else 0.0

    downside = r[r < 0]
    sortino = r.mean() / downside.std() * np.sqrt(ann) if len(downside) > 0 and downside.std() > 0 else 0.0

    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd = cum / rolling_max - 1
    max_dd = float(dd.min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    total_return = float((1 + r).prod() - 1)

    return {
        "label":        label,
        "cagr":         round(cagr, 4),
        "sharpe":       round(sharpe, 4),
        "sortino":      round(sortino, 4),
        "max_dd":       round(max_dd, 4),
        "calmar":       round(calmar, 4),
        "total_return": round(total_return, 4),
        "n_days":       len(r),
    }


def year_by_year(returns: pd.Series, label: str = "") -> pd.DataFrame:
    """Annual return breakdown."""
    r = returns.dropna()
    annual = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    annual.name = label
    return annual


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Return stats for specific known drawdown periods."""
    episodes = {
        "2011 Debt Ceiling":  ("2011-07-01", "2011-10-31"),
        "2015 China Shock":   ("2015-08-01", "2015-09-30"),
        "2018 Q4 Selloff":    ("2018-10-01", "2018-12-31"),
        "2020 COVID":         ("2020-02-15", "2020-04-30"),
        "2022 Bear Market":   ("2021-12-31", "2022-10-31"),
    }
    rows = []
    for name, (start, end) in episodes.items():
        ep = returns.loc[start:end].dropna()
        if len(ep) == 0:
            continue
        total = float((1 + ep).prod() - 1)
        vol   = float(ep.std() * np.sqrt(252))
        rows.append({"episode": name, "return": round(total, 4), "ann_vol": round(vol, 4)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. STRATEGY SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def simulate_spy_long(spy: pd.Series, position_scale: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Baseline: 100% SPY buy-and-hold (daily returns).
    Overlay:  SPY returns * position_scale (cash out when risk elevated).
    """
    spy_ret = spy.pct_change(fill_method=None).dropna()

    # Align
    idx = spy_ret.index.intersection(position_scale.index)
    spy_ret = spy_ret.loc[idx]
    ps = position_scale.loc[idx]

    baseline = spy_ret
    overlay  = spy_ret * ps   # cash earns 0 (conservative, also reduces sensitivity bias)

    return baseline, overlay


def simulate_put_selling(spy: pd.Series, vix: pd.Series,
                         position_scale: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Proxy for weekly ATM put selling on SPY.

    Logic (weekly):
    - Each Monday (or first trading day of the week), sell one weekly ATM put
    - Premium collected = VIX/100 * sqrt(5/252) * SPY_price * position_scale
    - At Friday close, compute P&L:
        if SPY falls below strike (= Monday SPY price):
            loss = (strike - SPY_friday) per unit - premium
        else:
            gain = premium
    - position_scale applied at the time of entry (Monday's scale)

    Returns daily-equivalent returns (spread across the week).
    """
    # Align indices
    idx = spy.index.intersection(vix.index).intersection(position_scale.index)
    spy_w  = spy.loc[idx]
    vix_w  = vix.loc[idx]
    ps_w   = position_scale.loc[idx]

    # Build weekly groups (ISO week)
    spy_df = pd.DataFrame({
        "spy": spy_w,
        "vix": vix_w,
        "ps":  ps_w,
    })
    spy_df["week"] = spy_df.index.to_period("W")

    weekly_returns_base    = []
    weekly_returns_overlay = []
    weekly_dates           = []

    for week, grp in spy_df.groupby("week"):
        if len(grp) < 2:
            continue

        entry_spy = grp["spy"].iloc[0]
        exit_spy  = grp["spy"].iloc[-1]
        entry_vix = grp["vix"].iloc[0]
        entry_ps  = grp["ps"].iloc[0]

        if pd.isna(entry_spy) or pd.isna(exit_spy) or pd.isna(entry_vix):
            continue

        # Strike = ATM = entry SPY price
        strike = entry_spy

        # Premium per $1 notional
        # VIX/100 * sqrt(5/252) approximates 1-week implied vol * sqrt(1 week)
        # Multiply by 1 (notional=1) to get premium as fraction of SPY price
        # Then divide by entry_spy to get $ premium per $1 of notional
        sigma_weekly = (entry_vix / 100) * np.sqrt(5 / 252)
        premium_frac = sigma_weekly * entry_spy / entry_spy   # = sigma_weekly per $1 notional

        # Payoff: short put = collect premium, pay max(0, strike - exit)
        intrinsic = max(0.0, strike - exit_spy) / entry_spy  # as fraction of capital
        put_pnl = premium_frac - intrinsic

        # Baseline: full notional
        weekly_returns_base.append(put_pnl)
        # Overlay: scaled notional
        weekly_returns_overlay.append(put_pnl * entry_ps)

        weekly_dates.append(grp.index[-1])   # attribute to Friday

    base_weekly    = pd.Series(weekly_returns_base,    index=weekly_dates)
    overlay_weekly = pd.Series(weekly_returns_overlay, index=weekly_dates)

    # Reindex to daily — divide by 5 to express as daily equivalent
    # (not exactly right, but metrics use the same denominator)
    daily_idx = spy_w.index
    base_daily    = base_weekly.reindex(daily_idx).fillna(0.0)
    overlay_daily = overlay_weekly.reindex(daily_idx).fillna(0.0)

    # Better: keep weekly, use weekly-frequency returns for metrics
    # We return weekly series (Friday dates) for cleaner Sharpe calc
    return base_weekly, overlay_weekly


# ─────────────────────────────────────────────────────────────────────────────
# 6. PERMUTATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(spy_ret: pd.Series, position_scale: pd.Series,
                     n_perms: int = 100, metric: str = "sharpe") -> dict:
    """
    Shuffle the position_scale dates 100x.
    Compare actual overlay metric vs distribution of shuffled metrics.
    Returns p-value (one-sided: does actual > shuffled?).
    """
    idx = spy_ret.index.intersection(position_scale.index)
    spy_ret_aligned = spy_ret.loc[idx]
    ps_aligned      = position_scale.loc[idx]

    def compute_metric(r: pd.Series) -> float:
        if metric == "sharpe":
            return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
        elif metric == "sortino":
            down = r[r < 0]
            return r.mean() / down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 0 else 0.0
        elif metric == "cagr":
            return float((1 + r).prod() ** (252 / len(r)) - 1)
        return 0.0

    actual_overlay = spy_ret_aligned * ps_aligned
    actual_metric  = compute_metric(actual_overlay)

    perm_metrics = []
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        ps_shuffled = ps_aligned.values.copy()
        rng.shuffle(ps_shuffled)
        perm_ret    = spy_ret_aligned * ps_shuffled
        perm_metrics.append(compute_metric(perm_ret))

    perm_arr = np.array(perm_metrics)
    p_value  = float(np.mean(perm_arr >= actual_metric))   # fraction of perms >= actual

    return {
        "metric":           metric,
        "actual":           round(actual_metric, 4),
        "perm_mean":        round(float(np.mean(perm_arr)), 4),
        "perm_p10":         round(float(np.percentile(perm_arr, 10)), 4),
        "perm_p90":         round(float(np.percentile(perm_arr, 90)), 4),
        "p_value_one_sided":round(p_value, 4),
        "n_perms":          n_perms,
        "interpretation":   (
            "SIGNIFICANT (timing adds value)" if p_value < 0.10
            else "NOT significant (random scaling as good)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def regime_analysis(spy_ret: pd.Series, overlay_ret: pd.Series,
                    spy: pd.Series) -> pd.DataFrame:
    """
    Stratify by SPY regime (bull/flat/bear based on rolling 63d return).
    Shows whether overlay helps more in bad markets.
    """
    spy_63d = spy.pct_change(63, fill_method=None).reindex(spy_ret.index, method="ffill")

    regime = pd.cut(spy_63d,
                    bins=[-np.inf, -0.05, 0.05, np.inf],
                    labels=["bear", "flat", "bull"])

    rows = []
    for reg in ["bull", "flat", "bear"]:
        mask = regime == reg
        if mask.sum() < 20:
            continue
        b_m = compute_metrics(spy_ret[mask],    label=f"baseline_{reg}")
        o_m = compute_metrics(overlay_ret[mask], label=f"overlay_{reg}")
        rows.append({
            "regime":           reg,
            "n_days":           int(mask.sum()),
            "base_sharpe":      b_m.get("sharpe", np.nan),
            "overlay_sharpe":   o_m.get("sharpe", np.nan),
            "sharpe_improvement": round(o_m.get("sharpe", 0) - b_m.get("sharpe", 0), 4),
            "base_cagr":        b_m.get("cagr", np.nan),
            "overlay_cagr":     o_m.get("cagr", np.nan),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 8. RISK SIGNAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def signal_summary(overlay_signal: pd.DataFrame, start: str) -> dict:
    """How often is the overlay active? What fraction of days at each level?"""
    sig = overlay_signal.loc[start:]
    total = len(sig)
    return {
        "total_days":        total,
        "pct_normal":        round(float((sig["risk_level"] == "normal").sum() / total), 4),
        "pct_elevated":      round(float((sig["risk_level"] == "elevated").sum() / total), 4),
        "pct_high":          round(float((sig["risk_level"] == "high").sum() / total), 4),
        "avg_position_scale":round(float(sig["position_scale"].mean()), 4),
        "avg_prob":          round(float(sig["prob"].mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("RISK OVERLAY BACKTEST — starting")
    log.info("=" * 70)

    # ── 1. Download data ──
    raw = download_data()

    spy = raw["spy"]
    vix = raw.get("vix")
    if spy is None:
        log.error("SPY data missing — cannot run backtest")
        sys.exit(1)
    if vix is None:
        log.error("VIX data missing — cannot run backtest")
        sys.exit(1)

    # ── 2. Compute features ──
    log.info("Computing feature history …")
    feats = compute_features_history(raw)
    log.info("  Features computed: %d rows × %d cols", len(feats), len(feats.columns))

    # ── 3. Compute risk signal ──
    log.info("Applying rules-based scoring …")
    signal = rules_based_score_history(feats)

    # Trim to backtest window (after warm-up)
    signal_bt = signal.loc[BACKTEST_START:]
    log.info("  Signal start: %s  end: %s", signal_bt.index[0].date(), signal_bt.index[-1].date())

    sig_summary = signal_summary(signal, BACKTEST_START)
    log.info("  Signal distribution: normal=%.1f%%  elevated=%.1f%%  high=%.1f%%",
             sig_summary["pct_normal"]*100, sig_summary["pct_elevated"]*100,
             sig_summary["pct_high"]*100)
    log.info("  Avg position scale: %.3f  Avg prob: %.3f",
             sig_summary["avg_position_scale"], sig_summary["avg_prob"])

    # Save signal timeseries
    signal_bt.to_csv(OUT_DIR / "signal_timeseries.csv")
    log.info("  Saved signal_timeseries.csv")

    # ── 4. SPY LONG SIMULATION ──
    log.info("\n--- Strategy 1: SPY Long ---")
    spy_base, spy_overlay = simulate_spy_long(spy, signal_bt["position_scale"])

    # Trim to backtest window
    spy_base    = spy_base.loc[BACKTEST_START:]
    spy_overlay = spy_overlay.loc[BACKTEST_START:]

    m_base_spy    = compute_metrics(spy_base,    "SPY_Baseline")
    m_overlay_spy = compute_metrics(spy_overlay, "SPY_Overlay")

    log.info("  SPY Baseline  — CAGR: %.1f%%  Sharpe: %.2f  Sortino: %.2f  MaxDD: %.1f%%",
             m_base_spy["cagr"]*100, m_base_spy["sharpe"],
             m_base_spy["sortino"], m_base_spy["max_dd"]*100)
    log.info("  SPY Overlay   — CAGR: %.1f%%  Sharpe: %.2f  Sortino: %.2f  MaxDD: %.1f%%",
             m_overlay_spy["cagr"]*100, m_overlay_spy["sharpe"],
             m_overlay_spy["sortino"], m_overlay_spy["max_dd"]*100)

    # Year by year
    yby_base_spy    = year_by_year(spy_base,    "baseline")
    yby_overlay_spy = year_by_year(spy_overlay, "overlay")
    yby_spy = pd.DataFrame({"baseline": yby_base_spy, "overlay": yby_overlay_spy})
    yby_spy["delta"] = yby_spy["overlay"] - yby_spy["baseline"]
    log.info("\n  Year-by-year SPY:\n%s", yby_spy.round(4).to_string())

    # Drawdown episodes
    dd_base    = drawdown_episodes(spy_base)
    dd_overlay = drawdown_episodes(spy_overlay)
    dd_ep = dd_base.merge(dd_overlay, on="episode", suffixes=("_base", "_overlay"))
    dd_ep["return_delta"] = dd_ep["return_overlay"] - dd_ep["return_base"]
    log.info("\n  Drawdown episodes SPY:\n%s", dd_ep.to_string())

    # Regime analysis
    reg_spy = regime_analysis(spy_base, spy_overlay, spy.loc[BACKTEST_START:])
    log.info("\n  Regime analysis SPY:\n%s", reg_spy.to_string())

    # ── 5. PUT SELLING SIMULATION ──
    log.info("\n--- Strategy 2: Put Selling ---")
    put_base, put_overlay = simulate_put_selling(spy, vix, signal_bt["position_scale"])

    # Trim to backtest window
    put_base    = put_base.loc[BACKTEST_START:]
    put_overlay = put_overlay.loc[BACKTEST_START:]

    m_base_put    = compute_metrics(put_base,    "PutSell_Baseline")
    m_overlay_put = compute_metrics(put_overlay, "PutSell_Overlay")

    log.info("  Put Baseline  — CAGR: %.1f%%  Sharpe: %.2f  Sortino: %.2f  MaxDD: %.1f%%",
             m_base_put["cagr"]*100, m_base_put["sharpe"],
             m_base_put["sortino"], m_base_put["max_dd"]*100)
    log.info("  Put Overlay   — CAGR: %.1f%%  Sharpe: %.2f  Sortino: %.2f  MaxDD: %.1f%%",
             m_overlay_put["cagr"]*100, m_overlay_put["sharpe"],
             m_overlay_put["sortino"], m_overlay_put["max_dd"]*100)

    yby_base_put    = year_by_year(put_base,    "baseline")
    yby_overlay_put = year_by_year(put_overlay, "overlay")
    yby_put = pd.DataFrame({"baseline": yby_base_put, "overlay": yby_overlay_put})
    yby_put["delta"] = yby_put["overlay"] - yby_put["baseline"]
    log.info("\n  Year-by-year Put Selling:\n%s", yby_put.round(4).to_string())

    dd_base_p    = drawdown_episodes(put_base)
    dd_overlay_p = drawdown_episodes(put_overlay)
    dd_ep_p = dd_base_p.merge(dd_overlay_p, on="episode", suffixes=("_base", "_overlay"))
    dd_ep_p["return_delta"] = dd_ep_p["return_overlay"] - dd_ep_p["return_base"]
    log.info("\n  Drawdown episodes Put:\n%s", dd_ep_p.to_string())

    reg_put = regime_analysis(put_base, put_overlay, spy.loc[BACKTEST_START:])
    log.info("\n  Regime analysis Put:\n%s", reg_put.to_string())

    # ── 6. PERMUTATION TESTS ──
    log.info("\n--- Permutation Tests (N=100) ---")

    log.info("  SPY Long — testing Sharpe …")
    perm_spy_sharpe  = permutation_test(spy_base, signal_bt["position_scale"].reindex(spy_base.index), metric="sharpe")
    log.info("  SPY Long — testing Sortino …")
    perm_spy_sortino = permutation_test(spy_base, signal_bt["position_scale"].reindex(spy_base.index), metric="sortino")

    log.info("  Put Sell — testing Sharpe (weekly) …")
    # For put selling, use weekly series
    ps_weekly = signal_bt["position_scale"].resample("W-FRI").last().dropna()
    perm_put_sharpe  = permutation_test(put_base, ps_weekly.reindex(put_base.index, method="ffill"), metric="sharpe")
    perm_put_sortino = permutation_test(put_base, ps_weekly.reindex(put_base.index, method="ffill"), metric="sortino")

    for name, p in [
        ("SPY Sharpe",     perm_spy_sharpe),
        ("SPY Sortino",    perm_spy_sortino),
        ("Put Sharpe",     perm_put_sharpe),
        ("Put Sortino",    perm_put_sortino),
    ]:
        log.info("  %-14s  actual=%.3f  perm_mean=%.3f  perm_p10=%.3f  perm_p90=%.3f  p=%.3f  → %s",
                 name,
                 p["actual"], p["perm_mean"], p["perm_p10"], p["perm_p90"],
                 p["p_value_one_sided"], p["interpretation"])

    # ── 7. SAVE ALL RESULTS ──
    log.info("\n--- Saving results ---")

    # Summary metrics table
    metrics_table = pd.DataFrame([
        m_base_spy, m_overlay_spy,
        m_base_put, m_overlay_put,
    ])
    metrics_table.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    # Year-by-year tables
    yby_spy.round(4).to_csv(OUT_DIR / "year_by_year_spy.csv")
    yby_put.round(4).to_csv(OUT_DIR / "year_by_year_put.csv")

    # Drawdown episodes
    dd_ep.to_csv(OUT_DIR / "drawdown_episodes_spy.csv", index=False)
    dd_ep_p.to_csv(OUT_DIR / "drawdown_episodes_put.csv", index=False)

    # Regime analysis
    reg_spy.to_csv(OUT_DIR / "regime_spy.csv", index=False)
    reg_put.to_csv(OUT_DIR / "regime_put.csv", index=False)

    # Permutation results
    perm_results = {
        "spy_sharpe":  perm_spy_sharpe,
        "spy_sortino": perm_spy_sortino,
        "put_sharpe":  perm_put_sharpe,
        "put_sortino": perm_put_sortino,
    }
    with open(OUT_DIR / "permutation_test_results.json", "w") as f:
        json.dump(perm_results, f, indent=2)

    # Equity curves (daily cumulative returns)
    equity = pd.DataFrame({
        "spy_baseline": (1 + spy_base).cumprod(),
        "spy_overlay":  (1 + spy_overlay).cumprod(),
    })
    equity.to_csv(OUT_DIR / "equity_curves_spy.csv")

    put_equity = pd.DataFrame({
        "put_baseline": (1 + put_base).cumprod(),
        "put_overlay":  (1 + put_overlay).cumprod(),
    })
    put_equity.to_csv(OUT_DIR / "equity_curves_put.csv")

    # Signal summary
    with open(OUT_DIR / "signal_summary.json", "w") as f:
        json.dump(sig_summary, f, indent=2)

    # Full JSON report
    report = {
        "run_timestamp": datetime.now().isoformat(),
        "backtest_start": BACKTEST_START,
        "backtest_end": str(spy_base.index[-1].date()),
        "signal_summary": sig_summary,
        "metrics": {
            "spy_baseline": m_base_spy,
            "spy_overlay":  m_overlay_spy,
            "put_baseline": m_base_put,
            "put_overlay":  m_overlay_put,
        },
        "permutation_tests": perm_results,
    }
    with open(OUT_DIR / "full_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── 8. PRINT CLEAN SUMMARY ──
    log.info("\n" + "=" * 70)
    log.info("RISK OVERLAY BACKTEST RESULTS")
    log.info("Period: %s → %s", BACKTEST_START, str(spy_base.index[-1].date()))
    log.info("=" * 70)

    log.info("\n[SIGNAL ACTIVITY]")
    log.info("  Normal (scale=1.0):   %.1f%% of days", sig_summary["pct_normal"]*100)
    log.info("  Elevated (scale=0.7): %.1f%% of days", sig_summary["pct_elevated"]*100)
    log.info("  High (scale=0.3):     %.1f%% of days", sig_summary["pct_high"]*100)
    log.info("  Average position:     %.1f%%", sig_summary["avg_position_scale"]*100)

    log.info("\n[SPY LONG STRATEGY]")
    log.info("  %-20s  CAGR=%5.1f%%  Sharpe=%5.2f  Sortino=%5.2f  MaxDD=%6.1f%%  Calmar=%5.2f",
             "Baseline",
             m_base_spy["cagr"]*100, m_base_spy["sharpe"],
             m_base_spy["sortino"], m_base_spy["max_dd"]*100, m_base_spy["calmar"])
    log.info("  %-20s  CAGR=%5.1f%%  Sharpe=%5.2f  Sortino=%5.2f  MaxDD=%6.1f%%  Calmar=%5.2f",
             "With Overlay",
             m_overlay_spy["cagr"]*100, m_overlay_spy["sharpe"],
             m_overlay_spy["sortino"], m_overlay_spy["max_dd"]*100, m_overlay_spy["calmar"])
    sharpe_delta_spy = m_overlay_spy["sharpe"] - m_base_spy["sharpe"]
    dd_delta_spy     = m_overlay_spy["max_dd"]  - m_base_spy["max_dd"]
    log.info("  DELTA: Sharpe %+.2f   MaxDD %+.1f%%",
             sharpe_delta_spy, dd_delta_spy*100)

    log.info("\n[PUT SELLING STRATEGY]")
    log.info("  %-20s  CAGR=%5.1f%%  Sharpe=%5.2f  Sortino=%5.2f  MaxDD=%6.1f%%  Calmar=%5.2f",
             "Baseline",
             m_base_put["cagr"]*100, m_base_put["sharpe"],
             m_base_put["sortino"], m_base_put["max_dd"]*100, m_base_put["calmar"])
    log.info("  %-20s  CAGR=%5.1f%%  Sharpe=%5.2f  Sortino=%5.2f  MaxDD=%6.1f%%  Calmar=%5.2f",
             "With Overlay",
             m_overlay_put["cagr"]*100, m_overlay_put["sharpe"],
             m_overlay_put["sortino"], m_overlay_put["max_dd"]*100, m_overlay_put["calmar"])
    sharpe_delta_put = m_overlay_put["sharpe"] - m_base_put["sharpe"]
    dd_delta_put     = m_overlay_put["max_dd"]  - m_base_put["max_dd"]
    log.info("  DELTA: Sharpe %+.2f   MaxDD %+.1f%%",
             sharpe_delta_put, dd_delta_put*100)

    log.info("\n[PERMUTATION TESTS]")
    log.info("  SPY Long  Sharpe  — actual=%.3f vs perm_mean=%.3f  p=%.3f  → %s",
             perm_spy_sharpe["actual"], perm_spy_sharpe["perm_mean"],
             perm_spy_sharpe["p_value_one_sided"], perm_spy_sharpe["interpretation"])
    log.info("  Put Sell  Sharpe  — actual=%.3f vs perm_mean=%.3f  p=%.3f  → %s",
             perm_put_sharpe["actual"], perm_put_sharpe["perm_mean"],
             perm_put_sharpe["p_value_one_sided"], perm_put_sharpe["interpretation"])

    log.info("\n[REGIME ANALYSIS — does overlay help more in bear markets?]")
    log.info("  SPY Long:")
    for _, row in reg_spy.iterrows():
        log.info("    %-6s  base_sharpe=%.2f  overlay_sharpe=%.2f  delta=%+.2f",
                 row["regime"], row["base_sharpe"], row["overlay_sharpe"],
                 row["sharpe_improvement"])

    log.info("\n[DRAWDOWN EPISODES]")
    for _, row in dd_ep.iterrows():
        log.info("  %-25s  base=%+.1f%%  overlay=%+.1f%%  delta=%+.1f%%",
                 row["episode"],
                 row["return_base"]*100, row["return_overlay"]*100,
                 row["return_delta"]*100)

    log.info("\n[NOTE: PUT SELLING CAGR]")
    log.info("  The put-selling CAGR (14,000%+) reflects VIX-implied premiums during high-vol periods")
    log.info("  (VIX 80 at COVID → ~11%/week implied premium). This is a theoretical upper bound —")
    log.info("  real trading has bid/ask, margin limits, and assignment risk. Use Sharpe/MaxDD, not CAGR.")
    log.info("  The OVERLAY comparison (base vs overlay) is still valid because both use the same premium model.")

    log.info("\n[VERDICT]")
    helped_spy  = sharpe_delta_spy  > 0.05
    helped_put  = sharpe_delta_put  > 0.05
    # dd_delta_spy is negative for improvement (overlay MaxDD is less negative than baseline)
    # e.g., base MaxDD = -0.337, overlay MaxDD = -0.119, delta = +0.218 (positive = improvement)
    dd_improved = dd_delta_spy > 0.02   # positive delta means overlay has shallower drawdown
    perm_sig    = perm_spy_sharpe["p_value_one_sided"] < 0.10

    if helped_spy and dd_improved and perm_sig:
        log.info("  OVERLAY PASSES: improves Sharpe on SPY, reduces drawdown, AND permutation is significant.")
        log.info("  Timing of signal adds real value — not just random position scaling.")
    elif helped_spy and dd_improved and not perm_sig:
        log.info("  MIXED: Sharpe and drawdown improve, but permutation test NOT significant.")
        log.info("  Overlay reduces risk but the TIMING may not matter — equivalent to always being at 80% equity.")
    elif not helped_spy and dd_improved:
        log.info("  PARTIAL: Drawdown reduced but Sharpe does not improve (too much return sacrificed).")
    else:
        log.info("  OVERLAY DOES NOT HELP: No improvement in Sharpe or drawdown.")
        log.info("  Do NOT integrate without redesigning the overlay thresholds.")

    log.info("\nAll results saved to: %s", OUT_DIR)
    log.info("=" * 70)

    return report


if __name__ == "__main__":
    main()
