"""
Rotation Strategy Monthly Regime Analysis
HC #428 R1 test at the strategy's natural operating frequency (21-day / 63-day).

Tests:
1. 21-day rolling regime Sharpe split (Green/Red/Flat by SPY 21-day return)
2. 63-day robustness check
3. Permutation test (2000 shuffles, HC #659)
4. Selection alpha: strategy vs SPY buy-and-hold in red months
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

np.random.seed(42)
OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/rotation_monthly_regime"

# ─── Load data ────────────────────────────────────────────────────────────────

returns = pd.read_parquet("/home/jupiter/Lvl3Quant/output/multi_strategy_portfolio_v2/returns_all_strategies.parquet")
returns.index = pd.to_datetime(returns.index)

# Also load rotation universe returns (daily series for each universe)
univ_files = {
    "US_Sectors_K3":      "output/rotation_universe_exploration/returns_U.S._Sectors_K3.csv",
    "Wide_Sector_K4":     "output/rotation_universe_exploration/returns_Wide_Sector+Intl_K4.csv",
    "Asset_Class_K4":     "output/rotation_universe_exploration/returns_Asset_Class_Factor_K4.csv",
    "Thematic_K4":        "output/rotation_universe_exploration/returns_Thematic_Industry_K4.csv",
    "Country_Region_K4":  "output/rotation_universe_exploration/returns_Country_Region_K4.csv",
}
base = "/home/jupiter/Lvl3Quant"
univ_returns = {}
for name, path in univ_files.items():
    df = pd.read_csv(f"{base}/{path}", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    univ_returns[name] = df["return"]

# Load SPY
spy_raw = pd.read_parquet("/home/jupiter/Lvl3Quant/results/tsmom_etf_v1/data/SPY_daily.parquet")
spy_raw["date"] = pd.to_datetime(spy_raw["date"])
spy_raw = spy_raw.sort_values("date").set_index("date")
spy_daily_ret = spy_raw["adj_close"].pct_change().rename("SPY")

print(f"Returns data: {returns.index.min().date()} to {returns.index.max().date()}, {len(returns)} rows")
print(f"Strategies: {returns.columns.tolist()}")
print(f"SPY data: {spy_daily_ret.index.min().date()} to {spy_daily_ret.index.max().date()}")

# ─── Strategy map ─────────────────────────────────────────────────────────────

# Primary strategies to test (map multi-strat columns + universe series)
strat_daily = {
    "ETF_v3":       returns["ETF_v3"],
    "ETF_v4_Wide":  returns["ETF_v4_Wide"],
    "Multi_Asset":  returns["Multi_Asset"],
}
# Add universe-level series
for k, v in univ_returns.items():
    strat_daily[k] = v

# ─── Core functions ───────────────────────────────────────────────────────────

def sharpe_ann(rets, periods_per_year=252):
    """Annualized Sharpe from daily returns."""
    rets = rets.dropna()
    if len(rets) < 5 or rets.std() == 0:
        return np.nan
    return (rets.mean() / rets.std()) * np.sqrt(periods_per_year)

def sortino_ann(rets, periods_per_year=252):
    rets = rets.dropna()
    downside = rets[rets < 0]
    if len(downside) < 3 or downside.std() == 0:
        return np.nan
    return (rets.mean() / downside.std()) * np.sqrt(periods_per_year)

def profit_factor(rets):
    rets = rets.dropna()
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    return wins / losses if losses > 0 else np.inf

def max_drawdown(rets):
    rets = rets.dropna()
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return dd.min()

def cagr(rets, periods_per_year=252):
    rets = rets.dropna()
    n = len(rets)
    total = (1 + rets).prod()
    return total ** (periods_per_year / n) - 1

def calmar(rets, periods_per_year=252):
    c = cagr(rets, periods_per_year)
    mdd = max_drawdown(rets)
    return c / abs(mdd) if mdd != 0 else np.inf

def win_rate(rets):
    rets = rets.dropna()
    return (rets > 0).mean()

def day_concentration(rets):
    """Fraction of cumulative PnL from single best day (HC #344 cap: <= 0.70)."""
    rets = rets.dropna()
    cum_pnl = rets.sum()
    if cum_pnl <= 0:
        return np.nan
    return rets.max() / cum_pnl

def full_metrics(rets, label="", periods_per_year=252):
    rets = rets.dropna()
    return {
        "label": label,
        "n_days": len(rets),
        "Sharpe": sharpe_ann(rets, periods_per_year),
        "Sortino": sortino_ann(rets, periods_per_year),
        "PF": profit_factor(rets),
        "WR": win_rate(rets),
        "CAGR": cagr(rets, periods_per_year),
        "MaxDD": max_drawdown(rets),
        "Calmar": calmar(rets, periods_per_year),
        "DayConc": day_concentration(rets),
    }

# ─── Build non-overlapping period blocks ──────────────────────────────────────

def build_period_returns(daily_rets, spy_daily, hold=21, spy_thresh=0.01):
    """
    Build series of non-overlapping hold-period returns.
    Each period: strategy cumulative return and SPY cumulative return.
    Periods are anchored at the rebalance dates (every hold days).
    
    Returns DataFrame with columns: strat_ret, spy_ret, regime
    """
    daily_rets = daily_rets.dropna()
    spy_daily = spy_daily.dropna()
    
    # Align to common dates
    common = daily_rets.index.intersection(spy_daily.index)
    s = daily_rets.loc[common].sort_index()
    spy = spy_daily.loc[common].sort_index()
    
    dates = s.index
    periods = []
    i = 0
    while i + hold <= len(dates):
        period_s = s.iloc[i:i+hold]
        period_spy = spy.iloc[i:i+hold]
        s_ret = (1 + period_s).prod() - 1
        spy_ret = (1 + period_spy).prod() - 1
        
        if spy_ret > spy_thresh:
            regime = "green"
        elif spy_ret < -spy_thresh:
            regime = "red"
        else:
            regime = "flat"
        
        periods.append({
            "start": dates[i],
            "end": dates[i+hold-1],
            "strat_ret": s_ret,
            "spy_ret": spy_ret,
            "regime": regime,
        })
        i += hold
    
    return pd.DataFrame(periods).set_index("start")

def regime_sharpe(period_df, periods_per_year_equiv):
    """
    Compute Sharpe for each regime bucket.
    periods_per_year_equiv: how many hold periods per year (252/hold).
    """
    results = {}
    for regime in ["green", "red", "flat"]:
        sub = period_df[period_df["regime"] == regime]["strat_ret"]
        if len(sub) < 3:
            results[regime] = {"n": len(sub), "Sharpe": np.nan, "mean_ret": np.nan}
        else:
            sh = (sub.mean() / sub.std()) * np.sqrt(periods_per_year_equiv)
            results[regime] = {
                "n": len(sub),
                "Sharpe": sh,
                "mean_ret": sub.mean(),
                "WR": (sub > 0).mean(),
                "PF": profit_factor(sub),
                "beat_spy": (period_df[period_df["regime"] == regime]["strat_ret"] >
                             period_df[period_df["regime"] == regime]["spy_ret"]).mean(),
            }
    
    # Compute regime gap
    g = results["green"]["Sharpe"]
    r = results["red"]["Sharpe"]
    if np.isnan(g) or np.isnan(r) or max(abs(g), abs(r)) == 0:
        gap = np.nan
        verdict = "NEEDS-DATA"
    else:
        gap = abs(g - r) / max(abs(g), abs(r))
        verdict = "PASS" if gap <= 0.50 else "FAIL"
    
    results["regime_gap"] = gap
    results["R1_verdict"] = verdict
    return results

def permutation_test(period_df, n_shuffles=2000, periods_per_year_equiv=12.0):
    """
    HC #659: shuffle period returns, recompute regime Sharpe gap.
    Returns p-value: fraction of shuffles with gap <= observed gap.
    """
    obs_g = period_df[period_df["regime"] == "green"]["strat_ret"]
    obs_r = period_df[period_df["regime"] == "red"]["strat_ret"]
    
    if len(obs_g) < 3 or len(obs_r) < 3:
        return {"p_value": np.nan, "note": "insufficient data for permutation"}
    
    obs_sh_g = (obs_g.mean() / obs_g.std()) * np.sqrt(periods_per_year_equiv) if obs_g.std() > 0 else np.nan
    obs_sh_r = (obs_r.mean() / obs_r.std()) * np.sqrt(periods_per_year_equiv) if obs_r.std() > 0 else np.nan
    
    if np.isnan(obs_sh_g) or np.isnan(obs_sh_r):
        return {"p_value": np.nan, "note": "nan Sharpe"}
    
    obs_overall_sharpe = (period_df["strat_ret"].mean() / period_df["strat_ret"].std()) * np.sqrt(periods_per_year_equiv)
    
    shuffled_sharpes = []
    rets_arr = period_df["strat_ret"].values.copy()
    
    for _ in range(n_shuffles):
        np.random.shuffle(rets_arr)
        sh = (rets_arr.mean() / rets_arr.std()) * np.sqrt(periods_per_year_equiv) if rets_arr.std() > 0 else 0
        shuffled_sharpes.append(sh)
    
    shuffled_sharpes = np.array(shuffled_sharpes)
    # p-value: fraction of shuffled runs with Sharpe >= observed (one-sided, H0: random)
    p_val = (shuffled_sharpes >= obs_overall_sharpe).mean()
    
    return {
        "observed_sharpe": obs_overall_sharpe,
        "p_value": p_val,
        "n_shuffles": n_shuffles,
        "perm_sharpe_mean": shuffled_sharpes.mean(),
        "perm_sharpe_p95": np.percentile(shuffled_sharpes, 95),
        "significant": p_val < 0.05,
    }

def selection_alpha_vs_spy(period_df, label=""):
    """
    Compare strategy vs SPY buy-and-hold in each regime.
    Proves selection alpha: strategy > SPY in red months.
    """
    results = {}
    for regime in ["green", "red", "flat", "all"]:
        if regime == "all":
            sub = period_df
        else:
            sub = period_df[period_df["regime"] == regime]
        
        if len(sub) < 2:
            results[regime] = {"n": len(sub)}
            continue
        
        strat_mean = sub["strat_ret"].mean()
        spy_mean = sub["spy_ret"].mean()
        beat_rate = (sub["strat_ret"] > sub["spy_ret"]).mean()
        
        results[regime] = {
            "n": len(sub),
            "strat_mean_period_ret": strat_mean,
            "spy_mean_period_ret": spy_mean,
            "excess_return": strat_mean - spy_mean,
            "beat_rate": beat_rate,
            "strat_pos_rate": (sub["strat_ret"] > 0).mean(),
            "spy_pos_rate": (sub["spy_ret"] > 0).mean(),
        }
    
    return results

# ─── Run analysis ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("ROTATION MONTHLY REGIME ANALYSIS")
print("="*70)

all_results = {}

for strat_name, daily_rets in strat_daily.items():
    print(f"\n{'─'*50}")
    print(f"Strategy: {strat_name}")
    
    strat_results = {}
    
    for hold, label in [(21, "21d"), (63, "63d")]:
        ppa = 252.0 / hold  # periods per year
        period_df = build_period_returns(daily_rets, spy_daily_ret, hold=hold, spy_thresh=0.01)
        
        if len(period_df) < 10:
            print(f"  {label}: insufficient periods ({len(period_df)}), skip")
            continue
        
        # Overall metrics on period-level returns
        overall_sh = (period_df["strat_ret"].mean() / period_df["strat_ret"].std()) * np.sqrt(ppa) if period_df["strat_ret"].std() > 0 else np.nan
        overall_wr = (period_df["strat_ret"] > 0).mean()
        overall_pf = profit_factor(period_df["strat_ret"])
        
        # Regime split
        reg = regime_sharpe(period_df, ppa)
        
        # Permutation test
        perm = permutation_test(period_df, n_shuffles=2000, periods_per_year_equiv=ppa)
        
        # Selection alpha vs SPY
        sel_alpha = selection_alpha_vs_spy(period_df, strat_name)
        
        print(f"\n  [{label}] n_periods={len(period_df)}, Overall Sharpe={overall_sh:.3f}, WR={overall_wr:.1%}, PF={overall_pf:.2f}")
        
        green = reg.get("green", {})
        red = reg.get("red", {})
        flat = reg.get("flat", {})
        
        print(f"  Regime split:")
        print(f"    Green (n={green.get('n',0)}): Sharpe={green.get('Sharpe', np.nan):.3f}, WR={green.get('WR', np.nan):.1%}, beat_SPY={green.get('beat_spy', np.nan):.1%}")
        print(f"    Red   (n={red.get('n',0)}): Sharpe={red.get('Sharpe', np.nan):.3f}, WR={red.get('WR', np.nan):.1%}, beat_SPY={red.get('beat_spy', np.nan):.1%}")
        print(f"    Flat  (n={flat.get('n',0)}): Sharpe={flat.get('Sharpe', np.nan):.3f}, WR={flat.get('WR', np.nan):.1%}, beat_SPY={flat.get('beat_spy', np.nan):.1%}")
        print(f"  Regime gap: {reg['regime_gap']:.3f} -> {reg['R1_verdict']}")
        
        print(f"  Permutation (HC #659): Sharpe={perm.get('observed_sharpe', np.nan):.3f}, p={perm.get('p_value', np.nan):.4f}, sig={perm.get('significant', False)}")
        
        red_alpha = sel_alpha.get("red", {})
        print(f"  Red month alpha: strat={red_alpha.get('strat_mean_period_ret', np.nan):.2%}, SPY={red_alpha.get('spy_mean_period_ret', np.nan):.2%}, excess={red_alpha.get('excess_return', np.nan):.2%}")
        
        strat_results[label] = {
            "n_periods": len(period_df),
            "overall_sharpe": overall_sh,
            "overall_wr": overall_wr,
            "overall_pf": overall_pf,
            "regime": reg,
            "permutation": perm,
            "selection_alpha": sel_alpha,
        }
    
    # Daily-bar full metrics
    d_rets = daily_rets.dropna()
    dm = full_metrics(d_rets, label=strat_name)
    strat_results["daily_metrics"] = dm
    
    all_results[strat_name] = strat_results

# ─── Summary table ─────────────────────────────────────────────────────────────

print("\n\n" + "="*70)
print("SUMMARY TABLE — 21-day periods")
print("="*70)
print(f"{'Strategy':<22} {'N':<5} {'Sharpe':<8} {'G_Sh':<8} {'R_Sh':<8} {'Gap':<8} {'R1':<6} {'p-val':<8} {'RedAlpha'}")
print("─" * 90)

for strat_name, res in all_results.items():
    r21 = res.get("21d", {})
    if not r21:
        continue
    
    reg = r21.get("regime", {})
    perm = r21.get("permutation", {})
    sel = r21.get("selection_alpha", {})
    
    g_sh = reg.get("green", {}).get("Sharpe", np.nan)
    r_sh = reg.get("red", {}).get("Sharpe", np.nan)
    gap = reg.get("regime_gap", np.nan)
    verdict = reg.get("R1_verdict", "?")
    p_val = perm.get("p_value", np.nan)
    red_alpha = sel.get("red", {}).get("excess_return", np.nan)
    
    sharpe = r21.get("overall_sharpe", np.nan)
    n = r21.get("n_periods", 0)
    
    gap_str = f"{gap:.3f}" if not np.isnan(gap) else "nan"
    p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "nan"
    alpha_str = f"{red_alpha:+.2%}" if not np.isnan(red_alpha) else "nan"
    
    print(f"{strat_name:<22} {n:<5} {sharpe:<8.3f} {g_sh:<8.3f} {r_sh:<8.3f} {gap_str:<8} {verdict:<6} {p_str:<8} {alpha_str}")

# ─── Save artifacts ────────────────────────────────────────────────────────────

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, 6)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

out_path = os.path.join(OUTPUT_DIR, "monthly_regime_results.json")
with open(out_path, "w") as f:
    json.dump(to_serializable(all_results), f, indent=2, default=str)

print(f"\nArtifacts saved to: {out_path}")
print("\nAnalysis complete.")
