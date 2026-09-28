"""
Portfolio Optimization Study: GP3 + VMR Daily + Consensus
Combines 3 validated growth strategies in a Robinhood ETF account.
2010-01-01 to 2026-07-17
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
from datetime import datetime
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

START = "2010-01-01"
END   = "2026-07-17"

TICKERS = ["SPY", "UPRO", "GLD", "TLT", "^VIX"]

print("Downloading market data...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)

prices = raw["Close"].copy()
prices.columns = [c.replace("^","") for c in prices.columns]
prices.dropna(subset=["SPY", "UPRO", "GLD", "TLT"], inplace=True)

vix = prices["VIX"].copy()
spy = prices["SPY"].copy()
upro = prices["UPRO"].copy()
gld = prices["GLD"].copy()
tlt = prices["TLT"].copy()

print(f"Data loaded: {prices.index[0].date()} → {prices.index[-1].date()} ({len(prices)} trading days)")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def wilder_rsi(series, period=10):
    """RSI using Wilder's smoothing (EWM with alpha = 1/period)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs   = avg_gain / avg_loss.replace(0, np.nan)
    rsi  = 100 - 100 / (1 + rs)
    return rsi

def annualized_vol(series, window=21):
    """Rolling realized vol: std of log returns × sqrt(252)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)

def compute_metrics(ret_series, rf=0.0, ann=252):
    """Compute canonical metrics from a daily return series."""
    ret = ret_series.dropna()
    if len(ret) < 30:
        return {}
    excess = ret - rf / ann
    sharpe = excess.mean() / excess.std() * np.sqrt(ann) if excess.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(ann) if len(ret[ret < 0]) > 0 else 1e-9
    sortino = (ret.mean() * ann) / downside if downside > 0 else np.nan
    cum  = (1 + ret).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    maxdd = dd.min()
    n_years = len(ret) / ann
    cagr = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    calmar = cagr / abs(maxdd) if maxdd != 0 else np.nan
    wr = (ret > 0).mean()
    # Day concentration: % of cumulative PnL from top-1 day
    cum_pnl = ret.cumsum()
    top1_pnl = ret.max()
    total_pnl = ret[ret > 0].sum()
    day_conc = top1_pnl / total_pnl if total_pnl > 0 else np.nan
    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "cagr": round(float(cagr), 4),
        "maxdd": round(float(maxdd), 4),
        "calmar": round(float(calmar), 3),
        "win_rate": round(float(wr), 4),
        "n_days": len(ret),
        "day_conc": round(float(day_conc), 4),
    }

def regime_split(ret_series, spy_close):
    """Compute Sharpe by green/red/flat ES (SPY) day."""
    spy_ret = spy_close.pct_change()
    common  = ret_series.index.intersection(spy_ret.index)
    r = ret_series.loc[common]
    s = spy_ret.loc[common]
    green = r[s > 0.0005]
    red   = r[s < -0.0005]
    flat  = r[(s >= -0.0005) & (s <= 0.0005)]
    def sh(x):
        if len(x) < 5: return np.nan
        return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    sg, sr, sf = sh(green), sh(red), sh(flat)
    skew = abs(sg - sr) / max(abs(sg), abs(sr)) if not (np.isnan(sg) or np.isnan(sr)) else np.nan
    return {
        "green_sharpe": round(float(sg), 3) if not np.isnan(sg) else None,
        "red_sharpe":   round(float(sr), 3) if not np.isnan(sr) else None,
        "flat_sharpe":  round(float(sf), 3) if not np.isnan(sf) else None,
        "skew": round(float(skew), 3) if not np.isnan(skew) else None,
        "regime_gate_pass": bool(skew <= 0.50) if not np.isnan(skew) else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: GAMEPLAN v3 (GP3)
# ─────────────────────────────────────────────────────────────────────────────

print("Building GP3 strategy...")

rsi10 = wilder_rsi(spy, period=10)
spy_log = np.log(spy / spy.shift(1))
vol21   = spy_log.rolling(21).std() * np.sqrt(252)

# 3-timeframe confluence
# Short: 5d momentum > 0 AND RSI(10) > 50
mom5    = spy / spy.shift(5) - 1
short_c = ((mom5 > 0) & (rsi10 > 50)).astype(float)

# Medium: SPY > 50d SMA AND 21d vol < threshold (using 20% as implied threshold)
sma50   = spy.rolling(50).mean()
vol_thr = 0.20
med_c   = ((spy > sma50) & (vol21 < vol_thr)).astype(float)

# Long: SPY > 200d SMA AND 63d vol trend declining (63d vol < 126d vol)
sma200  = spy.rolling(200).mean()
vol63   = spy_log.rolling(63).std() * np.sqrt(252)
vol126  = spy_log.rolling(126).std() * np.sqrt(252)
long_c  = ((spy > sma200) & (vol63 < vol126)).astype(float)

confluence = short_c + med_c + long_c

# Signal: confluence >= 2.5 AND 21d vol < 15%
# Since confluence is integer (0-3), >=2.5 means ==3
gp3_upro = (confluence >= 2.5) & (vol21 < 0.15)

# Daily returns: if signal on day t, hold UPRO on day t+1 (next-day execution)
upro_ret = upro.pct_change()
spy_ret  = spy.pct_change()

gp3_ret = pd.Series(index=spy_ret.index, dtype=float)
for i in range(1, len(gp3_upro)):
    sig_date = gp3_upro.index[i-1]
    ret_date = gp3_upro.index[i]
    if gp3_upro.iloc[i-1]:
        gp3_ret.iloc[i] = upro_ret.iloc[i]
    else:
        gp3_ret.iloc[i] = spy_ret.iloc[i]

gp3_ret.dropna(inplace=True)

gp3_metrics = compute_metrics(gp3_ret)
gp3_regime  = regime_split(gp3_ret, spy)
upro_pct_gp3 = gp3_upro.mean()

print(f"  GP3 Sharpe={gp3_metrics['sharpe']:.2f}, UPRO days={upro_pct_gp3:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: VMR DAILY
# ─────────────────────────────────────────────────────────────────────────────

print("Building VMR Daily strategy...")

vix10ma   = vix.rolling(10).mean()
vix20peak = vix.rolling(20).max()
vix20ma   = vix.rolling(20).mean()

def vmr_signal(idx):
    v       = vix.loc[idx]
    v10ma   = vix10ma.loc[idx]
    v20pk   = vix20peak.loc[idx]
    if pd.isna(v) or pd.isna(v10ma) or pd.isna(v20pk):
        return "SPY"
    if v < 15 and v < v10ma:
        return "UPRO"
    # Mean reversion: VIX > 20 AND < 85% of 20d peak AND declining (below 10d MA)
    if v > 20 and v < 0.85 * v20pk and v < v10ma:
        return "UPRO"
    # Defensive: VIX > 25 AND above 10d MA
    if v > 25 and v > v10ma:
        return "GLD_TLT"
    # Cautious: VIX > 20 AND above 10d MA
    if v > 20 and v > v10ma:
        return "SPY_TLT"
    return "SPY"

vmr_signals = pd.Series([vmr_signal(d) for d in vix.index], index=vix.index)

gld_ret = gld.pct_change()
tlt_ret = tlt.pct_change()

vmr_ret = pd.Series(index=spy_ret.index, dtype=float)
for i in range(1, len(vmr_signals)):
    sig = vmr_signals.iloc[i-1]
    rd  = vmr_signals.index[i]
    if rd not in spy_ret.index:
        continue
    if sig == "UPRO":
        vmr_ret.loc[rd] = upro_ret.loc[rd] if rd in upro_ret.index else spy_ret.loc[rd]
    elif sig == "GLD_TLT":
        g = gld_ret.loc[rd] if rd in gld_ret.index else 0.0
        t = tlt_ret.loc[rd] if rd in tlt_ret.index else 0.0
        vmr_ret.loc[rd] = 0.5 * g + 0.5 * t
    elif sig == "SPY_TLT":
        s = spy_ret.loc[rd]
        t = tlt_ret.loc[rd] if rd in tlt_ret.index else 0.0
        vmr_ret.loc[rd] = 0.5 * s + 0.5 * t
    else:
        vmr_ret.loc[rd] = spy_ret.loc[rd]

vmr_ret.dropna(inplace=True)

vmr_metrics = compute_metrics(vmr_ret)
vmr_regime  = regime_split(vmr_ret, spy)
upro_pct_vmr = (vmr_signals == "UPRO").mean()

print(f"  VMR Sharpe={vmr_metrics['sharpe']:.2f}, UPRO days={upro_pct_vmr:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: GP3+VMR CONSENSUS
# ─────────────────────────────────────────────────────────────────────────────

print("Building Consensus strategy...")

# UPRO only when BOTH say UPRO, else SPY
# GP3 UPRO signal is boolean series; VMR UPRO signal is from vmr_signals
vmr_upro_sig = (vmr_signals == "UPRO")

# Align on common index
common_idx = gp3_upro.index.intersection(vmr_upro_sig.index)
consensus_upro = gp3_upro.loc[common_idx] & vmr_upro_sig.loc[common_idx]

cons_ret = pd.Series(index=spy_ret.index, dtype=float)
for i in range(1, len(consensus_upro)):
    sig_date = consensus_upro.index[i-1]
    ret_date = consensus_upro.index[i]
    if ret_date not in spy_ret.index:
        continue
    if consensus_upro.iloc[i-1]:
        cons_ret.loc[ret_date] = upro_ret.loc[ret_date] if ret_date in upro_ret.index else spy_ret.loc[ret_date]
    else:
        cons_ret.loc[ret_date] = spy_ret.loc[ret_date]

cons_ret.dropna(inplace=True)

cons_metrics = compute_metrics(cons_ret)
cons_regime  = regime_split(cons_ret, spy)
upro_pct_cons = consensus_upro.mean()

print(f"  Consensus Sharpe={cons_metrics['sharpe']:.2f}, UPRO days={upro_pct_cons:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# ALIGN STRATEGY RETURNS TO COMMON DATE RANGE
# ─────────────────────────────────────────────────────────────────────────────

common = gp3_ret.index.intersection(vmr_ret.index).intersection(cons_ret.index)
gp3_r  = gp3_ret.loc[common]
vmr_r  = vmr_ret.loc[common]
cons_r = cons_ret.loc[common]

print(f"Common date range: {common[0].date()} → {common[-1].date()} ({len(common)} days)")

# Individual metrics on common range
gp3_metrics_c  = compute_metrics(gp3_r)
vmr_metrics_c  = compute_metrics(vmr_r)
cons_metrics_c = compute_metrics(cons_r)

# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION GRID
# ─────────────────────────────────────────────────────────────────────────────

print("Running allocation grid...")

allocations = {
    "GP3_only":       (1.00, 0.00, 0.00),
    "VMR_only":       (0.00, 1.00, 0.00),
    "Consensus_only": (0.00, 0.00, 1.00),
    "GP3_VMR_5050":   (0.50, 0.50, 0.00),
    "GP3_VMR_6040":   (0.60, 0.40, 0.00),
    "GP3_VMR_4060":   (0.40, 0.60, 0.00),
    "Equal_3way":     (1/3,  1/3,  1/3),
    "GP3_Cons_5050":  (0.50, 0.00, 0.50),
    "VMR_Cons_5050":  (0.00, 0.50, 0.50),
    "GP3_heavy":      (0.50, 0.30, 0.20),
    "VMR_heavy":      (0.20, 0.60, 0.20),
}

grid_results = {}
for name, (w1, w2, w3) in allocations.items():
    port_r = w1 * gp3_r + w2 * vmr_r + w3 * cons_r
    m = compute_metrics(port_r)
    m["weights"] = {"GP3": round(w1,3), "VMR": round(w2,3), "Consensus": round(w3,3)}
    m["regime"]  = regime_split(port_r, spy.loc[common])
    grid_results[name] = m

# ─────────────────────────────────────────────────────────────────────────────
# MEAN-VARIANCE OPTIMAL (MAXIMIZE SHARPE)
# ─────────────────────────────────────────────────────────────────────────────

print("Running mean-variance optimization...")

ret_matrix = pd.DataFrame({"GP3": gp3_r, "VMR": vmr_r, "Consensus": cons_r})

def neg_sharpe(w):
    w = np.array(w)
    port = ret_matrix.values @ w
    if port.std() == 0:
        return 0
    return -(port.mean() / port.std() * np.sqrt(252))

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds = [(0, 1), (0, 1), (0, 1)]
best_sh = np.inf
mv_result = None
for init in [(0.33, 0.33, 0.34), (1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0.2,0.6,0.2)]:
    res = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if res.fun < best_sh:
        best_sh = res.fun
        mv_result = res

w_mv = mv_result.x
mv_port = ret_matrix.values @ w_mv
mv_r = pd.Series(mv_port, index=ret_matrix.index)
mv_m = compute_metrics(mv_r)
mv_m["weights"] = {"GP3": round(w_mv[0],3), "VMR": round(w_mv[1],3), "Consensus": round(w_mv[2],3)}
mv_m["regime"]  = regime_split(mv_r, spy.loc[common])
grid_results["MV_MaxSharpe"] = mv_m
print(f"  MV Optimal weights: GP3={w_mv[0]:.2%}, VMR={w_mv[1]:.2%}, Cons={w_mv[2]:.2%}, Sharpe={mv_m['sharpe']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# RISK PARITY
# ─────────────────────────────────────────────────────────────────────────────

print("Running risk parity...")

def risk_parity_weights(cov_matrix):
    """Equal risk contribution via iterative algorithm."""
    n = cov_matrix.shape[0]
    w = np.ones(n) / n
    for _ in range(1000):
        sigma = np.sqrt(w @ cov_matrix @ w)
        mrc   = cov_matrix @ w / sigma
        rc    = w * mrc
        w_new = w * (1.0 / mrc)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-10:
            break
        w = w_new
    return w

cov = ret_matrix.cov().values * 252
rp_w = risk_parity_weights(cov)
rp_port = ret_matrix.values @ rp_w
rp_r = pd.Series(rp_port, index=ret_matrix.index)
rp_m = compute_metrics(rp_r)
rp_m["weights"] = {"GP3": round(rp_w[0],3), "VMR": round(rp_w[1],3), "Consensus": round(rp_w[2],3)}
rp_m["regime"]  = regime_split(rp_r, spy.loc[common])
grid_results["Risk_Parity"] = rp_m
print(f"  Risk Parity weights: GP3={rp_w[0]:.2%}, VMR={rp_w[1]:.2%}, Cons={rp_w[2]:.2%}, Sharpe={rp_m['sharpe']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION TEST ON BEST ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

print("Running permutation test on best allocation...")

best_name = max(grid_results, key=lambda k: grid_results[k].get("sharpe", -999))
best_w    = grid_results[best_name]["weights"]
w_arr     = np.array([best_w["GP3"], best_w["VMR"], best_w["Consensus"]])
best_port = ret_matrix.values @ w_arr
best_sharpe = grid_results[best_name]["sharpe"]

N_PERM = 1000
perm_sharpes = []
rng = np.random.default_rng(42)
for _ in range(N_PERM):
    shuffled = rng.permutation(best_port)
    s = pd.Series(shuffled)
    perm_sharpes.append(s.mean() / s.std() * np.sqrt(252))

perm_sharpes = np.array(perm_sharpes)
p_value = (perm_sharpes >= best_sharpe).mean()
print(f"  Best: {best_name}, Sharpe={best_sharpe:.2f}, p-value={p_value:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# ANNUAL BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

print("Computing annual breakdown...")

annual = {}
for name, (w1, w2, w3) in {**allocations, "MV_MaxSharpe": tuple(w_mv), "Risk_Parity": tuple(rp_w)}.items():
    port_r = w1 * gp3_r + w2 * vmr_r + w3 * cons_r
    port_r = pd.Series(port_r.values, index=ret_matrix.index)
    by_year = {}
    for year in range(port_r.index.year.min(), port_r.index.year.max() + 1):
        yr = port_r[port_r.index.year == year]
        if len(yr) < 20:
            continue
        m = compute_metrics(yr)
        by_year[str(year)] = {"sharpe": m["sharpe"], "cagr": m["cagr"], "maxdd": m["maxdd"]}
    annual[name] = by_year

# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

corr = ret_matrix.corr().round(4).to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL STRATEGY METRICS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

individual = {
    "GP3": {**gp3_metrics_c, "regime": gp3_regime, "upro_pct": round(float(upro_pct_gp3), 4)},
    "VMR": {**vmr_metrics_c, "regime": vmr_regime, "upro_pct": round(float(upro_pct_vmr), 4)},
    "Consensus": {**cons_metrics_c, "regime": cons_regime, "upro_pct": round(float(upro_pct_cons), 4)},
}

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

output = {
    "meta": {
        "generated_at": datetime.now().isoformat(),
        "start_date": str(common[0].date()),
        "end_date": str(common[-1].date()),
        "n_days": len(common),
    },
    "individual_strategies": individual,
    "correlation_matrix": corr,
    "allocations": grid_results,
    "best_allocation": {
        "name": best_name,
        "sharpe": best_sharpe,
        "weights": best_w,
        "permutation_test": {
            "n_permutations": N_PERM,
            "p_value": round(float(p_value), 4),
            "perm_sharpe_mean": round(float(perm_sharpes.mean()), 4),
            "perm_sharpe_p95": round(float(np.percentile(perm_sharpes, 95)), 4),
        },
    },
    "annual_breakdown": annual,
}

OUT_PATH = "/home/jupiter/Lvl3Quant/output/growth_research/portfolio_optimization_results.json"
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PORTFOLIO OPTIMIZATION RESULTS SUMMARY")
print("="*80)
print(f"{'Allocation':<22} {'GP3':>6} {'VMR':>6} {'Cons':>6} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'DayConc':>8}")
print("-"*90)

all_names = list(allocations.keys()) + ["MV_MaxSharpe", "Risk_Parity"]
for name in all_names:
    m = grid_results[name]
    w = m["weights"]
    print(f"{name:<22} {w['GP3']:>6.2f} {w['VMR']:>6.2f} {w['Consensus']:>6.2f} "
          f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['cagr']:>8.1%} "
          f"{m['maxdd']:>8.1%} {m['calmar']:>8.2f} {m['day_conc']:>8.1%}")

print("\n--- Regime Split for Best Allocation ---")
bm = grid_results[best_name]
r  = bm["regime"]
print(f"Best: {best_name}")
print(f"  Green Sharpe: {r['green_sharpe']}")
print(f"  Red Sharpe:   {r['red_sharpe']}")
print(f"  Flat Sharpe:  {r['flat_sharpe']}")
print(f"  Skew:         {r['skew']} {'PASS' if r['regime_gate_pass'] else 'FAIL'}")

print(f"\n--- Permutation Test ({N_PERM} shuffles) ---")
print(f"  Observed Sharpe: {best_sharpe:.2f}")
print(f"  Permuted mean:   {perm_sharpes.mean():.2f}")
print(f"  p-value:         {p_value:.4f} ({'PASS' if p_value < 0.05 else 'FAIL'})")

print("\n--- Correlation Matrix ---")
for k in corr:
    for k2 in corr[k]:
        if k < k2:
            print(f"  {k} vs {k2}: {corr[k][k2]:.3f}")

print("\nDone.")
