"""
Vol Regime Switching Strategy v1
=================================
Concept: Detect current vol regime (Low / Normal / High), switch between
three sub-strategies optimized for each environment.

Regimes:
  0 = Low Vol   (VIX < 15, contango)     → 100% leveraged risk parity (UPRO/TMF/UGL)
  1 = Normal    (VIX 15-25)              → 70% risk parity + 30% cross-asset momentum
  2 = High Vol  (VIX > 25/backwardation) → 50% TLT + 30% GLD + 20% cash

Walk-forward: 252d lookback, monthly rebalance, NO lookahead.
HC #0: sliding windows only. Commission-free (HC #694).
Outputs saved to /home/nick/Lvl3Quant/output/vol_regime_switch_v1/
"""

import sys, os, json, warnings
# Force unbuffered output so logs appear in real time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from hmmlearn import hmm

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/home/nick/Lvl3Quant/output/vol_regime_switch_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Vol Regime Switch v1", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["SPY", "^VIX", "^VIX3M", "UPRO", "TMF", "UGL", "TLT", "GLD", "IEF", "EFA", "EEM", "HYG"]
START = "2011-01-01"
END   = "2026-07-01"

print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading {len(TICKERS)} tickers...", flush=True)
raw = {}
for t in TICKERS:
    try:
        df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  WARNING: {t} returned empty", flush=True)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        raw[t] = df["Close"].rename(t)
        print(f"  {t}: {len(df)} rows  ({df.index[0].date()} - {df.index[-1].date()})", flush=True)
    except Exception as e:
        print(f"  ERROR {t}: {e}", flush=True)

price = pd.DataFrame(raw).sort_index()
price.index = pd.to_datetime(price.index)

vix   = price["^VIX"].rename("VIX").ffill()
vix3m = price["^VIX3M"].rename("VIX3M").ffill()

assets = ["SPY", "UPRO", "TMF", "UGL", "TLT", "GLD", "IEF", "EFA", "EEM", "HYG"]
asset_prices = price[assets].ffill()

print(f"[{datetime.now().strftime('%H:%M:%S')}] Price matrix: {asset_prices.shape}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

spy_ret = asset_prices["SPY"].pct_change()
realized_vol = spy_ret.rolling(21).std() * np.sqrt(252)
ts_ratio = vix / vix3m   # >1 = backwardation (stress)

features = pd.DataFrame({
    "vix": vix,
    "ts_ratio": ts_ratio,
    "rvol_21d": realized_vol,
}).dropna()

print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature matrix: {features.shape}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime_threshold(vix_val, ts_val):
    """Threshold classifier: 0=Low, 1=Normal, 2=High vol"""
    if vix_val > 25 or ts_val > 1.0:
        return 2
    elif vix_val < 15 and ts_val < 1.0:
        return 0
    else:
        return 1

# Threshold labels (fast, used as fallback and for first 252 days)
regime_thresh = features.apply(
    lambda r: classify_regime_threshold(r["vix"], r["ts_ratio"]), axis=1
)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Fitting HMM regimes (refit every 63 days)...", flush=True)

def fit_hmm_regimes(feature_df, lookback=252, refit_interval=63):
    """
    Rolling HMM classifier. Only refit every refit_interval days to save time.
    Always uses data up to (but not including) current day — no lookahead.
    """
    labels = regime_thresh.copy()   # start from threshold labels
    feat_arr = feature_df[["vix", "ts_ratio", "rvol_21d"]].values
    n = len(feat_arr)

    last_refit = -refit_interval
    current_label = None

    for i in range(lookback, n):
        if i - last_refit < refit_interval:
            if current_label is not None:
                labels.iloc[i] = current_label
            continue

        window = feat_arr[i - lookback:i]
        mu = window.mean(axis=0)
        sd = window.std(axis=0) + 1e-8
        window_norm = (window - mu) / sd

        try:
            model = hmm.GaussianHMM(
                n_components=3,
                covariance_type="diag",   # faster than 'full'
                n_iter=50,
                random_state=42,
                tol=1e-2,
            )
            model.fit(window_norm)
            window_states = model.predict(window_norm)

            # Map states to economic regimes by mean VIX in each state
            state_vix = {}
            for s in range(3):
                mask = window_states == s
                state_vix[s] = window[mask, 0].mean() if mask.sum() > 0 else 20.0
            sorted_states = sorted(state_vix.items(), key=lambda x: x[1])
            regime_map = {sorted_states[s][0]: s for s in range(3)}

            # Classify current point
            cur_norm = (feat_arr[i:i+1] - mu) / sd
            cur_state = model.predict(cur_norm)[0]
            current_label = regime_map[cur_state]
            labels.iloc[i] = current_label
            last_refit = i

        except Exception:
            # Fall back to threshold
            r = feature_df.iloc[i]
            current_label = classify_regime_threshold(r["vix"], r["ts_ratio"])
            labels.iloc[i] = current_label
            last_refit = i

    return labels

regime_labels_hmm = fit_hmm_regimes(features, lookback=252, refit_interval=63)

# For first 252 days (warmup), use threshold
regime_labels = regime_labels_hmm.copy()
regime_labels.iloc[:252] = regime_thresh.iloc[:252]

rc = regime_labels.value_counts().sort_index()
names = {0: "Low Vol", 1: "Normal", 2: "High Vol"}
print(f"[{datetime.now().strftime('%H:%M:%S')}] Regime distribution:", flush=True)
for r, cnt in rc.items():
    print(f"  Regime {r} ({names[r]}): {cnt} days ({100*cnt/len(regime_labels):.1f}%)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATEGY WEIGHTS PER REGIME
# ─────────────────────────────────────────────────────────────────────────────

def get_regime_weights(regime_id):
    if regime_id == 0:
        # Low vol: 100% leveraged risk parity
        return {"UPRO": 0.60, "TMF": 0.25, "UGL": 0.15}
    elif regime_id == 1:
        # Normal: 70% risk parity + 30% cross-asset momentum
        return {"SPY": 0.25, "TLT": 0.25, "GLD": 0.10, "IEF": 0.10,
                "EFA": 0.10, "EEM": 0.10, "HYG": 0.10}
    else:
        # High vol: safe haven (20% cash implicit)
        return {"TLT": 0.50, "GLD": 0.30}

# ─────────────────────────────────────────────────────────────────────────────
# 5. WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

BACKTEST_START = "2012-03-01"
bt_prices = asset_prices[asset_prices.index >= BACKTEST_START].copy()
bt_regime  = regime_labels[regime_labels.index >= BACKTEST_START].copy()
common_idx = bt_prices.index.intersection(bt_regime.index)
bt_prices  = bt_prices.loc[common_idx]
bt_regime  = bt_regime.loc[common_idx]

print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest: {bt_prices.index[0].date()} - {bt_prices.index[-1].date()} ({len(bt_prices)} days)", flush=True)

# Monthly rebalance dates
bt_prices["_month"] = bt_prices.index.to_period("M")
rebalance_dates = set(bt_prices.groupby("_month").apply(lambda g: g.index[0]).values)
bt_prices = bt_prices.drop(columns="_month")

daily_ret = bt_prices.pct_change()

# Fast vectorized backtest
portfolio_ret = pd.Series(0.0, index=bt_prices.index)
current_weights = {}
rebalance_log = []
MOMENTUM_ASSETS = ["EFA", "EEM", "HYG", "SPY", "TLT", "GLD"]

for i in range(1, len(bt_prices)):
    date = bt_prices.index[i]
    prev_date = bt_prices.index[i-1]

    if date in rebalance_dates or i == 1:
        regime = int(bt_regime.loc[prev_date])
        base_weights = get_regime_weights(regime)

        # Normal regime: momentum-rank the 30% sleeve
        if regime == 1:
            hist = bt_prices[MOMENTUM_ASSETS].loc[:prev_date].tail(148)
            if len(hist) >= 126:
                ret_12m = hist.iloc[-1] / hist.iloc[0] - 1
                ret_1m  = hist.iloc[-1] / hist.iloc[-22] - 1
                mom_score = (ret_12m - ret_1m).dropna()
                top3 = mom_score.nlargest(3)
                top3_pos = top3[top3 > 0]
                core = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.10, "IEF": 0.10}
                if len(top3_pos) > 0:
                    mom_w = (top3_pos / top3_pos.sum()) * 0.30
                    for tkr, w in mom_w.items():
                        core[tkr] = core.get(tkr, 0) + w
                current_weights = core
            else:
                current_weights = base_weights
        else:
            current_weights = base_weights

        rebalance_log.append({
            "date": str(date.date()),
            "regime": regime,
            "weights": {k: round(v, 4) for k, v in current_weights.items()},
        })

    # Daily return
    if current_weights:
        dr = sum(
            w * daily_ret.loc[date, tkr]
            for tkr, w in current_weights.items()
            if tkr in daily_ret.columns and not pd.isna(daily_ret.loc[date, tkr])
        )
        portfolio_ret.loc[date] = dr

portfolio_ret = portfolio_ret.iloc[1:]  # drop first row (NaN day)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest complete. {len(portfolio_ret)} return days.", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

spy_ret_bt   = daily_ret["SPY"].loc[portfolio_ret.index]
static_rp_w  = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.15, "IEF": 0.15, "EFA": 0.10, "EEM": 0.10}
static_rp    = sum(daily_ret[t].loc[portfolio_ret.index] * w for t, w in static_rp_w.items() if t in daily_ret.columns)
static_6040  = 0.60 * daily_ret["SPY"].loc[portfolio_ret.index] + 0.40 * daily_ret["TLT"].loc[portfolio_ret.index]

# ─────────────────────────────────────────────────────────────────────────────
# 7. PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def calc_metrics(ret, name="Strategy"):
    ret = ret.dropna()
    cum = (1 + ret).cumprod()
    n_years = len(ret) / 252
    cagr = cum.iloc[-1] ** (1 / n_years) - 1
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino  = ann_ret / downside if downside > 0 else 0.0
    roll_max = cum.cummax()
    max_dd   = ((cum - roll_max) / roll_max).min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0.0
    wr       = (ret > 0).mean()
    gross_up   = ret[ret > 0].sum()
    gross_dn   = abs(ret[ret < 0].sum())
    pf = gross_up / gross_dn if gross_dn > 0 else np.inf
    yearly = ret.groupby(ret.index.year).apply(lambda r: (1+r).prod() - 1)
    best_year  = yearly.max()
    worst_year = yearly.min()
    asymmetry  = best_year / abs(worst_year) if worst_year != 0 else np.inf
    return {
        "name": name,
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_dd": round(float(max_dd), 4),
        "calmar": round(float(calmar), 3),
        "ann_vol": round(float(ann_vol), 4),
        "win_rate": round(float(wr), 4),
        "profit_factor": round(float(pf), 3),
        "best_year": round(float(best_year), 4),
        "worst_year": round(float(worst_year), 4),
        "asymmetry_ratio": round(float(asymmetry), 3),
        "n_years": round(float(n_years), 2),
        "yearly_returns": {int(k): round(float(v), 4) for k, v in yearly.items()},
    }

strat_m = calc_metrics(portfolio_ret, "Vol Regime Switch v1")
spy_m   = calc_metrics(spy_ret_bt,    "SPY buy-hold")
rp_m    = calc_metrics(static_rp,     "Static Risk Parity")
bf_m    = calc_metrics(static_6040,   "Static 60/40")

print(f"\n{'='*60}", flush=True)
print("PERFORMANCE SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for m in [strat_m, spy_m, rp_m, bf_m]:
    print(f"\n{m['name']}", flush=True)
    print(f"  CAGR:      {m['cagr']*100:.2f}%", flush=True)
    print(f"  Sharpe:    {m['sharpe']:.3f}", flush=True)
    print(f"  Sortino:   {m['sortino']:.3f}", flush=True)
    print(f"  MaxDD:     {m['max_dd']*100:.2f}%", flush=True)
    print(f"  Calmar:    {m['calmar']:.3f}", flush=True)
    print(f"  Asymmetry: {m['asymmetry_ratio']:.3f}  (best_yr/|worst_yr|)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8. REGIME TEST (HC #428 R1)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}", flush=True)
print("REGIME TEST (HC #428 R1)", flush=True)
print(f"{'='*60}", flush=True)

regime_perf = {}
for r in [0, 1, 2]:
    mask = bt_regime.loc[portfolio_ret.index] == r
    r_ret = portfolio_ret[mask]
    if len(r_ret) < 20:
        regime_perf[r] = {"regime_name": names[r], "n_days": int(mask.sum()), "sharpe": 0.0}
        continue
    r_ann = r_ret.mean() * 252
    r_vol = r_ret.std() * np.sqrt(252)
    r_sh  = r_ann / r_vol if r_vol > 0 else 0.0
    regime_perf[r] = {
        "regime_name": names[r],
        "n_days": int(mask.sum()),
        "sharpe": round(float(r_sh), 3),
        "ann_ret": round(float(r_ann), 4),
    }
    print(f"  {names[r]}: {int(mask.sum())} days | Sharpe={r_sh:.3f} | AnnRet={r_ann*100:.1f}%", flush=True)

sh_low  = regime_perf[0]["sharpe"]
sh_high = regime_perf[2]["sharpe"]
denom   = max(abs(sh_low), abs(sh_high))
regime_bias = abs(sh_low - sh_high) / denom if denom > 0 else 0.0
hc428_pass  = regime_bias <= 0.50
print(f"\n  Regime bias: {regime_bias:.3f} (threshold 0.50) → {'PASS' if hc428_pass else 'FAIL'}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9. PERMUTATION TEST (shuffle regime labels, 1000 trials)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}", flush=True)
print("PERMUTATION TEST (N=1000, shuffle regime labels)", flush=True)
print(f"{'='*60}", flush=True)

observed_sharpe = strat_m["sharpe"]
perm_sharpes = []
np.random.seed(42)

regime_arr   = bt_regime.loc[portfolio_ret.index].values.copy()
ret_for_perm = daily_ret.loc[portfolio_ret.index]
rebal_set    = set(rebalance_dates)
date_list    = list(portfolio_ret.index)

# Pre-build rebalance index set for speed
rebal_idx_set = {i for i, d in enumerate(date_list) if d in rebal_set}

print(f"  Running 1000 permutations...", flush=True)
for perm_i in range(1000):
    if perm_i % 200 == 0:
        print(f"  ... perm {perm_i}/1000", flush=True)

    shuf = regime_arr.copy()
    np.random.shuffle(shuf)

    perm_ret = []
    pw = {}
    for i in range(len(date_list)):
        if i in rebal_idx_set or i == 0:
            pw = get_regime_weights(int(shuf[max(0, i-1)]))
        dr = 0.0
        for tkr, w in pw.items():
            if tkr in ret_for_perm.columns:
                v = ret_for_perm.iloc[i][tkr]
                if not pd.isna(v):
                    dr += w * v
        perm_ret.append(dr)

    pr = np.array(perm_ret)
    ann_r = pr.mean() * 252
    ann_v = pr.std() * np.sqrt(252)
    perm_sharpes.append(ann_r / ann_v if ann_v > 0 else 0.0)

perm_sharpes = np.array(perm_sharpes)
p_value = (perm_sharpes >= observed_sharpe).mean()
z_score = (observed_sharpe - perm_sharpes.mean()) / (perm_sharpes.std() + 1e-9)

print(f"\n  Observed Sharpe:     {observed_sharpe:.3f}", flush=True)
print(f"  Permutation mean:    {perm_sharpes.mean():.3f} ± {perm_sharpes.std():.3f}", flush=True)
print(f"  p-value:             {p_value:.4f}", flush=True)
print(f"  z-score:             {z_score:.2f}", flush=True)
print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10. YEARLY BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}", flush=True)
print("YEARLY RETURNS", flush=True)
print(f"{'Strategy':>12} {'SPY':>7} {'StaticRP':>9} {'60/40':>7}", flush=True)
for yr in sorted(strat_m["yearly_returns"].keys()):
    s  = strat_m["yearly_returns"].get(yr, 0)
    sp = spy_m["yearly_returns"].get(yr, 0)
    rp = rp_m["yearly_returns"].get(yr, 0)
    b  = bf_m["yearly_returns"].get(yr, 0)
    print(f"{yr}  {s*100:>9.1f}%  {sp*100:>5.1f}%  {rp*100:>7.1f}%  {b*100:>5.1f}%", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

results = {
    "run_date": datetime.now().isoformat(),
    "backtest_start": str(bt_prices.index[0].date()),
    "backtest_end":   str(bt_prices.index[-1].date()),
    "n_trading_days": len(portfolio_ret),
    "strategy": strat_m,
    "benchmarks": {"spy": spy_m, "static_risk_parity": rp_m, "static_6040": bf_m},
    "regime_performance": regime_perf,
    "regime_test_hc428": {
        "regime_bias_ratio": round(float(regime_bias), 4),
        "threshold": 0.50,
        "pass": bool(hc428_pass),
    },
    "permutation_test": {
        "n_permutations": 1000,
        "observed_sharpe": float(observed_sharpe),
        "perm_mean": float(perm_sharpes.mean()),
        "perm_std": float(perm_sharpes.std()),
        "p_value": float(p_value),
        "z_score": float(z_score),
        "significant_p05": bool(p_value < 0.05),
    },
    "rebalance_log_sample": rebalance_log[:12],
    "regime_counts": {str(k): int(v) for k, v in rc.items()},
}

out_path = os.path.join(OUTPUT_DIR, "results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# Save daily returns
pd.DataFrame({
    "date": portfolio_ret.index.strftime("%Y-%m-%d"),
    "vol_regime_switch": portfolio_ret.values,
    "spy": spy_ret_bt.reindex(portfolio_ret.index).values,
    "static_rp": static_rp.reindex(portfolio_ret.index).values,
    "static_6040": static_6040.reindex(portfolio_ret.index).values,
    "regime": bt_regime.reindex(portfolio_ret.index).values,
}).to_csv(os.path.join(OUTPUT_DIR, "daily_returns.csv"), index=False)

# Save equity curves
pd.DataFrame({
    "date": portfolio_ret.index.strftime("%Y-%m-%d"),
    "vol_regime_switch": (1 + portfolio_ret).cumprod().values,
    "spy": (1 + spy_ret_bt.reindex(portfolio_ret.index)).cumprod().values,
    "static_rp": (1 + static_rp.reindex(portfolio_ret.index)).cumprod().values,
    "static_6040": (1 + static_6040.reindex(portfolio_ret.index)).cumprod().values,
}).to_csv(os.path.join(OUTPUT_DIR, "equity_curves.csv"), index=False)

print(f"\n{'='*60}", flush=True)
print(f"[{datetime.now().strftime('%H:%M:%S')}] ALL DONE — results saved to {OUTPUT_DIR}", flush=True)
print(f"{'='*60}", flush=True)
print(f"\nFINAL SUMMARY:", flush=True)
print(f"  CAGR:         {strat_m['cagr']*100:.2f}%", flush=True)
print(f"  Sharpe:       {strat_m['sharpe']:.3f}", flush=True)
print(f"  Sortino:      {strat_m['sortino']:.3f}", flush=True)
print(f"  MaxDD:        {strat_m['max_dd']*100:.2f}%", flush=True)
print(f"  Calmar:       {strat_m['calmar']:.3f}", flush=True)
print(f"  Asymmetry:    {strat_m['asymmetry_ratio']:.3f}  (best_yr / |worst_yr|)", flush=True)
print(f"  Perm p-value: {p_value:.4f}  ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})", flush=True)
print(f"  HC428 R1:     {'PASS' if hc428_pass else 'FAIL'}", flush=True)
