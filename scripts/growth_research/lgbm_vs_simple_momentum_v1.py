#!/usr/bin/env python3
"""
LightGBM vs Simple Momentum — Is ML Actually Adding Value?
============================================================
Critical validation: our sector options rotation uses LightGBM to rank sectors.
But does ML ranking actually beat simple momentum (top N by recent return)?

Tests:
A) LightGBM ranked (our current approach)
B) Simple 21-day return ranking
C) Simple 63-day return ranking
D) Equal weight (buy all sectors)
E) Random picks (permutation baseline)

All use identical: bi-weekly rebalance, ATR-based option pricing, $645 start,
tiered position sizing, same confluence gate.

If B/C match A → LightGBM is wasted complexity.
If A >> B/C → ML genuinely adds value.
"""
import json, os, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_PATH / "output" / "growth_research" / "lgbm_vs_simple_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL_CAPITAL = 645.0
COMMISSION_PER_TRADE = 2.60
HAIRCUT = 0.15
SPREAD_PCT = 3.0
DTE = 30
TOP_K = 3
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
FEAT_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
             'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']

N_PERM = 200  # permutation shuffles

def fprint(*a, **kw): print(*a, **kw, flush=True)

def download_data():
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    ix = ix.intersection(sh.index).intersection(sl.index)
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

def compute_features(px, idx, ticker):
    s = px[ticker].iloc[:idx+1]
    if len(s) < 252: return None
    return {
        'ret_5d': float(s.pct_change(5).iloc[-1]),
        'ret_10d': float(s.pct_change(10).iloc[-1]),
        'ret_21d': float(s.pct_change(21).iloc[-1]),
        'ret_63d': float(s.pct_change(63).iloc[-1]),
        'ret_126d': float(s.pct_change(126).iloc[-1]),
        'ret_252d': float(s.pct_change(252).iloc[-1]),
        'vol_21d': float(s.pct_change().tail(21).std()),
        'vol_63d': float(s.pct_change().tail(63).std()),
        'sharpe_63d': float(s.pct_change().tail(63).mean() / max(s.pct_change().tail(63).std(), 1e-8)),
        'maxdd_63d': float((s.tail(63) / s.tail(63).cummax() - 1).min()),
        'pct_52w_high': float(s.iloc[-1] / s.tail(252).max()),
        'mom_accel': float(s.pct_change(21).iloc[-1] - s.pct_change(21).iloc[-22]) if len(s) > 43 else 0,
    }

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte / 252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val / 20.0)
    time_prem = atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S-K)/S)
    return intrinsic + time_prem

def tiered_position(equity):
    if equity < 2000: return min(200, equity / 3)
    elif equity < 10000: return min(500, equity / 3)
    else: return min(1000, equity / 3)

def build_lgbm_rankings(sc, dates):
    """Full walk-forward LGBM ranking."""
    fprint("  Building LightGBM WF rankings...")
    all_feats = []
    for di in range(252, len(dates)):
        dt = dates[di]
        for tk in sc.columns:
            f = compute_features(sc, di, tk)
            if f:
                fwd_ret = float(sc[tk].iloc[min(di+10, len(dates)-1)] / sc[tk].iloc[di] - 1)
                f['ticker'] = tk; f['date'] = dt; f['fwd_ret'] = fwd_ret
                all_feats.append(f)
    df = pd.DataFrame(all_feats)
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    udates = sorted(df['date'].unique())
    train_periods = 252
    rankings = {}
    for i in range(train_periods, len(udates)):
        td = udates[max(0, i-train_periods):i]; test_date = udates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[FEAT_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[FEAT_COLS].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    fprint(f"    LGBM rankings: {len(rankings)} dates")
    return rankings

def build_simple_rankings(sc, lookback=21):
    """Simple momentum ranking by N-day return."""
    rankings = {}
    dates = sc.index
    for di in range(max(252, lookback), len(dates)):
        dt = dates[di]
        scores = {}
        for tk in sc.columns:
            s = sc[tk].iloc[:di+1]
            if len(s) >= lookback and not pd.isna(s.iloc[-1]) and not pd.isna(s.iloc[-lookback]):
                scores[tk] = float(s.iloc[-1] / s.iloc[-lookback] - 1)
        if len(scores) >= 3:
            rankings[dt] = scores
    return rankings

def build_equal_rankings(sc):
    """Equal weight — all sectors get same score."""
    rankings = {}
    dates = sc.index
    for di in range(252, len(dates)):
        dt = dates[di]
        rankings[dt] = {tk: 1.0 for tk in sc.columns if not pd.isna(sc[tk].iloc[di])}
    return rankings

def build_random_rankings(sc, seed=0):
    """Random ranking — permutation baseline."""
    rng = np.random.default_rng(seed)
    rankings = {}
    dates = sc.index
    for di in range(252, len(dates)):
        dt = dates[di]
        tickers = [tk for tk in sc.columns if not pd.isna(sc[tk].iloc[di])]
        rankings[dt] = {tk: float(rng.random()) for tk in tickers}
    return rankings

def simulate(name, rankings, sc, sh, sl, spy, vix, top_k=TOP_K):
    """Run spread simulation with given rankings."""
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns
             if tk in sh.columns and tk in sl.columns}

    # Bi-weekly dates
    rdates = sorted(rankings.keys())
    rebal = rdates[::10]

    equity = INITIAL_CAPITAL
    trades = []

    for dt in rebal:
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])
        scores = rankings[dt]
        if not scores: continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]

        max_pos = tiered_position(equity)
        if max_pos < 30: continue

        for tk in picks:
            if tk not in sc.columns or tk not in atr_d: continue
            di = sc.index.get_loc(dt)
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if not pd.isna(atr_d[tk].loc[dt]) else S * 0.015

            K1 = round(S); K2 = round(S * (1 + SPREAD_PCT/100))
            lp = atr_premium(S, K1, DTE, av, cv, 'call') * (1 + HAIRCUT)
            sp = atr_premium(S, K2, DTE, av, cv, 'call') * (1 - HAIRCUT)
            debit = lp - sp; width = K2 - K1
            cost = debit * 100 + COMMISSION_PER_TRADE
            max_profit = (width - debit) * 100 - COMMISSION_PER_TRADE

            if cost <= 0 or cost > max_pos or cost > equity * 0.40: continue

            ei = min(di + DTE, len(sc) - 1)
            pnl = None
            for ci in range(di + 7, ei + 1):
                Sc = float(sc[tk].iloc[ci]); dh = ci - di; rd = max(0, DTE - dh)
                tm = np.sqrt(rd / max(DTE, 1))
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                si = (max(0, Sc-K1) - max(0, Sc-K2)) * 100 + ac * tm * 0.3 * 100
                cp = si - cost
                if cp >= max_profit * 0.50: pnl = cp; break
                if cp <= -cost * 0.80: pnl = cp; break

            if pnl is None:
                Se = float(sc[tk].iloc[ei])
                si = (max(0, Se-K1) - max(0, Se-K2)) * 100
                pnl = si - cost

            equity = max(0, equity + pnl)
            trades.append({"pnl": pnl, "date": str(dt.date())})

    wins = sum(1 for t in trades if t["pnl"] > 0)
    total = len(trades)
    wr = wins / max(1, total)
    total_pnl = sum(t["pnl"] for t in trades)
    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if total - wins > 0 else 0
    pf = abs(sum(t["pnl"] for t in trades if t["pnl"] > 0) /
             min(-1, sum(t["pnl"] for t in trades if t["pnl"] <= 0))) if total > 0 else 0

    # Regime analysis (R1)
    spy_daily = spy.pct_change()
    spy_monthly = spy.resample('ME').last().pct_change()

    green_pnl, red_pnl = [], []
    for t in trades:
        td = pd.Timestamp(t["date"])
        # Find closest month
        m = td.to_period('M').to_timestamp('M')
        if m in spy_monthly.index and not pd.isna(spy_monthly.loc[m]):
            if spy_monthly.loc[m] > 0:
                green_pnl.append(t["pnl"])
            else:
                red_pnl.append(t["pnl"])

    if green_pnl and red_pnl:
        green_sharpe = np.mean(green_pnl) / max(np.std(green_pnl), 1e-8)
        red_sharpe = np.mean(red_pnl) / max(np.std(red_pnl), 1e-8)
        r1_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 1e-8)
    else:
        r1_gap = 1.0

    return {
        "name": name,
        "trades": total,
        "wr": round(wr * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(equity, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "pf": round(pf, 2),
        "r1_gap": round(r1_gap, 3),
        "r1_pass": "PASS" if r1_gap < 0.50 else "FAIL",
    }

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("LightGBM vs Simple Momentum — ML Value Validation")
    fprint(f"Capital: ${INITIAL_CAPITAL:.0f} | Bi-weekly | Top {TOP_K}")
    fprint("=" * 70)

    sc, sh, sl, spy, vix = download_data()
    dates = sc.index
    fprint(f"Data: {len(dates)} days, {len(sc.columns)} sectors")

    results = {}

    # A) LightGBM
    lgbm_ranks = build_lgbm_rankings(sc, dates)
    r = simulate("A_LightGBM", lgbm_ranks, sc, sh, sl, spy, vix)
    results["A_LightGBM"] = r
    fprint(f"\n  A) LightGBM: {r['trades']} trades, WR {r['wr']}%, "
           f"PF {r['pf']}, Final ${r['final_equity']:,.0f}, R1 {r['r1_gap']:.3f} {r['r1_pass']}")

    # B) Simple 21d momentum
    fprint("  Building simple 21d momentum rankings...")
    mom21_ranks = build_simple_rankings(sc, lookback=21)
    r = simulate("B_Mom21d", mom21_ranks, sc, sh, sl, spy, vix)
    results["B_Mom21d"] = r
    fprint(f"  B) Mom 21d:  {r['trades']} trades, WR {r['wr']}%, "
           f"PF {r['pf']}, Final ${r['final_equity']:,.0f}, R1 {r['r1_gap']:.3f} {r['r1_pass']}")

    # C) Simple 63d momentum
    fprint("  Building simple 63d momentum rankings...")
    mom63_ranks = build_simple_rankings(sc, lookback=63)
    r = simulate("C_Mom63d", mom63_ranks, sc, sh, sl, spy, vix)
    results["C_Mom63d"] = r
    fprint(f"  C) Mom 63d:  {r['trades']} trades, WR {r['wr']}%, "
           f"PF {r['pf']}, Final ${r['final_equity']:,.0f}, R1 {r['r1_gap']:.3f} {r['r1_pass']}")

    # D) Equal weight
    fprint("  Building equal weight rankings...")
    eq_ranks = build_equal_rankings(sc)
    r = simulate("D_EqualWeight", eq_ranks, sc, sh, sl, spy, vix)
    results["D_EqualWeight"] = r
    fprint(f"  D) EqWeight: {r['trades']} trades, WR {r['wr']}%, "
           f"PF {r['pf']}, Final ${r['final_equity']:,.0f}, R1 {r['r1_gap']:.3f} {r['r1_pass']}")

    # E) Random baseline (average of N_PERM shuffles)
    fprint(f"  Running {N_PERM} random permutation baselines...")
    rand_finals = []
    rand_wrs = []
    rand_pfs = []
    for seed in range(N_PERM):
        rand_ranks = build_random_rankings(sc, seed)
        r = simulate(f"E_Random_{seed}", rand_ranks, sc, sh, sl, spy, vix)
        rand_finals.append(r["final_equity"])
        rand_wrs.append(r["wr"])
        rand_pfs.append(r["pf"])
        if seed % 50 == 0:
            fprint(f"    Permutation {seed}/{N_PERM}...")

    results["E_Random"] = {
        "name": "E_Random (avg of 200 shuffles)",
        "median_final": round(float(np.median(rand_finals)), 2),
        "mean_final": round(float(np.mean(rand_finals)), 2),
        "p5_final": round(float(np.percentile(rand_finals, 5)), 2),
        "p95_final": round(float(np.percentile(rand_finals, 95)), 2),
        "median_wr": round(float(np.median(rand_wrs)), 1),
        "median_pf": round(float(np.median(rand_pfs)), 2),
    }
    fprint(f"  E) Random:   median ${np.median(rand_finals):,.0f}, "
           f"WR {np.median(rand_wrs):.1f}%, PF {np.median(rand_pfs):.2f}")

    # ─── Summary ───
    fprint(f"\n{'='*70}")
    fprint("COMPARISON SUMMARY")
    fprint(f"{'='*70}")
    fprint(f"{'Method':<20} {'Trades':>7} {'WR':>6} {'PF':>7} {'Final':>12} {'R1':>6} {'R1 Pass':>8}")
    fprint("-" * 75)
    for key in ["A_LightGBM", "B_Mom21d", "C_Mom63d", "D_EqualWeight"]:
        r = results[key]
        fprint(f"{r['name']:<20} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>7.2f} "
               f"${r['final_equity']:>10,.0f} {r['r1_gap']:>6.3f} {r['r1_pass']:>8}")
    re = results["E_Random"]
    fprint(f"{'E_Random (median)':<20} {'N/A':>7} {re['median_wr']:>5.1f}% {re['median_pf']:>7.2f} "
           f"${re['median_final']:>10,.0f} {'N/A':>6} {'N/A':>8}")

    # ─── Key Question ───
    fprint(f"\n{'='*70}")
    fprint("KEY QUESTION: Does LightGBM Beat Simple Momentum?")
    fprint(f"{'='*70}")
    lgbm_final = results["A_LightGBM"]["final_equity"]
    mom21_final = results["B_Mom21d"]["final_equity"]
    mom63_final = results["C_Mom63d"]["final_equity"]
    eq_final = results["D_EqualWeight"]["final_equity"]
    rand_med = re["median_final"]

    fprint(f"  LightGBM: ${lgbm_final:,.0f}")
    fprint(f"  Mom 21d:  ${mom21_final:,.0f} ({(lgbm_final/max(1,mom21_final)-1)*100:+.0f}% vs LGBM)")
    fprint(f"  Mom 63d:  ${mom63_final:,.0f} ({(lgbm_final/max(1,mom63_final)-1)*100:+.0f}% vs LGBM)")
    fprint(f"  EqWeight: ${eq_final:,.0f} ({(lgbm_final/max(1,eq_final)-1)*100:+.0f}% vs LGBM)")
    fprint(f"  Random:   ${rand_med:,.0f} ({(lgbm_final/max(1,rand_med)-1)*100:+.0f}% vs LGBM)")

    # Permutation test: is LightGBM significantly better than random?
    perm_p = np.mean(np.array(rand_finals) >= lgbm_final)
    fprint(f"\n  Permutation test: p = {perm_p:.3f} "
           f"({'SIGNIFICANT' if perm_p < 0.05 else 'NOT significant'})")

    if lgbm_final > mom21_final * 1.10:
        fprint(f"\n  VERDICT: LightGBM adds {(lgbm_final/max(1,mom21_final)-1)*100:.0f}% "
               f"over simple momentum → ML IS VALUABLE")
    elif lgbm_final > mom21_final * 0.90:
        fprint(f"\n  VERDICT: LightGBM within 10% of simple momentum → ML adds MINIMAL value")
        fprint(f"    Consider simplifying to pure momentum ranking (lower complexity)")
    else:
        fprint(f"\n  VERDICT: LightGBM WORSE than simple momentum → ML is HARMFUL, switch to simple")

    fprint(f"\n{'='*70}")

    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.1f}s")

    # Save
    out = OUTPUT_DIR / "lgbm_vs_simple_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"Saved: {out}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("lgbm_vs_simple_momentum_v1")
        with mlflow.start_run(run_name=f"lgbm_vs_mom_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.log_params({"n_perm": N_PERM, "top_k": TOP_K, "dte": DTE})
            mlflow.log_metrics({
                "lgbm_final": lgbm_final,
                "mom21_final": mom21_final,
                "mom63_final": mom63_final,
                "eq_final": eq_final,
                "random_median": rand_med,
                "perm_p": perm_p,
                "lgbm_wr": results["A_LightGBM"]["wr"],
                "mom21_wr": results["B_Mom21d"]["wr"],
            })
            mlflow.log_artifact(str(out))
            fprint("Logged to MLflow")
    except Exception as e:
        fprint(f"MLflow skip: {e}")

    fprint("Done.")

if __name__ == "__main__":
    main()
