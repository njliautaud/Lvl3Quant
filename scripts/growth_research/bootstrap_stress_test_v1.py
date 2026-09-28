#!/usr/bin/env python3
"""Bootstrap Stress Test v1 — Honest confidence intervals from actual trade returns.

Previous Monte Carlo (item 946) was overcalibrated. This uses BOOTSTRAP RESAMPLING
of actual backtest trade PnLs to get honest confidence intervals.

Tests: Best validated strategies from our research pipeline.
Method: Block bootstrap (preserve autocorrelation), 10K paths, with adversarial shocks.
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'bootstrap_stress_test_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

CAP = 645.0

# ==================== STRATEGY TRADE GENERATORS ====================
# Instead of synthetic returns, we replicate the actual backtest logic and extract trade PnLs

def generate_sector_rotation_trades():
    """Replicate our best strategy (multi-factor sector rotation) and extract trade PnLs."""
    import yfinance as yf
    import lightgbm as lgb

    SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
    HAIRCUT = 0.15; SPREAD_COMM = 2.60

    fprint("  Generating sector rotation trades...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    sc, sh, sl, spy, vix = sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

    # Features + LGBM WF
    FEAT_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
                 'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    records = []
    for dt in bd:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            if len(px) < 260: continue
            f = {}
            for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
                f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
            rets = px.pct_change().dropna()
            f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
            f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
            r63 = rets.iloc[-63:]
            f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0.0
            pk63 = px.iloc[-63:].cummax()
            f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
            f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
            f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
            fi = min(idx+14, len(sc)-1)
            f.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(f)
    df = pd.DataFrame(records)
    if len(df) < 100: return []
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(12, len(dates)):
        td = dates[max(0,i-12):i]; test_date = dates[i]
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

    # Simulate trades
    def compute_atr(h, l, c, period=14):
        tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
        return tr.rolling(period).mean()

    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    trade_pnls = []
    for dt in sorted(rankings.keys()):
        if dt not in vix.index: continue
        cv = float(vix.loc[dt])
        scores = rankings[dt]
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:3]]
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt); ei = min(di+30, len(sc)-1)
            Se = float(sc[tk].iloc[ei])
            # ATR pricing
            K1, K2 = round(S), round(S*1.03)
            T = 30/252.0
            vol_f = max(0.3, cv/20.0)
            lp = (max(0,S-K1) + av*np.sqrt(T)*vol_f*np.exp(-3.0*abs(S-K1)/S)) * (1+HAIRCUT)
            sp = (max(0,S-K2) + av*np.sqrt(T)*vol_f*np.exp(-3.0*abs(S-K2)/S)) * (1-HAIRCUT)
            val = lp - sp; width = K2 - K1
            cost = val*100 + SPREAD_COMM
            if cost <= 0 or cost > 200: continue

            # Walk forward with early exit
            pnl = None
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci]); dh = ci-di; rd = max(0,30-dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/30.0)
                si = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                cp = si - cost
                mx_prof = (width-val)*100 - SPREAD_COMM
                if cp >= mx_prof*0.50 or rd < 7: pnl = cp; break
            if pnl is None:
                pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - val*100 - SPREAD_COMM
            trade_pnls.append(float(pnl))

    fprint(f"  Generated {len(trade_pnls)} actual trade PnLs")
    return trade_pnls

# ==================== BOOTSTRAP ENGINE ====================
def block_bootstrap(pnls, n_paths=10000, n_periods=260, block_size=10, cap=645.0,
                    max_concurrent=3, fee_drag=0.0):
    """Block bootstrap: resample blocks of trades to preserve local correlation."""
    pnls = np.array(pnls)
    n = len(pnls)
    if n < block_size: block_size = max(1, n//2)

    final_equities = []
    max_dds = []
    ruin_count = 0
    ruin_threshold = cap * 0.10  # Ruin = equity drops below 10% of start

    # Pre-compute blocks
    n_blocks = n - block_size + 1

    for path in range(n_paths):
        equity = cap
        peak = cap
        max_dd = 0.0
        ruined = False

        # Sample n_periods/block_size blocks
        n_block_draws = n_periods // block_size + 1
        for _ in range(n_block_draws):
            start = np.random.randint(0, n_blocks)
            block = pnls[start:start+block_size]

            # Apply fee drag
            if fee_drag > 0:
                block = block - fee_drag

            for pnl in block:
                equity += pnl
                if equity < ruin_threshold:
                    ruined = True; break
                peak = max(peak, equity)
                dd = (equity - peak) / peak
                max_dd = min(max_dd, dd)

            if ruined: break

        final_equities.append(equity if not ruined else 0)
        max_dds.append(max_dd)
        if ruined: ruin_count += 1

    return np.array(final_equities), np.array(max_dds), ruin_count / n_paths

def compute_bootstrap_stats(finals, max_dds, ruin_pct, name, cap=645.0):
    """Compute percentile-based confidence intervals."""
    valid = finals[finals > 0]
    stats = {
        'name': name,
        'n_paths': len(finals),
        'ruin_pct': round(ruin_pct * 100, 2),
        'median_final': round(float(np.median(finals)), 0),
        'mean_final': round(float(np.mean(finals)), 0),
        'p5_final': round(float(np.percentile(finals, 5)), 0),
        'p10_final': round(float(np.percentile(finals, 10)), 0),
        'p25_final': round(float(np.percentile(finals, 25)), 0),
        'p75_final': round(float(np.percentile(finals, 75)), 0),
        'p90_final': round(float(np.percentile(finals, 90)), 0),
        'p95_final': round(float(np.percentile(finals, 95)), 0),
        'median_maxdd': round(float(np.median(max_dds)) * 100, 1),
        'p5_maxdd': round(float(np.percentile(max_dds, 5)) * 100, 1),
        'pct_profit': round(float((finals > cap).mean() * 100), 1),
        'pct_double': round(float((finals > cap * 2).mean() * 100), 1),
        'pct_5x': round(float((finals > cap * 5).mean() * 100), 1),
        'pct_10x': round(float((finals > cap * 10).mean() * 100), 1),
    }
    if len(valid) > 0:
        years = 5.0  # 260 bi-weekly periods ~ 5 years
        stats['median_cagr'] = round((float(np.median(valid)) / cap) ** (1/years) - 1, 4) * 100
    else:
        stats['median_cagr'] = 0.0
    return stats

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"Bootstrap Stress Test v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"Capital: ${CAP:.0f} | Method: Block bootstrap (10K paths)")
    fprint(f"Testing: actual backtest trade PnLs → honest confidence intervals")
    fprint(f"{'='*80}")

    # Generate actual trade PnLs from our best strategy
    trade_pnls = generate_sector_rotation_trades()
    if not trade_pnls or len(trade_pnls) < 20:
        fprint("FATAL: Not enough trades generated")
        return

    pnls = np.array(trade_pnls)
    fprint(f"\n=== TRADE PNL STATISTICS ===")
    fprint(f"Total trades: {len(pnls)}")
    fprint(f"Win rate: {(pnls > 0).mean()*100:.1f}%")
    fprint(f"Mean PnL: ${pnls.mean():.2f}")
    fprint(f"Median PnL: ${np.median(pnls):.2f}")
    fprint(f"Std PnL: ${pnls.std():.2f}")
    fprint(f"Skew: {float(pd.Series(pnls).skew()):.2f}")
    fprint(f"Kurt: {float(pd.Series(pnls).kurtosis()):.2f}")
    fprint(f"Best trade: ${pnls.max():.2f}")
    fprint(f"Worst trade: ${pnls.min():.2f}")
    fprint(f"P10/P25/P50/P75/P90: ${np.percentile(pnls,10):.0f} / ${np.percentile(pnls,25):.0f} / ${np.percentile(pnls,50):.0f} / ${np.percentile(pnls,75):.0f} / ${np.percentile(pnls,90):.0f}")

    results = []

    # Scenario 1: Base case (actual trade PnLs, no modifications)
    fprint(f"\n=== SCENARIO 1: BASE CASE ===")
    f1, d1, r1 = block_bootstrap(pnls, n_paths=10000, n_periods=260, block_size=10, cap=CAP)
    s1 = compute_bootstrap_stats(f1, d1, r1, 'Base Case', CAP)
    results.append(s1)

    # Scenario 2: Reduced win rate (-10% WR)
    fprint(f"\n=== SCENARIO 2: DEGRADED WIN RATE (-10%) ===")
    pnls_deg = pnls.copy()
    # Flip 10% of winners to losers
    winners = np.where(pnls_deg > 0)[0]
    n_flip = int(len(winners) * 0.10)
    flip_idx = np.random.choice(winners, n_flip, replace=False)
    pnls_deg[flip_idx] = -abs(pnls_deg[flip_idx])
    f2, d2, r2 = block_bootstrap(pnls_deg, n_paths=10000, n_periods=260, block_size=10, cap=CAP)
    s2 = compute_bootstrap_stats(f2, d2, r2, 'Degraded WR (-10%)', CAP)
    results.append(s2)

    # Scenario 3: Fat tails (occasional large losses)
    fprint(f"\n=== SCENARIO 3: FAT TAILS (5% chance of 3x loss) ===")
    pnls_fat = pnls.copy()
    losers = np.where(pnls_fat < 0)[0]
    n_fat = max(1, int(len(losers) * 0.05))
    fat_idx = np.random.choice(losers, n_fat, replace=False)
    pnls_fat[fat_idx] = pnls_fat[fat_idx] * 3.0
    f3, d3, r3 = block_bootstrap(pnls_fat, n_paths=10000, n_periods=260, block_size=10, cap=CAP)
    s3 = compute_bootstrap_stats(f3, d3, r3, 'Fat Tails (3x loss)', CAP)
    results.append(s3)

    # Scenario 4: Commission increase (+50%)
    fprint(f"\n=== SCENARIO 4: HIGHER COMMISSIONS (+50%) ===")
    extra_comm = 1.30  # Additional $1.30 per trade
    f4, d4, r4 = block_bootstrap(pnls, n_paths=10000, n_periods=260, block_size=10, cap=CAP, fee_drag=extra_comm)
    s4 = compute_bootstrap_stats(f4, d4, r4, 'Higher Commissions', CAP)
    results.append(s4)

    # Scenario 5: Correlated losses (larger blocks = worse sequences)
    fprint(f"\n=== SCENARIO 5: CORRELATED LOSSES (block=30) ===")
    f5, d5, r5 = block_bootstrap(pnls, n_paths=10000, n_periods=260, block_size=30, cap=CAP)
    s5 = compute_bootstrap_stats(f5, d5, r5, 'Correlated (block=30)', CAP)
    results.append(s5)

    # Scenario 6: Combined adversarial (degraded WR + fat tails)
    fprint(f"\n=== SCENARIO 6: COMBINED ADVERSARIAL ===")
    pnls_adv = pnls.copy()
    winners_a = np.where(pnls_adv > 0)[0]
    n_flip_a = int(len(winners_a) * 0.10)
    if n_flip_a > 0:
        flip_a = np.random.choice(winners_a, n_flip_a, replace=False)
        pnls_adv[flip_a] = -abs(pnls_adv[flip_a])
    losers_a = np.where(pnls_adv < 0)[0]
    n_fat_a = max(1, int(len(losers_a) * 0.05))
    fat_a = np.random.choice(losers_a, min(n_fat_a, len(losers_a)), replace=False)
    pnls_adv[fat_a] = pnls_adv[fat_a] * 3.0
    f6, d6, r6 = block_bootstrap(pnls_adv, n_paths=10000, n_periods=260, block_size=10, cap=CAP, fee_drag=extra_comm)
    s6 = compute_bootstrap_stats(f6, d6, r6, 'Combined Adversarial', CAP)
    results.append(s6)

    # Scenario 7: Honest case (halve all PnLs — forward performance typically 50% of backtest)
    fprint(f"\n=== SCENARIO 7: HONEST FORWARD (50% of backtest) ===")
    pnls_half = pnls * 0.50
    f7, d7, r7 = block_bootstrap(pnls_half, n_paths=10000, n_periods=260, block_size=10, cap=CAP)
    s7 = compute_bootstrap_stats(f7, d7, r7, 'Honest Forward (50%)', CAP)
    results.append(s7)

    # Summary table
    fprint(f"\n{'='*110}")
    fprint(f"BOOTSTRAP STRESS TEST SUMMARY — 10K Paths, 5yr Horizon, ${CAP:.0f} Start")
    fprint(f"{'='*110}")
    fprint(f"{'Scenario':<28} {'Ruin%':>6} {'Med$':>8} {'P5$':>8} {'P25$':>8} {'P75$':>9} {'P95$':>9} {'MedDD':>7} {'%Prof':>6} {'%2x':>5} {'%5x':>5}")
    fprint("-"*110)
    for s in results:
        fprint(f"{s['name']:<28} {s['ruin_pct']:>5.1f}% ${s['median_final']:>7.0f} ${s['p5_final']:>7.0f} "
               f"${s['p25_final']:>7.0f} ${s['p75_final']:>8.0f} ${s['p95_final']:>8.0f} "
               f"{s['median_maxdd']:>6.1f}% {s['pct_profit']:>5.1f}% {s['pct_double']:>4.1f}% {s['pct_5x']:>4.1f}%")

    # Key findings
    base = results[0]
    honest = results[-1]
    combined = results[-2]
    fprint(f"\n=== KEY FINDINGS ===")
    fprint(f"BASE CASE: ${CAP:.0f} → ${base['median_final']:.0f} median (CAGR ~{base.get('median_cagr',0):.1f}%), "
           f"{base['ruin_pct']:.1f}% ruin, {base['pct_profit']:.0f}% profitable")
    fprint(f"HONEST FORWARD (50% of backtest): ${CAP:.0f} → ${honest['median_final']:.0f} median, "
           f"{honest['ruin_pct']:.1f}% ruin, {honest['pct_profit']:.0f}% profitable")
    fprint(f"WORST CASE (combined adversarial): ${CAP:.0f} → ${combined['median_final']:.0f} median, "
           f"{combined['ruin_pct']:.1f}% ruin")
    fprint(f"\nVERDICT: {'VIABLE' if honest['pct_profit'] > 60 and honest['ruin_pct'] < 20 else 'MARGINAL' if honest['pct_profit'] > 40 else 'RISKY'} "
           f"for live deployment (based on honest forward estimate)")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save
    save_data = {
        'timestamp': t0.isoformat(),
        'capital': CAP,
        'n_input_trades': len(pnls),
        'trade_stats': {
            'mean': round(float(pnls.mean()), 2),
            'median': round(float(np.median(pnls)), 2),
            'std': round(float(pnls.std()), 2),
            'win_rate': round(float((pnls > 0).mean() * 100), 1),
            'skew': round(float(pd.Series(pnls).skew()), 2),
            'kurtosis': round(float(pd.Series(pnls).kurtosis()), 2),
        },
        'scenarios': results,
        'runtime_s': round(elapsed, 1)
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow
    if MLFLOW_OK:
        try:
            en = 'bootstrap_stress_test_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"bootstrap_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'n_trades': len(pnls), 'n_paths': 10000, 'method': 'block_bootstrap'})
                for s in results:
                    pref = s['name'][:15].replace(' ','_').replace('(','').replace(')','')
                    mlflow.log_metrics({
                        f'{pref}_median': s['median_final'],
                        f'{pref}_ruin': s['ruin_pct'],
                        f'{pref}_pct_prof': s['pct_profit'],
                        f'{pref}_med_dd': s['median_maxdd'],
                    })
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}")
    fprint("DONE — Bootstrap Stress Test v1")
    fprint(f"{'='*80}")

if __name__ == '__main__':
    main()
