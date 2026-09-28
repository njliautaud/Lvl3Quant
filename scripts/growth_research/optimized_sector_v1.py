#!/usr/bin/env python3
"""Optimized Sector Momentum Bull Call Spread — v1

Sector-only universe (11 ETFs) with full adversarial validation.
Tests 6 configs varying DTE, spread width, VIX threshold, top-K, and lookback.

Uses HONEST equity-based Sharpe (v2 fix).
Full adversarial gates: permutation test, R1 regime check, sub-period stability, outlier removal.
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'optimized_sector_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable — will skip logging")

# Sector-only universe (11 ETFs)
UNIVERSE = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']

# 6 configs to test
CONFIGS = {
    'A_Baseline':     {'dte': 30, 'spread_pct': 3.0, 'top_k': 3, 'vix_min': 20, 'train_periods': 12},
    'B_OptDTE':       {'dte': 45, 'spread_pct': 3.0, 'top_k': 3, 'vix_min': 20, 'train_periods': 12},
    'C_OptDTE60':     {'dte': 60, 'spread_pct': 3.0, 'top_k': 3, 'vix_min': 20, 'train_periods': 12},
    'D_OptCombo':     {'dte': 45, 'spread_pct': 4.0, 'top_k': 3, 'vix_min': 20, 'train_periods': 12},
    'E_HighVIX':      {'dte': 45, 'spread_pct': 3.0, 'top_k': 3, 'vix_min': 25, 'train_periods': 12},
    'F_Conservative': {'dte': 45, 'spread_pct': 3.0, 'top_k': 2, 'vix_min': 20, 'train_periods': 18},
}


def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = list(set(UNIVERSE + ['SPY', '^VIX']))
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    avail = [c for c in UNIVERSE if c in close.columns and close[c].dropna().shape[0] > 500]
    sc = close[avail].dropna(how='all')
    sh = high[[c for c in avail if c in high.columns]].dropna(how='all')
    sl = low[[c for c in avail if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in avail if c in volume.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    fprint(f"Data: {len(ix)} days, {len(avail)} sector ETFs")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]


def compute_features(px, vol_data=None):
    if len(px) < 260: return None
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
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax(); mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    up = rets[rets > 0]; dn = rets[rets < 0]
    f['up_capture'] = float(up.iloc[-63:].mean()/(up.mean()+1e-10)) if len(up) > 10 else 1.0
    f['dn_capture'] = float(dn.iloc[-63:].mean()/(dn.mean()+1e-10)) if len(dn) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values+1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean()/(vol_data.iloc[-63:].mean()+1e-10)) if vol_data is not None and len(vol_data) >= 63 else 1.0
    return f


def build_rankings(sc, sv, rebal_dates, train_periods=12, fwd_days=14):
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if not feats: continue
            fi = min(idx+fwd_days, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in QM_COLS:
        if c not in df.columns: df[c] = 0.0
    df[QM_COLS] = df[QM_COLS].fillna(0.0)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(train_periods, len(dates)):
        td = dates[max(0,i-train_periods):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[QM_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[QM_COLS].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    return rankings


def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()


def atr_premium(S, K, dte, atr, vix_val):
    T = dte/252.0
    if T <= 0: return max(0, S-K)
    return max(0, S-K) + atr*np.sqrt(T)*max(0.3, vix_val/20.0)*np.exp(-3.0*abs(S-K)/S)


def simulate(rankings, sc, sh, sl, spy, vix, spread_pct=3.0, dte=30, top_k=3, vix_min=20):
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]
    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])
        if vix_min is not None and cv < vix_min: continue
        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_pos = min(200, equity/3)
        if max_pos < 30: eq_curve.append(equity); continue
        n_ent = 0
        for tk, _ in ranked:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt); ei = min(di+dte, len(sc)-1); Se = float(sc[tk].iloc[ei])
            K1, K2 = round(S), round(S*(1+spread_pct/100))
            lp = atr_premium(S, K1, dte, av, cv)*(1+HAIRCUT)
            sp = atr_premium(S, K2, dte, av, cv)*(1-HAIRCUT)
            val = lp-sp; cost = val*100+SPREAD_COMM; width = K2-K1
            mx_prof = (width-val)*100-SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue
            pnl = None
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci]); rd = max(0,dte-(ci-di))
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))
                si = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                if si-cost >= mx_prof*0.50 or rd < 7: pnl = si-cost; break
            if pnl is None:
                pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - val*100 - SPREAD_COMM
            equity += pnl; n_ent += 1
            trades.append({'pnl': pnl, 'win': pnl>0, 'regime': 'bull' if bull else 'bear',
                           'date': str(dt), 'ticker': tk})
        eq_curve.append(equity)
    return trades, equity, eq_curve


def compute_honest_monthly_returns(trades):
    """Compute honest equity-based monthly returns using CALENDAR months.

    Previous bug: used equal-chunk splitting (n//60) which artificially smooths
    variance and inflates Sharpe. Calendar month grouping is honest.
    """
    if not trades: return np.array([])
    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['date'])
    tdf['month'] = tdf['date'].dt.to_period('M')

    # Track equity at start of each month, compute returns honestly
    equity_track = CAP
    month_start_equity = {}
    monthly_pnl = {}
    current_month = None

    for _, row in tdf.iterrows():
        m = row['month']
        if m != current_month:
            month_start_equity[m] = equity_track
            current_month = m
            monthly_pnl[m] = 0
        monthly_pnl[m] += row['pnl']
        equity_track += row['pnl']

    months = sorted(monthly_pnl.keys())
    mr = np.array([monthly_pnl[m] / max(month_start_equity[m], 1.0) for m in months])
    return mr


def compute_sharpe_sortino(mr):
    """Compute honest Sharpe and Sortino from monthly returns."""
    if len(mr) < 4: return 0.0, 0.0
    sharpe = (np.mean(mr)*12) / (np.std(mr)*np.sqrt(12) + 1e-10)
    dr = mr[mr < 0]
    sortino = (np.mean(mr)*12) / (np.std(dr)*np.sqrt(12) + 1e-10) if len(dr) > 2 else sharpe
    return round(sharpe, 2), round(sortino, 2)


def permutation_test(mr, n_perms=500):
    """Permutation test: shuffle sign of returns, compute p-value."""
    if len(mr) < 5: return 1.0
    real_sharpe = np.mean(mr) / (np.std(mr) + 1e-10)
    count = 0
    for _ in range(n_perms):
        shuffled = mr * np.random.choice([-1, 1], len(mr))
        if np.mean(shuffled) / (np.std(shuffled) + 1e-10) >= real_sharpe:
            count += 1
    return count / n_perms


def r1_regime_check(trades):
    """R1 regime-agnostic check: |bull_WR - bear_WR| / max(bull_WR, bear_WR) < 0.50."""
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']
    if len(bull_trades) < 5 or len(bear_trades) < 5:
        # Not enough trades in one regime — flag but don't auto-fail
        return 0.99, False, len(bull_trades), len(bear_trades)
    bull_wr = sum(1 for t in bull_trades if t['win']) / len(bull_trades)
    bear_wr = sum(1 for t in bear_trades if t['win']) / len(bear_trades)
    gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1e-10)
    passed = gap < 0.50
    return round(gap, 3), passed, len(bull_trades), len(bear_trades)


def sub_period_stability(trades):
    """Split trades in half chronologically, check both halves profitable."""
    if len(trades) < 20: return False
    mid = len(trades) // 2
    first_half = trades[:mid]
    second_half = trades[mid:]
    pnl_1 = sum(t['pnl'] for t in first_half)
    pnl_2 = sum(t['pnl'] for t in second_half)
    return pnl_1 > 0 and pnl_2 > 0


def outlier_removal_test(trades):
    """Remove top 5% of wins, check if still profitable."""
    if len(trades) < 20: return False
    pnls = sorted([t['pnl'] for t in trades], reverse=True)
    n_remove = max(1, int(len(pnls) * 0.05))
    # Remove top n_remove wins
    wins_sorted = sorted([p for p in pnls if p > 0], reverse=True)
    if len(wins_sorted) < n_remove:
        return False
    removed_pnl = sum(wins_sorted[:n_remove])
    total_pnl = sum(pnls)
    remaining_pnl = total_pnl - removed_pnl
    return remaining_pnl > 0


def full_adversarial_validation(trades, eq_curve, final_eq):
    """Run all 4 adversarial gates. Returns dict with all metrics."""
    if not trades or len(trades) < 10:
        return None

    n = len(trades)
    wr = sum(1 for t in trades if t['win']) / n * 100
    pnls = [t['pnl'] for t in trades]

    # Honest monthly returns
    mr = compute_honest_monthly_returns(trades)
    ny = max(len(mr)/12, 0.5)
    sharpe, sortino = compute_sharpe_sortino(mr)
    cagr = (final_eq / CAP) ** (1/ny) - 1

    # Equity curve stats
    eq = np.array(eq_curve)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / (pk + 1e-10)).min())

    # Profit factor
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / (gl + 1e-10)

    # Gate 1: Permutation test (500 shuffles)
    perm_p = permutation_test(mr, n_perms=500)
    perm_pass = perm_p < 0.05

    # Gate 2: R1 regime-agnostic check
    r1_gap, r1_pass, n_bull, n_bear = r1_regime_check(trades)

    # Gate 3: Sub-period stability
    sp_pass = sub_period_stability(trades)

    # Gate 4: Outlier removal
    outlier_pass = outlier_removal_test(trades)

    # Overall: pass all 4 gates
    overall_pass = perm_pass and r1_pass and sp_pass and outlier_pass
    gates_passed = sum([perm_pass, r1_pass, sp_pass, outlier_pass])

    return {
        'n_trades': n,
        'wr': round(wr, 1),
        'honest_sharpe': sharpe,
        'sortino': sortino,
        'cagr': round(cagr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'pf': round(pf, 2),
        'final_eq': round(final_eq, 0),
        'perm_p': round(perm_p, 3),
        'perm_pass': perm_pass,
        'r1_gap': r1_gap,
        'r1_pass': r1_pass,
        'n_bull': n_bull,
        'n_bear': n_bear,
        'sub_period_pass': sp_pass,
        'outlier_pass': outlier_pass,
        'gates_passed': f"{gates_passed}/4",
        'overall_pass': overall_pass,
    }


def main():
    t0 = datetime.now()
    fprint(f"Optimized Sector Momentum v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*90}")
    fprint(f"Universe: {len(UNIVERSE)} sector ETFs | Capital: ${CAP} | Spread comm: ${SPREAD_COMM}")
    fprint(f"Configs: {len(CONFIGS)} | Adversarial gates: permutation, R1 regime, sub-period, outlier")
    fprint(f"{'='*90}")

    sc, sh, sl, sv, spy, vix = download_data()

    # Pre-build rankings for different lookback periods
    fprint("\nBuilding rankings...")
    bd_bw = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)

    # We need lookback 12 and 18
    fprint("  Bi-weekly, 12-period lookback...")
    rank_bw_12 = build_rankings(sc, sv, bd_bw, train_periods=12)
    fprint(f"    {len(rank_bw_12)} ranking dates")

    fprint("  Bi-weekly, 18-period lookback...")
    rank_bw_18 = build_rankings(sc, sv, bd_bw, train_periods=18)
    fprint(f"    {len(rank_bw_18)} ranking dates")

    # Run each config
    results = {}
    fprint(f"\n{'='*90}")
    fprint(f"{'Config':<18} {'#Tr':>4} {'WR%':>5} {'Sharpe':>7} {'Sort':>6} {'CAGR%':>6} {'MDD%':>6} {'PF':>5} {'Perm':>5} {'R1':>5} {'SubP':>5} {'Outl':>5} {'Gate':>5}")
    fprint(f"{'-'*90}")

    for name, cfg in CONFIGS.items():
        # Select rankings based on train_periods
        if cfg['train_periods'] == 18:
            rankings = rank_bw_18
        else:
            rankings = rank_bw_12

        trades, final_eq, eq_curve = simulate(
            rankings, sc, sh, sl, spy, vix,
            spread_pct=cfg['spread_pct'],
            dte=cfg['dte'],
            top_k=cfg['top_k'],
            vix_min=cfg['vix_min']
        )

        metrics = full_adversarial_validation(trades, eq_curve, final_eq)
        if metrics is None:
            fprint(f"  {name:<18} INSUFFICIENT TRADES")
            results[name] = {'config': cfg, 'metrics': None, 'error': 'insufficient_trades'}
            continue

        results[name] = {'config': cfg, 'metrics': metrics}

        # Print row
        pp = 'PASS' if metrics['perm_pass'] else 'FAIL'
        r1 = 'PASS' if metrics['r1_pass'] else 'FAIL'
        sp = 'PASS' if metrics['sub_period_pass'] else 'FAIL'
        ol = 'PASS' if metrics['outlier_pass'] else 'FAIL'
        ov = metrics['gates_passed']
        fprint(f"  {name:<18} {metrics['n_trades']:>4} {metrics['wr']:>5.1f} {metrics['honest_sharpe']:>7.2f} {metrics['sortino']:>6.2f} {metrics['cagr']:>6.1f} {metrics['mdd']:>6.1f} {metrics['pf']:>5.2f} {pp:>5} {r1:>5} {sp:>5} {ol:>5} {ov:>5}")

    # Summary
    fprint(f"\n{'='*90}")
    fprint("SUMMARY")
    fprint(f"{'='*90}")

    valid = {k: v for k, v in results.items() if v.get('metrics') is not None}
    if not valid:
        fprint("No configs produced enough trades. Check VIX/data filters.")
        return

    n_overall_pass = sum(1 for v in valid.values() if v['metrics']['overall_pass'])
    sharpes = [v['metrics']['honest_sharpe'] for v in valid.values()]
    sortinos = [v['metrics']['sortino'] for v in valid.values()]

    fprint(f"Configs tested: {len(valid)}")
    fprint(f"Full pass (4/4 gates): {n_overall_pass}/{len(valid)}")
    fprint(f"Honest Sharpe range: {min(sharpes):.2f} - {max(sharpes):.2f}")
    fprint(f"Honest Sharpe mean: {np.mean(sharpes):.2f} +/- {np.std(sharpes):.2f}")
    fprint(f"Sortino range: {min(sortinos):.2f} - {max(sortinos):.2f}")

    # Best config
    best_name = max(valid.keys(), key=lambda k: valid[k]['metrics']['honest_sharpe'])
    best = valid[best_name]['metrics']
    fprint(f"\nBest by Sharpe: {best_name}")
    fprint(f"  Sharpe={best['honest_sharpe']}, Sortino={best['sortino']}, WR={best['wr']}%, CAGR={best['cagr']}%, MDD={best['mdd']}%")
    fprint(f"  PF={best['pf']}, Perm p={best['perm_p']}, R1 gap={best['r1_gap']}, Gates={best['gates_passed']}")

    # Best that passes all gates
    passing = {k: v for k, v in valid.items() if v['metrics']['overall_pass']}
    if passing:
        best_pass = max(passing.keys(), key=lambda k: passing[k]['metrics']['honest_sharpe'])
        bp = passing[best_pass]['metrics']
        fprint(f"\nBest PASSING config: {best_pass}")
        fprint(f"  Sharpe={bp['honest_sharpe']}, Sortino={bp['sortino']}, WR={bp['wr']}%, CAGR={bp['cagr']}%, MDD={bp['mdd']}%")
    else:
        fprint("\nNo config passed all 4 adversarial gates.")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save results
    save_data = {
        'timestamp': t0.isoformat(),
        'universe': UNIVERSE,
        'capital': CAP,
        'spread_comm': SPREAD_COMM,
        'haircut': HAIRCUT,
        'configs': {k: v['config'] for k, v in results.items()},
        'results': {k: v['metrics'] for k, v in results.items() if v.get('metrics')},
        'summary': {
            'n_configs': len(valid),
            'n_full_pass': n_overall_pass,
            'sharpe_mean': round(float(np.mean(sharpes)), 2),
            'sharpe_std': round(float(np.std(sharpes)), 2),
            'sharpe_min': round(float(min(sharpes)), 2),
            'sharpe_max': round(float(max(sharpes)), 2),
            'best_config': best_name,
            'best_passing_config': best_pass if passing else None,
        },
        'runtime_s': round(elapsed, 1),
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            en = 'optimized_sector_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"sector_opt_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'universe': ','.join(UNIVERSE),
                    'n_configs': len(CONFIGS),
                    'capital': CAP,
                    'spread_comm': SPREAD_COMM,
                })
                mlflow.log_metrics({
                    'n_full_pass': n_overall_pass,
                    'sharpe_mean': float(np.mean(sharpes)),
                    'sharpe_std': float(np.std(sharpes)),
                    'sharpe_min': float(min(sharpes)),
                    'sharpe_max': float(max(sharpes)),
                })
                # Log per-config metrics
                for name, v in valid.items():
                    m = v['metrics']
                    safe_name = name.lower()
                    mlflow.log_metrics({
                        f'{safe_name}_sharpe': m['honest_sharpe'],
                        f'{safe_name}_sortino': m['sortino'],
                        f'{safe_name}_wr': m['wr'],
                        f'{safe_name}_cagr': m['cagr'],
                        f'{safe_name}_mdd': m['mdd'],
                        f'{safe_name}_pf': m['pf'],
                        f'{safe_name}_perm_p': m['perm_p'],
                        f'{safe_name}_r1_gap': m['r1_gap'],
                    })
                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow: logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint(f"\n{'='*90}\nDONE — Optimized Sector Momentum v1\n{'='*90}")


if __name__ == '__main__':
    main()
