#!/usr/bin/env python3
"""Lead-Lag Permutation Fix — Test if lead-lag selection beats RANDOM sector selection.

The original permutation test was broken (disabling lead-lag = zero trades = p=0.000 guaranteed).
This fix properly tests: does lead-lag selection beat picking random sectors each period?
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_PATH = RESULTS_DIR / 'sector_leadlag_perm_fix_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

def download_data():
    import yfinance as yf
    fprint("Downloading data...")
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
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

def compute_leadlag_matrix(returns, lookback=252, lag_days=10):
    n = len(returns.columns)
    result = pd.DataFrame(0.0, index=returns.columns, columns=returns.columns)
    for leader in returns.columns:
        for follower in returns.columns:
            if leader == follower: continue
            leader_rets = returns[leader].iloc[-lookback:-lag_days]
            follower_rets = returns[follower].iloc[-lookback+lag_days:]
            min_len = min(len(leader_rets), len(follower_rets))
            if min_len < 30: continue
            leader_rets = leader_rets.iloc[:min_len]
            follower_rets = follower_rets.iloc[:min_len]
            corr, _ = stats.pearsonr(leader_rets.values, follower_rets.values)
            result.loc[leader, follower] = corr
    return result

def get_leadlag_picks(sc, dt, top_k=3, threshold=0.10, min_leaders=1, mode='bull'):
    """Get lead-lag based picks for a given date."""
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 280: return []
    hist = sc.iloc[:idx+1]
    rets_5d = hist.pct_change(5).dropna()
    if len(rets_5d) < 252: return []

    ll_matrix = compute_leadlag_matrix(rets_5d, lookback=252, lag_days=2)

    candidates = []
    for follower in sc.columns:
        leaders = []
        for leader in sc.columns:
            if leader == follower: continue
            corr = ll_matrix.loc[leader, follower]
            if abs(corr) >= threshold:
                leader_recent = float(hist[leader].iloc[-1] / hist[leader].iloc[-5] - 1)
                if abs(leader_recent) > 0.005:
                    predicted = 'bull' if (corr > 0 and leader_recent > 0) or (corr < 0 and leader_recent < 0) else 'bear'
                    leaders.append({'dir': predicted, 'corr': abs(corr)})

        if not leaders: continue
        bull_count = sum(1 for l in leaders if l['dir'] == 'bull')
        bear_count = sum(1 for l in leaders if l['dir'] == 'bear')
        n_agree = max(bull_count, bear_count)

        if n_agree < min_leaders: continue

        direction = 'bull' if bull_count > bear_count else 'bear'
        if direction != mode: continue

        avg_corr = np.mean([l['corr'] for l in leaders])
        score = n_agree / len(leaders) * avg_corr
        candidates.append((follower, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in candidates[:top_k]]

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

def simulate_with_picks(picks_by_date, sc, sh, sl, spy, vix, mode='bull',
                         spread_pct=3.0, dte=30, early_exit_day=20):
    """Simulate given pre-determined picks per date."""
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt, picks in sorted(picks_by_date.items()):
        if dt not in vix.index: continue
        cv = float(vix.loc[dt])
        max_pos = min(200, equity/3)
        if max_pos < 30: continue

        for tk in picks:
            if tk not in sc.columns or tk not in atr_d: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)
            ei = min(di + early_exit_day, len(sc)-1)

            if mode == 'bull':
                K1, K2 = round(S), round(S*(1+spread_pct/100))
                lp = atr_premium(S, K1, dte, av, cv, 'call')*(1+HAIRCUT)
                sp = atr_premium(S, K2, dte, av, cv, 'call')*(1-HAIRCUT)
            else:
                K1, K2 = round(S*(1-spread_pct/100)), round(S)
                lp = atr_premium(S, K2, dte, av, cv, 'put')*(1+HAIRCUT)
                sp = atr_premium(S, K1, dte, av, cv, 'put')*(1-HAIRCUT)

            val = lp - sp; width = K2 - K1
            cost = val*100 + SPREAD_COMM; mx_prof = (width-val)*100 - SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

            pnl = None
            for ci in range(di+1, ei+1):
                if ci >= len(sc): break
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di; rd = max(0, dte - dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))
                if mode == 'bull':
                    intrinsic = (max(0, Sc-K1) - max(0, Sc-K2)) * 100
                else:
                    intrinsic = (max(0, K2-Sc) - max(0, K1-Sc)) * 100
                current_val = intrinsic + ac * tm * 0.3 * 100
                cp = current_val - cost
                if cp >= mx_prof * 0.50: pnl = cp; break
                if ci == ei: pnl = cp; break

            if pnl is None:
                Se = float(sc[tk].iloc[ei])
                if mode == 'bull':
                    pnl = (max(0, Se-K1) - max(0, Se-K2))*100 - cost
                else:
                    pnl = (max(0, K2-Se) - max(0, K1-Se))*100 - cost

            equity += pnl
            trades.append({'date': str(dt.date()), 'ticker': tk, 'pnl': round(pnl,2), 'equity': round(equity,2)})
            eq_curve.append(equity)

    return equity, trades, eq_curve

def compute_metrics(trades, eq_curve):
    if not trades: return {'sharpe': 0, 'cagr': 0, 'mdd': -1, 'wr': 0, 'pf': 0, 'sortino': 0, 'n_trades': 0, 'final_equity': CAP}
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    eq_at_month = df.groupby('month')['equity'].last()
    monthly_ret = monthly / (eq_at_month.shift(1).fillna(CAP))
    sharpe = float(monthly_ret.mean() / (monthly_ret.std()+1e-10) * np.sqrt(12))
    downside = monthly_ret[monthly_ret < 0]
    sortino = float(monthly_ret.mean() / (downside.std()+1e-10) * np.sqrt(12)) if len(downside) > 1 else sharpe
    final_eq = eq_curve[-1]
    # Calendar-based CAGR
    first_date = pd.to_datetime(df['date'].iloc[0])
    last_date = pd.to_datetime(df['date'].iloc[-1])
    years = (last_date - first_date).days / 365.25
    cagr = (final_eq/CAP)**(1/max(years,0.5)) - 1 if final_eq > 0 else -1
    peak = pd.Series(eq_curve).cummax()
    dd = (pd.Series(eq_curve) - peak) / peak
    mdd = float(dd.min())
    wins = df[df['pnl'] > 0]; losses = df[df['pnl'] <= 0]
    wr = len(wins) / len(df) if len(df) > 0 else 0
    pf = abs(wins['pnl'].sum()) / (abs(losses['pnl'].sum()) + 1e-10) if len(losses) > 0 else 99
    return {'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
            'cagr': round(cagr*100, 1), 'mdd': round(mdd*100, 1),
            'wr': round(wr*100, 1), 'pf': round(pf, 2),
            'n_trades': len(df), 'final_equity': round(final_eq, 0)}

def regime_r1_check(trades, spy):
    if not trades: return 1.0, 0, 0
    df = pd.DataFrame(trades); df['date'] = pd.to_datetime(df['date'])
    spy_monthly = spy.resample('ME').last().pct_change()
    bull_pnl, bear_pnl = [], []
    for _, row in df.iterrows():
        dt = row['date']
        closest = spy_monthly.index[spy_monthly.index.get_indexer([dt], method='ffill')[0]]
        if spy_monthly.loc[closest] >= 0: bull_pnl.append(row['pnl'])
        else: bear_pnl.append(row['pnl'])
    if not bull_pnl or not bear_pnl: return 1.0, 0, 0
    bull_sr = np.mean(bull_pnl) / (np.std(bull_pnl) + 1e-10)
    bear_sr = np.mean(bear_pnl) / (np.std(bear_pnl) + 1e-10)
    gap = abs(bull_sr - bear_sr) / (max(abs(bull_sr), abs(bear_sr)) + 1e-10)
    return round(gap, 3), round(bull_sr, 3), round(bear_sr, 3)

def subperiod_check(trades):
    if len(trades) < 20: return False, 0, 0
    df = pd.DataFrame(trades); mid = len(df) // 2
    h1_wr = (df.iloc[:mid]['pnl'] > 0).mean()
    h2_wr = (df.iloc[mid:]['pnl'] > 0).mean()
    return abs(h1_wr - h2_wr) < 0.15, round(h1_wr, 3), round(h2_wr, 3)

def outlier_check(trades):
    if len(trades) < 20: return False, 0
    df = pd.DataFrame(trades)
    thresh = df['pnl'].quantile(0.95)
    trimmed = df[df['pnl'] <= thresh]
    trimmed_pf = trimmed[trimmed['pnl'] > 0]['pnl'].sum() / (abs(trimmed[trimmed['pnl'] <= 0]['pnl'].sum()) + 1e-10)
    return trimmed_pf > 1.0, round(trimmed_pf, 2)

def main():
    fprint("=" * 70)
    fprint("SECTOR LEAD-LAG — PROPER PERMUTATION TEST")
    fprint("=" * 70)

    sc, sh, sl, spy, vix = download_data()
    all_dates = sc.index[sc.index >= '2010-01-01']
    rebal_dates = list(all_dates[::10])
    fprint(f"Rebalance dates: {len(rebal_dates)}")

    # Test configurations: best variants from initial run
    configs = {
        'C_BullBear': {'threshold': 0.10, 'min_leaders': 1},
        'E_MultiLeader': {'threshold': 0.08, 'min_leaders': 2},
        'F_HighThreshold': {'threshold': 0.15, 'min_leaders': 1},
    }

    results = {}
    for cname, cfg in configs.items():
        fprint(f"\n{'='*60}")
        fprint(f"Config {cname}: thresh={cfg['threshold']}, min_leaders={cfg['min_leaders']}")
        fprint(f"{'='*60}")

        # Build lead-lag picks for bull (VIX>=20) and bear (VIX<20)
        bull_picks, bear_picks = {}, {}
        fprint("  Computing lead-lag signals for each date...")
        for dt in rebal_dates:
            if dt not in vix.index: continue
            cv = float(vix.loc[dt])
            if cv >= 20:
                picks = get_leadlag_picks(sc, dt, top_k=3,
                    threshold=cfg['threshold'], min_leaders=cfg['min_leaders'], mode='bull')
                if picks: bull_picks[dt] = picks
            else:
                picks = get_leadlag_picks(sc, dt, top_k=3,
                    threshold=cfg['threshold'], min_leaders=cfg['min_leaders'], mode='bear')
                if picks: bear_picks[dt] = picks

        fprint(f"  Bull dates with picks: {len(bull_picks)}, Bear dates with picks: {len(bear_picks)}")

        # Real simulation
        eq_bull, trades_bull, curve_bull = simulate_with_picks(bull_picks, sc, sh, sl, spy, vix, mode='bull')
        eq_bear, trades_bear, curve_bear = simulate_with_picks(bear_picks, sc, sh, sl, spy, vix, mode='bear')

        all_trades = trades_bull + trades_bear
        all_trades.sort(key=lambda x: x['date'])
        combined_eq = CAP
        combined_curve = [CAP]
        for t in all_trades:
            combined_eq += t['pnl']
            t['equity'] = round(combined_eq, 2)
            combined_curve.append(combined_eq)

        metrics = compute_metrics(all_trades, combined_curve)
        fprint(f"  REAL: {metrics['n_trades']} trades, Sharpe {metrics['sharpe']}, "
               f"CAGR {metrics['cagr']}%, MDD {metrics['mdd']}%, WR {metrics['wr']}%, "
               f"${metrics['final_equity']}")

        # PROPER PERMUTATION: Random sector selection, same number of trades per date
        fprint(f"  Running PROPER permutation (200 trials, random sectors)...")
        n_perms = 200
        rand_equities = []
        rand_sharpes = []
        sectors_list = list(sc.columns)

        for p in range(n_perms):
            # Create random picks: same dates, same number of picks, random sectors
            rand_bull = {dt: list(np.random.choice(sectors_list, size=min(len(picks), len(sectors_list)), replace=False))
                         for dt, picks in bull_picks.items()}
            rand_bear = {dt: list(np.random.choice(sectors_list, size=min(len(picks), len(sectors_list)), replace=False))
                         for dt, picks in bear_picks.items()}

            eq_b, trd_b, _ = simulate_with_picks(rand_bull, sc, sh, sl, spy, vix, mode='bull')
            eq_br, trd_br, _ = simulate_with_picks(rand_bear, sc, sh, sl, spy, vix, mode='bear')

            all_r = trd_b + trd_br
            all_r.sort(key=lambda x: x['date'])
            req = CAP
            rcurve = [CAP]
            for t in all_r:
                req += t['pnl']
                rcurve.append(req)

            rm = compute_metrics(all_r, rcurve)
            rand_equities.append(req)
            rand_sharpes.append(rm['sharpe'])

        p_equity = np.mean([re >= combined_curve[-1] for re in rand_equities])
        p_sharpe = np.mean([rs >= metrics['sharpe'] for rs in rand_sharpes])
        rand_eq_mean = np.mean(rand_equities)
        rand_sh_mean = np.mean(rand_sharpes)

        fprint(f"  RANDOM: avg equity ${rand_eq_mean:.0f}, avg Sharpe {rand_sh_mean:.2f}")
        fprint(f"  p-value (equity): {p_equity:.3f}, p-value (Sharpe): {p_sharpe:.3f}")

        perm_pass = p_sharpe < 0.05
        improvement = (metrics['sharpe'] - rand_sh_mean) / (rand_sh_mean + 1e-10) * 100
        fprint(f"  Lead-lag Sharpe improvement vs random: {improvement:+.1f}%")
        fprint(f"  Perm PROPERLY: {'PASS' if perm_pass else 'FAIL'}")

        # Other gates
        r1_gap, bull_sr, bear_sr = regime_r1_check(all_trades, spy)
        r1_pass = r1_gap < 0.50
        sp_ok, h1_wr, h2_wr = subperiod_check(all_trades)
        out_ok, trimmed_pf = outlier_check(all_trades)

        fprint(f"  R1: gap={r1_gap} → {'PASS' if r1_pass else 'FAIL'}")
        fprint(f"  SubPeriod: {h1_wr}/{h2_wr} → {'PASS' if sp_ok else 'FAIL'}")
        fprint(f"  Outlier: PF={trimmed_pf} → {'PASS' if out_ok else 'FAIL'}")

        gates = sum([perm_pass, r1_pass, sp_ok, out_ok])
        fprint(f"  GATES (corrected): {gates}/4 {'✅' if gates == 4 else '❌'}")

        results[cname] = {
            'metrics': metrics,
            'perm_p_equity': round(p_equity, 3),
            'perm_p_sharpe': round(p_sharpe, 3),
            'rand_eq_mean': round(rand_eq_mean, 0),
            'rand_sharpe_mean': round(rand_sh_mean, 2),
            'sharpe_improvement': round(improvement, 1),
            'r1_gap': r1_gap,
            'gates': gates,
            'gates_detail': {'perm': perm_pass, 'r1': r1_pass, 'subperiod': sp_ok, 'outlier': out_ok},
        }

        if MLFLOW_OK:
            with mlflow.start_run(run_name=f"leadlag_permfix_{cname}"):
                mlflow.log_metrics({
                    'sharpe': metrics['sharpe'], 'cagr': metrics['cagr'],
                    'mdd': metrics['mdd'], 'wr': metrics['wr'],
                    'n_trades': metrics['n_trades'], 'final_equity': metrics['final_equity'],
                    'perm_p_sharpe': p_sharpe, 'rand_sharpe': rand_sh_mean,
                    'sharpe_improvement_pct': improvement,
                    'gates_passed': gates, 'r1_gap': r1_gap,
                })

    # Summary
    fprint(f"\n{'='*70}")
    fprint("CORRECTED SUMMARY — SECTOR LEAD-LAG vs RANDOM SELECTION")
    fprint(f"{'='*70}")
    fprint(f"{'Config':<20} {'Sharpe':>7} {'RandSh':>7} {'Impr%':>7} {'p_sh':>6} {'CAGR':>7} {'Gates':>6}")
    fprint("-" * 62)
    for cn, r in results.items():
        m = r['metrics']
        fprint(f"{cn:<20} {m['sharpe']:>7.2f} {r['rand_sharpe_mean']:>7.2f} {r['sharpe_improvement']:>6.1f}% {r['perm_p_sharpe']:>6.3f} {m['cagr']:>6.1f}% {r['gates']:>4}/4")

    # Verdict
    any_pass_perm = any(r['perm_p_sharpe'] < 0.05 for r in results.values())
    if any_pass_perm:
        best = max((r for r in results.values() if r['perm_p_sharpe'] < 0.05),
                   key=lambda r: r['metrics']['sharpe'])
        fprint(f"\nVERDICT: Lead-lag has GENUINE edge over random (p<0.05)")
        fprint(f"  Improvement: {best['sharpe_improvement']:+.1f}% Sharpe over random sectors")
    else:
        fprint(f"\nVERDICT: Lead-lag does NOT beat random sector selection (p>0.05)")
        fprint(f"  Edge is STRUCTURAL (any sector spread works) not from lead-lag signal")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    fprint("\nDONE")

if __name__ == '__main__':
    main()
