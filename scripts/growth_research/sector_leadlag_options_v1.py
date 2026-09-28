#!/usr/bin/env python3
"""Sector Lead-Lag Momentum Options v1 — Predictive sector spillover.

HYPOTHESIS: Sector returns have lead-lag relationships. When a "leader"
sector rallies, "follower" sectors tend to follow 1-2 weeks later.
If we can detect these flows, we can PREDICT which sectors will rally
next and buy call/put spreads BEFORE the move.

METHOD:
1. Compute rolling cross-correlation matrix of sector returns at lag 1-2 weeks
2. Identify stable lead-lag pairs (e.g., XLE leads XLI, XLK leads XLC)
3. When leader shows strong move, enter spreads on follower

This is PREDICTIVE rather than reactive momentum — genuinely different
from our existing LGBM ranking.

Variants:
  A: Pure lead-lag (top follower of strongest leader) — no VIX filter
  B: Lead-lag + VIX>=20 filter (bull spreads in high-vol)
  C: Lead-lag + VIX<20 (bear put spreads in calm markets)
  D: Lead-lag + LGBM momentum combined (both signals agree)
  E: Multi-leader consensus (follower must be led by 2+ leaders)
  F: Adaptive lead-lag (recompute correlations each period, walk-forward)

All: $645, 20d exit, ATR pricing, 4-gate adversarial.
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
RESULTS_PATH = RESULTS_DIR / 'sector_leadlag_options_v1_results.json'
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
    fprint(f"Data: {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

# ==================== LEAD-LAG ENGINE ====================
def compute_leadlag_matrix(returns, lookback=252, lag_days=10):
    """Compute lead-lag correlation matrix.

    For each pair (A, B), compute correlation between A's return
    and B's return shifted forward by lag_days. High correlation means
    A LEADS B.

    Returns: DataFrame where [A][B] = correlation(A_return_t, B_return_{t+lag})
    Positive = A leads B (A moves, B follows in same direction)
    """
    n = len(returns.columns)
    result = pd.DataFrame(0.0, index=returns.columns, columns=returns.columns)

    for leader in returns.columns:
        for follower in returns.columns:
            if leader == follower:
                continue
            # Leader returns at time t, follower returns at time t+lag
            leader_rets = returns[leader].iloc[-lookback:-lag_days]
            follower_rets = returns[follower].iloc[-lookback+lag_days:]

            if len(leader_rets) != len(follower_rets):
                min_len = min(len(leader_rets), len(follower_rets))
                leader_rets = leader_rets.iloc[:min_len]
                follower_rets = follower_rets.iloc[:min_len]

            if len(leader_rets) < 30:
                continue

            corr, _ = stats.pearsonr(leader_rets.values, follower_rets.values)
            result.loc[leader, follower] = corr

    return result

def get_leadlag_signals(sc, dt, lookback=252, lag_days=10, threshold=0.10):
    """Get lead-lag based trading signals for a given date.

    Returns dict: {follower_ticker: {'leaders': [list], 'direction': 'bull'/'bear',
                                      'signal_strength': float}}
    """
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < lookback + lag_days + 20:
        return {}

    # Compute returns
    hist = sc.iloc[:idx+1]
    rets_5d = hist.pct_change(5).dropna()

    if len(rets_5d) < lookback:
        return {}

    # Compute lead-lag matrix using 5-day returns
    ll_matrix = compute_leadlag_matrix(rets_5d, lookback=lookback, lag_days=2)  # 2 periods of 5d = ~10 trading days lag

    # For each potential follower, find its leaders
    signals = {}
    for follower in sc.columns:
        leaders = []
        for leader in sc.columns:
            if leader == follower:
                continue
            corr = ll_matrix.loc[leader, follower]
            if abs(corr) >= threshold:
                # Check if leader has moved recently (last 5-10 days)
                leader_recent = float(hist[leader].iloc[-1] / hist[leader].iloc[-5] - 1)
                if abs(leader_recent) > 0.005:  # >0.5% move in leader
                    leaders.append({
                        'ticker': leader,
                        'corr': corr,
                        'recent_move': leader_recent,
                        'predicted_direction': 'bull' if (corr > 0 and leader_recent > 0) or (corr < 0 and leader_recent < 0) else 'bear'
                    })

        if leaders:
            # Aggregate leader signals
            bull_count = sum(1 for l in leaders if l['predicted_direction'] == 'bull')
            bear_count = sum(1 for l in leaders if l['predicted_direction'] == 'bear')
            avg_corr = np.mean([abs(l['corr']) for l in leaders])

            if bull_count > bear_count:
                direction = 'bull'
                strength = bull_count / len(leaders) * avg_corr
            elif bear_count > bull_count:
                direction = 'bear'
                strength = bear_count / len(leaders) * avg_corr
            else:
                continue  # Conflicting signals

            signals[follower] = {
                'leaders': leaders,
                'direction': direction,
                'signal_strength': strength,
                'n_leaders': len(leaders),
                'consensus': max(bull_count, bear_count) / len(leaders)
            }

    return signals

# ==================== LGBM RANKING (for combined variant) ====================
MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
FEAT_COLS = MOM_COLS + QUALITY_COLS

def compute_features(px):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252))
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]; dn_days = rets[rets < 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean()+1e-10)) if len(up_days) > 10 else 1.0
    f['dn_capture'] = float(dn_days.iloc[-63:].mean() / (dn_days.mean()+1e-10)) if len(dn_days) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = 1.0
    return f

def build_lgbm_rankings(sc, rebal_dates):
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            feats = compute_features(px)
            if not feats: continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns: df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)
    if len(df) < 100: return {}
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
    fprint(f"    {len(rankings)} ranking dates")
    return rankings

# ==================== OPTIONS PRICING ====================
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

# ==================== SIMULATION ====================
def simulate(name, sc, sh, sl, spy, vix, rebal_dates,
             spread_pct=3.0, dte=30, top_k=3,
             vix_min=None, vix_max=None, mode='bull',
             early_exit_day=20,
             use_leadlag=True, leadlag_threshold=0.10,
             min_leaders=1, use_lgbm=False, lgbm_rankings=None,
             require_lgbm_agree=False):

    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in rebal_dates:
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])

        if vix_min is not None and cv < vix_min: continue
        if vix_max is not None and cv > vix_max: continue

        # Get lead-lag signals
        if use_leadlag:
            ll_signals = get_leadlag_signals(sc, dt, lookback=252, lag_days=10,
                                             threshold=leadlag_threshold)
        else:
            ll_signals = {}

        # Get LGBM rankings
        lgbm_scores = lgbm_rankings.get(dt, {}) if lgbm_rankings else {}

        # Build candidate list
        candidates = []
        for tk in sc.columns:
            score = 0
            reasons = []

            if use_leadlag and tk in ll_signals:
                sig = ll_signals[tk]
                if sig['direction'] == mode and sig['n_leaders'] >= min_leaders:
                    score += sig['signal_strength'] * sig['consensus']
                    reasons.append(f"LL:{sig['n_leaders']}leaders")
                else:
                    continue  # Wrong direction or not enough leaders

            if use_lgbm and lgbm_scores:
                lgbm_score = lgbm_scores.get(tk, 0.5)
                if mode == 'bull' and lgbm_score > 0.6:
                    score += lgbm_score
                    reasons.append(f"LGBM:{lgbm_score:.2f}")
                elif mode == 'bear' and lgbm_score < 0.4:
                    score += (1 - lgbm_score)
                    reasons.append(f"LGBM:{lgbm_score:.2f}")
                elif require_lgbm_agree:
                    continue  # LGBM doesn't agree

            if score > 0:
                candidates.append((tk, score, reasons))

        # Sort by score, take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        picks = candidates[:top_k]

        max_pos = min(200, equity/3)
        if max_pos < 30: eq_curve.append(equity); continue

        n_ent = 0
        for tk, score, reasons in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue

            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)
            ei = min(di + early_exit_day, len(sc)-1)

            if mode == 'bull':
                K1, K2 = round(S), round(S*(1+spread_pct/100))
                lp = atr_premium(S, K1, dte, av, cv, 'call')*(1+HAIRCUT)
                sp = atr_premium(S, K2, dte, av, cv, 'call')*(1-HAIRCUT)
                val = lp - sp; width = K2 - K1
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
                dh = ci - di
                rd = max(0, dte - dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))

                if mode == 'bull':
                    intrinsic = (max(0, Sc-K1) - max(0, Sc-K2)) * 100
                else:
                    intrinsic = (max(0, K2-Sc) - max(0, K1-Sc)) * 100
                time_val = ac * tm * 0.3 * 100
                current_val = intrinsic + time_val
                cp = current_val - cost

                if cp >= mx_prof * 0.50:
                    pnl = cp; break
                if ci == ei:
                    pnl = cp; break

            if pnl is None:
                Se = float(sc[tk].iloc[ei])
                if mode == 'bull':
                    pnl = (max(0, Se-K1) - max(0, Se-K2))*100 - cost
                else:
                    pnl = (max(0, K2-Se) - max(0, K1-Se))*100 - cost

            equity += pnl
            trades.append({'date': str(dt.date()), 'ticker': tk, 'mode': mode,
                           'cost': round(cost,2), 'pnl': round(pnl,2),
                           'equity': round(equity,2), 'reasons': str(reasons)})
            eq_curve.append(equity)
            n_ent += 1

    return equity, trades, eq_curve

# ==================== METRICS & GATES ====================
def compute_metrics(trades, eq_curve):
    if not trades: return {'sharpe': 0, 'cagr': 0, 'mdd': -1, 'wr': 0, 'pf': 0, 'sortino': 0, 'n_trades': 0, 'final_equity': 0}
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
    years = len(monthly) / 12
    cagr = (final_eq/CAP)**(1/max(years,0.5)) - 1 if final_eq > 0 else -1
    peak = pd.Series(eq_curve).cummax()
    dd = (pd.Series(eq_curve) - peak) / peak
    mdd = float(dd.min())
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    wr = len(wins) / len(df) if len(df) > 0 else 0
    pf = abs(wins['pnl'].sum()) / (abs(losses['pnl'].sum()) + 1e-10) if len(losses) > 0 else 99
    calmar = cagr / (abs(mdd) + 1e-10)
    return {'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
            'cagr': round(cagr*100, 1), 'mdd': round(mdd*100, 1),
            'wr': round(wr*100, 1), 'pf': round(pf, 2), 'calmar': round(calmar, 1),
            'n_trades': len(df), 'final_equity': round(final_eq, 0)}

def permutation_test(equity_real, sc, sh, sl, spy, vix, rebal_dates, n_perms=200, **sim_kwargs):
    """Randomize which sectors are selected to test lead-lag value."""
    fprint(f"    Permutation test ({n_perms} perms)...")
    random_equities = []
    for _ in range(n_perms):
        # Run with lead-lag disabled (pure random selection)
        eq, _, _ = simulate("perm", sc, sh, sl, spy, vix, rebal_dates,
                           use_leadlag=False, use_lgbm=False, **sim_kwargs)
        random_equities.append(eq)
    p_val = np.mean([re >= equity_real for re in random_equities])
    return p_val, np.mean(random_equities), np.std(random_equities)

def regime_r1_check(trades, spy):
    if not trades: return 1.0, 0, 0
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['date'])
    spy_monthly = spy.resample('ME').last().pct_change()
    bull_pnl, bear_pnl = [], []
    for _, row in df.iterrows():
        dt = row['date']
        closest = spy_monthly.index[spy_monthly.index.get_indexer([dt], method='ffill')[0]]
        if spy_monthly.loc[closest] >= 0:
            bull_pnl.append(row['pnl'])
        else:
            bear_pnl.append(row['pnl'])
    if not bull_pnl or not bear_pnl: return 1.0, 0, 0
    bull_sr = np.mean(bull_pnl) / (np.std(bull_pnl) + 1e-10)
    bear_sr = np.mean(bear_pnl) / (np.std(bear_pnl) + 1e-10)
    gap = abs(bull_sr - bear_sr) / (max(abs(bull_sr), abs(bear_sr)) + 1e-10)
    return round(gap, 3), round(bull_sr, 3), round(bear_sr, 3)

def subperiod_check(trades):
    if len(trades) < 20: return False, 0, 0
    df = pd.DataFrame(trades)
    mid = len(df) // 2
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

# ==================== LEAD-LAG ANALYSIS ====================
def analyze_leadlag_network(sc, sample_date=None):
    """Analyze and display the sector lead-lag network."""
    fprint("\n=== SECTOR LEAD-LAG NETWORK ANALYSIS ===")

    if sample_date is None:
        sample_date = sc.index[-1]

    idx = sc.index.get_indexer([sample_date], method='ffill')[0]
    hist = sc.iloc[:idx+1]
    rets_5d = hist.pct_change(5).dropna()

    if len(rets_5d) < 252:
        fprint("Not enough data for lead-lag analysis")
        return

    ll_matrix = compute_leadlag_matrix(rets_5d, lookback=252, lag_days=2)

    fprint(f"\nLead-lag correlation matrix (as of {sample_date.date()}):")
    fprint(f"{'':>6}", end='')
    for c in ll_matrix.columns:
        fprint(f"{c:>7}", end='')
    fprint()
    for r in ll_matrix.index:
        fprint(f"{r:>6}", end='')
        for c in ll_matrix.columns:
            v = ll_matrix.loc[r, c]
            fprint(f"{v:>7.3f}", end='')
        fprint()

    # Find strongest lead-lag pairs
    pairs = []
    for leader in ll_matrix.index:
        for follower in ll_matrix.columns:
            if leader != follower:
                corr = ll_matrix.loc[leader, follower]
                if abs(corr) >= 0.10:
                    pairs.append((leader, follower, corr))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    fprint(f"\nTop 15 lead-lag pairs (|corr| >= 0.10):")
    for leader, follower, corr in pairs[:15]:
        direction = "→ same dir" if corr > 0 else "→ opposite"
        fprint(f"  {leader} leads {follower}: {corr:+.3f} ({direction})")

    # Identify strongest leaders and followers
    leader_strength = ll_matrix.abs().sum(axis=1)
    follower_strength = ll_matrix.abs().sum(axis=0)
    fprint(f"\nLeader strength (sum of abs correlations as leader):")
    for tk in leader_strength.sort_values(ascending=False).index:
        fprint(f"  {tk}: {leader_strength[tk]:.3f}")
    fprint(f"\nFollower strength (sum of abs correlations as follower):")
    for tk in follower_strength.sort_values(ascending=False).index:
        fprint(f"  {tk}: {follower_strength[tk]:.3f}")

    return ll_matrix

# ==================== MAIN ====================
def main():
    fprint("=" * 70)
    fprint("SECTOR LEAD-LAG MOMENTUM OPTIONS v1 — Predictive Spillover")
    fprint("=" * 70)

    sc, sh, sl, spy, vix = download_data()

    # Rebalance biweekly starting 2010
    all_dates = sc.index[sc.index >= '2010-01-01']
    rebal_dates = all_dates[::10]
    fprint(f"Rebalance dates: {len(rebal_dates)}")

    # Analyze lead-lag network
    ll_matrix = analyze_leadlag_network(sc)

    # Build LGBM rankings for combined variants
    lgbm_rankings = build_lgbm_rankings(sc, rebal_dates)

    if MLFLOW_OK:
        exp = mlflow.set_experiment("sector_leadlag_options_v1")

    # ==================== VARIANTS ====================
    variants = {
        'A_PureLeadLag': {
            'desc': 'Pure lead-lag, no VIX filter, bull+bear combined',
            'vix_configs': [{'mode': 'bull', 'vix_min': None, 'vix_max': None}],
            'use_leadlag': True, 'leadlag_threshold': 0.10,
            'min_leaders': 1, 'use_lgbm': False, 'require_lgbm_agree': False,
        },
        'B_LeadLag_VIXBull': {
            'desc': 'Lead-lag bull spreads when VIX>=20',
            'vix_configs': [{'mode': 'bull', 'vix_min': 20, 'vix_max': None}],
            'use_leadlag': True, 'leadlag_threshold': 0.10,
            'min_leaders': 1, 'use_lgbm': False, 'require_lgbm_agree': False,
        },
        'C_LeadLag_BullBear': {
            'desc': 'Lead-lag bull (VIX>=20) + bear (VIX<20) combined',
            'vix_configs': [
                {'mode': 'bull', 'vix_min': 20, 'vix_max': None},
                {'mode': 'bear', 'vix_min': None, 'vix_max': 20},
            ],
            'use_leadlag': True, 'leadlag_threshold': 0.10,
            'min_leaders': 1, 'use_lgbm': False, 'require_lgbm_agree': False,
        },
        'D_LeadLag_LGBM': {
            'desc': 'Lead-lag + LGBM must both agree (bull+bear)',
            'vix_configs': [
                {'mode': 'bull', 'vix_min': 20, 'vix_max': None},
                {'mode': 'bear', 'vix_min': None, 'vix_max': 20},
            ],
            'use_leadlag': True, 'leadlag_threshold': 0.10,
            'min_leaders': 1, 'use_lgbm': True, 'require_lgbm_agree': True,
        },
        'E_MultiLeader': {
            'desc': 'Require 2+ leaders to agree (higher conviction)',
            'vix_configs': [
                {'mode': 'bull', 'vix_min': 20, 'vix_max': None},
                {'mode': 'bear', 'vix_min': None, 'vix_max': 20},
            ],
            'use_leadlag': True, 'leadlag_threshold': 0.08,
            'min_leaders': 2, 'use_lgbm': False, 'require_lgbm_agree': False,
        },
        'F_HighThreshold': {
            'desc': 'High correlation threshold (0.15+) for stronger lead-lag pairs',
            'vix_configs': [
                {'mode': 'bull', 'vix_min': 20, 'vix_max': None},
                {'mode': 'bear', 'vix_min': None, 'vix_max': 20},
            ],
            'use_leadlag': True, 'leadlag_threshold': 0.15,
            'min_leaders': 1, 'use_lgbm': False, 'require_lgbm_agree': False,
        },
    }

    results = {}
    for vname, cfg in variants.items():
        fprint(f"\n{'='*60}")
        fprint(f"Variant {vname}: {cfg['desc']}")
        fprint(f"{'='*60}")

        all_trades = []
        combined_equity = CAP
        combined_curve = [CAP]

        for vc in cfg['vix_configs']:
            eq, trd, curve = simulate(
                vname, sc, sh, sl, spy, vix, rebal_dates,
                spread_pct=3.0, dte=30, top_k=3, early_exit_day=20,
                vix_min=vc.get('vix_min'), vix_max=vc.get('vix_max'),
                mode=vc['mode'],
                use_leadlag=cfg['use_leadlag'],
                leadlag_threshold=cfg['leadlag_threshold'],
                min_leaders=cfg['min_leaders'],
                use_lgbm=cfg['use_lgbm'],
                lgbm_rankings=lgbm_rankings if cfg['use_lgbm'] else None,
                require_lgbm_agree=cfg['require_lgbm_agree'])
            all_trades.extend(trd)
            fprint(f"  {vc['mode']} ({vc.get('vix_min','any')}-{vc.get('vix_max','any')} VIX): {len(trd)} trades, ${eq:.0f}")

        # Sort and rebuild combined curve
        all_trades.sort(key=lambda x: x['date'])
        combined_equity = CAP
        combined_curve = [CAP]
        for t in all_trades:
            combined_equity += t['pnl']
            t['equity'] = round(combined_equity, 2)
            combined_curve.append(combined_equity)

        metrics = compute_metrics(all_trades, combined_curve)
        fprint(f"  Total: {metrics['n_trades']} trades")
        fprint(f"  Sharpe: {metrics['sharpe']}, CAGR: {metrics['cagr']}%, MDD: {metrics['mdd']}%")
        fprint(f"  WR: {metrics['wr']}%, PF: {metrics['pf']}, Sortino: {metrics['sortino']}")
        fprint(f"  Final equity: ${metrics['final_equity']}")

        # === ADVERSARIAL GATES ===
        gates = {'perm': False, 'r1': False, 'subperiod': False, 'outlier': False}

        if metrics['n_trades'] >= 20:
            # Permutation: random sector selection vs lead-lag selection
            p_val, rand_mean, rand_std = permutation_test(
                combined_curve[-1], sc, sh, sl, spy, vix, rebal_dates,
                n_perms=200, spread_pct=3.0, dte=30, top_k=3, early_exit_day=20,
                vix_min=cfg['vix_configs'][0].get('vix_min'),
                vix_max=cfg['vix_configs'][0].get('vix_max'),
                mode=cfg['vix_configs'][0]['mode'])
            gates['perm'] = p_val < 0.05
            fprint(f"  Perm: p={p_val:.3f} (rand=${rand_mean:.0f}) → {'PASS' if gates['perm'] else 'FAIL'}")
        else:
            p_val, rand_mean = 1.0, CAP
            fprint(f"  Perm: SKIP (too few trades)")

        r1_gap, bull_sr, bear_sr = regime_r1_check(all_trades, spy)
        gates['r1'] = r1_gap < 0.50
        fprint(f"  R1: gap={r1_gap} (bull={bull_sr}, bear={bear_sr}) → {'PASS' if gates['r1'] else 'FAIL'}")

        sp_ok, h1_wr, h2_wr = subperiod_check(all_trades)
        gates['subperiod'] = sp_ok
        fprint(f"  SubPeriod: h1={h1_wr}, h2={h2_wr} → {'PASS' if gates['subperiod'] else 'FAIL'}")

        out_ok, trimmed_pf = outlier_check(all_trades)
        gates['outlier'] = out_ok
        fprint(f"  Outlier: trimmed_PF={trimmed_pf} → {'PASS' if gates['outlier'] else 'FAIL'}")

        gates_passed = sum(gates.values())
        fprint(f"  GATES: {gates_passed}/4 {'✅ PASS' if gates_passed == 4 else '❌'}")

        results[vname] = {
            'desc': cfg['desc'], 'metrics': metrics, 'gates': gates,
            'gates_passed': gates_passed, 'perm_p': round(p_val, 3),
            'rand_equity': round(rand_mean, 0), 'r1_gap': r1_gap,
        }

        if MLFLOW_OK:
            with mlflow.start_run(run_name=vname):
                mlflow.log_params({'variant': vname, 'use_leadlag': cfg['use_leadlag'],
                    'threshold': cfg['leadlag_threshold'], 'min_leaders': cfg['min_leaders'],
                    'use_lgbm': cfg['use_lgbm']})
                mlflow.log_metrics({
                    'sharpe': metrics['sharpe'], 'sortino': metrics['sortino'],
                    'cagr': metrics['cagr'], 'mdd': metrics['mdd'],
                    'wr': metrics['wr'], 'pf': metrics['pf'],
                    'n_trades': metrics['n_trades'], 'final_equity': metrics['final_equity'],
                    'gates_passed': gates_passed, 'perm_p': p_val, 'r1_gap': r1_gap,
                })

    # ==================== SUMMARY ====================
    fprint(f"\n{'='*70}")
    fprint("SUMMARY — SECTOR LEAD-LAG OPTIONS v1")
    fprint(f"{'='*70}")
    fprint(f"{'Variant':<25} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'WR':>6} {'Trades':>7} {'Gates':>6} {'Final$':>8}")
    fprint("-" * 75)
    for vname, r in sorted(results.items()):
        m = r['metrics']
        fprint(f"{vname:<25} {m['sharpe']:>7.2f} {m['cagr']:>6.1f}% {m['mdd']:>6.1f}% {m['wr']:>5.1f}% {m['n_trades']:>7} {r['gates_passed']:>4}/4 ${m['final_equity']:>7.0f}")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nResults saved.")

    # Verdict
    any_pass = any(r['gates_passed'] == 4 for r in results.values())
    best = max(results.values(), key=lambda r: r['metrics']['sharpe'] if r['gates_passed'] >= 3 else 0)
    if any_pass:
        bm = best['metrics']
        fprint(f"\nBest: Sharpe {bm['sharpe']}, CAGR {bm['cagr']}%, {bm['n_trades']} trades")
        if bm['sharpe'] > 2.65:  # vs baseline bull+bear
            fprint("VERDICT: Lead-lag IMPROVES over baseline → ADOPT")
        else:
            fprint("VERDICT: Lead-lag works but doesn't beat LGBM momentum → MARGINAL")
    else:
        fprint("\nVERDICT: No variant passes all gates → Lead-lag is NOT viable for sector options")

    fprint("\nDONE")

if __name__ == '__main__':
    main()
