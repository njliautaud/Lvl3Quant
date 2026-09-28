#!/usr/bin/env python3
"""Focused Adversarial Validation — Top 3 Small-Account Options Strategies.

Tests: permutation shuffles, 2x commissions, bull/bear regime splits, yearly Sharpe consistency.
Uses HONEST Sharpe: mean(monthly_ret) / std(monthly_ret) * sqrt(12) where monthly_ret = monthly_pnl / equity_at_start_of_month.

Target runtime: <5 minutes.
"""
import json, sys, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'focused_adversarial_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

# MLflow setup
MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results saved locally only")

# ── Constants ──
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
BONDS = ['TLT','HYG','LQD','TIP']
CAP = 645.0
BASE_LEG_COMM = 0.65
HAIRCUT = 0.15
N_PERMS = 10
TRAIN_MONTHS = 24  # WF train period

QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']


def download_data():
    """Download sector ETFs + SPY + VIX."""
    import yfinance as yf
    all_tickers = list(set(SECTORS + BONDS + ['SPY', '^VIX']))
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    vol = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    fprint(f"  Data: {len(close)} days, {close.shape[1]} columns")
    return close, high, low, vol, spy, vix


def compute_features(px, vol_data=None):
    """Compute QM features for a single asset at a point in time."""
    if len(px) < 260:
        return None
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
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:]/pk)-1).min())
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


def build_rankings(sc, sv, rebal_dates, train_periods=12):
    """Walk-forward LightGBM ranking. train_periods = number of bi-weekly periods for training."""
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if not feats:
                continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)

    df = pd.DataFrame(records)
    for c in QM_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[QM_COLS] = df[QM_COLS].fillna(0.0)
    if len(df) < 100:
        return {}

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique())
    rankings = {}

    for i in range(train_periods, len(dates)):
        td = dates[max(0, i-train_periods):i]
        test_date = dates[i]
        tr = df[df['date'].isin(td)]
        te = df[df['date'] == test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue
        Xt = np.nan_to_num(tr[QM_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[QM_COLS].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except:
            continue

    return rankings


def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()


def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0:
        return max(0, S-K) if opt == 'call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt == 'call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)


def simulate(rankings, sc, sh, sl, spy, vix, atr_d, top_k=3, vix_min=20,
             comm_multiplier=1.0, shuffle_labels=False, seed=None):
    """Run the strategy simulation. Returns list of trades with dates and PnL.

    If shuffle_labels=True, randomly permute which ETFs get which scores (breaks signal).
    """
    spread_comm = 4 * BASE_LEG_COMM * comm_multiplier
    sma200 = spy.rolling(200).mean()
    equity = CAP
    trades = []
    # Track equity at start of each month for honest Sharpe
    monthly_start_equity = {}

    rng = np.random.RandomState(seed) if seed is not None else None

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue
        cv = float(vix.loc[dt])
        if vix_min is not None and cv < vix_min:
            continue

        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm

        scores = rankings[dt].copy()
        if not scores:
            continue

        # Permutation test: shuffle scores across tickers
        if shuffle_labels and rng is not None:
            tickers = list(scores.keys())
            vals = list(scores.values())
            rng.shuffle(vals)
            scores = dict(zip(tickers, vals))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]
        max_pos = min(200, equity/3)
        if max_pos < 30:
            continue

        # Track month-start equity
        month_key = dt.to_period('M')
        if month_key not in monthly_start_equity:
            monthly_start_equity[month_key] = equity

        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3:
                continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)
            ei = min(di+30, len(sc)-1)
            Se = float(sc[tk].iloc[ei])
            K1, K2 = round(S), round(S*1.03)
            lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
            sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
            val = lp - sp
            width = K2 - K1
            cost = val*100 + spread_comm
            mx_prof = (width - val)*100 - spread_comm
            if cost <= 0 or cost > max_pos or cost > equity*0.40:
                continue

            pnl = None
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di
                rd = max(0, 30-dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/30.0)
                si = (max(0, Sc-K1) - max(0, Sc-K2))*100 + ac*tm*0.3*100
                cp = si - cost
                if cp >= mx_prof*0.50 or rd < 7:
                    pnl = cp
                    break
            if pnl is None:
                pnl = (max(0, Se-K1) - max(0, Se-K2))*100 - val*100 - spread_comm

            equity += pnl
            n_ent += 1
            trades.append({
                'date': str(dt.date()),
                'month': str(month_key),
                'ticker': tk,
                'pnl': round(pnl, 2),
                'win': pnl > 0,
                'regime': 'bull' if bull else 'bear',
                'year': dt.year,
                'equity_before': equity - pnl,
            })

    return trades, equity, monthly_start_equity


def honest_sharpe(trades, monthly_start_equity):
    """Compute HONEST annualized Sharpe: mean(monthly_ret)/std(monthly_ret)*sqrt(12)
    where monthly_ret = monthly_pnl / equity_at_start_of_month."""
    if not trades:
        return 0.0, [], {}

    tdf = pd.DataFrame(trades)
    tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
    monthly_pnl = tdf.groupby('month')['pnl'].sum()

    monthly_rets = []
    month_details = {}
    for m in monthly_pnl.index:
        pnl = monthly_pnl[m]
        start_eq = monthly_start_equity.get(m, CAP)
        if start_eq <= 0:
            start_eq = CAP
        ret = pnl / start_eq
        monthly_rets.append(ret)
        month_details[str(m)] = {'pnl': round(pnl, 2), 'start_eq': round(start_eq, 2), 'ret': round(ret, 4)}

    rets = np.array(monthly_rets)
    if len(rets) < 3:
        return 0.0, rets.tolist(), month_details

    sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(12))
    return sharpe, rets.tolist(), month_details


def honest_sortino(monthly_rets):
    """Sortino from monthly returns."""
    rets = np.array(monthly_rets)
    if len(rets) < 3:
        return 0.0
    downside = rets[rets < 0]
    if len(downside) < 2:
        return float(rets.mean() / 1e-10 * np.sqrt(12))  # no downside = huge sortino
    return float(rets.mean() / (downside.std() + 1e-10) * np.sqrt(12))


def compute_metrics(trades, final_equity, monthly_start_equity, name):
    """Compute all metrics with honest Sharpe."""
    if not trades:
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins/n*100
    pnls = [t['pnl'] for t in trades]

    # Honest Sharpe
    sharpe, monthly_rets, month_details = honest_sharpe(trades, monthly_start_equity)
    sortino = honest_sortino(monthly_rets)

    # CAGR
    n_years = max(len(monthly_rets)/12, 0.5)
    cagr = (final_equity/CAP)**(1/n_years) - 1

    # Max drawdown from equity curve
    eq = [CAP]
    for t in trades:
        eq.append(eq[-1] + t['pnl'])
    eq = np.array(eq)
    pk = np.maximum.accumulate(eq)
    maxdd = float(((eq - pk)/(pk + 1e-10)).min())

    # Profit factor
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp/(gl + 1e-10)

    # Bull/bear split
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']
    bull_wr = sum(1 for t in bull_trades if t['win'])/max(len(bull_trades), 1)*100
    bear_wr = sum(1 for t in bear_trades if t['win'])/max(len(bear_trades), 1)*100
    r1_gap = abs(bull_wr - bear_wr)/max(bull_wr, bear_wr, 1)

    # Bull/bear Sharpe
    bull_pnls_by_month = {}
    bear_pnls_by_month = {}
    for t in trades:
        m = t['month']
        if t['regime'] == 'bull':
            bull_pnls_by_month[m] = bull_pnls_by_month.get(m, 0) + t['pnl']
        else:
            bear_pnls_by_month[m] = bear_pnls_by_month.get(m, 0) + t['pnl']

    def sharpe_from_dict(pnl_dict):
        if len(pnl_dict) < 3:
            return 0.0
        vals = np.array(list(pnl_dict.values()))
        # Use average equity as denominator for regime sub-Sharpes
        avg_eq = np.mean([monthly_start_equity.get(pd.Period(m, 'M'), CAP) for m in pnl_dict.keys()])
        rets = vals / max(avg_eq, 1)
        return float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(12))

    bull_sharpe = sharpe_from_dict(bull_pnls_by_month)
    bear_sharpe = sharpe_from_dict(bear_pnls_by_month)

    # Yearly Sharpe
    tdf = pd.DataFrame(trades)
    yearly_sharpes = {}
    for yr in sorted(tdf['year'].unique()):
        yr_trades = [t for t in trades if t['year'] == yr]
        yr_monthly = {}
        for t in yr_trades:
            m = t['month']
            yr_monthly[m] = yr_monthly.get(m, 0) + t['pnl']
        if len(yr_monthly) >= 3:
            vals = np.array(list(yr_monthly.values()))
            avg_eq = np.mean([monthly_start_equity.get(pd.Period(m, 'M'), CAP) for m in yr_monthly.keys()])
            rets = vals / max(avg_eq, 1)
            yearly_sharpes[int(yr)] = round(float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(12)), 2)

    return {
        'name': name,
        'n_trades': n,
        'win_rate': round(wr, 1),
        'honest_sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr_pct': round(cagr*100, 1),
        'maxdd_pct': round(maxdd*100, 1),
        'profit_factor': round(pf, 2),
        'final_equity': round(final_equity, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'r1_gap': round(r1_gap, 3),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'yearly_sharpes': yearly_sharpes,
        'n_months': len(monthly_rets),
    }


def prepare_strategy_data(universe, close, high, low, vol, spy, vix, train_periods=None):
    """Prepare data and build rankings ONCE. Returns cached data for fast re-simulation."""
    available = [c for c in universe if c in close.columns and close[c].dropna().shape[0] > 500]
    if len(available) < 3:
        return None

    sc = close[available].dropna(how='all')
    sh = high[[c for c in available if c in high.columns]].dropna(how='all')
    sl = low[[c for c in available if c in low.columns]].dropna(how='all')
    sv = vol[[c for c in available if c in vol.columns]].dropna(how='all')

    ix = sc.index.intersection(vix.index).intersection(spy.index)
    if len(sh) > 0: ix = ix.intersection(sh.index)
    if len(sl) > 0: ix = ix.intersection(sl.index)
    sc = sc.loc[ix]; sh = sh.loc[ix]; sl = sl.loc[ix]; sv = sv.reindex(ix)

    # ATR
    atr_d = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_d[tk] = compute_atr(sh[tk], sl[tk], sc[tk])

    # Bi-weekly rebalance dates
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)

    tp = train_periods if train_periods else 52
    fprint("  Building rankings (one-time)...")
    rankings = build_rankings(sc, sv, bd, train_periods=tp)
    if not rankings:
        return None

    return {'sc': sc, 'sh': sh, 'sl': sl, 'sv': sv, 'atr_d': atr_d, 'rankings': rankings}


def run_simulation(name, data, spy, vix, top_k=3, vix_min=20,
                   comm_multiplier=1.0, shuffle_labels=False, seed=None):
    """Run simulation using cached data. Fast — no retraining."""
    trades, final_eq, monthly_start_eq = simulate(
        data['rankings'], data['sc'], data['sh'], data['sl'], spy, vix, data['atr_d'],
        top_k=top_k, vix_min=vix_min,
        comm_multiplier=comm_multiplier,
        shuffle_labels=shuffle_labels, seed=seed
    )
    return compute_metrics(trades, final_eq, monthly_start_eq, name)


def main():
    t0 = time.time()
    fprint(f"Focused Adversarial Validation — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint("Testing top 3 strategies with honest Sharpe, permutation tests, 2x costs, regime splits")
    fprint(f"{'='*80}\n")

    # Download data once
    close, high, low, vol, spy, vix = download_data()

    # Define strategies to test
    strategies = [
        ('A_Sectors_Baseline', SECTORS, 3, 20),
        ('E_Sectors_Bonds', SECTORS + BONDS, 3, 20),
        ('B_Broad_Universe', SECTORS + BONDS + ['GLD','SLV','USO','DBA','EFA','EEM','VWO','VNQ','AMLP','BITO'], 3, 20),
    ]

    all_results = {}

    for strat_name, universe, top_k, vix_min in strategies:
        fprint(f"\n{'='*70}")
        fprint(f"STRATEGY: {strat_name} ({len(universe)} assets)")
        fprint(f"{'='*70}")

        # Prepare data and build rankings ONCE
        data = prepare_strategy_data(universe, close, high, low, vol, spy, vix)
        if not data:
            fprint(f"  SKIPPED — insufficient data")
            continue

        # ── 1. Base run with honest Sharpe ──
        fprint("\n[1] Base run (honest Sharpe, standard costs)...")
        base = run_simulation(strat_name, data, spy, vix,
                             top_k=top_k, vix_min=vix_min, comm_multiplier=1.0)
        if not base:
            fprint(f"  SKIPPED — no trades")
            continue

        fprint(f"  Trades: {base['n_trades']} | WR: {base['win_rate']}% | Honest Sharpe: {base['honest_sharpe']}")
        fprint(f"  CAGR: {base['cagr_pct']}% | MaxDD: {base['maxdd_pct']}% | PF: {base['profit_factor']}")
        fprint(f"  Bull WR: {base['bull_wr']}% ({base['bull_trades']} trades) | Bear WR: {base['bear_wr']}% ({base['bear_trades']} trades)")
        fprint(f"  Bull Sharpe: {base['bull_sharpe']} | Bear Sharpe: {base['bear_sharpe']}")
        fprint(f"  Yearly Sharpes: {base['yearly_sharpes']}")

        # ── 2. Permutation test (10 shuffles) — reuses cached rankings, only shuffles scores ──
        fprint(f"\n[2] Permutation test ({N_PERMS} shuffles — fast, reusing cached rankings)...")
        perm_sharpes = []
        for i in range(N_PERMS):
            perm = run_simulation(f"{strat_name}_perm{i}", data, spy, vix,
                                top_k=top_k, vix_min=vix_min, comm_multiplier=1.0,
                                shuffle_labels=True, seed=42+i)
            if perm:
                perm_sharpes.append(perm['honest_sharpe'])
                fprint(f"    Perm {i}: Sharpe {perm['honest_sharpe']}")
            else:
                perm_sharpes.append(0.0)

        perm_p = sum(1 for s in perm_sharpes if s >= base['honest_sharpe']) / len(perm_sharpes)
        perm_mean = np.mean(perm_sharpes)
        perm_max = max(perm_sharpes)
        fprint(f"  Base Sharpe: {base['honest_sharpe']} | Perm mean: {perm_mean:.2f} | Perm max: {perm_max:.2f}")
        fprint(f"  Perm p-value: {perm_p:.3f} {'PASS' if perm_p < 0.10 else 'FAIL'}")

        # ── 3. 2x commissions stress test ──
        fprint("\n[3] 2x commissions ($5.20 RT)...")
        stressed = run_simulation(f"{strat_name}_2x_cost", data, spy, vix,
                                 top_k=top_k, vix_min=vix_min, comm_multiplier=2.0)
        if stressed:
            fprint(f"  Trades: {stressed['n_trades']} | WR: {stressed['win_rate']}% | Sharpe: {stressed['honest_sharpe']}")
            fprint(f"  CAGR: {stressed['cagr_pct']}% | PF: {stressed['profit_factor']}")
            sharpe_decay = (base['honest_sharpe'] - stressed['honest_sharpe']) / max(abs(base['honest_sharpe']), 0.01) * 100
            fprint(f"  Sharpe decay from 2x costs: {sharpe_decay:.1f}%")

        # ── 4. Yearly consistency check ──
        fprint("\n[4] Yearly Sharpe consistency...")
        yearly = base['yearly_sharpes']
        positive_years = sum(1 for s in yearly.values() if s > 0)
        total_years = len(yearly)
        fprint(f"  Positive Sharpe years: {positive_years}/{total_years}")
        for yr, sh in sorted(yearly.items()):
            status = "OK" if sh > 0 else "NEGATIVE"
            fprint(f"    {yr}: Sharpe {sh:>6.2f}  {status}")

        # ── Compile results ──
        result = {
            'base': base,
            'permutation': {
                'n_perms': N_PERMS,
                'base_sharpe': base['honest_sharpe'],
                'perm_sharpes': [round(s, 2) for s in perm_sharpes],
                'perm_mean': round(perm_mean, 2),
                'perm_max': round(perm_max, 2),
                'perm_p': round(perm_p, 3),
                'verdict': 'PASS' if perm_p < 0.10 else 'FAIL',
            },
            'stress_2x_cost': stressed if stressed else None,
            'regime_split': {
                'bull_sharpe': base['bull_sharpe'],
                'bear_sharpe': base['bear_sharpe'],
                'bull_wr': base['bull_wr'],
                'bear_wr': base['bear_wr'],
                'r1_gap': base['r1_gap'],
                'regime_balanced': base['r1_gap'] < 0.50,
            },
            'yearly_consistency': {
                'yearly_sharpes': yearly,
                'positive_years': positive_years,
                'total_years': total_years,
                'pct_positive': round(positive_years/max(total_years, 1)*100, 1),
            },
        }

        # Adversarial verdict
        gates = 0
        gate_details = []

        # Gate 1: Permutation significance
        g1 = perm_p < 0.10
        gates += g1
        gate_details.append(f"Perm p={perm_p:.3f} {'PASS' if g1 else 'FAIL'}")

        # Gate 2: Regime balance
        g2 = base['r1_gap'] < 0.50
        gates += g2
        gate_details.append(f"R1 gap={base['r1_gap']:.3f} {'PASS' if g2 else 'FAIL'}")

        # Gate 3: Survives 2x costs
        g3 = stressed and stressed['honest_sharpe'] > 0.3
        gates += g3
        gate_details.append(f"2x cost Sharpe={stressed['honest_sharpe'] if stressed else 'N/A'} {'PASS' if g3 else 'FAIL'}")

        # Gate 4: Yearly consistency (>50% positive years)
        g4 = positive_years / max(total_years, 1) > 0.50
        gates += g4
        gate_details.append(f"Positive years={positive_years}/{total_years} {'PASS' if g4 else 'FAIL'}")

        # Gate 5: Honest Sharpe > 0.5
        g5 = base['honest_sharpe'] > 0.5
        gates += g5
        gate_details.append(f"Honest Sharpe={base['honest_sharpe']} {'PASS' if g5 else 'FAIL'}")

        result['adversarial_verdict'] = {
            'gates_passed': gates,
            'gates_total': 5,
            'gate_details': gate_details,
            'overall': 'PASS' if gates >= 4 else 'MARGINAL' if gates >= 3 else 'FAIL',
        }

        all_results[strat_name] = result
        fprint(f"\n  VERDICT: {gates}/5 gates passed → {result['adversarial_verdict']['overall']}")
        for gd in gate_details:
            fprint(f"    {gd}")

    # ── Summary ──
    elapsed = time.time() - t0
    fprint(f"\n{'='*80}")
    fprint(f"ADVERSARIAL VALIDATION SUMMARY")
    fprint(f"{'='*80}")
    fprint(f"{'Strategy':<25} {'Sharpe':>8} {'Perm-p':>8} {'2x Sh':>8} {'R1 gap':>8} {'Yr+':>5} {'Gates':>7} {'Verdict':>10}")
    fprint("-"*85)
    for name, r in all_results.items():
        b = r['base']
        s = r['stress_2x_cost']
        p = r['permutation']
        v = r['adversarial_verdict']
        yc = r['yearly_consistency']
        fprint(f"{name:<25} {b['honest_sharpe']:>8.2f} {p['perm_p']:>8.3f} "
               f"{s['honest_sharpe'] if s else 0:>8.2f} {b['r1_gap']:>8.3f} "
               f"{yc['positive_years']}/{yc['total_years']:>2} {v['gates_passed']}/{v['gates_total']:>2}   {v['overall']:>10}")

    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save results
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Focused adversarial validation of top 3 small-account options strategies',
        'methodology': {
            'honest_sharpe': 'mean(monthly_ret)/std(monthly_ret)*sqrt(12) where monthly_ret = monthly_pnl/equity_at_start_of_month',
            'permutation_test': f'{N_PERMS} shuffles of ranking scores across tickers',
            'stress_test': '2x commissions ($5.20 RT instead of $2.60)',
            'regime_split': 'Bull = SPY above 200-day SMA, Bear = below',
            'yearly_consistency': 'Sharpe computed per calendar year',
        },
        'results': all_results,
        'runtime_s': round(elapsed, 1),
    }

    def json_serializable(obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=json_serializable)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            en = 'focused_adversarial_validation'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"adv_val_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                for name, r in all_results.items():
                    p = name[:15].replace(' ', '_')
                    b = r['base']
                    mlflow.log_metrics({
                        f'{p}_sharpe': b['honest_sharpe'],
                        f'{p}_cagr': b['cagr_pct'],
                        f'{p}_mdd': b['maxdd_pct'],
                        f'{p}_wr': b['win_rate'],
                        f'{p}_perm_p': r['permutation']['perm_p'],
                        f'{p}_gates': r['adversarial_verdict']['gates_passed'],
                    })
                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint(f"\n{'='*80}")
    fprint("DONE — Focused Adversarial Validation")
    fprint(f"{'='*80}")


if __name__ == '__main__':
    main()
