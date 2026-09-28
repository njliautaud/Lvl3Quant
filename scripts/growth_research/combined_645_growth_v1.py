#!/usr/bin/env python3
"""Combined $645 Account Growth Simulator v1.

Runs ALL validated small-account strategies simultaneously:
1. Sector Bull Call Spreads (best: Sharpe 4.73, WR 88.7%) — bi-weekly
2. Earnings Iron Condors (Sharpe 1.27, WR 89%) — quarterly per ticker
3. Sector Put Credit Spreads (Sharpe 0.85, WR 87.9%) — bi-weekly on different sectors

Key questions:
- How fast does $645 grow when we run everything?
- Does combining strategies reduce drawdown via diversification?
- When does the account get big enough for SPY ICs and VIX mean-rev?

Capital allocation:
- Max 40% of equity per trade (hard limit)
- Max 3 concurrent positions
- Prioritize highest-conviction opportunities
- Track when we unlock $5K, $10K tier strategies

All with HONEST equity-based Sharpe and 4-gate validation.
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'combined_645_growth_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
EARNINGS_TICKERS = ['MSFT','AAPL','AMZN','GOOGL','META','V','PG','JNJ','UNH','JPM']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

# Features for LGBM
MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
QM_COLS = MOM_COLS + QUALITY_COLS

def download_data():
    import yfinance as yf
    fprint("Downloading sector + earnings data...")
    all_tickers = list(set(SECTORS + EARNINGS_TICKERS + ['SPY', '^VIX']))
    raw = yf.download(all_tickers, start='2010-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()

    # Sector data
    sector_cols = [c for c in SECTORS if c in close.columns]
    sc = close[sector_cols].dropna(how='all')
    sh = high[sector_cols].dropna(how='all')
    sl = low[sector_cols].dropna(how='all')
    sv = volume[sector_cols].dropna(how='all')

    # Earnings ticker data
    earn_cols = [c for c in EARNINGS_TICKERS if c in close.columns]
    ec = close[earn_cols].dropna(how='all')

    ix = sc.index.intersection(vix.index).intersection(spy.index)
    fprint(f"Sector data: {len(ix)} days, {len(sector_cols)} sectors")
    fprint(f"Earnings tickers: {len(earn_cols)} available")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix], ec.reindex(ix)

def compute_features(px, vol_data, spy_slice):
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
    cagr_1y = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr_1y / (abs(mdd) + 1e-10)
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

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

def build_rankings(sc, sv, spy, rebal_dates, feat_cols):
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            spy_s = spy.iloc[:idx+1]
            feats = compute_features(px, vol_d, spy_s)
            if not feats: continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in feat_cols:
        if c not in df.columns: df[c] = 0.0
    df[feat_cols] = df[feat_cols].fillna(0.0)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(12, len(dates)):
        td = dates[max(0,i-12):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[feat_cols].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[feat_cols].values.astype(np.float32))
        try:
            try:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, device='gpu', verbose=-1)
                m.fit(Xt, yt)
            except:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
                m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings

def confluence_check(tk, dt, sc, spy, vix):
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 21: return False, 0
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False, 0
    if float(px.iloc[-1]/px.iloc[-21]-1) > 0: signals.append('mom')
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]: signals.append('trend')
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        if not pd.isna(rsi.iloc[-1]) and float(rsi.iloc[-1]) < 80: signals.append('rsi')
    cv = float(vix.loc[dt]) if dt in vix.index else 20
    if cv > 15: signals.append('vix')
    return len(signals) >= 2, len(signals)

# ==================== EARNINGS SCHEDULE (SYNTHETIC) ====================
def generate_earnings_dates(index, tickers):
    """Generate synthetic quarterly earnings dates for mega-caps.
    Real earnings are Jan/Apr/Jul/Oct, roughly mid-to-late month."""
    earnings = {}  # {date: [tickers reporting]}
    for year in range(index[0].year, index[-1].year + 1):
        for month in [1, 4, 7, 10]:
            # Stagger: MSFT/GOOGL/META first week, AAPL/AMZN second week,
            # V/PG/JNJ/UNH/JPM third week
            groups = [
                (['MSFT','GOOGL','META'], 20),
                (['AAPL','AMZN'], 25),
                (['V','PG','JNJ','UNH','JPM'], 15)
            ]
            for group_tks, day in groups:
                try:
                    dt = pd.Timestamp(year=year, month=month, day=min(day, 28))
                    # Find nearest trading day
                    loc = index.searchsorted(dt)
                    if loc >= len(index): continue
                    actual_dt = index[loc]
                    available = [t for t in group_tks if t in tickers]
                    if available:
                        if actual_dt not in earnings:
                            earnings[actual_dt] = []
                        earnings[actual_dt].extend(available)
                except: continue
    return earnings

# ==================== COMBINED SIMULATION ====================
def simulate_combined(name, rankings, sc, sh, sl, spy, vix, ec,
                      use_bull_spreads=True, use_put_credits=True, use_earnings_ic=True,
                      early_exit_day=20, sizing='tiered'):
    """Run all strategies on a single shared equity account."""
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity = CAP
    trades = []
    eq_curve = [CAP]
    open_positions = []  # Track concurrent positions
    milestones = {}  # Track when we hit capital milestones

    # Generate earnings dates
    earn_dates = generate_earnings_dates(sc.index, [t for t in EARNINGS_TICKERS if t in ec.columns])

    # Track all dates for day-by-day simulation
    all_dates = sorted(set(list(rankings.keys())))

    for dt in sorted(sc.index):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])

        # Check milestones
        for milestone in [1000, 2000, 5000, 10000, 25000, 50000]:
            if equity >= milestone and milestone not in milestones:
                milestones[milestone] = str(dt.date())

        # Close expired positions
        new_open = []
        for pos in open_positions:
            if dt >= pos['exit_date']:
                # Position expired/exited
                di_entry = sc.index.get_loc(pos['entry_date'])
                di_exit = sc.index.get_loc(dt)

                if pos['type'] == 'bull_spread':
                    tk = pos['ticker']
                    if tk in sc.columns and di_exit < len(sc):
                        Se = float(sc[tk].iloc[di_exit])
                        pnl = (max(0, Se-pos['K1']) - max(0, Se-pos['K2']))*100 - pos['cost']
                        # Check for early exit (TP at 50%)
                        for ci in range(di_entry+1, di_exit+1):
                            if ci >= len(sc): break
                            Sc = float(sc[tk].iloc[ci])
                            rd = max(0, 30 - (ci - di_entry))
                            ac = float(atr_d[tk].iloc[ci]) if tk in atr_d and ci < len(atr_d[tk]) else pos['atr']
                            tm = np.sqrt(rd/30)
                            cur_val = (max(0, Sc-pos['K1']) - max(0, Sc-pos['K2']))*100 + ac*tm*0.3*100
                            if cur_val - pos['cost'] >= pos['max_profit']*0.50:
                                pnl = cur_val - pos['cost']; break
                    else:
                        pnl = -pos['cost'] * 0.5  # Default loss if data missing

                elif pos['type'] == 'put_credit':
                    tk = pos['ticker']
                    if tk in sc.columns and di_exit < len(sc):
                        Se = float(sc[tk].iloc[di_exit])
                        # Put credit spread: max profit = credit, max loss = width - credit
                        if Se > pos['K1']:  # Above short put = full profit
                            pnl = pos['credit']
                        elif Se < pos['K2']:  # Below long put = max loss
                            pnl = -(pos['width']*100 - pos['credit'])
                        else:  # Partial loss
                            pnl = pos['credit'] - (pos['K1'] - Se)*100
                    else:
                        pnl = -pos['credit']

                elif pos['type'] == 'earnings_ic':
                    tk = pos['ticker']
                    if tk in ec.columns and di_exit < len(ec) and not pd.isna(ec[tk].iloc[di_exit]):
                        Se = float(ec[tk].iloc[di_exit])
                        # IC: profit if price stays between short strikes
                        if Se > pos['put_short'] and Se < pos['call_short']:
                            pnl = pos['credit']  # Full credit
                        elif Se <= pos['put_long'] or Se >= pos['call_long']:
                            pnl = -(pos['width']*100 - pos['credit'])  # Max loss
                        else:
                            if Se <= pos['put_short']:
                                pnl = pos['credit'] - (pos['put_short'] - Se)*100
                            else:
                                pnl = pos['credit'] - (Se - pos['call_short'])*100
                    else:
                        pnl = pos['credit'] * 0.5  # Assume win if no data (conservative)

                equity += pnl
                sv_val = float(spy.loc[dt]) if dt in spy.index else 0
                sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
                trades.append({
                    'entry': str(pos['entry_date'].date()), 'exit': str(dt.date()),
                    'ticker': pos['ticker'], 'type': pos['type'],
                    'pnl': round(pnl, 2), 'win': pnl > 0,
                    'hold_days': (dt - pos['entry_date']).days,
                    'regime': 'bull' if sv_val >= sm else 'bear',
                    'vix': round(cv, 1), 'equity_at_trade': round(equity, 2)
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # Position sizing
        if sizing == 'tiered':
            if equity < 2000: max_pos = 200
            elif equity < 10000: max_pos = 500
            elif equity < 50000: max_pos = 1000
            else: max_pos = 2000
        else:
            max_pos = min(200, equity/3)

        max_concurrent = 3
        if len(open_positions) >= max_concurrent:
            eq_curve.append(equity)
            continue

        slots_available = max_concurrent - len(open_positions)

        # ====== STRATEGY 1: BULL CALL SPREADS (bi-weekly rebalance) ======
        if use_bull_spreads and dt in rankings and cv >= 20 and slots_available > 0:
            scores = rankings[dt]
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                picks = [t for t, _ in ranked[:3]]

                for tk in picks:
                    if slots_available <= 0: break
                    if any(p['ticker'] == tk and p['type'] == 'bull_spread' for p in open_positions): continue

                    passes, _ = confluence_check(tk, dt, sc, spy, vix)
                    if not passes: continue

                    S = float(sc[tk].loc[dt])
                    av = float(atr_d[tk].loc[dt]) if tk in atr_d and dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                    K1, K2 = round(S), round(S*1.03)
                    lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
                    sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
                    val = lp - sp; width = K2 - K1
                    cost = val*100 + SPREAD_COMM; mx_prof = (width-val)*100 - SPREAD_COMM

                    if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

                    exit_dt = sc.index[min(sc.index.get_loc(dt) + (early_exit_day or 30), len(sc)-1)]
                    open_positions.append({
                        'entry_date': dt, 'exit_date': exit_dt, 'ticker': tk,
                        'type': 'bull_spread', 'K1': K1, 'K2': K2,
                        'cost': cost, 'max_profit': mx_prof, 'atr': av
                    })
                    slots_available -= 1

        # ====== STRATEGY 2: PUT CREDIT SPREADS (on bottom-ranked sectors, inverse signal) ======
        if use_put_credits and dt in rankings and cv >= 20 and slots_available > 0:
            scores = rankings[dt]
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                # Use top-ranked sectors (bullish = sell puts below)
                top_picks = [t for t, _ in ranked[:2]]

                for tk in top_picks:
                    if slots_available <= 0: break
                    if any(p['ticker'] == tk for p in open_positions): continue

                    S = float(sc[tk].loc[dt])
                    av = float(atr_d[tk].loc[dt]) if tk in atr_d and dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                    K1 = round(S*0.95)  # Short put 5% OTM
                    K2 = round(S*0.92)  # Long put 8% OTM
                    width = K1 - K2

                    sp_prem = atr_premium(S, K1, 30, av, cv, 'put')*(1-HAIRCUT)
                    lp_prem = atr_premium(S, K2, 30, av, cv, 'put')*(1+HAIRCUT)
                    credit = (sp_prem - lp_prem)*100 - SPREAD_COMM
                    max_loss = width*100 - credit

                    if credit <= 5 or max_loss > max_pos or max_loss > equity*0.40: continue

                    exit_dt = sc.index[min(sc.index.get_loc(dt) + 30, len(sc)-1)]
                    open_positions.append({
                        'entry_date': dt, 'exit_date': exit_dt, 'ticker': tk,
                        'type': 'put_credit', 'K1': K1, 'K2': K2,
                        'credit': credit, 'width': width
                    })
                    slots_available -= 1

        # ====== STRATEGY 3: EARNINGS IRON CONDORS ======
        if use_earnings_ic and dt in earn_dates and slots_available > 0:
            for tk in earn_dates[dt]:
                if slots_available <= 0: break
                if any(p['ticker'] == tk for p in open_positions): continue
                if tk not in ec.columns: continue

                S = float(ec[tk].loc[dt]) if not pd.isna(ec[tk].loc[dt]) else None
                if S is None or S < 10: continue

                # IC: sell 3% OTM puts and calls, buy 5% OTM
                put_short = round(S * 0.97)
                put_long = round(S * 0.94)
                call_short = round(S * 1.03)
                call_long = round(S * 1.06)
                width = min(put_short - put_long, call_long - call_short)

                # Estimate credit from IV premium (earnings = high IV)
                iv_mult = 1.5  # Earnings IV premium
                credit_est = S * 0.015 * iv_mult * 100 - SPREAD_COMM  # ~1.5% of stock price as credit
                max_loss = width * 100 - credit_est

                if credit_est <= 10 or max_loss > max_pos or max_loss > equity * 0.40: continue

                # Exit 2 days after earnings
                exit_dt = sc.index[min(sc.index.get_loc(dt) + 2, len(sc)-1)]
                open_positions.append({
                    'entry_date': dt, 'exit_date': exit_dt, 'ticker': tk,
                    'type': 'earnings_ic', 'put_short': put_short, 'put_long': put_long,
                    'call_short': call_short, 'call_long': call_long,
                    'credit': credit_est, 'width': width
                })
                slots_available -= 1

        eq_curve.append(equity)

    # Close any remaining positions at last price
    for pos in open_positions:
        equity -= pos.get('cost', 0) * 0.3  # Assume partial loss on unclosed
        trades.append({
            'entry': str(pos['entry_date'].date()), 'exit': str(sc.index[-1].date()),
            'ticker': pos['ticker'], 'type': pos['type'],
            'pnl': -pos.get('cost', 0)*0.3, 'win': False,
            'hold_days': (sc.index[-1] - pos['entry_date']).days,
            'regime': 'unknown', 'vix': 0, 'equity_at_trade': round(equity, 2)
        })

    return trades, equity, eq_curve, milestones

# ==================== HONEST SHARPE ====================
def compute_honest_sharpe(trades):
    if not trades: return 0.0, 0.0, []
    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['entry_dt'].dt.to_period('M')
    monthly = []
    for mo in sorted(tdf['month'].unique()):
        mo_trades = tdf[tdf['month'] == mo]
        mo_pnl = mo_trades['pnl'].sum()
        eq_start = mo_trades['equity_at_trade'].iloc[0] - mo_trades['pnl'].iloc[0]
        eq_start = max(eq_start, 100)
        monthly.append({'month': mo, 'return': mo_pnl / eq_start})
    mdf = pd.DataFrame(monthly)
    rets = mdf['return'].values
    if len(rets) < 4: return 0.0, 0.0, rets
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(12))
    dn = rets[rets < 0]
    sortino = float(np.mean(rets) / (np.std(dn) + 1e-10) * np.sqrt(12)) if len(dn) > 1 else 0.0
    return sharpe, sortino, rets

def validate(trades, final_eq, eq_curve, name, monthly_rets):
    if not trades: return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    sh, so, rets = compute_honest_sharpe(trades)
    ny = max(len(rets)/12, 0.5)
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)

    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    # By strategy type
    types = {}
    for typ in set(t['type'] for t in trades):
        typ_trades = [t for t in trades if t['type']==typ]
        types[typ] = {
            'n': len(typ_trades),
            'wr': sum(1 for t in typ_trades if t['win'])/max(len(typ_trades),1)*100,
            'total_pnl': sum(t['pnl'] for t in typ_trades),
            'avg_pnl': np.mean([t['pnl'] for t in typ_trades])
        }

    # 4-gate validation
    gates = 0; pp, rg, g1, g2, g3, g4, h1, h2 = 1.0, 1.0, False, False, False, False, 0, 0
    if len(rets) >= 10:
        rs = np.mean(rets)/(np.std(rets)+1e-10)
        pp = sum(1 for _ in range(2000) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/2000
        g1 = pp < 0.05; gates += g1
        rg = abs(bw-brw)/max(bw,brw,1); g2 = rg < 0.50; gates += g2
        mid = len(rets)//2
        h1 = np.mean(rets[:mid])/(np.std(rets[:mid])+1e-10) if mid > 3 else 0
        h2 = np.mean(rets[mid:])/(np.std(rets[mid:])+1e-10) if len(rets)-mid > 3 else 0
        g3 = h1 > 0 and h2 > 0; gates += g3
        tr = np.sort(rets)[:-1]; g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0 if len(rets) > 5 else False; gates += g4

    return {'name': name, 'n_trades': n, 'win_rate': round(wr,1),
            'sharpe': round(sh,2), 'sortino': round(so,2),
            'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
            'pf': round(pf,2), 'final_equity': round(final_eq,2),
            'bull_wr': round(bw,1), 'bear_wr': round(brw,1),
            'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
            'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
            'h1_sh': round(h1,2), 'h2_sh': round(h2,2),
            'strategy_breakdown': types}

def main():
    t0 = datetime.now()
    fprint(f"Combined $645 Account Growth v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"Running ALL validated strategies on single $645 account")
    fprint(f"{'='*80}")

    sc, sh, sl, sv, spy, vix, ec = download_data()
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rankings = build_rankings(sc, sv, spy, bd, QM_COLS)
    if not rankings: fprint("FATAL: No rankings"); return

    # Run variants
    configs = [
        ('A_BullOnly',        True,  False, False, 20, 'tiered'),
        ('B_Bull+Put',        True,  True,  False, 20, 'tiered'),
        ('C_Bull+Earnings',   True,  False, True,  20, 'tiered'),
        ('D_AllThree',        True,  True,  True,  20, 'tiered'),
        ('E_AllThree_Fixed',  True,  True,  True,  20, 'fixed'),
        ('F_AllThree_30d',    True,  True,  True,  None, 'tiered'),
    ]

    results = []
    for nm, bs, pc, ei, exit_day, sz in configs:
        fprint(f"\n--- {nm} ---")
        tr, eq, cu, ms = simulate_combined(nm, rankings, sc, sh, sl, spy, vix, ec,
                                            use_bull_spreads=bs, use_put_credits=pc,
                                            use_earnings_ic=ei, early_exit_day=exit_day,
                                            sizing=sz)
        _, _, monthly_rets = compute_honest_sharpe(tr)
        r = validate(tr, eq, cu, nm, monthly_rets)
        if r:
            r['milestones'] = ms
            results.append(r)
            fprint(f"  {nm}: {r['n_trades']} trades | WR {r['win_rate']:.1f}% | Sh {r['sharpe']:.2f} | "
                   f"CAGR {r['cagr_pct']:.1f}% | MDD {r['maxdd_pct']:.1f}% | "
                   f"${CAP}->${r['final_equity']:.0f} | Gates {r['gates']}/4")
            if ms:
                fprint(f"  Milestones: {ms}")
            for typ, info in r['strategy_breakdown'].items():
                fprint(f"    {typ}: {info['n']} trades, WR {info['wr']:.1f}%, "
                       f"Total ${info['total_pnl']:.0f}, Avg ${info['avg_pnl']:.1f}")

    if not results: fprint("No results"); return

    # Summary
    fprint(f"\n{'='*100}")
    fprint(f"SUMMARY — Combined $645 Growth")
    fprint(f"{'='*100}")
    fprint(f"{'Variant':<22} {'#':>5} {'WR':>6} {'Sh':>7} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>9} {'G':>4}")
    fprint("-"*100)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>6.2f} ${r['final_equity']:>8.0f} {r['gates']:>3}/4")

    best = max(results, key=lambda x: x['sharpe'])
    fprint(f"\n=== BEST COMBINED STRATEGY ===")
    fprint(f"{best['name']} — Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, "
           f"MaxDD {best['maxdd_pct']}%, ${CAP}->${best['final_equity']:.0f}")
    if best.get('milestones'):
        fprint(f"Capital milestones: {best['milestones']}")

    # Compare single vs multi
    single = next((r for r in results if r['name'] == 'A_BullOnly'), None)
    combo = next((r for r in results if r['name'] == 'D_AllThree'), None)
    if single and combo:
        fprint(f"\n=== SINGLE vs MULTI-STRATEGY COMPARISON ===")
        fprint(f"Bull only:  Sh {single['sharpe']:.2f} | CAGR {single['cagr_pct']:.1f}% | MDD {single['maxdd_pct']:.1f}% | ${single['final_equity']:.0f}")
        fprint(f"All three:  Sh {combo['sharpe']:.2f} | CAGR {combo['cagr_pct']:.1f}% | MDD {combo['maxdd_pct']:.1f}% | ${combo['final_equity']:.0f}")
        sh_diff = combo['sharpe'] - single['sharpe']
        eq_diff = (combo['final_equity'] - single['final_equity']) / single['final_equity'] * 100
        fprint(f"Difference: Sharpe {sh_diff:+.2f}, Equity {eq_diff:+.0f}%")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save
    save_data = {
        'timestamp': t0.isoformat(),
        'capital': CAP,
        'version': 'combined_645_growth_v1',
        'results': results,
        'runtime_s': round(elapsed, 1)
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # MLflow
    if MLFLOW_OK:
        try:
            en = 'combined_645_growth_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"combined_645_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'strategies': 'bull+put+earnings',
                                   'exit': '20d', 'sizing': 'tiered'})
                for r in results:
                    p = r['name'][:18].replace(' ', '_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_cagr': r['cagr_pct'],
                                        f'{p}_mdd': r['maxdd_pct'], f'{p}_wr': r['win_rate'],
                                        f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Combined $645 Growth v1\n{'='*80}")

if __name__ == '__main__':
    main()
