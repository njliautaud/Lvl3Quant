#!/usr/bin/env python3
"""Triple Strategy Portfolio v1 — Three Validated Strategies, Single Equity Curve.

Combines ALL THREE validated $645-viable strategies:
  1. Sector Bull Spreads (VIX≥20): Buy call spreads on TOP LGBM-ranked sectors
  2. Sector Bear Put Spreads (VIX<20): Buy put spreads on BOTTOM-ranked sectors
  3. VIX Options Income (VIX>20): Sell VIX call spreads for mean-reversion income

Why this combination:
  - Bull+bear already gives us "always trading" — but VIX income adds a CONCURRENT
    income layer during VIX>20 periods, using a DIFFERENT instrument (VIX itself).
  - Low correlation: sector direction ≠ VIX mean-reversion.
  - VIX income is small ($50-150/contract) so doesn't blow budget.

Variants:
  A: Triple combined (all 3 active)
  B: Bull+Bear only (v1 baseline from prior research)
  C: Bull+VIX only (no bear side)
  D: Triple with VIX income at half-size
  E: Triple with momentum filter on bear side
  F: Triple with tiered sizing

$645 starting, 20d exit on sector, 14d on VIX, honest equity-based Sharpe.
"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import norm
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'triple_strategy_portfolio_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
QM_COLS = MOM_COLS + QUALITY_COLS

# ==================== DATA ====================
def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in SECTORS if c in volume.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index)
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]

# ==================== FEATURES ====================
def compute_features(px, vol_data, spy_slice):
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

def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ==================== LGBM RANKINGS ====================
def build_rankings(sc, sv, spy, rebal_dates):
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d, spy.iloc[:idx+1])
            if not feats: continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in QM_COLS:
        if c not in df.columns: df[c] = 0.0
    df[QM_COLS] = df[QM_COLS].fillna(0.0)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(12, len(dates)):
        td = dates[max(0,i-12):i]; test_date = dates[i]
        tr_df = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr_df) < 50: continue
        Xt = np.nan_to_num(tr_df[QM_COLS].values.astype(np.float32))
        yt = tr_df['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[QM_COLS].values.astype(np.float32))
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

# ==================== CONFLUENCE CHECKS ====================
def bull_confluence(tk, dt, sc, spy, vix):
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 63: return False
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False
    if float(px.iloc[-1]/px.iloc[-21]-1) > 0: signals.append(1)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]: signals.append(1)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        if not pd.isna(rsi.iloc[-1]) and float(rsi.iloc[-1]) < 80: signals.append(1)
    cv = float(vix.loc[dt]) if dt in vix.index else 20
    if cv > 15: signals.append(1)
    return len(signals) >= 2

def bear_confluence(tk, dt, sc, spy, vix):
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 63: return False
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False
    if float(px.iloc[-1]/px.iloc[-21]-1) < 0: signals.append(1)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] < sma50.iloc[-1]: signals.append(1)
    spy_s = spy.iloc[:idx+1]
    if len(spy_s) > 21:
        rel = px / spy_s
        if len(rel) > 21 and float(rel.iloc[-1]/rel.iloc[-21]-1) < 0: signals.append(1)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if 20 < r < 50: signals.append(1)
    return len(signals) >= 2

# ==================== VIX OPTION PRICING ====================
def vix_call_spread_trade(vix_val, dte=30, spread_width=5.0):
    """Price a VIX call spread (sell near, buy far) for mean-reversion income.

    When VIX > 20, sell a call spread expecting VIX to drop.
    Sell K1 call, buy K1+width call.
    Uses B-S with VIX-of-VIX ~80% (VIX is very volatile).
    """
    vvix = 0.80  # VIX-of-VIX typical
    T = dte / 365.0
    K1 = round(vix_val + 2)  # Sell slightly OTM
    K2 = K1 + spread_width   # Buy further OTM

    sell_premium = bs_price(vix_val, K1, T, vvix, opt='call')
    buy_premium = bs_price(vix_val, K2, T, vvix, opt='call')

    net_credit = sell_premium - buy_premium
    max_loss = spread_width - net_credit

    # Apply haircut for realism
    net_credit *= (1 - HAIRCUT)

    return {
        'K1': K1, 'K2': K2, 'credit': net_credit,
        'max_loss': max_loss, 'max_profit': net_credit,
        'cost_to_enter': max_loss * 100 + SPREAD_COMM,  # Margin requirement
        'credit_received': net_credit * 100 - SPREAD_COMM,
    }

def simulate_vix_exit(vix_series, entry_idx, entry_vix, trade, hold_days=14, profit_target=0.50):
    """Simulate VIX call spread exit. Profit if VIX drops, loss if VIX rises."""
    K1, K2 = trade['K1'], trade['K2']
    credit = trade['credit']
    vvix = 0.80

    for di in range(1, hold_days + 1):
        ci = entry_idx + di
        if ci >= len(vix_series): break

        cur_vix = float(vix_series.iloc[ci])
        days_left = max(1, hold_days - di)
        T = days_left / 365.0

        cur_sell = bs_price(cur_vix, K1, T, vvix, opt='call')
        cur_buy = bs_price(cur_vix, K2, T, vvix, opt='call')
        cur_spread = cur_sell - cur_buy

        # P&L = credit received - cost to close
        pnl_pts = credit - cur_spread

        # Early exit if hit profit target
        if pnl_pts >= credit * profit_target:
            return pnl_pts * 100 - SPREAD_COMM, ci

    # Hold to expiration — cash settlement
    ci = min(entry_idx + hold_days, len(vix_series) - 1)
    final_vix = float(vix_series.iloc[ci])
    intrinsic_sell = max(0, final_vix - K1)
    intrinsic_buy = max(0, final_vix - K2)
    pnl_pts = credit - (intrinsic_sell - intrinsic_buy)
    return pnl_pts * 100 - SPREAD_COMM, ci

# ==================== COMBINED SIMULATION ====================
def simulate(name, rankings, sc, sh, sl, spy, vix,
             use_bull=True, use_bear=True, use_vix=True,
             sizing='fixed', bear_size_mult=1.0, vix_size_mult=1.0,
             mom_filter_bear=False):

    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]
    bull_count, bear_count, vix_count = 0, 0, 0

    # Track VIX positions (avoid overlapping)
    vix_position_exit = -1

    # SPY 5d momentum for bear filter
    spy_mom_5d = spy.pct_change(5)

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])
        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull_regime = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue
        di_global = vix.index.get_loc(dt) if dt in vix.index else -1

        # Position sizing
        if sizing == 'tiered':
            if equity < 2000: base_pos = 200
            elif equity < 10000: base_pos = 500
            elif equity < 50000: base_pos = 1000
            else: base_pos = 2000
        else:
            base_pos = min(200, equity/3)

        if base_pos < 30: eq_curve.append(equity); continue

        # === VIX REGIME DETERMINES SECTOR DIRECTION ===
        is_bull_mode = cv >= 20

        # === STRATEGY 1: SECTOR BULL SPREADS (VIX >= 20) ===
        if is_bull_mode and use_bull:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            picks = [t for t, _ in ranked[:3]]
            max_pos = base_pos

            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                if not bull_confluence(tk, dt, sc, spy, vix): continue

                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + 20, len(sc)-1)
                K1, K2 = round(S), round(S*1.03)
                lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
                val = lp-sp; width = K2-K1
                cost = val*100+SPREAD_COMM; mx_prof = (width-val)*100-SPREAD_COMM
                if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

                pnl, aei = None, ei
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di))
                    ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    tm = np.sqrt(rd/30)
                    cur = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                    cp = cur - cost
                    if cp >= mx_prof*0.50: pnl = cp; aei = ci; break
                    if ci == ei:
                        pnl = cur - cost if rd > 0 else (max(0,Sc-K1)-max(0,Sc-K2))*100 - cost
                        aei = ci; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - cost

                equity += pnl; n_ent += 1; bull_count += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bull', 'strategy': 'sector_bull',
                               'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})

        # === STRATEGY 2: SECTOR BEAR PUT SPREADS (VIX < 20) ===
        elif not is_bull_mode and use_bear:
            # Optional momentum filter: skip if SPY 5d momentum is positive
            if mom_filter_bear and dt in spy_mom_5d.index:
                spy_5d = float(spy_mom_5d.loc[dt]) if not pd.isna(spy_mom_5d.loc[dt]) else 0
                if spy_5d > 0.02:  # Market strongly up, skip bear
                    eq_curve.append(equity); continue

            ranked = sorted(scores.items(), key=lambda x: x[1])
            picks = [t for t, _ in ranked[:3]]
            max_pos = base_pos * bear_size_mult

            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                if not bear_confluence(tk, dt, sc, spy, vix): continue

                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + 20, len(sc)-1)
                K1, K2 = round(S), round(S*0.97)
                if K2 >= K1: continue
                lp = atr_premium(S, K1, 30, av, cv, 'put')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'put')*(1-HAIRCUT)
                debit = lp-sp; width = K1-K2
                cost = debit*100+SPREAD_COMM; mx_prof = (width-debit)*100-SPREAD_COMM
                if cost <= 0 or cost > max_pos or cost > equity*0.40 or mx_prof <= 0: continue

                pnl, aei = None, ei
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di))
                    ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    tm = np.sqrt(rd/30)
                    intrinsic = (max(0,K1-Sc)-max(0,K2-Sc))*100
                    cur = intrinsic + ac*tm*0.3*100
                    cp = cur - cost
                    if cp >= mx_prof*0.50: pnl = cp; aei = ci; break
                    if ci == ei:
                        pnl = cur - cost if rd > 0 else intrinsic - cost
                        aei = ci; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,K1-Se)-max(0,K2-Se))*100 - cost

                equity += pnl; n_ent += 1; bear_count += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bear', 'strategy': 'sector_bear',
                               'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})

        # === STRATEGY 3: VIX CALL SPREAD INCOME (VIX > 20, concurrent with bull) ===
        if is_bull_mode and use_vix and cv > 20 and di_global > vix_position_exit:
            # Only enter if no existing VIX position AND budget allows
            vix_budget = base_pos * vix_size_mult

            trade_info = vix_call_spread_trade(cv, dte=30, spread_width=5.0)

            # Cost = margin requirement (max loss)
            vix_cost = trade_info['cost_to_enter']
            vix_credit = trade_info['credit_received']

            if vix_cost > 0 and vix_cost <= vix_budget and vix_cost <= equity * 0.25:
                pnl, exit_idx = simulate_vix_exit(
                    vix, di_global, cv, trade_info, hold_days=14, profit_target=0.50
                )

                vix_position_exit = exit_idx  # Block new VIX entries until this exits
                exit_date = str(vix.index[exit_idx].date()) if exit_idx < len(vix) else str(dt.date())

                equity += pnl; vix_count += 1
                trades.append({'entry': str(dt.date()), 'exit': exit_date,
                               'ticker': 'VIX', 'side': 'sell', 'strategy': 'vix_income',
                               'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': exit_idx - di_global if exit_idx >= 0 else 14,
                               'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})

        eq_curve.append(equity)

    return trades, equity, eq_curve, bull_count, bear_count, vix_count

# ==================== VALIDATION ====================
def compute_honest_sharpe(trades):
    if not trades: return 0.0, 0.0, []
    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['entry_dt'].dt.to_period('M')
    monthly = []
    for mo in sorted(tdf['month'].unique()):
        mt = tdf[tdf['month'] == mo]
        pnl = mt['pnl'].sum()
        eq = max(mt['equity_at_trade'].iloc[0] - mt['pnl'].iloc[0], 100)
        monthly.append(pnl / eq)
    rets = np.array(monthly)
    if len(rets) < 4: return 0.0, 0.0, rets
    sh = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(12))
    dn = rets[rets < 0]
    so = float(np.mean(rets) / (np.std(dn) + 1e-10) * np.sqrt(12)) if len(dn) > 1 else 0.0
    return sh, so, rets

def validate(trades, final_eq, eq_curve, name, bull_n, bear_n, vix_n):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    sh, so, rets = compute_honest_sharpe(trades)
    ny = max(len(rets)/12, 0.5)
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)

    # Regime analysis
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    # Strategy breakdown
    strats = {}
    for s in ['sector_bull', 'sector_bear', 'vix_income']:
        st = [t for t in trades if t.get('strategy') == s]
        if st:
            strats[s] = {
                'n': len(st), 'wr': sum(1 for t in st if t['win'])/len(st)*100,
                'pnl': sum(t['pnl'] for t in st),
                'avg_pnl': np.mean([t['pnl'] for t in st])
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

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1),
         'sharpe': round(sh,2), 'sortino': round(so,2),
         'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'final_equity': round(final_eq,2),
         'bull_trades': bull_n, 'bear_trades': bear_n, 'vix_trades': vix_n,
         'strategy_breakdown': strats,
         'regime_bull_wr': round(bw,1), 'regime_bear_wr': round(brw,1),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
         'h1_sh': round(h1,2), 'h2_sh': round(h2,2)}

    fprint(f"  {name}: {n} trades (Bull:{bull_n}/Bear:{bear_n}/VIX:{vix_n}) | "
           f"WR {wr:.1f}% | Sh {sh:.2f} | So {so:.2f} | CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | "
           f"PF {pf:.2f} | ${CAP}->${final_eq:.0f} | Gates {gates}/4")
    for sn, sd in strats.items():
        fprint(f"    {sn}: {sd['n']} trades, WR {sd['wr']:.0f}%, PnL ${sd['pnl']:.0f}, avg ${sd['avg_pnl']:.1f}")
    fprint(f"    G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'}")
    return r

# ==================== RANDOM DIRECTION CONTROL ====================
def simulate_random(name, rankings, sc, sh, sl, spy, vix):
    """Random direction control — same trades but random bull/bear assignment."""
    np.random.seed(42)
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])
        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull_regime = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue

        base_pos = min(200, equity/3)
        if base_pos < 30: eq_curve.append(equity); continue

        # RANDOM direction instead of VIX-based
        is_bull = np.random.random() > 0.5
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=is_bull)
        picks = [t for t, _ in ranked[:3]]

        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)
            ei = min(di + 20, len(sc)-1)

            if is_bull:
                K1, K2 = round(S), round(S*1.03)
                lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
                val = lp-sp; width = K2-K1
                cost = val*100+SPREAD_COMM; mx_prof = (width-val)*100-SPREAD_COMM
                if cost <= 0 or cost > base_pos or cost > equity*0.40: continue
                pnl = None
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di)); ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    cur = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*np.sqrt(rd/30)*0.3*100
                    if cur - cost >= mx_prof*0.50: pnl = cur - cost; break
                    if ci == ei: pnl = cur - cost if rd > 0 else (max(0,Sc-K1)-max(0,Sc-K2))*100 - cost; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - cost
            else:
                K1, K2 = round(S), round(S*0.97)
                if K2 >= K1: continue
                lp = atr_premium(S, K1, 30, av, cv, 'put')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'put')*(1-HAIRCUT)
                debit = lp-sp; width = K1-K2
                cost = debit*100+SPREAD_COMM; mx_prof = (width-debit)*100-SPREAD_COMM
                if cost <= 0 or cost > base_pos or cost > equity*0.40 or mx_prof <= 0: continue
                pnl = None
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di)); ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    intrinsic = (max(0,K1-Sc)-max(0,K2-Sc))*100
                    cur = intrinsic + ac*np.sqrt(rd/30)*0.3*100
                    if cur - cost >= mx_prof*0.50: pnl = cur - cost; break
                    if ci == ei: pnl = cur - cost if rd > 0 else intrinsic - cost; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,K1-Se)-max(0,K2-Se))*100 - cost

            equity += pnl; n_ent += 1
            trades.append({'entry': str(dt.date()), 'ticker': tk,
                           'side': 'bull' if is_bull else 'bear', 'strategy': 'random',
                           'pnl': round(pnl,2), 'win': pnl>0,
                           'regime': 'bull' if bull_regime else 'bear',
                           'vix': round(cv,1), 'equity_at_trade': round(equity,2)})
        eq_curve.append(equity)

    return trades, equity, eq_curve

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"Triple Strategy Portfolio v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*90}")
    fprint(f"Sector Bull (VIX>=20) + Sector Bear (VIX<20) + VIX Income (VIX>20)")
    fprint(f"{'='*90}")

    sc, sh, sl, sv, spy, vix = download_data()
    fprint(f"Data: {len(sc)} days")
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rankings = build_rankings(sc, sv, spy, bd)
    if not rankings: fprint("FATAL: No rankings"); return

    vix_above_20 = (vix >= 20).sum() / len(vix) * 100
    fprint(f"\nVIX stats: {vix_above_20:.1f}% of days VIX>=20, {100-vix_above_20:.1f}% VIX<20")

    configs = [
        # name, use_bull, use_bear, use_vix, sizing, bear_mult, vix_mult, mom_filter
        ('A_Triple_Full',        True,  True,  True,  'fixed',  1.0, 1.0, False),
        ('B_BullBear_Only',      True,  True,  False, 'fixed',  1.0, 1.0, False),
        ('C_BullVIX_Only',       True,  False, True,  'fixed',  1.0, 1.0, False),
        ('D_Triple_HalfVIX',     True,  True,  True,  'fixed',  1.0, 0.5, False),
        ('E_Triple_MomFilter',   True,  True,  True,  'fixed',  1.0, 1.0, True),
        ('F_Triple_Tiered',      True,  True,  True,  'tiered', 1.0, 1.0, False),
    ]

    results = []
    for nm, ub, ubr, uv, sz, bm, vm, mf in configs:
        fprint(f"\n--- {nm} ---")
        tr, eq, cu, bn, brn, vn = simulate(nm, rankings, sc, sh, sl, spy, vix,
                                            use_bull=ub, use_bear=ubr, use_vix=uv,
                                            sizing=sz, bear_size_mult=bm, vix_size_mult=vm,
                                            mom_filter_bear=mf)
        r = validate(tr, eq, cu, nm, bn, brn, vn)
        if r: results.append(r)

    # Random direction control
    fprint(f"\n--- RANDOM DIRECTION CONTROL ---")
    rtr, req, rcu = simulate_random('Random_Control', rankings, sc, sh, sl, spy, vix)
    if rtr:
        rsh, rso, rrets = compute_honest_sharpe(rtr)
        n = len(rtr); wins = sum(1 for t in rtr if t['win'])
        fprint(f"  Random: {n} trades | WR {wins/n*100:.1f}% | Sh {rsh:.2f} | ${CAP}->${req:.0f}")

    if not results: fprint("No results"); return

    # Summary
    fprint(f"\n{'='*120}")
    fprint(f"SUMMARY — Triple Strategy Portfolio v1")
    fprint(f"{'='*120}")
    fprint(f"{'Variant':<24} {'#':>5} {'B/S/V':>9} {'WR':>6} {'Sh':>7} {'So':>7} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>8} {'G':>4}")
    fprint("-"*120)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<24} {r['n_trades']:>5} {r['bull_trades']:>3}/{r['bear_trades']:<3}/{r.get('vix_trades',0):<3} "
               f"{r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['cagr_pct']:>6.1f}% "
               f"{r['maxdd_pct']:>6.1f}% {r['pf']:>6.2f} ${r['final_equity']:>7.0f} {r['gates']:>3}/4")

    # Triple vs Bull+Bear comparison
    triple = next((r for r in results if r['name']=='A_Triple_Full'), None)
    bb = next((r for r in results if r['name']=='B_BullBear_Only'), None)
    bv = next((r for r in results if r['name']=='C_BullVIX_Only'), None)
    if triple and bb:
        fprint(f"\n=== TRIPLE vs BULL+BEAR vs BULL+VIX ===")
        fprint(f"Bull+Bear:  Sh {bb['sharpe']:.2f} | CAGR {bb['cagr_pct']:.1f}% | "
               f"MDD {bb['maxdd_pct']:.1f}% | ${bb['final_equity']:.0f} | {bb['n_trades']} trades")
        fprint(f"Triple:     Sh {triple['sharpe']:.2f} | CAGR {triple['cagr_pct']:.1f}% | "
               f"MDD {triple['maxdd_pct']:.1f}% | ${triple['final_equity']:.0f} | {triple['n_trades']} trades")
        if bv:
            fprint(f"Bull+VIX:   Sh {bv['sharpe']:.2f} | CAGR {bv['cagr_pct']:.1f}% | "
                   f"MDD {bv['maxdd_pct']:.1f}% | ${bv['final_equity']:.0f} | {bv['n_trades']} trades")

        eq_diff = (triple['final_equity'] - bb['final_equity']) / max(bb['final_equity'], 1) * 100
        sh_diff = triple['sharpe'] - bb['sharpe']
        fprint(f"\nVIX income adds: Sharpe {sh_diff:+.2f}, Equity {eq_diff:+.1f}%")

    # Random control comparison
    if rtr:
        best = max(results, key=lambda x: x['sharpe'])
        fprint(f"\n=== RANDOM CONTROL ===")
        fprint(f"Random direction: Sh {rsh:.2f}, ${req:.0f}")
        fprint(f"Best strategy:    Sh {best['sharpe']:.2f}, ${best['final_equity']:.0f}")
        edge = (best['sharpe'] - rsh) / max(abs(rsh), 0.01) * 100
        fprint(f"VIX timing edge:  {edge:+.0f}% Sharpe improvement over random")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save
    save_data = {'timestamp': t0.isoformat(), 'capital': CAP,
                 'results': results, 'runtime_s': round(elapsed,1),
                 'random_sharpe': round(rsh,2) if rtr else None,
                 'random_final': round(req,2) if rtr else None}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'triple_strategy_portfolio_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"triple_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'strategy': 'bull+bear+vix', 'sector_exit': '20d', 'vix_hold': '14d'})
                for r in results:
                    p = r['name'][:20].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_wr': r['win_rate'],
                                        f'{p}_cagr': r['cagr_pct'], f'{p}_mdd': r['maxdd_pct'],
                                        f'{p}_gates': r['gates']})
                if rtr:
                    mlflow.log_metrics({'random_sharpe': round(rsh,2), 'random_final': round(req,2)})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*90}\nDONE — Triple Strategy Portfolio v1\n{'='*90}")

if __name__ == '__main__':
    main()
