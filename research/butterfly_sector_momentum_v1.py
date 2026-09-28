#!/usr/bin/env python3
"""
Butterfly Sector Momentum v1
==============================
Test long call butterflies on LightGBM-ranked sector ETFs.

Why butterflies?
- Very cheap entry ($30-80 per butterfly at $645 account)
- Defined max loss = debit paid
- Max profit when underlying lands at center strike at expiry
- Could work if our momentum model predicts direction + magnitude

Strategy: Buy call butterfly centered at predicted price target.
- Buy 1x lower strike call (ATM or slightly ITM)
- Sell 2x middle strike call (at predicted price)
- Buy 1x upper strike call (cap max profit)

If momentum ranking predicts +2-3% move over 30 days, center butterfly there.
Also test iron butterflies (sell ATM straddle + buy wings) for income.

HC compliance: 4-gate adversarial audit, permutation test, regime check.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json
import os
from scipy.stats import norm

try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    HAS_MLFLOW = True
except:
    HAS_MLFLOW = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

###############################################################################
# CONFIG
###############################################################################
UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
START_CAP = 645.0
COMMISSION_RT = 4.70
HAIRCUT = 0.15
RISK_FREE = 0.05

CONFIGS = {
    'A_CallBfly_30d_Top3': {
        'type': 'call_butterfly', 'dte': 30, 'wing_pct': 2.0, 'target_pct': 2.0,
        'top_k': 3, 'rebal_days': 21, 'vix_min': 0, 'train_periods': 12,
        'desc': 'Call butterfly, 30d, targeting +2% move'
    },
    'B_CallBfly_30d_VIX20': {
        'type': 'call_butterfly', 'dte': 30, 'wing_pct': 2.0, 'target_pct': 2.0,
        'top_k': 3, 'rebal_days': 21, 'vix_min': 20, 'train_periods': 12,
        'desc': 'Call butterfly + VIX>20 filter'
    },
    'C_IronBfly_30d_Income': {
        'type': 'iron_butterfly', 'dte': 30, 'wing_pct': 3.0, 'target_pct': 0,
        'top_k': 3, 'rebal_days': 21, 'vix_min': 20, 'train_periods': 12,
        'desc': 'Iron butterfly (income), sell ATM + buy wings'
    },
    'D_CallBfly_45d_Wide': {
        'type': 'call_butterfly', 'dte': 45, 'wing_pct': 3.0, 'target_pct': 3.0,
        'top_k': 3, 'rebal_days': 30, 'vix_min': 0, 'train_periods': 12,
        'desc': 'Wider wings, 45d DTE, +3% target'
    },
    'E_CallBfly_30d_Top1': {
        'type': 'call_butterfly', 'dte': 30, 'wing_pct': 2.0, 'target_pct': 2.0,
        'top_k': 1, 'rebal_days': 21, 'vix_min': 20, 'train_periods': 12,
        'desc': 'Single best sector, concentrated'
    },
    'F_BrokenWing_30d': {
        'type': 'broken_wing', 'dte': 30, 'wing_pct': 2.0, 'target_pct': 2.0,
        'top_k': 3, 'rebal_days': 21, 'vix_min': 0, 'train_periods': 12,
        'desc': 'Broken-wing butterfly (asymmetric risk)'
    },
}

###############################################################################
# BLACK-SCHOLES
###############################################################################
def bs_call(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

###############################################################################
# BUTTERFLY PNL CALCULATORS
###############################################################################
def call_butterfly_pnl(S_entry, S_exit, K_low, K_mid, K_high, T_entry, T_exit, sigma):
    """Long call butterfly: +1 K_low, -2 K_mid, +1 K_high"""
    # Entry
    T = max(T_entry, 1/365)
    c_low_entry = bs_call(S_entry, K_low, T, sigma)
    c_mid_entry = bs_call(S_entry, K_mid, T, sigma)
    c_high_entry = bs_call(S_entry, K_high, T, sigma)
    debit = c_low_entry - 2 * c_mid_entry + c_high_entry  # net debit to enter

    # Exit
    if T_exit <= 0.01:
        c_low_exit = max(S_exit - K_low, 0)
        c_mid_exit = max(S_exit - K_mid, 0)
        c_high_exit = max(S_exit - K_high, 0)
    else:
        T_rem = max(T_exit, 0.001)
        sig_exit = sigma * 0.95
        c_low_exit = bs_call(S_exit, K_low, T_rem, sig_exit)
        c_mid_exit = bs_call(S_exit, K_mid, T_rem, sig_exit)
        c_high_exit = bs_call(S_exit, K_high, T_rem, sig_exit)

    credit = c_low_exit - 2 * c_mid_exit + c_high_exit

    # Apply haircut
    debit *= (1 + HAIRCUT)
    credit *= (1 - HAIRCUT)

    pnl = (credit - debit) * 100  # per contract
    cost = debit * 100

    # 2 RT commissions (4 legs)
    pnl -= COMMISSION_RT * 2

    return pnl, max(cost, 1)

def iron_butterfly_pnl(S_entry, S_exit, K_mid, K_put_wing, K_call_wing, T_entry, T_exit, sigma):
    """Iron butterfly: sell ATM put + sell ATM call, buy put wing + buy call wing"""
    T = max(T_entry, 1/365)

    # Entry: sell ATM straddle, buy wings
    p_mid_entry = bs_put(S_entry, K_mid, T, sigma)
    c_mid_entry = bs_call(S_entry, K_mid, T, sigma)
    p_wing_entry = bs_put(S_entry, K_put_wing, T, sigma)
    c_wing_entry = bs_call(S_entry, K_call_wing, T, sigma)

    # Net credit = sell straddle - buy wings
    net_credit = (p_mid_entry + c_mid_entry) - (p_wing_entry + c_wing_entry)

    # Exit
    if T_exit <= 0.01:
        p_mid_exit = max(K_mid - S_exit, 0)
        c_mid_exit = max(S_exit - K_mid, 0)
        p_wing_exit = max(K_put_wing - S_exit, 0)
        c_wing_exit = max(S_exit - K_call_wing, 0)
    else:
        T_rem = max(T_exit, 0.001)
        sig_exit = sigma * 0.95
        p_mid_exit = bs_put(S_exit, K_mid, T_rem, sig_exit)
        c_mid_exit = bs_call(S_exit, K_mid, T_rem, sig_exit)
        p_wing_exit = bs_put(S_exit, K_put_wing, T_rem, sig_exit)
        c_wing_exit = bs_call(S_exit, K_call_wing, T_rem, sig_exit)

    net_debit_exit = (p_mid_exit + c_mid_exit) - (p_wing_exit + c_wing_exit)

    # Haircut: credit received is less, debit to close is more
    net_credit *= (1 - HAIRCUT)
    net_debit_exit *= (1 + HAIRCUT)

    pnl = (net_credit - net_debit_exit) * 100
    cost = (K_mid - K_put_wing) * 100 - net_credit * 100  # max loss = wing width - credit
    cost = max(cost, 1)

    pnl -= COMMISSION_RT * 2  # 4 legs

    return pnl, cost

def broken_wing_pnl(S_entry, S_exit, K_low, K_mid, K_high, T_entry, T_exit, sigma):
    """Broken-wing butterfly: +1 K_low, -2 K_mid, +1 K_high where K_high is further OTM"""
    # Wider upside wing = potentially zero cost or small credit
    K_high_adjusted = K_mid + 1.5 * (K_mid - K_low)  # 1.5x wider on upside

    T = max(T_entry, 1/365)
    c_low_entry = bs_call(S_entry, K_low, T, sigma)
    c_mid_entry = bs_call(S_entry, K_mid, T, sigma)
    c_high_entry = bs_call(S_entry, K_high_adjusted, T, sigma)
    debit = c_low_entry - 2 * c_mid_entry + c_high_entry

    if T_exit <= 0.01:
        c_low_exit = max(S_exit - K_low, 0)
        c_mid_exit = max(S_exit - K_mid, 0)
        c_high_exit = max(S_exit - K_high_adjusted, 0)
    else:
        T_rem = max(T_exit, 0.001)
        sig_exit = sigma * 0.95
        c_low_exit = bs_call(S_exit, K_low, T_rem, sig_exit)
        c_mid_exit = bs_call(S_exit, K_mid, T_rem, sig_exit)
        c_high_exit = bs_call(S_exit, K_high_adjusted, T_rem, sig_exit)

    credit = c_low_exit - 2 * c_mid_exit + c_high_exit

    debit *= (1 + HAIRCUT)
    credit *= (1 - HAIRCUT)

    pnl = (credit - debit) * 100
    cost = max(debit * 100, 1)
    pnl -= COMMISSION_RT * 2

    return pnl, cost

###############################################################################
# DATA + FEATURES (reuse from weekly DTE script)
###############################################################################
def load_data():
    import yfinance as yf
    cache_dir = '/home/jupiter/Lvl3Quant/research/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'sector_etf_weekly_data.parquet')
    vix_file = os.path.join(cache_dir, 'vix_data.parquet')

    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        if age_hours < 24:
            df = pd.read_parquet(cache_file)
            vix = pd.read_parquet(vix_file) if os.path.exists(vix_file) else None
            return df, vix

    tickers = UNIVERSE + ['^VIX']
    data = yf.download(tickers, start='2010-01-01', progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    vix_col = '^VIX' if '^VIX' in close.columns else None
    if vix_col:
        vix = close[[vix_col]].rename(columns={vix_col: 'VIX'})
        close = close.drop(columns=[vix_col])
    else:
        vix = None

    close.to_parquet(cache_file)
    if vix is not None:
        vix.to_parquet(vix_file)

    return close, vix

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def build_features(close, vix, lookback=12):
    # Monthly resampling for 30d DTE strategies
    monthly_close = close.resample('ME').last().dropna(how='all')
    monthly_vix = vix.resample('ME').last().dropna() if vix is not None else None

    features_list = []
    for ticker in UNIVERSE:
        if ticker not in monthly_close.columns:
            continue
        px = monthly_close[ticker].dropna()
        if len(px) < lookback + 5:
            continue

        feat = pd.DataFrame(index=px.index)
        feat['ticker'] = ticker
        feat['price'] = px

        for m in [1, 2, 3, 6, 12]:
            feat[f'ret_{m}m'] = px.pct_change(m)

        ret_1m = px.pct_change()
        for m in [3, 6, 12]:
            feat[f'vol_{m}m'] = ret_1m.rolling(m).std()

        feat['sharpe_6m'] = feat['ret_6m'] / (feat['vol_6m'] + 1e-8)
        feat['sharpe_12m'] = feat['ret_12m'] / (feat['vol_12m'] + 1e-8)
        feat['rsi_14'] = compute_rsi(px, 14)

        for m in [5, 10, 20]:
            sma = px.rolling(m).mean()
            feat[f'above_sma{m}'] = (px > sma).astype(float)

        ew_ret = monthly_close[UNIVERSE].pct_change(3).mean(axis=1)
        feat['rel_strength_3m'] = feat['ret_3m'] - ew_ret

        rolling_max = px.rolling(12).max()
        dd = (px - rolling_max) / rolling_max
        feat['max_dd_12m'] = dd.rolling(12).min()
        feat['calmar_12m'] = feat['ret_12m'] / (-feat['max_dd_12m'] + 1e-8)

        if monthly_vix is not None and 'VIX' in monthly_vix.columns:
            vix_a = monthly_vix['VIX'].reindex(feat.index, method='ffill')
            feat['vix'] = vix_a
            feat['vix_z'] = (vix_a - vix_a.rolling(24).mean()) / (vix_a.rolling(24).std() + 1e-8)

        feat['target'] = px.pct_change().shift(-1)
        features_list.append(feat)

    return pd.concat(features_list), monthly_close

###############################################################################
# BACKTEST
###############################################################################
def run_backtest(features, monthly_close, vix, config, name):
    dte = config['dte']
    wing_pct = config['wing_pct'] / 100.0
    target_pct = config['target_pct'] / 100.0
    top_k = config['top_k']
    rebal_days = config['rebal_days']
    vix_min = config['vix_min']
    train_periods = config['train_periods']
    trade_type = config['type']

    feat_cols = [c for c in features.columns if c not in ['ticker', 'price', 'target', 'vix']]
    dates = sorted(features.index.unique())

    equity = START_CAP
    trades = []
    equity_curve = []

    for i in range(train_periods, len(dates) - 1):
        date = dates[i]

        # VIX filter
        if vix_min > 0 and vix is not None:
            vix_val = vix.loc[:date, 'VIX']
            current_vix = vix_val.iloc[-1] if len(vix_val) > 0 else 15
            if current_vix < vix_min:
                equity_curve.append({'date': date, 'equity': equity})
                continue
        else:
            current_vix = 15

        # Train
        train_start = dates[max(0, i - train_periods)]
        train_mask = (features.index >= train_start) & (features.index < date)
        train_data = features[train_mask].copy()
        pred_mask = features.index == date
        pred_data = features[pred_mask].copy()

        if len(train_data) < 15 or len(pred_data) == 0:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        train_X = train_data[feat_cols].replace([np.inf, -np.inf], np.nan)
        train_y = train_data['target'].values
        pred_X = pred_data[feat_cols].replace([np.inf, -np.inf], np.nan)

        valid = ~(train_X.isna().any(axis=1) | np.isnan(train_y))
        train_X = train_X[valid]
        train_y = train_y[valid]

        if len(train_X) < 10:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        pred_X = pred_X.fillna(0)

        try:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1, random_state=42
            )
            model.fit(train_X, train_y)
            predictions = model.predict(pred_X)
        except:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        pred_df = pred_data[['ticker', 'price']].copy()
        pred_df['pred_ret'] = predictions
        pred_df = pred_df.sort_values('pred_ret', ascending=False)
        top_sectors = pred_df.head(top_k)

        next_date = dates[i + 1]

        for _, row in top_sectors.iterrows():
            ticker = row['ticker']
            S_entry = row['price']
            pred_ret = row['pred_ret']

            if ticker in monthly_close.columns and next_date in monthly_close.index:
                S_exit = monthly_close.loc[next_date, ticker]
            else:
                continue

            if pd.isna(S_entry) or pd.isna(S_exit) or S_entry <= 0:
                continue

            # IV estimate
            ticker_data = features[(features['ticker'] == ticker) & (features.index <= date)]
            if 'vol_3m' in ticker_data.columns and len(ticker_data) > 0:
                rv = ticker_data['vol_3m'].iloc[-1]
                if pd.isna(rv) or rv <= 0:
                    rv = 0.06
                iv = rv * np.sqrt(12) * 1.1
            else:
                iv = 0.25

            T_entry = dte / 365.0
            T_exit = max(0, (dte - rebal_days) / 365.0)

            # Position sizing
            max_trade = min(equity * 0.30 / top_k, 200)

            if trade_type == 'call_butterfly':
                K_mid = S_entry * (1 + target_pct)
                K_low = K_mid - S_entry * wing_pct
                K_high = K_mid + S_entry * wing_pct
                pnl, cost = call_butterfly_pnl(S_entry, S_exit, K_low, K_mid, K_high, T_entry, T_exit, iv)

            elif trade_type == 'iron_butterfly':
                K_mid = S_entry  # ATM
                K_put_wing = S_entry * (1 - wing_pct)
                K_call_wing = S_entry * (1 + wing_pct)
                pnl, cost = iron_butterfly_pnl(S_entry, S_exit, K_mid, K_put_wing, K_call_wing, T_entry, T_exit, iv)

            elif trade_type == 'broken_wing':
                K_mid = S_entry * (1 + target_pct)
                K_low = K_mid - S_entry * wing_pct
                K_high = K_mid + S_entry * wing_pct
                pnl, cost = broken_wing_pnl(S_entry, S_exit, K_low, K_mid, K_high, T_entry, T_exit, iv)

            else:
                continue

            if cost <= 0:
                continue

            n_contracts = max(1, int(max_trade / (cost / 100 + 0.01)))
            if n_contracts * cost / 100 > equity * 0.4:
                n_contracts = max(1, int(equity * 0.4 / (cost / 100 + 0.01)))

            actual_pnl = pnl * n_contracts
            actual_cost = cost * n_contracts / 100

            if actual_pnl < -actual_cost:
                actual_pnl = -actual_cost

            equity += actual_pnl

            # Regime
            mkt = monthly_close[UNIVERSE].mean(axis=1)
            spy_ret = None
            if date in mkt.index and next_date in mkt.index:
                spy_ret = (mkt.loc[next_date] - mkt.loc[date]) / mkt.loc[date]

            trades.append({
                'date': date, 'exit_date': next_date, 'ticker': ticker,
                'entry_price': S_entry, 'exit_price': S_exit,
                'pnl': actual_pnl, 'cost': actual_cost,
                'n_contracts': n_contracts, 'pred_ret': pred_ret,
                'vix': current_vix, 'type': trade_type,
                'regime': 'bull' if (spy_ret is not None and spy_ret > 0) else 'bear',
                'equity_after': equity,
            })

        equity_curve.append({'date': date, 'equity': equity})

    return trades, equity_curve

###############################################################################
# ADVERSARIAL AUDIT
###############################################################################
def adversarial_audit(trades, equity_curve, name):
    if len(trades) < 10:
        return {'name': name, 'n_trades': len(trades), 'gates_passed': 0, 'error': 'Too few trades'}

    trade_df = pd.DataFrame(trades)
    pnls = trade_df['pnl'].values
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n * 100
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls < 0].mean() if (pnls < 0).sum() > 0 else 0
    pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).sum() != 0 else 999

    eq_df = pd.DataFrame(equity_curve)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')
    monthly_eq = eq_df['equity'].resample('ME').last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    if len(monthly_ret) > 1:
        sharpe = monthly_ret.mean() / (monthly_ret.std() + 1e-8) * np.sqrt(12)
        neg_ret = monthly_ret[monthly_ret < 0]
        sortino = monthly_ret.mean() / (neg_ret.std() + 1e-8) * np.sqrt(12) if len(neg_ret) > 0 else sharpe * 1.5
    else:
        sharpe = sortino = 0

    final_eq = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else START_CAP
    years = max((eq_df.index[-1] - eq_df.index[0]).days / 365.25, 0.1)
    cagr = (final_eq / START_CAP) ** (1/years) - 1
    maxdd = ((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min()
    calmar = cagr / (-maxdd + 1e-8) if maxdd < 0 else 0

    # G1: Permutation
    n_perms = 500
    perm_sharpes = []
    for _ in range(n_perms):
        sh = pnls.copy()
        np.random.shuffle(sh)
        eq_p = np.cumsum(sh) + START_CAP
        nm = max(1, len(eq_p) // 3)
        chunks = np.array_split(eq_p, nm)
        mr = []
        prev = START_CAP
        for c in chunks:
            if len(c) > 0:
                mr.append((c[-1] - prev) / prev)
                prev = c[-1]
        if len(mr) > 1:
            a = np.array(mr)
            perm_sharpes.append(a.mean() / (a.std() + 1e-8) * np.sqrt(12))
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe) if perm_sharpes else 1.0
    g1 = perm_p < 0.05

    # G2: R1 regime
    bull = trade_df[trade_df['regime'] == 'bull']['pnl'].values
    bear = trade_df[trade_df['regime'] == 'bear']['pnl'].values
    if len(bull) > 5 and len(bear) > 5:
        bs = bull.mean() / (bull.std() + 1e-8) * np.sqrt(12)
        brs = bear.mean() / (bear.std() + 1e-8) * np.sqrt(12)
        r1 = abs(bs - brs) / max(abs(bs), abs(brs), 1e-8)
        bull_wr = (bull > 0).mean() * 100
        bear_wr = (bear > 0).mean() * 100
    else:
        bs = brs = sharpe
        r1 = 0
        bull_wr = bear_wr = wr
    g2 = r1 < 0.50

    # G3: Sub-period
    t = len(trade_df) // 3
    ss = []
    for s, e in [(0, t), (t, 2*t), (2*t, len(trade_df))]:
        sp = trade_df.iloc[s:e]['pnl'].values
        if len(sp) > 3:
            ss.append(sp.mean() / (sp.std() + 1e-8) * np.sqrt(12))
    g3 = len(ss) >= 2 and all(x > 0 for x in ss)

    # G4: Outlier removal
    p5, p95 = np.percentile(pnls, [5, 95])
    trimmed = pnls[(pnls >= p5) & (pnls <= p95)]
    if len(trimmed) > 3:
        ts = trimmed.mean() / (trimmed.std() + 1e-8) * np.sqrt(12)
        g4 = ts > 0
    else:
        ts = 0
        g4 = False

    return {
        'name': name, 'n_trades': n, 'win_rate': round(wr, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'total_pnl': round(pnls.sum(), 2), 'final_equity': round(final_eq, 2),
        'cagr_pct': round(cagr * 100, 1), 'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2), 'maxdd_pct': round(maxdd * 100, 1),
        'calmar': round(calmar, 2), 'pf': round(pf, 2),
        'r1_gap': round(r1, 3), 'bull_wr': round(bull_wr, 1), 'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull), 'bear_trades': len(bear),
        'bull_sharpe': round(bs, 2), 'bear_sharpe': round(brs, 2),
        'perm_p': round(perm_p, 4), 'g1_pass': g1, 'g2_pass': g2,
        'g3_pass': g3, 'g4_pass': g4, 'sub_sharpes': [round(x, 2) for x in ss],
        'gates_passed': sum([g1, g2, g3, g4]),
    }

###############################################################################
# MAIN
###############################################################################
def main():
    print("=" * 70)
    print("Butterfly Sector Momentum v1")
    print("=" * 70)
    t0 = datetime.now()

    close, vix = load_data()
    print(f"Data: {close.shape[0]} days, {close.shape[1]} tickers")

    features, monthly_close = build_features(close, vix)
    print(f"Features: {len(features)} rows")

    if HAS_MLFLOW:
        exp = mlflow.set_experiment("butterfly_sector_momentum_v1")
        exp_id = exp.experiment_id

    all_results = []
    best_result = None
    best_sharpe = -999

    for name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name} — {config['desc']}")
        print(f"{'='*60}")

        trades, eq_curve = run_backtest(features, monthly_close, vix, config, name)

        if len(trades) < 10:
            print(f"  SKIP: Only {len(trades)} trades")
            all_results.append({'name': name, 'n_trades': len(trades), 'gates_passed': 0, 'error': 'Too few trades'})
            continue

        result = adversarial_audit(trades, eq_curve, name)
        all_results.append(result)

        gates = f"{result['gates_passed']}/4"
        status = "✅" if result['gates_passed'] == 4 else "⚠️" if result['gates_passed'] >= 2 else "❌"

        print(f"  {status} {gates} | Sharpe {result['sharpe']:.2f} | WR {result['win_rate']:.1f}% | "
              f"CAGR {result['cagr_pct']:.1f}% | MDD {result['maxdd_pct']:.1f}% | "
              f"PF {result['pf']:.2f} | {result['n_trades']} trades | ${START_CAP}→${result['final_equity']:,.0f}")
        print(f"  R1 gap {result['r1_gap']:.3f} | Bull WR {result['bull_wr']:.1f}% vs Bear WR {result['bear_wr']:.1f}% | "
              f"Perm p={result['perm_p']:.4f}")

        if HAS_MLFLOW:
            with mlflow.start_run(experiment_id=exp_id, run_name=name):
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_params({k: str(v) for k, v in config.items()})

        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_result = result

    runtime = (datetime.now() - t0).total_seconds()

    print(f"\n{'='*70}")
    print(f"SUMMARY — Butterfly Sector Momentum v1")
    print(f"{'='*70}")

    p4 = [r for r in all_results if r.get('gates_passed', 0) == 4]
    p2 = [r for r in all_results if r.get('gates_passed', 0) >= 2]
    print(f"Configs: {len(all_results)} | 4/4 gates: {len(p4)} | ≥2/4: {len(p2)} | Runtime: {runtime:.0f}s")

    if best_result:
        print(f"\nBEST: {best_result['name']}")
        print(f"  Sharpe {best_result['sharpe']:.2f} | CAGR {best_result['cagr_pct']:.1f}% | "
              f"MDD {best_result['maxdd_pct']:.1f}% | WR {best_result['win_rate']:.1f}%")

    # Save
    findings_dir = '/home/jupiter/Lvl3Quant/research/findings'
    os.makedirs(findings_dir, exist_ok=True)
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'butterfly_sector_momentum_v1',
        'universe': UNIVERSE, 'capital': START_CAP,
        'configs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in CONFIGS.items()},
        'results': all_results, 'best': best_result, 'runtime_s': runtime,
    }
    with open(os.path.join(findings_dir, 'butterfly_sector_momentum_v1_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved.")
    return all_results

if __name__ == '__main__':
    main()
