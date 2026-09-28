#!/usr/bin/env python3
"""
Sector Iron Condor Income v1
==============================
NEW STRATEGY: Sell iron condors on LOW-momentum (range-bound) sectors.

Hypothesis: Our LightGBM momentum model identifies which sectors will move.
The BOTTOM-ranked sectors are predicted to be range-bound → ideal for selling premium.

Key advantages:
- Income generating (positive theta)
- Anti-correlated with bull call spread strategy
- Range-bound sectors exist in BOTH bull and bear markets → better R1
- Uses the same signal (inverted) so no new model needed

Compares to SPY iron condors (validated Sharpe 3.55) to see if sector selection adds value.

$645 starting capital, realistic B-S pricing, 15% haircut, commissions.
HC #428 compliant: 4-gate adversarial audit.
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
except:
    print("LightGBM required"); exit(1)

UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
START_CAP = 645.0
COMMISSION_RT = 4.70
HAIRCUT = 0.15

CONFIGS = {
    'A_Bottom3_30d_3pct': {
        'bottom_k': 3, 'dte': 30, 'wing_pct': 3.0, 'delta_target': 0.15,
        'vix_min': 0, 'train_periods': 12, 'tp_pct': 0.50,
        'desc': 'IC on bottom-3 momentum, 30d, 3% wings'
    },
    'B_Bottom3_30d_VIX20': {
        'bottom_k': 3, 'dte': 30, 'wing_pct': 3.0, 'delta_target': 0.15,
        'vix_min': 20, 'train_periods': 12, 'tp_pct': 0.50,
        'desc': 'IC on bottom-3 + VIX>20 (higher premiums)'
    },
    'C_Bottom2_45d': {
        'bottom_k': 2, 'dte': 45, 'wing_pct': 3.0, 'delta_target': 0.15,
        'vix_min': 0, 'train_periods': 12, 'tp_pct': 0.50,
        'desc': 'IC on bottom-2, 45d DTE (more theta)'
    },
    'D_Bottom3_Wide5pct': {
        'bottom_k': 3, 'dte': 30, 'wing_pct': 5.0, 'delta_target': 0.10,
        'vix_min': 0, 'train_periods': 12, 'tp_pct': 0.50,
        'desc': 'Wider 5% wings (higher WR, lower premium)'
    },
    'E_Random3_Baseline': {
        'bottom_k': 3, 'dte': 30, 'wing_pct': 3.0, 'delta_target': 0.15,
        'vix_min': 0, 'train_periods': 12, 'tp_pct': 0.50,
        'use_random': True,
        'desc': 'CONTROL: random sector selection (no momentum ranking)'
    },
    'F_Middle3': {
        'bottom_k': 3, 'dte': 30, 'wing_pct': 3.0, 'delta_target': 0.15,
        'vix_min': 0, 'train_periods': 12, 'tp_pct': 0.50,
        'use_middle': True,
        'desc': 'CONTROL: middle-ranked sectors (neither top nor bottom)'
    },
    'G_Combined_BullSpread_IC': {
        'bottom_k': 3, 'dte': 30, 'wing_pct': 3.0, 'delta_target': 0.15,
        'vix_min': 20, 'train_periods': 12, 'tp_pct': 0.50,
        'add_bull_spreads': True, 'top_k': 3,
        'desc': 'COMBINED: bull spreads on top-3 + IC on bottom-3'
    },
}

###############################################################################
# BLACK-SCHOLES
###############################################################################
def bs_call(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def iron_condor_trade(S_entry, S_exit, iv, dte, wing_pct, delta_target, T_exit_days=None):
    """
    Iron condor: sell put spread below + sell call spread above.
    Returns (pnl_per_contract, max_risk, credit_received).
    """
    wing = S_entry * wing_pct / 100.0
    T = dte / 365.0

    # Short strikes: target delta ~0.15 (roughly 1 stdev away)
    # Approximate: K_put_short = S * (1 - delta_target * sigma * sqrt(T))
    offset = iv * np.sqrt(T) * norm.ppf(1 - delta_target)
    K_put_short = S_entry * (1 - offset)
    K_put_long = K_put_short - wing
    K_call_short = S_entry * (1 + offset)
    K_call_long = K_call_short + wing

    # Entry premiums
    p_short = bs_put(S_entry, K_put_short, T, iv)
    p_long = bs_put(S_entry, K_put_long, T, iv)
    c_short = bs_call(S_entry, K_call_short, T, iv)
    c_long = bs_call(S_entry, K_call_long, T, iv)

    # Net credit
    credit = (p_short - p_long) + (c_short - c_long)
    credit *= (1 - HAIRCUT)  # slippage on entry

    # Exit
    if T_exit_days is not None:
        T_exit = max((dte - T_exit_days) / 365.0, 0.001)
    else:
        T_exit = 0.001  # near expiry

    iv_exit = iv * 0.95  # slight vol decay

    if T_exit <= 0.01:
        p_short_exit = max(K_put_short - S_exit, 0)
        p_long_exit = max(K_put_long - S_exit, 0)
        c_short_exit = max(S_exit - K_call_short, 0)
        c_long_exit = max(S_exit - K_call_long, 0)
    else:
        p_short_exit = bs_put(S_exit, K_put_short, T_exit, iv_exit)
        p_long_exit = bs_put(S_exit, K_put_long, T_exit, iv_exit)
        c_short_exit = bs_call(S_exit, K_call_short, T_exit, iv_exit)
        c_long_exit = bs_call(S_exit, K_call_long, T_exit, iv_exit)

    # Debit to close
    debit = (p_short_exit - p_long_exit) + (c_short_exit - c_long_exit)
    debit *= (1 + HAIRCUT)  # slippage on exit

    pnl = (credit - debit) * 100  # per contract
    pnl -= COMMISSION_RT * 2  # 4 legs = 2 RT commissions

    max_risk = wing * 100 - credit * 100  # max loss = wing width - credit
    max_risk = max(max_risk, 1)

    return pnl, max_risk, credit * 100

def bull_spread_trade(S_entry, S_exit, iv, dte, spread_pct=3.0):
    """Simple bull call spread for combined strategy."""
    T = dte / 365.0
    K_long = S_entry * (1 - spread_pct/200.0)
    K_short = S_entry * (1 + spread_pct/200.0)

    entry_val = bs_call(S_entry, K_long, T, iv) - bs_call(S_entry, K_short, T, iv)
    entry_cost = entry_val * (1 + HAIRCUT) * 100

    # At expiry
    exit_val = (max(S_exit - K_long, 0) - max(S_exit - K_short, 0)) * 100

    pnl = exit_val * (1 - HAIRCUT) - entry_cost - COMMISSION_RT
    return pnl, max(entry_cost, 1)

###############################################################################
# DATA + FEATURES
###############################################################################
def load_data():
    import yfinance as yf
    cache_dir = '/home/jupiter/Lvl3Quant/research/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'sector_etf_weekly_data.parquet')
    vix_file = os.path.join(cache_dir, 'vix_data.parquet')

    if os.path.exists(cache_file):
        if (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600 < 24:
            df = pd.read_parquet(cache_file)
            vix = pd.read_parquet(vix_file) if os.path.exists(vix_file) else None
            return df, vix

    tickers = UNIVERSE + ['^VIX']
    data = yf.download(tickers, start='2010-01-01', progress=False, auto_adjust=True)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    vix_col = '^VIX' if '^VIX' in close.columns else None
    if vix_col:
        vix = close[[vix_col]].rename(columns={vix_col: 'VIX'})
        close = close.drop(columns=[vix_col])
    else:
        vix = None
    close.to_parquet(cache_file)
    if vix is not None: vix.to_parquet(vix_file)
    return close, vix

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def build_features(close, vix):
    monthly = close.resample('ME').last().dropna(how='all')
    monthly_vix = vix.resample('ME').last().dropna() if vix is not None else None
    features_list = []

    for ticker in UNIVERSE:
        if ticker not in monthly.columns: continue
        px = monthly[ticker].dropna()
        if len(px) < 15: continue

        feat = pd.DataFrame(index=px.index)
        feat['ticker'] = ticker
        feat['price'] = px
        for m in [1,2,3,6,12]: feat[f'ret_{m}m'] = px.pct_change(m)
        ret1 = px.pct_change()
        for m in [3,6,12]: feat[f'vol_{m}m'] = ret1.rolling(m).std()
        feat['sharpe_6m'] = feat['ret_6m'] / (feat['vol_6m'] + 1e-8)
        feat['sharpe_12m'] = feat['ret_12m'] / (feat['vol_12m'] + 1e-8)
        feat['rsi_14'] = compute_rsi(px, 14)
        for m in [5,10,20]:
            sma = px.rolling(m).mean()
            feat[f'above_sma{m}'] = (px > sma).astype(float)
        ew = monthly[UNIVERSE].pct_change(3).mean(axis=1)
        feat['rel_strength_3m'] = feat['ret_3m'] - ew
        rm = px.rolling(12).max()
        dd = (px - rm) / rm
        feat['max_dd_12m'] = dd.rolling(12).min()
        feat['calmar_12m'] = feat['ret_12m'] / (-feat['max_dd_12m'] + 1e-8)
        if monthly_vix is not None and 'VIX' in monthly_vix.columns:
            v = monthly_vix['VIX'].reindex(feat.index, method='ffill')
            feat['vix'] = v
        feat['target'] = px.pct_change().shift(-1)
        features_list.append(feat)

    return pd.concat(features_list), monthly

###############################################################################
# BACKTEST
###############################################################################
def run_backtest(features, monthly_close, vix, config, name):
    bottom_k = config['bottom_k']
    dte = config['dte']
    wing_pct = config['wing_pct']
    delta_target = config['delta_target']
    vix_min = config['vix_min']
    train_periods = config['train_periods']
    tp_pct = config['tp_pct']
    use_random = config.get('use_random', False)
    use_middle = config.get('use_middle', False)
    add_bull = config.get('add_bull_spreads', False)
    top_k = config.get('top_k', 3)

    feat_cols = [c for c in features.columns if c not in ['ticker','price','target','vix']]
    dates = sorted(features.index.unique())

    equity = START_CAP
    trades = []
    equity_curve = []
    rng = np.random.RandomState(42)

    for i in range(train_periods, len(dates) - 1):
        date = dates[i]

        # VIX filter
        if vix_min > 0 and vix is not None:
            vv = vix.loc[:date, 'VIX']
            current_vix = vv.iloc[-1] if len(vv) > 0 else 15
            if current_vix < vix_min:
                equity_curve.append({'date': date, 'equity': equity})
                continue
        else:
            current_vix = 15

        # Train LightGBM (unless random)
        pred_mask = features.index == date
        pred_data = features[pred_mask].copy()

        if len(pred_data) == 0:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        if use_random:
            available = pred_data['ticker'].unique()
            selected_bottom = rng.choice(available, min(bottom_k, len(available)), replace=False)
            pred_df = pred_data[['ticker','price']].copy()
            pred_df['pred_ret'] = 0
        else:
            train_start = dates[max(0, i - train_periods)]
            train_mask = (features.index >= train_start) & (features.index < date)
            train_data = features[train_mask].copy()

            if len(train_data) < 15:
                equity_curve.append({'date': date, 'equity': equity})
                continue

            train_X = train_data[feat_cols].replace([np.inf,-np.inf], np.nan)
            train_y = train_data['target'].values
            valid = ~(train_X.isna().any(axis=1) | np.isnan(train_y))
            train_X = train_X[valid]; train_y = train_y[valid]

            if len(train_X) < 10:
                equity_curve.append({'date': date, 'equity': equity})
                continue

            pred_X = pred_data[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)

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

            pred_df = pred_data[['ticker','price']].copy()
            pred_df['pred_ret'] = predictions
            pred_df = pred_df.sort_values('pred_ret', ascending=True)  # ascending = bottom first

            if use_middle:
                n = len(pred_df)
                mid_start = n // 2 - bottom_k // 2
                selected_bottom = pred_df.iloc[mid_start:mid_start+bottom_k]['ticker'].values
            else:
                selected_bottom = pred_df.head(bottom_k)['ticker'].values

        next_date = dates[i + 1]

        # Market return for regime
        mkt = monthly_close[UNIVERSE].mean(axis=1)
        mkt_ret = None
        if date in mkt.index and next_date in mkt.index:
            mkt_ret = (mkt.loc[next_date] - mkt.loc[date]) / mkt.loc[date]

        regime = 'bull' if (mkt_ret is not None and mkt_ret > 0) else 'bear'

        # SELL ICs on bottom-ranked sectors
        for ticker in selected_bottom:
            row = pred_df[pred_df['ticker'] == ticker]
            if len(row) == 0: continue

            S_entry = row['price'].values[0]
            if ticker in monthly_close.columns and next_date in monthly_close.index:
                S_exit = monthly_close.loc[next_date, ticker]
            else:
                continue

            if pd.isna(S_entry) or pd.isna(S_exit) or S_entry <= 0: continue

            # IV estimate
            ticker_feat = features[(features['ticker']==ticker)&(features.index<=date)]
            if 'vol_3m' in ticker_feat.columns and len(ticker_feat) > 0:
                rv = ticker_feat['vol_3m'].iloc[-1]
                iv = (rv if not pd.isna(rv) and rv > 0 else 0.06) * np.sqrt(12) * 1.1
            else:
                iv = 0.25

            pnl, max_risk, credit = iron_condor_trade(
                S_entry, S_exit, iv, dte, wing_pct, delta_target
            )

            # Position sizing
            max_trade = min(equity * 0.25 / bottom_k, 200)
            n_contracts = max(1, int(max_trade / (max_risk / 100 + 0.01)))
            if n_contracts * max_risk / 100 > equity * 0.4:
                n_contracts = max(1, int(equity * 0.4 / (max_risk / 100 + 0.01)))

            actual_pnl = pnl * n_contracts
            if actual_pnl < -max_risk * n_contracts / 100:
                actual_pnl = -max_risk * n_contracts / 100

            # Take profit: if P&L > tp_pct * credit, take it
            if tp_pct is not None and actual_pnl > tp_pct * credit * n_contracts / 100:
                actual_pnl = tp_pct * credit * n_contracts / 100

            equity += actual_pnl

            trades.append({
                'date': date, 'exit_date': next_date, 'ticker': ticker,
                'type': 'iron_condor', 'pnl': actual_pnl,
                'cost': max_risk * n_contracts / 100,
                'credit': credit * n_contracts / 100,
                'n_contracts': n_contracts, 'vix': current_vix,
                'regime': regime, 'equity_after': equity,
            })

        # COMBINED: also buy bull spreads on top sectors
        if add_bull and not use_random:
            top_sectors = pred_df.sort_values('pred_ret', ascending=False).head(top_k)
            for _, row in top_sectors.iterrows():
                ticker = row['ticker']
                S_entry = row['price']
                if ticker in monthly_close.columns and next_date in monthly_close.index:
                    S_exit = monthly_close.loc[next_date, ticker]
                else:
                    continue
                if pd.isna(S_entry) or pd.isna(S_exit) or S_entry <= 0: continue

                ticker_feat = features[(features['ticker']==ticker)&(features.index<=date)]
                if 'vol_3m' in ticker_feat.columns and len(ticker_feat) > 0:
                    rv = ticker_feat['vol_3m'].iloc[-1]
                    iv = (rv if not pd.isna(rv) and rv > 0 else 0.06) * np.sqrt(12) * 1.1
                else:
                    iv = 0.25

                pnl, cost = bull_spread_trade(S_entry, S_exit, iv, dte)

                max_trade = min(equity * 0.20 / top_k, 150)
                n_contracts = max(1, int(max_trade / (cost / 100 + 0.01)))
                actual_pnl = pnl * n_contracts
                actual_cost = cost * n_contracts / 100
                if actual_pnl < -actual_cost:
                    actual_pnl = -actual_cost

                equity += actual_pnl

                trades.append({
                    'date': date, 'exit_date': next_date, 'ticker': ticker,
                    'type': 'bull_spread', 'pnl': actual_pnl,
                    'cost': actual_cost, 'n_contracts': n_contracts,
                    'vix': current_vix, 'regime': regime, 'equity_after': equity,
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
    n = len(pnls); wins = (pnls > 0).sum()
    wr = wins/n*100
    pf = abs(pnls[pnls>0].sum() / pnls[pnls<0].sum()) if (pnls<0).sum() != 0 else 999

    eq_df = pd.DataFrame(equity_curve)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')
    monthly_eq = eq_df['equity'].resample('ME').last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    if len(monthly_ret) > 1:
        sharpe = monthly_ret.mean() / (monthly_ret.std() + 1e-8) * np.sqrt(12)
        neg = monthly_ret[monthly_ret < 0]
        sortino = monthly_ret.mean() / (neg.std() + 1e-8) * np.sqrt(12) if len(neg) > 0 else sharpe * 1.5
    else:
        sharpe = sortino = 0

    final_eq = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else START_CAP
    years = max((eq_df.index[-1] - eq_df.index[0]).days / 365.25, 0.1)
    cagr = (final_eq / START_CAP) ** (1/years) - 1
    maxdd = ((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min()

    # G1: Permutation
    perm_sharpes = []
    for _ in range(500):
        sh = pnls.copy(); np.random.shuffle(sh)
        eq_p = np.cumsum(sh) + START_CAP
        nm = max(1, len(eq_p) // 3)
        chunks = np.array_split(eq_p, nm); mr = []; prev = START_CAP
        for c in chunks:
            if len(c) > 0: mr.append((c[-1]-prev)/prev); prev = c[-1]
        if len(mr) > 1:
            a = np.array(mr)
            perm_sharpes.append(a.mean()/(a.std()+1e-8)*np.sqrt(12))
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe) if perm_sharpes else 1.0
    g1 = perm_p < 0.05

    # G2: R1
    bull = trade_df[trade_df['regime']=='bull']['pnl'].values
    bear = trade_df[trade_df['regime']=='bear']['pnl'].values
    if len(bull) > 5 and len(bear) > 5:
        bs = bull.mean()/(bull.std()+1e-8)*np.sqrt(12)
        brs = bear.mean()/(bear.std()+1e-8)*np.sqrt(12)
        r1 = abs(bs-brs)/max(abs(bs),abs(brs),1e-8)
        bull_wr = (bull>0).mean()*100; bear_wr = (bear>0).mean()*100
    else:
        bs=brs=sharpe; r1=0; bull_wr=bear_wr=wr
    g2 = r1 < 0.50

    # G3: Sub-period
    t = len(trade_df)//3; ss = []
    for s,e in [(0,t),(t,2*t),(2*t,len(trade_df))]:
        sp = trade_df.iloc[s:e]['pnl'].values
        if len(sp) > 3: ss.append(sp.mean()/(sp.std()+1e-8)*np.sqrt(12))
    g3 = len(ss) >= 2 and all(x > 0 for x in ss)

    # G4: Outlier removal
    p5,p95 = np.percentile(pnls,[5,95])
    tr = pnls[(pnls>=p5)&(pnls<=p95)]
    g4 = (tr.mean()/(tr.std()+1e-8)*np.sqrt(12) > 0) if len(tr) > 3 else False

    # Type breakdown
    if 'type' in trade_df.columns:
        type_stats = {}
        for t_type in trade_df['type'].unique():
            t_pnls = trade_df[trade_df['type']==t_type]['pnl'].values
            type_stats[t_type] = {
                'n': len(t_pnls),
                'wr': round((t_pnls>0).mean()*100, 1),
                'avg_pnl': round(t_pnls.mean(), 2),
                'total_pnl': round(t_pnls.sum(), 2),
            }
    else:
        type_stats = {}

    return {
        'name': name, 'n_trades': n, 'win_rate': round(wr,1),
        'avg_win': round(pnls[pnls>0].mean(),2) if wins > 0 else 0,
        'avg_loss': round(pnls[pnls<0].mean(),2) if (pnls<0).sum() > 0 else 0,
        'total_pnl': round(pnls.sum(),2), 'final_equity': round(final_eq,2),
        'cagr_pct': round(cagr*100,1), 'sharpe': round(sharpe,2),
        'sortino': round(sortino,2), 'maxdd_pct': round(maxdd*100,1),
        'pf': round(pf,2), 'r1_gap': round(r1,3),
        'bull_wr': round(bull_wr,1), 'bear_wr': round(bear_wr,1),
        'bull_trades': len(bull), 'bear_trades': len(bear),
        'bull_sharpe': round(bs,2), 'bear_sharpe': round(brs,2),
        'perm_p': round(perm_p,4), 'g1_pass': g1, 'g2_pass': g2,
        'g3_pass': g3, 'g4_pass': g4,
        'sub_sharpes': [round(x,2) for x in ss],
        'gates_passed': sum([g1,g2,g3,g4]),
        'type_breakdown': type_stats,
    }

###############################################################################
# MAIN
###############################################################################
def main():
    print("="*70)
    print("Sector Iron Condor Income v1")
    print("="*70)
    t0 = datetime.now()

    close, vix = load_data()
    features, monthly_close = build_features(close, vix)
    print(f"Data: {close.shape[0]} days, {close.shape[1]} tickers, {len(features)} feature rows")

    if HAS_MLFLOW:
        exp = mlflow.set_experiment("sector_ic_income_v1")
        exp_id = exp.experiment_id

    all_results = []
    best = None; best_sharpe = -999

    for name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name} — {config['desc']}")

        trades, eq_curve = run_backtest(features, monthly_close, vix, config, name)

        if len(trades) < 10:
            print(f"  SKIP: Only {len(trades)} trades")
            all_results.append({'name': name, 'n_trades': len(trades), 'gates_passed': 0})
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
        if result.get('type_breakdown'):
            for tt, ts in result['type_breakdown'].items():
                print(f"  [{tt}] {ts['n']} trades, WR {ts['wr']}%, avg ${ts['avg_pnl']:.0f}")

        if HAS_MLFLOW:
            with mlflow.start_run(experiment_id=exp_id, run_name=name):
                for k,v in result.items():
                    if isinstance(v, (int,float)): mlflow.log_metric(k,v)
                mlflow.log_params({k: str(v) for k,v in config.items() if k != 'desc'})

        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']; best = result

    runtime = (datetime.now() - t0).total_seconds()

    print(f"\n{'='*70}")
    print(f"SUMMARY — Sector IC Income v1")
    print(f"{'='*70}")

    p4 = sum(1 for r in all_results if r.get('gates_passed',0) == 4)
    p2 = sum(1 for r in all_results if r.get('gates_passed',0) >= 2)
    print(f"Configs: {len(all_results)} | 4/4: {p4} | ≥2/4: {p2} | Runtime: {runtime:.0f}s")

    if best:
        print(f"\nBEST: {best['name']} — Sharpe {best['sharpe']:.2f}, CAGR {best['cagr_pct']:.1f}%, "
              f"MDD {best['maxdd_pct']:.1f}%, WR {best['win_rate']:.1f}%")

    # Key question: does ML ranking add value vs random for IC selling?
    ml_results = [r for r in all_results if 'Random' not in r.get('name','') and 'Middle' not in r.get('name','')]
    random_result = next((r for r in all_results if 'Random' in r.get('name','')), None)
    middle_result = next((r for r in all_results if 'Middle' in r.get('name','')), None)

    if random_result and ml_results:
        best_ml = max(ml_results, key=lambda x: x.get('sharpe', -999))
        print(f"\n--- ML vs RANDOM ---")
        print(f"  Best ML (bottom-ranked): Sharpe {best_ml['sharpe']:.2f}, WR {best_ml['win_rate']:.1f}%")
        print(f"  Random selection:        Sharpe {random_result.get('sharpe',0):.2f}, WR {random_result.get('win_rate',0):.1f}%")
        if middle_result:
            print(f"  Middle-ranked:           Sharpe {middle_result.get('sharpe',0):.2f}, WR {middle_result.get('win_rate',0):.1f}%")
        delta = best_ml.get('sharpe',0) - random_result.get('sharpe',0)
        print(f"  ML advantage: {delta:+.2f} Sharpe")
        if delta <= 0:
            print(f"  ⚠️ ML ranking does NOT add value for IC selling. Edge is structural.")

    # Save
    findings_dir = '/home/jupiter/Lvl3Quant/research/findings'
    os.makedirs(findings_dir, exist_ok=True)
    with open(os.path.join(findings_dir, 'sector_ic_income_v1_results.json'), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(), 'experiment': 'sector_ic_income_v1',
            'results': all_results, 'best': best, 'runtime_s': runtime,
        }, f, indent=2, default=str)

    print(f"\nResults saved.")
    return all_results

if __name__ == '__main__':
    main()
