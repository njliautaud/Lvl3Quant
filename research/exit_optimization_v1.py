#!/usr/bin/env python3
"""
Exit Optimization for Sector Bull Call Spreads v1
===================================================
Our sector bull spread strategy (Sharpe 1.5-4.2, 4/4 gates) currently holds
to monthly rebalancing. This tests if EARLY EXIT rules improve risk-adjusted returns:

1. Take-profit: Close at 50%, 75%, 100% of max spread value
2. Stop-loss: Close at -50%, -75% of cost
3. Time-based: Close at 15d, 20d (not full 30d hold)
4. Trailing stop: Close if P&L drops X% from peak

Uses DAILY price data for intra-month exit decisions (not just monthly resampling).
This should give more realistic results than the monthly-resampled backtests.

HC #428 compliant: 4-gate adversarial audit.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
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
    print("LightGBM required")
    exit(1)

###############################################################################
# CONFIG
###############################################################################
UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
START_CAP = 645.0
COMMISSION_RT = 4.70
HAIRCUT = 0.15
SPREAD_PCT = 0.03  # 3% spread width

CONFIGS = {
    'A_Hold30d_Baseline': {
        'tp_pct': None, 'sl_pct': None, 'max_hold': 30, 'trailing_pct': None,
        'desc': 'Baseline: hold full 30 days (current strategy)'
    },
    'B_TP50_SL75': {
        'tp_pct': 0.50, 'sl_pct': -0.75, 'max_hold': 30, 'trailing_pct': None,
        'desc': 'Take 50% profit, stop at 75% loss'
    },
    'C_TP75_SL50': {
        'tp_pct': 0.75, 'sl_pct': -0.50, 'max_hold': 30, 'trailing_pct': None,
        'desc': 'Take 75% profit, stop at 50% loss'
    },
    'D_TP50_NoStop': {
        'tp_pct': 0.50, 'sl_pct': None, 'max_hold': 30, 'trailing_pct': None,
        'desc': 'Take 50% profit, no stop loss'
    },
    'E_TimeExit_20d': {
        'tp_pct': None, 'sl_pct': None, 'max_hold': 20, 'trailing_pct': None,
        'desc': 'Exit after 20 days (capture theta, avoid expiry risk)'
    },
    'F_Trailing_30pct': {
        'tp_pct': None, 'sl_pct': None, 'max_hold': 30, 'trailing_pct': 0.30,
        'desc': 'Trailing stop: exit if P&L drops 30% from peak'
    },
    'G_TP50_SL50_20d': {
        'tp_pct': 0.50, 'sl_pct': -0.50, 'max_hold': 20, 'trailing_pct': None,
        'desc': 'Aggressive: 50% TP, 50% SL, 20d max hold'
    },
    'H_TP100_SL50': {
        'tp_pct': 1.00, 'sl_pct': -0.50, 'max_hold': 30, 'trailing_pct': None,
        'desc': 'Full profit target, tight stop'
    },
}

###############################################################################
# BS + SPREAD VALUATION
###############################################################################
def bs_call(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def spread_value(S, K_long, K_short, T, sigma):
    """Current value of bull call spread."""
    if T <= 0.001:
        return max(S - K_long, 0) - max(S - K_short, 0)
    return bs_call(S, K_long, T, sigma) - bs_call(S, K_short, T, sigma)

###############################################################################
# DATA
###############################################################################
def load_daily_data():
    import yfinance as yf
    cache_dir = '/home/jupiter/Lvl3Quant/research/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'sector_etf_daily_data.parquet')
    vix_file = os.path.join(cache_dir, 'vix_daily_data.parquet')

    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (datetime.now().timestamp() - mtime) / 3600 < 24:
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

def build_monthly_features(daily_close, vix):
    """Build monthly features for LightGBM sector ranking (same as validated strategy)."""
    monthly = daily_close.resample('ME').last().dropna(how='all')
    monthly_vix = vix.resample('ME').last().dropna() if vix is not None else None

    features_list = []
    for ticker in UNIVERSE:
        if ticker not in monthly.columns:
            continue
        px = monthly[ticker].dropna()
        if len(px) < 15:
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
        ew = monthly[UNIVERSE].pct_change(3).mean(axis=1)
        feat['rel_strength_3m'] = feat['ret_3m'] - ew
        rm = px.rolling(12).max()
        dd = (px - rm) / rm
        feat['max_dd_12m'] = dd.rolling(12).min()
        feat['calmar_12m'] = feat['ret_12m'] / (-feat['max_dd_12m'] + 1e-8)
        if monthly_vix is not None and 'VIX' in monthly_vix.columns:
            v = monthly_vix['VIX'].reindex(feat.index, method='ffill')
            feat['vix'] = v
            feat['vix_z'] = (v - v.rolling(24).mean()) / (v.rolling(24).std() + 1e-8)
        feat['target'] = px.pct_change().shift(-1)
        features_list.append(feat)

    return pd.concat(features_list)

###############################################################################
# BACKTEST WITH DAILY EXIT MONITORING
###############################################################################
def run_backtest(daily_close, vix, features, config, name):
    """Walk-forward with daily exit monitoring."""

    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']
    max_hold = config['max_hold']
    trailing_pct = config['trailing_pct']

    feat_cols = [c for c in features.columns if c not in ['ticker', 'price', 'target', 'vix']]
    dates = sorted(features.index.unique())

    equity = START_CAP
    trades = []
    equity_curve = []
    active_positions = []

    train_periods = 12
    top_k = 3
    vix_min = 20

    for i in range(train_periods, len(dates) - 1):
        entry_month_end = dates[i]

        # VIX filter
        if vix is not None:
            vv = vix.loc[:entry_month_end, 'VIX']
            current_vix = vv.iloc[-1] if len(vv) > 0 else 15
            if current_vix < vix_min:
                equity_curve.append({'date': entry_month_end, 'equity': equity})
                continue
        else:
            current_vix = 15

        # Train LightGBM
        train_start = dates[max(0, i - train_periods)]
        train_mask = (features.index >= train_start) & (features.index < entry_month_end)
        train_data = features[train_mask].copy()
        pred_mask = features.index == entry_month_end
        pred_data = features[pred_mask].copy()

        if len(train_data) < 15 or len(pred_data) == 0:
            equity_curve.append({'date': entry_month_end, 'equity': equity})
            continue

        train_X = train_data[feat_cols].replace([np.inf, -np.inf], np.nan)
        train_y = train_data['target'].values
        valid = ~(train_X.isna().any(axis=1) | np.isnan(train_y))
        train_X = train_X[valid]
        train_y = train_y[valid]

        if len(train_X) < 10:
            equity_curve.append({'date': entry_month_end, 'equity': equity})
            continue

        pred_X = pred_data[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        try:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1, random_state=42
            )
            model.fit(train_X, train_y)
            predictions = model.predict(pred_X)
        except:
            equity_curve.append({'date': entry_month_end, 'equity': equity})
            continue

        pred_df = pred_data[['ticker', 'price']].copy()
        pred_df['pred_ret'] = predictions
        top_sectors = pred_df.sort_values('pred_ret', ascending=False).head(top_k)

        # Entry: first trading day after month-end
        # Find entry date in daily data
        entry_candidates = daily_close.index[daily_close.index > entry_month_end]
        if len(entry_candidates) == 0:
            continue
        entry_date = entry_candidates[0]

        next_month_end = dates[i + 1]

        for _, row in top_sectors.iterrows():
            ticker = row['ticker']

            if ticker not in daily_close.columns:
                continue

            S_entry = daily_close.loc[entry_date, ticker] if entry_date in daily_close.index else None
            if S_entry is None or pd.isna(S_entry):
                continue

            # IV estimate
            if ticker in daily_close.columns:
                hist = daily_close[ticker].loc[:entry_date].tail(60)
                if len(hist) > 10:
                    rv = hist.pct_change().std() * np.sqrt(252)
                    iv = rv * 1.1
                else:
                    iv = 0.25
            else:
                iv = 0.25

            # Strikes
            K_long = S_entry * (1 - SPREAD_PCT / 2)
            K_short = S_entry * (1 + SPREAD_PCT / 2)

            # Entry value
            T_entry = 30 / 365.0
            entry_value = spread_value(S_entry, K_long, K_short, T_entry, iv)
            entry_cost = entry_value * (1 + HAIRCUT) * 100  # cost per contract

            if entry_cost <= 0:
                continue

            # Max spread value = K_short - K_long (at expiry, if deep ITM)
            max_spread = (K_short - K_long) * 100

            # Position sizing
            max_trade = min(equity * 0.30 / top_k, 200)
            n_contracts = max(1, int(max_trade / (entry_cost / 100 + 0.01)))

            # DAILY EXIT MONITORING
            exit_date = None
            exit_price = None
            exit_reason = 'expiry'
            peak_pnl = 0

            # Get daily prices for the hold period
            hold_start = entry_date
            hold_end = daily_close.index[daily_close.index <= entry_date + timedelta(days=max_hold + 5)]

            for d_idx, d in enumerate(daily_close.index):
                if d <= entry_date:
                    continue
                if d > entry_date + timedelta(days=max_hold + 5):
                    break

                days_held = (d - entry_date).days
                if days_held > max_hold:
                    exit_date = d
                    exit_reason = 'max_hold'
                    break

                if ticker not in daily_close.columns or d not in daily_close.index:
                    continue

                S_now = daily_close.loc[d, ticker]
                if pd.isna(S_now):
                    continue

                T_remaining = max((max_hold - days_held) / 365.0, 0.001)
                current_value = spread_value(S_now, K_long, K_short, T_remaining, iv)
                current_value_adj = current_value * (1 - HAIRCUT) * 100

                # P&L per contract
                unrealized_pnl = current_value_adj - entry_cost

                # Track peak
                if unrealized_pnl > peak_pnl:
                    peak_pnl = unrealized_pnl

                # Take profit check
                if tp_pct is not None:
                    target_pnl = tp_pct * (max_spread - entry_cost)
                    if unrealized_pnl >= target_pnl:
                        exit_date = d
                        exit_reason = 'take_profit'
                        exit_price = S_now
                        break

                # Stop loss check
                if sl_pct is not None:
                    stop_pnl = sl_pct * entry_cost  # sl_pct is negative
                    if unrealized_pnl <= stop_pnl:
                        exit_date = d
                        exit_reason = 'stop_loss'
                        exit_price = S_now
                        break

                # Trailing stop check
                if trailing_pct is not None and peak_pnl > 0:
                    if unrealized_pnl < peak_pnl * (1 - trailing_pct):
                        exit_date = d
                        exit_reason = 'trailing_stop'
                        exit_price = S_now
                        break

            # If no exit triggered, use max_hold date
            if exit_date is None:
                possible_exits = daily_close.index[daily_close.index >= entry_date + timedelta(days=max_hold)]
                if len(possible_exits) > 0:
                    exit_date = possible_exits[0]
                else:
                    continue

            if exit_price is None:
                if exit_date in daily_close.index and ticker in daily_close.columns:
                    exit_price = daily_close.loc[exit_date, ticker]
                else:
                    continue

            if pd.isna(exit_price):
                continue

            # Calculate actual exit PnL
            days_held = (exit_date - entry_date).days
            T_exit = max((30 - days_held) / 365.0, 0)

            if T_exit <= 0.001:
                exit_value = (max(exit_price - K_long, 0) - max(exit_price - K_short, 0)) * 100
            else:
                exit_value = spread_value(exit_price, K_long, K_short, T_exit, iv) * (1 - HAIRCUT) * 100

            pnl_per_contract = exit_value - entry_cost - COMMISSION_RT
            total_pnl = pnl_per_contract * n_contracts

            # Cap loss at cost
            if total_pnl < -entry_cost * n_contracts / 100:
                total_pnl = -entry_cost * n_contracts / 100

            equity += total_pnl

            # Regime
            mkt = daily_close[UNIVERSE].mean(axis=1)
            if entry_date in mkt.index and exit_date in mkt.index:
                mkt_ret = (mkt.loc[exit_date] - mkt.loc[entry_date]) / mkt.loc[entry_date]
                regime = 'bull' if mkt_ret > 0 else 'bear'
            else:
                regime = 'unknown'

            trades.append({
                'date': entry_date, 'exit_date': exit_date,
                'ticker': ticker, 'pnl': total_pnl,
                'cost': entry_cost * n_contracts / 100,
                'n_contracts': n_contracts,
                'days_held': days_held, 'exit_reason': exit_reason,
                'vix': current_vix, 'regime': regime,
                'equity_after': equity,
            })

        equity_curve.append({'date': entry_month_end, 'equity': equity})

    return trades, equity_curve

###############################################################################
# ADVERSARIAL AUDIT (same 4-gate)
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
        neg = monthly_ret[monthly_ret < 0]
        sortino = monthly_ret.mean() / (neg.std() + 1e-8) * np.sqrt(12) if len(neg) > 0 else sharpe * 1.5
    else:
        sharpe = sortino = 0

    final_eq = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else START_CAP
    years = max((eq_df.index[-1] - eq_df.index[0]).days / 365.25, 0.1)
    cagr = (final_eq / START_CAP) ** (1/years) - 1
    maxdd = ((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min()
    calmar = cagr / (-maxdd + 1e-8) if maxdd < 0 else 0

    # G1: Permutation
    perm_sharpes = []
    for _ in range(500):
        sh = pnls.copy(); np.random.shuffle(sh)
        eq_p = np.cumsum(sh) + START_CAP
        nm = max(1, len(eq_p) // 3)
        chunks = np.array_split(eq_p, nm)
        mr = []; prev = START_CAP
        for c in chunks:
            if len(c) > 0: mr.append((c[-1]-prev)/prev); prev = c[-1]
        if len(mr) > 1:
            a = np.array(mr)
            perm_sharpes.append(a.mean()/(a.std()+1e-8)*np.sqrt(12))
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe) if perm_sharpes else 1.0
    g1 = perm_p < 0.05

    # G2: R1
    bull = trade_df[trade_df['regime'] == 'bull']['pnl'].values
    bear = trade_df[trade_df['regime'] == 'bear']['pnl'].values
    if len(bull) > 5 and len(bear) > 5:
        bs = bull.mean()/(bull.std()+1e-8)*np.sqrt(12)
        brs = bear.mean()/(bear.std()+1e-8)*np.sqrt(12)
        r1 = abs(bs-brs)/max(abs(bs),abs(brs),1e-8)
        bull_wr = (bull>0).mean()*100
        bear_wr = (bear>0).mean()*100
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

    # Exit reason breakdown
    exit_reasons = trade_df['exit_reason'].value_counts().to_dict() if 'exit_reason' in trade_df.columns else {}
    avg_hold = trade_df['days_held'].mean() if 'days_held' in trade_df.columns else 30

    return {
        'name': name, 'n_trades': n, 'win_rate': round(wr,1),
        'avg_win': round(avg_win,2), 'avg_loss': round(avg_loss,2),
        'total_pnl': round(pnls.sum(),2), 'final_equity': round(final_eq,2),
        'cagr_pct': round(cagr*100,1), 'sharpe': round(sharpe,2),
        'sortino': round(sortino,2), 'maxdd_pct': round(maxdd*100,1),
        'calmar': round(calmar,2), 'pf': round(pf,2),
        'r1_gap': round(r1,3), 'bull_wr': round(bull_wr,1), 'bear_wr': round(bear_wr,1),
        'bull_trades': len(bull), 'bear_trades': len(bear),
        'perm_p': round(perm_p,4), 'g1_pass': g1, 'g2_pass': g2,
        'g3_pass': g3, 'g4_pass': g4,
        'sub_sharpes': [round(x,2) for x in ss],
        'gates_passed': sum([g1,g2,g3,g4]),
        'exit_reasons': exit_reasons,
        'avg_hold_days': round(avg_hold,1),
    }

###############################################################################
# MAIN
###############################################################################
def main():
    print("="*70)
    print("Exit Optimization for Sector Bull Call Spreads v1")
    print("="*70)
    t0 = datetime.now()

    daily_close, vix = load_daily_data()
    print(f"Daily data: {daily_close.shape[0]} days, {daily_close.shape[1]} tickers")

    features = build_monthly_features(daily_close, vix)
    print(f"Monthly features: {len(features)} rows")

    if HAS_MLFLOW:
        exp = mlflow.set_experiment("exit_optimization_v1")
        exp_id = exp.experiment_id

    all_results = []
    best_result = None
    best_sharpe = -999

    for name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name} — {config['desc']}")

        trades, eq_curve = run_backtest(daily_close, vix, features, config, name)

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
        print(f"  R1 gap {result['r1_gap']:.3f} | Perm p={result['perm_p']:.4f} | "
              f"Avg hold {result['avg_hold_days']:.1f}d")
        print(f"  Exit reasons: {result.get('exit_reasons', {})}")

        if HAS_MLFLOW:
            with mlflow.start_run(experiment_id=exp_id, run_name=name):
                for k,v in result.items():
                    if isinstance(v, (int,float)):
                        mlflow.log_metric(k,v)
                mlflow.log_params({k: str(v) for k,v in config.items()})

        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_result = result

    runtime = (datetime.now() - t0).total_seconds()

    print(f"\n{'='*70}")
    print(f"SUMMARY — Exit Optimization v1")
    print(f"{'='*70}")

    p4 = sum(1 for r in all_results if r.get('gates_passed',0) == 4)
    print(f"Configs: {len(all_results)} | 4/4 gates: {p4} | Runtime: {runtime:.0f}s")

    if best_result:
        print(f"\nBEST: {best_result['name']}")
        print(f"  Sharpe {best_result['sharpe']:.2f} | CAGR {best_result['cagr_pct']:.1f}% | "
              f"MDD {best_result['maxdd_pct']:.1f}% | WR {best_result['win_rate']:.1f}%")

    # Compare exit strategies
    print(f"\n--- EXIT STRATEGY COMPARISON ---")
    for r in sorted(all_results, key=lambda x: x.get('sharpe', -999), reverse=True):
        if 'error' not in r:
            delta = r['sharpe'] - (all_results[0].get('sharpe', 0) if all_results else 0)
            print(f"  {r['name']:25s} | Sharpe {r['sharpe']:5.2f} | MDD {r['maxdd_pct']:6.1f}% | "
                  f"WR {r['win_rate']:5.1f}% | Hold {r.get('avg_hold_days',30):4.1f}d | Δ {delta:+.2f}")

    # Save
    findings_dir = '/home/jupiter/Lvl3Quant/research/findings'
    os.makedirs(findings_dir, exist_ok=True)
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exit_optimization_v1',
        'configs': {k: {kk: str(vv) for kk,vv in v.items()} for k,v in CONFIGS.items()},
        'results': all_results, 'best': best_result, 'runtime_s': runtime,
    }
    with open(os.path.join(findings_dir, 'exit_optimization_v1_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved.")
    return all_results

if __name__ == '__main__':
    main()
