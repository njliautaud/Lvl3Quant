#!/usr/bin/env python3
"""
ML Post-Earnings Announcement Drift (PEAD) Strategy
=====================================================
Academic anomaly: stocks continue drifting in direction of earnings surprise
for 20-60 days post-announcement. We test if ML can predict which surprises
produce the strongest drift, and whether trading this produces alpha.

Universe: S&P 500 components (liquid, good data)
Data: yfinance earnings + price history
Signal: ML predicts drift direction/magnitude from gap + fundamentals
Hold: 5-60 day hold after earnings
Adversarial: permutation test, sub-period, outlier, R1 regime

HC #713: Fixed capital, no DCA
HC #714: Income + growth research
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from datetime import datetime, timedelta
import json, os, warnings, time
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/ml_pead'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

# High-liquidity stocks with long earnings history
UNIVERSE = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','BAC','GS',
    'WMT','HD','COST','UNH','JNJ','PFE','ABBV','XOM','CVX','COP',
    'DIS','NFLX','INTC','AMD','QCOM','MU','AVGO','TXN','CRM','ADBE',
    'V','MA','PYPL','BRK-B','BLK','SCHW','MS','C','WFC','USB',
    'PG','KO','PEP','MCD','SBUX','NKE','LOW','TGT','CMG','LULU'
]

def download_earnings_and_prices(tickers, start='2015-01-01'):
    """Download earnings calendar + price data for all tickers."""
    print(f"  Downloading data for {len(tickers)} tickers...")
    
    # Download all prices at once
    prices = yf.download(tickers, start=start, progress=False)
    if hasattr(prices.index, 'tz') and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices['Close']
        volume = prices['Volume']
    else:
        close = prices[['Close']] if 'Close' in prices.columns else prices
        volume = prices[['Volume']] if 'Volume' in prices.columns else None
    
    # Get SPY for market returns
    spy = yf.download('SPY', start=start, progress=False)['Close']
    if hasattr(spy.index, 'tz') and spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    spy_ret = spy.pct_change()
    
    # Collect earnings events
    earnings_events = []
    failed = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0:
            print(f"    Processing {i+1}/{len(tickers)}...")
        try:
            stock = yf.Ticker(ticker)
            
            # Get earnings dates from calendar
            try:
                cal = stock.earnings_dates
                if cal is None or len(cal) == 0:
                    failed.append(ticker)
                    continue
            except:
                failed.append(ticker)
                continue
            
            if ticker not in close.columns:
                failed.append(ticker)
                continue
            
            px = close[ticker].dropna()
            vol = volume[ticker].dropna() if volume is not None and ticker in volume.columns else None
            
            n_events_before = len(earnings_events)
            for date in cal.index:
              try:
                date_ts = pd.Timestamp(date)
                if date_ts.tzinfo is not None:
                    date_ts = date_ts.tz_localize(None)
                date_ts = pd.Timestamp(date_ts.date())

                pre_dates = px.index[px.index < date_ts]
                post_dates = px.index[px.index >= date_ts]

                if len(pre_dates) < 60 or len(post_dates) < 21:
                    continue

                earn_idx = post_dates[0]
                pre_idx = pre_dates[-1]
                gap = (px[earn_idx] / px[pre_idx] - 1) * 100

                ret_5d = (px[pre_idx] / px[pre_dates[-5]] - 1) if len(pre_dates) >= 5 else 0
                ret_21d = (px[pre_idx] / px[pre_dates[-21]] - 1) if len(pre_dates) >= 21 else 0
                ret_63d = (px[pre_idx] / px[pre_dates[-63]] - 1) if len(pre_dates) >= 63 else 0
                vol_21d = px[pre_dates[-21:]].pct_change().std() * np.sqrt(252) if len(pre_dates) >= 21 else 0

                if vol is not None and earn_idx in vol.index:
                    avg_vol_20 = vol[pre_dates[-20:]].mean()
                    vol_surge = vol[earn_idx] / avg_vol_20 if avg_vol_20 > 0 else 1.0
                else:
                    vol_surge = 1.0

                spy_ret_5d_val = 0
                try:
                    spy_ret_5d_val = float(spy_ret.reindex(pre_dates[-5:]).sum())
                except:
                    pass

                drift_5d = (px[post_dates[4]] / px[earn_idx] - 1) if len(post_dates) > 4 else np.nan
                drift_10d = (px[post_dates[9]] / px[earn_idx] - 1) if len(post_dates) > 9 else np.nan
                drift_21d = (px[post_dates[20]] / px[earn_idx] - 1) if len(post_dates) > 20 else np.nan

                try:
                    spy_drift_5d = float(spy_ret.reindex(post_dates[:5]).sum()) if len(post_dates) > 4 else np.nan
                    spy_drift_10d = float(spy_ret.reindex(post_dates[:10]).sum()) if len(post_dates) > 9 else np.nan
                    spy_drift_21d = float(spy_ret.reindex(post_dates[:21]).sum()) if len(post_dates) > 20 else np.nan
                except:
                    spy_drift_5d = spy_drift_10d = spy_drift_21d = np.nan

                def safe_excess(a, b):
                    try:
                        if np.isnan(a) or np.isnan(b): return np.nan
                        return float(a - b)
                    except:
                        return np.nan
                excess_5d = safe_excess(drift_5d, spy_drift_5d)
                excess_10d = safe_excess(drift_10d, spy_drift_10d)
                excess_21d = safe_excess(drift_21d, spy_drift_21d)

                eps_surprise = 0
                try:
                    if 'Surprise(%)' in cal.columns:
                        val = cal.loc[date, 'Surprise(%)']
                        if not pd.isna(val):
                            eps_surprise = float(val)
                except:
                    pass

                earnings_events.append({
                    'date': date_ts,
                    'ticker': ticker,
                    'gap_pct': float(gap),
                    'abs_gap': float(abs(gap)),
                    'gap_direction': 1 if gap > 0 else -1,
                    'eps_surprise': float(eps_surprise),
                    'ret_5d_pre': float(ret_5d),
                    'ret_21d_pre': float(ret_21d),
                    'ret_63d_pre': float(ret_63d),
                    'vol_21d': float(vol_21d),
                    'vol_surge': float(vol_surge),
                    'spy_ret_5d': float(spy_ret_5d_val),
                    'drift_5d': float(drift_5d) if not np.isnan(drift_5d) else np.nan,
                    'drift_10d': float(drift_10d) if not np.isnan(drift_10d) else np.nan,
                    'drift_21d': float(drift_21d) if not np.isnan(drift_21d) else np.nan,
                    'excess_5d': float(excess_5d) if not isinstance(excess_5d, float) or not np.isnan(excess_5d) else np.nan,
                    'excess_10d': float(excess_10d) if not isinstance(excess_10d, float) or not np.isnan(excess_10d) else np.nan,
                    'excess_21d': float(excess_21d) if not isinstance(excess_21d, float) or not np.isnan(excess_21d) else np.nan,
                })
              except Exception as inner_e:
                continue

            if len(earnings_events) == n_events_before:
                failed.append(ticker)  # No events from this ticker

        except Exception as e:
            if (i + 1) <= 3:
                print(f"    ERROR {ticker}: {e}")
            failed.append(ticker)
            continue
    
    df = pd.DataFrame(earnings_events)
    print(f"  Collected {len(df)} earnings events from {len(tickers) - len(failed)} tickers")
    if len(failed) > 0:
        print(f"  Failed: {len(failed)} ({', '.join(failed[:10])}{'...' if len(failed)>10 else ''})")
    if len(df) > 0:
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        print(f"  WARNING: No earnings events collected!")
    return df

def run_pead_backtest(df, hold_days='10d', use_ml=True, shuffle=False):
    """
    Run PEAD strategy backtest.
    - Buy stocks with positive gap, short stocks with negative gap
    - Hold for hold_days after earnings
    - If use_ml, use GBM to predict which gaps will drift more
    - Returns: daily return series
    """
    target_col = f'excess_{hold_days}'
    if target_col not in df.columns:
        return None
    
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean = df_clean.sort_values('date')
    
    feature_cols = ['gap_pct', 'abs_gap', 'eps_surprise', 'ret_5d_pre', 'ret_21d_pre', 
                    'ret_63d_pre', 'vol_21d', 'vol_surge', 'spy_ret_5d']
    
    # Ensure features are numeric
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=feature_cols)
    
    if len(df_clean) < 100:
        return None
    
    # Walk-forward
    dates = sorted(df_clean['date'].unique())
    train_window = pd.Timedelta(days=365)
    
    trade_returns = []
    
    for i, date in enumerate(dates):
        if date < dates[0] + train_window:
            continue
        
        today_events = df_clean[df_clean['date'] == date]
        if len(today_events) == 0:
            continue
        
        if use_ml:
            # Train on past events
            train = df_clean[(df_clean['date'] >= date - train_window) & (df_clean['date'] < date)]
            if len(train) < 30:
                continue
            
            X_tr = train[feature_cols].values
            # Target: will excess return be positive?
            y_tr = (train[target_col] > 0).astype(int).values
            
            X_te = today_events[feature_cols].values
            
            model = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1,
                                        subsample=0.8, verbose=-1, min_child_samples=10)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            
            if shuffle:
                probs = np.random.permutation(probs)
            
            # Trade only high-confidence predictions
            for j, (_, event) in enumerate(today_events.iterrows()):
                if probs[j] > 0.6:
                    trade_returns.append((date, event[target_col], 'long'))
                elif probs[j] < 0.4:
                    trade_returns.append((date, -event[target_col], 'short'))
        else:
            # Simple rule: trade in direction of gap
            for _, event in today_events.iterrows():
                if abs(event['gap_pct']) > 2.0:  # Only trade meaningful gaps
                    direction = 1 if event['gap_pct'] > 0 else -1
                    trade_returns.append((date, direction * event[target_col], 'gap_rule'))
    
    if not trade_returns:
        return None
    
    # Convert to daily returns
    trade_df = pd.DataFrame(trade_returns, columns=['date', 'return', 'type'])
    daily = trade_df.groupby('date')['return'].mean()
    
    # Fill missing dates with 0 (no trades)
    all_dates = pd.date_range(daily.index.min(), daily.index.max(), freq='B')
    daily = daily.reindex(all_dates, fill_value=0)
    
    return daily

def compute_metrics(returns, name=""):
    if returns is None or len(returns) < 20:
        return None
    ret = returns.values
    if np.std(ret) == 0:
        return None
    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252)
    neg = ret[ret < 0]
    sortino = np.mean(ret) / np.std(neg) * np.sqrt(252) if len(neg) > 0 and np.std(neg) > 0 else 0
    cum = (1 + pd.Series(ret)).cumprod()
    dd = (cum / cum.cummax() - 1)
    max_dd = dd.min() * 100
    years = len(ret) / 252
    cagr = ((cum.iloc[-1]) ** (1/years) - 1) * 100 if years > 0 else 0
    total_ret = (cum.iloc[-1] - 1) * 100
    wins = np.sum(ret > 0)
    trades = np.sum(ret != 0)
    wr = wins / trades * 100 if trades > 0 else 0
    pf = abs(ret[ret > 0].sum() / ret[ret < 0].sum()) if ret[ret < 0].sum() != 0 else 0
    
    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1), 'total_return': round(total_ret, 1),
        'profit_factor': round(pf, 3), 'win_rate': round(wr, 1), 'n_trades': int(trades),
        'years': round(years, 1)
    }

def run_adversarial(returns, df_events, name="PEAD"):
    """Full adversarial validation."""
    print(f"\n  ADVERSARIAL: {name}")
    print(f"  {'-'*60}")
    results = {}
    real = compute_metrics(returns, name)
    if not real:
        print("    Cannot compute metrics")
        return {'summary': {'gates_passed': 0, 'total': 4, 'verdict': 'FAIL'}}
    
    real_sharpe = real['sharpe']
    
    # 1. Permutation (signal shuffle)
    N_PERM = 100
    print(f"    [1/4] Permutation test ({N_PERM} iters)...")
    perm_sharpes = []
    for trial in range(N_PERM):
        perm_ret = run_pead_backtest(df_events, hold_days='10d', use_ml=True, shuffle=True)
        if perm_ret is not None:
            m = compute_metrics(perm_ret)
            if m:
                perm_sharpes.append(m['sharpe'])
    
    if perm_sharpes:
        perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
        perm_pass = perm_p < 0.05
        print(f"      Real={real_sharpe:.3f}, Perm={np.mean(perm_sharpes):.3f}±{np.std(perm_sharpes):.3f}, p={perm_p:.3f} -> {'PASS' if perm_pass else 'FAIL'}")
        results['permutation'] = {'p_value': round(float(perm_p), 4), 'pass': bool(perm_pass)}
    else:
        results['permutation'] = {'p_value': 1.0, 'pass': False}
        print(f"      No valid permutations")
    
    # 2. Sub-period
    print(f"    [2/4] Sub-period consistency...")
    n = len(returns)
    block_sz = n // 4
    block_sharpes = []
    for b in range(4):
        start = b * block_sz
        end = (b+1)*block_sz if b < 3 else n
        m = compute_metrics(returns.iloc[start:end])
        if m:
            block_sharpes.append(m['sharpe'])
    cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if block_sharpes and np.mean(block_sharpes) != 0 else 999
    sub_pass = cv < 0.50
    print(f"      Sharpes: {[f'{s:.2f}' for s in block_sharpes]}, CV={cv:.3f} -> {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(cv, 3), 'pass': bool(sub_pass)}
    
    # 3. Outlier
    print(f"    [3/4] Outlier robustness...")
    ret_vals = returns.values
    p5, p95 = np.percentile(ret_vals[ret_vals != 0], [5, 95]) if np.sum(ret_vals != 0) > 10 else (0, 0)
    trimmed = ret_vals.copy()
    trimmed[(trimmed < p5) | (trimmed > p95)] = 0
    m_trim = compute_metrics(pd.Series(trimmed))
    if m_trim and real_sharpe != 0:
        deg = (m_trim['sharpe'] - real_sharpe) / abs(real_sharpe)
        out_pass = deg > -0.50
        print(f"      Full={real_sharpe:.3f}, Trimmed={m_trim['sharpe']:.3f}, deg={deg:.3f} -> {'PASS' if out_pass else 'FAIL'}")
        results['outlier'] = {'degradation': round(deg, 3), 'pass': bool(out_pass)}
    else:
        results['outlier'] = {'degradation': -999, 'pass': False}
    
    # 4. R1 regime (bull vs bear using SPY)
    print(f"    [4/4] R1 regime test...")
    spy = yf.download('SPY', start=returns.index[0], end=returns.index[-1], progress=False)['Close']
    spy_monthly = spy.resample('ME').last().pct_change()
    
    green_months = spy_monthly[spy_monthly > 0].index
    red_months = spy_monthly[spy_monthly <= 0].index
    
    green_ret = returns[[d for d in returns.index if d.to_period('M').to_timestamp() in green_months]]
    red_ret = returns[[d for d in returns.index if d.to_period('M').to_timestamp() in red_months]]
    
    m_green = compute_metrics(green_ret) if len(green_ret) > 20 else None
    m_red = compute_metrics(red_ret) if len(red_ret) > 20 else None
    
    if m_green and m_red and max(abs(m_green['sharpe']), abs(m_red['sharpe'])) > 0:
        gap = abs(m_green['sharpe'] - m_red['sharpe']) / max(abs(m_green['sharpe']), abs(m_red['sharpe']))
        r1_pass = gap < 0.50
        print(f"      Green={m_green['sharpe']:.3f}, Red={m_red['sharpe']:.3f}, gap={gap:.3f} -> {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {'gap': round(gap, 3), 'pass': bool(r1_pass), 
                                'green_sharpe': m_green['sharpe'], 'red_sharpe': m_red['sharpe']}
    else:
        results['r1_regime'] = {'gap': 999, 'pass': False}
        print(f"      Insufficient data for regime test")
    
    gates = sum(1 for k in ['permutation','sub_period','outlier','r1_regime'] if results.get(k, {}).get('pass', False))
    results['summary'] = {'gates_passed': gates, 'total': 4, 'verdict': 'PASS' if gates >= 3 else 'FAIL'}
    print(f"      VERDICT: {gates}/4 -> {results['summary']['verdict']}")
    
    return results

# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("ML POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) STRATEGY")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Step 1: Data
print("\nSTEP 1: COLLECTING EARNINGS DATA")
df = download_earnings_and_prices(UNIVERSE)

# Save raw data
df.to_parquet(os.path.join(OUTPUT, 'earnings_events.parquet'), index=False)

# Step 2: Exploratory analysis
print("\n" + "=" * 70)
print("STEP 2: DRIFT ANALYSIS")
print("=" * 70)

for horizon in ['5d', '10d', '21d']:
    col = f'excess_{horizon}'
    if col not in df.columns:
        continue
    valid = df.dropna(subset=[col])
    
    pos_gap = valid[valid['gap_pct'] > 0][col]
    neg_gap = valid[valid['gap_pct'] < 0][col]
    big_gap = valid[valid['abs_gap'] > 5][col]
    
    print(f"\n  {horizon} excess drift:")
    print(f"    All events: {valid[col].mean()*100:.2f}% mean, {valid[col].median()*100:.2f}% median (n={len(valid)})")
    print(f"    After positive gap: {pos_gap.mean()*100:.2f}% (n={len(pos_gap)})")
    print(f"    After negative gap: {neg_gap.mean()*100:.2f}% (n={len(neg_gap)})")
    print(f"    After |gap| > 5%:   {big_gap.mean()*100:.2f}% (n={len(big_gap)})")

# Step 3: Backtests
print("\n" + "=" * 70)
print("STEP 3: STRATEGY BACKTESTS")
print("=" * 70)

results_all = {}

# 3a. Simple gap rule (baseline)
print("\n  [A] Simple gap rule (trade in direction of >2% gaps)...")
ret_simple = run_pead_backtest(df, hold_days='10d', use_ml=False)
if ret_simple is not None:
    m = compute_metrics(ret_simple, "Gap Rule (10d hold)")
    print(f"      {m}")
    results_all['gap_rule_10d'] = m

# 3b. ML-filtered PEAD
for hold in ['5d', '10d', '21d']:
    print(f"\n  [ML] ML PEAD ({hold} hold)...")
    ret_ml = run_pead_backtest(df, hold_days=hold, use_ml=True)
    if ret_ml is not None:
        m = compute_metrics(ret_ml, f"ML PEAD ({hold} hold)")
        print(f"      {m}")
        results_all[f'ml_pead_{hold}'] = m
    else:
        print(f"      No valid trades")

# Step 4: Adversarial on best
print("\n" + "=" * 70)
print("STEP 4: ADVERSARIAL VALIDATION")
print("=" * 70)

# Find best variant
best_key = max(results_all, key=lambda k: results_all[k].get('sharpe', -999)) if results_all else None
if best_key:
    print(f"\n  Best variant: {best_key} (Sharpe={results_all[best_key]['sharpe']})")
    best_hold = '10d'  # default
    if '5d' in best_key: best_hold = '5d'
    elif '21d' in best_key: best_hold = '21d'
    
    best_ret = run_pead_backtest(df, hold_days=best_hold, use_ml='ml' in best_key)
    if best_ret is not None:
        adv = run_adversarial(best_ret, df, results_all[best_key]['name'])
        results_all[best_key]['adversarial'] = adv

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
for key, m in results_all.items():
    adv_str = ""
    if 'adversarial' in m:
        adv = m['adversarial']
        gates = adv.get('summary', {}).get('gates_passed', '?')
        total = adv.get('summary', {}).get('total', '?')
        adv_str = f"  [{gates}/{total} gates]"
    print(f"  {m['name']:30s}  Sharpe={m['sharpe']:6.3f}  Sortino={m['sortino']:6.3f}  CAGR={m['cagr']:5.1f}%  MaxDD={m['max_dd']:5.1f}%  Trades={m['n_trades']:4d}{adv_str}")

# Save results
output = {
    'strategy': 'ML PEAD',
    'timestamp': datetime.now().isoformat(),
    'n_tickers': len(UNIVERSE),
    'n_events': len(df),
    'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}" if len(df) > 0 else "N/A",
    'results': results_all,
}
with open(os.path.join(OUTPUT, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nRuntime: {datetime.now().strftime('%H:%M:%S')}")
print(f"Output: {OUTPUT}")
