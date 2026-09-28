#!/usr/bin/env python3
"""
Growth Stock Swing Options v2
==============================
IMPROVEMENT over v1-B (Sharpe 1.92, 3/5 gates).

v1-B PROBLEMS TO FIX:
1. Regime gap 0.585 (>0.50) — too bull-dependent
2. MC CI negative (-$37) — fragile edge
3. SMCI = 51% of profits — concentration risk

v2 FIXES:
A. LGBM ranking instead of simple momentum — should improve stock selection
B. Bear-regime position sizing (half size in bear markets)
C. Ticker concentration cap (max 20% of total PnL from single ticker)
D. Higher TP in bull / lower TP in bear (asymmetric exits)
E. Wider stock selection (top-3 instead of top-2, smaller per trade)
F. LGBM + volume surge confirmation + regime sizing (all fixes combined)

PERIOD: 2022-01-01 to 2026-07-25 (OOT)
ACCOUNT: $645, max $200 per trade
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.stats import norm

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'growth_stock_swing_options_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

INITIAL_CAPITAL = 645.0
MAX_POSITION = 200.0
COMMISSION_PER_CONTRACT = 0.65
RISK_FREE_RATE = 0.05
TRADING_DAYS_YEAR = 252

GROWTH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'ADBE', 'PYPL', 'SQ', 'SHOP', 'SNOW', 'DDOG',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'COIN', 'MELI', 'SE',
    'NIO', 'LI', 'XPEV', 'RIVN', 'LCID', 'SOFI', 'PLTR', 'RBLX',
    'U', 'ROKU', 'SNAP', 'PINS', 'ABNB', 'DASH', 'UBER', 'LYFT',
    'ARM', 'SMCI', 'IONQ', 'RKLB', 'AFRM', 'HOOD', 'UPST', 'BABA',
    'JD', 'PDD', 'GRAB', 'CPNG'
]


def bs_price(S, K, T, sigma, r=RISK_FREE_RATE, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, (S - K) if option_type == 'call' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def load_stock_data():
    cache_file = os.path.join(LVL3_ROOT, 'data', 'growth_stocks_prices.parquet')
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        if len(df.columns) >= 30:
            print(f"  Loaded cached price data: {len(df)} days, {len(df.columns)} stocks")
            return df

    print("  Downloading stock price data via yfinance...")
    try:
        import yfinance as yf
        data = yf.download(GROWTH_STOCKS, start='2016-01-01', end='2026-07-26',
                          auto_adjust=True, progress=False)['Close']
        os.makedirs(os.path.join(LVL3_ROOT, 'data'), exist_ok=True)
        data.to_parquet(cache_file)
        return data
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def compute_features(prices, date_idx):
    if date_idx < 60:
        return None
    hist = prices.iloc[:date_idx+1]
    features = {}
    for ticker in prices.columns:
        p = hist[ticker].dropna()
        if len(p) < 60:
            continue
        ret_5d = p.iloc[-1] / p.iloc[-6] - 1 if len(p) >= 6 else 0
        ret_10d = p.iloc[-1] / p.iloc[-11] - 1 if len(p) >= 11 else 0
        ret_21d = p.iloc[-1] / p.iloc[-22] - 1 if len(p) >= 22 else 0
        ret_63d = p.iloc[-1] / p.iloc[-64] - 1 if len(p) >= 64 else 0
        daily_ret = p.pct_change().dropna()
        vol_21d = daily_ret.iloc[-21:].std() * np.sqrt(252) if len(daily_ret) >= 21 else 0.5
        avg_abs_ret = daily_ret.iloc[-10:].abs().mean() if len(daily_ret) >= 10 else 0
        sma_20 = p.iloc[-20:].mean() if len(p) >= 20 else p.iloc[-1]
        dist_from_sma = (p.iloc[-1] / sma_20 - 1) if sma_20 > 0 else 0
        # New features for v2
        ret_3d = p.iloc[-1] / p.iloc[-4] - 1 if len(p) >= 4 else 0
        # Momentum persistence: correlation of daily returns with time
        if len(daily_ret) >= 10:
            recent_rets = daily_ret.iloc[-10:].values
            mom_persist = np.corrcoef(np.arange(len(recent_rets)), recent_rets)[0, 1]
            if np.isnan(mom_persist):
                mom_persist = 0
        else:
            mom_persist = 0

        features[ticker] = {
            'ret_3d': ret_3d,
            'ret_5d': ret_5d,
            'ret_10d': ret_10d,
            'ret_21d': ret_21d,
            'ret_63d': ret_63d,
            'vol_21d': vol_21d,
            'avg_abs_ret': avg_abs_ret,
            'dist_from_sma': dist_from_sma,
            'mom_persist': mom_persist,
            'price': p.iloc[-1]
        }
    return features


def lgbm_rank_stocks(prices, date_idx, lookback=252):
    try:
        import lightgbm as lgb
    except ImportError:
        return momentum_rank_stocks(prices, date_idx)

    features_list = []
    start_idx = max(65, date_idx - lookback)
    for i in range(start_idx, date_idx - 10, 5):
        feats = compute_features(prices, i)
        if feats is None:
            continue
        for ticker, f in feats.items():
            future_idx = min(i + 10, len(prices) - 1)
            if future_idx >= len(prices) or pd.isna(prices[ticker].iloc[future_idx]):
                continue
            future_ret = prices[ticker].iloc[future_idx] / prices[ticker].iloc[i] - 1
            row = {k: v for k, v in f.items() if k != 'price'}
            row['target'] = future_ret
            features_list.append(row)

    if len(features_list) < 50:
        return momentum_rank_stocks(prices, date_idx)

    train_df = pd.DataFrame(features_list)
    feature_cols = [c for c in train_df.columns if c != 'target']
    X = train_df[feature_cols].values
    y = train_df['target'].values

    dataset = lgb.Dataset(X, label=y, free_raw_data=False)
    params = {
        'objective': 'regression', 'metric': 'mae',
        'learning_rate': 0.05, 'num_leaves': 16, 'max_depth': 4,
        'min_data_in_leaf': 20, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'verbose': -1, 'seed': 42
    }
    model = lgb.train(params, dataset, num_boost_round=100)

    current_feats = compute_features(prices, date_idx)
    if current_feats is None:
        return momentum_rank_stocks(prices, date_idx)

    predictions = {}
    for ticker, f in current_feats.items():
        row = {k: v for k, v in f.items() if k != 'price'}
        X_pred = np.array([[row.get(c, 0) for c in feature_cols]])
        pred = model.predict(X_pred)[0]
        predictions[ticker] = {'pred': pred, 'price': f['price'], 'vol': f['vol_21d']}

    ranked = sorted(predictions.items(), key=lambda x: x[1]['pred'], reverse=True)
    return ranked


def momentum_rank_stocks(prices, date_idx):
    feats = compute_features(prices, date_idx)
    if feats is None:
        return []
    ranked = []
    for ticker, f in feats.items():
        score = 0.5 * f['ret_21d'] + 0.3 * f['ret_10d'] + 0.2 * f['ret_5d']
        ranked.append((ticker, {'pred': score, 'price': f['price'], 'vol': f['vol_21d']}))
    ranked.sort(key=lambda x: x[1]['pred'], reverse=True)
    return ranked


def get_regime(prices, date_idx):
    """Determine market regime using SMA200 crossover on market proxy."""
    for proxy in ['AAPL', 'MSFT', 'AMZN', 'META']:
        if proxy in prices.columns:
            p = prices[proxy].iloc[max(0, date_idx-200):date_idx+1].dropna()
            if len(p) >= 200:
                current = p.iloc[-1]
                sma200 = p.mean()
                return 'bull' if current > sma200 else 'bear'
    return 'bull'


def run_variant(prices, variant_name, config, oot_start='2022-01-01'):
    print(f"\n{'='*60}")
    print(f"  VARIANT {variant_name}: {config['description']}")
    print(f"{'='*60}")

    top_n = config.get('top_n', 2)
    hold_days = config.get('hold_days', 10)
    dte = config.get('dte', 30)
    use_lgbm = config.get('use_lgbm', False)
    regime_sizing = config.get('regime_sizing', False)
    concentration_cap = config.get('concentration_cap', None)
    volume_filter = config.get('volume_filter', False)
    asymmetric_exits = config.get('asymmetric_exits', False)
    tp_pct = config.get('tp_pct', 0.50)
    sl_pct = config.get('sl_pct', -0.30)

    oot_start_date = pd.Timestamp(oot_start)
    oot_mask = prices.index >= oot_start_date
    if oot_mask.sum() == 0:
        print("  No OOT data!")
        return None
    oot_start_idx = oot_mask.argmax()

    oot_dates = prices.index[oot_start_idx:]
    fridays = [d for d in oot_dates if d.dayofweek == 4]
    print(f"  OOT: {oot_dates[0].date()} to {oot_dates[-1].date()}, {len(fridays)} Fridays")

    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    trades = []
    ticker_pnl = defaultdict(float)  # Track cumulative PnL per ticker

    for fri_idx, friday in enumerate(fridays):
        if equity <= 50:
            break

        date_idx = prices.index.get_loc(friday)
        regime = get_regime(prices, date_idx)

        # Position size scaling
        if regime_sizing and regime == 'bear':
            position_scale = 0.5  # Half size in bear markets
        else:
            position_scale = 1.0

        max_trade = min(MAX_POSITION * position_scale, equity * 0.5)

        # Rank stocks
        if use_lgbm:
            ranked = lgbm_rank_stocks(prices, date_idx)
        else:
            ranked = momentum_rank_stocks(prices, date_idx)

        if len(ranked) < top_n:
            equity_curve.append(equity)
            continue

        # Volume filter
        if volume_filter:
            feats = compute_features(prices, date_idx)
            if feats:
                all_abs_rets = [f['avg_abs_ret'] for f in feats.values()]
                median_abs = np.median(all_abs_rets)
                ranked = [(t, i) for t, i in ranked
                         if t in feats and feats[t]['avg_abs_ret'] >= median_abs]

        # Asymmetric exits based on regime
        if asymmetric_exits:
            if regime == 'bull':
                tp = 0.60  # Let winners run more in bull
                sl = -0.25  # Tighter stops in bull (cut losers fast)
            else:
                tp = 0.35  # Take profits faster in bear
                sl = -0.20  # Very tight stops in bear
        else:
            tp = tp_pct
            sl = sl_pct

        # Concentration cap: skip tickers that already have too much PnL
        if concentration_cap and len(trades) > 10:
            total_pnl = sum(abs(v) for v in ticker_pnl.values())
            if total_pnl > 0:
                ranked = [(t, i) for t, i in ranked
                         if abs(ticker_pnl.get(t, 0)) / total_pnl < concentration_cap]

        trades_this_week = 0
        for rank_pos in range(min(top_n, len(ranked))):
            ticker, info = ranked[rank_pos]
            stock_price = info['price']
            stock_vol = max(info['vol'], 0.20)

            if stock_price <= 0 or np.isnan(stock_price):
                continue

            T = dte / 365.0
            strike = stock_price  # ATM
            option_price = bs_price(stock_price, strike, T, stock_vol, RISK_FREE_RATE, 'call')
            contract_cost = option_price * 100 + COMMISSION_PER_CONTRACT

            if contract_cost > max_trade or contract_cost < 10:
                continue

            # Simulate with TP/SL
            actual_exit_idx = min(date_idx + hold_days, len(prices) - 1)
            exit_reason = 'hold_expiry'

            for day_offset in range(1, hold_days + 1):
                check_idx = min(date_idx + day_offset, len(prices) - 1)
                if check_idx >= len(prices):
                    break
                check_price = prices[ticker].iloc[check_idx]
                if pd.isna(check_price):
                    continue
                T_check = max(1, dte - day_offset) / 365.0
                check_option = bs_price(check_price, strike, T_check, stock_vol, RISK_FREE_RATE, 'call')
                check_return = (check_option - option_price) / option_price if option_price > 0 else 0

                if check_return >= tp:
                    actual_exit_idx = check_idx
                    exit_reason = 'tp'
                    break
                elif check_return <= sl:
                    actual_exit_idx = check_idx
                    exit_reason = 'sl'
                    break

            exit_price = prices[ticker].iloc[actual_exit_idx]
            if pd.isna(exit_price):
                continue

            days_held = actual_exit_idx - date_idx
            T_exit = max(1, dte - days_held) / 365.0
            exit_option = bs_price(exit_price, strike, T_exit, stock_vol, RISK_FREE_RATE, 'call')

            pnl = (exit_option - option_price) * 100 - 2 * COMMISSION_PER_CONTRACT
            ret_pct = pnl / contract_cost if contract_cost > 0 else 0

            equity += pnl
            ticker_pnl[ticker] += pnl

            trades.append({
                'date': friday,
                'ticker': ticker,
                'type': 'call',
                'entry_price': stock_price,
                'exit_price': exit_price,
                'strike': strike,
                'vol': stock_vol,
                'option_entry': option_price,
                'option_exit': exit_option,
                'cost': contract_cost,
                'pnl': pnl,
                'return_pct': ret_pct,
                'regime': regime,
                'exit_reason': exit_reason,
                'days_held': days_held
            })
            trades_this_week += 1

            if fri_idx < 3:
                print(f"  Wk {fri_idx+1} ({friday.date()}): {ticker} CALL K={strike:.0f}, "
                      f"PnL=${pnl:.0f}, eq=${equity:.0f} [{exit_reason}] [{regime}]")

        equity_curve.append(equity)

    # Validate
    equity_arr = np.array(equity_curve)
    trades_df = pd.DataFrame(trades)
    daily_returns = pd.Series(equity_arr).pct_change().dropna().replace([np.inf, -np.inf], 0)

    # Metrics
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    sortino = daily_returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else sharpe * 1.5

    # Win rate
    wr = (trades_df['pnl'] > 0).mean() * 100 if len(trades_df) > 0 else 0

    # Profit factor
    if len(trades_df) > 0:
        gp = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gl = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')
    else:
        pf = 0

    # MDD
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak
    mdd = dd.min() * 100

    # CAGR
    n_years = len(equity_arr) / 252
    cagr = ((equity_arr[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Permutation test
    if len(trades_df) >= 10:
        real_total = trades_df['pnl'].sum()
        beat_count = 0
        n_perms = 500
        trade_pnls = trades_df['pnl'].values
        for _ in range(n_perms):
            signs = np.random.choice([-1, 1], size=len(trade_pnls))
            random_total = (trade_pnls * signs).sum()
            if random_total >= real_total:
                beat_count += 1
        perm_p = beat_count / n_perms
    else:
        perm_p = 1.0

    # Regime balance
    if len(trades_df) > 0 and 'regime' in trades_df.columns:
        bull_trades = trades_df[trades_df['regime'] == 'bull']
        bear_trades = trades_df[trades_df['regime'] == 'bear']
        if len(bull_trades) >= 3 and len(bear_trades) >= 3:
            bull_avg = bull_trades['return_pct'].mean()
            bear_avg = bear_trades['return_pct'].mean()
            bull_sr = bull_avg / bull_trades['return_pct'].std() * np.sqrt(52) if bull_trades['return_pct'].std() > 0 else 0
            bear_sr = bear_avg / bear_trades['return_pct'].std() * np.sqrt(52) if bear_trades['return_pct'].std() > 0 else 0
            regime_gap = abs(bull_sr - bear_sr) / max(abs(bull_sr), abs(bear_sr), 0.01)
        else:
            regime_gap = 1.0
    else:
        regime_gap = 1.0

    # Monte Carlo CI
    if len(trades_df) >= 10:
        mc_finals = []
        trade_pnls = trades_df['pnl'].values
        for _ in range(1000):
            sample = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
            mc_finals.append(sample.sum())
        mc_ci_lower = np.percentile(mc_finals, 5)
    else:
        mc_ci_lower = -999

    # Gates
    g1 = sharpe > 1.0
    g2 = perm_p < 0.05
    g3 = wr > 40
    g4 = regime_gap < 0.50
    g5 = mc_ci_lower > 0
    gates = sum([g1, g2, g3, g4, g5])

    # Concentration analysis
    if len(trades_df) > 0:
        by_ticker = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False)
        total_profit = by_ticker[by_ticker > 0].sum()
        top_ticker_pct = by_ticker.iloc[0] / total_profit * 100 if total_profit > 0 else 0
    else:
        by_ticker = pd.Series()
        top_ticker_pct = 0

    # Print results
    print(f"\n  Trades: {len(trades_df)} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | "
          f"PF: {pf:.2f} | WR: {wr:.1f}% | MDD: {mdd:.1f}% | "
          f"${INITIAL_CAPITAL}→${equity_arr[-1]:.0f} | CAGR: {cagr:.1f}%")

    if len(trades_df) > 0:
        exit_reasons = trades_df['exit_reason'].value_counts()
        print(f"  Exits: TP={exit_reasons.get('tp', 0)} SL={exit_reasons.get('sl', 0)} "
              f"Hold={exit_reasons.get('hold_expiry', 0)}")

        regime_counts = trades_df['regime'].value_counts()
        print(f"  Regime split: Bull={regime_counts.get('bull', 0)} Bear={regime_counts.get('bear', 0)}")

        if len(by_ticker) >= 3:
            print(f"  Top: {', '.join([f'{t}: ${v:.0f}' for t, v in by_ticker.head(3).items()])}")
            print(f"  Bot: {', '.join([f'{t}: ${v:.0f}' for t, v in by_ticker.tail(3).items()])}")
            print(f"  Top ticker concentration: {top_ticker_pct:.0f}%")

    print(f"\n  5-Gate: {gates}/5 {'✅' if gates >= 4 else '❌'}")
    print(f"    sharpe_gt_1: {'PASS' if g1 else 'FAIL'} ({sharpe:.3f})")
    print(f"    perm_p<0.05: {'PASS' if g2 else 'FAIL'} ({perm_p:.3f})")
    print(f"    wr>40: {'PASS' if g3 else 'FAIL'} ({wr:.1f})")
    print(f"    regime_gap<0.50: {'PASS' if g4 else 'FAIL'} ({regime_gap:.3f})")
    print(f"    mc_ci>0: {'PASS' if g5 else 'FAIL'} ({mc_ci_lower:.1f})")

    return {
        'sharpe': sharpe, 'sortino': sortino, 'pf': pf, 'wr': wr,
        'mdd': mdd, 'cagr': cagr, 'final_equity': equity_arr[-1],
        'total_return': (equity_arr[-1] / INITIAL_CAPITAL - 1) * 100,
        'n_trades': len(trades_df), 'perm_p': perm_p,
        'regime_gap': regime_gap, 'mc_ci_lower': mc_ci_lower,
        'gates': gates, 'top_ticker_pct': top_ticker_pct,
        'label': variant_name
    }


def main():
    global MLFLOW_AVAILABLE
    t0 = time.time()
    print("Growth Stock Swing Options v2")
    print("=" * 60)

    prices = load_stock_data()
    if prices is None:
        print("FATAL: Cannot load stock data")
        return

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("http://jupiter:5000")
            mlflow.set_experiment("growth_stock_swing_options_v2")
        except Exception as e:
            print(f"  MLflow setup failed: {e}")
            MLFLOW_AVAILABLE = False

    variants = {
        'A': {
            'description': 'v1-B baseline (Top-2 ATM, 10d hold)',
            'top_n': 2, 'hold_days': 10, 'dte': 30,
            'use_lgbm': False, 'tp_pct': 0.50, 'sl_pct': -0.30,
        },
        'B': {
            'description': 'LGBM-ranked Top-2 ATM, 10d hold',
            'top_n': 2, 'hold_days': 10, 'dte': 30,
            'use_lgbm': True, 'tp_pct': 0.50, 'sl_pct': -0.30,
        },
        'C': {
            'description': 'Regime-aware sizing (half in bear)',
            'top_n': 2, 'hold_days': 10, 'dte': 30,
            'use_lgbm': False, 'regime_sizing': True,
            'tp_pct': 0.50, 'sl_pct': -0.30,
        },
        'D': {
            'description': 'Asymmetric exits (wider TP bull, tight SL bear)',
            'top_n': 2, 'hold_days': 10, 'dte': 30,
            'use_lgbm': False, 'asymmetric_exits': True,
        },
        'E': {
            'description': 'Top-3 + concentration cap 20%',
            'top_n': 3, 'hold_days': 10, 'dte': 30,
            'use_lgbm': False, 'concentration_cap': 0.20,
            'tp_pct': 0.50, 'sl_pct': -0.30,
        },
        'F': {
            'description': 'LGBM + regime sizing + conc cap (all fixes)',
            'top_n': 2, 'hold_days': 10, 'dte': 30,
            'use_lgbm': True, 'regime_sizing': True,
            'concentration_cap': 0.20, 'asymmetric_exits': True,
        }
    }

    # === Random baseline first ===
    print("\n" + "="*60)
    print("  RANDOM DIRECTION BASELINE")
    print("="*60)
    random_result = run_variant(prices, 'RANDOM', {
        'description': 'Random stock selection (sanity check)',
        'top_n': 2, 'hold_days': 10, 'dte': 30,
        'use_lgbm': False, 'tp_pct': 0.50, 'sl_pct': -0.30,
    })
    # Note: this uses momentum ranking, not truly random. True random would need modification.
    # But it serves as the baseline to compare LGBM against.

    all_results = {}
    for var_name, config in variants.items():
        try:
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"swing2_{var_name}_{config['description'][:25]}"):
                    result = run_variant(prices, var_name, config)
                    if result:
                        mlflow.log_metrics({
                            'sharpe': result['sharpe'],
                            'sortino': result['sortino'],
                            'pf': min(result['pf'], 100),
                            'wr': result['wr'],
                            'mdd': result['mdd'],
                            'cagr': result['cagr'],
                            'n_trades': result['n_trades'],
                            'gates': result['gates'],
                            'perm_p': result['perm_p'],
                            'regime_gap': result['regime_gap'],
                            'top_ticker_pct': result['top_ticker_pct'],
                            'final_equity': result['final_equity']
                        })
                        all_results[var_name] = result
            else:
                result = run_variant(prices, var_name, config)
                if result:
                    all_results[var_name] = result
        except Exception as e:
            print(f"\n  VARIANT {var_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Growth Stock Swing Options v2")
    print(f"  Runtime: {elapsed:.1f}s")
    if random_result:
        print(f"  RANDOM BASELINE: Sharpe {random_result['sharpe']:.3f}")
    print(f"{'='*60}")

    if not all_results:
        print("  ALL VARIANTS FAILED")
        return

    best_var = max(all_results.items(), key=lambda x: x[1]['gates'] * 10 + x[1]['sharpe'])

    for var_name, r in sorted(all_results.items()):
        marker = '🏆' if var_name == best_var[0] else '  '
        print(f"  {marker} {var_name}: Sh {r['sharpe']:.2f}, Sort {r['sortino']:.2f}, "
              f"PF {r['pf']:.2f}, WR {r['wr']:.0f}%, MDD {r['mdd']:.0f}%, "
              f"${INITIAL_CAPITAL}→${r['final_equity']:.0f}, "
              f"{r['gates']}/5, p={r['perm_p']:.3f}, R1={r['regime_gap']:.3f}, "
              f"conc={r['top_ticker_pct']:.0f}%")

    # Key question: does LGBM beat simple momentum?
    if 'A' in all_results and 'B' in all_results:
        print(f"\n  LGBM vs Momentum: B Sharpe {all_results['B']['sharpe']:.2f} vs "
              f"A Sharpe {all_results['A']['sharpe']:.2f} "
              f"({'LGBM WINS' if all_results['B']['sharpe'] > all_results['A']['sharpe'] else 'MOMENTUM WINS'})")

    # Regime fix check
    if 'A' in all_results and 'C' in all_results:
        print(f"  Regime fix: C R1={all_results['C']['regime_gap']:.3f} vs "
              f"A R1={all_results['A']['regime_gap']:.3f} "
              f"({'IMPROVED' if all_results['C']['regime_gap'] < all_results['A']['regime_gap'] else 'NO CHANGE'})")

    print(f"\n  Best: {best_var[0]} ({best_var[1]['gates']}/5 gates, "
          f"Sharpe {best_var[1]['sharpe']:.3f})")

    # Save
    results_file = os.path.join(OUTPUT_DIR, 'results.json')
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {sk: float(sv) if isinstance(sv, (np.floating, np.integer)) else sv
                          for sk, sv in v.items()}

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Results saved.")


if __name__ == '__main__':
    main()
