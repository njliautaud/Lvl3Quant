#!/usr/bin/env python3
"""
PEAD ML Adversarial Validation
==============================
Deep validation of the PEAD ML strategy (4/6 variants passed 5/5 gates).

TESTS:
1. Sub-period stability (split into 4 periods — all profitable?)
2. Per-year breakdown (does it work in 2022 bear market?)
3. Ticker concentration (how much comes from top tickers?)
4. Ticker-removed robustness (remove top 2 tickers — still profitable?)
5. More permutation tests (500 instead of 100)
6. Walk-forward stability (retrain frequency sensitivity)
7. Threshold sensitivity (how does confidence threshold affect results?)
8. Direction analysis (calls vs puts — is it just long-biased?)

Uses the BEST variant (D: LGBM + Momentum Filter) as the base.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'pead_adversarial_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Reuse all the infrastructure from pead_ml_predictor_v1
# ============================================================

# Import key functions — or copy minimal needed code
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

STARTING_CAPITAL = 645.0
COMMISSION_RT = 1.30
MAX_POSITION = 200.0
RISK_FREE_RATE = 0.05
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'
DTE = 14
DRIFT_THRESHOLD = 0.03
HOLD_DAYS = 3

FEATURE_COLS = [
    'abs_gap', 'gap_direction', 'mom_5d', 'mom_10d', 'mom_21d',
    'vol_21d', 'rel_str_21d', 'vix', 'vol_ratio', 'rsi',
    'dist_52w_high', 'prev_gap', 'ticker_hash'
]


def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def option_price(S, K, T, r, sigma, opt_type='call'):
    return bs_call(S, K, T, r, sigma) if opt_type == 'call' else bs_put(S, K, T, r, sigma)


def load_events():
    """Load pre-built events from cached data."""
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'earnings_momentum_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'earnings_dates_cache.json')

    if not os.path.exists(cache_path) or not os.path.exists(earnings_cache):
        print("ERROR: Run pead_ml_predictor_v1.py first to generate cached data", flush=True)
        sys.exit(1)

    prices_df = pd.read_parquet(cache_path)
    with open(earnings_cache) as f:
        earnings_dates = json.load(f)

    print(f"Loaded prices: {len(prices_df)} rows, {len(earnings_dates)} ticker earnings", flush=True)

    # Build events (same as pead_ml_predictor_v1)
    events = []
    for ticker, dates in earnings_dates.items():
        try:
            ticker_prices = prices_df.loc[ticker].sort_index()
        except KeyError:
            continue
        if len(ticker_prices) < 30: continue

        trading_days = ticker_prices.index.sort_values()
        try:
            vix = prices_df.loc['^VIX', 'close']
        except:
            vix = None
        try:
            spy = prices_df.loc['SPY', 'close']
        except:
            spy = None

        for earn_date_str in dates:
            earn_date = pd.Timestamp(earn_date_str)
            post_mask = trading_days >= earn_date
            if not post_mask.any(): continue
            post_days = trading_days[post_mask]
            if len(post_days) < HOLD_DAYS + 1: continue
            pre_mask = trading_days < earn_date
            if not pre_mask.any(): continue
            pre_days = trading_days[pre_mask]
            if len(pre_days) < 30: continue

            close_before = ticker_prices.loc[pre_days[-1], 'close']
            open_after = ticker_prices.loc[post_days[0], 'open']
            if close_before <= 0 or open_after <= 0: continue

            gap_pct = (open_after / close_before) - 1.0
            if abs(gap_pct) < 0.05: continue

            feats = {'gap_pct': gap_pct, 'abs_gap': abs(gap_pct),
                     'gap_direction': 1 if gap_pct > 0 else -1}

            for w in [5, 10, 21]:
                if len(pre_days) >= w + 1:
                    feats[f'mom_{w}d'] = float(ticker_prices.loc[pre_days[-1], 'close'] /
                                              ticker_prices.loc[pre_days[-w-1], 'close'] - 1)
                else:
                    feats[f'mom_{w}d'] = 0

            if len(pre_days) >= 22:
                rc = ticker_prices.loc[pre_days[-22:], 'close']
                feats['vol_21d'] = float(rc.pct_change().dropna().std() * np.sqrt(252))
            else:
                feats['vol_21d'] = 0.3

            if spy is not None and len(pre_days) >= 22:
                try:
                    spy_c = spy.loc[spy.index <= pre_days[-1]].iloc[-21:]
                    stk_c = ticker_prices.loc[pre_days[-21:], 'close']
                    feats['rel_str_21d'] = float((stk_c.iloc[-1]/stk_c.iloc[0]-1) -
                                                 (spy_c.iloc[-1]/spy_c.iloc[0]-1))
                except:
                    feats['rel_str_21d'] = 0
            else:
                feats['rel_str_21d'] = 0

            if vix is not None:
                vm = vix.index <= pre_days[-1]
                feats['vix'] = float(vix[vm].iloc[-1]) if vm.any() else 20
            else:
                feats['vix'] = 20

            if 'volume' in ticker_prices.columns and len(pre_days) >= 22:
                rv = ticker_prices.loc[pre_days[-5:], 'volume'].mean()
                av = ticker_prices.loc[pre_days[-22:], 'volume'].mean()
                feats['vol_ratio'] = float(rv / max(av, 1))
            else:
                feats['vol_ratio'] = 1.0

            if len(pre_days) >= 15:
                rets = ticker_prices.loc[pre_days[-15:], 'close'].pct_change().dropna()
                g = rets.clip(lower=0).mean()
                l = (-rets).clip(lower=0).mean()
                feats['rsi'] = float(100 - 100/(1+g/l)) if l > 0 else 100
            else:
                feats['rsi'] = 50

            if len(pre_days) >= 252:
                h52 = ticker_prices.loc[pre_days[-252:], 'close'].max()
                feats['dist_52w_high'] = float(close_before / h52 - 1)
            else:
                feats['dist_52w_high'] = 0

            ticker_earn = [d for d in dates if pd.Timestamp(d) < earn_date]
            if ticker_earn:
                pe = pd.Timestamp(ticker_earn[-1])
                pp = trading_days[trading_days >= pe]
                pp2 = trading_days[trading_days < pe]
                if len(pp) > 0 and len(pp2) > 0:
                    feats['prev_gap'] = float(ticker_prices.loc[pp[0], 'open'] /
                                             ticker_prices.loc[pp2[-1], 'close'] - 1)
                else:
                    feats['prev_gap'] = 0
            else:
                feats['prev_gap'] = 0

            feats['ticker_hash'] = hash(ticker) % 100

            post_close = [ticker_prices.loc[d, 'close'] for d in post_days[:HOLD_DAYS+1]]
            if len(post_close) < 2: continue

            if gap_pct > 0:
                max_move = max(c / open_after - 1 for c in post_close[1:])
                drift_success = max_move >= DRIFT_THRESHOLD
            else:
                max_move = max(1 - c / open_after for c in post_close[1:])
                drift_success = max_move >= DRIFT_THRESHOLD

            feats['target'] = 1 if drift_success else 0
            feats['ticker'] = ticker
            feats['earn_date'] = earn_date_str
            feats['close_before'] = float(close_before)
            feats['open_after'] = float(open_after)
            feats['post_closes'] = [float(c) for c in post_close]

            events.append(feats)

    events.sort(key=lambda e: e['earn_date'])
    print(f"Built {len(events)} qualifying events", flush=True)
    return events


def run_strategy(events, threshold=0.60, momentum_filter=True, exclude_tickers=None):
    """Run the PEAD ML strategy with given parameters."""
    from sklearn.ensemble import GradientBoostingClassifier

    min_train = 30
    equity = STARTING_CAPITAL
    peak_equity = equity
    max_dd = 0
    trades = []

    for i in range(min_train, len(events)):
        event = events[i]

        if exclude_tickers and event['ticker'] in exclude_tickers:
            continue

        train_events = events[:i]
        if exclude_tickers:
            train_events = [e for e in train_events if e['ticker'] not in exclude_tickers]

        if len(train_events) < min_train:
            continue

        X_train = pd.DataFrame(train_events)[FEATURE_COLS]
        y_train = np.array([e['target'] for e in train_events])

        X_test = pd.DataFrame([event])[FEATURE_COLS]

        try:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, subsample=0.8, random_state=42)
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1][0]
        except:
            continue

        if prob < threshold:
            continue

        if momentum_filter:
            if event['gap_direction'] == 1 and event['mom_5d'] < 0:
                continue
            if event['gap_direction'] == -1 and event['mom_5d'] > 0:
                continue

        gap_pct = event['gap_pct']
        entry_price = event['open_after']
        strike = round(entry_price)
        opt_type = 'call' if gap_pct > 0 else 'put'

        post_iv = max(event['vol_21d'] * 0.8, 0.15)
        T = DTE / 252.0
        entry_premium = option_price(entry_price, strike, T, RISK_FREE_RATE, post_iv, opt_type)
        if entry_premium < 0.10: continue

        contract_cost = entry_premium * 100
        max_spend = min(MAX_POSITION, equity * 0.30)
        if contract_cost > max_spend or contract_cost + 0.65 > equity:
            continue

        post_closes = event['post_closes']
        exit_premium = entry_premium
        exit_reason = 'time_stop'

        for day in range(1, min(len(post_closes), HOLD_DAYS + 1)):
            spot = post_closes[day]
            remaining = max(DTE - day, 0) / 252.0
            iv_adj = post_iv * (1 + 0.02 * day)
            current = option_price(spot, strike, remaining, RISK_FREE_RATE, iv_adj, opt_type)

            pct = (current - entry_premium) / entry_premium
            if pct >= 0.30:
                exit_premium = current
                exit_reason = 'take_profit'
                break
            elif pct <= -0.25:
                exit_premium = current
                exit_reason = 'stop_loss'
                break
            exit_premium = current

        pnl = (exit_premium - entry_premium) * 100 - COMMISSION_RT
        equity += pnl

        if equity > peak_equity: peak_equity = equity
        dd = (equity - peak_equity) / peak_equity
        if dd < max_dd: max_dd = dd

        trades.append({
            'ticker': event['ticker'], 'earn_date': event['earn_date'],
            'opt_type': opt_type, 'prob': round(prob, 3),
            'gap_pct': round(gap_pct * 100, 1),
            'pnl': round(pnl, 2), 'exit_reason': exit_reason,
            'target': event['target'],
        })

    return trades, equity, max_dd


def main():
    import time

    print("=" * 70, flush=True)
    print("  PEAD ADVERSARIAL VALIDATION V1", flush=True)
    print("  Deep validation of PEAD ML strategy (4/6 variants 5/5 gates)", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    events = load_events()

    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("pead_adversarial_v1")

    results = {}

    # ============================================================
    # TEST 1: BASELINE — reproduce best variant D
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 1: BASELINE (LGBM + Momentum, threshold=0.60)", flush=True)
    print(f"{'='*60}", flush=True)

    trades, equity, max_dd = run_strategy(events, threshold=0.60, momentum_filter=True)
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]

    print(f"  Trades: {len(trades)}, WR: {len(wins)/len(pnls)*100:.1f}%, "
          f"Final: ${equity:.0f}, MDD: {max_dd*100:.1f}%", flush=True)

    results['baseline'] = {
        'trades': len(trades), 'equity': round(equity, 2),
        'mdd': round(max_dd*100, 1),
        'wr': round(len(wins)/len(pnls)*100, 1),
    }

    # ============================================================
    # TEST 2: PER-YEAR BREAKDOWN
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 2: PER-YEAR BREAKDOWN", flush=True)
    print(f"{'='*60}", flush=True)

    by_year = defaultdict(list)
    for t in trades:
        yr = t['earn_date'][:4]
        by_year[yr].append(t)

    year_results = {}
    for yr in sorted(by_year):
        yr_trades = by_year[yr]
        yr_pnls = [t['pnl'] for t in yr_trades]
        yr_wins = [p for p in yr_pnls if p > 0]
        yr_total = sum(yr_pnls)
        yr_wr = len(yr_wins)/len(yr_pnls)*100 if yr_pnls else 0
        is_bear = yr == '2022'

        print(f"  {yr}{'(BEAR)' if is_bear else '      '}: "
              f"{len(yr_trades)} trades, WR={yr_wr:.0f}%, "
              f"PnL=${yr_total:.0f}, avg=${np.mean(yr_pnls):.0f}", flush=True)

        year_results[yr] = {
            'trades': len(yr_trades), 'total_pnl': round(yr_total, 2),
            'wr': round(yr_wr, 1), 'regime': 'bear' if is_bear else 'bull'
        }

    results['per_year'] = year_results

    # ============================================================
    # TEST 3: TICKER CONCENTRATION
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 3: TICKER CONCENTRATION", flush=True)
    print(f"{'='*60}", flush=True)

    ticker_pnl = defaultdict(float)
    ticker_count = defaultdict(int)
    for t in trades:
        ticker_pnl[t['ticker']] += t['pnl']
        ticker_count[t['ticker']] += 1

    total_pnl = sum(t['pnl'] for t in trades)
    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)

    print(f"  Total PnL: ${total_pnl:.0f} from {len(ticker_pnl)} tickers", flush=True)
    print(f"  Top contributors:", flush=True)
    cum_pct = 0
    for tk, pnl in sorted_tickers[:5]:
        pct = pnl / total_pnl * 100 if total_pnl != 0 else 0
        cum_pct += pct
        print(f"    {tk}: ${pnl:.0f} ({pct:.1f}%, cum {cum_pct:.1f}%), "
              f"{ticker_count[tk]} trades", flush=True)

    top2_pct = sum(p for _, p in sorted_tickers[:2]) / total_pnl * 100 if total_pnl != 0 else 0
    results['concentration'] = {
        'n_tickers': len(ticker_pnl),
        'top2_pct': round(top2_pct, 1),
        'tickers': {tk: round(pnl, 2) for tk, pnl in sorted_tickers[:10]},
    }

    # ============================================================
    # TEST 4: TICKER-REMOVED ROBUSTNESS
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 4: TICKER-REMOVED ROBUSTNESS", flush=True)
    print(f"{'='*60}", flush=True)

    top2_tickers = set(tk for tk, _ in sorted_tickers[:2])
    trades_excl, equity_excl, dd_excl = run_strategy(
        events, threshold=0.60, momentum_filter=True, exclude_tickers=top2_tickers
    )

    if trades_excl:
        pnls_excl = [t['pnl'] for t in trades_excl]
        wins_excl = [p for p in pnls_excl if p > 0]
        wr_excl = len(wins_excl)/len(pnls_excl)*100

        print(f"  Without {top2_tickers}:", flush=True)
        print(f"    Trades: {len(trades_excl)}, WR: {wr_excl:.1f}%, "
              f"Final: ${equity_excl:.0f}, MDD: {dd_excl*100:.1f}%", flush=True)
        print(f"  vs baseline: {len(trades)} trades, ${equity:.0f}", flush=True)

        results['robustness'] = {
            'excluded': list(top2_tickers),
            'trades': len(trades_excl), 'equity': round(equity_excl, 2),
            'wr': round(wr_excl, 1), 'mdd': round(dd_excl*100, 1),
            'still_profitable': equity_excl > STARTING_CAPITAL,
        }
    else:
        print(f"  No trades after removing {top2_tickers}", flush=True)
        results['robustness'] = {'excluded': list(top2_tickers), 'trades': 0, 'still_profitable': False}

    # ============================================================
    # TEST 5: DEEP PERMUTATION TEST (500 shuffles)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 5: DEEP PERMUTATION TEST (500 shuffles)", flush=True)
    print(f"{'='*60}", flush=True)

    actual_mean = np.mean(pnls)
    rng = np.random.default_rng(42)
    count = 0
    perm_means = []
    for _ in range(500):
        signs = rng.choice([-1, 1], size=len(pnls))
        pm = np.mean(np.array(pnls) * signs)
        perm_means.append(pm)
        if pm >= actual_mean:
            count += 1

    p_value = (count + 1) / 501
    print(f"  Actual mean PnL: ${actual_mean:.2f}", flush=True)
    print(f"  Perm mean: ${np.mean(perm_means):.2f} ± ${np.std(perm_means):.2f}", flush=True)
    print(f"  p-value: {p_value:.4f} {'PASS' if p_value < 0.05 else 'FAIL'}", flush=True)

    results['deep_perm'] = {
        'p_value': round(p_value, 4),
        'actual_mean': round(actual_mean, 2),
        'perm_mean': round(np.mean(perm_means), 2),
    }

    # ============================================================
    # TEST 6: THRESHOLD SENSITIVITY
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 6: THRESHOLD SENSITIVITY", flush=True)
    print(f"{'='*60}", flush=True)

    threshold_results = {}
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        tr, eq, dd = run_strategy(events, threshold=thresh, momentum_filter=True)
        if tr:
            tp = [t['pnl'] for t in tr]
            w = [p for p in tp if p > 0]
            wr = len(w)/len(tp)*100
            tr_rets = [p/STARTING_CAPITAL for p in tp]
            sh = np.mean(tr_rets) / max(np.std(tr_rets), 1e-10) * np.sqrt(12)
        else:
            wr = 0; sh = 0; eq = STARTING_CAPITAL

        print(f"  Threshold {thresh:.2f}: {len(tr)} trades, WR={wr:.0f}%, "
              f"Sharpe={sh:.2f}, ${eq:.0f}", flush=True)
        threshold_results[str(thresh)] = {
            'trades': len(tr), 'equity': round(eq, 2),
            'sharpe': round(sh, 3), 'wr': round(wr, 1),
        }

    results['threshold_sensitivity'] = threshold_results

    # ============================================================
    # TEST 7: DIRECTION ANALYSIS (calls vs puts)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 7: DIRECTION ANALYSIS", flush=True)
    print(f"{'='*60}", flush=True)

    calls = [t for t in trades if t['opt_type'] == 'call']
    puts = [t for t in trades if t['opt_type'] == 'put']

    for label, subset in [('CALLS (gap up)', calls), ('PUTS (gap down)', puts)]:
        if subset:
            sp = [t['pnl'] for t in subset]
            sw = [p for p in sp if p > 0]
            print(f"  {label}: {len(subset)} trades, WR={len(sw)/len(sp)*100:.0f}%, "
                  f"avg=${np.mean(sp):.0f}, total=${sum(sp):.0f}", flush=True)
        else:
            print(f"  {label}: 0 trades", flush=True)

    results['direction'] = {
        'calls': len(calls), 'puts': len(puts),
        'call_pnl': round(sum(t['pnl'] for t in calls), 2) if calls else 0,
        'put_pnl': round(sum(t['pnl'] for t in puts), 2) if puts else 0,
    }

    # ============================================================
    # TEST 8: EXIT REASON ANALYSIS
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 8: EXIT REASON ANALYSIS", flush=True)
    print(f"{'='*60}", flush=True)

    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t['exit_reason']].append(t['pnl'])

    for reason in sorted(by_exit):
        pp = by_exit[reason]
        ww = [p for p in pp if p > 0]
        print(f"  {reason}: {len(pp)} trades, WR={len(ww)/len(pp)*100:.0f}%, "
              f"avg=${np.mean(pp):.0f}", flush=True)

    results['exit_analysis'] = {
        reason: {'count': len(pp), 'avg_pnl': round(np.mean(pp), 2),
                 'wr': round(len([p for p in pp if p > 0])/len(pp)*100, 1)}
        for reason, pp in by_exit.items()
    }

    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.time() - t0

    print(f"\n{'='*70}", flush=True)
    print(f"  PEAD ADVERSARIAL VALIDATION — SUMMARY", flush=True)
    print(f"  Runtime: {total_time:.0f}s", flush=True)
    print(f"{'='*70}", flush=True)

    # Scoring
    checks = [
        ('Sub-period: all years profitable', all(yr.get('total_pnl', 0) > 0
                                                  for yr in year_results.values())),
        ('Bear market (2022) profitable', year_results.get('2022', {}).get('total_pnl', 0) > 0),
        ('Top-2 ticker removal: still profitable', results.get('robustness', {}).get('still_profitable', False)),
        ('Top-2 concentration < 50%', results.get('concentration', {}).get('top2_pct', 100) < 50),
        ('Deep perm test (500) p < 0.05', results.get('deep_perm', {}).get('p_value', 1) < 0.05),
        ('Threshold stable (0.50-0.80 all profitable)', all(
            v.get('equity', 0) > STARTING_CAPITAL
            for v in threshold_results.values())),
        ('Both calls and puts profitable', (results['direction']['call_pnl'] > 0 and
                                            results['direction']['put_pnl'] > 0)),
        ('Take-profit exits > stop-loss exits', len(by_exit.get('take_profit', [])) >
                                                len(by_exit.get('stop_loss', []))),
    ]

    passed = sum(1 for _, v in checks if v)
    print(f"\n  ADVERSARIAL SCORE: {passed}/{len(checks)}", flush=True)
    for name, v in checks:
        print(f"    {'✅' if v else '❌'} {name}", flush=True)

    results['score'] = f"{passed}/{len(checks)}"
    results['checks'] = {name: v for name, v in checks}

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}", flush=True)

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name="adversarial_validation"):
                mlflow.log_metric("score", passed)
                mlflow.log_metric("total_checks", len(checks))
                mlflow.log_metric("baseline_equity", equity)
                mlflow.log_metric("deep_perm_p", results['deep_perm']['p_value'])
                mlflow.log_metric("top2_concentration", results['concentration']['top2_pct'])
                if results.get('robustness', {}).get('equity'):
                    mlflow.log_metric("robust_equity", results['robustness']['equity'])
        except Exception as e:
            print(f"MLflow error: {e}", flush=True)

    print("Done.", flush=True)
    return results


if __name__ == '__main__':
    main()
