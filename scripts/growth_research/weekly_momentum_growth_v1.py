#!/usr/bin/env python3
"""
Weekly Momentum on Growth Stocks v1
====================================
HIGH-FREQUENCY strategy for $645 RH account — targets 200+ trades for
statistical validation.

RATIONALE:
- All PEAD strategies fail perm test due to ~20 trades in 4.5 years
- We need MORE TRADES to prove statistical significance
- Weekly rebalance on 52 growth stocks → ~230 trades/year → 1000+ in OOT

APPROACH:
- Each Friday: rank 52 growth stocks by recent momentum
- Buy top-N stocks with fractional shares
- Hold 1 week, sell and rebalance
- LGBM learns which momentum signals persist

KEY DIFFERENCES from sector rotation:
- Individual stocks (higher vol = bigger alpha opportunity)
- Fractional shares (exact position sizing even with $645)
- Weekly (not monthly) → more data points
- 52 stocks (not 11 sectors) → more diversification

6 VARIANTS:
  A. Top-3, 1-week momentum rank (simple)
  B. Top-5, LGBM-ranked (learned momentum persistence)
  C. Top-3, momentum + mean-reversion hybrid
  D. Top-3, VIX-scaled position sizing
  E. Top-3, sector-balanced (max 1 per sector)
  F. Top-1 concentrated (highest conviction)

UNIVERSE: 52 growth stocks (fractional shares on RH)
PERIOD: 2022-01-01 to 2026-07-25
ACCOUNT: $645
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'weekly_momentum_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

SECTOR_MAP = {
    'AAPL': 'tech', 'MSFT': 'tech', 'AMZN': 'consumer', 'GOOGL': 'tech', 'META': 'tech',
    'NVDA': 'semi', 'TSLA': 'auto', 'AMD': 'semi', 'NFLX': 'media', 'PYPL': 'fintech',
    'SHOP': 'ecom', 'ROKU': 'media', 'SNAP': 'social', 'PINS': 'social', 'COIN': 'crypto',
    'HOOD': 'fintech', 'PLTR': 'tech', 'RBLX': 'gaming', 'ENPH': 'clean', 'DXCM': 'health',
    'ALGN': 'health', 'CMG': 'restaurant', 'FSLR': 'clean', 'ARM': 'semi', 'SOFI': 'fintech',
    'RIVN': 'auto', 'ABNB': 'travel', 'UBER': 'transport', 'LYFT': 'transport', 'DASH': 'delivery',
    'NET': 'cloud', 'CRWD': 'cyber', 'ZS': 'cyber', 'PANW': 'cyber', 'MDB': 'cloud',
    'SNOW': 'cloud', 'DDOG': 'cloud', 'TTD': 'adtech', 'BILL': 'fintech', 'UPST': 'fintech',
    'AFRM': 'fintech', 'U': 'gaming', 'RKLB': 'space', 'SMCI': 'tech', 'MELI': 'ecom',
    'SE': 'ecom', 'BABA': 'ecom', 'JD': 'ecom', 'PDD': 'ecom', 'NIO': 'auto',
    'XPEV': 'auto', 'LI': 'auto',
}

STARTING_CAPITAL = 645.0
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'
N_PERMUTATIONS = 200
TRAIN_WEEKS = 26  # 6 months lookback for LGBM training

FEATURE_COLS = [
    'ret_1w', 'ret_2w', 'ret_4w', 'ret_13w', 'ret_26w',
    'vol_21d', 'rsi_14', 'dist_52w_high', 'rel_str_spy_4w',
    'vol_ratio', 'sector_avg_ret_1w',
]


def load_data():
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'weekly_momentum_cache.parquet')

    if os.path.exists(cache_path):
        prices = pd.read_parquet(cache_path)
        print(f"Loaded cache: {len(prices)} rows", flush=True)
        return prices

    print(f"Downloading {len(STOCK_UNIVERSE)+2} tickers...", flush=True)
    all_tickers = STOCK_UNIVERSE + ['SPY', '^VIX']
    frames = []
    for t in all_tickers:
        try:
            df = yf.download(t, start='2020-01-01', end=END_DATE, progress=False, auto_adjust=True)
            if len(df) < 100: continue
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df['ticker'] = t
            df.index.name = 'date'
            frames.append(df)
        except:
            pass
    prices = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices.to_parquet(cache_path)
    print(f"Saved {len(prices)} price rows", flush=True)
    return prices


def compute_features(prices, date, ticker, spy_data, vix_data, all_returns):
    """Compute features for a single stock on a single date."""
    try:
        tp = prices.loc[ticker]
        mask = tp.index <= date
        if mask.sum() < 60:
            return None
        hist = tp.loc[mask].tail(260)  # ~1 year
        close = hist['close']

        feats = {}
        # Returns at multiple horizons
        for w, label in [(5, '1w'), (10, '2w'), (21, '4w'), (63, '13w'), (126, '26w')]:
            if len(close) > w:
                feats[f'ret_{label}'] = float(close.iloc[-1] / close.iloc[-w] - 1)
            else:
                feats[f'ret_{label}'] = 0

        # Volatility
        feats['vol_21d'] = float(close.pct_change().tail(21).std() * np.sqrt(252))

        # RSI
        rets = close.pct_change().tail(15).dropna()
        gains = rets.clip(lower=0).mean()
        losses = (-rets).clip(lower=0).mean()
        feats['rsi_14'] = float(100 - 100 / (1 + gains/losses)) if losses > 0 else 50

        # Distance from 52w high
        if len(close) >= 252:
            feats['dist_52w_high'] = float(close.iloc[-1] / close.tail(252).max() - 1)
        else:
            feats['dist_52w_high'] = float(close.iloc[-1] / close.max() - 1)

        # Relative strength vs SPY
        feats['rel_str_spy_4w'] = 0
        if spy_data is not None:
            spy_mask = spy_data.index <= date
            if spy_mask.sum() >= 21:
                spy_close = spy_data[spy_mask].tail(21)
                feats['rel_str_spy_4w'] = feats['ret_4w'] - float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)

        # Volume ratio
        if 'volume' in hist.columns:
            recent = hist['volume'].tail(5).mean()
            avg = hist['volume'].tail(22).mean()
            feats['vol_ratio'] = float(recent / max(avg, 1))
        else:
            feats['vol_ratio'] = 1.0

        # Sector average return
        sector = SECTOR_MAP.get(ticker, 'other')
        sector_tickers = [t for t, s in SECTOR_MAP.items() if s == sector and t != ticker]
        sector_rets = []
        for st in sector_tickers:
            if st in all_returns:
                sector_rets.append(all_returns[st])
        feats['sector_avg_ret_1w'] = float(np.mean(sector_rets)) if sector_rets else 0

        return feats
    except:
        return None


def run_variant(vkey, cfg, prices):
    print(f"\n{'='*60}", flush=True)
    print(f"  VARIANT {vkey}: {cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    # Get trading days
    try:
        spy_prices = prices.loc['SPY']
    except:
        print("  ERROR: No SPY data", flush=True)
        return empty_result(vkey, cfg), []

    try:
        vix_data = prices.loc['^VIX', 'close']
    except:
        vix_data = None

    spy_close = spy_prices['close']
    all_dates = spy_prices.index.sort_values()

    # Identify Friday dates for rebalancing
    fridays = [d for d in all_dates if d.weekday() == 4]
    oot_start = pd.Timestamp(START_DATE)
    oot_fridays = [f for f in fridays if f >= oot_start]

    print(f"  OOT Fridays: {len(oot_fridays)}", flush=True)

    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0
    trades = []
    weekly_returns = []
    positions = {}  # ticker -> (shares, entry_price)

    train_start_idx = max(0, fridays.index(oot_fridays[0]) - TRAIN_WEEKS) if oot_fridays[0] in fridays else 0

    for fi, friday in enumerate(oot_fridays):
        # Skip if we can't find next Friday
        next_idx = oot_fridays.index(friday) + 1 if friday in oot_fridays else -1
        if next_idx >= len(oot_fridays):
            continue
        next_friday = oot_fridays[next_idx]

        # Compute 1-week returns for all stocks (for ranking)
        all_returns_1w = {}
        stock_features = {}

        for ticker in STOCK_UNIVERSE:
            try:
                tp = prices.loc[ticker]
                mask = tp.index <= friday
                if mask.sum() < 10:
                    continue
                close = tp.loc[mask, 'close']
                if len(close) < 6:
                    continue

                ret_1w = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 6 else 0
                all_returns_1w[ticker] = ret_1w

                feats = compute_features(prices, friday, ticker, spy_close, vix_data, all_returns_1w)
                if feats:
                    stock_features[ticker] = feats
            except:
                continue

        if len(stock_features) < 10:
            continue

        # Close existing positions
        week_pnl = 0
        for ticker, (shares, entry_p) in list(positions.items()):
            try:
                tp = prices.loc[ticker]
                mask = tp.index <= friday
                if mask.sum() > 0:
                    exit_p = float(tp.loc[mask, 'close'].iloc[-1])
                    pnl = (exit_p - entry_p) * shares
                    week_pnl += pnl
                    trades.append({
                        'ticker': ticker, 'date': str(friday.date()),
                        'direction': 'LONG', 'entry': entry_p, 'exit': exit_p,
                        'shares': shares, 'pnl': round(pnl, 2),
                    })
            except:
                pass

        equity += week_pnl
        if equity > peak: peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0
        if dd < max_dd: max_dd = dd
        weekly_returns.append(week_pnl / max(equity - week_pnl, 1))

        positions = {}

        # Select new positions
        if cfg.get('use_lgbm') and fi >= TRAIN_WEEKS:
            # LGBM ranking
            from sklearn.ensemble import GradientBoostingRegressor

            # Build training data from past weeks
            # Target: next week return
            # This is simplified — we use the features and next-week returns
            train_X = []
            train_y = []

            for past_fi in range(max(0, fi - TRAIN_WEEKS * 2), fi):
                past_friday = oot_fridays[past_fi] if past_fi < len(oot_fridays) else None
                if past_friday is None: continue
                nf_idx = past_fi + 1
                if nf_idx >= len(oot_fridays): continue
                nf = oot_fridays[nf_idx]

                for ticker in STOCK_UNIVERSE:
                    try:
                        tp = prices.loc[ticker]
                        mask_pf = tp.index <= past_friday
                        mask_nf = tp.index <= nf

                        if mask_pf.sum() < 10 or mask_nf.sum() < 10: continue
                        c_pf = float(tp.loc[mask_pf, 'close'].iloc[-1])
                        c_nf = float(tp.loc[mask_nf, 'close'].iloc[-1])

                        feats = compute_features(prices, past_friday, ticker, spy_close, vix_data, {})
                        if feats:
                            train_X.append([feats.get(f, 0) for f in FEATURE_COLS])
                            train_y.append(c_nf / c_pf - 1)
                    except:
                        continue

            if len(train_X) >= 50:
                try:
                    model = GradientBoostingRegressor(
                        n_estimators=50, max_depth=3, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    )
                    model.fit(np.array(train_X), np.array(train_y))

                    # Predict next week returns
                    predictions = {}
                    for ticker, feats in stock_features.items():
                        X = np.array([[feats.get(f, 0) for f in FEATURE_COLS]])
                        predictions[ticker] = model.predict(X)[0]

                    ranked = sorted(predictions.items(), key=lambda x: -x[1])
                except:
                    ranked = sorted(all_returns_1w.items(), key=lambda x: -x[1])
            else:
                ranked = sorted(all_returns_1w.items(), key=lambda x: -x[1])
        else:
            # Simple momentum ranking
            if cfg.get('mean_reversion_hybrid'):
                # Rank by 4-week momentum but filter for 1-week pullback
                scores = {}
                for ticker, feats in stock_features.items():
                    mom_4w = feats.get('ret_4w', 0)
                    ret_1w = feats.get('ret_1w', 0)
                    # Want stocks in uptrend (4w mom > 0) that pulled back this week (1w ret < 0)
                    if mom_4w > 0 and ret_1w < 0:
                        scores[ticker] = mom_4w - ret_1w  # favor big uptrend + big pullback
                    elif mom_4w > 0:
                        scores[ticker] = mom_4w * 0.5  # still consider but lower score
                ranked = sorted(scores.items(), key=lambda x: -x[1])
            else:
                ranked = sorted(all_returns_1w.items(), key=lambda x: -x[1])

        # Select top N
        top_n = cfg.get('top_n', 3)
        selected = []
        sectors_used = set()

        for ticker, score in ranked:
            if len(selected) >= top_n:
                break
            if cfg.get('sector_balanced'):
                sector = SECTOR_MAP.get(ticker, 'other')
                if sector in sectors_used:
                    continue
                sectors_used.add(sector)
            selected.append(ticker)

        if not selected:
            continue

        # Position sizing
        position_pct = 1.0 / len(selected)  # equal weight

        if cfg.get('vix_scaled') and vix_data is not None:
            vix_mask = vix_data.index <= friday
            if vix_mask.any():
                vix_val = float(vix_data[vix_mask].iloc[-1])
                if vix_val > 30:
                    position_pct *= 0.50
                elif vix_val > 25:
                    position_pct *= 0.75
                # Low VIX = normal or slightly larger
                elif vix_val < 15:
                    position_pct *= 1.10

        for ticker in selected:
            try:
                tp = prices.loc[ticker]
                mask = tp.index <= friday
                if mask.sum() == 0: continue
                entry_p = float(tp.loc[mask, 'close'].iloc[-1])
                alloc = equity * position_pct
                shares = alloc / entry_p  # fractional
                if shares * entry_p < 5: continue  # min $5
                positions[ticker] = (shares, entry_p)
            except:
                continue

        if fi < 3 and positions:
            print(f"  Week {fi+1} ({friday.date()}): eq=${equity:.2f}, hold {list(positions.keys())}", flush=True)

    # Close final positions at end
    final_pnl = 0
    for ticker, (shares, entry_p) in positions.items():
        try:
            tp = prices.loc[ticker]
            exit_p = float(tp['close'].iloc[-1])
            pnl = (exit_p - entry_p) * shares
            final_pnl += pnl
            trades.append({
                'ticker': ticker, 'date': str(all_dates[-1].date()),
                'direction': 'LONG', 'entry': entry_p, 'exit': exit_p,
                'shares': shares, 'pnl': round(pnl, 2),
            })
        except:
            pass
    equity += final_pnl

    # Results
    if not trades:
        return empty_result(vkey, cfg), []

    pnls = np.array([t['pnl'] for t in trades])
    n = len(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    sharpe = float(np.mean(pnls)/np.std(pnls)*np.sqrt(n)) if np.std(pnls) > 0 else 0

    # Better: use weekly returns for Sharpe
    wr_arr = np.array(weekly_returns)
    weekly_sharpe = float(np.mean(wr_arr)/np.std(wr_arr)*np.sqrt(52)) if len(wr_arr) > 1 and np.std(wr_arr) > 0 else 0

    down = pnls[pnls < 0]
    sortino = float(np.mean(pnls)/np.std(down)*np.sqrt(n)) if len(down)>0 and np.std(down)>0 else 0

    pf = float(sum(wins)/abs(sum(losses))) if len(losses)>0 and sum(losses)!=0 else float('inf')
    wr = float(len(wins)/n*100)

    # Ticker concentration
    ticker_pnl = defaultdict(float)
    for t in trades:
        ticker_pnl[t['ticker']] += t['pnl']
    t_abs = {t: abs(p) for t,p in ticker_pnl.items()}
    total_abs = sum(t_abs.values())
    top2 = sum(sorted(t_abs.values(), reverse=True)[:2])
    concentration = top2/total_abs if total_abs > 0 else 0

    # Permutation test (use weekly returns for proper Sharpe)
    perm_count = 0
    for _ in range(N_PERMUTATIONS):
        s = np.random.permutation(wr_arr)
        ps = float(np.mean(s)/np.std(s)*np.sqrt(52)) if np.std(s)>0 else 0
        if ps >= weekly_sharpe: perm_count += 1
    perm_p = perm_count / N_PERMUTATIONS

    # MC CI
    mc_final = [STARTING_CAPITAL * np.prod(1 + np.random.choice(wr_arr, len(wr_arr), replace=True)) for _ in range(1000)]
    mc_5th = np.percentile(mc_final, 5)

    # Regime: year-based
    year_returns = defaultdict(list)
    for i, wr_val in enumerate(weekly_returns):
        if i < len(oot_fridays):
            year = str(oot_fridays[i].year)
            year_returns[year].append(wr_val)

    year_sharpes = {}
    for year, rets in year_returns.items():
        r = np.array(rets)
        year_sharpes[year] = float(np.mean(r)/np.std(r)*np.sqrt(52)) if len(r)>=4 and np.std(r)>0 else 0

    green = [s for y,s in year_sharpes.items() if y != '2022']
    red = [s for y,s in year_sharpes.items() if y == '2022']
    if green and red:
        regime_gap = abs(np.mean(green)-np.mean(red)) / max(max(abs(s) for s in green+red), 0.01)
    else:
        regime_gap = 1.0

    # SPY comparison
    spy_start = float(spy_close.loc[spy_close.index >= pd.Timestamp(START_DATE)].iloc[0])
    spy_end = float(spy_close.iloc[-1])
    spy_return = (spy_end / spy_start - 1) * 100
    strat_return = (equity / STARTING_CAPITAL - 1) * 100
    alpha = strat_return - spy_return

    gates = {
        'sharpe_gt_1': weekly_sharpe >= 1.0,
        'perm_p_lt_005': perm_p < 0.05,
        'wr_gt_40': wr >= 40,
        'regime_balance': regime_gap < 0.50,
        'mc_ci_positive': mc_5th > STARTING_CAPITAL,
    }
    gates_passed = sum(gates.values())

    print(f"\n  Trades: {n} | Weekly Sharpe: {weekly_sharpe:.3f} | PF: {pf:.3f} | WR: {wr:.1f}%", flush=True)
    print(f"  MDD: {max_dd*100:.2f}% | Final: ${equity:.2f} | Return: {strat_return:.1f}%", flush=True)
    print(f"  SPY Return: {spy_return:.1f}% | Alpha: {alpha:.1f}%", flush=True)
    print(f"  Perm p: {perm_p:.3f} | Concentration: {concentration*100:.1f}% | Tickers: {len(ticker_pnl)}", flush=True)
    print(f"  Per-year Sharpe: {year_sharpes}", flush=True)

    print(f"  Top tickers:", flush=True)
    for t,p in sorted(ticker_pnl.items(), key=lambda x: -abs(x[1]))[:6]:
        print(f"    {t}: ${p:.2f}", flush=True)

    print(f"\n  5-Gate: {gates_passed}/5 {'✅' if gates_passed>=4 else '❌'}", flush=True)
    for g,v in gates.items():
        val = {'sharpe_gt_1': weekly_sharpe, 'perm_p_lt_005': perm_p, 'wr_gt_40': wr,
               'regime_balance': regime_gap, 'mc_ci_positive': mc_5th}.get(g)
        thr = {'sharpe_gt_1': 1.0, 'perm_p_lt_005': 0.05, 'wr_gt_40': 40,
               'regime_balance': 0.50, 'mc_ci_positive': 0}.get(g)
        print(f"    {g}: {'PASS' if v else 'FAIL'} ({val:.3f} vs {thr})", flush=True)

    return {
        'trades': n, 'sharpe': round(weekly_sharpe, 3), 'sortino': round(sortino, 3),
        'pf': round(pf, 3), 'wr': round(wr, 1), 'mdd': round(max_dd*100, 2),
        'final_equity': round(equity, 2), 'perm_p': round(perm_p, 3),
        'gates_passed': gates_passed, 'gates': gates,
        'variant': vkey, 'name': cfg['name'],
        'concentration': round(concentration*100, 1), 'n_tickers': len(ticker_pnl),
        'alpha_vs_spy': round(alpha, 1), 'spy_return': round(spy_return, 1),
        'year_sharpes': year_sharpes, 'regime_gap': round(regime_gap, 3),
    }, trades


def empty_result(vkey, cfg):
    return {
        'trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
        'final_equity': STARTING_CAPITAL, 'mdd': 0, 'perm_p': 1.0,
        'gates_passed': 0, 'gates': {}, 'variant': vkey, 'name': cfg['name'],
        'concentration': 0, 'n_tickers': 0, 'alpha_vs_spy': 0,
    }


VARIANTS = {
    'A': {'name': 'Top-3 Simple Momentum', 'top_n': 3, 'use_lgbm': False,
          'mean_reversion_hybrid': False, 'vix_scaled': False, 'sector_balanced': False},
    'B': {'name': 'Top-5 LGBM-Ranked', 'top_n': 5, 'use_lgbm': True,
          'mean_reversion_hybrid': False, 'vix_scaled': False, 'sector_balanced': False},
    'C': {'name': 'Top-3 Mom + Mean-Reversion Hybrid', 'top_n': 3, 'use_lgbm': False,
          'mean_reversion_hybrid': True, 'vix_scaled': False, 'sector_balanced': False},
    'D': {'name': 'Top-3 VIX-Scaled', 'top_n': 3, 'use_lgbm': False,
          'mean_reversion_hybrid': False, 'vix_scaled': True, 'sector_balanced': False},
    'E': {'name': 'Top-3 Sector-Balanced', 'top_n': 3, 'use_lgbm': False,
          'mean_reversion_hybrid': False, 'vix_scaled': False, 'sector_balanced': True},
    'F': {'name': 'Top-1 Concentrated', 'top_n': 1, 'use_lgbm': False,
          'mean_reversion_hybrid': False, 'vix_scaled': False, 'sector_balanced': False},
}


def main():
    print(f"Running on: {LVL3_ROOT}", flush=True)

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('weekly_momentum_growth_v1')
            print("MLflow OK", flush=True)
        except Exception as e:
            print(f"MLflow error: {e}", flush=True)

    print("="*70, flush=True)
    print("  WEEKLY MOMENTUM ON GROWTH STOCKS v1", flush=True)
    print(f"  Target: 200+ trades for statistical validation", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print("="*70, flush=True)

    prices = load_data()

    all_results = []
    for vkey in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vkey]
        t0 = datetime.now()
        result, trades = run_variant(vkey, cfg, prices)
        elapsed = (datetime.now() - t0).total_seconds()
        result['runtime_s'] = round(elapsed, 1)

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"wkly_{vkey}_{cfg['name'][:25]}"):
                    mlflow.log_params({k: str(v) for k,v in cfg.items()})
                    mlflow.log_metrics({
                        'sharpe': result['sharpe'], 'n_trades': result['trades'],
                        'pf': min(result.get('pf', 0), 999), 'wr': result['wr'],
                        'mdd': result['mdd'], 'final_equity': result['final_equity'],
                        'perm_p': result['perm_p'], 'gates_passed': result['gates_passed'],
                        'alpha_vs_spy': result.get('alpha_vs_spy', 0),
                    })
            except: pass

        all_results.append(result)

    # Summary
    print("\n" + "="*70, flush=True)
    print("  SUMMARY — WEEKLY MOMENTUM GROWTH v1", flush=True)
    print("="*70, flush=True)
    print(f"\n  {'V':<3} {'Name':<35} {'#':<6} {'Shrp':<7} {'PF':<6} {'WR%':<6} "
          f"{'MDD%':<7} {'Final$':<9} {'Alpha':<7} {'RegGap':<7} {'G':<4}", flush=True)

    for r in all_results:
        print(f"  {r['variant']:<3} {r['name'][:35]:<35} {r['trades']:<6} {r['sharpe']:<7} "
              f"{r.get('pf', 0):<6.1f} {r['wr']:<6.1f} {r['mdd']:<7.1f} ${r['final_equity']:<8.0f} "
              f"{r.get('alpha_vs_spy', 0):<7.1f} {r.get('regime_gap', 0):<7.3f} {r['gates_passed']}/5", flush=True)

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({'variants': all_results, 'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    best = max(all_results, key=lambda r: r['gates_passed']*10 + r['sharpe'])
    print(f"\n  BEST: {best['variant']} — {best['name']}", flush=True)
    print(f"    Sharpe {best['sharpe']}, {best['trades']} trades, "
          f"${STARTING_CAPITAL}→${best['final_equity']}, Alpha {best.get('alpha_vs_spy', 0)}%", flush=True)


if __name__ == '__main__':
    main()
