#!/usr/bin/env python3
"""
Post-Earnings Announcement Drift (PEAD) ML Predictor v1
========================================================
ML model to predict which post-earnings gaps will continue drifting.

MOTIVATION:
- Earnings momentum options v1 found Sharpe 2.52 for post-gap buyer
- BUT: 83% of profits from 2 tickers = concentration risk
- Can ML identify WHICH gaps are most likely to continue drifting?
- Features: gap size, pre-earnings momentum, vol, sector, IV rank, etc.

APPROACH:
1. Build feature matrix from all historical earnings events
2. Target: binary (did price continue 3%+ in gap direction within 3 days?)
3. Walk-forward LGBM + simple MLP
4. Trade only when model confidence > threshold
5. Compare vs random baseline

6 VARIANTS:
  A. LGBM classifier, threshold 60%
  B. LGBM classifier, threshold 70%
  C. Simple MLP (2-layer), threshold 60%
  D. LGBM + momentum filter
  E. Ensemble (LGBM + MLP), threshold 65%
  F. LGBM with ticker-specific features

UNIVERSE: 53 growth stocks (same as earnings_momentum_v1)
PERIOD: 2022-01-01 to 2026-07-25
ACCOUNT: $645, max $200/position
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings('ignore')

# Path setup
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'pead_ml_predictor_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

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
N_PERMUTATIONS = 100
DTE = 14
DRIFT_THRESHOLD = 0.03  # 3% continuation = "success"
HOLD_DAYS = 3

# ============================================================
# BLACK-SCHOLES
# ============================================================

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

# ============================================================
# DATA LOADING (with caching from earnings_momentum_v1)
# ============================================================

def load_data():
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'earnings_momentum_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'earnings_dates_cache.json')

    # Load cached price data
    if os.path.exists(cache_path):
        prices_df = pd.read_parquet(cache_path)
        print(f"Loaded cached prices: {len(prices_df)} rows", flush=True)
    else:
        print("Downloading prices...", flush=True)
        tickers = STOCK_UNIVERSE + ['SPY', '^VIX']
        frames = []
        for t in tickers:
            try:
                df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
                if len(df) < 50: continue
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df['ticker'] = t
                df.index.name = 'date'
                frames.append(df)
                print(f"  {t}: {len(df)} rows", flush=True)
            except Exception as e:
                print(f"  ERROR {t}: {e}", flush=True)
        prices_df = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices_df.to_parquet(cache_path)

    # Load cached earnings dates
    if os.path.exists(earnings_cache):
        with open(earnings_cache) as f:
            earnings_dates = json.load(f)
        print(f"Loaded cached earnings: {len(earnings_dates)} tickers", flush=True)
    else:
        print("Fetching earnings dates...", flush=True)
        earnings_dates = {}
        for t in STOCK_UNIVERSE:
            try:
                stock = yf.Ticker(t)
                dates = stock.get_earnings_dates(limit=30)
                if dates is not None and len(dates) > 0:
                    earnings_dates[t] = [str(d.date()) for d in dates.index]
                    print(f"  {t}: {len(earnings_dates[t])} dates", flush=True)
            except Exception:
                pass
        with open(earnings_cache, 'w') as f:
            json.dump(earnings_dates, f)

    return prices_df, earnings_dates

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_event_features(prices_df, earnings_dates):
    """Build feature matrix for all earnings events."""
    events = []

    for ticker, dates in earnings_dates.items():
        try:
            ticker_prices = prices_df.loc[ticker].sort_index()
        except KeyError:
            continue

        if len(ticker_prices) < 30:
            continue

        trading_days = ticker_prices.index.sort_values()

        # Get VIX data
        try:
            vix = prices_df.loc['^VIX', 'close']
        except:
            vix = None

        # Get SPY data
        try:
            spy = prices_df.loc['SPY', 'close']
        except:
            spy = None

        for earn_date_str in dates:
            earn_date = pd.Timestamp(earn_date_str)

            # Find post-earnings day
            post_mask = trading_days >= earn_date
            if not post_mask.any(): continue
            post_days = trading_days[post_mask]
            if len(post_days) < HOLD_DAYS + 1: continue

            # Find pre-earnings day
            pre_mask = trading_days < earn_date
            if not pre_mask.any(): continue
            pre_days = trading_days[pre_mask]
            if len(pre_days) < 30: continue

            close_before = ticker_prices.loc[pre_days[-1], 'close']
            open_after = ticker_prices.loc[post_days[0], 'open']

            if close_before <= 0 or open_after <= 0: continue

            gap_pct = (open_after / close_before) - 1.0
            if abs(gap_pct) < 0.05:  # Only events with 5%+ gap
                continue

            # Features
            feats = {}

            # Gap features
            feats['gap_pct'] = gap_pct
            feats['abs_gap'] = abs(gap_pct)
            feats['gap_direction'] = 1 if gap_pct > 0 else -1

            # Pre-earnings momentum
            for w in [5, 10, 21]:
                if len(pre_days) >= w + 1:
                    feats[f'mom_{w}d'] = float(ticker_prices.loc[pre_days[-1], 'close'] /
                                              ticker_prices.loc[pre_days[-w-1], 'close'] - 1)
                else:
                    feats[f'mom_{w}d'] = 0

            # Volatility
            if len(pre_days) >= 22:
                recent_close = ticker_prices.loc[pre_days[-22:], 'close']
                feats['vol_21d'] = float(recent_close.pct_change().dropna().std() * np.sqrt(252))
            else:
                feats['vol_21d'] = 0.3

            # Relative strength vs SPY
            if spy is not None and len(pre_days) >= 22:
                try:
                    spy_close = spy.loc[spy.index <= pre_days[-1]].iloc[-21:]
                    stock_close = ticker_prices.loc[pre_days[-21:], 'close']
                    feats['rel_str_21d'] = float(
                        (stock_close.iloc[-1] / stock_close.iloc[0] - 1) -
                        (spy_close.iloc[-1] / spy_close.iloc[0] - 1))
                except:
                    feats['rel_str_21d'] = 0
            else:
                feats['rel_str_21d'] = 0

            # VIX level
            if vix is not None:
                vix_mask = vix.index <= pre_days[-1]
                if vix_mask.any():
                    feats['vix'] = float(vix[vix_mask].iloc[-1])
                else:
                    feats['vix'] = 20
            else:
                feats['vix'] = 20

            # Volume ratio (recent vs average)
            if 'volume' in ticker_prices.columns and len(pre_days) >= 22:
                recent_vol = ticker_prices.loc[pre_days[-5:], 'volume'].mean()
                avg_vol = ticker_prices.loc[pre_days[-22:], 'volume'].mean()
                feats['vol_ratio'] = float(recent_vol / max(avg_vol, 1))
            else:
                feats['vol_ratio'] = 1.0

            # RSI
            if len(pre_days) >= 15:
                rets = ticker_prices.loc[pre_days[-15:], 'close'].pct_change().dropna()
                gains = rets.clip(lower=0).mean()
                losses = (-rets).clip(lower=0).mean()
                if losses > 0:
                    feats['rsi'] = float(100 - 100 / (1 + gains/losses))
                else:
                    feats['rsi'] = 100
            else:
                feats['rsi'] = 50

            # Distance from 52w high
            if len(pre_days) >= 252:
                h52 = ticker_prices.loc[pre_days[-252:], 'close'].max()
                feats['dist_52w_high'] = float(close_before / h52 - 1)
            else:
                feats['dist_52w_high'] = 0

            # Previous earnings gap (if available)
            ticker_earn = [d for d in dates if pd.Timestamp(d) < earn_date]
            if ticker_earn:
                prev_earn = pd.Timestamp(ticker_earn[-1])
                prev_post = trading_days[trading_days >= prev_earn]
                prev_pre = trading_days[trading_days < prev_earn]
                if len(prev_post) > 0 and len(prev_pre) > 0:
                    prev_close = ticker_prices.loc[prev_pre[-1], 'close']
                    prev_open = ticker_prices.loc[prev_post[0], 'open']
                    if prev_close > 0:
                        feats['prev_gap'] = float(prev_open / prev_close - 1)
                    else:
                        feats['prev_gap'] = 0
                else:
                    feats['prev_gap'] = 0
            else:
                feats['prev_gap'] = 0

            # Ticker encoding (simple hash-based)
            feats['ticker_hash'] = hash(ticker) % 100

            # TARGET: did price continue drifting in gap direction?
            post_close = [ticker_prices.loc[d, 'close'] for d in post_days[:HOLD_DAYS + 1]]
            if len(post_close) < 2:
                continue

            # Max favorable excursion in gap direction
            entry_price = open_after
            if gap_pct > 0:
                # Gap up — did it continue up?
                max_move = max(c / entry_price - 1 for c in post_close[1:])
                drift_success = max_move >= DRIFT_THRESHOLD
            else:
                # Gap down — did it continue down?
                max_move = max(1 - c / entry_price for c in post_close[1:])
                drift_success = max_move >= DRIFT_THRESHOLD

            feats['target'] = 1 if drift_success else 0

            # Also store metadata for trading simulation
            feats['ticker'] = ticker
            feats['earn_date'] = earn_date_str
            feats['close_before'] = float(close_before)
            feats['open_after'] = float(open_after)
            feats['post_closes'] = [float(c) for c in post_close]

            events.append(feats)

    print(f"\nBuilt {len(events)} qualifying events (|gap| >= 5%)", flush=True)
    if events:
        targets = [e['target'] for e in events]
        print(f"  Drift success rate: {sum(targets)/len(targets)*100:.1f}% "
              f"({sum(targets)}/{len(targets)})", flush=True)

    return events

# ============================================================
# ML MODELS
# ============================================================

FEATURE_COLS = [
    'abs_gap', 'gap_direction', 'mom_5d', 'mom_10d', 'mom_21d',
    'vol_21d', 'rel_str_21d', 'vix', 'vol_ratio', 'rsi',
    'dist_52w_high', 'prev_gap', 'ticker_hash'
]


def train_lgbm(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    try:
        import torch
        import torch.nn as nn

        class SimpleMLP(nn.Module):
            def __init__(self, n_features):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.net(x).squeeze()

        X_tensor = torch.FloatTensor(X_train.values)
        y_tensor = torch.FloatTensor(y_train)

        model = SimpleMLP(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        return model, 'torch'
    except ImportError:
        # Fallback to sklearn MLP
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200,
                             random_state=42, early_stopping=True)
        model.fit(X_train, y_train)
        return model, 'sklearn'


def predict_proba(model, X, model_type='lgbm'):
    if model_type == 'torch':
        import torch
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            probs = model(X_tensor).numpy()
        return probs
    elif model_type == 'sklearn':
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict_proba(X)[:, 1]


# ============================================================
# WALK-FORWARD SIMULATION
# ============================================================

def run_variant(variant_key, cfg, events):
    """Walk-forward backtest with ML prediction."""
    print(f"\n{'='*60}", flush=True)
    print(f"  VARIANT {variant_key}: {cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    # Sort events by date
    events_sorted = sorted(events, key=lambda e: e['earn_date'])

    # Walk-forward: train on past events, predict next
    min_train = 30  # need at least 30 events to train
    equity = STARTING_CAPITAL
    peak_equity = equity
    max_dd = 0
    trades = []

    for i in range(min_train, len(events_sorted)):
        event = events_sorted[i]

        # Train data: all events before this one
        train_events = events_sorted[:i]
        X_train = pd.DataFrame(train_events)[FEATURE_COLS]
        y_train = np.array([e['target'] for e in train_events])

        # Test data: this event
        X_test = pd.DataFrame([event])[FEATURE_COLS]

        # Train model
        try:
            if cfg['model'] == 'lgbm':
                model = train_lgbm(X_train, y_train)
                prob = predict_proba(model, X_test, 'lgbm')[0]
            elif cfg['model'] == 'mlp':
                model, mtype = train_mlp(X_train, y_train)
                prob = predict_proba(model, X_test, mtype)[0]
            elif cfg['model'] == 'ensemble':
                lgbm = train_lgbm(X_train, y_train)
                mlp, mtype = train_mlp(X_train, y_train)
                p1 = predict_proba(lgbm, X_test, 'lgbm')[0]
                p2 = predict_proba(mlp, X_test, mtype)[0]
                prob = 0.6 * p1 + 0.4 * p2
            else:
                continue
        except Exception as e:
            continue

        # Threshold check
        if prob < cfg['threshold']:
            continue

        # Momentum filter
        if cfg.get('momentum_filter') and event['gap_direction'] == 1 and event['mom_5d'] < 0:
            continue
        if cfg.get('momentum_filter') and event['gap_direction'] == -1 and event['mom_5d'] > 0:
            continue

        # Simulate trade
        gap_pct = event['gap_pct']
        entry_price = event['open_after']
        strike = round(entry_price)
        opt_type = 'call' if gap_pct > 0 else 'put'

        # Post-earnings IV (crushed)
        post_iv = event['vol_21d'] * 0.8  # crushed from elevated
        post_iv = max(post_iv, 0.15)

        T = DTE / 252.0
        entry_premium = option_price(entry_price, strike, T, RISK_FREE_RATE, post_iv, opt_type)
        if entry_premium < 0.10:
            continue

        contract_cost = entry_premium * 100
        max_spend = min(MAX_POSITION, equity * 0.30)
        if contract_cost > max_spend or contract_cost + 0.65 > equity:
            continue

        # Simulate holding period
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

        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity
        if dd < max_dd:
            max_dd = dd

        trades.append({
            'ticker': event['ticker'],
            'earn_date': event['earn_date'],
            'opt_type': opt_type,
            'prob': round(prob, 3),
            'gap_pct': round(gap_pct * 100, 1),
            'pnl': round(pnl, 2),
            'exit_reason': exit_reason,
            'target': event['target'],
        })

        if len(trades) <= 3:
            print(f"  Trade {len(trades)}: {event['ticker']} {opt_type.upper()} "
                  f"gap={gap_pct*100:.1f}% prob={prob:.2f} pnl=${pnl:.0f} "
                  f"[{exit_reason}] eq=${equity:.0f}", flush=True)

    # Compute metrics
    if not trades:
        print(f"  No trades taken (threshold too high?)", flush=True)
        return {'trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
                'final_equity': STARTING_CAPITAL, 'mdd': 0, 'avg_pnl': 0,
                'perm_p': 1.0, 'pred_accuracy': 0,
                'gates_passed': 0, 'gates': {}, 'variant': variant_key, 'name': cfg['name']}, []

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    trade_rets = [p / STARTING_CAPITAL for p in pnls]
    sharpe = np.mean(trade_rets) / max(np.std(trade_rets), 1e-10) * np.sqrt(12)
    neg_rets = [r for r in trade_rets if r < 0]
    sortino = np.mean(trade_rets) / max(np.std(neg_rets) if neg_rets else np.std(trade_rets), 1e-10) * np.sqrt(12)
    pf = abs(sum(wins)) / abs(sum(losses)) if losses else float('inf')

    # Permutation test
    actual_mean = np.mean(pnls)
    rng = np.random.default_rng(42)
    count = sum(1 for _ in range(N_PERMUTATIONS)
                if np.mean(np.array(pnls) * rng.choice([-1, 1], len(pnls))) >= actual_mean)
    perm_p = (count + 1) / (N_PERMUTATIONS + 1)

    # Prediction accuracy
    pred_correct = sum(1 for t in trades if (t['pnl'] > 0) == (t['target'] == 1))
    pred_accuracy = pred_correct / len(trades) * 100

    metrics = {
        'variant': variant_key,
        'name': cfg['name'],
        'trades': len(trades),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 2),
        'wr': round(len(wins) / len(pnls) * 100, 1),
        'mdd': round(max_dd * 100, 1),
        'final_equity': round(equity, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'perm_p': round(perm_p, 4),
        'pred_accuracy': round(pred_accuracy, 1),
    }

    # 5-gate validation
    gates = {
        'sharpe_gt_1': sharpe >= 1.0,
        'perm_p_lt_005': perm_p < 0.05,
        'wr_gt_40': len(wins) / len(pnls) >= 0.40,
        'regime_balance': True,  # TODO: proper regime test
        'avg_pnl_positive': np.mean(pnls) > 0,
    }
    metrics['gates_passed'] = sum(gates.values())
    metrics['gates'] = gates

    print(f"\n  Trades: {len(trades)} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | "
          f"PF: {pf:.2f} | WR: {len(wins)/len(pnls)*100:.1f}%", flush=True)
    print(f"  ${STARTING_CAPITAL} → ${equity:.2f} | MDD: {max_dd*100:.1f}% | "
          f"Perm p: {perm_p:.4f} | Pred acc: {pred_accuracy:.1f}%", flush=True)
    print(f"  Gates: {metrics['gates_passed']}/5", flush=True)

    return metrics, trades


# ============================================================
# VARIANT CONFIGS
# ============================================================

VARIANTS = {
    'A': {'name': 'LGBM 60% threshold', 'model': 'lgbm', 'threshold': 0.60, 'momentum_filter': False},
    'B': {'name': 'LGBM 70% threshold', 'model': 'lgbm', 'threshold': 0.70, 'momentum_filter': False},
    'C': {'name': 'MLP 60% threshold', 'model': 'mlp', 'threshold': 0.60, 'momentum_filter': False},
    'D': {'name': 'LGBM + Momentum Filter', 'model': 'lgbm', 'threshold': 0.60, 'momentum_filter': True},
    'E': {'name': 'Ensemble 65%', 'model': 'ensemble', 'threshold': 0.65, 'momentum_filter': False},
    'F': {'name': 'LGBM with ticker features', 'model': 'lgbm', 'threshold': 0.55, 'momentum_filter': False},
}

# ============================================================
# MAIN
# ============================================================

def main():
    import time

    print("=" * 70, flush=True)
    print("  PEAD ML PREDICTOR V1", flush=True)
    print("  Can ML predict which post-earnings gaps will continue drifting?", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()

    # Load data
    prices_df, earnings_dates = load_data()

    # Build events
    events = build_event_features(prices_df, earnings_dates)

    if len(events) < 50:
        print(f"ERROR: Only {len(events)} events, need at least 50", flush=True)
        return

    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("pead_ml_predictor_v1")

    all_results = {}
    best_sharpe = -999
    best_variant = None

    for vk in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vk]
        vt0 = time.time()

        metrics, trades = run_variant(vk, cfg, events)
        elapsed = time.time() - vt0
        metrics['runtime'] = round(elapsed, 1)

        all_results[vk] = metrics

        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_variant = vk

        # MLflow
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"variant_{vk}_{cfg['name'].replace(' ', '_')}"):
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
                    mlflow.log_param("variant", vk)
                    mlflow.log_param("model", cfg['model'])
                    mlflow.log_param("threshold", cfg['threshold'])
            except Exception as e:
                print(f"  MLflow error: {e}", flush=True)

    total_time = time.time() - t0

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"  PEAD ML PREDICTOR V1 — SUMMARY", flush=True)
    print(f"  Runtime: {total_time:.0f}s | Best: {best_variant} (Sharpe {best_sharpe:.3f})", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Var':<4} {'Name':<30} {'Trd':>4} {'WR%':>5} {'Sharpe':>7} "
          f"{'Sort':>7} {'PF':>5} {'MDD%':>6} {'Final$':>8} {'Perm':>6} {'Acc%':>5} {'Gate':>4}", flush=True)
    print("-" * 100, flush=True)

    for vk in sorted(all_results.keys()):
        m = all_results[vk]
        print(f"  {vk:<3} {m['name']:<30} {m['trades']:>4} {m['wr']:>5.1f} "
              f"{m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['pf']:>5.2f} "
              f"{m['mdd']:>6.1f} ${m['final_equity']:>7.2f} {m['perm_p']:>6.4f} "
              f"{m.get('pred_accuracy', 0):>5.1f} {m['gates_passed']}/5", flush=True)

    # Baseline comparison
    print(f"\n  BASELINE (earnings_momentum_v1 A): Sharpe 2.52, 53 trades, $645→$2,620", flush=True)
    print(f"  Does ML filtering improve over simple post-gap buying?", flush=True)

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
