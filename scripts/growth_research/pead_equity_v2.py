#!/usr/bin/env python3
"""
Post-Earnings Announcement Drift (PEAD) — EQUITY V2
====================================================
Addresses v1 weaknesses:
  1. CONCENTRATION FIX: max 3 trades per ticker, max 25% equity per ticker
  2. EQUITY TRADES: buy stock instead of options — no theta, no BS model needed
  3. WIDER UNIVERSE: 80 tickers (added large-cap diversifiers)
  4. TIGHTER RISK: position sizing by Kelly fraction, not fixed $200
  5. LONGER HOLD: test 3d, 5d, 10d holds (v1 only did 3d)
  6. DIRECTION BALANCE: ensure both long/short work independently

V1 FINDINGS (to build on):
  - Best D (LGBM+Mom): Sharpe 1.513, WR 52.4%, PF 3.64, $645→$2,230
  - Adversarial: 4/8 pass — concentration & top-2 removal = main failures
  - Call-biased: $1,216 calls vs $51 puts
  - Winners 3x losers on average

6 VARIANTS:
  A. Equity 3-day hold, LGBM+Mom, max 3/ticker
  B. Equity 5-day hold, LGBM+Mom, max 3/ticker
  C. Equity 10-day hold, LGBM, max 3/ticker
  D. Equity 3-day + Kelly sizing + diversification cap
  E. Equity 3-day, wider universe (80 tickers)
  F. Options v1 replay + diversification cap (compare apples-to-apples)

UNIVERSE: 80 growth+large-cap stocks
PERIOD: 2022-01-01 to 2026-07-25
ACCOUNT: $645
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

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'pead_equity_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

# Original 52 growth stocks
GROWTH_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

# 28 additional large-cap diversifiers (high earnings-gap frequency)
DIVERSIFIER_UNIVERSE = [
    'CRM', 'ADBE', 'INTC', 'QCOM', 'AVGO', 'NOW', 'ORCL', 'IBM',
    'V', 'MA', 'SQ', 'INTU', 'ISRG', 'DHR', 'ABT', 'TMO',
    'HD', 'LOW', 'TGT', 'COST', 'WMT', 'SBUX', 'MCD', 'NKE',
    'DIS', 'NFLX', 'BA', 'CAT',
]

STARTING_CAPITAL = 645.0
COMMISSION_EQUITY = 0.0  # Robinhood: $0 equity commission
MAX_POSITION_PCT = 0.30  # max 30% of equity per trade
MAX_TICKER_PCT = 0.25  # max 25% cumulative PnL from one ticker (diversification)
MAX_TRADES_PER_TICKER = 3  # v1 had unlimited → concentration
RISK_FREE_RATE = 0.05
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'
N_PERMUTATIONS = 200
DRIFT_THRESHOLD = 0.03  # 3% drift = positive target
MIN_GAP_PCT = 0.05  # 5% minimum gap

# BS for variant F (options comparison)
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
# DATA LOADING
# ============================================================

def load_data(universe):
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'pead_v2_prices_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'pead_v2_earnings_cache.json')

    unique_tickers = sorted(set(universe))

    # Load cached price data
    if os.path.exists(cache_path):
        prices_df = pd.read_parquet(cache_path)
        cached_tickers = set(prices_df.index.get_level_values(0).unique())
        missing = [t for t in unique_tickers if t not in cached_tickers and t not in ['^VIX']]
        if missing:
            print(f"Cache missing {len(missing)} tickers, re-downloading all...", flush=True)
        else:
            print(f"Loaded cached prices: {len(prices_df)} rows, {len(cached_tickers)} tickers", flush=True)
            # Load earnings
            if os.path.exists(earnings_cache):
                with open(earnings_cache) as f:
                    earnings_dates = json.load(f)
                print(f"Loaded cached earnings: {len(earnings_dates)} tickers", flush=True)
                return prices_df, earnings_dates

    # Download fresh
    print(f"Downloading {len(unique_tickers)} tickers + SPY + VIX...", flush=True)
    all_tickers = unique_tickers + ['SPY', '^VIX']
    frames = []
    for t in all_tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(df) < 50: continue
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df['ticker'] = t
            df.index.name = 'date'
            frames.append(df)
        except Exception as e:
            print(f"  SKIP {t}: {e}", flush=True)

    prices_df = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices_df.to_parquet(cache_path)
    print(f"Saved {len(prices_df)} price rows", flush=True)

    # Fetch earnings dates
    print("Fetching earnings dates...", flush=True)
    earnings_dates = {}
    for t in unique_tickers:
        try:
            stock = yf.Ticker(t)
            dates = stock.get_earnings_dates(limit=30)
            if dates is not None and len(dates) > 0:
                earnings_dates[t] = [str(d.date()) for d in dates.index]
        except Exception:
            pass

    with open(earnings_cache, 'w') as f:
        json.dump(earnings_dates, f)
    print(f"Cached {len(earnings_dates)} tickers' earnings dates", flush=True)

    return prices_df, earnings_dates


# ============================================================
# FEATURE ENGINEERING (same as v1 + new features)
# ============================================================

FEATURE_COLS = [
    'abs_gap', 'gap_direction', 'mom_5d', 'mom_10d', 'mom_21d',
    'vol_21d', 'rel_str_21d', 'vix', 'vol_ratio', 'rsi',
    'dist_52w_high', 'prev_gap', 'ticker_hash',
    # New v2 features
    'volume_spike', 'gap_vs_atr', 'sector_momentum',
]


def build_event_features(prices_df, earnings_dates, universe):
    """Build feature matrix for all earnings events."""
    events = []
    available_tickers = set(prices_df.index.get_level_values(0).unique())

    # Preload SPY and VIX
    try:
        spy = prices_df.loc['SPY', 'close']
    except:
        spy = None
    try:
        vix = prices_df.loc['^VIX', 'close']
    except:
        vix = None

    for ticker in universe:
        if ticker not in earnings_dates or ticker not in available_tickers:
            continue

        try:
            ticker_prices = prices_df.loc[ticker].sort_index()
        except KeyError:
            continue

        if len(ticker_prices) < 30:
            continue

        trading_days = ticker_prices.index.sort_values()
        dates = earnings_dates[ticker]

        for earn_date_str in dates:
            earn_date = pd.Timestamp(earn_date_str)

            # Find post-earnings day
            post_mask = trading_days >= earn_date
            if not post_mask.any(): continue
            post_days = trading_days[post_mask]
            if len(post_days) < 11: continue  # need up to 10d hold

            # Find pre-earnings day
            pre_mask = trading_days < earn_date
            if not pre_mask.any(): continue
            pre_days = trading_days[pre_mask]
            if len(pre_days) < 30: continue

            close_before = ticker_prices.loc[pre_days[-1], 'close']
            open_after = ticker_prices.loc[post_days[0], 'open']

            if close_before <= 0 or open_after <= 0: continue

            gap_pct = (open_after / close_before) - 1.0
            if abs(gap_pct) < MIN_GAP_PCT:
                continue

            # Features
            feats = {}
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
            feats['rel_str_21d'] = 0
            if spy is not None and len(pre_days) >= 22:
                try:
                    spy_close = spy.loc[spy.index <= pre_days[-1]].iloc[-21:]
                    stock_close = ticker_prices.loc[pre_days[-21:], 'close']
                    feats['rel_str_21d'] = float(
                        (stock_close.iloc[-1] / stock_close.iloc[0] - 1) -
                        (spy_close.iloc[-1] / spy_close.iloc[0] - 1))
                except:
                    pass

            # VIX level
            feats['vix'] = 20
            if vix is not None:
                vix_mask = vix.index <= pre_days[-1]
                if vix_mask.any():
                    feats['vix'] = float(vix[vix_mask].iloc[-1])

            # Volume ratio
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
                feats['rsi'] = float(100 - 100 / (1 + gains/losses)) if losses > 0 else 100
            else:
                feats['rsi'] = 50

            # Distance from 52w high
            if len(pre_days) >= 252:
                h52 = ticker_prices.loc[pre_days[-252:], 'close'].max()
                feats['dist_52w_high'] = float(close_before / h52 - 1)
            else:
                feats['dist_52w_high'] = 0

            # Previous earnings gap
            ticker_earn = [d for d in dates if pd.Timestamp(d) < earn_date]
            feats['prev_gap'] = 0
            if ticker_earn:
                prev_earn = pd.Timestamp(ticker_earn[-1])
                prev_post = trading_days[trading_days >= prev_earn]
                prev_pre = trading_days[trading_days < prev_earn]
                if len(prev_post) > 0 and len(prev_pre) > 0:
                    prev_close = ticker_prices.loc[prev_pre[-1], 'close']
                    prev_open = ticker_prices.loc[prev_post[0], 'open']
                    if prev_close > 0:
                        feats['prev_gap'] = float(prev_open / prev_close - 1)

            feats['ticker_hash'] = hash(ticker) % 100

            # NEW v2 features
            # Volume spike on earnings day
            if 'volume' in ticker_prices.columns and len(post_days) >= 1:
                earn_vol = ticker_prices.loc[post_days[0], 'volume']
                avg_vol = ticker_prices.loc[pre_days[-22:], 'volume'].mean() if len(pre_days) >= 22 else 1
                feats['volume_spike'] = float(earn_vol / max(avg_vol, 1))
            else:
                feats['volume_spike'] = 1.0

            # Gap vs ATR (how many ATRs is this gap)
            if len(pre_days) >= 15:
                highs = ticker_prices.loc[pre_days[-15:], 'high']
                lows = ticker_prices.loc[pre_days[-15:], 'low']
                closes_prev = ticker_prices.loc[pre_days[-16:-1], 'close'] if len(pre_days) >= 16 else highs
                tr = pd.DataFrame({
                    'hl': highs.values - lows.values,
                    'hc': abs(highs.values - closes_prev.values[:len(highs)]) if len(closes_prev) >= len(highs) else highs.values - lows.values,
                    'lc': abs(lows.values - closes_prev.values[:len(lows)]) if len(closes_prev) >= len(lows) else highs.values - lows.values,
                }).max(axis=1).mean()
                feats['gap_vs_atr'] = float(abs(gap_pct) * close_before / max(tr, 0.01))
            else:
                feats['gap_vs_atr'] = 1.0

            # Sector momentum (SPY momentum as proxy)
            feats['sector_momentum'] = 0
            if spy is not None and len(pre_days) >= 22:
                try:
                    spy_close = spy.loc[spy.index <= pre_days[-1]].iloc[-21:]
                    feats['sector_momentum'] = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)
                except:
                    pass

            # TARGET: did price continue drifting in gap direction?
            # Compute for multiple hold periods
            post_close = [float(ticker_prices.loc[d, 'close']) for d in post_days[:11]]

            for hold_days in [3, 5, 10]:
                if len(post_close) >= hold_days + 1:
                    entry_price = open_after
                    if gap_pct > 0:
                        max_move = max(c / entry_price - 1 for c in post_close[1:hold_days+1])
                    else:
                        max_move = max(1 - c / entry_price for c in post_close[1:hold_days+1])
                    feats[f'target_{hold_days}d'] = 1 if max_move >= DRIFT_THRESHOLD else 0

            # Store metadata
            feats['ticker'] = ticker
            feats['earn_date'] = earn_date_str
            feats['close_before'] = float(close_before)
            feats['open_after'] = float(open_after)
            feats['post_closes'] = post_close

            events.append(feats)

    print(f"\nBuilt {len(events)} qualifying events (|gap| >= {MIN_GAP_PCT*100}%)", flush=True)
    if events:
        for hd in [3, 5, 10]:
            targets = [e.get(f'target_{hd}d', 0) for e in events]
            print(f"  Drift success rate ({hd}d): {sum(targets)/len(targets)*100:.1f}% "
                  f"({sum(targets)}/{len(targets)})", flush=True)

    return events


# ============================================================
# ML MODELS
# ============================================================

def train_lgbm(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# WALK-FORWARD SIMULATION
# ============================================================

VARIANT_CONFIGS = {
    'A': {
        'name': 'Equity 3d Hold + LGBM+Mom + Max3/Ticker',
        'trade_type': 'equity',
        'hold_days': 3,
        'model': 'lgbm',
        'threshold': 0.60,
        'momentum_filter': True,
        'max_trades_per_ticker': 3,
        'tp_pct': 0.05,  # 5% take profit
        'sl_pct': -0.03,  # 3% stop loss
        'position_pct': 0.25,  # 25% of equity per trade
    },
    'B': {
        'name': 'Equity 5d Hold + LGBM+Mom',
        'trade_type': 'equity',
        'hold_days': 5,
        'model': 'lgbm',
        'threshold': 0.60,
        'momentum_filter': True,
        'max_trades_per_ticker': 3,
        'tp_pct': 0.08,
        'sl_pct': -0.05,
        'position_pct': 0.25,
    },
    'C': {
        'name': 'Equity 10d Hold + LGBM (Longer Drift)',
        'trade_type': 'equity',
        'hold_days': 10,
        'model': 'lgbm',
        'threshold': 0.60,
        'momentum_filter': False,
        'max_trades_per_ticker': 3,
        'tp_pct': 0.12,
        'sl_pct': -0.06,
        'position_pct': 0.20,
    },
    'D': {
        'name': 'Equity 3d + Kelly Sizing + Diversity Cap',
        'trade_type': 'equity',
        'hold_days': 3,
        'model': 'lgbm',
        'threshold': 0.60,
        'momentum_filter': True,
        'max_trades_per_ticker': 3,
        'kelly_sizing': True,
        'tp_pct': 0.05,
        'sl_pct': -0.03,
        'position_pct': 0.25,
        'max_ticker_pnl_pct': 0.25,  # max 25% of total PnL from any ticker
    },
    'E': {
        'name': 'Equity 3d + Wide Universe (80 tickers)',
        'trade_type': 'equity',
        'hold_days': 3,
        'model': 'lgbm',
        'threshold': 0.60,
        'momentum_filter': True,
        'max_trades_per_ticker': 3,
        'wide_universe': True,
        'tp_pct': 0.05,
        'sl_pct': -0.03,
        'position_pct': 0.25,
    },
    'F': {
        'name': 'Options v1 Replay + Diversity Cap (Comparison)',
        'trade_type': 'options',
        'hold_days': 3,
        'model': 'lgbm',
        'threshold': 0.60,
        'momentum_filter': True,
        'max_trades_per_ticker': 3,
        'tp_pct': 0.30,  # 30% option TP like v1
        'sl_pct': -0.25,  # 25% option SL like v1
        'position_pct': 0.30,
        'max_ticker_pnl_pct': 0.25,
    },
}


def run_variant(variant_key, cfg, events, all_events_wide=None):
    """Walk-forward backtest with ML prediction + equity/options trading."""
    print(f"\n{'='*60}", flush=True)
    print(f"  VARIANT {variant_key}: {cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    use_events = all_events_wide if cfg.get('wide_universe') and all_events_wide else events
    hold_days = cfg['hold_days']
    target_col = f'target_{hold_days}d'

    # Filter events that have the right target
    valid_events = [e for e in use_events if target_col in e]
    events_sorted = sorted(valid_events, key=lambda e: e['earn_date'])

    min_train = 30
    equity = STARTING_CAPITAL
    peak_equity = equity
    max_dd = 0
    trades = []
    ticker_trade_count = defaultdict(int)
    ticker_pnl = defaultdict(float)
    daily_equity = {}

    for i in range(min_train, len(events_sorted)):
        event = events_sorted[i]

        # Max trades per ticker check
        if ticker_trade_count[event['ticker']] >= cfg.get('max_trades_per_ticker', 999):
            continue

        # Diversity cap: skip if this ticker already has too much PnL share
        if cfg.get('max_ticker_pnl_pct') and trades:
            total_pnl = sum(t['pnl'] for t in trades)
            if total_pnl > 0 and ticker_pnl[event['ticker']] / total_pnl > cfg['max_ticker_pnl_pct']:
                continue

        # Train data
        train_events = events_sorted[:i]
        X_train = pd.DataFrame(train_events)[FEATURE_COLS]
        y_train = np.array([e.get(target_col, 0) for e in train_events])

        X_test = pd.DataFrame([event])[FEATURE_COLS]

        try:
            model = train_lgbm(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1][0]
        except Exception:
            continue

        if prob < cfg['threshold']:
            continue

        # Momentum filter
        if cfg.get('momentum_filter'):
            if event['gap_direction'] == 1 and event.get('mom_5d', 0) < 0:
                continue
            if event['gap_direction'] == -1 and event.get('mom_5d', 0) > 0:
                continue

        gap_pct = event['gap_pct']
        entry_price = event['open_after']

        if cfg['trade_type'] == 'equity':
            # EQUITY TRADE
            position_size = equity * cfg['position_pct']

            # Kelly sizing
            if cfg.get('kelly_sizing') and len(trades) >= 10:
                wins = [t for t in trades[-20:] if t['pnl'] > 0]
                losses = [t for t in trades[-20:] if t['pnl'] <= 0]
                if wins and losses:
                    wr = len(wins) / (len(wins) + len(losses))
                    avg_win = np.mean([t['pnl'] for t in wins])
                    avg_loss = abs(np.mean([t['pnl'] for t in losses]))
                    if avg_loss > 0:
                        kelly = wr - (1 - wr) / (avg_win / avg_loss)
                        kelly = max(0.05, min(kelly, 0.25))  # bound kelly
                        position_size = equity * kelly

            shares = int(position_size / entry_price)
            if shares < 1:
                continue

            actual_size = shares * entry_price
            if actual_size > equity:
                continue

            # Simulate holding period
            post_closes = event['post_closes']
            exit_price = entry_price
            exit_reason = 'time_stop'

            for day in range(1, min(len(post_closes), hold_days + 1)):
                spot = post_closes[day]
                if gap_pct > 0:
                    # Long trade
                    pct_change = (spot - entry_price) / entry_price
                else:
                    # Short trade (simulate via inverse)
                    pct_change = (entry_price - spot) / entry_price

                if pct_change >= cfg['tp_pct']:
                    exit_price = spot
                    exit_reason = 'take_profit'
                    break
                elif pct_change <= cfg['sl_pct']:
                    exit_price = spot
                    exit_reason = 'stop_loss'
                    break
                exit_price = spot

            # Calculate PnL
            if gap_pct > 0:
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares

            # No commission on Robinhood equity
            equity += pnl

        else:
            # OPTIONS TRADE (variant F — comparison)
            opt_type = 'call' if gap_pct > 0 else 'put'
            strike = round(entry_price)
            post_iv = event.get('vol_21d', 0.3) * 0.8
            post_iv = max(post_iv, 0.15)
            T = 14 / 252.0
            entry_premium = option_price(entry_price, strike, T, RISK_FREE_RATE, post_iv, opt_type)
            if entry_premium < 0.10:
                continue

            contract_cost = entry_premium * 100
            max_spend = min(equity * cfg['position_pct'], equity - 10)
            if contract_cost > max_spend or contract_cost + 1.30 > equity:
                continue

            post_closes = event['post_closes']
            exit_premium = entry_premium
            exit_reason = 'time_stop'

            for day in range(1, min(len(post_closes), hold_days + 1)):
                spot = post_closes[day]
                remaining = max(14 - day, 0) / 252.0
                iv_adj = post_iv * (1 + 0.02 * day)
                current = option_price(spot, strike, remaining, RISK_FREE_RATE, iv_adj, opt_type)
                pct = (current - entry_premium) / entry_premium
                if pct >= cfg['tp_pct']:
                    exit_premium = current
                    exit_reason = 'take_profit'
                    break
                elif pct <= cfg['sl_pct']:
                    exit_premium = current
                    exit_reason = 'stop_loss'
                    break
                exit_premium = current

            pnl = (exit_premium - entry_premium) * 100 - 1.30
            equity += pnl

        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        if dd < max_dd:
            max_dd = dd

        ticker_trade_count[event['ticker']] += 1
        ticker_pnl[event['ticker']] += pnl

        trades.append({
            'ticker': event['ticker'],
            'earn_date': event['earn_date'],
            'direction': 'LONG' if gap_pct > 0 else 'SHORT',
            'prob': round(prob, 3),
            'gap_pct': round(gap_pct * 100, 1),
            'pnl': round(pnl, 2),
            'exit_reason': exit_reason,
            'target': event.get(target_col, 0),
            'equity_after': round(equity, 2),
        })

        # Track daily equity
        daily_equity[event['earn_date']] = equity

        if len(trades) <= 3:
            print(f"  Trade {len(trades)}: {event['ticker']} "
                  f"{'LONG' if gap_pct > 0 else 'SHORT'} "
                  f"gap={gap_pct*100:.1f}% prob={prob:.2f} "
                  f"pnl=${pnl:.2f} eq=${equity:.2f} [{exit_reason}]", flush=True)

    # ---- Results ----
    if not trades:
        print(f"  NO TRADES generated", flush=True)
        return {
            'trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
            'final_equity': STARTING_CAPITAL, 'mdd': 0, 'avg_pnl': 0,
            'perm_p': 1.0, 'pred_accuracy': 0,
            'gates_passed': 0, 'gates': {}, 'variant': variant_key, 'name': cfg['name'],
            'concentration': 0, 'direction_balance': 0,
        }, []

    pnls = np.array([t['pnl'] for t in trades])
    n_trades = len(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252 / max(1, n_trades))) if np.std(pnls) > 0 else 0
    downside = pnls[pnls < 0]
    sortino = float(np.mean(pnls) / np.std(downside) * np.sqrt(252 / max(1, n_trades))) if len(downside) > 0 and np.std(downside) > 0 else 0
    pf = float(sum(wins) / abs(sum(losses))) if len(losses) > 0 and sum(losses) != 0 else float('inf')
    wr = float(len(wins) / n_trades * 100) if n_trades > 0 else 0
    avg_pnl = float(np.mean(pnls))

    # Prediction accuracy
    correct = sum(1 for t in trades if (t['pnl'] > 0 and t['target'] == 1) or (t['pnl'] <= 0 and t['target'] == 0))
    pred_accuracy = correct / n_trades * 100

    # Concentration analysis
    ticker_abs_pnl = {t: abs(p) for t, p in ticker_pnl.items()}
    total_abs_pnl = sum(ticker_abs_pnl.values())
    top2_pnl = sum(sorted(ticker_abs_pnl.values(), reverse=True)[:2])
    concentration = top2_pnl / total_abs_pnl if total_abs_pnl > 0 else 0

    # Direction balance
    long_pnl = sum(t['pnl'] for t in trades if t['direction'] == 'LONG')
    short_pnl = sum(t['pnl'] for t in trades if t['direction'] == 'SHORT')
    total_pnl = abs(long_pnl) + abs(short_pnl)
    dir_balance = abs(long_pnl - short_pnl) / total_pnl if total_pnl > 0 else 1.0

    # Permutation test
    observed_sharpe = sharpe
    perm_count = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(pnls)
        perm_sharpe = float(np.mean(shuffled) / np.std(shuffled) * np.sqrt(252 / max(1, n_trades))) if np.std(shuffled) > 0 else 0
        if perm_sharpe >= observed_sharpe:
            perm_count += 1
    perm_p = perm_count / N_PERMUTATIONS

    # Monte Carlo CI
    mc_final = []
    for _ in range(1000):
        mc_pnls = np.random.choice(pnls, size=n_trades, replace=True)
        mc_final.append(STARTING_CAPITAL + np.sum(mc_pnls))
    mc_5th = np.percentile(mc_final, 5)

    # Per-year breakdown (regime check)
    year_sharpes = {}
    for year in sorted(set(t['earn_date'][:4] for t in trades)):
        yr_pnls = [t['pnl'] for t in trades if t['earn_date'].startswith(year)]
        if len(yr_pnls) >= 2 and np.std(yr_pnls) > 0:
            year_sharpes[year] = float(np.mean(yr_pnls) / np.std(yr_pnls) * np.sqrt(252 / max(1, len(yr_pnls))))
        else:
            year_sharpes[year] = 0

    green_sharpes = [s for y, s in year_sharpes.items() if y in ['2023', '2024', '2025', '2026']]
    red_sharpes = [s for y, s in year_sharpes.items() if y in ['2022']]
    max_green = max(abs(s) for s in green_sharpes) if green_sharpes else 0.01
    max_red = max(abs(s) for s in red_sharpes) if red_sharpes else 0.01
    regime_gap = abs(np.mean(green_sharpes) - np.mean(red_sharpes)) / max(max_green, max_red) if green_sharpes and red_sharpes else 1.0

    # 5-Gate Validation
    gates = {
        'sharpe_gt_1': sharpe >= 1.0,
        'perm_p_lt_005': perm_p < 0.05,
        'wr_gt_40': wr >= 40,
        'regime_balance': regime_gap < 0.50,
        'mc_ci_positive': mc_5th > STARTING_CAPITAL,
    }
    gates_passed = sum(gates.values())

    # Print results
    print(f"\n  Trades: {n_trades} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | "
          f"PF: {pf:.3f} | WR: {wr:.1f}%", flush=True)
    print(f"  MDD: {max_dd*100:.2f}% | Final: ${equity:.2f} | "
          f"Return: {(equity/STARTING_CAPITAL - 1)*100:.1f}%", flush=True)
    print(f"  Avg PnL: ${avg_pnl:.2f} | Perm p: {perm_p:.3f} | Pred acc: {pred_accuracy:.1f}%", flush=True)
    print(f"  Top-2 ticker concentration: {concentration*100:.1f}% | Dir balance gap: {dir_balance:.2f}", flush=True)
    print(f"  Per-year: {year_sharpes}", flush=True)

    # Ticker breakdown
    print(f"  Ticker PnL breakdown:", flush=True)
    for t, p in sorted(ticker_pnl.items(), key=lambda x: -abs(x[1]))[:8]:
        print(f"    {t}: ${p:.2f} ({ticker_trade_count[t]} trades)", flush=True)

    print(f"\n  5-Gate Validation: {gates_passed}/5 {'PASS' if gates_passed >= 4 else 'FAIL'}", flush=True)
    for g, v in gates.items():
        val = {
            'sharpe_gt_1': sharpe, 'perm_p_lt_005': perm_p, 'wr_gt_40': wr,
            'regime_balance': regime_gap, 'mc_ci_positive': mc_5th,
        }.get(g, '?')
        thr = {
            'sharpe_gt_1': 1.0, 'perm_p_lt_005': 0.05, 'wr_gt_40': 40,
            'regime_balance': 0.50, 'mc_ci_positive': 0,
        }.get(g, '?')
        print(f"    {g}: {'PASS' if v else 'FAIL'} (value={val}, threshold={thr})", flush=True)

    result = {
        'trades': n_trades, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'pf': round(pf, 3), 'wr': round(wr, 1), 'mdd': round(max_dd * 100, 2),
        'final_equity': round(equity, 2), 'avg_pnl': round(avg_pnl, 2),
        'perm_p': round(perm_p, 3), 'pred_accuracy': round(pred_accuracy, 1),
        'gates_passed': gates_passed, 'gates': gates,
        'variant': variant_key, 'name': cfg['name'],
        'concentration': round(concentration * 100, 1),
        'direction_balance': round(dir_balance, 2),
        'year_sharpes': year_sharpes,
        'ticker_breakdown': {t: round(p, 2) for t, p in sorted(ticker_pnl.items(), key=lambda x: -abs(x[1]))[:10]},
    }

    return result, trades


# ============================================================
# RANDOM BASELINE
# ============================================================

def random_baseline(events, hold_days=3, n_iter=100):
    """Random trading baseline for comparison."""
    target_col = f'target_{hold_days}d'
    valid = [e for e in events if target_col in e]
    events_sorted = sorted(valid, key=lambda e: e['earn_date'])

    sharpes = []
    for _ in range(n_iter):
        equity = STARTING_CAPITAL
        peak = equity
        max_dd = 0
        pnls = []

        # Random subset of events, similar trade count
        n_trades = min(25, len(events_sorted) - 30)
        indices = sorted(np.random.choice(range(30, len(events_sorted)), size=n_trades, replace=False))

        for idx in indices:
            e = events_sorted[idx]
            entry = e['open_after']
            gap = e['gap_pct']
            shares = max(1, int(equity * 0.25 / entry))
            if shares * entry > equity: continue

            pc = e['post_closes']
            exit_p = pc[min(hold_days, len(pc)-1)]
            if gap > 0:
                pnl = (exit_p - entry) * shares
            else:
                pnl = (entry - exit_p) * shares
            equity += pnl
            pnls.append(pnl)

        if pnls and np.std(pnls) > 0:
            s = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / max(1, len(pnls)))
            sharpes.append(s)

    if sharpes:
        return {
            'mean_sharpe': round(float(np.mean(sharpes)), 3),
            'median_sharpe': round(float(np.median(sharpes)), 3),
            'p90_sharpe': round(float(np.percentile(sharpes, 90)), 3),
        }
    return {'mean_sharpe': 0, 'median_sharpe': 0, 'p90_sharpe': 0}


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Running on: {LVL3_ROOT}", flush=True)

    # MLflow setup
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('pead_equity_v2')
            print("MLflow OK", flush=True)
        except Exception as e:
            print(f"MLflow error: {e}", flush=True)

    print("=" * 70, flush=True)
    print("  PEAD EQUITY V2 — Concentration Fix + Equity Trades", flush=True)
    print(f"  Hypothesis: Equity trades remove theta risk,", flush=True)
    print(f"             diversification cap fixes concentration", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print("=" * 70, flush=True)

    # Load data for both universes
    print("\n--- Loading GROWTH universe (52 tickers) ---", flush=True)
    prices_growth, earnings_growth = load_data(GROWTH_UNIVERSE)

    print("\n--- Loading WIDE universe (80 tickers) ---", flush=True)
    wide_universe = sorted(set(GROWTH_UNIVERSE + DIVERSIFIER_UNIVERSE))
    prices_wide, earnings_wide = load_data(wide_universe)

    # Build events
    print("\n--- Building events (growth universe) ---", flush=True)
    events_growth = build_event_features(prices_growth, earnings_growth, GROWTH_UNIVERSE)

    print("\n--- Building events (wide universe) ---", flush=True)
    events_wide = build_event_features(prices_wide, earnings_wide, wide_universe)

    # Random baseline
    print("\n--- Random Baseline ---", flush=True)
    baseline = random_baseline(events_growth, hold_days=3, n_iter=100)
    print(f"  Random baseline (3d hold): mean Sharpe={baseline['mean_sharpe']}, "
          f"median={baseline['median_sharpe']}, p90={baseline['p90_sharpe']}", flush=True)

    # Run all variants
    all_results = []
    for vkey in sorted(VARIANT_CONFIGS.keys()):
        cfg = VARIANT_CONFIGS[vkey]
        t0 = datetime.now()
        result, trades = run_variant(vkey, cfg, events_growth, events_wide)
        elapsed = (datetime.now() - t0).total_seconds()
        result['runtime_s'] = round(elapsed, 1)

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"variant_{vkey}_{cfg['name'][:30]}"):
                    mlflow.log_params({
                        'variant': vkey,
                        'trade_type': cfg['trade_type'],
                        'hold_days': cfg['hold_days'],
                        'threshold': cfg['threshold'],
                        'momentum_filter': cfg.get('momentum_filter', False),
                        'max_trades_per_ticker': cfg.get('max_trades_per_ticker', 999),
                    })
                    mlflow.log_metrics({
                        'sharpe': result['sharpe'],
                        'sortino': result['sortino'],
                        'pf': min(result['pf'], 999),
                        'wr': result['wr'],
                        'mdd': result['mdd'],
                        'final_equity': result['final_equity'],
                        'n_trades': result['trades'],
                        'perm_p': result['perm_p'],
                        'gates_passed': result['gates_passed'],
                        'concentration_pct': result.get('concentration', 0),
                    })
            except Exception as e:
                print(f"  MLflow log error: {e}", flush=True)

        all_results.append(result)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("  SUMMARY — PEAD EQUITY V2", flush=True)
    print("=" * 70, flush=True)
    print(f"  Random baseline: Sharpe {baseline['mean_sharpe']} (mean), "
          f"{baseline['p90_sharpe']} (p90)", flush=True)
    print(f"\n  {'Var':<4} {'Name':<40} {'Trades':<7} {'Sharpe':<8} {'Sortino':<9} "
          f"{'PF':<7} {'WR%':<6} {'MDD%':<7} {'Final$':<9} {'Conc%':<7} {'Gates':<6}", flush=True)
    print(f"  {'-'*4} {'-'*40} {'-'*7} {'-'*8} {'-'*9} {'-'*7} {'-'*6} {'-'*7} {'-'*9} {'-'*7} {'-'*6}", flush=True)

    for r in all_results:
        v = r['variant']
        print(f"  {v:<4} {r['name'][:40]:<40} {r['trades']:<7} {r['sharpe']:<8} {r['sortino']:<9} "
              f"{r['pf']:<7.2f} {r['wr']:<6.1f} {r['mdd']:<7.1f} ${r['final_equity']:<8.0f} "
              f"{r.get('concentration', 0):<7.1f} {r['gates_passed']}/5", flush=True)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'baseline': baseline,
            'variants': all_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}", flush=True)

    # Best variant
    best = max(all_results, key=lambda r: r['gates_passed'] * 10 + r['sharpe'])
    print(f"\n  BEST: Variant {best['variant']} — {best['name']}", flush=True)
    print(f"    Sharpe {best['sharpe']}, {best['trades']} trades, "
          f"${STARTING_CAPITAL}→${best['final_equity']}, "
          f"Concentration {best.get('concentration', 0)}%", flush=True)

    if best['gates_passed'] >= 4:
        print(f"\n  ✅ VALIDATED — {best['gates_passed']}/5 gates passed", flush=True)
    else:
        print(f"\n  ❌ NOT VALIDATED — {best['gates_passed']}/5 gates passed", flush=True)


if __name__ == '__main__':
    main()
