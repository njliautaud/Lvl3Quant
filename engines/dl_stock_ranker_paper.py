#!/usr/bin/env python3
"""
DL Stock Ranker — PAPER ENGINE (LightGBM CPU variant)
======================================================
Cross-sectional stock ranking strategy. Ranks 50 large-cap stocks,
selects top 5, equal-weight portfolio, rebalances every 21 trading days.

Based on validated attention model backtest:
  Sharpe 2.37, CAGR 62%, WR 84%, MaxDD -13.8%
  Bear Sharpe 3.19 > Bull 1.96 (works better in bear markets)

This engine uses LightGBM (CPU-friendly, Sharpe 2.09 in backtest)
instead of the attention model which requires GPU on Neptune.

Walk-forward: 504-day training window, 21-day test window, sliding.
Rebalance: every 21 trading days at market close.

State: /home/jupiter/Lvl3Quant/state/dl_stock_ranker_paper_state.json
Logs:  /home/jupiter/Lvl3Quant/logs/dl_stock_ranker_paper.log
"""

import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_FILE = BASE / "state" / "dl_stock_ranker_paper_state.json"
LOG_FILE = BASE / "logs" / "dl_stock_ranker_paper.log"

STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("dl_stock_ranker")

# ── Strategy Parameters ───────────────────────────────────────────────────
TRAIN_WINDOW = 504          # trading days for training
REBAL_PERIOD = 21           # rebalance every 21 trading days
TOP_K = 5                   # select top 5 stocks
INITIAL_CAPITAL = 100_000
LOOKBACK_DAYS = 900         # calendar days to download (covers 504 trading days + buffer)

# Backtest reference
BACKTEST_SHARPE = 2.09      # LightGBM variant
BACKTEST_CAGR = 0.52        # approximate for LightGBM variant

# ── Stock Universe (50 large-caps from attention backtest) ─────────────────
UNIVERSE = [
    'AAPL', 'ABBV', 'ABT', 'ADBE', 'AMD', 'AMZN', 'AVGO', 'BAC', 'CAT',
    'CMCSA', 'COST', 'CRM', 'CSCO', 'CVX', 'DE', 'DIS', 'GOOGL', 'GS',
    'HD', 'INTC', 'JNJ', 'JPM', 'KO', 'LLY', 'LMT', 'MA', 'MCD', 'META',
    'MRK', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'PEP', 'PFE', 'PG',
    'QCOM', 'RTX', 'SHW', 'SO', 'T', 'TMO', 'TSLA', 'TXN', 'UNH', 'V',
    'VZ', 'WMT', 'XOM',
]

FEATURE_NAMES = [
    'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
    'vol_1m', 'vol_3m',
    'rs_rank',
    'rsi_14', 'rsi_5',
    'mom_accel',
    'bb_position',
    'vol_ratio',
    'obv_trend',
    'price_to_high_52w', 'price_to_low_52w',
    'avg_volume_ratio',
    'drawdown_20d',
    'skew_20d', 'kurt_20d',
    'beta_spy',
]


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download price and volume data for universe + SPY benchmark."""
    tickers = UNIVERSE + ['SPY']
    start = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    log.info(f"Downloading data for {len(tickers)} tickers from {start}")

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw['Close']
        volumes = raw['Volume']
        highs = raw['High']
        lows = raw['Low']
    else:
        closes = raw[['Close']].rename(columns={'Close': tickers[0]})
        volumes = raw[['Volume']].rename(columns={'Volume': tickers[0]})
        highs = raw[['High']].rename(columns={'High': tickers[0]})
        lows = raw[['Low']].rename(columns={'Low': tickers[0]})

    closes = closes.ffill().dropna(how='all')
    volumes = volumes.ffill().fillna(0)
    highs = highs.ffill().dropna(how='all')
    lows = lows.ffill().dropna(how='all')

    log.info(f"Data: {closes.shape[0]} days, {closes.index[0].date()} -> {closes.index[-1].date()}")
    return closes, volumes, highs, lows


# ── Feature Engineering ────────────────────────────────────────────────────
def compute_rsi(series, period):
    """Compute RSI for a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_features_for_day(closes, volumes, highs, lows, day_idx):
    """Compute 20 features for all stocks on a given day. Returns DataFrame."""
    rows = []
    rets = closes.pct_change()
    spy_rets = rets['SPY'] if 'SPY' in rets.columns else None

    for ticker in UNIVERSE:
        if ticker not in closes.columns:
            continue

        price = closes[ticker]
        vol = volumes[ticker] if ticker in volumes.columns else pd.Series(0, index=closes.index)
        ret = rets[ticker]
        i = day_idx

        if i < 252 or pd.isna(price.iloc[i]):
            continue

        feat = {'ticker': ticker}

        # Momentum at multiple horizons
        feat['ret_1m'] = price.iloc[i] / price.iloc[i - 21] - 1 if i >= 21 else np.nan
        feat['ret_3m'] = price.iloc[i] / price.iloc[i - 63] - 1 if i >= 63 else np.nan
        feat['ret_6m'] = price.iloc[i] / price.iloc[i - 126] - 1 if i >= 126 else np.nan
        feat['ret_12m'] = price.iloc[i] / price.iloc[i - 252] - 1 if i >= 252 else np.nan

        # Realized volatility
        feat['vol_1m'] = ret.iloc[max(0, i - 21):i].std() * np.sqrt(252)
        feat['vol_3m'] = ret.iloc[max(0, i - 63):i].std() * np.sqrt(252)

        # Relative strength rank (computed cross-sectionally later)
        feat['_ret_3m_raw'] = feat['ret_3m']

        # RSI
        rsi_series = compute_rsi(price, 14)
        feat['rsi_14'] = rsi_series.iloc[i] if i < len(rsi_series) else np.nan
        rsi5_series = compute_rsi(price, 5)
        feat['rsi_5'] = rsi5_series.iloc[i] if i < len(rsi5_series) else np.nan

        # Momentum acceleration
        feat['mom_accel'] = (feat['ret_3m'] or 0) - (feat['ret_6m'] or 0)

        # Bollinger band position
        sma20 = price.iloc[max(0, i - 20):i + 1].mean()
        std20 = price.iloc[max(0, i - 20):i + 1].std()
        if std20 > 0:
            feat['bb_position'] = (price.iloc[i] - sma20) / (2 * std20)
        else:
            feat['bb_position'] = 0.0

        # Volume ratio
        vol_20d = vol.iloc[max(0, i - 20):i].mean()
        vol_3m = vol.iloc[max(0, i - 63):i].mean()
        feat['vol_ratio'] = vol_20d / (vol_3m + 1e-10)

        # OBV trend (slope of OBV over 20 days, normalized)
        if i >= 20:
            obv_window = []
            obv_val = 0
            for j in range(i - 20, i + 1):
                if j > 0:
                    if ret.iloc[j] > 0:
                        obv_val += vol.iloc[j]
                    elif ret.iloc[j] < 0:
                        obv_val -= vol.iloc[j]
                obv_window.append(obv_val)
            obv_arr = np.array(obv_window, dtype=float)
            if len(obv_arr) > 1:
                x = np.arange(len(obv_arr))
                slope = np.polyfit(x, obv_arr, 1)[0]
                feat['obv_trend'] = slope / (np.abs(obv_arr).mean() + 1e-10)
            else:
                feat['obv_trend'] = 0.0
        else:
            feat['obv_trend'] = 0.0

        # Price relative to 52-week high/low
        high_52w = price.iloc[max(0, i - 252):i + 1].max()
        low_52w = price.iloc[max(0, i - 252):i + 1].min()
        feat['price_to_high_52w'] = price.iloc[i] / (high_52w + 1e-10) - 1
        feat['price_to_low_52w'] = price.iloc[i] / (low_52w + 1e-10) - 1

        # Average volume ratio (current vs 20d average)
        if vol.iloc[i] > 0 and vol_20d > 0:
            feat['avg_volume_ratio'] = vol.iloc[i] / vol_20d
        else:
            feat['avg_volume_ratio'] = 1.0

        # Max drawdown in last 20 days
        if i >= 20:
            window_prices = price.iloc[i - 20:i + 1]
            running_max = window_prices.cummax()
            dd = (window_prices / running_max - 1)
            feat['drawdown_20d'] = dd.min()
        else:
            feat['drawdown_20d'] = 0.0

        # Skewness and kurtosis of 20d returns
        if i >= 20:
            recent_rets = ret.iloc[i - 20:i]
            feat['skew_20d'] = recent_rets.skew()
            feat['kurt_20d'] = recent_rets.kurt()
        else:
            feat['skew_20d'] = 0.0
            feat['kurt_20d'] = 0.0

        # Rolling beta to SPY (60-day)
        if spy_rets is not None and i >= 60:
            stock_window = ret.iloc[i - 60:i]
            spy_window = spy_rets.iloc[i - 60:i]
            valid = stock_window.notna() & spy_window.notna()
            if valid.sum() > 10:
                cov = np.cov(stock_window[valid], spy_window[valid])
                feat['beta_spy'] = cov[0, 1] / (cov[1, 1] + 1e-10)
            else:
                feat['beta_spy'] = 1.0
        else:
            feat['beta_spy'] = 1.0

        rows.append(feat)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Cross-sectional relative strength rank
    if '_ret_3m_raw' in df.columns and len(df) > 0:
        df['rs_rank'] = df['_ret_3m_raw'].rank(pct=True)
        df.drop(columns=['_ret_3m_raw'], inplace=True)
    else:
        df['rs_rank'] = 0.5

    return df


def build_training_data(closes, volumes, highs, lows, end_idx, window=504, forward_days=21):
    """Build walk-forward training set.

    For each day in the window, compute features and label = forward 21-day return rank.
    The target is whether the stock ends up in the top quintile (top 10 of 50).
    """
    all_rows = []
    start_idx = max(252, end_idx - window)

    log.info(f"Building training data: days {start_idx} to {end_idx - forward_days}")

    for i in range(start_idx, end_idx - forward_days):
        day_df = compute_features_for_day(closes, volumes, highs, lows, i)
        if day_df.empty:
            continue

        # Forward return for labeling
        fwd_rets = {}
        for ticker in day_df['ticker'].values:
            if ticker in closes.columns:
                p_now = closes[ticker].iloc[i]
                p_fwd = closes[ticker].iloc[min(i + forward_days, len(closes) - 1)]
                if pd.notna(p_now) and pd.notna(p_fwd) and p_now > 0:
                    fwd_rets[ticker] = p_fwd / p_now - 1

        day_df['fwd_ret'] = day_df['ticker'].map(fwd_rets)
        day_df = day_df.dropna(subset=['fwd_ret'])

        if len(day_df) < 10:
            continue

        # Cross-sectional label: top quintile = 1, else 0
        threshold = day_df['fwd_ret'].quantile(0.80)
        day_df['target'] = (day_df['fwd_ret'] >= threshold).astype(int)
        day_df['train_day_idx'] = i

        all_rows.append(day_df)

    if not all_rows:
        return pd.DataFrame()

    train_df = pd.concat(all_rows, ignore_index=True)
    train_df.drop(columns=['fwd_ret'], inplace=True)
    return train_df


# ── Model Training ─────────────────────────────────────────────────────────
def train_model(train_df):
    """Train LightGBM ranker/classifier on the training data."""
    feat_cols = [c for c in FEATURE_NAMES if c in train_df.columns]
    X = train_df[feat_cols].fillna(0).values
    y = train_df['target'].values

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 200,
        'max_depth': 6,
        'verbose': -1,
        'random_state': 42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    train_auc = model.score(X, y)
    log.info(f"Training complete: {len(X)} samples, train accuracy: {train_auc:.3f}")
    log.info(f"Feature importances (top 10):")

    importance = dict(zip(feat_cols, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        log.info(f"  {feat}: {imp}")

    return model, feat_cols


# ── Ranking & Selection ────────────────────────────────────────────────────
def rank_stocks(model, feat_cols, closes, volumes, highs, lows, day_idx):
    """Rank all stocks by model prediction, return top K."""
    day_df = compute_features_for_day(closes, volumes, highs, lows, day_idx)
    if day_df.empty:
        log.warning("No features computed — cannot rank stocks")
        return [], day_df

    X = day_df[feat_cols].fillna(0).values
    probs = model.predict_proba(X)[:, 1]
    day_df['score'] = probs
    day_df = day_df.sort_values('score', ascending=False)

    top_k = day_df.head(TOP_K)['ticker'].tolist()
    log.info(f"Top {TOP_K} stocks: {top_k}")
    log.info(f"Full ranking (score):")
    for _, row in day_df.iterrows():
        marker = " <<<" if row['ticker'] in top_k else ""
        log.info(f"  {row['ticker']:6s} score={row['score']:.4f} "
                 f"ret_1m={row.get('ret_1m', 0):.3f} "
                 f"ret_3m={row.get('ret_3m', 0):.3f} "
                 f"rsi_14={row.get('rsi_14', 0):.1f}{marker}")

    return top_k, day_df


# ── State Management ───────────────────────────────────────────────────────
def load_state():
    """Load paper trading state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'portfolio_value': INITIAL_CAPITAL,
        'cash': INITIAL_CAPITAL,
        'positions': {},           # ticker -> {shares, entry_price, entry_date}
        'spy_shares': 0,
        'spy_entry_price': 0,
        'trade_history': [],       # list of trade records
        'rebalance_history': [],   # list of rebalance events
        'daily_values': [],        # list of {date, portfolio_value, spy_value}
        'last_rebal_date': None,
        'days_since_rebal': 0,
        'created': datetime.now().isoformat(),
    }


def save_state(state):
    """Save paper trading state to disk."""
    state['last_updated'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    log.info(f"State saved to {STATE_FILE}")


# ── Portfolio Tracking ─────────────────────────────────────────────────────
def compute_portfolio_value(state, closes):
    """Compute current portfolio value from positions + cash."""
    total = state['cash']
    last_idx = len(closes) - 1
    for ticker, pos in state['positions'].items():
        if ticker in closes.columns:
            current_price = closes[ticker].iloc[last_idx]
            if pd.notna(current_price):
                total += pos['shares'] * current_price
    return total


def compute_spy_value(state, closes):
    """Compute SPY benchmark value."""
    if state['spy_shares'] > 0 and 'SPY' in closes.columns:
        spy_price = closes['SPY'].iloc[-1]
        return state['spy_shares'] * spy_price
    return INITIAL_CAPITAL


def execute_rebalance(state, new_picks, closes):
    """Paper-execute a rebalance: sell old, buy new (equal weight)."""
    last_idx = len(closes) - 1
    today = closes.index[last_idx].strftime('%Y-%m-%d')
    log.info(f"=== REBALANCING on {today} ===")

    # Sell all current positions
    for ticker, pos in list(state['positions'].items()):
        if ticker in closes.columns:
            sell_price = closes[ticker].iloc[last_idx]
            if pd.notna(sell_price):
                proceeds = pos['shares'] * sell_price
                pnl = (sell_price - pos['entry_price']) * pos['shares']
                pnl_pct = (sell_price / pos['entry_price'] - 1) * 100

                state['cash'] += proceeds
                state['trade_history'].append({
                    'date': today,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': pos['shares'],
                    'price': float(sell_price),
                    'proceeds': float(proceeds),
                    'pnl': float(pnl),
                    'pnl_pct': float(pnl_pct),
                })
                log.info(f"  SELL {pos['shares']:.2f} {ticker} @ ${sell_price:.2f} "
                         f"(PnL: ${pnl:+.2f}, {pnl_pct:+.1f}%)")

    state['positions'] = {}

    # Buy new positions (equal weight)
    portfolio_value = state['cash']
    per_stock = portfolio_value / len(new_picks) if new_picks else 0

    for ticker in new_picks:
        if ticker in closes.columns:
            buy_price = closes[ticker].iloc[last_idx]
            if pd.notna(buy_price) and buy_price > 0:
                shares = per_stock / buy_price
                cost = shares * buy_price
                state['cash'] -= cost
                state['positions'][ticker] = {
                    'shares': float(shares),
                    'entry_price': float(buy_price),
                    'entry_date': today,
                }
                state['trade_history'].append({
                    'date': today,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': float(shares),
                    'price': float(buy_price),
                    'cost': float(cost),
                })
                log.info(f"  BUY  {shares:.2f} {ticker} @ ${buy_price:.2f} (${cost:.2f})")

    # Initialize SPY benchmark on first rebalance
    if state['spy_shares'] == 0 and 'SPY' in closes.columns:
        spy_price = closes['SPY'].iloc[last_idx]
        if pd.notna(spy_price) and spy_price > 0:
            state['spy_shares'] = float(INITIAL_CAPITAL / spy_price)
            state['spy_entry_price'] = float(spy_price)
            log.info(f"  SPY benchmark: {state['spy_shares']:.2f} shares @ ${spy_price:.2f}")

    state['last_rebal_date'] = today
    state['days_since_rebal'] = 0
    state['rebalance_history'].append({
        'date': today,
        'picks': new_picks,
        'portfolio_value': float(portfolio_value),
    })

    return state


def should_rebalance(state, closes):
    """Check if we need to rebalance (every 21 trading days, or first run)."""
    if state['last_rebal_date'] is None:
        return True

    last_rebal = pd.Timestamp(state['last_rebal_date'])
    trading_days_since = len(closes.index[closes.index > last_rebal])
    state['days_since_rebal'] = int(trading_days_since)

    if trading_days_since >= REBAL_PERIOD:
        log.info(f"{trading_days_since} trading days since last rebalance — time to rebalance")
        return True

    log.info(f"{trading_days_since}/{REBAL_PERIOD} trading days since last rebalance — holding")
    return False


# ── Performance Reporting ──────────────────────────────────────────────────
def report_performance(state, closes):
    """Log current performance metrics."""
    pv = compute_portfolio_value(state, closes)
    spy_v = compute_spy_value(state, closes)

    port_ret = (pv / INITIAL_CAPITAL - 1) * 100
    spy_ret = (spy_v / INITIAL_CAPITAL - 1) * 100
    excess = port_ret - spy_ret

    log.info(f"\n{'='*60}")
    log.info(f"PORTFOLIO STATUS")
    log.info(f"{'='*60}")
    log.info(f"Portfolio value:   ${pv:,.2f} ({port_ret:+.2f}%)")
    log.info(f"SPY benchmark:     ${spy_v:,.2f} ({spy_ret:+.2f}%)")
    log.info(f"Excess return:     {excess:+.2f}%")
    log.info(f"Cash:              ${state['cash']:,.2f}")
    log.info(f"Positions:         {len(state['positions'])}")
    log.info(f"Total trades:      {len(state['trade_history'])}")
    log.info(f"Rebalances:        {len(state['rebalance_history'])}")

    if state['positions']:
        log.info(f"\nCurrent Holdings:")
        for ticker, pos in state['positions'].items():
            if ticker in closes.columns:
                current = closes[ticker].iloc[-1]
                pnl_pct = (current / pos['entry_price'] - 1) * 100
                log.info(f"  {ticker:6s}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.2f} "
                         f"-> ${current:.2f} ({pnl_pct:+.1f}%)")

    # Win rate from trade history
    sells = [t for t in state['trade_history'] if t['action'] == 'SELL']
    if sells:
        wins = sum(1 for t in sells if t['pnl'] > 0)
        wr = wins / len(sells) * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in sells])
        log.info(f"\nTrade Stats:")
        log.info(f"  Completed trades: {len(sells)}")
        log.info(f"  Win rate:         {wr:.1f}%")
        log.info(f"  Avg return/trade: {avg_pnl:+.2f}%")

    # Track daily value
    today = closes.index[-1].strftime('%Y-%m-%d')
    state['daily_values'].append({
        'date': today,
        'portfolio_value': float(pv),
        'spy_value': float(spy_v),
    })

    # Compute Sharpe if enough history
    if len(state['daily_values']) > 5:
        vals = [d['portfolio_value'] for d in state['daily_values']]
        daily_rets = np.diff(vals) / np.array(vals[:-1])
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
            sortino_denom = np.std(daily_rets[daily_rets < 0]) if np.any(daily_rets < 0) else np.std(daily_rets)
            sortino = np.mean(daily_rets) / (sortino_denom + 1e-10) * np.sqrt(252)
            log.info(f"  Sharpe (live):    {sharpe:.2f} (backtest: {BACKTEST_SHARPE:.2f})")
            log.info(f"  Sortino (live):   {sortino:.2f}")

    state['portfolio_value'] = float(pv)
    return state


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    log.info(f"\n{'='*80}")
    log.info(f"DL STOCK RANKER PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    log.info(f"{'='*80}")

    try:
        # Load state
        state = load_state()
        is_first_run = state['last_rebal_date'] is None
        log.info(f"State loaded. Portfolio: ${state['portfolio_value']:,.2f}, "
                 f"Positions: {len(state['positions'])}, "
                 f"First run: {is_first_run}")

        # Download data
        closes, volumes, highs, lows = download_data()
        if closes.empty or len(closes) < 300:
            log.error(f"Insufficient data: {len(closes)} days (need 300+)")
            return

        # Check if today is a trading day
        today = pd.Timestamp.now().normalize()
        last_data_day = closes.index[-1].normalize()
        if (today - last_data_day).days > 3:
            log.warning(f"Latest data is from {last_data_day.date()}, may not be current")

        # Check if rebalance needed
        if should_rebalance(state, closes):
            log.info("Rebalance triggered — training model and ranking stocks")

            # Build training data
            last_idx = len(closes) - 1
            train_df = build_training_data(
                closes, volumes, highs, lows,
                end_idx=last_idx,
                window=TRAIN_WINDOW,
                forward_days=REBAL_PERIOD,
            )

            if train_df.empty or len(train_df) < 100:
                log.error(f"Insufficient training data: {len(train_df)} samples")
                return

            log.info(f"Training data: {len(train_df)} samples, "
                     f"positive rate: {train_df['target'].mean():.1%}")

            # Train model
            model, feat_cols = train_model(train_df)

            # Rank and select
            top_picks, rankings = rank_stocks(model, feat_cols, closes, volumes, highs, lows, last_idx)

            if not top_picks:
                log.error("No stocks selected — skipping rebalance")
                return

            # Execute paper rebalance
            state = execute_rebalance(state, top_picks, closes)
        else:
            log.info("No rebalance needed — reporting current state")

        # Report performance
        state = report_performance(state, closes)

        # Save state
        save_state(state)

        log.info(f"\nEngine completed successfully at {datetime.now().strftime('%H:%M:%S ET')}")

    except Exception as e:
        log.error(f"Engine failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
