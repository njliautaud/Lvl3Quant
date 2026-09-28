#!/usr/bin/env python3
"""
ML Sector Rotation — PAPER ENGINE
===================================
Runs daily at market close (4:50 PM ET via PM2 cron).
Generates sector rotation signals for next day's positions.

Based on the validated ML Sector Rotation strategy:
  Sharpe 1.99, Sortino 2.75, CAGR 16.4%, MaxDD -9.2%
  3/4 adversarial gates PASS, R1 regime gap 0.14 (PASS).

Strategy: 11-sector SPDR ETF rotation with GBM ML filter.
- Generates dual-MA (20/100) crossover signals for sector ETFs
- ML (GBM) trained on rolling 252d to predict trend continuation
- Only takes positions where ML confidence > 0.55
- Risk parity sizing (target 10% vol per position)

Paper tracking: logs positions, signals, and PnL to output/ml_trend_sectors/paper/
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_trend_sectors" / "paper"
OUTPUT.mkdir(parents=True, exist_ok=True)

POSITIONS_LOG = OUTPUT / "positions_log.json"
TRADES_LOG = OUTPUT / "trades_log.json"
DAILY_PNL = OUTPUT / "daily_pnl.csv"
PERF_SUMMARY = OUTPUT / "performance_summary.json"

# Strategy params (from validated backtest — results.json)
MA_SHORT = 20
MA_LONG = 100
ML_THRESHOLD = 0.55
TARGET_VOL = 0.10
TRAIN_WINDOW = 252
INITIAL_CAPITAL = 100_000
REBAL_COST_BPS = 10

# Backtest expectations (for tracking deviation)
BACKTEST_SHARPE = 1.986
BACKTEST_SORTINO = 2.745
BACKTEST_CAGR = 0.164
BACKTEST_MAX_DD = -0.092
BACKTEST_WIN_RATE = 0.521

UNIVERSE = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLY': 'Consumer Disc',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Svcs',
}


def get_data(lookback_days=500):
    """Download recent data for all sector ETFs + VIX + SPY (benchmark)."""
    tickers = list(UNIVERSE.keys()) + ['^VIX', 'SPY']
    start = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)
    return closes


def compute_trend_signals(df):
    """Compute dual-MA crossover signals for each sector."""
    signals = {}
    for ticker in UNIVERSE:
        if ticker not in df.columns:
            continue
        price = df[ticker]
        ma_short = price.rolling(MA_SHORT).mean()
        ma_long = price.rolling(MA_LONG).mean()
        trend = pd.Series(0.0, index=df.index)
        trend[ma_short > ma_long] = 1.0
        trend[ma_short < ma_long] = -1.0
        signals[ticker] = {
            'trend': trend,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'price': price,
        }
    return signals


def build_features_for_day(df, signals, day_idx):
    """Build ML features for a single day across all sectors."""
    rows = []
    for ticker, sig in signals.items():
        price = sig['price']
        trend = sig['trend']
        ma_s = sig['ma_short']
        ma_l = sig['ma_long']
        ret = price.pct_change()
        i = day_idx

        if trend.iloc[i] == 0:
            continue

        feat = {}
        feat['ticker'] = ticker
        feat['date'] = df.index[i]
        feat['direction'] = trend.iloc[i]

        # Trend strength (MA distance normalized by price)
        feat['ma_dist'] = (ma_s.iloc[i] - ma_l.iloc[i]) / (price.iloc[i] + 1e-8)

        # Trend duration
        dur = 0
        for j in range(i, max(i - 252, 0), -1):
            if trend.iloc[j] == trend.iloc[i]:
                dur += 1
            else:
                break
        feat['trend_duration'] = dur

        # Momentum at multiple horizons
        feat['mom_5d'] = price.pct_change(5).iloc[i]
        feat['mom_20d'] = price.pct_change(20).iloc[i]
        feat['mom_60d'] = price.pct_change(60).iloc[i]

        # Volatility
        feat['vol_20d'] = ret.iloc[max(0, i-20):i].std() * np.sqrt(252)
        feat['vol_60d'] = ret.iloc[max(0, i-60):i].std() * np.sqrt(252)

        # Drawdown from 252d high
        peak = price.iloc[max(0, i-252):i+1].max()
        feat['drawdown'] = price.iloc[i] / peak - 1

        # Cross-sector alignment (how many sectors agree on direction)
        same_dir = sum(1 for t2, s2 in signals.items()
                       if t2 != ticker and s2['trend'].iloc[i] == trend.iloc[i])
        feat['cross_align'] = same_dir / max(len(signals) - 1, 1)

        # Relative strength vs SPY
        if 'SPY' in df.columns:
            spy_ret_20 = df['SPY'].pct_change(20).iloc[i]
            sector_ret_20 = price.pct_change(20).iloc[i]
            feat['rel_strength_20d'] = sector_ret_20 - spy_ret_20 if pd.notna(spy_ret_20) else 0.0

            spy_ret_60 = df['SPY'].pct_change(60).iloc[i]
            sector_ret_60 = price.pct_change(60).iloc[i]
            feat['rel_strength_60d'] = sector_ret_60 - spy_ret_60 if pd.notna(spy_ret_60) else 0.0

        # VIX regime
        if 'VIX' in df.columns:
            feat['vix'] = df['VIX'].iloc[i]
            feat['vix_ma20'] = df['VIX'].rolling(20).mean().iloc[i]

        # Skew/kurtosis of recent returns
        if i > 20:
            feat['skew_20d'] = ret.iloc[i-20:i].skew()
            feat['kurt_20d'] = ret.iloc[i-20:i].kurt()

        rows.append(feat)
    return rows


def build_training_data(df, signals, end_idx, window=252):
    """Build training set using the last `window` days before end_idx."""
    rows = []
    start_idx = max(MA_LONG + 60, end_idx - window)

    for i in range(start_idx, end_idx):
        for ticker, sig in signals.items():
            price = sig['price']
            trend = sig['trend']
            ma_s = sig['ma_short']
            ma_l = sig['ma_long']
            ret = price.pct_change()

            if trend.iloc[i] == 0:
                continue

            feat = {}
            feat['ticker'] = ticker
            feat['direction'] = trend.iloc[i]
            feat['ma_dist'] = (ma_s.iloc[i] - ma_l.iloc[i]) / (price.iloc[i] + 1e-8)

            dur = 0
            for j in range(i, max(i - 252, 0), -1):
                if trend.iloc[j] == trend.iloc[i]:
                    dur += 1
                else:
                    break
            feat['trend_duration'] = dur
            feat['mom_5d'] = price.pct_change(5).iloc[i]
            feat['mom_20d'] = price.pct_change(20).iloc[i]
            feat['mom_60d'] = price.pct_change(60).iloc[i]
            feat['vol_20d'] = ret.iloc[max(0, i-20):i].std() * np.sqrt(252)
            feat['vol_60d'] = ret.iloc[max(0, i-60):i].std() * np.sqrt(252)
            peak = price.iloc[max(0, i-252):i+1].max()
            feat['drawdown'] = price.iloc[i] / peak - 1
            same_dir = sum(1 for t2, s2 in signals.items()
                           if t2 != ticker and s2['trend'].iloc[i] == trend.iloc[i])
            feat['cross_align'] = same_dir / max(len(signals) - 1, 1)

            # Relative strength vs SPY
            if 'SPY' in df.columns:
                spy_ret_20 = df['SPY'].pct_change(20).iloc[i]
                sector_ret_20 = price.pct_change(20).iloc[i]
                feat['rel_strength_20d'] = sector_ret_20 - spy_ret_20 if pd.notna(spy_ret_20) else 0.0
                spy_ret_60 = df['SPY'].pct_change(60).iloc[i]
                sector_ret_60 = price.pct_change(60).iloc[i]
                feat['rel_strength_60d'] = sector_ret_60 - spy_ret_60 if pd.notna(spy_ret_60) else 0.0

            if 'VIX' in df.columns:
                feat['vix'] = df['VIX'].iloc[i]
                feat['vix_ma20'] = df['VIX'].rolling(20).mean().iloc[i]
            if i > 20:
                feat['skew_20d'] = ret.iloc[i-20:i].skew()
                feat['kurt_20d'] = ret.iloc[i-20:i].kurt()

            # Target: trend continues over next 20d
            if i + 20 < len(df):
                future_ret = (price.iloc[i + 20] / price.iloc[i] - 1) * trend.iloc[i]
                feat['target'] = 1 if future_ret > 0 else 0
            else:
                feat['target'] = np.nan

            rows.append(feat)

    train_df = pd.DataFrame(rows).dropna(subset=['target'])
    return train_df


def generate_daily_signal():
    """Main: train ML on recent history, generate today's sector positions."""
    print(f"\n{'='*80}")
    print(f"ML SECTOR ROTATION PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")

    # Get data
    df = get_data(lookback_days=500)
    print(f"Data: {df.shape[0]} days, {df.index[0].date()} -> {df.index[-1].date()}")

    # Generate signals
    signals = compute_trend_signals(df)

    # Build training data (last 252 days, with 20d label lookahead buffer)
    today_idx = len(df) - 1
    train_df = build_training_data(df, signals, end_idx=today_idx - 20, window=TRAIN_WINDOW)

    feature_cols = [c for c in train_df.columns
                    if c not in ['ticker', 'date', 'direction', 'target']]

    print(f"Training samples: {len(train_df)} (positive rate: {train_df['target'].mean():.1%})")

    # Train GBM
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target'].astype(int)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"Train accuracy: {train_acc:.1%}")

    # Generate today's features and predictions
    today_features = build_features_for_day(df, signals, today_idx)

    if not today_features:
        print("No active sector signals today — all flat.")
        positions = {}
    else:
        pred_df = pd.DataFrame(today_features)
        X_today = pred_df[feature_cols].fillna(0)
        probs = model.predict_proba(X_today)[:, 1]
        pred_df['ml_prob'] = probs

        # Filter by threshold
        active = pred_df[pred_df['ml_prob'] >= ML_THRESHOLD].copy()
        print(f"\nSector signals passing ML filter ({ML_THRESHOLD}): {len(active)}/{len(pred_df)}")

        # Risk parity sizing
        positions = {}
        for _, row in active.iterrows():
            ticker = row['ticker']
            vol = row['vol_20d'] if row['vol_20d'] > 0 else 0.15
            weight = TARGET_VOL / vol
            weight = min(weight, 0.25)  # Cap at 25% per sector
            positions[ticker] = {
                'direction': int(row['direction']),
                'weight': round(weight, 4),
                'ml_confidence': round(row['ml_prob'], 4),
                'trend_duration': int(row['trend_duration']),
                'vol_20d': round(vol, 4),
                'sector': UNIVERSE.get(ticker, ''),
            }

    # Normalize weights if total > 1
    total_weight = sum(abs(p['weight']) for p in positions.values())
    if total_weight > 1.0:
        for t in positions:
            positions[t]['weight'] = round(positions[t]['weight'] / total_weight, 4)

    # Log positions
    today_str = df.index[-1].strftime('%Y-%m-%d')
    log_entry = {
        'date': today_str,
        'timestamp': datetime.now().isoformat(),
        'positions': positions,
        'n_signals': len(today_features),
        'n_active': len(positions),
        'vix': float(df['VIX'].iloc[-1]) if 'VIX' in df.columns else None,
    }

    # Append to positions log
    if POSITIONS_LOG.exists():
        with open(POSITIONS_LOG, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append(log_entry)
    with open(POSITIONS_LOG, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Print summary
    print(f"\n{'_'*60}")
    print(f"TODAY'S SECTOR POSITIONS ({today_str}):")
    print(f"{'_'*60}")
    if positions:
        for ticker, pos in sorted(positions.items(), key=lambda x: -abs(x[1]['weight'])):
            dir_str = "LONG" if pos['direction'] == 1 else "SHORT"
            print(f"  {ticker:5s} ({pos['sector']:18s}) {dir_str:5s} "
                  f"weight={pos['weight']:.2%} conf={pos['ml_confidence']:.3f} "
                  f"dur={pos['trend_duration']}d vol={pos['vol_20d']:.1%}")
    else:
        print("  ALL CASH (no sector signals pass ML filter)")

    if log_entry['vix']:
        print(f"\nVIX: {log_entry['vix']:.1f}")
    net_exp = sum(p['weight'] * p['direction'] for p in positions.values())
    gross_exp = sum(abs(p['weight']) for p in positions.values())
    print(f"Net exposure: {net_exp:.1%}")
    print(f"Gross exposure: {gross_exp:.1%}")
    print(f"Active sectors: {len(positions)}/{len(UNIVERSE)}")

    # Track PnL if we have prior day positions
    track_pnl(df, history)

    return positions


def track_pnl(df, history):
    """Calculate paper PnL from prior positions and track vs backtest."""
    if len(history) < 2:
        return

    prev = history[-2]

    if not prev['positions']:
        return

    daily_ret = 0.0
    for ticker, pos in prev['positions'].items():
        if ticker not in df.columns:
            continue
        today_ret = df[ticker].pct_change().iloc[-1]
        position_ret = today_ret * pos['direction'] * pos['weight']
        daily_ret += position_ret

    # Subtract rebalance cost estimate
    daily_ret -= REBAL_COST_BPS / 10000

    # Append to PnL csv
    curr_date = df.index[-1]
    pnl_row = pd.DataFrame([{
        'date': curr_date.strftime('%Y-%m-%d'),
        'daily_return': daily_ret,
        'n_positions': len(prev['positions']),
        'positions': json.dumps(prev['positions']),
    }])

    if DAILY_PNL.exists():
        existing = pd.read_csv(DAILY_PNL)
        pnl_df = pd.concat([existing, pnl_row], ignore_index=True)
    else:
        pnl_df = pnl_row

    pnl_df.to_csv(DAILY_PNL, index=False)

    # Summary stats and backtest comparison
    if len(pnl_df) >= 5:
        rets = pnl_df['daily_return'].astype(float)
        cum_ret = (1 + rets).prod() - 1
        sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
        downside = rets[rets < 0].std() + 1e-8
        sortino = rets.mean() / downside * np.sqrt(252)
        win_rate = (rets > 0).mean()
        max_dd = (((1 + rets).cumprod() / (1 + rets).cumprod().cummax()) - 1).min()

        print(f"\n  Paper PnL ({len(pnl_df)} days):")
        print(f"    Cumulative: {cum_ret:.2%}")
        print(f"    Sharpe:  {sharpe:.2f}  (backtest: {BACKTEST_SHARPE:.2f})")
        print(f"    Sortino: {sortino:.2f}  (backtest: {BACKTEST_SORTINO:.2f})")
        print(f"    Win rate: {win_rate:.1%} (backtest: {BACKTEST_WIN_RATE:.1%})")
        print(f"    Max DD:  {max_dd:.2%}  (backtest: {BACKTEST_MAX_DD:.1%})")

        # Save performance summary
        summary = {
            'updated': datetime.now().isoformat(),
            'paper_days': len(pnl_df),
            'paper': {
                'cumulative_return': round(cum_ret, 4),
                'sharpe': round(sharpe, 3),
                'sortino': round(sortino, 3),
                'win_rate': round(win_rate, 3),
                'max_dd': round(max_dd, 4),
            },
            'backtest': {
                'sharpe': BACKTEST_SHARPE,
                'sortino': BACKTEST_SORTINO,
                'cagr': BACKTEST_CAGR,
                'max_dd': BACKTEST_MAX_DD,
                'win_rate': BACKTEST_WIN_RATE,
            },
            'deviation': {
                'sharpe_gap': round(sharpe - BACKTEST_SHARPE, 3),
                'win_rate_gap': round(win_rate - BACKTEST_WIN_RATE, 3),
            },
        }
        with open(PERF_SUMMARY, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    positions = generate_daily_signal()
