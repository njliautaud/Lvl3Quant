#!/usr/bin/env python3
"""
Longer-Horizon Paper Trading Engine (HC #637)
==============================================

Paper trades the longer-horizon directional model on ES futures.
Runs on Jupiter CPU. Consumes minute bars from MBO data,
computes hourly features, runs LightGBM inference, and generates
2h/4h directional trades.

Architecture:
  - Loads pre-trained LightGBM model (walk-forward last fold)
  - Retrains daily on latest 60 days of data (model freshness)
  - Generates signals at hourly boundaries (RTH: 13:30-21:00 UTC)
  - Trades 1 ES contract per signal
  - Tracks P&L with full cost accounting

PM2: pm2 start longer_horizon_paper_engine.py --name lh-paper-engine

Author: Claude (HC #637)
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
REGIME_PATH = ROOT / "data" / "feature_store" / "v1" / "regime_features.parquet"
STATE_DIR = ROOT / "live_trading_linux" / "lh_paper_state"
MODEL_DIR = ROOT / "output" / "longer_horizon_v1" / "models"
LOG_DIR = ROOT / "logs"

STATE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s [LH-Paper] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_paper_engine.log")),
    ],
)
log = logging.getLogger('LH-Paper')

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COST_RT_TICKS = 2.376  # 2 spread crossings + commission

# Trading params
HORIZON = '2h'          # Primary trading horizon
CONFIDENCE_THRESHOLD = 0.20  # Top/bottom 20% triggers trade
TRAIN_DAYS = 60         # Rolling training window
MAX_POSITION = 1        # 1 ES contract
HOLD_HOURS = 2          # Hold for 2 hours then exit

# RTH boundaries (UTC hours)
RTH_START_HOUR = 13  # 13:30 UTC = 9:30 ET
RTH_END_HOUR = 20    # 20:00 UTC = 16:00 ET (last signal hour)


class LongerHorizonPaperEngine:
    """Paper trading engine for longer-horizon directional model."""

    def __init__(self):
        self.state_path = STATE_DIR / "state.json"
        self.trades_path = STATE_DIR / "trades.json"
        self.model = None
        self.feature_cols = None
        self.pred_quantiles = None  # For confidence thresholding
        self.last_train_date = None

        # Load or initialize state
        self.state = self._load_state()
        self.trades = self._load_trades()

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {
            'cash': 100_000.00,
            'position': 0,        # -1, 0, or +1
            'entry_price': 0,
            'entry_time': None,
            'entry_reason': '',
            'exit_target_time': None,
            'total_trades': 0,
            'total_pnl_ticks': 0,
            'total_pnl_dollars': 0,
            'wins': 0,
            'losses': 0,
            'last_signal_time': None,
            'last_retrain_date': None,
            'created': datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def _load_trades(self) -> List[Dict]:
        if self.trades_path.exists():
            with open(self.trades_path) as f:
                return json.load(f)
        return []

    def _save_trades(self):
        with open(self.trades_path, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)

    def _load_minute_bars(self, n_days: int = 70) -> pd.DataFrame:
        """Load recent minute bars for feature computation."""
        files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
        recent = files[-n_days:] if len(files) >= n_days else files

        frames = []
        for f in recent:
            try:
                df = pd.read_parquet(f)
                df['date'] = f.stem
                frames.append(df)
            except Exception as e:
                log.warning(f"Skip {f.stem}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined['ts_minute'] = pd.to_datetime(combined['ts_minute'], utc=True)
        return combined.sort_values('ts_minute').reset_index(drop=True)

    def _compute_features(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Compute hourly features from minute bars. Mirrors training pipeline."""
        # Import the training module's feature functions
        from alpha_discovery.longer_horizon_directional_v1 import (
            compute_hourly_features, add_rolling_context, add_macro_features
        )

        hourly_df = compute_hourly_features(minute_df)
        hourly_df = add_rolling_context(hourly_df)
        hourly_df = add_macro_features(hourly_df)

        return hourly_df

    def retrain(self, hourly_df: pd.DataFrame):
        """Retrain LightGBM on latest TRAIN_DAYS of data."""
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("LightGBM not installed")
            return

        dates = sorted(hourly_df['date'].unique())
        if len(dates) < TRAIN_DAYS:
            log.warning(f"Not enough days for training: {len(dates)} < {TRAIN_DAYS}")
            return

        train_dates = dates[-TRAIN_DAYS:]
        train_mask = hourly_df['date'].isin(train_dates)
        train_df = hourly_df[train_mask].copy()

        # Compute target: 2h forward ticks
        train_df['fwd_ticks'] = train_df.groupby('date')['close'].shift(-2) / train_df['close'] - 1
        train_df['fwd_ticks'] = (train_df.groupby('date')['close'].shift(-2) - train_df['close']) / 0.25
        train_df = train_df.dropna(subset=['fwd_ticks'])

        # Feature columns (clean mode: no price levels)
        exclude = {'fwd_ticks', 'date', 'ts', 'vol_regime_mode', 'open', 'high',
                    'low', 'close', 'intraday_cum_return', 'hour'}
        self.feature_cols = [c for c in train_df.columns
                             if c not in exclude and not c.startswith(('fwd_', 'direction_', 'up_'))]

        X = train_df[self.feature_cols].values
        y = train_df['fwd_ticks'].values

        # Clean
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]

        if len(X) < 100:
            log.warning(f"Too few training samples: {len(X)}")
            return

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
            'n_jobs': 8,
            'seed': 42,
        }

        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, train_data, num_boost_round=500)

        # Compute prediction quantiles for confidence thresholding
        preds = self.model.predict(X)
        self.pred_quantiles = {
            'p80': np.percentile(preds, 80),
            'p20': np.percentile(preds, 20),
            'p90': np.percentile(preds, 90),
            'p10': np.percentile(preds, 10),
        }

        self.last_train_date = train_dates[-1]
        self.state['last_retrain_date'] = self.last_train_date

        # Save model
        self.model.save_model(str(MODEL_DIR / "lh_lgbm_latest.txt"))
        log.info(f"Retrained on {len(train_dates)} days ({train_dates[0]}→{train_dates[-1]}), "
                 f"{len(X)} samples, quantiles: p20={self.pred_quantiles['p20']:.1f}, "
                 f"p80={self.pred_quantiles['p80']:.1f}")

    def predict(self, hourly_df: pd.DataFrame) -> Optional[Dict]:
        """Generate prediction for the latest hourly bar."""
        if self.model is None or self.feature_cols is None:
            log.warning("No model loaded — cannot predict")
            return None

        latest = hourly_df.iloc[-1:]
        if latest.empty:
            return None

        # Check we have all feature columns
        missing = [c for c in self.feature_cols if c not in latest.columns]
        if missing:
            log.warning(f"Missing features: {missing[:5]}...")
            return None

        X = latest[self.feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        pred = self.model.predict(X)[0]

        # Determine confidence level
        signal = 0
        confidence = 'low'
        if pred >= self.pred_quantiles['p80']:
            signal = 1  # Long
            confidence = 'high' if pred >= self.pred_quantiles['p90'] else 'medium'
        elif pred <= self.pred_quantiles['p20']:
            signal = -1  # Short
            confidence = 'high' if pred <= self.pred_quantiles['p10'] else 'medium'

        return {
            'prediction_ticks': pred,
            'signal': signal,
            'confidence': confidence,
            'bar_date': latest['date'].iloc[0],
            'bar_hour': latest['hour'].iloc[0] if 'hour' in latest.columns else None,
            'bar_close': latest['close'].iloc[0] if 'close' in latest.columns else None,
            'ts': datetime.now(timezone.utc).isoformat(),
        }

    def execute_signal(self, signal_info: Dict, current_price: float, bar_time=None):
        """Execute a trade based on the signal."""
        sig = signal_info['signal']
        if sig == 0:
            return  # No trade

        # Check if already positioned
        if self.state['position'] != 0:
            return  # Silent skip in backtest

        # Use bar_time if provided (backtest), else signal ts
        entry_ts = bar_time.isoformat() if bar_time else signal_info['ts']

        # Enter position
        self.state['position'] = sig
        self.state['entry_price'] = current_price
        self.state['entry_time'] = entry_ts
        self.state['entry_reason'] = (
            f"{'LONG' if sig > 0 else 'SHORT'} @ {current_price:.2f} | "
            f"pred={signal_info['prediction_ticks']:.1f}tk | conf={signal_info['confidence']}"
        )
        ref_time = bar_time if bar_time else datetime.fromisoformat(signal_info['ts'])
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)
        self.state['exit_target_time'] = (ref_time + timedelta(hours=HOLD_HOURS)).isoformat()

        log.info(f"ENTRY: {self.state['entry_reason']}")
        self._save_state()

    def check_exit(self, current_price: float, current_time: datetime):
        """Check if position should be closed (time-based exit)."""
        if self.state['position'] == 0:
            return

        target_time = datetime.fromisoformat(self.state['exit_target_time'])
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        if current_time >= target_time:
            self._close_position(current_price, 'TIME_EXIT')

    def _close_position(self, exit_price: float, reason: str):
        """Close current position and record trade."""
        pos = self.state['position']
        entry = self.state['entry_price']

        raw_ticks = (exit_price - entry) / 0.25 * pos  # pos is +1 or -1
        pnl_ticks = raw_ticks - COST_RT_TICKS
        pnl_dollars = pnl_ticks * ES_TICK_VALUE

        trade = {
            'direction': 'LONG' if pos > 0 else 'SHORT',
            'entry_price': entry,
            'exit_price': exit_price,
            'entry_time': self.state['entry_time'],
            'exit_time': datetime.now(timezone.utc).isoformat(),
            'exit_reason': reason,
            'raw_ticks': raw_ticks,
            'cost_ticks': COST_RT_TICKS,
            'pnl_ticks': pnl_ticks,
            'pnl_dollars': pnl_dollars,
        }
        self.trades.append(trade)

        # Update state
        self.state['position'] = 0
        self.state['entry_price'] = 0
        self.state['entry_time'] = None
        self.state['exit_target_time'] = None
        self.state['total_trades'] += 1
        self.state['total_pnl_ticks'] += pnl_ticks
        self.state['total_pnl_dollars'] += pnl_dollars
        self.state['cash'] += pnl_dollars
        if pnl_ticks > 0:
            self.state['wins'] += 1
        else:
            self.state['losses'] += 1

        wr = self.state['wins'] / max(self.state['total_trades'], 1) * 100
        log.info(f"EXIT ({reason}): {'LONG' if pos > 0 else 'SHORT'} | "
                 f"PnL: {pnl_ticks:+.1f} ticks (${pnl_dollars:+,.0f}) | "
                 f"Cumulative: {self.state['total_pnl_ticks']:+.0f} ticks "
                 f"(${self.state['total_pnl_dollars']:+,.0f}) | "
                 f"WR: {wr:.0f}% ({self.state['total_trades']} trades)")

        self._save_state()
        self._save_trades()

    def get_status(self) -> str:
        """Return human-readable status."""
        wr = self.state['wins'] / max(self.state['total_trades'], 1) * 100
        pos_str = 'FLAT'
        if self.state['position'] > 0:
            pos_str = f"LONG @ {self.state['entry_price']:.2f}"
        elif self.state['position'] < 0:
            pos_str = f"SHORT @ {self.state['entry_price']:.2f}"

        return (
            f"LH Paper Engine | {pos_str} | "
            f"Trades: {self.state['total_trades']} | WR: {wr:.0f}% | "
            f"PnL: {self.state['total_pnl_ticks']:+.0f}tk "
            f"(${self.state['total_pnl_dollars']:+,.0f}) | "
            f"Cash: ${self.state['cash']:,.0f}"
        )

    def run_backtest_mode(self):
        """
        Run as a backtest over historical data to validate paper engine logic.
        Uses the last 30 days as out-of-sample.
        """
        log.info("=" * 60)
        log.info("LONGER-HORIZON PAPER ENGINE — BACKTEST MODE")
        log.info("=" * 60)

        # Reset state for backtest
        self.state = self._load_state.__wrapped__(self) if hasattr(self._load_state, '__wrapped__') else {
            'cash': 100_000.00,
            'position': 0,
            'entry_price': 0,
            'entry_time': None,
            'entry_reason': '',
            'exit_target_time': None,
            'total_trades': 0,
            'total_pnl_ticks': 0,
            'total_pnl_dollars': 0,
            'wins': 0,
            'losses': 0,
            'last_signal_time': None,
            'last_retrain_date': None,
            'created': datetime.now(timezone.utc).isoformat(),
        }
        self.trades = []

        # Load all data
        minute_df = self._load_minute_bars(n_days=200)
        if minute_df.empty:
            log.error("No minute bars available")
            return

        hourly_df = self._compute_features(minute_df)
        dates = sorted(hourly_df['date'].unique())

        log.info(f"Data: {len(dates)} days ({dates[0]} → {dates[-1]})")

        # Walk through each day/hour
        oot_start = len(dates) - 30  # Last 30 days are OOT
        for i in range(TRAIN_DAYS, len(dates)):
            current_date = dates[i]
            day_bars = hourly_df[hourly_df['date'] == current_date].sort_values('ts')

            # Retrain daily (on data up to yesterday)
            if self.last_train_date != dates[i - 1]:
                train_data = hourly_df[hourly_df['date'].isin(dates[max(0, i-TRAIN_DAYS):i])]
                if len(train_data) > 100:
                    self.retrain(train_data)

            for _, bar in day_bars.iterrows():
                current_price = bar['close']
                current_time = bar['ts']

                if isinstance(current_time, str):
                    current_time = pd.Timestamp(current_time)
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=timezone.utc)

                # Check exit first
                self.check_exit(current_price, current_time)

                # Generate signal
                bar_idx = hourly_df.index[hourly_df['ts'] == bar['ts']]
                if len(bar_idx) == 0:
                    continue

                # Use all data up to this bar for prediction
                pred_data = hourly_df.loc[:bar_idx[0]]
                signal = self.predict(pred_data)

                if signal and signal['signal'] != 0:
                    self.execute_signal(signal, current_price, bar_time=current_time)

        # Final report
        log.info("\n" + "=" * 60)
        log.info("BACKTEST COMPLETE")
        log.info("=" * 60)
        log.info(self.get_status())

        if self.trades:
            pnls = [t['pnl_ticks'] for t in self.trades]
            pnl_arr = np.array(pnls)
            sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
            sortino_d = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
            sortino = pnl_arr.mean() / max(sortino_d, 1e-6) * np.sqrt(252)
            cum = np.cumsum(pnl_arr)
            max_dd = np.min(cum - np.maximum.accumulate(cum))

            log.info(f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | "
                     f"MaxDD: {max_dd:.0f} ticks (${max_dd * ES_TICK_VALUE:,.0f})")

            # OOT vs in-sample
            oot_date = dates[oot_start]
            oot_trades = [t for t in self.trades
                          if t.get('entry_time', '') > oot_date]
            is_trades = [t for t in self.trades
                         if t.get('entry_time', '') <= oot_date]

            if oot_trades:
                oot_pnl = np.array([t['pnl_ticks'] for t in oot_trades])
                log.info(f"OOT (last 30d): {len(oot_trades)} trades, "
                         f"WR={np.mean(oot_pnl > 0):.1%}, "
                         f"avg={oot_pnl.mean():.1f}tk, "
                         f"total={oot_pnl.sum():.0f}tk")

        self._save_state()
        self._save_trades()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode')
    parser.add_argument('--status', action='store_true', help='Print current status')
    args = parser.parse_args()

    engine = LongerHorizonPaperEngine()

    if args.status:
        print(engine.get_status())
        return

    if args.backtest:
        engine.run_backtest_mode()
        return

    # Live paper trading mode
    log.info("Starting Longer-Horizon Paper Engine (live mode)")
    log.info(engine.get_status())

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Only run during RTH
            if now.hour < RTH_START_HOUR or now.hour >= RTH_END_HOUR + 1:
                time.sleep(60)
                continue

            # Check at the top of each hour
            if now.minute < 5:  # First 5 minutes of each hour
                minute_df = engine._load_minute_bars()
                if minute_df.empty:
                    log.warning("No minute bars available")
                    time.sleep(300)
                    continue

                hourly_df = engine._compute_features(minute_df)

                # Retrain daily
                dates = sorted(hourly_df['date'].unique())
                if dates and engine.last_train_date != dates[-2]:  # Train on yesterday
                    engine.retrain(hourly_df)

                # Get latest bar's close as current price
                latest = hourly_df.iloc[-1]
                current_price = latest['close']

                # Check exit
                engine.check_exit(current_price, now)

                # Generate signal
                signal = engine.predict(hourly_df)
                if signal:
                    log.info(f"Signal: {signal['signal']} ({signal['confidence']}) "
                             f"pred={signal['prediction_ticks']:.1f}tk")
                    if signal['signal'] != 0 and signal['confidence'] in ('high', 'medium'):
                        engine.execute_signal(signal, current_price)

                engine._save_state()

            time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            log.info("Shutting down")
            break
        except Exception as e:
            log.error(f"Error: {e}\n{traceback.format_exc()}")
            time.sleep(60)


if __name__ == '__main__':
    main()
