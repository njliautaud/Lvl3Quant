"""
OFI Exhaustion Paper Engine — Minute-level mean-reversion signal
Fades extreme OFI spikes that align with recent trend (counter-trend exhaustion)

Config (validated IS/OOT):
  OFI z-score threshold: 3.0
  Volume z-score threshold: 1.0
  Hold period: 10 minutes
  30-min trend filter: same-direction OFI spike = fade

IS Sharpe 1.42, OOT Sharpe 1.51, regime-agnostic (green/red gap 0.11)
"""
import pandas as pd
import numpy as np
import glob
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Config
OFI_Z_THRESHOLD = 3.0
VOL_Z_THRESHOLD = 1.0
HOLD_MINUTES = 10
LOOKBACK_BARS = 60  # rolling window for z-scores
TREND_BARS = 30     # 30-min trend
COST_TICKS_RT = 2.376  # commission + 1 tick spread
TICK_VALUE = 12.50
TICK_SIZE = 0.25

STATE_DIR = Path('/home/jupiter/Lvl3Quant/lh_paper_state/ofi_exhaustion')
STATE_FILE = STATE_DIR / 'engine_state.json'
TRADE_LOG = STATE_DIR / 'trades.jsonl'


class OFIExhaustionEngine:
    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.positions = []  # active positions
        self.closed_trades = []
        self.ofi_buffer = []  # rolling buffer for z-score calc
        self.vol_buffer = []
        self.close_buffer = []
        self.bar_count = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.load_state()

    def load_state(self):
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                self.total_pnl = state.get('total_pnl', 0)
                self.trade_count = state.get('trade_count', 0)
                self.win_count = state.get('win_count', 0)
                self.positions = state.get('positions', [])
                print(f"Loaded state: {self.trade_count} trades, PnL ${self.total_pnl:.2f}")
            except Exception as e:
                print(f"State load error: {e}")

    def save_state(self):
        state = {
            'total_pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'positions': self.positions,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def log_trade(self, trade):
        with open(TRADE_LOG, 'a') as f:
            f.write(json.dumps(trade) + '\n')

    def compute_z(self, buffer, current):
        if len(buffer) < 20:
            return 0
        arr = np.array(buffer[-LOOKBACK_BARS:])
        mean = arr.mean()
        std = arr.std()
        if std < 1e-6:
            return 0
        return (current - mean) / std

    def process_bar(self, bar_time, ofi, volume, close):
        """Process a single minute bar and generate signals"""
        self.bar_count += 1
        self.ofi_buffer.append(ofi)
        self.vol_buffer.append(volume)
        self.close_buffer.append(close)

        # Keep buffer bounded
        max_buf = max(LOOKBACK_BARS, TREND_BARS) + 10
        if len(self.ofi_buffer) > max_buf:
            self.ofi_buffer = self.ofi_buffer[-max_buf:]
            self.vol_buffer = self.vol_buffer[-max_buf:]
            self.close_buffer = self.close_buffer[-max_buf:]

        # Check exits first
        self._check_exits(bar_time, close)

        # Need enough history
        if len(self.close_buffer) < TREND_BARS + 1:
            return None

        # Compute z-scores
        ofi_z = self.compute_z(self.ofi_buffer[:-1], ofi)
        vol_z = self.compute_z(self.vol_buffer[:-1], volume)

        # 30-min return (trend)
        if len(self.close_buffer) > TREND_BARS:
            ret_30m = (close / self.close_buffer[-TREND_BARS - 1]) - 1
        else:
            return None

        # Signal logic: fade OFI spikes aligned with trend
        signal = None

        # Bearish exhaustion → go LONG (OFI very negative + downtrend)
        if ofi_z < -OFI_Z_THRESHOLD and vol_z > VOL_Z_THRESHOLD and ret_30m < 0:
            signal = 'long'

        # Bullish exhaustion → go SHORT (OFI very positive + uptrend)
        elif ofi_z > OFI_Z_THRESHOLD and vol_z > VOL_Z_THRESHOLD and ret_30m > 0:
            signal = 'short'

        if signal and len(self.positions) < 3:  # max 3 concurrent
            position = {
                'direction': signal,
                'entry_price': close,
                'entry_time': bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
                'exit_bar': self.bar_count + HOLD_MINUTES,
                'ofi_z': round(ofi_z, 2),
                'vol_z': round(vol_z, 2),
                'ret_30m': round(ret_30m * 10000, 2)  # bps
            }
            self.positions.append(position)
            return signal

        return None

    def _check_exits(self, bar_time, close):
        """Exit positions that have reached hold period"""
        still_open = []
        for pos in self.positions:
            if self.bar_count >= pos['exit_bar']:
                # Exit
                entry = pos['entry_price']
                if pos['direction'] == 'long':
                    pnl_ticks = (close - entry) / TICK_SIZE - COST_TICKS_RT
                else:
                    pnl_ticks = (entry - close) / TICK_SIZE - COST_TICKS_RT

                pnl_dollars = pnl_ticks * TICK_VALUE
                self.total_pnl += pnl_dollars
                self.daily_pnl += pnl_dollars
                self.trade_count += 1
                if pnl_ticks > 0:
                    self.win_count += 1

                trade = {
                    'direction': pos['direction'],
                    'entry_price': entry,
                    'exit_price': close,
                    'entry_time': pos['entry_time'],
                    'exit_time': bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
                    'pnl_ticks': round(pnl_ticks, 2),
                    'pnl_dollars': round(pnl_dollars, 2),
                    'ofi_z': pos['ofi_z'],
                    'vol_z': pos['vol_z']
                }
                self.log_trade(trade)
                self.closed_trades.append(trade)
            else:
                still_open.append(pos)
        self.positions = still_open

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.ofi_buffer = []
        self.vol_buffer = []
        self.close_buffer = []
        self.bar_count = 0

    def stats(self):
        wr = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        return {
            'total_pnl': round(self.total_pnl, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'trades': self.trade_count,
            'win_rate': round(wr, 1),
            'open_positions': len(self.positions)
        }


def run_backtest():
    """Run backtest on historical minute bars"""
    files = sorted(glob.glob('/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/*.parquet'))
    print(f'Loading {len(files)} files...')
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['ts_minute'] = pd.to_datetime(df['ts_minute'], utc=True)
    df = df.sort_values('ts_minute').reset_index(drop=True)
    df['date'] = df['ts_minute'].dt.date.astype(str)

    dates = sorted(df['date'].unique())
    split_idx = int(len(dates) * 0.7)

    engine = OFIExhaustionEngine()
    daily_results = []

    for date in dates:
        day_df = df[df['date'] == date].sort_values('ts_minute')
        engine.reset_daily()

        for _, row in day_df.iterrows():
            engine.process_bar(
                bar_time=row['ts_minute'],
                ofi=row['ofi_1min'],
                volume=row['volume'],
                close=row['close']
            )

        # Force close any remaining positions at EOD
        if engine.positions and len(day_df) > 0:
            last_close = day_df.iloc[-1]['close']
            last_time = day_df.iloc[-1]['ts_minute']
            engine.bar_count = 9999  # force all exits
            engine._check_exits(last_time, last_close)

        daily_results.append({
            'date': date,
            'pnl': engine.daily_pnl,
            'trades': len([t for t in engine.closed_trades if True]),  # all today's
            'period': 'IS' if date <= dates[split_idx - 1] else 'OOT'
        })
        engine.closed_trades = []

    engine.save_state()

    # Results
    res_df = pd.DataFrame(daily_results)
    res_df['cum_pnl'] = res_df['pnl'].cumsum()

    for period in ['IS', 'OOT', 'ALL']:
        if period == 'ALL':
            sub = res_df
        else:
            sub = res_df[res_df['period'] == period]

        trading = sub[sub['pnl'] != 0]
        if len(trading) < 2:
            continue

        dpnl = trading['pnl'].values
        sharpe = np.mean(dpnl) / np.std(dpnl) * np.sqrt(252)
        down = dpnl[dpnl < 0]
        sortino = np.mean(dpnl) / np.std(down) * np.sqrt(252) if len(down) > 0 and np.std(down) > 0 else 0
        green = (dpnl > 0).sum()
        max_dd = 0
        peak = 0
        cum = 0
        for p in dpnl:
            cum += p
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)

        print(f'\n=== {period} ({len(sub)} days, {len(trading)} trading) ===')
        print(f'Total PnL: ${dpnl.sum():,.0f}')
        print(f'Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}')
        print(f'Avg daily: ${np.mean(dpnl):,.0f} | Green days: {green}/{len(trading)} ({green/len(trading)*100:.0f}%)')
        print(f'Max DD: ${max_dd:,.0f}')

    stats = engine.stats()
    print(f'\nTotal: {stats["trades"]} trades, WR {stats["win_rate"]}%, PnL ${stats["total_pnl"]:,.2f}')


def run_live():
    """Live mode — reads latest minute bars from MBO recorder output"""
    print("OFI Exhaustion Paper Engine — LIVE MODE")
    print(f"Config: OFI_z>{OFI_Z_THRESHOLD}, Vol_z>{VOL_Z_THRESHOLD}, Hold={HOLD_MINUTES}min")

    engine = OFIExhaustionEngine()

    # In live mode, we'd read from a streaming source
    # For now, poll the latest minute bar files
    last_processed = None

    while True:
        try:
            # Check for new minute bar data
            # The MBO recorder writes bars — check for latest
            bar_dir = '/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1'
            files = sorted(glob.glob(f'{bar_dir}/*.parquet'))
            if not files:
                time.sleep(60)
                continue

            latest_file = files[-1]
            if latest_file == last_processed:
                time.sleep(30)
                continue

            df = pd.read_parquet(latest_file)
            df['ts_minute'] = pd.to_datetime(df['ts_minute'], utc=True)
            df = df.sort_values('ts_minute')

            # Process new bars
            for _, row in df.iterrows():
                signal = engine.process_bar(
                    bar_time=row['ts_minute'],
                    ofi=row['ofi_1min'],
                    volume=row['volume'],
                    close=row['close']
                )
                if signal:
                    print(f"[{row['ts_minute']}] SIGNAL: {signal.upper()} @ {row['close']:.2f}")

            last_processed = latest_file
            engine.save_state()

            stats = engine.stats()
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M')}] "
                  f"Trades: {stats['trades']}, WR: {stats['win_rate']}%, "
                  f"PnL: ${stats['total_pnl']:,.2f}, Open: {stats['open_positions']}")

            time.sleep(60)

        except KeyboardInterrupt:
            print("\nShutting down...")
            engine.save_state()
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.live:
        run_live()
    else:
        print("Use --backtest or --live")
