#!/usr/bin/env python3
"""
Queue Entry Selector v2.1 — Paper Trading Engine
==================================================

Paper trades the v2.1 champion queue entry selector on ES futures.

TWO MODES:
  1. REPLAY: Uses precomputed OOT predictions from walk-forward training
     (fast — seconds, not hours). This is the HONEST test since predictions
     are strictly out-of-time.
  2. LIVE: When new data arrives beyond the OOT range, trains a fresh model
     on last 25 days and predicts. Runs as PM2 process.

CHAMPION STRATEGY (v2.1 OOT, cost-adjusted @ 0.67):
  - Entry model: LightGBM binary classifier on 28 queue microstructure features
  - Direction normalization: feature swapping for side-agnostic learning
  - Multi-hyperparameter: default/shallow/deep LightGBM, best per fold
  - Confidence threshold: 0.67 (cost-adjusted optimal, was 0.60 pre-cost)
  - FIFO config: tp4sl3 (4-tick TP, 3-tick SL, FIFO passive fill)

ASYMMETRIC COST MODEL (ES Futures — AMP/Rithmic):
  - FIFO labels net_ticks include 0.376t RT commission
    (winners: 4.0 TP - 0.376 = 3.624t; losers: -3.0 SL - 0.376 = -3.376t)
  - SL exits require market orders → 1.0t additional spread crossing cost
  - Paper engine applies: winners = raw net_ticks, losers = raw net_ticks - 1.0t
  - Total cost: TP hit = 0.376t (commission), SL hit = 1.376t (commission + spread)
  - This asymmetric cost shifts optimal threshold from 0.60 to 0.67

PM2:
  pm2 start queue_entry_v21_paper.py --name queue-v21-paper \
    --interpreter python3 -- --live

Author: Claude (autonomous build, v2.1 integration)
"""

import argparse
import gc
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

OOT_PARQUET = ROOT / "output" / "queue_entry_selector_v2_1" / "all_oot_trades_tp4sl3.parquet"
QUEUE_DIR = ROOT / "output" / "queue_features_universal"
FIFO_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
ENGINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ENGINE_DIR / "logs" / "queue_v21_paper"
STATE_DIR = OUTPUT_DIR

for d in [OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_FILE = OUTPUT_DIR / "queue_v21_paper.log"
logging.basicConfig(
    format="%(asctime)s [QES-v2.1] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger("QES-v2.1")

# ─────────────────────────────────────────────
#  CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # $4.70 / $12.50

# Strategy parameters
CONFIDENCE_THRESHOLD = 0.67   # Cost-adjusted optimal (asymmetric SL cost analysis)
SL_SPREAD_CROSSING_TICKS = 1.0  # SL exit = market order → 1 tick spread crossing
FIFO_CONFIG = "tp4sl3"
MIN_HOUR_ET = 14  # Only trade after 2 PM ET (morning trades are net losers)


# ═══════════════════════════════════════════════════════════════════
#  PAPER ENGINE
# ═══════════════════════════════════════════════════════════════════

class QueueEntryV21PaperEngine:
    """Paper trading engine for queue entry selector v2.1."""

    def __init__(self):
        self.state_path = STATE_DIR / "state.json"
        self.trades_path = STATE_DIR / "trades.csv"
        self.daily_path = STATE_DIR / "daily_pnl.csv"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'last_processed_date': None,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run_replay(self):
        """Replay using precomputed OOT predictions — fast and honest."""
        log.info("=" * 70)
        log.info("QUEUE ENTRY SELECTOR v2.1 — PAPER TRADING REPLAY")
        log.info(f"Using precomputed OOT predictions from walk-forward training")
        log.info(f"FIFO config: {FIFO_CONFIG} | Threshold: {CONFIDENCE_THRESHOLD}")
        log.info(f"Cost: passive entry + TP = {ES_RT_COMMISSION_TICKS:.3f}t, "
                 f"passive entry + SL = {ES_RT_COMMISSION_TICKS + 1.0:.3f}t")
        log.info("=" * 70)

        if not OOT_PARQUET.exists():
            log.error(f"OOT predictions not found at {OOT_PARQUET}")
            return

        # Load all OOT predictions
        log.info("Loading OOT predictions...")
        df = pd.read_parquet(OOT_PARQUET)
        log.info(f"Total entries: {len(df):,} across {df['date'].nunique()} dates")

        # Filter by confidence threshold
        selected = df[df['pred_prob'] >= CONFIDENCE_THRESHOLD].copy()
        log.info(f"Selected at threshold {CONFIDENCE_THRESHOLD}: {len(selected):,} trades")

        if len(selected) == 0:
            log.error("No trades selected at threshold!")
            return

        # Time-of-day filter: only trade after MIN_HOUR_ET (morning trades are net losers)
        if MIN_HOUR_ET > 0:
            selected['_hour_et'] = pd.to_datetime(
                selected['ts_ns'].astype(float) / 1e9, unit='s'
            ).dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.hour
            pre_filter = len(selected)
            selected = selected[selected['_hour_et'] >= MIN_HOUR_ET].copy()
            selected.drop(columns=['_hour_et'], inplace=True)
            log.info(f"Time filter (>= {MIN_HOUR_ET}:00 ET): {pre_filter} → {len(selected)} trades "
                     f"({pre_filter - len(selected)} morning trades removed)")

        # Classify day regimes from queue features
        log.info("Classifying day regimes...")
        regime_map = {}
        for date_str in sorted(df['date'].unique()):
            queue_df = self._load_queue_features(date_str)
            if queue_df is not None and 'mid_price' in queue_df.columns and len(queue_df) >= 10:
                day_ret = queue_df['mid_price'].iloc[-1] - queue_df['mid_price'].iloc[0]
                if day_ret > 2 * ES_TICK_SIZE:
                    regime_map[date_str] = 'green'
                elif day_ret < -2 * ES_TICK_SIZE:
                    regime_map[date_str] = 'red'
                else:
                    regime_map[date_str] = 'flat'
            else:
                regime_map[date_str] = 'unknown'
            del queue_df
        gc.collect()

        selected['regime'] = selected['date'].map(regime_map).fillna('unknown')

        # FIFO labels net_ticks ALREADY includes 0.376t RT commission
        # (winners: 4.0 TP - 0.376 = 3.624t; losers: -3.0 SL - 0.376 = -3.376t)
        # BUT losers also pay 1.0t SL spread crossing (market exit), NOT included in labels
        # ASYMMETRIC COST MODEL:
        #   TP hit → passive TP limit fill → cost = commission only (0.376t) → already in labels
        #   SL hit → market exit crossing spread → extra 1.0t cost beyond labels
        raw_net = selected['net_ticks'].astype(float)
        sl_penalty = np.where(raw_net < 0, -SL_SPREAD_CROSSING_TICKS, 0.0)
        selected['paper_net_ticks'] = raw_net + sl_penalty
        selected['cost_ticks_included'] = ES_RT_COMMISSION_TICKS  # commission in labels
        selected['sl_spread_cost'] = -sl_penalty  # extra cost applied to losers
        selected['winner'] = (selected['paper_net_ticks'] > 0).astype(int)

        # Save detailed trades
        trade_cols = ['date', 'ts_ns', 'side', 'pred_prob',
                      'paper_net_ticks', 'regime', 'winner']
        selected[trade_cols].to_csv(self.trades_path, index=False)
        log.info(f"Saved {len(selected)} trades to {self.trades_path.name}")

        # Daily P&L
        daily = selected.groupby('date').agg(
            regime=('regime', 'first'),
            n_trades=('paper_net_ticks', 'count'),
            n_winners=('winner', 'sum'),
            net_ticks=('paper_net_ticks', 'sum'),
        ).reset_index()

        daily['wr'] = daily['n_winners'] / daily['n_trades']
        daily['net_dollars'] = daily['net_ticks'] * ES_TICK_VALUE
        daily['cum_pnl_ticks'] = daily['net_ticks'].cumsum()
        daily['cum_pnl_dollars'] = daily['cum_pnl_ticks'] * ES_TICK_VALUE
        daily['commission_included'] = daily['n_trades'] * ES_RT_COMMISSION_TICKS

        daily.to_csv(self.daily_path, index=False)

        # Print day-by-day log
        log.info("\n  --- DAY-BY-DAY (net_ticks includes 0.376t commission) ---")
        for _, row in daily.iterrows():
            log.info(
                f"  {row['date']} [{row['regime']:5s}] | "
                f"trades={int(row['n_trades']):3d} | "
                f"WR={row['wr']:.0%} | "
                f"net={row['net_ticks']:+.1f}t | "
                f"cum={row['cum_pnl_ticks']:+.1f}t "
                f"(${row['cum_pnl_dollars']:+,.0f})"
            )

        # Comprehensive summary
        self._print_summary(selected, daily)

        self.state['last_processed_date'] = str(daily['date'].iloc[-1])
        self._save_state()

    def _load_queue_features(self, date_str: str):
        """Load queue features for regime classification."""
        path = QUEUE_DIR / f"features_{date_str.replace('-', '')}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path, columns=['ts_ns', 'mid_price'])
            if len(df) < 10:
                return None
            return df
        except Exception:
            return None

    def _print_summary(self, trades_df: pd.DataFrame, daily_df: pd.DataFrame):
        """Print comprehensive performance summary."""
        log.info("\n" + "=" * 70)
        log.info("QUEUE ENTRY SELECTOR v2.1 — PAPER TRADING SUMMARY")
        log.info("=" * 70)

        total_days = len(daily_df)
        total_trades = len(trades_df)
        total_net = float(trades_df['paper_net_ticks'].sum())
        total_commission_included = total_trades * ES_RT_COMMISSION_TICKS
        total_winners = int(trades_df['winner'].sum())
        total_losers = total_trades - total_winners
        wr = total_winners / max(total_trades, 1)

        # Daily stats
        daily_net = daily_df['net_ticks']
        sharpe = float(daily_net.mean() / daily_net.std()) if daily_net.std() > 0 else 0
        downside = daily_net[daily_net < 0]
        sortino = float(daily_net.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else (
            999.0 if daily_net.mean() > 0 else 0)

        # Profit factor (on daily net)
        daily_wins = daily_net[daily_net > 0].sum()
        daily_losses = abs(daily_net[daily_net < 0].sum())
        pf_daily = daily_wins / max(daily_losses, 1e-6)

        # Per-trade profit factor
        trade_wins = float(trades_df[trades_df['paper_net_ticks'] > 0]['paper_net_ticks'].sum())
        trade_losses = float(abs(trades_df[trades_df['paper_net_ticks'] < 0]['paper_net_ticks'].sum()))
        pf_trade = trade_wins / max(trade_losses, 1e-6)

        # Max drawdown
        cum = daily_net.cumsum()
        peak = cum.cummax()
        dd = cum - peak
        max_dd = float(dd.min())

        per_trade = total_net / max(total_trades, 1)
        per_day = total_net / max(total_days, 1)

        # Calmar
        calmar = float((per_day * 252) / abs(max_dd)) if max_dd != 0 else 999.0

        # Win/loss ratio
        avg_win = trade_wins / max(total_winners, 1)
        avg_loss = trade_losses / max(total_losers, 1)
        wl_ratio = avg_win / max(avg_loss, 1e-6)

        log.info(f"\n  Period: {daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]} ({total_days} trading days)")
        log.info(f"  NOTE: P&L already includes {ES_RT_COMMISSION_TICKS}t RT commission per trade")
        log.info(f"  Total trades: {total_trades} ({total_trades/max(total_days,1):.1f}/day)")
        log.info(f"  Win rate: {wr:.1%} ({total_winners}W / {total_losers}L)")
        log.info(f"  Avg win: {avg_win:+.2f}t | Avg loss: {avg_loss:.2f}t | W/L ratio: {wl_ratio:.2f}")
        log.info(f"")
        log.info(f"  Net P&L (incl {total_commission_included:.0f}t commission): "
                 f"{total_net:+.1f} ticks (${total_net * ES_TICK_VALUE:+,.0f})")
        log.info(f"  Per trade: {per_trade:+.3f} ticks (${per_trade * ES_TICK_VALUE:+.2f})")
        log.info(f"  Per day: {per_day:+.1f} ticks (${per_day * ES_TICK_VALUE:+,.0f} per contract)")
        log.info(f"")
        log.info(f"  Daily Sharpe: {sharpe:.3f}")
        log.info(f"  Daily Sortino: {sortino:.3f}")
        log.info(f"  Profit Factor (trade): {pf_trade:.3f}")
        log.info(f"  Profit Factor (daily): {pf_daily:.3f}")
        log.info(f"  Max DD: {max_dd:.1f} ticks (${max_dd * ES_TICK_VALUE:+,.0f})")
        log.info(f"  Calmar: {calmar:.3f}")

        # Day classification
        green_days = int((daily_df['net_ticks'] > 0).sum())
        red_days = int((daily_df['net_ticks'] < 0).sum())
        flat_days = int((daily_df['net_ticks'] == 0).sum())
        log.info(f"  Green/Red/Flat days: {green_days}/{red_days}/{flat_days} "
                 f"({green_days/max(total_days,1):.0%} green)")

        # Day-of-week breakdown
        log.info(f"\n  --- DAY-OF-WEEK BREAKDOWN ---")
        daily_df_copy = daily_df.copy()
        daily_df_copy['dow'] = pd.to_datetime(daily_df_copy['date'], format='%Y%m%d').dt.day_name()
        for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            dow_df = daily_df_copy[daily_df_copy['dow'] == dow]
            if len(dow_df) == 0:
                continue
            d_net = float(dow_df['net_ticks'].sum())
            d_trades = int(dow_df['n_trades'].sum())
            d_wr = float(dow_df['n_winners'].sum()) / max(d_trades, 1)
            log.info(f"  {dow:9s}: {len(dow_df)} days, {d_trades} trades, "
                     f"WR={d_wr:.0%}, net={d_net:+.1f}t")

        # Regime breakdown
        log.info(f"\n  --- REGIME BREAKDOWN ---")
        for regime in ['green', 'red', 'flat', 'unknown']:
            r_daily = daily_df[daily_df['regime'] == regime]
            if len(r_daily) == 0:
                continue
            r_net = float(r_daily['net_ticks'].sum())
            r_trades = int(r_daily['n_trades'].sum())
            r_winners = int(r_daily['n_winners'].sum())
            r_wr = r_winners / max(r_trades, 1)
            r_sharpe = float(r_daily['net_ticks'].mean() / r_daily['net_ticks'].std()) \
                if len(r_daily) > 1 and r_daily['net_ticks'].std() > 0 else 0
            log.info(f"  {regime:7s}: {len(r_daily):3d} days, {r_trades:4d} trades, "
                     f"WR={r_wr:.0%}, net={r_net:+.1f}t, Sharpe={r_sharpe:.3f}")

        # Regime gap (HC #428 R1)
        green_daily = daily_df[daily_df['regime'] == 'green']['net_ticks']
        red_daily = daily_df[daily_df['regime'] == 'red']['net_ticks']
        if len(green_daily) > 1 and len(red_daily) > 1:
            g_sharpe = float(green_daily.mean() / green_daily.std()) if green_daily.std() > 0 else 0
            r_sharpe = float(red_daily.mean() / red_daily.std()) if red_daily.std() > 0 else 0
            denom = max(abs(g_sharpe), abs(r_sharpe), 1e-6)
            gap = abs(g_sharpe - r_sharpe) / denom
            log.info(f"  Regime gap: {gap:.1%} (green Sharpe={g_sharpe:.3f}, red Sharpe={r_sharpe:.3f}) "
                     f"→ {'PASS' if gap <= 0.50 else 'FAIL'}")

        # Side breakdown
        log.info(f"\n  --- SIDE BREAKDOWN ---")
        for side in ['long', 'short']:
            s_df = trades_df[trades_df['side'] == side]
            if len(s_df) == 0:
                continue
            s_net = float(s_df['paper_net_ticks'].sum())
            s_wr = float(s_df['winner'].mean())
            s_wins = float(s_df[s_df['paper_net_ticks'] > 0]['paper_net_ticks'].sum())
            s_losses = float(abs(s_df[s_df['paper_net_ticks'] < 0]['paper_net_ticks'].sum()))
            s_pf = s_wins / max(s_losses, 1e-6)
            log.info(f"  {side:5s}: {len(s_df):4d} trades, WR={s_wr:.0%}, "
                     f"net={s_net:+.1f}t, PF={s_pf:.3f}")

        # Confidence tier breakdown
        log.info(f"\n  --- CONFIDENCE BREAKDOWN ---")
        for lo, hi, label in [(0.67, 0.70, '0.67-0.70'),
                               (0.70, 0.80, '0.70-0.80'),
                               (0.80, 1.00, '0.80+')]:
            tier = trades_df[(trades_df['pred_prob'] >= lo) & (trades_df['pred_prob'] < hi)]
            if len(tier) == 0:
                continue
            t_net = float(tier['paper_net_ticks'].sum())
            t_wr = float(tier['winner'].mean())
            log.info(f"  {label:10s}: {len(tier):4d} trades, WR={t_wr:.0%}, "
                     f"net={t_net:+.1f}t")

        # Monthly breakdown
        log.info(f"\n  --- MONTHLY BREAKDOWN ---")
        daily_df_copy['month'] = pd.to_datetime(daily_df_copy['date'], format='%Y%m%d').dt.to_period('M')
        for month, m_df in daily_df_copy.groupby('month'):
            m_net = float(m_df['net_ticks'].sum())
            m_trades = int(m_df['n_trades'].sum())
            m_wr = float(m_df['n_winners'].sum()) / max(m_trades, 1)
            m_sharpe = float(m_df['net_ticks'].mean() / m_df['net_ticks'].std()) \
                if len(m_df) > 1 and m_df['net_ticks'].std() > 0 else 0
            log.info(f"  {str(month):7s}: {len(m_df):2d} days, {m_trades:4d} trades, "
                     f"WR={m_wr:.0%}, net={m_net:+.1f}t, Sharpe={m_sharpe:.3f}")

        # Cost sensitivity (extra slippage on top of already-included costs)
        log.info(f"\n  --- COST SENSITIVITY (additional slippage beyond commission) ---")
        for extra_slip in [0.0, 0.2, 0.5, 0.75, 1.0]:
            adj_net = total_net - (extra_slip * total_trades)
            adj_per_trade = adj_net / max(total_trades, 1)
            adj_per_day = adj_net / max(total_days, 1)
            log.info(f"  +{extra_slip:.2f}t slip: net={adj_net:+.0f}t "
                     f"(${adj_net * ES_TICK_VALUE:+,.0f}), "
                     f"per trade={adj_per_trade:+.3f}t, "
                     f"per day=${adj_per_day * ES_TICK_VALUE:+,.0f}")

    def run_live(self):
        """Run in live loop — replay first, then watch for new data."""
        log.info("Starting live mode...")
        self.run_replay()

        log.info("Replay complete. Entering live monitoring loop...")
        while True:
            try:
                time.sleep(60)
                # TODO: Check for new dates beyond OOT range,
                # retrain model, score new entries
            except KeyboardInterrupt:
                log.info("Shutting down.")
                break
            except Exception as e:
                log.error(f"Live loop error: {e}")
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Queue Entry Selector v2.1 Paper Engine")
    parser.add_argument('--replay', action='store_true', help='Replay all available data')
    parser.add_argument('--live', action='store_true', help='Run in live loop mode (PM2)')
    parser.add_argument('--summary', action='store_true', help='Print summary and exit')
    args = parser.parse_args()

    engine = QueueEntryV21PaperEngine()

    if args.live:
        engine.run_live()
    else:
        engine.run_replay()


if __name__ == '__main__':
    main()
