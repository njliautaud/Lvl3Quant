#!/usr/bin/env python3
"""
Tick-Level Replay for 2h LGBM Model
====================================
Validates the 2h LGBM signal (IC=0.655, bar-level Sharpe 13+) at actual
tick resolution using Databento raw MBO data.

Key differences from tick_replay_engine.py (short-horizon):
- Signal: 1 per hour (not every 250 events)
- Hold: 2 hours (not 10-60 seconds)
- Entry: market or passive at hour boundary
- SL: tested at 10/20/40/80 ticks (wide, appropriate for 2h horizon)
- TP: tested at 20/40/80/120 ticks
- Exit: market at 2h mark if neither TP nor SL hit

HC #659 compliance:
- Uses actual tick data, not bar interpolation
- Permutation test mandatory
- If random directions profitable → artifact, reject

Author: Claude (tick-level 2h validation)
Date: 2026-07-11
"""

import numpy as np
import pandas as pd
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0

# Cost models
COST_MARKET_ENTRY_EXIT = 2 * (COMMISSION_RT_TICKS + SPREAD_TICKS)  # Both sides cross = 2.752 ticks
COST_PASSIVE_ENTRY_MARKET_EXIT = COMMISSION_RT_TICKS + (COMMISSION_RT_TICKS + SPREAD_TICKS)  # 1.752 ticks
COST_PASSIVE_BOTH = 2 * COMMISSION_RT_TICKS  # 0.752 ticks

# Default: market entry, market exit (worst case)
DEFAULT_COST = COST_MARKET_ENTRY_EXIT

RAW_MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PREDICTIONS_FILE = Path("/home/jupiter/Lvl3Quant/output/longer_horizon_v2_regime_gate/oot_predictions_2h.parquet")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/tick_replay_2h")

# RTH hours (UTC) - market open 14:30 UTC, close 21:00 UTC
RTH_OPEN_HOUR_UTC = 14   # 9:30 ET = 14:30 UTC (approx)
RTH_CLOSE_HOUR_UTC = 21  # 16:00 ET = 21:00 UTC

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trade2h:
    """A completed 2h trade with tick-level MFE/MAE."""
    trade_id: int
    date: int           # YYYYMMDD
    entry_hour: int     # UTC hour
    direction: str      # 'long' or 'short'
    prediction: float
    entry_price: float
    exit_price: float
    entry_time_ns: int
    exit_time_ns: int
    exit_reason: str    # 'tp', 'sl', 'time_stop', 'eod'
    gross_pnl_ticks: float = 0.0
    cost_ticks: float = 0.0
    net_pnl_ticks: float = 0.0
    mfe_ticks: float = 0.0      # Maximum favorable excursion
    mae_ticks: float = 0.0      # Maximum adverse excursion
    hold_seconds: float = 0.0
    n_ticks_seen: int = 0       # Tick events during hold


# =============================================================================
# Core Engine
# =============================================================================

class TickReplay2h:
    """Replay 2h model predictions against raw MBO tick data."""

    def __init__(self, tp_ticks: Optional[float] = None, sl_ticks: Optional[float] = None,
                 hold_hours: float = 2.0, entry_mode: str = 'market',
                 quantile_threshold: float = 0.20, direction: str = 'both'):
        """
        Args:
            tp_ticks: Take profit in ticks (None = no TP, hold full duration)
            sl_ticks: Stop loss in ticks (None = no SL)
            hold_hours: Max hold time in hours
            entry_mode: 'market' (immediate fill at ask/bid) or 'passive' (wait at bid/ask)
            quantile_threshold: Top N% of predictions to trade (0.10 = top 10%)
            direction: 'long', 'short', or 'both'
        """
        self.tp_ticks = tp_ticks
        self.sl_ticks = sl_ticks
        self.hold_ns = int(hold_hours * 3600 * 1e9)
        self.entry_mode = entry_mode
        self.quantile_threshold = quantile_threshold
        self.direction = direction
        self.trades: List[Trade2h] = []
        self.trade_count = 0

    def load_mbo_for_date(self, date_int: int) -> Optional[pd.DataFrame]:
        """Load raw MBO data for a given date."""
        try:
            import databento as db
        except ImportError:
            print("ERROR: databento package not installed")
            return None

        fname = f"glbx-mdp3-{date_int}.mbo.dbn.zst"
        fpath = RAW_MBO_DIR / fname
        if not fpath.exists():
            return None

        dbn = db.DBNStore.from_file(str(fpath))
        df = dbn.to_df()

        # Find front-month ES contract
        es_symbols = [s for s in df['symbol'].unique()
                      if s.startswith('ES') and '-' not in s]
        if not es_symbols:
            return None

        best_sym = max(es_symbols,
                       key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
        df = df[df['symbol'] == best_sym].copy()
        return df

    def extract_price_at_time(self, df: pd.DataFrame, target_ns: int,
                              direction: str) -> Tuple[float, int]:
        """
        Get the fill price at a given time.
        For market entry:
          - Long: fill at best ask
          - Short: fill at best bid

        Returns (price, actual_fill_time_ns)
        """
        # Find events near the target time
        mask = df.index.astype('int64') >= target_ns
        nearby = df[mask].head(100)

        if len(nearby) == 0:
            return 0.0, 0

        # Build BBO from recent events
        # Look at trades and BBO updates near this time
        pre_mask = df.index.astype('int64') < target_ns
        recent = df[pre_mask].tail(500)

        # Get last trade price as reference
        trades = recent[recent['action'] == 'T']
        if len(trades) > 0:
            last_trade = trades.iloc[-1]['price']
        else:
            last_trade = 0.0

        # For market order, we cross the spread
        # Use first trade after target time as fill price proxy
        post_trades = nearby[nearby['action'] == 'T']
        if len(post_trades) > 0:
            fill_price = post_trades.iloc[0]['price']
            fill_time = post_trades.index[0].value if hasattr(post_trades.index[0], 'value') else int(post_trades.index[0])
            return fill_price, fill_time

        # Fallback: use last known trade + spread adjustment
        if last_trade > 0:
            if direction == 'long':
                return last_trade + TICK_SIZE, target_ns  # Buy at ask
            else:
                return last_trade - TICK_SIZE, target_ns  # Sell at bid

        return 0.0, 0

    def replay_trade(self, df: pd.DataFrame, entry_ns: int, direction: str,
                     prediction: float, date_int: int, hour: int) -> Optional[Trade2h]:
        """
        Replay a single trade from entry to exit using tick data.

        Returns completed Trade2h or None if entry fails.
        """
        # Get entry price
        entry_price, actual_entry_ns = self.extract_price_at_time(df, entry_ns, direction)
        if entry_price <= 0:
            return None

        # Calculate TP/SL prices
        if direction == 'long':
            tp_price = entry_price + self.tp_ticks * TICK_SIZE if self.tp_ticks else None
            sl_price = entry_price - self.sl_ticks * TICK_SIZE if self.sl_ticks else None
        else:
            tp_price = entry_price - self.tp_ticks * TICK_SIZE if self.tp_ticks else None
            sl_price = entry_price + self.sl_ticks * TICK_SIZE if self.sl_ticks else None

        # Exit deadline
        exit_deadline_ns = actual_entry_ns + self.hold_ns

        # Scan tick-by-tick for TP/SL/time stop
        mask = df.index.astype('int64') > actual_entry_ns
        post_entry = df[mask]

        # Track MFE/MAE
        best_price = entry_price
        worst_price = entry_price
        exit_price = entry_price
        exit_time_ns = actual_entry_ns
        exit_reason = 'time_stop'
        n_ticks = 0

        for idx, row in post_entry.iterrows():
            ts = idx.value if hasattr(idx, 'value') else int(idx)

            # Only look at trade events for price movement
            if row['action'] != 'T':
                continue

            price = row['price']
            n_ticks += 1

            # Update MFE/MAE
            if direction == 'long':
                best_price = max(best_price, price)
                worst_price = min(worst_price, price)
            else:
                best_price = min(best_price, price)
                worst_price = max(worst_price, price)

            # Check TP
            if tp_price is not None:
                if (direction == 'long' and price >= tp_price) or \
                   (direction == 'short' and price <= tp_price):
                    exit_price = tp_price
                    exit_time_ns = ts
                    exit_reason = 'tp'
                    break

            # Check SL
            if sl_price is not None:
                if (direction == 'long' and price <= sl_price) or \
                   (direction == 'short' and price >= sl_price):
                    exit_price = price  # SL fills at market (may slip)
                    exit_time_ns = ts
                    exit_reason = 'sl'
                    break

            # Check time stop
            if ts >= exit_deadline_ns:
                exit_price = price
                exit_time_ns = ts
                exit_reason = 'time_stop'
                break
        else:
            # EOD - use last available price
            if len(post_entry) > 0:
                last_trades = post_entry[post_entry['action'] == 'T']
                if len(last_trades) > 0:
                    exit_price = last_trades.iloc[-1]['price']
                    exit_time_ns = last_trades.index[-1].value if hasattr(last_trades.index[-1], 'value') else int(last_trades.index[-1])
                    exit_reason = 'eod'

        # Calculate PnL
        if direction == 'long':
            gross_pnl_ticks = (exit_price - entry_price) / TICK_SIZE
            mfe_ticks = (best_price - entry_price) / TICK_SIZE
            mae_ticks = (entry_price - worst_price) / TICK_SIZE
        else:
            gross_pnl_ticks = (entry_price - exit_price) / TICK_SIZE
            mfe_ticks = (entry_price - best_price) / TICK_SIZE
            mae_ticks = (worst_price - entry_price) / TICK_SIZE

        # Cost depends on exit type
        if exit_reason == 'tp':
            cost = COST_PASSIVE_ENTRY_MARKET_EXIT if self.entry_mode == 'market' else COST_PASSIVE_BOTH
        else:
            cost = DEFAULT_COST if self.entry_mode == 'market' else COST_PASSIVE_ENTRY_MARKET_EXIT

        hold_seconds = (exit_time_ns - actual_entry_ns) / 1e9

        self.trade_count += 1
        return Trade2h(
            trade_id=self.trade_count,
            date=date_int,
            entry_hour=hour,
            direction=direction,
            prediction=prediction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time_ns=actual_entry_ns,
            exit_time_ns=exit_time_ns,
            exit_reason=exit_reason,
            gross_pnl_ticks=gross_pnl_ticks,
            cost_ticks=cost,
            net_pnl_ticks=gross_pnl_ticks - cost,
            mfe_ticks=mfe_ticks,
            mae_ticks=mae_ticks,
            hold_seconds=hold_seconds,
            n_ticks_seen=n_ticks
        )

    def run(self, predictions_df: pd.DataFrame, max_days: int = 999) -> List[Trade2h]:
        """
        Run tick-level replay on all predictions above threshold.

        Args:
            predictions_df: DataFrame with columns [date, hour, prediction, actual]
            max_days: Limit days processed (for testing)
        """
        self.trades = []
        self.trade_count = 0

        # Determine threshold from quantile
        abs_preds = predictions_df['prediction'].abs()
        threshold = abs_preds.quantile(1.0 - self.quantile_threshold)

        # Filter predictions above threshold
        if self.direction == 'long':
            signal_mask = predictions_df['prediction'] >= threshold
        elif self.direction == 'short':
            signal_mask = predictions_df['prediction'] <= -threshold
        else:  # both
            signal_mask = abs_preds >= threshold

        active_preds = predictions_df[signal_mask].copy()
        print(f"  Threshold: {threshold:.3f} ({self.quantile_threshold*100:.0f}% quantile)")
        print(f"  Active predictions: {len(active_preds)} / {len(predictions_df)}")

        dates = active_preds['date'].unique()
        n_days = min(len(dates), max_days)

        for i, date_int in enumerate(sorted(dates)[:max_days]):
            day_preds = active_preds[active_preds['date'] == date_int]

            # Load tick data for this day
            df = self.load_mbo_for_date(date_int)
            if df is None:
                continue

            for _, row in day_preds.iterrows():
                hour_utc = int(row['hour'])
                pred_val = row['prediction']

                # Determine direction
                if self.direction == 'long' and pred_val < 0:
                    continue
                elif self.direction == 'short' and pred_val > 0:
                    continue

                direction = 'long' if pred_val > 0 else 'short'

                # Entry time: start of the predicted hour (in nanoseconds)
                # Convert YYYYMMDD + hour to ns timestamp
                from datetime import datetime, timezone
                dt = datetime(int(str(date_int)[:4]), int(str(date_int)[4:6]),
                              int(str(date_int)[6:8]), hour_utc, 0, 0, tzinfo=timezone.utc)
                entry_ns = int(dt.timestamp() * 1e9)

                trade = self.replay_trade(df, entry_ns, direction, pred_val, date_int, hour_utc)
                if trade is not None:
                    self.trades.append(trade)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{n_days} days, {len(self.trades)} trades so far...")

        return self.trades


def compute_metrics(trades: List[Trade2h]) -> Dict:
    """Compute risk-adjusted metrics from trade list."""
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'pf': 0, 'wr': 0}

    pnls = np.array([t.net_pnl_ticks for t in trades])
    gross_pnls = np.array([t.gross_pnl_ticks for t in trades])

    n = len(pnls)
    total_pnl = pnls.sum()
    mean_pnl = pnls.mean()
    std_pnl = pnls.std() if n > 1 else 1.0
    sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = len(wins) / n if n > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Sortino
    downside = pnls[pnls < 0]
    downside_std = downside.std() if len(downside) > 1 else 1.0
    sortino = mean_pnl / downside_std * np.sqrt(252) if downside_std > 0 else 0.0

    # MFE/MAE stats
    mfes = np.array([t.mfe_ticks for t in trades])
    maes = np.array([t.mae_ticks for t in trades])

    # Exit reason distribution
    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1

    return {
        'n_trades': n,
        'total_pnl_ticks': float(total_pnl),
        'total_pnl_dollars': float(total_pnl * TICK_VALUE),
        'mean_pnl_ticks': float(mean_pnl),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'pf': float(pf),
        'wr': float(wr),
        'avg_mfe': float(mfes.mean()),
        'avg_mae': float(maes.mean()),
        'p90_mfe': float(np.percentile(mfes, 90)),
        'p90_mae': float(np.percentile(maes, 90)),
        'avg_hold_seconds': float(np.mean([t.hold_seconds for t in trades])),
        'exit_reasons': dict(reasons),
        'n_days': len(set(t.date for t in trades))
    }


def permutation_test(trades: List[Trade2h], n_perms: int = 100) -> Dict:
    """
    HC #659: Random direction permutation test.
    If random directions are also profitable → artifact.
    """
    if not trades:
        return {'p_value': 1.0, 'verdict': 'NO TRADES'}

    real_pnls = np.array([t.net_pnl_ticks for t in trades])
    real_sharpe = real_pnls.mean() / real_pnls.std() * np.sqrt(252) if real_pnls.std() > 0 else 0.0

    # For permutation: flip directions randomly
    gross_pnls = np.array([t.gross_pnl_ticks for t in trades])
    costs = np.array([t.cost_ticks for t in trades])

    random_sharpes = []
    for _ in range(n_perms):
        # Random sign flip (simulates random direction)
        signs = np.random.choice([-1, 1], size=len(gross_pnls))
        random_net = gross_pnls * signs - costs
        if random_net.std() > 0:
            random_sharpes.append(random_net.mean() / random_net.std() * np.sqrt(252))
        else:
            random_sharpes.append(0.0)

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= real_sharpe).mean()

    return {
        'real_sharpe': float(real_sharpe),
        'random_mean_sharpe': float(random_sharpes.mean()),
        'random_max_sharpe': float(random_sharpes.max()),
        'p_value': float(p_value),
        'n_perms': n_perms,
        'verdict': 'PASS — genuine edge' if p_value < 0.05 else 'FAIL — artifact'
    }


def run_config(tp, sl, quantile, direction, hold_hours=2.0, max_days=999):
    """Run a single config and return results."""
    preds = pd.read_parquet(PREDICTIONS_FILE)

    engine = TickReplay2h(
        tp_ticks=tp, sl_ticks=sl,
        hold_hours=hold_hours,
        quantile_threshold=quantile,
        direction=direction
    )

    print(f"\nConfig: TP={tp} SL={sl} Q={quantile} dir={direction} hold={hold_hours}h")
    trades = engine.run(preds, max_days=max_days)
    metrics = compute_metrics(trades)
    perm = permutation_test(trades, n_perms=100)

    return {
        'config': {
            'tp_ticks': tp, 'sl_ticks': sl,
            'quantile': quantile, 'direction': direction,
            'hold_hours': hold_hours
        },
        'metrics': metrics,
        'permutation_test': perm
    }


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("2h LGBM TICK-LEVEL REPLAY (HC #659 COMPLIANT)")
    print("=" * 70)

    # Load predictions
    preds = pd.read_parquet(PREDICTIONS_FILE)
    print(f"Predictions: {len(preds)} rows, {preds['date'].nunique()} days")
    print(f"Date range: {preds['date'].min()} to {preds['date'].max()}")
    print()

    # Run sweep of configs
    configs = [
        # (TP, SL, quantile, direction)
        # No TP/SL (raw signal, hold full 2h)
        (None, None, 0.10, 'both'),
        (None, None, 0.20, 'both'),
        (None, None, 0.30, 'both'),
        # With SL protection (wide stops for 2h horizon)
        (None, 40, 0.10, 'both'),
        (None, 40, 0.20, 'both'),
        (80, 40, 0.10, 'both'),
        (80, 40, 0.20, 'both'),
        (120, 60, 0.10, 'both'),
        # Direction-specific
        (None, None, 0.10, 'short'),
        (None, None, 0.10, 'long'),
        (None, 40, 0.10, 'short'),
        (None, 40, 0.10, 'long'),
    ]

    all_results = []
    for tp, sl, q, d in configs:
        try:
            result = run_config(tp, sl, q, d, max_days=999)
            all_results.append(result)
            m = result['metrics']
            p = result['permutation_test']
            print(f"  → Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} "
                  f"PF={m['pf']:.2f} WR={m['wr']:.1%} N={m['n_trades']} "
                  f"Perm p={p['p_value']:.3f} [{p['verdict']}]")
        except Exception as e:
            print(f"  → ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = OUTPUT_DIR / 'tick_replay_2h_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'Sharpe':>7} {'PF':>6} {'WR':>5} {'N':>5} {'Perm':>6}")
    print("-" * 70)
    for r in all_results:
        c = r['config']
        m = r['metrics']
        p = r['permutation_test']
        label = f"TP={c['tp_ticks']} SL={c['sl_ticks']} Q={c['quantile']} {c['direction']}"
        perm_str = f"p={p['p_value']:.2f}"
        print(f"{label:<35} {m['sharpe']:>7.2f} {m['pf']:>6.2f} {m['wr']:>5.1%} {m['n_trades']:>5} {perm_str:>6}")
