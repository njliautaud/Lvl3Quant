#!/usr/bin/env python3
"""
Leveraged Backtest: Top 3 Validated Strategies at 1x/2x/3x
============================================================
HC #800: Leverage explicitly approved (2x-3x) to beat SPY CAGR (~19%).

Tests our three lockbox-validated strategies at multiple leverage levels:
  1) Macro Regime Rotation (AVO v19, score 8.00, lockbox Sharpe 4.39)
  2) Flow Reversal (AVO v25, score 4.36, lockbox Sharpe 3.37)
  3) Sector Rotation WF (AVO v24, score 6.50, lockbox Sharpe 2.87)

Methodology:
  - Uses yfinance for real market data over Jan-Aug 2026 lockbox period
  - Each strategy's signal logic is implemented directly (not importing AVO modules)
  - Leverage = position size multiplier: 2x = $20K deployed on $10K capital
  - Margin cost: 6% annualized on the borrowed portion
  - Proper daily mark-to-market with leveraged P&L
  - Reports: CAGR, Sharpe, Sortino, Max Drawdown, PF, WR for each leverage level

Period: 2026-01-02 to 2026-08-22 (same as lockbox validation)
Capital: $10,000
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import sys
import time

# ─── CONFIG ─────────────────────────────────────────────────────────────────
CAPITAL = 10000.0
OOS_START = '2026-01-02'
OOS_END = '2026-08-22'
WARMUP_START = '2025-01-01'
LEVERAGE_LEVELS = [1.0, 2.0, 3.0]
MARGIN_RATE = 0.06  # 6% annualized borrowing cost
SLIPPAGE_PCT = 0.0001  # 1 bps slippage per side

# Strategy-specific configs
MACRO_REGIME = {
    'name': 'Macro Regime Rotation',
    'tradeable': ['XLU', 'XLP', 'XLV', 'XLRE'],
    'max_concurrent': 1,
    'max_per_trade_pct': 0.20,  # 20% of capital per trade at 1x
    'max_hold_days': 3,
    'take_profit': 0.035,
    'trailing_stop': -0.02,
    'dip_threshold': -0.015,
    'dip_threshold_highvol': -0.024,
    'dip_threshold_spystrong': -0.023,
    'trend_lookback': 30,
    'trend_min': -0.10,
}

FLOW_REVERSAL = {
    'name': 'Flow Reversal',
    'universe': ['ARKK', 'IGV', 'ITB', 'KRE', 'KWEB', 'OIH', 'SMH',
                 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
                 'XLRE', 'XLU', 'XLV', 'XLY'],
    'max_concurrent': 2,
    'max_per_trade_pct': 0.20,
    'max_hold_days': 3,
    'take_profit': 0.035,
    'trailing_stop': -0.02,
    'underwater_cut_days': 1,
    'underwater_tolerance': -0.004,
    'volume_lookback': 20,
    'volume_z_threshold': 2.0,
    'price_dip_threshold': -0.01,
}

SECTOR_ROTATION = {
    'name': 'Sector Rotation WF',
    'tradeable': ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
                  'XLRE', 'XLU', 'XLV', 'XLY'],
    'max_concurrent': 1,
    'max_per_trade_pct': 0.20,
    'max_hold_days': 3,
    'take_profit': 0.05,
    'trailing_stop': -0.008,
    'underwater_cut_days': 1,
    'underwater_cut_tol': -0.008,
    'min_composite_score': 0.28,
}


# ─── DATA DOWNLOAD ──────────────────────────────────────────────────────────
def download_data():
    """Download all needed tickers."""
    all_tickers = list(set(
        MACRO_REGIME['tradeable'] +
        FLOW_REVERSAL['universe'] +
        SECTOR_ROTATION['tradeable'] +
        ['SPY', '^VIX']
    ))
    print(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start=WARMUP_START, end='2026-08-23',
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        volume = raw['Volume']
        high = raw['High']
        low = raw['Low']
    else:
        close = raw[['Close']]
        volume = raw[['Volume']]
        high = raw[['High']]
        low = raw[['Low']]

    # Flatten MultiIndex if needed
    for df in [close, volume, high, low]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    close = close.ffill().dropna(how='all')
    volume = volume.ffill().fillna(0)
    high = high.ffill().dropna(how='all')
    low = low.ffill().dropna(how='all')

    # Extract VIX
    vix_cols = [c for c in close.columns if 'VIX' in str(c).upper()]
    if vix_cols:
        vix = close[vix_cols[0]].copy()
        close = close.drop(columns=vix_cols)
        volume = volume.drop(columns=[c for c in volume.columns if 'VIX' in str(c).upper()], errors='ignore')
        high = high.drop(columns=[c for c in high.columns if 'VIX' in str(c).upper()], errors='ignore')
        low = low.drop(columns=[c for c in low.columns if 'VIX' in str(c).upper()], errors='ignore')
    else:
        vix = pd.Series(20.0, index=close.index)

    print(f"Data range: {close.index.min().date()} to {close.index.max().date()}")
    print(f"Trading days: {len(close)}")
    return close, volume, high, low, vix


# ─── SIGNAL GENERATORS ──────────────────────────────────────────────────────

def macro_regime_signals(close, vix, spy):
    """
    Macro Regime Rotation: buy defensive sectors on dips.
    Adaptive dip thresholds based on VIX regime and SPY momentum.
    """
    cfg = MACRO_REGIME
    signals = []
    tradeable = [t for t in cfg['tradeable'] if t in close.columns]
    oos_dates = close.index[(close.index >= OOS_START) & (close.index <= OOS_END)]

    spy_mom_20 = spy.pct_change(20)

    for date in oos_dates:
        idx = close.index.get_loc(date)
        if idx < cfg['trend_lookback']:
            continue

        vix_val = vix.get(date, 20.0)
        if pd.isna(vix_val):
            vix_val = 20.0

        # Determine dip threshold based on regime
        spy_mom = spy_mom_20.iloc[idx] if idx < len(spy_mom_20) else 0
        if vix_val > 25:
            dip_thresh = cfg['dip_threshold_highvol']
        elif not pd.isna(spy_mom) and spy_mom > 0.03:
            dip_thresh = cfg['dip_threshold_spystrong']
        else:
            dip_thresh = cfg['dip_threshold']

        for etf in tradeable:
            if etf not in close.columns:
                continue

            # Check trend: not in deep downtrend
            trend_ret = (close[etf].iloc[idx] - close[etf].iloc[idx - cfg['trend_lookback']]) / close[etf].iloc[idx - cfg['trend_lookback']]
            if trend_ret < cfg['trend_min']:
                continue

            # Check for dip (5-day return below threshold)
            if idx < 5:
                continue
            dip_5d = (close[etf].iloc[idx] - close[etf].iloc[idx - 5]) / close[etf].iloc[idx - 5]
            if dip_5d <= dip_thresh:
                signals.append({'date': date, 'ticker': etf, 'score': abs(dip_5d)})

    return signals


def flow_reversal_signals(close, volume, vix):
    """
    Flow Reversal: buy ETFs showing volume surge + price dip (flow reversal pattern).
    Volume z-score > threshold + price dip = accumulation signal.
    """
    cfg = FLOW_REVERSAL
    signals = []
    universe = [t for t in cfg['universe'] if t in close.columns and t in volume.columns]
    oos_dates = close.index[(close.index >= OOS_START) & (close.index <= OOS_END)]

    for etf in universe:
        etf_close = close[etf]
        etf_vol = volume[etf]

        # Precompute rolling stats
        vol_mean = etf_vol.rolling(cfg['volume_lookback']).mean()
        vol_std = etf_vol.rolling(cfg['volume_lookback']).std()

        for date in oos_dates:
            idx = close.index.get_loc(date)
            if idx < cfg['volume_lookback'] + 5:
                continue

            # Volume z-score
            vm = vol_mean.iloc[idx]
            vs = vol_std.iloc[idx]
            if pd.isna(vm) or pd.isna(vs) or vs == 0:
                continue
            vol_z = (etf_vol.iloc[idx] - vm) / vs

            # Price dip (1d return)
            ret_1d = (etf_close.iloc[idx] - etf_close.iloc[idx - 1]) / etf_close.iloc[idx - 1]

            # Signal: high volume + price dip
            if vol_z >= cfg['volume_z_threshold'] and ret_1d <= cfg['price_dip_threshold']:
                signals.append({'date': date, 'ticker': etf, 'score': vol_z * abs(ret_1d)})

    return signals


def sector_rotation_signals(close, volume, high, low, vix, spy):
    """
    Sector Rotation WF: composite scoring across momentum, flow, relative strength,
    and mean-reversion indicators. Buy when composite score > threshold.
    """
    cfg = SECTOR_ROTATION
    signals = []
    tradeable = [t for t in cfg['tradeable'] if t in close.columns]
    oos_dates = close.index[(close.index >= OOS_START) & (close.index <= OOS_END)]

    spy_close = spy

    for date in oos_dates:
        idx = close.index.get_loc(date)
        if idx < 63:  # need 63d lookback
            continue

        scores = {}
        for etf in tradeable:
            score = 0.0
            n_signals = 0

            # 1. Relative strength (21d return vs SPY)
            etf_ret_21 = (close[etf].iloc[idx] - close[etf].iloc[idx - 21]) / close[etf].iloc[idx - 21]
            spy_ret_21 = (spy_close.iloc[idx] - spy_close.iloc[idx - 21]) / spy_close.iloc[idx - 21]
            rel_str = etf_ret_21 - spy_ret_21

            # 2. MFI (14-day simplified)
            if etf in high.columns and etf in low.columns and etf in volume.columns:
                tp = (high[etf].iloc[idx-14:idx+1] + low[etf].iloc[idx-14:idx+1] + close[etf].iloc[idx-14:idx+1]) / 3
                raw_mf = tp * volume[etf].iloc[idx-14:idx+1]
                tp_diff = tp.diff()
                pos_mf = raw_mf[tp_diff > 0].sum()
                neg_mf = raw_mf[tp_diff <= 0].sum()
                mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
            else:
                mfi = 50

            # 3. Drawdown from 63d high
            rolling_max_63 = close[etf].iloc[max(0, idx-63):idx+1].max()
            dd_63 = (close[etf].iloc[idx] - rolling_max_63) / rolling_max_63

            # 4. 5d momentum
            ret_5d = (close[etf].iloc[idx] - close[etf].iloc[idx - 5]) / close[etf].iloc[idx - 5]

            # Composite score (normalized)
            # Positive factors: relative strength, MFI (above 50), dip bounce
            # Mean-reversion component: buy dips that show flow support
            if rel_str > 0:
                score += 0.3
                n_signals += 1
            if mfi > 60:
                score += 0.25
                n_signals += 1
            if dd_63 < -0.03 and ret_5d > 0:  # bouncing from dip
                score += 0.25
                n_signals += 1
            if mfi > 50 and dd_63 < -0.02:  # flow support during dip
                score += 0.2
                n_signals += 1

            if n_signals > 0:
                scores[etf] = score

        # Signal the top-scoring ETF if it exceeds threshold
        if scores:
            best_etf = max(scores, key=scores.get)
            if scores[best_etf] >= cfg['min_composite_score']:
                signals.append({'date': date, 'ticker': best_etf, 'score': scores[best_etf]})

    return signals


# ─── TRADE SIMULATOR ────────────────────────────────────────────────────────

def simulate_trades(signals, close, strategy_cfg, leverage=1.0):
    """
    Simulate trades with given leverage level.

    Leverage mechanics:
      - 1x: deploy up to MAX_PER_TRADE from capital
      - 2x: deploy 2 * MAX_PER_TRADE (borrow the extra from margin)
      - 3x: deploy 3 * MAX_PER_TRADE (borrow 2x from margin)

    Margin cost: MARGIN_RATE * (leverage - 1) * position_value / 252 per day

    P&L is applied to the FULL leveraged position, but equity starts at CAPITAL.
    """
    max_concurrent = strategy_cfg['max_concurrent']
    max_hold = strategy_cfg['max_hold_days']
    tp_pct = strategy_cfg['take_profit']
    ts_pct = strategy_cfg['trailing_stop']

    oos_dates = close.index[(close.index >= OOS_START) & (close.index <= OOS_END)]
    if len(oos_dates) == 0:
        return [], pd.Series(dtype=float), pd.Series(dtype=float)

    # Build signal lookup
    signal_lookup = {}
    for s in signals:
        d = pd.Timestamp(s['date'])
        if d not in signal_lookup:
            signal_lookup[d] = []
        signal_lookup[d].append((s['ticker'], s['score']))

    capital = CAPITAL
    equity = CAPITAL
    open_positions = []
    trades = []
    daily_pnl = pd.Series(0.0, index=oos_dates)
    equity_curve = pd.Series(CAPITAL, index=oos_dates)
    peak_equity = CAPITAL
    pending_entries = []

    for i, date in enumerate(oos_dates):
        day_pnl = 0.0

        # --- Process exits ---
        new_open = []
        for pos in open_positions:
            ticker = pos['ticker']
            if ticker not in close.columns:
                new_open.append(pos)
                continue

            curr_price = close.loc[date, ticker]
            if pd.isna(curr_price):
                new_open.append(pos)
                continue

            # Update high water mark
            if curr_price > pos['hwm']:
                pos['hwm'] = curr_price

            entry_p = pos['entry_price']
            pnl_pct = (curr_price - entry_p) / entry_p
            days_held = int(np.busday_count(
                np.datetime64(pos['entry_date'], 'D'),
                np.datetime64(date, 'D')))
            dd_from_hwm = (curr_price - pos['hwm']) / pos['hwm']

            # Exit conditions
            do_exit = False
            exit_reason = ''

            if pnl_pct >= tp_pct:
                do_exit = True
                exit_reason = 'take_profit'
            elif days_held >= max_hold:
                do_exit = True
                exit_reason = 'max_hold'
            elif dd_from_hwm <= ts_pct:
                do_exit = True
                exit_reason = 'trailing_stop'
            # Underwater cut (flow reversal and sector rotation)
            elif ('underwater_cut_days' in strategy_cfg and
                  days_held >= strategy_cfg.get('underwater_cut_days', 99) and
                  pnl_pct < strategy_cfg.get('underwater_tolerance',
                                              strategy_cfg.get('underwater_cut_tol', -0.01))):
                do_exit = True
                exit_reason = 'underwater_cut'

            if do_exit:
                exit_price = curr_price * (1.0 - SLIPPAGE_PCT)
                # Leveraged P&L: shares * (exit - entry)
                trade_pnl = (exit_price - entry_p) * pos['shares']
                # Margin cost for days held
                margin_cost = (pos['position_value'] * (leverage - 1) *
                              MARGIN_RATE / 252 * max(days_held, 1))
                trade_pnl -= margin_cost

                day_pnl += trade_pnl
                capital += pos['capital_deployed'] + trade_pnl

                trades.append({
                    'ticker': ticker,
                    'entry_date': str(pos['entry_date'])[:10],
                    'exit_date': str(date)[:10],
                    'entry_price': round(float(entry_p), 4),
                    'exit_price': round(float(exit_price), 4),
                    'shares': pos['shares'],
                    'leverage': leverage,
                    'position_value': round(float(pos['position_value']), 2),
                    'capital_deployed': round(float(pos['capital_deployed']), 2),
                    'pnl': round(float(trade_pnl), 2),
                    'margin_cost': round(float(margin_cost), 2),
                    'return_pct': round(float(trade_pnl / pos['capital_deployed'] * 100), 4),
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                })
            else:
                # Mark-to-market
                if i > 0:
                    prev_date = oos_dates[i - 1]
                    prev_price = close.loc[prev_date, ticker]
                    if not pd.isna(prev_price):
                        mtm = (curr_price - prev_price) * pos['shares']
                        # Daily margin cost
                        daily_margin = (pos['position_value'] * (leverage - 1) *
                                       MARGIN_RATE / 252)
                        day_pnl += mtm - daily_margin
                new_open.append(pos)

        open_positions = new_open

        # --- Process pending entries (next-day execution) ---
        for ticker, score in pending_entries:
            if len(open_positions) >= max_concurrent:
                break
            if any(p['ticker'] == ticker for p in open_positions):
                continue
            if ticker not in close.columns:
                continue

            price = close.loc[date, ticker]
            if pd.isna(price) or price <= 0:
                continue

            # Position sizing: at 1x, use max_per_trade_pct of capital
            # At Nx leverage, deploy N * that amount
            base_size = CAPITAL * strategy_cfg['max_per_trade_pct']
            leveraged_size = base_size * leverage
            capital_needed = base_size  # Only lock up 1x worth of capital

            if capital < capital_needed * 0.5:
                continue

            entry_price = price * (1.0 + SLIPPAGE_PCT)
            shares = int(leveraged_size / entry_price)
            if shares < 1:
                continue

            position_value = shares * entry_price
            actual_capital = min(capital_needed, capital * 0.95)
            capital -= actual_capital

            open_positions.append({
                'ticker': ticker,
                'entry_date': date,
                'entry_price': entry_price,
                'hwm': entry_price,
                'shares': shares,
                'position_value': position_value,
                'capital_deployed': actual_capital,
            })

        pending_entries = []

        # --- Check for new signals ---
        if date in signal_lookup and len(open_positions) < max_concurrent:
            candidates = sorted(signal_lookup[date], key=lambda x: x[1], reverse=True)
            for ticker, score in candidates:
                if not any(p['ticker'] == ticker for p in open_positions):
                    pending_entries.append((ticker, score))

        # --- Update equity ---
        daily_pnl.iloc[i] = day_pnl
        if i > 0:
            equity_curve.iloc[i] = equity_curve.iloc[i - 1] + day_pnl
        else:
            equity_curve.iloc[i] = CAPITAL + day_pnl
        peak_equity = max(peak_equity, equity_curve.iloc[i])

    # Force close remaining
    if open_positions and len(oos_dates) > 0:
        last_date = oos_dates[-1]
        for pos in open_positions:
            ticker = pos['ticker']
            if ticker not in close.columns:
                continue
            price = close.loc[last_date, ticker]
            if pd.isna(price):
                continue
            exit_price = price * (1.0 - SLIPPAGE_PCT)
            trade_pnl = (exit_price - pos['entry_price']) * pos['shares']
            days_held = int(np.busday_count(
                np.datetime64(pos['entry_date'], 'D'),
                np.datetime64(last_date, 'D')))
            margin_cost = (pos['position_value'] * (leverage - 1) *
                          MARGIN_RATE / 252 * max(days_held, 1))
            trade_pnl -= margin_cost

            trades.append({
                'ticker': ticker,
                'entry_date': str(pos['entry_date'])[:10],
                'exit_date': str(last_date)[:10],
                'entry_price': round(float(pos['entry_price']), 4),
                'exit_price': round(float(exit_price), 4),
                'shares': pos['shares'],
                'leverage': leverage,
                'position_value': round(float(pos['position_value']), 2),
                'capital_deployed': round(float(pos['capital_deployed']), 2),
                'pnl': round(float(trade_pnl), 2),
                'margin_cost': round(float(margin_cost), 2),
                'return_pct': round(float(trade_pnl / pos['capital_deployed'] * 100), 4),
                'days_held': days_held,
                'exit_reason': 'force_close',
            })

    return trades, daily_pnl, equity_curve


# ─── METRICS ────────────────────────────────────────────────────────────────

def compute_metrics(trades, daily_pnl, equity_curve, leverage):
    """Compute comprehensive performance metrics."""
    n = len(trades)
    if n == 0:
        return {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'sharpe': 0, 'sortino': 0, 'cagr': 0,
            'total_return_pct': 0, 'max_drawdown_pct': 0,
        }

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    total_margin_cost = sum(t.get('margin_cost', 0) for t in trades)

    win_rate = len(wins) / n
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    downside = daily_ret[daily_ret < 0]
    ds_std = downside.std() if len(downside) > 5 else std_ret
    sortino = float(mean_ret / ds_std * np.sqrt(252)) if ds_std > 0 else 0

    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    max_dd = float(dd.min())

    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity - CAPITAL) / CAPITAL

    # CAGR: annualize the partial-year return
    n_days = len(equity_curve)
    years = n_days / 252.0
    if years > 0 and final_equity > 0:
        cagr = (final_equity / CAPITAL) ** (1.0 / years) - 1
    else:
        cagr = 0

    avg_days_held = np.mean([t['days_held'] for t in trades]) if trades else 0

    return {
        'leverage': leverage,
        'total_trades': n,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(float(win_rate), 4),
        'profit_factor': round(float(profit_factor), 3),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'cagr': round(float(cagr), 4),
        'cagr_pct': round(float(cagr * 100), 2),
        'total_return_pct': round(float(total_return * 100), 2),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
        'final_equity': round(float(final_equity), 2),
        'total_pnl': round(float(total_pnl), 2),
        'total_margin_cost': round(float(total_margin_cost), 2),
        'avg_days_held': round(float(avg_days_held), 1),
    }


def compute_spy_metrics(close):
    """Compute SPY buy-and-hold metrics over the same period."""
    spy = close['SPY']
    oos_spy = spy[(spy.index >= OOS_START) & (spy.index <= OOS_END)].dropna()

    if len(oos_spy) < 10:
        return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'max_drawdown_pct': 0}

    daily_ret = oos_spy.pct_change().dropna()
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    downside = daily_ret[daily_ret < 0]
    ds_std = downside.std() if len(downside) > 5 else std_ret
    sortino = float(mean_ret / ds_std * np.sqrt(252)) if ds_std > 0 else 0

    cum = (1 + daily_ret).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    total_ret = (oos_spy.iloc[-1] / oos_spy.iloc[0]) - 1
    n_days = len(oos_spy)
    years = n_days / 252.0
    cagr = (1 + total_ret) ** (1.0 / years) - 1 if years > 0 else 0

    return {
        'cagr': round(float(cagr), 4),
        'cagr_pct': round(float(cagr * 100), 2),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
        'total_return_pct': round(float(total_ret * 100), 2),
        'start_price': round(float(oos_spy.iloc[0]), 2),
        'end_price': round(float(oos_spy.iloc[-1]), 2),
        'n_days': n_days,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("LEVERAGED BACKTEST: Top 3 Validated Strategies at 1x / 2x / 3x")
    print("=" * 80)
    print(f"Period: {OOS_START} to {OOS_END}")
    print(f"Capital: ${CAPITAL:,.0f}")
    print(f"Margin rate: {MARGIN_RATE:.0%} annualized")
    print(f"Leverage levels: {LEVERAGE_LEVELS}")
    print()

    # Download data
    close, volume, high, low, vix = download_data()
    spy = close['SPY']

    # SPY benchmark
    spy_metrics = compute_spy_metrics(close)
    print(f"\nSPY BENCHMARK (buy & hold):")
    print(f"  CAGR: {spy_metrics['cagr_pct']:.1f}%")
    print(f"  Sharpe: {spy_metrics['sharpe']:.2f}")
    print(f"  Sortino: {spy_metrics['sortino']:.2f}")
    print(f"  Max DD: {spy_metrics['max_drawdown_pct']:.1f}%")
    print(f"  Total Return: {spy_metrics['total_return_pct']:.1f}%")

    # Generate signals for each strategy
    print("\n" + "=" * 80)
    print("GENERATING SIGNALS")
    print("=" * 80)

    print("\n  Macro Regime Rotation...")
    macro_signals = macro_regime_signals(close, vix, spy)
    print(f"    {len(macro_signals)} signals generated")

    print("  Flow Reversal...")
    flow_signals = flow_reversal_signals(close, volume, vix)
    print(f"    {len(flow_signals)} signals generated")

    print("  Sector Rotation WF...")
    sector_signals = sector_rotation_signals(close, volume, high, low, vix, spy)
    print(f"    {len(sector_signals)} signals generated")

    # Run backtests at each leverage level
    all_results = {}
    strategies = [
        ('Macro Regime Rotation', macro_signals, MACRO_REGIME),
        ('Flow Reversal', flow_signals, FLOW_REVERSAL),
        ('Sector Rotation WF', sector_signals, SECTOR_ROTATION),
    ]

    for strat_name, signals, cfg in strategies:
        print(f"\n{'=' * 80}")
        print(f"STRATEGY: {strat_name}")
        print(f"{'=' * 80}")

        strat_results = {}

        for lev in LEVERAGE_LEVELS:
            print(f"\n  --- {lev:.0f}x LEVERAGE ---")
            trades, dpnl, eq = simulate_trades(signals, close, cfg, leverage=lev)
            metrics = compute_metrics(trades, dpnl, eq, lev)

            print(f"    Trades:       {metrics['total_trades']}")
            print(f"    Win Rate:     {metrics['win_rate']:.1%}")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Sharpe:       {metrics['sharpe']:.2f}")
            print(f"    Sortino:      {metrics['sortino']:.2f}")
            print(f"    CAGR:         {metrics['cagr_pct']:.1f}%")
            print(f"    Total Return: {metrics['total_return_pct']:.1f}%")
            print(f"    Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
            print(f"    Final Equity: ${metrics['final_equity']:,.2f}")
            print(f"    Margin Cost:  ${metrics['total_margin_cost']:.2f}")
            print(f"    Avg Hold:     {metrics['avg_days_held']:.1f}d")

            # Check if it beats SPY
            beats_spy = metrics['cagr'] > spy_metrics['cagr']
            print(f"    Beats SPY:    {'YES' if beats_spy else 'NO'} "
                  f"({metrics['cagr_pct']:.1f}% vs {spy_metrics['cagr_pct']:.1f}%)")

            strat_results[f'{lev:.0f}x'] = metrics

        all_results[strat_name] = strat_results

    # ─── COMBINED PORTFOLIO ──────────────────────────────────────────────────
    # Equal-weight allocation across all 3 strategies (each gets 1/3 of capital)
    print(f"\n{'=' * 80}")
    print("COMBINED PORTFOLIO (Equal-weight 3 strategies)")
    print(f"{'=' * 80}")

    for lev in LEVERAGE_LEVELS:
        print(f"\n  --- COMBINED {lev:.0f}x ---")
        combined_pnl = 0
        combined_trades = 0
        combined_wins = 0
        combined_margin = 0

        for strat_name in ['Macro Regime Rotation', 'Flow Reversal', 'Sector Rotation WF']:
            m = all_results[strat_name][f'{lev:.0f}x']
            combined_pnl += m['total_pnl'] / 3  # 1/3 allocation each
            combined_trades += m['total_trades']
            combined_wins += m['wins']
            combined_margin += m['total_margin_cost'] / 3

        combined_return = combined_pnl / CAPITAL
        combined_wr = combined_wins / max(combined_trades, 1)

        # Approximate combined CAGR
        n_days = len(close.index[(close.index >= OOS_START) & (close.index <= OOS_END)])
        years = n_days / 252.0
        combined_cagr = ((1 + combined_return) ** (1.0 / years) - 1) if years > 0 else 0

        print(f"    Total trades: {combined_trades}")
        print(f"    Win Rate:     {combined_wr:.1%}")
        print(f"    Combined P&L: ${combined_pnl:.2f}")
        print(f"    Combined Return: {combined_return * 100:.1f}%")
        print(f"    Combined CAGR: {combined_cagr * 100:.1f}%")
        print(f"    Margin Cost:  ${combined_margin:.2f}")
        beats = combined_cagr > spy_metrics['cagr']
        print(f"    Beats SPY:    {'YES' if beats else 'NO'} "
              f"({combined_cagr * 100:.1f}% vs {spy_metrics['cagr_pct']:.1f}%)")

    # ─── SUMMARY TABLE ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"\n{'Strategy':<28} {'Lev':>4} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'PF':>6} {'WR':>6} {'Trades':>7} {'Beat SPY':>9}")
    print("-" * 100)

    # SPY row
    print(f"{'SPY (Buy & Hold)':<28} {'1x':>4} {spy_metrics['cagr_pct']:>6.1f}% "
          f"{spy_metrics['sharpe']:>7.2f} {spy_metrics['sortino']:>8.2f} "
          f"{spy_metrics['max_drawdown_pct']:>6.1f}% {'--':>6} {'--':>6} {'--':>7} {'--':>9}")
    print("-" * 100)

    for strat_name in ['Macro Regime Rotation', 'Flow Reversal', 'Sector Rotation WF']:
        for lev in LEVERAGE_LEVELS:
            m = all_results[strat_name][f'{lev:.0f}x']
            beats = 'YES' if m['cagr'] > spy_metrics['cagr'] else 'no'
            lev_str = f"{lev:.0f}x"
            print(f"{strat_name:<28} {lev_str:>4} {m['cagr_pct']:>6.1f}% "
                  f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
                  f"{m['max_drawdown_pct']:>6.1f}% {m['profit_factor']:>6.2f} "
                  f"{m['win_rate']:>5.1%} {m['total_trades']:>7} {beats:>9}")

    # ─── RISK-ADJUSTED ANALYSIS ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("RISK-ADJUSTED LEVERAGE ANALYSIS")
    print(f"{'=' * 80}")

    for strat_name in ['Macro Regime Rotation', 'Flow Reversal', 'Sector Rotation WF']:
        m1 = all_results[strat_name]['1x']
        m2 = all_results[strat_name]['2x']
        m3 = all_results[strat_name]['3x']

        print(f"\n  {strat_name}:")
        print(f"    CAGR scaling:  1x={m1['cagr_pct']:.1f}% -> 2x={m2['cagr_pct']:.1f}% -> 3x={m3['cagr_pct']:.1f}%")
        print(f"    Sharpe scaling: 1x={m1['sharpe']:.2f} -> 2x={m2['sharpe']:.2f} -> 3x={m3['sharpe']:.2f}")
        print(f"    MaxDD scaling: 1x={m1['max_drawdown_pct']:.1f}% -> 2x={m2['max_drawdown_pct']:.1f}% -> 3x={m3['max_drawdown_pct']:.1f}%")

        # Return per unit of drawdown
        if m1['max_drawdown_pct'] != 0:
            calmar_1x = m1['cagr_pct'] / abs(m1['max_drawdown_pct'])
        else:
            calmar_1x = float('inf')
        if m2['max_drawdown_pct'] != 0:
            calmar_2x = m2['cagr_pct'] / abs(m2['max_drawdown_pct'])
        else:
            calmar_2x = float('inf')
        if m3['max_drawdown_pct'] != 0:
            calmar_3x = m3['cagr_pct'] / abs(m3['max_drawdown_pct'])
        else:
            calmar_3x = float('inf')

        print(f"    Calmar ratio: 1x={calmar_1x:.2f} -> 2x={calmar_2x:.2f} -> 3x={calmar_3x:.2f}")

        # Recommendation
        if m2['cagr'] > spy_metrics['cagr'] and m2['max_drawdown_pct'] > -15:
            rec = "2x is the sweet spot: beats SPY with controlled drawdowns"
        elif m3['cagr'] > spy_metrics['cagr'] and m3['max_drawdown_pct'] > -20:
            rec = "3x needed to beat SPY, drawdowns acceptable"
        elif m1['cagr'] > spy_metrics['cagr']:
            rec = "Already beats SPY at 1x, leverage optional"
        else:
            rec = "Leverage alone may not be sufficient to beat SPY"
        print(f"    Recommendation: {rec}")

    # ─── SAVE RESULTS ────────────────────────────────────────────────────────
    output = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'capital': CAPITAL,
            'margin_rate': MARGIN_RATE,
            'slippage_pct': SLIPPAGE_PCT,
            'oos_period': f'{OOS_START} to {OOS_END}',
            'leverage_levels': LEVERAGE_LEVELS,
        },
        'spy_benchmark': spy_metrics,
        'strategies': {},
    }

    for strat_name in ['Macro Regime Rotation', 'Flow Reversal', 'Sector Rotation WF']:
        output['strategies'][strat_name] = {}
        for lev in LEVERAGE_LEVELS:
            key = f'{lev:.0f}x'
            output['strategies'][strat_name][key] = all_results[strat_name][key]

    output_path = '/home/jupiter/Lvl3Quant/strategies/leveraged_backtest_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
