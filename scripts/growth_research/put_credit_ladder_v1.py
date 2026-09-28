#!/usr/bin/env python3
"""
Put Credit Spread Ladder v1
============================
Income strategy for a $645 options account. Sells put credit spreads at
different strike levels on sector ETFs, creating a "ladder" of overlapping
expiries for smoother income.

Sector selection: rank by 3-month momentum, sell puts on strong sectors.
Option pricing: Black-Scholes with 1.2× vol multiplier + 10% haircut.

8 variants tested with 5-gate mandatory validation.
"""

import sys
import os
import json
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import norm
from itertools import combinations

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ─── LOGGING ─────────────────────────────────────────────────────────────────

LOG_PATH = '/home/jupiter/Lvl3Quant/logs/put_credit_ladder_v1.log'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/put_credit_ladder_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARKS = ['SPY', '^VIX']
START_DATE = '2020-01-01'
END_DATE = '2026-07-01'
RISK_FREE_RATE = 0.045
STARTING_CAPITAL = 645.0
COMMISSION_PER_CONTRACT = 0.65  # per contract per leg
MAX_CONCURRENT = 3

# ─── BLACK-SCHOLES ───────────────────────────────────────────────────────────

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_iv(prices, window=20, multiplier=1.2):
    """Estimate implied vol from realized vol with multiplier."""
    log_rets = np.log(prices / prices.shift(1))
    realized_vol = log_rets.rolling(window).std() * np.sqrt(252)
    return realized_vol * multiplier


def price_put_credit_spread(S, short_strike, long_strike, T, r, sigma, haircut=0.10):
    """Price a put credit spread (sell short_strike, buy long_strike).
    Returns net credit received after haircut."""
    short_put = bs_put_price(S, short_strike, T, r, sigma)
    long_put = bs_put_price(S, long_strike, T, r, sigma)
    gross_credit = short_put - long_put
    if gross_credit <= 0:
        return 0.0
    return gross_credit * (1.0 - haircut)  # bid-ask haircut


def spread_pnl_at_expiry(S_expiry, short_strike, long_strike, credit_received):
    """P&L of put credit spread at expiry (per share, not per contract)."""
    # Max profit = credit received (if S > short_strike)
    # Max loss = width - credit (if S < long_strike)
    if S_expiry >= short_strike:
        return credit_received
    elif S_expiry <= long_strike:
        return credit_received - (short_strike - long_strike)
    else:
        # Between strikes: partial loss
        return credit_received - (short_strike - S_expiry)


def spread_value_at_time(S, short_strike, long_strike, T_remaining, r, sigma):
    """Mark-to-market value of the spread (cost to close)."""
    if T_remaining <= 0:
        # At expiry
        if S >= short_strike:
            return 0.0
        elif S <= long_strike:
            return short_strike - long_strike
        else:
            return short_strike - S
    short_put = bs_put_price(S, short_strike, T_remaining, r, sigma)
    long_put = bs_put_price(S, long_strike, T_remaining, r, sigma)
    return max(short_put - long_put, 0.0)


# ─── DATA ────────────────────────────────────────────────────────────────────

def fetch_data():
    """Download sector ETF + SPY + VIX data via yfinance."""
    import yfinance as yf

    cache_path = os.path.join(OUTPUT_DIR, 'data_cache.parquet')
    if os.path.exists(cache_path):
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).days
        if age < 1:
            log.info("Using cached data")
            return pd.read_parquet(cache_path)

    tickers = SECTOR_ETFS + BENCHMARKS
    log.info(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}")

    all_frames = {}
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if len(df) > 60:
                all_frames[t] = df['Close'].rename(t)
                log.info(f"  {t}: {len(df)} days")
        except Exception as e:
            log.warning(f"  {t} failed: {e}")

    prices = pd.DataFrame(all_frames)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices.to_parquet(cache_path)
    log.info(f"Data: {prices.shape[0]} days, {prices.shape[1]} tickers")
    return prices


# ─── SECTOR RANKING ──────────────────────────────────────────────────────────

def rank_sectors_by_momentum(prices, date, lookback=63):
    """Rank sector ETFs by 3-month (63 trading day) momentum."""
    idx = prices.index.get_indexer([date], method='ffill')[0]
    if idx < lookback:
        return []
    window = prices.iloc[idx - lookback:idx + 1]
    mom = {}
    for etf in SECTOR_ETFS:
        if etf in window.columns and not window[etf].isna().any():
            mom[etf] = window[etf].iloc[-1] / window[etf].iloc[0] - 1.0
    ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────

class Position:
    def __init__(self, ticker, entry_date, expiry_date, short_strike, long_strike,
                 credit_received, num_contracts, entry_price):
        self.ticker = ticker
        self.entry_date = entry_date
        self.expiry_date = expiry_date
        self.short_strike = short_strike
        self.long_strike = long_strike
        self.width = short_strike - long_strike
        self.credit_received = credit_received  # per share
        self.num_contracts = num_contracts
        self.entry_price = entry_price  # underlying price at entry
        self.max_risk = (self.width - credit_received) * 100 * num_contracts
        self.total_credit = credit_received * 100 * num_contracts
        self.closed = False
        self.close_date = None
        self.pnl = 0.0
        self.close_reason = ''


class PutCreditLadderBacktest:
    def __init__(self, variant_name, params):
        self.name = variant_name
        self.p = params
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.capital = STARTING_CAPITAL
        self.capital_in_use = 0.0

    def run(self, prices):
        """Run backtest over price data."""
        sector_prices = prices[[c for c in SECTOR_ETFS if c in prices.columns]]
        vix = prices['^VIX'] if '^VIX' in prices.columns else None

        # Compute IV estimates for each sector
        iv_estimates = {}
        for etf in SECTOR_ETFS:
            if etf in prices.columns:
                iv_estimates[etf] = estimate_iv(prices[etf])

        dates = prices.index[70:]  # need lookback
        month_entry_tracker = {}  # track staggered entries

        for i, date in enumerate(dates):
            current_vix = vix.loc[date] if vix is not None and date in vix.index else 20.0

            # ── VIX FILTER (variant E) ──
            if self.p.get('vix_filter', False):
                if current_vix < self.p.get('vix_entry_min', 18):
                    self._mark_to_market(date, prices, iv_estimates)
                    continue

            # ── CHECK EXITS on open positions ──
            self._check_exits(date, prices, iv_estimates)

            # ── ENTRY LOGIC ──
            # Monthly cycle: enter on ~1st trading day of month (or rolling)
            is_new_month = (i == 0) or (date.month != dates[i-1].month)
            day_of_month = date.day

            if self.p.get('rolling_entry', False):
                # Variant G: enter whenever we have capacity
                should_enter = len([p for p in self.positions if not p.closed]) < MAX_CONCURRENT
            else:
                # Staggered ladder: enter one per week in first 3 weeks of month
                week_num = (day_of_month - 1) // 7  # 0, 1, 2, 3
                month_key = f"{date.year}-{date.month}"
                if month_key not in month_entry_tracker:
                    month_entry_tracker[month_key] = set()
                should_enter = (
                    week_num < 3 and
                    week_num not in month_entry_tracker.get(month_key, set()) and
                    # Only enter on first trading day of each week
                    (i == 0 or dates[i-1].day <= (week_num * 7) or is_new_month)
                )
                if should_enter:
                    month_entry_tracker.setdefault(month_key, set()).add(week_num)

            if should_enter:
                active_count = len([p for p in self.positions if not p.closed])
                if active_count < MAX_CONCURRENT:
                    self._enter_positions(date, prices, iv_estimates, current_vix,
                                          MAX_CONCURRENT - active_count)

            # ── RECORD EQUITY ──
            self._mark_to_market(date, prices, iv_estimates)

        # Close any remaining open positions at last date
        last_date = dates[-1]
        for pos in self.positions:
            if not pos.closed:
                self._close_position(pos, last_date, prices, iv_estimates, 'end_of_backtest')

        return self._compute_results()

    def _enter_positions(self, date, prices, iv_estimates, vix, max_new):
        """Enter new put credit spread positions."""
        ranked = rank_sectors_by_momentum(prices, date)
        if not ranked:
            return

        # Select sectors based on variant
        if self.p.get('contrarian', False):
            # Bottom N sectors
            selected = [r[0] for r in ranked[-self.p.get('top_n', 3):]]
        elif self.p.get('all_sector', False):
            # All sectors, 1 position each
            selected = [r[0] for r in ranked]
            max_new = min(max_new, len(selected))
        else:
            selected = [r[0] for r in ranked[:self.p.get('top_n', 3)]]

        # Don't duplicate existing positions
        active_tickers = {p.ticker for p in self.positions if not p.closed}
        selected = [s for s in selected if s not in active_tickers]

        entries = 0
        for ticker in selected:
            if entries >= max_new:
                break
            if ticker not in prices.columns or date not in prices.index:
                continue

            S = prices.loc[date, ticker]
            if pd.isna(S) or S <= 0:
                continue

            # Get IV estimate
            if ticker in iv_estimates and date in iv_estimates[ticker].index:
                sigma = iv_estimates[ticker].loc[date]
            else:
                sigma = 0.25  # fallback
            if pd.isna(sigma) or sigma <= 0.05:
                sigma = 0.20

            # Spread construction
            otm_pct = self.p.get('otm_pct', 5.0) / 100.0
            width = self.p.get('width', 3.0)
            dte = self.p.get('dte', 28)

            short_strike = round(S * (1.0 - otm_pct), 0)
            long_strike = short_strike - width
            T = dte / 365.0

            # Price the spread
            credit = price_put_credit_spread(S, short_strike, long_strike, T,
                                              RISK_FREE_RATE, sigma)

            # Minimum credit check
            if credit < 0.10:
                continue

            # Position sizing
            max_loss_per_spread = (width - credit) * 100  # per contract
            max_risk_budget = self.p.get('max_risk_per_spread', 200.0)
            available_capital = self.capital - self.capital_in_use

            if max_loss_per_spread <= 0 or max_loss_per_spread > max_risk_budget:
                # Width too large or credit exceeds width (shouldn't happen)
                num_contracts = 1 if max_loss_per_spread > 0 and max_loss_per_spread <= available_capital else 0
            else:
                num_contracts = min(
                    int(max_risk_budget / max_loss_per_spread),
                    int(available_capital / max_loss_per_spread)
                )

            if num_contracts < 1:
                continue

            # Commission
            commission = COMMISSION_PER_CONTRACT * 2 * num_contracts  # 2 legs
            total_credit = credit * 100 * num_contracts - commission
            total_risk = max_loss_per_spread * num_contracts + commission

            if total_risk > available_capital:
                continue

            expiry_date = date + timedelta(days=dte)

            pos = Position(
                ticker=ticker,
                entry_date=date,
                expiry_date=expiry_date,
                short_strike=short_strike,
                long_strike=long_strike,
                credit_received=credit,
                num_contracts=num_contracts,
                entry_price=S
            )
            self.positions.append(pos)
            self.capital_in_use += total_risk
            entries += 1
            log.debug(f"ENTER {ticker} {short_strike}/{long_strike} put spread, "
                      f"credit={credit:.2f}, contracts={num_contracts}")

    def _check_exits(self, date, prices, iv_estimates):
        """Check exit conditions for open positions."""
        for pos in self.positions:
            if pos.closed:
                continue
            if pos.ticker not in prices.columns or date not in prices.index:
                continue

            S = prices.loc[date, pos.ticker]
            if pd.isna(S):
                continue

            days_held = (date - pos.entry_date).days
            days_to_expiry = (pos.expiry_date - date).days
            T_remaining = max(days_to_expiry / 365.0, 0.001)

            # Get current IV
            if pos.ticker in iv_estimates and date in iv_estimates[pos.ticker].index:
                sigma = iv_estimates[pos.ticker].loc[date]
            else:
                sigma = 0.25
            if pd.isna(sigma) or sigma <= 0.05:
                sigma = 0.20

            # Current spread value (cost to close)
            current_value = spread_value_at_time(S, pos.short_strike, pos.long_strike,
                                                  T_remaining, RISK_FREE_RATE, sigma)

            current_pnl_per_share = pos.credit_received - current_value
            profit_target = self.p.get('profit_target_pct', 0.50)
            loss_cut_mult = self.p.get('loss_cut_mult', 2.0)

            # 1. Profit target: close at X% of max profit
            if current_pnl_per_share >= pos.credit_received * profit_target:
                self._close_position(pos, date, prices, iv_estimates, 'profit_target')
                continue

            # 2. Loss cut: close if loss > N× credit
            if current_pnl_per_share < 0 and abs(current_pnl_per_share) >= pos.credit_received * loss_cut_mult:
                self._close_position(pos, date, prices, iv_estimates, 'loss_cut')
                continue

            # 3. Expiry
            if days_to_expiry <= 0:
                self._close_position(pos, date, prices, iv_estimates, 'expiry')
                continue

            # 4. Roll at 7 DTE if profitable (not for rolling variant)
            if not self.p.get('rolling_entry', False) and days_to_expiry <= 7:
                if current_pnl_per_share > 0:
                    self._close_position(pos, date, prices, iv_estimates, 'roll')
                    continue

    def _close_position(self, pos, date, prices, iv_estimates, reason):
        """Close a position and record P&L."""
        S = prices.loc[date, pos.ticker] if date in prices.index else pos.entry_price
        if pd.isna(S):
            S = pos.entry_price

        days_to_expiry = max((pos.expiry_date - date).days, 0)

        if days_to_expiry <= 0 or reason == 'expiry' or reason == 'end_of_backtest':
            # Settle at expiry
            pnl_per_share = spread_pnl_at_expiry(S, pos.short_strike, pos.long_strike,
                                                   pos.credit_received)
        else:
            T_remaining = days_to_expiry / 365.0
            if pos.ticker in iv_estimates and date in iv_estimates[pos.ticker].index:
                sigma = iv_estimates[pos.ticker].loc[date]
            else:
                sigma = 0.25
            if pd.isna(sigma) or sigma <= 0.05:
                sigma = 0.20
            current_value = spread_value_at_time(S, pos.short_strike, pos.long_strike,
                                                  T_remaining, RISK_FREE_RATE, sigma)
            pnl_per_share = pos.credit_received - current_value

        # Commission to close
        close_commission = COMMISSION_PER_CONTRACT * 2 * pos.num_contracts
        total_pnl = pnl_per_share * 100 * pos.num_contracts - close_commission

        # Also subtract entry commission (already factored conceptually, but track net)
        entry_commission = COMMISSION_PER_CONTRACT * 2 * pos.num_contracts
        net_pnl = total_pnl - entry_commission  # total round-trip cost

        pos.closed = True
        pos.close_date = date
        pos.pnl = net_pnl
        pos.close_reason = reason

        # Release capital
        self.capital += net_pnl
        self.capital_in_use -= pos.max_risk
        self.capital_in_use = max(self.capital_in_use, 0)

        self.closed_trades.append({
            'ticker': pos.ticker,
            'entry_date': pos.entry_date.strftime('%Y-%m-%d'),
            'close_date': date.strftime('%Y-%m-%d'),
            'short_strike': pos.short_strike,
            'long_strike': pos.long_strike,
            'credit': pos.credit_received,
            'contracts': pos.num_contracts,
            'pnl': net_pnl,
            'reason': reason,
            'entry_price': pos.entry_price,
            'exit_price': S,
            'days_held': (date - pos.entry_date).days
        })

    def _mark_to_market(self, date, prices, iv_estimates):
        """Record equity curve point."""
        # Equity = cash + unrealized value of open positions
        unrealized = 0.0
        for pos in self.positions:
            if pos.closed:
                continue
            if pos.ticker not in prices.columns or date not in prices.index:
                continue
            S = prices.loc[date, pos.ticker]
            if pd.isna(S):
                continue
            days_to_expiry = max((pos.expiry_date - date).days, 0)
            T_remaining = max(days_to_expiry / 365.0, 0.001)
            if pos.ticker in iv_estimates and date in iv_estimates[pos.ticker].index:
                sigma = iv_estimates[pos.ticker].loc[date]
            else:
                sigma = 0.25
            if pd.isna(sigma) or sigma <= 0.05:
                sigma = 0.20
            current_value = spread_value_at_time(S, pos.short_strike, pos.long_strike,
                                                  T_remaining, RISK_FREE_RATE, sigma)
            unrealized_pnl = (pos.credit_received - current_value) * 100 * pos.num_contracts
            unrealized += unrealized_pnl

        total_equity = self.capital + unrealized
        active = len([p for p in self.positions if not p.closed])
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'capital': self.capital,
            'unrealized': unrealized,
            'active_positions': active,
            'capital_in_use': self.capital_in_use
        })

    def _compute_results(self):
        """Compute performance metrics."""
        if not self.equity_curve:
            return None

        eq_df = pd.DataFrame(self.equity_curve)
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df = eq_df.set_index('date')

        # Daily returns
        eq_df['daily_ret'] = eq_df['equity'].pct_change()
        eq_df = eq_df.dropna(subset=['daily_ret'])

        if len(eq_df) < 30:
            return None

        rets = eq_df['daily_ret'].values
        n_years = len(eq_df) / 252.0

        # Sharpe
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0.0

        # Sortino
        downside = rets[rets < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-6
        sortino = np.mean(rets) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0

        # Win rate from trades
        if self.closed_trades:
            wins = sum(1 for t in self.closed_trades if t['pnl'] > 0)
            wr = wins / len(self.closed_trades) * 100
            total_wins = sum(t['pnl'] for t in self.closed_trades if t['pnl'] > 0)
            total_losses = abs(sum(t['pnl'] for t in self.closed_trades if t['pnl'] < 0))
            pf = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            wr = 0.0
            pf = 0.0

        # Max drawdown
        equity_series = eq_df['equity']
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak
        mdd = dd.min() * 100

        # CAGR
        final_eq = eq_df['equity'].iloc[-1]
        cagr = (final_eq / STARTING_CAPITAL) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

        # Monthly income yield
        total_income = sum(t['pnl'] for t in self.closed_trades)
        n_months = n_years * 12
        monthly_income = total_income / n_months if n_months > 0 else 0.0
        monthly_yield = monthly_income / STARTING_CAPITAL * 100

        # Capital utilization
        avg_util = eq_df['capital_in_use'].mean() / STARTING_CAPITAL * 100

        return {
            'variant': self.name,
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'wr_pct': round(wr, 1),
            'profit_factor': round(pf, 3),
            'mdd_pct': round(mdd, 1),
            'income_yield_mo_pct': round(monthly_yield, 2),
            'cagr_pct': round(cagr * 100, 2),
            'capital_util_pct': round(avg_util, 1),
            'final_645': round(final_eq, 2),
            'total_trades': len(self.closed_trades),
            'total_pnl': round(total_income, 2),
            'avg_trade_pnl': round(total_income / len(self.closed_trades), 2) if self.closed_trades else 0,
            'equity_curve': eq_df,
            'trades': self.closed_trades,
            'daily_returns': rets
        }


# ─── VALIDATION GATES ────────────────────────────────────────────────────────

def gate_permutation_test(trades, sharpe, n_perms=100):
    """Shuffle trade P&L, recompute Sharpe. PASS if real > 95th percentile."""
    if len(trades) < 10:
        return False, 0.0
    pnls = np.array([t['pnl'] for t in trades])
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = np.random.permutation(pnls)
        # Simulate equity curve from shuffled P&L
        eq = np.cumsum(shuffled) + STARTING_CAPITAL
        rets = np.diff(eq) / eq[:-1]
        if np.std(rets) > 0:
            perm_sharpes.append(np.mean(rets) / np.std(rets) * np.sqrt(252))
        else:
            perm_sharpes.append(0.0)
    p_value = np.mean(np.array(perm_sharpes) >= sharpe)
    return p_value < 0.05, round(p_value, 4)


def gate_subperiod_stability(equity_curve):
    """Split into 4 quarters. PASS if 3/4 have positive Sharpe."""
    if len(equity_curve) < 100:
        return False, 0
    n = len(equity_curve)
    q_size = n // 4
    positive = 0
    for i in range(4):
        start = i * q_size
        end = start + q_size if i < 3 else n
        chunk = equity_curve['daily_ret'].iloc[start:end]
        if len(chunk) > 10 and np.std(chunk) > 0:
            s = np.mean(chunk) / np.std(chunk) * np.sqrt(252)
            if s > 0:
                positive += 1
    return positive >= 3, positive


def gate_outlier_removal(daily_returns, full_sharpe):
    """Trim 1% tails. PASS if trimmed Sharpe > 0.8× full."""
    if len(daily_returns) < 50:
        return False, 0.0
    lo, hi = np.percentile(daily_returns, [0.5, 99.5])
    trimmed = daily_returns[(daily_returns >= lo) & (daily_returns <= hi)]
    if np.std(trimmed) > 0:
        trimmed_sharpe = np.mean(trimmed) / np.std(trimmed) * np.sqrt(252)
    else:
        trimmed_sharpe = 0.0
    ratio = trimmed_sharpe / full_sharpe if full_sharpe != 0 else 0.0
    return ratio > 0.8, round(trimmed_sharpe, 3)


def gate_regime_balance(equity_curve, vix_series):
    """Sharpe on VIX>25 vs VIX<25 days. PASS if gap_ratio < 0.50."""
    merged = equity_curve[['daily_ret']].join(vix_series.rename('vix'), how='inner')
    high_vix = merged[merged['vix'] > 25]['daily_ret']
    low_vix = merged[merged['vix'] <= 25]['daily_ret']

    if len(high_vix) < 20 or len(low_vix) < 20:
        return True, 0.0  # Not enough data, pass by default

    s_high = np.mean(high_vix) / np.std(high_vix) * np.sqrt(252) if np.std(high_vix) > 0 else 0
    s_low = np.mean(low_vix) / np.std(low_vix) * np.sqrt(252) if np.std(low_vix) > 0 else 0

    max_s = max(abs(s_high), abs(s_low))
    gap = abs(s_high - s_low) / max_s if max_s > 0 else 0
    return gap < 0.50, round(gap, 3)


def gate_random_sector_selection(prices, real_sharpe, params, n_trials=100):
    """100 random sector picks. PASS if real Sharpe > 95th percentile."""
    random_sharpes = []
    for _ in range(n_trials):
        # Random variant: shuffle which sectors get picked
        rng_params = params.copy()
        rng_params['_random_sectors'] = True

        bt = PutCreditLadderBacktestRandom(f'random_{_}', rng_params)
        result = bt.run(prices)
        if result is not None:
            random_sharpes.append(result['sharpe'])
        else:
            random_sharpes.append(0.0)

    if not random_sharpes:
        return False, 0.0
    pct = np.percentile(random_sharpes, 95)
    return real_sharpe > pct, round(pct, 3)


class PutCreditLadderBacktestRandom(PutCreditLadderBacktest):
    """Variant that picks random sectors instead of momentum-ranked."""
    def _enter_positions(self, date, prices, iv_estimates, vix, max_new):
        # Random sector selection
        available = [e for e in SECTOR_ETFS if e in prices.columns]
        if not available:
            return
        selected = list(np.random.choice(available, size=min(3, len(available)), replace=False))

        active_tickers = {p.ticker for p in self.positions if not p.closed}
        selected = [s for s in selected if s not in active_tickers]

        entries = 0
        for ticker in selected:
            if entries >= max_new:
                break
            if date not in prices.index:
                continue
            S = prices.loc[date, ticker]
            if pd.isna(S) or S <= 0:
                continue

            if ticker in iv_estimates and date in iv_estimates[ticker].index:
                sigma = iv_estimates[ticker].loc[date]
            else:
                sigma = 0.25
            if pd.isna(sigma) or sigma <= 0.05:
                sigma = 0.20

            otm_pct = self.p.get('otm_pct', 5.0) / 100.0
            width = self.p.get('width', 3.0)
            dte = self.p.get('dte', 28)

            short_strike = round(S * (1.0 - otm_pct), 0)
            long_strike = short_strike - width
            T = dte / 365.0

            credit = price_put_credit_spread(S, short_strike, long_strike, T,
                                              RISK_FREE_RATE, sigma)
            if credit < 0.10:
                continue

            max_loss_per_spread = (width - credit) * 100
            available_capital = self.capital - self.capital_in_use
            if max_loss_per_spread <= 0:
                continue
            num_contracts = min(
                int(self.p.get('max_risk_per_spread', 200.0) / max_loss_per_spread),
                int(available_capital / max_loss_per_spread)
            )
            if num_contracts < 1:
                continue

            total_risk = max_loss_per_spread * num_contracts
            if total_risk > available_capital:
                continue

            expiry_date = date + timedelta(days=dte)
            pos = Position(ticker, date, expiry_date, short_strike, long_strike,
                          credit, num_contracts, S)
            self.positions.append(pos)
            self.capital_in_use += total_risk
            entries += 1


def run_all_gates(result, prices, params):
    """Run all 5 validation gates."""
    gates = {}
    vix = prices['^VIX'] if '^VIX' in prices.columns else None

    # Gate 1: Permutation test
    passed, pval = gate_permutation_test(result['trades'], result['sharpe'])
    gates['permutation'] = {'passed': passed, 'p_value': pval}

    # Gate 2: Sub-period stability
    passed, n_pos = gate_subperiod_stability(result['equity_curve'])
    gates['subperiod'] = {'passed': passed, 'positive_quarters': n_pos}

    # Gate 3: Outlier removal
    passed, trimmed_s = gate_outlier_removal(result['daily_returns'], result['sharpe'])
    gates['outlier'] = {'passed': passed, 'trimmed_sharpe': trimmed_s}

    # Gate 4: Regime balance
    if vix is not None:
        passed, gap = gate_regime_balance(result['equity_curve'], vix)
        gates['regime'] = {'passed': passed, 'gap_ratio': gap}
    else:
        gates['regime'] = {'passed': True, 'gap_ratio': 0.0}

    # Gate 5: Random sector selection
    passed, pct95 = gate_random_sector_selection(prices, result['sharpe'], params)
    gates['random_sector'] = {'passed': passed, 'random_95th': pct95}

    total_passed = sum(1 for g in gates.values() if g['passed'])
    return gates, total_passed


# ─── VARIANT DEFINITIONS ─────────────────────────────────────────────────────

VARIANTS = {
    'A_Base': {
        'top_n': 3, 'otm_pct': 5.0, 'width': 3.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 200.0
    },
    'B_Wider': {
        'top_n': 3, 'otm_pct': 5.0, 'width': 5.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 200.0
    },
    'C_DeeperOTM': {
        'top_n': 3, 'otm_pct': 8.0, 'width': 3.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 200.0
    },
    'D_Aggressive': {
        'top_n': 3, 'otm_pct': 3.0, 'width': 3.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 200.0
    },
    'E_VIXAdaptive': {
        'top_n': 3, 'otm_pct': 5.0, 'width': 3.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 200.0,
        'vix_filter': True, 'vix_entry_min': 18
    },
    'F_AllSector': {
        'top_n': 11, 'otm_pct': 5.0, 'width': 3.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 100.0,
        'all_sector': True
    },
    'G_Rolling': {
        'top_n': 3, 'otm_pct': 5.0, 'width': 3.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 200.0,
        'rolling_entry': True
    },
    'H_Contrarian': {
        'top_n': 3, 'otm_pct': 5.0, 'width': 3.0, 'dte': 28,
        'profit_target_pct': 0.50, 'loss_cut_mult': 2.0,
        'max_risk_per_spread': 200.0,
        'contrarian': True
    },
}


# ─── MONTHLY INCOME BREAKDOWN ────────────────────────────────────────────────

def monthly_income_breakdown(trades):
    """Generate monthly income summary from closed trades."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df['close_date'] = pd.to_datetime(df['close_date'])
    df['month'] = df['close_date'].dt.to_period('M')
    monthly = df.groupby('month').agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100),
    ).round(2)
    monthly['cumulative'] = monthly['total_pnl'].cumsum().round(2)
    monthly['yield_pct'] = (monthly['total_pnl'] / STARTING_CAPITAL * 100).round(2)
    return monthly


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("PUT CREDIT SPREAD LADDER v1 — $645 Account Backtest")
    log.info("=" * 70)

    # Fetch data
    prices = fetch_data()

    # MLflow setup
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('put_credit_ladder_v1')
            log.info("MLflow tracking enabled")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # Run all variants
    all_results = []

    for name, params in VARIANTS.items():
        log.info(f"\n{'─' * 60}")
        log.info(f"Running variant: {name}")
        log.info(f"Params: {params}")

        np.random.seed(42)  # reproducibility
        bt = PutCreditLadderBacktest(name, params)
        result = bt.run(prices)

        if result is None:
            log.warning(f"  {name}: No results (insufficient data/trades)")
            continue

        log.info(f"  Trades: {result['total_trades']}, Final: ${result['final_645']:.2f}")
        log.info(f"  Sharpe: {result['sharpe']}, Sortino: {result['sortino']}, "
                 f"WR: {result['wr_pct']}%, PF: {result['profit_factor']}")

        # Run validation gates
        log.info(f"  Running 5-gate validation...")
        gates, n_passed = run_all_gates(result, prices, params)
        result['gates'] = gates
        result['gates_passed'] = n_passed
        gate_str = f"{n_passed}/5"
        for gname, gdata in gates.items():
            status = "PASS" if gdata['passed'] else "FAIL"
            log.info(f"    {gname}: {status} ({gdata})")

        result['gate_summary'] = gate_str
        all_results.append(result)

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({k: str(v) for k, v in params.items()})
                    mlflow.log_metrics({
                        'sharpe': result['sharpe'],
                        'sortino': result['sortino'],
                        'win_rate': result['wr_pct'],
                        'profit_factor': result['profit_factor'],
                        'mdd_pct': result['mdd_pct'],
                        'cagr_pct': result['cagr_pct'],
                        'monthly_yield_pct': result['income_yield_mo_pct'],
                        'capital_util_pct': result['capital_util_pct'],
                        'final_equity': result['final_645'],
                        'total_trades': result['total_trades'],
                        'gates_passed': n_passed,
                    })
            except Exception as e:
                log.warning(f"MLflow logging failed for {name}: {e}")

    if not all_results:
        log.error("No variants produced results!")
        return

    # ── RESULTS TABLE ──
    log.info("\n" + "=" * 100)
    log.info("RESULTS SUMMARY (sorted by Sharpe)")
    log.info("=" * 100)

    summary_rows = []
    for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
        summary_rows.append({
            'Variant': r['variant'],
            'Sharpe': r['sharpe'],
            'Sortino': r['sortino'],
            'WR%': r['wr_pct'],
            'PF': r['profit_factor'],
            'MDD%': r['mdd_pct'],
            'Inc_Yld_Mo%': r['income_yield_mo_pct'],
            'CAGR%': r['cagr_pct'],
            'Cap_Util%': r['capital_util_pct'],
            '$645_Final': r['final_645'],
            'Trades': r['total_trades'],
            'Gates': r['gate_summary'],
        })

    summary_df = pd.DataFrame(summary_rows)
    log.info("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'variant_summary.csv'), index=False)

    # ── BEST VARIANT MONTHLY BREAKDOWN ──
    best = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[0]
    log.info(f"\n{'=' * 70}")
    log.info(f"MONTHLY INCOME BREAKDOWN — Best Variant: {best['variant']}")
    log.info(f"{'=' * 70}")

    monthly = monthly_income_breakdown(best['trades'])
    if not monthly.empty:
        log.info("\n" + monthly.to_string())
        monthly.to_csv(os.path.join(OUTPUT_DIR, 'best_variant_monthly_income.csv'))

    # ── TRADE LOG FOR BEST VARIANT ──
    if best['trades']:
        trades_df = pd.DataFrame(best['trades'])
        trades_df.to_csv(os.path.join(OUTPUT_DIR, 'best_variant_trades.csv'), index=False)

    # ── EQUITY CURVE FOR ALL VARIANTS ──
    eq_data = {}
    for r in all_results:
        eq_data[r['variant']] = r['equity_curve']['equity']
    eq_combined = pd.DataFrame(eq_data)
    eq_combined.to_csv(os.path.join(OUTPUT_DIR, 'equity_curves.csv'))

    # ── GATE DETAILS ──
    gate_details = {}
    for r in all_results:
        gate_details[r['variant']] = r['gates']
    with open(os.path.join(OUTPUT_DIR, 'gate_details.json'), 'w') as f:
        json.dump(gate_details, f, indent=2, default=str)

    # ── FULL RESULTS JSON ──
    results_json = []
    for r in all_results:
        rj = {k: v for k, v in r.items()
              if k not in ('equity_curve', 'daily_returns', 'trades')}
        results_json.append(rj)
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    log.info(f"\nAll outputs saved to {OUTPUT_DIR}")
    log.info("DONE")


if __name__ == '__main__':
    main()
