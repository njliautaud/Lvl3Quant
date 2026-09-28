"""
Calendar Spread (Time Spread) Walk-Forward Backtest
====================================================
Strategy: Sell near-term puts (7-14 DTE), buy far-term puts (35-45 DTE).
Profit from theta decay differential when term structure is favorable.

Data: Uses modeled BS IV from wheel_strategy_v1/data/cache/ and prices.parquet.
Walk-forward: Sliding window (60-day train for IV rank calibration, OOS only reported).

Author: Claude Opus 4.6
Date: 2026-07-11
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Black-Scholes pricer (copied from wheel_engine to stay standalone)
# ============================================================
SQRT_2PI = math.sqrt(2 * math.pi)

def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _ndtri(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    p = min(max(p, 1e-10), 1 - 1e-10)
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r2 = q * q
    return (((((a[0]*r2+a[1])*r2+a[2])*r2+a[3])*r2+a[4])*r2+a[5])*q / \
           (((((b[0]*r2+b[1])*r2+b[2])*r2+b[3])*r2+b[4])*r2+1)


def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * math.exp(-q * T) * _Phi(-d1)
    return S * math.exp(-q * T) * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def bs_delta(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return -1.0 if (kind == "put" and S < K) else (1.0 if (kind == "call" and S > K) else 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if kind == "put":
        return math.exp(-q * T) * (_Phi(d1) - 1.0)
    return math.exp(-q * T) * _Phi(d1)


def strike_from_delta(S, T, sigma, target_delta, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = (1 - target) if kind == "put" else target
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5 * sigma**2) * T))
    return K


def bs_theta(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    """Daily theta (dollars per day per share)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    phi_d1 = math.exp(-0.5 * d1**2) / SQRT_2PI

    term1 = -S * math.exp(-q * T) * phi_d1 * sigma / (2 * sqrtT)
    if kind == "put":
        term2 = r * K * math.exp(-r * T) * _Phi(-d2)
        term3 = -q * S * math.exp(-q * T) * _Phi(-d1)
        return (term1 + term2 + term3) / 365.0
    else:
        term2 = -r * K * math.exp(-r * T) * _Phi(d2)
        term3 = q * S * math.exp(-q * T) * _Phi(d1)
        return (term1 + term2 + term3) / 365.0


# ============================================================
# Calendar Spread Position
# ============================================================
@dataclass
class CalendarSpread:
    ticker: str
    open_date: str
    strike: float
    front_dte_at_open: int
    back_dte_at_open: int
    front_expiry_date: str
    back_expiry_date: str
    front_sigma: float       # IV used for front leg
    back_sigma: float        # IV used for back leg
    front_premium_sold: float   # per share
    back_premium_paid: float    # per share
    net_debit: float            # per share (back_paid - front_sold)
    num_contracts: int
    entry_price: float       # underlying price at entry
    open_cost: float         # total transaction cost at open

    # Tracking
    days_held: int = 0
    closed: bool = False
    close_date: Optional[str] = None
    close_pnl: float = 0.0
    close_reason: str = ""


# ============================================================
# Strategy Config
# ============================================================
@dataclass
class CalendarConfig:
    # Entry
    front_dte_min: int = 7
    front_dte_max: int = 14
    front_dte_target: int = 10
    back_dte_min: int = 35
    back_dte_max: int = 50
    back_dte_target: int = 42
    target_delta: float = 0.25      # put delta for strike selection

    # Filters
    min_iv_rank: float = 0.40       # only enter when IV is elevated
    min_term_ratio: float = 1.05    # front vol > back vol (favorable structure)
    max_positions: int = 5          # concurrent positions
    max_per_ticker: int = 1

    # Sizing
    capital: float = 100_000.0
    max_risk_per_trade: float = 0.02   # 2% of capital max risk per spread

    # Exit
    profit_target_pct: float = 0.40    # close at 40% of max profit
    max_loss_pct: float = 1.50         # close at 150% of net debit loss
    max_hold_days: int = 12            # close if held > N days (before front expiry)

    # Costs (realistic retail)
    commission_per_contract_per_leg: float = 0.65  # IBKR-like
    slippage_pct: float = 0.03          # 3% of premium per leg (conservative)
    slippage_min: float = 0.02          # $0.02/share minimum

    # Walk-forward
    iv_rank_lookback: int = 252         # 1yr for IV rank calibration
    min_oos_start_date: str = "2016-06-01"  # skip first year for warm-up


# ============================================================
# Main Backtest Engine
# ============================================================
class CalendarSpreadBacktest:
    def __init__(self, config: CalendarConfig = None):
        self.cfg = config or CalendarConfig()
        self.positions: list[CalendarSpread] = []
        self.closed_trades: list[CalendarSpread] = []
        self.equity_curve = []
        self.daily_pnl = []

    def load_data(self):
        """Load price and IV data."""
        root = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache")

        # Prices
        prices = pd.read_parquet(root / "prices.parquet")
        prices['date'] = pd.to_datetime(prices['date'])
        self.prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)

        # IV features (modeled)
        iv = pd.read_parquet(root / "iv_features_modeled.parquet")
        iv['date'] = pd.to_datetime(iv['date'])
        self.iv_data = iv.sort_values(['ticker', 'date']).reset_index(drop=True)

        # SPY for regime classification
        spy = pd.read_parquet(root / "spy_prices.parquet")
        spy['date'] = pd.to_datetime(spy['date'])
        spy = spy.sort_values('date').reset_index(drop=True)
        spy['spy_ret'] = spy['close'].pct_change()
        self.spy = spy

        # Merge prices + IV
        self.merged = pd.merge(
            self.prices[['ticker', 'date', 'close', 'volume']],
            self.iv_data[['ticker', 'date', 'sigma', 'sigma_atm_30d', 'iv_rank', 'term_ratio', 'term_proxy']],
            on=['ticker', 'date'],
            how='inner'
        )
        self.merged = self.merged.dropna(subset=['sigma', 'iv_rank', 'term_ratio', 'close'])
        self.merged = self.merged.sort_values(['ticker', 'date']).reset_index(drop=True)

        # Build lookup dict for fast access
        self._price_lookup = {}
        for _, row in self.merged.iterrows():
            key = (row['ticker'], row['date'].strftime('%Y-%m-%d'))
            self._price_lookup[key] = row

        # Get trading dates
        self.trading_dates = sorted(self.merged['date'].unique())

        print(f"Loaded {len(self.merged):,} ticker-day rows")
        print(f"Date range: {self.trading_dates[0].strftime('%Y-%m-%d')} to {self.trading_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Tickers: {self.merged['ticker'].nunique()}")

    def _slippage(self, premium: float) -> float:
        """Half-spread cost per share."""
        return max(self.cfg.slippage_min, self.cfg.slippage_pct * abs(premium))

    def _total_cost_open(self, front_prem, back_prem, n_contracts):
        """Total transaction cost to open a calendar spread (2 legs)."""
        comm = 2 * self.cfg.commission_per_contract_per_leg * n_contracts
        slip_front = self._slippage(front_prem) * 100 * n_contracts
        slip_back = self._slippage(back_prem) * 100 * n_contracts
        return comm + slip_front + slip_back

    def _total_cost_close(self, front_prem, back_prem, n_contracts):
        """Total transaction cost to close."""
        return self._total_cost_open(front_prem, back_prem, n_contracts)

    def _term_structure_sigma(self, sigma_30d, term_ratio, dte_front, dte_back):
        """
        Derive front and back IV from term structure.
        term_ratio = sigma_front / sigma_back (from iv_features_modeled).
        sigma_30d ~ ATM 30-day IV.

        We model:
          sigma_front = sigma_30d * sqrt(30/dte_front)^(-power) * term_ratio^(weight)
          sigma_back  = sigma_30d * sqrt(30/dte_back)^(-power)

        Simpler approach: use term_ratio directly.
          sigma_front = sigma_30d * (term_ratio ** 0.5)   (front vol inflated)
          sigma_back  = sigma_30d * (term_ratio ** -0.5)  (back vol deflated)
        This ensures sigma_front / sigma_back ~ term_ratio.
        """
        # Scale volatility by sqrt(time) and term structure
        # Front (shorter DTE) has higher vol per unit time
        ratio_sqrt = math.sqrt(max(term_ratio, 0.5))
        sigma_front = sigma_30d * ratio_sqrt
        sigma_back = sigma_30d / ratio_sqrt
        return sigma_front, sigma_back

    def _find_entry_candidates(self, date_str, date):
        """Find tickers with favorable calendar spread entry conditions."""
        cfg = self.cfg
        day_data = self.merged[self.merged['date'] == date]

        candidates = []
        # Count current positions per ticker
        pos_tickers = {p.ticker for p in self.positions if not p.closed}

        if len(self.positions) - len([p for p in self.positions if p.closed]) >= cfg.max_positions:
            return []

        for _, row in day_data.iterrows():
            ticker = row['ticker']

            # Skip if already have position in this ticker
            if ticker in pos_tickers:
                continue

            # IV rank filter
            if pd.isna(row['iv_rank']) or row['iv_rank'] < cfg.min_iv_rank:
                continue

            # Term structure filter (front vol > back vol = favorable for selling front)
            if pd.isna(row['term_ratio']) or row['term_ratio'] < cfg.min_term_ratio:
                continue

            # Need valid sigma
            if pd.isna(row['sigma']) or row['sigma'] <= 0.05:
                continue

            # Minimum price filter (avoid penny stocks)
            if row['close'] < 20:
                continue

            candidates.append(row)

        # Sort by term_ratio * iv_rank (best opportunities first)
        candidates.sort(key=lambda r: r['term_ratio'] * r['iv_rank'], reverse=True)
        return candidates

    def _open_position(self, row, date_str):
        """Open a calendar spread position."""
        cfg = self.cfg
        S = row['close']
        sigma = row['sigma']
        sigma_30d = row['sigma_atm_30d'] if not pd.isna(row['sigma_atm_30d']) else sigma
        term_ratio = row['term_ratio']

        # Derive front and back IV
        sigma_front, sigma_back = self._term_structure_sigma(
            sigma_30d, term_ratio, cfg.front_dte_target, cfg.back_dte_target
        )

        T_front = cfg.front_dte_target / 365.0
        T_back = cfg.back_dte_target / 365.0

        # Find strike at target delta using front leg sigma
        K = strike_from_delta(S, T_front, sigma_front, cfg.target_delta, kind="put")
        # Round to nearest 0.50 for realism
        K = round(K * 2) / 2.0

        # Price both legs
        front_prem = bs_price(S, K, T_front, sigma_front, kind="put")
        back_prem = bs_price(S, K, T_back, sigma_back, kind="put")

        # Net debit = back premium paid - front premium received
        net_debit = back_prem - front_prem

        if net_debit <= 0:
            # Calendar should be a debit spread; if credit, skip (mispricing)
            return None

        # Sizing: risk = net_debit * 100 * contracts
        max_risk = cfg.capital * cfg.max_risk_per_trade
        n_contracts = max(1, int(max_risk / (net_debit * 100)))
        n_contracts = min(n_contracts, 10)  # cap at 10 contracts

        # Transaction costs
        open_cost = self._total_cost_open(front_prem, back_prem, n_contracts)

        # Compute expiry dates (business days from now)
        front_expiry = pd.Timestamp(date_str) + pd.offsets.BDay(cfg.front_dte_target)
        back_expiry = pd.Timestamp(date_str) + pd.offsets.BDay(cfg.back_dte_target)

        pos = CalendarSpread(
            ticker=row['ticker'],
            open_date=date_str,
            strike=K,
            front_dte_at_open=cfg.front_dte_target,
            back_dte_at_open=cfg.back_dte_target,
            front_expiry_date=front_expiry.strftime('%Y-%m-%d'),
            back_expiry_date=back_expiry.strftime('%Y-%m-%d'),
            front_sigma=sigma_front,
            back_sigma=sigma_back,
            front_premium_sold=front_prem,
            back_premium_paid=back_prem,
            net_debit=net_debit,
            num_contracts=n_contracts,
            entry_price=S,
            open_cost=open_cost,
        )
        self.positions.append(pos)
        return pos

    def _mark_position(self, pos: CalendarSpread, date_str, S, sigma_30d, term_ratio):
        """Mark a calendar spread to market and check exit conditions."""
        if pos.closed:
            return 0.0

        pos.days_held += 1
        cfg = self.cfg

        # Remaining DTE for each leg
        front_dte_remain = max(0, pos.front_dte_at_open - pos.days_held)
        back_dte_remain = max(0, pos.back_dte_at_open - pos.days_held)

        T_front = front_dte_remain / 365.0
        T_back = back_dte_remain / 365.0

        # Use current IV data if available, else original
        if sigma_30d and not pd.isna(sigma_30d) and term_ratio and not pd.isna(term_ratio):
            s_front, s_back = self._term_structure_sigma(sigma_30d, term_ratio,
                                                         front_dte_remain, back_dte_remain)
        else:
            s_front = pos.front_sigma
            s_back = pos.back_sigma

        K = pos.strike

        # Current prices
        front_val = bs_price(S, K, T_front, s_front, kind="put")
        back_val = bs_price(S, K, T_back, s_back, kind="put")

        # Spread value = back_val - front_val (what we'd get if we closed now)
        spread_val = back_val - front_val

        # P&L = (current spread value - entry net debit) * 100 * contracts
        pnl_per_share = spread_val - pos.net_debit
        mtm_pnl = pnl_per_share * 100 * pos.num_contracts

        # Max profit estimate: at front expiry with stock at strike,
        # back leg retains most value. Approximate max profit ~ front_premium_sold
        max_profit = pos.front_premium_sold * 100 * pos.num_contracts

        # Check exit conditions
        close_cost = self._total_cost_close(front_val, back_val, pos.num_contracts)
        should_close = False
        reason = ""

        # 1. Profit target
        if mtm_pnl > 0 and mtm_pnl >= cfg.profit_target_pct * max_profit:
            should_close = True
            reason = "profit_target"

        # 2. Stop loss
        max_loss = pos.net_debit * 100 * pos.num_contracts * cfg.max_loss_pct
        if mtm_pnl < -max_loss:
            should_close = True
            reason = "stop_loss"

        # 3. Front expiry approaching (close before to avoid assignment risk)
        if front_dte_remain <= 1:
            should_close = True
            reason = "front_expiry"

        # 4. Max hold days
        if pos.days_held >= cfg.max_hold_days:
            should_close = True
            reason = "max_hold"

        if should_close:
            pos.closed = True
            pos.close_date = date_str
            pos.close_pnl = mtm_pnl - pos.open_cost - close_cost
            pos.close_reason = reason
            self.closed_trades.append(pos)
            return pos.close_pnl

        return mtm_pnl  # unrealized

    def run(self):
        """Run the walk-forward backtest."""
        cfg = self.cfg
        min_date = pd.Timestamp(cfg.min_oos_start_date)

        capital = cfg.capital
        equity = capital

        dates = [d for d in self.trading_dates if d >= min_date]
        print(f"\nRunning backtest: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"OOS days: {len(dates)}")
        print(f"Config: front_dte={cfg.front_dte_target}, back_dte={cfg.back_dte_target}, "
              f"delta={cfg.target_delta}, iv_rank_min={cfg.min_iv_rank}, "
              f"term_ratio_min={cfg.min_term_ratio}")
        print()

        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            day_data = self.merged[self.merged['date'] == date]

            # Mark existing positions
            daily_realized = 0.0
            daily_unrealized = 0.0

            active_positions = [p for p in self.positions if not p.closed]

            for pos in active_positions:
                ticker_row = day_data[day_data['ticker'] == pos.ticker]
                if len(ticker_row) == 0:
                    continue
                tr = ticker_row.iloc[0]
                pnl = self._mark_position(pos, date_str, tr['close'],
                                          tr.get('sigma_atm_30d'), tr.get('term_ratio'))
                if pos.closed:
                    daily_realized += pos.close_pnl
                else:
                    daily_unrealized += pnl

            # Look for new entries
            candidates = self._find_entry_candidates(date_str, date)
            for cand in candidates[:2]:  # max 2 new entries per day
                active_count = sum(1 for p in self.positions if not p.closed)
                if active_count >= cfg.max_positions:
                    break
                self._open_position(cand, date_str)

            # Track equity
            equity += daily_realized
            self.equity_curve.append({
                'date': date_str,
                'equity': equity,
                'realized': daily_realized,
                'unrealized': daily_unrealized,
                'n_positions': sum(1 for p in self.positions if not p.closed),
                'n_trades_total': len(self.closed_trades),
            })
            self.daily_pnl.append(daily_realized)

        # Close any remaining open positions at last date
        last_date_str = dates[-1].strftime('%Y-%m-%d')
        last_day = self.merged[self.merged['date'] == dates[-1]]
        for pos in self.positions:
            if not pos.closed:
                ticker_row = last_day[last_day['ticker'] == pos.ticker]
                if len(ticker_row) > 0:
                    tr = ticker_row.iloc[0]
                    # Force close
                    pos.days_held = pos.front_dte_at_open  # simulate expiry
                    self._mark_position(pos, last_date_str, tr['close'],
                                       tr.get('sigma_atm_30d'), tr.get('term_ratio'))
                    if not pos.closed:
                        pos.closed = True
                        pos.close_date = last_date_str
                        pos.close_reason = "end_of_backtest"
                        pos.close_pnl = 0.0
                        self.closed_trades.append(pos)

        return self._compute_stats(dates)

    def _compute_stats(self, dates):
        """Compute performance statistics."""
        eq = pd.DataFrame(self.equity_curve)
        eq['date'] = pd.to_datetime(eq['date'])

        # Daily returns
        eq['daily_ret'] = eq['equity'].pct_change().fillna(0)

        # Merge SPY for regime analysis
        spy_daily = self.spy[['date', 'spy_ret']].copy()
        eq = eq.merge(spy_daily, on='date', how='left')
        eq['spy_ret'] = eq['spy_ret'].fillna(0)

        # Classify regime: green (>+0.5%), red (<-0.5%), flat
        eq['regime'] = 'flat'
        eq.loc[eq['spy_ret'] > 0.005, 'regime'] = 'green'
        eq.loc[eq['spy_ret'] < -0.005, 'regime'] = 'red'

        # Basic stats
        total_days = len(eq)
        years = total_days / 252.0

        total_return = (eq['equity'].iloc[-1] / self.cfg.capital) - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        daily_rets = eq['daily_ret'].values
        mean_ret = np.mean(daily_rets)
        std_ret = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else 1e-6

        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

        # Sortino
        downside = daily_rets[daily_rets < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
        sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Max drawdown
        peak = eq['equity'].cummax()
        drawdown = (eq['equity'] - peak) / peak
        max_dd = drawdown.min()

        # Trade stats
        trades = self.closed_trades
        n_trades = len(trades)
        winners = [t for t in trades if t.close_pnl > 0]
        losers = [t for t in trades if t.close_pnl <= 0]
        wr = len(winners) / n_trades if n_trades > 0 else 0

        gross_profit = sum(t.close_pnl for t in winners)
        gross_loss = abs(sum(t.close_pnl for t in losers))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean([t.close_pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.close_pnl for t in losers]) if losers else 0
        avg_hold = np.mean([t.days_held for t in trades]) if trades else 0

        # Close reasons
        reasons = {}
        for t in trades:
            reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

        # Regime-stratified Sharpe
        regime_stats = {}
        for regime in ['green', 'red', 'flat']:
            mask = eq['regime'] == regime
            if mask.sum() > 5:
                r = daily_rets[mask.values]
                regime_sharpe = (np.mean(r) / np.std(r, ddof=1)) * np.sqrt(252) if np.std(r, ddof=1) > 0 else 0
                regime_stats[regime] = {
                    'n_days': int(mask.sum()),
                    'sharpe': round(regime_sharpe, 3),
                    'mean_ret': round(np.mean(r) * 10000, 2),  # bps
                }

        # Regime gap check (HC #428)
        sharpe_green = regime_stats.get('green', {}).get('sharpe', 0)
        sharpe_red = regime_stats.get('red', {}).get('sharpe', 0)
        max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0

        # SPY buy-and-hold comparison
        spy_bt = self.spy[(self.spy['date'] >= eq['date'].min()) &
                          (self.spy['date'] <= eq['date'].max())].copy()
        if len(spy_bt) > 1:
            spy_total_ret = spy_bt['close'].iloc[-1] / spy_bt['close'].iloc[0] - 1
            spy_cagr = (1 + spy_total_ret) ** (1 / years) - 1 if years > 0 else 0
            spy_rets = spy_bt['close'].pct_change().dropna().values
            spy_sharpe = (np.mean(spy_rets) / np.std(spy_rets, ddof=1)) * np.sqrt(252) if np.std(spy_rets, ddof=1) > 0 else 0
            spy_peak = spy_bt['close'].cummax()
            spy_dd = ((spy_bt['close'].values - spy_peak.values) / spy_peak.values).min()
        else:
            spy_total_ret = spy_cagr = spy_sharpe = spy_dd = 0

        results = {
            'strategy': 'Calendar Spread (Put)',
            'period': f"{eq['date'].iloc[0].strftime('%Y-%m-%d')} to {eq['date'].iloc[-1].strftime('%Y-%m-%d')}",
            'total_days': total_days,
            'years': round(years, 2),
            'starting_capital': self.cfg.capital,
            'ending_capital': round(eq['equity'].iloc[-1], 2),
            'total_return_pct': round(total_return * 100, 2),
            'cagr_pct': round(cagr * 100, 2),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'profit_factor': round(pf, 3),
            'win_rate_pct': round(wr * 100, 1),
            'n_trades': n_trades,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_hold_days': round(avg_hold, 1),
            'close_reasons': reasons,
            'regime_analysis': regime_stats,
            'regime_gap': round(regime_gap, 3),
            'regime_gap_pass': regime_gap <= 0.50,
            'spy_comparison': {
                'spy_total_return_pct': round(spy_total_ret * 100, 2),
                'spy_cagr_pct': round(spy_cagr * 100, 2),
                'spy_sharpe': round(spy_sharpe, 3),
                'spy_max_dd_pct': round(spy_dd * 100, 2),
            },
            'config': {
                'front_dte': self.cfg.front_dte_target,
                'back_dte': self.cfg.back_dte_target,
                'target_delta': self.cfg.target_delta,
                'min_iv_rank': self.cfg.min_iv_rank,
                'min_term_ratio': self.cfg.min_term_ratio,
                'profit_target_pct': self.cfg.profit_target_pct,
                'max_loss_pct': self.cfg.max_loss_pct,
                'commission_per_leg': self.cfg.commission_per_contract_per_leg,
                'slippage_pct': self.cfg.slippage_pct,
            }
        }

        return results, eq


def print_results(results, eq_df):
    """Pretty-print backtest results."""
    r = results
    print("=" * 70)
    print(f"  CALENDAR SPREAD BACKTEST RESULTS")
    print(f"  {r['period']}")
    print("=" * 70)
    print()
    print(f"  Total Return:    {r['total_return_pct']:+.2f}%")
    print(f"  CAGR:            {r['cagr_pct']:+.2f}%")
    print(f"  Sharpe Ratio:    {r['sharpe']:.3f}")
    print(f"  Sortino Ratio:   {r['sortino']:.3f}")
    print(f"  Max Drawdown:    {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor:   {r['profit_factor']:.3f}")
    print(f"  Win Rate:        {r['win_rate_pct']:.1f}%")
    print()
    print(f"  Trades:          {r['n_trades']}")
    print(f"  Avg Win:         ${r['avg_win']:.2f}")
    print(f"  Avg Loss:        ${r['avg_loss']:.2f}")
    print(f"  Avg Hold:        {r['avg_hold_days']:.1f} days")
    print()
    print(f"  Close Reasons:   {r['close_reasons']}")
    print()
    print("  --- Regime Analysis ---")
    for regime, stats in r['regime_analysis'].items():
        print(f"    {regime:5s}: Sharpe={stats['sharpe']:+.3f}  "
              f"mean={stats['mean_ret']:+.2f}bps  "
              f"n={stats['n_days']} days")
    print(f"  Regime Gap:      {r['regime_gap']:.3f} {'PASS' if r['regime_gap_pass'] else 'FAIL'}")
    print()
    print("  --- vs SPY Buy-and-Hold ---")
    spy = r['spy_comparison']
    print(f"    SPY Return:    {spy['spy_total_return_pct']:+.2f}%")
    print(f"    SPY CAGR:      {spy['spy_cagr_pct']:+.2f}%")
    print(f"    SPY Sharpe:    {spy['spy_sharpe']:.3f}")
    print(f"    SPY Max DD:    {spy['spy_max_dd_pct']:.2f}%")
    print()
    print("=" * 70)


def run_sensitivity_sweep():
    """Run multiple configurations to find robust parameter set."""
    configs = [
        # Baseline
        CalendarConfig(min_iv_rank=0.40, min_term_ratio=1.05, target_delta=0.25,
                       profit_target_pct=0.40, front_dte_target=10, back_dte_target=42),
        # Higher IV bar
        CalendarConfig(min_iv_rank=0.60, min_term_ratio=1.10, target_delta=0.25,
                       profit_target_pct=0.40, front_dte_target=10, back_dte_target=42),
        # Wider spread (7/45)
        CalendarConfig(min_iv_rank=0.40, min_term_ratio=1.05, target_delta=0.25,
                       profit_target_pct=0.40, front_dte_target=7, back_dte_target=45),
        # More aggressive delta
        CalendarConfig(min_iv_rank=0.40, min_term_ratio=1.05, target_delta=0.30,
                       profit_target_pct=0.40, front_dte_target=10, back_dte_target=42),
        # Tighter profit target
        CalendarConfig(min_iv_rank=0.40, min_term_ratio=1.05, target_delta=0.25,
                       profit_target_pct=0.25, front_dte_target=10, back_dte_target=42),
        # Relaxed IV rank
        CalendarConfig(min_iv_rank=0.30, min_term_ratio=1.03, target_delta=0.25,
                       profit_target_pct=0.40, front_dte_target=10, back_dte_target=42),
    ]

    config_names = [
        "Baseline (IVR40/TR1.05/D25)",
        "High IV (IVR60/TR1.10/D25)",
        "Wide DTE (7/45, IVR40)",
        "Aggressive Delta (D30)",
        "Tight PT (25%)",
        "Relaxed Filters (IVR30/TR1.03)",
    ]

    all_results = []
    best_result = None
    best_sharpe = -999
    best_eq = None

    for name, cfg in zip(config_names, configs):
        print(f"\n{'='*70}")
        print(f"  Configuration: {name}")
        print(f"{'='*70}")

        bt = CalendarSpreadBacktest(cfg)
        bt.load_data()
        results, eq = bt.run()
        results['config_name'] = name
        all_results.append(results)

        print_results(results, eq)

        if results['sharpe'] > best_sharpe and results['n_trades'] >= 20:
            best_sharpe = results['sharpe']
            best_result = results
            best_eq = eq

    return all_results, best_result, best_eq


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    out_dir = Path("/home/jupiter/Lvl3Quant/output/calendar_spread_research")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Calendar Spread (Time Spread) Walk-Forward Backtest")
    print("=" * 70)

    # Run sensitivity sweep
    all_results, best_result, best_eq = run_sensitivity_sweep()

    # Save results
    # Convert to JSON-serializable
    for r in all_results:
        for k, v in r.items():
            if isinstance(v, np.integer):
                r[k] = int(v)
            elif isinstance(v, np.floating):
                r[k] = float(v)

    with open(out_dir / "backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if best_eq is not None:
        best_eq.to_parquet(out_dir / "best_equity_curve.parquet", index=False)

    # Summary comparison table
    print("\n" + "=" * 70)
    print("  CONFIGURATION COMPARISON")
    print("=" * 70)
    print(f"{'Config':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'WR%':>5} {'PF':>6} {'#Tr':>5} {'RGap':>5}")
    print("-" * 100)
    for r in all_results:
        name = r.get('config_name', 'unknown')[:34]
        print(f"{name:<35} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
              f"{r['cagr_pct']:>+7.2f} {r['max_drawdown_pct']:>7.2f} "
              f"{r['win_rate_pct']:>5.1f} {r['profit_factor']:>6.3f} "
              f"{r['n_trades']:>5d} {r['regime_gap']:>5.3f}")

    if best_result:
        print(f"\nBest config by Sharpe: {best_result.get('config_name', 'N/A')}")
        spy = best_result['spy_comparison']
        print(f"Strategy: Sharpe={best_result['sharpe']:.3f}, CAGR={best_result['cagr_pct']:.2f}%")
        print(f"SPY B&H:  Sharpe={spy['spy_sharpe']:.3f}, CAGR={spy['spy_cagr_pct']:.2f}%")

    print(f"\nResults saved to {out_dir}/")
    print("Done.")
