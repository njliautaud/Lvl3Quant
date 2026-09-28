#!/usr/bin/env python3
"""
Calendar Spread Income Strategy Backtest v1
============================================
Thesis: Short-dated options decay faster than long-dated ones.
Sell 7-DTE ATM options, buy 30-DTE same-strike options, capture theta differential.

Uses Black-Scholes pricing with realized vol as IV proxy (conservative).
Includes 2% slippage, realistic commissions, regime-agnostic validation.
"""

import os
import sys
import json
import logging
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/calendar_spread_income_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "backtest.log"
RESULTS_FILE = OUTPUT_DIR / "results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Parameters ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "NFLX", "CRM", "SNOW", "SHOP", "SQ", "COIN", "ABNB", "UBER",
    "DASH", "PLTR", "RIVN", "SOFI",
]

START_DATE = "2020-01-02"
END_DATE = "2026-07-11"
STARTING_CAPITAL = 100_000.0
MAX_CONCURRENT = 5
MAX_RISK_PCT = 0.03          # 3% NAV per trade
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_PCT = 0.02          # 2% on each leg
SHORT_DTE = 7
LONG_DTE = 30
PROFIT_TARGET_PCT = 0.40     # 40% of max profit (middle of 25-50%)
STOP_LOSS_PCT = 0.50         # 50% of debit
ROLL_AT_DTE = 1              # Roll short leg at 1 DTE
RISK_FREE_RATE = 0.04        # ~current
REALIZED_VOL_WINDOW = 30     # days for realized vol calc
NUM_CONTRACTS_PER_TRADE = 1  # will scale by NAV
PERMUTATION_SHUFFLES = 100

# ── Black-Scholes ───────────────────────────────────────────────────────────

def _to_date(d):
    """Convert any date-like to datetime.date."""
    if hasattr(d, 'date') and callable(d.date):
        return d.date()
    return d


def bs_price(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes option price. T in years."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes delta."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


# ── Data Loading ────────────────────────────────────────────────────────────

def load_price_data(tickers, start, end):
    """Download daily OHLCV for universe + SPY."""
    all_tickers = list(set(tickers + ["SPY"]))
    log.info(f"Downloading price data for {len(all_tickers)} tickers...")
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) > 100:
                data[ticker] = df
                log.info(f"  {ticker}: {len(df)} days loaded")
            else:
                log.warning(f"  {ticker}: only {len(df)} days, skipping")
        except Exception as e:
            log.warning(f"  {ticker}: download failed: {e}")
    return data


def compute_realized_vol(prices_df, window=30):
    """Annualized realized volatility from close prices."""
    log_returns = np.log(prices_df["Close"] / prices_df["Close"].shift(1))
    rv = log_returns.rolling(window).std() * np.sqrt(252)
    return rv


# ── Calendar Spread Logic ──────────────────────────────────────────────────

class CalendarSpread:
    """Represents a single calendar spread position."""
    def __init__(self, ticker, entry_date, strike, spot, short_dte, long_dte,
                 sigma, option_type, num_contracts, debit_paid, commission):
        self.ticker = ticker
        self.entry_date = entry_date
        self.strike = strike
        self.spot_at_entry = spot
        self.short_expiry = entry_date + dt.timedelta(days=short_dte)
        self.long_expiry = entry_date + dt.timedelta(days=long_dte)
        self.sigma = sigma
        self.option_type = option_type
        self.num_contracts = num_contracts
        self.debit_paid = debit_paid  # total debit including slippage+commission
        self.commission = commission
        self.max_profit = self._calc_max_profit(spot, sigma)
        self.profit_target = self.max_profit * PROFIT_TARGET_PCT
        self.stop_loss = -self.debit_paid * STOP_LOSS_PCT
        self.closed = False
        self.close_date = None
        self.pnl = 0.0
        self.close_reason = ""
        self.rolled_count = 0

    def _calc_max_profit(self, spot, sigma):
        """Approximate max profit: happens when spot = strike at short expiry."""
        T_long_remaining = (self.long_expiry - self.short_expiry).days / 365.0
        long_value = bs_price(self.strike, self.strike, T_long_remaining,
                              RISK_FREE_RATE, sigma, self.option_type)
        # Max profit = long option value at short expiry (at strike) - debit paid
        max_p = long_value * 100 * self.num_contracts - self.debit_paid
        return max(max_p, self.debit_paid * 0.1)  # floor at 10% of debit

    def mark_to_market(self, current_date, spot, sigma):
        """Calculate current P&L of the spread."""
        cd = _to_date(current_date)
        short_T = max((self.short_expiry - cd).days / 365.0, 0)
        long_T = max((self.long_expiry - cd).days / 365.0, 0)

        short_price = bs_price(spot, self.strike, short_T, RISK_FREE_RATE,
                               sigma, self.option_type)
        long_price = bs_price(spot, self.strike, long_T, RISK_FREE_RATE,
                              sigma, self.option_type)

        spread_value = (long_price - short_price) * 100 * self.num_contracts
        # P&L = current spread value - debit paid
        return spread_value - self.debit_paid

    def days_to_short_expiry(self, current_date):
        return (self.short_expiry - _to_date(current_date)).days


class CalendarSpreadBacktest:
    """Main backtest engine."""

    def __init__(self, price_data, universe):
        self.price_data = price_data
        self.universe = [t for t in universe if t in price_data]
        self.spy_data = price_data.get("SPY")
        self.positions = []
        self.closed_positions = []
        self.nav_history = []
        self.trade_log = []
        self.capital = STARTING_CAPITAL
        self.cash = STARTING_CAPITAL

        # Precompute realized vol for all tickers
        self.vol_data = {}
        for ticker in self.universe:
            self.vol_data[ticker] = compute_realized_vol(
                self.price_data[ticker], REALIZED_VOL_WINDOW
            )

        # Get trading days from SPY
        self.trading_days = self.spy_data.index.tolist()
        log.info(f"Universe: {len(self.universe)} tickers, {len(self.trading_days)} trading days")

    def _get_spot(self, ticker, date):
        """Get close price for ticker on date."""
        df = self.price_data[ticker]
        if date in df.index:
            return float(df.loc[date, "Close"])
        # Find nearest prior date
        mask = df.index <= date
        if mask.any():
            return float(df.loc[df.index[mask][-1], "Close"])
        return None

    def _get_vol(self, ticker, date):
        """Get realized vol for ticker on date."""
        vol_series = self.vol_data[ticker]
        if date in vol_series.index:
            v = vol_series.loc[date]
            if pd.notna(v) and v > 0.05:
                return float(v)
        # Fallback
        mask = vol_series.index <= date
        valid = vol_series.loc[mask].dropna()
        if len(valid) > 0:
            v = float(valid.iloc[-1])
            if v > 0.05:
                return v
        return 0.30  # default

    def _select_option_type(self, ticker, date):
        """Pick put or call based on recent trend (sell what's richer)."""
        df = self.price_data[ticker]
        mask = df.index <= date
        recent = df.loc[mask].tail(10)
        if len(recent) < 5:
            return "put"
        ret = float(recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1)
        # In uptrend, sell calls (richer); in downtrend, sell puts (richer)
        return "call" if ret > 0.01 else "put"

    def _rank_candidates(self, date):
        """Rank tickers by IV rank (higher = better for selling)."""
        candidates = []
        for ticker in self.universe:
            vol = self._get_vol(ticker, date)
            spot = self._get_spot(ticker, date)
            if spot is None or vol is None:
                continue
            # Simple IV rank proxy: current vol vs 90-day range
            vol_series = self.vol_data[ticker]
            mask = vol_series.index <= date
            recent_vols = vol_series.loc[mask].tail(90).dropna()
            if len(recent_vols) < 20:
                continue
            iv_rank = (vol - recent_vols.min()) / max(recent_vols.max() - recent_vols.min(), 0.01)
            candidates.append((ticker, float(iv_rank), vol, spot))

        # Sort by IV rank descending (sell premium when IV is high)
        candidates.sort(key=lambda x: -x[1])
        return candidates

    def _open_spread(self, ticker, date, spot, sigma, option_type):
        """Open a calendar spread."""
        strike = round(spot, 0)  # ATM, round to nearest dollar

        short_T = SHORT_DTE / 365.0
        long_T = LONG_DTE / 365.0

        short_price = bs_price(spot, strike, short_T, RISK_FREE_RATE, sigma, option_type)
        long_price = bs_price(spot, strike, long_T, RISK_FREE_RATE, sigma, option_type)

        # Net debit = long premium - short premium (should be positive)
        net_debit_per_contract = (long_price - short_price) * 100
        if net_debit_per_contract <= 0:
            return None  # Inverted, skip

        # Apply slippage: pay more for long, receive less for short
        slippage = net_debit_per_contract * SLIPPAGE_PCT
        net_debit_per_contract += slippage

        # Commission: opening both legs
        commission_per_contract = COMMISSION_PER_CONTRACT * 2  # 2 legs

        # Position sizing: max risk = debit paid, cap at MAX_RISK_PCT of NAV
        max_risk_dollars = self.capital * MAX_RISK_PCT
        cost_per_contract = net_debit_per_contract + commission_per_contract
        if cost_per_contract <= 0:
            return None

        num_contracts = max(1, int(max_risk_dollars / cost_per_contract))
        # Also cap by practical limits (don't buy 100 contracts of a $500 stock option)
        num_contracts = min(num_contracts, 10)

        total_debit = cost_per_contract * num_contracts
        total_commission = commission_per_contract * num_contracts

        if total_debit > self.cash * 0.5:
            # Don't use more than 50% of cash on one trade
            num_contracts = max(1, int(self.cash * 0.5 / cost_per_contract))
            total_debit = cost_per_contract * num_contracts
            total_commission = commission_per_contract * num_contracts

        if total_debit > self.cash:
            return None

        self.cash -= total_debit

        spread = CalendarSpread(
            ticker=ticker,
            entry_date=_to_date(date),
            strike=strike,
            spot=spot,
            short_dte=SHORT_DTE,
            long_dte=LONG_DTE,
            sigma=sigma,
            option_type=option_type,
            num_contracts=num_contracts,
            debit_paid=total_debit,
            commission=total_commission,
        )

        log.debug(f"  OPEN {ticker} {option_type} calendar @ {strike:.0f}, "
                  f"{num_contracts} contracts, debit=${total_debit:.2f}")
        return spread

    def _close_spread(self, spread, date, spot, sigma, reason):
        """Close a calendar spread."""
        pnl = spread.mark_to_market(date, spot, sigma)

        # Closing commission (close long leg; short may have expired)
        close_commission = COMMISSION_PER_CONTRACT * spread.num_contracts
        # Slippage on exit
        spread_value_approx = pnl + spread.debit_paid
        slippage = abs(spread_value_approx) * SLIPPAGE_PCT
        pnl -= (close_commission + slippage)

        spread.closed = True
        close_dt = _to_date(date)
        spread.close_date = close_dt
        spread.pnl = pnl
        spread.close_reason = reason

        self.cash += spread.debit_paid + pnl  # return capital + P&L
        self.capital += pnl

        self.trade_log.append({
            "ticker": spread.ticker,
            "type": spread.option_type,
            "strike": spread.strike,
            "entry_date": str(spread.entry_date),
            "close_date": str(close_dt),
            "num_contracts": spread.num_contracts,
            "debit_paid": round(spread.debit_paid, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / spread.debit_paid * 100, 2) if spread.debit_paid > 0 else 0,
            "reason": reason,
            "rolled": spread.rolled_count,
        })

        log.debug(f"  CLOSE {spread.ticker} reason={reason} pnl=${pnl:.2f}")

    def _roll_short_leg(self, spread, date, spot, sigma):
        """Roll the short leg forward by SHORT_DTE days."""
        old_short_T = max((spread.short_expiry - _to_date(date)).days / 365.0, 0)

        # Close expiring short (buy back at intrinsic + small time value)
        close_short_price = bs_price(spot, spread.strike, old_short_T,
                                     RISK_FREE_RATE, sigma, spread.option_type)

        # Open new short at same strike, new SHORT_DTE
        new_short_T = SHORT_DTE / 365.0
        new_short_price = bs_price(spot, spread.strike, new_short_T,
                                   RISK_FREE_RATE, sigma, spread.option_type)

        # Roll credit/debit
        roll_credit = (new_short_price - close_short_price) * 100 * spread.num_contracts
        roll_commission = COMMISSION_PER_CONTRACT * 2 * spread.num_contracts
        roll_slippage = abs(roll_credit) * SLIPPAGE_PCT

        net_roll = roll_credit - roll_commission - roll_slippage

        # Update spread
        cd = _to_date(date)
        spread.short_expiry = cd + dt.timedelta(days=SHORT_DTE)
        spread.debit_paid -= net_roll  # Reduce cost basis by roll credit
        spread.rolled_count += 1
        self.cash += net_roll

        log.debug(f"  ROLL {spread.ticker} short leg, credit=${net_roll:.2f}")

    def run(self):
        """Run the full backtest."""
        log.info("=" * 60)
        log.info("CALENDAR SPREAD INCOME BACKTEST v1")
        log.info("=" * 60)
        log.info(f"Capital: ${STARTING_CAPITAL:,.0f}")
        log.info(f"Universe: {len(self.universe)} tickers")
        log.info(f"Period: {self.trading_days[0].strftime('%Y-%m-%d')} to "
                 f"{self.trading_days[-1].strftime('%Y-%m-%d')}")

        # Entry every Monday (weekly cadence)
        entry_days = set()
        for d in self.trading_days:
            if d.weekday() == 0:  # Monday
                entry_days.add(d)

        for i, date in enumerate(self.trading_days):
            # Skip first 60 days for vol warmup
            if i < 60:
                self.nav_history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "nav": STARTING_CAPITAL,
                    "positions": 0,
                })
                continue

            # ── Manage existing positions ──
            positions_to_remove = []
            for j, spread in enumerate(self.positions):
                ticker = spread.ticker
                spot = self._get_spot(ticker, date)
                sigma = self._get_vol(ticker, date)
                if spot is None:
                    continue

                cd = _to_date(date)
                dte = (spread.short_expiry - cd).days

                # Check P&L
                current_pnl = spread.mark_to_market(cd, spot, sigma)

                # Profit target hit
                if current_pnl >= spread.profit_target:
                    self._close_spread(spread, date, spot, sigma, "profit_target")
                    positions_to_remove.append(j)
                    continue

                # Stop loss hit
                if current_pnl <= spread.stop_loss:
                    self._close_spread(spread, date, spot, sigma, "stop_loss")
                    positions_to_remove.append(j)
                    continue

                # Long option expiring
                long_dte = (spread.long_expiry - cd).days
                if long_dte <= 1:
                    self._close_spread(spread, date, spot, sigma, "long_expiry")
                    positions_to_remove.append(j)
                    continue

                # Roll short leg at 1 DTE
                if dte <= ROLL_AT_DTE and long_dte > SHORT_DTE + 2:
                    self._roll_short_leg(spread, date, spot, sigma)

            # Remove closed positions (reverse order)
            for j in sorted(positions_to_remove, reverse=True):
                self.closed_positions.append(self.positions.pop(j))

            # ── Open new positions on entry days ──
            if date in entry_days and len(self.positions) < MAX_CONCURRENT:
                candidates = self._rank_candidates(date)
                # Avoid duplicates
                open_tickers = {p.ticker for p in self.positions}

                for ticker, iv_rank, sigma, spot in candidates:
                    if len(self.positions) >= MAX_CONCURRENT:
                        break
                    if ticker in open_tickers:
                        continue
                    if iv_rank < 0.30:
                        continue  # Only sell premium when IV is elevated

                    option_type = self._select_option_type(ticker, date)
                    spread = self._open_spread(ticker, date, spot, sigma, option_type)
                    if spread is not None:
                        self.positions.append(spread)
                        open_tickers.add(ticker)

            # ── Record NAV ──
            total_unrealized = 0
            for spread in self.positions:
                spot = self._get_spot(spread.ticker, date)
                sigma = self._get_vol(spread.ticker, date)
                if spot:
                    cd = _to_date(date)
                    total_unrealized += spread.mark_to_market(cd, spot, sigma)

            nav = self.cash + total_unrealized
            self.nav_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "nav": round(nav, 2),
                "positions": len(self.positions),
            })

            # Monthly progress log
            if date.day == 1:
                log.info(f"  {date.strftime('%Y-%m')}: NAV=${nav:,.2f}, "
                         f"positions={len(self.positions)}, "
                         f"trades={len(self.trade_log)}")

        # Close any remaining positions at end
        for spread in self.positions:
            last_date = self.trading_days[-1]
            spot = self._get_spot(spread.ticker, last_date)
            sigma = self._get_vol(spread.ticker, last_date)
            if spot:
                self._close_spread(spread, last_date, spot, sigma, "backtest_end")
                self.closed_positions.append(spread)
        self.positions = []

        log.info(f"Backtest complete. {len(self.trade_log)} total trades.")

    def compute_metrics(self):
        """Compute all performance metrics."""
        nav_df = pd.DataFrame(self.nav_history)
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.set_index("date")
        nav_df["returns"] = nav_df["nav"].pct_change()

        # SPY benchmark
        spy_start_price = float(self.spy_data["Close"].iloc[0])
        spy_nav = self.spy_data["Close"] / spy_start_price * STARTING_CAPITAL
        spy_returns = self.spy_data["Close"].pct_change()

        # ── Core metrics ──
        total_return = (nav_df["nav"].iloc[-1] / STARTING_CAPITAL - 1) * 100
        years = len(nav_df) / 252
        cagr = ((nav_df["nav"].iloc[-1] / STARTING_CAPITAL) ** (1 / years) - 1) * 100

        daily_returns = nav_df["returns"].dropna()
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        downside = daily_returns[daily_returns < 0].std()
        sortino = float(daily_returns.mean() / downside * np.sqrt(252)) if downside > 0 else 0

        # Max drawdown
        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_dd = float(drawdown.min() * 100)

        # Trade stats
        trades_df = pd.DataFrame(self.trade_log)
        if len(trades_df) > 0:
            win_rate = (trades_df["pnl"] > 0).mean() * 100
            avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() if (trades_df["pnl"] > 0).any() else 0
            avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean() if (trades_df["pnl"] < 0).any() else 0
            profit_factor = abs(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum() /
                               trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()) if (trades_df["pnl"] < 0).any() else float("inf")
            total_pnl = trades_df["pnl"].sum()
            avg_pnl = trades_df["pnl"].mean()
            total_commission = trades_df.get("debit_paid", pd.Series([0])).sum() * 0  # commissions already in PnL
        else:
            win_rate = avg_win = avg_loss = profit_factor = total_pnl = avg_pnl = 0

        # Monthly income
        if len(trades_df) > 0:
            trades_df["close_month"] = pd.to_datetime(trades_df["close_date"]).dt.to_period("M")
            monthly_pnl = trades_df.groupby("close_month")["pnl"].sum()
            avg_monthly_income = float(monthly_pnl.mean())
            median_monthly_income = float(monthly_pnl.median())
        else:
            avg_monthly_income = median_monthly_income = 0

        # SPY benchmark metrics
        spy_total_return = (float(self.spy_data["Close"].iloc[-1]) / spy_start_price - 1) * 100
        spy_daily = spy_returns.dropna()
        spy_sharpe = float(spy_daily.mean() / spy_daily.std() * np.sqrt(252)) if spy_daily.std() > 0 else 0

        # ── Regime Analysis (HC #428 R1) ──
        spy_close = self.spy_data["Close"].copy()
        spy_daily_ret = spy_close.pct_change()

        # Classify days
        regime = pd.Series(index=spy_daily_ret.index, dtype=str)
        regime[spy_daily_ret > 0.001] = "green"
        regime[spy_daily_ret < -0.001] = "red"
        regime[(spy_daily_ret >= -0.001) & (spy_daily_ret <= 0.001)] = "flat"

        # Map strategy returns to regimes
        strat_returns = nav_df["returns"].copy()
        strat_returns.index = pd.to_datetime(strat_returns.index)
        regime.index = pd.to_datetime(regime.index)

        merged = pd.DataFrame({"strat_ret": strat_returns, "regime": regime}).dropna()

        regime_sharpes = {}
        for r in ["green", "red", "flat"]:
            r_rets = merged.loc[merged["regime"] == r, "strat_ret"]
            if len(r_rets) > 20 and r_rets.std() > 0:
                regime_sharpes[r] = float(r_rets.mean() / r_rets.std() * np.sqrt(252))
            else:
                regime_sharpes[r] = 0.0

        # Regime agnostic test
        s_green = regime_sharpes.get("green", 0)
        s_red = regime_sharpes.get("red", 0)
        max_abs = max(abs(s_green), abs(s_red), 0.001)
        regime_divergence = abs(s_green - s_red) / max_abs
        regime_agnostic_pass = regime_divergence <= 0.50

        # ── Permutation Test ──
        log.info("Running permutation test (100 shuffles)...")
        actual_sharpe = sharpe
        if len(trades_df) > 0:
            pnl_array = trades_df["pnl"].values
            better_count = 0
            rng = np.random.RandomState(42)
            for _ in range(PERMUTATION_SHUFFLES):
                shuffled = rng.permutation(pnl_array)
                # Reconstruct equity curve from shuffled PnLs
                eq = np.cumsum(shuffled) + STARTING_CAPITAL
                eq_ret = np.diff(eq) / eq[:-1]
                if len(eq_ret) > 1 and np.std(eq_ret) > 0:
                    perm_sharpe = np.mean(eq_ret) / np.std(eq_ret) * np.sqrt(252)
                else:
                    perm_sharpe = 0
                if perm_sharpe >= actual_sharpe:
                    better_count += 1
            p_value = better_count / PERMUTATION_SHUFFLES
        else:
            p_value = 1.0

        # ── Capital needed for targets ──
        if avg_monthly_income > 0:
            capital_for_3k = STARTING_CAPITAL * (3000 / avg_monthly_income)
            capital_for_5k = STARTING_CAPITAL * (5000 / avg_monthly_income)
        else:
            capital_for_3k = capital_for_5k = float("inf")

        # ── Compile results ──
        results = {
            "strategy": "Calendar Spread Income v1",
            "period": f"{nav_df.index[0].strftime('%Y-%m-%d')} to {nav_df.index[-1].strftime('%Y-%m-%d')}",
            "starting_capital": STARTING_CAPITAL,
            "ending_nav": round(float(nav_df["nav"].iloc[-1]), 2),
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
            "total_trades": len(self.trade_log),
            "win_rate_pct": round(float(win_rate), 1),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "avg_pnl_per_trade": round(float(avg_pnl), 2),
            "total_pnl": round(float(total_pnl), 2),
            "avg_monthly_income": round(avg_monthly_income, 2),
            "median_monthly_income": round(median_monthly_income, 2),
            "capital_needed_3k_monthly": round(capital_for_3k, 0) if capital_for_3k != float("inf") else "N/A",
            "capital_needed_5k_monthly": round(capital_for_5k, 0) if capital_for_5k != float("inf") else "N/A",
            "benchmark_spy_total_return_pct": round(spy_total_return, 2),
            "benchmark_spy_sharpe": round(spy_sharpe, 3),
            "regime_analysis": {
                "sharpe_green_days": round(s_green, 3),
                "sharpe_red_days": round(s_red, 3),
                "sharpe_flat_days": round(regime_sharpes.get("flat", 0), 3),
                "regime_divergence": round(regime_divergence, 3),
                "regime_agnostic_PASS": regime_agnostic_pass,
            },
            "permutation_test": {
                "p_value": round(p_value, 3),
                "shuffles": PERMUTATION_SHUFFLES,
                "significant_at_05": p_value < 0.05,
            },
            "parameters": {
                "short_dte": SHORT_DTE,
                "long_dte": LONG_DTE,
                "profit_target_pct": PROFIT_TARGET_PCT,
                "stop_loss_pct": STOP_LOSS_PCT,
                "max_concurrent": MAX_CONCURRENT,
                "max_risk_pct": MAX_RISK_PCT,
                "slippage_pct": SLIPPAGE_PCT,
                "commission_per_contract": COMMISSION_PER_CONTRACT,
                "universe_size": len(self.universe),
            },
            "monthly_pnl_by_year": {},
            "trade_log_sample": self.trade_log[:20],
        }

        # Monthly PnL by year
        if len(trades_df) > 0:
            trades_df["close_year"] = pd.to_datetime(trades_df["close_date"]).dt.year
            for year in sorted(trades_df["close_year"].unique()):
                year_trades = trades_df[trades_df["close_year"] == year]
                year_trades_m = year_trades.copy()
                year_trades_m["month"] = pd.to_datetime(year_trades_m["close_date"]).dt.month
                monthly = year_trades_m.groupby("month")["pnl"].sum()
                results["monthly_pnl_by_year"][str(int(year))] = {
                    str(int(m)): round(float(v), 2) for m, v in monthly.items()
                }

        return results

    def print_summary(self, results):
        """Print plain-English summary."""
        print("\n" + "=" * 70)
        print("CALENDAR SPREAD INCOME STRATEGY — BACKTEST RESULTS")
        print("=" * 70)
        print(f"\nPeriod: {results['period']}")
        print(f"Starting Capital: ${results['starting_capital']:,.0f}")
        print(f"Ending NAV: ${results['ending_nav']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.1f}%")
        print(f"CAGR: {results['cagr_pct']:.1f}%")
        print(f"\n--- Risk-Adjusted Performance ---")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']}")
        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"Avg Win: ${results['avg_win']:.2f}")
        print(f"Avg Loss: ${results['avg_loss']:.2f}")
        print(f"Avg P&L/Trade: ${results['avg_pnl_per_trade']:.2f}")
        print(f"\n--- Income Analysis ---")
        print(f"Avg Monthly Income (@$100K): ${results['avg_monthly_income']:,.2f}")
        print(f"Median Monthly Income (@$100K): ${results['median_monthly_income']:,.2f}")
        print(f"Capital Needed for $3K/month: ${results['capital_needed_3k_monthly']:,}"
              if isinstance(results['capital_needed_3k_monthly'], (int, float)) else
              f"Capital Needed for $3K/month: {results['capital_needed_3k_monthly']}")
        print(f"Capital Needed for $5K/month: ${results['capital_needed_5k_monthly']:,}"
              if isinstance(results['capital_needed_5k_monthly'], (int, float)) else
              f"Capital Needed for $5K/month: {results['capital_needed_5k_monthly']}")
        print(f"\n--- Benchmark (SPY Buy & Hold) ---")
        print(f"SPY Total Return: {results['benchmark_spy_total_return_pct']:.1f}%")
        print(f"SPY Sharpe: {results['benchmark_spy_sharpe']:.3f}")
        print(f"\n--- Regime Analysis (HC #428 R1) ---")
        ra = results['regime_analysis']
        print(f"Sharpe on Green Days: {ra['sharpe_green_days']:.3f}")
        print(f"Sharpe on Red Days: {ra['sharpe_red_days']:.3f}")
        print(f"Sharpe on Flat Days: {ra['sharpe_flat_days']:.3f}")
        print(f"Regime Divergence: {ra['regime_divergence']:.3f} (threshold: 0.50)")
        print(f"Regime-Agnostic Test: {'PASS' if ra['regime_agnostic_PASS'] else 'FAIL'}")
        print(f"\n--- Permutation Test ---")
        pt = results['permutation_test']
        print(f"p-value: {pt['p_value']:.3f} ({'Significant' if pt['significant_at_05'] else 'Not significant'} at 0.05)")
        print(f"\n--- Monthly P&L by Year ---")
        for year, months in sorted(results.get("monthly_pnl_by_year", {}).items()):
            total = sum(months.values())
            print(f"  {year}: ${total:,.2f} total | " +
                  " ".join(f"M{m}:${v:,.0f}" for m, v in sorted(months.items(), key=lambda x: int(x[0]))))

        print("\n" + "=" * 70)
        verdict = "VIABLE" if (results["sharpe_ratio"] > 0.5 and
                               ra["regime_agnostic_PASS"] and
                               results["win_rate_pct"] > 45) else "NEEDS WORK"
        print(f"VERDICT: {verdict}")
        if results["sharpe_ratio"] <= 0.5:
            print("  - Sharpe below 0.5, risk-adjusted returns are weak")
        if not ra["regime_agnostic_PASS"]:
            print("  - FAILED regime-agnostic test (strategy is regime-dependent)")
        if results["win_rate_pct"] <= 45:
            print("  - Win rate below 45%, too many losing trades")
        if results["max_drawdown_pct"] < -20:
            print("  - Max drawdown exceeds 20%, risk is too high")
        print("=" * 70)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Calendar Spread Income Backtest v1")

    # Load data
    price_data = load_price_data(UNIVERSE, START_DATE, END_DATE)

    if len(price_data) < 5:
        log.error("Not enough tickers loaded. Aborting.")
        sys.exit(1)

    # Run backtest
    bt = CalendarSpreadBacktest(price_data, UNIVERSE)
    bt.run()

    # Compute and save results
    results = bt.compute_metrics()

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {RESULTS_FILE}")

    # Save NAV history
    nav_df = pd.DataFrame(bt.nav_history)
    nav_df.to_csv(OUTPUT_DIR / "nav_history.csv", index=False)

    # Save full trade log
    trades_df = pd.DataFrame(bt.trade_log)
    if len(trades_df) > 0:
        trades_df.to_csv(OUTPUT_DIR / "trade_log.csv", index=False)

    # Print summary
    bt.print_summary(results)

    log.info("Backtest complete.")
