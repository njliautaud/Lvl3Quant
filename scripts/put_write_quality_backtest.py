#!/usr/bin/env python3
"""
Cash-Secured Put Writing on Quality Stocks After Dips
=====================================================
SIMULATED — requires real options data validation.

6 variants:
  A: Sell ATM puts monthly on all quality stocks (no timing)
  B: Sell OTM (-5%) puts on stocks that dipped >3% in last 5 days
  C: Sell OTM (-5%) puts on stocks with RSI(14) < 40
  D: Sell ATM puts only when SPY < 200-SMA (bear market)
  E: Dual Signal D — dip >5% + RSI<35 + green-after-reds (most selective)
  F: Wheel — sell puts until assigned, then sell covered calls until called away

Synthetic premium: max(0.5%, IV_approx * sqrt(DTE/365) * delta_approx) * strike
Conservative IV = 30-day HV * 1.1, delta ~0.30 for OTM, ~0.50 for ATM.
"""

import json
import warnings
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
MAX_TRADE_NOTIONAL = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 5  # 5 bps for options
COMMISSION_PER_LEG = 0.65
DTE = 30
OTM_PCT = 0.05
IV_PREMIUM_FACTOR = 1.1
HOLD_DAYS_IF_ASSIGNED = 10

START_DATE = "2022-01-01"
END_DATE = "2026-07-31"

UNIVERSE = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "KO"]

# Validation gates
SHARPE_GATE = 0.5
MAX_DD_GATE = -0.50
MIN_TRADES = 20
REGIME_GAP_GATE = 0.50
PERM_PVAL_GATE = 0.05
N_PERMUTATIONS = 1000


# ── Helpers ────────────────────────────────────────────────────────────
def compute_hv30(prices: pd.Series) -> float:
    """30-day historical volatility (annualized)."""
    rets = np.log(prices / prices.shift(1)).dropna()
    if len(rets) < 20:
        return 0.30  # default
    return float(rets.tail(30).std() * np.sqrt(252))


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def synthetic_put_premium(spot: float, strike: float, hv30: float,
                          dte: int = 30, is_atm: bool = False) -> float:
    """
    Conservative synthetic put premium.
    premium = max(0.5%, IV * sqrt(DTE/365) * delta) * strike
    """
    iv = hv30 * IV_PREMIUM_FACTOR
    if iv < 0.10:
        iv = 0.10  # floor
    t = dte / 365.0
    delta = 0.50 if is_atm else 0.30
    prem_pct = iv * np.sqrt(t) * delta
    prem_pct = max(0.005, prem_pct)  # floor at 0.5%
    premium = prem_pct * strike
    # Apply bid-ask haircut — we sell at 80% of theoretical
    premium *= 0.80
    return premium


def apply_slippage(price: float, is_buy: bool) -> float:
    """Apply slippage to stock price."""
    slip = price * SLIPPAGE_BPS / 10000.0
    return price + slip if is_buy else price - slip


# ── Data Loading ───────────────────────────────────────────────────────
def load_data() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load price data for universe + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading data for {tickers}...")
    data = {}
    for t in tickers:
        df = yf.download(t, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 0:
            data[t] = df
    spy = data.pop("SPY")
    print(f"Loaded {len(data)} stocks, SPY has {len(spy)} days")
    return data, spy


# ── Trade Tracking ─────────────────────────────────────────────────────
@dataclass
class PutTrade:
    ticker: str
    entry_date: str
    expiry_date: str
    spot_at_entry: float
    strike: float
    premium_collected: float
    notional: float  # strike * fraction_of_contract
    is_atm: bool
    assigned: bool = False
    exit_date: str = ""
    pnl: float = 0.0
    status: str = "open"


@dataclass
class WheelPosition:
    ticker: str
    shares_notional: float  # fractional notional
    entry_price: float
    entry_date: str
    calls_sold: int = 0
    call_premium: float = 0.0


# ── Strategy Engines ───────────────────────────────────────────────────
class PutWriteBacktest:
    def __init__(self, variant: str, stock_data: Dict[str, pd.DataFrame],
                 spy_data: pd.DataFrame):
        self.variant = variant
        self.stock_data = stock_data
        self.spy = spy_data
        self.capital = INITIAL_CAPITAL
        self.trades: List[PutTrade] = []
        self.open_trades: List[PutTrade] = []
        self.wheel_positions: Dict[str, WheelPosition] = {}
        self.equity_curve = []
        self.daily_returns = []

        # Precompute SPY 200-SMA
        self.spy_sma200 = self.spy["Close"].rolling(200).mean()

        # Precompute per-stock indicators
        self.rsi = {}
        self.hv = {}
        for ticker, df in self.stock_data.items():
            self.rsi[ticker] = compute_rsi(df["Close"])
            # Rolling 30-day HV
            rets = np.log(df["Close"] / df["Close"].shift(1))
            self.hv[ticker] = rets.rolling(30).std() * np.sqrt(252)

    def _get_common_dates(self) -> pd.DatetimeIndex:
        """Get trading dates where SPY + all stocks have data."""
        dates = self.spy.index
        for df in self.stock_data.values():
            dates = dates.intersection(df.index)
        # Start after enough warmup
        warmup_date = dates[0] + pd.Timedelta(days=210)  # 200-SMA + buffer
        return dates[dates >= warmup_date]

    def _check_signal(self, ticker: str, date, df: pd.DataFrame) -> bool:
        """Check if signal fires for this variant on this date."""
        if date not in df.index:
            return False
        idx = df.index.get_loc(date)
        if idx < 30:
            return False

        if self.variant == "A":
            # Monthly: sell on first trading day each month
            if idx > 0:
                prev_date = df.index[idx - 1]
                return date.month != prev_date.month
            return False

        elif self.variant == "B":
            # Dip >3% in last 5 days
            if idx < 5:
                return False
            p5 = df["Close"].iloc[idx - 5]
            p_now = df["Close"].iloc[idx]
            dip = (p_now - p5) / p5
            return dip < -0.03

        elif self.variant == "C":
            # RSI(14) < 40
            rsi_val = self.rsi[ticker].iloc[idx]
            return not np.isnan(rsi_val) and rsi_val < 40

        elif self.variant == "D":
            # SPY < 200-SMA, monthly
            if date not in self.spy.index:
                return False
            spy_close = self.spy.loc[date, "Close"]
            spy_sma = self.spy_sma200.loc[date]
            if np.isnan(spy_sma):
                return False
            if spy_close >= spy_sma:
                return False
            # Monthly cadence
            if idx > 0:
                prev_date = df.index[idx - 1]
                return date.month != prev_date.month
            return False

        elif self.variant == "E":
            # Dual Signal D: dip >5% + RSI<35 + green after reds
            if idx < 5:
                return False
            p5 = df["Close"].iloc[idx - 5]
            p_now = df["Close"].iloc[idx]
            dip = (p_now - p5) / p5
            if dip >= -0.05:
                return False
            rsi_val = self.rsi[ticker].iloc[idx]
            if np.isnan(rsi_val) or rsi_val >= 35:
                return False
            # Green after reds: today green, prior 2 days red
            if idx < 3:
                return False
            today_green = df["Close"].iloc[idx] > df["Open"].iloc[idx]
            yest_red = df["Close"].iloc[idx-1] < df["Open"].iloc[idx-1]
            day2_red = df["Close"].iloc[idx-2] < df["Open"].iloc[idx-2]
            return today_green and yest_red and day2_red

        elif self.variant == "F":
            # Wheel: same entry as B (dip >3%) but with wheel mechanics
            if idx < 5:
                return False
            p5 = df["Close"].iloc[idx - 5]
            p_now = df["Close"].iloc[idx]
            dip = (p_now - p5) / p5
            return dip < -0.03

        return False

    def run(self):
        """Run the backtest."""
        dates = self._get_common_dates()
        prev_equity = INITIAL_CAPITAL

        for date in dates:
            # 1. Check expiring trades
            self._check_expiries(date)

            # 2. Check for new signals (if capacity available)
            if len(self.open_trades) < MAX_CONCURRENT:
                for ticker, df in self.stock_data.items():
                    if len(self.open_trades) >= MAX_CONCURRENT:
                        break
                    # Skip if already have open trade on this ticker
                    if any(t.ticker == ticker for t in self.open_trades):
                        continue
                    # For wheel: skip if we hold shares (we'd sell calls instead)
                    if self.variant == "F" and ticker in self.wheel_positions:
                        self._sell_covered_call(ticker, date, df)
                        continue

                    if self._check_signal(ticker, date, df):
                        self._open_put(ticker, date, df)

            # 3. Check wheel positions for call assignment
            if self.variant == "F":
                self._check_wheel_calls(date)

            # 4. Track equity
            # Collateral locked in open puts (will be returned at expiry)
            locked_collateral = sum(t.notional for t in self.open_trades)
            # Wheel stock value (mark to market)
            wheel_value = sum(
                wp.shares_notional * (self.stock_data[wp.ticker].loc[date, "Close"]
                                      / wp.entry_price)
                for wp in self.wheel_positions.values()
                if date in self.stock_data[wp.ticker].index
            )
            equity = self.capital + locked_collateral + wheel_value
            self.equity_curve.append({
                "date": str(date.date()),
                "equity": equity
            })
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_ret)
            prev_equity = equity

    def _open_put(self, ticker: str, date, df: pd.DataFrame):
        """Open a new put sale."""
        if date not in df.index:
            return
        spot = float(df.loc[date, "Close"])
        idx = df.index.get_loc(date)

        # ATM vs OTM
        is_atm = self.variant in ("A", "D")
        strike = spot if is_atm else spot * (1 - OTM_PCT)

        # Get HV
        hv_val = self.hv[ticker].iloc[idx]
        if np.isnan(hv_val) or hv_val <= 0:
            hv_val = 0.30

        # Premium
        premium_per_share = synthetic_put_premium(spot, strike, hv_val,
                                                   DTE, is_atm)

        # Size: fractional — FIXED $200 notional cap (no compounding)
        notional = min(MAX_TRADE_NOTIONAL, INITIAL_CAPITAL * 0.33)
        if notional < 10:
            return
        fraction = notional / (strike * 100)  # fraction of 1 contract
        premium = premium_per_share * 100 * fraction
        commission = COMMISSION_PER_LEG * 2 * fraction  # open + close

        # Net premium after commission
        net_premium = premium - commission
        if net_premium <= 0:
            return

        expiry = date + pd.Timedelta(days=DTE)

        trade = PutTrade(
            ticker=ticker,
            entry_date=str(date.date()),
            expiry_date=str(expiry.date()),
            spot_at_entry=spot,
            strike=strike,
            premium_collected=net_premium,
            notional=notional,
            is_atm=is_atm,
        )
        # Reserve collateral: cash-secured means we set aside the notional
        if self.capital < notional - net_premium:
            return  # can't afford the collateral
        self.open_trades.append(trade)
        # Capital effect: receive premium but lock up collateral
        self.capital -= (notional - net_premium)  # net cash outflow = collateral - premium

    def _check_expiries(self, date):
        """Check if any open trades have expired."""
        still_open = []
        for trade in self.open_trades:
            expiry = pd.Timestamp(trade.expiry_date)
            if date >= expiry:
                ticker = trade.ticker
                df = self.stock_data[ticker]
                # Find closest date to expiry
                valid_dates = df.index[df.index <= date]
                if len(valid_dates) == 0:
                    still_open.append(trade)
                    continue
                exp_date = valid_dates[-1]
                exp_price = float(df.loc[exp_date, "Close"])

                if exp_price < trade.strike:
                    # Assigned — we buy stock at strike price
                    trade.assigned = True
                    buy_price = apply_slippage(trade.strike, is_buy=True)

                    if self.variant == "F":
                        # Wheel: hold and sell calls
                        # Collateral stays locked (now it's stock)
                        wp = WheelPosition(
                            ticker=ticker,
                            shares_notional=trade.notional,
                            entry_price=buy_price,
                            entry_date=str(date.date()),
                        )
                        self.wheel_positions[ticker] = wp
                        trade.pnl = trade.premium_collected  # premium kept
                    else:
                        # Hold for HOLD_DAYS then sell
                        sell_date_target = date + pd.Timedelta(days=HOLD_DAYS_IF_ASSIGNED)
                        valid_sell = df.index[df.index >= sell_date_target]
                        if len(valid_sell) > 0:
                            sell_date = valid_sell[0]
                            sell_price = apply_slippage(
                                float(df.loc[sell_date, "Close"]), is_buy=False)
                        else:
                            sell_price = apply_slippage(exp_price, is_buy=False)
                            sell_date = exp_date

                        # PnL = premium + (sell_price - strike) * shares
                        shares_fraction = trade.notional / trade.strike
                        stock_pnl = (sell_price - buy_price) * shares_fraction
                        trade.pnl = trade.premium_collected + stock_pnl
                        # Release collateral + stock gain/loss
                        self.capital += trade.notional + stock_pnl
                        trade.exit_date = str(sell_date.date())
                else:
                    # Expired worthless — release collateral, keep premium
                    trade.pnl = trade.premium_collected
                    self.capital += trade.notional  # release collateral
                    trade.exit_date = str(exp_date.date())

                trade.status = "closed"
                self.trades.append(trade)
            else:
                still_open.append(trade)
        self.open_trades = still_open

    def _sell_covered_call(self, ticker: str, date, df: pd.DataFrame):
        """Sell a covered call on wheel position (monthly cadence)."""
        if ticker not in self.wheel_positions:
            return
        wp = self.wheel_positions[ticker]
        idx = df.index.get_loc(date)
        if idx < 1:
            return
        # Monthly cadence
        prev_date = df.index[idx - 1]
        if date.month == prev_date.month:
            return

        spot = float(df.loc[date, "Close"])
        hv_val = self.hv[ticker].iloc[idx]
        if np.isnan(hv_val) or hv_val <= 0:
            hv_val = 0.30

        # ATM call premium (symmetric to put for simplicity)
        call_strike = spot * 1.05  # 5% OTM call
        iv = hv_val * IV_PREMIUM_FACTOR
        t = DTE / 365.0
        call_prem = max(0.005, iv * np.sqrt(t) * 0.30) * call_strike * 0.80
        fraction = wp.shares_notional / (call_strike * 100)
        net_prem = call_prem * 100 * fraction - COMMISSION_PER_LEG * 2 * fraction
        if net_prem > 0:
            wp.calls_sold += 1
            wp.call_premium += net_prem
            self.capital += net_prem

    def _check_wheel_calls(self, date):
        """Check if any wheel positions should be called away."""
        to_remove = []
        for ticker, wp in self.wheel_positions.items():
            df = self.stock_data[ticker]
            if date not in df.index:
                continue
            entry_dt = pd.Timestamp(wp.entry_date)
            # Hold at least 30 days, then check if stock recovered above entry
            if (date - entry_dt).days >= 30:
                spot = float(df.loc[date, "Close"])
                if spot >= wp.entry_price * 1.05:
                    # Called away — sell at 5% above entry
                    sell_price = apply_slippage(spot, is_buy=False)
                    shares_fraction = wp.shares_notional / wp.entry_price
                    stock_pnl = (sell_price - wp.entry_price) * shares_fraction
                    self.capital += wp.shares_notional + stock_pnl
                    # Record as a trade
                    t = PutTrade(
                        ticker=ticker,
                        entry_date=wp.entry_date,
                        expiry_date="",
                        spot_at_entry=wp.entry_price,
                        strike=wp.entry_price,
                        premium_collected=wp.call_premium,
                        notional=wp.shares_notional,
                        is_atm=True,
                        assigned=True,
                        exit_date=str(date.date()),
                        pnl=wp.call_premium + stock_pnl,
                        status="wheel_closed",
                    )
                    self.trades.append(t)
                    to_remove.append(ticker)
        for t in to_remove:
            del self.wheel_positions[t]

    def results(self) -> Dict:
        """Compute strategy results."""
        if not self.trades:
            return {"variant": self.variant, "n_trades": 0, "sharpe": 0,
                    "total_return_pct": 0, "gates": {}}

        pnls = [t.pnl for t in self.trades]
        n = len(pnls)
        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / n if n > 0 else 0
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        profit_factor = (abs(avg_win * sum(1 for p in pnls if p > 0)) /
                         abs(avg_loss * sum(1 for p in pnls if p < 0))
                         if avg_loss != 0 and sum(1 for p in pnls if p < 0) > 0 else float('inf'))

        # Equity curve stats
        eq = [e["equity"] for e in self.equity_curve]
        if len(eq) < 2:
            return {"variant": self.variant, "n_trades": n, "sharpe": 0,
                    "total_return_pct": 0, "gates": {}}

        daily_rets = pd.Series(self.daily_returns)
        sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                  if daily_rets.std() > 0 else 0)

        # Sortino
        downside = daily_rets[daily_rets < 0]
        sortino = (daily_rets.mean() / downside.std() * np.sqrt(252)
                   if len(downside) > 0 and downside.std() > 0 else 0)

        # Max drawdown
        eq_series = pd.Series(eq)
        rolling_max = eq_series.cummax()
        dd = (eq_series - rolling_max) / rolling_max
        max_dd = float(dd.min())

        total_return = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
        ann_return = total_return / max(1, len(eq) / 252)

        # Assignment rate
        n_assigned = sum(1 for t in self.trades if t.assigned)
        assign_rate = n_assigned / n if n > 0 else 0

        # ── Regime analysis ──
        spy_close = self.spy["Close"]
        spy_sma = spy_close.rolling(200).mean()
        regime_pnls = {"bull": [], "bear": []}
        for t in self.trades:
            td = pd.Timestamp(t.entry_date)
            if td in spy_close.index and td in spy_sma.index:
                is_bull = spy_close.loc[td] >= spy_sma.loc[td]
                regime_pnls["bull" if is_bull else "bear"].append(t.pnl)

        bull_sharpe = 0
        bear_sharpe = 0
        if len(regime_pnls["bull"]) > 5:
            b = pd.Series(regime_pnls["bull"])
            bull_sharpe = b.mean() / b.std() * np.sqrt(12) if b.std() > 0 else 0
        if len(regime_pnls["bear"]) > 5:
            b = pd.Series(regime_pnls["bear"])
            bear_sharpe = b.mean() / b.std() * np.sqrt(12) if b.std() > 0 else 0

        regime_gap = (abs(bull_sharpe - bear_sharpe) /
                      max(abs(bull_sharpe), abs(bear_sharpe), 0.01))

        # ── Permutation test ──
        perm_pval = self._permutation_test(pnls)

        # ── Gates ──
        gates = {
            "sharpe_pass": sharpe > SHARPE_GATE,
            "permutation_pass": perm_pval < PERM_PVAL_GATE,
            "regime_gap_pass": regime_gap < REGIME_GAP_GATE,
            "max_dd_pass": max_dd > MAX_DD_GATE,
            "min_trades_pass": n >= MIN_TRADES,
        }
        gates_passed = sum(gates.values())

        return {
            "variant": self.variant,
            "n_trades": n,
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return * 100, 2),
            "ann_return_pct": round(ann_return * 100, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "profit_factor": round(profit_factor, 3),
            "win_rate": round(win_rate * 100, 1),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_pnl": round(np.mean(pnls), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "assignment_rate_pct": round(assign_rate * 100, 1),
            "n_assigned": n_assigned,
            "bull_sharpe": round(bull_sharpe, 3),
            "bear_sharpe": round(bear_sharpe, 3),
            "regime_gap": round(regime_gap, 3),
            "perm_pval": round(perm_pval, 4),
            "gates": gates,
            "gates_passed": f"{gates_passed}/5",
            "final_equity": round(eq[-1], 2),
        }

    def _permutation_test(self, pnls: List[float]) -> float:
        """
        Permutation test: generate random entry returns of same count/hold
        from the stock universe, compare to observed mean PnL.
        Tests whether our TIMING produces better returns than random entries.
        """
        if len(pnls) < 5:
            return 1.0
        observed_mean = np.mean(pnls)
        n_trades = len(pnls)
        rng = np.random.default_rng(42)

        # Collect all possible DTE-period returns across the universe
        all_period_returns = []
        for ticker, df in self.stock_data.items():
            closes = df["Close"].values
            for i in range(len(closes) - DTE):
                ret = (closes[i + DTE] - closes[i]) / closes[i]
                all_period_returns.append(ret)
        all_period_returns = np.array(all_period_returns)
        if len(all_period_returns) < n_trades:
            return 1.0

        # For each permutation, sample random entries and compute avg premium + stock PnL
        # A random put sale: premium collected minus assignment loss
        avg_premium_pct = np.mean([t.premium_collected / max(t.notional, 1)
                                   for t in self.trades])
        count_better = 0
        for _ in range(N_PERMUTATIONS):
            idxs = rng.choice(len(all_period_returns), size=n_trades, replace=True)
            random_rets = all_period_returns[idxs]
            # Simulate: collect premium, lose on assignment (stock below strike)
            random_pnls = []
            for r in random_rets:
                prem = avg_premium_pct * MAX_TRADE_NOTIONAL
                if r < -OTM_PCT:  # assigned
                    stock_loss = r * MAX_TRADE_NOTIONAL
                    random_pnls.append(prem + stock_loss)
                else:
                    random_pnls.append(prem)
            if np.mean(random_pnls) >= observed_mean:
                count_better += 1
        return count_better / N_PERMUTATIONS


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Cash-Secured Put Writing on Quality Stocks — Backtest")
    print("SIMULATED — requires real options data validation")
    print("=" * 70)

    stock_data, spy_data = load_data()

    variants = ["A", "B", "C", "D", "E", "F"]
    all_results = {}

    for v in variants:
        print(f"\n{'─'*50}")
        print(f"Running Variant {v}...")
        bt = PutWriteBacktest(v, stock_data, spy_data)
        bt.run()
        res = bt.results()
        all_results[v] = res

        print(f"  Trades: {res['n_trades']}")
        print(f"  Total Return: {res.get('total_return_pct', 0):.1f}%")
        print(f"  Sharpe: {res.get('sharpe', 0):.3f}")
        print(f"  Sortino: {res.get('sortino', 0):.3f}")
        print(f"  Win Rate: {res.get('win_rate', 0):.1f}%")
        print(f"  Max DD: {res.get('max_drawdown_pct', 0):.1f}%")
        print(f"  Profit Factor: {res.get('profit_factor', 0):.3f}")
        print(f"  Assignment Rate: {res.get('assignment_rate_pct', 0):.1f}%")
        print(f"  Gates: {res.get('gates_passed', '0/5')}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY — SIMULATED (requires real options data validation)")
    print(f"{'='*70}")
    print(f"{'Var':<4} {'Trades':<7} {'Return%':<9} {'Sharpe':<8} {'Sortino':<8} "
          f"{'WR%':<6} {'PF':<6} {'MaxDD%':<8} {'Assign%':<8} {'Gates':<6}")
    print("-" * 70)
    for v in variants:
        r = all_results[v]
        print(f"{v:<4} {r['n_trades']:<7} {r.get('total_return_pct',0):>7.1f}  "
              f"{r.get('sharpe',0):>6.3f}  {r.get('sortino',0):>6.3f}  "
              f"{r.get('win_rate',0):>5.1f} {r.get('profit_factor',0):>5.2f} "
              f"{r.get('max_drawdown_pct',0):>7.1f} {r.get('assignment_rate_pct',0):>7.1f} "
              f"{r.get('gates_passed','0/5')}")

    # ── Best variant ──
    best_v = max(variants,
                 key=lambda v: all_results[v].get("sharpe", -99)
                 if all_results[v]["n_trades"] >= MIN_TRADES else -99)
    best = all_results[best_v]
    print(f"\nBest variant: {best_v} (Sharpe={best.get('sharpe',0):.3f})")

    # ── Gate detail ──
    print(f"\n{'='*70}")
    print("5-GATE VALIDATION DETAIL")
    print(f"{'='*70}")
    for v in variants:
        r = all_results[v]
        g = r.get("gates", {})
        print(f"\nVariant {v}: {r.get('gates_passed', '0/5')}")
        for gate_name, passed in g.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {gate_name}")

    # ── Save results ──
    output = {
        "strategy": "Cash-Secured Put Writing on Quality Stocks",
        "note": "SIMULATED — requires real options data validation. "
                "Focus on RELATIVE comparison between variants.",
        "period": f"{START_DATE} to {END_DATE}",
        "universe": UNIVERSE,
        "capital": INITIAL_CAPITAL,
        "max_trade_notional": MAX_TRADE_NOTIONAL,
        "assumptions": {
            "iv_estimate": "30-day HV * 1.1",
            "bid_ask_haircut": "80% of theoretical",
            "slippage_bps": SLIPPAGE_BPS,
            "commission_per_leg": COMMISSION_PER_LEG,
            "dte": DTE,
            "otm_pct": OTM_PCT,
            "hold_days_if_assigned": HOLD_DAYS_IF_ASSIGNED,
        },
        "variants": {v: all_results[v] for v in variants},
        "best_variant": best_v,
    }

    out_path = "/home/jupiter/Lvl3Quant/data/put_write_quality_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
