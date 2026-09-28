#!/usr/bin/env python3
"""
Deep Rotation + Money Flow Strategy Backtest v1
================================================
8 strategy variants using sector ETF rotation with options overlay.
Walk-forward OOT: Jan 2022 - Jul 2026.
Target: 5-10x returns on $645 account.

Core thesis: Money moves between sectors BEFORE price does.
Volume surges, unusual options activity, and price-volume divergences
are leading indicators of rotation.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLB", "XLRE"]
BENCHMARK = "SPY"
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

INITIAL_CAPITAL = 645.0
MAX_TRADE_SIZE = 200.0
COMMISSION_RT = 1.30  # $0.65 each way

# Options cost model
ATM_30DTE_PREMIUM_PCT = 0.035  # 3.5% of underlying
ATM_14DTE_PREMIUM_PCT = 0.020  # 2% of underlying
BID_ASK_HAIRCUT = 0.10  # 10% on entry AND exit

# Walk-forward period
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-06-01"  # need lookback for indicators

# Regime
SMA_200_PERIOD = 200

# Validation
N_PERMUTATIONS = 1000
MIN_TRADES = 20

np.random.seed(42)


# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    """Download all sector ETF + SPY data."""
    print(f"Downloading data for {len(ALL_TICKERS)} tickers from {DATA_START} to {OOT_END}...")
    data = {}
    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, start=DATA_START, end=OOT_END, progress=False)
            if len(df) < 100:
                print(f"  WARNING: {ticker} only has {len(df)} rows")
                continue
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
            print(f"  {ticker}: {len(df)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
    return data


# ============================================================
# INDICATOR COMPUTATION
# ============================================================
def compute_indicators(data):
    """Compute all flow/rotation indicators for each sector."""
    spy = data[BENCHMARK].copy()
    spy["ret_20d"] = spy["Close"].pct_change(20)
    spy["sma_200"] = spy["Close"].rolling(200).mean()
    spy["regime"] = np.where(spy["Close"] > spy["sma_200"], "bull", "bear")

    indicators = {}
    for ticker in SECTOR_ETFS:
        if ticker not in data:
            continue
        df = data[ticker].copy()

        # Basic returns
        df["ret_1d"] = df["Close"].pct_change(1)
        df["ret_5d"] = df["Close"].pct_change(5)
        df["ret_20d"] = df["Close"].pct_change(20)
        df["ret_40d"] = df["Close"].pct_change(40)

        # 1. Relative Volume (RV)
        df["vol_20d_avg"] = df["Volume"].rolling(20).mean()
        df["vol_30d_avg"] = df["Volume"].rolling(30).mean()
        df["rv_20d"] = df["Volume"] / df["vol_20d_avg"]

        # 2. Price-Volume Divergence (PVD)
        df["vol_trend_5d"] = df["Volume"].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
        )
        df["price_trend_5d"] = df["Close"].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
        )
        # Accumulation: vol up, price flat/down
        df["pvd_accum"] = ((df["vol_trend_5d"] > 0) & (df["price_trend_5d"] <= 0)).astype(int)
        # Distribution: vol down, price up
        df["pvd_distrib"] = ((df["vol_trend_5d"] < 0) & (df["price_trend_5d"] > 0)).astype(int)

        # 3. Money Flow Index (MFI) - 14-period
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        raw_money_flow = typical_price * df["Volume"]
        pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        pos_flow_sum = pd.Series(pos_flow, index=df.index).rolling(14).sum()
        neg_flow_sum = pd.Series(neg_flow, index=df.index).rolling(14).sum()
        money_ratio = pos_flow_sum / neg_flow_sum.replace(0, np.nan)
        df["mfi"] = 100 - (100 / (1 + money_ratio))

        # 4. On-Balance Volume (OBV)
        obv = [0.0]
        closes = df["Close"].values
        volumes = df["Volume"].values
        for i in range(1, len(df)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        df["obv_trend_5d"] = df["obv"].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x / (x.iloc[0] if x.iloc[0] != 0 else 1), 1)[0]
            if len(x) == 5 else 0, raw=False
        )
        df["obv_trend_20d"] = df["obv"].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x / (x.iloc[0] if x.iloc[0] != 0 else 1), 1)[0]
            if len(x) == 20 else 0, raw=False
        )

        # 5. Sector Relative Strength vs SPY
        spy_ret_20d = spy["ret_20d"].reindex(df.index)
        df["rs_vs_spy"] = df["ret_20d"] - spy_ret_20d
        df["rs_trend"] = df["rs_vs_spy"].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
        )

        # 6. Volume-Weighted Momentum
        df["vol_weighted_mom"] = df["ret_20d"] * df["rv_20d"]

        # 7. Accumulation/Distribution Line
        clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
        df["ad_line"] = (clv * df["Volume"]).cumsum()
        df["ad_trend_5d"] = df["ad_line"].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x / (abs(x.iloc[0]) + 1), 1)[0]
            if len(x) == 5 else 0, raw=False
        )

        # 8. EMA 20
        df["ema_20"] = df["Close"].ewm(span=20).mean()

        # Range position (where did close land in day's range)
        df["range_pct"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)

        indicators[ticker] = df

    return indicators, spy


def compute_dispersion(indicators, spy):
    """Cross-sector dispersion: std of all sector 20d returns."""
    ret_20d_all = pd.DataFrame({
        t: indicators[t]["ret_20d"] for t in indicators
    })
    dispersion = ret_20d_all.std(axis=1)
    disp_252_rank = dispersion.rolling(252).rank(pct=True)
    return dispersion, disp_252_rank


def compute_cyclical_defensive_spread(indicators):
    """Spread between cyclical and defensive sector returns."""
    cyclicals = ["XLY", "XLI", "XLF"]
    defensives = ["XLU", "XLP", "XLV"]

    cyc_rets = pd.DataFrame({t: indicators[t]["ret_20d"] for t in cyclicals if t in indicators})
    def_rets = pd.DataFrame({t: indicators[t]["ret_20d"] for t in defensives if t in indicators})

    cyc_avg = cyc_rets.mean(axis=1)
    def_avg = def_rets.mean(axis=1)
    spread = cyc_avg - def_avg
    spread_20d_chg = spread - spread.shift(20)
    return spread, spread_20d_chg


# ============================================================
# OPTIONS P&L MODEL
# ============================================================
def option_pnl(entry_price, exit_price, is_call=True, dte=30, hold_days=None,
               position_size=200.0):
    """
    Model option P&L realistically.

    For ATM options:
    - Premium = pct of underlying (3.5% for 30-DTE, 2% for 14-DTE)
    - Delta starts ~0.50 for ATM
    - Gamma effect: delta increases as underlying moves favorably
    - Theta decay: ~1/sqrt(DTE) accelerating near expiry
    - Bid-ask haircut: 10% on entry AND exit

    Returns: (pnl_dollars, premium_paid, num_contracts)
    """
    if hold_days is None:
        hold_days = max(1, dte - 5)  # exit 5 days before expiry

    premium_pct = ATM_30DTE_PREMIUM_PCT if dte >= 25 else ATM_14DTE_PREMIUM_PCT
    premium_per_share = entry_price * premium_pct
    contract_cost = premium_per_share * 100  # 1 contract = 100 shares

    # How many contracts can we buy?
    # Account for bid-ask haircut on entry
    effective_entry_cost = contract_cost * (1 + BID_ASK_HAIRCUT)
    num_contracts = max(1, int(position_size / effective_entry_cost))
    total_premium = num_contracts * contract_cost

    # Underlying move
    underlying_move = (exit_price - entry_price) / entry_price
    if not is_call:
        underlying_move = -underlying_move

    # Option value at exit (simplified but realistic model)
    # ATM delta ~0.50, gamma effect for larger moves
    intrinsic_at_exit = max(0, underlying_move * entry_price) * 100 * num_contracts

    # Time value remaining
    remaining_dte = max(0, dte - hold_days)
    time_decay_factor = np.sqrt(remaining_dte / dte) if dte > 0 else 0

    # Effective delta (accounts for gamma - delta increases as we go ITM)
    if underlying_move > 0:
        # Favorable move - delta increases
        effective_delta = min(0.95, 0.50 + underlying_move * 3)
    else:
        # Unfavorable move - delta decreases
        effective_delta = max(0.05, 0.50 + underlying_move * 3)

    # Option value = intrinsic + remaining time value
    pct_move = abs(exit_price - entry_price) / entry_price
    if underlying_move > 0:
        # Favorable: option value = delta * move * underlying * 100 + remaining_tv
        option_value = (effective_delta * pct_move * entry_price * 100 +
                        premium_per_share * 100 * time_decay_factor * (1 - pct_move * 2))
        option_value = max(option_value, max(0, (exit_price - entry_price) * 100) if is_call
                          else max(0, (entry_price - exit_price) * 100))
    else:
        # Unfavorable: option loses value
        option_value = premium_per_share * 100 * time_decay_factor * max(0, 1 - abs(pct_move) * 5)

    option_value = max(0, option_value)  # Can't go below 0
    total_exit_value = option_value * num_contracts

    # Bid-ask haircut on exit
    total_exit_value *= (1 - BID_ASK_HAIRCUT)

    # Commissions
    total_commission = COMMISSION_RT * num_contracts

    # P&L
    pnl = total_exit_value - total_premium * (1 + BID_ASK_HAIRCUT) - total_commission

    # Cap loss at premium paid + commissions
    max_loss = -(total_premium * (1 + BID_ASK_HAIRCUT) + total_commission)
    pnl = max(pnl, max_loss)

    return pnl, total_premium * (1 + BID_ASK_HAIRCUT), num_contracts


# ============================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================

class BacktestEngine:
    """Core backtest engine for all strategies."""

    def __init__(self, indicators, spy, dispersion, disp_rank,
                 cyc_def_spread, cyc_def_spread_chg, initial_capital=INITIAL_CAPITAL):
        self.indicators = indicators
        self.spy = spy
        self.dispersion = dispersion
        self.disp_rank = disp_rank
        self.cyc_def_spread = cyc_def_spread
        self.cyc_def_spread_chg = cyc_def_spread_chg
        self.initial_capital = initial_capital

        # Common date range (OOT period only)
        self.oot_dates = spy.loc[OOT_START:OOT_END].index

    def run_strategy(self, strategy_func, name):
        """Run a strategy and collect trades."""
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        trades = strategy_func()
        if not trades:
            print(f"  No trades generated!")
            return None

        return self.analyze_trades(trades, name)

    def analyze_trades(self, trades, name):
        """Analyze a list of trades and compute metrics."""
        df = pd.DataFrame(trades)
        if len(df) == 0:
            return None

        # Equity curve
        capital = self.initial_capital
        equity = [capital]
        df = df.sort_values("entry_date").reset_index(drop=True)

        for _, row in df.iterrows():
            capital += row["pnl"]
            capital = max(capital, 0)  # Can't go negative
            equity.append(capital)

        final_capital = equity[-1]
        total_return = (final_capital / self.initial_capital - 1) * 100

        # Win/loss stats
        winners = df[df["pnl"] > 0]
        losers = df[df["pnl"] <= 0]
        win_rate = len(winners) / len(df) * 100 if len(df) > 0 else 0

        avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers["pnl"].mean()) if len(losers) > 0 else 1
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Monthly returns for Sharpe/Sortino
        df["entry_month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M")
        monthly_pnl = df.groupby("entry_month")["pnl"].sum()

        # Fill missing months with 0
        all_months = pd.period_range(OOT_START, OOT_END, freq="M")
        monthly_returns = monthly_pnl.reindex(all_months, fill_value=0)
        monthly_ret_pct = monthly_returns / self.initial_capital

        # Risk metrics
        if monthly_ret_pct.std() > 0:
            sharpe = monthly_ret_pct.mean() / monthly_ret_pct.std() * np.sqrt(12)
        else:
            sharpe = 0

        downside = monthly_ret_pct[monthly_ret_pct < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = monthly_ret_pct.mean() / downside.std() * np.sqrt(12)
        else:
            sortino = 0

        # Max drawdown from equity curve
        eq_arr = np.array(equity)
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - running_max) / running_max
        max_dd = drawdowns.min() * 100

        # Profit factor
        gross_profit = winners["pnl"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["pnl"].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Regime analysis
        df["regime"] = df["entry_date"].apply(
            lambda d: self.spy.loc[:d, "regime"].iloc[-1]
            if d in self.spy.index or True else "unknown"
        )
        # Safer regime lookup
        regimes = []
        for d in df["entry_date"]:
            try:
                mask = self.spy.index <= pd.Timestamp(d)
                if mask.any():
                    regimes.append(self.spy.loc[mask, "regime"].iloc[-1])
                else:
                    regimes.append("unknown")
            except:
                regimes.append("unknown")
        df["regime"] = regimes

        bull_trades = df[df["regime"] == "bull"]
        bear_trades = df[df["regime"] == "bear"]

        bull_sharpe = 0
        bear_sharpe = 0
        if len(bull_trades) > 5:
            bull_monthly = bull_trades.groupby("entry_month")["pnl"].sum().reindex(all_months, fill_value=0)
            bull_ret = bull_monthly / self.initial_capital
            if bull_ret.std() > 0:
                bull_sharpe = bull_ret.mean() / bull_ret.std() * np.sqrt(12)
        if len(bear_trades) > 5:
            bear_monthly = bear_trades.groupby("entry_month")["pnl"].sum().reindex(all_months, fill_value=0)
            bear_ret = bear_monthly / self.initial_capital
            if bear_ret.std() > 0:
                bear_sharpe = bear_ret.mean() / bear_ret.std() * np.sqrt(12)

        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)

        # Monthly trade frequency
        n_months = len(all_months)
        monthly_freq = len(df) / n_months if n_months > 0 else 0

        results = {
            "name": name,
            "n_trades": len(df),
            "total_return_pct": round(total_return, 2),
            "final_capital": round(final_capital, 2),
            "initial_capital": self.initial_capital,
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "profit_factor": round(profit_factor, 3),
            "win_rate": round(win_rate, 1),
            "avg_winner": round(avg_win, 2),
            "avg_loser": round(avg_loss, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "best_trade": round(df["pnl"].max(), 2),
            "worst_trade": round(df["pnl"].min(), 2),
            "monthly_trade_freq": round(monthly_freq, 1),
            "bull_sharpe": round(bull_sharpe, 3),
            "bear_sharpe": round(bear_sharpe, 3),
            "regime_gap": round(regime_gap, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
        }

        # Print summary
        print(f"  Trades: {len(df)} | Return: {total_return:+.1f}% | Final: ${final_capital:.0f}")
        print(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {profit_factor:.2f} | WR: {win_rate:.0f}%")
        print(f"  Payoff: {payoff_ratio:.2f} | MaxDD: {max_dd:.1f}% | Monthly freq: {monthly_freq:.1f}")
        print(f"  Regime: Bull Sharpe={bull_sharpe:.3f} ({len(bull_trades)} trades) | Bear Sharpe={bear_sharpe:.3f} ({len(bear_trades)} trades) | Gap={regime_gap:.3f}")

        return results, df

    def permutation_test(self, strategy_func, observed_sharpe, n_perms=N_PERMUTATIONS):
        """Shuffle which sector is selected to test if selection adds value."""
        print(f"  Running {n_perms} permutations...")
        perm_sharpes = []

        for i in range(n_perms):
            try:
                trades = strategy_func(permute=True)
                if not trades or len(trades) < 5:
                    continue
                df = pd.DataFrame(trades)
                df["entry_month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M")
                monthly_pnl = df.groupby("entry_month")["pnl"].sum()
                all_months = pd.period_range(OOT_START, OOT_END, freq="M")
                monthly_returns = monthly_pnl.reindex(all_months, fill_value=0)
                monthly_ret_pct = monthly_returns / self.initial_capital
                if monthly_ret_pct.std() > 0:
                    s = monthly_ret_pct.mean() / monthly_ret_pct.std() * np.sqrt(12)
                else:
                    s = 0
                perm_sharpes.append(s)
            except:
                continue

        if len(perm_sharpes) < 50:
            return 1.0  # Not enough permutations

        p_value = np.mean([s >= observed_sharpe for s in perm_sharpes])
        print(f"  Permutation p-value: {p_value:.4f} (observed Sharpe={observed_sharpe:.3f}, "
              f"mean perm Sharpe={np.mean(perm_sharpes):.3f})")
        return p_value


# ============================================================
# STRATEGY A: Volume-Confirmed Momentum Rotation
# ============================================================
def strategy_a(engine, permute=False):
    """Weekly: rank sectors by 20d momentum * relative_volume, buy calls on top."""
    trades = []
    # Trade every Monday (weekly)
    mondays = [d for d in engine.oot_dates if d.weekday() == 0]

    for entry_date in mondays:
        # Rank sectors by volume-weighted momentum
        scores = {}
        for ticker in SECTOR_ETFS:
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if entry_date not in ind.index:
                continue
            row = ind.loc[entry_date]
            if pd.isna(row.get("vol_weighted_mom")) or pd.isna(row.get("rv_20d")):
                continue
            scores[ticker] = row["vol_weighted_mom"]

        if len(scores) < 3:
            continue

        if permute:
            selected = np.random.choice(list(scores.keys()))
        else:
            selected = max(scores, key=scores.get)

        # Buy ATM calls 30-DTE on top sector
        ind = engine.indicators[selected]
        entry_price = ind.loc[entry_date, "Close"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Hold for up to 25 days (exit 5 before expiry), check for 50% gain/loss stops
        exit_date = None
        exit_price = entry_price
        dte = 30
        hold_days = 0

        future_dates = ind.index[ind.index > entry_date][:25]
        premium_pct = ATM_30DTE_PREMIUM_PCT
        stop_loss_pct = -0.03  # ~50% option loss ≈ ~3% underlying drop for ATM
        target_pct = 0.04     # ~50% option gain ≈ ~4% underlying rise (accounting for theta)

        for fd in future_dates:
            hold_days += 1
            current_price = ind.loc[fd, "Close"]
            move_pct = (current_price - entry_price) / entry_price

            if move_pct >= target_pct:  # Target hit
                exit_price = current_price
                exit_date = fd
                break
            elif move_pct <= stop_loss_pct:  # Stop hit
                exit_price = current_price
                exit_date = fd
                break

        if exit_date is None and len(future_dates) > 0:
            exit_date = future_dates[-1]
            exit_price = ind.loc[exit_date, "Close"]
            hold_days = len(future_dates)

        if exit_date is None:
            continue

        pnl, premium, n_contracts = option_pnl(
            entry_price, exit_price, is_call=True, dte=30,
            hold_days=hold_days, position_size=MAX_TRADE_SIZE
        )

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "ticker": selected,
            "direction": "call",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "hold_days": hold_days,
            "pnl": float(pnl),
            "premium_paid": float(premium),
            "n_contracts": n_contracts,
            "strategy": "A_VolConfirmedMomentum",
        })

    return trades


# ============================================================
# STRATEGY B: Accumulation Detector -> Calls
# ============================================================
def strategy_b(engine, permute=False):
    """Detect rising OBV + flat/falling price -> buy calls."""
    trades = []
    # Check daily, but limit to one position per sector at a time
    active_positions = set()

    for i, date in enumerate(engine.oot_dates):
        if i % 3 != 0:  # Check every 3 days to avoid overtrading
            continue

        for ticker in SECTOR_ETFS:
            if ticker in active_positions:
                continue
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if date not in ind.index:
                continue

            row = ind.loc[date]

            # Accumulation signal: rising OBV trend + flat/falling price
            obv_rising = row.get("obv_trend_5d", 0) > 0.001
            price_flat_or_down = row.get("price_trend_5d", 1) <= 0
            rv_elevated = row.get("rv_20d", 0) > 1.2  # Some volume confirmation

            if not (obv_rising and price_flat_or_down and rv_elevated):
                continue

            if permute:
                # Randomly decide whether to take this trade
                if np.random.random() > 0.5:
                    continue

            entry_price = row["Close"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Hold until price catches up (5d return > 2%) or 25 days
            future_dates = ind.index[ind.index > date][:25]
            exit_date = None
            exit_price = entry_price
            hold_days = 0

            for fd in future_dates:
                hold_days += 1
                current_price = ind.loc[fd, "Close"]
                pct_move = (current_price - entry_price) / entry_price

                if pct_move >= 0.02:  # Price caught up to OBV
                    exit_price = current_price
                    exit_date = fd
                    break
                elif pct_move <= -0.04:  # Stop loss
                    exit_price = current_price
                    exit_date = fd
                    break

            if exit_date is None and len(future_dates) > 0:
                exit_date = future_dates[-1]
                exit_price = ind.loc[exit_date, "Close"]
                hold_days = len(future_dates)

            if exit_date is None:
                continue

            pnl, premium, n_contracts = option_pnl(
                entry_price, exit_price, is_call=True, dte=30,
                hold_days=hold_days, position_size=MAX_TRADE_SIZE
            )

            trades.append({
                "entry_date": date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "direction": "call",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "hold_days": hold_days,
                "pnl": float(pnl),
                "premium_paid": float(premium),
                "n_contracts": n_contracts,
                "strategy": "B_AccumulationDetector",
            })

    return trades


# ============================================================
# STRATEGY C: Money Flow Divergence
# ============================================================
def strategy_c(engine, permute=False):
    """MFI crossovers: above 20 -> calls, below 80 -> puts."""
    trades = []

    for i, date in enumerate(engine.oot_dates):
        if i < 1:
            continue

        prev_date = engine.oot_dates[i - 1]

        for ticker in SECTOR_ETFS:
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if date not in ind.index or prev_date not in ind.index:
                continue

            mfi_today = ind.loc[date, "mfi"]
            mfi_yesterday = ind.loc[prev_date, "mfi"]

            if pd.isna(mfi_today) or pd.isna(mfi_yesterday):
                continue

            is_call = None
            # Bullish: MFI crosses above 20 from below
            if mfi_yesterday < 20 and mfi_today >= 20:
                is_call = True
            # Bearish: MFI crosses below 80 from above
            elif mfi_yesterday > 80 and mfi_today <= 80:
                is_call = False

            if is_call is None:
                continue

            if permute:
                is_call = np.random.choice([True, False])

            entry_price = ind.loc[date, "Close"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Hold up to 20 days
            future_dates = ind.index[ind.index > date][:20]
            exit_date = None
            exit_price = entry_price
            hold_days = 0

            target_pct = 0.03 if is_call else -0.03
            stop_pct = -0.04 if is_call else 0.04

            for fd in future_dates:
                hold_days += 1
                current_price = ind.loc[fd, "Close"]
                move = (current_price - entry_price) / entry_price

                if is_call and move >= target_pct:
                    exit_price = current_price
                    exit_date = fd
                    break
                elif not is_call and move <= target_pct:
                    exit_price = current_price
                    exit_date = fd
                    break
                elif is_call and move <= stop_pct:
                    exit_price = current_price
                    exit_date = fd
                    break
                elif not is_call and move >= abs(stop_pct):
                    exit_price = current_price
                    exit_date = fd
                    break

            if exit_date is None and len(future_dates) > 0:
                exit_date = future_dates[-1]
                exit_price = ind.loc[exit_date, "Close"]
                hold_days = len(future_dates)

            if exit_date is None:
                continue

            pnl, premium, n_contracts = option_pnl(
                entry_price, exit_price, is_call=is_call, dte=30,
                hold_days=hold_days, position_size=MAX_TRADE_SIZE
            )

            trades.append({
                "entry_date": date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "direction": "call" if is_call else "put",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "hold_days": hold_days,
                "pnl": float(pnl),
                "premium_paid": float(premium),
                "n_contracts": n_contracts,
                "strategy": "C_MoneyFlowDivergence",
            })

    return trades


# ============================================================
# STRATEGY D: Sector Pair Rotation
# ============================================================
def strategy_d(engine, permute=False):
    """Cyclical vs defensive spread rotation."""
    trades = []
    cyclicals = ["XLY", "XLI", "XLF"]
    defensives = ["XLU", "XLP", "XLV"]

    # Trade weekly
    mondays = [d for d in engine.oot_dates if d.weekday() == 0]

    for entry_date in mondays:
        if entry_date not in engine.cyc_def_spread.index:
            continue

        spread = engine.cyc_def_spread.loc[entry_date]
        spread_chg = engine.cyc_def_spread_chg.loc[entry_date]

        if pd.isna(spread) or pd.isna(spread_chg):
            continue

        # Cyclicals recovering (spread rising from negative territory)
        if spread < 0 and spread_chg > 0.005:
            # Buy calls on best cyclical
            target_group = cyclicals
            is_call = True
        # Defensives taking over (spread falling from positive territory)
        elif spread > 0 and spread_chg < -0.005:
            # Buy calls on best defensive (or puts on cyclical)
            target_group = defensives
            is_call = True
        else:
            continue

        # Pick best in group by momentum
        scores = {}
        for ticker in target_group:
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if entry_date not in ind.index:
                continue
            scores[ticker] = ind.loc[entry_date, "ret_20d"]

        if not scores:
            continue

        if permute:
            selected = np.random.choice(list(scores.keys()))
        else:
            selected = max(scores, key=scores.get)

        ind = engine.indicators[selected]
        entry_price = ind.loc[entry_date, "Close"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        future_dates = ind.index[ind.index > entry_date][:25]
        exit_date = None
        exit_price = entry_price
        hold_days = 0

        for fd in future_dates:
            hold_days += 1
            current_price = ind.loc[fd, "Close"]
            move = (current_price - entry_price) / entry_price

            if move >= 0.05:  # 5% target
                exit_price = current_price
                exit_date = fd
                break
            elif move <= -0.03:  # 3% stop
                exit_price = current_price
                exit_date = fd
                break

        if exit_date is None and len(future_dates) > 0:
            exit_date = future_dates[-1]
            exit_price = ind.loc[exit_date, "Close"]
            hold_days = len(future_dates)

        if exit_date is None:
            continue

        pnl, premium, n_contracts = option_pnl(
            entry_price, exit_price, is_call=is_call, dte=30,
            hold_days=hold_days, position_size=MAX_TRADE_SIZE
        )

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "ticker": selected,
            "direction": "call" if is_call else "put",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "hold_days": hold_days,
            "pnl": float(pnl),
            "premium_paid": float(premium),
            "n_contracts": n_contracts,
            "strategy": "D_SectorPairRotation",
        })

    return trades


# ============================================================
# STRATEGY E: Dispersion + Momentum
# ============================================================
def strategy_e(engine, permute=False):
    """Only trade when dispersion is high. Long best, short worst."""
    trades = []
    mondays = [d for d in engine.oot_dates if d.weekday() == 0]

    for entry_date in mondays:
        if entry_date not in engine.disp_rank.index:
            continue

        disp_rank = engine.disp_rank.loc[entry_date]
        if pd.isna(disp_rank) or disp_rank < 0.75:  # Top quartile only
            continue

        # Rank sectors by momentum
        scores = {}
        for ticker in SECTOR_ETFS:
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if entry_date not in ind.index:
                continue
            ret = ind.loc[entry_date, "ret_20d"]
            if not pd.isna(ret):
                scores[ticker] = ret

        if len(scores) < 5:
            continue

        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if permute:
            best = np.random.choice(list(scores.keys()))
            worst = np.random.choice([t for t in scores.keys() if t != best])
        else:
            best = sorted_sectors[0][0]
            worst = sorted_sectors[-1][0]

        # Buy calls on best sector
        ind_best = engine.indicators[best]
        entry_price = ind_best.loc[entry_date, "Close"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        future_dates = ind_best.index[ind_best.index > entry_date][:20]
        exit_date = None
        exit_price = entry_price
        hold_days = 0

        for fd in future_dates:
            hold_days += 1
            current_price = ind_best.loc[fd, "Close"]
            move = (current_price - entry_price) / entry_price
            if move >= 0.04:
                exit_price = current_price
                exit_date = fd
                break
            elif move <= -0.035:
                exit_price = current_price
                exit_date = fd
                break

        if exit_date is None and len(future_dates) > 0:
            exit_date = future_dates[-1]
            exit_price = ind_best.loc[exit_date, "Close"]
            hold_days = len(future_dates)

        if exit_date is None:
            continue

        pnl_call, prem_call, nc_call = option_pnl(
            entry_price, exit_price, is_call=True, dte=30,
            hold_days=hold_days, position_size=MAX_TRADE_SIZE // 2
        )

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "ticker": best,
            "direction": "call",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "hold_days": hold_days,
            "pnl": float(pnl_call),
            "premium_paid": float(prem_call),
            "n_contracts": nc_call,
            "strategy": "E_DispersionMomentum",
        })

        # Buy puts on worst sector
        ind_worst = engine.indicators[worst]
        entry_price_w = ind_worst.loc[entry_date, "Close"]
        if pd.isna(entry_price_w) or entry_price_w <= 0:
            continue

        future_dates_w = ind_worst.index[ind_worst.index > entry_date][:20]
        exit_date_w = None
        exit_price_w = entry_price_w
        hold_days_w = 0

        for fd in future_dates_w:
            hold_days_w += 1
            current_price = ind_worst.loc[fd, "Close"]
            move = (current_price - entry_price_w) / entry_price_w
            if move <= -0.04:
                exit_price_w = current_price
                exit_date_w = fd
                break
            elif move >= 0.035:
                exit_price_w = current_price
                exit_date_w = fd
                break

        if exit_date_w is None and len(future_dates_w) > 0:
            exit_date_w = future_dates_w[-1]
            exit_price_w = ind_worst.loc[exit_date_w, "Close"]
            hold_days_w = len(future_dates_w)

        if exit_date_w is None:
            continue

        pnl_put, prem_put, nc_put = option_pnl(
            entry_price_w, exit_price_w, is_call=False, dte=30,
            hold_days=hold_days_w, position_size=MAX_TRADE_SIZE // 2
        )

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date_w.strftime("%Y-%m-%d"),
            "ticker": worst,
            "direction": "put",
            "entry_price": float(entry_price_w),
            "exit_price": float(exit_price_w),
            "hold_days": hold_days_w,
            "pnl": float(pnl_put),
            "premium_paid": float(prem_put),
            "n_contracts": nc_put,
            "strategy": "E_DispersionMomentum",
        })

    return trades


# ============================================================
# STRATEGY F: Multi-Signal Confluence Rotation
# ============================================================
def strategy_f(engine, permute=False):
    """Composite scoring: momentum + volume + OBV + MFI + RS. Need 3+ agreeing."""
    trades = []
    mondays = [d for d in engine.oot_dates if d.weekday() == 0]

    for entry_date in mondays:
        sector_scores = {}

        for ticker in SECTOR_ETFS:
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if entry_date not in ind.index:
                continue
            row = ind.loc[entry_date]

            score = 0
            signals_bullish = 0
            signals_total = 0

            # 1. Momentum rank (20d return)
            ret_20 = row.get("ret_20d", np.nan)
            if not pd.isna(ret_20):
                signals_total += 1
                if ret_20 > 0.02:
                    score += 20
                    signals_bullish += 1
                elif ret_20 < -0.02:
                    score -= 10

            # 2. Volume confirmation
            rv = row.get("rv_20d", np.nan)
            if not pd.isna(rv):
                signals_total += 1
                if rv > 1.3:
                    score += 20
                    signals_bullish += 1
                elif rv < 0.7:
                    score -= 10

            # 3. OBV trend
            obv_t = row.get("obv_trend_20d", np.nan)
            if not pd.isna(obv_t):
                signals_total += 1
                if obv_t > 0:
                    score += 20
                    signals_bullish += 1
                else:
                    score -= 10

            # 4. MFI zone
            mfi = row.get("mfi", np.nan)
            if not pd.isna(mfi):
                signals_total += 1
                if 20 < mfi < 50:  # Oversold recovering
                    score += 20
                    signals_bullish += 1
                elif mfi > 80:
                    score -= 15

            # 5. RS vs SPY
            rs = row.get("rs_vs_spy", np.nan)
            if not pd.isna(rs):
                signals_total += 1
                if rs > 0.01:
                    score += 20
                    signals_bullish += 1
                elif rs < -0.01:
                    score -= 10

            sector_scores[ticker] = {
                "score": max(0, min(100, score + 50)),  # Normalize to 0-100
                "signals_bullish": signals_bullish,
                "signals_total": signals_total,
            }

        if len(sector_scores) < 5:
            continue

        # Find clear rotation: one sector > 80, another < 20
        scores_list = [(t, s["score"], s["signals_bullish"]) for t, s in sector_scores.items()]
        scores_list.sort(key=lambda x: x[1], reverse=True)

        best_ticker, best_score, best_bullish = scores_list[0]
        worst_ticker, worst_score, _ = scores_list[-1]

        if best_score < 75 or worst_score > 30:
            continue  # No clear rotation
        if best_bullish < 3:
            continue  # Need 3+ signals agreeing

        if permute:
            best_ticker = np.random.choice([s[0] for s in scores_list])

        ind = engine.indicators[best_ticker]
        entry_price = ind.loc[entry_date, "Close"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        future_dates = ind.index[ind.index > entry_date][:25]
        exit_date = None
        exit_price = entry_price
        hold_days = 0

        for fd in future_dates:
            hold_days += 1
            current_price = ind.loc[fd, "Close"]
            move = (current_price - entry_price) / entry_price
            if move >= 0.05:
                exit_price = current_price
                exit_date = fd
                break
            elif move <= -0.03:
                exit_price = current_price
                exit_date = fd
                break

        if exit_date is None and len(future_dates) > 0:
            exit_date = future_dates[-1]
            exit_price = ind.loc[exit_date, "Close"]
            hold_days = len(future_dates)

        if exit_date is None:
            continue

        pnl, premium, n_contracts = option_pnl(
            entry_price, exit_price, is_call=True, dte=30,
            hold_days=hold_days, position_size=MAX_TRADE_SIZE
        )

        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "ticker": best_ticker,
            "direction": "call",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "hold_days": hold_days,
            "pnl": float(pnl),
            "premium_paid": float(premium),
            "n_contracts": n_contracts,
            "strategy": "F_MultiSignalConfluence",
        })

    return trades


# ============================================================
# STRATEGY G: CTA Momentum Following
# ============================================================
def strategy_g(engine, permute=False):
    """CTA proxy: 40+ day trend + accelerating volume -> calls."""
    trades = []

    # Check every 5 days
    check_dates = [d for i, d in enumerate(engine.oot_dates) if i % 5 == 0]

    for entry_date in check_dates:
        for ticker in SECTOR_ETFS:
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if entry_date not in ind.index:
                continue

            row = ind.loc[entry_date]

            # 40-day trend UP
            ret_40 = row.get("ret_40d", np.nan)
            if pd.isna(ret_40) or ret_40 <= 0.02:  # At least 2% in 40 days
                continue

            # Volume accelerating (recent volume > average)
            rv = row.get("rv_20d", np.nan)
            if pd.isna(rv) or rv <= 1.1:
                continue

            # Price above 20-day EMA
            ema_20 = row.get("ema_20", np.nan)
            price = row["Close"]
            if pd.isna(ema_20) or price <= ema_20:
                continue

            if permute:
                if np.random.random() > 0.3:
                    continue

            entry_price = price
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Hold until trend breaks (price drops below 20-day EMA) or 30 days
            future_dates = ind.index[ind.index > entry_date][:30]
            exit_date = None
            exit_price = entry_price
            hold_days = 0

            for fd in future_dates:
                hold_days += 1
                current_price = ind.loc[fd, "Close"]
                current_ema = ind.loc[fd, "ema_20"]

                if not pd.isna(current_ema) and current_price < current_ema:
                    exit_price = current_price
                    exit_date = fd
                    break

                move = (current_price - entry_price) / entry_price
                if move >= 0.08:  # 8% target
                    exit_price = current_price
                    exit_date = fd
                    break
                elif move <= -0.04:  # 4% stop
                    exit_price = current_price
                    exit_date = fd
                    break

            if exit_date is None and len(future_dates) > 0:
                exit_date = future_dates[-1]
                exit_price = ind.loc[exit_date, "Close"]
                hold_days = len(future_dates)

            if exit_date is None:
                continue

            pnl, premium, n_contracts = option_pnl(
                entry_price, exit_price, is_call=True, dte=30,
                hold_days=hold_days, position_size=MAX_TRADE_SIZE
            )

            trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "direction": "call",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "hold_days": hold_days,
                "pnl": float(pnl),
                "premium_paid": float(premium),
                "n_contracts": n_contracts,
                "strategy": "G_CTAMomentum",
            })

    return trades


# ============================================================
# STRATEGY H: Volume Surge Breakout
# ============================================================
def strategy_h(engine, permute=False):
    """Volume > 2x 30-day avg + strong close -> calls, 14-DTE."""
    trades = []

    for date in engine.oot_dates:
        for ticker in SECTOR_ETFS:
            if ticker not in engine.indicators:
                continue
            ind = engine.indicators[ticker]
            if date not in ind.index:
                continue

            row = ind.loc[date]

            # Volume > 2x 30-day average
            vol = row.get("Volume", 0)
            vol_30avg = row.get("vol_30d_avg", np.nan)
            if pd.isna(vol_30avg) or vol_30avg == 0 or vol / vol_30avg < 2.0:
                continue

            # Strong close (top 25% of range)
            range_pct = row.get("range_pct", np.nan)
            if pd.isna(range_pct) or range_pct < 0.75:
                continue

            # Positive day
            ret_1d = row.get("ret_1d", np.nan)
            if pd.isna(ret_1d) or ret_1d <= 0:
                continue

            if permute:
                if np.random.random() > 0.4:
                    continue

            entry_price = row["Close"]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # 14-DTE for faster payoff, stop 50% loss, target 100% gain
            future_dates = ind.index[ind.index > date][:10]  # Exit before expiry
            exit_date = None
            exit_price = entry_price
            hold_days = 0

            for fd in future_dates:
                hold_days += 1
                current_price = ind.loc[fd, "Close"]
                move = (current_price - entry_price) / entry_price

                if move >= 0.03:  # ~100% option gain for 14-DTE
                    exit_price = current_price
                    exit_date = fd
                    break
                elif move <= -0.025:  # ~50% option loss
                    exit_price = current_price
                    exit_date = fd
                    break

            if exit_date is None and len(future_dates) > 0:
                exit_date = future_dates[-1]
                exit_price = ind.loc[exit_date, "Close"]
                hold_days = len(future_dates)

            if exit_date is None:
                continue

            pnl, premium, n_contracts = option_pnl(
                entry_price, exit_price, is_call=True, dte=14,
                hold_days=hold_days, position_size=MAX_TRADE_SIZE
            )

            trades.append({
                "entry_date": date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "direction": "call",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "hold_days": hold_days,
                "pnl": float(pnl),
                "premium_paid": float(premium),
                "n_contracts": n_contracts,
                "strategy": "H_VolumeSurgeBreakout",
            })

    return trades


# ============================================================
# VALIDATION
# ============================================================
def validate_strategy(results, p_value):
    """Apply 5-gate validation framework."""
    if results is None:
        return {"passed": False, "reason": "No results"}

    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = results["sharpe"] > 0.5

    # Gate 2: Permutation p < 0.05
    gates["perm_p_lt_0.05"] = p_value < 0.05

    # Gate 3: Regime gap < 0.5
    gates["regime_gap_lt_0.5"] = results["regime_gap"] < 0.5

    # Gate 4: MaxDD > -50%
    gates["maxdd_gt_neg50"] = results["max_drawdown_pct"] > -50

    # Gate 5: >= 20 trades
    gates["trades_gte_20"] = results["n_trades"] >= MIN_TRADES

    passed = all(gates.values())
    gates_passed = sum(gates.values())

    return {
        "passed": passed,
        "gates_passed": f"{gates_passed}/5",
        "gates": gates,
        "p_value": round(p_value, 4),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("DEEP ROTATION + MONEY FLOW STRATEGY BACKTEST v1")
    print(f"Period: {OOT_START} to {OOT_END}")
    print(f"Initial Capital: ${INITIAL_CAPITAL}")
    print(f"Target: 5-10x returns (${INITIAL_CAPITAL * 5:.0f}-${INITIAL_CAPITAL * 10:.0f})")
    print("=" * 70)

    # Download data
    data = download_data()
    if len(data) < 5:
        print("ERROR: Not enough data downloaded")
        sys.exit(1)

    # Compute indicators
    print("\nComputing indicators...")
    indicators, spy = compute_indicators(data)
    dispersion, disp_rank = compute_dispersion(indicators, spy)
    cyc_def_spread, cyc_def_spread_chg = compute_cyclical_defensive_spread(indicators)

    # Create engine
    engine = BacktestEngine(indicators, spy, dispersion, disp_rank,
                            cyc_def_spread, cyc_def_spread_chg)

    # Run all strategies
    all_results = {}
    all_trades = {}

    strategies = [
        ("A_VolConfirmedMomentum", lambda p=False: strategy_a(engine, permute=p)),
        ("B_AccumulationDetector", lambda p=False: strategy_b(engine, permute=p)),
        ("C_MoneyFlowDivergence", lambda p=False: strategy_c(engine, permute=p)),
        ("D_SectorPairRotation", lambda p=False: strategy_d(engine, permute=p)),
        ("E_DispersionMomentum", lambda p=False: strategy_e(engine, permute=p)),
        ("F_MultiSignalConfluence", lambda p=False: strategy_f(engine, permute=p)),
        ("G_CTAMomentum", lambda p=False: strategy_g(engine, permute=p)),
        ("H_VolumeSurgeBreakout", lambda p=False: strategy_h(engine, permute=p)),
    ]

    for name, strat_func in strategies:
        result = engine.run_strategy(lambda sf=strat_func: sf(False), name)
        if result is not None:
            results, trades_df = result
            all_results[name] = results
            all_trades[name] = trades_df

            # Permutation test
            if results["n_trades"] >= MIN_TRADES:
                p_val = engine.permutation_test(
                    lambda p=True, sf=strat_func: sf(p),
                    results["sharpe"]
                )
            else:
                p_val = 1.0

            validation = validate_strategy(results, p_val)
            all_results[name]["validation"] = validation
            print(f"  Validation: {validation['gates_passed']} gates | p={p_val:.4f} | "
                  f"{'PASSED' if validation['passed'] else 'FAILED'}")
        else:
            all_results[name] = {"name": name, "error": "No trades generated"}

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30} {'Trades':>6} {'Return%':>8} {'Final$':>8} {'Sharpe':>7} "
          f"{'Sortino':>8} {'PF':>6} {'WR%':>5} {'MaxDD%':>7} {'Valid':>6}")
    print("-" * 100)

    ranked = []
    for name in sorted(all_results.keys()):
        r = all_results[name]
        if "error" in r:
            print(f"{name:<30} {'N/A':>6} {'N/A':>8} {'N/A':>8} {'N/A':>7} "
                  f"{'N/A':>8} {'N/A':>6} {'N/A':>5} {'N/A':>7} {'FAIL':>6}")
            continue

        v = r.get("validation", {})
        valid_str = v.get("gates_passed", "?/?")
        passed = "YES" if v.get("passed", False) else "NO"

        print(f"{name:<30} {r['n_trades']:>6} {r['total_return_pct']:>7.1f}% "
              f"${r['final_capital']:>7.0f} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
              f"{r['profit_factor']:>6.2f} {r['win_rate']:>4.0f}% {r['max_drawdown_pct']:>6.1f}% "
              f"{valid_str:>6}")

        ranked.append((name, r.get("sharpe", 0), r.get("total_return_pct", 0),
                        v.get("passed", False)))

    # Rank by total return (the user wants 5-10x)
    ranked.sort(key=lambda x: x[2], reverse=True)
    print(f"\n{'='*70}")
    print("RANKING BY TOTAL RETURN (target: +500% to +900%):")
    for i, (name, sharpe, ret, passed) in enumerate(ranked):
        status = "PASSED ALL GATES" if passed else "FAILED VALIDATION"
        print(f"  {i+1}. {name}: {ret:+.1f}% return, Sharpe={sharpe:.3f} [{status}]")

    # Combination analysis: what if we trade ALL passing strategies?
    print(f"\n{'='*70}")
    print("COMBINATION ANALYSIS: Trading all strategies together")
    combined_trades = []
    for name, trades_df in all_trades.items():
        v = all_results[name].get("validation", {})
        if v.get("passed", False) or all_results[name].get("sharpe", 0) > 0:
            for _, row in trades_df.iterrows():
                combined_trades.append(row.to_dict())

    if combined_trades:
        combined_df = pd.DataFrame(combined_trades).sort_values("entry_date")
        capital = INITIAL_CAPITAL
        peak = capital

        # Track concurrent positions (can't exceed capital)
        equity_curve = []
        for _, trade in combined_df.iterrows():
            if capital >= trade.get("premium_paid", 0):
                capital += trade["pnl"]
                capital = max(capital, 0)
            equity_curve.append(capital)

        final = capital
        total_ret = (final / INITIAL_CAPITAL - 1) * 100
        max_dd = 0
        peak = INITIAL_CAPITAL
        for v in equity_curve:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)

        print(f"  Combined trades: {len(combined_df)}")
        print(f"  Final capital: ${final:.0f} ({total_ret:+.1f}%)")
        print(f"  Max drawdown: {max_dd*100:.1f}%")
        print(f"  NOTE: Does not account for position sizing limits with small account")

    # Save results
    output = {
        "metadata": {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "oot_period": f"{OOT_START} to {OOT_END}",
            "initial_capital": INITIAL_CAPITAL,
            "target_return_pct": "500-900%",
            "n_strategies": len(strategies),
            "sector_etfs": SECTOR_ETFS,
            "options_model": {
                "atm_30dte_premium_pct": ATM_30DTE_PREMIUM_PCT,
                "atm_14dte_premium_pct": ATM_14DTE_PREMIUM_PCT,
                "bid_ask_haircut": BID_ASK_HAIRCUT,
                "commission_rt": COMMISSION_RT,
            },
        },
        "strategies": all_results,
        "ranking": [
            {"rank": i + 1, "name": name, "total_return_pct": ret, "sharpe": sharpe,
             "passed_validation": passed}
            for i, (name, sharpe, ret, passed) in enumerate(ranked)
        ],
    }

    output_path = Path("/home/jupiter/Lvl3Quant/data/deep_rotation_flow_v1_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Also save trade details
    trades_output = {}
    for name, trades_df in all_trades.items():
        trades_output[name] = trades_df.to_dict(orient="records")

    trades_path = Path("/home/jupiter/Lvl3Quant/data/deep_rotation_flow_v1_trades.json")
    with open(trades_path, "w") as f:
        json.dump(trades_output, f, indent=2, default=str)
    print(f"Trade details saved to {trades_path}")

    return output


if __name__ == "__main__":
    main()
