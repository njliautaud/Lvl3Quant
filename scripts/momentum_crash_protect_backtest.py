#!/usr/bin/env python3
"""
Momentum with Crash Protection Backtest
========================================
6 variants combining momentum factor with systematic drawdown protection.
Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

Universe: Growth/tech + safe havens (GLD, TLT, UUP, SHY).
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
TOP_N = 3

START_DATE = "2021-07-01"  # buffer for 60d lookback + 200 SMA
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"

GROWTH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "UBER", "COIN",
]
SAFE_HAVENS = ["GLD", "TLT", "UUP", "SHY"]
ALL_TICKERS = GROWTH_STOCKS + SAFE_HAVENS
SPY = "SPY"

MOM_LOOKBACK = 60  # trading days (~3 months)
TRAILING_STOP_PCT = 0.15  # 15% trailing stop
OVEREXTENDED_THRESHOLD = 0.40  # 40% gain in 60 days = overextended
RSI_ENTRY_THRESHOLD = 40  # RSI(5) < 40 = dip buy

N_PERM = 1000
RANDOM_SEED = 42

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/momentum_crash_protect_results.json")


# ── Data Download ─────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    tickers = list(set(ALL_TICKERS + [SPY]))
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
    else:
        close = data
        high = data
        low = data

    close = close.ffill().dropna(how="all")
    high = high.ffill().dropna(how="all")
    low = low.ffill().dropna(how="all")

    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {close.shape[1]} tickers")
    return close, high, low


# ── Helpers ───────────────────────────────────────────────────────────────────
def monthly_rebalance_dates(idx, start, end):
    """Last trading day of each month."""
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    sub = idx[mask]
    monthly = sub.to_series().groupby([sub.year, sub.month]).last()
    return list(monthly.values)


def apply_slippage(price, direction="buy"):
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def get_price(df, ticker, date):
    """Safely get price."""
    if ticker not in df.columns:
        return np.nan
    val = df[ticker].loc[date] if date in df.index else np.nan
    if isinstance(val, pd.Series):
        val = val.iloc[-1]
    return val


def compute_momentum_scores(close, date, tickers, lookback=MOM_LOOKBACK):
    """Compute momentum (return over lookback) for each ticker."""
    loc = close.index.get_loc(date)
    if loc < lookback:
        return {}
    scores = {}
    for t in tickers:
        if t not in close.columns:
            continue
        cur = close[t].iloc[loc]
        prev = close[t].iloc[loc - lookback]
        if pd.notna(cur) and pd.notna(prev) and prev > 0:
            scores[t] = (cur / prev) - 1.0
    return scores


def compute_atr(close, high, low, ticker, date, period=20):
    """ATR(20) / price for vol scaling."""
    loc = close.index.get_loc(date)
    if loc < period + 1 or ticker not in close.columns:
        return None
    c = close[ticker].iloc[loc - period:loc + 1]
    h = high[ticker].iloc[loc - period:loc + 1] if ticker in high.columns else c
    l = low[ticker].iloc[loc - period:loc + 1] if ticker in low.columns else c

    tr_vals = []
    for i in range(1, len(c)):
        tr = max(
            h.iloc[i] - l.iloc[i],
            abs(h.iloc[i] - c.iloc[i - 1]),
            abs(l.iloc[i] - c.iloc[i - 1])
        )
        tr_vals.append(tr)
    if not tr_vals:
        return None
    atr = np.mean(tr_vals)
    price = c.iloc[-1]
    return atr / price if price > 0 else None


def compute_rsi(close, ticker, date, period=5):
    """RSI over given period."""
    loc = close.index.get_loc(date)
    if loc < period + 1 or ticker not in close.columns:
        return 50.0  # neutral default
    prices = close[ticker].iloc[loc - period:loc + 1]
    changes = prices.diff().dropna()
    gains = changes.clip(lower=0)
    losses = (-changes.clip(upper=0))
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def spy_regime(close, date):
    """Bull if SPY > 200-SMA, else Bear."""
    loc = close.index.get_loc(date)
    if loc < 200:
        return "bull"
    sma200 = close[SPY].iloc[loc - 199:loc + 1].mean()
    return "bull" if close[SPY].iloc[loc] > sma200 else "bear"


# ── Portfolio Tracking ────────────────────────────────────────────────────────
class Portfolio:
    """Track positions, equity, trades with mark-to-market."""

    def __init__(self, cash=ACCOUNT_SIZE):
        self.cash = cash
        self.positions = {}  # ticker -> {shares, entry_price, peak_price}
        self.trades = []
        self.equity_curve = []
        self.regime_returns = {"bull": [], "bear": []}
        self.prev_equity = cash

    def mark_to_market(self, close, date):
        """Current portfolio value."""
        val = self.cash
        for t, pos in self.positions.items():
            p = get_price(close, t, date)
            if pd.notna(p):
                val += pos["shares"] * p
        return val

    def sell(self, ticker, close, date, reason=""):
        """Sell entire position."""
        if ticker not in self.positions:
            return
        pos = self.positions[ticker]
        price = get_price(close, ticker, date)
        if pd.isna(price):
            return
        sell_price = apply_slippage(price, "sell")
        proceeds = pos["shares"] * sell_price
        cost_basis = pos["shares"] * pos["entry_price"]
        pnl = proceeds - cost_basis
        self.cash += proceeds
        self.trades.append({
            "date": str(date.date()), "ticker": ticker, "side": "sell",
            "price": round(sell_price, 2), "shares": round(pos["shares"], 4),
            "pnl": round(pnl, 2), "reason": reason,
        })
        del self.positions[ticker]

    def buy(self, ticker, allocation, close, date, reason=""):
        """Buy shares worth `allocation` dollars."""
        price = get_price(close, ticker, date)
        if pd.isna(price) or price <= 0 or allocation <= 0:
            return
        buy_price = apply_slippage(price, "buy")
        shares = allocation / buy_price
        if shares <= 0:
            return
        self.cash -= shares * buy_price
        self.positions[ticker] = {
            "shares": shares,
            "entry_price": buy_price,
            "peak_price": buy_price,
        }
        self.trades.append({
            "date": str(date.date()), "ticker": ticker, "side": "buy",
            "price": round(buy_price, 2), "shares": round(shares, 4),
            "reason": reason,
        })

    def sell_all(self, close, date, reason=""):
        for t in list(self.positions.keys()):
            self.sell(t, close, date, reason)

    def update_peaks(self, close, date):
        """Update trailing peak prices."""
        for t, pos in self.positions.items():
            p = get_price(close, t, date)
            if pd.notna(p) and p > pos["peak_price"]:
                pos["peak_price"] = p

    def record_day(self, close, date, regime):
        eq = self.mark_to_market(close, date)
        daily_ret = (eq - self.prev_equity) / self.prev_equity if self.prev_equity > 0 else 0
        self.regime_returns[regime].append(daily_ret)
        self.equity_curve.append({"date": str(date.date()), "equity": round(eq, 2), "regime": regime})
        self.prev_equity = eq


# ── Strategy Variants ─────────────────────────────────────────────────────────

def run_variant_A(close, high, low):
    """Simple Top-3 Momentum: buy 3 best 60d return stocks, rebalance monthly, no protection."""
    pf = Portfolio()
    reb_dates = monthly_rebalance_dates(close.index, OOT_START, OOT_END)
    oot_mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        regime = spy_regime(close, date)
        pf.update_peaks(close, date)

        if date in reb_dates:
            mom = compute_momentum_scores(close, date, GROWTH_STOCKS)
            if len(mom) >= TOP_N:
                ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                picks = [t for t, _ in ranked[:TOP_N]]

                # Sell positions not in new picks
                for t in list(pf.positions.keys()):
                    if t not in picks:
                        pf.sell(t, close, date, "rebalance")

                # Calculate allocation
                eq = pf.mark_to_market(close, date)
                n_to_buy = sum(1 for t in picks if t not in pf.positions)
                if n_to_buy > 0:
                    # Equal weight across all 3, but only buy new ones
                    target_per = eq / TOP_N
                    for t in picks:
                        if t not in pf.positions:
                            pf.buy(t, target_per, close, date, "momentum pick")

        pf.record_day(close, date, regime)
    return pf


def run_variant_B(close, high, low):
    """Vol-Scaled Momentum: scale position size inversely to ATR(20)/price."""
    pf = Portfolio()
    reb_dates = monthly_rebalance_dates(close.index, OOT_START, OOT_END)
    oot_mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        regime = spy_regime(close, date)
        pf.update_peaks(close, date)

        if date in reb_dates:
            mom = compute_momentum_scores(close, date, GROWTH_STOCKS)
            if len(mom) >= TOP_N:
                ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                picks = [t for t, _ in ranked[:TOP_N]]

                pf.sell_all(close, date, "rebalance")

                # Inverse-vol weighting
                vol_scores = {}
                for t in picks:
                    atr_pct = compute_atr(close, high, low, t, date)
                    if atr_pct and atr_pct > 0:
                        vol_scores[t] = 1.0 / atr_pct
                    else:
                        vol_scores[t] = 1.0

                total_score = sum(vol_scores.values())
                eq = pf.mark_to_market(close, date)

                for t in picks:
                    weight = vol_scores.get(t, 1.0 / TOP_N) / total_score
                    alloc = eq * weight
                    pf.buy(t, alloc, close, date, f"vol-scaled mom (w={weight:.2f})")

        pf.record_day(close, date, regime)
    return pf


def run_variant_C(close, high, low):
    """Trailing Stop Momentum: top-3 momentum + 15% trailing stop."""
    pf = Portfolio()
    reb_dates = monthly_rebalance_dates(close.index, OOT_START, OOT_END)
    oot_mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[oot_mask]
    stopped_out = set()  # tickers stopped out between rebalances

    for date in oot_dates:
        regime = spy_regime(close, date)
        pf.update_peaks(close, date)

        # Check trailing stops
        for t in list(pf.positions.keys()):
            pos = pf.positions[t]
            cur_p = get_price(close, t, date)
            if pd.notna(cur_p) and pos["peak_price"] > 0:
                drawdown = (cur_p - pos["peak_price"]) / pos["peak_price"]
                if drawdown < -TRAILING_STOP_PCT:
                    pf.sell(t, close, date, f"trailing stop ({drawdown:.1%} from peak)")
                    stopped_out.add(t)

        if date in reb_dates:
            stopped_out.clear()  # reset on rebalance
            mom = compute_momentum_scores(close, date, GROWTH_STOCKS)
            if len(mom) >= TOP_N:
                ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                picks = [t for t, _ in ranked[:TOP_N]]

                for t in list(pf.positions.keys()):
                    if t not in picks:
                        pf.sell(t, close, date, "rebalance")

                eq = pf.mark_to_market(close, date)
                n_to_buy = sum(1 for t in picks if t not in pf.positions)
                if n_to_buy > 0:
                    target_per = eq / TOP_N
                    for t in picks:
                        if t not in pf.positions:
                            pf.buy(t, target_per, close, date, "momentum pick")

        pf.record_day(close, date, regime)
    return pf


def run_variant_D(close, high, low):
    """Dual Momentum: only buy top-3 if absolute momentum > 0. Otherwise GLD."""
    pf = Portfolio()
    reb_dates = monthly_rebalance_dates(close.index, OOT_START, OOT_END)
    oot_mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        regime = spy_regime(close, date)
        pf.update_peaks(close, date)

        if date in reb_dates:
            mom = compute_momentum_scores(close, date, GROWTH_STOCKS)
            if len(mom) >= TOP_N:
                ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)

                # Filter: only stocks with positive absolute momentum
                positive = [(t, m) for t, m in ranked if m > 0]

                picks = []
                for t, m in positive[:TOP_N]:
                    picks.append(t)

                # Fill remaining slots with GLD
                while len(picks) < TOP_N:
                    if "GLD" not in picks:
                        picks.append("GLD")
                        break
                    break  # can't add more GLD

                pf.sell_all(close, date, "rebalance")

                eq = pf.mark_to_market(close, date)
                if picks:
                    target_per = eq / len(picks)
                    for t in picks:
                        pf.buy(t, target_per, close, date,
                               "dual momentum" if t != "GLD" else "safe haven (GLD)")

        pf.record_day(close, date, regime)
    return pf


def run_variant_E(close, high, low):
    """Time-Series Momentum with Reversal Filter: skip overextended stocks (>40% in 60d)."""
    pf = Portfolio()
    reb_dates = monthly_rebalance_dates(close.index, OOT_START, OOT_END)
    oot_mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[oot_mask]

    for date in oot_dates:
        regime = spy_regime(close, date)
        pf.update_peaks(close, date)

        if date in reb_dates:
            mom = compute_momentum_scores(close, date, GROWTH_STOCKS)
            if len(mom) >= TOP_N:
                # Filter: positive momentum AND not overextended
                ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                picks = []
                for t, m in ranked:
                    if m <= 0:
                        continue  # must have positive momentum
                    if m > OVEREXTENDED_THRESHOLD:
                        continue  # skip overextended
                    picks.append(t)
                    if len(picks) >= TOP_N:
                        break

                pf.sell_all(close, date, "rebalance")

                eq = pf.mark_to_market(close, date)
                if picks:
                    target_per = eq / len(picks)
                    for t in picks:
                        pf.buy(t, target_per, close, date, "filtered momentum")

        pf.record_day(close, date, regime)
    return pf


def run_variant_F(close, high, low):
    """Momentum + Mean Reversion Combo: top-3 by 60d momentum, but only enter when RSI(5) < 40."""
    pf = Portfolio()
    reb_dates = monthly_rebalance_dates(close.index, OOT_START, OOT_END)
    oot_mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[oot_mask]

    # Track which stocks are "wanted" but waiting for RSI dip
    wanted = []  # list of tickers we want to buy on RSI dip
    last_reb_picks = []

    for date in oot_dates:
        regime = spy_regime(close, date)
        pf.update_peaks(close, date)

        if date in reb_dates:
            mom = compute_momentum_scores(close, date, GROWTH_STOCKS)
            if len(mom) >= TOP_N:
                ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                picks = [t for t, _ in ranked[:TOP_N]]
                last_reb_picks = picks

                # Sell positions not in new top-3
                for t in list(pf.positions.keys()):
                    if t not in picks:
                        pf.sell(t, close, date, "rebalance")

                # Set wanted list (stocks we want but haven't entered yet)
                wanted = [t for t in picks if t not in pf.positions]

        # Check RSI for wanted stocks — buy on dip
        if wanted:
            eq = pf.mark_to_market(close, date)
            n_current = len(pf.positions)
            for t in list(wanted):
                rsi = compute_rsi(close, t, date, period=5)
                if rsi < RSI_ENTRY_THRESHOLD:
                    # Dip detected — enter
                    n_slots = TOP_N - n_current
                    if n_slots > 0:
                        target_per = eq / TOP_N
                        pf.buy(t, target_per, close, date, f"RSI dip entry (RSI={rsi:.0f})")
                        wanted.remove(t)
                        n_current += 1

        pf.record_day(close, date, regime)
    return pf


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pf):
    """Compute strategy performance metrics from Portfolio."""
    eq = pd.DataFrame(pf.equity_curve)
    eq["equity"] = eq["equity"].astype(float)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")

    daily_returns = eq["equity"].pct_change().dropna()
    if len(daily_returns) < 10:
        return None

    total_return = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) - 1
    n_days = len(daily_returns)
    years = n_days / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 and years > 0 else -1

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    downside = daily_returns[daily_returns < 0]
    down_std = downside.std() if len(downside) > 0 else 1e-6
    sortino = (mean_ret / down_std) * np.sqrt(252) if down_std > 0 else 0

    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    closed = [t for t in pf.trades if t["side"] == "sell"]
    n_trades = len(closed)
    wins = [t for t in closed if t.get("pnl", 0) > 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0

    gross_profit = sum(t["pnl"] for t in closed if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in closed if t.get("pnl", 0) < 0))
    pf_ratio = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0)

    bull_rets = pf.regime_returns.get("bull", [])
    bear_rets = pf.regime_returns.get("bear", [])
    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets)) * np.sqrt(252) if len(bull_rets) > 20 and np.std(bull_rets) > 0 else 0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets)) * np.sqrt(252) if len(bear_rets) > 20 and np.std(bear_rets) > 0 else 0
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "profit_factor": round(pf_ratio, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "n_trades": n_trades,
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_days": len(bull_rets),
        "bear_days": len(bear_rets),
    }


# ── Permutation Test ──────────────────────────────────────────────────────────
def permutation_test(close, actual_sharpe, n_perm=N_PERM):
    """
    Vectorized permutation test: pre-compute daily returns per ticker,
    then for each permutation randomly assign 3 stocks per month and
    compute the portfolio return as equal-weighted average.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    oot_mask = (close.index >= pd.Timestamp(OOT_START)) & (close.index <= pd.Timestamp(OOT_END))
    oot_dates = close.index[oot_mask]

    universe = [t for t in GROWTH_STOCKS if t in close.columns]

    # Pre-compute daily returns for all universe stocks over OOT period
    oot_close = close.loc[oot_dates, universe].ffill()
    daily_rets = oot_close.pct_change().fillna(0)

    # Build month labels for each day
    months = pd.Series(oot_dates.to_period("M"), index=oot_dates)
    unique_months = months.unique()

    perm_sharpes = []
    n_universe = len(universe)

    for _ in range(n_perm):
        # For each month, randomly pick TOP_N stocks
        port_daily = np.zeros(len(oot_dates))
        for m in unique_months:
            mask = (months == m).values
            picks_idx = rng.choice(n_universe, size=min(TOP_N, n_universe), replace=False)
            pick_tickers = [universe[i] for i in picks_idx]
            # Equal-weight daily return for this month
            port_daily[mask] = daily_rets.loc[mask, pick_tickers].mean(axis=1).values

        std = port_daily.std()
        s = (port_daily.mean() / std) * np.sqrt(252) if std > 0 else 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)
    return round(float(p_value), 4), round(float(np.mean(perm_sharpes)), 3), round(float(np.std(perm_sharpes)), 3)


# ── 5-Gate Validation ─────────────────────────────────────────────────────────
def validate_5gate(metrics, p_value):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ──────────────────────────────────────────────────────────────────────
VARIANT_RUNNERS = {
    "A": ("Simple Top-3 Momentum (60d, monthly, no protection)", run_variant_A),
    "B": ("Vol-Scaled Momentum (inverse ATR sizing)", run_variant_B),
    "C": ("Trailing Stop Momentum (15% trailing stop)", run_variant_C),
    "D": ("Dual Momentum (absolute + relative, GLD hedge)", run_variant_D),
    "E": ("Time-Series Momentum + Reversal Filter (skip >40%)", run_variant_E),
    "F": ("Momentum + Mean Reversion Combo (RSI(5)<40 entry)", run_variant_F),
}


def main():
    print("=" * 70)
    print("MOMENTUM WITH CRASH PROTECTION BACKTEST")
    print(f"Walk-Forward OOT: {OOT_START} to {OOT_END} | Account: ${ACCOUNT_SIZE}")
    print("=" * 70)

    close, high, low = download_data()

    results = {}
    all_metrics = {}

    for variant_id, (desc, runner) in VARIANT_RUNNERS.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {variant_id}: {desc}")
        print(f"{'─' * 60}")

        pf = runner(close, high, low)
        metrics = compute_metrics(pf)

        if metrics is None:
            print(f"  !! Not enough data for metrics")
            continue

        print(f"  Return: {metrics['total_return_pct']:+.1f}%  |  CAGR: {metrics['cagr_pct']:+.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}  |  Calmar: {metrics['calmar']:.3f}")
        print(f"  MaxDD: {metrics['max_drawdown_pct']:.1f}%  |  WR: {metrics['win_rate_pct']:.0f}%  |  PF: {metrics['profit_factor']:.2f}")
        print(f"  Trades: {metrics['n_trades']}  |  Final: ${metrics['final_equity']:.2f}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}  |  Bear Sharpe: {metrics['bear_sharpe']:.3f}  |  Gap: {metrics['regime_gap']:.3f}")

        # Permutation test
        print(f"  Running permutation test (n={N_PERM})...", end=" ", flush=True)
        p_val, perm_mean, perm_std = permutation_test(close, metrics["sharpe"])
        print(f"p={p_val:.4f} (perm Sharpe: {perm_mean:.3f} +/- {perm_std:.3f})")

        # 5-gate
        gates = validate_5gate(metrics, p_val)
        passed = sum(1 for k, v in gates.items() if k != "all_passed" and v)
        print(f"  5-Gate: {passed}/5 passed {'✓ PASS' if gates['all_passed'] else '✗ FAIL'}")
        for k, v in gates.items():
            if k != "all_passed":
                print(f"    {'✓' if v else '✗'} {k}")

        all_metrics[variant_id] = metrics

        results[f"variant_{variant_id}"] = {
            "name": desc,
            "metrics": metrics,
            "permutation": {
                "p_value": p_val,
                "perm_sharpe_mean": perm_mean,
                "perm_sharpe_std": perm_std,
            },
            "gates": gates,
            "sample_trades": pf.trades[:10],
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — MOMENTUM WITH CRASH PROTECTION")
    print("=" * 70)
    print(f"{'Var':<4} {'Description':<50} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'WR':>5} {'Gates':>5}")
    print("-" * 85)

    best_sharpe = -999
    best_var = None
    for vid in sorted(all_metrics.keys()):
        m = all_metrics[vid]
        desc = VARIANT_RUNNERS[vid][0][:48]
        g = results[f"variant_{vid}"]["gates"]
        passed = sum(1 for k, v in g.items() if k != "all_passed" and v)
        tag = "PASS" if g["all_passed"] else "FAIL"
        print(f"  {vid:<3} {desc:<50} {m['sharpe']:>7.3f} {m['total_return_pct']:>+7.1f}% {m['max_drawdown_pct']:>6.1f}% {m['win_rate_pct']:>4.0f}% {passed}/5 {tag}")
        if m["sharpe"] > best_sharpe and g["all_passed"]:
            best_sharpe = m["sharpe"]
            best_var = vid

    if best_var:
        print(f"\n>>> BEST PASSING VARIANT: {best_var} — {VARIANT_RUNNERS[best_var][0]}")
    else:
        print(f"\n>>> NO VARIANT PASSED ALL 5 GATES")
        # Find best Sharpe regardless
        best_vid = max(all_metrics, key=lambda v: all_metrics[v]["sharpe"])
        print(f"    Best Sharpe: {best_vid} ({all_metrics[best_vid]['sharpe']:.3f})")

    # Save results
    results["_meta"] = {
        "strategy": "momentum_crash_protection",
        "account_size": ACCOUNT_SIZE,
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "universe": ALL_TICKERS,
        "n_permutations": N_PERM,
        "momentum_lookback": MOM_LOOKBACK,
        "trailing_stop_pct": TRAILING_STOP_PCT,
        "overextended_threshold": OVEREXTENDED_THRESHOLD,
        "rsi_entry_threshold": RSI_ENTRY_THRESHOLD,
        "best_passing_variant": best_var,
        "timestamp": dt.datetime.now().isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
