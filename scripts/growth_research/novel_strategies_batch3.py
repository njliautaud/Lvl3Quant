"""
Novel Strategies Batch 3 — Six Fundamentally Different Strategy Concepts
=========================================================================
Moving BEYOND mean-reversion / dip-buying. Testing momentum, event-driven,
factor rotation, and volatility risk premium approaches.

STRATEGIES:
  A. Momentum Breakout with Volume Confirmation
  B. Earnings Calendar Strangle Proxy (vol compression → expansion)
  C. Sector Momentum Rotation (long top 2, short bottom 2 SPDR sectors)
  D. Post-Earnings Drift with Quality Filter (PEAD — buy gap-ups)
  E. Dividend Capture on Dips
  F. Volatility Risk Premium Harvest (VIX vs realized vol)

Universe: AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, UNH, LLY, AVGO,
          AMD, HD, ABBV, MRK, COST, CRM, NFLX, ADBE, PG, JNJ
Period:   2020-01-01 to 2026-07-01, $300 position, max 2 concurrent

VALIDATION GATES (5-gate):
  1. Sharpe > 0.5
  2. Win rate > 55%
  3. Profit factor > 1.5
  4. Regime gap < 0.50 (bull vs bear: SPY vs 200-SMA)
  5. Permutation test p < 0.05 (200 shuffles)
"""

import sys, os, json, warnings, time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/novel_strategies_batch3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_SIZE = 300.0
MAX_CONCURRENT = 2
BUFFER_START = "2019-06-01"  # extra lookback for indicators
START = "2020-01-01"
END = "2026-07-01"
N_PERMUTATIONS = 200
np.random.seed(42)

QUALITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH",
    "LLY", "AVGO", "AMD", "HD", "ABBV", "MRK", "COST", "CRM",
    "NFLX", "ADBE", "PG", "JNJ"
]

SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]

ts = lambda: datetime.now().strftime('%H:%M:%S')

# =============================================================================
# 1. DATA DOWNLOAD
# =============================================================================
print(f"[{ts()}] === NOVEL STRATEGIES BATCH 3 ===", flush=True)
print(f"[{ts()}] Downloading stock data...", flush=True)

ALL_TICKERS = list(set(QUALITY_UNIVERSE + SECTOR_ETFS + ["SPY", "^VIX"]))

raw_close = {}
raw_high = {}
raw_low = {}
raw_volume = {}

for t in ALL_TICKERS:
    try:
        df = yf.download(t, start=BUFFER_START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  WARNING: {t} empty", flush=True)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        raw_close[t] = df["Close"].rename(t)
        raw_high[t] = df["High"].rename(t)
        raw_low[t] = df["Low"].rename(t)
        raw_volume[t] = df["Volume"].rename(t)
    except Exception as e:
        print(f"  ERROR {t}: {e}", flush=True)

print(f"[{ts()}] Downloaded {len(raw_close)} tickers", flush=True)

close_df = pd.DataFrame(raw_close).sort_index().ffill()
high_df = pd.DataFrame(raw_high).sort_index().ffill()
low_df = pd.DataFrame(raw_low).sort_index().ffill()
volume_df = pd.DataFrame(raw_volume).sort_index().ffill().fillna(0)
close_df.index = pd.to_datetime(close_df.index)
high_df.index = pd.to_datetime(high_df.index)
low_df.index = pd.to_datetime(low_df.index)
volume_df.index = pd.to_datetime(volume_df.index)

# SPY for regime classification
spy_close = close_df["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

vix = close_df["^VIX"].copy() if "^VIX" in close_df.columns else None

print(f"[{ts()}] Data ready: {close_df.shape}, {close_df.index[0].date()} to {close_df.index[-1].date()}", flush=True)

# =============================================================================
# HELPERS
# =============================================================================
def compute_metrics(trades_df):
    """Compute strategy metrics from a trades DataFrame with 'pnl_pct' and 'exit_date'."""
    if trades_df is None or len(trades_df) == 0:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "n_trades": 0,
                "total_pnl": 0, "avg_hold": 0, "max_dd_pct": 0}

    pnl = trades_df["pnl_pct"].values
    n = len(pnl)

    # Annualize assuming ~20 trades/month is daily-ish
    avg_hold = trades_df["hold_days"].mean() if "hold_days" in trades_df.columns else 5
    trades_per_year = 252 / max(avg_hold, 1)

    mean_ret = np.mean(pnl)
    std_ret = np.std(pnl)
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside = np.std(pnl[pnl < 0])
    sortino = (mean_ret / downside) * np.sqrt(trades_per_year) if downside > 0 else 0

    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    wr = (pnl > 0).mean()

    # Drawdown from cumulative PnL
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "n_trades": n,
        "total_pnl_pct": round(np.sum(pnl) * 100, 2),
        "avg_pnl_pct": round(mean_ret * 100, 3),
        "avg_hold": round(avg_hold, 1),
        "max_dd_pct": round(max_dd * 100, 2),
    }


def regime_gap(trades_df, regime_series):
    """Compute |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    if trades_df is None or len(trades_df) < 10:
        return 1.0, 0, 0

    bull_trades = trades_df[trades_df["entry_date"].apply(
        lambda d: regime.get(d, 1) if hasattr(regime, 'get') else regime.reindex([d]).iloc[0] if d in regime.index else 1
    ) == 1]
    bear_trades = trades_df[trades_df["entry_date"].apply(
        lambda d: regime.reindex([d], method='ffill').iloc[0] if d in regime.index or True else 1
    ) == 0]

    # Simpler: use the regime at entry date
    entry_regimes = []
    for d in trades_df["entry_date"]:
        idx = regime.index.searchsorted(d)
        if idx > 0 and idx <= len(regime):
            entry_regimes.append(regime.iloc[min(idx, len(regime)-1)])
        else:
            entry_regimes.append(1)
    trades_df = trades_df.copy()
    trades_df["regime"] = entry_regimes

    bull = trades_df[trades_df["regime"] == 1]["pnl_pct"]
    bear = trades_df[trades_df["regime"] == 0]["pnl_pct"]

    if len(bull) < 5 or len(bear) < 5:
        return 0.99, 0, 0  # not enough data in one regime

    bull_sharpe = (bull.mean() / bull.std()) * np.sqrt(252 / max(trades_df["hold_days"].mean(), 1)) if bull.std() > 0 else 0
    bear_sharpe = (bear.mean() / bear.std()) * np.sqrt(252 / max(trades_df["hold_days"].mean(), 1)) if bear.std() > 0 else 0

    denom = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / denom if denom > 0 else 1.0

    return round(gap, 3), round(bull_sharpe, 3), round(bear_sharpe, 3)


def permutation_test(trades_df, n_perms=N_PERMUTATIONS):
    """Random-entry permutation test: keep hold periods, randomize entry dates.

    For each permutation, randomly pick entry dates from the tradeable date range,
    compute PnL using actual price changes over the same hold period, and compare
    the resulting Sharpe to the actual strategy Sharpe.
    """
    if trades_df is None or len(trades_df) < 10:
        return 1.0

    actual_sharpe = compute_metrics(trades_df)["sharpe"]
    avg_hold_days = int(trades_df["hold_days"].mean())
    n_trades = len(trades_df)

    # Get tickers used and their available dates
    tickers_used = trades_df["ticker"].unique().tolist()
    # Filter to tickers that exist in close_df
    valid_tickers = [t for t in tickers_used if t in close_df.columns and t != "BASKET"]

    if not valid_tickers:
        # For basket strategies (F), use SPY
        valid_tickers = ["SPY"]

    effective_dates = close_df.loc[START:END].index
    max_start_idx = len(effective_dates) - avg_hold_days - 1
    if max_start_idx < 10:
        return 1.0

    count_better = 0
    for _ in range(n_perms):
        # Generate random trades: random ticker + random entry date, same hold period
        random_pnls = []
        for _ in range(n_trades):
            ticker = np.random.choice(valid_tickers)
            start_idx = np.random.randint(0, max_start_idx)
            entry_date = effective_dates[start_idx]
            exit_idx = min(start_idx + avg_hold_days, len(effective_dates) - 1)

            entry_price = close_df[ticker].iloc[close_df.index.get_loc(entry_date)]
            exit_price = close_df[ticker].iloc[close_df.index.get_loc(effective_dates[exit_idx])]

            if entry_price > 0:
                random_pnls.append((exit_price - entry_price) / entry_price)

        if len(random_pnls) < 5:
            continue

        pnl_arr = np.array(random_pnls)
        shuf_mean = np.mean(pnl_arr)
        shuf_std = np.std(pnl_arr)
        shuf_sharpe = (shuf_mean / shuf_std) * np.sqrt(252 / max(avg_hold_days, 1)) if shuf_std > 0 else 0

        if shuf_sharpe >= actual_sharpe:
            count_better += 1

    return round((count_better + 1) / (n_perms + 1), 4)


# =============================================================================
# STRATEGY A: MOMENTUM BREAKOUT WITH VOLUME CONFIRMATION
# =============================================================================
def strategy_a_momentum_breakout():
    print(f"\n[{ts()}] === STRATEGY A: Momentum Breakout + Volume ===", flush=True)

    trades = []
    active_positions = {}  # ticker -> {entry_date, entry_price, highest_since_entry, atr_at_entry}

    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 20:
            continue

        # Check exits first
        to_close = []
        for ticker, pos in active_positions.items():
            if ticker not in close_df.columns:
                continue
            current_price = close_df[ticker].iloc[date_idx]
            days_held = (date - pos["entry_date"]).days

            # Update trailing high
            pos["highest_since_entry"] = max(pos["highest_since_entry"], current_price)

            # Trailing stop: 2 * ATR below the high since entry
            trailing_stop = pos["highest_since_entry"] - 2 * pos["atr_at_entry"]

            exit_signal = False
            if current_price <= trailing_stop:
                exit_signal = True
            elif days_held >= 21:  # max hold
                exit_signal = True

            if exit_signal:
                pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": days_held,
                })
                to_close.append(ticker)

        for t in to_close:
            del active_positions[t]

        # Check entries (if room)
        if len(active_positions) >= MAX_CONCURRENT:
            continue

        for ticker in QUALITY_UNIVERSE:
            if ticker in active_positions or ticker not in close_df.columns:
                continue
            if len(active_positions) >= MAX_CONCURRENT:
                break

            # 20-day high breakout
            lookback_close = close_df[ticker].iloc[max(0, date_idx-20):date_idx+1]
            current_price = close_df[ticker].iloc[date_idx]

            if len(lookback_close) < 20:
                continue

            high_20d = lookback_close.iloc[:-1].max()  # previous 20 days, not including today

            # Volume: 2x average
            lookback_vol = volume_df[ticker].iloc[max(0, date_idx-20):date_idx]
            avg_vol = lookback_vol.mean()
            today_vol = volume_df[ticker].iloc[date_idx]

            # ATR (14-day)
            h = high_df[ticker].iloc[max(0, date_idx-14):date_idx+1]
            l = low_df[ticker].iloc[max(0, date_idx-14):date_idx+1]
            c = close_df[ticker].iloc[max(0, date_idx-14):date_idx+1]
            if len(h) < 14:
                continue
            tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
            atr = tr.iloc[-14:].mean()

            if current_price > high_20d and today_vol > 2 * avg_vol and avg_vol > 0:
                active_positions[ticker] = {
                    "entry_date": date,
                    "entry_price": current_price,
                    "highest_since_entry": current_price,
                    "atr_at_entry": atr,
                }

    # Close remaining
    last_date = effective_dates[-1]
    last_idx = close_df.index.get_loc(last_date)
    for ticker, pos in active_positions.items():
        current_price = close_df[ticker].iloc[last_idx]
        pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "pnl_pct": pnl_pct,
            "hold_days": (last_date - pos["entry_date"]).days,
        })

    trades_df = pd.DataFrame(trades) if trades else None
    print(f"  Trades: {len(trades)}", flush=True)
    return trades_df


# =============================================================================
# STRATEGY B: EARNINGS CALENDAR STRANGLE PROXY
# =============================================================================
def strategy_b_earnings_strangle_proxy():
    print(f"\n[{ts()}] === STRATEGY B: Earnings Strangle Proxy ===", flush=True)
    print(f"  Fetching earnings dates...", flush=True)

    # Get earnings dates from yfinance
    earnings_dates = {}
    for ticker in QUALITY_UNIVERSE:
        try:
            tk = yf.Ticker(ticker)
            # Get historical earnings dates
            cal = tk.earnings_dates
            if cal is not None and len(cal) > 0:
                dates = cal.index.tz_localize(None) if cal.index.tz else cal.index
                earnings_dates[ticker] = sorted(dates.tolist())
                print(f"    {ticker}: {len(earnings_dates[ticker])} earnings dates", flush=True)
            else:
                print(f"    {ticker}: no earnings data", flush=True)
        except Exception as e:
            print(f"    {ticker}: error - {e}", flush=True)

    trades = []
    active_positions = {}

    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 60:
            continue

        # Check exits: exit day after earnings
        to_close = []
        for ticker, pos in active_positions.items():
            # Exit if we've passed the earnings date
            if date > pos["earnings_date"]:
                current_price = close_df[ticker].iloc[date_idx]
                pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                # Take absolute value of pnl to simulate straddle (profit from move either direction)
                straddle_pnl = abs(pnl_pct) - 0.005  # subtract premium proxy (0.5%)
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": straddle_pnl,
                    "hold_days": (date - pos["entry_date"]).days,
                })
                to_close.append(ticker)

        for t in to_close:
            del active_positions[t]

        if len(active_positions) >= MAX_CONCURRENT:
            continue

        # Check entries
        for ticker in QUALITY_UNIVERSE:
            if ticker in active_positions or ticker not in earnings_dates:
                continue
            if len(active_positions) >= MAX_CONCURRENT:
                break

            # Find next earnings date
            future_earnings = [d for d in earnings_dates[ticker] if d > pd.Timestamp(date)]
            if not future_earnings:
                continue
            next_earnings = future_earnings[0]
            days_to_earnings = (next_earnings - pd.Timestamp(date)).days

            if not (5 <= days_to_earnings <= 10):
                continue

            # Vol compression check: 10-day realized vol < 60-day realized vol
            rets = close_df[ticker].pct_change().iloc[max(0, date_idx-60):date_idx+1]
            if len(rets) < 60:
                continue

            vol_10d = rets.iloc[-10:].std() * np.sqrt(252)
            vol_60d = rets.std() * np.sqrt(252)

            if vol_10d < vol_60d:
                active_positions[ticker] = {
                    "entry_date": date,
                    "entry_price": close_df[ticker].iloc[date_idx],
                    "earnings_date": next_earnings,
                }

    # Close remaining
    last_date = effective_dates[-1]
    last_idx = close_df.index.get_loc(last_date)
    for ticker, pos in active_positions.items():
        current_price = close_df[ticker].iloc[last_idx]
        pnl_pct = abs((current_price - pos["entry_price"]) / pos["entry_price"]) - 0.005
        trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "pnl_pct": pnl_pct,
            "hold_days": (last_date - pos["entry_date"]).days,
        })

    trades_df = pd.DataFrame(trades) if trades else None
    print(f"  Trades: {len(trades)}", flush=True)
    return trades_df


# =============================================================================
# STRATEGY C: SECTOR MOMENTUM ROTATION
# =============================================================================
def strategy_c_sector_rotation():
    print(f"\n[{ts()}] === STRATEGY C: Sector Momentum Rotation ===", flush=True)

    sector_close = close_df[[s for s in SECTOR_ETFS if s in close_df.columns]].copy()
    sector_close = sector_close.loc[START:END]

    trades = []

    # Monthly rebalance
    months = sector_close.resample("MS").first().index

    for i in range(1, len(months)):
        rebal_date = months[i]
        # Find actual trading day
        actual_dates = sector_close.index[sector_close.index >= rebal_date]
        if len(actual_dates) == 0:
            continue
        entry_date = actual_dates[0]
        entry_idx = sector_close.index.get_loc(entry_date)

        # Lookback: 1 month return
        lookback_start = entry_idx - 21
        if lookback_start < 0:
            continue

        one_month_returns = {}
        for sector in sector_close.columns:
            ret = (sector_close[sector].iloc[entry_idx] / sector_close[sector].iloc[lookback_start]) - 1
            one_month_returns[sector] = ret

        ranked = sorted(one_month_returns.items(), key=lambda x: x[1], reverse=True)

        # Long top 2, short bottom 2
        longs = [r[0] for r in ranked[:2]]
        shorts = [r[0] for r in ranked[-2:]]

        # Exit date: next rebalance or end
        if i + 1 < len(months):
            next_rebal = months[i + 1]
            exit_dates = sector_close.index[sector_close.index >= next_rebal]
            exit_date = exit_dates[0] if len(exit_dates) > 0 else sector_close.index[-1]
        else:
            exit_date = sector_close.index[-1]

        exit_idx = sector_close.index.get_loc(exit_date)

        for sector in longs:
            entry_price = sector_close[sector].iloc[entry_idx]
            exit_price = sector_close[sector].iloc[exit_idx]
            pnl_pct = (exit_price - entry_price) / entry_price
            trades.append({
                "ticker": sector,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "hold_days": (exit_date - entry_date).days,
                "side": "long",
            })

        for sector in shorts:
            entry_price = sector_close[sector].iloc[entry_idx]
            exit_price = sector_close[sector].iloc[exit_idx]
            pnl_pct = -(exit_price - entry_price) / entry_price  # short
            trades.append({
                "ticker": sector,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "hold_days": (exit_date - entry_date).days,
                "side": "short",
            })

    trades_df = pd.DataFrame(trades) if trades else None
    print(f"  Trades: {len(trades)}", flush=True)
    return trades_df


# =============================================================================
# STRATEGY D: POST-EARNINGS DRIFT (PEAD)
# =============================================================================
def strategy_d_pead():
    print(f"\n[{ts()}] === STRATEGY D: Post-Earnings Drift ===", flush=True)
    print(f"  Fetching earnings dates...", flush=True)

    earnings_dates = {}
    for ticker in QUALITY_UNIVERSE:
        try:
            tk = yf.Ticker(ticker)
            cal = tk.earnings_dates
            if cal is not None and len(cal) > 0:
                dates = cal.index.tz_localize(None) if cal.index.tz else cal.index
                earnings_dates[ticker] = sorted(dates.tolist())
        except:
            pass

    trades = []
    active_positions = {}

    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 2:
            continue

        # Check exits: 30-day hold
        to_close = []
        for ticker, pos in active_positions.items():
            days_held = (date - pos["entry_date"]).days
            if days_held >= 30:
                current_price = close_df[ticker].iloc[date_idx]
                pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": days_held,
                })
                to_close.append(ticker)

        for t in to_close:
            del active_positions[t]

        if len(active_positions) >= MAX_CONCURRENT:
            continue

        # Check: did any stock have earnings yesterday AND gap up >2%?
        for ticker in QUALITY_UNIVERSE:
            if ticker in active_positions or ticker not in earnings_dates:
                continue
            if len(active_positions) >= MAX_CONCURRENT:
                break

            # Check if yesterday was earnings
            yesterday = close_df.index[date_idx - 1] if date_idx > 0 else None
            if yesterday is None:
                continue

            # Was yesterday an earnings date (within 1 day tolerance)?
            is_earnings_day = False
            for ed in earnings_dates.get(ticker, []):
                if abs((pd.Timestamp(yesterday) - pd.Timestamp(ed)).days) <= 1:
                    is_earnings_day = True
                    break

            if not is_earnings_day:
                continue

            # Gap up >2%?
            prev_close = close_df[ticker].iloc[date_idx - 1]
            today_open = close_df[ticker].iloc[date_idx]  # approx with close
            gap_pct = (today_open - prev_close) / prev_close

            if gap_pct > 0.02:
                active_positions[ticker] = {
                    "entry_date": date,
                    "entry_price": today_open,
                }

    # Close remaining
    last_date = effective_dates[-1]
    last_idx = close_df.index.get_loc(last_date)
    for ticker, pos in active_positions.items():
        current_price = close_df[ticker].iloc[last_idx]
        pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "pnl_pct": pnl_pct,
            "hold_days": (last_date - pos["entry_date"]).days,
        })

    trades_df = pd.DataFrame(trades) if trades else None
    print(f"  Trades: {len(trades)}", flush=True)
    return trades_df


# =============================================================================
# STRATEGY E: DIVIDEND CAPTURE ON DIPS
# =============================================================================
def strategy_e_dividend_capture():
    print(f"\n[{ts()}] === STRATEGY E: Dividend Capture on Dips ===", flush=True)

    # Get dividend info
    div_dates = {}
    for ticker in QUALITY_UNIVERSE:
        try:
            tk = yf.Ticker(ticker)
            divs = tk.dividends
            if divs is not None and len(divs) > 0:
                if divs.index.tz:
                    divs.index = divs.index.tz_localize(None)
                # Filter to our period
                divs = divs[(divs.index >= BUFFER_START) & (divs.index <= END)]
                div_dates[ticker] = list(zip(divs.index.tolist(), divs.values.tolist()))
                print(f"    {ticker}: {len(div_dates[ticker])} dividends", flush=True)
        except Exception as e:
            print(f"    {ticker}: error - {e}", flush=True)

    trades = []
    active_positions = {}

    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 20:
            continue

        # Check exits: 5 days after ex-div
        to_close = []
        for ticker, pos in active_positions.items():
            days_since_exdiv = (date - pos["ex_div_date"]).days
            if days_since_exdiv >= 5:
                current_price = close_df[ticker].iloc[date_idx]
                price_pnl = (current_price - pos["entry_price"]) / pos["entry_price"]
                div_yield = pos["div_amount"] / pos["entry_price"]
                total_pnl = price_pnl + div_yield
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": total_pnl,
                    "hold_days": (date - pos["entry_date"]).days,
                    "div_yield": div_yield,
                })
                to_close.append(ticker)

        for t in to_close:
            del active_positions[t]

        if len(active_positions) >= MAX_CONCURRENT:
            continue

        # Check entries: 3-5 days before ex-div AND >3% below 20-SMA
        for ticker in QUALITY_UNIVERSE:
            if ticker in active_positions or ticker not in div_dates:
                continue
            if len(active_positions) >= MAX_CONCURRENT:
                break

            current_price = close_df[ticker].iloc[date_idx]
            sma20 = close_df[ticker].iloc[max(0, date_idx-20):date_idx+1].mean()

            # Must be >3% below 20-SMA
            if current_price >= sma20 * 0.97:
                continue

            # Check if ex-div is 3-5 days away
            for ex_date, div_amount in div_dates[ticker]:
                days_to_exdiv = (pd.Timestamp(ex_date) - pd.Timestamp(date)).days
                if 3 <= days_to_exdiv <= 5:
                    active_positions[ticker] = {
                        "entry_date": date,
                        "entry_price": current_price,
                        "ex_div_date": pd.Timestamp(ex_date),
                        "div_amount": div_amount,
                    }
                    break

    # Close remaining
    last_date = effective_dates[-1]
    last_idx = close_df.index.get_loc(last_date)
    for ticker, pos in active_positions.items():
        current_price = close_df[ticker].iloc[last_idx]
        price_pnl = (current_price - pos["entry_price"]) / pos["entry_price"]
        div_yield = pos["div_amount"] / pos["entry_price"]
        trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "pnl_pct": price_pnl + div_yield,
            "hold_days": (last_date - pos["entry_date"]).days,
        })

    trades_df = pd.DataFrame(trades) if trades else None
    print(f"  Trades: {len(trades)}", flush=True)
    return trades_df


# =============================================================================
# STRATEGY F: VOLATILITY RISK PREMIUM HARVEST
# =============================================================================
def strategy_f_vol_risk_premium():
    print(f"\n[{ts()}] === STRATEGY F: Vol Risk Premium Harvest ===", flush=True)

    if vix is None:
        print("  ERROR: No VIX data available", flush=True)
        return None

    trades = []
    active_position = None

    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 20:
            continue

        current_vix = vix.iloc[date_idx] if date_idx < len(vix) else None
        if current_vix is None or np.isnan(current_vix):
            continue

        # 20-day realized vol of SPY (annualized)
        spy_rets = spy_close.pct_change().iloc[max(0, date_idx-20):date_idx+1]
        realized_vol = spy_rets.std() * np.sqrt(252) * 100  # as percentage points

        vol_premium = current_vix - realized_vol  # VIX - RV

        # Check exit
        if active_position is not None:
            days_held = (date - active_position["entry_date"]).days

            # Exit when vol premium normalizes (<2) or max hold 21 days
            exit_signal = vol_premium < 2 or days_held >= 21

            if exit_signal:
                # We buy a basket of quality stocks equally weighted
                basket_pnl = 0
                n_stocks = 0
                for ticker in QUALITY_UNIVERSE[:10]:  # top 10 for simplicity
                    if ticker in close_df.columns:
                        entry_p = close_df[ticker].iloc[close_df.index.get_loc(active_position["entry_date"])]
                        exit_p = close_df[ticker].iloc[date_idx]
                        basket_pnl += (exit_p - entry_p) / entry_p
                        n_stocks += 1

                if n_stocks > 0:
                    avg_pnl = basket_pnl / n_stocks
                    trades.append({
                        "ticker": "BASKET",
                        "entry_date": active_position["entry_date"],
                        "exit_date": date,
                        "entry_price": 0,
                        "exit_price": 0,
                        "pnl_pct": avg_pnl,
                        "hold_days": days_held,
                        "vol_premium_entry": active_position["vol_premium"],
                    })
                active_position = None

        # Check entry: VIX > realized vol by 5+ points
        if active_position is None and vol_premium >= 5:
            active_position = {
                "entry_date": date,
                "vol_premium": vol_premium,
            }

    # Close remaining
    if active_position is not None:
        last_date = effective_dates[-1]
        last_idx = close_df.index.get_loc(last_date)
        basket_pnl = 0
        n_stocks = 0
        for ticker in QUALITY_UNIVERSE[:10]:
            if ticker in close_df.columns:
                entry_p = close_df[ticker].iloc[close_df.index.get_loc(active_position["entry_date"])]
                exit_p = close_df[ticker].iloc[last_idx]
                basket_pnl += (exit_p - entry_p) / entry_p
                n_stocks += 1
        if n_stocks > 0:
            trades.append({
                "ticker": "BASKET",
                "entry_date": active_position["entry_date"],
                "exit_date": last_date,
                "entry_price": 0,
                "exit_price": 0,
                "pnl_pct": basket_pnl / n_stocks,
                "hold_days": (last_date - active_position["entry_date"]).days,
            })

    trades_df = pd.DataFrame(trades) if trades else None
    print(f"  Trades: {len(trades)}", flush=True)
    return trades_df


# =============================================================================
# MAIN: RUN ALL STRATEGIES + 5-GATE VALIDATION
# =============================================================================
print(f"\n[{ts()}] ========== RUNNING ALL 6 STRATEGIES ==========", flush=True)

strategies = {
    "A: Momentum Breakout": strategy_a_momentum_breakout,
    "B: Earnings Strangle Proxy": strategy_b_earnings_strangle_proxy,
    "C: Sector Momentum Rotation": strategy_c_sector_rotation,
    "D: Post-Earnings Drift": strategy_d_pead,
    "E: Dividend Capture Dips": strategy_e_dividend_capture,
    "F: Vol Risk Premium": strategy_f_vol_risk_premium,
}

results = {}
all_trades = {}

for name, func in strategies.items():
    t0 = time.time()
    try:
        trades_df = func()
        elapsed = time.time() - t0
        print(f"  [{name}] completed in {elapsed:.1f}s", flush=True)

        if trades_df is not None and len(trades_df) > 0:
            metrics = compute_metrics(trades_df)
            gap, bull_s, bear_s = regime_gap(trades_df, regime)

            print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...", flush=True)
            p_val = permutation_test(trades_df)

            # 5-gate validation
            gate1 = metrics["sharpe"] > 0.5
            gate2 = metrics["wr"] > 0.55
            gate3 = metrics["pf"] > 1.5
            gate4 = gap < 0.50
            gate5 = p_val < 0.05
            gates_passed = sum([gate1, gate2, gate3, gate4, gate5])

            results[name] = {
                **metrics,
                "regime_gap": gap,
                "bull_sharpe": bull_s,
                "bear_sharpe": bear_s,
                "perm_p": p_val,
                "gate1_sharpe": "PASS" if gate1 else "FAIL",
                "gate2_wr": "PASS" if gate2 else "FAIL",
                "gate3_pf": "PASS" if gate3 else "FAIL",
                "gate4_regime": "PASS" if gate4 else "FAIL",
                "gate5_perm": "PASS" if gate5 else "FAIL",
                "gates_passed": f"{gates_passed}/5",
            }
            all_trades[name] = trades_df

            print(f"  Sharpe={metrics['sharpe']}, WR={metrics['wr']:.1%}, PF={metrics['pf']}, "
                  f"Regime Gap={gap}, Perm p={p_val}, Gates={gates_passed}/5", flush=True)
        else:
            results[name] = {"error": "No trades generated", "gates_passed": "0/5"}
            print(f"  NO TRADES GENERATED", flush=True)
    except Exception as e:
        import traceback
        results[name] = {"error": str(e), "gates_passed": "0/5"}
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print(f"\n\n[{ts()}] ============================================", flush=True)
print(f"         NOVEL STRATEGIES BATCH 3 — RESULTS", flush=True)
print(f"         ============================================\n", flush=True)

header = f"{'Strategy':<32} {'Sharpe':>7} {'WR':>7} {'PF':>7} {'Trades':>7} {'Regime':>7} {'Perm-p':>7} {'Gates':>7}"
print(header, flush=True)
print("-" * len(header), flush=True)

five_of_five = []

for name, r in results.items():
    if "error" in r:
        print(f"{name:<32} {'ERROR':>7} {r['error'][:40]}", flush=True)
    else:
        line = (f"{name:<32} {r['sharpe']:>7.3f} {r['wr']:>6.1%} {r['pf']:>7.3f} "
                f"{r['n_trades']:>7} {r['regime_gap']:>7.3f} {r['perm_p']:>7.4f} {r['gates_passed']:>7}")
        print(line, flush=True)
        if r["gates_passed"] == "5/5":
            five_of_five.append(name)

print(f"\n{'='*80}", flush=True)
print(f"\nGATE DETAIL:", flush=True)
for name, r in results.items():
    if "error" not in r:
        print(f"\n  {name}:", flush=True)
        print(f"    G1 Sharpe>0.5:   {r['gate1_sharpe']} (Sharpe={r['sharpe']})", flush=True)
        print(f"    G2 WR>55%:       {r['gate2_wr']} (WR={r['wr']:.1%})", flush=True)
        print(f"    G3 PF>1.5:       {r['gate3_pf']} (PF={r['pf']})", flush=True)
        print(f"    G4 Regime<0.50:  {r['gate4_regime']} (Gap={r['regime_gap']}, Bull={r['bull_sharpe']}, Bear={r['bear_sharpe']})", flush=True)
        print(f"    G5 Perm p<0.05:  {r['gate5_perm']} (p={r['perm_p']})", flush=True)
        print(f"    Total PnL:       {r['total_pnl_pct']:.2f}%, Avg hold: {r['avg_hold']:.0f} days", flush=True)
        print(f"    Sortino:         {r['sortino']}", flush=True)

if five_of_five:
    print(f"\n*** STRATEGIES PASSING 5/5 GATES (ready for adversarial): {', '.join(five_of_five)} ***", flush=True)
else:
    print(f"\n*** No strategy passed all 5 gates. ***", flush=True)
    # Find best
    best = max(results.items(), key=lambda x: int(x[1].get("gates_passed", "0/5").split("/")[0]) if "error" not in x[1] else 0)
    print(f"*** Best: {best[0]} with {best[1].get('gates_passed', '0/5')} ***", flush=True)

# Save results
with open(OUTPUT_DIR / "results.json", "w") as f:
    # Convert non-serializable types
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {kk: str(vv) if not isinstance(vv, (int, float, str)) else vv for kk, vv in v.items()}
    json.dump(clean_results, f, indent=2)

# Save trade logs
for name, trades_df in all_trades.items():
    safe_name = name.replace(":", "").replace(" ", "_").lower()
    trades_df.to_csv(OUTPUT_DIR / f"trades_{safe_name}.csv", index=False)

print(f"\n[{ts()}] Results saved to {OUTPUT_DIR}", flush=True)
print(f"[{ts()}] DONE.", flush=True)
