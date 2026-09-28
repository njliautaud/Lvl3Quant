"""
Novel Strategies Batch 7 — Anomaly-Based Share Trading
=======================================================
Genuinely novel strategies NOT previously tested:
  A. Overnight Drift Capture (buy close, sell open — overnight equity premium)
  B. Intraday Momentum / Morning Continuation (gap-up → continuation proxy)
  C. Options Expiration Pinning (OpEx Friday round-number magnet)
  D. Tax-Loss Harvesting Reversal (Nov losers → Jan reversal)
  E. Buyback Execution Window (post-earnings quiet period floor)
  F. Cross-Stock Lead-Lag (NVDA→AMD, AAPL→suppliers, META→GOOGL)

Universe: quality mega-caps. Period: 2020-01-01 to 2026-07-01.
Position: $300, max 2 concurrent. 5-gate validation.

VALIDATION GATES:
  1. Sharpe > 0.5
  2. Win rate > 55%
  3. Profit factor > 1.5
  4. Regime gap < 0.50
  5. Permutation test p < 0.05 (200 shuffles)
"""

import sys, os, warnings, time
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/novel_strategies_batch7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_SIZE = 300.0
MAX_CONCURRENT = 2
BUFFER_START = "2019-01-01"
START = "2020-01-01"
END = "2026-07-01"
N_PERMUTATIONS = 200
np.random.seed(42)

QUALITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH",
    "LLY", "AVGO", "AMD", "HD", "ABBV", "MRK", "COST", "CRM",
    "NFLX", "ADBE", "PG", "JNJ"
]

# Lead-lag pairs for Strategy F
LEAD_LAG_PAIRS = [
    ("NVDA", "AMD", "GPU sector"),
    ("AAPL", "AVGO", "AAPL→supplier"),
    ("AAPL", "QCOM", "AAPL→supplier2"),
    ("META", "GOOGL", "Ad spending"),
]

ts = lambda: datetime.now().strftime('%H:%M:%S')

# =============================================================================
# 1. DATA DOWNLOAD
# =============================================================================
print(f"[{ts()}] === NOVEL STRATEGIES BATCH 7 ===", flush=True)
print(f"[{ts()}] Downloading data...", flush=True)

ALL_TICKERS = list(set(
    QUALITY_UNIVERSE +
    [p[0] for p in LEAD_LAG_PAIRS] + [p[1] for p in LEAD_LAG_PAIRS] +
    ["SPY", "^VIX"]
))

raw_close = {}
raw_open = {}
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
        raw_open[t] = df["Open"].rename(t)
        raw_high[t] = df["High"].rename(t)
        raw_low[t] = df["Low"].rename(t)
        raw_volume[t] = df["Volume"].rename(t)
    except Exception as e:
        print(f"  ERROR {t}: {e}", flush=True)

print(f"[{ts()}] Downloaded {len(raw_close)} tickers", flush=True)

close_df = pd.DataFrame(raw_close).sort_index().ffill()
open_df = pd.DataFrame(raw_open).sort_index().ffill()
high_df = pd.DataFrame(raw_high).sort_index().ffill()
low_df = pd.DataFrame(raw_low).sort_index().ffill()
volume_df = pd.DataFrame(raw_volume).sort_index().ffill().fillna(0)

for df_ in [close_df, open_df, high_df, low_df, volume_df]:
    df_.index = pd.to_datetime(df_.index)

# SPY for regime classification
spy_close = close_df["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

print(f"[{ts()}] Data ready: {close_df.shape}, {close_df.index[0].date()} to {close_df.index[-1].date()}", flush=True)

# =============================================================================
# HELPERS
# =============================================================================
def compute_metrics(trades_df):
    """Compute strategy metrics from trades DataFrame with 'pnl_pct'."""
    if trades_df is None or len(trades_df) == 0:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "n_trades": 0,
                "total_pnl_pct": 0, "avg_pnl_pct": 0, "avg_hold": 0, "max_dd_pct": 0}

    pnl = trades_df["pnl_pct"].values
    n = len(pnl)
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


def regime_gap(trades_df):
    """Compute |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    if trades_df is None or len(trades_df) < 10:
        return 1.0, 0, 0

    entry_regimes = []
    for d in trades_df["entry_date"]:
        idx = regime.index.searchsorted(d)
        if idx > 0 and idx <= len(regime):
            entry_regimes.append(regime.iloc[min(idx, len(regime)-1)])
        else:
            entry_regimes.append(1)
    tdf = trades_df.copy()
    tdf["regime"] = entry_regimes

    bull = tdf[tdf["regime"] == 1]["pnl_pct"]
    bear = tdf[tdf["regime"] == 0]["pnl_pct"]

    if len(bull) < 5 or len(bear) < 5:
        return 0.99, 0, 0

    avg_hold = max(tdf["hold_days"].mean(), 1)
    bull_sharpe = (bull.mean() / bull.std()) * np.sqrt(252 / avg_hold) if bull.std() > 0 else 0
    bear_sharpe = (bear.mean() / bear.std()) * np.sqrt(252 / avg_hold) if bear.std() > 0 else 0

    denom = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / denom if denom > 0 else 1.0

    return round(gap, 3), round(bull_sharpe, 3), round(bear_sharpe, 3)


def permutation_test(trades_df, valid_tickers=None, n_perms=N_PERMUTATIONS):
    """Random-entry permutation test."""
    if trades_df is None or len(trades_df) < 10:
        return 1.0

    actual_sharpe = compute_metrics(trades_df)["sharpe"]
    avg_hold_days = max(int(trades_df["hold_days"].mean()), 1)
    n_trades = len(trades_df)

    if valid_tickers is None:
        tickers_used = trades_df["ticker"].unique().tolist()
        valid_tickers = [t for t in tickers_used if t in close_df.columns]
    if not valid_tickers:
        valid_tickers = ["SPY"]

    effective_dates = close_df.loc[START:END].index
    max_start_idx = len(effective_dates) - avg_hold_days - 1
    if max_start_idx < 10:
        return 1.0

    count_better = 0
    for _ in range(n_perms):
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
        shuf_sharpe = (np.mean(pnl_arr) / np.std(pnl_arr)) * np.sqrt(252 / avg_hold_days) if np.std(pnl_arr) > 0 else 0
        if shuf_sharpe >= actual_sharpe:
            count_better += 1

    return round((count_better + 1) / (n_perms + 1), 4)


# =============================================================================
# STRATEGY A: OVERNIGHT DRIFT CAPTURE
# =============================================================================
def strategy_a_overnight_drift():
    """Buy at close, sell at open next day. Equity premium concentrates overnight."""
    print(f"\n[{ts()}] === STRATEGY A: Overnight Drift Capture ===", flush=True)

    trades = []
    effective_dates = close_df.loc[START:END].index

    # For each day, compute overnight return (close→open next day) for each stock
    # Pick the best candidates based on recent overnight drift persistence
    active_positions = {}

    for i, date in enumerate(effective_dates):
        date_idx = close_df.index.get_loc(date)
        if date_idx < 60:
            continue

        # Close out any positions from yesterday (sell at today's open)
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            ticker = pos["ticker"]
            if ticker in open_df.columns:
                exit_price = open_df[ticker].iloc[date_idx]
                entry_price = pos["entry_price"]
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": 1,
                    "signal": "overnight_drift"
                })
            active_positions[key] = None

        # New entries: buy at today's close
        # Rank stocks by their average overnight return over the past 20 days
        scores = {}
        for ticker in QUALITY_UNIVERSE:
            if ticker not in close_df.columns or ticker not in open_df.columns:
                continue
            # Compute trailing overnight returns: close[t-1] → open[t]
            overnight_rets = []
            for j in range(1, 21):
                idx = date_idx - j
                if idx < 1:
                    break
                prev_close = close_df[ticker].iloc[idx - 1]
                curr_open = open_df[ticker].iloc[idx]
                if prev_close > 0:
                    overnight_rets.append((curr_open - prev_close) / prev_close)

            if len(overnight_rets) < 10:
                continue

            avg_on = np.mean(overnight_rets)
            consistency = (np.array(overnight_rets) > 0).mean()

            # Only enter if overnight returns have been consistently positive
            if avg_on > 0.0005 and consistency > 0.55:
                scores[ticker] = avg_on * consistency

        # Pick top 2 by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        for ticker, score in ranked[:MAX_CONCURRENT]:
            entry_price = close_df[ticker].iloc[date_idx]
            active_positions[ticker] = {
                "ticker": ticker,
                "entry_date": date,
                "entry_price": entry_price,
            }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY B: INTRADAY MOMENTUM (MORNING CONTINUATION PROXY)
# =============================================================================
def strategy_b_morning_momentum():
    """If stock gaps up >0.5% at open, buy. Proxy: test gap-up days → same-day close performance."""
    print(f"\n[{ts()}] === STRATEGY B: Intraday Momentum (Morning Continuation) ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 20:
            continue

        # Close out positions from previous day (next day open)
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            ticker = pos["ticker"]
            # Exit at today's close (same-day trade proxy) or next day open
            exit_price = close_df[ticker].iloc[date_idx]
            if hold_days >= 1:
                entry_price = pos["entry_price"]
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": "morning_momentum"
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Scan for gap-ups >0.5%
        candidates = []
        for ticker in QUALITY_UNIVERSE:
            if ticker not in close_df.columns or ticker not in open_df.columns:
                continue
            if ticker in active_positions and active_positions.get(ticker) is not None:
                continue

            prev_close = close_df[ticker].iloc[date_idx - 1]
            today_open = open_df[ticker].iloc[date_idx]
            if prev_close <= 0:
                continue

            gap_pct = (today_open - prev_close) / prev_close

            # Gap up >0.5% but <3% (avoid earnings gaps which reverse)
            if 0.005 < gap_pct < 0.03:
                # Additional filter: 5-day momentum positive (trend confirmation)
                ret_5d = (close_df[ticker].iloc[date_idx - 1] - close_df[ticker].iloc[date_idx - 6]) / close_df[ticker].iloc[date_idx - 6]
                if ret_5d > 0:
                    candidates.append((ticker, gap_pct, ret_5d))

        # Rank by gap size * momentum
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)

        for ticker, gap, mom in candidates[:MAX_CONCURRENT - n_active]:
            # Entry at open price (gap-up buy)
            entry_price = open_df[ticker].iloc[date_idx]
            active_positions[ticker] = {
                "ticker": ticker,
                "entry_date": date,
                "entry_price": entry_price,
            }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY C: OPTIONS EXPIRATION PINNING
# =============================================================================
def strategy_c_opex_pinning():
    """On monthly OpEx Fridays, buy stocks near round numbers expecting pinning."""
    print(f"\n[{ts()}] === STRATEGY C: Options Expiration Pinning ===", flush=True)

    trades = []
    effective_dates = close_df.loc[START:END].index

    # Find 3rd Fridays (monthly options expiration)
    opex_fridays = []
    for date in effective_dates:
        if date.weekday() == 4:  # Friday
            if 15 <= date.day <= 21:  # 3rd week
                opex_fridays.append(date)

    print(f"  Found {len(opex_fridays)} OpEx Fridays", flush=True)

    for opex_date in opex_fridays:
        date_idx = close_df.index.get_loc(opex_date)
        if date_idx < 5:
            continue

        # Look at Wednesday before OpEx (entry day = 2 days before)
        entry_idx = date_idx - 2
        if entry_idx < 0:
            continue
        entry_date = close_df.index[entry_idx]

        # Find stocks near round numbers
        candidates = []
        for ticker in QUALITY_UNIVERSE:
            if ticker not in close_df.columns:
                continue

            price = close_df[ticker].iloc[entry_idx]
            if price <= 0:
                continue

            # Determine pin level (nearest $5 or $10 increment)
            if price > 100:
                pin_increment = 10
            elif price > 50:
                pin_increment = 5
            else:
                pin_increment = 5

            nearest_round = round(price / pin_increment) * pin_increment
            distance_pct = abs(price - nearest_round) / price

            # Within 1% of a round number
            if distance_pct < 0.01:
                # Direction: if price is below pin, expect pull up; if above, pull down
                direction = 1 if price < nearest_round else -1
                candidates.append((ticker, nearest_round, distance_pct, direction, price))

        # Pick top 2 by closest to round number
        candidates.sort(key=lambda x: x[2])

        n_entered = 0
        for ticker, pin, dist, direction, entry_price in candidates:
            if n_entered >= MAX_CONCURRENT:
                break

            # Only take long trades (buy expecting pull toward pin)
            if direction != 1:
                continue

            exit_price = close_df[ticker].iloc[date_idx]  # OpEx Friday close
            pnl_pct = (exit_price - entry_price) / entry_price

            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": opex_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "hold_days": 2,
                "signal": f"opex_pin_{pin}"
            })
            n_entered += 1

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY D: TAX-LOSS HARVESTING REVERSAL
# =============================================================================
def strategy_d_tax_loss_reversal():
    """Buy quality stocks that are YTD losers in late Nov, hold through Jan (tax-loss reversal)."""
    print(f"\n[{ts()}] === STRATEGY D: Tax-Loss Harvesting Reversal ===", flush=True)

    trades = []
    effective_dates = close_df.loc[START:END].index

    # For each year, find stocks down >10% YTD by late November
    years = sorted(set(d.year for d in effective_dates))

    for year in years:
        # Entry: last trading day of November or first few days of December
        nov_dates = [d for d in effective_dates if d.year == year and d.month == 11 and d.day >= 20]
        jan_dates = [d for d in effective_dates if d.year == year + 1 and d.month == 1 and d.day >= 25]

        if not nov_dates or not jan_dates:
            continue

        entry_date = nov_dates[-1]  # Last trading day of Nov (or close to it)
        exit_date = jan_dates[-1]   # Late January

        entry_idx = close_df.index.get_loc(entry_date)

        # Find Jan 1 price for YTD calculation
        jan_start_dates = [d for d in close_df.index if d.year == year and d.month == 1]
        if not jan_start_dates:
            continue
        jan_start_idx = close_df.index.get_loc(jan_start_dates[0])

        # Find stocks down >10% YTD
        losers = []
        for ticker in QUALITY_UNIVERSE:
            if ticker not in close_df.columns:
                continue

            jan_price = close_df[ticker].iloc[jan_start_idx]
            nov_price = close_df[ticker].iloc[entry_idx]
            if jan_price <= 0:
                continue

            ytd_return = (nov_price - jan_price) / jan_price
            if ytd_return < -0.10:
                losers.append((ticker, ytd_return, nov_price))

        # Sort by worst performers (most tax-loss selling pressure)
        losers.sort(key=lambda x: x[1])

        exit_idx = close_df.index.get_loc(exit_date)

        for ticker, ytd_ret, entry_price in losers[:MAX_CONCURRENT]:
            exit_price = close_df[ticker].iloc[exit_idx]
            pnl_pct = (exit_price - entry_price) / entry_price
            hold_days = (exit_date - entry_date).days

            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "hold_days": hold_days,
                "signal": f"tax_loss_ytd_{ytd_ret:.1%}"
            })

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY E: BUYBACK EXECUTION WINDOW
# =============================================================================
def strategy_e_buyback_window():
    """Buy quality stocks during estimated buyback windows (post-earnings quiet period)
    when they're also in a dip. Proxy: 3-6 weeks after quarter end (when buyback windows open)
    + stock is down >3% from 20-day high."""
    print(f"\n[{ts()}] === STRATEGY E: Buyback Execution Window ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index

    # Quarter end dates (approximate earnings release + 2 weeks = buyback window opens)
    # Typical earnings: Q1 late Apr, Q2 late Jul, Q3 late Oct, Q4 late Jan
    # Buyback windows open ~2 weeks after earnings, close ~2 weeks before next quarter end
    # Proxy: weeks 3-6 after quarter end month

    def is_buyback_window(date):
        """Estimate if date falls in a typical buyback execution window."""
        month, day = date.month, date.day
        # Post Q4 earnings (late Jan → Feb-Mar buyback window)
        if (month == 2 and day >= 10) or month == 3:
            return True
        # Post Q1 earnings (late Apr → May-Jun buyback window)
        if (month == 5 and day >= 10) or month == 6:
            return True
        # Post Q2 earnings (late Jul → Aug-Sep buyback window)
        if (month == 8 and day >= 10) or month == 9:
            return True
        # Post Q3 earnings (late Oct → Nov-early Dec buyback window)
        if (month == 11 and day >= 10) or (month == 12 and day <= 15):
            return True
        return False

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 30:
            continue

        # Check exits
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            ticker = pos["ticker"]
            current_price = close_df[ticker].iloc[date_idx]
            entry_price = pos["entry_price"]
            pnl_pct = (current_price - entry_price) / entry_price

            # Exit: 21 day max hold, +5% profit, -4% stop
            if hold_days >= 21 or pnl_pct >= 0.05 or pnl_pct <= -0.04:
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": "buyback_window_dip"
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        if not is_buyback_window(date):
            continue

        # Find stocks in a dip during buyback window
        candidates = []
        for ticker in QUALITY_UNIVERSE:
            if ticker not in close_df.columns:
                continue
            if ticker in active_positions and active_positions.get(ticker) is not None:
                continue

            current_price = close_df[ticker].iloc[date_idx]
            high_20d = close_df[ticker].iloc[date_idx-20:date_idx].max()

            if high_20d <= 0:
                continue

            drawdown = (current_price - high_20d) / high_20d

            # Stock is down >3% from 20-day high (dip) but not crashing (>-10%)
            if -0.10 < drawdown < -0.03:
                # Also check: stock is above 50-day SMA (not in breakdown)
                sma50 = close_df[ticker].iloc[date_idx-50:date_idx].mean()
                if current_price > sma50:
                    candidates.append((ticker, drawdown, current_price))

        # Pick the ones with moderate dips (not too deep)
        candidates.sort(key=lambda x: x[1], reverse=True)  # least negative first

        for ticker, dd, entry_price in candidates[:MAX_CONCURRENT - n_active]:
            active_positions[ticker] = {
                "ticker": ticker,
                "entry_date": date,
                "entry_price": entry_price,
            }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY F: CROSS-STOCK LEAD-LAG
# =============================================================================
def strategy_f_lead_lag():
    """If leader stock gaps up >2%, buy laggard expecting it to follow within 1-3 days.
    Hold for 5 trading days max."""
    print(f"\n[{ts()}] === STRATEGY F: Cross-Stock Lead-Lag ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 20:
            continue

        # Check exits
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            ticker = pos["ticker"]
            current_price = close_df[ticker].iloc[date_idx]
            entry_price = pos["entry_price"]
            pnl_pct = (current_price - entry_price) / entry_price

            # Exit: 5 day hold, or +3% profit (quick follow trade), or -3% stop
            if hold_days >= 5 or pnl_pct >= 0.03 or pnl_pct <= -0.03:
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": pos["signal"]
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Check lead-lag pairs
        for leader, lagger, pair_name in LEAD_LAG_PAIRS:
            if leader not in close_df.columns or lagger not in close_df.columns:
                continue
            if lagger in active_positions and active_positions.get(lagger) is not None:
                continue

            # Leader's return today
            leader_ret = (close_df[leader].iloc[date_idx] - close_df[leader].iloc[date_idx-1]) / close_df[leader].iloc[date_idx-1]

            # Lagger's return today
            lagger_ret = (close_df[lagger].iloc[date_idx] - close_df[lagger].iloc[date_idx-1]) / close_df[lagger].iloc[date_idx-1]

            # Signal: leader up >2%, lagger hasn't followed yet (<1% move)
            if leader_ret > 0.02 and lagger_ret < 0.01:
                entry_price = close_df[lagger].iloc[date_idx]
                active_positions[lagger] = {
                    "ticker": lagger,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "signal": f"lead_lag_{pair_name}"
                }
                n_active += 1
                if n_active >= MAX_CONCURRENT:
                    break

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# RUN ALL STRATEGIES
# =============================================================================
print(f"\n[{ts()}] ========== RUNNING ALL 6 STRATEGIES ==========", flush=True)

strategies = {
    "A: Overnight Drift Capture": strategy_a_overnight_drift,
    "B: Morning Momentum (Gap)": strategy_b_morning_momentum,
    "C: OpEx Pinning": strategy_c_opex_pinning,
    "D: Tax-Loss Reversal": strategy_d_tax_loss_reversal,
    "E: Buyback Window Dip": strategy_e_buyback_window,
    "F: Cross-Stock Lead-Lag": strategy_f_lead_lag,
}

results = {}

for name, func in strategies.items():
    t0 = time.time()
    trades_df, metrics = func()
    elapsed = time.time() - t0
    print(f"  [{ts()}] Completed in {elapsed:.1f}s", flush=True)

    # Validation gates
    if trades_df is not None and len(trades_df) >= 10:
        print(f"  Running regime gap analysis...", flush=True)
        gap, bull_s, bear_s = regime_gap(trades_df)
        print(f"  Running permutation test (200 shuffles)...", flush=True)
        perm_p = permutation_test(trades_df)
    else:
        gap, bull_s, bear_s = 1.0, 0, 0
        perm_p = 1.0

    # 5-gate validation
    g1 = metrics["sharpe"] > 0.5
    g2 = metrics["wr"] > 0.55
    g3 = metrics["pf"] > 1.5
    g4 = gap < 0.50
    g5 = perm_p < 0.05

    gates_passed = sum([g1, g2, g3, g4, g5])
    validated = gates_passed == 5

    results[name] = {
        **metrics,
        "regime_gap": gap,
        "bull_sharpe": bull_s,
        "bear_sharpe": bear_s,
        "perm_p": perm_p,
        "gates_passed": f"{gates_passed}/5",
        "validated": validated,
        "g1_sharpe": g1, "g2_wr": g2, "g3_pf": g3, "g4_regime": g4, "g5_perm": g5,
    }

    status = "VALIDATED" if validated else f"FAILED ({gates_passed}/5)"
    print(f"  Result: {status} | Sharpe={metrics['sharpe']} WR={metrics['wr']:.1%} PF={metrics['pf']} Gap={gap} p={perm_p}", flush=True)


# =============================================================================
# RESULTS TABLE
# =============================================================================
print(f"\n[{ts()}] ========== FINAL RESULTS TABLE ==========\n", flush=True)

header = f"{'Strategy':<35} {'N':>4} {'Sharpe':>7} {'Sort':>7} {'WR':>6} {'PF':>6} {'AvgPnL':>7} {'TotPnL':>8} {'Hold':>5} {'Gap':>5} {'p-val':>6} {'Gates':>6} {'Pass':>5}"
print(header, flush=True)
print("-" * len(header), flush=True)

for name, r in results.items():
    pass_str = "YES" if r["validated"] else "NO"
    print(f"{name:<35} {r['n_trades']:>4} {r['sharpe']:>7.3f} {r['sortino']:>7.3f} {r['wr']:>5.1%} {r['pf']:>6.2f} {r['avg_pnl_pct']:>6.3f}% {r['total_pnl_pct']:>7.2f}% {r['avg_hold']:>5.1f} {r['regime_gap']:>5.3f} {r['perm_p']:>6.4f} {r['gates_passed']:>6} {pass_str:>5}", flush=True)

print(f"\n", flush=True)

# Gate detail breakdown
print(f"GATE DETAIL BREAKDOWN:", flush=True)
print(f"{'Strategy':<35} {'G1:Sh>0.5':>10} {'G2:WR>55%':>10} {'G3:PF>1.5':>10} {'G4:Gap<.5':>10} {'G5:p<.05':>10}", flush=True)
print("-" * 85, flush=True)
for name, r in results.items():
    def gstr(v): return "PASS" if v else "FAIL"
    print(f"{name:<35} {gstr(r['g1_sharpe']):>10} {gstr(r['g2_wr']):>10} {gstr(r['g3_pf']):>10} {gstr(r['g4_regime']):>10} {gstr(r['g5_perm']):>10}", flush=True)

print(f"\n", flush=True)

# Regime breakdown
print(f"REGIME BREAKDOWN:", flush=True)
print(f"{'Strategy':<35} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Gap':>8}", flush=True)
print("-" * 67, flush=True)
for name, r in results.items():
    print(f"{name:<35} {r['bull_sharpe']:>12.3f} {r['bear_sharpe']:>12.3f} {r['regime_gap']:>8.3f}", flush=True)

# Summary
n_validated = sum(1 for r in results.values() if r["validated"])
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY: {n_validated}/6 strategies fully validated (all 5 gates)", flush=True)
close_calls = [name for name, r in results.items() if int(r["gates_passed"].split("/")[0]) >= 4 and not r["validated"]]
if close_calls:
    print(f"CLOSE CALLS (4/5): {', '.join(close_calls)}", flush=True)
print(f"{'='*60}", flush=True)

# Save results
import json
results_file = OUTPUT_DIR / "batch7_results.json"
with open(results_file, "w") as f:
    json.dump({k: {kk: str(vv) if isinstance(vv, (bool, np.bool_)) else vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, default=str)
print(f"\n[{ts()}] Results saved to {results_file}", flush=True)
