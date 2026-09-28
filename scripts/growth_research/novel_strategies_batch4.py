"""
Novel Strategies Batch 4 — Money Flow, Rotation & Informed Flow Patterns
=========================================================================
HC #768: Build strategies around smart money detection, cross-asset timing,
mid-cap inefficiency, institutional accumulation proxy, and vol term structure.

STRATEGIES:
  A. Smart Money Divergence (sector ETF flow vs price divergence)
  B. Gamma Squeeze Detector (ATR contraction + volume spike → breakout)
  C. Cross-Asset Momentum Timing (gold/dollar/oil/yields → equity timing)
  D. Mean Reversion on Mid-Caps (same RSI<30 signals, less efficient universe)
  E. Institutional Accumulation Proxy (quarter-end unusual volume, flat price)
  F. Volatility Term Structure Trade (VIX inversion → buy quality, sell on normalization)

Universe varies by strategy. Period: 2020-01-01 to 2026-07-01.
Position: $300, max 2 concurrent, 21-day max hold (30 for E).

VALIDATION GATES (5-gate):
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

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/novel_strategies_batch4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_SIZE = 300.0
MAX_CONCURRENT = 2
BUFFER_START = "2019-01-01"  # extra lookback for long indicators
START = "2020-01-01"
END = "2026-07-01"
N_PERMUTATIONS = 200
np.random.seed(42)

# Universes
QUALITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH",
    "LLY", "AVGO", "AMD", "HD", "ABBV", "MRK", "COST", "CRM",
    "NFLX", "ADBE", "PG", "JNJ"
]

MIDCAP_UNIVERSE = [
    "PANW", "SNPS", "CDNS", "FTNT", "MCHP", "KLAC", "LRCX", "ANET",
    "DXCM", "IDXX", "ODFL", "ROP", "TDG", "WST", "POOL"
]

SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]

# Map sector ETFs to top individual stocks in each sector
SECTOR_TOP_STOCKS = {
    "XLK": ["AAPL", "MSFT", "NVDA"],
    "XLV": ["UNH", "LLY", "ABBV"],
    "XLF": ["JPM", "BRK-B", "V"],
    "XLE": ["XOM", "CVX", "COP"],
    "XLI": ["CAT", "GE", "UNP"],
    "XLC": ["META", "GOOGL", "NFLX"],
    "XLY": ["AMZN", "TSLA", "HD"],
    "XLP": ["PG", "COST", "KO"],
    "XLU": ["NEE", "SO", "DUK"],
    "XLRE": ["PLD", "AMT", "EQIX"],
    "XLB": ["LIN", "FCX", "NEM"],
}

CROSS_ASSET_TICKERS = ["GLD", "UUP", "USO", "^TNX"]
ENERGY_MATERIALS = ["XOM", "CVX", "FCX", "NEM"]

ts = lambda: datetime.now().strftime('%H:%M:%S')

# =============================================================================
# 1. DATA DOWNLOAD
# =============================================================================
print(f"[{ts()}] === NOVEL STRATEGIES BATCH 4 ===", flush=True)
print(f"[{ts()}] Downloading data...", flush=True)

# Collect all unique tickers
ALL_TICKERS = list(set(
    QUALITY_UNIVERSE + MIDCAP_UNIVERSE + SECTOR_ETFS +
    [s for stocks in SECTOR_TOP_STOCKS.values() for s in stocks] +
    CROSS_ASSET_TICKERS + ENERGY_MATERIALS +
    ["SPY", "^VIX"]
))

raw_close = {}
raw_high = {}
raw_low = {}
raw_volume = {}
raw_open = {}

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
        raw_open[t] = df["Open"].rename(t)
    except Exception as e:
        print(f"  ERROR {t}: {e}", flush=True)

print(f"[{ts()}] Downloaded {len(raw_close)} tickers", flush=True)

close_df = pd.DataFrame(raw_close).sort_index().ffill()
high_df = pd.DataFrame(raw_high).sort_index().ffill()
low_df = pd.DataFrame(raw_low).sort_index().ffill()
volume_df = pd.DataFrame(raw_volume).sort_index().ffill().fillna(0)
open_df = pd.DataFrame(raw_open).sort_index().ffill()

close_df.index = pd.to_datetime(close_df.index)
high_df.index = pd.to_datetime(high_df.index)
low_df.index = pd.to_datetime(low_df.index)
volume_df.index = pd.to_datetime(volume_df.index)
open_df.index = pd.to_datetime(open_df.index)

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
        valid_tickers = [t for t in tickers_used if t in close_df.columns and t != "BASKET"]
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


def count_active(active_positions, date):
    """Count currently active positions."""
    return sum(1 for v in active_positions.values() if v is not None)


# =============================================================================
# STRATEGY A: SMART MONEY DIVERGENCE
# =============================================================================
def strategy_a_smart_money_divergence():
    print(f"\n[{ts()}] === STRATEGY A: Smart Money Divergence ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 60:
            continue

        # Check exits first
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            current_price = close_df[pos["ticker"]].iloc[date_idx]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

            # Exit: 21 day max hold, or +8% profit, or -5% stop
            if hold_days >= 21 or pnl_pct >= 0.08 or pnl_pct <= -0.05:
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": pos["signal"]
                })
                active_positions[key] = None

        # Count active
        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Scan sector ETFs for smart money divergence
        best_signal = None
        best_score = 0

        for etf in SECTOR_ETFS:
            if etf not in close_df.columns or etf not in volume_df.columns:
                continue

            # Volume surge: today's volume vs 20-day average
            vol_20avg = volume_df[etf].iloc[date_idx-20:date_idx].mean()
            if vol_20avg <= 0:
                continue
            vol_ratio = volume_df[etf].iloc[date_idx] / vol_20avg

            # Price change of the ETF today
            etf_ret = (close_df[etf].iloc[date_idx] - close_df[etf].iloc[date_idx-1]) / close_df[etf].iloc[date_idx-1]

            # SPY return today
            spy_ret = (spy_close.iloc[date_idx] - spy_close.iloc[date_idx-1]) / spy_close.iloc[date_idx-1]

            # Smart money signal 1: ETF volume surges 2x+ on UP day, diverging from SPY
            if vol_ratio >= 2.0 and etf_ret > 0.003 and etf_ret > spy_ret + 0.003:
                score = vol_ratio * etf_ret * 100
                if score > best_score:
                    best_score = score
                    best_signal = {"etf": etf, "type": "breakout_ahead", "score": score}

            # Smart money signal 2: ETF volume surges 2x+ on DOWN day while SPY flat/up
            # → smart money buying the dip in this sector
            if vol_ratio >= 2.0 and etf_ret < -0.003 and spy_ret > etf_ret + 0.003:
                score = vol_ratio * abs(etf_ret) * 100
                if score > best_score:
                    best_score = score
                    best_signal = {"etf": etf, "type": "dip_accumulation", "score": score}

        if best_signal is not None:
            # Buy the top stock in the sector
            etf = best_signal["etf"]
            if etf in SECTOR_TOP_STOCKS:
                candidates = SECTOR_TOP_STOCKS[etf]
                # Pick the one with best recent momentum (5-day return)
                best_ticker = None
                best_mom = -999
                for t in candidates:
                    if t not in close_df.columns:
                        continue
                    mom = (close_df[t].iloc[date_idx] - close_df[t].iloc[date_idx-5]) / close_df[t].iloc[date_idx-5]
                    if best_signal["type"] == "dip_accumulation":
                        # For dip accumulation, pick the most beaten down
                        mom = -mom
                    if mom > best_mom:
                        best_mom = mom
                        best_ticker = t

                if best_ticker and best_ticker not in active_positions:
                    entry_price = close_df[best_ticker].iloc[date_idx]
                    active_positions[best_ticker] = {
                        "ticker": best_ticker,
                        "entry_date": date,
                        "entry_price": entry_price,
                        "signal": f"{best_signal['type']}_{etf}"
                    }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY B: GAMMA SQUEEZE DETECTOR
# =============================================================================
def strategy_b_gamma_squeeze():
    print(f"\n[{ts()}] === STRATEGY B: Gamma Squeeze Detector ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index
    universe = QUALITY_UNIVERSE + MIDCAP_UNIVERSE

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 30:
            continue

        # Check exits first
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            current_price = close_df[pos["ticker"]].iloc[date_idx]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

            # Exit: 21 day max hold, or +10% profit (squeeze target), or -5% stop
            if hold_days >= 21 or pnl_pct >= 0.10 or pnl_pct <= -0.05:
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": "gamma_squeeze"
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Scan for gamma squeeze setups
        best_ticker = None
        best_score = 0

        for ticker in universe:
            if ticker not in close_df.columns or ticker in active_positions:
                continue

            # ATR contraction: current 5-day ATR vs 20-day ATR
            highs = high_df[ticker].iloc[date_idx-20:date_idx+1]
            lows = low_df[ticker].iloc[date_idx-20:date_idx+1]
            closes = close_df[ticker].iloc[date_idx-20:date_idx+1]

            if len(highs) < 20:
                continue

            tr = pd.concat([
                highs - lows,
                (highs - closes.shift(1)).abs(),
                (lows - closes.shift(1)).abs()
            ], axis=1).max(axis=1)

            atr_5 = tr.iloc[-5:].mean()
            atr_20 = tr.mean()

            if atr_20 <= 0:
                continue

            atr_contraction = atr_5 / atr_20  # < 1 means contracting

            # Volume spike: today's volume vs 20-day average
            vol_20avg = volume_df[ticker].iloc[date_idx-20:date_idx].mean()
            if vol_20avg <= 0:
                continue
            vol_ratio = volume_df[ticker].iloc[date_idx] / vol_20avg

            # Gamma squeeze signal: ATR contracts by >25% AND volume 1.5x+
            if atr_contraction <= 0.75 and vol_ratio >= 1.5:
                # Price should be near a local high (breakout direction)
                price = close_df[ticker].iloc[date_idx]
                high_20 = close_df[ticker].iloc[date_idx-20:date_idx+1].max()
                near_high = price >= high_20 * 0.95  # within 5% of 20-day high

                if near_high:
                    score = vol_ratio * (1 - atr_contraction)  # higher is better
                    if score > best_score:
                        best_score = score
                        best_ticker = ticker

        if best_ticker:
            entry_price = close_df[best_ticker].iloc[date_idx]
            active_positions[best_ticker] = {
                "ticker": best_ticker,
                "entry_date": date,
                "entry_price": entry_price,
                "signal": "gamma_squeeze"
            }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY C: CROSS-ASSET MOMENTUM TIMING
# =============================================================================
def strategy_c_cross_asset_momentum():
    print(f"\n[{ts()}] === STRATEGY C: Cross-Asset Momentum Timing ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index
    last_entry_date = None

    # Quality stocks for risk-on buys
    quality_for_riskon = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "UNH", "LLY"]
    # Energy + materials for inflation hedge
    inflation_hedge = [t for t in ENERGY_MATERIALS if t in close_df.columns]

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 60:
            continue

        # Check exits
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            current_price = close_df[pos["ticker"]].iloc[date_idx]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

            if hold_days >= 21 or pnl_pct >= 0.08 or pnl_pct <= -0.05:
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": pos["signal"]
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Cooldown: don't enter within 5 days of last entry
        if last_entry_date is not None and (date - last_entry_date).days < 5:
            continue

        # Cross-asset signals
        signal_type = None
        buy_universe = []

        # Signal 1: Gold up >2% in 10 days + Dollar down → risk-on
        if "GLD" in close_df.columns and "UUP" in close_df.columns:
            gld_10d = (close_df["GLD"].iloc[date_idx] - close_df["GLD"].iloc[date_idx-10]) / close_df["GLD"].iloc[date_idx-10]
            uup_10d = (close_df["UUP"].iloc[date_idx] - close_df["UUP"].iloc[date_idx-10]) / close_df["UUP"].iloc[date_idx-10]

            if gld_10d > 0.02 and uup_10d < -0.005:
                signal_type = "risk_on_gold_dollar"
                buy_universe = quality_for_riskon

        # Signal 2: Oil up >3% in 10 days + yields up → inflation hedge
        if signal_type is None and "USO" in close_df.columns and "^TNX" in close_df.columns:
            uso_10d = (close_df["USO"].iloc[date_idx] - close_df["USO"].iloc[date_idx-10]) / close_df["USO"].iloc[date_idx-10]
            tnx_10d = (close_df["^TNX"].iloc[date_idx] - close_df["^TNX"].iloc[date_idx-10]) / close_df["^TNX"].iloc[date_idx-10]

            if uso_10d > 0.03 and tnx_10d > 0:
                signal_type = "inflation_hedge"
                buy_universe = inflation_hedge

        # Signal 3: Both gold AND bonds up sharply → flight to safety, AVOID equities
        # (No trade signal — this is a "stay out" filter)

        if signal_type and buy_universe:
            # Pick the stock with best 5-day momentum from the universe
            best_ticker = None
            best_mom = -999
            for t in buy_universe:
                if t not in close_df.columns or t in active_positions:
                    continue
                mom = (close_df[t].iloc[date_idx] - close_df[t].iloc[date_idx-5]) / close_df[t].iloc[date_idx-5]
                if mom > best_mom:
                    best_mom = mom
                    best_ticker = t

            if best_ticker:
                entry_price = close_df[best_ticker].iloc[date_idx]
                active_positions[best_ticker] = {
                    "ticker": best_ticker,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "signal": signal_type
                }
                last_entry_date = date

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY D: MEAN REVERSION ON MID-CAPS
# =============================================================================
def strategy_d_midcap_mean_reversion():
    print(f"\n[{ts()}] === STRATEGY D: Mid-Cap Mean Reversion ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index

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
            current_price = close_df[pos["ticker"]].iloc[date_idx]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

            # Exit: 21 day hold, or RSI > 60 (mean reverted), or -7% stop
            rsi_val = _compute_rsi(close_df[pos["ticker"]], date_idx, 14)
            if hold_days >= 21 or rsi_val > 60 or pnl_pct >= 0.10 or pnl_pct <= -0.07:
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": "midcap_rsi_dip"
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Scan mid-caps for RSI < 30 + >7% below 20-day high
        best_ticker = None
        best_rsi = 100

        for ticker in MIDCAP_UNIVERSE:
            if ticker not in close_df.columns or ticker in active_positions:
                continue

            rsi = _compute_rsi(close_df[ticker], date_idx, 14)
            high_20 = close_df[ticker].iloc[date_idx-20:date_idx+1].max()
            current_price = close_df[ticker].iloc[date_idx]
            drawdown = (current_price - high_20) / high_20

            if rsi < 35 and drawdown < -0.05:
                if rsi < best_rsi:
                    best_rsi = rsi
                    best_ticker = ticker

        if best_ticker:
            entry_price = close_df[best_ticker].iloc[date_idx]
            active_positions[best_ticker] = {
                "ticker": best_ticker,
                "entry_date": date,
                "entry_price": entry_price,
                "signal": "midcap_rsi_dip"
            }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


def _compute_rsi(series, idx, period=14):
    """Compute RSI at a specific index."""
    if idx < period + 1:
        return 50
    changes = series.iloc[idx-period:idx+1].diff().dropna()
    gains = changes.clip(lower=0).mean()
    losses = (-changes.clip(upper=0)).mean()
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))


# =============================================================================
# STRATEGY E: INSTITUTIONAL ACCUMULATION PROXY
# =============================================================================
def strategy_e_institutional_accumulation():
    print(f"\n[{ts()}] === STRATEGY E: Institutional Accumulation Proxy ===", flush=True)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index

    # Quarter-end windows: last 5 trading days of Mar, Jun, Sep, Dec
    def is_quarter_end_window(dt):
        """Check if date is in the last 10 trading days of a quarter-end month."""
        month = dt.month
        if month not in [3, 6, 9, 12]:
            return False
        # Wider window: day >= 18 (covers last ~10 trading days)
        return dt.day >= 18

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 30:
            continue

        # Check exits (30-day hold for 13F strategy)
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            current_price = close_df[pos["ticker"]].iloc[date_idx]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

            # Exit: 30 day hold (until 13F disclosure), or +10% profit, or -6% stop
            if hold_days >= 30 or pnl_pct >= 0.10 or pnl_pct <= -0.06:
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": "institutional_accumulation"
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Only look during quarter-end windows
        if not is_quarter_end_window(date):
            continue

        # Scan quality stocks for unusual volume + flat/down price
        best_ticker = None
        best_vol_ratio = 0

        for ticker in QUALITY_UNIVERSE:
            if ticker not in close_df.columns or ticker in active_positions:
                continue

            # Volume: today vs 20-day avg
            vol_20avg = volume_df[ticker].iloc[date_idx-20:date_idx].mean()
            if vol_20avg <= 0:
                continue
            vol_ratio = volume_df[ticker].iloc[date_idx] / vol_20avg

            # Price change: should be flat or slightly down (accumulation, not news-driven)
            price_ret = (close_df[ticker].iloc[date_idx] - close_df[ticker].iloc[date_idx-1]) / close_df[ticker].iloc[date_idx-1]

            # Signal: 1.5x+ volume, price flat or slightly down (-2% to +0.5%)
            if vol_ratio >= 1.5 and -0.02 <= price_ret <= 0.005:
                if vol_ratio > best_vol_ratio:
                    best_vol_ratio = vol_ratio
                    best_ticker = ticker

        if best_ticker:
            entry_price = close_df[best_ticker].iloc[date_idx]
            active_positions[best_ticker] = {
                "ticker": best_ticker,
                "entry_date": date,
                "entry_price": entry_price,
                "signal": "institutional_accumulation"
            }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# STRATEGY F: VOLATILITY TERM STRUCTURE TRADE
# =============================================================================
def strategy_f_vol_term_structure():
    print(f"\n[{ts()}] === STRATEGY F: Volatility Term Structure Trade ===", flush=True)

    if vix is None:
        print("  ERROR: VIX data not available", flush=True)
        return None, compute_metrics(None)

    trades = []
    active_positions = {}
    effective_dates = close_df.loc[START:END].index

    for date in effective_dates:
        date_idx = close_df.index.get_loc(date)
        if date_idx < 60:
            continue

        vix_val = vix.iloc[date_idx]
        vix_60avg = vix.iloc[date_idx-60:date_idx].mean()

        # Check exits
        for key in list(active_positions.keys()):
            pos = active_positions[key]
            if pos is None:
                continue
            hold_days = (date - pos["entry_date"]).days
            current_price = close_df[pos["ticker"]].iloc[date_idx]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

            # Exit when VIX drops below 60-day average (normalization), or max hold, or stops
            vix_normalized = vix_val < vix_60avg
            if hold_days >= 21 or vix_normalized or pnl_pct >= 0.12 or pnl_pct <= -0.06:
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_days,
                    "signal": "vol_term_structure"
                })
                active_positions[key] = None

        n_active = sum(1 for v in active_positions.values() if v is not None)
        if n_active >= MAX_CONCURRENT:
            continue

        # Entry: VIX > 1.15x its 60-day average (term structure inversion proxy)
        # AND VIX has started declining from peak (momentum confirmation - don't catch falling knife)
        vix_3d_ago = vix.iloc[date_idx-3] if date_idx >= 3 else vix_val
        vix_declining = vix_val < vix_3d_ago  # VIX peaked and starting to drop

        if vix_val > 1.15 * vix_60avg and vix_declining:
            # Buy the quality stock with the biggest recent drawdown (most upside on normalization)
            best_ticker = None
            best_dd = 0

            for ticker in QUALITY_UNIVERSE:
                if ticker not in close_df.columns or ticker in active_positions:
                    continue

                high_20 = close_df[ticker].iloc[date_idx-20:date_idx+1].max()
                current_price = close_df[ticker].iloc[date_idx]
                dd = (current_price - high_20) / high_20

                if dd < best_dd:
                    best_dd = dd
                    best_ticker = ticker

            if best_ticker and best_dd < -0.02:  # at least 2% below recent high
                entry_price = close_df[best_ticker].iloc[date_idx]
                active_positions[best_ticker] = {
                    "ticker": best_ticker,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "signal": "vol_term_structure"
                }

    trades_df = pd.DataFrame(trades) if trades else None
    metrics = compute_metrics(trades_df)
    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']}", flush=True)
    return trades_df, metrics


# =============================================================================
# RUN ALL STRATEGIES + VALIDATION
# =============================================================================
print(f"\n[{ts()}] ========== RUNNING ALL 6 STRATEGIES ==========", flush=True)

strategies = {
    "A: Smart Money Divergence": strategy_a_smart_money_divergence,
    "B: Gamma Squeeze Detector": strategy_b_gamma_squeeze,
    "C: Cross-Asset Momentum": strategy_c_cross_asset_momentum,
    "D: Mid-Cap Mean Reversion": strategy_d_midcap_mean_reversion,
    "E: Institutional Accumulation": strategy_e_institutional_accumulation,
    "F: Vol Term Structure": strategy_f_vol_term_structure,
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
results_file = OUTPUT_DIR / "batch4_results.json"
with open(results_file, "w") as f:
    json.dump({k: {kk: str(vv) if isinstance(vv, (bool, np.bool_)) else vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, default=str)
print(f"\n[{ts()}] Results saved to {results_file}", flush=True)
print(f"[{ts()}] === BATCH 4 COMPLETE ===", flush=True)
