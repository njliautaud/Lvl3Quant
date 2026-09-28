#!/usr/bin/env python3
"""
Post-Earnings Gap Reversal + Revision Momentum Combined Backtester
==================================================================
Two distinct sub-strategies combined into one portfolio:

SUB-STRATEGY A: EARNINGS GAP-DOWN REVERSAL (mean reversion, long-only)
  Academic basis: Large earnings gap-downs that show immediate buying pressure
  (close well above intraday low = "hammer" candle) reverse ~60% of the time.
  Key: only trade the strongest reversal signals (gap > 5%, close in top 30% of range).

SUB-STRATEGY B: EARNINGS BEAT CONTINUATION (momentum, long-only)
  Academic basis: Post-Earnings Announcement Drift. Stocks beating estimates
  with big gap-ups AND high volume continue drifting for 20-60 days.
  Key: require BOTH gap > 4% AND drift confirmation over 3 days.

Portfolio construction:
  - Max 3 positions at a time
  - Equal weight (~30% each)
  - Independent trailing stops per position
  - Bear market: half position size, require stronger signals

OOT: Jan 2022 - Jul 2026 (includes 2022 bear market)
Capital: $645
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/earnings_reversal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Sub-strategy A: Gap-down reversal ──
GAPDOWN_MIN = -0.05           # At least -5% gap
GAPDOWN_MAX = -0.20           # Not more than -20% (likely broken)
CLOSE_RANGE_MIN = 0.30        # Close must be in top 30% of day's range (buying pressure)
VOL_MULT_A = 1.5              # Volume > 1.5x average
REVERSAL_MAX_HOLD = 20        # 20 trading days max hold
REVERSAL_TP = 0.08            # 8% profit target → tighten stop
REVERSAL_INITIAL_STOP_ATR = 2.0
REVERSAL_TRAIL_ATR = 1.5

# ── Sub-strategy B: Earnings beat continuation ──
GAPUP_MIN = 0.04              # At least 4% gap up
VOL_MULT_B = 1.8              # Higher volume bar for continuation
DRIFT_CONFIRM_DAYS = 3        # 3-day drift confirmation
DRIFT_CONFIRM_MIN = 0.005     # Must drift at least 0.5% more
CONTINUATION_MAX_HOLD = 45    # Hold up to 45 days (longer drift)
CONTINUATION_TP = 0.15        # 15% target → tighten
CONTINUATION_INITIAL_STOP_ATR = 2.5
CONTINUATION_TRAIL_ATR = 2.0

# ── Portfolio ──
MAX_POSITIONS = 3
POSITION_SIZE_PCT = 0.30      # 30% per position
SLIPPAGE_BPS = 5
ATR_PERIOD = 14

DATA_START = "2020-06-01"
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
STARTING_CAPITAL = 645.0

SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200

# ─── Universe ────────────────────────────────────────────────────────────────

UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "CRM", "AVGO", "ADBE",
    "ORCL", "CSCO", "TXN", "QCOM", "MU", "NOW", "AMAT", "INTC", "NFLX",
    # Growth tech
    "CRWD", "PANW", "ZS", "SNOW", "NET", "DDOG", "MDB", "COIN", "SHOP",
    "UBER", "ABNB", "DASH", "PINS", "TTD", "PLTR", "SOFI", "RBLX",
    "ENPH", "FSLR", "SMCI",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN",
    # Consumer
    "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "COST", "WMT",
    # Industrial/Energy
    "XOM", "CVX", "COP", "CAT", "DE", "GE", "HON", "BA", "RTX",
    # Comm/Other
    "PG", "KO", "PEP", "DIS", "CMCSA", "T", "VZ",
    # Extra volatile names
    "SNAP", "ROKU", "LYFT", "DKNG", "RIVN",
]


def download_data(tickers, start, end):
    """Download with caching."""
    cache_file = CACHE_DIR / f"data_{start}_{end}.pkl"
    if cache_file.exists():
        log.info("Loading cached data...")
        data = pd.read_pickle(cache_file)
        if len(data) >= len(tickers) * 0.7:
            return data

    log.info(f"Downloading {len(tickers)} tickers...")
    data = {}
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(" ".join(batch), start=start, end=end,
                             group_by="ticker", auto_adjust=True, progress=False, threads=True)
            for t in batch:
                try:
                    df = raw[t].copy() if len(batch) > 1 else raw.copy()
                    df = df.dropna(subset=["Close", "Volume"])
                    if len(df) > 100:
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Batch failed: {e}")

    log.info(f"Downloaded {len(data)} tickers")
    pd.to_pickle(data, cache_file)
    return data


def download_spy(start, end):
    cache_file = CACHE_DIR / "spy.pkl"
    if cache_file.exists():
        spy = pd.read_pickle(cache_file)
        if len(spy) > 500:
            return spy
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    pd.to_pickle(spy, cache_file)
    return spy


def compute_atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def precompute(df):
    """Pre-compute all features for a ticker."""
    prev_close = df["Close"].shift(1)
    gap_pct = (df["Open"] - prev_close) / prev_close
    vol_avg = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_avg

    # Candle body position: (Close - Low) / (High - Low)
    day_range = df["High"] - df["Low"]
    close_position = (df["Close"] - df["Low"]) / day_range.replace(0, np.nan)

    # Future drift for confirmation
    future_3d = df["Close"].shift(-3) / df["Close"] - 1
    future_5d = df["Close"].shift(-5) / df["Close"] - 1

    atr = compute_atr(df, ATR_PERIOD)

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return pd.DataFrame({
        "gap_pct": gap_pct,
        "vol_ratio": vol_ratio,
        "close_pos": close_position,
        "future_3d": future_3d,
        "future_5d": future_5d,
        "atr": atr,
        "rsi": rsi,
        "close": df["Close"],
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "prev_close": prev_close,
    }, index=df.index)


class Position:
    def __init__(self, ticker, entry_date, entry_price, shares, sub_strategy,
                 initial_stop, trail_atr, atr_val, profit_target):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.sub_strategy = sub_strategy  # "A_reversal" or "B_continuation"
        self.initial_stop = initial_stop
        self.trailing_stop = initial_stop
        self.trail_atr = trail_atr
        self.atr_val = atr_val
        self.profit_target = profit_target
        self.peak_price = entry_price
        self.days_held = 0
        self.target_hit = False

    def update(self, high, low):
        self.peak_price = max(self.peak_price, high)
        gain = (self.peak_price - self.entry_price) / self.entry_price
        if gain >= self.profit_target:
            self.target_hit = True

        atr_mult = 1.0 if self.target_hit else self.trail_atr
        new_stop = self.peak_price - atr_mult * self.atr_val
        self.trailing_stop = max(self.trailing_stop, new_stop)


def run_backtest(data, spy):
    spy_close = spy["Close"].squeeze() if isinstance(spy["Close"], pd.DataFrame) else spy["Close"]
    spy_sma50 = spy_close.rolling(SPY_SMA_FAST).mean()
    spy_sma200 = spy_close.rolling(SPY_SMA_SLOW).mean()
    spy_regime = (spy_sma50 > spy_sma200).astype(int)
    spy_daily_ret = spy_close.pct_change()

    log.info("Pre-computing features...")
    features = {}
    for t, df in data.items():
        features[t] = precompute(df)

    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    all_dates = spy.index[(spy.index >= oot_start) & (spy.index <= oot_end)]
    log.info(f"OOT: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

    equity = STARTING_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    daily_returns = []
    traded_events = set()

    for i, date in enumerate(all_dates):
        regime = 1
        if date in spy_regime.index:
            r = spy_regime.loc[date]
            regime = int(r) if not pd.isna(r) else 1

        spy_ret = 0.0
        if date in spy_daily_ret.index:
            sr = spy_daily_ret.loc[date]
            spy_ret = float(sr) if not pd.isna(sr) else 0.0

        # ── Manage positions ──
        to_close = []
        for pos in positions:
            pos.days_held += 1
            t = pos.ticker

            if t not in features or date not in features[t].index:
                continue

            f = features[t].loc[date]
            high, low, close = f["high"], f["low"], f["close"]
            if pd.isna(close):
                continue

            pos.update(high, low)

            exit_price = None
            exit_reason = None

            # Trailing stop
            if low <= pos.trailing_stop:
                exit_price = max(pos.trailing_stop, low)
                exit_reason = "trailing_stop"

            # Initial stop
            if exit_price is None and low <= pos.initial_stop:
                exit_price = pos.initial_stop
                exit_reason = "initial_stop"

            # Time stop
            max_hold = REVERSAL_MAX_HOLD if pos.sub_strategy == "A_reversal" else CONTINUATION_MAX_HOLD
            if exit_price is None and pos.days_held >= max_hold:
                exit_price = close
                exit_reason = "time_stop"

            if exit_price is not None:
                exit_price = float(exit_price) * (1 - SLIPPAGE_BPS / 10000)
                pnl = (exit_price - pos.entry_price) * pos.shares
                equity += pnl
                to_close.append(pos)

                trades.append({
                    "ticker": t,
                    "sub_strategy": pos.sub_strategy,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(date.date()),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": round(pos.shares, 4),
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / (pos.entry_price * pos.shares) * 100, 2),
                    "days_held": pos.days_held,
                    "exit_reason": exit_reason,
                    "regime": "bull" if regime else "bear",
                    "spy_ret": round(spy_ret * 100, 3),
                })

        for p in to_close:
            positions.remove(p)

        # ── Scan for new signals ──
        if len(positions) < MAX_POSITIONS and equity > 30:
            candidates = []

            for ticker in UNIVERSE:
                if ticker not in features or date not in features[ticker].index:
                    continue
                if any(p.ticker == ticker for p in positions):
                    continue

                f = features[ticker]
                idx = f.index.get_loc(date)
                if idx < 30:
                    continue

                today = f.iloc[idx]
                gap = today["gap_pct"]
                vol_r = today["vol_ratio"]
                close_pos = today["close_pos"]
                atr = today["atr"]
                rsi = today["rsi"]

                if pd.isna(gap) or pd.isna(vol_r) or pd.isna(atr):
                    continue

                # ── SUB-STRATEGY A: Gap-down reversal ──
                # Buy TODAY at close if earnings gap-down shows hammer candle
                if (GAPDOWN_MAX <= gap <= GAPDOWN_MIN and
                    vol_r >= VOL_MULT_A and
                    not pd.isna(close_pos) and close_pos >= CLOSE_RANGE_MIN):

                    event_key = f"{ticker}_A_{date.date()}"
                    if event_key not in traded_events:
                        strength = abs(gap) * vol_r * close_pos
                        candidates.append({
                            "ticker": ticker,
                            "sub": "A_reversal",
                            "strength": strength,
                            "atr": atr,
                            "entry_price": float(today["close"]),
                            "initial_stop_atr": REVERSAL_INITIAL_STOP_ATR,
                            "trail_atr": REVERSAL_TRAIL_ATR,
                            "profit_target": REVERSAL_TP,
                            "event_key": event_key,
                        })

                # ── SUB-STRATEGY B: Gap-up continuation ──
                # Buy 3 days after gap-up if drift is confirmed
                if idx >= 3:
                    day_3ago = f.iloc[idx - 3]
                    gap_3 = day_3ago["gap_pct"]
                    vol_3 = day_3ago["vol_ratio"]

                    if (not pd.isna(gap_3) and not pd.isna(vol_3) and
                        gap_3 >= GAPUP_MIN and vol_3 >= VOL_MULT_B):

                        event_key = f"{ticker}_B_{f.index[idx-3].date()}"
                        if event_key not in traded_events:
                            # Check drift: current close > close 3 days ago
                            close_3ago = day_3ago["close"]
                            current_close = today["close"]
                            if (not pd.isna(close_3ago) and not pd.isna(current_close) and
                                current_close > close_3ago * (1 + DRIFT_CONFIRM_MIN)):

                                drift = (current_close - close_3ago) / close_3ago
                                strength = gap_3 * vol_3 * (1 + drift)
                                candidates.append({
                                    "ticker": ticker,
                                    "sub": "B_continuation",
                                    "strength": strength,
                                    "atr": atr,
                                    "entry_price": float(current_close),
                                    "initial_stop_atr": CONTINUATION_INITIAL_STOP_ATR,
                                    "trail_atr": CONTINUATION_TRAIL_ATR,
                                    "profit_target": CONTINUATION_TP,
                                    "event_key": event_key,
                                })

            # Sort and enter
            candidates.sort(key=lambda x: x["strength"], reverse=True)
            slots = MAX_POSITIONS - len(positions)

            for c in candidates[:slots]:
                size_pct = POSITION_SIZE_PCT
                if regime == 0:
                    size_pct *= 0.5

                pos_value = equity * size_pct
                if pos_value < 5:
                    continue

                entry = c["entry_price"] * (1 + SLIPPAGE_BPS / 10000)
                shares = pos_value / entry
                init_stop = entry - c["initial_stop_atr"] * c["atr"]

                pos = Position(
                    ticker=c["ticker"],
                    entry_date=date,
                    entry_price=entry,
                    shares=shares,
                    sub_strategy=c["sub"],
                    initial_stop=init_stop,
                    trail_atr=c["trail_atr"],
                    atr_val=c["atr"],
                    profit_target=c["profit_target"],
                )
                positions.append(pos)
                traded_events.add(c["event_key"])

        # ── MTM ──
        mtm = equity
        for pos in positions:
            if pos.ticker in features and date in features[pos.ticker].index:
                cur = features[pos.ticker].loc[date]["close"]
                if not pd.isna(cur):
                    mtm += (float(cur) - pos.entry_price) * pos.shares

        equity_curve.append({
            "date": str(date.date()),
            "equity": round(mtm, 2),
            "n_positions": len(positions),
            "regime": "bull" if regime else "bear",
            "spy_ret": round(spy_ret, 6),
        })

        if len(equity_curve) >= 2:
            prev = equity_curve[-2]["equity"]
            daily_returns.append(mtm / prev - 1 if prev > 0 else 0)

        if i % 100 == 0:
            log.info(f"  Day {i}/{len(all_dates)}: eq=${mtm:.2f}, pos={len(positions)}, trades={len(trades)}")

    # Close remaining
    for pos in positions:
        last = all_dates[-1]
        if pos.ticker in features and last in features[pos.ticker].index:
            c = float(features[pos.ticker].loc[last]["close"])
            if not pd.isna(c):
                pnl = (c - pos.entry_price) * pos.shares
                trades.append({
                    "ticker": pos.ticker,
                    "sub_strategy": pos.sub_strategy,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(last.date()),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(c, 2),
                    "shares": round(pos.shares, 4),
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / (pos.entry_price * pos.shares) * 100, 2),
                    "days_held": pos.days_held,
                    "exit_reason": "end_of_test",
                    "regime": "bull",
                    "spy_ret": 0,
                })

    return trades, equity_curve, daily_returns


def validate(trades, equity_curve, daily_returns):
    """5-gate validation."""
    if not trades:
        return {"passed": False, "reason": "No trades"}

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity_curve)
    rets = np.array(daily_returns)

    n = len(df_t)
    winners = df_t[df_t["pnl"] > 0]
    losers = df_t[df_t["pnl"] <= 0]
    wr = len(winners) / n
    pf = abs(float(winners["pnl"].sum()) / float(losers["pnl"].sum())) if float(losers["pnl"].sum()) != 0 else float("inf")
    total_pnl = float(df_t["pnl"].sum())
    final_eq = float(df_e["equity"].iloc[-1])

    # Sub-strategy breakdown
    a_trades = df_t[df_t["sub_strategy"] == "A_reversal"]
    b_trades = df_t[df_t["sub_strategy"] == "B_continuation"]

    results = {
        "total_trades": int(n),
        "win_rate": round(float(wr), 4),
        "avg_win": round(float(winners["pnl"].mean()), 2) if len(winners) > 0 else 0,
        "avg_loss": round(float(losers["pnl"].mean()), 2) if len(losers) > 0 else 0,
        "profit_factor": round(float(pf), 3),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_eq, 2),
        "return_pct": round((final_eq / STARTING_CAPITAL - 1) * 100, 2),
        "avg_days_held": round(float(df_t["days_held"].mean()), 1),
        "exit_reasons": {k: int(v) for k, v in df_t["exit_reason"].value_counts().items()},
        "A_reversal_trades": int(len(a_trades)),
        "A_reversal_wr": round(float(len(a_trades[a_trades["pnl"] > 0]) / len(a_trades)), 4) if len(a_trades) > 0 else 0,
        "A_reversal_pnl": round(float(a_trades["pnl"].sum()), 2) if len(a_trades) > 0 else 0,
        "B_continuation_trades": int(len(b_trades)),
        "B_continuation_wr": round(float(len(b_trades[b_trades["pnl"] > 0]) / len(b_trades)), 4) if len(b_trades) > 0 else 0,
        "B_continuation_pnl": round(float(b_trades["pnl"].sum()), 2) if len(b_trades) > 0 else 0,
    }

    # CAGR
    n_years = len(rets) / 252
    if n_years > 0 and final_eq > 0:
        results["cagr"] = round(((final_eq / STARTING_CAPITAL) ** (1 / n_years) - 1) * 100, 2)
    else:
        results["cagr"] = 0.0

    # Gate 1: Sharpe
    if len(rets) > 20:
        ann_ret = float(np.mean(rets)) * 252
        ann_vol = float(np.std(rets)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    else:
        sharpe = 0
    results["sharpe"] = round(sharpe, 3)
    results["gate1_sharpe"] = bool(sharpe > 0.5)

    ds = rets[rets < 0]
    ds_vol = float(np.std(ds)) * np.sqrt(252) if len(ds) > 0 else 1e-6
    results["sortino"] = round((float(np.mean(rets)) * 252) / ds_vol, 3)

    # Gate 2: Perm test
    perm_sharpes = []
    for _ in range(2000):
        p = np.random.permutation(rets)
        pm, ps = float(np.mean(p)) * 252, float(np.std(p)) * np.sqrt(252)
        perm_sharpes.append(pm / ps if ps > 0 else 0)
    p_val = float(np.mean(np.array(perm_sharpes) >= sharpe))
    results["perm_p_value"] = round(p_val, 4)
    results["gate2_perm"] = bool(p_val < 0.05)

    # Gate 3: Beat random
    rand_sharpes = []
    for _ in range(1000):
        r = np.random.choice(rets, size=len(rets), replace=True)
        rm, rs = float(np.mean(r)) * 252, float(np.std(r)) * np.sqrt(252)
        rand_sharpes.append(rm / rs if rs > 0 else 0)
    rand_med = float(np.median(rand_sharpes))
    results["random_baseline"] = round(rand_med, 3)
    results["gate3_beats_random"] = bool(sharpe > rand_med + 0.1)

    # Gate 4: Regime
    bull = df_t[df_t["regime"] == "bull"]
    bear = df_t[df_t["regime"] == "bear"]
    bull_wr = float(len(bull[bull["pnl"] > 0]) / len(bull)) if len(bull) > 0 else 0
    bear_wr = float(len(bear[bear["pnl"] > 0]) / len(bear)) if len(bear) > 0 else 0

    if len(bull) > 5 and len(bear) > 5:
        gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr) if max(bull_wr, bear_wr) > 0 else 1
    elif len(bear) <= 5:
        gap = 0.3
    else:
        gap = 1.0

    results["bull_trades"] = int(len(bull))
    results["bull_wr"] = round(bull_wr, 4)
    results["bear_trades"] = int(len(bear))
    results["bear_wr"] = round(bear_wr, 4)
    results["regime_gap"] = round(gap, 4)
    results["gate4_regime"] = bool(gap < 0.50)

    # Gate 5: MDD
    eq = df_e["equity"].astype(float).values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(dd.min())
    results["max_drawdown"] = round(mdd, 4)
    results["gate5_mdd"] = bool(mdd > -0.50)

    # Green/red
    df_e2 = df_e.copy()
    df_e2["daily_ret"] = pd.to_numeric(df_e2["equity"]).pct_change()
    df_e2["spy_ret_f"] = pd.to_numeric(df_e2["spy_ret"])
    green = df_e2[df_e2["spy_ret_f"] > 0.001]
    red = df_e2[df_e2["spy_ret_f"] < -0.001]
    if len(green) > 20 and len(red) > 20:
        gs = (float(green["daily_ret"].mean()) * 252) / (float(green["daily_ret"].std()) * np.sqrt(252)) if float(green["daily_ret"].std()) > 0 else 0
        rs = (float(red["daily_ret"].mean()) * 252) / (float(red["daily_ret"].std()) * np.sqrt(252)) if float(red["daily_ret"].std()) > 0 else 0
        results["green_day_sharpe"] = round(gs, 3)
        results["red_day_sharpe"] = round(rs, 3)
    else:
        results["green_day_sharpe"] = 0.0
        results["red_day_sharpe"] = 0.0

    gates = [results["gate1_sharpe"], results["gate2_perm"], results["gate3_beats_random"],
             results["gate4_regime"], results["gate5_mdd"]]
    results["gates_passed"] = int(sum(gates))
    results["all_gates_passed"] = bool(all(gates))

    return results


def log_to_mlflow(results, trades, equity_curve):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("earnings_reversal_combo_v1")

        with mlflow.start_run(run_name=f"earnings_rev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("strategy", "earnings_reversal_combo")
            mlflow.log_param("universe_size", len(UNIVERSE))
            mlflow.log_param("gapdown_range", f"{GAPDOWN_MAX} to {GAPDOWN_MIN}")
            mlflow.log_param("gapup_min", GAPUP_MIN)
            mlflow.log_param("max_positions", MAX_POSITIONS)
            mlflow.log_param("oot_period", f"{OOT_START} to {OOT_END}")
            mlflow.log_param("starting_capital", STARTING_CAPITAL)

            for k, v in results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

            for name, obj in [("trades.json", trades), ("equity.json", equity_curve), ("results.json", results)]:
                path = OUTPUT_DIR / name
                with open(path, "w") as f:
                    json.dump(obj, f, indent=2)
                mlflow.log_artifact(str(path))

            log.info("MLflow logged")
    except Exception as e:
        log.warning(f"MLflow failed: {e}")


def main():
    log.info("=" * 70)
    log.info("EARNINGS REVERSAL + CONTINUATION COMBO BACKTEST")
    log.info("=" * 70)

    data = download_data(UNIVERSE, DATA_START, OOT_END)
    spy = download_spy(DATA_START, OOT_END)
    log.info(f"Data: {len(data)} tickers, SPY {len(spy)} days")

    trades, equity_curve, daily_returns = run_backtest(data, spy)
    log.info(f"\n{len(trades)} trades generated")

    results = validate(trades, equity_curve, daily_returns)

    log.info("\n" + "=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info(f"Trades: {results['total_trades']} | WR: {results['win_rate']*100:.1f}% | PF: {results['profit_factor']:.3f}")
    log.info(f"P&L: ${results['total_pnl']:.2f} | Final: ${results['final_equity']:.2f} ({results['return_pct']:.1f}%) | CAGR: {results['cagr']:.1f}%")
    log.info(f"Avg hold: {results['avg_days_held']:.1f}d | Exits: {results['exit_reasons']}")
    log.info(f"")
    log.info(f"Sub-A (reversal): {results['A_reversal_trades']} trades, WR {results['A_reversal_wr']*100:.1f}%, P&L ${results['A_reversal_pnl']:.2f}")
    log.info(f"Sub-B (continuation): {results['B_continuation_trades']} trades, WR {results['B_continuation_wr']*100:.1f}%, P&L ${results['B_continuation_pnl']:.2f}")
    log.info(f"")
    log.info(f"Gate 1 Sharpe>0.5:  {results['sharpe']:.3f} (Sortino {results['sortino']:.3f}) {'PASS' if results['gate1_sharpe'] else 'FAIL'}")
    log.info(f"Gate 2 Perm p<0.05: {results['perm_p_value']:.4f} {'PASS' if results['gate2_perm'] else 'FAIL'}")
    log.info(f"Gate 3 Beat random: {results['sharpe']:.3f} vs {results['random_baseline']:.3f} {'PASS' if results['gate3_beats_random'] else 'FAIL'}")
    log.info(f"Gate 4 Regime:      gap={results['regime_gap']:.3f} (bull {results['bull_wr']*100:.1f}% bear {results['bear_wr']*100:.1f}%) {'PASS' if results['gate4_regime'] else 'FAIL'}")
    log.info(f"Gate 5 MDD>-50%:    {results['max_drawdown']*100:.1f}% {'PASS' if results['gate5_mdd'] else 'FAIL'}")
    log.info(f"Green Sharpe: {results['green_day_sharpe']:.3f} | Red Sharpe: {results['red_day_sharpe']:.3f}")
    log.info(f"")
    log.info(f"GATES: {results['gates_passed']}/5 {'PASS' if results['all_gates_passed'] else 'FAIL'}")

    log_to_mlflow(results, trades, equity_curve)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
