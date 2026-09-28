#!/usr/bin/env python3
"""
Sentiment Contrarian Paper Engine (AVO-evolved v22)
====================================================
ALL-TIME RECORD AVO score 8.25. Lockbox validated: Sharpe 2.73 at MAX_CONCURRENT=3.

Multi-signal mean reversion: Buy stocks beaten down relative to SPY when macro
conditions aren't deteriorating. Two signal tiers (deep oversold + moderate oversold).
Sentiment boost when Reddit polarity is negative. Cross-asset sector boosts.

Strategy source: AVO run sentiment_contrarian-20260824-015120, step 22.
Data source: Master panel at data/feature_store/master_panel/daily.parquet

Cron: 5 16 * * 1-5  (4:05 PM ET, after market close)
State: paper_engines/state/sentiment_contrarian_avo_state.json
"""
import json
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "sentiment_contrarian_avo_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "sentiment_contrarian_avo.log"
CALLBACK_SCRIPT = BASE / "scripts" / "run_engine_with_callback.sh"
MASTER_PANEL = BASE / "data" / "feature_store" / "master_panel" / "daily.parquet"

# ── Strategy Parameters (from AVO v22 -- LOCKBOX VALIDATED) ──

INITIAL_CAPITAL = 10_000.0

# Tier 1: Deep oversold (high conviction)
T1_REL_STRENGTH = -0.07
T1_RET_5D = -0.07
T1_VOL_SPIKE = 1.15
T1_SCORE_BASE = 1.0

# Tier 2: Moderate oversold (lower conviction, broader)
T2_REL_STRENGTH = -0.03
T2_RET_10D = -0.055
T2_SCORE_BASE = 0.6

# Common filters
MIN_CLOSE_PRICE = 5.0
TREND_GUARD_20D = -0.10
MAX_SIGNALS_PER_DAY = 3

# Regime filter
SKIP_RISK_OFF = True  # skip risk_off AND risk_off_severe

# Sentiment boost (when available)
SENTIMENT_BOOST = 1.35

# Position sizing -- LOCKBOX: MAX_CONCURRENT=3 (Sharpe 2.73 at 3, -0.53 at 5)
MAX_PER_TRADE = 2000.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0005

# Exit parameters
TRAILING_STOP_PCT = -0.025
TRAILING_STOP_TIGHT = -0.015  # tighter after profit threshold
PROFIT_LOCK_THRESHOLD = 0.008  # tighten trail after +0.8% unrealized
TAKE_PROFIT_PCT = 0.043
MAX_HOLD_DAYS = 5
UNDERWATER_CUT_PCT = -0.002
UNDERWATER_CUT_DAYS = 1

# SPY regime filter (post-lockbox addition)
SPY_5D_DROP_THRESHOLD = -0.02  # skip when SPY dropped >2% in 5 days

# Data windows
TRAIN_WINDOW_DAYS = 126  # ~6 months of trading days
SIGNAL_WINDOW_DAYS = 30  # last 30 days for generate_signals (needs 25 rows min)


# ── Strategy Functions (verbatim from AVO v22 strategy.py) ──

def fit(train_data):
    """Learn sector-specific mean-reversion strength from training data."""
    sector_stats = {}
    tickers = train_data['ticker'].unique()

    for ticker in tickers:
        tdf = train_data[train_data['ticker'] == ticker].sort_values('date').copy()
        if len(tdf) < 30:
            continue

        close = tdf['close'].values
        if close[-1] < MIN_CLOSE_PRICE:
            continue

        ret_5d = pd.Series(close).pct_change(5).values
        rel_str = tdf['sr_rel_strength_spy'].values if 'sr_rel_strength_spy' in tdf.columns else None
        if rel_str is None:
            continue

        vol = tdf['volume'].values
        vol_ma20 = pd.Series(vol).rolling(20, min_periods=10).mean().values
        sector = tdf['sector'].iloc[0] if 'sector' in tdf.columns else 'unknown'

        fwd_rets = []
        for i in range(25, len(tdf) - 5):
            if np.isnan(ret_5d[i]) or np.isnan(rel_str[i]):
                continue
            if vol_ma20[i] <= 0 or np.isnan(vol_ma20[i]):
                continue

            vol_ratio = vol[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0

            if (ret_5d[i] < T1_RET_5D and
                rel_str[i] < T1_REL_STRENGTH and
                vol_ratio > T1_VOL_SPIKE):
                fwd = (close[i + 5] - close[i]) / close[i]
                fwd_rets.append(fwd)

        if fwd_rets:
            if sector not in sector_stats:
                sector_stats[sector] = {'fwd_rets': [], 'n_events': 0}
            sector_stats[sector]['fwd_rets'].extend(fwd_rets)
            sector_stats[sector]['n_events'] += len(fwd_rets)

    sector_scores = {}
    for sector, stats in sector_stats.items():
        if stats['n_events'] >= 3:
            avg_ret = np.mean(stats['fwd_rets'])
            hit_rate = np.mean([r > 0 for r in stats['fwd_rets']])
            sector_scores[sector] = float(avg_ret * (hit_rate - 0.4))
        else:
            sector_scores[sector] = 0.0

    return {'sector_scores': sector_scores}


def generate_signals(data, params):
    """Generate mean-reversion buy signals using two tiers."""
    sector_scores = params.get('sector_scores', {})
    raw_signals = []
    tickers = data['ticker'].unique()

    for ticker in tickers:
        tdf = data[data['ticker'] == ticker].sort_values('date').copy()
        if len(tdf) < 25:
            continue

        close = tdf['close'].values
        if close[-1] < MIN_CLOSE_PRICE:
            continue

        close_s = pd.Series(close, index=tdf.index)
        ret_5d = close_s.pct_change(5)
        ret_10d = close_s.pct_change(10)
        ret_20d = close_s.pct_change(20)

        rel_str = tdf['sr_rel_strength_spy'] if 'sr_rel_strength_spy' in tdf.columns else pd.Series(np.nan, index=tdf.index)

        vol = tdf['volume']
        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        vol_ratio = vol / vol_ma20.clip(lower=1)

        regime = tdf['regime_state'] if 'regime_state' in tdf.columns else pd.Series('neutral', index=tdf.index)
        sent = tdf['rh_sentiment_polarity'] if 'rh_sentiment_polarity' in tdf.columns else pd.Series(0.0, index=tdf.index)

        sector = tdf['sector'].iloc[0] if 'sector' in tdf.columns else 'unknown'
        sector_bonus = max(sector_scores.get(sector, 0), 0)

        sector_xa_map = {
            'Energy': 'xa_OIL_zscore_60d',
            'Basic Materials': 'xa_COPPER_zscore_60d',
            'Financial Services': 'xa_UST10Y_zscore_60d',
        }
        xa_col = sector_xa_map.get(sector, None)

        for i in range(25, len(tdf)):
            r5 = ret_5d.iloc[i]
            r10 = ret_10d.iloc[i] if i >= 10 else np.nan
            r20 = ret_20d.iloc[i] if i >= 20 else np.nan
            rs = rel_str.iloc[i]
            vr = vol_ratio.iloc[i]
            rg = regime.iloc[i]
            s_pol = sent.iloc[i]

            if pd.isna(rs):
                continue

            # Trend guard: skip severe downtrends
            if not pd.isna(r20) and r20 < TREND_GUARD_20D:
                continue

            # Regime filter: skip risk_off (both regular and severe)
            if SKIP_RISK_OFF and isinstance(rg, str):
                if 'risk_off' in rg.lower():
                    continue

            # SPY 5-day regime filter (post-lockbox addition)
            if not pd.isna(r5) and 'spy' not in ticker.lower():
                spy_df = data[data['ticker'] == 'SPY']
                if len(spy_df) > 0:
                    spy_close = spy_df['close'].values
                    if i < len(spy_close) and i >= 5:
                        spy_5d_ret = (spy_close[i] - spy_close[i - 5]) / spy_close[i - 5]
                        if spy_5d_ret < SPY_5D_DROP_THRESHOLD:
                            continue

            signal = None

            # TIER 1: Deep oversold + volume spike
            if not pd.isna(r5) and r5 < T1_RET_5D and rs < T1_REL_STRENGTH and vr > T1_VOL_SPIKE:
                score = np.sqrt(abs(r5) * abs(rs)) * 10 * T1_SCORE_BASE
                score *= (1.0 + sector_bonus)
                signal = ('tier1', score)

            # TIER 2: Moderate oversold on 10d, no volume requirement
            elif not pd.isna(r10) and r10 < T2_RET_10D and rs < T2_REL_STRENGTH:
                score = np.sqrt(abs(r10) * abs(rs)) * 5 * T2_SCORE_BASE
                score *= (1.0 + sector_bonus)
                signal = ('tier2', score)

            if signal is not None:
                tier, score = signal

                # Sentiment boost
                if not pd.isna(s_pol) and s_pol < -0.1:
                    score *= SENTIMENT_BOOST

                # Cross-asset boost
                if xa_col and xa_col in tdf.columns:
                    xa_val = tdf[xa_col].iloc[i]
                    if not pd.isna(xa_val) and xa_val < -1.0:
                        score *= 1.15

                raw_signals.append({
                    'date': tdf['date'].iloc[i],
                    'ticker': ticker,
                    'score': float(score),
                    'direction': 'long',
                    'tier': tier,
                })

    if not raw_signals:
        return pd.DataFrame(columns=['date', 'ticker', 'score', 'direction', 'tier'])

    # Cap signals per day to avoid day concentration
    df_sig = pd.DataFrame(raw_signals)
    df_sig = df_sig.sort_values('score', ascending=False)
    capped = df_sig.groupby('date').head(MAX_SIGNALS_PER_DAY)

    return capped.reset_index(drop=True)


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Exit logic: underwater cut, max hold, trailing stop, take profit."""
    entry_price = pos['entry_price_adj']
    direction = pos.get('direction', 'long')
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )

    if direction == 'long':
        pnl_pct = (current_price - entry_price) / entry_price
        high_water = pos.get('high_water_mark', entry_price)
        drawdown_from_high = (current_price - high_water) / high_water if high_water > 0 else 0
    else:
        pnl_pct = (entry_price - current_price) / entry_price
        low_water = pos.get('low_water_mark', entry_price)
        drawdown_from_high = (low_water - current_price) / low_water if low_water > 0 else 0

    # 1. Max hold days
    if days_held >= MAX_HOLD_DAYS:
        return True, "max_hold"

    # 2. Take profit
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, "take_profit"

    # 3. Trailing stop (tighter when in profit to lock gains)
    trail = TRAILING_STOP_TIGHT if pnl_pct >= PROFIT_LOCK_THRESHOLD else TRAILING_STOP_PCT
    if drawdown_from_high <= trail:
        return True, "trailing_stop"

    # 4. Underwater cut (1 day, tighter when portfolio is in drawdown)
    cut_level = UNDERWATER_CUT_PCT * 0.5 if portfolio_dd < -0.015 else UNDERWATER_CUT_PCT
    if days_held >= UNDERWATER_CUT_DAYS and pnl_pct < cut_level:
        return True, "underwater_cut"

    # 5. Portfolio circuit breaker
    if portfolio_dd < -0.05:
        return True, "circuit_breaker"

    return False, ""


# ── Engine Infrastructure ──

ENGINE_NAME = "SENTIMENT-CONTRARIAN-AVO"


def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [{ENGINE_NAME}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "positions": [],
        "closed_trades": [],
        "equity_curve": [],
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "peak_equity": INITIAL_CAPITAL,
        "max_drawdown": 0.0,
        "total_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "n_trades": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "created": datetime.now(ET).isoformat(),
        "last_run_date": None,
        "last_updated": None,
        "sector_scores": {},
        "last_fit_date": None,
    }


def save_state(state: dict):
    state["last_updated"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def fire_callback(trade_info: dict):
    """Fire signal callback for trade taken."""
    if not CALLBACK_SCRIPT.exists():
        return
    try:
        import os
        env_vars = {
            "ENGINE_NAME": "sentiment_contrarian_avo",
            "TRADE_TICKER": trade_info.get("ticker", ""),
            "TRADE_DIRECTION": trade_info.get("direction", "long"),
            "TRADE_ACTION": trade_info.get("action", ""),
            "TRADE_PRICE": str(trade_info.get("price", 0)),
            "TRADE_PNL": str(trade_info.get("pnl", 0)),
        }
        full_env = {**os.environ, **env_vars}
        subprocess.Popen(
            [str(CALLBACK_SCRIPT)],
            env=full_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log(f"WARN: Callback failed: {e}")


def load_master_panel():
    """Load master panel parquet and return as DataFrame."""
    if not MASTER_PANEL.exists():
        log(f"ERROR: Master panel not found at {MASTER_PANEL}")
        return None

    try:
        df = pd.read_parquet(MASTER_PANEL)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        log(f"ERROR: Failed to load master panel: {e}")
        return None


def get_train_and_signal_data(panel, today):
    """
    Split master panel into training window (last 6 months) and
    signal window (last 30 days). Both end at the latest available date <= today.
    """
    panel = panel[panel['date'] <= pd.Timestamp(today)]
    if panel.empty:
        return None, None

    max_date = panel['date'].max()

    # Training window: last ~6 months
    train_start = max_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS * 1.5)  # calendar days
    train_data = panel[panel['date'] >= train_start].copy()

    # Signal window: last 30 days (needs at least 25 rows per ticker)
    signal_start = max_date - pd.Timedelta(days=SIGNAL_WINDOW_DAYS * 1.5)
    signal_data = panel[panel['date'] >= signal_start].copy()

    return train_data, signal_data


def run():
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends
    if today.weekday() >= 5:
        log("Weekend -- skipping.")
        return

    log(f"=== Sentiment Contrarian AVO Paper Engine -- {today_str} ===")
    state = load_state()

    # Skip if already ran today
    if state.get("last_run_date") == today_str:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # Load master panel
    panel = load_master_panel()
    if panel is None:
        log("ERROR: No data. Aborting.")
        save_state(state)
        return

    train_data, signal_data = get_train_and_signal_data(panel, today)
    if train_data is None or signal_data is None:
        log("ERROR: Insufficient data after filtering. Aborting.")
        save_state(state)
        return

    n_train_tickers = train_data['ticker'].nunique()
    n_signal_tickers = signal_data['ticker'].nunique()
    train_dates = train_data['date'].nunique()
    signal_dates = signal_data['date'].nunique()
    log(f"Train: {n_train_tickers} tickers x {train_dates} days | Signal: {n_signal_tickers} tickers x {signal_dates} days")

    if signal_dates < 20:
        log(f"ERROR: Too few signal dates ({signal_dates}). Need at least 20. Aborting.")
        save_state(state)
        return

    # Get latest date in data
    last_data_date = signal_data['date'].max()
    data_age = (pd.Timestamp(today) - last_data_date).days
    if data_age > 3:
        log(f"WARNING: Latest data is {data_age} days old ({last_data_date.date()}). Possible holiday/stale data.")

    # --- 0. Fit sector scores (re-fit weekly or on first run) ---
    need_fit = (
        state.get("last_fit_date") is None or
        (today - datetime.fromisoformat(state["last_fit_date"]).date()).days >= 7
    )
    if need_fit:
        log("Fitting sector scores on training window...")
        fit_result = fit(train_data)
        state["sector_scores"] = fit_result['sector_scores']
        state["last_fit_date"] = today_str
        n_sectors = len([s for s, v in state["sector_scores"].items() if v != 0])
        log(f"  Fit complete: {n_sectors} sectors with non-zero scores")
    else:
        log(f"  Using cached sector scores (last fit: {state['last_fit_date']})")

    params = {'sector_scores': state.get('sector_scores', {})}

    # Get current prices for each ticker (latest row in signal data)
    latest_date = signal_data['date'].max()
    latest_rows = signal_data[signal_data['date'] == latest_date]
    current_prices = dict(zip(latest_rows['ticker'], latest_rows['close']))

    if not current_prices:
        log("ERROR: No current prices available. Aborting.")
        save_state(state)
        return

    log(f"  Latest data date: {latest_date.date()} | {len(current_prices)} tickers with prices")

    # --- 1. Check exits on open positions ---
    portfolio_dd = 0.0
    if state["peak_equity"] > 0:
        portfolio_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]

    still_open = []
    exits_today = []
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in current_prices:
            log(f"  WARN: No price for {ticker}, keeping position open")
            still_open.append(pos)
            continue

        price = current_prices[ticker]

        # Update high water mark
        if price > pos.get("high_water_mark", pos["entry_price_adj"]):
            pos["high_water_mark"] = price

        exit_flag, exit_reason = should_exit(pos, price, today_str, portfolio_dd)

        if exit_flag:
            # Apply slippage on exit
            exit_price = price * (1 - SLIPPAGE_PCT)
            shares = pos["shares"]
            pnl = (exit_price - pos["entry_price_adj"]) * shares
            pnl_pct = (exit_price / pos["entry_price_adj"] - 1) * 100
            days_held = np.busday_count(
                np.datetime64(pos["entry_date"], 'D'),
                np.datetime64(today_str, 'D'))

            state["capital"] += shares * exit_price
            state["total_pnl"] += pnl
            state["n_trades"] += 1
            if pnl >= 0:
                state["wins"] += 1
                state["gross_profit"] += pnl
            else:
                state["losses"] += 1
                state["gross_loss"] += abs(pnl)

            trade_record = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": today_str,
                "entry_price": pos["entry_price_adj"],
                "exit_price": round(exit_price, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": int(days_held),
                "exit_reason": exit_reason,
                "direction": pos.get("direction", "long"),
                "tier": pos.get("tier", "unknown"),
            }
            state["closed_trades"].append(trade_record)
            state["closed_trades"] = state["closed_trades"][-200:]  # Keep last 200
            exits_today.append(trade_record)

            log(f"  EXIT {ticker}: ${pnl:+.2f} ({pnl_pct:+.2f}%) after {days_held}d [{exit_reason}]")

            fire_callback({
                "ticker": ticker, "action": "exit", "direction": "sell",
                "price": exit_price, "pnl": round(pnl, 2),
            })
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Generate signals and check for new entries ---
    entries_today = []
    n_open = len(state["positions"])

    if n_open < MAX_CONCURRENT:
        signals_df = generate_signals(signal_data, params)

        # Filter to today's signals only (latest date in data)
        today_signals = signals_df[signals_df['date'] == latest_date]
        today_signals = today_signals.sort_values('score', ascending=False)

        if len(today_signals) > 0:
            log(f"  Signals firing: {', '.join(today_signals['ticker'].tolist())} "
                f"(scores: {', '.join(f'{s:.3f}' for s in today_signals['score'])})")

        held_tickers = {p["ticker"] for p in state["positions"]}

        # Anti-re-entry: skip tickers that exited in last 10 trading days (HC #807 R2 / HC #812)
        recent_exit_tickers = set()
        for ct in state.get("closed_trades", []):
            exit_date = ct.get("exit_date", "")
            if exit_date:
                try:
                    days_since = np.busday_count(
                        np.datetime64(exit_date, 'D'),
                        np.datetime64(today_str, 'D')
                    )
                    if days_since <= 10:
                        recent_exit_tickers.add(ct["ticker"])
                except Exception:
                    pass

        for _, sig in today_signals.iterrows():
            if n_open >= MAX_CONCURRENT:
                log(f"  SKIP {sig['ticker']}: max concurrent ({MAX_CONCURRENT}) reached")
                break
            if sig['ticker'] in held_tickers:
                log(f"  SKIP {sig['ticker']}: already holding")
                continue
            if sig['ticker'] in recent_exit_tickers:
                log(f"  SKIP {sig['ticker']}: recent exit within 10d (HC #807 R2)")
                continue
            if sig['ticker'] not in current_prices:
                continue

            ticker = sig['ticker']
            price = current_prices[ticker]

            # Apply slippage on entry
            entry_price = price * (1 + SLIPPAGE_PCT)

            # Position sizing: max $2,000 per position, limited by available capital
            position_size = min(MAX_PER_TRADE, state["capital"] * 0.95)  # Keep 5% cash buffer

            if position_size < 50:
                log(f"  SKIP {ticker}: insufficient capital (${state['capital']:.0f})")
                continue

            shares = position_size / entry_price
            cost = shares * entry_price
            state["capital"] -= cost

            pos = {
                "ticker": ticker,
                "shares": round(shares, 6),
                "entry_price_adj": round(entry_price, 4),
                "entry_date": today_str,
                "high_water_mark": round(entry_price, 4),
                "direction": sig.get("direction", "long"),
                "tier": sig.get("tier", "unknown"),
                "score": round(sig["score"], 4),
            }
            state["positions"].append(pos)
            held_tickers.add(ticker)
            n_open += 1
            entries_today.append(pos)

            log(f"  ENTRY {ticker}: {shares:.4f} shares @ ${entry_price:.2f} "
                f"(tier={sig.get('tier', '?')}, score={sig['score']:.3f})")

            fire_callback({
                "ticker": ticker, "action": "entry", "direction": "long",
                "price": entry_price, "pnl": 0,
            })
    else:
        log(f"  Max concurrent positions ({MAX_CONCURRENT}) -- skipping signal scan")

    # --- 3. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in current_prices:
            portfolio_value += pos["shares"] * current_prices[ticker]
        else:
            portfolio_value += pos["shares"] * pos["entry_price_adj"]

    state["equity"] = round(portfolio_value, 2)

    # Update peak equity and drawdown
    if state["equity"] > state["peak_equity"]:
        state["peak_equity"] = state["equity"]
    current_dd = 0.0
    if state["peak_equity"] > 0:
        current_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]
    if current_dd < state["max_drawdown"]:
        state["max_drawdown"] = round(current_dd, 6)

    # Record equity curve point
    state["equity_curve"].append({
        "date": today_str,
        "equity": state["equity"],
        "positions": len(state["positions"]),
    })
    state["equity_curve"] = state["equity_curve"][-500:]  # Keep last 500 days

    state["last_run_date"] = today_str
    save_state(state)

    # --- 4. Print summary ---
    print_summary(state, entries_today, exits_today)
    log("Done.")


def print_summary(state, entries_today=None, exits_today=None):
    """Print clean status summary to stdout."""
    entries_today = entries_today or []
    exits_today = exits_today or []

    total_trades = state["n_trades"]
    wr = state["wins"] / total_trades * 100 if total_trades > 0 else 0
    pf = state["gross_profit"] / state["gross_loss"] if state["gross_loss"] > 0 else float('inf')
    avg_win = state["gross_profit"] / state["wins"] if state["wins"] > 0 else 0
    avg_loss = state["gross_loss"] / state["losses"] if state["losses"] > 0 else 0
    return_pct = (state["equity"] / INITIAL_CAPITAL - 1) * 100
    dd_pct = state["max_drawdown"] * 100

    print("\n" + "=" * 60)
    print(f"  {ENGINE_NAME} -- Paper Trading Status")
    print("=" * 60)
    print(f"  Equity:     ${state['equity']:,.2f}  ({return_pct:+.2f}%)")
    print(f"  Cash:       ${state['capital']:,.2f}")
    print(f"  Max DD:     {dd_pct:.2f}%")
    print(f"  Total PnL:  ${state['total_pnl']:+,.2f}")
    print("-" * 60)
    print(f"  Trades:     {total_trades}  |  W/L: {state['wins']}/{state['losses']}  |  WR: {wr:.1f}%")
    print(f"  PF: {pf:.2f}  |  Avg Win: ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print("-" * 60)

    if state["positions"]:
        print("  Open Positions:")
        for pos in state["positions"]:
            print(f"    {pos['ticker']:5s}  {pos['shares']:.2f} sh @ ${pos['entry_price_adj']:.2f}  "
                  f"(entered {pos['entry_date']}, {pos.get('tier', '?')})")
    else:
        print("  No open positions.")

    if entries_today:
        print(f"\n  Today's Entries: {', '.join(p['ticker'] for p in entries_today)}")
    if exits_today:
        exit_strs = [f"{t['ticker']} ${t['pnl']:+.2f}" for t in exits_today]
        print(f"  Today's Exits:  {', '.join(exit_strs)}")

    print("=" * 60 + "\n")

    log(f"Equity: ${state['equity']:.2f} | Positions: {len(state['positions'])} | "
        f"Trades: {total_trades} | WR: {wr:.0f}% | PF: {pf:.2f} | PnL: ${state['total_pnl']:+.2f} | DD: {dd_pct:.2f}%")


if __name__ == "__main__":
    run()
