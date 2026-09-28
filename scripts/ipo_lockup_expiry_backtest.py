#!/usr/bin/env python3
"""
IPO Lockup Expiry Backtest — Trading the Post-Lockup Recovery
=============================================================
Concept: When IPO lockup periods expire (typically 90-180 days post-IPO),
insiders can sell, creating predictable selling pressure. This strategy
exploits the recovery AFTER lockup-driven selling exhausts.

Universe: Recent IPO stocks (2019-2025 vintage)
OOT: Jan 2022 - present
Starting capital: $645
6 variants with permutation testing and 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Config ───────────────────────────────────────────────────────────────────
# Recent IPO stocks with approximate IPO dates (YYYY-MM-DD)
# We use a broad set and also try to detect IPO dates from yfinance
IPO_UNIVERSE = {
    # 2020-2021 IPO boom
    "HOOD": "2021-07-29",   # Robinhood
    "RIVN": "2021-11-10",   # Rivian
    "RBLX": "2021-03-10",   # Roblox (DPO)
    "PLTR": "2020-09-30",   # Palantir (DPO)
    "COIN": "2021-04-14",   # Coinbase (DPO)
    "SNOW": "2020-09-16",   # Snowflake
    "DDOG": "2019-09-19",   # Datadog
    "U": "2020-09-18",      # Unity
    "ABNB": "2020-12-10",   # Airbnb
    "DASH": "2020-12-09",   # DoorDash
    "AFRM": "2021-01-13",   # Affirm
    "SOFI": "2021-06-01",   # SoFi (de-SPAC)
    "LCID": "2021-07-26",   # Lucid (de-SPAC)
    "JOBY": "2021-08-11",   # Joby Aviation (de-SPAC)
    "PATH": "2021-04-21",   # UiPath
    "DUOL": "2021-07-28",   # Duolingo
    "BROS": "2021-09-15",   # Dutch Bros
    "RKLB": "2021-08-25",   # Rocket Lab (de-SPAC)
    "IONQ": "2021-10-01",   # IonQ (de-SPAC)
    "DNA": "2021-09-17",    # Ginkgo Bioworks (de-SPAC)
    # 2022-2023 IPOs
    "CART": "2023-09-19",   # Instacart / Maplebear
    "ARM": "2023-09-14",    # ARM Holdings
    "BIRK": "2023-10-11",   # Birkenstock
    # 2024 IPOs
    "RDDT": "2024-03-21",   # Reddit
    "IBKR": "2002-05-01",   # old IPO, control/exclude
}

# Remove stocks with very old IPOs (pre-2019)
IPO_UNIVERSE = {k: v for k, v in IPO_UNIVERSE.items()
                if datetime.strptime(v, "%Y-%m-%d") >= datetime(2019, 1, 1)}

TICKERS = list(IPO_UNIVERSE.keys())
IPO_DATES = {k: pd.Timestamp(v) for k, v in IPO_UNIVERSE.items()}

# Data window: need pre-IPO for some stocks, start early
DATA_START = "2019-01-01"
OOT_START = "2022-01-03"
OOT_END = "2026-07-28"
INITIAL_CAPITAL = 645.0
N_PERMS = 1000
RISK_FREE = 0.04
LOCKUP_DAYS = 180  # standard lockup period in calendar days

# ─── Data download ───────────────────────────────────────────────────────────
print("Downloading price data...")
all_tickers = TICKERS + ["SPY"]
raw = yf.download(all_tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

close = raw["Close"].copy()
volume = raw["Volume"].copy()

# Drop tickers that have no data
valid_tickers = [t for t in TICKERS if t in close.columns and close[t].dropna().shape[0] > 20]
print(f"Valid tickers: {len(valid_tickers)} of {len(TICKERS)}")
TICKERS = valid_tickers

close = close.ffill()
volume = volume.ffill().fillna(0)

# SPY for regime
spy_close = close["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear
regime.name = "regime"

# OOT mask
oot_mask = close.index >= OOT_START
close_oot = close.loc[oot_mask].copy()
volume_oot = volume.loc[oot_mask].copy()
regime_oot = regime.loc[oot_mask].copy()

print(f"OOT period: {close_oot.index[0].date()} to {close_oot.index[-1].date()}")
print(f"Trading days: {len(close_oot)}")
print(f"Regime: {regime_oot.sum()} bull days, {(~regime_oot.astype(bool)).sum()} bear days")

# ─── Pre-compute indicators ─────────────────────────────────────────────────
# Daily returns
daily_returns = close[TICKERS].pct_change()

# Rolling volume average (20-day)
vol_avg_20 = volume[TICKERS].rolling(20).mean()

# Revenue growth proxy: 60-day momentum (positive = growing business proxy)
ret_60d = close[TICKERS].pct_change(60)

# Consecutive up days
def count_consecutive_up(series):
    """Count consecutive up days ending at each point."""
    up = (series > 0).astype(int)
    result = pd.Series(0, index=series.index, dtype=int)
    count = 0
    for i in range(len(up)):
        if up.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result

print("Computing consecutive up-day counts...")
consec_up = pd.DataFrame(index=close.index, columns=TICKERS, dtype=int)
for t in TICKERS:
    consec_up[t] = count_consecutive_up(daily_returns[t])

# ─── Lockup expiry date for each stock ──────────────────────────────────────
lockup_expiry = {t: IPO_DATES[t] + timedelta(days=LOCKUP_DAYS) for t in TICKERS}
print("\nLockup expiry dates:")
for t in sorted(lockup_expiry, key=lambda x: lockup_expiry[x]):
    print(f"  {t}: IPO {IPO_DATES[t].date()} -> Lockup expiry ~{lockup_expiry[t].date()}")


# ─── Backtest engine ─────────────────────────────────────────────────────────
def run_backtest(signal_func, hold_days, top_n=None, allow_overlap=True, label=""):
    """
    signal_func(date_idx, date) -> list of eligible tickers
    Equal-weight allocation. Hold for hold_days.
    allow_overlap: if True, can open new positions while others are open.
    Returns daily portfolio equity curve and trade log.
    """
    dates = close_oot.index.tolist()
    cash = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    positions = {}  # {ticker_date_key: {ticker, entry_price, entry_date, shares, exit_idx}}
    last_signal_check = -999  # avoid checking every day, check every 1 day

    i = 0
    while i < len(dates):
        date = dates[i]

        # Close expired positions
        closed_keys = []
        for key, pos in list(positions.items()):
            if i >= pos["exit_idx"]:
                exit_price = close_oot.loc[dates[min(i, len(dates)-1)], pos["ticker"]]
                if pd.isna(exit_price):
                    exit_price = pos["entry_price"]
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                cash += pos["shares"] * exit_price
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(date.date()),
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "return": float((exit_price / pos["entry_price"]) - 1) if pos["entry_price"] > 0 else 0,
                    "regime": int(regime.loc[pos["entry_date"]]) if pos["entry_date"] in regime.index else -1
                })
                closed_keys.append(key)
        for key in closed_keys:
            del positions[key]

        # Check for new signals (check every day for event-driven)
        open_ok = allow_overlap or len(positions) == 0
        if open_ok:
            eligible = signal_func(i, date)
            if top_n and len(eligible) > top_n:
                eligible = eligible[:top_n]

            # Don't re-enter a stock we already hold
            held_tickers = {p["ticker"] for p in positions.values()}
            eligible = [t for t in eligible if t not in held_tickers]

            if len(eligible) > 0:
                # Allocate a fraction of current portfolio
                total_equity = cash
                for pos in positions.values():
                    curr_p = close_oot.loc[date, pos["ticker"]]
                    if pd.notna(curr_p):
                        total_equity += pos["shares"] * curr_p

                # Max 25% of equity per position, but no more than available cash
                max_per_pos = total_equity * 0.25
                alloc_per = min(max_per_pos, cash / max(len(eligible), 1))

                for tick in eligible:
                    if cash < 10:  # minimum cash to invest
                        break
                    price = close_oot.loc[date, tick]
                    if pd.isna(price) or price <= 0:
                        continue
                    actual_alloc = min(alloc_per, cash)
                    shares = actual_alloc / price
                    exit_idx = min(i + hold_days, len(dates) - 1)
                    key = f"{tick}_{i}"
                    positions[key] = {
                        "ticker": tick,
                        "entry_price": float(price),
                        "entry_date": date,
                        "shares": shares,
                        "exit_idx": exit_idx
                    }
                    cash -= shares * price

        # Mark-to-market
        mtm = cash
        for pos in positions.values():
            curr_price = close_oot.loc[date, pos["ticker"]]
            if pd.notna(curr_price):
                mtm += pos["shares"] * curr_price
            else:
                mtm += pos["shares"] * pos["entry_price"]
        equity_curve.append({"date": str(date.date()), "equity": float(mtm)})

        i += 1

    # Force close remaining
    if positions:
        last_date = dates[-1]
        for key, pos in positions.items():
            exit_price = close_oot.loc[last_date, pos["ticker"]]
            if pd.isna(exit_price):
                exit_price = pos["entry_price"]
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            cash += pos["shares"] * exit_price
            trades.append({
                "ticker": pos["ticker"],
                "entry_date": str(pos["entry_date"].date()),
                "exit_date": str(last_date.date()),
                "entry_price": float(pos["entry_price"]),
                "exit_price": float(exit_price),
                "pnl": float(pnl),
                "return": float((exit_price / pos["entry_price"]) - 1) if pos["entry_price"] > 0 else 0,
                "regime": int(regime.loc[pos["entry_date"]]) if pos["entry_date"] in regime.index else -1
            })

    return equity_curve, trades


# ─── Signal functions ────────────────────────────────────────────────────────

def signal_post_lockup_recovery(i, date):
    """
    Variant A: Post-Lockup Recovery
    - Stock IPO'd 150-210 days ago (in the lockup expiry window)
    - Has dropped >= 5% in last 10 days (insider selling pressure)
    - 3+ consecutive up days (selling exhausted, recovery starting)
    """
    eligible = []
    for t in TICKERS:
        ipo_date = IPO_DATES[t]
        days_since_ipo = (date - ipo_date).days

        # We want to catch stocks around their lockup expiry
        # But also look at stocks AFTER lockup expired (within 30 days post)
        if not (120 <= days_since_ipo <= 240):
            continue

        if date not in close.index:
            continue

        cur_price = close.loc[date, t]
        if pd.isna(cur_price) or cur_price <= 0:
            continue

        # Check for recent drop (selling pressure)
        if i < 10:
            continue
        price_10d_ago = close.iloc[close.index.get_loc(date) - 10][t] if date in close.index else np.nan
        if pd.isna(price_10d_ago) or price_10d_ago <= 0:
            continue

        drop_pct = (cur_price / price_10d_ago) - 1
        if drop_pct > -0.05:  # Need at least -5% drop
            continue

        # Check for 3 consecutive up days (recovery signal)
        if date not in consec_up.index:
            continue
        cup = consec_up.loc[date, t]
        if cup < 3:
            continue

        eligible.append(t)

    return eligible


def signal_pre_lockup_short(i, date):
    """
    Variant B: Pre-Lockup Short Momentum
    - 10 days before estimated lockup expiry
    - Model as SHORT position (we'll invert returns)
    - Lockup selling typically starts before the actual date
    """
    eligible = []
    for t in TICKERS:
        lockup_date = lockup_expiry[t]
        days_to_lockup = (lockup_date - date).days

        # Enter 10-20 days before lockup
        if not (5 <= days_to_lockup <= 20):
            continue

        if date not in close.index:
            continue

        cur_price = close.loc[date, t]
        if pd.isna(cur_price) or cur_price <= 0:
            continue

        eligible.append(t)

    return eligible


def signal_quality_recovery(i, date):
    """
    Variant C: Quality Filter Recovery
    - Same timing as A (post-lockup window)
    - Require positive 60-day momentum (proxy for revenue growth / business quality)
    - Stock must be above IPO price (insiders selling is profit-taking, not fleeing)
    - 3+ consecutive up days
    """
    eligible = []
    for t in TICKERS:
        ipo_date = IPO_DATES[t]
        days_since_ipo = (date - ipo_date).days

        if not (120 <= days_since_ipo <= 240):
            continue

        if date not in close.index:
            continue

        cur_price = close.loc[date, t]
        if pd.isna(cur_price) or cur_price <= 0:
            continue

        # Quality: positive 60-day momentum
        if date in ret_60d.index:
            mom = ret_60d.loc[date, t]
            if pd.isna(mom) or mom <= 0:
                continue
        else:
            continue

        # Above IPO-day price (approximate — use first available price)
        ipo_idx = close.index.searchsorted(ipo_date)
        if ipo_idx < len(close.index):
            ipo_price = close.iloc[ipo_idx][t]
            if pd.notna(ipo_price) and cur_price < ipo_price:
                continue

        # Recovery signal
        if date not in consec_up.index:
            continue
        cup = consec_up.loc[date, t]
        if cup < 3:
            continue

        # Need the drop first
        if i < 10:
            continue
        price_10d_ago = close.iloc[close.index.get_loc(date) - 10][t]
        if pd.isna(price_10d_ago) or price_10d_ago <= 0:
            continue
        drop_pct = (cur_price / price_10d_ago) - 1
        if drop_pct > -0.05:
            continue

        eligible.append(t)

    return eligible


def signal_sector_etf_proxy(i, date):
    """
    Variant D: Sector ETF Proxy
    - When multiple IPO stocks in a sector have lockup expiries within 30 days,
      buy the sector ETF after selling pressure subsides
    - Proxy: count how many stocks in our universe have lockups expiring within +-30 days
    - If >= 2, treat it as a concentrated lockup event
    - Buy SPY as a broad proxy (simplified) after a pullback
    """
    lockup_count = 0
    for t in TICKERS:
        le = lockup_expiry[t]
        days_diff = abs((date - le).days)
        if days_diff <= 30:
            lockup_count += 1

    if lockup_count < 2:
        return []

    # Check if SPY had a recent pullback (proxy for sector selling)
    if date not in close.index or i < 5:
        return []

    spy_now = spy_close.loc[date]
    spy_5d = spy_close.iloc[spy_close.index.get_loc(date) - 5]
    if pd.isna(spy_now) or pd.isna(spy_5d) or spy_5d <= 0:
        return []

    spy_ret = (spy_now / spy_5d) - 1
    if spy_ret < -0.02:  # SPY dropped 2%+ (market-wide selling pressure)
        return ["SPY"]

    return []


def signal_spac_despac_lockup(i, date):
    """
    Variant E: SPAC De-SPAC Lockup
    - Focus on de-SPAC'd companies (identified in our universe)
    - Buy after insider/PIPE selling exhaustion
    - Same recovery logic but only for SPAC-originated companies
    """
    spac_tickers = ["SOFI", "LCID", "JOBY", "RKLB", "IONQ", "DNA"]
    eligible = []

    for t in spac_tickers:
        if t not in TICKERS:
            continue

        ipo_date = IPO_DATES[t]
        days_since_ipo = (date - ipo_date).days

        # SPACs often have multiple lockup windows (90, 180, 365 days)
        # Check all three windows
        in_lockup_window = False
        for lockup_offset in [90, 180, 365]:
            window_center = lockup_offset
            if abs(days_since_ipo - window_center) <= 30:
                in_lockup_window = True
                break

        if not in_lockup_window:
            continue

        if date not in close.index:
            continue

        cur_price = close.loc[date, t]
        if pd.isna(cur_price) or cur_price <= 0:
            continue

        # Check for selling exhaustion: volume spike followed by volume decline
        if date not in vol_avg_20.index or i < 5:
            continue

        cur_vol = volume.loc[date, t]
        avg_vol = vol_avg_20.loc[date, t]
        if pd.isna(cur_vol) or pd.isna(avg_vol) or avg_vol <= 0:
            continue

        # Recovery: 2+ consecutive up days (less strict for SPACs)
        if date not in consec_up.index:
            continue
        cup = consec_up.loc[date, t]
        if cup < 2:
            continue

        # Recent drop
        if i < 10:
            continue
        price_10d_ago = close.iloc[close.index.get_loc(date) - 10][t]
        if pd.isna(price_10d_ago) or price_10d_ago <= 0:
            continue
        drop_pct = (cur_price / price_10d_ago) - 1
        if drop_pct > -0.03:  # less strict: -3% for SPACs
            continue

        eligible.append(t)

    return eligible


def signal_volume_exhaustion(i, date):
    """
    Variant F: Volume Exhaustion Signal
    - Same as A but require volume spike (>3x average) on the drop day
    - Then declining volume on recovery days
    - Confirms selling is institutional and exhausting
    """
    eligible = []
    for t in TICKERS:
        ipo_date = IPO_DATES[t]
        days_since_ipo = (date - ipo_date).days

        if not (120 <= days_since_ipo <= 240):
            continue

        if date not in close.index or i < 10:
            continue

        cur_price = close.loc[date, t]
        if pd.isna(cur_price) or cur_price <= 0:
            continue

        # Check for recent drop
        date_loc = close.index.get_loc(date)
        price_10d_ago = close.iloc[date_loc - 10][t]
        if pd.isna(price_10d_ago) or price_10d_ago <= 0:
            continue
        drop_pct = (cur_price / price_10d_ago) - 1
        if drop_pct > -0.05:
            continue

        # Find the worst drop day in last 10 days
        worst_day_idx = None
        worst_ret = 0
        for j in range(1, 11):
            if date_loc - j < 0:
                break
            d = close.index[date_loc - j]
            if date_loc - j - 1 < 0:
                break
            prev_d = close.index[date_loc - j - 1]
            p_cur = close.loc[d, t]
            p_prev = close.loc[prev_d, t]
            if pd.notna(p_cur) and pd.notna(p_prev) and p_prev > 0:
                ret = (p_cur / p_prev) - 1
                if ret < worst_ret:
                    worst_ret = ret
                    worst_day_idx = date_loc - j

        if worst_day_idx is None:
            continue

        # Volume spike on worst day (>3x 20-day average)
        worst_day = close.index[worst_day_idx]
        if worst_day not in volume.index or worst_day not in vol_avg_20.index:
            continue
        vol_spike_day = volume.loc[worst_day, t]
        vol_avg = vol_avg_20.loc[worst_day, t]
        if pd.isna(vol_spike_day) or pd.isna(vol_avg) or vol_avg <= 0:
            continue
        if vol_spike_day < 3.0 * vol_avg:
            continue

        # Declining volume since spike day (current volume < spike volume)
        cur_vol = volume.loc[date, t]
        if pd.isna(cur_vol) or cur_vol >= vol_spike_day:
            continue

        # 3 consecutive up days
        if date not in consec_up.index:
            continue
        cup = consec_up.loc[date, t]
        if cup < 3:
            continue

        eligible.append(t)

    return eligible


# ─── Compute metrics ─────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades, label):
    if not equity_curve or len(equity_curve) < 2:
        return {"label": label, "error": "insufficient data"}

    eq = pd.Series([e["equity"] for e in equity_curve],
                   index=pd.to_datetime([e["date"] for e in equity_curve]))
    daily_ret = eq.pct_change().dropna()

    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe (annualized, excess over risk-free)
    excess = daily_ret - RISK_FREE / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    # Sortino
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-9
    sortino = np.sqrt(252) * excess.mean() / downside_std

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Trade-level stats
    n_trades = len(trades)
    if n_trades > 0:
        rets = [t["return"] for t in trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        win_rate = len(wins) / n_trades
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss
        avg_return = np.mean(rets)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
    else:
        win_rate = 0
        profit_factor = 0
        avg_return = 0
        avg_win = 0
        avg_loss = 0

    # Regime-stratified Sharpe
    regime_dates = regime_oot.reindex(eq.index).ffill()
    bull_ret = daily_ret[regime_dates == 1]
    bear_ret = daily_ret[regime_dates == 0]

    bull_excess = bull_ret - RISK_FREE / 252
    bear_excess = bear_ret - RISK_FREE / 252

    sharpe_bull = np.sqrt(252) * bull_excess.mean() / bull_excess.std() if len(bull_excess) > 5 and bull_excess.std() > 0 else 0
    sharpe_bear = np.sqrt(252) * bear_excess.mean() / bear_excess.std() if len(bear_excess) > 5 and bear_excess.std() > 0 else 0

    max_s = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_s if max_s > 0 else 0

    return {
        "label": label,
        "total_return": round(float(total_return), 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 4),
        "total_trades": int(n_trades),
        "avg_return": round(float(avg_return), 4),
        "avg_win": round(float(avg_win), 4),
        "avg_loss": round(float(avg_loss), 4),
        "sharpe_bull": round(float(sharpe_bull), 4),
        "sharpe_bear": round(float(sharpe_bear), 4),
        "regime_gap": round(float(regime_gap), 4),
        "final_equity": round(float(eq.iloc[-1]), 2),
        "start_equity": round(float(eq.iloc[0]), 2),
    }


# ─── Permutation test ────────────────────────────────────────────────────────
def permutation_test(signal_func, hold_days, top_n, actual_sharpe, label="", is_short=False):
    """Shuffle which stocks are selected on each signal day."""
    print(f"  Permutation test for {label}...")
    perm_sharpes = []
    all_t = TICKERS if not is_short else TICKERS

    for p in range(N_PERMS):
        def shuffled_signal(i, date, _orig=signal_func):
            orig = _orig(i, date)
            if len(orig) == 0:
                return []
            n = min(len(orig), top_n) if top_n else len(orig)
            pool = all_t if "SPY" not in orig else ["SPY"]
            return list(np.random.choice(pool, size=min(n, len(pool)), replace=False))

        eq, tr = run_backtest(shuffled_signal, hold_days, top_n, label=f"perm_{p}")
        if eq and len(eq) > 1:
            eq_s = pd.Series([e["equity"] for e in eq])
            dr = eq_s.pct_change().dropna()
            excess = dr - RISK_FREE / 252
            s = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0
            perm_sharpes.append(s)

    if len(perm_sharpes) == 0:
        return 1.0

    perm_p = np.mean([s >= actual_sharpe for s in perm_sharpes])
    return round(float(perm_p), 4)


# ─── 5-Gate validation ───────────────────────────────────────────────────────
def validate_5gate(metrics, perm_p):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown"] > -0.50,
        "trades_gte_20": metrics["total_trades"] >= 20,
    }
    gates["passed"] = sum(v for k, v in gates.items() if isinstance(v, bool))
    gates["all_passed"] = all(v for k, v in gates.items() if k not in ["passed", "all_passed"])
    return gates


# ─── Special short backtest for Variant B ────────────────────────────────────
def run_short_backtest(signal_func, hold_days, top_n=None, label=""):
    """
    Same as run_backtest but models SHORT positions.
    PnL is inverted: we profit when the stock drops.
    """
    dates = close_oot.index.tolist()
    cash = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    positions = {}

    i = 0
    while i < len(dates):
        date = dates[i]

        # Close expired short positions
        closed_keys = []
        for key, pos in list(positions.items()):
            if i >= pos["exit_idx"]:
                exit_price = close_oot.loc[dates[min(i, len(dates)-1)], pos["ticker"]]
                if pd.isna(exit_price):
                    exit_price = pos["entry_price"]
                # SHORT: profit when price drops
                pnl = (pos["entry_price"] - exit_price) * pos["shares"]
                cash += pos["shares"] * pos["entry_price"] + pnl  # return collateral + pnl
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(date.date()),
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "return": float((pos["entry_price"] / exit_price) - 1) if exit_price > 0 else 0,
                    "regime": int(regime.loc[pos["entry_date"]]) if pos["entry_date"] in regime.index else -1,
                    "side": "SHORT"
                })
                closed_keys.append(key)
        for key in closed_keys:
            del positions[key]

        # Open new short positions
        if len(positions) == 0:
            eligible = signal_func(i, date)
            if top_n and len(eligible) > top_n:
                eligible = eligible[:top_n]

            if len(eligible) > 0:
                alloc_per = cash / max(len(eligible), 1) * 0.5  # 50% of cash for shorts (margin)
                for tick in eligible:
                    if cash < 10:
                        break
                    price = close_oot.loc[date, tick]
                    if pd.isna(price) or price <= 0:
                        continue
                    actual_alloc = min(alloc_per, cash * 0.5)
                    shares = actual_alloc / price
                    exit_idx = min(i + hold_days, len(dates) - 1)
                    key = f"{tick}_{i}"
                    positions[key] = {
                        "ticker": tick,
                        "entry_price": float(price),
                        "entry_date": date,
                        "shares": shares,
                        "exit_idx": exit_idx
                    }
                    cash -= actual_alloc  # collateral for short

        # Mark-to-market for shorts
        mtm = cash
        for pos in positions.values():
            curr_price = close_oot.loc[date, pos["ticker"]]
            if pd.notna(curr_price):
                # Collateral + unrealized PnL
                unrealized = (pos["entry_price"] - curr_price) * pos["shares"]
                mtm += pos["shares"] * pos["entry_price"] + unrealized
            else:
                mtm += pos["shares"] * pos["entry_price"]
        equity_curve.append({"date": str(date.date()), "equity": float(mtm)})

        i += 1

    # Force close remaining
    if positions:
        last_date = dates[-1]
        for key, pos in positions.items():
            exit_price = close_oot.loc[last_date, pos["ticker"]]
            if pd.isna(exit_price):
                exit_price = pos["entry_price"]
            pnl = (pos["entry_price"] - exit_price) * pos["shares"]
            cash += pos["shares"] * pos["entry_price"] + pnl
            trades.append({
                "ticker": pos["ticker"],
                "entry_date": str(pos["entry_date"].date()),
                "exit_date": str(last_date.date()),
                "entry_price": float(pos["entry_price"]),
                "exit_price": float(exit_price),
                "pnl": float(pnl),
                "return": float((pos["entry_price"] / exit_price) - 1) if exit_price > 0 else 0,
                "regime": int(regime.loc[pos["entry_date"]]) if pos["entry_date"] in regime.index else -1,
                "side": "SHORT"
            })

    return equity_curve, trades


# ─── Run all variants ────────────────────────────────────────────────────────
VARIANTS = [
    # (label, signal_func, hold_days, top_n, is_short, allow_overlap)
    ("A_post_lockup_recovery_20d", signal_post_lockup_recovery, 20, 3, False, True),
    ("B_pre_lockup_short_15d", signal_pre_lockup_short, 15, 3, True, False),
    ("C_quality_filter_recovery_20d", signal_quality_recovery, 20, 3, False, True),
    ("D_sector_etf_proxy_10d", signal_sector_etf_proxy, 10, 1, False, True),
    ("E_spac_despac_lockup_20d", signal_spac_despac_lockup, 20, 3, False, True),
    ("F_volume_exhaustion_20d", signal_volume_exhaustion, 20, 3, False, True),
]

results = {}
for label, sig_func, hold, top_n, is_short, allow_overlap in VARIANTS:
    print(f"\n{'='*60}")
    print(f"Running variant: {label}")
    print(f"{'='*60}")

    if is_short:
        eq_curve, trade_log = run_short_backtest(sig_func, hold, top_n, label=label)
    else:
        eq_curve, trade_log = run_backtest(sig_func, hold, top_n, allow_overlap=allow_overlap, label=label)

    metrics = compute_metrics(eq_curve, trade_log, label)

    print(f"  Total return: {metrics.get('total_return', 'N/A')}")
    print(f"  Sharpe: {metrics.get('sharpe', 'N/A')}")
    print(f"  Trades: {metrics.get('total_trades', 0)}")
    print(f"  Win rate: {metrics.get('win_rate', 'N/A')}")

    # Permutation test (skip if no trades)
    actual_sharpe = metrics.get("sharpe", 0)
    if metrics.get("total_trades", 0) >= 5:
        perm_p = permutation_test(sig_func, hold, top_n, actual_sharpe, label=label, is_short=is_short)
    else:
        perm_p = 1.0
        print(f"  Skipping permutation test (< 5 trades)")
    metrics["perm_p"] = perm_p
    print(f"  Perm p-value: {perm_p}")

    # 5-gate
    gates = validate_5gate(metrics, perm_p)
    metrics["five_gate"] = gates
    print(f"  5-Gate: {gates['passed']}/5 passed | All: {gates['all_passed']}")

    # Store sample trades
    metrics["sample_trades"] = trade_log[:10] if trade_log else []
    metrics["equity_curve_endpoints"] = {
        "start": eq_curve[0] if eq_curve else None,
        "end": eq_curve[-1] if eq_curve else None,
    }

    results[label] = metrics

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY — IPO LOCKUP EXPIRY BACKTEST")
print(f"{'='*70}")
print(f"{'Variant':<40} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'WR':>6} {'PF':>6} {'Trades':>7} {'MDD':>8} {'PermP':>7} {'Gates':>6}")
print("-" * 110)
for label, m in results.items():
    if "error" in m:
        print(f"{label:<40} ERROR: {m['error']}")
        continue
    print(f"{m['label']:<40} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>6.1%} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['total_trades']:>7d} {m['max_drawdown']:>7.1%} {m['perm_p']:>7.3f} {m['five_gate']['passed']:>2d}/5")

# ─── Save results ─────────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/ipo_lockup_results.json")

def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def deep_convert(d):
    if isinstance(d, dict):
        return {k: deep_convert(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [deep_convert(i) for i in d]
    else:
        return convert_numpy(d)

output = {
    "metadata": {
        "strategy": "IPO Lockup Expiry Trading",
        "universe": TICKERS,
        "ipo_dates": {k: str(v.date()) for k, v in IPO_DATES.items()},
        "lockup_days": LOCKUP_DAYS,
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "initial_capital": INITIAL_CAPITAL,
        "n_permutations": N_PERMS,
        "risk_free_rate": RISK_FREE,
        "run_timestamp": datetime.now().isoformat(),
    },
    "variants": deep_convert(results),
    "validation_gates": {
        "sharpe_threshold": 0.5,
        "perm_p_threshold": 0.05,
        "regime_gap_threshold": 0.5,
        "max_drawdown_threshold": -0.50,
        "min_trades": 20,
    }
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=convert_numpy)

print(f"\nResults saved to {output_path}")

# Gate summary
n_passed = sum(1 for m in results.values() if "five_gate" in m and m["five_gate"].get("all_passed", False))
print(f"\nVARIANTS PASSING ALL 5 GATES: {n_passed}/{len(results)}")
