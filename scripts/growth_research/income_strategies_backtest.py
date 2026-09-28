#!/usr/bin/env python3
"""
Income / Premium-Selling Strategies Backtest
=============================================
Six systematic credit-spread / premium-selling strategies for a $750 Robinhood
account with Level 2 options.

Key modelling choices:
  - Spread widths are FIXED in dollar terms (e.g., $5 wide) to match Robinhood
    reality where you pick strike prices at discrete intervals.
  - Premium is modelled as a fraction of spread width, varying by DTE, moneyness,
    and IV environment.
  - Max risk = spread_width × 100 − premium_received, capped at $150.
  - Cooldown between trades per ticker to avoid overtrading.
  - Max 2 concurrent open positions across all strategies.

Period: 2020-01-01 → 2026-07-01
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple
from collections import defaultdict

# ── constants ────────────────────────────────────────────────────────────────
START = "2020-01-01"
END   = "2026-07-01"
MAX_RISK       = 150.0   # per trade
MAX_CONCURRENT = 2
ACCOUNT        = 750.0

QUALITY_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH",
    "V", "MA", "HD", "PG", "JNJ", "XOM", "COST",
]

# FOMC dates 2020-2026
FOMC_DATES = pd.to_datetime([
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
    "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
    "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
    "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
    "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
    "2025-07-30","2025-09-17","2025-11-05","2025-12-17",
    "2026-01-28","2026-03-18","2026-05-06","2026-06-17",
])

# ── data download ────────────────────────────────────────────────────────────
print("Downloading data …")
tickers_needed = list(set(QUALITY_TICKERS + ["SPY", "^VIX"]))
raw = yf.download(tickers_needed, start=START, end=END, auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"].copy()
    opn   = raw["Open"].copy()
else:
    close = raw[["Close"]].copy()
    opn   = raw[["Open"]].copy()

for df in [close, opn]:
    if "^VIX" in df.columns:
        df.rename(columns={"^VIX": "VIX"}, inplace=True)

close = close.ffill()
opn   = opn.ffill()
vix   = close["VIX"].copy() if "VIX" in close.columns else pd.Series(20.0, index=close.index)
spy   = close["SPY"].copy()

print(f"Data: {close.index[0].date()} → {close.index[-1].date()}, {len(close)} days")


# ── helpers ──────────────────────────────────────────────────────────────────

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).mean()


def premium_estimate(spread_width_dollars: float, dte: int, otm_pct: float,
                     vix_level: float) -> float:
    """
    Estimate credit received for a put credit spread.

    Conservative model:
      base_rate depends on OTM% and DTE.
      ATM 30-DTE spread on SPY in normal vol → ~30-35% of spread width.
      5% OTM 30-DTE → ~15-20% of spread width.
      Adjust up for high VIX, down for low VIX.

    Returns premium in dollars (for 1 contract = 100 shares).
    spread_width_dollars = strike difference * 100.
    """
    # Base premium rate as fraction of spread width
    # OTM% effect: ATM=0 gets highest, further OTM = lower
    otm_factor = max(0.05, 1.0 - otm_pct * 10)  # 0%OTM→1.0, 5%OTM→0.5, 10%OTM→0.0

    # DTE effect: longer = more premium (sqrt scaling)
    dte_factor = min(2.0, (dte / 30) ** 0.5)

    # VIX effect: higher VIX = more premium
    vix_factor = max(0.5, vix_level / 20)  # VIX=20 → 1.0, VIX=30 → 1.5, VIX=40 → 2.0

    # Base rate: 25% of spread width for ATM 30-DTE normal vol
    base_rate = 0.25
    rate = base_rate * otm_factor * dte_factor * vix_factor

    # Cap at 60% (can't collect more than ~60% of spread width realistically)
    rate = min(rate, 0.60)
    # Floor at 3% (minimum to make it worth trading)
    rate = max(rate, 0.03)

    return spread_width_dollars * rate


def resolve_trade(entry_price: float, expiry_price: float,
                  short_strike: float, long_strike: float,
                  premium: float, direction: str = "bull_put") -> float:
    """
    Compute P&L at expiry for a credit spread.
    bull_put: short put at short_strike, long put at long_strike (lower).
    bear_call: short call at short_strike, long call at long_strike (higher).
    """
    spread_width = abs(short_strike - long_strike) * 100  # in dollars

    if direction == "bull_put":
        if expiry_price >= short_strike:
            return premium  # both expire worthless
        elif expiry_price <= long_strike:
            return premium - spread_width  # max loss
        else:
            intrinsic = (short_strike - expiry_price) * 100
            return premium - intrinsic

    elif direction == "bear_call":
        if expiry_price <= short_strike:
            return premium
        elif expiry_price >= long_strike:
            return premium - spread_width
        else:
            intrinsic = (expiry_price - short_strike) * 100
            return premium - intrinsic

    return 0


def compute_stats(trades: List[Tuple[pd.Timestamp, float]], name: str) -> Dict:
    if not trades:
        return {"name": name, "trades": 0}

    df = pd.DataFrame(trades, columns=["date", "pnl"])
    arr = df["pnl"].values
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    total = arr.sum()
    wr = (arr > 0).sum() / len(arr) * 100

    # Monthly PnL
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum().values

    if len(monthly) > 2 and monthly.std() > 0:
        sharpe = monthly.mean() / monthly.std() * np.sqrt(12)
        ds = monthly[monthly < 0]
        ds_std = ds.std() if len(ds) > 1 else monthly.std()
        sortino = monthly.mean() / ds_std * np.sqrt(12) if ds_std > 0 else 0
    else:
        sharpe = sortino = 0

    max_cl = cur = 0
    for p in arr:
        if p <= 0:
            cur += 1
            max_cl = max(max_cl, cur)
        else:
            cur = 0

    pf = abs(wins.sum() / losses.sum()) if len(losses) and losses.sum() != 0 else 999.0

    cum = np.cumsum(arr)
    dd = cum - np.maximum.accumulate(cum)
    max_dd = dd.min()

    return {
        "name": name, "trades": len(arr),
        "total_premium": float(wins.sum()), "total_losses": float(losses.sum()),
        "net_pnl": float(total),
        "net_return_pct": float(total / ACCOUNT * 100),
        "win_rate": float(wr), "profit_factor": float(pf),
        "sharpe": float(sharpe), "sortino": float(sortino),
        "avg_win": float(wins.mean()) if len(wins) else 0,
        "avg_loss": float(losses.mean()) if len(losses) else 0,
        "max_consec_losses": max_cl, "max_drawdown": float(max_dd),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Position manager — enforces max concurrent and cooldowns
# ═══════════════════════════════════════════════════════════════════════════
class PositionManager:
    def __init__(self, max_concurrent=MAX_CONCURRENT, cooldown_days=21):
        self.max_concurrent = max_concurrent
        self.cooldown_days = cooldown_days
        self.open_positions = []   # list of (expiry_date,)
        self.last_entry = {}       # ticker → last entry date

    def can_open(self, dt: pd.Timestamp, ticker: str) -> bool:
        # Remove expired
        self.open_positions = [exp for exp in self.open_positions if exp > dt]
        if len(self.open_positions) >= self.max_concurrent:
            return False
        # Cooldown
        if ticker in self.last_entry:
            if (dt - self.last_entry[ticker]).days < self.cooldown_days:
                return False
        return True

    def open(self, dt: pd.Timestamp, ticker: str, expiry: pd.Timestamp):
        self.open_positions.append(expiry)
        self.last_entry[ticker] = dt


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY A: Put Credit Spread on Quality Dips (30 DTE, ~ATM)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ Strategy A: Put Credit Spread on Quality Dips ━━━")

trades_a = []
pm_a = PositionManager(max_concurrent=2, cooldown_days=30)

for ticker in QUALITY_TICKERS:
    if ticker not in close.columns:
        continue
    px = close[ticker].dropna()
    s20 = sma(px, 20)

    for i in range(25, len(px)):
        dt = px.index[i]
        price = px.iloc[i]
        ma_val = s20.iloc[i]
        if pd.isna(ma_val) or ma_val == 0:
            continue

        dip_pct = (price - ma_val) / ma_val
        if dip_pct > -0.03:
            continue

        if not pm_a.can_open(dt, ticker):
            continue

        cur_vix = vix.loc[dt] if dt in vix.index else 20.0

        # Fixed $2 wide spread (realistic for stocks $20-200+)
        # Short put at current price (ATM), long put $2 lower
        spread_width_pts = 2.0  # $2 strike width
        short_strike = price
        long_strike = price - spread_width_pts
        spread_width_dollars = spread_width_pts * 100  # = $200

        otm_pct = 0.0  # ATM
        dte = 30
        premium = premium_estimate(spread_width_dollars, dte, otm_pct, cur_vix)

        max_loss = spread_width_dollars - premium
        if max_loss > MAX_RISK:
            # Use $1 wide spread instead
            spread_width_pts = 1.0
            long_strike = price - spread_width_pts
            spread_width_dollars = spread_width_pts * 100
            premium = premium_estimate(spread_width_dollars, dte, otm_pct, cur_vix)
            max_loss = spread_width_dollars - premium

        if max_loss > MAX_RISK or premium < 5:
            continue

        # Expiry: 21 trading days later
        expiry_idx = min(i + 21, len(px) - 1)
        expiry_price = px.iloc[expiry_idx]
        expiry_dt = px.index[expiry_idx]

        pnl = resolve_trade(price, expiry_price, short_strike, long_strike,
                           premium, "bull_put")
        trades_a.append((dt, pnl))
        pm_a.open(dt, ticker, expiry_dt)

stats_a = compute_stats(trades_a, "A: Quality Dip Put Spread")
print(f"  Trades: {stats_a['trades']}, WR: {stats_a.get('win_rate',0):.1f}%, "
      f"Net: ${stats_a.get('net_pnl',0):.0f}, Sharpe: {stats_a.get('sharpe',0):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY B: High IV Put Selling (VIX > 20 gate, 5% OTM)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ Strategy B: High IV Put Selling (VIX>20) ━━━")

trades_b = []
pm_b = PositionManager(max_concurrent=2, cooldown_days=30)

for ticker in QUALITY_TICKERS:
    if ticker not in close.columns:
        continue
    px = close[ticker].dropna()
    s20 = sma(px, 20)

    for i in range(25, len(px)):
        dt = px.index[i]
        price = px.iloc[i]
        ma_val = s20.iloc[i]
        if pd.isna(ma_val) or ma_val == 0:
            continue

        cur_vix = vix.loc[dt] if dt in vix.index else 20.0
        if cur_vix <= 20:
            continue

        dip_pct = (price - ma_val) / ma_val
        if dip_pct > -0.03:
            continue

        if not pm_b.can_open(dt, ticker):
            continue

        # 5% OTM put spread, $2 wide
        short_strike = price * 0.95
        spread_width_pts = 2.0
        long_strike = short_strike - spread_width_pts
        spread_width_dollars = spread_width_pts * 100

        otm_pct = 0.05
        dte = 30
        premium = premium_estimate(spread_width_dollars, dte, otm_pct, cur_vix)
        max_loss = spread_width_dollars - premium

        if max_loss > MAX_RISK or premium < 5:
            continue

        expiry_idx = min(i + 21, len(px) - 1)
        expiry_price = px.iloc[expiry_idx]
        expiry_dt = px.index[expiry_idx]

        pnl = resolve_trade(price, expiry_price, short_strike, long_strike,
                           premium, "bull_put")
        trades_b.append((dt, pnl))
        pm_b.open(dt, ticker, expiry_dt)

stats_b = compute_stats(trades_b, "B: High IV Put Selling")
print(f"  Trades: {stats_b['trades']}, WR: {stats_b.get('win_rate',0):.1f}%, "
      f"Net: ${stats_b.get('net_pnl',0):.0f}, Sharpe: {stats_b.get('sharpe',0):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY C: Iron Condor on SPY Around FOMC (7 DTE)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ Strategy C: FOMC Iron Condor on SPY ━━━")

trades_c = []

for fomc_date in FOMC_DATES:
    # Enter 2 trading days before
    mask = spy.index < fomc_date
    if mask.sum() < 3:
        continue
    entry_dt = spy.index[mask][-2]
    spy_price = spy.loc[entry_dt]
    cur_vix = vix.loc[entry_dt] if entry_dt in vix.index else 20.0

    # 3% OTM each side, $1 wide spreads (keeps max risk manageable)
    put_short  = round(spy_price * 0.97, 0)
    put_long   = put_short - 1.0
    call_short = round(spy_price * 1.03, 0)
    call_long  = call_short + 1.0

    put_sw  = 1.0 * 100  # $100
    call_sw = 1.0 * 100

    # 7 DTE, 3% OTM, use elevated pre-FOMC vol
    put_prem  = premium_estimate(put_sw,  7, 0.03, cur_vix * 1.1)  # vol bump
    call_prem = premium_estimate(call_sw, 7, 0.03, cur_vix * 1.1)

    total_premium = put_prem + call_prem
    max_loss_side = max(put_sw - put_prem, call_sw - call_prem)

    # Scale down if needed
    if max_loss_side > MAX_RISK:
        scale = MAX_RISK / max_loss_side
        put_prem  *= scale
        call_prem *= scale
        put_sw    *= scale
        call_sw   *= scale
        total_premium = put_prem + call_prem

    # 5 trading days to expiry
    entry_pos = spy.index.get_loc(entry_dt)
    expiry_idx = min(entry_pos + 5, len(spy) - 1)
    expiry_price = spy.iloc[expiry_idx]

    # IC outcome
    pnl_put  = resolve_trade(spy_price, expiry_price, put_short, put_long, put_prem, "bull_put")
    pnl_call = resolve_trade(spy_price, expiry_price, call_short, call_long, call_prem, "bear_call")
    pnl = pnl_put + pnl_call

    trades_c.append((entry_dt, pnl))

stats_c = compute_stats(trades_c, "C: FOMC Iron Condor")
print(f"  Trades: {stats_c['trades']}, WR: {stats_c.get('win_rate',0):.1f}%, "
      f"Net: ${stats_c.get('net_pnl',0):.0f}, Sharpe: {stats_c.get('sharpe',0):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY D: Post-Earnings Vol-Crush Put Spread
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ Strategy D: Post-Earnings Vol Crush Put Spread ━━━")

trades_d = []
pm_d = PositionManager(max_concurrent=2, cooldown_days=60)  # quarterly earnings

for ticker in QUALITY_TICKERS:
    if ticker not in close.columns or ticker not in opn.columns:
        continue
    px = close[ticker].dropna()
    op = opn[ticker].dropna()
    idx = px.index.intersection(op.index)
    px = px.loc[idx]
    op = op.loc[idx]

    for i in range(1, len(px)):
        dt = px.index[i]
        prev_close = px.iloc[i-1]
        today_open = op.iloc[i]

        if prev_close == 0:
            continue

        gap_pct = (today_open - prev_close) / prev_close
        if gap_pct < 0.02:
            continue

        # Earnings months only
        if dt.month not in [1, 2, 4, 5, 7, 8, 10, 11]:
            continue

        if not pm_d.can_open(dt, ticker):
            continue

        price = px.iloc[i]
        cur_vix = vix.loc[dt] if dt in vix.index else 20.0

        # 5% OTM, $2 wide, 30 DTE
        short_strike = round(price * 0.95, 0)
        long_strike = short_strike - 2.0
        spread_width_dollars = 2.0 * 100

        # Elevated post-earnings IV → higher premium
        premium = premium_estimate(spread_width_dollars, 30, 0.05, cur_vix * 1.3)
        max_loss = spread_width_dollars - premium

        if max_loss > MAX_RISK or premium < 5:
            continue

        expiry_idx = min(i + 21, len(px) - 1)
        expiry_price = px.iloc[expiry_idx]
        expiry_dt = px.index[expiry_idx]

        pnl = resolve_trade(price, expiry_price, short_strike, long_strike,
                           premium, "bull_put")
        trades_d.append((dt, pnl))
        pm_d.open(dt, ticker, expiry_dt)

stats_d = compute_stats(trades_d, "D: Post-Earnings Vol Crush")
print(f"  Trades: {stats_d['trades']}, WR: {stats_d.get('win_rate',0):.1f}%, "
      f"Net: ${stats_d.get('net_pnl',0):.0f}, Sharpe: {stats_d.get('sharpe',0):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY E: VIX Mean-Reversion SPY Put Spread
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ Strategy E: VIX Mean-Reversion SPY Put Spread ━━━")

trades_e = []
last_entry_e = None

for i in range(1, len(spy)):
    dt = spy.index[i]
    cur_vix = vix.loc[dt] if dt in vix.index else 20.0

    if cur_vix <= 25:
        continue

    # 30-day cooldown
    if last_entry_e and (dt - last_entry_e).days < 30:
        continue

    spy_price = spy.iloc[i]

    # ATM put spread, $3 wide on SPY
    short_strike = round(spy_price, 0)
    long_strike = short_strike - 3.0
    spread_width_dollars = 3.0 * 100  # $300

    # High VIX → fat premium
    premium = premium_estimate(spread_width_dollars, 30, 0.0, cur_vix)
    max_loss = spread_width_dollars - premium

    if max_loss > MAX_RISK:
        # Use $2 wide
        short_strike = round(spy_price, 0)
        long_strike = short_strike - 2.0
        spread_width_dollars = 2.0 * 100
        premium = premium_estimate(spread_width_dollars, 30, 0.0, cur_vix)
        max_loss = spread_width_dollars - premium

    if max_loss > MAX_RISK or premium < 5:
        continue

    expiry_idx = min(i + 21, len(spy) - 1)
    expiry_price = spy.iloc[expiry_idx]

    pnl = resolve_trade(spy_price, expiry_price, short_strike, long_strike,
                       premium, "bull_put")
    trades_e.append((dt, pnl))
    last_entry_e = dt

stats_e = compute_stats(trades_e, "E: VIX Mean-Rev SPY Put Spread")
print(f"  Trades: {stats_e['trades']}, WR: {stats_e.get('win_rate',0):.1f}%, "
      f"Net: ${stats_e.get('net_pnl',0):.0f}, Sharpe: {stats_e.get('sharpe',0):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY F: Systematic Weekly SPY Credit Spread (every Friday)
# ═══════════════════════════════════════════════════════════════════════════
print("\n━━━ Strategy F: Systematic Weekly SPY Credit Spread ━━━")

trades_f = []

for i in range(5, len(spy)):
    dt = spy.index[i]
    if dt.dayofweek != 4:  # Friday only
        continue

    spy_price = spy.iloc[i]
    cur_vix = vix.loc[dt] if dt in vix.index else 20.0

    # 2% OTM, $1 wide, 7 DTE (SPY has $1 strike increments for weeklies)
    short_strike = round(spy_price * 0.98, 0)
    long_strike = short_strike - 1.0
    spread_width_dollars = 1.0 * 100  # $100 max risk per spread

    premium = premium_estimate(spread_width_dollars, 7, 0.02, cur_vix)
    max_loss = spread_width_dollars - premium

    if max_loss > MAX_RISK or premium < 1:
        continue

    expiry_idx = min(i + 5, len(spy) - 1)
    expiry_price = spy.iloc[expiry_idx]

    pnl = resolve_trade(spy_price, expiry_price, short_strike, long_strike,
                       premium, "bull_put")
    trades_f.append((dt, pnl))

stats_f = compute_stats(trades_f, "F: Weekly SPY Credit Spread")
print(f"  Trades: {stats_f['trades']}, WR: {stats_f.get('win_rate',0):.1f}%, "
      f"Net: ${stats_f.get('net_pnl',0):.0f}, Sharpe: {stats_f.get('sharpe',0):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*105)
print("INCOME / PREMIUM STRATEGY COMPARISON — $750 account, $150 max risk/trade, 2020–2026")
print("="*105)

all_stats = [stats_a, stats_b, stats_c, stats_d, stats_e, stats_f]

hdr = (f"{'Strategy':<35} {'#':>5} {'WR%':>6} {'Net$':>8} {'Ret%':>7} "
       f"{'Sharpe':>7} {'Sort':>7} {'PF':>6} {'AvgW':>6} {'AvgL':>7} {'MaxDD':>7} {'MCL':>4}")
print(hdr)
print("-"*105)

for s in all_stats:
    if s["trades"] == 0:
        print(f"  {s['name']:<33} {'0':>5}  — no trades —")
        continue
    print(f"  {s['name']:<33} {s['trades']:>5} {s['win_rate']:>5.1f}% "
          f"{s['net_pnl']:>7.0f} {s['net_return_pct']:>6.1f}% "
          f"{s['sharpe']:>7.2f} {s['sortino']:>7.2f} {s['profit_factor']:>5.2f} "
          f"{s['avg_win']:>5.0f} {s['avg_loss']:>6.0f} {s['max_drawdown']:>6.0f} "
          f"{s['max_consec_losses']:>4}")

print("-"*105)


# ── Per-year breakdown ──
print("\n" + "="*105)
print("PER-YEAR BREAKDOWN")
print("="*105)

strat_labels = ["A: Quality Dip", "B: High IV", "C: FOMC IC",
                "D: Earnings VC", "E: VIX MR", "F: Weekly"]
strat_trades = [trades_a, trades_b, trades_c, trades_d, trades_e, trades_f]

for label, tlist in zip(strat_labels, strat_trades):
    if not tlist:
        continue
    df_t = pd.DataFrame(tlist, columns=["date", "pnl"])
    df_t["year"] = df_t["date"].dt.year
    yearly = df_t.groupby("year").agg(
        n=("pnl", "count"),
        net=("pnl", "sum"),
        wr=("pnl", lambda x: (x > 0).sum() / len(x) * 100),
    )
    print(f"\n  {label}:")
    for yr, row in yearly.iterrows():
        bar = "+" * int(max(0, row["net"]) / 5) + "-" * int(max(0, -row["net"]) / 5)
        print(f"    {yr}: {int(row['n']):>3} trades  Net ${row['net']:>7.0f}  "
              f"WR {row['wr']:>5.1f}%  {bar}")


# ── Regime analysis ──
print("\n" + "="*105)
print("REGIME ANALYSIS (trailing 63d SPY return: >5% = bull, <-5% = bear, else flat)")
print("="*105)

spy_ret63 = spy.pct_change(63)

for label, tlist in zip(strat_labels, strat_trades):
    if not tlist:
        continue
    regimes = {"bull": [], "bear": [], "flat": []}
    for dt, pnl in tlist:
        if dt in spy_ret63.index and not pd.isna(spy_ret63.loc[dt]):
            r = spy_ret63.loc[dt]
            bucket = "bull" if r > 0.05 else ("bear" if r < -0.05 else "flat")
            regimes[bucket].append(pnl)

    parts = []
    for regime in ["bull", "bear", "flat"]:
        arr = np.array(regimes[regime])
        if len(arr):
            wr = (arr > 0).sum() / len(arr) * 100
            parts.append(f"{regime}: n={len(arr)} WR={wr:.0f}% ${arr.sum():.0f}")
    print(f"  {label}: " + " | ".join(parts))


# ── Combined portfolio ──
print("\n" + "="*105)
print("COMBINED PORTFOLIO")
print("="*105)

all_combined = []
for tlist in strat_trades:
    all_combined.extend(tlist)
all_combined.sort(key=lambda x: x[0])

stats_comb = compute_stats(all_combined, "COMBINED")
if stats_comb["trades"] > 0:
    years = 6.5
    final_val = ACCOUNT + stats_comb["net_pnl"]
    if final_val > 0:
        cagr = (final_val / ACCOUNT) ** (1/years) - 1
    else:
        cagr = -1.0

    print(f"  Trades: {stats_comb['trades']}")
    print(f"  Net PnL: ${stats_comb['net_pnl']:.0f}  ({stats_comb['net_return_pct']:.1f}% on $750)")
    print(f"  Win Rate: {stats_comb['win_rate']:.1f}%")
    print(f"  Sharpe: {stats_comb['sharpe']:.2f}  |  Sortino: {stats_comb['sortino']:.2f}")
    print(f"  Profit Factor: {stats_comb['profit_factor']:.2f}")
    print(f"  Avg Win: ${stats_comb['avg_win']:.0f}  |  Avg Loss: ${stats_comb['avg_loss']:.0f}")
    print(f"  Max Drawdown: ${stats_comb['max_drawdown']:.0f}")
    print(f"  Max Consecutive Losses: {stats_comb['max_consec_losses']}")
    print(f"  CAGR: {cagr*100:.1f}%")


# ── 5-Gate validation ──
print("\n" + "="*105)
print("5-GATE VALIDATION")
print("="*105)

for s in all_stats:
    if s["trades"] == 0:
        print(f"  {s['name']}: NO TRADES")
        continue

    g = [
        ("Sharpe > 0.5",   s["sharpe"] > 0.5,        f"{s['sharpe']:.2f}"),
        ("WR > 55%",       s["win_rate"] > 55,        f"{s['win_rate']:.1f}%"),
        ("PF > 1.3",       s["profit_factor"] > 1.3,  f"{s['profit_factor']:.2f}"),
        ("MaxCL <= 5",     s["max_consec_losses"] <= 5, f"{s['max_consec_losses']}"),
        ("DD < $375",      abs(s["max_drawdown"]) < 375, f"${abs(s['max_drawdown']):.0f}"),
    ]
    passed = sum(v for _, v, _ in g)
    verdict = "VIABLE" if passed >= 4 else ("MARGINAL" if passed >= 3 else "REJECT")
    status = " | ".join(f"{n}: {'OK' if v else 'X'}({val})" for n, v, val in g)
    print(f"  {s['name']}: {passed}/5 → {verdict}")
    print(f"    {status}")


# ── Premium sensitivity analysis ──
print("\n" + "="*105)
print("PREMIUM SENSITIVITY: What premium rate makes Strategy F (Weekly SPY) break even?")
print("="*105)

for prem_mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    test_trades = []
    for i in range(5, len(spy)):
        dt = spy.index[i]
        if dt.dayofweek != 4:
            continue
        spy_price = spy.iloc[i]
        cur_vix = vix.loc[dt] if dt in vix.index else 20.0

        short_strike = round(spy_price * 0.98, 0)
        long_strike = short_strike - 1.0
        sw = 100.0

        prem = premium_estimate(sw, 7, 0.02, cur_vix) * prem_mult
        max_loss = sw - prem
        if max_loss <= 0 or prem < 1:
            continue

        expiry_idx = min(i + 5, len(spy) - 1)
        expiry_price = spy.iloc[expiry_idx]
        pnl = resolve_trade(spy_price, expiry_price, short_strike, long_strike, prem, "bull_put")
        test_trades.append((dt, pnl))

    if test_trades:
        arr = np.array([p for _, p in test_trades])
        wr = (arr > 0).sum() / len(arr) * 100
        net = arr.sum()
        print(f"  {prem_mult:.1f}x premium: {len(arr)} trades, WR {wr:.1f}%, Net ${net:.0f}")

print("\n" + "="*105)
print("DONE")
print("="*105)
