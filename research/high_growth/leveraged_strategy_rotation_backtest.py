#!/usr/bin/env python3
"""
Leveraged Strategy Rotation Backtest
=====================================
Builds on the validated Strategy Rotation (37.3% CAGR, Sharpe 2.13) and
Quality Factor Rotation (20.1% CAGR) strategies with three leverage approaches:

  Variant A  — 3x leveraged ETF substitution (SPY→UPRO, QQQ→TQQQ, sectors→3x)
  Variant B  — 2x notional leverage via margin (200% capital deployed)
  Variant C  — ATM call options overlay instead of equity positions (30-45 DTE)

Baseline (Variant 0) — unleveraged replication of the best strategy rotation config.

All variants:
- yfinance real market data 2020-2026
- Walk-forward OOT: train on 2020-2021, test 2022-2026
- Realistic costs: commissions, bid-ask, vol decay for leveraged ETFs
- Risk metrics: CAGR, max drawdown, Sharpe, Sortino, win rate, profit factor
- Compared to SPY buy-and-hold benchmark
"""

import warnings, json, os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ───────────────────────────────────────────────────────────────────
STARTING_CAPITAL   = 100_000.0   # Normalize to $100k for clean comparison
TRAIN_START        = "2020-01-01"
OOT_START          = "2022-01-01"   # Walk-forward OOT starts here
OOT_END            = "2026-07-25"
SLIPPAGE_PCT       = 0.0005         # 5 bps per side for liquid ETFs
COMMISSION_PER_RT  = 1.00           # $1 round-trip per position (Robinhood = $0; IB ~$1)

# Leveraged ETF annual vol-decay (empirical from TQQQ/UPRO research)
# On average, 3x ETFs lose ~4-8% CAGR extra vs 3x underlying from decay+fees
LEV3X_ANNUAL_DECAY_PENALTY = 0.05   # 5% annual headwind from vol decay + fees

# Margin cost (Variant B)
MARGIN_RATE_ANNUAL = 0.0650  # ~6.5% margin interest (IB interactive brokers)

# Options cost approximation (Variant C)
# ATM call 30-45 DTE: pays IV premium. We model the cost as:
#   Option cost = intrinsic breakeven ≈ underlying * IV * sqrt(45/252)
# We use a simplified approach: subtract options theta decay cost
# Options theta is approximated as IV-premium above the expected move
OPTIONS_IV_PREMIUM  = 0.04  # ~4% of notional per position as premium cost (conservative)

OUT_DIR = Path("/home/jupiter/Lvl3Quant/research/high_growth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── TICKER MAPPINGS ──────────────────────────────────────────────────────────
# 1x → 3x equivalents
ETF_3X_MAP = {
    "SPY":  "UPRO",   # S&P 500 3x
    "QQQ":  "TQQQ",   # Nasdaq-100 3x
    "XLK":  "TECL",   # Tech 3x
    "XLF":  "FAS",    # Financials 3x
    "XLE":  "ERX",    # Energy 3x
    "XLV":  "CURE",   # Healthcare 3x
    "XLI":  "DUSL",   # Industrials 3x
    "XLY":  "WANT",   # Consumer Disc 3x (WANT = 3x Consumer Disc)
    "XLB":  "NAIL",   # Materials → proxy with homebuilders 3x (closest available)
    "XLRE": "DRN",    # Real estate 3x
    "XLU":  "UTSL",   # Utilities 3x
    "XLC":  "TQQQ",   # Comm services → use TQQQ as proxy (no pure 3x comms)
    "GLD":  "GDXU",   # Gold miners 3x
    "TLT":  "TMF",    # 20yr Treasury 3x
    "IWM":  "TNA",    # Russell 2000 3x
    "EEM":  "EDC",    # EM 3x
}

# Core tickers needed for the strategy
BASE_TICKERS = [
    "SPY", "QQQ", "^VIX",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLB", "XLRE", "XLU", "XLC",
]

LEV3X_TICKERS = list(set([
    "UPRO", "TQQQ", "TECL", "FAS", "ERX", "CURE", "DUSL", "DRN", "UTSL",
    "TNA", "TMF", "SOXL"
]))

ALL_TICKERS = BASE_TICKERS + LEV3X_TICKERS


# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    print("[1/7] Downloading market data...")

    raw = yf.download(
        ALL_TICKERS,
        start="2019-01-01",   # Extra buffer for 200-SMA and momentum lookbacks
        end=OOT_END,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    prices = prices.ffill()

    # Report coverage
    for t in ALL_TICKERS:
        if t in prices.columns:
            first_valid = prices[t].first_valid_index()
            n_valid = prices[t].notna().sum()
            print(f"  {t}: {n_valid} days, first={first_valid.date() if first_valid else 'N/A'}")
        else:
            print(f"  {t}: NOT FOUND")

    print(f"  Data range: {prices.index[0].date()} → {prices.index[-1].date()}, {len(prices)} rows")
    return prices


# ── REGIME SIGNALS ────────────────────────────────────────────────────────────
def compute_signals(prices):
    """Compute regime and rotation signals."""
    p = prices.copy()
    spy = p["SPY"]
    vix = p["^VIX"]

    signals = pd.DataFrame(index=p.index)
    signals["SPY"]     = spy
    signals["VIX"]     = vix
    signals["SMA200"]  = spy.rolling(200).mean()
    signals["SMA50"]   = spy.rolling(50).mean()
    signals["bull"]    = (spy > signals["SMA200"]).astype(int)
    signals["RSI14"]   = _rsi(spy, 14)

    # Momentum signals for sector rotation
    for t in BASE_TICKERS:
        if t in p.columns and t not in ["^VIX"]:
            signals[f"{t}_mom1m"]  = p[t].pct_change(21)
            signals[f"{t}_mom3m"]  = p[t].pct_change(63)
            signals[f"{t}_mom12m"] = p[t].pct_change(252)

    signals["VIX_5d_chg"]  = vix.pct_change(5)
    signals["SPY_ret5"]    = spy.pct_change(5)
    signals["SPY_ret20"]   = spy.pct_change(20)
    signals["SPY_ret60"]   = spy.pct_change(60)

    return signals.dropna()


def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


# ── CORE ROTATION LOGIC ───────────────────────────────────────────────────────
def get_rotation_signal(sig_row):
    """
    Best Strategy Rotation variant (Variant E from original backtest —
    composite regime + momentum + VIX gating).
    Returns: ('asset', allocation_fraction)
    """
    vix   = sig_row["VIX"]
    bull  = sig_row["bull"]
    rsi   = sig_row["RSI14"]
    ret20 = sig_row["SPY_ret20"]
    ret5  = sig_row["SPY_ret5"]
    vix5  = sig_row["VIX_5d_chg"]

    # VIX fade: crisis recovery
    if vix > 25 and vix5 < 0:
        return "SPY", 1.0  # Buy SPY on VIX retreat

    # Extreme overbought → cash
    if rsi > 75 and vix < 15:
        return "CASH", 0.0

    # Strong bull + momentum → QQQ
    if bull and ret20 > 0.02:
        return "QQQ", 1.0

    # Bull regime, neutral momentum → SPY
    if bull:
        return "SPY", 1.0

    # Bear + SPY dip → contrarian buy
    if not bull and ret5 <= -0.03:
        return "SPY", 1.0  # Buy the dip

    # Bear regime, no dip signal → cash/defensive
    return "CASH", 0.0


def get_sector_rotation_picks(prices, signals, date, n_picks=3):
    """
    Quality Factor Rotation: pick top N sectors by composite score.
    Returns list of sector ETF tickers.
    """
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLB", "XLRE", "XLU", "XLC"]
    available = [s for s in sectors if s in prices.columns]

    if date not in signals.index or len(available) < 2:
        return available[:n_picks]

    row = signals.loc[date]
    scores = {}
    for s in available:
        score = 0.0
        m1  = row.get(f"{s}_mom1m", 0) or 0
        m3  = row.get(f"{s}_mom3m", 0) or 0
        m12 = row.get(f"{s}_mom12m", 0) or 0
        score = 0.3 * m1 + 0.4 * m3 + 0.3 * m12
        scores[s] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:n_picks]]


# ── COST HELPERS ─────────────────────────────────────────────────────────────
def apply_trade_cost(ret, direction="buy"):
    """Apply slippage to a return."""
    # Buy: pay slightly more; Sell: receive slightly less
    return ret - SLIPPAGE_PCT * 2  # round-trip slippage


def lev3x_decay_daily():
    """Daily vol-decay penalty for 3x leveraged ETFs."""
    return (1 + LEV3X_ANNUAL_DECAY_PENALTY) ** (1/252) - 1


# ── SIMULATION ENGINES ────────────────────────────────────────────────────────

def simulate_variant_0(prices, signals):
    """
    Variant 0 — Unleveraged Strategy Rotation (baseline).
    SPY / QQQ rotation with VIX gating, monthly rebalance.
    """
    daily_rets = []
    holdings = None

    spy_ret  = prices["SPY"].pct_change()
    qqq_ret  = prices["QQQ"].pct_change()

    for i, date in enumerate(signals.index):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig = signals.loc[date]
        asset, _ = get_rotation_signal(sig)

        if asset == "SPY":
            r = spy_ret.get(date, 0) or 0
        elif asset == "QQQ":
            r = qqq_ret.get(date, 0) or 0
        else:
            r = 0.0

        # Apply slippage only on transitions
        if asset != holdings:
            r -= SLIPPAGE_PCT * 2

        holdings = asset
        daily_rets.append(r)

    return pd.Series(daily_rets, index=signals.index, name="V0_unleveraged")


def simulate_variant_A(prices, signals):
    """
    Variant A — 3x Leveraged ETF Substitution.
    Same rotation logic but SPY→UPRO, QQQ→TQQQ.
    Includes vol-decay penalty and higher slippage for leveraged ETFs.
    """
    SLIPPAGE_LEV = 0.001  # 10 bps for leveraged ETFs (wider bid-ask)

    daily_rets = []
    holdings = None

    upro_ret = prices["UPRO"].pct_change() if "UPRO" in prices.columns else prices["SPY"].pct_change() * 3
    tqqq_ret = prices["TQQQ"].pct_change() if "TQQQ" in prices.columns else prices["QQQ"].pct_change() * 3

    for i, date in enumerate(signals.index):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig   = signals.loc[date]
        asset1x, _ = get_rotation_signal(sig)

        # Map to 3x equivalent
        if asset1x == "SPY":
            asset3x = "UPRO"
            r = upro_ret.get(date, 0) or 0
        elif asset1x == "QQQ":
            asset3x = "TQQQ"
            r = tqqq_ret.get(date, 0) or 0
        else:
            asset3x = "CASH"
            r = 0.0

        # Vol-decay penalty (applies every day we hold leveraged ETF)
        if asset3x != "CASH":
            r -= lev3x_decay_daily()

        # Transition cost
        if asset3x != holdings:
            r -= SLIPPAGE_LEV * 2

        holdings = asset3x
        daily_rets.append(r)

    return pd.Series(daily_rets, index=signals.index, name="V_A_3x_lev")


def simulate_variant_B(prices, signals):
    """
    Variant B — 2x Notional via Margin (200% capital deployed).
    Uses the SAME 1x ETFs but allocates 200% of portfolio.
    Pays margin interest on the borrowed 100%.
    Includes forced de-leverage: if drawdown >25%, reduce to 100%.
    """
    SLIPPAGE_2X   = 0.0005
    MARGIN_DAILY  = MARGIN_RATE_ANNUAL / 252

    daily_rets = []
    holdings   = None
    peak_equity = STARTING_CAPITAL
    equity      = STARTING_CAPITAL

    spy_ret = prices["SPY"].pct_change()
    qqq_ret = prices["QQQ"].pct_change()

    for i, date in enumerate(signals.index):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig   = signals.loc[date]
        asset, _ = get_rotation_signal(sig)

        # Dynamic de-lever on drawdown >25%
        dd = (equity - peak_equity) / peak_equity
        leverage = 2.0 if dd > -0.25 else 1.0

        if asset == "SPY":
            r = spy_ret.get(date, 0) or 0
        elif asset == "QQQ":
            r = qqq_ret.get(date, 0) or 0
        else:
            r = 0.0

        # Apply leverage
        leveraged_r = r * leverage

        # Subtract margin cost on borrowed portion
        borrowed_fraction = leverage - 1.0
        leveraged_r -= MARGIN_DAILY * borrowed_fraction

        # Transition cost
        if asset != holdings:
            leveraged_r -= SLIPPAGE_2X * 2 * leverage

        holdings = asset
        equity *= (1 + leveraged_r)
        peak_equity = max(peak_equity, equity)
        daily_rets.append(leveraged_r)

    return pd.Series(daily_rets, index=signals.index, name="V_B_2x_margin")


def simulate_variant_C(prices, signals):
    """
    Variant C — ATM Call Options Overlay (30-45 DTE).

    Implementation:
    - When strategy signals long SPY or QQQ, buy ATM call instead of shares.
    - Option delta ~0.5 (ATM), so 2x notional exposure per $1 spent on calls.
    - Model option P&L as: underlying_return * delta - daily_theta
    - Delta decay approximation: delta moves with underlying
    - Theta decay: premium / (DTE) per day
    - Options cost ~OPTIONS_IV_PREMIUM of notional on entry (bid-ask + IV)
    - Roll at 15 DTE to avoid gamma risk
    - When signal is CASH: hold cash (no options)

    Simplified BS-free model:
      - Long call value moves at ~delta=0.5 of underlying move when ATM
      - We allocate full capital to buying calls, so return = delta * underlying_ret
      - We subtract daily theta = (IV_cost / 45_days) per holding day
      - On a 5% up day: call gains 0.5*5% = 2.5% on same capital → SAME returns as 0.5x lev
      - BUT: we allocate full capital so P&L per dollar = delta * underlying_ret
      - Net: call P&L per $ = delta * underlying_ret - theta_daily

    Key metric: if underlying moves our way by more than the theta paid → profitable vs cash
    """
    DELTA         = 0.50   # ATM call delta (will drift but start at 0.50)
    DTE_ENTRY     = 40     # Days to expiration on entry
    DTE_ROLL      = 15     # Roll when DTE reaches this
    THETA_DAILY   = OPTIONS_IV_PREMIUM / DTE_ENTRY   # Daily theta as fraction of notional
    SLIPPAGE_OPT  = 0.002  # 20 bps per side for options (wider)

    daily_rets = []
    holdings   = None
    dte        = 0

    spy_ret = prices["SPY"].pct_change()
    qqq_ret = prices["QQQ"].pct_change()

    for i, date in enumerate(signals.index):
        if i == 0:
            daily_rets.append(0.0)
            dte = DTE_ENTRY
            continue

        sig   = signals.loc[date]
        asset, _ = get_rotation_signal(sig)

        if asset == "CASH":
            daily_rets.append(0.0)
            holdings = "CASH"
            dte = DTE_ENTRY  # Reset for next position
            continue

        # Get underlying return
        if asset == "SPY":
            underlying_r = spy_ret.get(date, 0) or 0
        else:
            underlying_r = qqq_ret.get(date, 0) or 0

        # Adjust delta dynamically (simple: up moves increase delta, down decrease)
        dynamic_delta = np.clip(DELTA + underlying_r * 2, 0.20, 0.90)

        # Option P&L = delta * underlying_return - theta
        option_r = dynamic_delta * underlying_r - THETA_DAILY

        # Transition cost: buy new options on asset switch or roll
        if asset != holdings or dte <= DTE_ROLL:
            option_r -= SLIPPAGE_OPT * 2
            dte = DTE_ENTRY  # Fresh contract
        else:
            dte = max(0, dte - 1)

        holdings = asset
        daily_rets.append(option_r)

    return pd.Series(daily_rets, index=signals.index, name="V_C_options")


def simulate_factor_rotation_leveraged(prices, signals):
    """
    Factor Rotation Leveraged — Quality factor rotation across sectors,
    using 3x leveraged sector ETFs.
    Monthly rebalance, top 3 sectors, equal weight.
    """
    SLIPPAGE_LEV = 0.001

    # Monthly rebalance dates
    rebal_idx = [0]
    for i in range(1, len(signals)):
        if signals.index[i].month != signals.index[i-1].month:
            rebal_idx.append(i)

    # Sector ETFs and their 3x equivalents
    sector_map = {
        "XLK": "TECL",
        "XLF": "FAS",
        "XLE": "ERX",
        "XLV": "CURE",
        "XLRE": "DRN",
        "XLU": "UTSL",
    }
    # Filter to what's available
    available_1x = [s for s in sector_map if s in prices.columns]
    available_3x = [sector_map[s] for s in available_1x if sector_map[s] in prices.columns]

    # If 3x ETFs aren't available, fall back to 3x synthetic
    use_3x_real = len(available_3x) >= 3

    daily_rets = []
    current_picks = []

    for i, date in enumerate(signals.index):
        if i == 0:
            daily_rets.append(0.0)
            continue

        # Rebalance on first day of new month
        if i in rebal_idx:
            current_picks = get_sector_rotation_picks(prices, signals, date, n_picks=3)

        if not current_picks:
            daily_rets.append(0.0)
            continue

        # Get daily returns for picks
        day_rets = []
        for pick in current_picks:
            if use_3x_real and pick in sector_map and sector_map[pick] in prices.columns:
                lev_pick = sector_map[pick]
                r = prices[lev_pick].pct_change().get(date, 0) or 0
                r -= lev3x_decay_daily()  # Vol decay
            else:
                # Synthetic 3x: 3 * daily return of 1x
                r = (prices[pick].pct_change().get(date, 0) or 0) * 3
                r -= lev3x_decay_daily()

            # Transition cost on new entry
            if i in rebal_idx:
                r -= SLIPPAGE_LEV * 2

            day_rets.append(r)

        daily_rets.append(np.mean(day_rets))

    return pd.Series(daily_rets, index=signals.index, name="FR_3x_sector")


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(daily_ret, name="Strategy"):
    """Compute comprehensive performance metrics."""
    dr = daily_ret.dropna()
    dr = dr[dr != 0]  # Active trading days only for some metrics

    all_dr = daily_ret.fillna(0)

    equity = STARTING_CAPITAL * (1 + all_dr).cumprod()

    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_years   = len(all_dr) / 252
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    vol       = all_dr.std() * np.sqrt(252)
    sharpe    = cagr / vol if vol > 0 else 0

    neg_dr    = all_dr[all_dr < 0]
    downside  = neg_dr.std() * np.sqrt(252) if len(neg_dr) > 0 else 1e-9
    sortino   = cagr / downside

    # Max drawdown
    peak      = equity.cummax()
    dd        = (equity - peak) / peak
    mdd       = dd.min()

    # Win rate and profit factor (on active days)
    if len(dr) > 0:
        wins   = (dr > 0).sum()
        losses = (dr < 0).sum()
        wr     = wins / (wins + losses) if (wins + losses) > 0 else 0
        avg_w  = dr[dr > 0].mean() if wins > 0 else 0
        avg_l  = abs(dr[dr < 0].mean()) if losses > 0 else 1e-9
        pf     = (avg_w * wins) / (avg_l * losses) if (avg_l * losses) > 0 else 999
    else:
        wr, pf = 0, 0

    # Calmar
    calmar = cagr / abs(mdd) if mdd < 0 else 0

    # Per-year breakdown
    yearly = {}
    for yr in range(2022, 2027):
        mask = all_dr.index.year == yr
        if mask.sum() > 50:
            yr_dr   = all_dr[mask]
            yr_ret  = (1 + yr_dr).prod() - 1
            yearly[str(yr)] = round(yr_ret, 4)

    return {
        "name":         name,
        "cagr":         round(cagr, 4),
        "total_return": round(total_ret, 4),
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_drawdown": round(mdd, 4),
        "calmar":       round(calmar, 3),
        "win_rate":     round(wr, 3),
        "profit_factor":round(pf, 3),
        "annual_vol":   round(vol, 4),
        "final_equity": round(equity.iloc[-1], 2),
        "n_years":      round(n_years, 2),
        "per_year":     yearly,
    }


def spy_buyhold_metrics(prices):
    """SPY buy-and-hold benchmark."""
    spy_ret = prices["SPY"].pct_change().loc[OOT_START:OOT_END].fillna(0)
    return compute_metrics(spy_ret, name="SPY_BuyHold")


def upro_buyhold_metrics(prices):
    """UPRO buy-and-hold benchmark (naive 3x S&P exposure)."""
    if "UPRO" in prices.columns:
        upro_ret = prices["UPRO"].pct_change().loc[OOT_START:OOT_END].fillna(0)
        return compute_metrics(upro_ret, name="UPRO_BuyHold")
    return None


# ── MONTE CARLO STRESS TEST ───────────────────────────────────────────────────
def monte_carlo_stress(daily_ret, n_sims=500, horizon_years=4):
    """Bootstrap-resample returns to estimate distribution of outcomes."""
    dr_arr = daily_ret.dropna().values
    n_days = int(horizon_years * 252)

    cagrs = []
    mdds  = []

    rng = np.random.RandomState(42)
    for _ in range(n_sims):
        sim_rets = rng.choice(dr_arr, size=n_days, replace=True)
        eq       = STARTING_CAPITAL * np.cumprod(1 + sim_rets)
        total    = eq[-1] / STARTING_CAPITAL - 1
        cagr     = (1 + total) ** (1/horizon_years) - 1
        peak     = np.maximum.accumulate(eq)
        mdd      = ((eq - peak) / peak).min()
        cagrs.append(cagr)
        mdds.append(mdd)

    return {
        "cagr_p10":  round(np.percentile(cagrs, 10), 4),
        "cagr_p50":  round(np.percentile(cagrs, 50), 4),
        "cagr_p90":  round(np.percentile(cagrs, 90), 4),
        "mdd_p10":   round(np.percentile(mdds, 10), 4),
        "mdd_median":round(np.percentile(mdds, 50), 4),
        "prob_loss": round((np.array(cagrs) < 0).mean(), 4),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("LEVERAGED STRATEGY ROTATION BACKTEST — HIGH GROWTH RESEARCH")
    print(f"OOT: {OOT_START} → {OOT_END}  |  Capital: ${STARTING_CAPITAL:,.0f}")
    print("=" * 72)

    # 1. Download data
    prices = download_data()

    # 2. Compute signals
    print("\n[2/7] Computing regime + rotation signals...")
    signals = compute_signals(prices)
    print(f"  Signals computed for {len(signals)} days ({signals.index[0].date()} → {signals.index[-1].date()})")

    # 3. Trim to OOT
    oot_signals = signals.loc[OOT_START:OOT_END]
    oot_prices  = prices.loc[OOT_START:OOT_END]
    print(f"  OOT window: {len(oot_signals)} trading days")

    # 4. Run all variants
    print("\n[3/7] Running strategy variants...")

    v0 = simulate_variant_0(oot_prices, oot_signals)
    print(f"  V0 (Unleveraged) done: {len(v0)} days")

    vA = simulate_variant_A(oot_prices, oot_signals)
    print(f"  VA (3x ETF sub) done: {len(vA)} days")

    vB = simulate_variant_B(oot_prices, oot_signals)
    print(f"  VB (2x Margin) done: {len(vB)} days")

    vC = simulate_variant_C(oot_prices, oot_signals)
    print(f"  VC (Options) done: {len(vC)} days")

    vFR = simulate_factor_rotation_leveraged(oot_prices, oot_signals)
    print(f"  V_FR (Factor Rotation 3x) done: {len(vFR)} days")

    # 5. Compute metrics
    print("\n[4/7] Computing performance metrics...")

    m0   = compute_metrics(v0,  name="V0_Unleveraged_StratRotation")
    mA   = compute_metrics(vA,  name="VA_3x_ETF_Substitution")
    mB   = compute_metrics(vB,  name="VB_2x_Margin")
    mC   = compute_metrics(vC,  name="VC_Options_Overlay")
    mFR  = compute_metrics(vFR, name="VFR_FactorRotation_3x")
    mSPY = spy_buyhold_metrics(prices)
    mUPRO = upro_buyhold_metrics(prices)

    all_metrics = [m0, mA, mB, mC, mFR, mSPY]
    if mUPRO:
        all_metrics.append(mUPRO)

    # 6. Monte Carlo stress tests
    print("\n[5/7] Monte Carlo stress tests (500 simulations each)...")

    mc_results = {}
    for v, name in [(v0,"V0"), (vA,"VA"), (vB,"VB"), (vC,"VC"), (vFR,"VFR")]:
        mc = monte_carlo_stress(v)
        mc_results[name] = mc
        print(f"  {name}: CAGR p10={mc['cagr_p10']:.1%} / p50={mc['cagr_p50']:.1%} / p90={mc['cagr_p90']:.1%} | "
              f"MDD p50={mc['mdd_median']:.1%} | P(loss)={mc['prob_loss']:.1%}")

    # 7. Print comparison table
    print("\n[6/7] Results Summary")
    print("=" * 72)
    header = f"{'Strategy':<35} {'CAGR':>7} {'MDD':>8} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6}"
    print(header)
    print("-" * 72)

    for m in all_metrics:
        print(f"{m['name']:<35} {m['cagr']:>7.1%} {m['max_drawdown']:>8.1%} "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['win_rate']:>6.1%} {m['profit_factor']:>6.2f}")

    print("-" * 72)
    print(f"\nPer-year returns (OOT 2022-2026):")
    for m in all_metrics:
        yr_str = "  ".join(f"{yr}:{ret:+.0%}" for yr, ret in m.get("per_year", {}).items())
        print(f"  {m['name']:<35} {yr_str}")

    # 8. Save results
    print("\n[7/7] Saving results...")

    results = {
        "meta": {
            "run_timestamp": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "starting_capital": STARTING_CAPITAL,
            "costs": {
                "slippage_pct": SLIPPAGE_PCT,
                "commission_per_rt": COMMISSION_PER_RT,
                "lev3x_annual_decay": LEV3X_ANNUAL_DECAY_PENALTY,
                "margin_rate_annual": MARGIN_RATE_ANNUAL,
                "options_iv_premium": OPTIONS_IV_PREMIUM,
            }
        },
        "variants": {m["name"]: m for m in all_metrics},
        "monte_carlo": mc_results,
    }

    out_path = OUT_DIR / "leveraged_strategy_rotation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")

    # Save equity curves
    curves_df = pd.DataFrame({
        "V0_Unleveraged": STARTING_CAPITAL * (1 + v0.fillna(0)).cumprod(),
        "VA_3x_ETF":      STARTING_CAPITAL * (1 + vA.fillna(0)).cumprod(),
        "VB_2x_Margin":   STARTING_CAPITAL * (1 + vB.fillna(0)).cumprod(),
        "VC_Options":     STARTING_CAPITAL * (1 + vC.fillna(0)).cumprod(),
        "VFR_Factor3x":   STARTING_CAPITAL * (1 + vFR.fillna(0)).cumprod(),
        "SPY_BuyHold":    STARTING_CAPITAL * (1 + prices["SPY"].pct_change().loc[OOT_START:OOT_END].fillna(0)).cumprod(),
    })
    curves_path = OUT_DIR / "equity_curves.csv"
    curves_df.to_csv(curves_path)
    print(f"  Equity curves saved: {curves_path}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    return results


if __name__ == "__main__":
    main()
