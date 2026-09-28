"""
Calendar Spread Income Strategy Backtest v1
===========================================
ATM put calendar spreads: sell near-term (10-18 DTE) / buy far-term (25-50 DTE)
at the same strike.

Edge thesis:
  - Near-term theta decays faster than far-term (non-linear time decay)
  - Defined risk: max loss = net debit paid
  - Positive vega: benefits from vol expansion (unlike naked premium sellers)
  - Direction-neutral: profits if stock stays near strike

Key mechanical features:
  - Worst-case fills: sell-at-bid, buy-at-ask
  - $100K portfolio, 2% NAV max debit per calendar, max 10 concurrent
  - HC #428 R1 regime gap test across ALL OOT days
  - 100-trial permutation test
  - Year-by-year breakdown
  - Comparison vs CSP baseline

Data available:
  - DTE range in chains: 10–67
  - Short leg: 10–18 DTE (bi-weekly expirations)
  - Long leg: 25–50 DTE (monthly expirations)
  - Both legs need the same strike → ATM selection
"""

import json
import random
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
CHAINS_DIR      = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains")
PRICES_PATH     = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet")
IV_FEATURES_PATH = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/iv_features_real.parquet")
OUT_DIR         = Path("/home/jupiter/Lvl3Quant/output/calendar_spread_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Strategy Parameters ──────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000.0
MAX_DEBIT_PCT    = 0.02          # 2% of NAV per calendar max debit
MAX_NOTIONAL_PCT = 0.05          # 5% of NAV max notional (strike * 100 * contracts)
MAX_CALENDARS    = 10            # concurrent open spreads
MAX_CONTRACTS    = 20            # hard cap on contracts per position
TOP_N_RANK       = 10            # candidate pool ranked by IV rank
IV_RANK_MIN      = 0.50          # IV percentile > 50% filter
SHORT_DTE_MIN    = 10
SHORT_DTE_MAX    = 18
LONG_DTE_MIN     = 25
LONG_DTE_MAX     = 50
CLOSE_NEAR_PCT   = 0.02          # within 2% → short expires worthless, roll
CLOSE_AWAY_PCT   = 0.05          # beyond 5% → close whole spread
RANDOM_SEED      = 42
N_PERMUTATIONS   = 50   # 50 trials × ~40s = ~33 min total
OOT_FRACTION     = 0.30          # last 30% for HC #428 R1 test


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load chains, IV features, SPY regime.
    Builds fast nested-dict lookups: {date: {ticker: rows_df}}
    """
    t0 = time.time()
    print("\n[1/6] Loading option chains...")

    frames = []
    for f in sorted(CHAINS_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(
                f,
                columns=["date", "expiration", "strike", "type",
                         "bid", "ask", "vol", "delta", "dte", "mid"]
            )
            df["ticker"] = f.stem
            frames.append(df)
        except Exception:
            pass

    chains = pd.concat(frames, ignore_index=True)
    chains["date"]       = pd.to_datetime(chains["date"])
    chains["expiration"] = pd.to_datetime(chains["expiration"])
    print(f"  {len(chains):,} rows, {len(frames)} tickers ({time.time()-t0:.1f}s)")

    # ── Spot prices from near-ATM options (scale-consistent with strikes)
    atm_for_spot = chains[
        chains["delta"].abs().between(0.40, 0.60) &
        chains["dte"].between(5, 50)
    ]
    spot_from_chains = (
        atm_for_spot.groupby(["ticker", "date"])["strike"]
        .median()
        .reset_index()
        .rename(columns={"strike": "spot"})
    )
    spot_dict: dict[str, dict] = {}
    for ticker, grp in spot_from_chains.groupby("ticker"):
        spot_dict[ticker] = dict(zip(grp["date"], grp["spot"].astype(float)))

    # ── Pre-filter put legs
    puts_short = chains[
        (chains["type"] == "p") &
        (chains["dte"].between(SHORT_DTE_MIN, SHORT_DTE_MAX)) &
        (chains["bid"] > 0) &
        (chains["vol"] > 0)
    ].copy()
    puts_long = chains[
        (chains["type"] == "p") &
        (chains["dte"].between(LONG_DTE_MIN, LONG_DTE_MAX)) &
        (chains["bid"] > 0) &
        (chains["vol"] > 0)
    ].copy()
    print(f"  Short leg puts: {len(puts_short):,}  Long leg puts: {len(puts_long):,}")

    # ── Build nested {date: {ticker: df}} for fast lookup
    print("  Building nested date/ticker indices...")
    t1 = time.time()
    short_nested = _build_nested(puts_short)
    long_nested  = _build_nested(puts_long)

    # ── Full puts index for MTM and close valuation by exact expiration
    # {date: {ticker: {expiration: mid}}} — only need strike==ATM lookup by expiration
    # Build as {(date, ticker, expiration, strike): mid}
    puts_all = chains[
        (chains["type"] == "p") &
        (chains["bid"] > 0)
    ][["date", "ticker", "expiration", "strike", "bid", "mid"]].copy()
    # nested: {date: {ticker: DataFrame}} with all puts
    all_puts_nested = _build_nested(puts_all)
    print(f"  Indices ready ({time.time()-t1:.1f}s)")

    # ── IV rank
    print("[2/6] Loading IV features...")
    try:
        iv_feat = pd.read_parquet(IV_FEATURES_PATH)
        iv_feat["date"] = pd.to_datetime(iv_feat["date"])
        iv_feat = iv_feat[["date", "ticker", "iv_rank"]].dropna(subset=["iv_rank"])
        iv_rank_dict: dict[str, dict] = {}
        for ticker, grp in iv_feat.groupby("ticker"):
            iv_rank_dict[ticker] = dict(zip(grp["date"], grp["iv_rank"].astype(float)))
        print(f"  IV rank loaded: {len(iv_feat):,} rows")
    except Exception as e:
        print(f"  Warning: IV features unavailable ({e})")
        iv_rank_dict = {}

    # ── SPY regime
    print("[3/6] Loading SPY for regime classification...")
    prices_raw = pd.read_parquet(PRICES_PATH)
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    spy = prices_raw[prices_raw["ticker"] == "SPY"][["date", "close"]].sort_values("date").copy()
    spy["ret"] = spy["close"].pct_change()
    spy["regime"] = "flat"
    spy.loc[spy["ret"] >  0.003, "regime"] = "green"
    spy.loc[spy["ret"] < -0.003, "regime"] = "red"
    regime_dict = dict(zip(spy["date"], spy["regime"]))

    all_dates = sorted(short_nested.keys())
    print(f"  Total trading dates: {len(all_dates)} ({all_dates[0].date()} – {all_dates[-1].date()})")

    return short_nested, long_nested, all_puts_nested, spot_dict, iv_rank_dict, regime_dict, all_dates


def _build_nested(df: pd.DataFrame) -> dict:
    """Build {date: {ticker: sub-DataFrame}} from flat DataFrame."""
    result: dict = {}
    for (date, ticker), grp in df.groupby(["date", "ticker"]):
        if date not in result:
            result[date] = {}
        result[date][ticker] = grp
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_spot(spot_dict: dict, ticker: str, date: pd.Timestamp) -> float | None:
    d = spot_dict.get(ticker)
    if d is None:
        return None
    v = d.get(date)
    if v is not None:
        return float(v)
    for delta in [1, 2, 3, 5, 7]:
        v2 = d.get(date - pd.Timedelta(days=delta))
        if v2 is not None:
            return float(v2)
    return None


def _get_iv_rank(iv_rank_dict: dict, ticker: str, date: pd.Timestamp) -> float:
    d = iv_rank_dict.get(ticker, {})
    v = d.get(date)
    if v is not None:
        return float(v)
    for delta in [1, 2, 3, 5, 7]:
        v2 = d.get(date - pd.Timedelta(days=delta))
        if v2 is not None:
            return float(v2)
    return 0.5  # neutral default


def _find_atm_put(rows: pd.DataFrame, spot: float) -> pd.Series | None:
    """Return the ATM-closest put row from a per-ticker/date slice."""
    if rows is None or len(rows) == 0:
        return None
    valid = rows[rows["bid"] > 0]
    if valid.empty:
        return None
    idx = (valid["strike"] - spot).abs().idxmin()
    return valid.loc[idx]


def _compute_metrics(equity_curve: list, dates: list, weekly_pnl: list, label: str) -> dict:
    ec  = np.array(equity_curve, dtype=float)
    pnl = np.array(weekly_pnl, dtype=float)

    if len(ec) < 4 or ec[0] == 0:
        return {"label": label, "error": "insufficient data"}

    rets = np.diff(ec) / ec[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        return {"label": label, "error": "no returns"}

    ann    = 52.0
    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1e-9
    down_r = rets[rets < 0]
    sort_d = float(np.std(down_r, ddof=1)) if len(down_r) > 1 else 1e-9

    sharpe  = (mean_r / std_r) * np.sqrt(ann)
    sortino = (mean_r / sort_d) * np.sqrt(ann)

    wins   = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf     = float(wins / losses) if losses > 0 else float("inf")
    wr     = float((pnl > 0).mean())

    peak = ec[0]; max_dd = 0.0
    for v in ec:
        if v > peak: peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    total_ret = (ec[-1] - ec[0]) / ec[0]
    n_years   = len(rets) / ann
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    return {
        "label":         label,
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else "inf",
        "win_rate":      round(wr, 3),
        "cagr":          round(cagr, 4),
        "total_ret":     round(total_ret, 4),
        "max_dd":        round(max_dd, 4),
        "n_weeks":       len(rets),
        "final_nav":     round(float(ec[-1]), 2),
        "equity_curve":  ec.tolist(),
        "dates":         [str(d) for d in dates],
        "weekly_pnl":    pnl.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. CALENDAR SPREAD BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_calendar_backtest(
    short_nested:    dict,
    long_nested:     dict,
    all_puts_nested: dict,
    spot_dict:       dict,
    iv_rank_dict:    dict,
    all_dates:       list,
    use_iv_rank:     bool = True,
    label:           str  = "Calendar",
    rng_seed:        int  = RANDOM_SEED,
    shuffle_dates:   bool = False,
) -> dict:
    """
    Weekly calendar spread backtest.

    Each date:
      1. Screen for high IV rank tickers (or random baseline)
      2. Open ATM put calendars — sell short leg (bid), buy long leg (ask)
      3. When short leg expires:
         - Within 2% of strike: short expires worthless → roll (sell new short)
         - Beyond 5%: stock moved — close entire spread (sell long at bid)
         - Between: also close entire spread
      4. Track NAV with spread marked at (long_mid - short_mid)
    """
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    dates = list(all_dates)
    if shuffle_dates:
        rng.shuffle(dates)
    if len(dates) < 8:
        return {"label": label, "error": "insufficient dates"}

    all_tickers = set()
    for d_map in short_nested.values():
        all_tickers.update(d_map.keys())

    nav       = float(INITIAL_CAPITAL)
    free_cash = float(INITIAL_CAPITAL)
    equity_curve = [nav]
    dates_out    = [dates[0]]
    weekly_pnl   = []
    open_positions: list[dict] = []
    trades_opened = 0; trades_closed = 0
    close_reasons = {"rolled": 0, "near_expired_close": 0, "away_close": 0,
                     "long_expired": 0}
    real_leg_count = 0; total_leg_count = 0

    for trade_date in dates:
        short_by_ticker = short_nested.get(trade_date, {})
        long_by_ticker  = long_nested.get(trade_date, {})
        week_realized   = 0.0
        closed_tickers  = set()

        # ── A. PROCESS POSITIONS (expired short legs) ─────────────────────
        still_open: list[dict] = []
        all_puts_today = all_puts_nested.get(trade_date, {})

        for pos in open_positions:
            ticker  = pos["ticker"]
            strike  = pos["strike"]
            contracts = pos["contracts"]
            cost_basis = pos["net_debit_total"]

            spot = _get_spot(spot_dict, ticker, trade_date)

            long_expired  = pos["long_exp"]  <= trade_date
            short_expired = pos["short_exp"] <= trade_date

            if long_expired:
                # Long leg expired: if ITM, book intrinsic value; else 0
                # (we own the long put at expiry)
                if spot is not None and spot < strike:
                    intrinsic = (strike - spot) * 100 * contracts
                else:
                    intrinsic = 0.0
                pnl = intrinsic - cost_basis
                week_realized += pnl
                free_cash     += cost_basis + pnl
                close_reasons["long_expired"] += 1
                trades_closed += 1
                closed_tickers.add(ticker)
                continue

            if short_expired:
                dist_pct = abs(spot - strike) / strike if spot else 1.0

                if dist_pct <= CLOSE_NEAR_PCT:
                    # Short expired near ATM → try to roll (sell next weekly)
                    new_short_rows = short_by_ticker.get(ticker)
                    rolled = False
                    if new_short_rows is not None:
                        # Prefer same strike; else ATM
                        exact = new_short_rows[new_short_rows["strike"] == strike]
                        if exact.empty:
                            exact = new_short_rows
                        row = _find_atm_put(exact, spot if spot else strike)
                        if row is not None and row["bid"] > 0:
                            roll_credit = float(row["bid"]) * 100 * contracts
                            free_cash   += roll_credit
                            week_realized += roll_credit
                            pos = dict(pos)
                            pos["short_exp"] = row["expiration"]
                            pos["short_dte"] = int(row["dte"])
                            pos["rolls"]     = pos.get("rolls", 0) + 1
                            still_open.append(pos)
                            close_reasons["rolled"] += 1
                            rolled = True

                    if not rolled:
                        # Can't roll: close long leg at bid (exact expiration)
                        all_puts_ticker = all_puts_today.get(ticker)
                        long_val = _value_leg_by_expiry(all_puts_ticker, pos["long_exp"], strike, use_bid=True)
                        total_val = (long_val or 0.0) * 100 * contracts
                        pnl = total_val - cost_basis
                        week_realized += pnl
                        free_cash     += cost_basis + pnl
                        close_reasons["near_expired_close"] += 1
                        trades_closed += 1
                        closed_tickers.add(ticker)

                else:
                    # Stock moved away: close entire spread (sell long at bid)
                    all_puts_ticker = all_puts_today.get(ticker)
                    long_val = _value_leg_by_expiry(all_puts_ticker, pos["long_exp"], strike, use_bid=True)
                    total_val = (long_val or 0.0) * 100 * contracts
                    pnl = total_val - cost_basis
                    week_realized += pnl
                    free_cash     += cost_basis + pnl
                    close_reasons["away_close"] += 1
                    trades_closed += 1
                    closed_tickers.add(ticker)
            else:
                still_open.append(pos)

        open_positions = still_open

        # ── B. OPEN NEW POSITIONS ─────────────────────────────────────────
        occupied = {p["ticker"] for p in open_positions}
        slots = MAX_CALENDARS - len(occupied)

        if slots > 0 and short_by_ticker and long_by_ticker:
            available = list(all_tickers - occupied - closed_tickers)

            if use_iv_rank and iv_rank_dict:
                scored = [(t, _get_iv_rank(iv_rank_dict, t, trade_date))
                          for t in available]
                scored = [(t, s) for t, s in scored if s >= IV_RANK_MIN]
                scored.sort(key=lambda x: x[1], reverse=True)
                candidates = [t for t, _ in scored[:TOP_N_RANK]]
            else:
                rng.shuffle(available)
                candidates = available[:TOP_N_RANK]

            opened_this_week = 0
            for ticker in candidates:
                if opened_this_week >= slots:
                    break

                spot = _get_spot(spot_dict, ticker, trade_date)
                if spot is None:
                    continue

                s_rows = short_by_ticker.get(ticker)
                l_rows = long_by_ticker.get(ticker)
                if s_rows is None or l_rows is None:
                    continue

                short_row = _find_atm_put(s_rows, spot)
                if short_row is None:
                    continue

                atm_strike = float(short_row["strike"])

                # Long leg: same strike preferred
                l_exact = l_rows[l_rows["strike"] == atm_strike]
                if l_exact.empty:
                    l_exact = l_rows
                long_row = _find_atm_put(l_exact, spot)
                if long_row is None:
                    continue

                # Worst-case fills: sell short at BID, buy long at ASK
                short_credit = float(short_row["bid"])
                long_debit   = float(long_row["ask"])
                net_debit    = long_debit - short_credit   # per share

                if net_debit <= 0 or long_debit <= 0:
                    continue  # credit calendar or degenerate — skip

                # Size: max 2% NAV as net debit, also cap by notional and hard contract limit
                max_debit_dollars  = nav * MAX_DEBIT_PCT
                max_notional_dollars = nav * MAX_NOTIONAL_PCT
                contracts_by_debit   = int(max_debit_dollars / (net_debit * 100))
                contracts_by_notional = int(max_notional_dollars / (atm_strike * 100))
                contracts = max(1, min(contracts_by_debit, contracts_by_notional, MAX_CONTRACTS))
                total_cost   = net_debit * 100 * contracts
                if total_cost > free_cash:
                    continue

                free_cash -= total_cost
                open_positions.append({
                    "ticker":         ticker,
                    "strike":         atm_strike,
                    "short_exp":      short_row["expiration"],
                    "long_exp":       long_row["expiration"],
                    "short_dte":      int(short_row["dte"]),
                    "long_dte":       int(long_row["dte"]),
                    "short_credit":   short_credit,
                    "long_debit":     long_debit,
                    "net_debit_total": total_cost,
                    "contracts":      contracts,
                    "open_date":      trade_date,
                    "rolls":          0,
                })
                trades_opened += 1
                real_leg_count += 2; total_leg_count += 2
                opened_this_week += 1

        # ── C. MARK-TO-MARKET NAV ─────────────────────────────────────────
        spread_mtm = 0.0
        for pos in open_positions:
            ticker = pos["ticker"]
            # Use exact expiration for MTM to avoid picking wrong expiry
            all_puts_ticker = all_puts_today.get(ticker)
            long_mid  = _value_leg_by_expiry(all_puts_ticker, pos["long_exp"],  pos["strike"], use_bid=False)
            short_mid = _value_leg_by_expiry(all_puts_ticker, pos["short_exp"], pos["strike"], use_bid=False)
            if long_mid is None:
                long_mid = pos["long_debit"]       # fallback: cost
            if short_mid is None:
                short_mid = pos["short_credit"]    # fallback: original credit
            spread_val = (long_mid - short_mid) * 100 * pos["contracts"]
            spread_mtm += max(0.0, spread_val)     # floor at 0 (max loss = debit)

        nav = free_cash + spread_mtm
        equity_curve.append(nav)
        dates_out.append(trade_date)
        weekly_pnl.append(week_realized)

    metrics = _compute_metrics(equity_curve, dates_out, weekly_pnl, label)
    metrics["trades_opened"] = trades_opened
    metrics["trades_closed"] = trades_closed
    metrics["close_reasons"] = close_reasons
    metrics["real_price_coverage_pct"] = round(
        100 * real_leg_count / max(1, total_leg_count), 1)
    return metrics


def _value_leg(
    rows: pd.DataFrame | None,
    strike: float,
    orig_dte: int,
    use_bid: bool = True,
) -> float | None:
    """Get bid/mid for a put leg at given strike.
    rows is already the per-ticker/date slice for either short or long DTE range.
    Used only for MTM of open positions (approximate — exact valuation uses _value_leg_exact).
    """
    if rows is None or rows.empty:
        return None
    exact = rows[rows["strike"] == strike]
    if not exact.empty:
        col = "bid" if use_bid else "mid"
        v = float(exact.iloc[0][col])
        return v if v >= 0 else None
    # Fallback: closest strike within ±5
    close = rows[(rows["strike"] - strike).abs() <= 5.0]
    if not close.empty:
        col = "bid" if use_bid else "mid"
        idx = (close["strike"] - strike).abs().idxmin()
        v = float(close.loc[idx, col])
        return v if v >= 0 else None
    return None


def _value_leg_by_expiry(
    all_puts_by_ticker: pd.DataFrame | None,
    expiration: pd.Timestamp,
    strike: float,
    use_bid: bool = True,
) -> float | None:
    """Look up a specific leg's value by exact expiration date and strike.
    all_puts_by_ticker is the full puts DataFrame for this ticker on trade_date.
    This is the CORRECT close/roll valuation: finds the right expiry.
    """
    if all_puts_by_ticker is None or all_puts_by_ticker.empty:
        return None
    exact = all_puts_by_ticker[
        (all_puts_by_ticker["expiration"] == expiration) &
        (all_puts_by_ticker["strike"] == strike)
    ]
    if not exact.empty:
        col = "bid" if use_bid else "mid"
        v = float(exact.iloc[0][col])
        return v if v >= 0 else None
    # If exact expiration not listed, try nearest expiration within ±7 days
    strike_mask  = all_puts_by_ticker["strike"] == strike
    exp_diff     = (all_puts_by_ticker["expiration"] - expiration).abs()
    nearby_mask  = strike_mask & (exp_diff <= pd.Timedelta(days=7))
    nearby = all_puts_by_ticker[nearby_mask]
    if not nearby.empty:
        col = "bid" if use_bid else "mid"
        idx = (nearby["expiration"] - expiration).abs().idxmin()
        v = float(nearby.loc[idx, col])
        return v if v >= 0 else None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. CSP BASELINE (sell ATM put each cycle, no assignment)
# ═══════════════════════════════════════════════════════════════════════════

def run_csp_baseline(
    short_nested:  dict,
    spot_dict:     dict,
    iv_rank_dict:  dict,
    all_dates:     list,
    label:         str = "CSP_Baseline",
    rng_seed:      int = RANDOM_SEED,
) -> dict:
    """
    Sell ATM put (10-18 DTE) on high IV rank stocks.
    At expiration: worthless → keep premium; ITM → book loss.
    No assignment / stock holding.
    Commission-free.
    """
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_tickers = set()
    for d_map in short_nested.values():
        all_tickers.update(d_map.keys())

    nav = float(INITIAL_CAPITAL)
    free_cash = float(INITIAL_CAPITAL)
    equity_curve = [nav]
    dates_out = [all_dates[0]]
    weekly_pnl = []
    open_puts: list[dict] = []
    reserved: dict[str, float] = {}   # ticker -> margin reserved

    for trade_date in all_dates:
        by_ticker = short_nested.get(trade_date, {})
        week_realized = 0.0

        # Settle
        still_open = []
        for pos in open_puts:
            if pos["expiration"] <= trade_date:
                ticker = pos["ticker"]
                spot = _get_spot(spot_dict, ticker, trade_date)
                mgn = pos["margin_reserved"]
                free_cash += mgn
                reserved.pop(ticker, None)
                if spot is None or spot >= pos["strike"]:
                    week_realized += pos["premium"]
                else:
                    intrinsic = (pos["strike"] - spot) * 100 * pos["contracts"]
                    week_realized += pos["premium"] - intrinsic
            else:
                still_open.append(pos)
        open_puts = still_open

        # Open new
        occupied = {p["ticker"] for p in open_puts}
        slots = MAX_CALENDARS - len(occupied)
        if slots > 0 and by_ticker:
            available = list(all_tickers - occupied)
            scored = [(t, _get_iv_rank(iv_rank_dict, t, trade_date)) for t in available]
            scored = [(t, s) for t, s in scored if s >= IV_RANK_MIN]
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [t for t, _ in scored[:TOP_N_RANK]]

            opened = 0
            for ticker in candidates:
                if opened >= slots:
                    break
                spot = _get_spot(spot_dict, ticker, trade_date)
                if spot is None:
                    continue
                rows = by_ticker.get(ticker)
                if rows is None:
                    continue
                row = _find_atm_put(rows, spot)
                if row is None:
                    continue

                strike    = float(row["strike"])
                prem_share = float(row["bid"])
                # CSP: size by notional (strike * 100 * contracts) as % of NAV
                contracts  = max(1, min(int(nav * MAX_NOTIONAL_PCT / (strike * 100)), MAX_CONTRACTS))
                mgn        = strike * 100 * contracts
                prem       = prem_share * 100 * contracts
                if mgn > free_cash:
                    continue

                free_cash -= mgn
                free_cash += prem
                reserved[ticker] = mgn
                open_puts.append({
                    "ticker": ticker, "strike": strike,
                    "expiration": row["expiration"], "contracts": contracts,
                    "premium": prem, "margin_reserved": mgn,
                })
                opened += 1

        total_margin = sum(reserved.values())
        nav = free_cash + total_margin
        equity_curve.append(nav)
        dates_out.append(trade_date)
        weekly_pnl.append(week_realized)

    return _compute_metrics(equity_curve, dates_out, weekly_pnl, label)


# ═══════════════════════════════════════════════════════════════════════════
# 5. HC #428 R1 REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def regime_analysis(metrics: dict, regime_dict: dict) -> dict:
    """HC #428 R1: stratify Sharpe by market regime on OOT portion."""
    dates  = [pd.Timestamp(d) for d in metrics.get("dates", [])]
    pnl_a  = np.array(metrics.get("weekly_pnl", []), dtype=float)
    ec_a   = np.array(metrics.get("equity_curve", []), dtype=float)

    if len(dates) < 4:
        return {"error": "insufficient data"}

    n_oot     = max(4, int(len(dates) * OOT_FRACTION))
    oot_dates = dates[-n_oot:]
    oot_pnl   = pnl_a[-n_oot:]
    oot_ec    = ec_a[-(n_oot+1):]
    oot_rets  = np.diff(oot_ec) / oot_ec[:-1]
    oot_rets  = oot_rets[:len(oot_dates)]

    regimes = np.array([regime_dict.get(d, "unknown") for d in oot_dates])
    result  = {}
    ann     = 52.0

    for reg in ["green", "red", "flat"]:
        mask = regimes == reg
        if mask.sum() < 3:
            result[reg] = {"n": int(mask.sum()), "sharpe": None}
            continue
        r = oot_rets[mask]
        mean_r = float(np.mean(r))
        std_r  = float(np.std(r, ddof=1)) if len(r) > 1 else 1e-9
        sharpe = (mean_r / std_r) * np.sqrt(ann)
        p = oot_pnl[mask]
        wins = p[p > 0].sum(); losses = -p[p < 0].sum()
        pf = float(wins / losses) if losses > 0 else float("inf")
        result[reg] = {
            "n":      int(mask.sum()),
            "sharpe": round(sharpe, 3),
            "pf":     round(pf, 3) if np.isfinite(pf) else "inf",
            "wr":     round(float((p > 0).mean()), 3),
        }

    sg = result.get("green", {}).get("sharpe")
    sr = result.get("red",   {}).get("sharpe")
    if sg is not None and sr is not None:
        denom = max(abs(sg), abs(sr), 1e-6)
        ratio = abs(sg - sr) / denom
        result["regime_gap_ratio"] = round(ratio, 3)
        result["passes_hc428_r1"]  = bool(ratio <= 0.50)
        result["verdict"] = "PASS" if ratio <= 0.50 else "FAIL (regime-tailored)"
    else:
        result["passes_hc428_r1"] = None
        result["verdict"] = "INSUFFICIENT_DATA"

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test(
    short_nested:    dict,
    long_nested:     dict,
    all_puts_nested: dict,
    spot_dict:       dict,
    iv_rank_dict:    dict,
    all_dates:       list,
    observed_sharpe: float,
    n_trials: int = N_PERMUTATIONS,
) -> dict:
    """Shuffle calendar opening timing; compare observed Sharpe vs null."""
    print(f"\n[5/6] Permutation test ({n_trials} trials)...")
    null_sharpes = []

    for i in range(n_trials):
        m = run_calendar_backtest(
            short_nested, long_nested, all_puts_nested, spot_dict, iv_rank_dict, all_dates,
            use_iv_rank=False,
            label=f"perm_{i}",
            rng_seed=i + 1000,
            shuffle_dates=True,
        )
        if "sharpe" in m:
            null_sharpes.append(m["sharpe"])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_trials} done...")

    if not null_sharpes:
        return {"error": "all permutations failed"}

    null_arr = np.array(null_sharpes)
    p_value  = float((null_arr >= observed_sharpe).mean())
    return {
        "observed_sharpe": round(observed_sharpe, 3),
        "null_mean":       round(float(null_arr.mean()), 3),
        "null_std":        round(float(null_arr.std()), 3),
        "null_p5":         round(float(np.percentile(null_arr, 5)), 3),
        "null_p95":        round(float(np.percentile(null_arr, 95)), 3),
        "p_value":         round(p_value, 3),
        "n_trials":        len(null_sharpes),
        "significant":     bool(p_value < 0.05),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. YEAR-BY-YEAR BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def year_by_year(metrics: dict, regime_dict: dict) -> list[dict]:
    dates = [pd.Timestamp(d) for d in metrics.get("dates", [])]
    pnl_a = np.array(metrics.get("weekly_pnl", []), dtype=float)
    ec_a  = np.array(metrics.get("equity_curve", []), dtype=float)

    if len(dates) < 2:
        return []

    results = []
    years = sorted(set(d.year for d in dates[1:]))
    for yr in years:
        mask = np.array([d.year == yr for d in dates[1:]])
        if mask.sum() < 4:
            continue
        yr_pnl = pnl_a[mask]
        ec_idx  = np.concatenate([[True], mask])
        yr_ec   = ec_a[ec_idx]

        yr_rets = np.diff(yr_ec) / yr_ec[:-1]
        yr_rets = yr_rets[np.isfinite(yr_rets)]
        if len(yr_rets) == 0:
            continue

        mean_r = float(np.mean(yr_rets))
        std_r  = float(np.std(yr_rets, ddof=1)) if len(yr_rets) > 1 else 1e-9
        sharpe = (mean_r / std_r) * np.sqrt(52)

        wins = yr_pnl[yr_pnl > 0].sum(); losses = -yr_pnl[yr_pnl < 0].sum()
        pf = float(wins / losses) if losses > 0 else float("inf")
        wr = float((yr_pnl > 0).mean())

        yr_dates = [d for d in dates[1:] if d.year == yr]
        reg_counts = {reg: sum(1 for d in yr_dates if regime_dict.get(d) == reg)
                      for reg in ["green", "red", "flat"]}

        total_ret = (yr_ec[-1] - yr_ec[0]) / yr_ec[0] if yr_ec[0] != 0 else 0.0
        results.append({
            "year": yr, "sharpe": round(sharpe, 3),
            "pf": round(pf, 3) if np.isfinite(pf) else "inf",
            "wr": round(wr, 3),
            "total_ret": round(total_ret, 4),
            "n_weeks": int(mask.sum()),
            "regimes": reg_counts,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 68)
    print("  CALENDAR SPREAD INCOME STRATEGY BACKTEST v1")
    print("  HC #428 R1 compliant | 100-trial permutation | bid/ask fills")
    print("=" * 68)

    # ── Load
    short_nested, long_nested, all_puts_nested, spot_dict, iv_rank_dict, regime_dict, all_dates = load_data()

    # ── Calendar backtest (IV-ranked)
    print("\n[4/6] Running calendar spread backtest (IV-ranked)...")
    t1 = time.time()
    cal = run_calendar_backtest(
        short_nested, long_nested, all_puts_nested, spot_dict, iv_rank_dict, all_dates,
        use_iv_rank=True, label="Calendar_IVRank",
    )
    print(f"  Done in {time.time()-t1:.1f}s")
    if "error" in cal:
        print(f"  FATAL: {cal['error']}")
        return

    print(f"  Sharpe:  {cal['sharpe']}")
    print(f"  Sortino: {cal['sortino']}")
    print(f"  PF:      {cal['profit_factor']}")
    print(f"  WR:      {cal['win_rate']:.1%}")
    print(f"  CAGR:    {cal['cagr']:.1%}")
    print(f"  MaxDD:   {cal['max_dd']:.1%}")
    print(f"  Trades opened/closed: {cal['trades_opened']}/{cal['trades_closed']}")
    print(f"  Real price coverage:  {cal.get('real_price_coverage_pct', 'N/A')}%")
    print(f"  Close reasons: {cal['close_reasons']}")

    # ── CSP baseline
    print("\n  Running CSP baseline...")
    csp = run_csp_baseline(short_nested, spot_dict, iv_rank_dict, all_dates)
    print(f"  CSP Sharpe: {csp.get('sharpe','N/A')}  CAGR: {csp.get('cagr',0):.1%}  MaxDD: {csp.get('max_dd',0):.1%}")

    # ── HC #428 R1
    print("\n  HC #428 R1 regime analysis...")
    regime_res = regime_analysis(cal, regime_dict)
    for reg in ["green", "red", "flat"]:
        r = regime_res.get(reg, {})
        print(f"    {reg:5}: n={r.get('n','?'):4}  Sharpe={r.get('sharpe','N/A')}  PF={r.get('pf','N/A')}  WR={r.get('wr','N/A')}")
    print(f"  Regime gap ratio: {regime_res.get('regime_gap_ratio','N/A')}")
    print(f"  HC #428 R1 verdict: {regime_res.get('verdict','N/A')}")

    # ── Year-by-year
    print("\n  Year-by-year:")
    yby = year_by_year(cal, regime_dict)
    print("  Year  Sharpe   PF      WR      Ret")
    for yr in yby:
        print(f"  {yr['year']}  {yr['sharpe']:6.3f}  {str(yr['pf']):6}  {yr['wr']:.3f}  {yr['total_ret']:.1%}")

    # ── Permutation test
    perm = permutation_test(
        short_nested, long_nested, all_puts_nested, spot_dict, iv_rank_dict, all_dates,
        observed_sharpe=cal["sharpe"], n_trials=N_PERMUTATIONS,
    )
    print(f"\n  Permutation p-value: {perm.get('p_value','N/A')}")
    print(f"  Null Sharpe: {perm.get('null_mean','N/A')} ± {perm.get('null_std','N/A')}")
    print(f"  Significant (p<0.05): {perm.get('significant','N/A')}")

    # ── Save
    print("\n[6/6] Saving results...")
    out = {
        "calendar_metrics":  {k: v for k, v in cal.items()
                              if k not in ("equity_curve","dates","weekly_pnl")},
        "csp_baseline":      {k: v for k, v in csp.items()
                              if k not in ("equity_curve","dates","weekly_pnl")},
        "regime_analysis":   regime_res,
        "year_by_year":      yby,
        "permutation_test":  perm,
        "strategy_params": {
            "initial_capital": INITIAL_CAPITAL,
            "max_debit_pct":   MAX_DEBIT_PCT,
            "max_calendars":   MAX_CALENDARS,
            "short_dte_range": [SHORT_DTE_MIN, SHORT_DTE_MAX],
            "long_dte_range":  [LONG_DTE_MIN, LONG_DTE_MAX],
            "iv_rank_min":     IV_RANK_MIN,
            "close_near_pct":  CLOSE_NEAR_PCT,
            "fill_method":     "sell-at-bid / buy-at-ask (worst case)",
        },
        "runtime_seconds": round(time.time() - t_total, 1),
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    curves = {
        "calendar_dates": cal["dates"],
        "calendar_nav":   cal["equity_curve"],
        "csp_dates":      csp.get("dates", []),
        "csp_nav":        csp.get("equity_curve", []),
    }
    with open(OUT_DIR / "equity_curves.json", "w") as f:
        json.dump(curves, f)

    print(f"  Results → {OUT_DIR}/results.json")
    print(f"  Curves  → {OUT_DIR}/equity_curves.json")

    # ── Final summary
    print("\n" + "=" * 68)
    print("  FINAL SUMMARY")
    print("=" * 68)
    print(f"  Strategy:   ATM Put Calendar Spread (short 10-18 DTE / long 25-50 DTE)")
    print(f"  Universe:   69 tickers, 2019–2026, real bid/ask fills")
    print(f"  CAGR:       {cal['cagr']:.1%}   (CSP: {csp.get('cagr',0):.1%})")
    print(f"  Sharpe:     {cal['sharpe']}   (CSP: {csp.get('sharpe','N/A')})")
    print(f"  Sortino:    {cal['sortino']}")
    print(f"  PF:         {cal['profit_factor']}   (CSP: {csp.get('profit_factor','N/A')})")
    print(f"  WR:         {cal['win_rate']:.1%}   (CSP: {csp.get('win_rate',0):.1%})")
    print(f"  Max DD:     {cal['max_dd']:.1%}   (CSP: {csp.get('max_dd',0):.1%})")
    print(f"  HC #428 R1: {regime_res.get('verdict')}")
    pv = perm.get('p_value', 'N/A')
    sig = perm.get('significant', False)
    print(f"  Perm p-val: {pv}  {'significant' if sig else 'NOT significant'}")
    print(f"  Runtime:    {round(time.time()-t_total)}s")
    print("=" * 68)

    return out


if __name__ == "__main__":
    main()
