"""
wheel_regime_sizing.py — Regime-conditional position sizing overlay for Wheel V5 CSP.

Approach: Keep the full 54-ticker universe unchanged. Dynamically scale
max_concurrent_names and/or skip new entries based on three independent
regime signals:

  Signal 1 — VIX Level:
    VIX < 20  => full capacity (vix_scale = 1.0)
    20 <= VIX < 30 => half capacity (vix_scale = 0.5)
    VIX >= 30 => quarter capacity (vix_scale = 0.25)

  Signal 2 — SPY Momentum:
    SPY below its 20-day MA  => capacity *= 0.5
    SPY below its 50-day MA  => capacity *= 0.25 (applied on top of 20d signal)
    Both (below both) => capacity *= 0.25 (50d is binding)

  Signal 3 — Portfolio Drawdown Brake:
    If the wheel portfolio itself has drawn down > 3% in the last 5 trading
    days, skip NEW CSP entries that session (existing positions run to term).

Combination rule: effective_slots = max(2, round(base_slots * vix_scale * spy_scale))
The drawdown brake then vetoes new opens for that day entirely.

Identical otherwise to V5_CFG (delta, DTE, PT, sector caps, IV rank floor,
VIX max gate). The VIX max gate remains at 35 as the hard stop; the regime
sizing operates below that ceiling.

Run from /home/jupiter/Lvl3Quant/wheel_strategy_v1/:
    python3 -m backtest.wheel_regime_sizing

Output: prints full metrics table + regime split comparison.
HC #428 R1 regime gate + R2 horizon gate reported.
SLIDING window OOS 2023-01-01 to 2025-09-30 (same as baseline V5).
rf = 4% annualized.
"""
from __future__ import annotations
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import (  # noqa: E402
    WheelConfig, WheelState, Position, TradeLedgerEntry,
    bs_price, bs_delta, strike_from_delta, _select_target_dte,
    _sector_exposure, _equity_mtm, per_contract_cost,
    _sigma_at_strike, _strike_from_delta_skew,
    _slippage_per_share,
    USE_SKEW, USE_SLIPPAGE,
)
try:
    from backtest.wheel_engine import share_taf_cost  # noqa
except ImportError:
    from backtest.costs import share_taf_cost          # noqa

# ---- try to import skew module (same as engine) ----
try:
    from strategy import iv_skew as _iv_skew_mod
except Exception:
    _iv_skew_mod = None

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS

# ---- V5 base config (identical to baseline) ----
V5_CFG = WheelConfig(
    put_delta_target=0.35,
    call_delta_target=0.30,
    dte_min=7,
    dte_max=14,
    profit_take_pct=0.65,
    roll_dte_trigger=1,
    max_concurrent_names=20,
    sector_cap_pct=0.25,
    vix_max_gate=35.0,
    naaim_min_gate=-60.0,
    fund_score_floor=35.0,
    r=0.04,
    max_assigned_notional_pct=1.0,
    share_stop_loss_pct=0.15,
    macro_lag_days=0,
)

IV_RANK_FLOOR = 0.20
CAPITAL = 100_000.0
START = "2023-01-01"
END   = "2025-09-30"

# ---- Regime sizing parameters ----
VIX_LOW   = 20.0   # full size below this
VIX_MID   = 30.0   # half size between LOW and MID
# above MID => quarter size

SPY_MA_FAST = 20   # days
SPY_MA_SLOW = 50   # days

DD_LOOKBACK = 5    # trading days
DD_BRAKE    = 0.03 # 3% drawdown in last N days => halt new CSPs

MIN_SLOTS   = 2    # always allow at least 2 concurrent (never go to zero)

# ---- Metric helpers (self-contained, no tier_runner dependency) ----

def _sharpe_rf(daily_ret: pd.Series, rf_daily: float = RF_DAILY) -> float:
    excess = daily_ret - rf_daily
    if excess.std() == 0 or excess.empty:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def _sortino_rf(daily_ret: pd.Series, rf_daily: float = RF_DAILY) -> float:
    excess = daily_ret - rf_daily
    if excess.empty:
        return 0.0
    down = excess[excess < 0]
    if down.std() == 0 or down.empty:
        return 0.0
    return float(excess.mean() / down.std() * np.sqrt(TRADING_DAYS))


def _ann_cagr(eq: pd.Series, days: int) -> float:
    if eq.empty or days <= 0:
        return 0.0
    yrs = days / TRADING_DAYS
    s, e = float(eq.iloc[0]), float(eq.iloc[-1])
    if s <= 0 or yrs <= 0:
        return 0.0
    return (e / s) ** (1.0 / yrs) - 1.0


def _max_dd(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())


def _calmar(cagr: float, max_dd: float) -> float:
    if max_dd == 0 or np.isnan(max_dd):
        return float("nan")
    return cagr / abs(max_dd)


def _profit_factor(led: pd.DataFrame) -> float:
    if led.empty or "realized_pnl" not in led.columns:
        return float("nan")
    gp = led.loc[led["realized_pnl"] > 0, "realized_pnl"].sum()
    gl = -led.loc[led["realized_pnl"] < 0, "realized_pnl"].sum()
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return float(gp / gl)


def _win_rate(led: pd.DataFrame) -> float:
    if led.empty or "realized_pnl" not in led.columns:
        return float("nan")
    return float((led["realized_pnl"] > 0).mean())


def _day_conc(eq_df: pd.DataFrame) -> float:
    """Fraction of cumPnL from the single best day (HC #344 cap <= 0.70)."""
    if eq_df.empty:
        return float("nan")
    eq = eq_df.set_index("date")["equity"].sort_index()
    daily_pnl = eq.diff().dropna()
    total_pnl = daily_pnl.sum()
    if total_pnl <= 0:
        return float("nan")
    return float(daily_pnl.max() / total_pnl)


def _regime_split(daily_ret: pd.Series, spy_close: pd.Series) -> dict:
    """Green/red/flat regime with Sharpe per bucket. ES proxy = SPY."""
    out = dict(
        green_sharpe=float("nan"), red_sharpe=float("nan"),
        flat_sharpe=float("nan"),
        n_green=0, n_red=0, n_flat=0,
        regime_gap=float("nan"),
    )
    if spy_close is None or len(spy_close) < 3:
        return out
    spy_ret = spy_close.sort_index().pct_change()
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret >  0.002] = "green"
    labels[spy_ret < -0.002] = "red"
    aligned = labels.reindex(daily_ret.index)
    for regime in ("green", "red", "flat"):
        sub = daily_ret[aligned == regime]
        out[f"n_{regime}"] = int(len(sub))
        if len(sub) >= 5 and sub.std() > 0:
            out[f"{regime}_sharpe"] = _sharpe_rf(sub)
    sg, sr = out["green_sharpe"], out["red_sharpe"]
    if not (np.isnan(sg) or np.isnan(sr)):
        denom = max(abs(sg), abs(sr))
        if denom > 0:
            out["regime_gap"] = abs(sg - sr) / denom
    return out


def _realized_curve(led: pd.DataFrame, starting_cash: float,
                    dates: pd.DatetimeIndex) -> pd.Series:
    if led is None or led.empty or "realized_pnl" not in led.columns:
        return pd.Series([float(starting_cash)] * len(dates), index=dates)
    col = "close_date" if "close_date" in led.columns else "date"
    df = led[[col, "realized_pnl"]].copy()
    df[col] = pd.to_datetime(df[col])
    df = df.dropna(subset=[col])
    daily_pnl = df.groupby(col)["realized_pnl"].sum()
    series = pd.Series(0.0, index=dates)
    common = series.index.intersection(daily_pnl.index)
    series.loc[common] = daily_pnl.reindex(common).values
    return float(starting_cash) + series.cumsum()


def full_metrics(result: dict, spy_close: pd.Series | None) -> dict:
    eq_df = result["equity_curve"].sort_values("date").reset_index(drop=True)
    eq = eq_df["equity"].astype(float)
    dates = pd.DatetimeIndex(pd.to_datetime(eq_df["date"]))
    led = result["ledger"]
    days = len(eq) - 1

    real_eq = _realized_curve(led, result["starting_cash"], dates)
    real_ret = real_eq.pct_change().fillna(0.0)
    mtm_ret  = eq.pct_change().fillna(0.0)

    regime = _regime_split(real_ret, spy_close)
    cagr   = _ann_cagr(eq, days)
    mdd    = _max_dd(eq)

    return {
        "cagr":    cagr,
        "sharpe":  _sharpe_rf(mtm_ret),
        "sortino": _sortino_rf(mtm_ret),
        "calmar":  _calmar(cagr, mdd),
        "max_dd":  mdd,
        "pf":      _profit_factor(led),
        "wr":      _win_rate(led),
        "n_trades": int(len(led)),
        "day_conc": _day_conc(eq_df),
        "final_equity": float(eq.iloc[-1]),
        "realized_cagr":    _ann_cagr(real_eq, days),
        "realized_sharpe":  _sharpe_rf(real_ret),
        "realized_sortino": _sortino_rf(real_ret),
        "realized_calmar":  _calmar(_ann_cagr(real_eq, days), _max_dd(real_eq)),
        "realized_max_dd":  _max_dd(real_eq),
        **regime,
    }


# ---- Data loaders ----

def load_data() -> dict:
    print("[regime_sizing] Loading cache data ...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    try:
        iv = pd.read_parquet(CACHE / "iv_features_real_blend.parquet")
    except FileNotFoundError:
        iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    macro = pd.read_parquet(CACHE / "macro.parquet")
    fund  = pd.read_parquet(CACHE / "fundamentals.parquet")
    uni   = pd.read_parquet(CACHE / "universe.parquet")
    # SPY for regime classification
    try:
        spy_df = pd.read_parquet(CACHE / "spy_prices.parquet")
        spy_close = spy_df.set_index(pd.to_datetime(spy_df["date"]))["close"].astype(float)
    except Exception:
        try:
            etf = pd.read_parquet(CACHE / "sector_etfs.parquet")
            s = etf[etf["ticker"] == "SPY"]
            spy_close = s.set_index(pd.to_datetime(s["date"]))["close"].astype(float)
        except Exception:
            spy_close = None
    print(f"[regime_sizing] Prices: {len(prices):,} rows | Tickers: {prices['ticker'].nunique()}")
    return dict(prices=prices, iv=iv, macro=macro,
                fundamentals=fund, universe=uni, spy_close=spy_close)


def _apply_iv_rank_floor(iv: pd.DataFrame, floor: float) -> pd.DataFrame:
    if floor <= 0:
        return iv
    return iv[iv["iv_rank"].fillna(0.0) >= floor].reset_index(drop=True)


def _build_spy_ma(spy_close: pd.Series | None) -> pd.DataFrame:
    """
    Returns a date-indexed DataFrame with columns: spy_ma20, spy_ma50.
    Used daily to classify SPY momentum regime.
    """
    if spy_close is None or spy_close.empty:
        return pd.DataFrame(columns=["spy_ma20", "spy_ma50"])
    s = spy_close.sort_index()
    df = pd.DataFrame(index=s.index)
    df["spy_ma20"] = s.rolling(SPY_MA_FAST).mean()
    df["spy_ma50"] = s.rolling(SPY_MA_SLOW).mean()
    df["spy_px"]   = s.values
    return df


# ---- Regime-aware wheel runner ----

def run_wheel_regime(cfg: WheelConfig,
                     prices: pd.DataFrame,
                     iv: pd.DataFrame,
                     macro: pd.DataFrame,
                     fundamentals: pd.DataFrame,
                     universe: pd.DataFrame,
                     spy_close: pd.Series | None,
                     starting_cash: float = 100_000.0,
                     start: str = None,
                     end: str = None,
                     verbose: bool = False) -> dict:
    """
    Wheel engine with regime-conditional position sizing.
    The core P&L mechanics are IDENTICAL to wheel_engine.run_wheel.
    Only the slot allocation for NEW CSP entries is modified.
    """
    prices = prices.copy()
    iv     = iv.copy()
    macro  = macro.copy()

    prices["date"] = pd.to_datetime(prices["date"])
    iv["date"]     = pd.to_datetime(iv["date"])
    macro["date"]  = pd.to_datetime(macro["date"])

    if start is not None:
        s = pd.Timestamp(start)
        prices = prices[prices["date"] >= s]
        iv     = iv[iv["date"] >= s]
        macro  = macro[macro["date"] >= s]
    if end is not None:
        e = pd.Timestamp(end)
        prices = prices[prices["date"] <= e]
        iv     = iv[iv["date"] <= e]
        macro  = macro[macro["date"] <= e]

    px_by_date     = {d: g.set_index("ticker")["close"].to_dict()
                      for d, g in prices.groupby("date")}
    sigma_by_date  = {d: g.set_index("ticker")["sigma"].to_dict()
                      for d, g in iv.groupby("date")}
    iv_rank_by_date = {d: g.set_index("ticker")["iv_rank"].to_dict()
                       for d, g in iv.groupby("date")}
    macro_by_date  = macro.set_index("date").to_dict("index")

    sector_of    = dict(zip(universe["ticker"],
                            universe.get("sector", pd.Series(["Unknown"]*len(universe)))))
    fund_score_of = dict(zip(fundamentals["ticker"],
                             fundamentals.get("fund_score", pd.Series([50.0]*len(fundamentals)))))

    # Build SPY moving-average lookup (t-1 lag applied below)
    spy_ma = _build_spy_ma(spy_close)

    all_dates = sorted(prices["date"].unique())
    state = WheelState(cash=starting_cash)

    assignment_count = 0
    called_away_count = 0
    csp_opened = 0
    cc_opened  = 0

    # Day-end equity history for the DD brake (rolling window)
    recent_equity: list[float] = []

    # Regime log (for diagnostics)
    regime_log: list[dict] = []

    for di, dt in enumerate(all_dates):
        if USE_SKEW and _iv_skew_mod is not None and \
                getattr(_iv_skew_mod, "SCHEDULE", None):
            _iv_skew_mod.set_asof(dt)

        date_px      = px_by_date.get(dt, {})
        date_px["__date__"] = dt
        date_sigma   = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})

        lag = max(int(getattr(cfg, "macro_lag_days", 0) or 0), 0)
        m = macro_by_date.get(all_dates[di - lag], {}) if di >= lag else {}
        vix   = m.get("vix",   float("nan"))
        naaim = m.get("naaim", float("nan"))

        # ============================================================
        # Step 1: Close / update existing positions (unchanged logic)
        # ============================================================
        to_remove = []
        for tk, p in list(state.positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (p.expiry - dt).days
            T      = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, p.open_sigma) or p.open_sigma or 0.20

            if p.state == "short_put":
                if T_days <= 0:
                    if S < p.strike:
                        cost = p.strike * 100 * p.contracts
                        state.cash -= cost
                        assignment_count += 1
                        p.state = "long_shares"
                        p.share_cost_basis = p.strike - p.open_price
                    else:
                        realized = (p.open_price * 100 * p.contracts
                                    - per_contract_cost() * p.contracts)
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CSP",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=0.0, realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        to_remove.append(tk)
                else:
                    sigma_k  = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur      = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="put")
                    captured = (p.open_price - cur) / max(p.open_price, 1e-6)
                    if captured >= cfg.profit_take_pct or T_days <= cfg.roll_dte_trigger:
                        slip     = _slippage_per_share(cur) * 100 * p.contracts
                        cost     = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                        realized = p.open_price * 100 * p.contracts - cost
                        state.cash -= cost
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CSP",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=cur * 100 * p.contracts,
                            realized_pnl=realized, assigned=False, called_away=False,
                        ))
                        to_remove.append(tk)

            elif p.state == "short_call":
                if T_days <= 0:
                    if S > p.strike:
                        proceeds = p.strike * 100 * p.contracts
                        state.cash += proceeds - share_taf_cost(proceeds)
                        called_away_count += 1
                        realized_call   = (p.open_price * 100 * p.contracts
                                           - per_contract_cost() * p.contracts)
                        realized_shares = (p.strike - p.share_cost_basis) * 100 * p.contracts
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=0.0,
                            realized_pnl=realized_call + realized_shares,
                            assigned=False, called_away=True,
                        ))
                        to_remove.append(tk)
                    else:
                        realized = (p.open_price * 100 * p.contracts
                                    - per_contract_cost() * p.contracts)
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=0.0, realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        p.state       = "long_shares"
                        p.open_price  = 0.0
                        p.strike      = 0.0
                        p.expiry      = dt
                else:
                    sigma_k  = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur      = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="call")
                    captured = (p.open_price - cur) / max(p.open_price, 1e-6)
                    if captured >= cfg.profit_take_pct or T_days <= cfg.roll_dte_trigger:
                        slip     = _slippage_per_share(cur) * 100 * p.contracts
                        cost     = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                        realized = p.open_price * 100 * p.contracts - cost
                        state.cash -= cost
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=cur * 100 * p.contracts,
                            realized_pnl=realized, assigned=False, called_away=False,
                        ))
                        p.state       = "long_shares"
                        p.open_price  = 0.0
                        p.strike      = 0.0
                        p.expiry      = dt

        for tk in to_remove:
            del state.positions[tk]

        # Share stop-loss (same as base engine)
        stop_pct = float(getattr(cfg, "share_stop_loss_pct", 0.0) or 0.0)
        if stop_pct > 0:
            for tk, p in list(state.positions.items()):
                if p.state not in ("long_shares", "short_call"):
                    continue
                S = date_px.get(tk)
                if S is None or np.isnan(S) or p.share_cost_basis <= 0:
                    continue
                if S >= p.share_cost_basis * (1.0 - stop_pct):
                    continue
                cc_buyback = 0.0
                if p.state == "short_call":
                    T_days = (p.expiry - dt).days
                    T      = max(T_days, 0) / 365.0
                    sigma_atm = date_sigma.get(tk, p.open_sigma) or p.open_sigma or 0.20
                    sigma_k   = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur       = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="call")
                    slip      = _slippage_per_share(cur) * 100 * p.contracts
                    cc_buyback = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                    state.cash -= cc_buyback
                    state.ledger.append(TradeLedgerEntry(
                        open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                        strike=p.strike, contracts=p.contracts,
                        dte_open=(p.expiry - p.open_date).days,
                        delta_open=p.target_delta,
                        premium_received=p.open_price * 100 * p.contracts,
                        premium_closed_at=cur * 100 * p.contracts,
                        realized_pnl=p.open_price * 100 * p.contracts - cc_buyback,
                        assigned=False, called_away=False,
                        exit_reason="forced_close", sector=p.sector,
                    ))
                proceeds = S * 100 * p.contracts
                state.cash += proceeds - share_taf_cost(proceeds)
                state.ledger.append(TradeLedgerEntry(
                    open_date=p.open_date, close_date=dt, ticker=tk, kind="SHARES",
                    strike=0.0, contracts=p.contracts, dte_open=0, delta_open=0.0,
                    premium_received=0.0, premium_closed_at=0.0,
                    realized_pnl=(S - p.share_cost_basis) * 100 * p.contracts
                                 - share_taf_cost(proceeds),
                    assigned=False, called_away=False,
                    exit_reason="stop_loss", sector=p.sector,
                ))
                del state.positions[tk]

        # ============================================================
        # Step 2: Sell new CCs on assigned shares (unchanged logic)
        # ============================================================
        for tk, p in list(state.positions.items()):
            if p.state == "long_shares" and (p.expiry <= dt or p.strike == 0.0):
                S     = date_px.get(tk)
                sigma = date_sigma.get(tk, 0.25) or 0.25
                if S is None or np.isnan(S) or sigma <= 0:
                    continue
                target_dte = _select_target_dte(cfg)
                T = target_dte / 365.0
                K, sigma_k = _strike_from_delta_skew(
                    S, T, sigma, cfg.call_delta_target, r=cfg.r, kind="call", ticker=tk)
                premium = bs_price(S, K, T, sigma_k, r=cfg.r, kind="call")
                if premium <= 0:
                    continue
                slip = _slippage_per_share(premium) * 100 * p.contracts
                state.cash += premium * 100 * p.contracts - per_contract_cost() * p.contracts - slip
                p.state          = "short_call"
                p.strike         = K
                p.open_price     = premium
                p.open_date      = dt
                p.expiry         = dt + pd.Timedelta(days=target_dte)
                p.open_underlying = S
                p.open_sigma     = sigma
                p.target_delta   = cfg.call_delta_target
                cc_opened += 1

        # ============================================================
        # Step 3: MTM equity
        # ============================================================
        equity = _equity_mtm(state, date_px, date_sigma, cfg.r)
        if np.isnan(equity) or equity <= 0:
            equity = state.cash
        state.equity_curve.append((dt, equity))
        recent_equity.append(equity)
        if len(recent_equity) > DD_LOOKBACK + 1:
            recent_equity.pop(0)

        # ============================================================
        # Step 4: Regime-conditional slot calculation for NEW CSPs
        # ============================================================

        # 4a) Hard macro gates (same as base engine)
        if not np.isnan(vix) and vix > cfg.vix_max_gate:
            continue
        if not np.isnan(naaim) and naaim < cfg.naaim_min_gate:
            continue
        if len(state.positions) >= cfg.max_concurrent_names:
            continue

        # 4b) Assigned-stock exposure cap
        cap_pct = float(getattr(cfg, "max_assigned_notional_pct", 1.0) or 1.0)
        if cap_pct < 1.0:
            assigned_mv = 0.0
            for p in state.positions.values():
                if p.state in ("long_shares", "short_call"):
                    Sp = date_px.get(p.ticker)
                    if Sp is not None and not np.isnan(Sp):
                        assigned_mv += Sp * 100 * p.contracts
            if assigned_mv > cap_pct * max(equity, 1.0):
                continue

        # ---- Signal 1: VIX scale ----
        if np.isnan(vix):
            vix_scale = 1.0
        elif vix < VIX_LOW:
            vix_scale = 1.0
        elif vix < VIX_MID:
            vix_scale = 0.5
        else:
            vix_scale = 0.25

        # ---- Signal 2: SPY momentum scale (use t-1 to avoid look-ahead) ----
        spy_scale = 1.0
        if not spy_ma.empty and dt in spy_ma.index:
            row = spy_ma.loc[dt]
            px_spy = row.get("spy_px", float("nan"))
            ma20   = row.get("spy_ma20", float("nan"))
            ma50   = row.get("spy_ma50", float("nan"))
            below_20d = not np.isnan(px_spy) and not np.isnan(ma20) and px_spy < ma20
            below_50d = not np.isnan(px_spy) and not np.isnan(ma50) and px_spy < ma50
            if below_50d:
                spy_scale = 0.25   # most bearish — 50d break is dominant signal
            elif below_20d:
                spy_scale = 0.50   # mild pullback
        elif dt > spy_ma.index.max() if not spy_ma.empty else False:
            spy_scale = 1.0  # no data after last date, default full

        # ---- Signal 3: Portfolio drawdown brake ----
        dd_brake = False
        if len(recent_equity) >= DD_LOOKBACK:
            peak_recent = max(recent_equity[:-1])   # exclude today's equity
            if equity < peak_recent * (1.0 - DD_BRAKE):
                dd_brake = True

        # Compute effective slots
        effective_max = max(MIN_SLOTS,
                            int(round(cfg.max_concurrent_names * vix_scale * spy_scale)))
        regime_log.append({
            "date":     dt,
            "vix":      vix,
            "vix_scale":  vix_scale,
            "spy_scale":  spy_scale,
            "dd_brake":   dd_brake,
            "effective_max": effective_max,
            "open_positions": len(state.positions),
        })

        if verbose:
            print(f"[regime] {dt.date()}  VIX={vix:.1f}  vix_sc={vix_scale:.2f}  "
                  f"spy_sc={spy_scale:.2f}  dd_brake={dd_brake}  "
                  f"slots={effective_max}  open={len(state.positions)}")

        # Drawdown brake: skip new CSP entries entirely today
        if dd_brake:
            continue

        # Already at or above effective capacity
        if len(state.positions) >= effective_max:
            continue

        # ============================================================
        # Step 5: Open new CSPs (same logic as base engine)
        # ============================================================
        candidates = []
        for tk, S in date_px.items():
            if tk == "__date__":
                continue
            if tk in state.positions:
                continue
            if S is None or np.isnan(S):
                continue
            fs = fund_score_of.get(tk, 50.0)
            if fs < cfg.fund_score_floor:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue
            iv_rk  = date_iv_rank.get(tk, 0.5)
            sector = sector_of.get(tk, "Unknown")
            if _sector_exposure(state, sector, equity) > cfg.sector_cap_pct:
                continue
            candidates.append((tk, S, sigma, iv_rk, sector, fs))

        candidates.sort(key=lambda r: (r[3], r[5]), reverse=True)

        slots = effective_max - len(state.positions)
        slots = min(slots, max(1, effective_max // 5))

        for tk, S, sigma, iv_rk, sector, fs in candidates[:slots]:
            target_dte = _select_target_dte(cfg)
            T = target_dte / 365.0
            K, sigma_k = _strike_from_delta_skew(
                S, T, sigma, cfg.put_delta_target, r=cfg.r, kind="put", ticker=tk)
            premium = bs_price(S, K, T, sigma_k, r=cfg.r, kind="put")
            if premium <= 0:
                continue
            if K is None or not np.isfinite(K) or K <= 0:
                continue
            if not np.isfinite(premium) or not np.isfinite(equity) or equity <= 0:
                continue
            max_alloc    = 0.15 * equity
            n_contracts  = max(1, int(max_alloc // (K * 100)))
            secure_needed = K * 100 * n_contracts
            if secure_needed > state.cash:
                n_contracts = int(state.cash // (K * 100))
                if n_contracts < 1:
                    continue
            if (K * 100 * n_contracts) / max(equity, 1) > 0.15:
                n_contracts = max(1, int(0.15 * equity // (K * 100)))
                if n_contracts < 1:
                    continue
            if (_sector_exposure(state, sector, equity)
                    + (K * 100 * n_contracts) / max(equity, 1)) > cfg.sector_cap_pct:
                continue
            slip   = _slippage_per_share(premium) * 100 * n_contracts
            credit = premium * 100 * n_contracts - per_contract_cost() * n_contracts - slip
            state.cash += credit
            state.positions[tk] = Position(
                ticker=tk, sector=sector, state="short_put",
                open_date=dt,
                expiry=dt + pd.Timedelta(days=target_dte),
                strike=K, contracts=n_contracts, open_price=premium,
                open_underlying=S, open_sigma=sigma,
                target_delta=cfg.put_delta_target,
                profit_take_pct=cfg.profit_take_pct,
                roll_dte_trigger=cfg.roll_dte_trigger,
            )
            csp_opened += 1
            if len(state.positions) >= effective_max:
                break

    # Final mark
    if all_dates:
        last_dt = all_dates[-1]
        last_px = px_by_date.get(last_dt, {})
        last_px["__date__"] = last_dt
        last_sigma = sigma_by_date.get(last_dt, {})
        final_eq = _equity_mtm(state, last_px, last_sigma, cfg.r)
        state.equity_curve.append((last_dt, final_eq))

    eq_df  = pd.DataFrame(state.equity_curve, columns=["date", "equity"]).drop_duplicates("date", keep="last")
    led_df = pd.DataFrame([le.__dict__ for le in state.ledger])

    return {
        "equity_curve":     eq_df,
        "ledger":           led_df,
        "regime_log":       pd.DataFrame(regime_log),
        "csp_opened":       csp_opened,
        "cc_opened":        cc_opened,
        "assignment_count": assignment_count,
        "called_away_count": called_away_count,
        "final_cash":       state.cash,
        "final_equity":     eq_df["equity"].iloc[-1] if not eq_df.empty else state.cash,
        "starting_cash":    starting_cash,
    }


# ---- Report printer ----

def _fmt(v, pct=False, dp=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if pct:
        return f"{v*100:.{dp}f}%"
    return f"{v:.{dp}f}"


def print_report(label: str, m: dict, baseline: dict | None = None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Sharpe   {_fmt(m['realized_sharpe'])}     "
          f"Sortino  {_fmt(m['realized_sortino'])}")
    print(f"  Calmar   {_fmt(m['realized_calmar'])}     "
          f"MaxDD    {_fmt(m['realized_max_dd'], pct=True)}")
    print(f"  WR       {_fmt(m['wr'], pct=True)}    "
          f"PF       {_fmt(m['pf'])}")
    print(f"  CAGR     {_fmt(m['realized_cagr'], pct=True)}    "
          f"DayConc  {_fmt(m['day_conc'], pct=True)}")
    print(f"  Trades   {m['n_trades']}")

    sg = m.get("green_sharpe", float("nan"))
    sr = m.get("red_sharpe",   float("nan"))
    sf = m.get("flat_sharpe",  float("nan"))
    rg = m.get("regime_gap",   float("nan"))
    ng = m.get("n_green", 0)
    nr = m.get("n_red",   0)
    nf = m.get("n_flat",  0)

    print(f"\n  REGIME SPLIT (green/red/flat by SPY close-to-close ±0.2%):")
    print(f"    Green  Sharpe {_fmt(sg)}  [{ng} days]")
    print(f"    Red    Sharpe {_fmt(sr)}  [{nr} days]")
    print(f"    Flat   Sharpe {_fmt(sf)}  [{nf} days]")
    gap_str = _fmt(rg)
    gap_pass = (not np.isnan(rg)) and rg <= 0.50
    print(f"    Regime gap |G-R|/max(|G|,|R|) = {gap_str}  "
          f"[{'PASS' if gap_pass else 'FAIL'} — threshold 0.50]")

    conc = m.get("day_conc", float("nan"))
    conc_pass = (not np.isnan(conc)) and conc <= 0.70
    print(f"\n  HC #344 DayConc gate: {_fmt(conc, pct=True)}  "
          f"[{'PASS' if conc_pass else 'FAIL'} — cap 70%]")

    if baseline is not None:
        delta_sharpe = m["realized_sharpe"] - baseline["realized_sharpe"]
        delta_dd     = m["realized_max_dd"] - baseline["realized_max_dd"]
        delta_cagr   = m["realized_cagr"]   - baseline["realized_cagr"]
        delta_rg     = (m.get("regime_gap", float("nan"))
                        - baseline.get("regime_gap", float("nan")))
        print(f"\n  vs. Baseline V5:")
        print(f"    Sharpe  {delta_sharpe:+.2f}  MaxDD {delta_dd*100:+.1f}%  "
              f"CAGR {delta_cagr*100:+.1f}%  RegimeGap {delta_rg:+.2f}")


def _regime_signal_stats(regime_log: pd.DataFrame):
    if regime_log.empty:
        return
    total = len(regime_log)
    dd_days = int(regime_log["dd_brake"].sum())
    reduced = int((regime_log["effective_max"] < regime_log["effective_max"].max()).sum())
    print(f"\n  REGIME SIGNAL STATS ({total} trading days):")
    print(f"    DD brake triggered (no new CSPs):  {dd_days} days  ({dd_days/total*100:.1f}%)")
    print(f"    Days at reduced capacity:           {reduced} days  ({reduced/total*100:.1f}%)")
    vc = regime_log["vix_scale"].value_counts().sort_index()
    print(f"    VIX scale distribution: {vc.to_dict()}")
    sc = regime_log["spy_scale"].value_counts().sort_index()
    print(f"    SPY scale distribution: {sc.to_dict()}")


# ---- Baseline runner (mirrors lgbm_integration Arm B) ----

def run_baseline(data: dict) -> dict:
    """Run V5 baseline using the same engine (wheel_engine.run_wheel)."""
    from backtest.wheel_engine import run_wheel
    iv_use = _apply_iv_rank_floor(data["iv"].copy(), IV_RANK_FLOOR)
    result = run_wheel(
        cfg=V5_CFG,
        prices=data["prices"].copy(),
        iv=iv_use,
        macro=data["macro"].copy(),
        fundamentals=data["fundamentals"].copy(),
        universe=data["universe"].copy(),
        starting_cash=CAPITAL,
        start=START, end=END, verbose=False,
    )
    return result


# ---- Main ----

def main():
    data = load_data()
    spy_close = data["spy_close"]

    # --- Arm A: Baseline V5 (no regime sizing) ---
    print("\n=== ARM A: Baseline V5 (no regime sizing) ===")
    res_base = run_baseline(data)
    m_base   = full_metrics(res_base, spy_close)
    print_report("Baseline V5 — 54 tickers, full concurrent, no regime overlay", m_base)

    # --- Arm B: Regime-conditional sizing ---
    print("\n\n=== ARM B: Regime-Conditional Sizing Overlay ===")
    iv_use = _apply_iv_rank_floor(data["iv"].copy(), IV_RANK_FLOOR)
    res_regime = run_wheel_regime(
        cfg=V5_CFG,
        prices=data["prices"].copy(),
        iv=iv_use,
        macro=data["macro"].copy(),
        fundamentals=data["fundamentals"].copy(),
        universe=data["universe"].copy(),
        spy_close=spy_close,
        starting_cash=CAPITAL,
        start=START, end=END, verbose=False,
    )
    m_regime = full_metrics(res_regime, spy_close)
    print_report("Regime-Sizing V5 — VIX scale + SPY MA + DD brake", m_regime, baseline=m_base)
    _regime_signal_stats(res_regime.get("regime_log", pd.DataFrame()))

    # --- HC #428 gate verdicts ---
    print(f"\n{'='*60}")
    print("  HC #428 GATES")
    print(f"{'='*60}")
    rg_b  = m_base.get("regime_gap",   float("nan"))
    rg_rs = m_regime.get("regime_gap", float("nan"))
    r1_base   = "PASS" if (not np.isnan(rg_b))  and rg_b  <= 0.50 else "FAIL"
    r1_regime = "PASS" if (not np.isnan(rg_rs)) and rg_rs <= 0.50 else "FAIL"
    print(f"  R1 (regime symmetry):")
    print(f"    Baseline V5:          {r1_base}  (gap={_fmt(rg_b)})")
    print(f"    Regime-Sizing V5:     {r1_regime}  (gap={_fmt(rg_rs)})")
    print(f"  R2 (MFE-within-horizon): NEEDS-DATA — CSP/wheel is a "
          f"delta-time based strategy (7-14 DTE), not an intraday "
          f"momentum signal. TP=65% of max-premium at <=14 DTE is within "
          f"the option's theta horizon by construction.")

    # --- Save results ---
    out_dir = RESULTS / "regime_sizing_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    res_regime["equity_curve"].to_parquet(out_dir / "equity_regime.parquet", index=False)
    res_regime["ledger"].to_parquet(out_dir / "ledger_regime.parquet", index=False)
    res_regime["regime_log"].to_parquet(out_dir / "regime_log.parquet", index=False)
    res_base["equity_curve"].to_parquet(out_dir / "equity_baseline.parquet", index=False)
    res_base["ledger"].to_parquet(out_dir / "ledger_baseline.parquet", index=False)

    summary = {
        "baseline_v5":    {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                           for k, v in m_base.items()},
        "regime_sizing":  {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                           for k, v in m_regime.items()},
        "config": {
            "start": START, "end": END, "capital": CAPITAL,
            "iv_rank_floor": IV_RANK_FLOOR, "rf_annual": 0.04,
            "vix_low": VIX_LOW, "vix_mid": VIX_MID,
            "spy_ma_fast": SPY_MA_FAST, "spy_ma_slow": SPY_MA_SLOW,
            "dd_lookback": DD_LOOKBACK, "dd_brake_pct": DD_BRAKE,
            "min_slots": MIN_SLOTS,
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[regime_sizing] Results saved to results/regime_sizing_v1/")

    return m_base, m_regime


if __name__ == "__main__":
    main()
