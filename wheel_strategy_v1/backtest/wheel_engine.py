"""
wheel_engine.py — Cash-secured-put + covered-call wheel simulator.

Mechanics:
  STATE = CASH | SHORT_PUT | LONG_SHARES | SHORT_CALL

  CASH -> SHORT_PUT:
    Pick a name that passes gates (fund_score >= floor, vix <= vix_max,
    naaim >= naaim_min, sector allocation under cap, single-name allocation
    under 0.15, max_concurrent not breached). Sell a CSP at target delta /
    target DTE. Premium credited.

  SHORT_PUT -> CASH:
    Each day, advance time, mark-to-market via BS. If profit_take % hit
    (e.g. 50% of max profit captured) or DTE <= roll_dte_trigger and OTM
    -> close for credit minus residual price. Pocket realized P&L.

  SHORT_PUT -> LONG_SHARES (assignment):
    If at expiry the underlying is below the strike, we get assigned 100
    shares per contract at strike. Cash debited by strike * 100 * contracts.

  LONG_SHARES -> SHORT_CALL:
    Immediately sell a CC at target delta / target DTE. Premium credited.

  SHORT_CALL -> LONG_SHARES (call expires OTM):
    Keep shares, keep premium. Re-sell next cycle.

  SHORT_CALL -> CASH (called away):
    Shares sold at strike. SEC TAF charged on sale notional.

P&L tracking: per-day mark-to-market equity curve. Returns full trade ledger.

Pricing:
  Black-Scholes on the realized 20d vol (annualized) of the underlying.
  r = 0.04, q = dividend yield from fundamentals (0 if missing).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .costs import per_contract_cost, assignment_cost, share_taf_cost

# Skew-aware per-strike IV. Import is at module scope but never required;
# if iv_skew is unavailable the engine falls back to ATM sigma transparently.
try:
    from strategy import iv_skew as _iv_skew_mod  # type: ignore
    from strategy.iv_skew import iv_at_strike as _iv_at_strike_impl  # type: ignore
except Exception:
    try:
        from ..strategy import iv_skew as _iv_skew_mod  # type: ignore
        from ..strategy.iv_skew import iv_at_strike as _iv_at_strike_impl  # type: ignore
    except Exception:
        _iv_skew_mod = None
        _iv_at_strike_impl = None

# Toggleable globally. tier_runner can flip this OFF via wheel_engine.USE_SKEW = False
# to reproduce the old ATM-only behaviour for ablation runs.
USE_SKEW = True

# ---- Slippage model ----
# Real option fills cost a fraction of the bid-ask spread to cross. Without
# a vendor-supplied bid/ask in the cache, approximate the half-spread as a
# fraction of the mid premium with a minimum tick floor.  Per-share cost
# applies BOTH at open (we cross to sell short) AND at close (we cross to buy
# back), so a round-trip CSP that gets closed eats 2× half-spread.
#
# Defaults are calibrated to roughly match ATM/near-ATM options on
# medium-liquidity single names. Toggleable for ablation.
USE_SLIPPAGE = True
SLIPPAGE_FRAC = 0.025       # 2.5% of premium per leg
SLIPPAGE_MIN_TICKS = 0.03   # $0.03/share minimum half-spread (~ $3/contract)


def _slippage_per_share(premium: float) -> float:
    """Estimated half-spread cost per share to cross the option market.
    Returns dollars per underlying share (multiply by 100 to get per-contract)."""
    if not USE_SLIPPAGE or premium is None or premium <= 0:
        return 0.0
    return max(SLIPPAGE_MIN_TICKS, SLIPPAGE_FRAC * premium)


def _sigma_at_strike(sigma_atm: float, S: float, K: float, T: float,
                     ticker: str | None = None) -> float:
    """Skew-bumped sigma at strike K. Falls back to sigma_atm if skew lib missing
    or globally disabled. `ticker` enables per-ticker walk-forward (a, b)
    when a per-ticker schedule is loaded (v9); otherwise it is a no-op."""
    if not USE_SKEW or _iv_at_strike_impl is None or sigma_atm is None:
        return sigma_atm
    try:
        return _iv_at_strike_impl(sigma_atm, S, K, T, ticker=ticker)
    except TypeError:
        # Older iv_skew without the ticker kwarg — keep global behaviour.
        try:
            return _iv_at_strike_impl(sigma_atm, S, K, T)
        except Exception:
            return sigma_atm
    except Exception:
        return sigma_atm


def _strike_from_delta_skew(S, T, sigma_atm, target_delta, r=0.04, q=0.0,
                            kind="put", iters: int = 2,
                            ticker: str | None = None):
    """
    Two-pass strike solver that respects skew. First pass uses ATM sigma;
    second pass re-bumps sigma at K0 and re-solves. Converges in 1-2 iters
    for typical equity wheels (15-40 delta).
    """
    K = strike_from_delta(S, T, sigma_atm, target_delta, r=r, q=q, kind=kind)
    if not USE_SKEW or _iv_at_strike_impl is None:
        return K, sigma_atm
    for _ in range(max(iters, 1)):
        sigma_k = _sigma_at_strike(sigma_atm, S, K, T, ticker=ticker)
        if sigma_k is None or sigma_k <= 0:
            return K, sigma_atm
        K = strike_from_delta(S, T, sigma_k, target_delta, r=r, q=q, kind=kind)
    sigma_k = _sigma_at_strike(sigma_atm, S, K, T, ticker=ticker)
    return K, (sigma_k if sigma_k and sigma_k > 0 else sigma_atm)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

# -------- Black-Scholes --------
SQRT_2PI = math.sqrt(2 * math.pi)
def _phi(x): return math.exp(-0.5 * x * x) / SQRT_2PI
def _Phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if kind == "put":
            return max(K - S, 0.0)
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * math.exp(-q * T) * _Phi(-d1)
    else:
        return S * math.exp(-q * T) * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def bs_delta(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        # at expiry intrinsic
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if kind == "put":
        return math.exp(-q * T) * (_Phi(d1) - 1.0)  # negative
    return math.exp(-q * T) * _Phi(d1)              # positive

def strike_from_delta(S, T, sigma, target_delta, r=0.04, q=0.0, kind="put"):
    """
    Solve for strike K such that |delta| = target_delta. Uses closed form
    inversion of d1: Phi(d1) = target (call) or Phi(d1) = 1 - target (put).
    """
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    # invert Phi via Acklam (good enough)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5 * sigma * sigma) * T))
    return K

def _ndtri(p):
    # Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


# -------- Open positions --------
@dataclass
class Position:
    ticker: str
    sector: str
    state: str            # 'short_put' | 'long_shares' | 'short_call'
    open_date: pd.Timestamp
    expiry: pd.Timestamp
    strike: float
    contracts: int        # = number of 100-share lots
    open_price: float     # option premium received per share
    open_underlying: float
    open_sigma: float
    target_delta: float
    profit_take_pct: float
    roll_dte_trigger: int
    # For long_shares legs (between CC sells)
    share_cost_basis: float = 0.0  # per share

    @property
    def notional(self):
        return self.strike * 100 * self.contracts


@dataclass
class TradeLedgerEntry:
    # --- Existing fields (unchanged for back-compat) ---
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    ticker: str
    kind: str          # 'CSP' | 'CC' | 'SHARES'
    strike: float
    contracts: int
    dte_open: int
    delta_open: float
    premium_received: float   # gross
    premium_closed_at: float  # 0 if expired
    realized_pnl: float
    assigned: bool
    called_away: bool
    # --- HC #544 R5 additions (leakage audit + intraday tracking) ---
    open_underlying: float = 0.0     # underlying spot at entry (NOT a future-bar close)
    close_underlying: float = 0.0    # underlying spot at exit
    open_sigma: float = 0.0          # IV used at entry
    close_sigma: float = 0.0         # IV at exit
    open_time: str = "16:00"         # HH:MM at entry — defaults to close-of-bar; intraday cells override
    close_time: str = "16:00"        # HH:MM at exit
    dte_close: int = 0
    fees_paid: float = 0.0           # commission + reg fees on this trade
    exit_reason: str = ""            # 'expired' | 'profit_take' | 'rolled' | 'assigned' | 'called_away' | 'forced_close'
    entry_bar_used: str = "close"    # 'open' | 'close' | 'next_open' (leakage audit toggles this)
    sector: str = ""


# -------- Engine --------
@dataclass
class WheelConfig:
    put_delta_target: float
    call_delta_target: float
    dte_min: int
    dte_max: int
    profit_take_pct: float       # 0..1 (fraction of max profit before closing)
    roll_dte_trigger: int
    max_concurrent_names: int
    sector_cap_pct: float        # 0..1 (max fraction of equity in any one sector)
    vix_max_gate: float
    naaim_min_gate: float
    fund_score_floor: float
    r: float = 0.04
    # --- wheel_dd_fix_v6 risk controls (defaults preserve legacy behavior) ---
    max_assigned_notional_pct: float = 1.0   # cap: share-position MV / equity. 1.0 = off
    share_stop_loss_pct: float = 0.0         # sell assigned shares if S < basis*(1-x). 0 = off
    macro_lag_days: int = 0                  # use t-N macro for gates (1 = t-1, no look-ahead)


@dataclass
class WheelState:
    cash: float
    equity_curve: list = field(default_factory=list)  # (date, equity)
    positions: dict = field(default_factory=dict)     # ticker -> Position
    ledger: list = field(default_factory=list)


def _select_target_dte(cfg: WheelConfig) -> int:
    return int(round((cfg.dte_min + cfg.dte_max) / 2))


def _sector_exposure(state: WheelState, sector: str, equity: float) -> float:
    # Approx exposure: sum of notionals for positions in the sector.
    sec = 0.0
    for p in state.positions.values():
        if p.sector == sector:
            sec += p.notional
    if equity <= 0:
        return 1.0
    return sec / equity


def _equity_mtm(state: WheelState, date_px: dict, sigmas: dict, r: float) -> float:
    """Mark to market across all positions."""
    equity = state.cash
    for p in state.positions.values():
        S = date_px.get(p.ticker)
        if S is None or np.isnan(S):
            continue
        T = max((p.expiry - date_px["__date__"]).days, 0) / 365.0
        sigma_atm = sigmas.get(p.ticker, p.open_sigma) or p.open_sigma or 0.20
        # BUGFIX 2026-06-10 (wheel_dd_fix_v6): premium credits and share
        # purchases already flow through state.cash at the moment they occur
        # (CSP open credits cash; assignment debits full strike; CC open
        # credits cash; called-away credits strike proceeds). Therefore MTM
        # must ONLY add current asset values and subtract current option
        # liabilities. The old code re-added open premium AND re-subtracted
        # share cost basis, which double-counted assignment cost and crushed
        # NAV by ~full strike notional per assigned name (-84%/-98% phantom
        # drawdowns in the v5_REAL FullWheel books).
        if p.state == "short_put":
            sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=p.ticker)
            opt_val = bs_price(S, p.strike, T, sigma_k, r=r, kind="put")
            equity -= opt_val * 100 * p.contracts          # liability only
        elif p.state == "short_call":
            sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=p.ticker)
            opt_val = bs_price(S, p.strike, T, sigma_k, r=r, kind="call")
            equity += S * 100 * p.contracts                # shares we hold
            equity -= opt_val * 100 * p.contracts          # call liability
        elif p.state == "long_shares":
            equity += S * 100 * p.contracts                # shares we hold
    return equity


def run_wheel(cfg: WheelConfig,
              prices: pd.DataFrame,
              iv: pd.DataFrame,
              macro: pd.DataFrame,
              fundamentals: pd.DataFrame,
              universe: pd.DataFrame,
              starting_cash: float = 100_000.0,
              start: str = None,
              end: str = None,
              verbose: bool = False) -> dict:
    """
    Run the wheel backtest over the joint date range.

    prices: long df (ticker, date, open, high, low, close, ...) with rv_20.
    iv:     long df (date, ticker, sigma, iv_rank).
    macro:  wide df (date, vix, naaim, ...).
    fundamentals: per-ticker (ticker, fund_score, dividend_yield).
    universe: per-ticker (ticker, sector).
    """
    # -- Index everything by date for fast lookups
    prices = prices.copy()
    iv = iv.copy()
    macro = macro.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    iv["date"] = pd.to_datetime(iv["date"])
    macro["date"] = pd.to_datetime(macro["date"])

    if start is not None:
        s = pd.Timestamp(start)
        prices = prices[prices["date"] >= s]
        iv = iv[iv["date"] >= s]
        macro = macro[macro["date"] >= s]
    if end is not None:
        e = pd.Timestamp(end)
        prices = prices[prices["date"] <= e]
        iv = iv[iv["date"] <= e]
        macro = macro[macro["date"] <= e]

    # Per-day price snapshot
    px_by_date = {d: g.set_index("ticker")["close"].to_dict()
                  for d, g in prices.groupby("date")}
    sigma_by_date = {d: g.set_index("ticker")["sigma"].to_dict()
                     for d, g in iv.groupby("date")}
    iv_rank_by_date = {d: g.set_index("ticker")["iv_rank"].to_dict()
                       for d, g in iv.groupby("date")}
    macro_by_date = macro.set_index("date").to_dict("index")

    # Per-ticker static
    sector_of = dict(zip(universe["ticker"], universe.get("sector", pd.Series(["Unknown"]*len(universe)))))
    fund_score_of = dict(zip(fundamentals["ticker"], fundamentals.get("fund_score", pd.Series([50.0]*len(fundamentals)))))
    div_yield_of = dict(zip(fundamentals["ticker"], fundamentals.get("dividend_yield", pd.Series([0.0]*len(fundamentals)))))

    all_dates = sorted(prices["date"].unique())
    state = WheelState(cash=starting_cash)

    assignment_count = 0
    called_away_count = 0
    csp_opened = 0
    cc_opened = 0

    for di, dt in enumerate(all_dates):
        # Walk-forward skew: advance (a, b) to the leak-free period covering
        # this simulated day (no-op unless a schedule was loaded — lane A3).
        if USE_SKEW and _iv_skew_mod is not None and \
                getattr(_iv_skew_mod, "SCHEDULE", None):
            _iv_skew_mod.set_asof(dt)
        date_px = px_by_date.get(dt, {})
        date_px["__date__"] = dt
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        # Macro gates use t-N values when macro_lag_days > 0 (no look-ahead:
        # decisions at today's close use yesterday's published macro).
        lag = max(int(getattr(cfg, "macro_lag_days", 0) or 0), 0)
        if lag > 0:
            m = macro_by_date.get(all_dates[di - lag], {}) if di >= lag else {}
        else:
            m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))
        naaim = m.get("naaim", float("nan"))

        # -- 1) Update / close existing positions
        to_remove = []
        for tk, p in list(state.positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (p.expiry - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, p.open_sigma) or p.open_sigma or 0.20

            if p.state == "short_put":
                if T_days <= 0:
                    # expiry
                    if S < p.strike:
                        # assigned
                        cost = p.strike * 100 * p.contracts
                        state.cash -= cost
                        # net premium kept (already received at open)
                        assignment_count += 1
                        p.state = "long_shares"
                        p.share_cost_basis = p.strike - p.open_price  # net basis
                        # immediately schedule a CC next loop pass
                    else:
                        # expire worthless — keep premium
                        realized = p.open_price * 100 * p.contracts - per_contract_cost() * p.contracts
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
                    # check profit-take — mark at skew-bumped sigma at strike
                    sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="put")
                    captured = (p.open_price - cur) / max(p.open_price, 1e-6)
                    if captured >= cfg.profit_take_pct or T_days <= cfg.roll_dte_trigger:
                        # close — buy back the short put, paying half-spread to cross
                        slip = _slippage_per_share(cur) * 100 * p.contracts
                        cost = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                        realized = p.open_price * 100 * p.contracts - cost
                        # BUGFIX: debit full buy-back cost incl. per-contract fee
                        state.cash -= (cur * 100 * p.contracts
                                       + per_contract_cost() * p.contracts + slip)
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CSP",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=cur * 100 * p.contracts,
                            realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        to_remove.append(tk)

            elif p.state == "short_call":
                if T_days <= 0:
                    if S > p.strike:
                        # called away
                        proceeds = p.strike * 100 * p.contracts
                        state.cash += proceeds - share_taf_cost(proceeds)
                        called_away_count += 1
                        realized_call = p.open_price * 100 * p.contracts - per_contract_cost() * p.contracts
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
                        # CC expires worthless -> back to long_shares (reset for next CC)
                        realized = p.open_price * 100 * p.contracts - per_contract_cost() * p.contracts
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=0.0, realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        # BUGFIX: premium was ALREADY credited to cash when the
                        # CC was sold — do NOT credit it again here.
                        p.state = "long_shares"
                        p.open_price = 0.0
                        p.strike = 0.0
                        p.expiry = dt  # placeholder; will re-sell CC
                else:
                    sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="call")
                    captured = (p.open_price - cur) / max(p.open_price, 1e-6)
                    if captured >= cfg.profit_take_pct or T_days <= cfg.roll_dte_trigger:
                        slip = _slippage_per_share(cur) * 100 * p.contracts
                        cost = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                        realized = p.open_price * 100 * p.contracts - cost
                        # BUGFIX: premium was already credited at open; only the
                        # buy-back cost moves cash now.
                        state.cash -= cost
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=cur * 100 * p.contracts,
                            realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        p.state = "long_shares"
                        p.open_price = 0.0
                        p.strike = 0.0
                        p.expiry = dt

        for tk in to_remove:
            # finalize cash for the closed leg (premium was held; for expired worthless, add it now)
            p = state.positions[tk]
            if p.state == "short_put":
                # we never added premium to cash for short_put; add now if expired worthless
                pass
            del state.positions[tk]

        # -- 1b) Share stop-loss (wheel_dd_fix_v6): liquidate assigned shares
        # if price falls more than share_stop_loss_pct below cost basis.
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
                # Buy back any open CC first.
                cc_buyback = 0.0
                if p.state == "short_call":
                    T_days = (p.expiry - dt).days
                    T = max(T_days, 0) / 365.0
                    sigma_atm = date_sigma.get(tk, p.open_sigma) or p.open_sigma or 0.20
                    sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="call")
                    slip = _slippage_per_share(cur) * 100 * p.contracts
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
                # Sell shares at market close price.
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

        # -- 2) For LONG_SHARES that need a new CC, sell it
        for tk, p in list(state.positions.items()):
            if p.state == "long_shares" and (p.expiry <= dt or p.strike == 0.0):
                S = date_px.get(tk)
                sigma = date_sigma.get(tk, 0.25) or 0.25
                if S is None or np.isnan(S) or sigma <= 0:
                    continue
                target_dte = _select_target_dte(cfg)
                T = target_dte / 365.0
                # Skew-aware strike: solve K and σ(K) jointly, then price at σ(K).
                K, sigma_k = _strike_from_delta_skew(
                    S, T, sigma, cfg.call_delta_target, r=cfg.r, kind="call",
                    ticker=tk,
                )
                premium = bs_price(S, K, T, sigma_k, r=cfg.r, kind="call")
                if premium <= 0:
                    continue
                slip = _slippage_per_share(premium) * 100 * p.contracts
                state.cash += premium * 100 * p.contracts - per_contract_cost() * p.contracts - slip
                p.state = "short_call"
                p.strike = K
                p.open_price = premium
                p.open_date = dt
                p.expiry = dt + pd.Timedelta(days=target_dte)
                p.open_underlying = S
                p.open_sigma = sigma
                p.target_delta = cfg.call_delta_target
                cc_opened += 1

        # -- 3) Try to open new CSPs on names not already in book
        equity = _equity_mtm(state, date_px, date_sigma, cfg.r)
        if np.isnan(equity) or equity <= 0:
            equity = state.cash
        state.equity_curve.append((dt, equity))

        # macro gates
        if not np.isnan(vix) and vix > cfg.vix_max_gate:
            continue
        if not np.isnan(naaim) and naaim < cfg.naaim_min_gate:
            continue
        if len(state.positions) >= cfg.max_concurrent_names:
            continue

        # Assigned-stock exposure cap (wheel_dd_fix_v6): if the market value of
        # held shares exceeds max_assigned_notional_pct of equity, do not add
        # new short puts (which add further assignment risk).
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

        # candidate names = in universe, in price/IV today, fund_score >= floor, not already held
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
            iv_rk = date_iv_rank.get(tk, 0.5)
            sector = sector_of.get(tk, "Unknown")
            # sector cap
            if _sector_exposure(state, sector, equity) > cfg.sector_cap_pct:
                continue
            candidates.append((tk, S, sigma, iv_rk, sector, fs))

        # rank: prefer higher iv_rank then higher fund_score
        candidates.sort(key=lambda r: (r[3], r[5]), reverse=True)

        # how many new positions can we open this day?
        slots = cfg.max_concurrent_names - len(state.positions)
        # don't blast them all open at once — pace
        slots = min(slots, max(1, cfg.max_concurrent_names // 5))

        for tk, S, sigma, iv_rk, sector, fs in candidates[:slots]:
            target_dte = _select_target_dte(cfg)
            T = target_dte / 365.0
            # Skew-aware strike: solve K and σ(K) jointly, then price at σ(K).
            K, sigma_k = _strike_from_delta_skew(
                S, T, sigma, cfg.put_delta_target, r=cfg.r, kind="put",
                ticker=tk,
            )
            premium = bs_price(S, K, T, sigma_k, r=cfg.r, kind="put")
            if premium <= 0:
                continue
            # Defensive NaN guards (sigma can be NaN if data is sparse).
            if K is None or not np.isfinite(K) or K <= 0:
                continue
            if not np.isfinite(premium) or not np.isfinite(equity) or equity <= 0:
                continue
            # contracts = floor(allocation / strike notional). cap allocation per name at 15% of equity.
            max_alloc = 0.15 * equity
            n_contracts = max(1, int(max_alloc // (K * 100)))
            # cash secure: need to be able to cover K * 100 * n if assigned
            secure_needed = K * 100 * n_contracts
            if secure_needed > state.cash:
                # scale down
                n_contracts = int(state.cash // (K * 100))
                if n_contracts < 1:
                    continue
            # double-check single-name cap relative to equity
            if (K * 100 * n_contracts) / max(equity, 1) > 0.15:
                n_contracts = max(1, int(0.15 * equity // (K * 100)))
                if n_contracts < 1:
                    continue
            # sector cap re-check
            if (_sector_exposure(state, sector, equity) + (K * 100 * n_contracts) / max(equity, 1)) > cfg.sector_cap_pct:
                continue
            # open — selling short put, half-spread to cross
            slip = _slippage_per_share(premium) * 100 * n_contracts
            credit = premium * 100 * n_contracts - per_contract_cost() * n_contracts - slip
            state.cash += credit
            state.positions[tk] = Position(
                ticker=tk, sector=sector, state="short_put",
                open_date=dt, expiry=dt + pd.Timedelta(days=target_dte),
                strike=K, contracts=n_contracts, open_price=premium,
                open_underlying=S, open_sigma=sigma,
                target_delta=cfg.put_delta_target,
                profit_take_pct=cfg.profit_take_pct,
                roll_dte_trigger=cfg.roll_dte_trigger,
            )
            csp_opened += 1
            if len(state.positions) >= cfg.max_concurrent_names:
                break

    # Final mark
    if all_dates:
        last_dt = all_dates[-1]
        last_px = px_by_date.get(last_dt, {})
        last_px["__date__"] = last_dt
        last_sigma = sigma_by_date.get(last_dt, {})
        final_eq = _equity_mtm(state, last_px, last_sigma, cfg.r)
        state.equity_curve.append((last_dt, final_eq))

    eq_df = pd.DataFrame(state.equity_curve, columns=["date", "equity"]).drop_duplicates("date", keep="last")
    led_df = pd.DataFrame([le.__dict__ for le in state.ledger])

    return {
        "equity_curve": eq_df,
        "ledger": led_df,
        "csp_opened": csp_opened,
        "cc_opened": cc_opened,
        "assignment_count": assignment_count,
        "called_away_count": called_away_count,
        "final_cash": state.cash,
        "final_equity": eq_df["equity"].iloc[-1] if not eq_df.empty else state.cash,
        "starting_cash": starting_cash,
    }
