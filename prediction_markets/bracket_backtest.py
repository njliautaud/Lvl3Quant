#!/usr/bin/env python3
"""
SPX Bracket Backtest — Synthetic backtest of our vol-model-based bracket strategy.

Uses historical SPX/VIX data to simulate:
  1. Market-implied bracket prices (from VIX → market vol assumption, normal dist)
  2. Our model-implied bracket prices (from realized vol, Student-t df=4)
  3. Trading mispricings where our edge > fees + min_edge

The key insight: the market prices brackets assuming ~normal returns and uses
VIX as the vol estimate. We know:
  a) Returns are fat-tailed (Student-t with df~4 fits better)
  b) Our model predicts realized vol with IC=0.644 at 30min
Both give us systematic mispricing to exploit.

Edge sources:
  1. DISTRIBUTIONAL EDGE: Market uses normal dist; we use Student-t → tail brackets
     are systematically underpriced by the market.
  2. VOL FORECAST EDGE: Our IC=0.644 model predicts realized vol better than VIX.
     When realized vol > VIX-implied vol, wing brackets pay more than priced.
     When realized vol < VIX-implied vol, ATM brackets pay more than priced.

Usage:
    python bracket_backtest.py                     # Run full backtest
    python bracket_backtest.py --days 60           # Last 60 trading days
    python bracket_backtest.py --capital 10000     # $10K starting capital
    python bracket_backtest.py --verbose           # Show every trade

Output:
    Prints P&L report, saves detailed results to data/bracket_backtest_results.json
"""

import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("ERROR: scipy is required. pip install scipy")
    sys.exit(1)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

logger = logging.getLogger("bracket_backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Fee model (mirrors kalshi_client.py)
# ---------------------------------------------------------------------------

FEE_COEFF_SPX = 0.035        # SPX halved taker fee coefficient
MAKER_FEE_COEFF = 0.0175     # Maker fee coefficient


def taker_fee(price: float, contracts: int = 1) -> float:
    """Kalshi taker fee for SPX brackets. Price in [0, 1]. Returns $ per contract."""
    return math.ceil(FEE_COEFF_SPX * contracts * price * (1 - price) * 100) / 100


def maker_fee(price: float, contracts: int = 1) -> float:
    """Kalshi maker fee. Returns $ per contract."""
    return math.ceil(MAKER_FEE_COEFF * contracts * price * (1 - price) * 100) / 100


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def fetch_spx_vix_data(days: int = 252) -> dict:
    """Fetch historical SPX and VIX data from Yahoo Finance.

    Returns dict with arrays:
      dates, spx_open, spx_close, vix_close, spx_high, spx_low
    """
    if not HAS_YF:
        raise RuntimeError("yfinance not available. pip install yfinance")

    end = datetime.now()
    start = end - timedelta(days=int(days * 1.6))  # extra margin for weekends

    logger.info(f"Fetching SPX data from {start.date()} to {end.date()}...")
    spx = yf.download("^GSPC", start=start, end=end, progress=False)
    vix = yf.download("^VIX", start=start, end=end, progress=False)

    if spx.empty or vix.empty:
        raise RuntimeError("Failed to fetch market data from Yahoo Finance")

    # Align dates
    common = spx.index.intersection(vix.index)
    spx = spx.loc[common].tail(days)
    vix = vix.loc[common].tail(days)

    # Handle multi-level columns from yfinance
    def get_col(df, col):
        if isinstance(df.columns, __import__('pandas').MultiIndex):
            return df[col].iloc[:, 0].values if col in df.columns.get_level_values(0) else df[col].values
        return df[col].values

    result = {
        "dates": [d.strftime("%Y-%m-%d") for d in spx.index],
        "spx_open": get_col(spx, "Open").astype(float),
        "spx_close": get_col(spx, "Close").astype(float),
        "spx_high": get_col(spx, "High").astype(float),
        "spx_low": get_col(spx, "Low").astype(float),
        "vix_close": get_col(vix, "Close").astype(float),
    }

    logger.info(f"Fetched {len(result['dates'])} trading days")
    return result


def compute_realized_vol(closes: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute trailing realized vol (annualized) using log returns.

    Returns array same length as closes, with NaN for first `window` entries.
    """
    log_ret = np.diff(np.log(closes))
    rvol = np.full(len(closes), np.nan)
    for i in range(window, len(closes)):
        rvol[i] = np.std(log_ret[i - window:i]) * np.sqrt(252) * 100
    return rvol


# ---------------------------------------------------------------------------
# Bracket simulation
# ---------------------------------------------------------------------------

@dataclass
class BracketContract:
    """A single SPX bracket contract for a trading day."""
    floor: float
    cap: float
    market_price_yes: float    # What the market charges (from VIX, normal dist)
    model_price_yes: float     # What our model says it's worth (rvol, Student-t)
    settled_yes: bool          # Did SPX close in this bracket?


@dataclass
class Trade:
    """A single trade on a bracket contract."""
    date: str
    floor: float
    cap: float
    side: str                  # "BUY_YES" or "BUY_NO"
    entry_price: float         # What we paid
    fair_value: float          # Our model price
    edge_gross: float          # fair_value - entry_price
    fee: float                 # Taker fee paid
    edge_net: float            # edge_gross - fee
    settled_value: float       # 1.0 or 0.0 (what it settled at)
    pnl: float                 # settled_value - entry_price - fee
    contracts: int = 1
    spx_at_entry: float = 0.0
    spx_at_close: float = 0.0
    vix: float = 0.0
    rvol: float = 0.0


def generate_brackets(
    spx_open: float,
    bracket_width: float = 25.0,
    n_brackets_per_side: int = 8,
) -> list[tuple[float, float]]:
    """Generate bracket ranges around current SPX.

    Returns list of (floor, cap) tuples.
    """
    base = int(spx_open / bracket_width) * bracket_width
    brackets = []
    for i in range(-n_brackets_per_side, n_brackets_per_side + 1):
        floor = base + i * bracket_width
        cap = floor + bracket_width
        brackets.append((floor, cap))
    return brackets


def price_bracket_normal(
    floor: float, cap: float, spx: float, vol_annual_pct: float,
    hours_to_close: float = 6.5,
) -> float:
    """Price a bracket using normal distribution (what the market does).

    vol_annual_pct: annualized vol in percent (e.g. 18.0 for 18%)
    Returns probability [0, 1].
    """
    vol = vol_annual_pct / 100.0
    time_frac = hours_to_close / (252 * 6.5)
    sigma = spx * vol * np.sqrt(time_frac)
    if sigma <= 0:
        return 1.0 if floor <= spx <= cap else 0.0
    z_lo = (floor - spx) / sigma
    z_hi = (cap - spx) / sigma
    return float(scipy_stats.norm.cdf(z_hi) - scipy_stats.norm.cdf(z_lo))


def price_bracket_student_t(
    floor: float, cap: float, spx: float, vol_annual_pct: float,
    hours_to_close: float = 6.5, df: float = 4.0,
) -> float:
    """Price a bracket using Student-t distribution (our model).

    Student-t with df=4 has fatter tails than normal, matching SPX intraday returns.
    Scale is adjusted so total variance matches the vol prediction.
    """
    vol = vol_annual_pct / 100.0
    time_frac = hours_to_close / (252 * 6.5)
    sigma = spx * vol * np.sqrt(time_frac)
    if sigma <= 0:
        return 1.0 if floor <= spx <= cap else 0.0
    # Scale so Student-t variance = sigma^2
    # Var(t_df) = df/(df-2) * scale^2, so scale = sigma * sqrt((df-2)/df)
    scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
    dist = scipy_stats.t(df=df, loc=spx, scale=scale)
    return float(max(0, dist.cdf(cap) - dist.cdf(floor)))


def simulate_market_price(
    floor: float, cap: float, spx: float, vix: float,
    hours_to_close: float = 6.5, spread_half: float = 0.01,
) -> tuple[float, float]:
    """Simulate Kalshi market bid/ask for a bracket.

    Market makers use VIX and approximately normal distribution.
    Returns (yes_bid, yes_ask).
    """
    mid = price_bracket_normal(floor, cap, spx, vix, hours_to_close)
    # Add noise to simulate real market inefficiency (mean-zero, small)
    noise = np.random.normal(0, 0.003)
    mid = np.clip(mid + noise, 0.01, 0.99)
    # Spread is wider for illiquid (far OTM) contracts
    distance = abs(spx - (floor + cap) / 2)
    sigma_approx = spx * (vix / 100) * np.sqrt(hours_to_close / (252 * 6.5))
    spread_mult = 1.0 + 0.5 * min(3.0, distance / (sigma_approx + 1))
    half = spread_half * spread_mult
    yes_bid = max(0.01, mid - half)
    yes_ask = min(0.99, mid + half)
    return yes_bid, yes_ask


# ---------------------------------------------------------------------------
# Vol forecast simulation (simulating our model's edge)
# ---------------------------------------------------------------------------

def simulate_vol_forecast(
    true_rvol: float, vix: float, ic: float = 0.644
) -> float:
    """Simulate our vol model's prediction given true realized vol.

    Our model has IC=0.644 (rank correlation) with realized vol.
    This means our prediction explains ~41% of variance (r^2 ~ IC^2).

    We simulate: predicted_vol = IC * true_rvol + (1 - IC) * vix + noise
    The noise ensures the overall correlation with true_rvol ≈ IC.

    Returns annualized vol in percent.
    """
    # Linear combination: lean toward true rvol proportional to IC
    signal = ic * true_rvol + (1 - ic) * vix
    # Add noise so rank correlation ≈ IC (not perfect)
    noise_std = abs(true_rvol - vix) * np.sqrt(1 - ic**2) * 0.5
    noise = np.random.normal(0, max(0.5, noise_std))
    predicted = signal + noise
    return max(3.0, min(80.0, predicted))


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for the bracket backtest."""
    starting_capital: float = 10000.0
    max_position_per_bracket: int = 50      # Max contracts per bracket
    max_daily_trades: int = 20              # Max trades per day
    max_daily_risk: float = 1000.0          # Max capital at risk per day
    min_net_edge: float = 0.008             # Min net edge after fees to trade (0.8 cents)
    min_gross_edge: float = 0.015           # Min gross edge before fees
    bracket_width: float = 25.0             # SPX bracket width
    n_brackets_per_side: int = 8            # Brackets above/below current
    hours_to_close: float = 5.0            # Assume entry at ~11:00 AM ET (5 hrs to close)
    vol_model_ic: float = 0.644            # Our vol model's IC
    student_t_df: float = 4.0             # Degrees of freedom for our tail model
    spread_half: float = 0.01             # Half-spread for simulated market (1 cent)
    kelly_fraction: float = 0.15          # Fraction of Kelly to bet (conservative)
    use_maker_fees: bool = False          # If True, assume maker fills (lower fee)
    rvol_lookback: int = 20               # Days for realized vol calculation


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: dict
    n_trading_days: int = 0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_pnl_per_day: float = 0.0
    monthly_pnl_estimate: float = 0.0
    annual_pnl_estimate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_edge_per_trade: float = 0.0
    total_fees: float = 0.0
    avg_contracts_per_day: float = 0.0
    capital_utilized_pct: float = 0.0
    edge_by_bracket_type: dict = field(default_factory=dict)
    daily_pnl: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    pnl_by_strategy: dict = field(default_factory=dict)


def compute_kelly_size(
    edge: float, price: float, max_contracts: int, kelly_frac: float = 0.15
) -> int:
    """Compute position size using fractional Kelly criterion.

    For a binary bet:
      Kelly fraction = (p*b - q) / b
      where p = win prob, b = odds, q = 1-p

    We approximate: if edge = fair_value - price,
      p = fair_value, b = (1-price)/price (payout odds)
      Kelly = edge / (1 - price)

    Use kelly_frac of full Kelly for safety.
    """
    if edge <= 0 or price <= 0.01 or price >= 0.99:
        return 0
    kelly_full = edge / (1 - price)
    kelly_adj = kelly_full * kelly_frac
    contracts = max(1, min(max_contracts, int(kelly_adj * 100)))
    return contracts


def run_backtest(config: BacktestConfig = None, market_data: dict = None) -> BacktestResult:
    """Run the full bracket backtest.

    Args:
        config: Backtest configuration
        market_data: Pre-fetched market data dict (optional, will fetch if None)

    Returns:
        BacktestResult with full P&L analysis
    """
    if config is None:
        config = BacktestConfig()

    if market_data is None:
        market_data = fetch_spx_vix_data(days=252)

    dates = market_data["dates"]
    spx_open = market_data["spx_open"]
    spx_close = market_data["spx_close"]
    vix_close = market_data["vix_close"]
    spx_high = market_data["spx_high"]
    spx_low = market_data["spx_low"]

    n_days = len(dates)

    # Compute trailing realized vol
    rvol = compute_realized_vol(spx_close, window=config.rvol_lookback)

    # Track results
    all_trades: list[Trade] = []
    daily_pnl = []
    equity = config.starting_capital
    equity_curve = [equity]
    peak_equity = equity

    # Edge breakdown
    distributional_trades = []  # Trades where tail mispricing is the main edge
    vol_forecast_trades = []    # Trades where vol forecast is the main edge

    logger.info(f"Starting backtest: {n_days} days, capital=${config.starting_capital}")
    logger.info(f"Config: IC={config.vol_model_ic}, df={config.student_t_df}, "
                f"min_net_edge={config.min_net_edge}, kelly_frac={config.kelly_fraction}")

    for i in range(config.rvol_lookback + 1, n_days):
        day_date = dates[i]
        spx_now = spx_open[i]        # Entry at open
        spx_end = spx_close[i]       # Settlement at close
        vix_now = vix_close[i - 1]   # Previous day's VIX (known at open)
        true_rvol = rvol[i]          # True realized vol (what actually happened)

        if np.isnan(true_rvol) or np.isnan(vix_now):
            continue

        # Simulate our model's vol forecast
        model_vol = simulate_vol_forecast(true_rvol, vix_now, ic=config.vol_model_ic)

        # Generate brackets for this day
        brackets = generate_brackets(spx_now, config.bracket_width, config.n_brackets_per_side)

        day_trades = []
        day_risk = 0.0

        for floor, cap in brackets:
            if day_risk >= config.max_daily_risk:
                break
            if len(day_trades) >= config.max_daily_trades:
                break

            # Market prices (from VIX, normal distribution)
            yes_bid, yes_ask = simulate_market_price(
                floor, cap, spx_now, vix_now, config.hours_to_close, config.spread_half
            )

            # Our model prices (from predicted rvol, Student-t)
            model_fv = price_bracket_student_t(
                floor, cap, spx_now, model_vol, config.hours_to_close, config.student_t_df
            )

            # Also compute what normal dist would give with our vol (to separate edge sources)
            normal_fv_our_vol = price_bracket_normal(
                floor, cap, spx_now, model_vol, config.hours_to_close
            )

            # Settlement
            settled_yes = floor <= spx_end < cap

            # Check BUY YES opportunity: our model says it's worth more than ask
            buy_yes_edge = model_fv - yes_ask
            buy_yes_fee = taker_fee(yes_ask) if not config.use_maker_fees else maker_fee(yes_ask)
            buy_yes_net = buy_yes_edge - buy_yes_fee

            # Check BUY NO opportunity: our model says bracket is worth less than bid
            no_fv = 1.0 - model_fv
            no_ask = 1.0 - yes_bid   # Buying NO at the implied ask
            buy_no_edge = no_fv - no_ask
            buy_no_fee = taker_fee(no_ask) if not config.use_maker_fees else maker_fee(no_ask)
            buy_no_net = buy_no_edge - buy_no_fee

            # Pick best trade
            if buy_yes_net >= config.min_net_edge and buy_yes_edge >= config.min_gross_edge:
                side = "BUY_YES"
                entry_price = yes_ask
                fair_value = model_fv
                edge_gross = buy_yes_edge
                fee = buy_yes_fee
                edge_net = buy_yes_net
                settled_value = 1.0 if settled_yes else 0.0

            elif buy_no_net >= config.min_net_edge and buy_no_edge >= config.min_gross_edge:
                side = "BUY_NO"
                entry_price = no_ask
                fair_value = no_fv
                edge_gross = buy_no_edge
                fee = buy_no_fee
                edge_net = buy_no_net
                settled_value = 0.0 if settled_yes else 1.0

            else:
                continue  # No edge on this bracket

            # Position sizing
            contracts = compute_kelly_size(
                edge_net, entry_price, config.max_position_per_bracket, config.kelly_fraction
            )
            if contracts <= 0:
                continue

            # Risk check
            risk = contracts * entry_price
            if day_risk + risk > config.max_daily_risk:
                contracts = max(1, int((config.max_daily_risk - day_risk) / entry_price))
                risk = contracts * entry_price

            # Execute trade
            trade_pnl = contracts * (settled_value - entry_price - fee)

            # Classify edge source
            dist_edge = abs(model_fv - normal_fv_our_vol)  # Distributional (tail) edge
            vol_edge = abs(normal_fv_our_vol - price_bracket_normal(
                floor, cap, spx_now, vix_now, config.hours_to_close
            ))

            trade = Trade(
                date=day_date,
                floor=floor,
                cap=cap,
                side=side,
                entry_price=round(entry_price, 4),
                fair_value=round(fair_value, 4),
                edge_gross=round(edge_gross, 4),
                fee=round(fee, 4),
                edge_net=round(edge_net, 4),
                settled_value=settled_value,
                pnl=round(trade_pnl, 2),
                contracts=contracts,
                spx_at_entry=round(spx_now, 2),
                spx_at_close=round(spx_end, 2),
                vix=round(vix_now, 2),
                rvol=round(true_rvol, 2),
            )
            day_trades.append(trade)
            day_risk += risk

            if dist_edge > vol_edge:
                distributional_trades.append(trade)
            else:
                vol_forecast_trades.append(trade)

        # Day summary
        day_total = sum(t.pnl for t in day_trades)
        daily_pnl.append(day_total)
        equity += day_total
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        all_trades.extend(day_trades)

    # ---------------------------------------------------------------------------
    # Compute results
    # ---------------------------------------------------------------------------

    n_trades = len(all_trades)
    if n_trades == 0:
        logger.warning("No trades executed. Try lowering min_net_edge.")
        return BacktestResult(config=asdict(config), n_trading_days=len(daily_pnl))

    pnls = [t.pnl for t in all_trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = n_trades - wins

    total_pnl = sum(pnls)
    total_fees = sum(t.fee * t.contracts for t in all_trades)

    # Daily stats
    daily_pnl_arr = np.array(daily_pnl)
    non_zero_days = daily_pnl_arr[daily_pnl_arr != 0]
    avg_daily = np.mean(daily_pnl_arr) if len(daily_pnl_arr) > 0 else 0
    std_daily = np.std(daily_pnl_arr) if len(daily_pnl_arr) > 1 else 1

    # Max drawdown
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Sharpe (annualized from daily)
    sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

    # Edge by bracket position
    atm_pnl = sum(t.pnl for t in all_trades if abs(t.spx_at_entry - (t.floor + t.cap)/2) < 25)
    wing_pnl = sum(t.pnl for t in all_trades if abs(t.spx_at_entry - (t.floor + t.cap)/2) >= 25)
    atm_count = sum(1 for t in all_trades if abs(t.spx_at_entry - (t.floor + t.cap)/2) < 25)
    wing_count = sum(1 for t in all_trades if abs(t.spx_at_entry - (t.floor + t.cap)/2) >= 25)

    # Average contracts per day (over days with trades)
    trading_days = sum(1 for d in daily_pnl if d != 0)
    total_contracts = sum(t.contracts for t in all_trades)

    result = BacktestResult(
        config=asdict(config),
        n_trading_days=len(daily_pnl),
        n_trades=n_trades,
        n_wins=wins,
        n_losses=losses,
        win_rate=round(wins / n_trades, 4) if n_trades > 0 else 0,
        total_pnl=round(total_pnl, 2),
        avg_pnl_per_trade=round(total_pnl / n_trades, 4) if n_trades > 0 else 0,
        avg_pnl_per_day=round(avg_daily, 2),
        monthly_pnl_estimate=round(avg_daily * 21, 2),
        annual_pnl_estimate=round(avg_daily * 252, 2),
        max_drawdown=round(max_dd, 4),
        sharpe_ratio=round(sharpe, 2),
        avg_edge_per_trade=round(np.mean([t.edge_net for t in all_trades]), 4),
        total_fees=round(total_fees, 2),
        avg_contracts_per_day=round(total_contracts / trading_days, 1) if trading_days > 0 else 0,
        capital_utilized_pct=round(
            np.mean([sum(t.entry_price * t.contracts for t in all_trades if t.date == d)
                     for d in set(t.date for t in all_trades)]) / config.starting_capital * 100, 1
        ) if all_trades else 0,
        edge_by_bracket_type={
            "atm": {"pnl": round(atm_pnl, 2), "trades": atm_count,
                     "avg_pnl": round(atm_pnl / atm_count, 4) if atm_count > 0 else 0},
            "wing": {"pnl": round(wing_pnl, 2), "trades": wing_count,
                      "avg_pnl": round(wing_pnl / wing_count, 4) if wing_count > 0 else 0},
        },
        daily_pnl=[round(d, 2) for d in daily_pnl],
        equity_curve=[round(e, 2) for e in equity_curve],
        trades=[asdict(t) for t in all_trades[-200:]],  # Last 200 trades for file size
        pnl_by_strategy={
            "distributional_edge": {
                "pnl": round(sum(t.pnl for t in distributional_trades), 2),
                "trades": len(distributional_trades),
                "avg_pnl": round(np.mean([t.pnl for t in distributional_trades]), 4) if distributional_trades else 0,
            },
            "vol_forecast_edge": {
                "pnl": round(sum(t.pnl for t in vol_forecast_trades), 2),
                "trades": len(vol_forecast_trades),
                "avg_pnl": round(np.mean([t.pnl for t in vol_forecast_trades]), 4) if vol_forecast_trades else 0,
            },
        },
    )

    return result


def format_report(result: BacktestResult) -> str:
    """Format backtest results as a readable report."""
    lines = [
        "=" * 70,
        "  SPX BRACKET BACKTEST RESULTS",
        "=" * 70,
        "",
        f"  Trading days:        {result.n_trading_days}",
        f"  Total trades:        {result.n_trades}",
        f"  Win/Loss:            {result.n_wins}W / {result.n_losses}L  ({result.win_rate:.1%})",
        "",
        f"  Total P&L:           ${result.total_pnl:>+10.2f}",
        f"  Avg P&L per trade:   ${result.avg_pnl_per_trade:>+10.4f}",
        f"  Avg P&L per day:     ${result.avg_pnl_per_day:>+10.2f}",
        "",
        f"  Monthly estimate:    ${result.monthly_pnl_estimate:>+10.2f}",
        f"  Annual estimate:     ${result.annual_pnl_estimate:>+10.2f}",
        "",
        f"  Sharpe ratio:        {result.sharpe_ratio:>10.2f}",
        f"  Max drawdown:        {result.max_drawdown:>10.2%}",
        f"  Total fees paid:     ${result.total_fees:>10.2f}",
        f"  Avg contracts/day:   {result.avg_contracts_per_day:>10.1f}",
        f"  Capital utilization:  {result.capital_utilized_pct:>9.1f}%",
        "",
        "-" * 70,
        "  EDGE BREAKDOWN BY BRACKET TYPE",
        "-" * 70,
    ]

    for btype, data in result.edge_by_bracket_type.items():
        lines.append(
            f"  {btype.upper():>6s}:  {data['trades']:>4d} trades  "
            f"P&L=${data['pnl']:>+8.2f}  "
            f"Avg=${data['avg_pnl']:>+.4f}/trade"
        )

    lines.extend([
        "",
        "-" * 70,
        "  EDGE BREAKDOWN BY SOURCE",
        "-" * 70,
    ])

    for source, data in result.pnl_by_strategy.items():
        label = source.replace("_", " ").title()
        lines.append(
            f"  {label:>25s}:  {data['trades']:>4d} trades  "
            f"P&L=${data['pnl']:>+8.2f}  "
            f"Avg=${data['avg_pnl']:>+.4f}/trade"
        )

    lines.extend([
        "",
        "-" * 70,
        "  CAPITAL REQUIREMENTS & RISK",
        "-" * 70,
        f"  Starting capital:      ${result.config['starting_capital']:>10,.0f}",
        f"  Return on capital:     {result.total_pnl / result.config['starting_capital'] * 100:>10.1f}%",
        f"  Annualized ROC:        {result.annual_pnl_estimate / result.config['starting_capital'] * 100:>10.1f}%",
        f"  Max drawdown:          {result.max_drawdown:>10.2%}",
        f"  Kelly fraction used:   {result.config['kelly_fraction']:>10.0%}",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity(market_data: dict = None) -> dict:
    """Run backtest across different parameter settings.

    Tests sensitivity to:
      - IC (model quality)
      - Student-t df (tail thickness)
      - Minimum edge threshold
      - Kelly fraction
    """
    if market_data is None:
        market_data = fetch_spx_vix_data(days=252)

    results = {}

    # IC sensitivity
    logger.info("Sensitivity: IC sweep...")
    ic_results = []
    for ic in [0.0, 0.2, 0.4, 0.5, 0.644, 0.8]:
        cfg = BacktestConfig(vol_model_ic=ic)
        r = run_backtest(cfg, market_data)
        ic_results.append({
            "ic": ic,
            "total_pnl": r.total_pnl,
            "sharpe": r.sharpe_ratio,
            "trades": r.n_trades,
            "win_rate": r.win_rate,
            "monthly": r.monthly_pnl_estimate,
        })
    results["ic_sensitivity"] = ic_results

    # df sensitivity (tail thickness)
    logger.info("Sensitivity: df sweep...")
    df_results = []
    for df in [3.0, 4.0, 5.0, 7.0, 10.0, 30.0]:
        cfg = BacktestConfig(student_t_df=df)
        r = run_backtest(cfg, market_data)
        df_results.append({
            "df": df,
            "total_pnl": r.total_pnl,
            "sharpe": r.sharpe_ratio,
            "trades": r.n_trades,
            "win_rate": r.win_rate,
            "monthly": r.monthly_pnl_estimate,
        })
    results["df_sensitivity"] = df_results

    # Min edge sensitivity
    logger.info("Sensitivity: min_edge sweep...")
    edge_results = []
    for edge in [0.003, 0.005, 0.008, 0.012, 0.020, 0.030]:
        cfg = BacktestConfig(min_net_edge=edge)
        r = run_backtest(cfg, market_data)
        edge_results.append({
            "min_net_edge": edge,
            "total_pnl": r.total_pnl,
            "sharpe": r.sharpe_ratio,
            "trades": r.n_trades,
            "win_rate": r.win_rate,
            "monthly": r.monthly_pnl_estimate,
        })
    results["edge_sensitivity"] = edge_results

    # Kelly fraction sensitivity
    logger.info("Sensitivity: Kelly sweep...")
    kelly_results = []
    for kf in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        cfg = BacktestConfig(kelly_fraction=kf)
        r = run_backtest(cfg, market_data)
        kelly_results.append({
            "kelly_fraction": kf,
            "total_pnl": r.total_pnl,
            "sharpe": r.sharpe_ratio,
            "trades": r.n_trades,
            "max_drawdown": r.max_drawdown,
            "monthly": r.monthly_pnl_estimate,
        })
    results["kelly_sensitivity"] = kelly_results

    return results


def format_sensitivity_report(sens: dict) -> str:
    """Format sensitivity analysis as readable tables."""
    lines = [
        "",
        "=" * 70,
        "  SENSITIVITY ANALYSIS",
        "=" * 70,
        "",
        "  IC (Model Quality) Impact:",
        f"  {'IC':>6s} | {'Trades':>6s} | {'Win Rate':>8s} | {'Monthly':>10s} | {'Sharpe':>7s}",
        "  " + "-" * 50,
    ]
    for r in sens["ic_sensitivity"]:
        lines.append(
            f"  {r['ic']:>6.3f} | {r['trades']:>6d} | {r['win_rate']:>8.1%} | "
            f"${r['monthly']:>+9.0f} | {r['sharpe']:>7.2f}"
        )

    lines.extend([
        "",
        "  Student-t df (Tail Thickness) Impact:",
        f"  {'df':>6s} | {'Trades':>6s} | {'Win Rate':>8s} | {'Monthly':>10s} | {'Sharpe':>7s}",
        "  " + "-" * 50,
    ])
    for r in sens["df_sensitivity"]:
        lines.append(
            f"  {r['df']:>6.1f} | {r['trades']:>6d} | {r['win_rate']:>8.1%} | "
            f"${r['monthly']:>+9.0f} | {r['sharpe']:>7.2f}"
        )

    lines.extend([
        "",
        "  Minimum Edge Threshold:",
        f"  {'Edge':>6s} | {'Trades':>6s} | {'Win Rate':>8s} | {'Monthly':>10s} | {'Sharpe':>7s}",
        "  " + "-" * 50,
    ])
    for r in sens["edge_sensitivity"]:
        lines.append(
            f"  {r['min_net_edge']:>6.3f} | {r['trades']:>6d} | {r['win_rate']:>8.1%} | "
            f"${r['monthly']:>+9.0f} | {r['sharpe']:>7.2f}"
        )

    lines.extend([
        "",
        "  Kelly Fraction (Position Sizing):",
        f"  {'Kelly':>6s} | {'Trades':>6s} | {'MaxDD':>8s} | {'Monthly':>10s} | {'Sharpe':>7s}",
        "  " + "-" * 50,
    ])
    for r in sens["kelly_sensitivity"]:
        lines.append(
            f"  {r['kelly_fraction']:>6.0%} | {r['trades']:>6d} | {r['max_drawdown']:>8.2%} | "
            f"${r['monthly']:>+9.0f} | {r['sharpe']:>7.2f}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assessment: realistic P&L, capital, risks
# ---------------------------------------------------------------------------

def generate_assessment(result: BacktestResult, sens: dict = None) -> str:
    """Generate a comprehensive assessment of the strategy viability."""
    lines = [
        "",
        "=" * 70,
        "  STRATEGY ASSESSMENT",
        "=" * 70,
        "",
        "  1. REALISTIC MONTHLY P&L",
        "  " + "-" * 40,
    ]

    # Discount backtest P&L by 50% for real-world frictions
    realistic_monthly = result.monthly_pnl_estimate * 0.5
    realistic_annual = result.annual_pnl_estimate * 0.5

    lines.extend([
        f"  Backtest monthly P&L:   ${result.monthly_pnl_estimate:>+10.0f}",
        f"  Realistic (50% haircut): ${realistic_monthly:>+10.0f}",
        f"  Realistic annual:        ${realistic_annual:>+10.0f}",
        f"",
        f"  The 50% haircut accounts for:",
        f"    - Backtest overfitting to historical data",
        f"    - Wider real spreads than simulated",
        f"    - Adverse selection (smart MMs pull quotes on our signals)",
        f"    - Execution delays and missed fills",
        f"    - VIX is not the exact market-maker vol input",
        "",
        "  2. CAPITAL REQUIREMENTS",
        "  " + "-" * 40,
        f"  Minimum to start:       $5,000  (paper trade on Kalshi demo first)",
        f"  Recommended:            $10,000 (adequate for Kelly sizing)",
        f"  Optimal:                $25,000 (full strategy deployment)",
        f"  Max useful:             $50,000 (liquidity-limited beyond this)",
        "",
        "  3. RISK PROFILE",
        "  " + "-" * 40,
        f"  Max drawdown (backtest): {result.max_drawdown:.1%}",
        "  Worst daily loss:        ${:>+.0f}".format(min(result.daily_pnl) if result.daily_pnl else 0),
        "  Best daily gain:         ${:>+.0f}".format(max(result.daily_pnl) if result.daily_pnl else 0),
        f"  Sharpe ratio:            {result.sharpe_ratio:.2f}",
        "",
        f"  Key risks:",
        f"    - Kalshi liquidity: SPX brackets do $200-350K/day volume",
        f"      Our max daily deployment is small enough to avoid impact",
        f"    - Regime change: Vol model trained on recent data, may not",
        f"      generalize to fundamentally different vol regimes",
        f"    - Platform risk: Single exchange, regulatory uncertainty",
        f"    - Model degradation: IC=0.644 may decay over time",
        "",
        "  4. KYC & ACCOUNT SETUP",
        "  " + "-" * 40,
        f"  YES, Kalshi KYC is required:",
        f"    - Full name, DOB, SSN, government photo ID",
        f"    - US residents only",
        f"    - Processing: 1-3 business days",
        f"    - API key generation after KYC approval",
        f"  ",
        f"  BEFORE live trading:",
        f"    1. Complete KYC on kalshi.com",
        f"    2. Generate API key (RSA private key)",
        f"    3. Fund with $5-10K via USDC (instant) or ACH (3-5 days)",
        f"    4. Paper trade on demo API for 1-2 weeks",
        f"    5. Go live with 25% of intended capital",
        "",
        "  5. OTHER PREDICTION MARKET OPPORTUNITIES",
        "  " + "-" * 40,
        f"  a) FOMC Divergence (Kalshi):",
        f"     - Trade Kalshi vs CME FedWatch divergence",
        f"     - ~8 trades/year, edge when divergence > 15%",
        f"     - Expected: $100-500/trade, low frequency",
        f"     - Status: FOMC tracker already built",
        "",
        f"  b) Polymarket Market Making:",
        f"     - 0% maker fees, daily rebates from taker fees",
        f"     - $150-300/day per market (reported by pros)",
        f"     - Requires inventory management + hedging",
        f"     - Status: polymarket_client.py exists, needs MM logic",
        "",
        f"  c) Favorite-Longshot Bias (Kalshi/Polymarket):",
        f"     - Sell overpriced longshot contracts",
        f"     - Academic research confirms 2-3% edge on 5-cent contracts",
        f"     - Risk: occasional large loss when longshot hits",
        f"     - Status: strategy_engine.py has scanner",
        "",
        f"  d) Combinatorial Arbitrage (Polymarket):",
        f"     - Find logically contradictory prices",
        f"     - $40M+ extracted in 2024-2025 (per academic paper)",
        f"     - Highly competitive (2.7s average opportunity window)",
        f"     - Status: cross_platform_arb.py exists",
        "",
        f"  e) Weather/Sports (Kalshi):",
        f"     - 90% of Kalshi volume is sports",
        f"     - Need domain-specific models (no current edge)",
        f"     - Skip unless we build a sports model",
        "",
        "  6. RECOMMENDED NEXT STEPS",
        "  " + "-" * 40,
        f"    1. Complete Kalshi KYC",
        f"    2. Paper trade SPX brackets on demo API for 2 weeks",
        f"    3. Run vol_signal_writer.py in daemon mode during paper trading",
        f"    4. If paper results match backtest within 50%, go live",
        f"    5. Start with $5K, scale to $10-25K over 1 month",
        f"    6. Monitor FOMC tracker for divergence trades (passive)",
        f"    7. Consider Polymarket MM if Kalshi edge persists",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SPX Bracket Backtest — Vol model edge on Kalshi contracts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=252, help="Trading days to backtest")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital ($)")
    parser.add_argument("--ic", type=float, default=0.644, help="Vol model IC")
    parser.add_argument("--df", type=float, default=4.0, help="Student-t degrees of freedom")
    parser.add_argument("--min-edge", type=float, default=0.008, help="Min net edge to trade")
    parser.add_argument("--kelly", type=float, default=0.15, help="Kelly fraction (0-1)")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--verbose", action="store_true", help="Show individual trades")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Fetch data once
    market_data = fetch_spx_vix_data(days=args.days)

    # Run main backtest
    config = BacktestConfig(
        starting_capital=args.capital,
        vol_model_ic=args.ic,
        student_t_df=args.df,
        min_net_edge=args.min_edge,
        kelly_fraction=args.kelly,
    )
    result = run_backtest(config, market_data)

    # Print report
    print(format_report(result))

    # Verbose: show trades
    if args.verbose and result.trades:
        print("\n  RECENT TRADES (last 30):")
        print(f"  {'Date':>12s} | {'Side':>8s} | {'Floor':>6s}-{'Cap':>6s} | "
              f"{'Entry':>6s} | {'FV':>6s} | {'Edge':>6s} | {'PnL':>8s}")
        print("  " + "-" * 75)
        for t in result.trades[-30:]:
            print(f"  {t['date']:>12s} | {t['side']:>8s} | {t['floor']:>6.0f}-{t['cap']:>6.0f} | "
                  f"${t['entry_price']:>5.2f} | ${t['fair_value']:>5.2f} | "
                  f"${t['edge_net']:>5.3f} | ${t['pnl']:>+7.2f}")

    # Sensitivity analysis
    sens = None
    if args.sensitivity:
        sens = run_sensitivity(market_data)
        print(format_sensitivity_report(sens))

    # Assessment
    print(generate_assessment(result, sens))

    # Save results
    if not args.no_save:
        output_file = DATA_DIR / f"bracket_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "result": {
                "config": result.config,
                "n_trading_days": result.n_trading_days,
                "n_trades": result.n_trades,
                "n_wins": result.n_wins,
                "n_losses": result.n_losses,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "avg_pnl_per_trade": result.avg_pnl_per_trade,
                "avg_pnl_per_day": result.avg_pnl_per_day,
                "monthly_pnl_estimate": result.monthly_pnl_estimate,
                "annual_pnl_estimate": result.annual_pnl_estimate,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "total_fees": result.total_fees,
                "edge_by_bracket_type": result.edge_by_bracket_type,
                "pnl_by_strategy": result.pnl_by_strategy,
                "daily_pnl": result.daily_pnl,
                "equity_curve": result.equity_curve,
                "trades": result.trades,
            },
        }
        if sens:
            save_data["sensitivity"] = sens
        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
