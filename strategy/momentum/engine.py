"""
Momentum Strategy — Backtest Engine
Walk-forward momentum scoring + portfolio construction + regime-stratified analytics.

Key design choices:
- SLIDING windows only (never expanding) — hard constraint.
- Momentum = total return over lookback, skipping most recent skip_months to avoid reversal.
- Walk-forward: score → rank → buy top N → hold for rebal_period → repeat.
- Transaction costs applied on turnover.

Enhanced (v2):
- Trend filter: SPY below 200d SMA → reduce/cut equity exposure.
- Sector-neutral: pick top stocks per GICS sector to avoid concentration.
- Volatility-managed: scale position size inversely to realized vol.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import json
import datetime as dt


# ---------------------------------------------------------------------------
# GICS Sector Mapping (for sector-neutral selection)
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    # Technology
    "AAPL": "Tech", "ADBE": "Tech", "ADI": "Tech", "ADP": "Tech", "ADSK": "Tech",
    "AMAT": "Tech", "AMD": "Tech", "AVGO": "Tech", "CDNS": "Tech", "CRM": "Tech",
    "CSCO": "Tech", "GOOG": "Tech", "INTC": "Tech", "INTU": "Tech", "META": "Tech",
    "MSFT": "Tech", "NOW": "Tech", "NVDA": "Tech", "ORCL": "Tech", "PYPL": "Tech",
    "QCOM": "Tech", "TXN": "Tech",
    # Healthcare
    "ABBV": "Health", "ABT": "Health", "AMGN": "Health", "BDX": "Health", "BMY": "Health",
    "CI": "Health", "DHR": "Health", "GILD": "Health", "ISRG": "Health", "JNJ": "Health",
    "LLY": "Health", "MDT": "Health", "MRK": "Health", "PFE": "Health", "TMO": "Health",
    "UNH": "Health", "ZTS": "Health",
    # Financials
    "AIG": "Fin", "AXP": "Fin", "BAC": "Fin", "BLK": "Fin", "BRK-B": "Fin",
    "C": "Fin", "CME": "Fin", "COF": "Fin", "GS": "Fin", "ICE": "Fin",
    "JPM": "Fin", "MET": "Fin", "MS": "Fin", "SCHW": "Fin", "USB": "Fin",
    "V": "Fin", "MA": "Fin", "WFC": "Fin", "SPG": "Fin",
    # Consumer Discretionary
    "AMZN": "ConsDisc", "DIS": "ConsDisc", "F": "ConsDisc", "GM": "ConsDisc",
    "HD": "ConsDisc", "LOW": "ConsDisc", "MCD": "ConsDisc", "NKE": "ConsDisc",
    "NFLX": "ConsDisc", "SBUX": "ConsDisc", "TGT": "ConsDisc",
    # Consumer Staples
    "CL": "ConsStp", "COST": "ConsStp", "EL": "ConsStp", "KO": "ConsStp",
    "MDLZ": "ConsStp", "MO": "ConsStp", "PEP": "ConsStp", "PG": "ConsStp",
    "PM": "ConsStp", "WMT": "ConsStp",
    # Industrials
    "BA": "Indust", "CAT": "Indust", "DE": "Indust", "EMR": "Indust",
    "FDX": "Indust", "GD": "Indust", "GE": "Indust", "HON": "Indust",
    "LMT": "Indust", "NOC": "Indust", "RTX": "Indust", "UNP": "Indust",
    "UPS": "Indust",
    # Energy
    "COP": "Energy", "CVX": "Energy", "EOG": "Energy", "XOM": "Energy",
    # Utilities
    "D": "Util", "DUK": "Util", "EXC": "Util", "NEE": "Util", "SO": "Util",
    # Communication
    "CMCSA": "Comm", "T": "Comm", "TMUS": "Comm", "VZ": "Comm",
    # Materials
    "ECL": "Matls", "LIN": "Matls", "SHW": "Matls",
    # Real Estate
    "CCI": "RE",
    # Other / unmapped
    "ACN": "Tech", "IBM": "Tech", "MMM": "Indust",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MomentumConfig:
    """All tunable parameters in one place."""
    # Momentum lookback (trading days)
    mom_12m_days: int = 252          # ~12 months
    mom_6m_days: int = 126           # ~6 months
    mom_skip_days: int = 21          # skip most recent month (reversal avoidance)

    # Composite score weights
    w_12m: float = 0.6
    w_6m: float = 0.4

    # Portfolio construction
    top_n: int = 20                  # number of stocks to hold
    rebal_freq: str = "monthly"      # "weekly" or "monthly"
    weighting: str = "equal"         # "equal" or "inv_vol"
    inv_vol_lookback_days: int = 63  # ~3 months for vol estimate

    # Transaction costs
    cost_per_trade: float = 0.001    # 0.1% per trade (slippage + commission)

    # Walk-forward
    warmup_days: int = 280           # need at least 12m+skip of history before first signal

    # Regime check threshold
    regime_gap_threshold: float = 0.50

    # --- v2 enhancements ---

    # Trend filter: when SPY < SMA(trend_sma_days), cut equity exposure
    trend_filter: bool = False
    trend_sma_days: int = 200        # 200-day SMA (or ~10 months)
    trend_cash_frac: float = 1.0     # fraction to move to cash when below SMA
                                     # 1.0 = 100% cash; 0.5 = 50% cash / 50% equities
    trend_bond_ticker: str | None = None  # if set (e.g. "SHY"), allocate cash portion here

    # Sector-neutral: pick top_per_sector from each sector instead of overall top_n
    sector_neutral: bool = False
    top_per_sector: int = 2

    # Volatility-managed: scale exposure inversely to realized vol
    vol_managed: bool = False
    vol_target: float = 0.15         # annualized vol target (15%)
    vol_lookback_days: int = 63      # lookback for realized vol estimate
    vol_max_leverage: float = 1.5    # cap leverage at 1.5x
    vol_min_exposure: float = 0.2    # floor exposure at 20%

    # Long/short: short bottom-N momentum stocks
    short_in_downtrend: bool = False  # short only when trend is off
    always_short: bool = False       # short bottom-N at ALL times (market-neutral-ish)
    short_n: int = 10               # number of stocks to short
    short_weight: float = 0.5       # total short exposure as fraction (0.5 = 50% short)

    # Defensive rotation: when below SMA, rotate into defensive sectors only
    defensive_rotation: bool = False
    defensive_sectors: tuple = ("Util", "ConsStp", "Health")  # traditionally defensive

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Momentum Scoring
# ---------------------------------------------------------------------------

def compute_momentum(prices: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    """
    Compute composite momentum score for each stock on each day.

    momentum_Nm = price[t - skip] / price[t - N - skip] - 1
    composite = w_12m * mom_12m + w_6m * mom_6m

    Returns DataFrame same shape as prices with momentum scores.
    """
    skip = config.mom_skip_days

    # 12-month momentum (skip recent month)
    mom_12m = prices.shift(skip) / prices.shift(config.mom_12m_days + skip) - 1

    # 6-month momentum (skip recent month)
    mom_6m = prices.shift(skip) / prices.shift(config.mom_6m_days + skip) - 1

    composite = config.w_12m * mom_12m + config.w_6m * mom_6m

    return composite


def _select_sector_neutral(
    scores_row: pd.Series,
    config: MomentumConfig,
) -> pd.Series:
    """
    Sector-neutral selection: pick top_per_sector from each GICS sector.
    Returns a Series of selected tickers with their momentum scores.
    """
    selected = []
    # Group tickers by sector
    sector_groups: dict[str, list] = {}
    for ticker, score in scores_row.items():
        sector = SECTOR_MAP.get(ticker, "Other")
        sector_groups.setdefault(sector, []).append((ticker, score))

    for sector, members in sector_groups.items():
        # Sort by score descending, pick top_per_sector
        members.sort(key=lambda x: x[1], reverse=True)
        for ticker, score in members[:config.top_per_sector]:
            selected.append((ticker, score))

    if not selected:
        return pd.Series(dtype=float)
    tickers, vals = zip(*selected)
    return pd.Series(vals, index=tickers)


def rank_and_select(
    scores: pd.DataFrame,
    date_idx: int,
    config: MomentumConfig,
    prices: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    On a given date, rank stocks by momentum score and return top N with weights.

    Returns dict of {ticker: weight}.
    """
    row = scores.iloc[date_idx].dropna()

    if config.sector_neutral:
        top = _select_sector_neutral(row, config)
    else:
        if len(row) < config.top_n:
            top = row.nlargest(len(row))
        else:
            top = row.nlargest(config.top_n)

    if len(top) == 0:
        return {}

    if config.weighting == "equal":
        w = 1.0 / len(top)
        return {t: w for t in top.index}

    elif config.weighting == "inv_vol":
        if prices is None:
            raise ValueError("Need prices for inv_vol weighting")
        # Inverse volatility weighting
        vol_end = date_idx + 1
        vol_start = max(0, vol_end - config.inv_vol_lookback_days)
        rets = prices.iloc[vol_start:vol_end].pct_change().dropna()
        valid_tickers = [t for t in top.index if t in rets.columns]
        if not valid_tickers:
            w = 1.0 / len(top)
            return {t: w for t in top.index}
        vols = rets[valid_tickers].std()
        vols = vols.replace(0, np.nan).dropna()
        if len(vols) == 0:
            w = 1.0 / len(top)
            return {t: w for t in top.index}
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return weights.to_dict()

    else:
        raise ValueError(f"Unknown weighting: {config.weighting}")


# ---------------------------------------------------------------------------
# Walk-Forward Backtest
# ---------------------------------------------------------------------------

def get_rebal_dates(dates: pd.DatetimeIndex, freq: str) -> list[int]:
    """Return indices in `dates` where rebalancing occurs."""
    if freq == "monthly":
        # First trading day of each month
        months = dates.to_period("M")
        indices = []
        for i in range(1, len(months)):
            if months[i] != months[i - 1]:
                indices.append(i)
        return indices
    elif freq == "weekly":
        # First trading day of each week
        weeks = dates.to_period("W")
        indices = []
        for i in range(1, len(weeks)):
            if weeks[i] != weeks[i - 1]:
                indices.append(i)
        return indices
    else:
        raise ValueError(f"Unknown rebal freq: {freq}")


def _compute_trend_signal(spy_prices: pd.Series, sma_days: int) -> pd.Series:
    """
    Compute trend signal: 1.0 when SPY >= SMA(sma_days), 0.0 when below.
    Uses ONLY past data (SMA computed on data up to and including current day).
    """
    sma = spy_prices.rolling(window=sma_days, min_periods=sma_days).mean()
    signal = (spy_prices >= sma).astype(float)
    # Before SMA is available, default to 1.0 (invested)
    signal = signal.fillna(1.0)
    return signal


def _compute_vol_scalar(
    equity: np.ndarray,
    day_idx: int,
    config: MomentumConfig,
) -> float:
    """
    Compute volatility-managed scalar: vol_target / realized_vol.
    Capped at vol_max_leverage, floored at vol_min_exposure.
    Uses ONLY past data (lookback ending at day_idx-1).
    """
    lookback = config.vol_lookback_days
    if day_idx < lookback + 2:
        return 1.0  # not enough history

    # Compute realized daily returns over lookback
    recent_values = equity[max(0, day_idx - lookback):day_idx]
    if len(recent_values) < 10:
        return 1.0
    daily_rets = np.diff(recent_values) / recent_values[:-1]
    realized_vol = np.std(daily_rets) * np.sqrt(252)

    if realized_vol <= 0:
        return 1.0

    scalar = config.vol_target / realized_vol
    scalar = min(scalar, config.vol_max_leverage)
    scalar = max(scalar, config.vol_min_exposure)
    return scalar


def run_backtest(
    prices: pd.DataFrame,
    config: MomentumConfig | None = None,
    spy_prices: pd.Series | None = None,
    bond_prices: pd.Series | None = None,
) -> dict:
    """
    Run walk-forward momentum backtest.

    Uses SLIDING windows: momentum lookback is fixed-length, never expanding.

    Args:
        prices: stock price DataFrame
        config: backtest configuration
        spy_prices: SPY prices (needed for trend filter)
        bond_prices: bond ETF prices (e.g. SHY) for cash alternative

    Returns dict with:
        - equity_curve: pd.Series (daily portfolio value, starts at 1.0)
        - trades: list of dicts
        - rebal_log: list of dicts (date, holdings, turnover)
        - config: the config used
    """
    if config is None:
        config = MomentumConfig()

    extras = []
    if config.trend_filter:
        extras.append("trend_filter")
    if config.sector_neutral:
        extras.append("sector_neutral")
    if config.vol_managed:
        extras.append("vol_managed")
    extra_str = f" [{', '.join(extras)}]" if extras else ""

    print(f"[engine] Running backtest: top_n={config.top_n}, rebal={config.rebal_freq}, "
          f"weighting={config.weighting}, cost={config.cost_per_trade*100:.2f}%{extra_str}")

    # Compute trend signal if needed
    trend_signal = None
    if config.trend_filter:
        if spy_prices is None:
            raise ValueError("spy_prices required when trend_filter=True")
        trend_signal = _compute_trend_signal(spy_prices, config.trend_sma_days)
        # Align to prices index
        trend_signal = trend_signal.reindex(prices.index).ffill().fillna(1.0)

    # Compute bond daily returns if using bond alternative
    bond_daily_rets = None
    if config.trend_filter and config.trend_bond_ticker and bond_prices is not None:
        bond_daily_rets = bond_prices.reindex(prices.index).ffill().pct_change().fillna(0)

    scores = compute_momentum(prices, config)
    dates = prices.index
    rebal_indices = get_rebal_dates(dates, config.rebal_freq)

    # Filter rebal dates to after warmup
    rebal_indices = [i for i in rebal_indices if i >= config.warmup_days]

    if len(rebal_indices) == 0:
        raise ValueError("No rebalance dates after warmup period")

    # Initialize
    daily_returns = prices.pct_change()
    start_idx = rebal_indices[0]
    rebal_set = set(rebal_indices)

    portfolio_value = np.ones(len(dates))
    current_weights: dict[str, float] = {}
    current_equity_frac = 1.0  # fraction allocated to equities (vs cash/bonds)
    rebal_log = []
    total_turnover = 0.0
    n_trend_off = 0  # count days trend filter was active

    # Track the "unscaled" (full-equity) weights separately from the trend-adjusted ones
    base_long_weights: dict[str, float] = {}   # long momentum picks
    base_short_weights: dict[str, float] = {}  # short picks (bottom momentum, for downtrend)
    prev_trend_on = True

    def _compute_short_weights(scores_df, day_i, cfg):
        """Pick bottom-N momentum stocks for shorting."""
        row = scores_df.iloc[day_i].dropna()
        if len(row) < cfg.short_n:
            bottom = row.nsmallest(len(row))
        else:
            bottom = row.nsmallest(cfg.short_n)
        if len(bottom) == 0:
            return {}
        w = cfg.short_weight / len(bottom)
        return {t: -w for t in bottom.index}  # negative = short

    def _compute_defensive_weights(scores_df, day_i, cfg):
        """Select only from defensive sectors."""
        row = scores_df.iloc[day_i].dropna()
        defensive_tickers = [t for t in row.index if SECTOR_MAP.get(t, "Other") in cfg.defensive_sectors]
        if not defensive_tickers:
            return {}
        def_scores = row[defensive_tickers]
        n = min(cfg.top_n, len(def_scores))
        top = def_scores.nlargest(n)
        w = 1.0 / len(top)
        return {t: w for t in top.index}

    for day_idx in range(start_idx, len(dates)):
        # --- DAILY trend filter check ---
        if config.trend_filter and trend_signal is not None:
            trend_on = trend_signal.iloc[day_idx] >= 0.5
        else:
            trend_on = True

        is_rebal = day_idx in rebal_set
        trend_changed = trend_on != prev_trend_on

        if is_rebal:
            # Recompute stock selection
            base_long_weights = rank_and_select(scores, day_idx, config, prices)
            if config.short_in_downtrend or config.always_short:
                base_short_weights = _compute_short_weights(scores, day_idx, config)
            if config.defensive_rotation:
                defensive_weights = _compute_defensive_weights(scores, day_idx, config)

        if is_rebal or trend_changed:
            # Build final weights based on trend state
            if trend_on:
                # Uptrend: full long momentum
                scaled_weights = dict(base_long_weights)
                # If always_short, add shorts even in uptrend
                if config.always_short:
                    # Remove any overlap (don't short something you're long)
                    short_filtered = {t: w for t, w in base_short_weights.items()
                                     if t not in scaled_weights}
                    scaled_weights.update(short_filtered)
            else:
                # Downtrend: reduce/eliminate longs, optionally add shorts or defensive
                equity_frac = 1.0 - config.trend_cash_frac

                if config.defensive_rotation:
                    if is_rebal or trend_changed:
                        defensive_weights = _compute_defensive_weights(scores, day_idx, config)
                    scaled_weights = dict(defensive_weights)
                elif config.short_in_downtrend or config.always_short:
                    # Reduce longs + add short positions
                    scaled_weights = {t: w * equity_frac for t, w in base_long_weights.items()}
                    short_filtered = {t: w for t, w in base_short_weights.items()
                                     if t not in scaled_weights}
                    scaled_weights.update(short_filtered)
                else:
                    # Just reduce long exposure
                    scaled_weights = {t: w * equity_frac for t, w in base_long_weights.items()}

            current_equity_frac = 1.0 if trend_on else (1.0 - config.trend_cash_frac)

            # Calculate turnover
            all_tickers = set(list(current_weights.keys()) + list(scaled_weights.keys()))
            turnover = sum(
                abs(scaled_weights.get(t, 0) - current_weights.get(t, 0))
                for t in all_tickers
            )
            total_turnover += turnover

            # Transaction cost
            cost = turnover * config.cost_per_trade
            portfolio_value[day_idx] = portfolio_value[day_idx - 1] * (1 - cost) if day_idx > start_idx else 1.0 * (1 - cost)

            current_weights = scaled_weights

            if not trend_on:
                n_trend_off += 1

            if is_rebal:
                rebal_log.append({
                    "date": str(dates[day_idx].date()),
                    "n_holdings": sum(1 for w in scaled_weights.values() if w > 0),
                    "n_shorts": sum(1 for w in scaled_weights.values() if w < 0),
                    "equity_frac": round(current_equity_frac, 2),
                    "trend_on": trend_on,
                    "turnover": round(turnover, 4),
                    "top_5": [t for t, w in list(scaled_weights.items())[:5]],
                })

            prev_trend_on = trend_on

        elif day_idx > start_idx and current_weights:
            # Normal day: compute portfolio return
            # Weights can be positive (long) or negative (short)
            port_ret = sum(
                w * daily_returns.iloc[day_idx].get(t, 0)
                for t, w in current_weights.items()
            )

            # Bond/cash portion (unallocated fraction)
            total_abs_weight = sum(abs(w) for w in current_weights.values())
            unallocated = max(0, 1.0 - total_abs_weight)
            if unallocated > 0.01 and bond_daily_rets is not None:
                port_ret += unallocated * bond_daily_rets.iloc[day_idx]

            # Vol management scalar
            if config.vol_managed:
                vol_scalar = _compute_vol_scalar(portfolio_value, day_idx, config)
                port_ret = port_ret * vol_scalar

            portfolio_value[day_idx] = portfolio_value[day_idx - 1] * (1 + port_ret)
        else:
            if day_idx > 0:
                portfolio_value[day_idx] = portfolio_value[day_idx - 1]

    # Trim to active period
    equity = pd.Series(portfolio_value[start_idx:], index=dates[start_idx:], name="equity")

    trend_info = ""
    if config.trend_filter:
        trend_info = f", trend_off_rebal={n_trend_off}"
    print(f"[engine] Backtest complete: {len(rebal_log)} rebalances, "
          f"total turnover={total_turnover:.1f}x{trend_info}")

    return {
        "equity_curve": equity,
        "rebal_log": rebal_log,
        "config": config.to_dict(),
        "start_date": str(dates[start_idx].date()),
        "end_date": str(dates[-1].date()),
    }


# ---------------------------------------------------------------------------
# Performance Analytics
# ---------------------------------------------------------------------------

def compute_metrics(
    equity: pd.Series,
    spy: pd.Series | None = None,
    config: MomentumConfig | None = None,
) -> dict:
    """
    Compute performance metrics from equity curve.

    Returns dict with CAGR, Sharpe, Sortino, MaxDD, win rate, regime analysis.
    """
    if config is None:
        config = MomentumConfig()

    daily_rets = equity.pct_change().dropna()

    # --- Core metrics ---
    n_years = len(daily_rets) / 252
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = (daily_rets.mean() * 252) / downside if downside > 0 else 0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # Win rate (daily)
    win_rate = (daily_rets > 0).mean()

    # Monthly returns
    monthly_rets = equity.resample("ME").last().pct_change().dropna()
    monthly_win_rate = (monthly_rets > 0).mean()

    metrics = {
        "cagr": round(cagr * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "annual_vol_pct": round(ann_vol * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "daily_win_rate": round(win_rate * 100, 2),
        "monthly_win_rate": round(monthly_win_rate * 100, 2),
        "n_years": round(n_years, 2),
        "n_months": len(monthly_rets),
    }

    # --- Regime-stratified analysis ---
    if spy is not None:
        # Align SPY to equity dates
        spy_aligned = spy.reindex(equity.index).ffill()
        spy_monthly = spy_aligned.resample("ME").last().pct_change().dropna()

        # Classify months: green (SPY > 0) vs red (SPY <= 0)
        common_months = monthly_rets.index.intersection(spy_monthly.index)
        monthly_rets_aligned = monthly_rets.reindex(common_months)
        spy_monthly_aligned = spy_monthly.reindex(common_months)

        green_mask = spy_monthly_aligned > 0
        red_mask = spy_monthly_aligned <= 0

        green_rets = monthly_rets_aligned[green_mask]
        red_rets = monthly_rets_aligned[red_mask]

        def monthly_sharpe(rets):
            if len(rets) < 3 or rets.std() == 0:
                return 0
            return (rets.mean() * 12) / (rets.std() * np.sqrt(12))

        sharpe_green = monthly_sharpe(green_rets)
        sharpe_red = monthly_sharpe(red_rets)

        # Regime gap check
        denom = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / denom if denom > 0 else 0
        regime_pass = regime_gap < config.regime_gap_threshold

        metrics["regime"] = {
            "green_months": len(green_rets),
            "red_months": len(red_rets),
            "sharpe_green": round(sharpe_green, 3),
            "sharpe_red": round(sharpe_red, 3),
            "mean_return_green_pct": round(green_rets.mean() * 100, 3) if len(green_rets) > 0 else 0,
            "mean_return_red_pct": round(red_rets.mean() * 100, 3) if len(red_rets) > 0 else 0,
            "regime_gap": round(regime_gap, 3),
            "regime_gap_threshold": config.regime_gap_threshold,
            "regime_pass": regime_pass,
        }

    return metrics


def print_report(metrics: dict, config: dict | None = None):
    """Pretty-print backtest results."""
    print("\n" + "=" * 60)
    print("  MOMENTUM STRATEGY BACKTEST RESULTS")
    print("=" * 60)

    if config:
        print(f"\n  Config: top_n={config.get('top_n')}, rebal={config.get('rebal_freq')}, "
              f"weighting={config.get('weighting')}, cost={config.get('cost_per_trade', 0)*100:.1f}%")

    print(f"\n  Period: {metrics.get('n_years', 0):.1f} years ({metrics.get('n_months', 0)} months)")
    print(f"  CAGR:           {metrics['cagr']:>8.2f}%")
    print(f"  Total Return:   {metrics['total_return_pct']:>8.2f}%")
    print(f"  Sharpe:         {metrics['sharpe']:>8.3f}")
    print(f"  Sortino:        {metrics['sortino']:>8.3f}")
    print(f"  Annual Vol:     {metrics['annual_vol_pct']:>8.2f}%")
    print(f"  Max Drawdown:   {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Daily Win Rate: {metrics['daily_win_rate']:>8.2f}%")
    print(f"  Monthly WR:     {metrics['monthly_win_rate']:>8.2f}%")

    if "regime" in metrics:
        r = metrics["regime"]
        print(f"\n  --- Regime Analysis (SPY months) ---")
        print(f"  Green months: {r['green_months']}  |  Red months: {r['red_months']}")
        print(f"  Sharpe (green): {r['sharpe_green']:>7.3f}  |  Sharpe (red): {r['sharpe_red']:>7.3f}")
        print(f"  Mean ret (green): {r['mean_return_green_pct']:>6.3f}%  |  Mean ret (red): {r['mean_return_red_pct']:>6.3f}%")
        print(f"  Regime gap: {r['regime_gap']:.3f} (threshold: {r['regime_gap_threshold']:.2f}) "
              f"{'PASS' if r['regime_pass'] else 'FAIL'}")

    print("=" * 60)
