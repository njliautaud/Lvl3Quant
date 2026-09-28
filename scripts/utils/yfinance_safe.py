"""
yfinance_safe.py — Split-safe wrapper around yfinance downloads.

THE PROBLEM
-----------
yfinance applies stock splits retroactively when auto_adjust=True. For example,
AMZN's 20:1 split in June 2022 causes all pre-split prices to be divided by 20.
This is correct within a SINGLE download call, but causes bugs when:

  1. Entry and exit prices come from DIFFERENT download calls (different adjustment
     factors if splits occurred between calls).
  2. Prices are cached/stored and later compared to freshly downloaded prices.
  3. Batch downloads vs single-ticker downloads apply adjustments differently.

SYMPTOMS:
  - AMZN exit price shows $9.85 instead of ~$200 (20:1 split applied to exit but
    not entry, or vice versa).
  - Individual strategy Sharpe ratios of 5-9 (impossibly high — caused by price
    discontinuities inflating apparent returns).
  - Any single-stock trade showing >200% return or <-80% loss that isn't a
    leveraged product.

THE FIX
-------
Always download the FULL date range in a SINGLE call with auto_adjust=True. Never
mix prices from different download batches. This module enforces that pattern and
provides validation utilities.

USAGE
-----
    from scripts.utils.yfinance_safe import safe_download, validate_trade_pnl

    # Download — single call, consistent split adjustment
    close, open_, volume = safe_download(['AMZN', 'AAPL'], '2021-01-01', '2024-12-31')

    # Validate a trade before recording P&L
    ok, msg = validate_trade_pnl(
        entry_price=150.0, exit_price=160.0,
        ticker='AMZN', entry_date='2023-01-15', exit_date='2023-02-15'
    )
    if not ok:
        print(f"SUSPICIOUS TRADE: {msg}")
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required: pip install yfinance")


# ---------------------------------------------------------------------------
# Known stock splits (ticker -> list of (date, ratio) where ratio is new/old).
# Used for validation, not adjustment — yfinance handles adjustment internally.
# ---------------------------------------------------------------------------
KNOWN_SPLITS: Dict[str, List[Tuple[str, float]]] = {
    "AMZN": [("2022-06-06", 20.0)],
    "GOOGL": [("2022-07-18", 20.0)],
    "GOOG": [("2022-07-18", 20.0)],
    "TSLA": [("2022-08-25", 3.0), ("2020-08-31", 5.0)],
    "AAPL": [("2020-08-31", 4.0)],
    "NVDA": [("2024-06-10", 10.0), ("2021-07-20", 4.0)],
    "SHOP": [("2022-06-29", 10.0)],
    "CMG": [("2024-06-26", 50.0)],
    "WMT": [("2024-02-26", 3.0)],
}

# Leveraged / inverse products where extreme returns are expected
LEVERAGED_PRODUCTS = {
    "TQQQ", "SQQQ", "UPRO", "SPXU", "UDOW", "SDOW", "TNA", "TZA",
    "LABU", "LABD", "FNGU", "FNGD", "SOXL", "SOXS", "UVXY", "SVXY",
    "NUGT", "DUST", "JNUG", "JDST", "ERX", "ERY", "FAS", "FAZ",
    "TECL", "TECS", "CURE", "DRIP", "GUSH", "UCO", "SCO",
}

# Maximum plausible single-trade return thresholds (for non-leveraged equities)
MAX_GAIN_PCT = 200.0   # >200% gain on a single trade is suspicious
MAX_LOSS_PCT = -80.0   # >80% loss on a single trade is suspicious


def safe_download(
    tickers: Union[str, List[str]],
    start: str,
    end: str,
    *,
    progress: bool = False,
    threads: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download price data with guaranteed consistent split adjustment.

    Downloads the FULL date range in a SINGLE yfinance call with auto_adjust=True,
    ensuring all prices use the same split adjustment factors.

    Parameters
    ----------
    tickers : str or list of str
        Ticker symbol(s) to download.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    progress : bool
        Show yfinance download progress bar (default False).
    threads : bool
        Use threaded downloads for multiple tickers (default True).

    Returns
    -------
    close : pd.DataFrame
        Adjusted close prices. Columns = tickers, index = dates.
    open_ : pd.DataFrame
        Adjusted open prices. Same shape as close.
    volume : pd.DataFrame
        Volume. Same shape as close.

    Raises
    ------
    ValueError
        If no data is returned for any ticker.

    Examples
    --------
    >>> close, open_, volume = safe_download(['AMZN', 'AAPL'], '2021-01-01', '2024-12-31')
    >>> close.head()
                  AMZN    AAPL
    2021-01-04  163.20  129.41
    ...

    Notes
    -----
    - NEVER cache the output and mix with a later safe_download() call if a split
      may have occurred between calls. Re-download the full range instead.
    - For backtests, call safe_download ONCE covering the entire backtest period
      and slice the resulting DataFrames by date.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        t_upper = t.upper()
        if t_upper not in seen:
            seen.add(t_upper)
            unique_tickers.append(t)

    # Single download call — this is the critical part.
    # auto_adjust=True ensures OHLC prices are split-adjusted consistently.
    raw = yf.download(
        unique_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=progress,
        threads=threads,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {unique_tickers} "
            f"between {start} and {end}"
        )

    # Handle column structure differences between single and multi-ticker downloads
    if len(unique_tickers) == 1:
        ticker = unique_tickers[0]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw[["Close"]].rename(columns={"Close": ticker})
        open_ = raw[["Open"]].rename(columns={"Open": ticker})
        volume = raw[["Volume"]].rename(columns={"Volume": ticker})
    else:
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
            open_ = raw["Open"]
            volume = raw["Volume"]
        else:
            # Fallback: single-ticker-like structure even with multiple tickers
            close = raw[["Close"]].rename(columns={"Close": unique_tickers[0]})
            open_ = raw[["Open"]].rename(columns={"Open": unique_tickers[0]})
            volume = raw[["Volume"]].rename(columns={"Volume": unique_tickers[0]})

    # Run split detection on close prices
    split_warnings = detect_splits(close)
    if split_warnings:
        for warn_msg in split_warnings:
            warnings.warn(f"[yfinance_safe] {warn_msg}", stacklevel=2)

    # Drop tickers that are entirely NaN (delisted, invalid symbol, etc.)
    valid_mask = close.notna().any()
    dropped = valid_mask[~valid_mask].index.tolist()
    if dropped:
        warnings.warn(
            f"[yfinance_safe] Dropped tickers with no data: {dropped}",
            stacklevel=2,
        )
        close = close[valid_mask[valid_mask].index]
        open_ = open_[valid_mask[valid_mask].index]
        volume = volume[valid_mask[valid_mask].index]

    return close, open_, volume


def detect_splits(
    prices_df: pd.DataFrame,
    threshold: float = 0.20,
) -> List[str]:
    """
    Detect potential stock-split artifacts in a price DataFrame.

    Flags any single-day price change exceeding `threshold` (default 20%) as a
    potential unadjusted split. In properly adjusted data, splits should NOT
    cause large jumps — so any detected jump indicates either:
      (a) A genuine large move (earnings, news), or
      (b) A split-adjustment failure.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Close prices. Columns = tickers, index = dates.
    threshold : float
        Minimum absolute daily return to flag (default 0.20 = 20%).

    Returns
    -------
    list of str
        Warning messages for each detected anomaly.

    Examples
    --------
    >>> import pandas as pd
    >>> # Simulate unadjusted AMZN 20:1 split
    >>> dates = pd.date_range('2022-06-01', '2022-06-10', freq='B')
    >>> prices = pd.DataFrame({'AMZN': [2500, 2480, 2510, 125, 126, 127, 128][:len(dates)]},
    ...                       index=dates)
    >>> warnings = detect_splits(prices)
    >>> len(warnings) > 0
    True
    """
    warnings_list = []
    returns = prices_df.pct_change()

    for col in returns.columns:
        series = returns[col].dropna()
        large_moves = series[series.abs() > threshold]

        for date, ret in large_moves.items():
            pct = ret * 100
            direction = "drop" if ret < 0 else "jump"

            # Check if this matches a known split date
            known = False
            if col.upper() in KNOWN_SPLITS:
                for split_date_str, ratio in KNOWN_SPLITS[col.upper()]:
                    split_date = pd.Timestamp(split_date_str)
                    # Allow 3-day window around split date
                    if abs((pd.Timestamp(date) - split_date).days) <= 3:
                        known = True
                        warnings_list.append(
                            f"SPLIT ARTIFACT? {col} on {date:%Y-%m-%d}: "
                            f"{pct:+.1f}% {direction} — matches known "
                            f"{ratio:.0f}:1 split on {split_date_str}. "
                            f"Verify auto_adjust=True was used."
                        )
                        break

            if not known:
                # Could be genuine (earnings, M&A) or unknown split
                warnings_list.append(
                    f"LARGE MOVE: {col} on {date:%Y-%m-%d}: {pct:+.1f}% "
                    f"{direction}. Check if this is a real move or "
                    f"unadjusted split."
                )

    return warnings_list


def validate_trade_pnl(
    entry_price: float,
    exit_price: float,
    ticker: str,
    entry_date: str,
    exit_date: str,
    *,
    max_gain_pct: float = MAX_GAIN_PCT,
    max_loss_pct: float = MAX_LOSS_PCT,
    is_short: bool = False,
) -> Tuple[bool, str]:
    """
    Sanity-check a single trade's P&L for split-adjustment errors.

    Rejects trades with implausibly large returns that typically indicate
    entry and exit prices were downloaded with inconsistent split adjustments.

    Parameters
    ----------
    entry_price : float
        Price at trade entry.
    exit_price : float
        Price at trade exit.
    ticker : str
        Ticker symbol.
    entry_date : str
        Entry date ('YYYY-MM-DD').
    exit_date : str
        Exit date ('YYYY-MM-DD').
    max_gain_pct : float
        Maximum allowed gain % (default 200).
    max_loss_pct : float
        Maximum allowed loss % as negative number (default -80).
    is_short : bool
        True if this is a short trade (inverts the return calculation).

    Returns
    -------
    (is_valid, message) : (bool, str)
        is_valid=True if the trade passes sanity checks.
        message describes the issue if invalid.

    Examples
    --------
    >>> # Normal trade
    >>> ok, msg = validate_trade_pnl(150.0, 160.0, 'AMZN', '2023-01-01', '2023-02-01')
    >>> ok
    True

    >>> # Split-corrupted trade (AMZN pre-split entry vs post-split exit)
    >>> ok, msg = validate_trade_pnl(2500.0, 130.0, 'AMZN', '2022-05-01', '2022-07-01')
    >>> ok
    False
    >>> 'split' in msg.lower()
    True

    >>> # Leveraged ETF — extreme returns are expected
    >>> ok, msg = validate_trade_pnl(10.0, 35.0, 'TQQQ', '2023-01-01', '2023-12-01')
    >>> ok
    True
    """
    if entry_price <= 0 or exit_price <= 0:
        return False, (
            f"{ticker}: Invalid price (entry={entry_price}, exit={exit_price}). "
            f"Prices must be positive."
        )

    # Calculate return
    if is_short:
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
    else:
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

    # Leveraged products get a pass — extreme returns are normal
    if ticker.upper() in LEVERAGED_PRODUCTS:
        return True, f"{ticker}: Leveraged product, skip sanity check (return: {pnl_pct:+.1f}%)"

    # Check against thresholds
    if pnl_pct > max_gain_pct:
        # Check if a known split could explain this
        split_info = _check_split_between_dates(ticker, entry_date, exit_date)
        return False, (
            f"{ticker}: Implausible gain of {pnl_pct:+.1f}% "
            f"(entry={entry_price:.2f} on {entry_date}, "
            f"exit={exit_price:.2f} on {exit_date}). "
            f"Likely a stock-split adjustment error. {split_info}"
        )

    if pnl_pct < max_loss_pct:
        split_info = _check_split_between_dates(ticker, entry_date, exit_date)
        return False, (
            f"{ticker}: Implausible loss of {pnl_pct:+.1f}% "
            f"(entry={entry_price:.2f} on {entry_date}, "
            f"exit={exit_price:.2f} on {exit_date}). "
            f"Likely a stock-split adjustment error. {split_info}"
        )

    # Price ratio sanity: if exit/entry ratio matches a known split ratio, flag it
    ratio = exit_price / entry_price if entry_price > 0 else 0
    inv_ratio = entry_price / exit_price if exit_price > 0 else 0
    for check_ratio in [ratio, inv_ratio]:
        if ticker.upper() in KNOWN_SPLITS:
            for split_date, split_ratio in KNOWN_SPLITS[ticker.upper()]:
                # Check if the price ratio is suspiciously close to the split ratio
                if abs(check_ratio - split_ratio) / split_ratio < 0.05:
                    return False, (
                        f"{ticker}: Price ratio {check_ratio:.1f} matches known "
                        f"{split_ratio:.0f}:1 split on {split_date}. "
                        f"Entry/exit prices likely have inconsistent split adjustment."
                    )

    return True, f"{ticker}: Trade OK (return: {pnl_pct:+.1f}%)"


def _check_split_between_dates(ticker: str, start_date: str, end_date: str) -> str:
    """Check if a known split occurred between two dates."""
    ticker_upper = ticker.upper()
    if ticker_upper not in KNOWN_SPLITS:
        return "No known splits in database for this ticker."

    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except Exception:
        return ""

    splits_between = []
    for split_date_str, ratio in KNOWN_SPLITS[ticker_upper]:
        split_date = pd.Timestamp(split_date_str)
        if start <= split_date <= end:
            splits_between.append(f"{ratio:.0f}:1 on {split_date_str}")

    if splits_between:
        return f"KNOWN SPLITS in trade window: {', '.join(splits_between)}"
    return "No known splits in this date range (may be an unknown split)."


def safe_get_price(
    prices_df: pd.DataFrame,
    ticker: str,
    target_date: str,
    *,
    max_lookback_days: int = 5,
) -> Optional[float]:
    """
    Get a price from a pre-downloaded DataFrame, with business-day fallback.

    Always use this with a DataFrame from safe_download() to ensure consistent
    split adjustment. Never mix with prices from a separate download call.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Close prices from safe_download().
    ticker : str
        Ticker symbol (must be a column in prices_df).
    target_date : str
        Target date ('YYYY-MM-DD').
    max_lookback_days : int
        Maximum business days to look back if target_date is not a trading day.

    Returns
    -------
    float or None
        The price, or None if not found within lookback window.

    Examples
    --------
    >>> close, _, _ = safe_download(['AAPL'], '2023-01-01', '2023-12-31')
    >>> price = safe_get_price(close, 'AAPL', '2023-07-04')  # Holiday
    >>> price is not None  # Falls back to 2023-07-03
    True
    """
    if ticker not in prices_df.columns:
        return None

    target = pd.Timestamp(target_date)

    # Try exact date first
    if target in prices_df.index:
        val = prices_df.loc[target, ticker]
        if pd.notna(val):
            return float(val)

    # Look back up to max_lookback_days
    for i in range(1, max_lookback_days + 1):
        check_date = target - timedelta(days=i)
        if check_date in prices_df.index:
            val = prices_df.loc[check_date, ticker]
            if pd.notna(val):
                return float(val)

    return None


def download_with_validation(
    tickers: Union[str, List[str]],
    start: str,
    end: str,
    *,
    progress: bool = False,
    threads: bool = True,
    strict: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Download data and run full validation. Returns data + list of warnings.

    Like safe_download() but also returns detailed validation warnings.
    If strict=True, raises ValueError on any detected split artifacts.

    Parameters
    ----------
    tickers : str or list of str
        Ticker symbol(s).
    start, end : str
        Date range ('YYYY-MM-DD').
    progress, threads : bool
        Passed to yfinance.
    strict : bool
        If True, raise on detected split artifacts. If False, return warnings.

    Returns
    -------
    close, open_, volume : pd.DataFrame
        Price/volume data.
    warnings : list of str
        Validation warnings (empty if clean).
    """
    close, open_, volume = safe_download(
        tickers, start, end, progress=progress, threads=threads
    )

    all_warnings = detect_splits(close)

    # Additional validation: check for suspiciously low prices that might
    # indicate unadjusted splits applied incorrectly
    for col in close.columns:
        recent = close[col].dropna().tail(20)
        historical = close[col].dropna().head(60)

        if len(recent) > 0 and len(historical) > 0:
            recent_mean = recent.mean()
            hist_mean = historical.mean()

            if hist_mean > 0 and recent_mean > 0:
                ratio = recent_mean / hist_mean
                # If recent prices are 5x+ or 0.2x historical, something is off
                if ratio > 5 or ratio < 0.2:
                    all_warnings.append(
                        f"PRICE DISCONTINUITY: {col} recent mean "
                        f"${recent_mean:.2f} vs historical mean "
                        f"${hist_mean:.2f} (ratio {ratio:.1f}x). "
                        f"Possible split adjustment issue."
                    )

    if strict and any("SPLIT ARTIFACT" in w for w in all_warnings):
        raise ValueError(
            "Split artifacts detected in downloaded data:\n"
            + "\n".join(w for w in all_warnings if "SPLIT ARTIFACT" in w)
        )

    return close, open_, volume, all_warnings
