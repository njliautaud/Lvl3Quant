"""
Instrument validation module for live trading.

Ensures we never accidentally trade the wrong contract or apply a model
trained on one instrument to a different one. Provides price-range sanity
checks and a startup audit hook with Discord alerting.
"""

import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approved symbols (ES contract months through 2027-H)
# ---------------------------------------------------------------------------
APPROVED_SYMBOLS: set[str] = {"ESM6", "ESU6", "ESZ6", "ESH7"}

# ---------------------------------------------------------------------------
# Expected price ranges per product family
# ---------------------------------------------------------------------------
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "ES": (4000.0, 8000.0),
    "NQ": (15000.0, 30000.0),
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIT_LOG_DIR = Path(__file__).resolve().parent / "logs"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "instrument_audit.log"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class InstrumentMismatchError(Exception):
    """Raised when a symbol or model checkpoint does not match the expected instrument."""
    pass


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------
def validate_instrument(symbol: str, model_weights_path: str = None) -> bool:
    """Validate that *symbol* is approved and optionally matches the model checkpoint.

    Args:
        symbol: Contract symbol to validate (e.g. ``"ESM6"``).
        model_weights_path: Optional path to a ``.pt`` checkpoint. If provided
            the function attempts to load the checkpoint and verify that its
            ``training_instrument`` metadata matches *symbol*.

    Returns:
        ``True`` when validation passes.

    Raises:
        InstrumentMismatchError: If the symbol is not approved or does not
            match the checkpoint metadata.
    """
    if symbol not in APPROVED_SYMBOLS:
        raise InstrumentMismatchError(
            f"Symbol '{symbol}' is not in APPROVED_SYMBOLS {APPROVED_SYMBOLS}"
        )

    if model_weights_path is not None:
        _validate_checkpoint_instrument(symbol, model_weights_path)

    return True


def _validate_checkpoint_instrument(symbol: str, weights_path: str) -> None:
    """Load a torch checkpoint and compare its training_instrument to *symbol*."""
    try:
        import torch
    except ImportError:
        logger.warning("torch not available — skipping checkpoint instrument check")
        return

    if not os.path.isfile(weights_path):
        logger.warning("Weights file not found: %s — skipping checkpoint check", weights_path)
        return

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Could not load checkpoint %s: %s", weights_path, exc)
        return

    training_instrument = None
    if isinstance(checkpoint, dict):
        training_instrument = checkpoint.get("training_instrument")

    if training_instrument is None:
        logger.info("Checkpoint has no 'training_instrument' key — skipping mismatch check")
        return

    if training_instrument != symbol:
        raise InstrumentMismatchError(
            f"Model was trained on '{training_instrument}' but trading symbol is '{symbol}'"
        )

    logger.info("Checkpoint instrument '%s' matches trading symbol", symbol)


# ---------------------------------------------------------------------------
# Price-range sanity check
# ---------------------------------------------------------------------------
def check_price_range(symbol: str, price: float) -> bool:
    """Check whether *price* falls in the expected range for the product family.

    The product family is derived from the leading alphabetic characters of
    *symbol* (e.g. ``"ESM6"`` → ``"ES"``).

    Args:
        symbol: Contract symbol.
        price: Last/mid price to validate.

    Returns:
        ``True`` if the price is within range (or if no range is defined for
        the product family).  ``False`` if outside range — a strong hint that
        the wrong instrument is being quoted.
    """
    # Extract product family: first 2 chars for standard CME futures (ES, NQ, YM, etc.)
    # Symbol format: ESM6 = ES (product) + M (month) + 6 (year)
    product = symbol[:2] if len(symbol) >= 2 else symbol

    expected = PRICE_RANGES.get(product)
    if expected is None:
        logger.info("No price range defined for product '%s' — skipping check", product)
        return True

    lo, hi = expected
    if lo <= price <= hi:
        return True

    logger.warning(
        "Price %.2f for %s is OUTSIDE expected range [%.0f, %.0f] — possible wrong instrument!",
        price, symbol, lo, hi,
    )
    return False


# ---------------------------------------------------------------------------
# Discord webhook alert (best-effort, no external deps)
# ---------------------------------------------------------------------------
def _send_discord_alert(message: str) -> None:
    """Send a best-effort Discord webhook alert."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set — cannot send alert")
        return

    payload = json.dumps({"content": message}).encode("utf-8")
    req = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urlopen(req, timeout=5)
        logger.info("Discord alert sent")
    except Exception as exc:
        logger.warning("Failed to send Discord alert: %s", exc)


# ---------------------------------------------------------------------------
# Startup audit hook
# ---------------------------------------------------------------------------
def startup_instrument_audit(
    symbol: str,
    exchange: str = "",
    weights_path: str = None,
) -> bool:
    """Run a full instrument audit at system startup.

    1. Validates the symbol (and optional checkpoint).
    2. Logs the result to ``logs/instrument_audit.log``.
    3. Sends a Discord webhook alert on failure.

    Args:
        symbol: Contract symbol to trade.
        exchange: Exchange name (informational, logged but not validated).
        weights_path: Optional model checkpoint path.

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
    """
    ts = datetime.now(timezone.utc).isoformat()

    try:
        validate_instrument(symbol, model_weights_path=weights_path)
    except InstrumentMismatchError as exc:
        msg = f"[{ts}] INSTRUMENT AUDIT FAILED — {exc}"
        logger.error(msg)
        _write_audit_log(ts, symbol, exchange, weights_path, passed=False, detail=str(exc))
        _send_discord_alert(f"🚨 **INSTRUMENT VALIDATION FAILED**\n{exc}")
        return False

    _write_audit_log(ts, symbol, exchange, weights_path, passed=True)
    logger.info("Instrument audit PASSED for %s on %s", symbol, exchange or "(no exchange)")
    return True


def _write_audit_log(
    ts: str,
    symbol: str,
    exchange: str,
    weights_path: str | None,
    passed: bool,
    detail: str = "",
) -> None:
    """Append a line to the audit log file."""
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        status = "PASS" if passed else "FAIL"
        line = f"{ts} | {status} | symbol={symbol} | exchange={exchange} | weights={weights_path or 'N/A'}"
        if detail:
            line += f" | detail={detail}"
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("Could not write audit log: %s", exc)
