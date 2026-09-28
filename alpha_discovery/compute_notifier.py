"""
compute_notifier.py — Lightweight compute signal system.

Any orchestrator or training script calls notify_complete() when it finishes.
Signals are written to a JSON file that can be polled by a monitor or Discord bot.

No external dependencies — stdlib only.

Usage (direct call):
    from alpha_discovery.compute_notifier import notify_complete
    notify_complete("overnight_v2", "completed", result_summary="4 phases done, 3h12m")

Usage (decorator):
    from alpha_discovery.compute_notifier import notify_on_complete

    @notify_on_complete("bookcnn_v2_gpu")
    def run_training():
        ...

Usage (context manager):
    from alpha_discovery.compute_notifier import ComputeTask

    with ComputeTask("novel_targets_30s"):
        run_novel_targets(...)
"""

import json
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path

# ---------------------------------------------------------------------------
# Signal file location
# Relative to this file: alpha_discovery/data/compute_signals.json
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
SIGNALS_FILE = _THIS_DIR / "data" / "compute_signals.json"

# Maximum signals kept in the file (trim oldest when exceeded)
_MAX_SIGNALS = 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_signals() -> list:
    """Load existing signals from disk, or return empty list."""
    if not SIGNALS_FILE.exists():
        return []
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_signals(signals: list) -> None:
    """Write signals list to disk atomically (write to tmp then rename)."""
    SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = SIGNALS_FILE.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2, default=str)
        # Atomic replace (works on Windows too in Python 3.3+)
        os.replace(tmp_path, SIGNALS_FILE)
    except OSError as e:
        # Last-resort: try direct write
        try:
            with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
                json.dump(signals, f, indent=2, default=str)
        except OSError:
            raise e


def _append_signal(signal: dict) -> None:
    """
    Load existing signals, append the new one, trim to _MAX_SIGNALS, save.

    Uses a simple retry loop to handle concurrent writers without requiring
    any file-locking library.
    """
    for attempt in range(5):
        try:
            signals = _load_signals()
            signals.append(signal)
            # Keep only the most recent _MAX_SIGNALS entries
            if len(signals) > _MAX_SIGNALS:
                signals = signals[-_MAX_SIGNALS:]
            _save_signals(signals)
            return
        except OSError:
            if attempt < 4:
                time.sleep(0.1 * (attempt + 1))
            else:
                raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def notify_complete(
    task_name: str,
    status: str,
    result_summary: str = None,
    result_file: str = None,
    extra: dict = None,
) -> None:
    """
    Called by any script when it finishes. Appends a signal record to the
    signals file.

    Args:
        task_name:      Human-readable name, e.g. "overnight_v2" or "train_walkforward_book".
        status:         One of 'completed', 'failed', 'timeout'.
        result_summary: Optional short text summary of results.
        result_file:    Optional path to a JSON/log file with full results.
        extra:          Optional dict of additional key-value data.
    """
    signal = {
        "task": task_name,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "result_summary": result_summary,
        "result_file": str(result_file) if result_file else None,
        "extra": extra or {},
        "read": False,
    }
    try:
        _append_signal(signal)
    except Exception:
        # Never crash the calling script because of a notification failure
        pass


def get_unread_signals() -> list:
    """
    Returns all unread signals and marks them as read in the file.

    Returns:
        List of signal dicts where 'read' was False.
    """
    try:
        signals = _load_signals()
    except Exception:
        return []

    unread = [s for s in signals if not s.get("read", False)]
    if not unread:
        return []

    # Mark all as read
    for s in signals:
        s["read"] = True

    try:
        _save_signals(signals)
    except Exception:
        pass

    return unread


def get_all_signals(last_n: int = 20) -> list:
    """
    Get recent signals (read or unread).

    Args:
        last_n: How many of the most recent signals to return.

    Returns:
        List of signal dicts, most recent last.
    """
    try:
        signals = _load_signals()
    except Exception:
        return []
    return signals[-last_n:] if len(signals) > last_n else signals


def clear_signals() -> None:
    """Remove all signals from the file (useful for testing)."""
    try:
        _save_signals([])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def notify_on_complete(task_name: str, result_file: str = None):
    """
    Decorator that wraps a function and calls notify_complete() when it returns
    or raises.

    Usage:
        @notify_on_complete("bookcnn_v2_gpu")
        def run_training():
            ...

    On success the decorator tries to use the function's return value as
    result_summary if it is a string, or converts it via str() if it is a dict.
    On failure it records status='failed' with the exception message.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.time() - t0
                # Build a short summary from the return value
                if isinstance(result, str):
                    summary = result
                elif isinstance(result, dict):
                    # Pull common metric keys if available
                    parts = []
                    for key in ("agg_ic", "total_pnl_dollars", "sharpe", "n_folds", "elapsed"):
                        if key in result:
                            parts.append(f"{key}={result[key]}")
                    summary = f"elapsed={elapsed:.0f}s" + ("; " + ", ".join(parts) if parts else "")
                else:
                    summary = f"elapsed={elapsed:.0f}s"
                notify_complete(task_name, "completed", result_summary=summary,
                                result_file=result_file)
                return result
            except Exception as exc:
                elapsed = time.time() - t0
                notify_complete(
                    task_name, "failed",
                    result_summary=f"elapsed={elapsed:.0f}s; {type(exc).__name__}: {exc}",
                    result_file=result_file,
                )
                raise
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class ComputeTask:
    """
    Context manager that records start/end of a compute task and calls
    notify_complete() on exit.

    Usage:
        with ComputeTask("novel_targets_30s"):
            run_novel_targets(...)

        # With a result file path:
        with ComputeTask("overnight_v2", result_file="/path/to/summary.json") as task:
            result = run_overnight()
            task.set_summary(f"phases={len(result['phases'])}")
    """

    def __init__(self, task_name: str, result_file: str = None):
        self.task_name = task_name
        self.result_file = result_file
        self._summary = None
        self._t0 = None

    def set_summary(self, summary: str) -> None:
        """Optionally set a human-readable result summary before exit."""
        self._summary = summary

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._t0 if self._t0 else 0.0

        if exc_type is None:
            status = "completed"
            summary = self._summary or f"elapsed={elapsed:.0f}s"
        elif exc_type is KeyboardInterrupt:
            status = "failed"
            summary = f"interrupted after {elapsed:.0f}s"
        else:
            status = "failed"
            summary = (self._summary or "") + f" elapsed={elapsed:.0f}s; "
            summary += f"{exc_type.__name__}: {exc_val}"

        notify_complete(
            self.task_name, status,
            result_summary=summary,
            result_file=self.result_file,
        )
        # Do not suppress exceptions
        return False


# ---------------------------------------------------------------------------
# CLI: poll and print unread signals
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"

    if cmd == "poll":
        signals = get_unread_signals()
        if not signals:
            print("No unread signals.")
        else:
            print(f"{len(signals)} unread signal(s):")
            for s in signals:
                print(f"  [{s['timestamp']}] {s['task']} => {s['status']}")
                if s.get("result_summary"):
                    print(f"    {s['result_summary']}")
                if s.get("result_file"):
                    print(f"    result_file: {s['result_file']}")

    elif cmd == "all":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        signals = get_all_signals(last_n=n)
        if not signals:
            print("No signals found.")
        else:
            print(f"Last {len(signals)} signal(s):")
            for s in signals:
                read_flag = "" if s.get("read") else " [UNREAD]"
                print(f"  [{s['timestamp']}]{read_flag} {s['task']} => {s['status']}")
                if s.get("result_summary"):
                    print(f"    {s['result_summary']}")

    elif cmd == "clear":
        clear_signals()
        print("Signal file cleared.")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python compute_notifier.py [poll|all [N]|clear]")
        sys.exit(1)
