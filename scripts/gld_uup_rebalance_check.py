#!/usr/bin/env python3
"""
Weekly GLD+UUP Rebalance Check
Run every Friday to check if allocation has drifted >5% from 50/50 target.
Outputs a rebalance instruction if needed.
"""

import json
import sys
from datetime import datetime

# Target allocation
TARGET_GLD_PCT = 0.50
TARGET_UUP_PCT = 0.50
DRIFT_THRESHOLD = 0.05  # 5% drift triggers rebalance

def check_rebalance(gld_value, uup_value):
    """Check if rebalancing is needed."""
    total = gld_value + uup_value
    if total < 10:  # Account too small or positions closed
        return {"action": "SKIP", "reason": "Total value too small or positions closed"}

    gld_pct = gld_value / total
    uup_pct = uup_value / total

    gld_drift = abs(gld_pct - TARGET_GLD_PCT)
    uup_drift = abs(uup_pct - TARGET_UUP_PCT)
    max_drift = max(gld_drift, uup_drift)

    result = {
        "timestamp": datetime.now().isoformat(),
        "gld_value": round(gld_value, 2),
        "uup_value": round(uup_value, 2),
        "total_value": round(total, 2),
        "gld_pct": round(gld_pct * 100, 1),
        "uup_pct": round(uup_pct * 100, 1),
        "max_drift_pct": round(max_drift * 100, 1),
        "threshold_pct": DRIFT_THRESHOLD * 100,
    }

    if max_drift > DRIFT_THRESHOLD:
        # Calculate trades needed
        target_gld = total * TARGET_GLD_PCT
        target_uup = total * TARGET_UUP_PCT

        gld_trade = target_gld - gld_value
        uup_trade = target_uup - uup_value

        result["action"] = "REBALANCE"
        result["gld_trade_dollars"] = round(gld_trade, 2)  # positive = buy, negative = sell
        result["uup_trade_dollars"] = round(uup_trade, 2)
        result["reason"] = f"Drift {max_drift*100:.1f}% exceeds {DRIFT_THRESHOLD*100}% threshold"
    else:
        result["action"] = "HOLD"
        result["reason"] = f"Drift {max_drift*100:.1f}% within {DRIFT_THRESHOLD*100}% threshold"

    return result

if __name__ == "__main__":
    # When called from autonomy_inject or cron, values will be passed as args
    # For now, just output the check logic
    if len(sys.argv) >= 3:
        gld_val = float(sys.argv[1])
        uup_val = float(sys.argv[2])
        result = check_rebalance(gld_val, uup_val)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python3 gld_uup_rebalance_check.py <gld_value> <uup_value>")
        print("Example: python3 gld_uup_rebalance_check.py 340.50 328.20")
