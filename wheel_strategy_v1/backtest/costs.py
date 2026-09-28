"""
costs.py — Retail options costs.

Assumptions (CANONICAL — used everywhere in this package):
  - Commission per contract:        $0.00 (Schwab thinkorswim, Robinhood)
                                    For IBKR Lite: $0.65/contract — set BROKER='ibkr' to use.
  - Regulatory + exchange fees:     ~$0.03/contract per side  (ORF + OCC + exchange fee, average)
  - Assignment / exercise fee:      $0.00 (Schwab, Robinhood, IBKR Lite all $0)
  - Share assignment cost:          $0.00 commission (Robinhood/Schwab); SEC TAF still applies
                                    on the share leg sells (~$0.00229% on sells = ~0.23 bps).

Net cost per opened contract (one-side open OR close):
  Schwab/Robinhood:  ~$0.03
  IBKR Lite:         ~$0.68
"""
from __future__ import annotations

BROKER = "schwab"  # 'schwab' | 'robinhood' | 'ibkr'

def per_contract_cost(broker: str = None) -> float:
    b = (broker or BROKER).lower()
    if b in ("schwab", "robinhood"):
        commission = 0.0
    elif b == "ibkr":
        commission = 0.65
    else:
        commission = 0.0
    reg_fee = 0.03  # ORF/OCC/exchange fees, rough average per contract per side
    return commission + reg_fee


def assignment_cost(broker: str = None) -> float:
    # All three brokers in scope: $0 assignment fee.
    return 0.0


def share_taf_cost(notional: float) -> float:
    # SEC Trading Activity Fee applied on share-leg SELLS only (~0.00229%).
    # Applied when shares are called away (covered-call exercise) or liquidated.
    return max(0.0, notional * 0.0000229)
