"""Standalone bracket scanner - called by the JS agent.

Supports two modes:
  1. Manual vol: pass vol as argument (backward compatible)
  2. Auto vol:   pass vol=0 or omit to use the enhanced VolModel
                 (reads walk-forward calibration + signal file)
"""
import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from kalshi_client import BracketPricer, VolModel


def scan(spx, vol=None, hours=4.0, min_net_edge=0.005, vix_level=None):
    """Scan brackets for trading opportunities.

    Args:
        spx: Current SPX price
        vol: Annualized vol (decimal). If None or 0, uses VolModel auto-estimate.
        hours: Hours to market close
        min_net_edge: Minimum net edge after fees to include
        vix_level: Optional VIX level (improves auto vol estimate)

    Returns:
        list of opportunity dicts sorted by net edge
    """
    vm = VolModel()
    pricer = BracketPricer(vm, df=4.0)

    # Auto vol estimation if not provided
    if vol is None or vol <= 0:
        est = vm.get_current_vol_estimate(
            current_spx=spx, hours_to_close=hours, vix_level=vix_level
        )
        vol = est["annualized_vol"]
        vol_info = {
            "source": est["source"],
            "confidence": est["confidence"],
            "horizon": est["horizon"],
            "model_ic": est["model_ic"],
        }
    else:
        vol_info = {"source": "manual", "confidence": 1.0}

    # Generate representative brackets around current SPX
    # In production, these come from the Kalshi API
    brackets = []
    base = int(spx / 25) * 25
    for i in range(-6, 7):
        floor = base + i * 25
        cap = floor + 25
        distance = abs(spx - (floor + cap) / 2)
        sigma = spx * vol * (hours / (252 * 6.5)) ** 0.5
        if sigma > 0:
            mock_price = max(0.02, 0.35 * (1 - (distance / (3 * sigma + 1))))
        else:
            mock_price = 0.02
        brackets.append({
            'ticker': f'INXD-SIM-B{floor}-{cap}',
            'floor': floor, 'cap': cap,
            'yes_ask': round(mock_price + 0.01, 2),
            'no_ask': round(1 - mock_price + 0.01, 2),
            'yes_bid': round(mock_price - 0.01, 2),
            'no_bid': round(1 - mock_price - 0.01, 2),
        })

    results = pricer.price_all_brackets(brackets, spx, vol, hours)
    opps = [r for r in results if r['net_edge_after_fees'] > min_net_edge]

    # Attach vol metadata to output
    output = {
        "vol_used": round(vol, 4),
        "vol_info": vol_info,
        "n_brackets": len(brackets),
        "n_opportunities": len(opps),
        "opportunities": opps,
    }
    return output


if __name__ == '__main__':
    spx = float(sys.argv[1]) if len(sys.argv) > 1 else 5900
    vol = float(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = auto
    hours = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    min_edge = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
    vix = float(sys.argv[5]) if len(sys.argv) > 5 else None

    result = scan(spx, vol if vol > 0 else None, hours, min_edge, vix_level=vix)
    print(json.dumps(result))
