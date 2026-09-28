# Wheel + Picker Blend — HC #558 R3 + HC #557 R7

Window: 2020-01-01 → 2025-12-31. $100k starting per leg.
Wheel: v7 canonical Balanced FullWheel (real IV + skew + slippage + regime overlay).
Picker: v2 thematic rotation (sectors + themes + acceleration signal).
Blend: static weight, daily rebalanced (mathematical, no friction added because both legs already charged friction internally).

## Grid Results

| Wheel% | Picker% | CAGR | Sharpe | Sortino | MaxDD | Green$/mo | Red$/mo | Flat$/mo | Sh>0.70 | DD<-47% | Red≥0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100% | 0% | 10.6% | 1.02 | 0.41 | -11.9% | 0.24% | 0.06% | 0.58% | ✓ | ✓ | ✓ |
| 70% | 30% | 11.9% | 1.33 | 1.65 | -11.1% | 1.59% | -1.16% | 0.57% | ✓ | ✓ | ✗ |
| 60% | 40% | 12.3% | 1.33 | 1.71 | -10.9% | 2.04% | -1.56% | 0.56% | ✓ | ✓ | ✗ |
| 50% | 50% | 12.6% | 1.29 | 1.68 | -12.3% | 2.50% | -1.96% | 0.56% | ✓ | ✓ | ✗ |
| 30% | 70% | 13.2% | 1.12 | 1.46 | -16.3% | 3.42% | -2.75% | 0.55% | ✓ | ✓ | ✗ |
| 0% | 100% | 13.7% | 0.89 | 1.13 | -23.5% | 4.83% | -3.92% | 0.53% | ✓ | ✓ | ✗ |

## Best Blend (by red-month closest to zero among qualified)

**Wheel 100% / Picker 0%**
- CAGR 10.6%, Sharpe 1.02, MaxDD -11.9%
- Monthly returns: green 0.24%, red 0.06%, flat 0.58%
- vs SPY 1.5× margin (Sharpe 0.70, MaxDD -47%, CAGR 18.5%): WINS Sharpe + DD

## Notes
- Picker v2 alone has Red monthly of -3.93% — pure long-only equity bleeds in red months.
- Wheel Balanced alone red monthly varies — premium income usually positive but small.
- Blend should ideally produce non-negative red months from wheel premium covering picker losses.
