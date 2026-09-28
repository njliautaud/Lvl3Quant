# ETF Rotation — Capacity Analysis at RETAIL Sizes ($20K Baseline)

**Date:** 2026-06-08
**Binding directive:** HC #580 (capacity must be re-tested at retail sizes; $20K baseline; % returns primary)
**Source of original test:** `output/macro_picker/capacity_20260608_224842/` (mean impact 5.34 bps RT at $1M; sweep $1M → $1B)
**Strategy:** ETF rotation v1, hold-21 long-only, n_long=2, regime-gated (SPY 60d MA bull-only), target_vol=15%, lev∈[0.25,2.0]
**Universe (11):** XLK XLF XLE XLY XLP XLU XLI XLV XLB XLC XLRE
**Most-picked sectors:** XLE (29), XLI (16), XLC (16), XLU (10), XLRE (8), XLP/XLV (5 each)
**Bottleneck ETFs (smallest ADV):** XLRE ($271M/d), XLB ($637M/d), XLC ($668M/d)
**OOT span:** 419 trading days (~1.66 years), 47 rebal events

---

## Impact model (unchanged from original — apples-to-apples)

Square-root market-impact model, identical parameterization to `strategy/macro_picker/capacity_analysis.py`:

```
per_etf_dollar     = AUM * gross_lev / n_long
participation_pct  = 100 * per_etf_dollar / ADV_20d
impact_bps_per_side = k * sqrt(participation_pct),  k=10
impact_rt_bps      = 2 * impact_bps_per_side
incremental_drag   = max(0, impact_rt_bps − 5bps baseline) * gross_lev
```

Because impact scales as sqrt(AUM), and the $1M run measured 5.34 bps RT mean, the impact at any other AUM is `5.34 * sqrt(AUM / $1M)`.

## Retail-size impact (the headline result)

| AUM     | Mean RT impact (bps) | Incremental above 5bps baseline |
|---------|---------------------:|--------------------------------:|
| $20K    | 0.755                | **0.000** (clamped to zero)     |
| $50K    | 1.194                | **0.000**                        |
| $100K   | 1.689                | **0.000**                        |
| $250K   | 2.670                | **0.000**                        |
| $500K   | 3.776                | **0.000**                        |
| $1M     | 5.340                | 0.340                            |
| $10M    | 16.89                | 11.89                            |
| $100M   | 53.40                | 48.40                            |

**Implication:** At every retail size from $20K through $500K, raw market-impact is *less than the 5bps baseline transaction-cost assumption already in the backtest book*. The impact-drag adjustment is **identically zero**, so performance metrics at $20K, $50K, $100K, and $250K are **exactly identical** to the headline pre-capacity result.

In plain English: SPY-sector ETFs (XLK/XLE/XLF/XLU/etc.) trade hundreds of millions to billions of dollars a day. Putting $20K — or even $250K — through them is a *rounding error*. There is no detectable price impact at retail size on these instruments.

## Capacity ladder (% returns primary, $20K-account dollars secondary)

| AUM     | CAGR    | MaxDD   | Sharpe | Sortino | Calmar | PF    | WR    |
|---------|--------:|--------:|-------:|--------:|-------:|------:|------:|
| $20K    | 26.13%  | -8.16%  | 1.90   | 2.75    | 3.20   | 1.41  | 46.5% |
| $50K    | 26.13%  | -8.16%  | 1.90   | 2.75    | 3.20   | 1.41  | 46.5% |
| $100K   | 26.13%  | -8.16%  | 1.90   | 2.75    | 3.20   | 1.41  | 46.5% |
| $250K   | 26.13%  | -8.16%  | 1.90   | 2.75    | 3.20   | 1.41  | 46.5% |
| (ref) $1M | 26.13% | -8.16% | 1.90   | 2.75    | 3.20   | 1.41  | 46.5% |
| (ref) $10M | 21.52% | -8.71% | 1.60  | 2.41    | 2.47   | 1.33  | 46.5% |
| (ref) $50M | 13.39% | -9.73% | 1.04  | 1.63    | 1.38   | 1.20  | 46.3% |
| (ref) $100M | 7.63% | -10.74% | 0.62 | 0.98    | 0.71   | 1.11  | 46.1% |

### Dollar P&L on $20K account (over 419-day OOT span)
- Total return ≈ **+47.1%** → $20,000 → **~$29,400** (gain **~+$9,400**)
- Annualized: **~+$5,230/yr**
- Maximum peak-to-trough equity drawdown experienced: **~-$1,630** (-8.16%)

(Reported dollar values are for the $20K baseline only, per HC #580 R2.)

## Where does capacity actually start mattering?

- **$0 → ~$500K:** No measurable effect. Strategy is uncapped at retail size.
- **$1M:** First place where mean impact (5.34 bps) starts to nudge above the 5 bps baseline. Trivial drag.
- **$10M:** First place with a real haircut — Sharpe falls from 1.90 → 1.60, CAGR 26% → 22%.
- **$50M:** First place with a *meaningful* haircut — Sharpe 1.04, CAGR 13%.
- **~$53M:** Capacity (defined as Sharpe ≥ 1.0) breakpoint per the original analysis.
- **$100M+:** Strategy is meaningfully impaired; $500M and $1B make it unprofitable.

## Bottleneck ETFs (don't apply at retail)

At $1B AUM the smallest-ADV ETFs (XLRE, XLB, XLC) absorb ~41% of total impact, driven largely by XLC's high pick frequency (16/47 rebals) combined with its small ADV (~$668M/day). XLRE is even smaller (~$271M/day). At $20K-$250K none of this matters — even a 100% position in XLC at $250K is ~0.04% of XLC's daily volume, far below the threshold where any sane market-impact model registers a print.

## Headline (for HC #580 R3)

The capacity analysis re-run at the user's actual scale ($20K → $250K) returns the strategy's *uncapped* numbers across the board. Capacity does not constrain a retail operator on liquid SPDR sector ETFs. The earlier "$40M clean / $100M erodes" framing was institutionally-framed and not relevant to a retail-scale operator — the relevant answer is **"capacity is not a binding constraint for you on this strategy."**

## Caveats / honest qualifiers

- The 5 bps baseline already assumed in the book is conservative for retail; real costs at $20K may be slightly *higher* (commission per share on small notional, half-spread on penny-wide ETFs ≈ 0.5–1 bp, SEC fees) but the order of magnitude is well below 10 bps RT and does not change the conclusion.
- The CAGR/Sharpe figures derive from the leader strategy's measured OOT book; any general OOT-leakage / overfitting / regime-shift risk applies equally at all account sizes — capacity analysis only addresses market-impact, not edge persistence.
- This re-run uses the *exact same impact model* (sqrt-law, k=10) as the original test, per HC #580 R3 ("apples-to-apples"). No model recalibration was performed.
