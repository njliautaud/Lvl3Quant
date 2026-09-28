# HC #425 Alternative-Label Base-Rate Verdict
**Dataset**: 5 OOT days (2026-02-23 → 2026-02-27), 4 label geometries (time-stop and 3 TP/SL combos).
**Cost floor**: 0.376t passive commission (HC #426 R4). Edge = mean_gross_ticks − 0.376.

## Aggregate (5-day total)

| Geometry | Side  | n_signals | n_filled | fill % | mean_gross | WR %  | edge_after_cost |
|----------|-------|----------:|---------:|-------:|-----------:|------:|----------------:|
| time60 | long  |    241692 |    49683 |  20.6% |     -4.769t |  34.4% |          -5.145t |
| time60 | short |    241692 |    59177 |  24.5% |     -7.392t |  27.3% |          -7.768t |
| tp4sl4 | long  |    241692 |    49683 |  20.6% |     -0.181t |  48.7% |          -0.557t |
| tp4sl4 | short |    241692 |    59177 |  24.5% |     -0.217t |  48.7% |          -0.593t |
| tp6sl2 | long  |    241692 |    49683 |  20.6% |     -0.378t |  25.9% |          -0.754t |
| tp6sl2 | short |    241692 |    59177 |  24.5% |     -0.413t |  25.1% |          -0.789t |
| tp8sl3 | long  |    241692 |    49683 |  20.6% |     -0.724t |  32.2% |          -1.100t |
| tp8sl3 | short |    241692 |    59177 |  24.5% |     -1.010t |  29.5% |          -1.386t |

## Ranking by net-of-passive-commission edge

### LONG side

| rank | geometry | mean_gross | n_filled | net_after_passive_cost |
|-----:|----------|-----------:|---------:|-----------------------:|
| 1 | tp4sl4 |     -0.181t |    49683 |                 -0.557t |
| 2 | tp6sl2 |     -0.378t |    49683 |                 -0.754t |
| 3 | tp8sl3 |     -0.724t |    49683 |                 -1.100t |
| 4 | time60 |     -4.769t |    49683 |                 -5.145t |

### SHORT side

| rank | geometry | mean_gross | n_filled | net_after_passive_cost |
|-----:|----------|-----------:|---------:|-----------------------:|
| 1 | tp4sl4 |     -0.217t |    59177 |                 -0.593t |
| 2 | tp6sl2 |     -0.413t |    59177 |                 -0.789t |
| 3 | tp8sl3 |     -1.010t |    59177 |                 -1.386t |
| 4 | time60 |     -7.392t |    59177 |                 -7.768t |

