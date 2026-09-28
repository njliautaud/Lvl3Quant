# Microprice Adverse Selection v1 — REPORT

**Verdict: REJECT**

- elapsed: 135.5s
- OOT dates processed: 15 / 15
- failures: 0
- fills with valid microprice features: 31086
- baseline net_ticks/trade (no filter, pooled): -0.351
- baseline pressure-aligned rate: 0.496
- total sweep cells: 215
- winning cells (HC #428 gates): 0

## HC #428 gates
- net_ticks > +0.1
- Sharpe > 0.3
- PF > 1.1
- day_coverage >= 60% of OOT days profitable
- n_trades >= 50

## Baseline (no filter) per config
```
                         config  n_trades  net_ticks_per_trade    sharpe       wr       pf  day_coverage
          pair01_logret1s+pup5s      7291            -0.331150 -1.520310 0.435743 0.825464      0.142857
  pair07_logret10s+logret60sq50      2295            -0.604322 -2.859547 0.399129 0.695651      0.000000
          pair08_logret5s+pup5s     10349            -0.341504 -1.569031 0.434825 0.820376      0.200000
 trip03_logret5s+pup5s+logret1s      7099            -0.317400 -1.456297 0.437667 0.832143      0.142857
trip01_pup5s+logret1s+logret10s      4052            -0.329480 -1.513130 0.437068 0.826157      0.090909
```

## Closest misses (top 10 by composite gate-score)
```
                         config filter_type            alignment  drift_thresh_ticks  n_trades  net_ticks_per_trade    sharpe       wr       pf  day_coverage  gate_score
 trip03_logret5s+pup5s+logret1s       combo  drift>=0.0&imb>=0.3                0.00       974            -0.077745 -0.354366 0.471253 0.956182      0.307692   -0.576596
 trip03_logret5s+pup5s+logret1s       combo  drift>=0.0&imb>=0.1                0.00      1436            -0.082825 -0.377272 0.470752 0.953454      0.307692   -0.706220
 trip03_logret5s+pup5s+logret1s       combo drift>=0.05&imb>=0.1                0.05      1397            -0.091104 -0.415074 0.469578 0.948910      0.384615   -0.790946
 trip03_logret5s+pup5s+logret1s       combo drift>=0.05&imb>=0.3                0.05       945            -0.087640 -0.399574 0.469841 0.950728      0.307692   -0.831198
 trip03_logret5s+pup5s+logret1s       combo  drift>=0.1&imb>=0.1                0.10      1369            -0.099521 -0.453403 0.468225 0.944327      0.461538   -0.878841
          pair01_logret1s+pup5s       combo  drift>=0.0&imb>=0.3                0.00       998            -0.096441 -0.439893 0.468938 0.945891      0.307692   -1.057999
          pair01_logret1s+pup5s       combo drift>=0.05&imb>=0.3                0.05       968            -0.103273 -0.471169 0.467975 0.942153      0.307692   -1.233966
          pair01_logret1s+pup5s       combo  drift>=0.0&imb>=0.1                0.00      1476            -0.107369 -0.489408 0.467480 0.940040      0.307692   -1.337645
 trip03_logret5s+pup5s+logret1s       combo  drift>=0.1&imb>=0.3                0.10       927            -0.112785 -0.514302 0.466019 0.937040      0.307692   -1.477517
trip01_pup5s+logret1s+logret10s       combo  drift>=0.0&imb>=0.3                0.00       532            -0.116602 -0.532803 0.466165 0.934757      0.333333   -1.536690
```


## Interpretation

- **pair01_logret1s+pup5s**: baseline Sharpe=-1.52/net=-0.331t → best filtered Sharpe=-0.44/net=-0.096t (combo/drift>=0.0&imb>=0.3/thr=0.0, n=998, days=13, day_cov=0.31)
- **pair07_logret10s+logret60sq50**: baseline Sharpe=-2.86/net=-0.604t → best filtered Sharpe=-2.04/net=-0.436t (queue_imb/aligned/thr=0.5, n=433, days=14, day_cov=0.36)
- **pair08_logret5s+pup5s**: baseline Sharpe=-1.57/net=-0.342t → best filtered Sharpe=-0.80/net=-0.176t (combo/drift>=0.05&imb>=0.1/thr=0.05, n=2079, days=14, day_cov=0.36)
- **trip03_logret5s+pup5s+logret1s**: baseline Sharpe=-1.46/net=-0.317t → best filtered Sharpe=-0.35/net=-0.078t (combo/drift>=0.0&imb>=0.3/thr=0.0, n=974, days=13, day_cov=0.31)
- **trip01_pup5s+logret1s+logret10s**: baseline Sharpe=-1.51/net=-0.329t → best filtered Sharpe=-0.53/net=-0.117t (combo/drift>=0.0&imb>=0.3/thr=0.0, n=532, days=9, day_cov=0.33)

## What to try next if REJECT

- Combine microprice drift with regime classifier (green/red day per HC #428 R1)
- Trade-side asymmetry: short signals had better historical edge — restrict filter to short side only
- Replace static threshold with rolling-percentile (e.g. drift >= p70 of last 1000 entries)
- Try shorter lookback (250ms-500ms) — signal predictive horizon is sub-second
- Stack: microprice + signed-volume + spread state — 3-feature gate
- Re-examine fillsim — if adverse selection comes from queue wait (not signal), filter may need queue-ahead bucket

## Methodology notes & caveats

- L1 from `data/processed/mbo_book_features/{date}_book_features.npz` (raw bid/ask price + size, 30-feature schema).
- Prices encoded as ticks RELATIVE to session anchor; microprice and drift are differences so anchor cancels.
- Microprice snapshot taken at (entry - 50 ms): strictly causal.
- 5s lookback rows that pre-date session start are dropped.
- Pooled OOT analysis only (v1 spec). Walk-forward deferred.
- Cost 0.376 t passive-limit already in fills.net_ticks.
- Sharpe annualized at sqrt(252) on per-trade returns (so it's a trade-Sharpe, not a daily-Sharpe — use for cell-comparison only).
