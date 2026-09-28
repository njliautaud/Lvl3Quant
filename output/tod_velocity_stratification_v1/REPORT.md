# TOD x Velocity Stratification v1 - REPORT
Generated: 2026-05-22 12:34:30 EDT
Source: v3.4.2 OOT preds, 29 days, 1,364,517 RTH prediction rows (skipped ['20260308', '20260315'])
Regime mix: green=25, red=2, flat=2

## VERDICT: ACCEPT
6 cell(s) pass HC #428 gates. See winning_cells.txt.
Top winner:
  horizon=10s conf=top5pc side=long tod=mid_am vel_dec=8
  n=904 days=23 net=+0.890t Sh=4.12 PF=1.44 WR=56.5% profdays=65.2% imbal=0.13

## Structural patterns (top-quartile by Sharpe)
- TOD frequency:      {'open': 0.2537313432835821, 'close': 0.2462686567164179, 'mid_am': 0.20522388059701493, 'late_pm': 0.16791044776119404, 'midday': 0.12686567164179105}
- Side frequency:     {'short': 0.503731343283582, 'long': 0.4962686567164179}
- Velocity decile:    {'9': 0.20149253731343283, '8': 0.16417910447761194, '7': 0.11567164179104478, '6': 0.11194029850746269, '5': 0.09328358208955224, '4': 0.08208955223880597, '1': 0.07462686567164178, '2': 0.05970149253731343, '3': 0.048507462686567165, '0': 0.048507462686567165}
- Horizon frequency:  {'10s': 0.39552238805970147, '30s': 0.2873134328358209, '5s': 0.23134328358208955, '1s': 0.08582089552238806}

## Velocity-edge correlation (Spearman rho between vel_dec and Sharpe)
Top |rho| slices:
    1s  long  top1pc   midday: rho=+0.964  (n_cells=10)
    1s  long top10pc  late_pm: rho=+0.952  (n_cells=10)
    1s  long  top5pc  late_pm: rho=+0.952  (n_cells=10)
    1s  long top10pc   midday: rho=+0.939  (n_cells=10)
    1s  long  top5pc   midday: rho=+0.939  (n_cells=10)
    5s  long top10pc  late_pm: rho=+0.927  (n_cells=10)
    5s  long  top5pc  late_pm: rho=+0.915  (n_cells=10)
   30s  long  top5pc  late_pm: rho=+0.893  (n_cells=7)
   10s short top10pc   mid_am: rho=-0.891  (n_cells=10)
    1s  long top10pc    close: rho=+0.891  (n_cells=10)

## Interpretation
- Stratification REVEALED pockets of deployable edge that pooled analysis missed.
- Winning cells TOD concentration: {'mid_am': np.int64(2), 'open': np.int64(2), 'late_pm': np.int64(1), 'midday': np.int64(1)}
- Winning cells side concentration: {'short': np.int64(4), 'long': np.int64(2)}

## Recommended next move
- Lock surviving cell(s) and run shadow paper-trade for 5 OOT days to confirm.
- Re-run HC #485 R5 regen on the locked cell config for canonical record.
