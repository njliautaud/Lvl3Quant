# Task 6 — FIFO Queue Back-off Sim (proxy)

Modeled: place limit `backoff` ticks behind touch on signal direction. Filled iff price moves against us by `backoff` ticks first (MAE proxy). Edge after fill = MFE-in-direction − backoff − commission.

| Side | Backoff (t) | Fill% | n_filled | Net PnL/fill (t) |
|---|---:|---:|---:|---:|
| long | 0 | 4.9 | 9 | 14.51288890838623 |
| long | 1 | 0.0 | 0 | None |
| long | 2 | 0.0 | 0 | None |
| long | 3 | 0.0 | 0 | None |
| short | 0 | 100.0 | 1206 | -3.909996271133423 |
| short | 1 | 88.4 | 1066 | -4.625530242919922 |
| short | 2 | 70.5 | 850 | -5.323646545410156 |
| short | 3 | 53.3 | 643 | -6.2959065437316895 |
