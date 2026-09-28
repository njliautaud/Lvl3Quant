# HC #405 — WINNERS

Produced: 2026-05-17 01:08:59 ET

## Bar to beat (trial 278)

| metric | value |
|---|---|
| Sharpe | 13.48 |
| tk/fill | 1.99 |
| day_conc | 0.186 |
| n_fills | 195 |

Trial 278 is 30s SHORT passive_at_touch_plus_2 (the +2 credit accounts for ~+2.0 of the +1.99 tk/fill).

## Win gates (signal-driven configs)

- Sharpe ≥ 5.0
- tk/fill ≥ 0.5
- day_conc ≤ 0.2
- n_fills ≥ 30
- entry_order = passive_at_touch (NO passive credit)
- exit_family ≠ fixed_hold (must be dynamic)

## SIGNAL-DRIVEN winners (passive_at_touch, dynamic exit)

**NONE.** No dynamic exit family at passive_at_touch entry passes the win gates on LONG side / 10s or 30s heads.

## ANY-ENTRY-ORDER winners (incl. passive_+1, passive_+2 credit)

| side | head | entry_order | exit_family | exit_param | n_fills | sharpe | mean_net | day_conc | pf | wr | mean_exit_sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long | 30s | passive_at_touch_plus_2 | trailing_stop | 0.500 | 66 | 5.972 | 1.760 | 0.182 | 3.033 | 71.212 | 18.258 |
| long | 30s | passive_at_touch_plus_2 | trailing_stop | 1.000 | 66 | 5.642 | 1.669 | 0.192 | 2.855 | 68.182 | 18.712 |
