# K=1 Floor Verdict — P3 book-OFI, h=5s, long, forward

_Generated 2026-05-22 08:36:51 — no lookahead, entry-event only._

## Config

- **pressure**: P3_book_ofi_5s_proxy (ofi_book_5s)
- **K**: 1
- **horizon**: 5s
- **policy**: entry-event sign + |p|>=per-day p75, no forward peek
- **cost_ticks_rt**: 0.376
- **n_days_loaded**: 32

## LONG (deployable floor candidate)

- net_ticks/event: **-0.2349**
- win_rate: **0.4406**
- sharpe_daily: **-9.40**
- prof_days: **5/32**
- n_events: **203,361**
- sharpe_green/red: -5.73 / -12.37 (imbalance 0.54)
- day_concentration: 0.05

## SHORT MIRROR (sign-accident check)

- net_ticks/event: **-0.4525**
- win_rate: **0.4386**
- sharpe_daily: **-8.79**
- prof_days: **4/32**
- n_events: **191,006**
- sharpe_green/red: -8.61 / -17.23 (imbalance 0.50)
- day_concentration: 0.05

## Verdict

- net_ticks >= 0.1: FAIL (-0.2349)
- WR >= 0.52: FAIL (0.4406)
- prof_days >= 30/32: FAIL (5)
- regime_imbalance <= 0.5: FAIL (0.54)
- day_concentration <= 0.7: PASS (0.05)

**REJECT HEADLINE.** Edge collapses without the forward-peek window. The +0.67 net at K=4 is lookahead-driven (forward stream-coherence is peeking at ~1s of post-entry pressure, which is not available live).

_Mirror (short, same magnitude gate): net=-0.4525, WR=0.4386. Mirror clearly worse, so long-side sign is not a coin-flip accident (but that alone doesn't make long deployable).