# Chase + Exit Optimization Sweep — Mar 2-5, z3

**Run:** 2026-04-27 00:16 ET
**Setup:** Real Rust FIFO sim, chase entry (2t/5repr), z=3 threshold, 60s hold, prime hours

## Results Table

| Config              | CNN-Mamba PnL | CNN-Mamba WR / PF | Mamba_v7 PnL | Mamba_v7 WR / PF |
|---------------------|---------------|-------------------|--------------|------------------|
| base (SL20)         | -$211         | 49.2% / 0.97      | +$168        | 43.4% / 1.04     |
| tight_sl10          | -$589         | 43.0% / 0.93      | -$885        | 37.2% / 0.82     |
| mae_exit (SL15)     | -$66          | 44.7% / 0.99      | -$437        | 37.7% / 0.91     |
| ratchet (SL15)      | -$1872        | 30.2% / 0.66      | -$1035       | 25.6% / 0.66     |
| **vol_exit (SL15+vol5/5b)** | **+$273** | **48.4% / 1.03** | **+$140** | **42.3% / 1.03** |
| combo               | +$58          | 43.7% / 1.01      | -$174        | 37.7% / 0.96     |

## Key Findings

### 1. Volatility-based exit is the winner ✅
Both CNN-Mamba and Mamba_v7 turn profitable with `--vol-exit-ticks 5 --vol-exit-bars 5`.
This catches fast adverse moves (5 ticks in 500ms) BEFORE they hit the 20-tick SL.
Converts a -$261 SL hit into ~-$120 early-exit. Modest but enough to flip profitable.

### 2. Tighter fixed SL is COUNTERPRODUCTIVE ❌
Dropping SL from 20 to 10 makes everything worse. Many winning trades take a normal drawdown
before recovering — tight SL stops them out before MFE materializes. **Don't fight the noise floor.**

### 3. Ratchet stop is destructive ❌
WR collapses to 25-30%. Locks in profits too aggressively, exits trades that would have continued.
Probably needs a much higher MFE threshold before activating.

### 4. MAE-exit alone barely helps
Setting MAE-exit at 8t @ 30s sec hold helps slightly on CNN-Mamba but hurts Mamba_v7.
Needs to be combined with vol-exit (combo config did better than mae_exit alone).

## What this means for live trading

**Production execution config (current best):**
```
--chase-entry --chase-max-ticks 2 --chase-max-reprices 5
--stop-loss-ticks 15
--vol-exit-ticks 5 --vol-exit-bars 5
--hold-ms 60000 --latency-ms 5
--conviction-exit-bars 20 --conviction-exit-mag 0.5
--prime-hours
```

**Caveats:**
- Total PnL +$273 over 4 days = ~$68/day per contract. Modest.
- Alpha is real but execution costs eat most of it. To scale we need either:
  (a) Larger size at high z (z>=4 if signal supports it after this filter)
  (b) More signals per day (more models / fusion)
  (c) Lower latency (5ms is already optimistic)
- Mar 4 still loses on every config — that day is structurally adverse for our signal.

## Next experiments queue

1. Apply vol_exit config across z=1,2,3,4 to find the sweet spot trade-count vs WR.
2. Test on full Mar 9-13 OOT (need new fold predictions — ask Neptune to extend).
3. Combine vol_exit + time-of-day filter (drop 11:30 + 13:30 buckets).
4. Try size-scaling: 1 contract @ z3, 2 @ z4, 3 @ z5.
