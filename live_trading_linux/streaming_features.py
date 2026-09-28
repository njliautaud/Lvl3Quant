#!/usr/bin/env python3
"""
streaming_features.py — Online feature computation matching compute_derived().

The 21 features expected by the LGBM model:
    time_delta, event_type, side, price, qty, spread,
    rolling_ofi_100, cancel_side_asym_100, event_density_50,
    price_mom_20, qty_price_mom_20, cum_delta, roll_delta_500,
    ofi_short_20, cancel_rate_100, trade_rate_50, add_side_asym_100,
    spread_velocity_50, qty_add_imbalance_100, price_sign_mom_200,
    fill_recovery_20

Event type codes: 0=add, 1=cancel, 3=trade.
Side codes:       0=bid, 1=ask.

Training reference: /home/saturn/Lvl3Quant/lgbm_prod_wf_60_5.py::compute_derived

IMPORTANT CORRECTNESS NOTE:
    The training code uses `np.convolve(x, W_k, mode='full')[:N]` where
    W_k = np.ones(k)/k. For position i this equals:
        (1/k) * sum_{j=max(0, i-k+1)}^{i} x[j]
    i.e. a causal rolling SUM divided by k (NOT by the number of samples
    actually in the window). The streaming equivalent is therefore an
    incremental sum over a deque of maxlen=k, always divided by k.

    Warm-up period: for i < k-1 the denominator is still k, so early values
    are depressed until the window fills.  We reproduce that behaviour
    exactly so online predictions match offline training.

We avoid numpy convolutions entirely — just deques + running sums.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Feature name / index table — kept identical to training script
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'time_delta', 'event_type', 'side', 'price', 'qty', 'spread',
    'rolling_ofi_100', 'cancel_side_asym_100', 'event_density_50',
    'price_mom_20', 'qty_price_mom_20', 'cum_delta', 'roll_delta_500',
    'ofi_short_20', 'cancel_rate_100', 'trade_rate_50', 'add_side_asym_100',
    'spread_velocity_50', 'qty_add_imbalance_100', 'price_sign_mom_200',
    'fill_recovery_20',
]
assert len(FEATURE_NAMES) == 21


class _RollingSum:
    """Fixed-window rolling sum over the last `window` samples.

    Matches `np.convolve(x, np.ones(window)/window, mode='full')[:N]` when
    caller divides sum() by window (see RollingMean below).
    """

    __slots__ = ("window", "buf", "s")

    def __init__(self, window: int) -> None:
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.s: float = 0.0

    def push(self, x: float) -> None:
        if len(self.buf) == self.window:
            # About to evict the oldest
            self.s -= self.buf[0]
        self.buf.append(float(x))
        self.s += float(x)

    def sum(self) -> float:
        return self.s

    def mean_over_window(self) -> float:
        """Sum divided by fixed window size — matches convolve(W_k, mode='full')."""
        return self.s / self.window


class StreamingFeatures:
    """Incrementally compute the 21 features expected by the LGBM model.

    Call `update(event_type, side, price, qty, spread, time_delta)` once per
    MBO event; it returns an np.ndarray of shape (21,) dtype float32.

    The class is stateful: do not share across symbols.
    """

    # Window sizes referenced in compute_derived
    W20  = 20
    W50  = 50
    W100 = 100
    W200 = 200
    W500 = 500

    def __init__(self) -> None:
        # --- Raw-value deques needed for multi-step differences ---------------
        # price history to compute pdiff[i] = price[i] - price[i-20]
        self._price_hist: deque = deque(maxlen=self.W20 + 1)  # need index -20
        # spread history for sdiff[i] = spread[i] - spread[i-1]
        self._prev_spread: Optional[float] = None

        # --- Rolling-sum accumulators (one per convolve in compute_derived) --
        self.rofi      = _RollingSum(self.W100)   # rolling_ofi_100     (col 6)
        self.casym     = _RollingSum(self.W100)   # cancel_side_asym_100(col 7)
        self.dens      = _RollingSum(self.W50)    # event_density_50    (col 8)
        self.pmom      = _RollingSum(self.W20)    # price_mom_20        (col 9)
        self.qpmom     = _RollingSum(self.W20)    # qty_price_mom_20    (col 10)
        self.cr_ma     = _RollingSum(self.W500)   # crma = conv(cr, W500) (cd)
        self.cr2_ma    = _RollingSum(self.W500)   # conv(cr^2, W500)     (cd)
        self.rd5       = _RollingSum(self.W500)   # roll_delta_500      (col 12)
        self.os20      = _RollingSum(self.W20)    # ofi_short_20        (col 13)
        self.crate     = _RollingSum(self.W100)   # cancel_rate_100     (col 14)
        self.trate     = _RollingSum(self.W50)    # trade_rate_50       (col 15)
        self.aasym     = _RollingSum(self.W100)   # add_side_asym_100   (col 16)
        self.svel      = _RollingSum(self.W50)    # spread_velocity_50  (col 17)
        self.qai_num   = _RollingSum(self.W100)   # qai numerator
        self.qai_den   = _RollingSum(self.W100)   # qai denominator
        self.psm       = _RollingSum(self.W200)   # price_sign_mom_200  (col 19)
        # fill_recovery: first stage = conv(is_t*is_bid, W20), conv(is_t*is_ask, W20)
        #                second stage = conv(br + ar, W20)
        self.t_bid_ma  = _RollingSum(self.W20)
        self.t_ask_ma  = _RollingSum(self.W20)
        self.fr_ma     = _RollingSum(self.W20)    # fill_recovery_20    (col 20)

        # Cumulative OFI (for cum_delta column 11)
        self.cr_cum: float = 0.0

        # Event counter (for diagnostics / unit test parity)
        self.n_events: int = 0

    # ------------------------------------------------------------- update
    def update(
        self,
        event_type: int,
        side:       int,
        price:      float,
        qty:        float,
        spread:     float,
        time_delta: float,
    ) -> np.ndarray:
        """Ingest one event and return the 21-dim feature vector.

        Parameters match the first 6 columns of the training feature matrix.
        """
        et = float(event_type)
        sd = float(side)
        p  = float(price)
        q  = float(qty)
        sp = float(spread)
        td = float(time_delta)

        # --- Indicator flags (match compute_derived exactly) -----------------
        is_t   = 1.0 if et == 3.0 else 0.0
        is_c   = 1.0 if et == 1.0 else 0.0
        is_a   = 1.0 if et == 0.0 else 0.0
        is_bid = 1.0 if sd == 0.0 else 0.0
        is_ask = 1.0 if sd == 1.0 else 0.0

        # --- Core intermediaries ---------------------------------------------
        # ofi: trade-buy-flow minus trade-sell-flow
        bf  = is_t * is_ask * q     # buyer-initiated (trade lifts offer)
        sf  = is_t * is_bid * q     # seller-initiated (trade hits bid)
        ofi = bf - sf

        # cumulative OFI (for cum_delta)
        self.cr_cum += ofi
        cr = self.cr_cum

        # pdiff[i] = price[i] - price[i-20]; first 20 events = 0
        self._price_hist.append(p)
        if len(self._price_hist) == self.W20 + 1:
            pdiff = p - self._price_hist[0]
        else:
            pdiff = 0.0
        sign_pdiff = 0.0 if pdiff == 0.0 else (1.0 if pdiff > 0.0 else -1.0)

        # sdiff[i] = spread[i] - spread[i-1]; sdiff[0] = 0
        if self._prev_spread is None:
            sdiff = 0.0
        else:
            sdiff = sp - self._prev_spread
        self._prev_spread = sp

        # event_density input: np.where(td > 1e-9, 1/(td+1e-6), 1)
        if td > 1e-9:
            dens_in = 1.0 / (td + 1e-6)
        else:
            dens_in = 1.0

        # Cancel side asymmetry input
        casym_in = is_c * is_bid - is_c * is_ask
        # Add side asymmetry input
        aasym_in = is_a * is_bid - is_a * is_ask
        # Quantity-add imbalance pieces
        baq = is_a * is_bid * q
        aaq = is_a * is_ask * q
        qai_num_in = baq - aaq
        qai_den_in = baq + aaq

        # --- Push all values into rolling accumulators -----------------------
        # NOTE: order matters only for our own bookkeeping — each sum is
        # independent, but we must push BEFORE reading the current mean so
        # that the current event is included in the window (matches convolve).
        self.rofi.push(ofi)
        self.casym.push(casym_in)
        self.dens.push(dens_in)
        self.pmom.push(pdiff)
        self.qpmom.push(q * sign_pdiff)
        self.cr_ma.push(cr)
        self.cr2_ma.push(cr * cr)
        self.rd5.push(ofi)
        self.os20.push(ofi)
        self.crate.push(is_c)
        self.trate.push(is_t)
        self.aasym.push(aasym_in)
        self.svel.push(sdiff)
        self.qai_num.push(qai_num_in)
        self.qai_den.push(qai_den_in)
        self.psm.push(sign_pdiff)

        # Fill-recovery is a two-stage convolution. Stage 1 feeds point-wise
        # scaled values into stage 2.
        self.t_bid_ma.push(is_t * is_bid)
        self.t_ask_ma.push(is_t * is_ask)
        br = self.t_bid_ma.mean_over_window() * is_a * is_bid
        ar = self.t_ask_ma.mean_over_window() * is_a * is_ask
        self.fr_ma.push(br + ar)

        # --- Read current feature values -------------------------------------
        rofi_v  = self.rofi.mean_over_window()
        casym_v = self.casym.mean_over_window()
        dens_v  = self.dens.mean_over_window()
        pmom_v  = self.pmom.mean_over_window()
        qpmom_v = self.qpmom.mean_over_window()
        crma    = self.cr_ma.mean_over_window()
        cr2ma   = self.cr2_ma.mean_over_window()
        var     = cr2ma - crma * crma
        if var < 1e-8:
            var = 1e-8
        crstd = math.sqrt(var)
        cd_v    = (cr - crma) / (crstd + 1e-6)
        rd5_v   = self.rd5.mean_over_window()
        os20_v  = self.os20.mean_over_window()
        crate_v = self.crate.mean_over_window()
        trate_v = self.trate.mean_over_window()
        aasym_v = self.aasym.mean_over_window()
        svel_v  = self.svel.mean_over_window()
        qnum    = self.qai_num.mean_over_window()
        qden    = self.qai_den.mean_over_window()
        qai_v   = qnum / (qden + 1e-6)
        psm_v   = self.psm.mean_over_window()
        fr_v    = self.fr_ma.mean_over_window()

        self.n_events += 1

        out = np.array([
            td, et, sd, p, q, sp,
            rofi_v, casym_v, dens_v,
            pmom_v, qpmom_v, cd_v, rd5_v,
            os20_v, crate_v, trate_v, aasym_v,
            svel_v, qai_v,
            psm_v, fr_v,
        ], dtype=np.float32)
        return out


# ---------------------------------------------------------------------------
# Self-test: feed synthetic events through the streaming class and compare
# against compute_derived() (the offline batch version).
# ---------------------------------------------------------------------------
def _compute_derived_reference(ev: np.ndarray) -> np.ndarray:
    """Verbatim copy of compute_derived() from lgbm_prod_wf_60_5.py, for parity
    testing only.  Kept here so this module is self-contained."""
    N = len(ev)
    td = ev[:, 0].astype('f4'); et = ev[:, 1].astype('f4'); side = ev[:, 2].astype('f4')
    price = ev[:, 3].astype('f4'); qty = ev[:, 4].astype('f4'); sprd = ev[:, 5].astype('f4')
    W20 = np.ones(20, 'f4') / 20; W50 = np.ones(50, 'f4') / 50; W100 = np.ones(100, 'f4') / 100
    W200 = np.ones(200, 'f4') / 200; W500 = np.ones(500, 'f4') / 500
    is_t = (et == 3).astype('f4'); is_c = (et == 1).astype('f4'); is_a = (et == 0).astype('f4')
    is_bid = (side == 0).astype('f4'); is_ask = (side == 1).astype('f4')
    bf = is_t * is_ask * qty; sf = is_t * is_bid * qty; ofi = bf - sf
    rofi  = np.convolve(ofi, W100, mode='full')[:N]
    casym = np.convolve(is_c * is_bid - is_c * is_ask, W100, mode='full')[:N]
    dens  = np.convolve(np.where(td > 1e-9, 1. / (td + 1e-6), 1.).astype('f4'), W50, mode='full')[:N]
    pdiff = np.zeros(N, 'f4'); pdiff[20:] = price[20:] - price[:-20]
    pmom  = np.convolve(pdiff, W20, mode='full')[:N]
    qpmom = np.convolve(qty * np.sign(pdiff), W20, mode='full')[:N]
    cr    = np.cumsum(ofi).astype('f4')
    crma  = np.convolve(cr, W500, mode='full')[:N]
    crstd = np.sqrt(np.maximum(np.convolve(cr ** 2, W500, mode='full')[:N] - crma ** 2, 1e-8))
    cd    = (cr - crma) / (crstd + 1e-6)
    rd5   = np.convolve(ofi, W500, mode='full')[:N]
    os20  = np.convolve(ofi, W20, mode='full')[:N]
    crate = np.convolve(is_c, W100, mode='full')[:N]
    trate = np.convolve(is_t, W50, mode='full')[:N]
    aasym = np.convolve(is_a * is_bid - is_a * is_ask, W100, mode='full')[:N]
    sdiff = np.zeros(N, 'f4'); sdiff[1:] = sprd[1:] - sprd[:-1]
    svel  = np.convolve(sdiff, W50, mode='full')[:N]
    baq = is_a * is_bid * qty; aaq = is_a * is_ask * qty
    qai = (np.convolve(baq - aaq, W100, mode='full')[:N]
           / (np.convolve(baq + aaq, W100, mode='full')[:N] + 1e-6))
    psm = np.convolve(np.sign(pdiff), W200, mode='full')[:N]
    br_ma = np.convolve(is_t * is_bid, W20, mode='full')[:N]
    ar_ma = np.convolve(is_t * is_ask, W20, mode='full')[:N]
    br = br_ma * is_a * is_bid
    ar = ar_ma * is_a * is_ask
    fr = np.convolve(br + ar, W20, mode='full')[:N]
    d = np.stack([rofi, casym, dens, pmom, qpmom, cd, rd5, os20,
                  crate, trate, aasym, svel, qai, psm, fr], axis=1)
    return np.concatenate([ev, d], axis=1).astype('f4')


def _self_test(n: int = 2_000, tol: float = 2e-3, seed: int = 7) -> None:
    """Compare streaming vs batch on synthetic events.  Tolerance is a bit
    loose because f4 convolution accumulates FP error differently than an
    incremental sum, but both should agree to within a few millis."""
    rng = np.random.default_rng(seed)
    # Build plausible events: time_delta ~ exp, mixture of add/cancel/trade,
    # side ~ uniform, price random-walk, qty small int, spread near 1 tick.
    td_arr    = rng.exponential(scale=0.01, size=n).astype('f4')
    et_choices = rng.choice([0, 1, 3], size=n, p=[0.45, 0.45, 0.10]).astype('f4')
    side_arr  = rng.integers(0, 2, size=n).astype('f4')
    price_arr = (5000.0 + np.cumsum(rng.normal(0, 0.25, size=n))).astype('f4')
    qty_arr   = rng.integers(1, 5, size=n).astype('f4')
    spread_arr = (0.25 + 0.25 * rng.integers(0, 3, size=n)).astype('f4')

    ev = np.stack([td_arr, et_choices, side_arr, price_arr, qty_arr, spread_arr], axis=1)
    ref = _compute_derived_reference(ev)  # (N, 21)

    sf = StreamingFeatures()
    online = np.empty((n, 21), dtype=np.float32)
    for i in range(n):
        online[i] = sf.update(
            int(ev[i, 1]), int(ev[i, 2]),
            float(ev[i, 3]), float(ev[i, 4]),
            float(ev[i, 5]), float(ev[i, 0]),
        )

    # Compare column-by-column
    max_diffs = np.max(np.abs(online - ref), axis=0)
    ok = True
    for name, d in zip(FEATURE_NAMES, max_diffs):
        status = "OK " if d <= tol else "BAD"
        print(f"  [{status}] {name:22s} max|diff|={d:.6f}")
        if d > tol:
            ok = False

    # For cum_delta (col 11) tolerance is more generous: it involves a sqrt
    # of a subtracted variance, which amplifies float error.  We special-case
    # it and re-check with a bigger tolerance.
    cd_idx = FEATURE_NAMES.index('cum_delta')
    cd_diff = max_diffs[cd_idx]
    if cd_diff > tol:
        cd_tol = 5e-2
        still_bad = cd_diff > cd_tol
        print(f"  cum_delta re-check with tol={cd_tol}: "
              f"{'BAD' if still_bad else 'OK '} (diff={cd_diff:.4f})")
        if not still_bad:
            # If cum_delta passes the looser threshold and nothing else is bad,
            # we consider the test ok.
            other_bad = any(
                d > tol and i != cd_idx for i, d in enumerate(max_diffs)
            )
            ok = not other_bad

    if ok:
        print(f"\n  PARITY OK on N={n} events.")
    else:
        print(f"\n  PARITY FAILED — feature computation differs from training!")
    return ok


if __name__ == "__main__":
    import sys as _sys
    ok = _self_test(n=2_000)
    _sys.exit(0 if ok else 1)
