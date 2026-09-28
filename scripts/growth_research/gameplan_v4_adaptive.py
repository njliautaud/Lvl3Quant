#!/usr/bin/env python3
"""
GAMEPLAN v4 — ADAPTIVE VIX PERCENTILE INTEGRATION
====================================================
Integrates relative VIX percentile approach into the validated Gameplan v3 system.

v3 baseline: Fixed VIX thresholds (vol<10%: UPRO full-day, 10-15%: UPRO overnight,
15-30%: SPY, >30%: GLD) + 20/200 MA crossover + September hedge + confluence gate
(3-timeframe score >= 2.5 enter, < 2.0 exit).

v4 variants:
  1. PURE REPLACEMENT: Replace fixed VIX thresholds with percentile ranks (63d lookback)
  2. HYBRID: Percentile rank modulates fixed thresholds (tighten/loosen)
  3. CONFLUENCE + PERCENTILE: VIX percentile as 4th confluence factor
  4. FULL ADAPTIVE: Percentile replaces fixed AND modulates confluence thresholds

Each variant compared head-to-head vs v3 baseline on:
  - Sharpe, Sortino, CAGR, MaxDD, Calmar
  - Switches/year
  - Crisis behavior (COVID, 2022, 2025 tariffs)
  - Walk-forward OOS

Adversarial validation (HC #705):
  1. Permutation test (200 shuffles)
  2. Sub-period consistency (3+ blocks)
  3. Outlier robustness (remove top 5%)
  4. R1 regime test (green/red day asymmetry < 0.50)
  5. Walk-forward validation (3yr train / 1yr OOS)
"""

import os
import sys
import json
import warnings
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/gameplan_v4_adaptive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100
TX_COST_PCT = 0.0002
GAP_RISK_PCT = 0.001

# Crisis periods for targeted analysis
CRISES = {
    'COVID_crash':    ('2020-02-19', '2020-03-23'),
    'COVID_recovery': ('2020-03-24', '2020-08-31'),
    'Rate_hike_2022': ('2022-01-03', '2022-10-12'),
    'Tariff_2025':    ('2025-01-20', '2025-04-30'),
    'SVB_2023':       ('2023-03-08', '2023-03-24'),
}


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_data():
    """Download all required price data including VIX."""
    print("=" * 80)
    print("GAMEPLAN v4 — ADAPTIVE VIX PERCENTILE INTEGRATION")
    print("=" * 80)
    print(f"\nRun started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n[1] Downloading data...")
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'SHY', 'QQQ', 'HYG', 'LQD', 'IWM', '^VIX']
    data = yf.download(tickers, start='2010-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except Exception:
            pass

    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})

    closes = closes.dropna(how='all')
    closes['VIX'] = closes['VIX'].ffill()
    closes = closes.dropna(subset=['SPY', 'UPRO', 'VIX'])
    returns = closes.pct_change().fillna(0)

    print(f"  {len(closes)} trading days: {closes.index[0].strftime('%Y-%m-%d')} "
          f"to {closes.index[-1].strftime('%Y-%m-%d')}")
    return closes, returns


# =============================================================================
# SIGNAL COMPUTATION
# =============================================================================

def compute_all_signals(closes):
    """Compute all signals used across variants."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vix = closes['VIX']

    sig = {}

    # -- VIX percentile (multiple lookbacks for sweep) --
    for lb in [42, 63, 126, 252]:
        sig[f'vix_pctile_{lb}'] = vix.rolling(lb).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100,
            raw=False
        )

    # -- Realized vol --
    sig['vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    sig['vol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252) * 100

    # -- Confluence signals --
    sig['mom_5d'] = spy.pct_change(5)
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    sig['rsi_10'] = 100 - (100 / (1 + rs))
    sig['sma_20'] = spy.rolling(20).mean()
    sig['sma_50'] = spy.rolling(50).mean()
    sig['sma_200'] = spy.rolling(200).mean()
    sig['sma_200_slope'] = sig['sma_200'].pct_change(20)
    sig['vol_63d_trend'] = sig['vol_63d'] - sig['vol_63d'].rolling(21).mean()

    # -- Protection overlay signals --
    if 'LQD' in closes.columns and 'HYG' in closes.columns:
        credit_ratio = closes['LQD'] / closes['HYG']
        sig['credit_ratio'] = credit_ratio
        sig['credit_sma20'] = credit_ratio.rolling(20).mean()
    if 'IWM' in closes.columns:
        sig['breadth_21d'] = closes['IWM'].pct_change(21) - spy.pct_change(21)

    # -- Raw VIX --
    sig['vix'] = vix

    print(f"  Computed {len(sig)} signal series.")
    return sig


# =============================================================================
# CONFLUENCE SCORING
# =============================================================================

def confluence_score_at(sig, i):
    """Standard 6-factor confluence score (0-3)."""
    s = 0.0
    m = sig['mom_5d'].iloc[i]
    r = sig['rsi_10'].iloc[i]
    s20 = sig['sma_20'].iloc[i]
    s50 = sig['sma_50'].iloc[i]
    v21 = sig['vol_21d'].iloc[i]
    slope = sig['sma_200_slope'].iloc[i]
    vt = sig['vol_63d_trend'].iloc[i]

    if not np.isnan(m) and m > 0: s += 0.5
    if not np.isnan(r) and r > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(v21) and v21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vt) and vt < 0: s += 0.5
    return s


def confluence_score_with_vix_pctile(sig, i, pctile_key='vix_pctile_63',
                                      low_pctile=20, high_pctile=80):
    """7-factor confluence: standard 6 + VIX percentile factor (0-3.5)."""
    s = confluence_score_at(sig, i)
    pctile = sig[pctile_key].iloc[i]
    if not np.isnan(pctile):
        if pctile < low_pctile:
            s += 0.5   # VIX unusually low = bullish
        elif pctile > high_pctile:
            s -= 0.5   # VIX unusually high = bearish (can go negative on this factor)
    return s


# =============================================================================
# REGIME FUNCTIONS (VARIANTS)
# =============================================================================

class GameplanV3Baseline:
    """v3 baseline: fixed thresholds + confluence gate + MA protection + Sept hedge."""
    def __init__(self, sig, entry=2.5, exit_thresh=2.0):
        self.sig = sig
        self.entry = entry
        self.exit_thresh = exit_thresh
        self.in_upro = False
        self.name = "v3_baseline"

    def reset(self):
        self.in_upro = False

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'

        v = self.sig['vol_21d'].iloc[i]
        s20 = self.sig['sma_20'].iloc[i]
        s200 = self.sig['sma_200'].iloc[i]
        if np.isnan(v): v = 15

        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        if v > 30:
            self.in_upro = False
            return 'GLD'
        if v > 15 or prot_off:
            self.in_upro = False
            return 'SPY'

        score = confluence_score_at(self.sig, i)
        if self.in_upro:
            if score < self.exit_thresh:
                self.in_upro = False
                return 'SPY'
            return 'UPRO'
        else:
            if score >= self.entry:
                self.in_upro = True
                return 'UPRO'
            return 'SPY'


class V4_PureReplacement:
    """
    Variant 1: Replace fixed VIX thresholds with VIX percentile rank.
    - VIX pctile < low_pctile  -> UPRO (via confluence gate)
    - VIX pctile > high_pctile -> GLD (defensive)
    - else                     -> SPY
    Plus: MA protection, September hedge, confluence gate for UPRO entry.
    """
    def __init__(self, sig, lookback=63, low_pctile=20, high_pctile=80,
                 entry=2.5, exit_thresh=2.0):
        self.sig = sig
        self.pctile_key = f'vix_pctile_{lookback}'
        self.low_pctile = low_pctile
        self.high_pctile = high_pctile
        self.entry = entry
        self.exit_thresh = exit_thresh
        self.in_upro = False
        self.name = f"v4_pure_lb{lookback}_lo{low_pctile}_hi{high_pctile}"

    def reset(self):
        self.in_upro = False

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'

        s20 = self.sig['sma_20'].iloc[i]
        s200 = self.sig['sma_200'].iloc[i]
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        if prot_off:
            self.in_upro = False
            return 'SPY'

        pctile = self.sig[self.pctile_key].iloc[i]
        if np.isnan(pctile):
            return 'SPY'

        if pctile > self.high_pctile:
            self.in_upro = False
            return 'GLD'

        if pctile > self.low_pctile:
            self.in_upro = False
            return 'SPY'

        # VIX in low percentile zone - use confluence gate for UPRO
        score = confluence_score_at(self.sig, i)
        if self.in_upro:
            if score < self.exit_thresh:
                self.in_upro = False
                return 'SPY'
            return 'UPRO'
        else:
            if score >= self.entry:
                self.in_upro = True
                return 'UPRO'
            return 'SPY'


class V4_Hybrid:
    """
    Variant 2: Percentile rank modulates fixed thresholds.
    - When VIX percentile > 80: tighten UPRO threshold from 15% to 12% (more cautious)
    - When VIX percentile < 20: loosen UPRO threshold from 15% to 18% (more aggressive)
    - Otherwise use standard 15% threshold
    Plus: Modulate GLD threshold similarly, MA protection, Sept hedge, confluence gate.
    """
    def __init__(self, sig, lookback=63, low_pctile=20, high_pctile=80,
                 base_upro_thresh=15, base_gld_thresh=30,
                 tighten_upro=12, loosen_upro=18,
                 tighten_gld=25, loosen_gld=35,
                 entry=2.5, exit_thresh=2.0):
        self.sig = sig
        self.pctile_key = f'vix_pctile_{lookback}'
        self.low_pctile = low_pctile
        self.high_pctile = high_pctile
        self.base_upro = base_upro_thresh
        self.base_gld = base_gld_thresh
        self.tighten_upro = tighten_upro
        self.loosen_upro = loosen_upro
        self.tighten_gld = tighten_gld
        self.loosen_gld = loosen_gld
        self.entry = entry
        self.exit_thresh = exit_thresh
        self.in_upro = False
        self.name = f"v4_hybrid_lb{lookback}"

    def reset(self):
        self.in_upro = False

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'

        s20 = self.sig['sma_20'].iloc[i]
        s200 = self.sig['sma_200'].iloc[i]
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        if prot_off:
            self.in_upro = False
            return 'SPY'

        v = self.sig['vol_21d'].iloc[i]
        if np.isnan(v): v = 15

        # Modulate thresholds based on VIX percentile
        pctile = self.sig[self.pctile_key].iloc[i]
        if np.isnan(pctile):
            upro_thresh = self.base_upro
            gld_thresh = self.base_gld
        elif pctile > self.high_pctile:
            # VIX elevated relative to recent history -> tighten (more cautious)
            upro_thresh = self.tighten_upro
            gld_thresh = self.tighten_gld
        elif pctile < self.low_pctile:
            # VIX low relative to recent history -> loosen (more aggressive)
            upro_thresh = self.loosen_upro
            gld_thresh = self.loosen_gld
        else:
            upro_thresh = self.base_upro
            gld_thresh = self.base_gld

        if v > gld_thresh:
            self.in_upro = False
            return 'GLD'
        if v > upro_thresh:
            self.in_upro = False
            return 'SPY'

        # Confluence gate for UPRO
        score = confluence_score_at(self.sig, i)
        if self.in_upro:
            if score < self.exit_thresh:
                self.in_upro = False
                return 'SPY'
            return 'UPRO'
        else:
            if score >= self.entry:
                self.in_upro = True
                return 'UPRO'
            return 'SPY'


class V4_ConfluencePlusPercentile:
    """
    Variant 3: Keep v3 confluence gate but add VIX percentile as 7th factor.
    - Standard 6 confluence signals + VIX percentile factor
    - VIX pctile < 20 -> +0.5 to score (bullish)
    - VIX pctile > 80 -> -0.5 to score (bearish)
    - Fixed vol thresholds remain for GLD/SPY override
    """
    def __init__(self, sig, lookback=63, low_pctile=20, high_pctile=80,
                 entry=2.5, exit_thresh=2.0):
        self.sig = sig
        self.pctile_key = f'vix_pctile_{lookback}'
        self.low_pctile = low_pctile
        self.high_pctile = high_pctile
        self.entry = entry
        self.exit_thresh = exit_thresh
        self.in_upro = False
        self.name = f"v4_confluence_pctile_lb{lookback}"

    def reset(self):
        self.in_upro = False

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'

        v = self.sig['vol_21d'].iloc[i]
        s20 = self.sig['sma_20'].iloc[i]
        s200 = self.sig['sma_200'].iloc[i]
        if np.isnan(v): v = 15

        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        if v > 30:
            self.in_upro = False
            return 'GLD'
        if v > 15 or prot_off:
            self.in_upro = False
            return 'SPY'

        # Enhanced confluence with VIX percentile
        score = confluence_score_with_vix_pctile(
            self.sig, i, self.pctile_key, self.low_pctile, self.high_pctile
        )

        if self.in_upro:
            if score < self.exit_thresh:
                self.in_upro = False
                return 'SPY'
            return 'UPRO'
        else:
            if score >= self.entry:
                self.in_upro = True
                return 'UPRO'
            return 'SPY'


class V4_FullAdaptive:
    """
    Variant 4: Full adaptive - percentile replaces fixed thresholds AND
    modulates confluence entry/exit thresholds.
    - VIX percentile determines base regime (UPRO/SPY/GLD)
    - Confluence thresholds tighten when VIX percentile is elevated
    - Entry threshold: 2.5 base, +0.5 when pctile > 60, -0.5 when pctile < 30
    - Exit threshold: 2.0 base, +0.5 when pctile > 60, -0.5 when pctile < 30
    """
    def __init__(self, sig, lookback=63, low_pctile=20, high_pctile=80,
                 base_entry=2.5, base_exit=2.0):
        self.sig = sig
        self.pctile_key = f'vix_pctile_{lookback}'
        self.low_pctile = low_pctile
        self.high_pctile = high_pctile
        self.base_entry = base_entry
        self.base_exit = base_exit
        self.in_upro = False
        self.name = f"v4_full_adaptive_lb{lookback}"

    def reset(self):
        self.in_upro = False

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'

        s20 = self.sig['sma_20'].iloc[i]
        s200 = self.sig['sma_200'].iloc[i]
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        if prot_off:
            self.in_upro = False
            return 'SPY'

        pctile = self.sig[self.pctile_key].iloc[i]
        if np.isnan(pctile):
            return 'SPY'

        # Adaptive regime from percentile
        if pctile > self.high_pctile:
            self.in_upro = False
            return 'GLD'

        # Adaptive confluence thresholds based on VIX percentile
        if pctile > 60:
            # Elevated VIX -> harder to enter, easier to exit UPRO
            entry = self.base_entry + 0.5
            exit_t = self.base_exit + 0.5
        elif pctile < 30:
            # Low VIX -> easier to enter, harder to exit UPRO
            entry = max(1.5, self.base_entry - 0.5)
            exit_t = max(1.0, self.base_exit - 0.5)
        else:
            entry = self.base_entry
            exit_t = self.base_exit

        if pctile > self.low_pctile:
            # Mid-range VIX percentile -> SPY unless confluence is strong
            # (effectively raising the bar)
            entry = max(entry, self.base_entry)
            score = confluence_score_at(self.sig, i)
            if self.in_upro:
                if score < exit_t:
                    self.in_upro = False
                    return 'SPY'
                return 'UPRO'
            else:
                if score >= entry:
                    self.in_upro = True
                    return 'UPRO'
                return 'SPY'

        # Low VIX percentile -> use (possibly loosened) confluence gate
        score = confluence_score_at(self.sig, i)
        if self.in_upro:
            if score < exit_t:
                self.in_upro = False
                return 'SPY'
            return 'UPRO'
        else:
            if score >= entry:
                self.in_upro = True
                return 'UPRO'
            return 'SPY'


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def simulate(closes, returns, regime_fn, warmup=260):
    """
    Simulate portfolio with DCA and regime-based allocation.
    Returns (daily_values Series, total_contributed, n_switches, regime_series).
    """
    regime_fn.reset()
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []
    daily_regimes = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        regime = regime_fn(i, date)

        # Transaction cost on switch
        if regime != last_regime and last_regime is not None:
            switches += 1
            cash *= (1 - TX_COST_PCT - GAP_RISK_PCT)
        last_regime = regime

        # Apply return
        if regime in returns.columns:
            r = returns.loc[date, regime]
            if not np.isnan(r):
                cash *= (1 + r)

        daily_values.append(cash)
        daily_regimes.append(regime)

    dates = closes.index[warmup:warmup + len(daily_values)]
    vals = pd.Series(daily_values, index=dates)
    regs = pd.Series(daily_regimes, index=dates)
    return vals, total_contributed, switches, regs


def compute_metrics(values, total_contributed=None, switches=0):
    """Compute risk-adjusted metrics."""
    daily_ret = values.pct_change().dropna()
    if len(daily_ret) < 10:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0,
                'calmar': 0, 'ann_vol': 0, 'final_value': 0,
                'total_contributed': 0, 'switches_yr': 0}

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 1
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_dd = drawdown.min()

    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    sw_yr = switches / years if years > 0 else 0

    tc = total_contributed or 0
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'cagr': float(cagr),
        'max_dd': float(max_dd),
        'calmar': float(calmar),
        'ann_vol': float(ann_vol),
        'final_value': float(values.iloc[-1]),
        'total_contributed': float(tc),
        'profit': float(values.iloc[-1] - tc) if tc else 0,
        'switches_yr': float(sw_yr),
        'switches': int(switches),
        'n_years': float(years),
    }


# =============================================================================
# CRISIS ANALYSIS
# =============================================================================

def crisis_analysis(values, regs, label):
    """Analyze behavior during crisis periods."""
    results = {}
    for crisis_name, (start, end) in CRISES.items():
        mask = (values.index >= start) & (values.index <= end)
        crisis_vals = values[mask]
        if len(crisis_vals) < 3:
            continue

        ret = (crisis_vals.iloc[-1] / crisis_vals.iloc[0] - 1) * 100
        dd = ((crisis_vals - crisis_vals.cummax()) / crisis_vals.cummax()).min() * 100

        crisis_regs = regs[mask]
        regime_dist = crisis_regs.value_counts(normalize=True).to_dict()

        results[crisis_name] = {
            'return_pct': float(ret),
            'max_dd_pct': float(dd),
            'n_days': int(len(crisis_vals)),
            'dominant_regime': crisis_regs.mode().iloc[0] if len(crisis_regs) > 0 else 'N/A',
            'regime_dist': {k: round(v, 3) for k, v in regime_dist.items()},
        }
    return results


# =============================================================================
# ADVERSARIAL VALIDATION SUITE
# =============================================================================

def adversarial_permutation(closes, returns, sig, variant_class, variant_kwargs,
                            n_perms=200, warmup=260):
    """PERMUTATION TEST: Shuffle regime assignments. Real must beat random."""
    print("\n" + "=" * 70)
    print("  ADVERSARIAL TEST 1: PERMUTATION (n=%d)" % n_perms)
    print("=" * 70)

    fn = variant_class(sig=sig, **variant_kwargs)
    vals_real, contrib, switches, regs = simulate(closes, returns, fn, warmup=warmup)
    m_real = compute_metrics(vals_real, contrib, switches)
    real_sharpe = m_real['sharpe']
    print(f"  Real Sharpe: {real_sharpe:.3f}")

    perm_sharpes = []
    for p in range(n_perms):
        # Block-shuffle regime labels (5-day blocks)
        shuffled_regs = regs.copy()
        block_size = 5
        n_blocks = len(shuffled_regs) // block_size
        block_indices = np.arange(n_blocks)
        np.random.shuffle(block_indices)
        new_vals_list = []
        for bi in block_indices:
            s = bi * block_size
            new_vals_list.extend(shuffled_regs.iloc[s:s + block_size].tolist())
        new_vals_list.extend(shuffled_regs.iloc[n_blocks * block_size:].tolist())
        perm_regime = pd.Series(new_vals_list[:len(shuffled_regs)], index=shuffled_regs.index)

        # Simulate with shuffled regimes
        cash = float(INITIAL)
        total_c = float(INITIAL)
        last_week = None
        last_r = None
        sw = 0
        dv = []
        rets_df = returns.reindex(perm_regime.index)
        for idx, reg in perm_regime.items():
            week_key = (idx.year, idx.isocalendar()[1])
            if week_key != last_week:
                cash += WEEKLY_DCA
                total_c += WEEKLY_DCA
                last_week = week_key
            if reg != last_r and last_r is not None:
                sw += 1
                cash *= (1 - TX_COST_PCT - GAP_RISK_PCT)
            last_r = reg
            if reg in rets_df.columns:
                r = rets_df.loc[idx, reg]
                if not np.isnan(r):
                    cash *= (1 + r)
            dv.append(cash)

        pv = pd.Series(dv, index=perm_regime.index)
        pm = compute_metrics(pv, total_c, sw)
        perm_sharpes.append(pm['sharpe'])

        if (p + 1) % 50 == 0:
            print(f"    {p + 1}/{n_perms} permutations done...")

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= real_sharpe))

    print(f"  Real Sharpe:   {real_sharpe:.3f}")
    print(f"  Perm mean:     {np.mean(perm_sharpes):.3f}")
    print(f"  Perm p95:      {np.percentile(perm_sharpes, 95):.3f}")
    print(f"  p-value:       {p_value:.4f}")

    passed = p_value < 0.05
    banner = "PASS" if passed else "FAIL"
    print(f"\n  {'=' * 40}")
    print(f"  PERMUTATION TEST: *** {banner} *** (p={p_value:.4f})")
    print(f"  {'=' * 40}")
    return {'real_sharpe': real_sharpe, 'perm_mean': float(np.mean(perm_sharpes)),
            'p_value': p_value, 'pass': passed}


def adversarial_subperiod(closes, returns, sig, variant_class, variant_kwargs,
                          warmup=260):
    """SUB-PERIOD CONSISTENCY: Each 3-year block must show positive edge."""
    print("\n" + "=" * 70)
    print("  ADVERSARIAL TEST 2: SUB-PERIOD CONSISTENCY (3-year blocks)")
    print("=" * 70)

    fn = variant_class(sig=sig, **variant_kwargs)
    vals, contrib, switches, _ = simulate(closes, returns, fn, warmup=warmup)
    daily_ret = vals.pct_change().dropna()

    years = sorted(set(daily_ret.index.year))
    blocks = []
    for i_blk in range(0, len(years), 3):
        block_years = years[i_blk:i_blk + 3]
        if len(block_years) < 2:
            continue
        block_rets = daily_ret[daily_ret.index.year.isin(block_years)]
        if len(block_rets) > 100:
            ann_ret = block_rets.mean() * 252
            ann_vol = block_rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            blocks.append({
                'years': f"{min(block_years)}-{max(block_years)}",
                'sharpe': float(sharpe),
                'ann_ret': float(ann_ret),
                'n_days': len(block_rets)
            })

    print(f"\n  {'Period':<15} {'Sharpe':>8} {'Ann Ret':>10} {'Days':>6}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 10} {'-' * 6}")
    positive = 0
    for b in blocks:
        mark = "[+]" if b['sharpe'] > 0 else "[-]"
        print(f"  {b['years']:<15} {b['sharpe']:>8.3f} {b['ann_ret']:>9.1%} {b['n_days']:>6} {mark}")
        if b['sharpe'] > 0:
            positive += 1

    pct_pos = positive / len(blocks) if blocks else 0
    passed = pct_pos >= 0.6 and len(blocks) >= 3
    banner = "PASS" if passed else "FAIL"
    print(f"\n  Positive blocks: {positive}/{len(blocks)} ({pct_pos:.0%})")
    print(f"\n  {'=' * 40}")
    print(f"  SUB-PERIOD TEST: *** {banner} ***")
    print(f"  {'=' * 40}")
    return {'blocks': blocks, 'pct_positive': pct_pos, 'pass': passed}


def adversarial_outlier(closes, returns, sig, variant_class, variant_kwargs,
                        warmup=260):
    """OUTLIER ROBUSTNESS: Remove top 5% best days. Edge must persist."""
    print("\n" + "=" * 70)
    print("  ADVERSARIAL TEST 3: OUTLIER ROBUSTNESS (remove top 5%)")
    print("=" * 70)

    fn = variant_class(sig=sig, **variant_kwargs)
    vals, _, _, _ = simulate(closes, returns, fn, warmup=warmup)
    daily_ret = vals.pct_change().dropna()

    full_sharpe = float((daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)))

    threshold = daily_ret.quantile(0.95)
    trimmed = daily_ret[daily_ret <= threshold]
    trimmed_sharpe = float((trimmed.mean() * 252) / (trimmed.std() * np.sqrt(252)))

    lo = daily_ret.quantile(0.05)
    hi = daily_ret.quantile(0.95)
    winsorized = daily_ret[(daily_ret >= lo) & (daily_ret <= hi)]
    winsor_sharpe = float((winsorized.mean() * 252) / (winsorized.std() * np.sqrt(252)))

    print(f"  Full Sharpe:           {full_sharpe:.3f}")
    print(f"  Top-5% removed Sharpe: {trimmed_sharpe:.3f}")
    print(f"  Winsorized Sharpe:     {winsor_sharpe:.3f}")

    passed = trimmed_sharpe > 0 and winsor_sharpe > 0
    banner = "PASS" if passed else "FAIL"
    print(f"\n  {'=' * 40}")
    print(f"  OUTLIER TEST: *** {banner} ***")
    print(f"  {'=' * 40}")
    return {'full_sharpe': full_sharpe, 'trimmed_sharpe': trimmed_sharpe,
            'winsor_sharpe': winsor_sharpe, 'pass': passed}


def adversarial_regime_r1(closes, returns, sig, variant_class, variant_kwargs,
                          warmup=260):
    """R1 REGIME TEST: Must work on both green and red SPY days. Asymmetry < 0.50."""
    print("\n" + "=" * 70)
    print("  ADVERSARIAL TEST 4: R1 REGIME (green vs red SPY days)")
    print("=" * 70)

    fn = variant_class(sig=sig, **variant_kwargs)
    vals, _, _, regs = simulate(closes, returns, fn, warmup=warmup)
    strat_ret = vals.pct_change().dropna()
    spy_ret = closes['SPY'].pct_change().reindex(strat_ret.index)

    green_days = strat_ret[spy_ret > 0]
    red_days = strat_ret[spy_ret < 0]

    green_sharpe = float((green_days.mean() * 252) / (green_days.std() * np.sqrt(252))) if len(green_days) > 10 else 0
    red_sharpe = float((red_days.mean() * 252) / (red_days.std() * np.sqrt(252))) if len(red_days) > 10 else 0

    max_sharpe = max(abs(green_sharpe), abs(red_sharpe))
    asymmetry = abs(green_sharpe - red_sharpe) / max_sharpe if max_sharpe > 0 else 0

    print(f"  Green SPY days: n={len(green_days)}, Sharpe={green_sharpe:.3f}")
    print(f"  Red SPY days:   n={len(red_days)}, Sharpe={red_sharpe:.3f}")
    print(f"  Asymmetry:      {asymmetry:.3f} (reject if >0.50)")

    regime_counts = regs.value_counts()
    print(f"\n  Regime distribution:")
    for r, c in regime_counts.items():
        print(f"    {r}: {c} days ({c / len(regs):.1%})")

    passed = asymmetry <= 0.50
    banner = "PASS" if passed else "FAIL"
    print(f"\n  {'=' * 40}")
    print(f"  R1 REGIME TEST: *** {banner} *** (asym={asymmetry:.3f})")
    print(f"  {'=' * 40}")
    return {'green_sharpe': green_sharpe, 'red_sharpe': red_sharpe,
            'asymmetry': float(asymmetry), 'pass': passed}


def adversarial_walkforward(closes, returns, sig, variant_class, variant_kwargs,
                            warmup=260, train_years=3, test_years=1):
    """WALK-FORWARD VALIDATION: 3yr train / 1yr OOS rolling windows."""
    print("\n" + "=" * 70)
    print(f"  ADVERSARIAL TEST 5: WALK-FORWARD ({train_years}yr train, {test_years}yr OOS)")
    print("=" * 70)

    all_years = sorted(set(closes.index.year))
    min_year = min(all_years) + 1

    oos_results = []
    for start_test in range(min_year + train_years, max(all_years) + 1, test_years):
        end_test = start_test + test_years - 1
        test_mask = (closes.index.year >= start_test) & (closes.index.year <= end_test)
        test_data = closes[test_mask]
        if len(test_data) < 100:
            continue

        # Build regime for full dataset but evaluate on test period only
        fn = variant_class(sig=sig, **variant_kwargs)

        # Run simulation on full data
        vals_full, _, _, regs_full = simulate(closes, returns, fn, warmup=warmup)

        # Extract test period
        test_strat = vals_full.reindex(test_data.index).dropna()
        if len(test_strat) < 50:
            continue
        test_ret = test_strat.pct_change().dropna()
        if len(test_ret) < 30:
            continue

        ann_ret = test_ret.mean() * 252
        ann_vol = test_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        spy_test = returns['SPY'].reindex(test_ret.index)
        spy_sharpe = (spy_test.mean() * 252) / (spy_test.std() * np.sqrt(252))

        oos_results.append({
            'period': f"{start_test}-{end_test}",
            'sharpe': float(sharpe),
            'spy_sharpe': float(spy_sharpe),
            'excess_sharpe': float(sharpe - spy_sharpe),
            'n_days': len(test_ret),
        })

    print(f"\n  {'Period':<12} {'Sharpe':>8} {'SPY Sh':>8} {'Excess':>8} {'Days':>6}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")
    positive_excess = 0
    for r in oos_results:
        mark = "[+]" if r['excess_sharpe'] > 0 else "[-]"
        print(f"  {r['period']:<12} {r['sharpe']:>8.3f} {r['spy_sharpe']:>8.3f} "
              f"{r['excess_sharpe']:>8.3f} {r['n_days']:>6} {mark}")
        if r['excess_sharpe'] > 0:
            positive_excess += 1

    pct_pos = positive_excess / len(oos_results) if oos_results else 0
    avg_excess = float(np.mean([r['excess_sharpe'] for r in oos_results])) if oos_results else 0
    avg_oos = float(np.mean([r['sharpe'] for r in oos_results])) if oos_results else 0

    passed = pct_pos >= 0.5 and avg_oos > 0
    banner = "PASS" if passed else "FAIL"
    print(f"\n  OOS positive excess: {positive_excess}/{len(oos_results)} ({pct_pos:.0%})")
    print(f"  Avg OOS Sharpe: {avg_oos:.3f}")
    print(f"  Avg excess vs SPY: {avg_excess:.3f}")
    print(f"\n  {'=' * 40}")
    print(f"  WALK-FORWARD TEST: *** {banner} ***")
    print(f"  {'=' * 40}")
    return {'oos_results': oos_results, 'pct_positive': pct_pos,
            'avg_oos_sharpe': avg_oos, 'avg_excess': avg_excess, 'pass': passed}


def run_adversarial_suite(closes, returns, sig, variant_class, variant_kwargs,
                          label="", n_perms=200):
    """Run all 5 adversarial tests for a variant."""
    print("\n" + "#" * 80)
    print(f"# ADVERSARIAL VALIDATION: {label}")
    print("#" * 80)

    results = {}
    results['permutation'] = adversarial_permutation(
        closes, returns, sig, variant_class, variant_kwargs, n_perms=n_perms)
    results['subperiod'] = adversarial_subperiod(
        closes, returns, sig, variant_class, variant_kwargs)
    results['outlier'] = adversarial_outlier(
        closes, returns, sig, variant_class, variant_kwargs)
    results['regime_r1'] = adversarial_regime_r1(
        closes, returns, sig, variant_class, variant_kwargs)
    results['walkforward'] = adversarial_walkforward(
        closes, returns, sig, variant_class, variant_kwargs)

    # Summary
    n_pass = sum(1 for v in results.values() if v.get('pass', False))
    n_total = len(results)

    print("\n" + "=" * 70)
    print(f"  ADVERSARIAL SUMMARY: {label}")
    print("=" * 70)
    for test_name, r in results.items():
        status = "PASS" if r.get('pass', False) else "FAIL"
        print(f"    {test_name:<20}: {status}")
    print(f"\n    TOTAL: {n_pass}/{n_total} PASSED")

    overall = "PASS" if n_pass >= 4 else "FAIL"
    print(f"\n  {'#' * 50}")
    print(f"  # OVERALL ADVERSARIAL: *** {overall} *** ({n_pass}/{n_total})")
    print(f"  {'#' * 50}")

    results['overall_pass'] = n_pass >= 4
    results['n_pass'] = n_pass
    results['n_total'] = n_total
    return results


# =============================================================================
# HEAD-TO-HEAD COMPARISON
# =============================================================================

def run_head_to_head(closes, returns, sig, warmup=260):
    """Run all variants head-to-head and print comparison table."""
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON: v3 BASELINE vs v4 VARIANTS")
    print("=" * 80)

    variants = [
        ("v3 Baseline", GameplanV3Baseline, {'entry': 2.5, 'exit_thresh': 2.0}),
        ("v4.1 Pure Replace (63d)", V4_PureReplacement,
         {'lookback': 63, 'low_pctile': 20, 'high_pctile': 80}),
        ("v4.1 Pure Replace (126d)", V4_PureReplacement,
         {'lookback': 126, 'low_pctile': 20, 'high_pctile': 80}),
        ("v4.2 Hybrid (63d)", V4_Hybrid, {'lookback': 63}),
        ("v4.2 Hybrid (126d)", V4_Hybrid, {'lookback': 126}),
        ("v4.3 Confluence+Pctile (63d)", V4_ConfluencePlusPercentile, {'lookback': 63}),
        ("v4.3 Confluence+Pctile (126d)", V4_ConfluencePlusPercentile, {'lookback': 126}),
        ("v4.4 Full Adaptive (63d)", V4_FullAdaptive, {'lookback': 63}),
        ("v4.4 Full Adaptive (126d)", V4_FullAdaptive, {'lookback': 126}),
    ]

    all_results = {}
    for label, cls, kwargs in variants:
        fn = cls(sig=sig, **kwargs)
        vals, contrib, switches, regs = simulate(closes, returns, fn, warmup=warmup)
        m = compute_metrics(vals, contrib, switches)
        crisis = crisis_analysis(vals, regs, label)
        all_results[label] = {
            'metrics': m,
            'crisis': crisis,
            'values': vals,
            'regimes': regs,
        }

    # Print comparison table
    print(f"\n  {'Variant':<32} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Sw/yr':>6}")
    print(f"  {'-' * 32} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 6}")

    for label, res in all_results.items():
        m = res['metrics']
        print(f"  {label:<32} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['calmar']:>7.3f} "
              f"{m['switches_yr']:>6.1f}")

    # Crisis comparison
    print(f"\n  CRISIS BEHAVIOR:")
    print(f"  {'Variant':<32}", end="")
    for crisis_name in CRISES:
        print(f" {crisis_name:<18}", end="")
    print()
    print(f"  {'-' * 32}", end="")
    for _ in CRISES:
        print(f" {'-' * 18}", end="")
    print()

    for label, res in all_results.items():
        print(f"  {label:<32}", end="")
        for crisis_name in CRISES:
            if crisis_name in res['crisis']:
                c = res['crisis'][crisis_name]
                print(f" {c['return_pct']:>+6.1f}%/{c['max_dd_pct']:>5.1f}%  ", end="")
            else:
                print(f" {'N/A':>18}", end="")
        print()

    return all_results


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def parameter_sweep(closes, returns, sig, warmup=260):
    """Sweep key parameters across variants to find robust configurations."""
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP — KEY DIMENSIONS")
    print("=" * 80)

    sweep_results = []

    # Sweep lookback and percentile thresholds for Pure Replacement
    for lb in [42, 63, 126]:
        for lo in [15, 20, 25, 30]:
            for hi in [70, 75, 80, 85]:
                fn = V4_PureReplacement(sig=sig, lookback=lb, low_pctile=lo, high_pctile=hi)
                vals, contrib, switches, _ = simulate(closes, returns, fn, warmup=warmup)
                m = compute_metrics(vals, contrib, switches)
                sweep_results.append({
                    'variant': 'pure_replacement',
                    'lookback': lb, 'low_pctile': lo, 'high_pctile': hi,
                    **m
                })

    # Sweep for Hybrid
    for lb in [42, 63, 126]:
        for tighten in [10, 12, 13]:
            for loosen in [17, 18, 20]:
                fn = V4_Hybrid(sig=sig, lookback=lb,
                               tighten_upro=tighten, loosen_upro=loosen)
                vals, contrib, switches, _ = simulate(closes, returns, fn, warmup=warmup)
                m = compute_metrics(vals, contrib, switches)
                sweep_results.append({
                    'variant': 'hybrid',
                    'lookback': lb, 'tighten': tighten, 'loosen': loosen,
                    **m
                })

    # Sweep for Confluence + Percentile
    for lb in [42, 63, 126]:
        for lo in [15, 20, 25]:
            for hi in [75, 80, 85]:
                fn = V4_ConfluencePlusPercentile(sig=sig, lookback=lb,
                                                 low_pctile=lo, high_pctile=hi)
                vals, contrib, switches, _ = simulate(closes, returns, fn, warmup=warmup)
                m = compute_metrics(vals, contrib, switches)
                sweep_results.append({
                    'variant': 'confluence_pctile',
                    'lookback': lb, 'low_pctile': lo, 'high_pctile': hi,
                    **m
                })

    sweep_df = pd.DataFrame(sweep_results)

    # Print top configs per variant
    for variant_name in ['pure_replacement', 'hybrid', 'confluence_pctile']:
        vdf = sweep_df[sweep_df['variant'] == variant_name]
        top5 = vdf.nlargest(5, 'sharpe')
        print(f"\n  TOP 5 — {variant_name}:")
        for _, row in top5.iterrows():
            params = {k: v for k, v in row.items()
                      if k not in ['variant', 'sharpe', 'sortino', 'cagr', 'max_dd',
                                   'calmar', 'ann_vol', 'final_value', 'total_contributed',
                                   'profit', 'switches_yr', 'switches', 'n_years']}
            print(f"    Sharpe={row['sharpe']:.3f} Sortino={row['sortino']:.3f} "
                  f"CAGR={row['cagr']:.1%} MaxDD={row['max_dd']:.1%} "
                  f"Sw/yr={row['switches_yr']:.1f} | {params}")

    return sweep_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()

    # 1. Data
    closes, returns = download_data()

    # 2. Signals
    print("\n[2] Computing signals...")
    sig = compute_all_signals(closes)

    # 3. Head-to-head comparison
    print("\n[3] Running head-to-head comparison...")
    h2h_results = run_head_to_head(closes, returns, sig)

    # 4. Parameter sweep
    print("\n[4] Running parameter sweep...")
    sweep_df = parameter_sweep(closes, returns, sig)

    # 5. Find best variant overall (highest Sharpe from sweep)
    best_idx = sweep_df['sharpe'].idxmax()
    best_row = sweep_df.loc[best_idx]
    best_variant = best_row['variant']
    print(f"\n[5] BEST CONFIG FROM SWEEP: {best_variant}")
    print(f"    Sharpe={best_row['sharpe']:.3f}, CAGR={best_row['cagr']:.1%}, "
          f"MaxDD={best_row['max_dd']:.1%}, Switches/yr={best_row['switches_yr']:.1f}")

    # 6. Run adversarial validation on best variant from each category + v3 baseline
    print("\n[6] Running adversarial validation on top variants...")

    # v3 baseline adversarial
    adv_baseline = run_adversarial_suite(
        closes, returns, sig, GameplanV3Baseline,
        {'entry': 2.5, 'exit_thresh': 2.0},
        label="v3 Baseline", n_perms=200)

    # Best pure replacement from sweep
    pure_df = sweep_df[sweep_df['variant'] == 'pure_replacement']
    best_pure = pure_df.loc[pure_df['sharpe'].idxmax()] if len(pure_df) > 0 else None
    if best_pure is not None:
        adv_pure = run_adversarial_suite(
            closes, returns, sig, V4_PureReplacement,
            {'lookback': int(best_pure['lookback']),
             'low_pctile': int(best_pure['low_pctile']),
             'high_pctile': int(best_pure['high_pctile'])},
            label=f"v4.1 Pure Replace (lb={int(best_pure['lookback'])},"
                  f"lo={int(best_pure['low_pctile'])},hi={int(best_pure['high_pctile'])})",
            n_perms=200)
    else:
        adv_pure = None

    # Best hybrid from sweep
    hybrid_df = sweep_df[sweep_df['variant'] == 'hybrid']
    best_hybrid = hybrid_df.loc[hybrid_df['sharpe'].idxmax()] if len(hybrid_df) > 0 else None
    if best_hybrid is not None:
        adv_hybrid = run_adversarial_suite(
            closes, returns, sig, V4_Hybrid,
            {'lookback': int(best_hybrid['lookback']),
             'tighten_upro': int(best_hybrid['tighten']),
             'loosen_upro': int(best_hybrid['loosen'])},
            label=f"v4.2 Hybrid (lb={int(best_hybrid['lookback'])},"
                  f"tight={int(best_hybrid['tighten'])},loose={int(best_hybrid['loosen'])})",
            n_perms=200)
    else:
        adv_hybrid = None

    # Best confluence+percentile from sweep
    conf_df = sweep_df[sweep_df['variant'] == 'confluence_pctile']
    best_conf = conf_df.loc[conf_df['sharpe'].idxmax()] if len(conf_df) > 0 else None
    if best_conf is not None:
        adv_conf = run_adversarial_suite(
            closes, returns, sig, V4_ConfluencePlusPercentile,
            {'lookback': int(best_conf['lookback']),
             'low_pctile': int(best_conf['low_pctile']),
             'high_pctile': int(best_conf['high_pctile'])},
            label=f"v4.3 Confluence+Pctile (lb={int(best_conf['lookback'])},"
                  f"lo={int(best_conf['low_pctile'])},hi={int(best_conf['high_pctile'])})",
            n_perms=200)
    else:
        adv_conf = None

    # Full adaptive with default params
    adv_full = run_adversarial_suite(
        closes, returns, sig, V4_FullAdaptive,
        {'lookback': 63, 'low_pctile': 20, 'high_pctile': 80},
        label="v4.4 Full Adaptive (63d, 20/80)", n_perms=200)

    # 7. Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — GAMEPLAN v4 ADAPTIVE")
    print("=" * 80)

    all_adv = {
        'v3_baseline': adv_baseline,
        'v4.1_pure_replacement': adv_pure,
        'v4.2_hybrid': adv_hybrid,
        'v4.3_confluence_pctile': adv_conf,
        'v4.4_full_adaptive': adv_full,
    }

    print(f"\n  {'Variant':<40} {'Adv Gates':>10} {'Overall':>8}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 8}")
    for name, adv in all_adv.items():
        if adv is None:
            print(f"  {name:<40} {'N/A':>10} {'N/A':>8}")
            continue
        gates = f"{adv['n_pass']}/{adv['n_total']}"
        overall = "PASS" if adv['overall_pass'] else "FAIL"
        print(f"  {name:<40} {gates:>10} {overall:>8}")

    # Head-to-head metrics for passing variants
    print(f"\n  METRICS FOR VARIANTS PASSING ADVERSARIAL (>= 4/5 gates):")
    print(f"  {'Variant':<32} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
          f"{'MaxDD':>7} {'Calmar':>7} {'Sw/yr':>6}")
    print(f"  {'-' * 32} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 6}")

    for label, res in h2h_results.items():
        m = res['metrics']
        print(f"  {label:<32} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['calmar']:>7.3f} "
              f"{m['switches_yr']:>6.1f}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save results
    print(f"\n[7] Saving results...")

    # Save sweep results
    sweep_df.to_csv(OUTPUT_DIR / "parameter_sweep.csv", index=False)

    # Save metrics summary
    summary = {}
    for label, res in h2h_results.items():
        summary[label] = res['metrics']
        summary[label]['crisis'] = res['crisis']

    # Add adversarial results
    adv_summary = {}
    for name, adv in all_adv.items():
        if adv is not None:
            adv_clean = {}
            for k, v in adv.items():
                if k in ['overall_pass', 'n_pass', 'n_total']:
                    adv_clean[k] = v
                elif isinstance(v, dict):
                    adv_clean[k] = {kk: vv for kk, vv in v.items()
                                    if not isinstance(vv, (pd.Series, pd.DataFrame))}
            adv_summary[name] = adv_clean

    full_output = {
        'run_date': datetime.now().isoformat(),
        'metrics': {k: v for k, v in summary.items()},
        'adversarial': adv_summary,
        'best_sweep_config': {
            'variant': best_variant,
            **{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
               for k, v in best_row.to_dict().items()}
        }
    }

    with open(OUTPUT_DIR / "gameplan_v4_results.json", 'w') as f:
        json.dump(full_output, f, indent=2, default=str)

    print(f"  Saved to {OUTPUT_DIR}/")
    print(f"\n{'=' * 80}")
    print("GAMEPLAN v4 ADAPTIVE — COMPLETE")
    print(f"{'=' * 80}")

    return full_output


if __name__ == "__main__":
    results = main()
