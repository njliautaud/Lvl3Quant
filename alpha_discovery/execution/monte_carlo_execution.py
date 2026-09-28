"""
Monte Carlo Simulation of Execution Research Results
======================================================
Runs 10,000 simulations of daily P&L paths using empirical parameters extracted from:

Primary data sources:
1. /output/backtest_continuous_reeval/backtest_results.json
   - CNN-Mamba v2 OOT predictions through FIFO fill simulator (folds 5-15, March 2026)
   - Passive cost model: 0.376 ticks = $4.70 RT commission
   - Metrics: n_trades, win_rate, avg_winner_ticks, avg_loser_ticks, profit_factor, sortino

2. /output/blended_cost_execution_analysis.json
   - 10 OOT folds, CNN-Mamba v2 with blended exit costs
   - trades_per_day, daily_pnl by confidence tier

3. /output/fifo_regime_sweep/ (9 actual trading days, per-day P&L)
   - Real observed daily std: ~$2,500 (from 9 days of actual simulation data)
   - This provides ground truth for regime-level daily variance

Key modeling decision:
   CLT averaging of 250+ trades/day makes daily P&L nearly Gaussian with very low std.
   This is WRONG because:
   - Signal strength varies by regime/date (some days IC=0.20, others IC=0.05)
   - We only have 10 OOT days => t-distribution uncertainty is large
   - Adverse selection and fill quality degrade during volatile regimes

   Solution: Model daily P&L as t-distributed with:
   - Location = empirical daily_mean from blended_cost analysis
   - Scale = observed daily_std from fifo_regime_sweep ($2,500)
   - df = 10 (our sample size, gives heavy tails reflecting true uncertainty)

Cost constants (canonical from CLAUDE.md, HC #231(A)):
  ES_TICK_VALUE   = $12.50
  ES_RT_COMMISSION = $4.70  (round-trip)
  All order types: 0.376 ticks (commission only — no spread crossing cost)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
ES_TICK_VALUE        = 12.50       # USD per tick
ES_RT_COMMISSION     = 4.70        # USD round-trip
ES_RT_COMMISSION_TICKS = 0.376     # ticks equivalent

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/monte_carlo_execution')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIM = 10_000
RUIN_DRAWDOWN_PCT = 0.20           # 20% peak-to-trough = ruin threshold
TRADING_DAYS_PER_YEAR = 252
N_DAYS_QUARTER = 63               # 1 trading quarter
INITIAL_CAPITAL = 50_000          # USD, for drawdown % calculation

# Ground truth daily std from 9 observed simulation days (fifo_regime_sweep)
# This reflects REAL regime-level variance, not CLT trade-level smoothing
OBSERVED_DAILY_STD_USD = 2_529.0
OBSERVED_DAILY_STD_DF  = 9         # degrees of freedom for t-distribution (9 observed days)


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def load_backtest_configs() -> dict:
    """Load per-config aggregate stats from continuous re-eval backtest."""
    p = Path('/home/jupiter/Lvl3Quant/output/backtest_continuous_reeval/backtest_results.json')
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def load_blended_cost_configs() -> dict:
    """Load profitable per-config stats from blended cost analysis."""
    p = Path('/home/jupiter/Lvl3Quant/output/blended_cost_execution_analysis.json')
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())['configs']


def load_observed_daily_pnl() -> dict:
    """
    Load per-day P&L from fifo_regime_sweep to get empirical daily variance.

    Returns a dict mapping regime -> list of daily P&L values (USD).
    This is the ONLY source of real daily-level variance in our data.
    """
    import glob
    import os
    results = {}
    for regime in ['wide_conviction', 'medium_balanced']:
        pattern = f'/home/jupiter/Lvl3Quant/output/fifo_regime_sweep/{regime}_*.json'
        daily_pnl = []
        for fpath in sorted(glob.glob(pattern)):
            d = json.loads(Path(fpath).read_text())
            daily_pnl.append(d.get('total_pnl_dollars', 0.0))
        if daily_pnl:
            results[regime] = daily_pnl
    return results


# ─── Configuration Definition ─────────────────────────────────────────────────

class ExecConfig:
    """
    Execution strategy configuration for Monte Carlo simulation.

    Combines:
    - Empirical daily mean from blended_cost analysis (10 OOT folds)
    - Realistic daily std from t-distribution centered on observed fillsim variance
    - Parameter uncertainty via t-distribution (df = N_BACKTEST_DAYS)

    The t-distribution with low df gives fat tails, reflecting:
    1. Only 10 days of OOT data (small sample)
    2. Regime uncertainty (signal IC varies 0.05-0.21 day-to-day)
    3. Unknown fill quality in live trading
    """

    def __init__(self, name: str, daily_mean_usd: float, daily_std_usd: float,
                 trades_per_day: float, win_rate: float,
                 avg_win_ticks: float, avg_loss_ticks: float,
                 profit_factor: float, sortino_trade_level: float,
                 n_observed_days: int = 10, description: str = ''):
        self.name = name
        self.daily_mean = daily_mean_usd
        self.daily_std = daily_std_usd
        self.n_day = trades_per_day
        self.wr = win_rate
        self.avg_win = avg_win_ticks * ES_TICK_VALUE
        self.avg_loss = avg_loss_ticks * ES_TICK_VALUE    # should be negative
        self.pf = profit_factor
        self.sortino_trade = sortino_trade_level
        self.df = n_observed_days - 1   # t-distribution degrees of freedom
        self.description = description

        # Daily Sharpe (annualized) from these parameters
        self.daily_sharpe = (daily_mean_usd / daily_std_usd) * np.sqrt(TRADING_DAYS_PER_YEAR)

    def sample_daily_pnl(self, n_sim: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample daily P&L using t-distribution.

        The t-distribution with df = n_observed_days - 1 captures:
        - Parameter uncertainty from small sample (only 10 OOT days)
        - Regime risk (some days signal doesn't work at all)
        - Fat tails that Gaussian would miss

        We also add a regime component: with probability p_bad_regime,
        the daily P&L is drawn from a worse distribution (simulating days
        where signal decays or market regime is hostile).
        """
        # Base daily P&L from t-distribution
        # scipy.stats.t: rvs gives t-distributed samples, then scale
        t_samples = rng.standard_t(df=self.df, size=n_sim)
        daily = self.daily_mean + self.daily_std * t_samples

        # Regime shocks: ~15% of days, signal underperforms by 2-3 std
        p_bad_regime = 0.15
        bad_days = rng.random(n_sim) < p_bad_regime
        # On bad days: scale down mean to 30% and increase noise
        regime_shock = rng.standard_t(df=3, size=n_sim)  # heavy tails for shocks
        bad_day_pnl = self.daily_mean * 0.30 + self.daily_std * 1.5 * regime_shock
        daily = np.where(bad_days, bad_day_pnl, daily)

        return daily

    def sample_cumulative_paths(self, n_sim: int, n_days: int,
                                 rng: np.random.Generator):
        """
        Simulate n_sim equity curves of n_days each.

        Returns:
          paths: shape (n_sim, n_days+1) — cumulative P&L from 0
          daily: shape (n_sim, n_days)  — day-by-day P&L
        """
        daily = np.zeros((n_sim, n_days))
        for d in range(n_days):
            daily[:, d] = self.sample_daily_pnl(n_sim, rng)

        paths = np.zeros((n_sim, n_days + 1))
        paths[:, 1:] = np.cumsum(daily, axis=1)
        return paths, daily


# ─── Config Builder ────────────────────────────────────────────────────────────

def build_configs_from_data() -> list:
    """
    Build ExecConfig objects from actual backtest data.

    Key modeling:
    - Daily mean: from blended_cost analysis (most realistic cost model)
    - Daily std: from observed fillsim daily variance ($2,529 from 9 days)
      scaled by signal quality (better signal = slightly less variance)
    - n_observed_days = 10 (from continuous re-eval backtest)
    """
    bc = load_backtest_configs()
    blended = load_blended_cost_configs()
    N_BACKTEST_DAYS = 10.0

    configs = []

    # ── Config 1: Instant flip, Top 1% signal confidence ─────────────────────
    # Source: backtest_continuous_reeval passive cost
    # Source: blended_cost 1s_top_1pct for daily_pnl
    cfg = bc['instant_flip_Top1%']
    bl1 = blended.get('1s_top_1pct', {})
    # Use blended daily P&L (more realistic) if available
    daily_mean = bl1.get('daily_pnl_dollars', cfg['net_pnl_usd'] / N_BACKTEST_DAYS)
    configs.append(ExecConfig(
        name='Top1%_Passive',
        daily_mean_usd=daily_mean,
        daily_std_usd=OBSERVED_DAILY_STD_USD,  # grounded in observed data
        trades_per_day=cfg['n_trades'] / N_BACKTEST_DAYS,
        win_rate=cfg['win_rate'],
        avg_win_ticks=cfg['avg_winner_ticks'],
        avg_loss_ticks=cfg['avg_loser_ticks'],
        profit_factor=cfg['profit_factor'],
        sortino_trade_level=cfg['sortino'],
        n_observed_days=int(N_BACKTEST_DAYS),
        description='Top 1% confidence signals, instant flip exit, passive entry',
    ))

    # ── Config 2: Instant flip, Top 5% signal confidence ─────────────────────
    cfg5 = bc['instant_flip_Top5%']
    bl5 = blended.get('1s_top_5pct', {})
    daily_mean5 = bl5.get('daily_pnl_dollars', cfg5['net_pnl_usd'] / N_BACKTEST_DAYS)
    configs.append(ExecConfig(
        name='Top5%_Passive',
        daily_mean_usd=daily_mean5,
        daily_std_usd=OBSERVED_DAILY_STD_USD * 1.2,  # more trades = more correlated risk
        trades_per_day=cfg5['n_trades'] / N_BACKTEST_DAYS,
        win_rate=cfg5['win_rate'],
        avg_win_ticks=cfg5['avg_winner_ticks'],
        avg_loss_ticks=cfg5['avg_loser_ticks'],
        profit_factor=cfg5['profit_factor'],
        sortino_trade_level=cfg5['sortino'],
        n_observed_days=int(N_BACKTEST_DAYS),
        description='Top 5% confidence signals, instant flip exit, passive entry',
    ))

    # ── Config 3: Instant flip, Top 0.1% (very high selectivity) ─────────────
    cfg01 = bc['instant_flip_Top0.1%']
    daily_mean01 = cfg01['net_pnl_usd'] / N_BACKTEST_DAYS
    configs.append(ExecConfig(
        name='Top0.1%_HighSelect',
        daily_mean_usd=daily_mean01,
        daily_std_usd=OBSERVED_DAILY_STD_USD * 0.6,  # fewer trades, lower daily variance
        trades_per_day=cfg01['n_trades'] / N_BACKTEST_DAYS,
        win_rate=cfg01['win_rate'],
        avg_win_ticks=cfg01['avg_winner_ticks'],
        avg_loss_ticks=cfg01['avg_loser_ticks'],
        profit_factor=cfg01['profit_factor'],
        sortino_trade_level=cfg01['sortino'],
        n_observed_days=int(N_BACKTEST_DAYS),
        description='Top 0.1% confidence signals only (very selective), ~33 trades/day',
    ))

    # ── Config 4: Razer-style reference (conservative, per task spec) ─────────
    # User said "Razer PPO produced $300-500/day with basic CNN-Mamba signal only"
    # Use this as a conservative reference scenario
    configs.append(ExecConfig(
        name='Razer_PPO_Ref',
        daily_mean_usd=400.0,    # midpoint of $300-500/day reference
        daily_std_usd=OBSERVED_DAILY_STD_USD * 0.8,
        trades_per_day=30.0,
        win_rate=0.62,
        avg_win_ticks=3.0,
        avg_loss_ticks=-2.0,
        profit_factor=1.8,
        sortino_trade_level=0.45,
        n_observed_days=int(N_BACKTEST_DAYS),
        description='Razer PPO reference: $300-500/day with CNN-Mamba signal only',
    ))

    return configs


# ─── Monte Carlo Engine ────────────────────────────────────────────────────────

def max_drawdown_usd(cum_pnl: np.ndarray) -> float:
    """Compute max peak-to-trough drawdown in USD from a cumulative P&L path."""
    equity = INITIAL_CAPITAL + cum_pnl
    running_max = np.maximum.accumulate(equity)
    return float(np.max(running_max - equity))


def run_monte_carlo(cfg: ExecConfig, n_sim: int = N_SIM, n_days: int = N_DAYS_QUARTER,
                    rng: np.random.Generator = None) -> dict:
    """
    Run full Monte Carlo simulation for one ExecConfig.

    Returns dict of all statistics plus the raw simulation arrays.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    print(f'  Simulating {cfg.name}: {n_sim:,} paths x {n_days} days ... ', end='', flush=True)
    paths, daily = cfg.sample_cumulative_paths(n_sim, n_days, rng)
    print('done')

    # ── Daily P&L distribution ──────────────────────────────────────────────
    daily_flat = daily.ravel()
    path_means = np.mean(daily, axis=1)

    # 95% CI on the expected daily mean (using path distribution)
    ci_lo = np.percentile(path_means, 2.5)
    ci_hi = np.percentile(path_means, 97.5)

    # ── Per-path Sharpe and Sortino (annualized) ────────────────────────────
    path_mean = np.mean(daily, axis=1)
    path_std  = np.std(daily, axis=1, ddof=1)

    # Sharpe
    sharpe = np.where(path_std > 0,
                      path_mean / path_std * np.sqrt(TRADING_DAYS_PER_YEAR),
                      np.nan)

    # Sortino (downside std = sqrt(mean of negative returns^2))
    neg_daily = np.where(daily < 0, daily, 0.0)
    downside_std = np.sqrt(np.mean(neg_daily ** 2, axis=1))
    sortino = np.where(downside_std > 0,
                       path_mean / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR),
                       np.nan)

    # ── Max drawdown per path ───────────────────────────────────────────────
    # Compute for subset to save time (3000 paths is sufficient for percentiles)
    n_dd_sample = min(3000, n_sim)
    mdd_usd = np.array([max_drawdown_usd(paths[i]) for i in range(n_dd_sample)])
    mdd_pct = mdd_usd / INITIAL_CAPITAL

    # ── Ruin probability ────────────────────────────────────────────────────
    # Ruin: drawdown hits 20% of initial capital at any point during quarter
    prob_ruin = np.mean(mdd_pct >= RUIN_DRAWDOWN_PCT)

    # Probability of profitable quarter
    prob_pos_quarter = np.mean(paths[:, -1] > 0)

    # Profit factor from simulated daily P&L
    pos_sum = daily_flat[daily_flat > 0].sum()
    neg_sum = abs(daily_flat[daily_flat < 0].sum())
    sim_pf  = pos_sum / neg_sum if neg_sum > 0 else np.inf

    return {
        'config_name': cfg.name,
        'description': cfg.description,
        'params': {
            'daily_mean_usd': cfg.daily_mean,
            'daily_std_usd': cfg.daily_std,
            'daily_sharpe_theoretical': cfg.daily_sharpe,
            'trades_per_day': cfg.n_day,
            'win_rate': cfg.wr,
            'avg_win_usd': cfg.avg_win,
            'avg_loss_usd': cfg.avg_loss,
            'profit_factor_empirical': cfg.pf,
            'sortino_trade_level': cfg.sortino_trade,
            'df_t_distribution': cfg.df,
        },
        'daily_pnl': {
            'mean': float(np.mean(daily_flat)),
            'std': float(np.std(daily_flat)),
            'p5': float(np.percentile(daily_flat, 5)),
            'p25': float(np.percentile(daily_flat, 25)),
            'p50': float(np.percentile(daily_flat, 50)),
            'p75': float(np.percentile(daily_flat, 75)),
            'p95': float(np.percentile(daily_flat, 95)),
            'ci_95_lo': float(ci_lo),
            'ci_95_hi': float(ci_hi),
            'pct_positive_days': float(np.mean(daily_flat > 0)),
        },
        'sharpe': {
            'mean': float(np.nanmean(sharpe)),
            'std': float(np.nanstd(sharpe)),
            'p10': float(np.nanpercentile(sharpe, 10)),
            'p25': float(np.nanpercentile(sharpe, 25)),
            'p50': float(np.nanpercentile(sharpe, 50)),
            'p75': float(np.nanpercentile(sharpe, 75)),
            'p90': float(np.nanpercentile(sharpe, 90)),
        },
        'sortino': {
            'mean': float(np.nanmean(sortino)),
            'std': float(np.nanstd(sortino)),
            'p10': float(np.nanpercentile(sortino, 10)),
            'p25': float(np.nanpercentile(sortino, 25)),
            'p50': float(np.nanpercentile(sortino, 50)),
            'p75': float(np.nanpercentile(sortino, 75)),
            'p90': float(np.nanpercentile(sortino, 90)),
        },
        'max_drawdown': {
            'mean_usd': float(np.mean(mdd_usd)),
            'std_usd': float(np.std(mdd_usd)),
            'p50_usd': float(np.percentile(mdd_usd, 50)),
            'p75_usd': float(np.percentile(mdd_usd, 75)),
            'p90_usd': float(np.percentile(mdd_usd, 90)),
            'p95_usd': float(np.percentile(mdd_usd, 95)),
            'p99_usd': float(np.percentile(mdd_usd, 99)),
            'mean_pct': float(np.mean(mdd_pct)),
            'p90_pct': float(np.percentile(mdd_pct, 90)),
            'p95_pct': float(np.percentile(mdd_pct, 95)),
        },
        'risk': {
            'prob_ruin_20pct_dd': float(prob_ruin),
            'prob_positive_quarter': float(prob_pos_quarter),
            'sim_profit_factor': float(sim_pf) if sim_pf != np.inf else 999.0,
        },
        'quarterly_pnl': {
            'mean': float(np.mean(paths[:, -1])),
            'std': float(np.std(paths[:, -1])),
            'p5': float(np.percentile(paths[:, -1], 5)),
            'p25': float(np.percentile(paths[:, -1], 25)),
            'p50': float(np.percentile(paths[:, -1], 50)),
            'p75': float(np.percentile(paths[:, -1], 75)),
            'p95': float(np.percentile(paths[:, -1], 95)),
        },
        # Store paths for charting (full array)
        '_paths': paths,
        '_daily': daily,
    }


# ─── Chart ────────────────────────────────────────────────────────────────────

def plot_results(all_results: list, save_path: Path):
    """Generate 5-panel Monte Carlo chart with dark theme."""
    BG    = '#0d1117'
    PANEL = '#161b22'
    GRID  = '#21262d'
    EDGE  = '#30363d'
    TEXT  = '#e6edf3'
    PALETTE = ['#00d4ff', '#ff6b35', '#98ff98', '#ffd700']

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32,
                           top=0.94, bottom=0.04, left=0.06, right=0.97)

    def stylize(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8.5)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(EDGE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, color=TEXT, fontsize=10.5, fontweight='bold', pad=7)
        ax.grid(True, color=GRID, alpha=0.6, linewidth=0.5)

    usd_fmt = plt.FuncFormatter(lambda x, _: f'${x:,.0f}')

    # ─── Panel 1 (top, full width): Equity fan chart ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    stylize(ax1, f'Equity Curve Fan — {N_SIM:,} Simulated Paths x {N_DAYS_QUARTER} Days '
                 f'(1 Quarter)  |  Initial Capital ${INITIAL_CAPITAL:,}')

    best = all_results[0]
    paths = best['_paths']
    days  = np.arange(paths.shape[1])

    rng_d = np.random.default_rng(7)
    idx   = rng_d.choice(len(paths), size=min(2500, len(paths)), replace=False)
    s_paths = paths[idx]
    finals  = s_paths[:, -1]

    for i, p in enumerate(s_paths):
        col = '#00d4ff' if finals[i] > 0 else '#ff4444'
        ax1.plot(days, p, color=col, alpha=0.025, linewidth=0.35)

    p5, p25, p50, p75, p95 = [np.percentile(paths, q, axis=0) for q in [5, 25, 50, 75, 95]]
    ax1.fill_between(days, p5,  p95,  alpha=0.12, color='#00d4ff')
    ax1.fill_between(days, p25, p75,  alpha=0.22, color='#00d4ff')
    ax1.plot(days, p50, color='#ffd700', lw=2.0, zorder=5,
             label=f'Median final: ${p50[-1]:,.0f}')
    ax1.axhline(0, color='#666', lw=0.8, linestyle='--')

    ruin_usd = -INITIAL_CAPITAL * RUIN_DRAWDOWN_PCT
    ax1.axhline(ruin_usd, color='#ff6b35', lw=1.5, linestyle=':',
                label=f'Ruin threshold: ${ruin_usd:,.0f}  '
                      f'(P(ruin)={best["risk"]["prob_ruin_20pct_dd"]:.1%})')

    ax1.yaxis.set_major_formatter(usd_fmt)
    ax1.set_xlabel('Trading Days', color=TEXT, fontsize=9)
    ax1.set_ylabel('Cumulative P&L (USD)', color=TEXT, fontsize=9)
    leg = ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.3, labelcolor=TEXT)
    leg.get_frame().set_facecolor(BG)

    pq = best['risk']['prob_positive_quarter']
    ax1.text(0.995, 0.97,
             f"{best['config_name']}\n"
             f"Daily mean: ${best['daily_pnl']['mean']:,.0f} ± ${best['daily_pnl']['std']:,.0f}\n"
             f"P(profitable quarter) = {pq:.1%}",
             transform=ax1.transAxes, ha='right', va='top',
             color=TEXT, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=BG, alpha=0.8, edgecolor=EDGE))

    # ─── Panel 2: Daily P&L histogram comparison ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    stylize(ax2, 'Daily P&L Distribution by Configuration')

    for i, res in enumerate(all_results):
        color = PALETTE[i % len(PALETTE)]
        df_flat = res['_daily'].ravel()
        clip = np.percentile(np.abs(df_flat), 99.5)
        df_clipped = np.clip(df_flat, -clip, clip)
        ax2.hist(df_clipped, bins=80, density=True, alpha=0.52, color=color,
                 histtype='stepfilled',
                 label=f"{res['config_name']} (μ=${res['daily_pnl']['mean']:,.0f})")
        ax2.axvline(res['daily_pnl']['mean'], color=color, lw=1.8, linestyle='--')

    ax2.axvline(0, color='white', lw=0.8, alpha=0.5)
    ax2.set_xlabel('Daily P&L (USD)', color=TEXT, fontsize=9)
    ax2.set_ylabel('Density', color=TEXT, fontsize=9)
    ax2.xaxis.set_major_formatter(usd_fmt)
    leg2 = ax2.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT)
    leg2.get_frame().set_facecolor(BG)

    # ─── Panel 3: Sharpe distribution ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    stylize(ax3, 'Annualized Sharpe Distribution (63-day sims)')

    for i, res in enumerate(all_results):
        color = PALETTE[i % len(PALETTE)]
        daily_i = res['_daily']
        pm = np.mean(daily_i, axis=1)
        ps = np.std(daily_i, axis=1, ddof=1)
        sh = np.where(ps > 0, pm / ps * np.sqrt(TRADING_DAYS_PER_YEAR), np.nan)
        sh_clip = np.clip(sh, -10, 15)
        p50 = res['sharpe']['p50']
        ax3.hist(sh_clip, bins=60, density=True, alpha=0.52, color=color,
                 histtype='stepfilled',
                 label=f"{res['config_name']} (p50={p50:.2f})")

    for thresh, lbl in [(0, '0'), (1, '1.0'), (2, '2.0'), (3, '3.0')]:
        ax3.axvline(thresh, color='#888' if thresh == 0 else '#ffd700' if thresh > 0 else '#666',
                    lw=1.0, linestyle='--', alpha=0.7)
        if thresh > 0:
            ax3.text(thresh + 0.05, ax3.get_ylim()[1] * 0.85 if ax3.get_ylim()[1] > 0 else 0.01,
                     lbl, color='#ffd700', fontsize=7, va='top')

    ax3.set_xlabel('Annualized Sharpe Ratio', color=TEXT, fontsize=9)
    ax3.set_ylabel('Density', color=TEXT, fontsize=9)
    leg3 = ax3.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT)
    leg3.get_frame().set_facecolor(BG)

    # ─── Panel 4: Max drawdown distribution ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    stylize(ax4, 'Max Drawdown Distribution — 1 Quarter (%  of $50k capital)')

    for i, res in enumerate(all_results):
        color = PALETTE[i % len(PALETTE)]
        p95_pct = res['max_drawdown']['p95_pct'] * 100
        mdd_sample = np.array([
            max_drawdown_usd(res['_paths'][j]) for j in range(min(3000, N_SIM))
        ]) / INITIAL_CAPITAL * 100
        ax4.hist(mdd_sample, bins=60, density=True, alpha=0.52, color=color,
                 histtype='stepfilled',
                 label=f"{res['config_name']} (p95={p95_pct:.1f}%)")

    ax4.axvline(RUIN_DRAWDOWN_PCT * 100, color='#ff4444', lw=1.8, linestyle='--',
                label=f'Ruin: {RUIN_DRAWDOWN_PCT:.0%}')
    ax4.set_xlabel('Max Drawdown (%)', color=TEXT, fontsize=9)
    ax4.set_ylabel('Density', color=TEXT, fontsize=9)
    leg4 = ax4.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT)
    leg4.get_frame().set_facecolor(BG)

    # ─── Panel 5: Quarterly P&L distribution ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    stylize(ax5, 'Quarterly P&L Distribution (63 Days)')

    for i, res in enumerate(all_results):
        color = PALETTE[i % len(PALETTE)]
        qpnl = res['_paths'][:, -1]
        clip_q = np.percentile(np.abs(qpnl), 99)
        qpnl_clip = np.clip(qpnl, -clip_q, clip_q)
        p50_q = res['quarterly_pnl']['p50']
        ax5.hist(qpnl_clip, bins=80, density=True, alpha=0.52, color=color,
                 histtype='stepfilled',
                 label=f"{res['config_name']} (p50=${p50_q:,.0f})")
        ax5.axvline(p50_q, color=color, lw=1.5, linestyle='--')

    ax5.axvline(0, color='white', lw=0.8, alpha=0.5)
    ax5.set_xlabel('Quarterly P&L (USD)', color=TEXT, fontsize=9)
    ax5.set_ylabel('Density', color=TEXT, fontsize=9)
    ax5.xaxis.set_major_formatter(usd_fmt)
    leg5 = ax5.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT)
    leg5.get_frame().set_facecolor(BG)

    fig.suptitle(
        f'CNN-Mamba v2 Execution Monte Carlo  |  {N_SIM:,} Sims x {N_DAYS_QUARTER} Days  |  '
        f'Cost: $4.70 RT FIFO Passive  |  t-dist(df={OBSERVED_DAILY_STD_DF}) Daily Model  |  '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}',
        color=TEXT, fontsize=12, fontweight='bold',
    )

    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Chart: {save_path}')


# ─── Summary Printer ──────────────────────────────────────────────────────────

def print_summary(all_results: list):
    print('\n' + '=' * 80)
    print('MONTE CARLO — FINAL RESULTS')
    print(f'N_SIM={N_SIM:,} | N_DAYS={N_DAYS_QUARTER} (1 quarter) | Capital=${INITIAL_CAPITAL:,}')
    print(f'Daily model: t-distribution(df={OBSERVED_DAILY_STD_DF}) + 15% bad-regime shocks')
    print(f'Observed daily std: ${OBSERVED_DAILY_STD_USD:,.0f} (from {OBSERVED_DAILY_STD_DF+1} fillsim days)')
    print(f'Cost: FIFO passive ${ES_RT_COMMISSION} RT = {ES_RT_COMMISSION_TICKS} ticks')
    print('=' * 80)

    for res in all_results:
        d = res['daily_pnl']
        sh = res['sharpe']
        so = res['sortino']
        dd = res['max_drawdown']
        rk = res['risk']
        pm = res['params']
        qp = res['quarterly_pnl']

        print(f'\n  ── {res["config_name"]} ──')
        print(f'     {res["description"]}')
        print(f'     Trades/day={pm["trades_per_day"]:.0f}  |  WR={pm["win_rate"]:.1%}  '
              f'|  PF={pm["profit_factor_empirical"]:.2f}  |  Sortino(trade)={pm["sortino_trade_level"]:.3f}')
        print()
        print(f'  Daily P&L (${N_SIM:,} samples):')
        print(f'    Mean:             ${d["mean"]:>10,.2f}')
        print(f'    Std dev:          ${d["std"]:>10,.2f}')
        print(f'    95% CI on mean:   [${d["ci_95_lo"]:,.0f}, ${d["ci_95_hi"]:,.0f}]')
        print(f'    p5/p50/p95:       ${d["p5"]:,.0f} / ${d["p50"]:,.0f} / ${d["p95"]:,.0f}')
        print(f'    % positive days:  {d["pct_positive_days"]:.1%}')
        print()
        print(f'  Annualized Sharpe (from {N_DAYS_QUARTER}-day path distribution):')
        print(f'    p10 / p50 / p90:  {sh["p10"]:.2f} / {sh["p50"]:.2f} / {sh["p90"]:.2f}')
        print()
        print(f'  Annualized Sortino:')
        print(f'    p10 / p50 / p90:  {so["p10"]:.2f} / {so["p50"]:.2f} / {so["p90"]:.2f}')
        print()
        print(f'  Max Drawdown (1 quarter, ${INITIAL_CAPITAL:,} capital):')
        print(f'    Mean / p90 / p95: ${dd["mean_usd"]:,.0f} / ${dd["p90_usd"]:,.0f} / ${dd["p95_usd"]:,.0f}')
        print(f'    As % of capital:  {dd["mean_pct"]:.1%} mean, {dd["p95_pct"]:.1%} p95')
        print()
        print(f'  Risk:')
        print(f'    P(ruin ≥20% DD):      {rk["prob_ruin_20pct_dd"]:.2%}')
        print(f'    P(profitable quarter):{rk["prob_positive_quarter"]:.2%}')
        print()
        print(f'  Quarterly P&L:')
        print(f'    p5 / p50 / p95:   ${qp["p5"]:,.0f} / ${qp["p50"]:,.0f} / ${qp["p95"]:,.0f}')


def save_stats_json(all_results: list, path: Path):
    """Save all stats except large arrays."""
    out = []
    for res in all_results:
        r = {k: v for k, v in res.items() if not k.startswith('_')}
        out.append(r)
    path.write_text(json.dumps(out, indent=2))
    print(f'  Stats: {path}')


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 70)
    print('Monte Carlo Execution Simulator — CNN-Mamba v2')
    print(f'Run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'N_SIM={N_SIM:,}  N_DAYS={N_DAYS_QUARTER}  Capital=${INITIAL_CAPITAL:,}')
    print(f'Daily model: t-distribution  |  Observed std=${OBSERVED_DAILY_STD_USD:,.0f}')
    print('=' * 70)

    rng = np.random.default_rng(2026_05_03)

    print('\nLoading data and building configs...')
    try:
        configs = build_configs_from_data()
        print(f'  Loaded {len(configs)} configs from empirical backtest data')
    except FileNotFoundError as e:
        print(f'  WARNING: {e}')
        print('  Using fallback reference configs from documented metrics...')
        configs = [
            ExecConfig(
                name='Ref_Top1pct',
                daily_mean_usd=1652,
                daily_std_usd=OBSERVED_DAILY_STD_USD,
                trades_per_day=252,
                win_rate=0.65,
                avg_win_ticks=1.73,
                avg_loss_ticks=-1.21,
                profit_factor=2.02,
                sortino_trade_level=0.345,
                n_observed_days=10,
                description='Reference: 1s Top1% from blended cost analysis',
            ),
            ExecConfig(
                name='Ref_RazerPPO',
                daily_mean_usd=400,
                daily_std_usd=OBSERVED_DAILY_STD_USD * 0.8,
                trades_per_day=30,
                win_rate=0.62,
                avg_win_ticks=3.0,
                avg_loss_ticks=-2.0,
                profit_factor=1.8,
                sortino_trade_level=0.45,
                n_observed_days=10,
                description='Razer PPO reference: $300-500/day',
            ),
        ]

    print('\nConfig summary:')
    for c in configs:
        print(f'  {c.name}: daily_mean=${c.daily_mean:,.0f}, '
              f'daily_std=${c.daily_std:,.0f}, Sharpe_theor={c.daily_sharpe:.2f}')

    print('\nRunning simulations...')
    all_results = []
    for cfg in configs:
        res = run_monte_carlo(cfg, n_sim=N_SIM, n_days=N_DAYS_QUARTER, rng=rng)
        all_results.append(res)

    print_summary(all_results)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = OUTPUT_DIR / f'monte_carlo_{ts}.png'
    stats_path = OUTPUT_DIR / f'mc_stats_{ts}.json'

    print('\nGenerating chart...')
    plot_results(all_results, chart_path)

    print('Saving stats...')
    save_stats_json(all_results, stats_path)

    print('\n' + '=' * 70)
    print('DONE')
    print(f'  Chart: {chart_path}')
    print(f'  Stats: {stats_path}')
    print('=' * 70)
