"""
Dividend Aristocrat Momentum Backtest v1
=========================================
Buy strongest-performing Dividend Aristocrats (25+ yr dividend growth),
rotate monthly. Quality universe + momentum signal = regime-robust edge.

Variants:
A) Top-5 Momentum: equal weight top 5 by 3-month return
B) Top-3 Concentrated: top 3 only
C) Top-10 Diversified: top 10
D) Momentum + Value: above-median yield AND strong momentum
E) Sector-Balanced: top 5 but max 2 per sector
F) Quality Filter: above 200-SMA + positive 12-month return, then rank by 3-month momentum
"""

import numpy as np, pandas as pd, yfinance as yf, json, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Dividend Aristocrats universe
UNIVERSE = [
    'JNJ','PG','KO','PEP','MMM','ABT','ABBV','MCD','WMT','CL',
    'T','XOM','CVX','EMR','GPC','SWK','ITW','ADP','BDX','ED',
    'LOW','SHW','CINF','TGT','AFL','WBA','CAH','PPG','GD','AOS',
    'CTAS','ROP','SYY','NUE','BEN','LEG','HRL','FRT','TROW','MKC'
]

# GICS sector mapping for variant E
SECTOR_MAP = {
    'JNJ':'Healthcare','PG':'Staples','KO':'Staples','PEP':'Staples',
    'MMM':'Industrials','ABT':'Healthcare','ABBV':'Healthcare','MCD':'Discretionary',
    'WMT':'Staples','CL':'Staples','T':'Comm','XOM':'Energy','CVX':'Energy',
    'EMR':'Industrials','GPC':'Discretionary','SWK':'Industrials','ITW':'Industrials',
    'ADP':'Tech','BDX':'Healthcare','ED':'Utilities','LOW':'Discretionary',
    'SHW':'Materials','CINF':'Financials','TGT':'Discretionary','AFL':'Financials',
    'WBA':'Staples','CAH':'Healthcare','PPG':'Materials','GD':'Industrials',
    'AOS':'Industrials','CTAS':'Industrials','ROP':'Tech','SYY':'Staples',
    'NUE':'Materials','BEN':'Financials','LEG':'Discretionary','HRL':'Staples',
    'FRT':'RealEstate','TROW':'Financials','MKC':'Staples'
}

ACCOUNT = 645.0
OOT_S, OOT_E = '2022-01-01', '2026-07-31'
LOOKBACK_START = '2021-06-01'  # need 6 months before OOT for 3-month momentum
N_PERMS = 1000
SLIP_PCT = 0.0002  # 0.02% each way

def load():
    """Download all tickers + SPY."""
    data = {}
    print(f"Downloading {len(UNIVERSE)} aristocrats + SPY...")
    for t in UNIVERSE:
        try:
            df = yf.download(t, start=LOOKBACK_START, end=OOT_E, progress=False, auto_adjust=False)
            if len(df) > 200:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[t] = df
        except:
            pass
    spy = yf.download('SPY', start=LOOKBACK_START, end=OOT_E, progress=False, auto_adjust=False)
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy['sma200'] = spy['Close'].rolling(200).mean()
    spy['bull'] = spy['Close'] > spy['sma200']
    print(f"  Loaded {len(data)}/{len(UNIVERSE)} tickers")
    return data, spy


def compute_features(data):
    """Compute momentum, yield, and quality features for each stock."""
    featured = {}
    for t, df in data.items():
        df = df.copy()
        # 3-month (63 trading day) total return (approx, includes dividends via Adj Close)
        if 'Adj Close' in df.columns:
            df['ret_3m'] = df['Adj Close'].pct_change(63)
            df['ret_12m'] = df['Adj Close'].pct_change(252)
        else:
            df['ret_3m'] = df['Close'].pct_change(63)
            df['ret_12m'] = df['Close'].pct_change(252)
        # Dividend yield proxy: (Adj Close - Close) cumulative difference suggests dividends
        # Better: use dividends from yfinance
        if 'Adj Close' in df.columns:
            # Approximate annual dividend yield from price ratio
            adj_ratio = df['Adj Close'] / df['Close']
            df['div_yield_proxy'] = (1 - adj_ratio.pct_change(252).fillna(0)) * 0.05  # rough
        # Use a simpler proxy: trailing 12m dividend yield from adj close vs close divergence
        if 'Adj Close' in df.columns:
            df['annual_div_pct'] = (df['Close'] - df['Adj Close']).rolling(252).sum() / df['Close']
            df['annual_div_pct'] = df['annual_div_pct'].clip(0, 0.15)  # sanity
        else:
            df['annual_div_pct'] = 0.025  # assume 2.5% for aristocrats

        df['sma200'] = df['Close'].rolling(200).mean()
        df['above200'] = df['Close'] > df['sma200']
        featured[t] = df
    return featured


def generate_monthly_signals(featured, spy):
    """Generate cross-sectional rankings at each monthly rebalance."""
    # Get first trading day of each month in OOT period
    oot_dates = pd.date_range(OOT_S, OOT_E, freq='B')
    months_seen = set()
    rebal_dates = []
    for d in oot_dates:
        m = (d.year, d.month)
        if m not in months_seen:
            months_seen.add(m)
            rebal_dates.append(d)

    all_signals = []
    for rebal_d in rebal_dates:
        scores = []
        div_yields = []
        for t, df in featured.items():
            # Find nearest prior date
            prior = df.index[df.index <= rebal_d]
            if len(prior) == 0:
                continue
            d_use = prior[-1]
            if (rebal_d - d_use).days > 5:
                continue  # too stale

            row = df.loc[d_use]
            ret_3m = row.get('ret_3m', np.nan)
            if pd.isna(ret_3m):
                continue

            # Forward return: 21 trading days
            fwd_idx = df.index[df.index > d_use]
            if len(fwd_idx) < 21:
                continue
            if 'Adj Close' in df.columns:
                fwd_price = df.loc[fwd_idx[20], 'Adj Close']
                cur_price = row['Adj Close']
            else:
                fwd_price = df.loc[fwd_idx[20], 'Close']
                cur_price = row['Close']
            fwd_ret = fwd_price / cur_price - 1

            entry = {
                'date': rebal_d,
                'date_used': d_use,
                'ticker': t,
                'close': float(row['Close']),
                'ret_3m': float(ret_3m),
                'ret_12m': float(row.get('ret_12m', np.nan)),
                'div_yield': float(row.get('annual_div_pct', 0.025)),
                'above200': bool(row.get('above200', True)),
                'sector': SECTOR_MAP.get(t, 'Other'),
                'fwd_ret_21d': float(fwd_ret),
            }
            scores.append(entry)
            div_yields.append(entry['div_yield'])

        if len(scores) < 5:
            continue

        # Rank by 3-month momentum (descending)
        sdf = sorted(scores, key=lambda x: x['ret_3m'], reverse=True)
        median_yield = np.median(div_yields)
        for rank, s in enumerate(sdf, 1):
            s['mom_rank'] = rank
            s['median_yield'] = median_yield
            all_signals.append(s)

    return pd.DataFrame(all_signals)


def variant_A(sigs):
    """Top-5 Momentum: buy top 5 ranked aristocrats each month."""
    return sigs[sigs['mom_rank'] <= 5]


def variant_B(sigs):
    """Top-3 Concentrated."""
    return sigs[sigs['mom_rank'] <= 3]


def variant_C(sigs):
    """Top-10 Diversified."""
    return sigs[sigs['mom_rank'] <= 10]


def variant_D(sigs):
    """Momentum + Value: above-median yield AND top-10 momentum."""
    return sigs[(sigs['div_yield'] >= sigs['median_yield']) & (sigs['mom_rank'] <= 10)]


def variant_E(sigs):
    """Sector-Balanced: top-5 momentum but max 2 from any sector."""
    results = []
    for date in sigs['date'].unique():
        day = sigs[sigs['date'] == date].sort_values('mom_rank')
        sector_counts = {}
        selected = []
        for _, row in day.iterrows():
            sec = row['sector']
            if sector_counts.get(sec, 0) >= 2:
                continue
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected) >= 5:
                break
        if selected:
            results.append(pd.DataFrame(selected))
    return pd.concat(results) if results else pd.DataFrame()


def variant_F(sigs):
    """Quality Filter: above 200-SMA + positive 12-month return, then top-5 momentum."""
    quality = sigs[(sigs['above200'] == True) & (sigs['ret_12m'] > 0)]
    # Re-rank within quality subset each month
    results = []
    for date in quality['date'].unique():
        day = quality[quality['date'] == date].sort_values('ret_3m', ascending=False).head(5)
        results.append(day)
    return pd.concat(results) if results else pd.DataFrame()


def backtest(trades, n_positions):
    """Run backtest with equal-weight position sizing and monthly rebalance."""
    if trades is None or len(trades) == 0:
        return None
    t = trades.dropna(subset=['fwd_ret_21d']).sort_values('date')
    if len(t) == 0:
        return None

    equity = ACCOUNT
    equity_curve = [ACCOUNT]
    pnls = []
    trade_count = 0

    for date in sorted(t['date'].unique()):
        day_trades = t[t['date'] == date]
        n_stocks = len(day_trades)
        if n_stocks == 0:
            continue

        # Equal weight across selected stocks
        weight = 1.0 / n_stocks
        month_ret = 0.0
        for _, row in day_trades.iterrows():
            alloc = equity * weight
            shares = int(alloc / row['close'])
            if shares < 1:
                shares = 1
            cost_basis = shares * row['close']
            # Gross return - 2-way slippage
            gross_ret = row['fwd_ret_21d']
            net_ret = gross_ret - 2 * SLIP_PCT
            pnl = cost_basis * net_ret
            pnls.append({
                'date': date, 'ticker': row['ticker'],
                'pnl': pnl, 'ret': net_ret, 'shares': shares
            })
            month_ret += weight * net_ret
            trade_count += 1

        equity *= (1 + month_ret)
        equity_curve.append(equity)

    if trade_count < 10:
        return None

    p = pd.DataFrame(pnls)
    # Monthly aggregation
    monthly = p.groupby('date').agg({'pnl': 'sum', 'ret': 'mean'}).reset_index()
    monthly = monthly.sort_values('date')

    r = monthly['ret'].values
    total_pnl = equity_curve[-1] - ACCOUNT

    # Annualized Sharpe (monthly returns * sqrt(12))
    sh = np.mean(r) / np.std(r) * np.sqrt(12) if np.std(r) > 0 else 0
    # Sortino
    dr = r[r < 0]
    ds = np.std(dr) if len(dr) > 1 else np.std(r)
    so = np.mean(r) / ds * np.sqrt(12) if ds > 0 else 0
    # Profit factor
    gp = monthly[monthly['pnl'] > 0]['pnl'].sum()
    gl = abs(monthly[monthly['pnl'] < 0]['pnl'].sum())
    pf = gp / gl if gl > 0 else 999
    # Max drawdown from equity curve
    ec = np.array(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    mdd = dd.min()
    # Win rate (monthly)
    wr = (monthly['pnl'] > 0).mean() * 100

    return {
        'n_months': len(monthly),
        'n_trades': trade_count,
        'total_pnl': round(total_pnl, 2),
        'final_equity': round(equity_curve[-1], 2),
        'total_return_pct': round((equity_curve[-1] / ACCOUNT - 1) * 100, 2),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'sharpe': round(sh, 3),
        'sortino': round(so, 3),
        'mdd_pct': round(mdd * 100, 1),
        'returns': r.tolist()
    }


def regime_analysis(trades, spy):
    """Split performance by bull/bear regime."""
    if trades is None or len(trades) == 0:
        return 0, 0, 99

    t = trades.dropna(subset=['fwd_ret_21d']).sort_values('date')
    bull_map = spy['bull'].to_dict()

    bull_rets, bear_rets = [], []
    for _, row in t.iterrows():
        d = row['date']
        # Find regime at rebalance date
        prior_spy = spy.index[spy.index <= d]
        if len(prior_spy) == 0:
            continue
        regime_date = prior_spy[-1]
        is_bull = spy.loc[regime_date, 'bull']
        net_ret = row['fwd_ret_21d'] - 2 * SLIP_PCT
        if is_bull:
            bull_rets.append(net_ret)
        else:
            bear_rets.append(net_ret)

    def ann_sharpe(a):
        a = np.array(a)
        if len(a) < 3 or np.std(a) == 0:
            return 0
        return float(np.mean(a) / np.std(a) * np.sqrt(12))

    sb = ann_sharpe(bull_rets)
    sr = ann_sharpe(bear_rets)
    gap = abs(sb - sr) / max(abs(sb), abs(sr), 0.001)
    return round(sb, 3), round(sr, 3), round(gap, 3)


def permutation_test(trades, all_sigs, observed_sharpe, n=N_PERMS):
    """Shuffle monthly stock selections randomly, test if observed Sharpe is significant."""
    if trades is None or len(trades) < 10:
        return 1.0

    # Group by date to preserve monthly structure
    t = trades.dropna(subset=['fwd_ret_21d'])
    monthly_counts = t.groupby('date').size().to_dict()

    better = 0
    all_valid = all_sigs.dropna(subset=['fwd_ret_21d'])

    for _ in range(n):
        shuffled_trades = []
        for date, count in monthly_counts.items():
            pool = all_valid[all_valid['date'] == date]
            if len(pool) == 0:
                continue
            samp = pool.sample(n=min(count, len(pool)), replace=False)
            shuffled_trades.append(samp)

        if not shuffled_trades:
            continue
        shuf_df = pd.concat(shuffled_trades)
        n_pos = max(monthly_counts.values()) if monthly_counts else 5
        r = backtest(shuf_df, n_pos)
        if r and r['sharpe'] >= observed_sharpe:
            better += 1

    return round(better / n, 4)


def main():
    print("=" * 65)
    print("DIVIDEND ARISTOCRAT MOMENTUM BACKTEST v1")
    print(f"Walk-forward OOT: {OOT_S} to {OOT_E}")
    print(f"Starting capital: ${ACCOUNT}")
    print("=" * 65)

    data, spy = load()
    featured = compute_features(data)
    sigs = generate_monthly_signals(featured, spy)
    print(f"Total signal rows: {len(sigs)} across {sigs['date'].nunique()} months")

    variants = {
        'A_Top5_Momentum': (variant_A, 5),
        'B_Top3_Concentrated': (variant_B, 3),
        'C_Top10_Diversified': (variant_C, 10),
        'D_MomentumValue': (variant_D, 10),
        'E_SectorBalanced': (variant_E, 5),
        'F_QualityFilter': (variant_F, 5),
    }

    results = {}
    for name, (func, n_pos) in variants.items():
        print(f"\n{'=' * 55}")
        print(f"VARIANT {name}")
        tr = func(sigs)
        print(f"  Signals: {len(tr)} across {tr['date'].nunique() if len(tr) > 0 else 0} months")

        r = backtest(tr, n_pos)
        if r is None:
            print("  SKIP - too few trades")
            results[name] = {'variant': name, 'status': 'TOO_FEW_TRADES'}
            continue

        sb, sr, gap = regime_analysis(tr, spy)
        print(f"  Running {N_PERMS} permutations...")
        pp = permutation_test(tr, sigs, r['sharpe'])

        # 5-gate validation
        g1 = r['sharpe'] > 0.5
        g2 = pp < 0.05
        g3 = gap < 0.5
        g4 = r['mdd_pct'] > -50
        g5 = r['n_trades'] >= 20
        passed = sum([g1, g2, g3, g4, g5])

        gate_str = (
            f"{'PASS' if g1 else 'FAIL'} Sharpe({r['sharpe']:.2f}>0.5)  "
            f"{'PASS' if g2 else 'FAIL'} Perm(p={pp})  "
            f"{'PASS' if g3 else 'FAIL'} Regime(gap={gap:.2f})  "
            f"{'PASS' if g4 else 'FAIL'} MDD({r['mdd_pct']:.1f}%)  "
            f"{'PASS' if g5 else 'FAIL'} Trades({r['n_trades']}>=20)"
        )

        results[name] = {
            'variant': name,
            'n_months': r['n_months'],
            'n_trades': r['n_trades'],
            'total_pnl': r['total_pnl'],
            'final_equity': r['final_equity'],
            'total_return_pct': r['total_return_pct'],
            'win_rate': r['win_rate'],
            'profit_factor': r['profit_factor'],
            'sharpe': r['sharpe'],
            'sortino': r['sortino'],
            'mdd_pct': r['mdd_pct'],
            'sharpe_bull': sb,
            'sharpe_bear': sr,
            'regime_gap': gap,
            'perm_p': pp,
            'gates_passed': passed,
            'gate_detail': gate_str,
        }

        print(f"  Months: {r['n_months']}  Trades: {r['n_trades']}")
        print(f"  PnL: ${r['total_pnl']:+.2f}  Return: {r['total_return_pct']:+.1f}%  Final: ${r['final_equity']:.2f}")
        print(f"  Sharpe: {r['sharpe']:.3f}  Sortino: {r['sortino']:.3f}  WR: {r['win_rate']:.1f}%  PF: {r['profit_factor']:.2f}")
        print(f"  MDD: {r['mdd_pct']:.1f}%")
        print(f"  Bull Sharpe: {sb:.3f}  Bear Sharpe: {sr:.3f}  Gap: {gap:.3f}")
        print(f"  Perm p-value: {pp}")
        print(f"  GATES: {passed}/5 -- {gate_str}")

    # Find champion
    valid = {k: v for k, v in results.items() if 'sharpe' in v}
    if valid:
        champ = max(valid, key=lambda k: valid[k]['gates_passed'] * 100 + valid[k].get('sharpe', -99))
        champ_data = valid[champ]
    else:
        champ = "NONE"
        champ_data = {}

    # Summary
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} {'MDD%':>7} {'Perm':>6} {'Gates':>6}")
    print("-" * 75)
    for name, r in results.items():
        if 'sharpe' not in r:
            print(f"{name:<25} {'SKIP':>7}")
            continue
        print(f"{name:<25} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} {r['mdd_pct']:>6.1f}% {r['perm_p']:>6.3f} {r['gates_passed']:>3}/5")

    print(f"\nCHAMPION: {champ}")
    if champ_data:
        print(f"  Sharpe: {champ_data.get('sharpe')}, Return: {champ_data.get('total_return_pct')}%, Gates: {champ_data.get('gates_passed')}/5")

    # Save results
    output = {
        'metadata': {
            'strategy': 'Dividend Aristocrat Momentum',
            'script': 'dividend_aristocrat_momentum_backtest.py',
            'run_date': datetime.now().isoformat(),
            'oot_period': f'{OOT_S} to {OOT_E}',
            'starting_capital': ACCOUNT,
            'universe_size': len(data),
            'slippage_each_way': SLIP_PCT,
            'permutations': N_PERMS,
        },
        'variant_results': {k: {kk: vv for kk, vv in v.items() if kk != 'returns'} for k, v in results.items()},
        'champion': champ,
        'champion_detail': {k: v for k, v in champ_data.items() if k != 'returns'} if champ_data else {},
    }

    out_path = '/home/jupiter/Lvl3Quant/data/dividend_aristocrat_momentum_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
