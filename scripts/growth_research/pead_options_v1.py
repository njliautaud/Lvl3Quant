#!/usr/bin/env python3
"""PEAD Options v1 — Post-Earnings Drift with Call/Put Spreads at $645.

PEAD equity was validated (Sharpe 1.18, CAGR 36.8%, WR 63.5%, 3/4 gates).
But at $645 we can't buy stocks. This converts PEAD to OPTIONS:

  - Earnings gap up > 3%: Buy call spread on the stock (capture drift up)
  - Hold 20-40 days
  - $645 starting capital, max $200/trade

Key advantages of options for PEAD:
  - Defined risk (max loss = debit paid)
  - Leverage (can trade $100+ stocks with $100-200)
  - Works at $645 (equity PEAD needs $10K+)

Key risks:
  - Theta decay over 40-day hold (use 45-60 DTE to slow it)
  - IV crush post-earnings reduces option value
  - Spread width limits upside

Variants:
  A: Gap >3%, call spread, 30 DTE, 20d exit
  B: Gap >3%, call spread, 45 DTE, 30d exit
  C: Gap >5%, call spread, 45 DTE, 30d exit (higher quality)
  D: Gap >3%, call spread, 45 DTE, 40d exit (full PEAD)
  E: Gap >3% + IV rank <50, call spread, 45 DTE, 30d exit
  F: Gap >3%, call spread, 45 DTE, 30d exit, half-size ($100 max)

Random control: random stocks on same dates (is PEAD the edge or general drift?).
"""
import json, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy.stats import norm

BASE = Path('/home/jupiter/Lvl3Quant')
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'pead_options_v1_results.json'
CACHE_DIR = BASE / 'output' / 'pead_options_v1'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

# 30 large-caps with liquid options (same universe as PEAD equity v1)
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
    'BA', 'DIS', 'SBUX', 'HD', 'LOW', 'MCD', 'NKE', 'COST', 'WMT',
    'JPM', 'GS', 'BAC', 'MS', 'JNJ', 'PG', 'KO', 'UNH', 'ABBV', 'CRM', 'NOW',
]

CAP = 645.0
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # 4 legs round-trip
HAIRCUT = 0.15  # bid-ask haircut
EARNINGS_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}

# ==================== DATA ====================
def download_data():
    """Download price data for all tickers + SPY + VIX."""
    import yfinance as yf
    cache = CACHE_DIR / 'prices_cache.parquet'

    if cache.exists():
        df = pd.read_parquet(cache)
        fprint(f"  Loaded cached prices: {len(df)} rows")
        return df

    fprint(f"  Downloading {len(TICKERS)} tickers...")
    all_data = []
    for tk in TICKERS + ['SPY']:
        try:
            d = yf.download(tk, start='2016-01-01', end='2026-07-26', progress=False, auto_adjust=True)
            if len(d) > 0:
                d = d.reset_index()
                d.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in d.columns]
                d['ticker'] = tk
                all_data.append(d[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']])
                fprint(f"    {tk}: {len(d)} days")
        except Exception as e:
            fprint(f"    {tk}: FAILED {e}")
        time.sleep(0.15)

    df = pd.concat(all_data, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(cache)
    fprint(f"  Cached {len(df)} rows")
    return df

def download_vix():
    import yfinance as yf
    cache = CACHE_DIR / 'vix_cache.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    d = yf.download('^VIX', start='2016-01-01', end='2026-07-26', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        close = d['Close']['^VIX'] if '^VIX' in d['Close'].columns else d['Close'].iloc[:, 0]
    else:
        close = d['Close']
    close = close.dropna()
    vdf = pd.DataFrame({'date': close.index, 'vix': close.values.flatten()}).dropna()
    vdf.to_parquet(cache)
    return vdf

# ==================== EARNINGS DETECTION ====================
def detect_earnings(prices_df, ticker):
    """Detect earnings events from overnight gaps during earnings season.

    Heuristic: if |gap| > 2% during earnings months (Jan/Feb, Apr/May, Jul/Aug, Oct/Nov),
    it's likely an earnings event.
    """
    tk_data = prices_df[prices_df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
    if len(tk_data) < 60:
        return []

    events = []
    for i in range(1, len(tk_data)):
        row = tk_data.iloc[i]
        prev = tk_data.iloc[i-1]
        month = row['date'].month

        if month not in EARNINGS_MONTHS:
            continue

        gap = (row['open'] - prev['close']) / prev['close']

        if abs(gap) < 0.02:  # At least 2% gap to be earnings
            continue

        # Check we haven't had another gap in last 60 days (avoid double-counting)
        recent_events = [e for e in events if (row['date'] - e['date']).days < 60]
        if recent_events:
            continue

        # Compute IV rank proxy: 20d realized vol relative to 60d
        if i >= 60:
            ret_20 = tk_data['close'].iloc[i-20:i].pct_change().std() * np.sqrt(252)
            ret_60 = tk_data['close'].iloc[i-60:i].pct_change().std() * np.sqrt(252)
            iv_rank = ret_20 / (ret_60 + 1e-10)  # >1 means elevated vol
        else:
            iv_rank = 1.0

        events.append({
            'date': row['date'],
            'ticker': ticker,
            'gap_pct': gap * 100,
            'prev_close': prev['close'],
            'open_price': row['open'],
            'close_price': row['close'],
            'idx': i,
            'iv_rank': iv_rank,
        })

    return events

# ==================== OPTION PRICING ====================
def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def price_call_spread(S, width_pct, dte, vol, haircut=HAIRCUT):
    """Price a bull call spread: buy ATM call, sell OTM call."""
    T = dte / 365.0
    K_long = round(S, 2)  # ATM
    K_short = round(S * (1 + width_pct), 2)  # OTM

    long_price = bs_price(S, K_long, T, vol, opt='call') * (1 + haircut)  # Pay more
    short_price = bs_price(S, K_short, T, vol, opt='call') * (1 - haircut)  # Receive less

    debit = long_price - short_price
    width = K_short - K_long
    max_profit = width - debit

    cost = debit * 100 + SPREAD_COMM
    potential = max_profit * 100 - SPREAD_COMM

    return {
        'K_long': K_long, 'K_short': K_short,
        'debit': debit, 'width': width,
        'cost': cost, 'max_profit': potential,
        'T': T, 'vol': vol,
    }

def value_spread_at_time(S, K_long, K_short, T_remaining, vol):
    """Value a call spread at some future time."""
    if T_remaining <= 0:
        # At expiration — intrinsic only
        return (max(0, S - K_long) - max(0, S - K_short)) * 100

    long_val = bs_price(S, K_long, T_remaining, vol, opt='call')
    short_val = bs_price(S, K_short, T_remaining, vol, opt='call')
    return (long_val - short_val) * 100

# ==================== SIMULATION ====================
def simulate(name, events, prices_df, spy_df, vix_df,
             gap_threshold=3.0, dte=45, exit_days=30,
             max_position=200, iv_rank_max=999, width_pct=0.05):
    """Simulate PEAD options strategy."""
    fprint(f"\n--- {name} ---")

    equity = CAP
    trades = []
    eq_curve = [CAP]

    # SPY regime classification
    spy_close = spy_df.set_index('date')['close']
    spy_sma200 = spy_close.rolling(200).mean()

    for event in sorted(events, key=lambda x: x['date']):
        gap = event['gap_pct']
        tk = event['ticker']
        dt = event['date']
        idx = event['idx']

        # Gap direction filter (long-only, validated)
        if gap < gap_threshold:
            continue

        # IV rank filter
        if event['iv_rank'] > iv_rank_max:
            continue

        # Get ticker data
        tk_data = prices_df[prices_df['ticker'] == tk].sort_values('date').reset_index(drop=True)
        if idx + exit_days + 5 >= len(tk_data):
            continue

        # Position sizing
        pos_size = min(max_position, equity * 0.35)
        if pos_size < 50:
            eq_curve.append(equity)
            continue

        # Price the spread at entry
        S_entry = event['open_price']

        # Post-earnings IV crush: use lower vol than pre-earnings
        # Typical IV drops 30-50% after earnings
        pre_vol = tk_data['close'].iloc[max(0,idx-21):idx].pct_change().std() * np.sqrt(252)
        post_vol = pre_vol * 0.65  # 35% IV crush

        spread = price_call_spread(S_entry, width_pct, dte, post_vol)

        if spread['cost'] <= 0 or spread['cost'] > pos_size or spread['max_profit'] <= 0:
            continue

        # Simulate forward
        entry_cost = spread['cost']
        best_pnl = -entry_cost
        exit_pnl = None
        exit_idx = idx + exit_days
        exit_date = tk_data['date'].iloc[min(exit_idx, len(tk_data)-1)]

        for di in range(1, exit_days + 1):
            ci = idx + di
            if ci >= len(tk_data):
                break

            S_cur = tk_data['close'].iloc[ci]
            days_remaining = max(1, dte - di)
            T_rem = days_remaining / 365.0

            # Vol mean-reverts further after earnings
            cur_vol = post_vol * (0.9 + 0.1 * di / exit_days)

            spread_val = value_spread_at_time(S_cur, spread['K_long'], spread['K_short'], T_rem, cur_vol)
            pnl = spread_val - entry_cost

            # Profit target: 50% of max profit
            if pnl >= spread['max_profit'] * 0.50:
                exit_pnl = pnl
                exit_idx = ci
                exit_date = tk_data['date'].iloc[ci]
                break

        if exit_pnl is None:
            # Exit at hold period end
            ci = min(idx + exit_days, len(tk_data) - 1)
            S_exit = tk_data['close'].iloc[ci]
            days_remaining = max(1, dte - exit_days)
            T_rem = days_remaining / 365.0
            cur_vol = post_vol * 0.95
            spread_val = value_spread_at_time(S_exit, spread['K_long'], spread['K_short'], T_rem, cur_vol)
            exit_pnl = spread_val - entry_cost
            exit_idx = ci
            exit_date = tk_data['date'].iloc[ci]

        # Regime at entry
        if dt in spy_close.index and dt in spy_sma200.index:
            bull_regime = spy_close.loc[dt] >= spy_sma200.loc[dt]
        else:
            # Find nearest
            nearest = spy_close.index[spy_close.index.get_indexer([dt], method='nearest')]
            bull_regime = True if len(nearest) == 0 else spy_close.loc[nearest[0]] >= spy_sma200.loc[nearest[0]] if nearest[0] in spy_sma200.index else True

        equity += exit_pnl
        trades.append({
            'entry': str(dt.date()),
            'exit': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
            'ticker': tk,
            'gap_pct': round(gap, 1),
            'entry_price': round(S_entry, 2),
            'spread_cost': round(entry_cost, 2),
            'pnl': round(exit_pnl, 2),
            'win': exit_pnl > 0,
            'hold_days': exit_idx - idx,
            'regime': 'bull' if bull_regime else 'bear',
            'iv_rank': round(event['iv_rank'], 2),
            'equity_at_trade': round(equity, 2),
        })
        eq_curve.append(equity)

    return trades, equity, eq_curve

# ==================== VALIDATION ====================
def compute_honest_sharpe(trades):
    if not trades: return 0.0, 0.0, []
    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['entry_dt'].dt.to_period('M')
    monthly = []
    for mo in sorted(tdf['month'].unique()):
        mt = tdf[tdf['month'] == mo]
        pnl = mt['pnl'].sum()
        eq = max(mt['equity_at_trade'].iloc[0] - mt['pnl'].iloc[0], 100)
        monthly.append(pnl / eq)
    rets = np.array(monthly)
    if len(rets) < 4: return 0.0, 0.0, rets
    sh = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(12))
    dn = rets[rets < 0]
    so = float(np.mean(rets) / (np.std(dn) + 1e-10) * np.sqrt(12)) if len(dn) > 1 else 0.0
    return sh, so, rets

def validate(trades, final_eq, eq_curve, name):
    if not trades:
        fprint(f"  {name}: No trades")
        return None

    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    sh, so, rets = compute_honest_sharpe(trades)
    ny = max(len(rets)/12, 0.5)
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)

    # Ticker concentration
    tdf = pd.DataFrame(trades)
    top_ticker = tdf['ticker'].value_counts().iloc[0] / n * 100 if n > 0 else 0
    unique_tickers = tdf['ticker'].nunique()

    # Regime
    bt = [t for t in trades if t['regime']=='bull']
    brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    # 4-gate validation
    gates = 0; pp, rg, g1, g2, g3, g4, h1, h2 = 1.0, 1.0, False, False, False, False, 0, 0
    if len(rets) >= 10:
        rs = np.mean(rets)/(np.std(rets)+1e-10)
        pp = sum(1 for _ in range(2000) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/2000
        g1 = pp < 0.05; gates += g1
        rg = abs(bw-brw)/max(bw,brw,1); g2 = rg < 0.50; gates += g2
        mid = len(rets)//2
        h1 = np.mean(rets[:mid])/(np.std(rets[:mid])+1e-10) if mid > 3 else 0
        h2 = np.mean(rets[mid:])/(np.std(rets[mid:])+1e-10) if len(rets)-mid > 3 else 0
        g3 = h1 > 0 and h2 > 0; gates += g3
        tr = np.sort(rets)[:-1]; g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0 if len(rets) > 5 else False; gates += g4

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1),
         'sharpe': round(sh,2), 'sortino': round(so,2),
         'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'final_equity': round(final_eq,2),
         'unique_tickers': unique_tickers, 'top_ticker_pct': round(top_ticker,1),
         'avg_gap': round(np.mean([t['gap_pct'] for t in trades]),1),
         'avg_pnl': round(np.mean(pnls),1),
         'regime_bull_wr': round(bw,1), 'regime_bear_wr': round(brw,1),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
         'h1_sh': round(h1,2), 'h2_sh': round(h2,2)}

    fprint(f"  {name}: {n} trades ({unique_tickers} tickers) | "
           f"WR {wr:.1f}% | Sh {sh:.2f} | So {so:.2f} | CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | "
           f"PF {pf:.2f} | ${CAP}->${final_eq:.0f} | Gates {gates}/4")
    fprint(f"    Avg gap: {np.mean([t['gap_pct'] for t in trades]):.1f}% | Avg PnL: ${np.mean(pnls):.1f}")
    fprint(f"    G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'}")
    return r

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"PEAD Options v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*90}")
    fprint(f"Post-Earnings Drift with Call Spreads at $645")
    fprint(f"{'='*90}")

    prices_df = download_data()
    vix_df = download_vix()

    # SPY data
    spy_df = prices_df[prices_df['ticker'] == 'SPY'].copy()
    fprint(f"SPY: {len(spy_df)} days")

    # Detect earnings events
    fprint("\nDetecting earnings events...")
    all_events = []
    for tk in TICKERS:
        events = detect_earnings(prices_df, tk)
        fprint(f"  {tk}: {len(events)} earnings events")
        all_events.extend(events)

    fprint(f"\nTotal: {len(all_events)} earnings events across {len(TICKERS)} stocks")
    gap_up = sum(1 for e in all_events if e['gap_pct'] > 3)
    gap_dn = sum(1 for e in all_events if e['gap_pct'] < -3)
    fprint(f"  Gap up >3%: {gap_up}, Gap down <-3%: {gap_dn}")

    # Configs
    configs = [
        # name, gap_thresh, dte, exit_days, max_pos, iv_rank_max, width_pct
        ('A_30DTE_20d',     3.0, 30, 20, 200, 999, 0.05),
        ('B_45DTE_30d',     3.0, 45, 30, 200, 999, 0.05),
        ('C_Gap5_45DTE',    5.0, 45, 30, 200, 999, 0.05),
        ('D_45DTE_40d',     3.0, 45, 40, 200, 999, 0.05),
        ('E_IVRank_45DTE',  3.0, 45, 30, 200, 1.0, 0.05),  # IV rank < 1.0 (normal vol)
        ('F_HalfSize_45DTE',3.0, 45, 30, 100, 999, 0.05),
    ]

    results = []
    for nm, gt, dte, ed, mp, ivr, wp in configs:
        tr, eq, cu = simulate(nm, all_events, prices_df, spy_df, vix_df,
                              gap_threshold=gt, dte=dte, exit_days=ed,
                              max_position=mp, iv_rank_max=ivr, width_pct=wp)
        r = validate(tr, eq, cu, nm)
        if r: results.append(r)

    # Random control: use random stocks on same dates
    fprint(f"\n--- RANDOM CONTROL ---")
    np.random.seed(42)
    random_events = []
    for e in all_events:
        if e['gap_pct'] < 3.0: continue
        # Substitute random ticker
        random_tk = np.random.choice(TICKERS)
        tk_data = prices_df[prices_df['ticker'] == random_tk].sort_values('date')
        # Find closest date
        date_idx = tk_data['date'].searchsorted(e['date'])
        if date_idx < 1 or date_idx >= len(tk_data) - 50: continue
        random_events.append({
            'date': tk_data['date'].iloc[date_idx],
            'ticker': random_tk,
            'gap_pct': e['gap_pct'],  # Keep same gap magnitude
            'prev_close': tk_data['close'].iloc[date_idx-1],
            'open_price': tk_data['open'].iloc[date_idx],
            'close_price': tk_data['close'].iloc[date_idx],
            'idx': date_idx,
            'iv_rank': 1.0,
        })

    rtr, req, rcu = simulate('Random_Stock', random_events, prices_df, spy_df, vix_df,
                              gap_threshold=0.0, dte=45, exit_days=30, max_position=200)
    rsh, rso, rrets = compute_honest_sharpe(rtr) if rtr else (0, 0, [])
    if rtr:
        rn = len(rtr); rwins = sum(1 for t in rtr if t['win'])
        fprint(f"  Random: {rn} trades | WR {rwins/rn*100:.1f}% | Sh {rsh:.2f} | ${CAP}->${req:.0f}")

    if not results:
        fprint("No results")
        return

    # Summary
    fprint(f"\n{'='*110}")
    fprint(f"SUMMARY — PEAD Options v1")
    fprint(f"{'='*110}")
    fprint(f"{'Variant':<22} {'#':>5} {'Tks':>4} {'WR':>6} {'Sh':>7} {'So':>7} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>8} {'G':>4}")
    fprint("-"*110)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['n_trades']:>5} {r['unique_tickers']:>4} "
               f"{r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['cagr_pct']:>6.1f}% "
               f"{r['maxdd_pct']:>6.1f}% {r['pf']:>6.2f} ${r['final_equity']:>7.0f} {r['gates']:>3}/4")

    # Random comparison
    if rtr:
        best = max(results, key=lambda x: x['sharpe'])
        fprint(f"\n=== RANDOM CONTROL ===")
        fprint(f"Random stock on same dates: Sh {rsh:.2f}, ${req:.0f}")
        fprint(f"Best PEAD config:           Sh {best['sharpe']:.2f}, ${best['final_equity']:.0f}")
        if abs(rsh) > 0.01:
            edge = (best['sharpe'] - rsh) / max(abs(rsh), 0.01) * 100
            fprint(f"PEAD selection edge: {edge:+.0f}% Sharpe improvement")
        fprint(f"If random is also profitable → call spreads themselves work, not PEAD selection")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save
    save_data = {'timestamp': t0.isoformat(), 'capital': CAP,
                 'results': results, 'runtime_s': round(elapsed,1),
                 'total_earnings_events': len(all_events),
                 'random_sharpe': round(rsh,2) if rtr else None}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'pead_options_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"pead_opt_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'strategy': 'PEAD_call_spreads',
                                   'universe_size': len(TICKERS), 'total_events': len(all_events)})
                for r in results:
                    p = r['name'][:18].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_wr': r['win_rate'],
                                        f'{p}_cagr': r['cagr_pct'], f'{p}_mdd': r['maxdd_pct'],
                                        f'{p}_gates': r['gates']})
                if rtr:
                    mlflow.log_metrics({'random_sharpe': round(rsh,2)})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*90}\nDONE — PEAD Options v1\n{'='*90}")

if __name__ == '__main__':
    main()
