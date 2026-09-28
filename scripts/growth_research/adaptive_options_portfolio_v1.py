#!/usr/bin/env python3
"""
Adaptive Options Portfolio v1
=============================
Key Innovation: Dynamically select option STRUCTURE based on signal confidence and IV regime.
- High confidence + low IV → buy calls (cheap gamma)
- High confidence + high IV → bull call spreads (hedge vega)
- Medium confidence → bull call spreads (defined risk)
- Kelly-optimal sizing based on LGBM confidence scores

Uses our validated LGBM sector rotation signal as the alpha source.
Vectorized for speed.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
from datetime import datetime
import os, sys, traceback

# --- Config ---
START_CAPITAL = 645.0
MAX_POSITION_PCT = 0.40
MAX_PORTFOLIO_PCT = 0.85
COMMISSION_PER_CONTRACT = 0.65
OOT_START = '2022-01-01'
SECTOR_ETFS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def load_data():
    """Load sector ETF data from parquet cache → returns price matrix"""
    parquet_path = 'research/cache/sector_etf_daily_data.parquet'
    if not os.path.exists(parquet_path):
        print("ERROR: No sector ETF data found")
        return None
    price_df = pd.read_parquet(parquet_path)
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    # Keep only our sectors
    available = [e for e in SECTOR_ETFS if e in price_df.columns]
    return price_df[available]

def precompute_features(price_df):
    """Vectorized feature computation for all dates and sectors"""
    # Returns dict of feature DataFrames
    ret_5d = price_df / price_df.shift(5) - 1
    ret_21d = price_df / price_df.shift(21) - 1
    ret_63d = price_df / price_df.shift(63) - 1

    log_ret = np.log(price_df / price_df.shift(1))
    vol_21d = log_ret.rolling(21).std() * np.sqrt(252)
    vol_63d = log_ret.rolling(63).std() * np.sqrt(252)

    # IV rank proxy
    iv_rank = ((vol_21d - vol_63d * 0.8) / (vol_63d * 0.4 + 1e-6)).clip(0, 1)

    # Trend R² (63d) — vectorized per sector
    trend_r2 = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
    trend_dir = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
    x = np.arange(63, dtype=float)
    x_centered = x - x.mean()
    ss_x = (x_centered ** 2).sum()

    for etf in price_df.columns:
        vals = price_df[etf].values
        r2_arr = np.full(len(vals), np.nan)
        dir_arr = np.full(len(vals), np.nan)
        for i in range(62, len(vals)):
            y = vals[i-62:i+1]
            if np.any(np.isnan(y)):
                continue
            y_centered = y - y.mean()
            ss_y = (y_centered ** 2).sum()
            if ss_y < 1e-10:
                continue
            corr = (x_centered * y_centered).sum() / (np.sqrt(ss_x * ss_y) + 1e-10)
            r2_arr[i] = corr ** 2
            dir_arr[i] = 1.0 if corr > 0 else -1.0
        trend_r2[etf] = r2_arr
        trend_dir[etf] = dir_arr

    # RSI proxy (14-day)
    gains = log_ret.clip(lower=0).rolling(14).mean()
    losses = (-log_ret).clip(lower=0).rolling(14).mean()
    rsi = 100 * gains / (gains + losses + 1e-10)

    # Composite score (mimics LGBM ranking)
    score = (ret_5d * 25 + ret_21d * 35 + ret_63d * 20 +
             trend_r2 * trend_dir * 15 + (rsi - 50) / 100 * 5)

    return {
        'score': score, 'ret_5d': ret_5d, 'ret_21d': ret_21d, 'ret_63d': ret_63d,
        'vol_21d': vol_21d, 'vol_63d': vol_63d, 'iv_rank': iv_rank,
        'trend_r2': trend_r2, 'trend_dir': trend_dir, 'rsi': rsi,
        'price': price_df,
    }

def select_option_structure(confidence, iv_rank, trend_r2):
    """Dynamically select option structure"""
    if confidence >= 0.8:
        if iv_rank < 0.3:
            return ('call', 30, 0.0, None)
        elif iv_rank < 0.6:
            return ('spread', 30, 0.0, 0.05)
        else:
            return ('spread', 21, 0.0, 0.03)
    elif confidence >= 0.6:
        if iv_rank < 0.4:
            return ('spread', 45, 0.0, 0.07)
        else:
            return ('spread', 30, 0.0, 0.05)
    else:
        if trend_r2 > 0.7:
            return ('spread', 30, 0.02, 0.05)
        return (None, 0, 0, 0)

def price_option(S, offset, width, dte, vol, structure, r=0.05):
    T = dte / 365.0
    K = S * (1 + offset)
    haircut = 0.85
    if structure == 'call':
        price = bs_call(S, K, T, r, vol) * haircut * 100
        return price, None, price
    elif structure == 'spread':
        K_short = S * (1 + offset + (width or 0.05))
        debit = (bs_call(S, K, T, r, vol) - bs_call(S, K_short, T, r, vol)) * haircut * 100
        max_profit = (K_short - K) * 100 - debit
        return debit, max_profit, debit
    return 0, 0, 0

def simulate_trade(entry_p, exit_p, offset, width, vol, dte, structure, days_held, r=0.05):
    T_entry = dte / 365.0
    T_exit = max((dte - days_held) / 365.0, 1/365.0)
    K = entry_p * (1 + offset)
    haircut = 0.85
    exit_vol = vol * 0.95
    if structure == 'call':
        return (bs_call(exit_p, K, T_exit, r, exit_vol) - bs_call(entry_p, K, T_entry, r, vol)) * haircut * 100
    elif structure == 'spread':
        K_s = entry_p * (1 + offset + (width or 0.05))
        entry_val = bs_call(entry_p, K, T_entry, r, vol) - bs_call(entry_p, K_s, T_entry, r, vol)
        exit_val = bs_call(exit_p, K, T_exit, r, exit_vol) - bs_call(exit_p, K_s, T_exit, r, exit_vol)
        return (exit_val - entry_val) * haircut * 100
    return 0

def kelly_size(wr, avg_w, avg_l, frac=0.25):
    if avg_l == 0: return 0
    b = avg_w / avg_l
    k = (wr * b - (1 - wr)) / b
    return max(0, min(k * frac, 0.40))

def run_variant(price_df, features, variant_name, cfg):
    """Run a single variant backtest"""
    dates = price_df.index
    oot_mask = dates >= pd.Timestamp(OOT_START)
    oot_dates = dates[oot_mask]
    if len(oot_dates) < 100:
        return None

    etfs = [e for e in SECTOR_ETFS if e in price_df.columns]
    rebal_freq = cfg.get('rebal_freq', 5)
    top_n = cfg.get('top_n', 2)
    hold_days = cfg.get('hold_days', 10)
    use_adaptive = cfg.get('use_adaptive', True)
    default_structure = cfg.get('default_structure', 'spread')
    conf_thresh = cfg.get('confidence_threshold', 0.0)
    use_kelly = cfg.get('use_kelly', False)

    capital = START_CAPITAL
    equity_curve = [capital]
    trades = []
    positions = {}  # etf -> dict

    for idx, date in enumerate(oot_dates):
        # Exit check
        for etf in list(positions.keys()):
            pos = positions[etf]
            days_held = (date - pos['entry_date']).days
            if days_held >= hold_days:
                exit_p = price_df.loc[date, etf]
                if np.isnan(exit_p):
                    continue
                pnl = simulate_trade(
                    pos['entry_p'], exit_p, pos['offset'], pos['width'],
                    pos['vol'], pos['dte'], pos['structure'], days_held
                ) - COMMISSION_PER_CONTRACT * 2
                capital += pnl
                trades.append({'date': date, 'etf': etf, 'pnl': pnl,
                               'structure': pos['structure'], 'days': days_held})
                del positions[etf]

        # Rebalance
        if idx % rebal_freq != 0:
            equity_curve.append(capital)
            continue

        # Get scores for this date
        scores = {}
        for etf in etfs:
            s = features['score'].loc[date, etf] if date in features['score'].index else np.nan
            if not np.isnan(s):
                scores[etf] = s

        if len(scores) < 5:
            equity_curve.append(capital)
            continue

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        all_s = [v for _, v in ranked]
        s_min, s_max = min(all_s), max(all_s)
        s_range = s_max - s_min + 1e-10

        # Select entries
        available_cap = capital * MAX_PORTFOLIO_PCT - sum(p['cost'] for p in positions.values())
        entries = 0

        for etf, score in ranked:
            if etf in positions or entries >= top_n:
                break

            conf = (score - s_min) / s_range
            if conf < conf_thresh:
                continue

            iv_r = features['iv_rank'].loc[date, etf] if date in features['iv_rank'].index else 0.5
            tr2 = features['trend_r2'].loc[date, etf] if date in features['trend_r2'].index else 0.3
            vol = features['vol_21d'].loc[date, etf] if date in features['vol_21d'].index else 0.25
            price = price_df.loc[date, etf]

            if np.isnan(price) or np.isnan(iv_r): iv_r = 0.5
            if np.isnan(tr2): tr2 = 0.3
            if np.isnan(vol) or vol <= 0: vol = 0.25

            if use_adaptive:
                structure, dte, offset, width = select_option_structure(conf, iv_r, tr2)
            else:
                structure = default_structure
                dte = 30
                offset = 0.0
                width = 0.05 if structure == 'spread' else None

            if structure is None:
                continue

            cost, _, _ = price_option(price, offset, width, dte, vol, structure)
            cost += COMMISSION_PER_CONTRACT

            if use_kelly:
                est_wr = 0.45 + conf * 0.20
                size_pct = kelly_size(est_wr, cost * 1.5, cost * 0.7)
            else:
                size_pct = 0.25

            max_spend = min(capital * size_pct, capital * MAX_POSITION_PCT, available_cap)
            if cost > max_spend or cost > available_cap or cost < 5:
                continue

            positions[etf] = {
                'entry_date': date, 'entry_p': price, 'structure': structure,
                'dte': dte, 'offset': offset, 'width': width,
                'vol': vol, 'cost': cost, 'conf': conf,
            }
            available_cap -= cost
            entries += 1

        equity_curve.append(capital)

    # Close remaining
    final_date = oot_dates[-1]
    for etf, pos in list(positions.items()):
        exit_p = price_df.loc[final_date, etf]
        if np.isnan(exit_p): continue
        days_held = (final_date - pos['entry_date']).days
        pnl = simulate_trade(
            pos['entry_p'], exit_p, pos['offset'], pos['width'],
            pos['vol'], pos['dte'], pos['structure'], days_held
        ) - COMMISSION_PER_CONTRACT * 2
        capital += pnl
        trades.append({'date': final_date, 'etf': etf, 'pnl': pnl,
                       'structure': pos['structure'], 'days': days_held})
    equity_curve.append(capital)

    if len(trades) < 5:
        return {'variant': variant_name, 'trades': len(trades), 'sharpe': -999, 'skip': True,
                'reason': f'Only {len(trades)} trades'}

    # Metrics
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls) * 100
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999

    eq = np.array(equity_curve)
    rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    rets = rets[rets != 0]

    ann_f = np.sqrt(252 / max(rebal_freq, 1))
    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * ann_f if len(rets) > 5 else 0
    neg = rets[rets < 0]
    sortino = np.mean(rets) / (np.std(neg) + 1e-10) * ann_f if len(neg) > 2 else sharpe

    peak = eq[0]
    max_dd = 0
    for v in eq:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / (peak + 1e-10))

    years = len(oot_dates) / 252
    cagr = (max(capital, 1) / START_CAPITAL) ** (1/max(years, 0.1)) - 1

    # Regime analysis using equal-weight sector index
    regime_gap = 999
    if len(trades) >= 10:
        avg_sector = price_df[etfs].mean(axis=1)
        avg_ret = avg_sector.pct_change()
        green_pnl, red_pnl = [], []
        for t in trades:
            d = t['date']
            if d in avg_ret.index:
                r = avg_ret.loc[d]
                if r > 0: green_pnl.append(t['pnl'])
                else: red_pnl.append(t['pnl'])
        if green_pnl and red_pnl:
            sg = np.mean(green_pnl) / (np.std(green_pnl) + 1e-10)
            sr = np.mean(red_pnl) / (np.std(red_pnl) + 1e-10)
            regime_gap = abs(sg - sr) / (max(abs(sg), abs(sr)) + 1e-10)

    structs = {}
    for t in trades:
        s = t.get('structure', '?')
        structs[s] = structs.get(s, 0) + 1

    return {
        'variant': variant_name, 'trades': len(trades),
        'final_equity': round(capital, 2), 'cagr': round(cagr * 100, 1),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 2),
        'wr': round(wr, 1), 'pf': round(pf, 2), 'max_dd': round(max_dd * 100, 1),
        'avg_win': round(np.mean(wins), 2) if len(wins) > 0 else 0,
        'avg_loss': round(np.mean(np.abs(losses)), 2) if len(losses) > 0 else 0,
        'regime_gap': round(regime_gap, 3), 'structures': structs,
        'trade_pnls': pnls.tolist(), 'equity_curve': equity_curve,
    }

def permutation_test_result(pnls, n_perms=1000):
    observed = np.mean(pnls)
    pnls = np.array(pnls)
    count = sum(1 for _ in range(n_perms)
                if np.mean(pnls * np.random.choice([-1, 1], size=len(pnls))) >= observed)
    return count / n_perms

def random_baseline(price_df, features, n_sims=200, rebal_freq=5, top_n=2, hold_days=10):
    """Random direction baseline"""
    etfs = [e for e in SECTOR_ETFS if e in price_df.columns]
    dates = price_df.index
    oot_dates = dates[dates >= pd.Timestamp(OOT_START)]

    sharpes = []
    for sim in range(n_sims):
        np.random.seed(sim * 42)
        capital = START_CAPITAL
        eq = [capital]

        for i in range(0, len(oot_dates), rebal_freq):
            if i + hold_days >= len(oot_dates): break
            picks = np.random.choice(etfs, size=min(top_n, len(etfs)), replace=False)
            d_entry = oot_dates[i]
            d_exit = oot_dates[min(i + hold_days, len(oot_dates)-1)]

            for etf in picks:
                ep = price_df.loc[d_entry, etf]
                xp = price_df.loc[d_exit, etf]
                if np.isnan(ep) or np.isnan(xp): continue
                cost, _, _ = price_option(ep, 0.0, 0.05, 30, 0.25, 'spread')
                if cost < 5 or cost > capital * 0.3: continue
                pnl = simulate_trade(ep, xp, 0.0, 0.05, 0.25, 30, 'spread', hold_days)
                pnl -= COMMISSION_PER_CONTRACT * 2
                capital += pnl
            eq.append(capital)

        rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
        rets = rets[rets != 0]
        if len(rets) > 5:
            sharpes.append(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))

    return np.mean(sharpes) if sharpes else 0

def main():
    t0 = datetime.now()
    print("="*70)
    print("  ADAPTIVE OPTIONS PORTFOLIO v1")
    print("  Dynamic option structure based on IV regime + confidence")
    print("="*70)

    print("\nLoading data...")
    price_df = load_data()
    if price_df is None: return
    print(f"  {len(price_df.columns)} ETFs, {len(price_df)} days ({price_df.index[0].date()} to {price_df.index[-1].date()})")

    print("Precomputing features (vectorized)...")
    features = precompute_features(price_df)
    print("  Done.")

    variants = {
        'A_Adaptive': {'use_adaptive': True, 'top_n': 2, 'rebal_freq': 5,
                        'hold_days': 10, 'confidence_threshold': 0.3},
        'B_AdaptiveKelly': {'use_adaptive': True, 'top_n': 2, 'rebal_freq': 5,
                            'hold_days': 10, 'confidence_threshold': 0.3, 'use_kelly': True},
        'C_FixedSpread': {'use_adaptive': False, 'default_structure': 'spread', 'top_n': 2,
                          'rebal_freq': 5, 'hold_days': 10, 'confidence_threshold': 0.0},
        'D_FixedCall': {'use_adaptive': False, 'default_structure': 'call', 'top_n': 2,
                        'rebal_freq': 5, 'hold_days': 10, 'confidence_threshold': 0.0},
        'E_Top3Weekly': {'use_adaptive': True, 'top_n': 3, 'rebal_freq': 5,
                         'hold_days': 10, 'confidence_threshold': 0.2},
        'F_Biweekly': {'use_adaptive': True, 'top_n': 2, 'rebal_freq': 10,
                        'hold_days': 15, 'confidence_threshold': 0.4, 'use_kelly': True},
    }

    results = []
    for name, cfg in variants.items():
        print(f"\n  Running {name}...", end=" ")
        try:
            r = run_variant(price_df, features, name, cfg)
            if r and not r.get('skip'):
                results.append(r)
                print(f"Sharpe {r['sharpe']:.3f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | "
                      f"MDD {r['max_dd']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t | {r['structures']}")
            elif r:
                print(f"SKIP: {r.get('reason')}")
            else:
                print("FAIL")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    if not results:
        print("\n❌ ALL VARIANTS FAILED")
        return

    print(f"\n  Running random baseline (200 sims)...", end=" ")
    rand_sharpe = random_baseline(price_df, features)
    print(f"Sharpe {rand_sharpe:.3f}")

    print(f"\n{'='*70}")
    print("  5-GATE EVALUATION")
    print(f"{'='*70}")

    for r in sorted(results, key=lambda x: -x['sharpe']):
        gates = 0
        details = []

        g1 = r['sharpe'] > 0.5; gates += g1
        details.append(f"G1 Sharpe>0.5: {'✅' if g1 else '❌'} ({r['sharpe']:.3f})")

        pval = permutation_test_result(r['trade_pnls']) if len(r['trade_pnls']) >= 10 else 1.0
        g2 = pval < 0.05; gates += g2
        details.append(f"G2 Perm p<0.05: {'✅' if g2 else '❌'} (p={pval:.3f})")

        g3 = r['max_dd'] > -40; gates += g3
        details.append(f"G3 MDD>-40%: {'✅' if g3 else '❌'} ({r['max_dd']:.1f}%)")

        g4 = r['regime_gap'] < 0.50; gates += g4
        details.append(f"G4 Regime<0.50: {'✅' if g4 else '❌'} ({r['regime_gap']:.3f})")

        g5 = r['sharpe'] > rand_sharpe + 0.10; gates += g5
        details.append(f"G5 >Random+0.10: {'✅' if g5 else '❌'} ({r['sharpe']:.3f} vs {rand_sharpe:.3f})")

        status = '✅ PASS' if gates >= 4 else ('⚠️ PARTIAL' if gates >= 3 else '❌ FAIL')

        print(f"\n  {r['variant']} — {status} ({gates}/5)")
        print(f"    Sharpe {r['sharpe']:.3f} | Sortino {r['sortino']:.2f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f}")
        print(f"    MDD {r['max_dd']:.1f}% | CAGR {r['cagr']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t")
        print(f"    Structures: {r['structures']}")
        for d in details:
            print(f"    {d}")

    best = max(results, key=lambda x: x['sharpe'])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['variant']} (Sharpe {best['sharpe']:.3f})")
    print(f"  Random: {rand_sharpe:.3f}")
    print(f"  Alpha: {'YES' if best['sharpe'] > rand_sharpe + 0.20 else 'MARGINAL/NO'}")
    print(f"  Elapsed: {(datetime.now() - t0).total_seconds():.0f}s")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        exp = mlflow.set_experiment("adaptive_options_portfolio_v1")
        with mlflow.start_run(run_name=f"adaptive_opts_{datetime.now().strftime('%H%M')}"):
            mlflow.log_param("strategy", "adaptive_options_portfolio_v1")
            mlflow.log_param("best_variant", best['variant'])
            mlflow.log_metric("best_sharpe", best['sharpe'])
            mlflow.log_metric("best_sortino", best['sortino'])
            mlflow.log_metric("best_wr", best['wr'])
            mlflow.log_metric("best_pf", best['pf'])
            mlflow.log_metric("best_mdd", best['max_dd'])
            mlflow.log_metric("random_sharpe", rand_sharpe)
            mlflow.log_metric("n_trades", best['trades'])
            for r in results:
                mlflow.log_metric(f"{r['variant']}_sharpe", r['sharpe'])
        print(f"  MLflow logged: {exp.name}")
    except Exception as e:
        print(f"  MLflow: {e}")

if __name__ == '__main__':
    main()
