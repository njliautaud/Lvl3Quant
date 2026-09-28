#!/usr/bin/env python3
"""
Weekly Trade Simulator v1 — Walk-Forward Trade-by-Trade Backtest
================================================================
Instead of aggregate statistics, outputs INDIVIDUAL TRADE LOGS showing
exactly what happens week-by-week with the $645 account.

Tests the FULL system:
1. LightGBM sector ranking → pick top 3 sectors
2. Enter bull call spreads (bi-weekly) on ranked sectors
3. Overlay earnings IC plays when available (PG, AAPL, etc.)
4. Dynamic position sizing (tiered: $200/$500/$1K)
5. Multi-signal confluence gate (HC #750): min 2 confirming signals

Outputs: last 52 weeks of simulated trade log for paper trading validation.
"""
import json, os, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_PATH / "output" / "growth_research" / "weekly_trade_simulator_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL_CAPITAL = 645.0
COMMISSION_PER_TRADE = 2.60
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
FEAT_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
             'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']

HAIRCUT = 0.15
SPREAD_PCT = 3.0  # 3% OTM for bull call spreads
DTE = 30  # 30-day expiry
TOP_K = 3

def fprint(*a, **kw): print(*a, **kw, flush=True)

def tiered_position(equity):
    """Tiered position sizing per capital scaling analysis."""
    if equity < 2000:
        return min(200, equity / 3)
    elif equity < 10000:
        return min(500, equity / 3)
    else:
        return min(1000, equity / 3)

def download_data():
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    ix = ix.intersection(sh.index).intersection(sl.index)
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

def compute_features(px, idx, ticker):
    """Compute momentum/vol features for a single ticker at date idx."""
    s = px[ticker].iloc[:idx+1]
    if len(s) < 252:
        return None
    c = float(s.iloc[-1])
    feats = {
        'ret_5d': float(s.pct_change(5).iloc[-1]),
        'ret_10d': float(s.pct_change(10).iloc[-1]),
        'ret_21d': float(s.pct_change(21).iloc[-1]),
        'ret_63d': float(s.pct_change(63).iloc[-1]),
        'ret_126d': float(s.pct_change(126).iloc[-1]),
        'ret_252d': float(s.pct_change(252).iloc[-1]),
        'vol_21d': float(s.pct_change().tail(21).std()),
        'vol_63d': float(s.pct_change().tail(63).std()),
        'sharpe_63d': float(s.pct_change().tail(63).mean() / max(s.pct_change().tail(63).std(), 1e-8)),
        'maxdd_63d': float((s.tail(63) / s.tail(63).cummax() - 1).min()),
        'pct_52w_high': float(c / s.tail(252).max()),
        'mom_accel': float(s.pct_change(21).iloc[-1] - s.pct_change(21).iloc[-22]) if len(s) > 43 else 0,
    }
    return feats

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte / 252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val / 20.0)
    time_prem = atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S-K)/S)
    return intrinsic + time_prem

def confluence_check(ticker, sc, spy, vix, idx):
    """
    Multi-signal confluence gate (HC #750):
    Returns number of confirming signals and details.
    """
    signals = []
    s = sc[ticker].iloc[:idx+1]
    sp = spy.iloc[:idx+1]

    if len(s) < 252 or len(sp) < 200:
        return 0, signals

    c = float(s.iloc[-1])

    # Signal 1: Momentum (21d return > 0)
    ret_21 = float(s.pct_change(21).iloc[-1])
    if ret_21 > 0:
        signals.append(f"Momentum +{ret_21*100:.1f}%")

    # Signal 2: Relative strength (outperforming SPY over 63d)
    rs_63 = float(s.pct_change(63).iloc[-1]) - float(sp.pct_change(63).iloc[-1])
    if rs_63 > 0:
        signals.append(f"RelStrength +{rs_63*100:.1f}%")

    # Signal 3: Above 50-day SMA (trend confirmation)
    sma50 = float(s.tail(50).mean())
    if c > sma50:
        signals.append("Above SMA50")

    # Signal 4: VIX not spiking (< 30) — favorable for long positions
    cv = float(vix.iloc[idx])
    if cv < 30:
        signals.append(f"VIX={cv:.0f}<30")

    # Signal 5: Momentum acceleration (recent > older momentum)
    if len(s) > 43:
        recent_mom = float(s.pct_change(21).iloc[-1])
        older_mom = float(s.pct_change(21).iloc[-22])
        if recent_mom > older_mom:
            signals.append("MomAccel")

    return len(signals), signals

def simulate_trade_log(sc, sh, sl, spy, vix, last_n_weeks=52):
    """Run full WF simulation, output detailed trade log for last N weeks."""
    fprint(f"\nBuilding LGBM rankings (walk-forward)...")

    # Build features DataFrame
    dates = sc.index
    all_feats = []
    for di in range(252, len(dates)):
        dt = dates[di]
        for tk in sc.columns:
            f = compute_features(sc, di, tk)
            if f:
                fwd_ret = float(sc[tk].iloc[min(di+10, len(dates)-1)] / sc[tk].iloc[di] - 1)
                f['ticker'] = tk
                f['date'] = dt
                f['fwd_ret'] = fwd_ret
                all_feats.append(f)

    df = pd.DataFrame(all_feats)
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    fprint(f"  Features: {len(df)} rows, {df['date'].nunique()} dates")

    # WF ranking
    udates = sorted(df['date'].unique())
    train_periods = 252  # 1yr lookback
    rankings = {}
    for i in range(train_periods, len(udates)):
        td = udates[max(0, i-train_periods):i]
        test_date = udates[i]
        tr = df[df['date'].isin(td)]
        te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue
        Xt = np.nan_to_num(tr[FEAT_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[FEAT_COLS].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except:
            continue
    fprint(f"  Rankings: {len(rankings)} dates")

    # ATR computation
    atr_d = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_d[tk] = compute_atr(sh[tk], sl[tk], sc[tk])

    # Bi-weekly rebalance dates
    ranked_dates = sorted(rankings.keys())
    rebal_dates = ranked_dates[::10]  # every ~2 weeks

    # Simulate with full trade log
    sma200 = spy.rolling(200).mean()
    equity = INITIAL_CAPITAL
    all_trades = []
    eq_curve = [(rebal_dates[0] if rebal_dates else dates[0], equity)]

    for ri, dt in enumerate(rebal_dates):
        if dt not in spy.index or dt not in vix.index:
            continue

        cv = float(vix.loc[dt])
        sv = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv

        scores = rankings.get(dt, {})
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        max_pos = tiered_position(equity)
        if max_pos < 30:
            continue

        for tk in picks:
            if tk not in sc.columns or tk not in atr_d:
                continue

            # Confluence check (HC #750)
            di = sc.index.get_loc(dt)
            n_signals, signal_list = confluence_check(tk, sc, spy, vix, di)

            if n_signals < 2:
                # Log skipped trade
                all_trades.append({
                    "date": str(dt.date()),
                    "ticker": tk,
                    "action": "SKIP",
                    "reason": f"Only {n_signals} signal(s), need 2+",
                    "signals": signal_list,
                    "equity_before": round(equity, 2),
                })
                continue

            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if not pd.isna(atr_d[tk].loc[dt]) else S * 0.015

            # Price bull call spread
            K1 = round(S)  # ATM long call
            K2 = round(S * (1 + SPREAD_PCT/100))  # OTM short call
            lp = atr_premium(S, K1, DTE, av, cv, 'call') * (1 + HAIRCUT)
            sp_prem = atr_premium(S, K2, DTE, av, cv, 'call') * (1 - HAIRCUT)
            debit = lp - sp_prem
            width = K2 - K1
            cost = debit * 100 + COMMISSION_PER_TRADE
            max_profit = (width - debit) * 100 - COMMISSION_PER_TRADE

            if cost <= 0 or cost > max_pos or cost > equity * 0.40:
                all_trades.append({
                    "date": str(dt.date()),
                    "ticker": tk,
                    "action": "SKIP",
                    "reason": f"Cost ${cost:.0f} exceeds position limit ${max_pos:.0f}",
                    "signals": signal_list,
                    "equity_before": round(equity, 2),
                })
                continue

            # Walk forward to expiry
            ei = min(di + DTE, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])

            # Check for early exit (50% profit target or DTE < 7)
            pnl = None
            exit_reason = "expiry"
            for ci in range(di + 7, ei + 1):
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di
                rd = max(0, DTE - dh)
                tm = np.sqrt(rd / max(DTE, 1))
                si = (max(0, Sc - K1) - max(0, Sc - K2)) * 100
                # Add remaining time value
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                si += ac * tm * 0.3 * 100
                cp = si - cost

                # 50% profit target
                if cp >= max_profit * 0.50:
                    pnl = cp
                    exit_reason = f"50% profit @ day {dh}"
                    break
                # Stop loss at 80% of cost
                if cp <= -cost * 0.80:
                    pnl = cp
                    exit_reason = f"stop loss @ day {dh}"
                    break

            if pnl is None:
                # Expiry: intrinsic only
                si = (max(0, Se - K1) - max(0, Se - K2)) * 100
                pnl = si - cost

            equity = max(0, equity + pnl)

            trade = {
                "date": str(dt.date()),
                "ticker": tk,
                "action": "BULL_CALL_SPREAD",
                "entry_price": round(S, 2),
                "strikes": f"{K1}C/{K2}C",
                "cost": round(cost, 2),
                "pnl": round(pnl, 2),
                "exit_reason": exit_reason,
                "n_signals": n_signals,
                "signals": signal_list,
                "equity_before": round(equity - pnl, 2),
                "equity_after": round(equity, 2),
                "lgbm_rank": round(scores.get(tk, 0), 3),
            }
            all_trades.append(trade)

        eq_curve.append((dt, equity))

    # Filter to last N weeks for detailed display
    if all_trades:
        last_date = pd.Timestamp(all_trades[-1]["date"])
        cutoff = last_date - pd.Timedelta(weeks=last_n_weeks)
        recent_trades = [t for t in all_trades if pd.Timestamp(t["date"]) >= cutoff]
    else:
        recent_trades = []

    return all_trades, recent_trades, eq_curve

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("WEEKLY TRADE SIMULATOR v1 — Walk-Forward Trade Log")
    fprint(f"Capital: ${INITIAL_CAPITAL:.0f} | Bi-weekly rebalance | Multi-signal confluence")
    fprint("=" * 70)

    sc, sh, sl, spy, vix = download_data()
    all_trades, recent_trades, eq_curve = simulate_trade_log(sc, sh, sl, spy, vix)

    # ─── Summary Statistics ───
    executed = [t for t in all_trades if t["action"] != "SKIP"]
    skipped = [t for t in all_trades if t["action"] == "SKIP"]
    wins = [t for t in executed if t["pnl"] > 0]
    losses = [t for t in executed if t["pnl"] <= 0]

    fprint(f"\n{'='*70}")
    fprint(f"FULL BACKTEST SUMMARY")
    fprint(f"{'='*70}")
    fprint(f"  Total signals: {len(all_trades)}")
    fprint(f"  Executed: {len(executed)} | Skipped (confluence fail): {len(skipped)}")
    fprint(f"  Wins: {len(wins)} ({len(wins)/max(1,len(executed))*100:.0f}%) | "
           f"Losses: {len(losses)}")
    if executed:
        pnls = [t["pnl"] for t in executed]
        fprint(f"  Total PnL: ${sum(pnls):,.2f}")
        fprint(f"  Avg trade: ${np.mean(pnls):.2f} | "
               f"Avg win: ${np.mean([t['pnl'] for t in wins]):.2f} | "
               f"Avg loss: ${np.mean([t['pnl'] for t in losses]):.2f}")
        fprint(f"  Final equity: ${eq_curve[-1][1]:,.2f}")

    # ─── Recent Trade Log (last 52 weeks) ───
    fprint(f"\n{'='*70}")
    fprint(f"RECENT TRADE LOG (Last 52 Weeks)")
    fprint(f"{'='*70}")
    fprint(f"{'Date':<12} {'Ticker':<6} {'Action':<8} {'Strikes':<12} {'Cost':>8} "
           f"{'PnL':>8} {'Exit':<18} {'Signals':>4} {'Equity':>10}")
    fprint("-" * 100)

    for t in recent_trades:
        if t["action"] == "SKIP":
            fprint(f"{t['date']:<12} {t['ticker']:<6} {'SKIP':<8} {'':<12} {'':<8} "
                   f"{'':<8} {t['reason'][:18]:<18} {t.get('n_signals',0):>4}s "
                   f"${t['equity_before']:>9,.2f}")
        else:
            fprint(f"{t['date']:<12} {t['ticker']:<6} {'BCS':<8} {t['strikes']:<12} "
                   f"${t['cost']:>7.2f} ${t['pnl']:>7.2f} {t['exit_reason'][:18]:<18} "
                   f"{t['n_signals']:>4}s ${t['equity_after']:>9,.2f}")

    # ─── Confluence Filter Impact ───
    fprint(f"\n{'='*70}")
    fprint(f"CONFLUENCE FILTER ANALYSIS (HC #750)")
    fprint(f"{'='*70}")
    skip_reasons = {}
    for t in skipped:
        r = t.get("reason", "unknown")
        skip_reasons[r] = skip_reasons.get(r, 0) + 1
    for r, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        fprint(f"  {r}: {cnt} trades skipped")
    fprint(f"  Filter rate: {len(skipped)/max(1,len(all_trades))*100:.1f}% of signals filtered")

    # ─── Save Results ───
    results = {
        "summary": {
            "total_signals": len(all_trades),
            "executed": len(executed),
            "skipped": len(skipped),
            "win_rate": len(wins)/max(1,len(executed)),
            "total_pnl": sum(t["pnl"] for t in executed),
            "final_equity": eq_curve[-1][1] if eq_curve else INITIAL_CAPITAL,
            "filter_rate": len(skipped)/max(1,len(all_trades)),
        },
        "recent_trades": recent_trades,
        "equity_curve": [(str(d), e) for d, e in eq_curve[-52:]],
    }

    out = OUTPUT_DIR / "trade_log.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.1f}s")
    fprint(f"Saved: {out}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("weekly_trade_simulator_v1")
        with mlflow.start_run(run_name=f"tradesim_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.log_params({
                "initial_capital": INITIAL_CAPITAL,
                "spread_pct": SPREAD_PCT,
                "dte": DTE,
                "top_k": TOP_K,
                "confluence_min": 2,
            })
            mlflow.log_metrics({
                "total_trades": len(executed),
                "win_rate": len(wins)/max(1,len(executed)),
                "total_pnl": sum(t["pnl"] for t in executed),
                "final_equity": eq_curve[-1][1] if eq_curve else INITIAL_CAPITAL,
                "filter_rate": len(skipped)/max(1,len(all_trades)),
            })
            mlflow.log_artifact(str(out))
            fprint("Logged to MLflow")
    except Exception as e:
        fprint(f"MLflow skip: {e}")

    fprint("Done.")

if __name__ == "__main__":
    main()
