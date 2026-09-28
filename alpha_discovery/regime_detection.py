#!/usr/bin/env python3
"""Regime detection via Gaussian Mixture Model on daily microstructure statistics.
Daily features: event_rate, spread_mean, ofi_imbalance, volatility, cancel_rate, autocorr.
Fits GMM k=2,3,4 - picks best BIC. Output: regime labels per day + timeline.
Key question: do regime labels predict whether models will work?
"""
import logging,json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
log=logging.getLogger()
ROOT=Path('/home/jupiter/Lvl3Quant')
DATA_DIR=ROOT/'data'/'processed'/'mbo_events'
OUT_DIR=ROOT/'alpha_discovery'/'results'/'regime_detection'
OUT_DIR.mkdir(parents=True,exist_ok=True)
DATE_CUTOFF='20251201';MAX_EVENTS=500_000

def daily_stats(fp):
    d=np.load(fp,allow_pickle=True)
    ev=d['events'].astype('f4')
    if len(ev)>MAX_EVENTS:ev=ev[:MAX_EVENTS]
    td=ev[:,0];et=ev[:,1];side=ev[:,2];price=ev[:,3];qty=ev[:,4];sprd=ev[:,5]
    is_t=(et==2);is_c=(et==1);is_a=(et==0)
    is_bid=(side==0);is_ask=(side==1)
    bvol=is_t*is_ask*qty;svol=is_t*is_bid*qty
    ofi=bvol-svol
    n=len(ev)
    dur=float(td[td>0].sum()) if (td>0).any() else 1.0
    event_rate=float(n/max(dur,1.0))
    spread_mean=float(sprd.mean())
    spread_std=float(sprd.std())
    ofi_imbal=float(ofi.mean()/(np.abs(ofi).mean()+1e-8))
    ofi_std=float(ofi.std())
    cancel_rate=float(is_c.mean())
    trade_rate=float(is_t.mean())
    add_rate=float(is_a.mean())
    # Price volatility
    valid_price=price[price>0]
    if len(valid_price)>1:
        pdiffs=np.diff(valid_price)
        vol=float(np.std(pdiffs))
        skew=float(np.mean(pdiffs**3)/(np.std(pdiffs)**3+1e-8))
    else:
        vol=0.0;skew=0.0
    # AR1 of price changes (mean reversion vs momentum)
    pdiff_all=np.zeros(n,'f4');pdiff_all[1:]=price[1:]-price[:-1]
    ar1=float(np.corrcoef(pdiff_all[1:],pdiff_all[:-1])[0,1]) if n>50 else 0.0
    # Buy/sell imbalance
    buy_imbal=float((bvol.sum()-svol.sum())/(bvol.sum()+svol.sum()+1e-8))
    # Spread percentiles (bimodal spread = tick clustering)
    spread_p90=float(np.percentile(sprd,90))
    spread_p10=float(np.percentile(sprd,10))
    spread_range=spread_p90-spread_p10
    # labels if available
    ic_1s=None;ic_10s=None
    if 'labels_1s' in d and len(ev)>100:
        labs=d['labels_1s'].astype('f4')[:MAX_EVENTS]
        # simple feature: std of labels (how much predictability there is)
        ic_1s=float(np.std(labs))
    if 'labels_10s' in d and len(ev)>100:
        labs=d['labels_10s'].astype('f4')[:MAX_EVENTS]
        ic_10s=float(np.std(labs))
    return {'date':fp.stem[:8],'event_rate':event_rate,'spread_mean':spread_mean,
            'spread_std':spread_std,'spread_range':spread_range,
            'ofi_imbal':ofi_imbal,'ofi_std':ofi_std,'cancel_rate':cancel_rate,
            'trade_rate':trade_rate,'add_rate':add_rate,'vol':vol,'skew':skew,
            'ar1':ar1,'buy_imbal':buy_imbal,'label_std_1s':ic_1s,'label_std_10s':ic_10s,
            'n_events':n}

def main():
    # Load all files (not just Dec 2025+ - want full history for regime context)
    all_files=sorted(DATA_DIR.glob('*_mbo_events.npz'))
    cutoff_files=sorted([f for f in all_files if f.stem[:8]>=DATE_CUTOFF])
    log.info(f'All files:{len(all_files)} | Dec2025+:{len(cutoff_files)}')
    # Use full history to build regime model, apply to cutoff window
    log.info('Computing daily stats (full history)...')
    daily=[]
    for i,f in enumerate(all_files):
        try:
            st=daily_stats(f)
            daily.append(st)
            if (i+1)%20==0:log.info(f'  {i+1}/{len(all_files)} done')
        except Exception as e:log.warning(f' skip {f.name}:{e}')
    log.info(f'Daily stats: {len(daily)} days total')
    feat_cols=['event_rate','spread_mean','spread_std','ofi_imbal','cancel_rate',
               'trade_rate','vol','ar1','buy_imbal','spread_range']
    X=np.array([[d[c] for c in feat_cols] for d in daily])
    # Clip extreme outliers
    for j in range(X.shape[1]):
        p1,p99=np.percentile(X[:,j],[1,99])
        X[:,j]=np.clip(X[:,j],p1,p99)
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    # Fit GMM k=2,3,4
    best_bic=np.inf;best_k=3;best_gmm=None
    for k in [2,3,4]:
        gmm=GaussianMixture(n_components=k,covariance_type='full',n_init=10,random_state=42)
        gmm.fit(Xs)
        bic=gmm.bic(Xs)
        aic=gmm.aic(Xs)
        log.info(f'  GMM k={k}: BIC={bic:.1f} AIC={aic:.1f}')
        if bic<best_bic:best_bic=bic;best_k=k;best_gmm=gmm
    log.info(f'Best k={best_k} (BIC={best_bic:.1f})')
    labels=best_gmm.predict(Xs)
    probs=best_gmm.predict_proba(Xs)
    # Analyze each regime
    log.info('\n=== REGIME CHARACTERISTICS ===')
    for r in range(best_k):
        mask=(labels==r)
        n_days=int(mask.sum())
        log.info(f'\nRegime {r}: {n_days} days ({100*mask.mean():.1f}%)')
        regime_days=[d['date'] for i,d in enumerate(daily) if mask[i]]
        log.info(f'  First dates: {regime_days[:5]}')
        log.info(f'  Last dates:  {regime_days[-5:]}')
        for c in feat_cols:
            vals=[daily[i][c] for i in range(len(daily)) if mask[i]]
            log.info(f'  {c:20s}: mean={np.mean(vals):8.4f} std={np.std(vals):.4f}')
        # Post-Dec label std (proxy for predictability)
        dec_dates=[d for d in daily if mask[daily.index(d)] and d['date']>=DATE_CUTOFF]
        if dec_dates:
            ls1=[d['label_std_1s'] for d in dec_dates if d['label_std_1s'] is not None]
            ls10=[d['label_std_10s'] for d in dec_dates if d['label_std_10s'] is not None]
            if ls1:log.info(f'  label_std_1s (Dec2025+ only): mean={np.mean(ls1):.4f}')
            if ls10:log.info(f'  label_std_10s (Dec2025+ only): mean={np.mean(ls10):.4f}')
    # Timeline: show regime transitions
    log.info('\n=== REGIME TIMELINE ===')
    prev=-1
    for i,(d,l) in enumerate(zip(daily,labels)):
        if l!=prev:
            log.info(f'  {d["date"]}: -> REGIME {l}  (vol={d["vol"]:.4f} '
                     f'spread={d["spread_mean"]:.4f} event_rate={d["event_rate"]:.1f} '
                     f'ar1={d["ar1"]:+.3f})')
            prev=l
    # Identify current regime (latest date)
    cur_regime=int(labels[-1])
    log.info(f'\nCURRENT REGIME: {cur_regime} (as of {daily[-1]["date"]})')
    # How has the regime distribution shifted Dec2025 vs earlier?
    pre_dec={r:sum(1 for i,d in enumerate(daily) if labels[i]==r and d['date']<DATE_CUTOFF)
             for r in range(best_k)}
    post_dec={r:sum(1 for i,d in enumerate(daily) if labels[i]==r and d['date']>=DATE_CUTOFF)
              for r in range(best_k)}
    n_pre=sum(pre_dec.values());n_post=sum(post_dec.values())
    log.info('\n=== REGIME SHIFT: Pre vs Post Dec 2025 ===')
    for r in range(best_k):
        pre_pct=100*pre_dec[r]/max(n_pre,1)
        post_pct=100*post_dec[r]/max(n_post,1)
        log.info(f'  Regime {r}: pre={pre_dec[r]:3d}d ({pre_pct:.1f}%) -> post={post_dec[r]:3d}d ({post_pct:.1f}%)')
    # Save
    out_data={'n_regimes':int(best_k),'feat_cols':feat_cols,
              'dates':[d['date'] for d in daily],'regime_labels':labels.tolist(),
              'regime_probs':probs.tolist(),'daily_stats':daily,
              'current_regime':cur_regime,'best_bic':float(best_bic)}
    out=OUT_DIR/f'regimes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    json.dump(out_data,open(out,'w'),indent=2)
    log.info(f'Saved:{out}')

if __name__=='__main__':main()
