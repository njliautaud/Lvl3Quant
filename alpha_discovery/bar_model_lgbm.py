#!/usr/bin/env python3
"""100ms OHLCV bar model LGBM -- baseline comparison to event-driven model.
Converts MBO events -> 100ms OHLCV bars -> LGBM features -> predict labels_10s.
Key question: do bar models have the same regime cliff as event models?
Confidence-stratified IC is MANDATORY per hard rule.
"""
import logging,json
import numpy as np
from pathlib import Path
import lightgbm as lgb
from scipy.stats import spearmanr
from datetime import datetime,timedelta
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
log=logging.getLogger()
ROOT=Path('/home/jupiter/Lvl3Quant')
DATA_DIR=ROOT/'data'/'processed'/'mbo_events'
OUT_DIR=ROOT/'alpha_discovery'/'results'/'bar_model_100ms'
OUT_DIR.mkdir(parents=True,exist_ok=True)
DATE_CUTOFF='20251201';MAX_EVENTS=2_000_000;BAR_MS=100;LOOKBACK=30
TRAIN_DAYS=45;TEST_DAYS=15;HORIZON='labels_10s'

def events_to_bars(ev,labels,bar_ms=100):
    """Vectorized: convert events to OHLCV bars at bar_ms resolution."""
    td=ev[:,0];et=ev[:,1];side=ev[:,2];price=ev[:,3];qty=ev[:,4];sprd=ev[:,5]
    cum_ms=(np.cumsum(td)*1000).astype('f8')
    bar_ids=(cum_ms/bar_ms).astype('i4')
    max_bar=int(bar_ids[-1])+1
    if max_bar<LOOKBACK+10:return None
    is_t=(et==2);is_bid=(side==0);is_ask=(side==1)
    bvol_ev=is_t*is_bid*qty;svol_ev=is_t*is_ask*qty
    n_bar=np.bincount(bar_ids,minlength=max_bar).astype('f4')
    vol_bar=np.bincount(bar_ids,weights=qty,minlength=max_bar).astype('f4')
    bvol_bar=np.bincount(bar_ids,weights=bvol_ev,minlength=max_bar).astype('f4')
    svol_bar=np.bincount(bar_ids,weights=svol_ev,minlength=max_bar).astype('f4')
    sprd_sum=np.bincount(bar_ids,weights=sprd,minlength=max_bar)
    sprd_bar=(sprd_sum/(n_bar+1e-8)).astype('f4')
    # OHLC via sorted reduceat
    sort_idx=np.argsort(bar_ids,kind='stable')
    sb=bar_ids[sort_idx];sp=price[sort_idx]
    ubars,fi=np.unique(sb,return_index=True)
    li=np.append(fi[1:],len(sort_idx))-1
    open_bar=np.zeros(max_bar,'f4');close_bar=np.zeros(max_bar,'f4')
    high_bar=np.zeros(max_bar,'f4');low_bar=np.full(max_bar,np.inf,'f4')
    open_bar[ubars]=sp[fi];close_bar[ubars]=sp[li]
    high_bar[ubars]=np.maximum.reduceat(sp,fi)
    low_bar[ubars]=np.minimum.reduceat(sp,fi)
    low_bar[low_bar==np.inf]=0.0
    # Mean label per bar
    lbl_bar=np.zeros(max_bar,'f4')
    if labels is not None:
        lbl_sum=np.bincount(bar_ids,weights=labels,minlength=max_bar)
        lbl_bar=(lbl_sum/(n_bar+1e-8)).astype('f4')
    bars=np.stack([open_bar,high_bar,low_bar,close_bar,vol_bar,n_bar,
                   bvol_bar,svol_bar,sprd_bar,lbl_bar],axis=1)
    return bars

def bar_features(bars,lookback=30):
    """Roll lookback window over bars to create LGBM features."""
    N=len(bars)
    o=bars[:,0];h=bars[:,1];l=bars[:,2];c=bars[:,3]
    vol=bars[:,4];n=bars[:,5];bvol=bars[:,6];svol=bars[:,7];sprd=bars[:,8]
    lbl=bars[:,9]
    rows=[];lbls=[]
    for i in range(lookback,N):
        wc=c[i-lookback:i];wvol=vol[i-lookback:i]
        wbvol=bvol[i-lookback:i];wsvol=svol[i-lookback:i]
        wsprd=sprd[i-lookback:i];wn=n[i-lookback:i]
        wh=h[i-lookback:i];wl=l[i-lookback:i]
        rets=np.diff(wc)/(wc[:-1]+1e-8) if len(wc)>1 else np.array([0.0])
        ret5=float(wc[-1]-wc[-5])/(float(wc[-5])+1e-8) if len(wc)>=5 else 0.0
        ret10=float(wc[-1]-wc[-10])/(float(wc[-10])+1e-8) if len(wc)>=10 else 0.0
        ret_all=float(wc[-1]-wc[0])/(float(wc[0])+1e-8) if wc[0]>0 else 0.0
        ret_mean=float(rets.mean()) if len(rets)>0 else 0.0
        ret_std=float(rets.std()) if len(rets)>1 else 0.0
        ar1=float(np.corrcoef(rets[1:],rets[:-1])[0,1]) if len(rets)>5 else 0.0
        vol_mean=float(wvol.mean())
        vol_ratio=float(wvol[-1]/(vol_mean+1e-8))
        tot=wbvol.sum()+wsvol.sum()
        buy_imbal=float((wbvol.sum()-wsvol.sum())/(tot+1e-8))
        sprd_mean=float(wsprd.mean())
        sprd_dev=float(wsprd[-1]-sprd_mean)
        n_mean=float(wn.mean())
        full_range=float(wh.max()-wl.min())
        pos_in_range=float(wc[-1]-wl.min())/(full_range+1e-8)
        body_size=float(abs(wc[-1]-o[i-lookback+lookback-1]))/(full_range+1e-8)
        # Higher timeframe: 5-bar OHLCV (coarser view)
        if len(wc)>=5:
            hf_rets=np.diff(wc[-5:])/(wc[-5:-1]+1e-8)
            hf_vol=float(wvol[-5:].mean())
        else:
            hf_rets=np.array([0.0]);hf_vol=0.0
        hf_dir=float(np.sign(hf_rets).mean())
        row=[ret5,ret10,ret_all,ret_mean,ret_std,ar1,
             vol_mean,vol_ratio,buy_imbal,sprd_mean,sprd_dev,
             n_mean,full_range,pos_in_range,body_size,hf_dir,hf_vol]
        rows.append(row)
        lbls.append(float(lbl[i]))
    if not rows:return None,None
    return np.array(rows,'f4'),np.array(lbls,'f4')

def conf_ic(preds,labels):
    """HARD RULE: stratify by confidence."""
    abs_p=np.abs(preds);res={}
    for pct,lbl in [(100,'all'),(50,'top50'),(25,'top25'),(10,'top10')]:
        mask=abs_p>=np.percentile(abs_p,100-pct) if pct<100 else np.ones(len(preds),bool)
        if mask.sum()<50:continue
        ic=float(spearmanr(preds[mask],labels[mask]).correlation)
        da=float(np.mean(np.sign(preds[mask])==np.sign(labels[mask])))
        n=int(mask.sum())
        res[lbl]={'ic':ic,'dir_acc':da,'n':n}
        log.info(f'    [{lbl:6s}] n={n:6d} ({100*n/len(preds):4.1f}%) IC={ic:+.4f} DirAcc={da:.4f}')
    return res

def load_bars(fp):
    d=np.load(fp,allow_pickle=True)
    ev=d['events'].astype('f4')
    if len(ev)>MAX_EVENTS:ev=ev[:MAX_EVENTS]
    labs=d[HORIZON].astype('f4')[:MAX_EVENTS] if HORIZON in d else None
    bars=events_to_bars(ev,labs)
    if bars is None:return None,None
    return bar_features(bars,LOOKBACK)

def main():
    files=sorted([f for f in DATA_DIR.glob('*_mbo_events.npz') if f.stem[:8]>=DATE_CUTOFF])
    log.info(f'Files:{len(files)} {files[0].stem[:8]}..{files[-1].stem[:8]}')
    start=datetime.strptime(files[0].stem[:8],'%Y%m%d')
    end=datetime.strptime(files[-1].stem[:8],'%Y%m%d')
    results={'folds':[],'model':'100ms_bar_lgbm','horizon':HORIZON};fold=0;ts=start
    while True:
        te=ts+timedelta(days=TRAIN_DAYS);xs=te+timedelta(days=1);xe=xs+timedelta(days=TEST_DAYS)
        if xe>end:break
        trs=ts.strftime('%Y%m%d');tre=te.strftime('%Y%m%d')
        xes=xs.strftime('%Y%m%d');xee=xe.strftime('%Y%m%d')
        trf=[f for f in files if trs<=f.stem[:8]<=tre]
        tef=[f for f in files if xes<=f.stem[:8]<=xee]
        if not trf or not tef:ts+=timedelta(days=TEST_DAYS);continue
        log.info(f'\nFOLD {fold:02d} train:{trs}..{tre}({len(trf)}f) test:{xes}..{xee}({len(tef)}f)')
        Xtr=[];ytr=[]
        for f in trf:
            try:
                X,y=load_bars(f)
                if X is not None:Xtr.append(X);ytr.append(y)
            except Exception as e:log.warning(f' skip {f.name}:{e}')
        if not Xtr:ts+=timedelta(days=TEST_DAYS);fold+=1;continue
        Xtr=np.vstack(Xtr);ytr=np.concatenate(ytr)
        log.info(f'  Train bars:{Xtr.shape}')
        Xte=[];yte=[]
        for f in tef:
            try:
                X,y=load_bars(f)
                if X is not None:Xte.append(X);yte.append(y)
            except Exception as e:log.warning(f' skip {f.name}:{e}')
        if not Xte:ts+=timedelta(days=TEST_DAYS);fold+=1;continue
        Xte=np.vstack(Xte);yte=np.concatenate(yte)
        log.info(f'  Test bars:{Xte.shape}')
        m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=63,
                              min_child_samples=50,n_jobs=16,verbose=-1)
        m.fit(Xtr,ytr)
        preds=m.predict(Xte)
        log.info(f'  {HORIZON} confidence-stratified IC:')
        fr={'fold':fold,'train':f'{trs}..{tre}','test':f'{xes}..{xee}',
            'n_train_bars':int(len(Xtr)),'n_test_bars':int(len(Xte))}
        fr['conf_ic']=conf_ic(preds,yte)
        results['folds'].append(fr);ts+=timedelta(days=TEST_DAYS);fold+=1
    log.info('\n=== BAR MODEL SUMMARY (100ms OHLCV) ===')
    for bkt in ['all','top50','top25','top10']:
        ics=[f['conf_ic'][bkt]['ic'] for f in results['folds'] if bkt in f.get('conf_ic',{})]
        das=[f['conf_ic'][bkt]['dir_acc'] for f in results['folds'] if bkt in f.get('conf_ic',{})]
        if ics:
            log.info(f'  {HORIZON} [{bkt:6s}] avg_IC={np.mean(ics):+.4f} '
                     f'avg_DirAcc={np.mean(das):.4f} folds={[round(x,4) for x in ics]}')
    out=OUT_DIR/f'bar_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    json.dump(results,open(out,'w'),indent=2);log.info(f'Saved:{out}')

if __name__=='__main__':main()
