#!/usr/bin/env python3
"""LGBM walkforward with confidence-stratified IC.
HARD RULE: report IC + directional accuracy by |pred| percentile (all/top50/top25/top10).
We only trade high-confidence signals -- full-distribution IC is noise floor, not alpha.
v2: NaN label filtering added.
"""
import os,logging,json
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import lightgbm as lgb
from datetime import datetime,timedelta
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
log=logging.getLogger()
ROOT=Path('/home/jupiter/Lvl3Quant')
DATA_DIR=ROOT/'data'/'processed'/'mbo_events'
LABEL_DIR=ROOT/'data'/'processed'/'mbo_events_smart_v3'  # Labels stored here since raw files were regenerated
OUT_DIR=ROOT/'alpha_discovery'/'results'/'lgbm_confidence_ic'
OUT_DIR.mkdir(parents=True,exist_ok=True)
DATE_CUTOFF='20251201';MAX_EVENTS=1_500_000;TRAIN_DAYS=45;TEST_DAYS=15
HORIZONS=['labels_1s','labels_10s']
FEAT=['time_delta','event_type','side','price','qty','spread','rolling_ofi_100',
      'cancel_side_asym_100','event_density_50','price_mom_20','qty_price_mom_20',
      'cum_delta','roll_delta_500','ofi_short_20','cancel_rate_100','trade_rate_50',
      'add_side_asym_100','spread_velocity_50','qty_add_imbalance_100',
      'price_sign_mom_200','fill_recovery_20']

def derived(ev):
    N=len(ev);td=ev[:,0].astype('f4');et=ev[:,1].astype('f4');side=ev[:,2].astype('f4')
    price=ev[:,3].astype('f4');qty=ev[:,4].astype('f4');sprd=ev[:,5].astype('f4')
    w20=np.ones(20,'f4')/20;w50=np.ones(50,'f4')/50;w100=np.ones(100,'f4')/100
    w200=np.ones(200,'f4')/200;w500=np.ones(500,'f4')/500
    # CAUSAL: mode='full'[:N] — no look-forward (HC audit 2026-07-01)
    def cc(a,k): return np.convolve(a,k,mode='full')[:N]
    it=(et==2).astype('f4');ic2=(et==1).astype('f4');ia=(et==0).astype('f4')
    ibid=(side==0).astype('f4');iask=(side==1).astype('f4')
    ofi=it*iask*qty-it*ibid*qty
    rofi=cc(ofi,w100)
    casym=cc(ic2*ibid-ic2*iask,w100)
    dens=cc(np.where(td>1e-9,1./(td+1e-6),1.).astype('f4'),w50)
    pdiff=np.zeros(N,'f4');pdiff[20:]=price[20:]-price[:-20]
    pmom=cc(pdiff,w20)
    qpmom=cc(qty*np.sign(pdiff),w20)
    cr=np.cumsum(ofi).astype('f4');crma=cc(cr,w500)
    crstd=np.sqrt(np.maximum(cc(cr**2,w500)-crma**2,1e-8))
    cd=(cr-crma)/(crstd+1e-6)
    rd5=cc(ofi,w500);os20=cc(ofi,w20)
    crate=cc(ic2,w100);trate=cc(it,w50)
    aasym=cc(ia*ibid-ia*iask,w100)
    sdiff=np.zeros(N,'f4');sdiff[1:]=sprd[1:]-sprd[:-1]
    svel=cc(sdiff,w50)
    baq=ia*ibid*qty;aaq=ia*iask*qty
    qai=cc(baq-aaq,w100)/(cc(baq+aaq,w100)+1e-6)
    psm=cc(np.sign(pdiff),w200)
    br=cc(it*ibid,w20)*ia*ibid
    ar=cc(it*iask,w20)*ia*iask
    fr=cc(br+ar,w20)
    d=np.stack([rofi,casym,dens,pmom,qpmom,cd,rd5,os20,crate,trate,aasym,svel,qai,psm,fr],axis=1)
    return np.concatenate([ev,d],axis=1).astype('f4')

def load_file(fp):
    d=np.load(fp,allow_pickle=True);ev=d['events'].astype('f4')
    N_raw=len(ev)
    if N_raw>MAX_EVENTS:N_raw=MAX_EVENTS;ev=ev[:N_raw]
    # Try labels from this file first; if missing, load from smart_v3 companion
    labs={}
    for h in HORIZONS:
        if h in d:
            labs[h]=d[h].astype('f4')[:N_raw]
    if not labs:
        date_str=fp.stem[:8]
        label_fp=LABEL_DIR/f'{date_str}_mbo_events.npz'
        if label_fp.exists():
            ld=np.load(label_fp,allow_pickle=True)
            for h in HORIZONS:
                if h in ld:
                    labs[h]=ld[h].astype('f4')[:N_raw]
            ld.close()
    # Align lengths: labels may have fewer rows than raw events
    if labs:
        min_n=min(N_raw,min(len(v) for v in labs.values()))
        ev=ev[:min_n]
        labs={h:v[:min_n] for h,v in labs.items()}
    ev=derived(ev)
    d.close()
    return ev,labs

def conf_ic(preds,labels,tag):
    """HARD RULE: always stratify IC and directional accuracy by confidence."""
    abs_p=np.abs(preds);res={}
    for pct,lbl in [(100,'all'),(50,'top50'),(25,'top25'),(10,'top10')]:
        mask=abs_p>=np.percentile(abs_p,100-pct) if pct<100 else np.ones(len(preds),bool)
        if mask.sum()<50:continue
        ic=float(spearmanr(preds[mask],labels[mask]).correlation)
        nz = mask & (labels != 0)
        da = float(np.mean(np.sign(preds[nz])==np.sign(labels[nz]))) if nz.sum() > 10 else float('nan')
        lm_nz = nz & (preds > 0); sm_nz = nz & (preds < 0)
        dl = float(np.mean(np.sign(preds[lm_nz])==np.sign(labels[lm_nz]))) if lm_nz.sum() > 10 else float('nan')
        ds = float(np.mean(np.sign(preds[sm_nz])==np.sign(labels[sm_nz]))) if sm_nz.sum() > 10 else float('nan')
        n=int(mask.sum()); n_nz=int(nz.sum())
        res[lbl]={'ic':ic,'dir_acc':da,'dir_long':dl,'dir_short':ds,'n':n,'n_nonzero':n_nz,'pct':float(100*n/len(preds))}
        log.info(f'    {tag} [{lbl:6s}] n={n:7d}(nz={n_nz}) ({100*n/len(preds):4.1f}%) IC={ic:+.4f} DirAcc={da:.4f} Long={dl:.4f} Short={ds:.4f}')
    return res

def main():
    files=sorted([f for f in DATA_DIR.glob('*.npz') if f.stem[:8]>=DATE_CUTOFF and len(f.stem)==8])
    log.info(f'Files:{len(files)} {files[0].stem[:8]}..{files[-1].stem[:8]}')
    start=datetime.strptime(files[0].stem[:8],'%Y%m%d')
    end=datetime.strptime(files[-1].stem[:8],'%Y%m%d')
    results={'folds':[]};fold=0;ts=start
    while True:
        te=ts+timedelta(days=TRAIN_DAYS);xs=te+timedelta(days=1);xe=xs+timedelta(days=TEST_DAYS)
        if xe>end:break
        trs=ts.strftime('%Y%m%d');tre=te.strftime('%Y%m%d')
        xes=xs.strftime('%Y%m%d');xee=xe.strftime('%Y%m%d')
        trf=[f for f in files if trs<=f.stem[:8]<=tre]
        tef=[f for f in files if xes<=f.stem[:8]<=xee]
        if not trf or not tef:ts+=timedelta(days=TEST_DAYS);continue
        log.info(f'\nFOLD {fold:02d} train:{trs}..{tre}({len(trf)}f) test:{xes}..{xee}({len(tef)}f)')
        Xtr=[];ytr={h:[] for h in HORIZONS}
        for f in trf:
            try:
                X,lb=load_file(f);Xtr.append(X)
                for h in HORIZONS:
                    if h in lb:ytr[h].append(lb[h])
            except Exception as e:log.warning(f' skip {f.name}:{e}')
        if not Xtr:ts+=timedelta(days=TEST_DAYS);fold+=1;continue
        Xtr=np.vstack(Xtr);log.info(f'  Train:{Xtr.shape}')
        Xte=[];yte={h:[] for h in HORIZONS}
        for f in tef:
            try:
                X,lb=load_file(f);Xte.append(X)
                for h in HORIZONS:
                    if h in lb:yte[h].append(lb[h])
            except Exception as e:log.warning(f' skip {f.name}:{e}')
        if not Xte:ts+=timedelta(days=TEST_DAYS);fold+=1;continue
        Xte=np.vstack(Xte);log.info(f'  Test:{Xte.shape}')
        fr={'fold':fold,'train':f'{trs}..{tre}','test':f'{xes}..{xee}','conf_ic':{}}
        for h in HORIZONS:
            if not ytr[h] or not yte[h]:continue
            Ytr=np.concatenate(ytr[h]);Yte=np.concatenate(yte[h])
            # Filter NaN labels
            valid_tr=np.isfinite(Ytr)
            if not valid_tr.all():
                log.warning(f'  {h}: dropping {(~valid_tr).sum()} NaN train labels')
                Xtr_fit=Xtr[valid_tr];Ytr=Ytr[valid_tr]
            else:
                Xtr_fit=Xtr
            valid_te=np.isfinite(Yte)
            if not valid_te.all():
                log.warning(f'  {h}: dropping {(~valid_te).sum()} NaN test labels')
                Xte_fit=Xte[valid_te];Yte=Yte[valid_te]
            else:
                Xte_fit=Xte
            if len(Ytr)<1000 or len(Yte)<100:
                log.warning(f'  {h}: too few samples after NaN drop, skip');continue
            m=lgb.LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=63,
                                  min_child_samples=50,n_jobs=16,verbose=-1)
            m.fit(Xtr_fit,Ytr,feature_name=FEAT)
            preds=m.predict(Xte_fit)
            log.info(f'  {h}:')
            fr['conf_ic'][h]=conf_ic(preds,Yte,h)
            np.savez_compressed(str(OUT_DIR/f'fold{fold:02d}_{h}.npz'),
                                preds=preds.astype('f4'),labels=Yte.astype('f4'))
        results['folds'].append(fr);ts+=timedelta(days=TEST_DAYS);fold+=1
    log.info('\n=== CONFIDENCE-STRATIFIED IC SUMMARY ===')
    for h in HORIZONS:
        for bkt in ['all','top50','top25','top10']:
            ics=[f['conf_ic'][h][bkt]['ic'] for f in results['folds']
                 if h in f.get('conf_ic',{}) and bkt in f['conf_ic'][h]]
            das=[f['conf_ic'][h][bkt]['dir_acc'] for f in results['folds']
                 if h in f.get('conf_ic',{}) and bkt in f['conf_ic'][h]]
            if ics:
                log.info(f'  {h} [{bkt:6s}] avg_IC={np.mean(ics):+.4f} avg_DirAcc={np.mean(das):.4f} '
                         f'folds={[round(x,4) for x in ics]}')
    out=OUT_DIR/f'confidence_ic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    json.dump(results,open(out,'w'),indent=2);log.info(f'Saved:{out}')

if __name__=='__main__':main()
