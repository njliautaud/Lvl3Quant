import os,sys,json,numpy as np
from datetime import datetime,timezone

JDATA='/home/jupiter/Lvl3Quant/data'
EVENTS_DIR=JDATA+'/processed/mbo_events'
OUTPUT_PATH=JDATA+'/fillsim_cnn_s76_mbo_results.json'
FOLD_FILES={0:JDATA+'/cnn_s76_fold_00_oot.npz',1:JDATA+'/cnn_s76_fold_01_oot_predictions.npz',2:JDATA+'/cnn_s76_fold_02_oot_predictions.npz',3:JDATA+'/cnn_s76_fold_03_oot_predictions.npz',4:JDATA+'/cnn_s76_fold_04_oot_predictions.npz'}
WINDOW_SIZE=500
Z_THRESHOLD=2.5
TP_TICKS=2.0
SL_TICKS=1.0
ORDER_TYPE='limit'
CANCEL_AFTER_NS=30_000_000_000
RTH_START=14*3600*1_000_000_000+30*60*1_000_000_000
RTH_END=21*3600*1_000_000_000

def day_start_ns(ds):
    y,m,d=int(ds[:4]),int(ds[4:6]),int(ds[6:8])
    return int(datetime(y,m,d,tzinfo=timezone.utc).timestamp())*1_000_000_000

def date_from_fname(fn):
    for p in fn.split('_'):
        if len(p)==8 and p.isdigit(): return p
    raise ValueError(fn)

def infer_stride(n_ev,n_pr,win):
    for s in [1,50,100,200,250,500,1000]:
        if n_pr and (n_ev-win)//s+1==n_pr: return s
    return max(1,(n_ev-win)//(n_pr-1)) if n_pr else 250

def load_fold(fid,fpath):
    fold=np.load(fpath,allow_pickle=True)
    keys=list(fold.keys())
    if 'predictions' in keys:
        preds=fold['predictions']; pcol=2
    elif 'preds_10s' in keys:
        preds=fold['preds_10s'].reshape(-1,1); pcol=0
    else:
        raise KeyError('no pred key in fold '+str(fid)+': '+str(keys))
    oot_files=list(fold.get('oot_files',[]))
    print('  Fold '+str(fid)+': '+str(len(preds))+' preds, '+str(len(oot_files))+' oot files')
    signals=[]
    offset=0
    for evpath_raw in oot_files:
        evfname=str(evpath_raw).replace(chr(92),chr(47)).split(chr(47))[-1]
        evlocal=os.path.join(EVENTS_DIR,evfname)
        if not os.path.exists(evlocal):
            print('  WARN not found: '+evlocal); continue
        ev=np.load(evlocal,allow_pickle=True)
        ts_all=ev['timestamps']
        n_ev=len(ts_all)
        n_pr_file=len(preds)-offset if len(oot_files)==1 else None
        stride=infer_stride(n_ev,n_pr_file,WINDOW_SIZE)
        n_pr_file=min((n_ev-WINDOW_SIZE)//stride+1,len(preds)-offset)
        print('    '+evfname+': n_ev='+str(n_ev)+' stride='+str(stride)+' n_pr='+str(n_pr_file))
        pth=preds[offset:offset+n_pr_file,pcol]
        mu=float(np.mean(pth)); sig=float(np.std(pth))
        if sig<1e-9: offset+=n_pr_file; continue
        z=(pth-mu)/sig
        idx=np.arange(int(n_pr_file))*stride+WINDOW_SIZE-1
        idx=np.clip(idx,0,n_ev-1)
        pts=ts_all[idx]
        ds=date_from_fname(evfname)
        dns=day_start_ns(ds)
        tsoff=pts-dns
        rth=(tsoff>=RTH_START)&(tsoff<=RTH_END)
        lm=(z>Z_THRESHOLD)&rth; sm=(z<-Z_THRESHOLD)&rth
        nl=int(lm.sum()); ns=int(sm.sum())
        print('    z>'+str(Z_THRESHOLD)+': '+str(nl)+' long '+str(ns)+' short')
        for ts in pts[lm]: signals.append((ds,int(ts),'long'))
        for ts in pts[sm]: signals.append((ds,int(ts),'short'))
        offset+=n_pr_file
    return signals

def agg_stats(pnl,lab):
    if not pnl: return {}
    p=np.array(pnl); neg=p[p<0]
    sor=float(p.mean()/neg.std()) if len(neg)>1 else float('inf')
    return {'label':lab,'n':len(p),'mean_pnl':float(p.mean()),'std':float(p.std()),
            'sortino':sor,'win_rate':float((p>0).mean()),
            'total_ticks':float(p.sum()),'total_usd':float(p.sum()*12.5)}

def run_batch():
    sys.path.insert(0,'/home/jupiter/Lvl3Quant/alpha_discovery/deep_models')
    from mbo_replay_server import run_fill_sim
    all_sig=[]
    for fid,fp in FOLD_FILES.items():
        if not os.path.exists(fp): print('Fold '+str(fid)+' not found'); continue
        print('\n=== Fold '+str(fid)+' ===')
        all_sig.extend(load_fold(fid,fp))
    if not all_sig: print('No signals'); return
    by_date={}
    for ds,ts,dr in all_sig: by_date.setdefault(ds,[]).append((ts,dr))
    print('\nSIGNALS: '+str(len(all_sig))+' total')
    for d in sorted(by_date):
        nl=sum(1 for _,dr in by_date[d] if dr=='long')
        print('  '+d+': '+str(len(by_date[d]))+' ('+str(nl)+'L/'+str(len(by_date[d])-nl)+'S)')
    agg_l,agg_s,agg_c=[],[],[]
    day_res={}
    for ds in sorted(by_date):
        sigs=sorted(by_date[ds],key=lambda x:x[0])
        tsl=[s[0] for s in sigs]; dl=[s[1] for s in sigs]
        print('\n--- '+ds+' ('+str(len(tsl))+' sigs) ---')
        try:
            r=run_fill_sim(date=ds,instrument_id=42140878,signal_ts_ns=tsl,directions=dl,
                          tp_ticks=TP_TICKS,sl_ticks=SL_TICKS,order_type=ORDER_TYPE,
                          cancel_after_ns=CANCEL_AFTER_NS)
            day_res[ds]={'long':r['long'],'short':r['short'],'combined':r['combined']}
            L=r['long'];S=r['short'];C=r['combined']
            print('  L:'+str(L.get('n',0))+'/'+str(L.get('n_signals',0))+' pnl='+str(round(L.get('mean_pnl_net_ticks',0),3))+'t wr='+str(round(L.get('win_rate',0)*100,1))+'%')
            print('  S:'+str(S.get('n',0))+'/'+str(S.get('n_signals',0))+' pnl='+str(round(S.get('mean_pnl_net_ticks',0),3))+'t wr='+str(round(S.get('win_rate',0)*100,1))+'%')
            print('  C:'+str(C.get('n',0))+'/'+str(C.get('n_signals',0))+' pnl='+str(round(C.get('mean_pnl_net_ticks',0),3))+'t sortino='+str(round(C.get('sortino',0),3)))
            agg_l.extend([x.pnl_ticks_net for x in r.get('raw_results',[]) if x.direction=='long'])
            agg_s.extend([x.pnl_ticks_net for x in r.get('raw_results',[]) if x.direction=='short'])
            agg_c.extend([x.pnl_ticks_net for x in r.get('raw_results',[])])
        except Exception as e:
            import traceback; traceback.print_exc()
            day_res[ds]={'error':str(e)}
    out={'config':{'model':'CNN s76','z':Z_THRESHOLD,'tp':TP_TICKS,'sl':SL_TICKS,'commission_rt':4.70,'commission_ticks':0.376,'fill_sim':'MBO-v2'},
         'aggregate':{'long':agg_stats(agg_l,'LONG'),'short':agg_stats(agg_s,'SHORT'),'combined':agg_stats(agg_c,'COMBINED')},
         'by_day':day_res}
    print('\n=== AGGREGATE ===')
    for k in ['long','short','combined']:
        s=out['aggregate'][k]
        if not s: continue
        print(k.upper()+': n='+str(s.get('n',0))+' mean='+str(round(s.get('mean_pnl',0),4))+'t sortino='+str(round(s.get('sortino',0),4))+' wr='+str(round(s.get('win_rate',0)*100,1))+'% total=$'+str(round(s.get('total_usd',0),2)))
    with open(OUTPUT_PATH,'w') as f: json.dump(out,f,indent=2,default=str)
    print('Results -> '+OUTPUT_PATH)

if __name__=='__main__': run_batch()
