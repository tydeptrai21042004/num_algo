#!/usr/bin/env python3
"""Aggregate measured results; distinguish attempted, qualified and paired cases."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.common import *
import pandas as pd

def cluster_ci(df,value,seed_value=2026):
    v=df.groupby('host')[value].mean().dropna().to_numpy();n=len(v)
    if n<2:return (float(v.mean()) if n else None,None,None,n)
    rng=np.random.default_rng(seed_value);s=np.mean(rng.choice(v,size=(2000,n),replace=True),axis=1);return float(v.mean()),float(np.percentile(s,2.5)),float(np.percentile(s,97.5)),n

def summarize(folder,out):
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for path in folder.rglob('quality.csv'):
        q=pd.read_csv(path);a=pd.read_csv(path.with_name('attacks.csv'));rows=[];pairs=[];fams=[]
        for col in ('ber','nc01'):
            if col not in a:a[col]=np.nan
        if 'psnr_db' not in q:q['psnr_db']=np.nan
        label=str(path.parent.relative_to(folder));dest=out/label;dest.mkdir(parents=True,exist_ok=True)
        for (method,variant),g in q.groupby(['method','variant']):
            raw=a[(a.method==method)&(a.variant==variant)];valid=raw[(raw.status=='ok')&(raw.qualified==1)];returned=raw[raw.status=='ok'];qual=g[g.qualified==1];ci=cluster_ci(valid,'ber');row=dict(method=method,variant=variant,attempts=len(g),qualified=len(qual),qualification_rate=len(qual)/len(g),attack_rows=len(raw),valid_attacks=len(valid),returned_attacks=len(returned),qualified_mean_ber=ci[0],ber_ci_low=ci[1],ber_ci_high=ci[2],n_hosts=ci[3],qualified_mean_nc01=float(valid.nc01.mean()) if len(valid) else None,all_returned_mean_ber=float(returned.ber.mean()) if len(returned) else None,all_returned_mean_nc01=float(returned.nc01.mean()) if len(returned) else None,all_returned_balanced_accuracy=float(returned.balanced_accuracy.mean()) if len(returned) and 'balanced_accuracy' in returned else None,median_psnr=float(pd.to_numeric(qual.psnr_db,errors='coerce').median()) if len(qual) else None,median_embed_ms=float(g.embed_ms.median()) if 'embed_ms' in g else None,worst_host_ber=float(valid.groupby('host').ber.mean().max()) if len(valid) else None,worst_condition_ber=float(valid.groupby('condition_id').ber.mean().max()) if len(valid) else None)
            rows.append(row)
            for (cat,level),z in valid.groupby(['category','level']):fams.append(dict(method=method,variant=variant,category=cat,level=level,mean_ber=float(z.ber.mean()),mean_nc01=float(z.nc01.mean()),n=len(z)))
            if method=='FC-AOIP':continue
            base=a[(a.method=='FC-AOIP')&(a.variant=='default')&(a.status=='ok')&(a.qualified==1)];join=valid.merge(base,on=['host','key_index','watermark','condition_id'],suffixes=('_baseline','_proposal'),validate='one_to_one') if 'watermark' in valid else valid.merge(base,on=['host','key_index','condition_id'],suffixes=('_baseline','_proposal'),validate='one_to_one');join['difference']=join.ber_baseline-join.ber_proposal;d=cluster_ci(join,'difference')
            pairs.append(dict(method=method,variant=variant,paired_attacks=len(join),paired_hosts=d[3],mean_baseline_minus_proposal_ber=d[0],ci_low=d[1],ci_high=d[2]))
        if 'watermark' in q:
            per_logo=[]
            for (m,v,w),g in q.groupby(['method','variant','watermark']):
                z=a[(a.method==m)&(a.variant==v)&(a.watermark==w)&(a.status=='ok')&(a.qualified==1)]
                per_logo.append(dict(method=m,variant=v,watermark=w,attempts=len(g),qualified=int(g.qualified.sum()),mean_clean_psnr=float(pd.to_numeric(g.psnr_db,errors='coerce').mean()),mean_attacked_ber=float(z.ber.mean()) if len(z) else None,mean_attacked_nc01=float(z.nc01.mean()) if len(z) else None))
            write_csv(dest/'watermark_summary.csv',per_logo)
        write_csv(dest/'method_summary.csv',rows);write_csv(dest/'family_summary.csv',fams);write_csv(dest/'paired_summary.csv',pairs)
        pd.DataFrame(rows).to_latex(dest/'method_summary.tex',index=False,float_format='%.4f',escape=True)
    for path in folder.rglob('raw_results.csv'):
        df=pd.read_csv(path);dest=out/str(path.parent.relative_to(folder));dest.mkdir(parents=True,exist_ok=True)
        if 'sample' in df and 'embed_ms' in df:
            cols=['experiment','method','height','width','payload_bits','kind','kernel_only','psnr_floor'];rows=[]
            for key,g in df[df['sample'].notna()].groupby(cols,dropna=False):
                r=dict(zip(cols,key));r.update(samples=len(g),qualified=int((g.status=='ok').sum()))
                for col in ('embed_ms','extract_ms','api_total_ms','rss_mb','peak_rss_mb'):
                    vals=pd.to_numeric(g[col],errors='coerce').dropna() if col in g else []
                    if len(vals):
                        for name,pct in [('min',0),('median',50),('p90',90),('p95',95),('p99',99),('max',100)]:r[f'{col}_{name}']=float(np.percentile(vals,pct))
                        r[f'{col}_mean']=float(np.mean(vals));r[f'{col}_std']=float(np.std(vals))
                rows.append(r)
            write_csv(dest/'latency_summary.csv',rows)
            g=df[(df.experiment=='C01')&(df.status=='ok')].dropna(subset=['embed_ms','pixels','payload_bits']);
            if len(g)>=3:
                X=np.column_stack([g.pixels,g.payload_bits,np.ones(len(g))]);coef=np.linalg.lstsq(X,g.embed_ms,rcond=None)[0];pred=X@coef;atomic_json(dest/'complexity_fit.json',dict(model='a*N+b*K+c, descriptive only',coefficients=coef.tolist(),rank=int(np.linalg.matrix_rank(X)),rmse_ms=float(np.sqrt(np.mean((pred-g.embed_ms)**2))),observations=len(g)))
        if 'trial' in df:
            if 'ber' not in df:df['ber']=np.nan
            rows=[]
            for (method,watermark),g in df.groupby(['method','watermark']):
                for trial,z in g.groupby('trial'):
                    valid=z[z.status=='ok'];r=dict(method=method,watermark=watermark,trial=trial,trials=len(z),valid=len(valid),mean_ber=float(valid.ber.mean()) if len(valid) else None)
                    for threshold in (.05,.1,.2,.3,.4):r[f'accept_rate_ber_le_{threshold}']=float((valid.ber<=threshold).mean()) if len(valid) else None
                    rows.append(r)
            write_csv(dest/'detection_summary.csv',rows)
    print('Summaries:',out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/paper');p.add_argument('--out',default='results/paper_ready');a=p.parse_args();summarize(a.root,a.out)
