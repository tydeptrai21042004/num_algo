#!/usr/bin/env python3
"""Paired host-level sign tests for the held-out 60-attack comparison.
No pseudoreplication over keys or attacks. No fabricated missing scores.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import METHODS, atomic_json, write_csv, WATERMARK_IDS
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [None]*len(values)
    prev = 0.
    for rank, i in enumerate(order):
        prev = max(prev, min(1., (len(values)-rank)*values[i]))
        adjusted[i] = prev
    return adjusted


def analyze(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = root/'robustness60'/'attacks.csv'
    if not path.exists():
        raise FileNotFoundError(f'Run the held-out test stage first: {path}')
    df = pd.read_csv(path)
    ids = ['host','key_index','watermark','condition_id']
    required = set(ids+['method','variant','status','qualified','split'])
    if not required <= set(df):
        raise ValueError(f'Missing columns: {required-set(df)}')
    if set(df.split) != {'test'} or set(df.method) != set(METHODS):
        raise ValueError('Expected held-out results for all six methods')
    df = df[((df.method=='FC-AOIP') & (df.variant=='default')) | ((df.method!='FC-AOIP') & (df.variant=='matched'))].copy()
    if df.duplicated(ids+['method']).any():
        raise ValueError('Duplicate method/host/key/condition rows')
    import json
    cfg = json.loads((path.parent/'manifest.json').read_text())['config']
    hosts = {Path(p).stem for p in cfg['hosts']}
    expected = len(hosts)*cfg['keys']*len(WATERMARK_IDS)*60
    expected_ids = {(h,k,w,f'A{i:03d}') for h in hosts for k in range(cfg['keys']) for w in WATERMARK_IDS for i in range(1,61)}
    if len(hosts)!=8 or any(set(map(tuple, df.loc[df.method==m,ids].itertuples(index=False,name=None)))!=expected_ids for m in METHODS):
        raise ValueError('Incomplete or unexpected held-out attack schedule')
    if 'ber' not in df: df['ber'] = np.nan
    df['ber'] = pd.to_numeric(df.ber, errors='coerce')
    df = df[(df.status=='ok') & (df.qualified==1) & df.ber.between(0,1)]
    base = df[df.method=='FC-AOIP'][ids+['ber']]
    rows, details = [], []
    for m in METHODS[1:]:
        j = df[df.method==m][ids+['ber']].merge(base,on=ids,suffixes=('_baseline','_proposal'),validate='one_to_one')
        j['delta'] = j.ber_baseline-j.ber_proposal
        g = j.groupby('host').delta.agg(['mean','size'])
        d = g['mean'].to_numpy(); nonzero = d[~np.isclose(d,0,atol=1e-12,rtol=0)]
        wins = int((nonzero>0).sum()); losses = int((nonzero<0).sum())
        p = float(binomtest(wins,len(nonzero),.5).pvalue) if len(nonzero) else (1. if len(d) else None)
        rows.append(dict(baseline=m,expected_pairs=expected,qualified_pairs=len(j),pair_coverage=len(j)/expected,
                         paired_hosts=len(g),expected_hosts=len(hosts),wins=wins,losses=losses,ties=len(d)-len(nonzero),
                         mean_host_ber_difference=float(d.mean()) if len(d) else None,
                         median_host_ber_difference=float(np.median(d)) if len(d) else None,p_sign=p))
        details.extend(dict(baseline=m,host=h,paired_attacks=int(v['size']),mean_ber_difference=float(v['mean'])) for h,v in g.iterrows())
    # Missing comparisons count conservatively in the five-test family.
    adjusted=holm([r['p_sign'] if r['p_sign'] is not None else 1. for r in rows])
    for r,p in zip(rows,adjusted):
        r['p_holm']=p if r['p_sign'] is not None else None
        r['significant_005']=bool(p<.05) if r['p_sign'] is not None else None
    write_csv(out/'paired_sign_tests.csv',rows)
    write_csv(out/'paired_host_differences.csv',details)
    pd.DataFrame(rows).to_latex(out/'paired_sign_tests.tex',index=False,float_format='%.4f',escape=True)
    atomic_json(out/'methodology.json',dict(unit='host',test='two-sided exact paired sign test; Holm across five baselines',
        estimand='baseline minus proposal BER, averaged over jointly qualified matching keys, watermarks and attacks within each host',
        caveats=['Conditional on jointly qualified cases; not an unconditional robustness comparison.', 'Missing or failed cases are not imputed; coverage is reported.', 'Eight previously used hosts; inference is exploratory and assumes independent hosts.', 'Positive difference favors FC-AOIP.']))
    print('Statistical tests:',out)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/compact12_real_watermarks'));p.add_argument('--out',type=Path,default=Path('results/compact12_real_watermarks_summary/statistics'));a=p.parse_args();analyze(a.root,a.out)
