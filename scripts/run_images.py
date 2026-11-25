#!/usr/bin/env python3
"""Image benchmark: 1 proposal + 5 paper-derived baseline *adapters*; 60 attacks.

No baseline is labeled 'paper-reproduced result': mappings from signals to 8x8
RGB blocks are declared implementation choices, not authors' experiments.
"""
from __future__ import annotations
import argparse,csv,json,sys,time,platform,hashlib,traceback
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.watermarks import WATERMARK_IDS, watermark_bits, watermark_manifest
from fcaoip.evaluation import METHODS,apply_embed,apply_extract,psnr,ber,nc01,FIXED_STRENGTHS
from attack_suite60 import canonical_conditions, attack_rgb,protocol_sha256

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)

def stable_int(*args):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,args)).encode()).digest()[:4],'little') & 0x7fffffff

def run(args):
    data=ROOT/'data'/'host'
    host_names=sorted(p.stem for p in data.glob('*.bmp')) if args.hosts=='all' else args.hosts.split(',')
    methods=list(METHODS) if args.methods=='all' else args.methods.split(',')
    if not all(m in METHODS for m in methods):raise ValueError(f'Only methods {METHODS}')
    conds=canonical_conditions()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    quality=[];attacks=[]
    for host_name in host_names:
        path=data/f'{host_name}.bmp'
        with Image.open(path) as p:im=np.asarray(p.convert('RGB'),dtype=np.uint8)
        for sidx in range(args.keys):
            seed=stable_int(args.seed,host_name,sidx,'schedule')
            for watermark in WATERMARK_IDS:
                bits=watermark_bits(watermark,args.bits)
                for method in methods:
                    record={'host':host_name,'key_index':sidx,'watermark':watermark,'method':method,'bits':args.bits,
                            'seed':seed,'status':'ok','psnr_db':'','clean_ber':'',
                            'sse':'','mode':'','embed_ms':'','extract_ms':'','error':''}
                    try:
                        # Warm-up JIT using this method and image before timing; report cold run separately if needed.
                        marked,public=apply_embed(method,im,bits,seed=seed,psnr=args.psnr,
                                                 steps=tuple(args.steps), offsets=tuple(args.offsets))
                        start=time.perf_counter_ns()
                        for _ in range(args.repeats):
                            marked,public=apply_embed(method,im,bits,seed=seed,psnr=args.psnr,
                                                     steps=tuple(args.steps), offsets=tuple(args.offsets))
                        elapsed=(time.perf_counter_ns()-start)/1e6/args.repeats
                        start=time.perf_counter_ns()
                        decoded=apply_extract(method,marked,public)
                        extract_ms=(time.perf_counter_ns()-start)/1e6
                        diff=marked.astype(np.int32)-im.astype(np.int32)
                        record.update(psnr_db=psnr(im,marked),clean_ber=ber(bits,decoded),
                                      sse=int(np.sum(diff.astype(np.int64)**2)),
                                      mode=public.get('mode',''),embed_ms=elapsed,extract_ms=extract_ms)
                        record['psnr55_clean0']=int(record['psnr_db']>=args.psnr and record['clean_ber']==0)
                        for c in conds:
                            if c['category']=='Clean':continue
                            attack_seed=stable_int(args.seed,host_name,sidx,c['condition_id'])
                            altered=attack_rgb(marked,c,seed=attack_seed)
                            pred=apply_extract(method,altered,public)
                            attacks.append({'host':host_name,'key_index':sidx,'watermark':watermark,'method':method,
                                            'condition_id':c['condition_id'],'category':c['category'],
                                            'attack':c['attack'],'level':c['level'],
                                            'ber':ber(bits,pred),'nc01':nc01(bits,pred),
                                            'clean_qualified':record['psnr55_clean0']})
                    except Exception as e:
                        record.update(status='distortion_limit' if 'DISTORTION_LIMIT' in str(e) else 'error',error=str(e)[:500])
                    quality.append(record)
                    print(f'{host_name} key{sidx} {method}: {record["status"]} PSNR={record["psnr_db"]} BER={record["clean_ber"]} ms={record["embed_ms"]}',flush=True)
                    write_csv(out/'quality_timing.csv',quality)
                    write_csv(out/'attacks.csv',attacks)
    manifest=dict(watermarks=watermark_manifest(),methods=methods,profile=args.profile,hosts=host_names,keys=args.keys,bits=args.bits,
                  seed=args.seed,psnr_floor_db=args.psnr,steps=args.steps,offsets=args.offsets,
                  timing='warm end-to-end embed API, copy-out, excludes image load/attack, includes keyed schedule and quality solving',
                  library_versions=dict(numpy=np.__version__),
                  cpu=platform.processor(),python=platform.python_version(),
                  attack_protocol_sha256=protocol_sha256(),
                  baseline_default_strengths=FIXED_STRENGTHS,
                  fair_comparison='Use only host/seed rows satisfying PSNR floor and clean BER0 for each method; unmatched-host aggregates are NOT paired comparisons.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    print('Wrote',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--hosts',default='Girl,house')
    p.add_argument('--methods',default='all')
    p.add_argument('--bits',type=int,default=4096)
    p.add_argument('--keys',type=int,default=1)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--psnr',type=float,default=55.0)
    p.add_argument('--steps',type=int,nargs='+',default=[2,4])
    p.add_argument('--offsets',type=int,nargs='+',default=[-1,0,1])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--profile',choices=['smoke','focused','full'],default='full')
    p.add_argument('--out',default='results/image_smoke')
    run(p.parse_args())
