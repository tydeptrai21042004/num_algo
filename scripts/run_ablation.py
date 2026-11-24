#!/usr/bin/env python3
"""Ablations for the SAME FC-AOIP proposal; not extra proposed methods.

Fixed carrier, complementary carrier, optional orbit-size variations, and
separate PSNR budget thresholds; all are tested on identical keyed payloads.
"""
import argparse,csv,sys,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fcaoip.proposal import embed,extract,check_restricted_orbit
from fcaoip.evaluation import ber,psnr,nc01
from src.attack_suite60 import canonical_conditions,attack_rgb

ABLATIONS={
    'full':dict(mode_policy='adaptive',orbit_steps=(2,4),orbit_offsets=(-1,0,1)),
    'fixed_dct44':dict(mode_policy='dct44',orbit_steps=(2,4),orbit_offsets=(-1,0,1)),
    'fixed_checker':dict(mode_policy='checker',orbit_steps=(2,4),orbit_offsets=(-1,0,1)),
    'sign_only':dict(mode_policy='adaptive',orbit_steps=(1,),orbit_offsets=(0,)),
    'stronger_quantized_orbit':dict(mode_policy='adaptive',orbit_steps=(2,4,8),orbit_offsets=(-2,0,2)),
}

def run(args):
    hosts=sorted(p.stem for p in (ROOT/'data/host').glob('*.bmp')) if args.hosts=='all' else args.hosts.split(',')
    rows=[];attack_rows=[]
    conditions=canonical_conditions()[1:]
    if args.attacks=='none':conditions=[]
    for h in hosts:
        with Image.open(ROOT/'data/host'/f'{h}.bmp') as p:x=np.asarray(p.convert('RGB'))
        for ki in range(args.keys):
            seed=int.from_bytes(hashlib.sha256(f'{h}/{ki}/{args.seed}'.encode()).digest()[:4],'little')&0x7fffffff
            bits=np.random.default_rng(seed).integers(0,2,args.bits,dtype=np.uint8)
            for ablation,kw in ABLATIONS.items():
                row=dict(host=h,key_index=ki,ablation=ablation,status='ok',psnr_db='',sse='',clean_ber='',
                         margin='',mode='',embed_ms='',orbit_certified='',error='')
                try:
                    start=time.perf_counter()
                    y,info=embed(x,bits,seed=seed,psnr_floor_db=args.psnr,**kw)
                    row.update(psnr_db=psnr(x,y),sse=info.sse,
                               clean_ber=ber(bits,extract(y,mode=info.mode,seed=seed,length=args.bits)),
                               margin=info.margin,mode=info.mode,embed_ms=(time.perf_counter()-start)*1000,
                               orbit_certified=check_restricted_orbit(y,bits,info))
                    for cond in conditions:
                        attacked=attack_rgb(y,cond,seed=int.from_bytes(hashlib.sha256(f'{h}/{ki}/{cond["condition_id"]}/{args.seed}'.encode()).digest()[:4],'little'))
                        pred=extract(attacked,mode=info.mode,seed=seed,length=args.bits)
                        attack_rows.append(dict(host=h,key_index=ki,ablation=ablation,
                            condition_id=cond['condition_id'],category=cond['category'],attack=cond['attack'],
                            level=cond['level'],ber=ber(bits,pred),nc01=nc01(bits,pred)))
                except Exception as e:row.update(status='infeasible' if 'INFEASIBLE' in str(e) else 'error',error=str(e))
                rows.append(row);print(h,ki,ablation,row['status'],row['psnr_db'],flush=True)
    out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if attack_rows:
        atk_path=out.with_name(out.stem+'_attacks.csv')
        with atk_path.open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(attack_rows[0]));wr.writeheader();wr.writerows(attack_rows)
        print('Wrote',atk_path.resolve())
    print('Wrote',out.resolve())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--hosts',default='Girl,house')
    a.add_argument('--keys',type=int,default=1);a.add_argument('--bits',type=int,default=4096)
    a.add_argument('--seed',type=int,default=2026);a.add_argument('--psnr',type=float,default=55)
    a.add_argument('--attacks',choices=['none','focused','full'],default='full')
    a.add_argument('--out',default='results/ablation.csv');run(a.parse_args())
