#!/usr/bin/env python3
"""Multi-key clean feasibility, no claim of universal PSNR guarantees."""
import argparse,csv,hashlib,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.watermarks import WATERMARK_IDS, watermark_bits
from fcaoip.proposal import embed,extract
from fcaoip.evaluation import ber

def run(args):
    hosts=sorted(p.stem for p in (ROOT/'data/host').glob('*.bmp')) if args.hosts=='all' else args.hosts.split(',')
    rows=[]
    for host in hosts:
        x=np.array(Image.open(ROOT/'data/host'/f'{host}.bmp').convert('RGB'))
        for key_i in range(args.keys):
            seed=int.from_bytes(hashlib.sha256(f'{args.seed}|{host}|{key_i}'.encode()).digest()[:4],'little')&0x7fffffff
            for watermark in WATERMARK_IDS:
                bits=watermark_bits(watermark,args.bits)
                row=dict(host=host,key_index=key_i,watermark=watermark,seed=seed,status='ok',psnr_db='',mode='',margin='',sse='',clean_ber='',error='')
                try:
                    y,meta=embed(x,bits,seed=seed,psnr_floor_db=args.psnr)
                    row.update(psnr_db=meta.psnr_db,mode=meta.mode,margin=meta.margin,sse=meta.sse,
                               clean_ber=ber(bits,extract(y,mode=meta.mode,seed=seed,length=args.bits)))
                except ValueError as e:row.update(status='infeasible',error=str(e)[:200])
                rows.append(row)
        print(host,'qualified',sum(r['status']=='ok' for r in rows if r['host']==host),'/',args.keys*len(WATERMARK_IDS),flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('Total',sum(r['status']=='ok' for r in rows),'/',len(rows),'Wrote',out)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--hosts',default='all');a.add_argument('--keys',type=int,default=25)
    a.add_argument('--bits',type=int,default=4096);a.add_argument('--seed',type=int,default=2026)
    a.add_argument('--psnr',type=float,default=55);a.add_argument('--out',default='results/key_audit_25.csv')
    run(a.parse_args())
