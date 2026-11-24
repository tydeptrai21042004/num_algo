#!/usr/bin/env python3
"""End-to-end video comparison: exactly one proposal + five baseline adapters."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from fcaoip.evaluation import METHODS
from scripts.run_video import process

def main(args):
    if str(args.input).startswith('camera:'):
        raise ValueError('Use a video file to compare methods on exactly the same frames; live cameras cannot be replayed identically')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method in METHODS:
        cfg=argparse.Namespace(input=args.input,method=method,frames=args.frames,bits=4096,
                 width=512,height=512,fps=args.fps,seed=args.seed,psnr=args.psnr,
                 out=str(out/method.replace('-','_')))
        result=process(cfg)
        rows.append(dict(method=method,frames=result['frames_processed'],
            qualified=result['qualified_frames'],
            throughput_fps=result['throughput_fps'],
            mean_embed_ms=result['mean_embed_ms'],
            mean_post_mp4_ber=result['mean_encoded_ber']))
    with (out/'all_methods_video_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'all_methods_video_summary.json').write_text(json.dumps(rows,indent=2))
    print('Video comparison written to',out)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True)
    a.add_argument('--frames',type=int,default=30)
    a.add_argument('--fps',type=float,default=20)
    a.add_argument('--seed',type=int,default=2026)
    a.add_argument('--psnr',type=float,default=55)
    a.add_argument('--out',default='results/video_all')
    main(a.parse_args())
