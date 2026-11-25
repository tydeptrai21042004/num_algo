#!/usr/bin/env python3
"""Local clean video evaluation, no lossy codec or frame attacks."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.video import run_video

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',default='results/omip_video_local');p.add_argument('--frames',type=int,default=100);p.add_argument('--profile',choices=['lite','paper','smoke'],default='lite');p.add_argument('--resume',action='store_true');p.add_argument('--fps',type=float,nargs='+',default=[24,30,60]);a=p.parse_args()
    run_video(a.profile,a.out,a.resume,explicit=a.input,frames=a.frames,fps_list=a.fps,keep=True,method_filter=('FC-AOIP',))
if __name__=='__main__':main()
