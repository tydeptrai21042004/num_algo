#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.common import *
from experiments.images import run_images,run_security,calibration_selection,read_frozen
from experiments.mathematics import run_math
from experiments.compute import run_compute
from experiments.video import run_video
from experiments.datasets import verify_splits

def main():
    p=argparse.ArgumentParser(description='Frozen FC-AOIP experimental suite');p.add_argument('--group',required=True,choices=['mathematics','hyperparameters','calibration','validation','robustness60','ablation','payload_robustness','security','compute','compute_stress','video']);p.add_argument('--profile',choices=PROFILES,default='paper');p.add_argument('--split',choices=['legacy','train','development','validation','test'],default=None);p.add_argument('--resume',action='store_true');p.add_argument('--out');p.add_argument('--limit',type=int);p.add_argument('--keys',type=int);p.add_argument('--methods',default='all');p.add_argument('--frozen',default='results/frozen_config.json');p.add_argument('--calibration',default='results/paper/calibration');p.add_argument('--operations',type=int);p.add_argument('--seconds',type=float);p.add_argument('--input');p.add_argument('--frames',type=int);p.add_argument('--video-mode',choices=['all','codec','stream'],default='all');p.add_argument('--loop-input',action='store_true');p.add_argument('--fps-list',type=float,nargs='+');p.add_argument('--keep-videos',action='store_true');a=p.parse_args()
    verify_core();group=a.group;split=a.split or ('legacy' if a.profile=='smoke' else 'train' if group in ('hyperparameters','calibration') else 'validation' if group=='validation' else 'test');out=Path(a.out or f'results/{a.profile}/{group}_{split}');methods=None if a.methods=='all' else a.methods.split(',');
    if a.fps_list and any(f<=0 or f>1000 for f in a.fps_list):p.error('FPS must be in (0,1000]')
    if methods and not set(methods)<=set(METHODS):p.error('Unknown method')
    if any(v is not None and v<=0 for v in (a.limit,a.keys,a.operations,a.seconds,a.frames)):p.error('Counts and durations must be positive')
    if group=='mathematics':run_math(a.profile,out,a.resume)
    elif group in ('compute','compute_stress'):run_compute(a.profile,out,a.resume,a.operations,a.seconds)
    elif group=='video':run_video(a.profile,out,a.resume,split,a.input,a.frames,a.keep_videos,a.video_mode,a.seconds,a.loop_input,methods,a.fps_list)
    else:
        if a.profile!='smoke' and split=='legacy' and group in ('hyperparameters','calibration','validation'):raise ValueError('Legacy images are not calibration or held-out validation data')
        if split=='development':raise ValueError('development is a video-only split')
        if a.profile!='smoke' and split!='legacy':
            verify_splits()
            dataset_profile=json.loads((ROOT/'data/manifests/image_sources.json').read_text())['profile']
            if a.profile in ('paper','stress') and dataset_profile=='lite':raise ValueError('Paper/stress requires full selected independent datasets; run download_images.py --profile paper')
        if group in ('hyperparameters','calibration') and split!='train' and not (a.profile=='smoke' and split=='legacy'):raise ValueError('Hyperparameter/calibration experiments require train split')
        if group=='validation' and split!='validation' and not (a.profile=='smoke' and split=='legacy'):raise ValueError('Validation requires held-out split')
        matched=None
        if group=='validation' and a.profile!='smoke':matched,_=calibration_selection(a.calibration)
        elif group=='validation':matched=FIXED_STRENGTHS
        if split=='test':matched=read_frozen(a.frozen)
        elif group=='robustness60' and Path(a.frozen).exists() and a.profile!='smoke':matched=read_frozen(a.frozen)
        if group=='security':run_security(split,a.profile,out,a.resume,a.limit,a.keys,method_filter=methods or ("FC-AOIP",))
        else:run_images(group,split,a.profile,out,a.resume,a.limit,a.keys,matched,methods)
    print('Finished',out,flush=True)
if __name__=='__main__':main()
