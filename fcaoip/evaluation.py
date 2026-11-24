"""Shared method API and metrics; exactly one proposal + five classical adapters."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
from .proposal import embed as proposal_embed, extract as proposal_extract
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baselines.registry import METHODS as BASELINES, PAPERS, embed as baseline_embed, extract as baseline_extract

METHODS=('FC-AOIP',*BASELINES)
FIXED_STRENGTHS={'Scalar-QIM':12.0,'STDM-QIM':10.0,'SCS':12.0,'ISS':3.0,'SS':3.0}

def apply_embed(method,image,bits,*,seed=2026,psnr=55.,steps=(2,4),offsets=(-1,0,1),policy='adaptive',strengths=None):
    if method=='FC-AOIP':
        y,info=proposal_embed(image,bits,seed=int(seed),psnr_floor_db=float(psnr),
            orbit_steps=steps,orbit_offsets=offsets,mode_policy=policy)
        return y,{'method':method,'mode':info.mode,'length':len(bits),'seed':seed,'info':info.as_dict()}
    if method not in BASELINES:raise KeyError(f'Unknown method {method}')
    if len(bits)!=4096 or image.shape!=(512,512,3):
        raise ValueError('The supplied classical adapters require 512x512 RGB and 4096 bits')
    step=float((strengths or FIXED_STRENGTHS)[method])
    step=int(step) if method in ('Scalar-QIM','SCS') else step
    y,key=baseline_embed(method,image,np.asarray(bits,dtype=np.uint8),step=step,seed=int(seed))
    return y,{'method':method,'key':key,'seed':int(seed),'step':step,'length':len(bits)}

def apply_extract(method,image,public):
    if method=='FC-AOIP':
        return proposal_extract(image,mode=public['mode'],seed=public['seed'],length=public['length'])
    return np.asarray(baseline_extract(method,image,public['key']),dtype=np.uint8).reshape(-1)

def psnr(x,y):
    d=x.astype(np.int32)-y.astype(np.int32)
    sse=int(np.sum(d.astype(np.int64)*d.astype(np.int64)))
    return float('inf') if sse==0 else 10*math.log10(255**2*x.size/sse)

def ber(a,b):
    a=np.asarray(a).reshape(-1)
    b=np.asarray(b).reshape(-1)
    if len(a)!=len(b):raise ValueError('Payload size mismatch')
    return float(np.mean(a!=b))

def nc01(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    b=np.asarray(b,dtype=np.float64).reshape(-1)
    d=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/d) if d>0 else float(np.array_equal(a,b))
