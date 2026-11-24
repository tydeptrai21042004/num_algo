"""Eggers--Baeuml--Tzschoppe--Girod (2003) SCS principle, image adapter.

Source: IEEE Trans Signal Processing 51(4), 1003--1019.
DOI https://doi.org/10.1109/TSP.2003.809366
Equation: y=x+alpha*(Q_b(x)-x), with keyed dither and binary cosets;
blind decoder: choose nearest scalar codebook from marked value.

One keyed GREEN sample per 8x8 block is an EXPLICIT new image-domain
adapter, NOT the original authors' image implementation or tables.
Use an announced alpha and delta; no optimize-to-test attack strengths.
"""
from __future__ import annotations
import numpy as np
from . import scalar_qim


def embed(image,bits,*,step=20,alpha=0.8,seed=2026):
    if not (0.5<alpha<=1.0):raise ValueError('alpha in (0.5,1]')
    if step<2 or step%2:raise ValueError('even step >=2')
    x=scalar_qim._blocks(image).copy()
    b=np.asarray(bits,dtype=np.uint8).reshape(-1)
    if b.size!=4096 or np.any(b>1):raise ValueError('4096 bits required')
    # Separate keyed dither, identical decoder schedule.
    pos,phase=scalar_qim.schedule(int(seed),int(step))
    idx=np.arange(4096)
    v=x[idx,pos].astype(np.int32)
    residue=(phase+b.astype(np.int32)*(step//2))%step
    quantized=scalar_qim._nearest_valid(v,residue,step)
    modified=np.clip(np.rint(v+alpha*(quantized-v)),0,255).astype(np.uint8)
    x[idx,pos]=modified
    return x.reshape(64,64,8,8,3).transpose(0,2,1,3,4).reshape(512,512,3)


def extract(image,*,step=20,alpha=0.8,seed=2026):
    # Alpha affects encoding only, not codebook decision.
    return scalar_qim.extract(image,step=int(step),seed=int(seed))
