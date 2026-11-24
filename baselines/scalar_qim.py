"""Standalone spatial scalar-QIM comparison (Chen--Wornell QIM family).

A one-sample-per-block image-domain adapter, NOT a replication of an image
experiment in Chen and Wornell 2001. Integer-valued dither cosets support
exact clean recovery even at 0/255, with no transforms or ECC.
"""
from __future__ import annotations
import functools
import numpy as np

N=4096
H=W=512

@functools.lru_cache(maxsize=64)
def schedule(seed:int,delta:int):
    if delta < 2 or delta%2: raise ValueError('Delta must be even >=2')
    rng=np.random.default_rng(seed)
    # Distinct sample for each 8x8 RGB image block.
    position=rng.integers(0, 192, N, dtype=np.int32)
    phase=rng.integers(0, delta, N, dtype=np.int32)
    return position,phase

def _blocks(rgb):
    im=np.asarray(rgb)
    if im.shape!=(H,W,3) or im.dtype!=np.uint8:
        raise ValueError('Expected uint8 RGB 512x512')
    return im.reshape(64,8,64,8,3).transpose(0,2,1,3,4).reshape(N,192)

def _nearest_valid(v, residue, step):
    """Nearest integer in {residue + k*step: k>=0, value<=255}."""
    k=np.floor((v-residue)/step+0.5).astype(np.int32)
    k=np.clip(k,0,(255-residue)//step)
    return residue+step*k

def embed(rgb,bits,*,step:int=16,seed:int=2026):
    bits=np.asarray(bits,dtype=np.uint8).reshape(-1)
    if bits.size != N or not np.all((bits==0)|(bits==1)):
        raise ValueError('Expected 4096 bits')
    blocks=_blocks(rgb).copy()
    pos,phase=schedule(seed,step)
    v=blocks[np.arange(N),pos].astype(np.int32)
    rem=(phase+bits.astype(np.int32)*(step//2))%step
    blocks[np.arange(N),pos]=_nearest_valid(v,rem,step).astype(np.uint8)
    out=blocks.reshape(64,64,8,8,3).transpose(0,2,1,3,4).reshape(H,W,3)
    return out

def extract(rgb,*,step:int=16,seed:int=2026):
    blocks=_blocks(rgb)
    pos,phase=schedule(seed,step)
    v=blocks[np.arange(N),pos].astype(np.int32)
    d0=np.abs(v-_nearest_valid(v,phase,step))
    r1=(phase+step//2)%step
    d1=np.abs(v-_nearest_valid(v,r1,step))
    return (d1<d0).astype(np.uint8)

def psnr(x,y):
    dif=x.astype(np.int32)-y.astype(np.int32)
    sse=np.sum(dif*dif,dtype=np.int64)
    return float('inf') if sse==0 else 10*np.log10(255**2*int(dif.size)/int(sse))

if __name__=='__main__':
    x=np.zeros((H,W,3),np.uint8);r=np.random.default_rng(42);b=r.integers(0,2,N,dtype=np.uint8)
    for value in (0,255,None):
        if value is None:x=r.integers(0,256,(H,W,3),dtype=np.uint8)
        else:x.fill(value)
        for step in (2,4,12,16,20,24,32):
            y=embed(x,b,step=step)
            rec=extract(y,step=step)
            assert np.array_equal(rec,b),(value,step)
    print('QIM exactness tests PASS')
