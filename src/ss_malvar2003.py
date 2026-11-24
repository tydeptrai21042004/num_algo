"""Blind traditional spread spectrum (SS), Eq. (2) of Malvar & Florencio (2003).

Direct paper-equation implementation with declared 8x8 DCT RGB image adapter.
The whole image host adapter is NOT a bit-exact reproduction of the original paper's
illustrative transforms. No non-blind source host is used for extraction.
"""
from __future__ import annotations
import numpy as np
from qr64_certified.baselines.common.dct.block_dct import (
    rgb_to_dct_blocks,dct_blocks_to_rgb,dct_blocks_from_rgb)
from qr64_certified.baselines.common.dct.keys import _regenerate_block_indices_from_key
from qr64_certified.baselines.common.dct.embedding import _vote_bits
from qr64_certified.baselines.common.dct.core import make_block_indices
from qr64_certified.baselines.common.dct.types import WatermarkKey
from qr64_certified.baselines.implementations.dct.dct_iss_malvar2003 import (
    COEFFICIENTS,_pn_sequences,BLOCK_SIZE)

METHOD_ID='ss_malvar2003'
DEFAULT_ALPHA=8.0


def embed(host_rgb:np.ndarray,watermark_binary:np.ndarray,*,seed:int=2026,
          repeat:int|str=1,step:float|None=None,method_params=None):
    host=np.asarray(host_rgb,dtype=np.uint8)
    bits=(np.asarray(watermark_binary).reshape(-1)>127).astype(np.uint8)
    if bits.size!=4096 or host.shape!=(512,512,3) or repeat!=1:
        raise ValueError('SS common adapter only supports 4096 independent bits, repeat=1, 512x512 RGB')
    alpha=float(DEFAULT_ALPHA if step is None else step)
    if alpha<=0:raise ValueError('SS alpha must be positive')
    coeffs,y,cb,cr,hc,wc=rgb_to_dct_blocks(host,BLOCK_SIZE)
    indices=make_block_indices(hc,wc,BLOCK_SIZE,bits.size,1,int(seed)).astype(np.int32)
    pn=_pn_sequences(len(indices),int(seed))
    b=2*bits.astype(np.float64)-1
    coords=tuple(tuple(int(v) for v in pair) for pair in COEFFICIENTS)
    # Traditional SS modulator: s=x+alpha*b*u, with unit-norm pseudorandom u.
    for j,(u,v) in enumerate(coords):
        coeffs[indices,u,v]+=alpha*b*pn[:,j]
    watermarked=dct_blocks_to_rgb(coeffs,y,cb,cr,hc,wc,BLOCK_SIZE)
    key=WatermarkKey(
        method_id=METHOD_ID,host_shape=tuple(host.shape),
        watermark_shape=(64,64),seed=int(seed),repeat=1,step=alpha,
        arnold_iter=0,arnold_period=1,threshold=127,
        params={'domain':'8x8 luminance DCT','block_size':BLOCK_SIZE,
                'coefficients':[list(z) for z in coords],
                'method':'traditional spread spectrum, paper Eq(2)',
                'adapter':'28 upper-mid/high DCT coefficients, 4096 blocks'},
        schedule=[],fully_blind=True,
        side_information='seed, alpha, 28D DCT PN coefficients and payload shape')
    return watermarked,key


def extract(image_rgb:np.ndarray,key:WatermarkKey)->np.ndarray:
    coeffs,_,_=dct_blocks_from_rgb(image_rgb,BLOCK_SIZE)
    indices=_regenerate_block_indices_from_key(key,np.asarray(image_rgb).shape)
    pn=_pn_sequences(len(indices),int(key.seed))
    coords=tuple(tuple(int(v) for v in pair) for pair in COEFFICIENTS)
    vectors=np.stack([coeffs[indices,u,v] for u,v in coords],axis=1)
    raw=(np.sum(vectors*pn,axis=1)>0).astype(np.uint8)
    return (_vote_bits(raw,4096).reshape(64,64)*255).astype(np.uint8)
