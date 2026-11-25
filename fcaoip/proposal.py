"""OMIP: one fixed integer carrier, optimal feature margin, exact projection."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .kernel import run_embed, decode, features as kernel_features



@dataclass(frozen=True)
class EmbedInfo:
    margin: int
    sse: int
    psnr_db: float
    payload_bits: int
    height: int
    width: int
    seed: int
    orbit_steps: tuple[int,...]
    orbit_offsets: tuple[int,...]
    certified_min_margin: int
    psnr_floor_db: float

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


def quant_index(z: int, step: int, offset: int=0) -> int:
    """Nearest integer scalar quantization index with ties toward +infinity."""
    return (int(z)+int(offset)+int(step)//2)//int(step)


def minimum_orbit_margin(steps=(2,4), offsets=(-1,0,1)):
    """Exact uniform |feature| margin for nonzero signed quantized decisions.

    If s=+1, require Q_q(F+o)>=1, and if s=-1 require Q_q(F+o)<=-1,
    for every pair (q,o) in the Cartesian product of the specified families.
    This is a sufficient and sharp *uniform signed-magnitude* integer threshold.
    """
    if not steps or not offsets or any(int(q)<=0 for q in steps):
        raise ValueError('Nonempty positive integer quantizer steps/offsets required')
    return max(max((int(q)+1)//2-int(o), int(q)//2+int(o)+1)
               for q in steps for o in offsets)


def _validate(image,bits,seed):
    x=np.asarray(image)
    b=np.asarray(bits,dtype=np.uint8).reshape(-1)
    if x.ndim!=3 or x.shape[2]!=3 or x.dtype!=np.uint8:
        raise ValueError('Expected 8-bit RGB image')
    if x.shape[0]%8 or x.shape[1]%8:
        raise ValueError('Height and width must be divisible by 8')
    capacity=(x.shape[0]//8)*(x.shape[1]//8)
    if not 1<=b.size<=capacity or np.any(b>1):
        raise ValueError(f'Payload must contain 1..{capacity} binary bits')
    if not isinstance(seed,(int,np.integer)) or not 0<=seed<2**128:
        raise ValueError('seed must be in [0, 2^128)')
    return x,b


def psnr(x,y):
    d=x.astype(np.int32)-y.astype(np.int32)
    sse=int(np.square(d,dtype=np.int64).sum())
    return float('inf') if sse==0 else 10*math.log10(255**2*x.size/sse)


def _project(image,bits,*,seed=2026,psnr_floor_db=55.,orbit_steps=(2,4),orbit_offsets=(-1,0,1),kind=0,pred=3,optimize=True,optimal=True):
    x,b=_validate(image,bits,seed)
    if not math.isfinite(psnr_floor_db) or psnr_floor_db<0:raise ValueError('Invalid PSNR floor')
    minimum=minimum_orbit_margin(orbit_steps,orbit_offsets)
    budget=math.floor(x.size*255**2 * 10**(-float(psnr_floor_db)/10))
    y,margin,claimed=run_embed(x,b,kind,pred,int(seed),budget,minimum,optimize,optimal)
    if margin<0:raise ValueError('DISTORTION_LIMIT: requested margin exceeds distortion budget')
    sse=int(np.square(y.astype(np.int32)-x.astype(np.int32),dtype=np.int64).sum())
    if sse!=claimed or sse>budget:raise AssertionError('SSE discrepancy')
    v=kernel_features(y,len(b),kind,pred,int(seed))
    if not np.array_equal((v>0).astype(np.uint8),b) or np.min(np.where(b>0,v,-v))<margin:raise AssertionError('Signed margin violation')
    return y,EmbedInfo(int(margin),sse,psnr(x,y),b.size,x.shape[0],x.shape[1],int(seed),tuple(map(int,orbit_steps)),tuple(map(int,orbit_offsets)),minimum,float(psnr_floor_db))

def embed(image,bits,*,seed=2026,psnr_floor_db=55.,orbit_steps=(2,4),orbit_offsets=(-1,0,1)):
    return _project(image,bits,seed=seed,psnr_floor_db=psnr_floor_db,orbit_steps=orbit_steps,orbit_offsets=orbit_offsets)

def features(image,*,seed=2026,length=4096):
    x,_=_validate(image,np.zeros(length,np.uint8),seed)
    return kernel_features(x,length,0,3,int(seed))

def extract(image,*,seed=2026,length=4096):
    return (features(image,seed=seed,length=length)>0).astype(np.uint8)

def check_restricted_orbit(marked,bits,info):
    """Certify only exact integer signed-feature quantizer family, not JPEG/video."""
    vals=features(marked,seed=info.seed,length=info.payload_bits)
    b=np.asarray(bits,dtype=np.uint8).ravel()
    if b.size!=len(vals):raise ValueError('payload length mismatch')
    for q in info.orbit_steps:
        for o in info.orbit_offsets:
            v=np.array([quant_index(int(z),q,o) for z in vals],dtype=np.int64)
            if not np.array_equal((v>0).astype(np.uint8),b):return False
            if np.any((b==0)&(v>=0)):return False # nonzero negative index requirement
    return True
