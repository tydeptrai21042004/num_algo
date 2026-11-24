"""One proposal: Frontier-Certified Attack-Orbit Integer Projection (FC-AOIP).

Practical path: 8x8 keyed nonoverlapping color-residual sign carriers with exact
minimum-SSE G-pixel edits, two complementary carrier modes, and a quality-budget
feasibility check. The public per-image mode is required for blind extraction.

Certified attack orbit is RESTRICTED: integer feature z -> Q_q(z+o), where
Q_q(w) = floor((w+floor(q/2))/q), and offsets o belong to a finite declared set.
It is NOT a certified JPEG/H.264 attack model. Image/video attack results must
be measured separately. No ECC, repetition, CRC, denoising, or attack classifier.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .kernel import run_embed, decode

MODES = {0:(0,3), 1:(3,2)}  # DCT44(3G-R-B), checkerboard(2G-R-B)

@dataclass(frozen=True)
class EmbedInfo:
    mode: int
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
    if not isinstance(seed,(int,np.integer)) or not 0<=seed<2**31:
        raise ValueError('seed must be in [0, 2^31)')
    return x,b


def psnr(x,y):
    d=x.astype(np.int32)-y.astype(np.int32)
    sse=int(np.square(d,dtype=np.int64).sum())
    return float('inf') if sse==0 else 10*math.log10(255**2*x.size/sse)


def embed(image,bits,*,seed=2026,psnr_floor_db=55.0,
          orbit_steps=(2,4),orbit_offsets=(-1,0,1),
          mode_policy='adaptive',max_margin=512):
    x,b=_validate(image,bits,seed)
    min_margin=minimum_orbit_margin(orbit_steps,orbit_offsets)
    if min_margin>max_margin: raise ValueError('max_margin below orbit certificate')
    budget=math.floor(x.size*255**2 * 10**(-float(psnr_floor_db)/10))
    if budget<0:raise ValueError('Invalid PSNR floor')
    choices={'adaptive':(0,1),'dct44':(0,),'checker':(1,)}
    if mode_policy not in choices:raise ValueError('Unsupported mode_policy')
    for mode in choices[mode_policy]:
        kind,pred=MODES[mode]
        y,margin,claimed=run_embed(x,b,kind,pred,int(seed),int(max_margin),int(budget),int(min_margin))
        if margin<0:continue
        sse=int(np.square(y.astype(np.int32)-x.astype(np.int32),dtype=np.int64).sum())
        if sse!=claimed or sse>budget: raise AssertionError('SSE discrepancy or PSNR floor breached')
        recovered=extract(y,mode=mode,seed=seed,length=b.size)
        if not np.array_equal(b,recovered): raise AssertionError('Nonzero clean BER')
        # Strict mathematical checks of the idealized integer feature orbit.
        vals=features(y,mode=mode,seed=seed,length=b.size)
        signed=np.where(b>0,vals,-vals)
        if np.any(signed<min_margin):raise AssertionError('Orbit margin violation')
        return y,EmbedInfo(mode,int(margin),sse,psnr(x,y),b.size,x.shape[0],x.shape[1],
                           int(seed),tuple(map(int,orbit_steps)),tuple(map(int,orbit_offsets)),
                           min_margin,float(psnr_floor_db))
    raise ValueError(f'INFEASIBLE: no permitted mode fits PSNR {psnr_floor_db:.2f} dB and quantized-orbit margin {min_margin}')


def extract(image,*,mode,seed=2026,length=4096):
    x=np.asarray(image)
    if x.ndim!=3 or x.shape[2]!=3 or x.dtype!=np.uint8 or x.shape[0]%8 or x.shape[1]%8:
        raise ValueError('Expected RGB uint8 dimensions divisible by 8')
    if mode not in MODES:raise ValueError('Public 1-bit mode descriptor required')
    if not 1<=length<=(x.shape[0]//8)*(x.shape[1]//8):raise ValueError('Invalid payload length')
    kind,pred=MODES[mode]
    return decode(x,int(length),kind,pred,int(seed))


def features(image,*,mode,seed=2026,length=4096):
    """Vectorized exact integer signed color-residual features in keyed block order."""
    x=np.asarray(image)
    h,w=x.shape[:2]; nr=h//8;nc=w//8;K=nr*nc
    a=2*(int(seed)%65536)+1
    while math.gcd(a,K)!=1:a+=2
    b=(int(seed)*23+59)%K
    blocks=x.reshape(nr,8,nc,8,3).transpose(0,2,1,3,4).reshape(K,8,8,3).astype(np.int32)
    rr=np.asarray([1,-1,-1,1,1,-1,-1,1],dtype=np.int32)
    weights=np.outer(rr,rr) if mode==0 else np.fromfunction(lambda y,z:1-2*((y+z)%2),(8,8),dtype=int)
    coef=(3 if mode==0 else 2)*blocks[:,:,:,1]-blocks[:,:,:,0]-blocks[:,:,:,2]
    all_features=np.sum(coef*weights,axis=(1,2),dtype=np.int32)
    index=(np.arange(length,dtype=np.int64)*int(a)+b)%K
    return all_features[index]


def check_restricted_orbit(marked,bits,info):
    """Certify only exact integer signed-feature quantizer family, not JPEG/video."""
    vals=features(marked,mode=info.mode,seed=info.seed,length=info.payload_bits)
    b=np.asarray(bits,dtype=np.uint8).ravel()
    if b.size!=len(vals):raise ValueError('payload length mismatch')
    for q in info.orbit_steps:
        for o in info.orbit_offsets:
            v=np.array([quant_index(int(z),q,o) for z in vals],dtype=np.int64)
            if not np.array_equal((v>0).astype(np.uint8),b):return False
            if np.any((b==0)&(v>=0)):return False # nonzero negative index requirement
    return True
