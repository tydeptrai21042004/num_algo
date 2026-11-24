"""Exactly five paper-derived classical baseline adapters for the original ALgo-main tree.

These are reproductions of published modulation principles using openly declared
image block mappings, not original authors' complete experimental reference code.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from . import scalar_qim, scs_eggers2003
from qr64_certified.baselines.implementations.dct import dct_stdm_qim_chen2001 as stdm
from qr64_certified.baselines.implementations.dct import dct_iss_malvar2003 as iss
import ss_malvar2003 as ss

METHODS = ('Scalar-QIM', 'STDM-QIM', 'SCS', 'ISS', 'SS')
PAPERS = {
    'Scalar-QIM': 'https://doi.org/10.1109/18.923725',
    'STDM-QIM': 'https://doi.org/10.1109/18.923725',
    'SCS': 'https://doi.org/10.1109/TSP.2003.809366',
    'ISS': 'https://doi.org/10.1109/TSP.2003.809385',
    'SS': 'https://doi.org/10.1109/TSP.2003.809385',
}

def embed(method, image, bits, *, step, seed):
    if method == 'Scalar-QIM':
        s = int(step)
        return scalar_qim.embed(image,bits,step=s,seed=seed), {'step':s,'seed':seed}
    if method == 'SCS':
        s = int(step)
        return scs_eggers2003.embed(image,bits,step=s,alpha=0.8,seed=seed), {'step':s,'alpha':0.8,'seed':seed}
    wm = (bits.reshape(64,64).astype('uint8')*255)
    if method == 'STDM-QIM':
        return stdm.embed(image,wm,seed=seed,repeat=1,step=float(step))
    if method == 'ISS':
        return iss.embed(image,wm,seed=seed,repeat=1,step=float(step))
    if method == 'SS':
        return ss.embed(image,wm,seed=seed,repeat=1,step=float(step))
    raise ValueError(method)

def extract(method, image, key):
    if method == 'Scalar-QIM':
        return scalar_qim.extract(image,step=key['step'],seed=key['seed'])
    if method == 'SCS':
        return scs_eggers2003.extract(image,step=key['step'],alpha=key['alpha'],seed=key['seed'])
    if method == 'STDM-QIM':
        raw = stdm.extract(image,key)
    elif method == 'ISS':
        raw = iss.extract(image,key)
    elif method == 'SS':
        raw = ss.extract(image,key)
    else:
        raise ValueError(method)
    return (raw.reshape(-1)>127).astype('uint8')
