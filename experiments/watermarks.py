"""Load the two original bundled logos; no synthetic payload in media benchmarks."""
from functools import lru_cache
from pathlib import Path
import hashlib
import math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WATERMARKS = {'C': 'wm.png', 'CC': 'Cc.logo.circle.png'}
WATERMARK_IDS = tuple(WATERMARKS)

@lru_cache(maxsize=None)
def _logo(name):
    path = ROOT / 'data/watermark' / WATERMARKS[name]
    with Image.open(path) as im:
        if im.size != (64, 64):
            raise ValueError(f'Expected a 64x64 original watermark: {path}')
        rgba = im.convert('RGBA')
        # CC is black with transparent background; dropping alpha destroys it.
        gray = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('L')
        bits = (np.asarray(gray) >= 128).astype(np.uint8)
    if np.unique(bits).size != 2:
        raise ValueError(f'Watermark must have both bit values: {path}')
    bits.setflags(write=False)
    return bits


def watermark_bits(name, length=4096):
    """Row-major logo bits (white=1). Other lengths spatially resample the logo.

    Power-of-two lengths use a balanced rectangular grid; arbitrary lengths
    use a near-square grid and its first `length` samples. Larger sizes are
    resampled logo images, not independent random information or ECC.
    """
    if length < 1:
        raise ValueError('Payload length must be positive')
    logo = _logo(name)
    if length == 4096:
        return logo.ravel().copy()
    w = math.isqrt(length)
    h = (length + w - 1) // w
    if length & (length - 1) == 0:
        w = 1 << ((length.bit_length() - 1) // 2)
        h = length // w
    res = Image.fromarray(logo * 255).resize((w, h), Image.Resampling.NEAREST)
    return (np.asarray(res).ravel()[:length] >= 128).astype(np.uint8)


def watermark_for_index(i):
    return WATERMARK_IDS[int(i) % len(WATERMARK_IDS)]


def watermark_manifest():
    return dict(protocol='bundled_C_CC_v1', threshold=128, white_bit=1,
                alpha_background='white', nondefault_lengths='nearest spatial resampling',
                files={name: dict(path='data/watermark/'+file,
                    sha256=hashlib.sha256((ROOT/'data/watermark'/file).read_bytes()).hexdigest(),
                    payload_sha256=hashlib.sha256(watermark_bits(name).tobytes()).hexdigest())
                    for name,file in WATERMARKS.items()})
