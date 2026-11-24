"""Balanced fixed 60-attack protocol: 5 families x 4 attacks x 3 levels.

All methods receive identical conditions and deterministic stochastic attack draws.
The clean condition C000 is separate and is NOT one of the 60 attacks.

This module keeps the original file ``cirmm_attacks.py`` intact for provenance.
It reuses its published-method-agnostic attack implementations, except AWGN,
which is now explicitly parameterized as pixel standard deviation sigma.

Geometric attacks are evaluated without oracle re-registration in the primary
experiment, and therefore measure alignment failure AND payload fragility.
CIRMM translation-pilot registration is reported in a separate table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from cirmm_attacks import ATTACKS, attack as legacy_attack

VERSION = 'CIRMM_BALANCED_5x4x3_v2_pixel_AWGN'
LEVELS = ('Mild', 'Moderate', 'Strong')
FAMILIES = ('Compression', 'Noise', 'Filtering', 'Geometry', 'Photometric')
ATTACKS_60 = {
    family: {
        (('AWGN pixel sigma' if name == 'Gaussian variance [0,1]' else name)):
            ((2.0, 5.0, 10.0) if name == 'Gaussian variance [0,1]' else tuple(values))
        for name, values in methods.items()
    }
    for family, methods in ATTACKS.items()
}
assert tuple(ATTACKS_60) == FAMILIES
assert all(len(subtypes) == 4 for subtypes in ATTACKS_60.values())
assert all(len(levels) == 3 for subtypes in ATTACKS_60.values() for levels in subtypes.values())


def canonical_conditions() -> list[dict]:
    rows = [dict(condition_id='C000', category='Clean', attack='none',
                 level='Clean', severity_rank=0, parameter='', attack_index=0)]
    idx = 0
    for family, methods in ATTACKS_60.items():
        for name, values in methods.items():
            for rank, (level, param) in enumerate(zip(LEVELS, values), 1):
                idx += 1
                rows.append(dict(condition_id=f'A{idx:03d}', category=family,
                                 attack=name, level=level, severity_rank=rank,
                                 parameter=repr(param), attack_index=idx))
    assert idx == 60
    assert len({c['condition_id'] for c in rows}) == 61
    for family in FAMILIES:
        assert sum(c['category'] == family for c in rows) == 12
    for level in LEVELS:
        assert sum(c['level'] == level for c in rows) == 20
    return rows


def protocol_sha256() -> str:
    implementation = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('attack_suite60.py', 'cirmm_attacks.py')
    }
    body = {'version': VERSION, 'conditions': canonical_conditions(), 'source_sha256': implementation}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def attack_rgb(rgb: np.ndarray, cond: dict, seed: int) -> np.ndarray:
    """Return fixed-size uint8 RGB; stochastic masks/noise are shared via seed.

    Do not change image geometry dimensions; original attack semantics are used:
    cropped regions are resized back, and rotations preserve the 512x512 canvas.
    """
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError('Input must be uint8 RGB with three channels')
    name = cond['attack']
    if name == 'none':
        result = a.copy()
    elif name == 'AWGN pixel sigma':
        sigma = float(ATTACKS_60['Noise'][name][LEVELS.index(cond['level'])])
        noise = np.random.default_rng(int(seed)).normal(0.0, sigma, size=a.shape)
        result = np.clip(np.rint(a.astype(np.float64) + noise), 0, 255).astype(np.uint8)
    else:
        bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        observed = legacy_attack(bgr, name, cond['level'], seed=int(seed))
        if observed.shape != bgr.shape or observed.dtype != np.uint8:
            raise AssertionError(f'{name}: attack returned bad shape or type')
        result = cv2.cvtColor(observed, cv2.COLOR_BGR2RGB)
    if result.shape != a.shape or result.dtype != np.uint8:
        raise AssertionError(f'{name}: RGB contract violated')
    return result
