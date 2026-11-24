#!/usr/bin/env python3
"""Reproducible exact attack-frontier math tests and JPEG-Q50 idealized DCT no-go bounds.

No claim about full JPEG (YCbCr, inverse DCT, clipping, spatial rounding, subsampling).
"""
from __future__ import annotations
import argparse
import csv
import itertools
import math
import os
import random
from pathlib import Path


def qi(z: int, step: int, offset: int = 0) -> int:
    """Monotone integer index: floor((z+offset)/step+1/2), ties upward."""
    assert step >= 1
    return (2 * (int(z) + offset) + step) // (2 * step)


def scalar_cost_table(x: tuple[int, ...], M: int) -> list[int]:
    """Exact min sum_j (y_j-x_j)^2 for every possible integer sum(y)."""
    import heapq
    total = len(x)*M
    base = sum(x)
    result = [10**30] * (total+1)
    result[base] = 0
    # Increasing and decreasing totals have separate exact marginal-cost queues.
    for sign in (-1,+1):
        heap = []
        for j, xi in enumerate(x):
            cap = (M - xi) if sign == +1 else xi
            if cap:
                heapq.heappush(heap, (1,j,cap,1))
        dist = 0
        current = base
        while heap:
            cost, j, cap, level = heapq.heappop(heap)
            current += sign
            dist += cost
            result[current] = dist
            if level < cap:
                heapq.heappush(heap, (2 * (level+1) - 1,j,cap,level+1))
    assert all(v < 10**30 for v in result)
    return result


def signature(u: int, attacks: tuple[tuple[int,int,int,int], ...]) -> tuple[int,...]:
    return tuple(qi(u, qu, ou) for qu,qv,ou,ov in attacks)


def constant_quantizer_intervals(umax: int, attacks: tuple[tuple[int,int,int,int],...]):
    """Enumerate only quantizer breakpoints, not all feature values.

    The index qi(z,q,o) jumps at z=k*q-floor(q/2)-o.
    """
    boundaries={0,umax+1}
    for qu,qv,ou,ov in attacks:
        k=qi(0,qu,ou)
        nxt=(k+1)*qu-(qu//2)-ou
        assert nxt>=1
        while nxt<=umax:
            boundaries.add(nxt)
            nxt+=qu
    bb=sorted(boundaries)
    for a,b in zip(bb[:-1],bb[1:]):
        yield a,b-1,signature(a,attacks)


def frontier_for_signature(sig, attacks, vmax, bit):
    """Exact robust staircase bound on second feature's sum."""
    if bit == 1:
        hi_all = vmax
        for target, (qu,qv,ou,ov) in zip(sig,attacks):
            lo,hi = -1,vmax+1
            # largest v with qi(v) < target; predicate true for low values
            while hi-lo > 1:
                mid=(lo+hi)//2
                if qi(mid,qv,ov) < target:
                    lo=mid
                else:
                    hi=mid
            hi_all=min(hi_all,lo)
        return hi_all
    lo_all=0
    for target,(qu,qv,ou,ov) in zip(sig,attacks):
        lo,hi = -1,vmax+1
        # smallest v with qi(v) >= target, predicate true for high values
        while hi-lo>1:
            mid=(lo+hi)//2
            if qi(mid,qv,ov) >= target:
                hi=mid
            else:
                lo=mid
        lo_all=max(lo_all,hi)
    return lo_all


def optimal_staircase(xu, xv, M, attacks, bit, return_count=False):
    cu=scalar_cost_table(tuple(xu),M)
    cv=scalar_cost_table(tuple(xv),M)
    u0=sum(xu); v0=sum(xv)
    umax=len(xu)*M; vmax=len(xv)*M
    best=None
    intervals=0
    for a,z,sig in constant_quantizer_intervals(umax, attacks):
        intervals+=1
        u=max(a,min(z,u0))
        frontier=frontier_for_signature(sig,attacks,vmax,bit)
        if bit==1:
            if frontier<0: continue
            v=min(v0,frontier)
        else:
            if frontier>vmax: continue
            v=max(v0,frontier)
        value=cu[u]+cv[v]
        if best is None or value<best[0]:
            best=(value,u,v)
    if return_count: return best,intervals,umax+1
    return best


def brute_optimizer(xu,xv,M,attacks,bit):
    best=None
    for yu in itertools.product(range(M+1),repeat=len(xu)):
        U=sum(yu)
        costu=sum((a-b)**2 for a,b in zip(yu,xu))
        for yv in itertools.product(range(M+1),repeat=len(xv)):
            V=sum(yv)
            if not all((qi(U,qu,ou)>qi(V,qv,ov))==bool(bit) for qu,qv,ou,ov in attacks):
                continue
            cost=costu+sum((a-b)**2 for a,b in zip(yv,xv))
            if best is None or cost<best: best=cost
    return best


def run_exact_tests(cases=512,seed=20261123):
    rng=random.Random(seed)
    saved=0
    for case in range(cases):
        M=rng.randint(2,4)
        xu=tuple(rng.randint(0,M) for _ in range(2))
        xv=tuple(rng.randint(0,M) for _ in range(2))
        nt=rng.randint(1,3)
        attacks=tuple((rng.randint(1,5),rng.randint(1,5),rng.randint(-2,2),rng.randint(-2,2)) for _ in range(nt))
        bit=rng.randint(0,1)
        best,k,R=optimal_staircase(xu,xv,M,attacks,bit,True)
        brute=brute_optimizer(xu,xv,M,attacks,bit)
        assert (None if best is None else best[0]) == brute,(case,xu,xv,attacks,bit,best,brute)
        if best is not None:
            _,u,v=best
            assert all((qi(u,qu,ou)>qi(v,qv,ov))==bool(bit) for qu,qv,ou,ov in attacks)
        saved+=(R-k)
    print(f'Exact staircase optimizer: {cases}/{cases} comparisons with exhaustive pixel search PASS')
    print(f'Total integer U states skipped by quantizer signature intervals: {saved}')
    # Verify exact rounding-threshold distances in signed domain.
    for _ in range(1000):
        q=2*rng.randint(1,40)
        z=rng.randint(-1000,1000)
        k=qi(z,q)
        inc=(k+1)*q-q//2-z
        dec=z-(k*q-q//2-1)
        assert qi(z+inc-1,q)==k and qi(z+inc,q)>k
        assert qi(z-dec+1,q)==k and qi(z-dec,q)<k
    print('1,000 exact signed quantizer threshold-boundary checks PASS')


def standard_q50_mode44_lower_bounds(host_dir:Path,out_csv:Path|None):
    try: import numpy as np; import cv2
    except ImportError as exc: raise SystemExit('Image audit needs numpy and opencv-python') from exc
    # DCT(4,4) basis = outer(pattern,pattern) /8. Q50 JPEG standard luminance entry at (4,4) =68.
    p=np.array([1,-1,-1,1,1,-1,-1,1],dtype=np.int64)
    weights=np.outer(p,p)
    q_luma=68
    qnum=8*q_luma # quantizer in integer signed feature S=sum w*G
    D55=math.floor(512*512*3*255**2 * 10**(-55/10))
    entries=[]
    for f in sorted(host_dir.glob('*.bmp')):
        img=cv2.imread(str(f))
        if img is None or img.shape != (512,512,3): continue
        G=img[:,:,1].astype(np.int64)
        feature=(G.reshape(64,8,64,8).transpose(0,2,1,3)*weights).sum(axis=(2,3)).ravel()
        bits=np.random.default_rng(2026).integers(0,2,len(feature))
        # Q(S)=floor(S/qnum+1/2), decoder bit=1 if quantized coefficient >0.
        # If bit=1, S>=qnum/2. If bit=0, S<=qnum/2-1 (ties up).
        needed=np.where(bits==1,np.maximum(0,qnum//2-feature),np.maximum(0,feature-(qnum//2-1)))
        lb=(needed.astype(np.float64)**2).sum()/64.0
        entries.append(dict(image=f.name,minimum_SSE_lower_bound=f'{lb:.4f}',D55_budget=D55,ratio=f'{lb/D55:.4f}',wrong_blocks=int(np.count_nonzero(needed))))
    if out_csv:
        out_csv.parent.mkdir(exist_ok=True,parents=True)
        with open(out_csv,'w',newline='') as fw:
            ww=csv.DictWriter(fw,fieldnames=list(entries[0].keys()))
            ww.writeheader(); ww.writerows(entries)
    print('Idealized independent-green-plane DCT-(4,4) Q50 quantizer sign detector:')
    for a in entries:
        print(f" {a['image']:16s} SSE >= {float(a['minimum_SSE_lower_bound']):12.0f}  budget ratio {a['ratio']:>7s}x")
    print(f'Images certified infeasible under this restricted model: {sum(float(e["minimum_SSE_lower_bound"])>D55 for e in entries)}/{len(entries)}')
    return entries


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host-dir',default=None,type=Path)
    p.add_argument('--out-csv',default=None,type=Path)
    p.add_argument('--cases',default=512,type=int)
    p.add_argument('--all',action='store_true',help='Also run exact integer and capacity audits')
    a=p.parse_args()
    run_exact_tests(a.cases)
    if a.host_dir:
        standard_q50_mode44_lower_bounds(a.host_dir,a.out_csv)
        if a.all:
            exact_green_sign_q50_costs(a.host_dir,Path('q50_sign_exact_integer_costs.csv'))
            run_capacity_audit(a.host_dir,Path('idealized_quantized_sign_capacity.csv'))
            run_sign_capacity_tests(600)
            run_embedding_smoke(a.host_dir)


# The stand-alone image audit above uses a continuous relaxation. The following
# optional function computes the full exact *integer* optimum for that model.
def exact_green_sign_q50_costs(host_dir: Path, out_csv:Path|None=None):
    import numpy as np, cv2
    p=np.array([1,-1,-1,1,1,-1,-1,1],dtype=np.int16)
    w=np.outer(p,p).reshape(1,64)
    rows=[]
    for f in sorted(host_dir.glob('*.bmp')):
        im=cv2.imread(str(f))
        if im is None or im.shape!=(512,512,3): continue
        pixels=im[:,:,1].reshape(64,8,64,8).transpose(0,2,1,3).reshape(-1,64).astype(np.int16)
        sums=np.sum(pixels*w,axis=1,dtype=np.int64)
        bits=np.random.default_rng(2026).integers(0,2,len(sums))
        move_dir=(2*bits-1).astype(np.int16)[:,None]
        required=np.where(bits==1,np.maximum(0,272-sums),np.maximum(0,sums-271)).astype(np.int64)
        favorable_dir=move_dir*w
        capacities=np.where(favorable_dir>0,255-pixels,pixels)
        assert np.all(capacities.sum(axis=1)>=required),f.name
        remaining=required.copy(); exact=np.zeros(len(sums),dtype=np.int64)
        for lev in range(1,256):
            if not np.any(remaining): break
            available=np.sum(capacities>=lev,axis=1,dtype=np.int64)
            used=np.minimum(available,remaining)
            exact+=(2*lev-1)*used
            remaining-=used
        assert np.all(remaining==0)
        budget=math.floor(512*512*3*255**2*10**(-55/10))
        tot=int(exact.sum())
        psnr=10*math.log10((512*512*3*255**2)/tot)
        rows.append(dict(image=f.name,exact_integer_minimum_SSE=tot,min_PSNR_dB=f'{psnr:.5f}',budget_55dB=budget,ratio=f'{tot/budget:.3f}',max_required_increments=int(required.max()),max_used_increment_level=lev))
    if out_csv and rows:
        with open(out_csv,'w',newline='') as fo:
            writer=csv.DictWriter(fo,list(rows[0].keys()));writer.writeheader();writer.writerows(rows)
    for x in rows:
        print(' EXACT',x['image'],x['exact_integer_minimum_SSE'],x['min_PSNR_dB'])
    return rows

def exact_sign_cost_batch(pixels, signed_w, threshold: int, bit: int, max_pixel:int=255):
    """Vectorized exact minimum integer SSE for sign threshold and one raw bit per block.
    signed_w has +/-1 entries, pixels 2-D [block,64].
    """
    import numpy as np
    orig=np.sum(pixels*signed_w,axis=1,dtype=np.int64)
    direction=1 if bit else -1
    n=(np.maximum(0,threshold-orig) if bit else np.maximum(0,orig-(threshold-1)))
    capacity=np.where(direction*signed_w>0,max_pixel-pixels,pixels)
    assert np.all(np.sum(capacity,axis=1,dtype=np.int64)>=n)
    left=n.copy()
    cost=np.zeros(len(n),dtype=np.int64)
    for lev in range(1,max_pixel+1):
        if not np.any(left):break
        active=np.sum(capacity>=lev,axis=1,dtype=np.int64)
        taken=np.minimum(active,left)
        cost+=(2*lev-1)*taken
        left-=taken
    assert np.all(left==0)
    return cost


def run_capacity_audit(host_dir: Path, out_csv:Path|None=None):
    import numpy as np,cv2
    p=np.array([1,-1,-1,1,1,-1,-1,1],dtype=np.int16)
    w=np.outer(p,p).reshape(1,64)
    budget=math.floor(512*512*3*255**2*10**(-55/10))
    rows=[]
    for f in sorted(host_dir.glob('*.bmp')):
        im=cv2.imread(str(f))
        if im is None or im.shape!=(512,512,3):continue
        pixels=im[:,:,1].reshape(64,8,64,8).transpose(0,2,1,3).reshape(-1,64).astype(np.int16)
        for name, qco in [('Q90',14),('Q70',41),('Q50',68)]:
            # Exact one-feature sign decoder after idealized scalar coefficient quantization.
            threshold=4*qco # (1/2) * (8*qco)
            c0=exact_sign_cost_batch(pixels,w,threshold,0)
            c1=exact_sign_cost_batch(pixels,w,threshold,1)
            worst=np.maximum(c0,c1)
            cumulative=np.cumsum(np.sort(worst))
            capacity=int(np.searchsorted(cumulative,budget,side='right'))
            # Fixed public raster ordering, worst-case arbitrary 0/1 payload on first B blocks.
            prefix=np.cumsum(worst)
            raster_capacity=int(np.searchsorted(prefix,budget,side='right'))
            bits=np.random.default_rng(2026).integers(0,2,len(c0))
            random_cost=int(np.sum(np.where(bits,c1,c0)))
            rows.append(dict(image=f.name,idealized_quantizer=name,oracle_best_subset_worst_case_bits=capacity,fixed_raster_worst_case_bits=raster_capacity,random_4096_bits_exact_SSE=random_cost,random_4096_bits_feasible=int(random_cost<=budget),budget=budget))
    if out_csv and rows:
        with open(out_csv,'w',newline='') as fo:
            writer=csv.DictWriter(fo,fieldnames=list(rows[0].keys()));writer.writeheader();writer.writerows(rows)
    for row in rows:
        print(' CAPACITY',row['image'],row['idealized_quantizer'],row['oracle_best_subset_worst_case_bits'],row['fixed_raster_worst_case_bits'],row['random_4096_bits_feasible'])
    return rows

def run_sign_capacity_tests(cases=240, seed=1234):
    import numpy as np
    rng=random.Random(seed)
    for i in range(cases):
        M=rng.randint(2,4)
        n=rng.randint(2,4)
        xs=tuple(rng.randint(0,M) for _ in range(n))
        ws=tuple(rng.choice((-1,+1)) for _ in range(n))
        threshold=rng.randint(-M*n,M*n)
        x=np.array(xs,dtype=np.int16).reshape(1,n)
        w=np.array(ws,dtype=np.int16).reshape(1,n)
        for bit in (0,1):
            ref=None
            for ys in itertools.product(range(M+1),repeat=n):
                sval=sum(v*sgn for v,sgn in zip(ys,ws))
                if (sval>=threshold)!=bool(bit):continue
                cost=sum((v-xi)**2 for v,xi in zip(ys,xs))
                ref=cost if ref is None else min(ref,cost)
            try:
                got=int(exact_sign_cost_batch(x,w,threshold,bit,max_pixel=M)[0])
            except AssertionError:
                got=None
            assert got==ref,(xs,ws,M,threshold,bit,got,ref)
    print(f'{cases} random bounded signed-feature scenarios checked against exhaustive integer projection PASS')



def construct_signed_sign_embedding(pixels, weights, target_bit, threshold=272, max_pixel=255):
    """Exact constructive integer projection for one signed +/-1 block."""
    import numpy as np
    x=np.asarray(pixels,dtype=np.int64)
    w=np.asarray(weights,dtype=np.int64)
    assert x.ndim==w.ndim==1 and np.all(abs(w)==1)
    u=int(np.dot(x,w))
    direction=1 if target_bit else -1
    n=max(0,threshold-u) if target_bit else max(0,u-threshold+1)
    caps=np.where(direction*w>0,max_pixel-x,x)
    assert int(sum(caps))>=n
    t=np.zeros(len(x),dtype=np.int64)
    remaining=n
    for lev in range(1,max_pixel+1):
        if remaining==0:break
        for j,c in enumerate(caps):
            if c>=lev and remaining:
                t[j]+=1
                remaining-=1
    assert remaining==0
    y=x+direction*w*t
    assert np.all(y>=0) and np.all(y<=max_pixel)
    achieved=int(np.dot(y,w))
    assert (qi(achieved,threshold*2)>0)==bool(target_bit)
    return y,int(np.sum((y-x)**2))


def run_embedding_smoke(host_dir:Path,cases=300):
    import numpy as np,cv2
    files=sorted(host_dir.glob('*.bmp'))
    r=random.Random(20261123)
    pattern=np.outer(np.array([1,-1,-1,1,1,-1,-1,1]),np.array([1,-1,-1,1,1,-1,-1,1])).ravel()
    checked=0
    for p in files:
        im=cv2.imread(str(p))
        if im is None:continue
        blocks=im[:,:,1].reshape(64,8,64,8).transpose(0,2,1,3).reshape(-1,64)
        for _ in range(max(1,cases//len(files))):
            x=blocks[r.randrange(len(blocks))].astype(np.int16)
            bit=r.randrange(2)
            y,energy=construct_signed_sign_embedding(x,pattern,bit,threshold=272)
            exact=int(exact_sign_cost_batch(x.reshape(1,-1),pattern.reshape(1,-1),272,bit)[0])
            assert energy==exact
            checked+=1
    print(f'Constructed {checked} 8-bit marked blocks; exact SSE and idealized Q50 detection match PASS')

if __name__=='__main__': main()
