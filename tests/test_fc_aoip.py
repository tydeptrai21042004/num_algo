import math
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from fcaoip.proposal import embed,extract,minimum_orbit_margin,quant_index,check_restricted_orbit,features
from fcaoip.evaluation import METHODS,BASELINES,apply_embed,apply_extract,psnr,ber,nc01
from fcaoip.frontier_exact_reference import run_exact_tests,optimal_staircase,brute_optimizer

@pytest.fixture(scope='module')
def small_image():
    return np.random.default_rng(12).integers(30,220,(64,64,3),dtype=np.uint8)

@pytest.fixture(scope='module')
def host():
    from PIL import Image
    return np.asarray(Image.open(ROOT/'data/host/Girl.bmp').convert('RGB'))

def test_exact_uniform_orbit_margin():
    for qs in ((1,), (2,), (2,4), (4,8,12), (3,7)):
        for offsets in ((0,),(-1,0,1),(-3,2)):
            m=minimum_orbit_margin(qs,offsets)
            assert all(quant_index(m,q,o)>=1 and quant_index(-m,q,o)<=-1 for q in qs for o in offsets)
            if m>0:assert any(quant_index(m-1,q,o)<1 or quant_index(-m+1,q,o)>-1 for q in qs for o in offsets)

def test_quantizer_negative_rounding():
    assert quant_index(-1,2,0)==0
    assert quant_index(-2,2,0)==-1
    assert quant_index(2,4,0)==1
    assert quant_index(-3,4,0)==-1

def test_frontier_exhaustive_64():
    run_exact_tests(cases=64,seed=20261123)

def test_restricted_frontier_specific():
    attacks=((2,3,0,1),(3,2,-1,0))
    xu=(1,3);xv=(2,1);M=4
    for bit in (0,1):
        got=optimal_staircase(xu,xv,M,attacks,bit)
        brute=brute_optimizer(xu,xv,M,attacks,bit)
        assert (got[0] if got else None)==brute

def test_small_clean_exact_and_orbit(small_image):
    b=np.random.default_rng(3).integers(0,2,64,dtype=np.uint8)
    y,info=embed(small_image,b,seed=42,psnr_floor_db=25)
    assert ber(b,extract(y,mode=info.mode,seed=42,length=64))==0
    assert check_restricted_orbit(y,b,info)
    assert psnr(small_image,y)>=25
    assert info.sse==int(np.sum((small_image.astype(np.int32)-y.astype(np.int32)).astype(np.int64)**2))

def test_full_4096_clean_psnr55(host):
    b=np.random.default_rng(33).integers(0,2,4096,dtype=np.uint8)
    y,info=embed(host,b,seed=2026)
    assert info.sse<=math.floor(host.size*255**2*10**(-55/10))
    assert np.array_equal(extract(y,mode=info.mode,seed=2026,length=4096),b)
    assert check_restricted_orbit(y,b,info)

def test_public_mode_required(host):
    with pytest.raises(ValueError):extract(host,mode=3,length=4096)

def test_infeasibility_not_silently_ignored():
    x=np.full((64,64,3),128,np.uint8)
    b=np.random.default_rng(2).integers(0,2,64,dtype=np.uint8)
    with pytest.raises(ValueError,match='INFEASIBLE'):
        embed(x,b,seed=4,psnr_floor_db=95)

def test_baseline_registry_exactly_five():
    assert METHODS==('FC-AOIP','Scalar-QIM','STDM-QIM','SCS','ISS','SS')
    assert len(BASELINES)==5

def test_baseline_output_shapes(host):
    b=np.random.default_rng(7).integers(0,2,4096,dtype=np.uint8)
    for m in BASELINES:
        y,key=apply_embed(m,host,b,seed=25)
        pred=apply_extract(m,y,key)
        assert y.shape==host.shape and y.dtype==np.uint8
        assert pred.shape==(4096,)
        assert 0<=ber(b,pred)<=1 and math.isfinite(psnr(host,y))

def test_feature_sign_in_clean_image(host):
    b=np.random.default_rng(17).integers(0,2,4096,dtype=np.uint8)
    for policy in ('dct44','checker'):
        try:y,info=embed(host,b,seed=19,mode_policy=policy)
        except ValueError as e:
            assert 'INFEASIBLE' in str(e)
            continue
        f=features(y,mode=info.mode,seed=19,length=4096)
        assert np.all((f>0)==(b>0))
