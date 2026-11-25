import numpy as np
from pathlib import Path
from experiments import common,images,video
from fcaoip.kernel import schedule,forward,allcost
from fcaoip import proposal
from src.attack_suite60 import canonical_conditions,attack_rgb,operation

def test_new_attack_registry_and_execution():
    rows=canonical_conditions();assert len(rows)==61 and sum(r.get('reference',False) for r in rows)==45
    x=common.load_image(common.ROOT/'data/host/airplane.bmp');original=x.copy()
    for c in rows:
        y=attack_rgb(x,c,12);assert y.shape==x.shape and y.dtype==np.uint8
        assert np.array_equal(y,attack_rgb(x,c,12))
        if c['operations']:assert not np.array_equal(x,y),c['condition_id']
    assert np.array_equal(x,original)

def test_gaussian_units_and_combination():
    x=np.full((512,512,3),128,np.uint8);c=canonical_conditions()[16];y=attack_rgb(x,c,1)
    assert abs(np.std(y.astype(float)-128)-255*np.sqrt(.003))<.4
    x=common.load_image(common.ROOT/'data/host/airplane.bmp');c=canonical_conditions()[55];rng=np.random.default_rng(7);z=x.copy()
    for name,v in c['operations']:z=operation(z,name,v,rng)
    assert np.array_equal(z,attack_rgb(x,c,7))

def test_optimal_margin_and_key():
    x=common.load_image(common.ROOT/'data/host/airplane.bmp');b=common.watermark_bits('C');key=common.proposal_seed('test');y,info=proposal.embed(x,b,seed=key)
    assert np.array_equal(proposal.extract(y,seed=key),b)
    assert not np.array_equal(schedule(512,512,key),schedule(512,512,key^4096))
    order=schedule(512,512,key);H,ones,caps,alpha=forward(x,b,0,3,order);budget=int(x.size*255**2*10**(-5.5))
    assert allcost(x,b,0,info.margin,H,ones,caps,order,alpha,budget)==info.sse
    assert allcost(x,b,0,info.margin+1,H,ones,caps,order,alpha,budget)>budget
    assert allcost(x,b,0,info.margin,H,ones,caps,order,alpha,2**60,False)>=info.sse

def test_video_no_attack_registry():
    assert set(video.CODECS)=={'ffv1'} and not hasattr(video,'run_video_frames60')
    for i in range(3):assert np.array_equal(video.frame_bits(4096,'sample',i),common.watermark_bits('C'))


def test_fast_ssim_matches_reference():
    from skimage.metrics import structural_similarity
    rng=np.random.default_rng(2);a=rng.integers(0,256,(32,64,3),dtype=np.uint8);b=rng.integers(0,256,a.shape,dtype=np.uint8)
    assert abs(common.image_ssim(a,b)-structural_similarity(a,b,channel_axis=-1,data_range=255))<1e-12
