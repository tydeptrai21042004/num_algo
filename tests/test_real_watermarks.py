from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from experiments import common, video
from experiments.watermarks import WATERMARK_IDS, WATERMARKS, watermark_bits, watermark_manifest


def test_original_logo_pixels_and_alpha():
    for name in WATERMARK_IDS:
        with Image.open(common.ROOT/'data/watermark'/WATERMARKS[name]) as image:
            rgba=image.convert('RGBA')
            expected=np.asarray(Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('L'))>=128
        bits=watermark_bits(name)
        assert bits.shape==(4096,) and bits.dtype==np.uint8
        assert np.array_equal(bits.reshape(64,64),expected)
        assert set(bits)=={0,1}
        assert set(watermark_bits(name,64))=={0,1}
        for n in (1,16,64,256,512,1024,2048,4096,8192,129600):
            assert len(watermark_bits(name,n))==n
    assert not np.array_equal(watermark_bits('C'),watermark_bits('CC'))
    assert len(watermark_manifest()['files'])==2


def test_frame_payload_independent_of_key_and_source():
    assert np.array_equal(video.frame_bits(4096,'source1',0),watermark_bits('C'))
    assert np.array_equal(video.frame_bits(4096,'source2',2),watermark_bits('C'))
    assert np.array_equal(video.frame_bits(4096,'source1',1),watermark_bits('C'))
    assert video.frame_seed('source1',0)==video.frame_seed('source1',2)


def test_actual_logo_clean_recovery():
    x=common.load_image(common.ROOT/'data/host/airplane.bmp')
    for name in WATERMARK_IDS:
        b=watermark_bits(name)
        for sd in (42,2026):
            y,p=common.embed_method('FC-AOIP',x,b,sd)
            assert np.array_equal(common.apply_extract('FC-AOIP',y,p),b)
            assert common.psnr(x,y)>=55-1e-8
