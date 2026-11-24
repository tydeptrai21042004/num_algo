import sys, json, zipfile
from pathlib import Path
import pytest, numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import common,datasets,images,mathematics,video,compute

def test_core_and_attack_protocol():
    assert common.verify_core()
    assert [r['condition_id'] for r in common.CONDITIONS]==['C000']+[f'A{i:03d}' for i in range(1,61)]

def test_hyperparameter_grid_and_split_ids():
    specs=images.specs_for('hyperparameters','paper')
    assert len([v for v in specs if v[1].startswith('B06')])==81
    assert len(specs)==115
    a,k=datasets.split_ids('paper');b,j=datasets.split_ids('lite')
    assert [len(a[x]) for x in ('train','validation','test')]==[100,50,100] and k==24
    assert all(set(a[x]).isdisjoint(a[y]) for x,y in [('train','validation'),('train','test'),('validation','test')])
    assert all(set(b[x])<=set(a[x]) for x in a)

def test_checkpoint_resume_and_reject_change(tmp_path):
    s=common.Store(tmp_path,{'id':1});s.put('a',[{'table':'x','status':'ok','a':1},{'table':'x','status':'error','error':'test'}]);s.close()
    s=common.Store(tmp_path,{'id':1},True);assert s.has('a');s.close()
    assert json.loads((tmp_path/'counts.json').read_text())['tables']['x']==2
    with pytest.raises(RuntimeError,match='fingerprint'):common.Store(tmp_path,{'id':2},True)

def test_download_and_cache_tamper(tmp_path):
    source=tmp_path/'source.bin';source.write_bytes(b'1234567890');target=tmp_path/'cache.bin'
    meta=datasets.download(source.as_uri(),target,expected_sha=common.sha(source));assert meta['sha256']==common.sha(source)
    assert datasets.download(source.as_uri(),target)==meta
    target.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Cache verification'):datasets.download(source.as_uri(),target)

def test_selected_extraction_no_traversal(tmp_path):
    z=tmp_path/'x.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr('../../0001.png',b'one');f.writestr('a/0002.png',b'two')
    paths=datasets.extract_selected(z,[1],tmp_path/'dest');assert paths[0].read_bytes()==b'one' and paths[0].parent==tmp_path/'dest'
    assert not (tmp_path/'0001.png').exists()

def test_split_leakage(tmp_path,monkeypatch):
    monkeypatch.setattr(datasets,'ROOT',tmp_path);d=tmp_path/'data/manifests';d.mkdir(parents=True)
    for i,split in enumerate(('train','validation','test')):
        p=tmp_path/f'{split}.png';Image.fromarray(np.full((16,16,3),i*60,np.uint8)).save(p);common.atomic_json(d/f'{split}.json',[datasets.image_record(p,split,'test fixture')])
    assert datasets.verify_splits()
    p=tmp_path/'test.png';p.write_bytes((tmp_path/'train.png').read_bytes());common.atomic_json(d/'test.json',[datasets.image_record(p,'test','test fixture')])
    with pytest.raises(RuntimeError,match='Duplicate'):datasets.verify_splits()

def test_infeasible_still_has_60_rows(monkeypatch):
    def fail(*args,**kw):raise ValueError('INFEASIBLE synthetic test')
    monkeypatch.setattr(images,'embed_method',fail)
    rows=images.case_rows(np.zeros((8,8,3),np.uint8),np.array([1]),'FC-AOIP',common.DEFAULT,None,{'host':'test'},1,('test',))
    assert len(rows)==61 and rows[0]['status']=='infeasible'
    assert all(r['status']=='not_embedded' for r in rows[1:])

def test_independent_allocation_dp_heap():
    assert mathematics.dp_cost([0,1,2,5],5)==mathematics.heap_cost([0,1,2,5],5)==9
    assert mathematics.dp_cost([1],2)==-1

def test_lossless_video_one_frame(tmp_path):
    import shutil
    if not shutil.which('ffmpeg'):pytest.skip('FFmpeg not installed')
    src=str(ROOT/'data/video/synthetic_motion.mp4');out=tmp_path/'lossless.mkv'
    rows,keys,hashes,fps=video.mark_video(src,'test','FC-AOIP',512,512,4096,1,out)
    got,_=video.decode_video(out,'test','FC-AOIP',512,512,4096,rows,keys,hashes,'ffv1',1)
    assert len(got)==1 and got[0]['lossless_pixel_match']==1 and got[0]['ber']==0

def test_compute_stress_has_8k_and_all_baselines():
    c=list(compute.cases('stress'))
    assert any(x[1:3]==(8192,8192) for x in c)
    assert {x[6] for x in c if x[0]=='C03'}==set(common.METHODS)
