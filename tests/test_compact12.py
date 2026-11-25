import sys
from pathlib import Path
from unittest.mock import Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_12_images as runner


def test_split_exactly_twelve_disjoint(tmp_path):
    a=runner.prepare_split(tmp_path);b=runner.prepare_split(tmp_path)
    assert a==b and len(a['train'])==4 and len(a['test'])==8
    assert set(a['train']).isdisjoint(a['test'])
    assert set(a['train']+a['test'])==set((ROOT/'data/host').glob('*.bmp'))


def test_main_routes_calibrated_methods_no_image_download(monkeypatch,tmp_path):
    train_test=runner.prepare_split(tmp_path)
    run=Mock();videos=Mock();frames=Mock();prepare=Mock();security=Mock()
    strengths=runner.common.FIXED_STRENGTHS
    folder=tmp_path/'calibration';folder.mkdir()
    runner.common.atomic_json(folder/'manifest.json',dict(config=dict(hosts={str(p.relative_to(ROOT)):runner.common.sha(p) for p in train_test['train']}),runner=runner.common.source_hash()))
    monkeypatch.setattr(runner.images,'calibration_selection',lambda p:(strengths,{}))
    monkeypatch.setattr(runner.images,'run_images',run)
    monkeypatch.setattr(runner.images,'run_security',security)
    monkeypatch.setattr(runner.video,'run_video',videos)
    monkeypatch.setattr(runner.compute,'run_compute',Mock())
    monkeypatch.setattr(runner,'prepare_videos',prepare)
    monkeypatch.setattr(runner.mathematics,'run_math',Mock())
    monkeypatch.setattr(runner.datasets,'download_images',Mock(side_effect=AssertionError('image download forbidden')))
    from scripts import summarize_results,generate_figures
    monkeypatch.setattr(summarize_results,'summarize',Mock())
    monkeypatch.setattr(generate_figures,'plot',Mock())
    from scripts import statistical_tests
    monkeypatch.setattr(statistical_tests,'analyze',Mock())
    monkeypatch.setattr(sys,'argv',['run_12_images.py','--out',str(tmp_path)])
    from scripts import audit_attacks,audit_results,generate_visual_examples
    monkeypatch.setattr(audit_attacks,'audit',Mock())
    monkeypatch.setattr(audit_results,'audit',Mock())
    monkeypatch.setattr(generate_visual_examples,'render',Mock())
    runner.main()
    assert [c.args[:2] for c in run.call_args_list]==[('calibration','train'),('hyperparameters','train'),('robustness60','test'),('ablation','test'),('payload_robustness','test')]
    for c in run.call_args_list:assert c.kwargs['hosts']==train_test[c.args[1]]
    assert run.call_args_list[2].kwargs['matched_only'] is True
    assert run.call_args_list[2].kwargs['matched']==strengths
    assert videos.call_args.kwargs['method_filter']==runner.PROPOSAL_ONLY
    assert security.call_args.kwargs['method_filter']==runner.PROPOSAL_ONLY
    assert all(c.kwargs['method_filter']==runner.PROPOSAL_ONLY for c in run.call_args_list[3:])
    assert run.call_args_list[2].kwargs['method_filter'] is None
    assert videos.call_args.kwargs['frames']==100
    assert security.call_args.kwargs['hosts']==train_test['test']
    prepare.assert_called_once_with(False)


def test_video_only_downloads_test_sources(monkeypatch,tmp_path):
    monkeypatch.setattr(runner.common,'ROOT',tmp_path)
    calls=[]
    def fake_download(url,p,**kwargs):
        calls.append(p.name);return dict(sha256=p.name,url=url)
    monkeypatch.setattr(runner.datasets,'download',fake_download)
    monkeypatch.setattr(runner.datasets,'probe',lambda p:{})
    runner.prepare_videos()
    assert calls==['vidyo1.webm','vidyo3.webm','vidyo4.webm','FourPeople.webm']


def test_security_and_frame_attacks_actual(tmp_path):
    import pandas as pd
    r=runner
    r.images.run_security('test','paper',tmp_path/'security',False,keys=2,
        hosts=[ROOT/'data/host/Girl.bmp'],method_filter=r.PROPOSAL_ONLY)
    q=pd.read_csv(tmp_path/'security/raw_results.csv')
    assert set(q.method)=={'FC-AOIP'} and len(q)==18 and set(q.status)=={'ok'}
    assert set(q.watermark)=={'C','CC'} and q[q.trial=='correct'].ber.eq(0).all()
    assert not hasattr(r.video,'run_video_frames60')


def test_proposal_only_video_execution(tmp_path):
    import pandas as pd
    from shutil import which
    import pytest
    if not which('ffmpeg'):pytest.skip('FFmpeg missing')
    r=runner
    r.video.run_video('smoke',tmp_path/'video',False,
        explicit=str(ROOT/'data/video/synthetic_motion.mp4'),frames=2,
        method_filter=r.PROPOSAL_ONLY, fps_list=[30])
    q=pd.read_csv(tmp_path/'video/video_codec_summary.csv')
    assert set(q.method)=={'FC-AOIP'} and set(q.codec)=={'ffv1'}
    assert set(q.status)=={'ok'}
    assert set(pd.read_csv(tmp_path/'video/stream_summary.csv').method)=={'FC-AOIP'}


def test_statistics_known_fixture_and_missing_pairs(tmp_path):
    from scripts.statistical_tests import analyze
    import pandas as pd
    folder=tmp_path/'robustness60';folder.mkdir()
    hosts={f'data/host/h{i}.bmp':'fixture' for i in range(8)}
    runner.common.atomic_json(folder/'manifest.json',{'config':{'hosts':hosts,'keys':1}})
    rows=[]
    for m in runner.common.METHODS:
        for p in hosts:
            for w in ('C','CC'):
                for i in range(1,61):
                    rows.append(dict(host=Path(p).stem,key_index=0,watermark=w,condition_id=f'A{i:03}',method=m,
                        variant='default' if m=='FC-AOIP' else 'matched',split='test',
                        status='ok',qualified=int(m!='SS'),ber=.1 if m=='FC-AOIP' else .2))
    pd.DataFrame(rows).to_csv(folder/'attacks.csv',index=False)
    res=analyze(tmp_path,tmp_path/'stats')
    assert len(res)==5
    for x in res:
        if x['baseline']=='SS':assert x['p_sign'] is None and x['qualified_pairs']==0
        else:assert x['paired_hosts']==8 and x['p_sign']==.0078125 and x['p_holm']==.0390625
    assert (tmp_path/'stats/paired_sign_tests.tex').exists()


def test_image_robustness_and_payload_actual(tmp_path):
    import pandas as pd
    r=runner
    hosts=[ROOT/'data/host/airplane.bmp']
    r.images.run_images('robustness60','test','paper',tmp_path/'robustness',keys=1,
        matched=r.common.FIXED_STRENGTHS,hosts=hosts,matched_only=True)
    a=pd.read_csv(tmp_path/'robustness/attacks.csv')
    assert set(a.method)==set(r.common.METHODS) and len(a)==720
    assert a.groupby(['method','watermark']).condition_id.nunique().eq(60).all()
    assert a.groupby(['method','watermark']).size().eq(60).all()
    assert set(a.watermark)=={'C','CC'}
    assert a.groupby("method").seed.nunique().eq(1).all()
    assert a.groupby("condition_id").attack_seed.nunique().eq(1).all()
    assert set(a[a.method=='FC-AOIP'].status)=={'ok'}
    r.images.run_images('payload_robustness','test','paper',tmp_path/'payload',keys=1,
        hosts=hosts,method_filter=r.PROPOSAL_ONLY)
    b=pd.read_csv(tmp_path/'payload/attacks.csv')
    assert set(b.method)=={'FC-AOIP'} and len(b)==720
    assert set(b.payload_bits)=={64,256,512,1024,2048,4096}
    assert b.groupby(['payload_bits','watermark']).size().eq(60).all()
