#!/usr/bin/env python3
"""Audit expected image identities and row counts, video lossless integrity and stage failures."""
import argparse,sys,sqlite3,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.common import atomic_json,verify_core,protocol_sha256,watermark_manifest,ROOT,sha
import pandas as pd
BAD={'error','mismatch','missing_frame','decode_error','extra_frames'}

def audit(root,require_all=True):
    root=Path(root);issues=[];stages={}
    required=['mathematics','calibration','hyperparameters','robustness60','ablation','payload_robustness','security','video','compute']
    if require_all:
        preflight=root/'attack_audit'/'audit.json'
        if not preflight.exists() or not json.loads(preflight.read_text()).get('success'):issues.append('Missing or unsuccessful attack preflight')
        for stage in required:
            if not (root/stage/'manifest.json').is_file():issues.append(f'missing stage {stage}')
    for p in sorted(root.glob('*/manifest.json')):
        folder=p.parent;stage=folder.name;m=json.loads(p.read_text());cfg=m['config'];counts={}
        if m.get('core')!=verify_core() or m.get('attack_protocol_sha256')!=protocol_sha256():issues.append(f'{stage}: core/protocol mismatch')
        if isinstance(cfg.get('hosts'),dict):
            for path,expected_hash in cfg['hosts'].items():
                if not (ROOT/path).exists() or sha(ROOT/path)!=expected_hash:issues.append(f'{stage}: changed input {path}')
        if stage!='mathematics' and cfg.get('watermarks')!=watermark_manifest():issues.append(f'{stage}: watermark mismatch')
        if not (folder/'checkpoint.sqlite3').exists():issues.append(f'{stage}: missing checkpoint');continue
        with sqlite3.connect(folder/'checkpoint.sqlite3') as db:
            if db.execute('pragma quick_check').fetchone()[0]!='ok':issues.append(f'{stage}: corrupt checkpoint')
            for (raw,) in db.execute('select rows from cases'):
                for r in json.loads(raw):
                    status=r.get('status','ok');counts[status]=counts.get(status,0)+1
        if any(counts.get(s,0) for s in BAD):issues.append(f'{stage}: unexpected errors {counts}')
        stages[stage]=counts
        if stage in ('calibration','hyperparameters','robustness60','ablation','payload_robustness'):
            q=pd.read_csv(folder/'quality.csv');a=pd.read_csv(folder/'attacks.csv');key=['host','key_index','watermark','method','variant'];expected={(Path(h).stem,k,w,s[0],s[1]) for h in cfg['hosts'] for k in range(cfg['keys']) for w in ('C','CC') for s in cfg['specs']}
            got=set(q[key].itertuples(index=False,name=None));ids={f'A{i:03}' for i in range(1,61)}
            if got!=expected or q.duplicated(key).any() or len(a)!=len(expected)*60 or a.duplicated(key+['condition_id']).any():issues.append(f'{stage}: missing/duplicate image rows')
            if set(a.condition_id)!=ids or any(set(g.condition_id)!=ids for _,g in a.groupby(key)):issues.append(f'{stage}: attack IDs incomplete')
            if any(r.status=='ok' and r.method=='FC-AOIP' and (r.clean_ber!=0 or float(r.psnr_db)+1e-8<float(r.psnr)) for r in q.itertuples()):issues.append(f'{stage}: proposal invariant violation')
        if stage=='security':
            q=pd.read_csv(folder/'raw_results.csv');expected=len(cfg['hosts'])*len(cfg['methods'])*2
            if len(q[q.trial=='correct'])!=expected:issues.append('security: missing correct-key cases')
            for _,g in q.groupby(['host','watermark','method']):
                if g[g.trial=='correct'].status.eq('ok').any() and (len(g[g.trial=='wrong_key'])!=cfg['keys'] or len(g[g.trial=='unmarked'])!=cfg['keys']):issues.append('security: missing negative trials')
        if stage=='video':
            q=pd.read_csv(folder/'video_codec_summary.csv')
            if set(q.codec)!={'ffv1'} or not q.status.eq('ok').all() or not q.pixel_matches.eq(q.frames).all() or not q.mean_qualified_ber.eq(0).all():issues.append('video: lossless check failed')
            if q.duplicated(['case_id']).any() or not q.frames.eq(cfg['frames']).all():issues.append('video: duplicate cases or missing requested frames')
            frames=pd.read_csv(folder/'video_codec_frames.csv');timing=pd.read_csv(folder/'video_frame_timing.csv')
            for t in (frames,timing):
                if t.duplicated(['case_id','frame']).any() or set(t.watermark)!={'C'} or set(t.case_id)!=set(q.case_id):issues.append('video: frame identities or watermark mismatch')
                if not t.groupby('case_id').size().eq(cfg['frames']).all():issues.append('video: incomplete frame table')
            if (folder/'attacks.csv').exists() or (root/'video_frames60').exists():issues.append('video: forbidden attack outputs')
    result=dict(success=not issues,issues=issues,stages=stages);atomic_json(root/'audit_report.json',result)
    if issues:raise RuntimeError('; '.join(issues))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();print(audit(a.root,not a.partial))
