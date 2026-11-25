#!/usr/bin/env python3
"""12 bundled hosts only: disjoint calibration/test; real video evaluation."""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import common, datasets, images, video, mathematics, compute

PROPOSAL_ONLY = ("FC-AOIP",)


def prepare_split(out):
    hosts = common.get_hosts('legacy', 'paper')
    if len(hosts) != 12:
        raise RuntimeError('Expected exactly 12 bundled BMP hosts')
    random.Random(2026).shuffle(hosts)
    splits = {'train': sorted(hosts[:4]), 'test': sorted(hosts[4:])}
    records = {s: [datasets.image_record(p,s,'bundled legacy BMP') for p in paths]
               for s, paths in splits.items()}
    for field in ('sha256','native_hash','processed_hash'):
        if len({r[field] for rows in records.values() for r in rows}) != 12:
            raise RuntimeError(f'Duplicate bundled image: {field}')
    common.atomic_json(out/'image_split.json', dict(seed=2026, splits=records,
        fingerprint=common.digest(records), previously_used_legacy=True))
    for s, paths in splits.items():
        print(s, [p.name for p in paths], flush=True)
    return splits


def prepare_videos(skip=False):
    # Only the four original real test sources; no video development/validation.
    manifest=common.ROOT/'data/manifests/video_sources.json'
    if skip:
        cached=common.json.loads(manifest.read_text())
        names={name for split,name,url,license,large in datasets.VIDEO_SOURCES if split=='test' and not large}
        records=[r for r in cached if r['split']=='test' and Path(r['path']).name in names]
        if len(records)!=len(names) or {Path(r['path']).name for r in records}!=names:
            raise RuntimeError('Missing required compact test video cache')
        for r in records:
            if common.sha(common.ROOT/r['path'])!=r['sha256']:raise RuntimeError('Changed cached video')
        common.atomic_json(manifest,records)
        return
    records=[]; hashes=set()
    for split,name,url,license,large in datasets.VIDEO_SOURCES:
        if split!='test' or large:
            continue
        p=common.ROOT/'data/videos'/split/name
        m=datasets.download(url,p,min_bytes=10000)
        if m['sha256'] in hashes: raise RuntimeError('Duplicate test video')
        hashes.add(m['sha256'])
        records.append(dict(split=split,path=str(p.relative_to(common.ROOT)),license=license,**m,**datasets.probe(p)))
        print('Verified test video:',name,flush=True)
    common.atomic_json(common.ROOT/'data/manifests/video_sources.json',records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['main','calibration','test','hyperparameters','ablation','payload_robustness','security','video','video_frames60','mathematics','compute','summary','statistics'],default='main')
    p.add_argument('--keys',type=int,default=5)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--skip-download',action='store_true',help='Use previously verified video cache; images are never downloaded')
    p.add_argument('--video-input',type=Path,help='Use one local real video instead of downloading test videos')
    p.add_argument('--video-profile',choices=['lite','paper'],default='paper')
    p.add_argument('--video-frames',type=int,default=100)
    p.add_argument('--attack-frames',type=int,default=10)
    p.add_argument('--out',type=Path,default=common.ROOT/'results/compact12_proposal_only')
    a=p.parse_args()
    if min(a.keys,a.video_frames,a.attack_frames)<1:p.error('Counts must be positive')
    if a.video_input and not a.video_input.is_file():p.error('Local video does not exist')
    common.verify_core();a.out.mkdir(parents=True,exist_ok=True)
    splits=prepare_split(a.out)
    def run(group,split,keys,matched=None,matched_only=False,method_filter=None):
        images.run_images(group,split,'paper',a.out/group,a.resume,keys=keys,
                          matched=matched,hosts=splits[split],matched_only=matched_only,method_filter=method_filter)
    def calibrated():
        folder=a.out/'calibration'
        chosen,details=images.calibration_selection(folder)
        manifest=common.json.loads((folder/'manifest.json').read_text())
        expected={str(p.relative_to(common.ROOT)):common.sha(p) for p in splits['train']}
        if manifest['config']['hosts']!=expected or manifest['runner']!=common.source_hash():
            raise RuntimeError('Calibration does not match this train split or source')
        obj=dict(proposal=common.DEFAULT,strengths=chosen,details=details,
                 image_split_sha256=common.sha(a.out/'image_split.json'),
                 calibration_manifest=common.sha(folder/'manifest.json'),core=common.verify_core(),
                 attack_protocol=common.protocol_sha256())
        common.atomic_json(a.out/'frozen_config.json',dict(obj,fingerprint=common.digest(obj)))
        return chosen
    if a.stage in ('main','mathematics'):
        mathematics.run_math('paper',a.out/'mathematics',a.resume)
    if a.stage in ('main','calibration'):
        run('calibration','train',1)
        calibrated()
    if a.stage in ('main','hyperparameters'):
        run('hyperparameters','train',1)
    if a.stage in ('main','test'):
        run('robustness60','test',a.keys,calibrated(),True)
    for group in ('ablation','payload_robustness'):
        if a.stage in ('main',group):run(group,'test',a.keys,method_filter=PROPOSAL_ONLY)
    if a.stage in ('main','security'):
        images.run_security('test','paper',a.out/'security',a.resume,hosts=splits['test'],method_filter=PROPOSAL_ONLY)
    if a.stage in ('main','video','video_frames60'):
        if not a.video_input:prepare_videos(a.skip_download)
        explicit=str(a.video_input.resolve()) if a.video_input else None
        if a.stage in ('main','video'):
            video.run_video(a.video_profile,a.out/'video',a.resume,split='test',explicit=explicit,frames=a.video_frames,method_filter=PROPOSAL_ONLY)
        if a.stage in ('main','video_frames60'):
            video.run_video_frames60(a.video_profile,a.out/'video_frames60',a.resume,split='test',explicit=explicit,frame_count=a.attack_frames,methods=PROPOSAL_ONLY)
    if a.stage=='compute':compute.run_compute('paper',a.out/'compute',a.resume)
    if a.stage in ('main','summary'):
        from scripts.summarize_results import summarize
        from scripts.generate_figures import plot
        dest=a.out.parent/(a.out.name+'_summary')
        summarize(a.out,dest);plot(a.out,dest/'figures')
    if a.stage in ('main','summary','statistics'):
        from scripts.statistical_tests import analyze
        analyze(a.out, a.out.parent/(a.out.name+'_summary')/'statistics')
    print('Completed stage:',a.stage,flush=True)

if __name__=='__main__':main()
