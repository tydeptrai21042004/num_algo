#!/usr/bin/env python3
"""Run every image attack on the same real marked host and preserve evidence."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import common,images

def audit(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    x=common.load_image(common.ROOT/'data/host/airplane.bmp');b=common.watermark_bits('C')
    rows=images.case_rows(x,b,'FC-AOIP',common.DEFAULT,None,dict(host='airplane',watermark='C',variant='default',key_index=0,method='FC-AOIP'),common.proposal_seed('schedule','airplane',0),('attacks','airplane',0),out/'evidence')
    common.write_csv(out/'attack_audit_quality.csv',[rows[0]]);common.write_csv(out/'attack_audit_rows.csv',rows[1:]);common.atomic_json(out/'protocol.json',common.CONDITIONS)
    good=rows[0]['status']=='ok' and len(rows)==61 and all(r['status']=='ok' for r in rows[1:]);report=dict(success=good,conditions=len(rows)-1,changed=sum(r.get('changed_pixel_ratio',0)>0 for r in rows[1:]),identity_conditions=[r['condition_id'] for r in rows[1:] if r.get('changed_pixel_ratio')==0],protocol_sha256=common.protocol_sha256());common.atomic_json(out/'audit.json',report)
    if not good:raise RuntimeError('Attack audit failed')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/attack_audit');a=p.parse_args();print(audit(a.out))
