"""Experimental infrastructure; no changes to embedding or attack kernels."""
import os
# Set before NumPy, OpenCV and BLAS imports in entrypoints.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]=os.environ.get('FCAOIP_THREADS','1')
import csv, hashlib, json, math, platform, sqlite3, sys, time, resource
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import cv2
from PIL import Image
from .watermarks import WATERMARK_IDS, watermark_bits, watermark_for_index, watermark_manifest
cv2.setNumThreads(int(os.environ.get('FCAOIP_THREADS','1')))
from fcaoip import proposal
from fcaoip.evaluation import METHODS, FIXED_STRENGTHS, apply_embed, apply_extract, psnr, ber, nc01
from src.attack_suite60 import canonical_conditions, attack_rgb, protocol_sha256
DEFAULT=dict(psnr=55.,steps=[2,4],offsets=[-1,0,1],policy='adaptive',max_margin=512)
CONDITIONS=canonical_conditions()
assert len(CONDITIONS)==61
PROFILES={'smoke':dict(keys=1,hosts=1,reps=3,warm=1,rounds=1,frames=3), 'lite':dict(keys=1,hosts=6,reps=20,warm=3,rounds=2,frames=300), 'paper':dict(keys=5,hosts=None,reps=200,warm=20,rounds=5,frames=1000), 'stress':dict(keys=5,hosts=None,reps=200,warm=20,rounds=5,frames=10000)}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def seed(*parts):return int(digest(parts)[:8],16)&0x7fffffff
def bits_for(n,*parts):return np.random.default_rng(seed(*parts)).integers(0,2,n,dtype=np.uint8)
def clean_json(obj):
    if isinstance(obj,dict):return {k:clean_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean_json(v) for v in obj]
    if isinstance(obj,np.generic):return clean_json(obj.item())
    if isinstance(obj,float) and not math.isfinite(obj):return 'inf' if obj>0 else '-inf' if obj<0 else None
    return obj

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean_json(obj),indent=2,sort_keys=True,default=str,allow_nan=False));tmp.replace(path)

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r));tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def core_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for d in ('fcaoip','baselines','src') for p in sorted((ROOT/d).rglob('*.py'))}
def verify_core():
    expected=json.loads((ROOT/'configs/frozen_core.json').read_text());actual=core_hashes()
    if actual!=expected:raise RuntimeError('Frozen proposal, baselines or attack sources changed')
    return digest(actual)
def environment():
    packages={}
    for p in ['numpy','numba','opencv-python-headless','Pillow','scipy','pandas','matplotlib','psutil']:
        try:packages[p]=version(p)
        except PackageNotFoundError:packages[p]='not installed'
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),cpu)
    return dict(python=sys.version,platform=platform.platform(),cpu=cpu,logical_cpus=os.cpu_count(),threads=cv2.getNumThreads(),packages=packages,affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)
def source_hash():return digest({str(p.relative_to(ROOT)):sha(p) for d in ('experiments','scripts') for p in sorted((ROOT/d).glob('*.py'))})

class Store:
    """Atomic case commits. A completed case contains all expected rows, including failures."""
    def __init__(self,out,config,resume=False):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        if config.get('group') != 'mathematics':
            config={**config, 'watermarks':watermark_manifest()}
        self.manifest=dict(config=config,core=verify_core(),attack_protocol_sha256=protocol_sha256(),runner=source_hash(),environment=environment())
        self.fingerprint=digest(self.manifest);p=self.out/'manifest.json'
        if p.exists():
            if not resume:raise RuntimeError(f'{self.out} already exists; use --resume or new --out')
            if json.loads(p.read_text())['fingerprint']!=self.fingerprint:raise RuntimeError('Resume fingerprint mismatch; choose new output directory')
        else:atomic_json(p,dict(self.manifest,fingerprint=self.fingerprint))
        self.db=sqlite3.connect(self.out/'checkpoint.sqlite3');self.db.execute('CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, rows TEXT NOT NULL)');self.db.commit()
    def has(self,key):return self.db.execute('SELECT 1 FROM cases WHERE id=?',(digest(key),)).fetchone() is not None
    def put(self,key,rows):
        with self.db:self.db.execute('INSERT OR REPLACE INTO cases VALUES (?,?)',(digest(key),json.dumps(clean_json(rows),allow_nan=False)))
    def export(self):
        # Two passes to stream results and avoid retaining hundreds of thousands of rows.
        fields={};counts={}
        for (raw,) in self.db.execute('SELECT rows FROM cases ORDER BY rowid'):
            for r in json.loads(raw):
                name=r.get('table','raw_results');fields.setdefault(name,{}) .update(dict.fromkeys(k for k in r if k!='table'));counts[name]=counts.get(name,0)+1
        files={};writers={};failure_count=0
        try:
            for name,cols in fields.items():
                f=(self.out/(name+'.csv.tmp')).open('w',newline='');files[name]=f;writers[name]=csv.DictWriter(f,fieldnames=list(cols));writers[name].writeheader()
            with (self.out/'failures.jsonl.tmp').open('w') as fail:
                for (raw,) in self.db.execute('SELECT rows FROM cases ORDER BY rowid'):
                    for r in json.loads(raw):
                        name=r.pop('table','raw_results');writers[name].writerow(r)
                        if r.get('status','ok')!='ok':fail.write(json.dumps(r)+'\n');failure_count+=1
        finally:
            for f in files.values():f.close()
        for name in fields:(self.out/(name+'.csv.tmp')).replace(self.out/(name+'.csv'))
        (self.out/'failures.jsonl.tmp').replace(self.out/'failures.jsonl');atomic_json(self.out/'counts.json',dict(tables=counts,failure_rows=failure_count,cases=self.db.execute('SELECT count(*) FROM cases').fetchone()[0]))
    def close(self):self.export();self.db.close()

def load_image(path,size=(512,512)):
    with Image.open(path) as im:
        im=im.convert('RGB');im=im.resize(size,Image.Resampling.LANCZOS) if size and im.size!=size else im
        return np.asarray(im,dtype=np.uint8).copy()
def pixel_hash(x):return hashlib.sha256(str(x.shape).encode()+x.tobytes()).hexdigest()
def get_hosts(split,profile='paper',limit=None):
    if split=='legacy': paths=sorted((ROOT/'data/host').glob('*.bmp'))
    else:
        path=ROOT/'data/manifests'/f'{split}.json'
        if not path.exists():raise FileNotFoundError(f'{path}: run download_images.py first')
        rows=json.loads(path.read_text());paths=[ROOT/r['path'] for r in rows]
        for p,r in zip(paths,rows):
            if sha(p)!=r['sha256']:raise RuntimeError(f'Input image changed: {p}')
    n=limit if limit is not None else (PROFILES[profile]['hosts'] if split=='legacy' or profile=='smoke' else None);paths=paths[:n] if n else paths
    if not paths:raise ValueError(f'Empty split {split}')
    return paths

def embed_method(method,x,b,sd,params=None,strengths=None):
    p={**DEFAULT,**(params or {})}
    if method=='FC-AOIP':
        y,i=proposal.embed(x,b,seed=sd,psnr_floor_db=p['psnr'],orbit_steps=tuple(p['steps']),orbit_offsets=tuple(p['offsets']),mode_policy=p['policy'],max_margin=p['max_margin'])
        return y,dict(method=method,mode=i.mode,seed=sd,length=len(b),info=i.as_dict())
    return apply_embed(method,x,b,seed=sd,psnr=p['psnr'],strengths=strengths)

def accuracy(b,p):
    e=ber(b,p);return dict(ber=e,nc01=nc01(b,p),bipolar_nc=1-2*e,accuracy=1-e)
def elapsed(t):return (time.perf_counter_ns()-t)/1e6
def peak_rss_mb():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def quality(x,y,b,pub,method,p):
    t=time.perf_counter_ns();pred=apply_extract(method,y,pub);ex=elapsed(t);q=psnr(x,y);acc=accuracy(b,pred);d=y.astype(np.int32)-x.astype(np.int32);sse=int(np.square(d,dtype=np.int64).sum());del d
    result=dict(psnr_db=q,sse=sse,mse=sse/x.size,clean_ber=acc['ber'],clean_nc01=acc['nc01'],extract_ms=ex,qualified=int(q>=p['psnr']-1e-8 and acc['ber']==0))
    if method=='FC-AOIP':
        info=pub['info'];v=proposal.features(y,mode=pub['mode'],seed=pub['seed'],length=len(b));m=int(np.min(np.where(b>0,v,-v)));budget=math.floor(x.size*255**2*10**(-p['psnr']/10));result.update(mode=pub['mode'],margin=info['margin'],actual_min_signed_feature=m,orbit_threshold=info['certified_min_margin'],orbit_certified=int(m>=info['certified_min_margin']),budget=budget,budget_remaining=budget-sse,cap_active=int(info['margin']==p['max_margin']))
    return result

def error_status(e):return 'infeasible' if 'INFEASIBLE' in str(e) or 'max_margin below' in str(e) else 'error'
