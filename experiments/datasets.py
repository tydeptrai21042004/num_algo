"""Official-source downloads, deterministic source-level partitions, no synthetic fallback."""
import shutil, subprocess, zipfile, random, tempfile
from .common import *
DIV='https://data.vision.ee.ethz.ch/cvl/DIV2K/'
KOD='https://www.r0k.us/graphics/kodak/kodak/'
XIPH='https://media.xiph.org/video/derf/'
VIDEO_SOURCES=[('development','bunny.mp4','https://media.w3.org/2010/05/bunny/trailer.mp4','Blender Foundation, CC BY 3.0',False),('validation','sintel.mp4','https://media.w3.org/2010/05/sintel/trailer.mp4','Blender Foundation, CC BY 3.0',False),('test','vidyo1.webm',XIPH+'webm/vidyo1_720p_60fps.webm','Xiph, public domain; lossy source',False),('test','vidyo3.webm',XIPH+'webm/vidyo3_720p_60fps.webm','Xiph, public domain; lossy source',False),('test','vidyo4.webm',XIPH+'webm/vidyo4_720p_60fps.webm','Xiph, public domain; lossy source',False),('test','FourPeople.webm',XIPH+'webm/FourPeople_1280x720_60.webm','Xiph, public domain; lossy source',False),('test','crowd_run_2160p50.y4m',XIPH+'y4m/crowd_run_2160p50.y4m','Xiph/SVT; see source terms',True),('test','ducks_take_off_1080p50.y4m',XIPH+'y4m/ducks_take_off_1080p50.y4m','Xiph/SVT; see source terms',True)]

def download(url,path,min_bytes=1,expected_sha=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);meta=path.with_suffix(path.suffix+'.source.json');part=path.with_suffix(path.suffix+'.part')
    if path.exists():
        if not meta.exists():raise RuntimeError(f'Untracked cache: {path}; remove or register it explicitly')
        m=json.loads(meta.read_text());actual=sha(path)
        if m['url']!=url or m['sha256']!=actual or (expected_sha and actual!=expected_sha):raise RuntimeError(f'Cache verification failed: {path}')
        return m
    if shutil.disk_usage(path.parent).free<min_bytes:raise OSError('Insufficient disk space')
    cmd=['curl','--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','20','--speed-time','120','--speed-limit','1024','--continue-at','-','--output',str(part),url]
    run=subprocess.run(cmd)
    if run.returncode==33: # server rejects Range; restart rather than append bad data
        part.unlink(missing_ok=True);subprocess.run(cmd,check=True)
    elif run.returncode:raise RuntimeError(f'Download failed: {url} (curl {run.returncode})')
    if part.stat().st_size<min_bytes:raise RuntimeError(f'Truncated download {url}')
    h=sha(part)
    if expected_sha and h!=expected_sha:raise RuntimeError(f'Checksum mismatch {url}')
    part.replace(path);m=dict(url=url,sha256=h,size_bytes=path.stat().st_size,checksum_origin='publisher' if expected_sha else 'first successful download; subsequent local integrity')
    atomic_json(meta,m);return m

def split_ids(profile):
    ids=list(range(1,801));random.Random(2026).shuffle(ids)
    n,m,k,kod=(6,3,6,4) if profile=='lite' else (100,50,100,24)
    return {'train':sorted(ids[:n]),'validation':sorted(ids[100:100+m]),'test':list(range(801,801+k))},kod

def extract_selected(archive,ids,dest):
    dest.mkdir(parents=True,exist_ok=True);result=[]
    with zipfile.ZipFile(archive) as z:
        lookup={}
        for i in z.infolist():
            name=Path(i.filename).name
            if name.lower().endswith('.png') and name[:-4].isdigit():
                if name in lookup:raise ValueError(f'Duplicate archive member {name}')
                lookup[name]=i
        for ix in ids:
            name=f'{ix:04d}.png';entry=lookup[name];p=dest/('div2k_'+name)
            if entry.file_size>100_000_000 or shutil.disk_usage(dest).free<entry.file_size:raise RuntimeError('Unsafe archive member or insufficient disk')
            # Use chosen flat names, not archive paths. zipfile validates CRC on read.
            with z.open(entry) as src, p.with_suffix('.part').open('wb') as dst:shutil.copyfileobj(src,dst)
            p.with_suffix('.part').replace(p);result.append(p)
    return result

def image_record(p,split,source):
    with Image.open(p) as im:im.verify()
    native=load_image(p,None);processed=load_image(p)
    return dict(path=str(p.relative_to(ROOT)),split=split,source=source,sha256=sha(p),native_hash=pixel_hash(native),processed_hash=pixel_hash(processed),width=native.shape[1],height=native.shape[0],preprocessing='RGB, resize 512x512 Lanczos')

def verify_splits():
    seen={k:{} for k in ('sha256','native_hash','processed_hash')};manifest={}
    for split in ('train','validation','test'):
        rows=json.loads((ROOT/'data/manifests'/f'{split}.json').read_text());manifest[split]=rows
        if not rows:raise ValueError(f'Empty {split}')
        for r in rows:
            p=ROOT/r['path'];actual=image_record(p,split,r['source'])
            if any(r[k]!=actual[k] for k in seen):raise RuntimeError(f'Modified image {p}')
            for k,s in seen.items():
                if r[k] in s:raise RuntimeError(f'Duplicate image in {split} and {s[r[k]]}: {p}')
                s[r[k]]=split
    # Legacy data was previously used to develop the proposal. It is not an unseen test set.
    for p in sorted((ROOT/'data/host').glob('*.bmp')):
        for k,h in [('sha256',sha(p)),('native_hash',pixel_hash(load_image(p,None))),('processed_hash',pixel_hash(load_image(p)))]:
            if h in seen[k]:raise RuntimeError(f'Independent split duplicates legacy image: {p}')
    return digest(manifest)

def download_images(profile):
    if profile=='smoke':raise ValueError('Smoke is offline legacy-only; use lite or paper for independent data')
    ids,kod=split_ids(profile);cache=ROOT/'data/cache';data=ROOT/'data/images';sources={}
    for name in ('train','valid'):
        cache.mkdir(parents=True,exist_ok=True)
        if not (cache/f'DIV2K_{name}_HR.zip').exists() and shutil.disk_usage(cache).free < (8 if name=='train' else 2)*2**30:raise OSError('Insufficient disk space for DIV2K archives')
        f=f'DIV2K_{name}_HR.zip';p=cache/f;sources[name]=download(DIV+f,p,min_bytes=1000000)
        if not zipfile.is_zipfile(p):raise RuntimeError(f'Invalid ZIP: {p}; remove corrupt cache and retry')
    for split,subset in ids.items():
        archive=cache/f'DIV2K_{"valid" if split=="test" else "train"}_HR.zip';paths=extract_selected(archive,subset,data/split)
        rows=[image_record(p,split,DIV+archive.name) for p in paths]
        if split=='test':
            for i in range(1,kod+1):
                name=f'kodim{i:02d}.png';p=data/split/name;download(KOD+name,p,min_bytes=1000);rows.append(image_record(p,split,KOD+name))
        atomic_json(ROOT/'data/manifests'/f'{split}.json',rows)
    fingerprint=verify_splits();atomic_json(ROOT/'data/manifests/image_sources.json',dict(profile=profile,split_seed=2026,sources=sources,counts={k:len(v)+(kod if k=='test' else 0) for k,v in ids.items()},fingerprint=fingerprint))
    print('Verified independent image splits:',fingerprint,flush=True)

def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_format','-of','json',str(path)]));s=data['streams'][0];return {k:s.get(k) for k in ('width','height','codec_name','pix_fmt','r_frame_rate','avg_frame_rate','nb_frames','duration')}

def download_videos(profile):
    if profile=='smoke':raise ValueError('Smoke uses only bundled synthetic diagnostic')
    records=[];hashes=set()
    for split,name,url,license,large in VIDEO_SOURCES:
        if large and profile!='stress':continue
        if large and shutil.disk_usage(ROOT/'data').free<8*2**30:raise OSError('Stress video download requires at least 8 GiB free for source cache')
        if profile=='lite' and name in ('vidyo3.webm','FourPeople.webm'):continue
        p=ROOT/'data/videos'/split/name;m=download(url,p,min_bytes=10000);v=probe(p)
        if m['sha256'] in hashes:raise RuntimeError('Duplicate source video')
        hashes.add(m['sha256']);records.append(dict(split=split,path=str(p.relative_to(ROOT)),license=license,**m,**v));print('Verified',name,flush=True)
    atomic_json(ROOT/'data/manifests/video_sources.json',records)
