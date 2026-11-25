"""Clean video watermarking, FFV1 integrity control and paced streaming; no attacks."""
from .common import *
from .datasets import probe
from .compute import available_memory
from dataclasses import asdict,is_dataclass
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import subprocess, shutil, tempfile, threading, queue, psutil

CODECS={'ffv1':['-c:v','ffv1','-pix_fmt','bgr0']}

def ffmpeg(*args):return ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-filter_threads','1',*map(str,args)]
def frame_bits(k,source,i):return watermark_bits("C",k)
def frame_seed(source,i):return proposal_seed('video_schedule',source)
def serial_public(pub):
    if pub is None:return None
    p=dict(pub)
    if is_dataclass(p.get('key')):p['key']=asdict(p['key'])
    return p

def capture(src):
    cap=cv2.VideoCapture(int(src.split(':')[1]) if src.startswith('camera:') else src)
    if not cap.isOpened():raise OSError(f'Cannot open {src}')
    return cap

def read_exact(stream,n):
    b=bytearray()
    while len(b)<n:
        part=stream.read(n-len(b))
        if not part:break
        b.extend(part)
    if not b:return None
    if len(b)!=n:raise OSError('Truncated raw frame')
    return bytes(b)

def video_input(profile,split='test',explicit=None):
    if explicit:
        path=Path(explicit)
        if profile!='smoke' and path.resolve()==(ROOT/'data/video/synthetic_motion.mp4').resolve():raise ValueError('Synthetic fixture is test-only')
        return [(str(explicit),path.name)]
    if profile=='smoke':return [(str(ROOT/'data/video/synthetic_motion.mp4'),'synthetic_fixture')]
    rows=json.loads((ROOT/'data/manifests/video_sources.json').read_text());out=[]
    for r in rows:
        if r['split']!=split:continue
        p=ROOT/r['path']
        if sha(p)!=r['sha256']:raise RuntimeError(f'Changed video {p}')
        out.append((str(p),p.name))
    if not out:raise RuntimeError(f'No source videos for split {split}')
    return out

def mark_video(src,source,method,h,w,k,frames,output,fps=None,strengths=None):
    """Encode marked RGB into lossless FFV1. No repeated payload, no frame padding."""
    cap=capture(src);native_fps=cap.get(cv2.CAP_PROP_FPS);native_size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];fps=float(fps or (native_fps if native_fps>0 else 30));rows=[];keys=[];mark_hashes=[];output.parent.mkdir(parents=True,exist_ok=True)
    log=output.with_suffix('.ffmpeg.log');wall=time.perf_counter_ns();enc=None
    try:
        with log.open('wb') as err:
            enc=subprocess.Popen(ffmpeg('-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',fps,'-i','pipe:0','-an','-threads','1',*CODECS['ffv1'],output),stdin=subprocess.PIPE,stderr=err)
            for i in range(frames):
                start=time.perf_counter_ns();ok,frame=cap.read();read_ms=elapsed(start)
                if not ok:break
                t=time.perf_counter_ns();x=cv2.cvtColor(cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB);pre=elapsed(t);b=frame_bits(k,source,i);sd=frame_seed(source,i);p=None
                row=dict(frame=i,watermark="C",source=source,method=method,width=w,height=h,payload_bits=k,source_fps=native_fps,output_fps=fps,read_ms=read_ms,preprocess_ms=pre,status='ok',qualified=0,first_frame=i==0)
                try:
                    if i==0:
                        t=time.perf_counter_ns()
                        try:embed_method(method,x,b,sd,strengths=strengths)
                        except ValueError:pass
                        row['warmup_ms']=elapsed(t)
                    t=time.perf_counter_ns();y,p=embed_method(method,x,b,sd,strengths=strengths);row['embed_ms']=elapsed(t);row.update(quality(x,y,b,p,method,DEFAULT));row['status']='ok' if row['qualified'] else 'unqualified'
                except Exception as e:raise RuntimeError(f'Frame {i} embedding failed: {e}') from e
                if i==0:
                    Image.fromarray(x).save(output.parent/'first_original.png');Image.fromarray(y).save(output.parent/'first_marked.png');Image.fromarray(apply_extract(method,y,p).reshape(64,64)*255).save(output.parent/'first_recovered.png')
                mark_hashes.append(pixel_hash(y));keys.append(p);t=time.perf_counter_ns();enc.stdin.write(y.tobytes());row['pipe_write_ms']=elapsed(t);row['capture_to_write_ms']=elapsed(start);row['deadline_ms']=1000/fps;row['algorithm_deadline_miss']=int(row.get('embed_ms',0)+row.get('extract_ms',0)>1000/fps) if 'embed_ms' in row else None;rows.append(row)
            enc.stdin.close();rc=enc.wait();
            if rc:raise RuntimeError(f'FFV1 encoding failed: {log.read_text()[-1000:]}')
    finally:
        cap.release()
        if enc is not None and enc.poll() is None:enc.kill();enc.wait()
    if len(rows)!=frames:raise RuntimeError(f'Expected {frames} source frames, got {len(rows)}')
    write_csv(output.parent/'frame_timing.csv',rows);atomic_json(output.parent/'public_metadata.json',dict(source=source,method=method,payload='original C logo, fixed per-video key; no temporal vote or ECC',watermarks=watermark_manifest(),frame_alignment='sequential frame indices, no synchronization recovery'))
    atomic_json(output.parent/'source_manifest.json',dict(source=src,native_size=native_size,native_fps=native_fps,processing_size=[w,h],output_fps=fps,requested_frames=frames,actual_frames=len(rows),source_exhausted=len(rows)<frames,wall_s=elapsed(wall)/1000,output_sha256=sha(output),lossless_control='FFV1 RGB via bgr0'))
    return rows,keys,mark_hashes,fps

def decode_video(path,source,method,h,w,k,rows,keys,hashes,codec,generation,reference=None):
    if codec!='ffv1':raise ValueError('Video attacks disabled')
    proc=subprocess.Popen(ffmpeg('-threads','1','-i',path,'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE);out=[];t0=time.perf_counter_ns()
    refproc=subprocess.Popen(ffmpeg('-threads','1','-i',reference,'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL) if reference is not None and codec!='ffv1' else None
    try:
        for i,row in enumerate(rows):
            t=time.perf_counter_ns();raw=read_exact(proc.stdout,h*w*3);ms=elapsed(t);r=dict(table='video_codec_frames',source=source,method=method,width=w,height=h,payload_bits=k,codec=codec,generation=generation,frame=i,watermark="C",qualified=row['qualified'],status='missing_frame')
            refraw=read_exact(refproc.stdout,h*w*3) if refproc else None
            if raw is not None:
                rgb=np.frombuffer(raw,np.uint8).reshape(h,w,3);r.update(decode_read_ms=ms,lossless_pixel_match=int(pixel_hash(rgb)==hashes[i]),status='not_embedded')
                if codec=='ffv1':r['psnr_to_marked_db']=float('inf') if r['lossless_pixel_match'] else None
                elif refraw is not None:r['psnr_to_marked_db']=psnr(np.frombuffer(refraw,np.uint8).reshape(h,w,3),rgb)
                if keys[i] is not None:
                    try:
                        b=frame_bits(k,source,i);t=time.perf_counter_ns();pred=apply_extract(method,rgb,keys[i]);r['post_extract_ms']=elapsed(t);r.update(accuracy(b,pred),status='ok')
                    except Exception as e:r.update(status='error',error=str(e))
            out.append(r)
        extra=read_exact(proc.stdout,h*w*3)
        if extra is not None:
            proc.kill();proc.wait();out.append(dict(table='video_codec_frames',source=source,method=method,codec=codec,generation=generation,status='extra_frames'))
        else:
            stderr=proc.stderr.read();rc=proc.wait()
            if rc:out.append(dict(table='video_codec_frames',source=source,method=method,codec=codec,generation=generation,status='decode_error',error=stderr.decode()[-500:]))
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        proc.stdout.close();proc.stderr.close()
        if refproc is not None:
            if refproc.poll() is None:refproc.kill();refproc.wait()
            refproc.stdout.close()
    return out,elapsed(t0)/1000

def codec_case(store,src,source,method,h,w,k,frames,profile,codecs,keep=False,strengths=None):
    if codecs!=['ffv1']:raise ValueError('Only lossless FFV1 integrity control is permitted')
    key=('lossless',source,method,h,w,k,frames);case_id=digest(key)
    if store.has(key):return
    job=store.out/'artifacts'/case_id[:16];job.mkdir(parents=True,exist_ok=True)
    native=probe(src);base=dict(source=source,method=method,width=w,height=h,native_width=int(native['width']),native_height=int(native['height']),upscaled=int(w>int(native['width']) or h>int(native['height'])),payload_bits=k,case_id=case_id,source_sha256=sha(src),watermark='C')
    collected=[];lossless=job/'marked_ffv1.mkv'
    try:
        rows,keys,hashes,fps=mark_video(src,source,method,h,w,k,frames,lossless,strengths=strengths)
        collected.extend(dict(r,table='video_frame_timing',case_id=case_id,source_sha256=base['source_sha256'],native_width=base['native_width'],native_height=base['native_height'],upscaled=base['upscaled']) for r in rows)
        out,dec_s=decode_video(lossless,source,method,h,w,k,rows,keys,hashes,'ffv1',1)
        collected.extend(dict(r,case_id=case_id,source_sha256=base['source_sha256'],native_width=base['native_width'],native_height=base['native_height'],upscaled=base['upscaled']) for r in out)
        good=len(out)==len(rows) and all(r['status']=='ok' and r.get('lossless_pixel_match')==1 and r.get('ber')==0 for r in out)
        collected.append(dict(base,table='video_codec_summary',codec='ffv1',generation=1,status='ok' if good else 'error',frames=len(rows),qualified_frames=sum(r['qualified'] for r in rows),decoded_frames=len(out),pixel_matches=sum(r.get('lossless_pixel_match',0) for r in out),mean_qualified_ber=float(np.mean([r['ber'] for r in out if 'ber' in r])) if any('ber' in r for r in out) else None,decode_wall_s=dec_s,size_bytes=lossless.stat().st_size))
        if not good:lossless.unlink(missing_ok=True)
    except Exception as e:
        collected.append(dict(base,table='video_codec_summary',codec='ffv1',status='error',error=str(e)))
        lossless.unlink(missing_ok=True)
    finally:
        if not keep:lossless.unlink(missing_ok=True)
    store.put(key,collected)

def paced_stream(src,source,method,h,w,k,fps,frames,seconds=None,loop=False,queue_size=4,streams=1,strengths=None):
    """Real producer/consumer deadline stress. Camera hardware losses cannot be inferred."""
    cap=capture(src);camera=src.startswith('camera:');
    if camera:cap.set(cv2.CAP_PROP_FPS,fps)
    reported_fps=cap.get(cv2.CAP_PROP_FPS);q=queue.Queue(maxsize=queue_size);rows=[];dropped=[];stop=threading.Event();errors=[];proc=psutil.Process();origin=time.monotonic();period=1/fps
    def produce():
        i=0
        try:
            while not stop.is_set() and (i<frames if seconds is None else time.monotonic()-origin<seconds):
                scheduled=origin+i*period
                if not camera:
                    wait=scheduled-time.monotonic()
                    if wait>0:time.sleep(wait)
                start=time.monotonic();ok,frame=cap.read();captured=time.monotonic()
                if not ok and loop and not camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,frame=cap.read();captured=time.monotonic()
                if not ok:break
                try:q.put_nowait((i,scheduled,captured,frame,(captured-start)*1000))
                except queue.Full:dropped.append(dict(frame=i,watermark="C",status='queue_drop',scheduled_s=scheduled-origin,captured_s=captured-origin))
                i+=1
        except Exception as e:errors.append(str(e))
        finally:
            cap.release();q.put(None)
    # Warm kernels outside measured streaming window; no warmup frames silently discarded.
    try:
        x=np.full((h,w,3),128,np.uint8);b=watermark_bits('C',k);embed_method(method,x,b,1,strengths=strengths)
    except Exception:pass
    origin=time.monotonic();thread=threading.Thread(target=produce,daemon=True);thread.start()
    try:
        while True:
            item=q.get()
            if item is None:break
            i,scheduled,captured,frame,read_ms=item;t=time.perf_counter_ns();queue_ms=(time.monotonic()-captured)*1000;x=cv2.cvtColor(cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB);prep=elapsed(t);b=frame_bits(k,source,i);sd=frame_seed(source,i);r=dict(frame=i,watermark="C",status='ok',qualified=0,read_ms=read_ms,preprocess_ms=prep,queue_ms=queue_ms,queue_depth=q.qsize())
            try:
                t=time.perf_counter_ns();y,p=embed_method(method,x,b,sd,strengths=strengths);r['embed_ms']=elapsed(t);t=time.perf_counter_ns();pred=apply_extract(method,y,p);r['extract_ms']=elapsed(t);r.update(clean_ber=ber(b,pred),psnr_db=psnr(x,y));r['qualified']=int(r['clean_ber']==0 and r['psnr_db']>=55-1e-8);r['status']='ok' if r['qualified'] else 'unqualified'
            except Exception as e:r.update(status=error_status(e),error=str(e))
            end=time.monotonic();r.update(capture_to_extract_ms=(end-captured)*1000,scheduled_to_extract_ms=(end-scheduled)*1000,deadline_miss=int(end>(captured if camera else scheduled)+period),rss_mb=proc.memory_info().rss/2**20,cpu_percent=proc.cpu_percent());rows.append(r)
    finally:stop.set();thread.join(timeout=5)
    wall=time.monotonic()-origin;valid=sum(r['qualified'] for r in rows);attempts=len(rows)+len(dropped);lat=[r['capture_to_extract_ms'] for r in rows];summary=dict(table='stream_summary',source=source,method=method,width=w,height=h,payload_bits=k,target_fps=fps,mode='camera' if camera else 'paced_file',looped_input=loop,requested_frames=frames,requested_seconds=seconds,processed=len(rows),qualified=valid,queue_drops=len(dropped),attempts=attempts,queue_drop_rate=len(dropped)/max(attempts,1),deadline_miss_rate=(sum(r['deadline_miss'] for r in rows)+len(dropped))/max(attempts,1),wall_s=wall,processed_fps=len(rows)/max(wall,1e-9),successful_fps=valid/max(wall,1e-9),median_ms=float(np.median(lat)) if lat else None,p95_ms=float(np.percentile(lat,95)) if lat else None,p99_ms=float(np.percentile(lat,99)) if lat else None,status='ok' if (attempts==frames if seconds is None else attempts>0) and not errors and all(r['status']=='ok' for r in rows) else 'error',error='; '.join(errors),camera_hardware_drops='unknown',reported_capture_fps=reported_fps,codec_included=False)
    return [dict(table='stream_frames',source=source,method=method,target_fps=fps,**r) for r in rows+dropped]+[summary]

def run_video(profile,out,resume,split='test',explicit=None,frames=None,keep=True,mode='all',seconds=None,loop=False,method_filter=None,fps_list=None,strengths=None):
    sources=video_input(profile,split,explicit);n=frames or 100
    if method_filter and tuple(method_filter)!=('FC-AOIP',):raise ValueError('Video is proposal only')
    if mode not in ('all','codec','stream'):raise ValueError('Only clean video and streaming supported')
    store=Store(out,dict(group='video',profile=profile,split=split,sources=[(s,name,sha(s)) for s,name in sources],frames=n,mode=mode,keep=keep,fps_list=fps_list),resume)
    try:
        for src,source in sources:
            native=probe(src);nh,nw=int(native['height']),int(native['width'])
            sizes=[(nh//8*8,nw//8*8)] if profile=='smoke' else list(dict.fromkeys([(nh//8*8,nw//8*8)]+[(h,round(h*nw/nh/8)*8) for h in ([480,720,1080] if profile=='lite' else [480,720,1080,1440,2160])]))
            if profile=='smoke' and sizes[0][0]*sizes[0][1]//64<4096:sizes=[(512,512)]
            if mode in ('all','codec'):
                for h,w in sizes:
                    if h*w//64<4096:raise ValueError('Video grid too small for 4096 bits')
                    codec_case(store,src,source,'FC-AOIP',h,w,4096,n,profile,['ffv1'],keep=keep and (h,w)==sizes[0])
            if mode in ('all','stream'):
                h,w=sizes[0]
                for fps in (fps_list or ([30] if profile=='smoke' else [24,30,60])):
                    key=('stream',source,h,w,fps,n)
                    if store.has(key):continue
                    rows=paced_stream(src,source,'FC-AOIP',h,w,4096,fps,n,seconds,loop)
                    store.put(key,[dict(r,case_id=digest(key),width=w,height=h,payload_bits=4096,source_sha256=sha(src)) for r in rows])
    finally:store.close()
