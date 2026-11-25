"""Actual FFmpeg RGB lossless control, lossy codec tests, paced file and webcam stress."""
from .common import *
from .datasets import probe
from .compute import available_memory
from dataclasses import asdict,is_dataclass
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import subprocess, shutil, tempfile, threading, queue, psutil

CODECS={'ffv1':['-c:v','ffv1','-pix_fmt','bgr0'],'mpeg4':['-c:v','mpeg4','-q:v','3','-pix_fmt','yuv420p']}
for c,enc in [('h264','libx264'),('h265','libx265')]:
    for q in (18,23,28,35):CODECS[f'{c}_{q}']=['-c:v',enc,'-preset','veryfast' if c=='h264' else 'ultrafast','-crf',str(q),'-pix_fmt','yuv420p']+(['-x265-params','pools=1:frame-threads=1:log-level=error'] if c=='h265' else [])
for q in (20,32,44):CODECS[f'vp9_{q}']=['-c:v','libvpx-vp9','-crf',str(q),'-b:v','0','-deadline','realtime','-cpu-used','6','-pix_fmt','yuv420p']
CODECS['h264_23_444']=['-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv444p']
CODECS['h264_lossless_rgb']=['-c:v','libx264rgb','-preset','ultrafast','-crf','0','-pix_fmt','rgb24']
CODECS['av1_32']=['-c:v','libaom-av1','-crf','32','-b:v','0','-cpu-used','8','-pix_fmt','yuv420p']

def ffmpeg(*args):return ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-filter_threads','1',*map(str,args)]
def frame_bits(k,source,i):return bits_for(k,'video_payload',source,i)
def frame_seed(source,i):return seed('video_schedule',source,i)
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
    if explicit:return [(str(explicit),Path(explicit).name)]
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
                row=dict(frame=i,source=source,method=method,width=w,height=h,payload_bits=k,source_fps=native_fps,output_fps=fps,read_ms=read_ms,preprocess_ms=pre,status='ok',qualified=0,first_frame=i==0)
                try:
                    if i==0:
                        t=time.perf_counter_ns()
                        try:embed_method(method,x,b,sd,strengths=strengths)
                        except ValueError:pass
                        row['warmup_ms']=elapsed(t)
                    t=time.perf_counter_ns();y,p=embed_method(method,x,b,sd,strengths=strengths);row['embed_ms']=elapsed(t);row.update(quality(x,y,b,p,method,DEFAULT));row['status']='ok' if row['qualified'] else 'unqualified'
                except Exception as e:y=x;p=None;row.update(status=error_status(e),error=str(e))
                mark_hashes.append(pixel_hash(y));keys.append(p);t=time.perf_counter_ns();enc.stdin.write(y.tobytes());row['pipe_write_ms']=elapsed(t);row['capture_to_write_ms']=elapsed(start);row['deadline_ms']=1000/fps;row['algorithm_deadline_miss']=int(row.get('embed_ms',0)+row.get('extract_ms',0)>1000/fps) if 'embed_ms' in row else None;rows.append(row)
            enc.stdin.close();rc=enc.wait();
            if rc:raise RuntimeError(f'FFV1 encoding failed: {log.read_text()[-1000:]}')
    finally:
        cap.release()
        if enc is not None and enc.poll() is None:enc.kill();enc.wait()
    if not rows:raise RuntimeError('No source frames')
    write_csv(output.parent/'frame_timing.csv',rows);atomic_json(output.parent/'public_metadata.json',dict(source=source,method=method,keys=[serial_public(p) for p in keys],payload='deterministic independent raw bits, no repetition',frame_alignment='sequential frame indices, no synchronization recovery'))
    atomic_json(output.parent/'source_manifest.json',dict(source=src,native_size=native_size,native_fps=native_fps,processing_size=[w,h],output_fps=fps,requested_frames=frames,actual_frames=len(rows),source_exhausted=len(rows)<frames,wall_s=elapsed(wall)/1000,output_sha256=sha(output),lossless_control='FFV1 RGB via bgr0'))
    return rows,keys,mark_hashes,fps

def decode_video(path,source,method,h,w,k,rows,keys,hashes,codec,generation,reference=None):
    proc=subprocess.Popen(ffmpeg('-threads','1','-i',path,'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE);out=[];t0=time.perf_counter_ns()
    refproc=subprocess.Popen(ffmpeg('-threads','1','-i',reference,'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1'),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL) if reference is not None and codec!='ffv1' else None
    try:
        for i,row in enumerate(rows):
            t=time.perf_counter_ns();raw=read_exact(proc.stdout,h*w*3);ms=elapsed(t);r=dict(table='video_codec_frames',source=source,method=method,codec=codec,generation=generation,frame=i,qualified=row['qualified'],status='missing_frame')
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

def encode_codec(source,target,codec):
    t=time.perf_counter_ns();cmd=ffmpeg('-threads','1','-i',source,'-map','0:v:0','-an','-threads','1',*CODECS[codec],target);r=subprocess.run(cmd,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr.decode()[-800:])
    return elapsed(t)/1000

def codec_case(store,src,source,method,h,w,k,frames,profile,codecs,keep=False,strengths=None):
    key=('E03',source,method,h,w,k,frames,codecs)
    if store.has(key):return
    job=store.out/'artifacts'/digest(key)[:16];job.mkdir(parents=True,exist_ok=True);base=dict(source=source,method=method,width=w,height=h,payload_bits=k)
    collected=[];lossless=job/'marked_ffv1.mkv'
    try:
        rows,keys,hashes,fps=mark_video(src,source,method,h,w,k,frames,lossless,strengths=strengths)
        collected.extend(dict(table='video_frame_timing',**r) for r in rows)
        for codec in codecs:
            prev=lossless;generations=3 if profile in ('paper','stress') and codec in ('h264_23','h265_23') else 1
            for g in range(1,generations+1):
                target=job/f'{codec}_{g}.mkv';r=dict(base,table='video_codec_summary',codec=codec,generation=g,status='ok')
                try:
                    sec=0. if codec=='ffv1' else encode_codec(prev,target,codec);actual=lossless if codec=='ffv1' else target
                    out,dec_s=decode_video(actual,source,method,h,w,k,rows,keys,hashes,codec,g,reference=lossless);collected.extend(out);valid=[x for x in out if x.get('qualified') and x.get('status')=='ok'];r.update(encode_wall_s=sec,decode_and_reference_evaluation_wall_s=dec_s,frames=len(rows),qualified_frames=sum(x['qualified'] for x in rows),decoded_frames=sum(x.get('status') in ('ok','not_embedded') for x in out),mean_qualified_ber=float(np.mean([x['ber'] for x in valid])) if valid else None,size_bytes=actual.stat().st_size,bitrate_bps=actual.stat().st_size*8*fps/len(rows),pixel_matches=sum(x.get('lossless_pixel_match',0) for x in out),decoded_qualified_frames=len(valid))
                    if codec=='ffv1' and r['pixel_matches']!=len(rows):r.update(status='error',error='Lossless control changed RGB pixels')
                    if any(x['status'] in ('decode_error','missing_frame','extra_frames','error') for x in out):r['status']='error'
                    if prev!=lossless and not keep:prev.unlink(missing_ok=True)
                    prev=actual
                except Exception as e:r.update(status='error',error=str(e));collected.append(r);break
                collected.append(r)
            if prev!=lossless and not keep:prev.unlink(missing_ok=True)
    except Exception as e:collected.append(dict(base,table='video_codec_summary',status='error',error=str(e)))
    finally:
        if not keep:
            for p in job.glob('*.mkv'):p.unlink(missing_ok=True)
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
                except queue.Full:dropped.append(dict(frame=i,status='queue_drop',scheduled_s=scheduled-origin,captured_s=captured-origin))
                i+=1
        except Exception as e:errors.append(str(e))
        finally:
            cap.release();q.put(None)
    # Warm kernels outside measured streaming window; no warmup frames silently discarded.
    try:
        x=np.full((h,w,3),128,np.uint8);b=bits_for(k,'warm');embed_method(method,x,b,1,strengths=strengths)
    except Exception:pass
    origin=time.monotonic();thread=threading.Thread(target=produce,daemon=True);thread.start()
    try:
        while True:
            item=q.get()
            if item is None:break
            i,scheduled,captured,frame,read_ms=item;t=time.perf_counter_ns();queue_ms=(time.monotonic()-captured)*1000;x=cv2.cvtColor(cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB);prep=elapsed(t);b=frame_bits(k,source,i);sd=frame_seed(source,i);r=dict(frame=i,status='ok',qualified=0,read_ms=read_ms,preprocess_ms=prep,queue_ms=queue_ms,queue_depth=q.qsize())
            try:
                t=time.perf_counter_ns();y,p=embed_method(method,x,b,sd,strengths=strengths);r['embed_ms']=elapsed(t);t=time.perf_counter_ns();pred=apply_extract(method,y,p);r['extract_ms']=elapsed(t);r.update(clean_ber=ber(b,pred),psnr_db=psnr(x,y));r['qualified']=int(r['clean_ber']==0 and r['psnr_db']>=55-1e-8);r['status']='ok' if r['qualified'] else 'unqualified'
            except Exception as e:r.update(status=error_status(e),error=str(e))
            end=time.monotonic();r.update(capture_to_extract_ms=(end-captured)*1000,scheduled_to_extract_ms=(end-scheduled)*1000,deadline_miss=int(end>(captured if camera else scheduled)+period),rss_mb=proc.memory_info().rss/2**20,cpu_percent=proc.cpu_percent());rows.append(r)
    finally:stop.set();thread.join(timeout=5)
    wall=time.monotonic()-origin;valid=sum(r['qualified'] for r in rows);attempts=len(rows)+len(dropped);lat=[r['capture_to_extract_ms'] for r in rows];summary=dict(table='stream_summary',source=source,method=method,width=w,height=h,payload_bits=k,target_fps=fps,mode='camera' if camera else 'paced_file',looped_input=loop,requested_frames=frames,requested_seconds=seconds,processed=len(rows),qualified=valid,queue_drops=len(dropped),attempts=attempts,queue_drop_rate=len(dropped)/max(attempts,1),deadline_miss_rate=(sum(r['deadline_miss'] for r in rows)+len(dropped))/max(attempts,1),wall_s=wall,processed_fps=len(rows)/max(wall,1e-9),successful_fps=valid/max(wall,1e-9),median_ms=float(np.median(lat)) if lat else None,p95_ms=float(np.percentile(lat,95)) if lat else None,p99_ms=float(np.percentile(lat,99)) if lat else None,status='ok' if attempts and not errors else 'error',error='; '.join(errors),camera_hardware_drops='unknown',reported_capture_fps=reported_fps,codec_included=False)
    return [dict(table='stream_frames',source=source,method=method,target_fps=fps,**r) for r in rows+dropped]+[summary]

def run_video(profile,out,resume,split='test',explicit=None,frames=None,keep=False,mode='all',seconds=None,loop=False,method_filter=None,fps_list=None,strengths=None):
    sources=video_input(profile,split,explicit);n=frames or PROFILES[profile]['frames'];methods=method_filter or METHODS
    if any(s.startswith('camera:') for s,_ in sources) and mode!='stream':raise ValueError('Camera requires --video-mode stream')
    store=Store(out,dict(group='video',profile=profile,split=split,sources=[(s,source,sha(s) if Path(s).is_file() else None) for s,source in sources],frames=n,seconds=seconds,mode=mode,loop=loop,keep=keep,methods=methods,fps_list=fps_list,strengths=strengths),resume)
    try:
        if mode in ('all','codec'):
            codecs=['ffv1','h264_23'] if profile=='smoke' else ['ffv1','mpeg4','h264_23','h265_23','vp9_32'] if profile=='lite' else list(CODECS)
            for src,source in sources:
                for method in methods:codec_case(store,src,source,method,512,512,4096,n,profile,codecs,keep,strengths=strengths)
                if 'FC-AOIP' in methods and profile!='smoke':
                    sizes=[(480,856),(720,1280),(1080,1920)] if profile=='lite' else [(480,856),(720,1280),(1080,1920),(1440,2560),(2160,3840)]
                    if profile=='stress':sizes.append((4320,7680))
                    if not src.startswith('camera:'):
                        native=probe(src);nh,nw=int(native['height']),int(native['width'])
                        if nh%8==0 and nw%8==0 and nh*nw//64>=4096:sizes=list(dict.fromkeys(sizes+[(nh,nw)]))
                    for h,w in sizes:
                        for k in sorted({4096,h*w//64}):
                            if h*w*64>available_memory()*.65:store.put(('memory',source,h,w,k),[dict(table='video_codec_summary',source=source,method='FC-AOIP',width=w,height=h,payload_bits=k,status='skipped_memory_guard')]);continue
                            codec_case(store,src,source,'FC-AOIP',h,w,k,min(n,300),profile,['ffv1','h264_23'],keep,strengths=strengths)
        if mode in ('all','stream'):
            for src,source in sources:
                for m in methods:
                    for fps in (fps_list or ([30] if profile=='smoke' else [24,30,60] if profile=='lite' else [24,30,60,120,240])):
                        key=('E04',source,m,fps,n,seconds,loop)
                        if store.has(key):continue
                        try:r=paced_stream(src,source,m,512,512,4096,fps,n,seconds,loop,strengths=strengths);store.put(key,r)
                        except Exception as e:store.put(key,[dict(table='stream_summary',source=source,method=m,target_fps=fps,status='error',error=str(e))])
                if profile=='stress' and 'FC-AOIP' in methods and not src.startswith('camera:'):
                    for h,w in [(720,1280),(1080,1920),(2160,3840)]:
                        for fps in (30,60,120):
                            key=('E04large',source,h,w,fps)
                            if not store.has(key):store.put(key,paced_stream(src,source,'FC-AOIP',h,w,4096,fps,min(n,1000),None,False))
            if profile=='stress' and 'FC-AOIP' in methods:
                src,source=sources[0]
                if not src.startswith('camera:'):
                    for count in (1,2,4,8):
                        key=('E05multi',source,count)
                        if store.has(key):continue
                        start=time.monotonic()
                        with ThreadPoolExecutor(max_workers=count) as pool:
                            futures=[pool.submit(paced_stream,src,f'{source}:stream{i}','FC-AOIP',512,512,4096,30,300,None,False) for i in range(count)];r=[item for f in futures for item in f.result()]
                        sums=[v for v in r if v['table']=='stream_summary'];r.append(dict(table='multi_stream_summary',source=source,streams=count,wall_s=time.monotonic()-start,successful_frames=sum(v['qualified'] for v in sums),attempts=sum(v['attempts'] for v in sums),drops=sum(v['queue_drops'] for v in sums),status='ok' if all(v['status']=='ok' for v in sums) else 'error'));store.put(key,r)
    finally:store.close()


def run_video_frames60(profile,out,resume,split='test',explicit=None,frame_count=None,methods=None,strengths=None):
    from .images import case_rows
    sources=video_input(profile,split,explicit);methods=methods or METHODS;n=frame_count or (1 if profile=='smoke' else 3 if profile=='lite' else 10)
    store=Store(out,dict(group='video_frames60',split=split,profile=profile,sources=[(src,source,sha(src)) for src,source in sources],frames=n,methods=methods,strengths=strengths),resume)
    try:
        for src,source in sources:
            cap=capture(src);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));indices=sorted(set(np.linspace(0,max(total-1,0),n,dtype=int))) if total>0 else list(range(n))
            try:
                for index in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
                    if not ok:
                        store.put(('read',source,int(index)),[dict(table='source_failures',source=source,frame=int(index),status='error',error='Frame unavailable')]);continue
                    x=cv2.cvtColor(cv2.resize(frame,(512,512),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB);host=f'{source}:{int(index)}';sd=frame_seed(source,int(index));b=frame_bits(4096,source,int(index))
                    for method in methods:
                        key=(source,int(index),method)
                        if store.has(key):continue
                        base=dict(experiment='video_frames60',split=split,source=source,frame=int(index),host=host,key_index=0,seed=sd,method=method,variant='default',payload_bits=len(b))
                        store.put(key,case_rows(x,b,method,DEFAULT,strengths,base,sd,('video_attacks',source,int(index))))
            finally:cap.release()
    finally:store.close()
