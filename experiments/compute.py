"""Single-call latency distributions, large-image stress, endurance and independent workers."""
from .common import *
from .mathematics import synthetic,KINDS
from fcaoip.kernel import run_embed,decode
from concurrent.futures import ProcessPoolExecutor
import psutil, random

def timing_case(x,b,method,p,sd,reps,warm,round_idx=0,kernel=False,strengths=None):
    rows=[]
    def call():
        if not kernel:return embed_method(method,x,b,sd,p,strengths)
        budget=math.floor(x.size*255**2*10**(-p['psnr']/10));minimum=proposal.minimum_orbit_margin(p['steps'],p['offsets'])
        y,m,s=run_embed(x,b,0,3,sd,budget,minimum)
        if m<0:raise ValueError('DISTORTION_LIMIT kernel')
        return y,dict(seed=sd,length=len(b),info={'margin':m,'certified_min_margin':minimum})
    cold=time.perf_counter_ns()
    try:
        y,pub=call();first=elapsed(cold);apply_extract(method,y,pub)
        for _ in range(warm):call()
    except Exception as e:return [dict(status=error_status(e),error=str(e),first_call_ms=elapsed(cold),sample=-1)]
    for i in range(reps):
        try:
            t=time.perf_counter_ns();y,pub=call();em=elapsed(t);t=time.perf_counter_ns();pred=apply_extract(method,y,pub);ex=elapsed(t);q=psnr(x,y);be=ber(b,pred)
            rows.append(dict(sample=i,status='ok' if q>=p['psnr']-1e-8 and be==0 else 'unqualified',first_call_ms=first,embed_ms=em,extract_ms=ex,api_total_ms=em+ex,psnr_db=q,clean_ber=be,peak_rss_mb=peak_rss_mb(),rss_mb=psutil.Process().memory_info().rss/2**20,megapixels_per_s=x.shape[0]*x.shape[1]/(max(em,1e-9)*1000),bits_per_s=len(b)*1000/max(em,1e-9)))
        except Exception as e:rows.append(dict(sample=i,status=error_status(e),error=str(e)))
    return rows

def available_memory():
    available=psutil.virtual_memory().available
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip();used=int(Path('/sys/fs/cgroup/memory.current').read_text())
        if limit!='max':available=min(available,max(0,int(limit)-used))
    except (OSError,ValueError):pass
    return available

def cases(profile):
    sizes=[(64,64),(128,128),(256,256),(512,512),(1024,1024),(2048,2048),(2160,3840),(4096,4096)]
    if profile=='stress':sizes.extend([(4320,7680),(8192,8192)])
    if profile=='smoke':sizes=[(64,64),(512,512)]
    if profile=='lite':sizes=sizes[:5]
    for h,w in sizes:
        cap=h*w//64
        for k in sorted({min(cap,4096),cap}):yield ('C01',h,w,k,'photo',DEFAULT,'FC-AOIP',False)
    for k in ([1,64,4096] if profile=='smoke' else [1,16,64,256,512,1024,2048,4096]):yield ('C02',512,512,k,'photo',DEFAULT,'FC-AOIP',False)
    for h,w in ([(512,512)] if profile=='smoke' else [(512,512),(1024,1024),(2160,3840)]):
        for rho in ([.1,1.] if profile=='smoke' else [.01,.05,.1,.25,.5,.75,1.]):yield ('C02_density',h,w,max(1,int(h*w//64*rho)),'photo',DEFAULT,'FC-AOIP',False)
    for m in METHODS:yield ('C03',512,512,4096,'photo',DEFAULT,m,False)
    yield ('C03_kernel',512,512,4096,'photo',DEFAULT,'FC-AOIP',True)
    for kind in (KINDS if profile!='smoke' else ['zero','random']):
        for p in ([55,70] if profile=='smoke' else [35,55,65,70,95]):yield ('C04',512,512,4096,kind,{**DEFAULT,'psnr':p,'steps':[2,4,8,16,32,64],'offsets':list(range(-8,9))},'FC-AOIP',False)

def make_input(h,w,kind):
    if kind=='photo':return load_image(ROOT/'data/host/Girl.bmp',(w,h))
    return synthetic(kind,h,w)

def worker_batch(task):
    path,n,sd=task;x=load_image(path);b=watermark_bits('C');count=0;errors=0;t=time.perf_counter_ns()
    for _ in range(n):
        try:
            y,p=embed_method('FC-AOIP',x,b,sd);count+=int(np.array_equal(b,apply_extract('FC-AOIP',y,p)))
        except Exception:errors+=1
    return dict(success=count,errors=errors,worker_ms=elapsed(t),worker_peak_rss_mb=peak_rss_mb())

def endurance(store,profile,operations=None,seconds=None):
    x=load_image(ROOT/'data/host/Girl.bmp');b=watermark_bits('C');y,pub=embed_method('FC-AOIP',x,b,42)
    sizes=[10] if profile=='smoke' else [1000] if profile=='lite' else [1000,10000] if profile=='paper' else [1000,10000,100000]
    if operations:sizes=[operations]
    for count in sizes:
        for op in ('embed','extract'):
            key=('C05',count,op,seconds)
            if store.has(key):continue
            rows=[];proc=psutil.Process();t0=time.monotonic();window=[];ok=0;errors=0;start_rss=proc.memory_info().rss/2**20;i=0;last=time.monotonic()
            while i<count or (seconds and time.monotonic()-t0<seconds):
                t=time.perf_counter_ns()
                try:
                    if op=='embed':yy,pp=embed_method('FC-AOIP',x,b,42);window.append(elapsed(t));pred=apply_extract('FC-AOIP',yy,pp)
                    else:pred=apply_extract('FC-AOIP',y,pub);window.append(elapsed(t))
                    ok+=int(np.array_equal(b,pred))
                except Exception:errors+=1
                i+=1
                if i%100==0 or (i>=count and (not seconds or time.monotonic()-t0>=seconds)):
                    now=time.monotonic();rows.append(dict(watermark='C',experiment='C05',operation=op,requested=count,requested_seconds=seconds,operations=i,success=ok,errors=errors,status='ok' if errors==0 and ok==i else 'error',wall_s=now-t0,window_wall_s=now-last,window_count=len(window),median_ms=float(np.median(window)) if window else None,p95_ms=float(np.percentile(window,95)) if window else None,p99_ms=float(np.percentile(window,99)) if window else None,rss_mb=proc.memory_info().rss/2**20,start_rss_mb=start_rss,cpu_percent=proc.cpu_percent(),peak_rss_mb=peak_rss_mb()));window=[];last=now
            store.put(key,rows)

def parallel(store,profile):
    import multiprocessing as mp
    counts=[1,2] if profile=='smoke' else [1,2,4,8,16];n=2 if profile=='smoke' else 30 if profile=='lite' else 100
    for workers in counts:
        key=('C06',workers,n)
        if store.has(key):continue
        t=time.perf_counter_ns()
        try:
            # Spawn prevents forking a process with initialized OpenMP/Numba state.
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:r=list(pool.map(worker_batch,[(str(ROOT/'data/host/Girl.bmp'),n,seed('worker',i)) for i in range(workers)]))
            wall=elapsed(t)/1000;success=sum(x['success'] for x in r);row=dict(watermark='C',experiment='C06',workers=workers,operations=workers*n,success=success,errors=sum(x['errors'] for x in r),status='ok' if success==workers*n else 'error',wall_s=wall,successful_images_s=success/wall,includes_worker_startup=True,max_worker_peak_rss_mb=max(x['worker_peak_rss_mb'] for x in r))
        except Exception as e:row=dict(watermark='C',experiment='C06',workers=workers,status='error',error=str(e))
        store.put(key,[row])

def run_compute(profile,out,resume,operations=None,seconds=None,strengths=None,hosts=None):
    specs=list(cases(profile));store=Store(out,dict(group='compute',profile=profile,specs=specs,operations=operations,seconds=seconds,strengths=strengths,hosts=[str(p) for p in (hosts or [])]),resume);settings=PROFILES[profile]
    try:
        for round_idx in range(settings['rounds']):
            order=list(specs);random.Random(2026+round_idx).shuffle(order)
            for experiment,h,w,k,kind,p,method,kernel in order:
                key=(round_idx,experiment,h,w,k,kind,p,method,kernel)
                if store.has(key):continue
                if experiment!='C03' and experiment!='C03_kernel' and round_idx:continue
                base=dict(watermark='C',experiment=experiment,height=h,width=w,pixels=h*w,payload_bits=k,kind=kind,method=method,round=round_idx,kernel_only=kernel,psnr_floor=p['psnr'])
                # Full frozen API materializes several arrays. Skip if conservative estimate exceeds available RAM.
                if h*w*64>available_memory()*.65:rows=[dict(status='skipped_memory_guard',error='Conservative memory estimate exceeds 65% of available RAM')]
                else:
                    try:
                        x=make_input(h,w,kind);b=watermark_bits('C',k);reps=settings['reps'] if experiment in ('C03','C03_kernel') else min(settings['reps'],3 if h*w>4_000_000 else 10);rows=timing_case(x,b,method,p,42,reps,settings['warm'] if experiment.startswith('C03') else 1,round_idx,kernel,strengths);del x,b
                    except Exception as e:rows=[dict(status=error_status(e),error=str(e))]
                store.put(key,[dict(base,**r) for r in rows]);print(experiment,h,w,k,method,rows[0]['status'],flush=True)
        if hosts:
            for path in hosts:
                for method in METHODS:
                    key=('heldout_timing',path.stem,method)
                    if store.has(key):continue
                    x=load_image(path);b=watermark_bits('C');r=timing_case(x,b,method,DEFAULT,42,min(settings['reps'],20),settings['warm'],strengths=strengths)
                    store.put(key,[dict(host=path.stem,experiment='C03_heldout',method=method,height=512,width=512,payload_bits=4096,kind='photo',kernel_only=False,psnr_floor=55,**row) for row in r])
        endurance(store,profile,operations,seconds);parallel(store,profile)
    finally:store.close()
