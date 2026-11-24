"""Independent exact reference checks; frontier model is not the practical embedder."""
from .common import *
from fcaoip.kernel import cost,run_embed,forward,allcost,sgn
from fcaoip.frontier_exact_reference import optimal_staircase,brute_optimizer
from fcaoip.proposal import minimum_orbit_margin,quant_index,MODES
import heapq

def dp_cost(caps,n):
    if n>sum(caps):return -1
    d=[0]+[10**18]*n
    for cap in caps:
        d=[min(d[j-i]+i*i for i in range(min(j,cap)+1)) for j in range(n+1)]
    return d[n]
def heap_cost(caps,n):
    if n>sum(caps):return -1
    heap=[(1,i) for i,c in enumerate(caps) if c];heapq.heapify(heap);total=0
    for _ in range(n):
        v,i=heapq.heappop(heap);total+=v
        if (v+1)//2<caps[i]:heapq.heappush(heap,(v+2,i))
    return total

def caps_for(x,kind,sign):return [255-int(x[i,j,1]) if sign*sgn(i,j,kind)>0 else int(x[i,j,1]) for i in range(8) for j in range(8)]
def synthetic(kind,h,w,sd=3):
    rng=np.random.default_rng(sd)
    if kind=='random':return rng.integers(0,256,(h,w,3),dtype=np.uint8)
    if kind=='zero':return np.zeros((h,w,3),np.uint8)
    if kind=='white':return np.full((h,w,3),255,np.uint8)
    if kind=='gray':return np.full((h,w,3),128,np.uint8)
    if kind=='saturated':return np.broadcast_to(np.array([255,0,255],np.uint8),(h,w,3)).copy()
    if kind=='gradient':return np.broadcast_to(np.linspace(0,255,w,dtype=np.uint8)[None,:,None],(h,w,3)).copy()
    if kind=='checker':return np.repeat((((np.arange(h)[:,None]+np.arange(w)[None,:])%2)*255).astype(np.uint8)[:,:,None],3,axis=2)
    if kind=='binary':return rng.integers(0,2,(h,w,3),dtype=np.uint8)*255
    raise ValueError(kind)
KINDS=['zero','white','gray','saturated','checker','gradient','random','binary']
def run_math(profile,out,resume):
    n=8 if profile=='smoke' else 64 if profile=='lite' else 512;store=Store(out,dict(group='mathematics',profile=profile,cases=n),resume);rng=np.random.default_rng(2026)
    try:
        for i in range(n):
            x=rng.integers(0,256,(8,8,3),dtype=np.uint8);kind=int(rng.choice([0,3]));sign=int(rng.choice([-1,1]));caps=caps_for(x,kind,sign);need=int(rng.integers(0,180))
            if not store.has(('A01',i)):
                t=time.perf_counter_ns();v=cost(x,0,0,kind,sign,need);ms=elapsed(t);t=time.perf_counter_ns();ref=dp_cost(caps,need);refms=elapsed(t);t=time.perf_counter_ns();hv=heap_cost(caps,need);hms=elapsed(t);store.put(('A01',i),[dict(experiment='A01/A04',case=i,status='ok' if v==ref==hv else 'mismatch',required_units=need,solver_sse=v,dp_sse=ref,heap_sse=hv,optimality_gap=(v-ref)/max(ref,1),kernel_ms=ms,dp_ms=refms,heap_ms=hms)])
            xu=tuple(map(int,rng.integers(0,5,2)));xv=tuple(map(int,rng.integers(0,5,2)));att=tuple(tuple(map(int,[rng.integers(1,6),rng.integers(1,6),rng.integers(-2,3),rng.integers(-2,3)])) for _ in range(3));bit=int(rng.integers(2))
            if not store.has(('frontier',i)):
                t=time.perf_counter_ns();res,intervals,states=optimal_staircase(xu,xv,4,att,bit,True);ms=elapsed(t);t=time.perf_counter_ns();ref=brute_optimizer(xu,xv,4,att,bit);refms=elapsed(t);v=res[0] if res else None;store.put(('frontier',i),[dict(experiment='A04_frontier_separate_model',case=i,status='ok' if v==ref else 'mismatch',solver_sse=v,reference_sse=ref,solver_ms=ms,reference_ms=refms,intervals=intervals,states=states)])
        for steps in ([1],[2,4],[3,7],[2,4,8,16,32,64]):
            for r in (0,1,2,4,8,16,32):
                key=('A02',steps,r)
                if store.has(key):continue
                offsets=range(-r,r+1);m=minimum_orbit_margin(steps,offsets);good=all(quant_index(m,q,o)>0 and quant_index(-m,q,o)<0 for q in steps for o in offsets);sharp=any(quant_index(m-1,q,o)<=0 or quant_index(-m+1,q,o)>=0 for q in steps for o in offsets);store.put(key,[dict(experiment='A02',steps=steps,radius=r,threshold=m,status='ok' if good and sharp else 'mismatch',sufficient=good,sharp=sharp)])
        for kind in KINDS:
            for fraction in ([0,.5,1] if profile=='smoke' else [0,.1,.25,.5,.75,.9,1]):
                for p in ([55] if profile=='smoke' else [35,45,55,65,70]):
                    key=('A03',kind,fraction,p)
                    if store.has(key):continue
                    x=synthetic(kind,64,64);b=np.zeros(64,np.uint8);b[:round(64*fraction)]=1;np.random.default_rng(3).shuffle(b);row=dict(experiment='A03',kind=kind,fraction=fraction,psnr_floor=p,status='ok')
                    try:
                        t=time.perf_counter_ns();y,pub=embed_method('FC-AOIP',x,b,42,{'psnr':p});row['embed_ms']=elapsed(t);row.update(quality(x,y,b,pub,'FC-AOIP',{**DEFAULT,'psnr':p}))
                    except Exception as e:row.update(status=error_status(e),error=str(e))
                    store.put(key,[row])
        for i in range(n):
            x=synthetic('random',16,16,i);b=bits_for(4,i);budget=[100,500,5000,50000][i%4];cap=64 if profile=='smoke' else 512
            for mode,(kind,alpha) in MODES.items():
                key=('A05',i,mode)
                if store.has(key):continue
                H,ones,caps,a,z,coef=forward(x,b,kind,alpha,42);costs=[];prefixes=[];j=z
                for k in range(len(b)):
                    rr,cc=(j//2)*8,(j%2)*8;local=caps_for(x[rr:rr+8,cc:cc+8],kind,2*int(b[k])-1)
                    increments=sorted(2*t+1 for c in local for t in range(c));prefixes.append(np.concatenate(([0],np.cumsum(increments))));j=(j+a)%4
                # Independent heap reference per block, explicit scan over every candidate margin.
                for mu in range(4,cap+1):
                    total=0;j=z
                    for k in range(len(b)):
                        rr,cc=(j//2)*8,(j%2)*8;need=max(0,(mu*coef-int(H[k])+coef-1)//coef);c=int(prefixes[k][need]) if need<len(prefixes[k]) else -1
                        if c<0:total=budget+1;break
                        total+=c
                        if total>budget:break
                        j=(j+a)%4
                    if total>budget:break
                    costs.append((mu,total))
                y,margin,sse=run_embed(x,b,kind,alpha,42,cap,budget,4);ref=costs[-1] if costs else (-1,-1);store.put(key,[dict(experiment='A05',case=i,mode=mode,budget=budget,margin=margin,sse=sse,reference_margin=ref[0],reference_sse=ref[1],status='ok' if (margin,sse)==ref else 'mismatch')])
    finally:store.close()
