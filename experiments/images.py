"""All robustness cases use C000 + A001..A060, including parameter sweeps."""
from itertools import product
from .common import *

ABLATIONS={'full':{},'fixed_margin':{'optimize':False},'sequential_projection':{'optimal':False},'green_only':{'color':0},'checker_control':{'carrier':3}}
SWEEPS={'B01':('psnr',[40,45,50,55,58,60,62,65]),'B02':('steps',[[1],[2],[2,4],[2,4,8]]),'B03':('offsets',[list(range(-r,r+1)) for r in (0,1,2,4,8)]),'B04':('color',[0,1,2,3,4]),'B05':('carrier',[0,1,2,3])}
STRENGTHS={'Scalar-QIM':[2,4,8,12,16,24,32],'SCS':[2,4,8,12,16,24,32],'STDM-QIM':[2,4,6,8,10,12,16,24],'ISS':[.5,1,2,3,4,6,8],'SS':[.5,1,2,3,4,6,8]}

def specs_for(group,profile,matched=None):
    if group=='robustness60':
        specs=[(m,'default',DEFAULT,None) for m in METHODS]
        if matched:specs.extend((m,'matched',DEFAULT,matched) for m in METHODS if m!='FC-AOIP')
        return specs
    if group=='payload_robustness':return [('FC-AOIP',f'payload_{k}',DEFAULT,None) for k in ([64,4096] if profile=='smoke' else [64,256,512,1024,2048,4096])]
    if group=='ablation':return [('FC-AOIP',k,{**DEFAULT,**v},None) for k,v in ABLATIONS.items()]
    if group=='hyperparameters':
        specs=[]
        for name,(k,values) in SWEEPS.items():
            if profile=='smoke':values=[values[0],values[-1]]
            for i,v in enumerate(values):specs.append(('FC-AOIP',f'{name}_{i:02}',{**DEFAULT,k:v},None))
        return specs
    if group=='calibration':
        return [(m,f'strength_{v:g}',DEFAULT,{**FIXED_STRENGTHS,m:v}) for m,vs in STRENGTHS.items() for v in (vs if profile!='smoke' else vs[:2])]
    if group=='validation':return [('FC-AOIP','default',DEFAULT,None)]+[(m,'matched',DEFAULT,matched) for m in METHODS if m!='FC-AOIP']
    raise ValueError(group)

def case_rows(x,b,method,params,strengths,base,sd,attack_seed_base,visual_dir=None):
    q=dict(base,table='quality',status='ok',qualified=0,**params,strength=(strengths or FIXED_STRENGTHS).get(method))
    rows=[q];y=pub=None
    try:
        t=time.perf_counter_ns();y,pub=embed_method(method,x,b,sd,params,strengths);q['first_call_ms']=elapsed(t)
        t=time.perf_counter_ns();y,pub=embed_method(method,x,b,sd,params,strengths);q['embed_ms']=elapsed(t);q.update(quality(x,y,b,pub,method,params))
        q['ssim']=image_ssim(x,y)
        q['status']='ok' if q['qualified'] else 'unqualified'
        q['watermarked_sha256']=pixel_hash(y)
        if visual_dir is not None:
            visual_dir.mkdir(parents=True,exist_ok=True);Image.fromarray(x).save(visual_dir/'host.png');Image.fromarray(y).save(visual_dir/'marked.png');Image.fromarray(b.reshape(64,64)*255).save(visual_dir/'watermark.png');Image.fromarray(apply_extract(method,y,pub).reshape(64,64)*255).save(visual_dir/'clean_recovered.png');Image.fromarray(np.clip(abs(y.astype(int)-x.astype(int))*16,0,255).astype(np.uint8)).save(visual_dir/'embedding_diff_x16.png')
        if base.get('variant')=='full' and method=='FC-AOIP':
            from fcaoip.kernel import forward,allcost,schedule
            order=schedule(x.shape[0],x.shape[1],int(sd));H,ones,caps,alpha=forward(x,b,0,3,order);q['sequential_sse_at_equal_margin']=int(allcost(x,b,0,pub['info']['margin'],H,ones,caps,order,alpha,2**60,False))
    except Exception as e:q.update(status=error_status(e),error=str(e)[:500])
    for c in CONDITIONS[1:]:
        a=dict(base,table='attacks',**c,qualified=q['qualified'],status='not_embedded',attack_seed=seed(*attack_seed_base,c['condition_id']))
        if pub is not None and 'clean_ber' in q:
            try:
                t=time.perf_counter_ns();altered=attack_rgb(y,c,a['attack_seed']);a['attack_ms']=elapsed(t);t=time.perf_counter_ns();pred=apply_extract(method,altered,pub);a['extract_ms']=elapsed(t);a.update(accuracy(b,pred),status='ok');d=altered.astype(np.int32)-y.astype(np.int32);a.update(host_psnr_db=psnr(x,altered),attack_psnr_db=psnr(y,altered),attack_ssim=image_ssim(y,altered),attack_mse=float(np.mean(d*d)),attack_mae=float(np.mean(abs(d))),changed_pixel_ratio=float(np.mean(np.any(d!=0,axis=2))),max_pixel_difference=int(abs(d).max()),watermarked_sha256=pixel_hash(y),attacked_sha256=pixel_hash(altered),extracted_bits_sha256=pixel_hash(pred))
                if visual_dir is not None:
                    aid=c['condition_id'];Image.fromarray(altered).save(visual_dir/f'{aid}_attacked.png');Image.fromarray(pred.reshape(64,64)*255).save(visual_dir/f'{aid}_recovered.png');Image.fromarray(np.clip(abs(d)*4,0,255).astype(np.uint8)).save(visual_dir/f'{aid}_diff_x4.png')
            except Exception as e:a.update(status='error',error=str(e)[:500])
        rows.append(a)
    return rows

def run_images(group,split,profile,out,resume=False,limit=None,keys=None,matched=None,method_filter=None,hosts=None,matched_only=False):
    hosts=get_hosts(split,profile,limit) if hosts is None else list(hosts);specs=specs_for(group,profile,matched)
    if matched_only:
        if group!='robustness60' or not matched:raise ValueError('Matched-only requires calibrated robustness60')
        specs=[s for s in specs if s[0]=='FC-AOIP' or s[1]=='matched']
    if method_filter:specs=[s for s in specs if s[0] in method_filter]
    if not specs:raise ValueError('No methods selected')
    keys=keys or PROFILES[profile]['keys'];store=Store(out,dict(group=group,split=split,profile=profile,keys=keys,hosts={str(p.relative_to(ROOT)):sha(p) for p in hosts},specs=specs),resume)
    try:
        for path in hosts:
            x=load_image(path);host=path.stem
            for k in range(keys):
                sd=seed('schedule',host,k)
                for watermark in WATERMARK_IDS:
                    for method,variant,p,strength in specs:
                        key=(host,k,watermark,method,variant)
                        if store.has(key):continue
                        b=watermark_bits(watermark,int(variant.split('_')[1]) if group=='payload_robustness' else 4096)
                        sd=proposal_seed('schedule',host,k) if method=='FC-AOIP' else seed('schedule',host,k)
                        base=dict(experiment=group,split=split,host=host,key_index=k,watermark=watermark,seed=sd,method=method,variant=variant,payload_bits=len(b))
                        visual=Path(out)/'visual_examples'/host/watermark if group=='robustness60' and method=='FC-AOIP' and k==0 and path==hosts[0] else None
                        rows=case_rows(x,b,method,p,strength,base,sd,('attacks',host,k),visual);store.put(key,rows)
                        print(group,host,k,watermark,method,variant,rows[0]['status'],flush=True)
    finally:store.close()

def calibration_selection(folder):
    import pandas as pd
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    if m['config']['split']!='train' or m['config']['group']!='calibration':raise ValueError('Only training calibration results allowed')
    q=pd.read_csv(folder/'quality.csv');a=pd.read_csv(folder/'attacks.csv');records={};chosen={}
    cfg=m['config'];expected=len(cfg['hosts'])*cfg['keys']*len(cfg['specs'])*len(WATERMARK_IDS)
    if q.status.eq('error').any() or a.status.eq('error').any():raise RuntimeError('Unexpected calibration errors; resume before freezing')
    if len(q)!=expected or len(a)!=expected*60 or {s[0] for s in cfg['specs']}!=set(STRENGTHS):raise RuntimeError('Incomplete training calibration')
    if cfg.get('watermarks')!=watermark_manifest():raise RuntimeError('Calibration watermark protocol mismatch')
    if m['core']!=verify_core() or m['attack_protocol_sha256']!=protocol_sha256():raise RuntimeError('Calibration core/protocol mismatch')
    for method in STRENGTHS:
        qs=q[q.method==method]; candidates=[]
        for variant,g in qs.groupby('variant'):
            ar=a[(a.method==method)&(a.variant==variant)&(a.status=='ok')&(a.qualified==1)];ps=pd.to_numeric(g.psnr_db,errors='coerce');valid=g.qualified==1
            # First maximize successful clean+quality fraction; then nearest actual PSNR to 55; then BER.
            rec=dict(variant=variant,strength=float(g.strength.iloc[0]),qualification=float(valid.mean()),distance=float(abs(ps[valid]-55).mean()) if valid.any() else 1e9,ber=float(ar.ber.mean()) if len(ar) else 1.)
            candidates.append(rec)
        if not candidates:raise RuntimeError(f'Missing calibration {method}')
        best=min(candidates,key=lambda r:(-r['qualification'],r['distance'],r['ber'],r['strength']));chosen[method]=best['strength'];records[method]=candidates
    return chosen,records

def freeze(out,calibration,validation):
    from .datasets import verify_splits
    chosen,details=calibration_selection(calibration);v=Path(validation);m=json.loads((v/'manifest.json').read_text());cfg=m['config'];
    if cfg.get('watermarks')!=watermark_manifest():raise RuntimeError('Validation watermark protocol mismatch')
    if cfg['group']!='validation' or cfg['split']!='validation':raise RuntimeError('Held-out validation required')
    for sp,manifest in [('train',json.loads((Path(calibration)/'manifest.json').read_text())),('validation',m)]:
        allowed={r['path']:r['sha256'] for r in json.loads((ROOT/'data/manifests'/f'{sp}.json').read_text())}
        if any(allowed.get(p)!=h for p,h in manifest['config']['hosts'].items()):raise RuntimeError(f'{sp} manifest inputs do not match current split')
    if any(s[3]!=chosen for s in cfg['specs'] if s[0]!='FC-AOIP'):raise RuntimeError('Validation strengths mismatch')
    if {s[0] for s in cfg['specs']}!=set(METHODS):raise RuntimeError('Incomplete validation method set')
    if not (v/'quality.csv').exists():raise RuntimeError('Validation incomplete')
    import pandas as pd
    q=pd.read_csv(v/'quality.csv');a=pd.read_csv(v/'attacks.csv');expected=len(cfg['hosts'])*cfg['keys']*len(cfg['specs'])*len(WATERMARK_IDS)
    if len(q)!=expected or len(a)!=expected*60:raise RuntimeError('Incomplete validation cases')
    obj=dict(proposal=DEFAULT,strengths=chosen,watermarks=watermark_manifest(),calibration_details=details,dataset_fingerprint=verify_splits(),core=verify_core(),attack_protocol=protocol_sha256(),calibration_manifest=sha(Path(calibration)/'manifest.json'),validation_manifest=sha(v/'manifest.json'))
    atomic_json(out,dict(obj,fingerprint=digest(obj)));return chosen

def read_frozen(path):
    from .datasets import verify_splits
    o=json.loads(Path(path).read_text());h=o.pop('fingerprint')
    if o.get('watermarks')!=watermark_manifest() or digest(o)!=h or o['core']!=verify_core() or o['attack_protocol']!=protocol_sha256() or o['dataset_fingerprint']!=verify_splits() or o['proposal']!=DEFAULT:raise RuntimeError('Frozen config verification failed')
    return o['strengths']

def run_security(split,profile,out,resume,limit=None,keys=None,hosts=None,strengths=None,method_filter=None):
    methods=tuple(METHODS if method_filter is None else method_filter)
    if not methods or not set(methods)<=set(METHODS):raise ValueError("Invalid security methods")
    hosts=get_hosts(split,profile,limit) if hosts is None else list(hosts);keys=keys or (2 if profile=='smoke' else 100 if profile=='stress' else 25);store=Store(out,dict(group='security',split=split,hosts={str(p.relative_to(ROOT)):sha(p) for p in hosts},keys=keys,strengths=strengths,methods=methods),resume)
    try:
        for path in hosts:
            x=load_image(path);host=path.stem;sd=proposal_seed(host,'correct')
            for watermark in WATERMARK_IDS:
                b=watermark_bits(watermark)
                for method in methods:
                    key=(host,watermark,method)
                    if store.has(key):continue
                    rows=[];base=dict(host=host,method=method,split=split,watermark=watermark)
                    try:
                        sd=proposal_seed(host,'correct') if method=='FC-AOIP' else seed(host,'correct');y,pub=embed_method(method,x,b,sd,strengths=strengths);q=quality(x,y,b,pub,method,DEFAULT)
                        rows.append(dict(base,trial='correct',status='ok',qualified=q['qualified'],**accuracy(b,apply_extract(method,y,pub))))
                        if method=='FC-AOIP':
                            wp={**pub,'seed':sd^4096};rows.append(dict(base,trial='old_equivalent_key',status='ok',qualified=q['qualified'],**accuracy(b,apply_extract(method,y,wp))))
                            for label,pred in [('all_zero',np.zeros_like(b)),('all_one',np.ones_like(b)),('random_control',bits_for(len(b),host,watermark,'control'))]:rows.append(dict(base,trial=label,status='ok',qualified=q['qualified'],**accuracy(b,pred)))
                        # Baseline DCT keys are not just seed dictionaries. Derive a new valid key from a separate embedding.
                        for i in range(keys):
                            wrong=proposal_seed(host,'wrong',i) if method=='FC-AOIP' else seed(host,'wrong',i)
                            try:
                                if method=='FC-AOIP':wp={**pub,'seed':wrong}
                                else:_,wp=embed_method(method,x,b,wrong,strengths=strengths)
                                for kind,img,p in [('wrong_key',y,wp),('unmarked',x,wp)]:
                                    target=b
                                    rows.append(dict(base,trial=kind,index=i,test_seed=wrong,status='ok',qualified=q['qualified'],**accuracy(target,apply_extract(method,img,p))))
                            except Exception as e:rows.append(dict(base,trial='wrong_key',index=i,status=error_status(e),error=str(e)))
                    except Exception as e:rows.append(dict(base,trial='correct',status=error_status(e),error=str(e)))
                    store.put(key,rows)
    finally:store.close()
