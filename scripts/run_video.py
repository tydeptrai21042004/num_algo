#!/usr/bin/env python3
"""Real-video/file/camera experiment with actual MP4 re-encoding evaluation.

Frames alternate between the two bundled logo payloads; no temporal
voting, ECC, CRC, denoising, attack-adaptive decoder, or transform search.
Public one-bit per-frame carrier mode is recorded in sidecar CSV, so decoder
is blind to the cover but NOT descriptor-free. 512x512 / 4096 bits for all
five provided classical image adapters. Unqualified frames are reported and
written unmodified; they are never counted as successful embedding.
"""
import argparse,csv,hashlib,json,sys,time,platform
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.watermarks import watermark_bits, watermark_for_index, watermark_manifest
from fcaoip.evaluation import METHODS,apply_embed,apply_extract,psnr,ber,nc01

def seed_for(master,frame,kind):
    return int.from_bytes(hashlib.sha256(f'{master}|{frame}|{kind}'.encode()).digest()[:4],'little') & 0x7fffffff

def measured_source(src):
    if str(src).startswith('camera:'):
        return cv2.VideoCapture(int(str(src).split(':',1)[1])),True
    return cv2.VideoCapture(str(src)),False

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def process(args):
    if args.method not in METHODS:raise ValueError(f'Method must be one of {METHODS}')
    if args.method!='FC-AOIP' and args.bits!=4096:raise ValueError('Five baseline adapters require 4096 bits')
    cap,is_camera=measured_source(args.input)
    if not cap.isOpened():raise FileNotFoundError(f'Cannot open source {args.input}')
    input_fps=float(cap.get(cv2.CAP_PROP_FPS))
    fps=input_fps if 1<=input_fps<=240 else float(args.fps)
    width,height=int(args.width),int(args.height)
    if width%8 or height%8:raise ValueError('Width and height must be divisible by 8')
    if args.bits>(width//8)*(height//8):raise ValueError('Payload exceeds frame block capacity')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    result=out/f'{args.method.replace("-","_").lower()}_marked.mp4'
    writer=cv2.VideoWriter(str(result),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('No MP4 encoder supported by OpenCV')
    # Compile JIT / initialize decoder once without counting startup in steady-state FPS.
    warm_start=time.perf_counter()
    initial_ok,initial_frame=cap.read()
    if not initial_ok:raise RuntimeError('No readable frames in video source')
    initial=cv2.cvtColor(cv2.resize(initial_frame,(width,height)),cv2.COLOR_BGR2RGB)
    warm_seed=seed_for(args.seed,0,'key')
    warm_bits=watermark_bits('C',args.bits)
    try:apply_embed(args.method,initial,warm_bits,seed=warm_seed,psnr=args.psnr)
    except ValueError:pass  # Infeasibility is recorded during normal frame processing.
    warmup_ms=(time.perf_counter()-warm_start)*1000
    rows=[];public_keys=[]
    first=time.perf_counter()
    for i in range(args.frames):
        total_start=time.perf_counter()
        if i==0:ok,frame=initial_ok,initial_frame
        else:ok,frame=cap.read()
        if not ok:break
        bgr=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
        x=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        seed=seed_for(args.seed,i,'key')
        bits=watermark_bits(watermark_for_index(i),args.bits)
        start=time.perf_counter()
        status='ok';error='';public=None
        try:
            y,public=apply_embed(args.method,x,bits,seed=seed,psnr=args.psnr)
            decoded=apply_extract(args.method,y,public)
            clean_ber=ber(bits,decoded)
            clean_psnr=psnr(x,y)
            if clean_ber!=0 or clean_psnr+1e-8<args.psnr:
                status='unqualified'
        except (ValueError,AssertionError) as exc:
            status='infeasible';error=str(exc)[:180];y=x.copy()
            clean_ber=float('nan');clean_psnr=float('inf')
        embed_ms=(time.perf_counter()-start)*1000
        public_keys.append(public)
        writer.write(cv2.cvtColor(y,cv2.COLOR_RGB2BGR))
        total_ms=(time.perf_counter()-total_start)*1000
        rows.append(dict(frame=i,watermark=watermark_for_index(i),method=args.method,status=status,seed=seed,
            public_mode='' if public is None else public.get('mode',''),
            payload_bits=args.bits,clean_psnr_db=clean_psnr,clean_ber=clean_ber,
            embed_ms=embed_ms,frame_total_ms=total_ms,post_mp4_ber='',post_mp4_nc01='',
            error=error))
        print(f'frame {i+1}: {status}, embed {embed_ms:.2f} ms, PSNR {clean_psnr:.2f}',flush=True)
    cap.release();writer.release()
    elapsed=time.perf_counter()-first
    # Second pass: read the *encoded* output. Re-encoding itself can change bits.
    compressed=cv2.VideoCapture(str(result))
    for r in rows:
        ok,frame=compressed.read()
        if not ok:break
        if r['status']=='infeasible':continue
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        seed=int(r['seed']);bits=watermark_bits(r['watermark'],args.bits)
        public=public_keys[int(r['frame'])]
        decoded=apply_extract(args.method,rgb,public)
        r['post_mp4_ber']=ber(bits,decoded)
        r['post_mp4_nc01']=nc01(bits,decoded)
    compressed.release()
    write_csv(out/'video_frames.csv',rows)
    valid=[r for r in rows if r['status']=='ok']
    manifest=dict(source=args.input,source_is_camera=is_camera,source_fps=input_fps,
        output_fps=fps,warmup_ms_excluded_from_steady_state=warmup_ms,frames_processed=len(rows),qualified_frames=len(valid),
        payload_bits=args.bits,output_video=str(result),
        encode_codec='mp4v',processing_wall_seconds=elapsed,
        throughput_fps=(len(rows)/elapsed if elapsed else None),
        mean_embed_ms=(float(np.mean([r['embed_ms'] for r in valid])) if valid else None),
        mean_encoded_ber=(float(np.mean([float(r['post_mp4_ber']) for r in valid if isinstance(r['post_mp4_ber'],float)])) if any(isinstance(r['post_mp4_ber'],float) for r in valid) else None),
        throughput_exceeds_nominal_source_fps=(len(rows)/elapsed>=fps if elapsed and len(rows) else False),
        qualification_note='File processing throughput is NOT live camera frame-drop evidence; camera:N measures an actual connected webcam.',
        actual_jpeg_h264_certificate=False,
        raw_payload_per_frame='Original C/CC logos alternating by frame index.',watermarks=watermark_manifest(),
        public_metadata='FC-AOIP decoder requires the 1-bit mode per accepted frame; mode available from video_frames.csv.')
    (out/'video_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Video experiment:',json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True,help='Local video filename or camera:0')
    a.add_argument('--method',choices=METHODS,default='FC-AOIP')
    a.add_argument('--frames',type=int,default=30);a.add_argument('--bits',type=int,default=4096)
    a.add_argument('--width',type=int,default=512);a.add_argument('--height',type=int,default=512)
    a.add_argument('--fps',type=float,default=30);a.add_argument('--seed',type=int,default=2026)
    a.add_argument('--psnr',type=float,default=55);a.add_argument('--out',default='results/video_demo')
    process(a.parse_args())
