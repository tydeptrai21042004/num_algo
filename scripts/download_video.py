#!/usr/bin/env python3
"""Download an authorized direct MP4, with streaming, size bounds, hash and provenance.

W3C-hosted Sintel test trailer is the default (attribution/license are external).
Use only when you have permission to download and use the selected video.
"""
import argparse,hashlib,json,urllib.request,urllib.parse
from pathlib import Path

DEFAULT='https://media.w3.org/2010/05/sintel/trailer.mp4'

def run(url,output,max_mb=80):
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or not parsed.netloc:
        raise ValueError('Only direct HTTPS media URLs are supported (no YouTube/DRM scraping)')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    limit=int(max_mb*1024*1024);size=0;h=hashlib.sha256()
    request=urllib.request.Request(url,headers={'User-Agent':'FC-AOIP-Research/1.0'})
    temp=out.with_name(out.name+'.part')
    try:
        with urllib.request.urlopen(request,timeout=45) as resp,temp.open('wb') as target:
            ctype=resp.headers.get('Content-Type','')
            if not ('video' in ctype or 'octet-stream' in ctype):
                raise ValueError(f'Unexpected MIME type: {ctype}')
            while True:
                chunk=resp.read(65536)
                if not chunk:break
                size+=len(chunk)
                if size>limit:raise ValueError('Download exceeds --max-mb limit')
                target.write(chunk);h.update(chunk)
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    meta={'url':url,'size_bytes':size,'sha256':h.hexdigest(),'license':'User must verify permissions/attribution for media'}
    out.with_suffix('.download.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2));return out

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--url',default=DEFAULT)
    a.add_argument('--output',default='data/video/sintel_trailer.mp4')
    a.add_argument('--max-mb',type=int,default=80)
    args=a.parse_args();run(args.url,args.output,args.max_mb)
