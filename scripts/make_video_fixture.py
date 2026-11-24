#!/usr/bin/env python3
"""Generate a SYNTHETIC motion diagnostic from supplied still images.

This is never labeled a real camera/online-video test.
"""
from pathlib import Path
import cv2,numpy as np,argparse
ROOT=Path(__file__).resolve().parents[1]
def generate(output,frames=20,fps=20,width=512,height=512):
    x=cv2.imread(str(ROOT/'data/host/Girl.bmp'))
    if x is None:raise FileNotFoundError('Girl.bmp')
    p=Path(output);p.parent.mkdir(exist_ok=True,parents=True)
    writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
    for i in range(frames):
        M=np.array([[1.,0.,float(i%9-4)],[0.,1.,float((i//3)%7-3)]])
        xx=cv2.warpAffine(x,M,(x.shape[1],x.shape[0]),borderMode=cv2.BORDER_REFLECT)
        xx=cv2.resize(xx,(width,height),interpolation=cv2.INTER_LINEAR)
        writer.write(xx)
    writer.release();print('Synthetic diagnostic:',p,'frames',frames,'fps',fps)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',default='data/video/synthetic_motion.mp4')
    a.add_argument('--frames',type=int,default=20);a.add_argument('--fps',type=int,default=20)
    args=a.parse_args();generate(args.output,args.frames,args.fps)
