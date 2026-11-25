"""Canonical manuscript-derived protocol: 45 reference conditions + 15 additions.
All geometry returns the original canvas; no registration. Combined operations
execute in order with fixed canvas normalization after each operation.
"""
import hashlib,json
from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
from PIL import Image,ImageFilter,ImageEnhance
VERSION='omip-manuscript45-plus15-v1'
FAMILIES=('Compression','Noise','Filtering','Geometry','Photometric','Combined')
LEVELS=('1','2','3','4','5','Single')

def canonical_conditions():
    rows=[dict(condition_id='C000',category='Clean',attack='none',level='Clean',severity_rank=0,parameter='',attack_index=0,operations=[])]
    def add(cat,name,values,ref=True):
        for i,v in enumerate(values,1):
            idx=len(rows);rows.append(dict(condition_id=f'A{idx:03d}',category=cat,attack=name,level=str(i),severity_rank=i,parameter=repr(v),attack_index=idx,operations=[[name,v]],reference=ref if isinstance(ref,bool) else v in ref))
    add('Compression','JPEG',[90,70,50,40,30]);add('Compression','JPEG2000',[3,5,10,7],(3,5,10));add('Compression','WebP',[90,70,50],False);add('Compression','Bit depth',[7,6,5],False)
    add('Noise','Gaussian variance',[.003,.005,.05]);add('Noise','Salt and pepper',[.01,.05]);add('Noise','Speckle variance',[.001,.005,.03]);add('Noise','Poisson peak',[255,128,64],False)
    add('Filtering','Median',[3,5]);add('Filtering','Average',[3,5]);add('Filtering','Lowpass',[5]);add('Filtering','Gaussian blur',[.5,1.,1.5],(1.,));add('Filtering','Sharpness',[1.5])
    add('Geometry','Rotation',[5,10,15,30,45]);add('Geometry','Translation',[(5,5),(10,0)]);add('Geometry','Resize',[.5]);add('Geometry','Occlusion',[.25]);add('Geometry','Border crop pad',[16]);add('Geometry','Upscale',[2.]);add('Geometry','Crop resize',[.5]);add('Geometry','Shear',[.1])
    add('Photometric','Brightness',[.9]);add('Photometric','Contrast',[.9]);add('Photometric','Histogram equalization',[True]);add('Photometric','Gamma',[1.2,1.5],False);add('Photometric','Saturation',[.5],False)
    combos=[[['JPEG2000',7],['Resize',.5]],[['JPEG2000',7],['Rotation',10]],[['Resize',.5],['Gaussian blur',.5]],[['Sharpness',.5],['JPEG',90]],[['Speckle variance',.001],['Resize',.5]],[['Sharpness',.5],['Speckle variance',.001],['JPEG2000',7]]]
    for i,ops in enumerate(combos,1):
        idx=len(rows);rows.append(dict(condition_id=f'A{idx:03d}',category='Combined',attack=' + '.join(x[0] for x in ops),level='Single',severity_rank=1,parameter=repr(ops),attack_index=idx,operations=ops,reference=True))
    assert len(rows)==61 and sum(r.get('reference',False) for r in rows)==45
    return rows

def protocol_sha256():
    return hashlib.sha256(json.dumps({'version':VERSION,'conditions':canonical_conditions(),'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()

def clip(x):return np.clip(np.rint(x),0,255).astype(np.uint8)
def operation(x,name,v,rng):
    h,w=x.shape[:2];im=Image.fromarray(x)
    if name in ('JPEG','JPEG2000','WebP'):
        buf=BytesIO();fmt={'JPEG':'JPEG','JPEG2000':'JPEG2000','WebP':'WEBP'}[name]
        kw={'quality_mode':'rates','quality_layers':[int(v)],'irreversible':True} if name=='JPEG2000' else {'quality':int(v)} if name=='JPEG' else {'quality':int(v),'lossless':False,'method':6}
        im.save(buf,format=fmt,**kw);buf.seek(0)
        with Image.open(buf) as dec:return np.asarray(dec.convert('RGB')).copy()
    if name=='Bit depth':return ((x>>(8-v))<<(8-v)).astype(np.uint8)
    if name=='Gaussian variance':return clip(x.astype(float)+rng.normal(0,255*np.sqrt(v),x.shape))
    if name=='Speckle variance':return clip(x.astype(float)*(1+rng.normal(0,np.sqrt(v),x.shape)))
    if name=='Salt and pepper':
        y=x.copy();m=rng.random((h,w));y[m<v/2]=0;y[(m>=v/2)&(m<v)]=255;return y
    if name=='Poisson peak':return clip(rng.poisson(x.astype(float)*v/255)*255/v)
    if name=='Median':return cv2.medianBlur(x,v)
    if name=='Average':return cv2.blur(x,(v,v))
    if name=='Lowpass':return cv2.GaussianBlur(x,(v,v),0) # documented binomial Gaussian low-pass
    if name=='Gaussian blur':return np.asarray(im.filter(ImageFilter.GaussianBlur(v))).copy()
    if name=='Sharpness':return np.asarray(ImageEnhance.Sharpness(im).enhance(v)).copy()
    if name=='Rotation':return cv2.warpAffine(x,cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),v,1),(w,h),borderMode=cv2.BORDER_CONSTANT)
    if name=='Translation':return cv2.warpAffine(x,np.float32([[1,0,v[0]],[0,1,v[1]]]),(w,h),borderMode=cv2.BORDER_CONSTANT)
    if name in ('Resize','Upscale'):
        small=cv2.resize(x,(max(1,round(w*v)),max(1,round(h*v))),interpolation=cv2.INTER_AREA if v<1 else cv2.INTER_CUBIC)
        return cv2.resize(small,(w,h),interpolation=cv2.INTER_CUBIC if v<1 else cv2.INTER_AREA)
    if name=='Occlusion':
        hh,ww=round(h*np.sqrt(v)),round(w*np.sqrt(v));y=x.copy();y[(h-hh)//2:(h-hh)//2+hh,(w-ww)//2:(w-ww)//2+ww]=0;return y
    if name=='Border crop pad':
        if min(h,w)<=2*v:raise ValueError('Image too small for border crop')
        return cv2.copyMakeBorder(x[v:h-v,v:w-v],v,v,v,v,cv2.BORDER_CONSTANT)
    if name=='Crop resize':
        hh,ww=max(1,round(h*np.sqrt(v))),max(1,round(w*np.sqrt(v)));return cv2.resize(x[(h-hh)//2:(h-hh)//2+hh,(w-ww)//2:(w-ww)//2+ww],(w,h),interpolation=cv2.INTER_CUBIC)
    if name=='Shear':return cv2.warpAffine(x,np.float32([[1,v,-v*(h-1)/2],[0,1,0]]),(w,h),borderMode=cv2.BORDER_CONSTANT)
    if name=='Brightness':return np.asarray(ImageEnhance.Brightness(im).enhance(v)).copy()
    if name=='Contrast':return np.asarray(ImageEnhance.Contrast(im).enhance(v)).copy()
    if name=='Histogram equalization':
        y=cv2.cvtColor(x,cv2.COLOR_RGB2YCrCb);y[:,:,0]=cv2.equalizeHist(y[:,:,0]);return cv2.cvtColor(y,cv2.COLOR_YCrCb2RGB)
    if name=='Gamma':return clip(255*(x.astype(float)/255)**v)
    if name=='Saturation':return np.asarray(ImageEnhance.Color(im).enhance(v)).copy()
    raise ValueError(name)

def attack_rgb(rgb,cond,seed):
    x=np.asarray(rgb)
    if x.ndim!=3 or x.shape[2]!=3 or x.dtype!=np.uint8:raise ValueError('Expected RGB uint8')
    y=x.copy();rng=np.random.default_rng(int(seed))
    for name,v in cond['operations']:y=operation(y,name,v,rng)
    if y.shape!=x.shape or y.dtype!=np.uint8:raise AssertionError('Attack output contract')
    return y
