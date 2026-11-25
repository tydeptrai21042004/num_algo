"""Exact minimum-SSE fixed-carrier sign embedding, derived from the supplied C3RE prototype.
Numerical correctness does NOT certify actual JPEG attacks.
"""
import hashlib
import numpy as np
from numba import njit
ROW=(1,-1,-1,1,1,-1,-1,1)
@njit(cache=True)
def sgn(y,x,kind):
 if kind==0:return ROW[y]*ROW[x] # 44
 if kind==1:return ROW[y] # 40
 if kind==2:return ROW[x] # 04
 if kind==3:return 1 if (x+y)%2==0 else -1 # checker
 if kind==4:return 1 if x<4 else -1 # half horizontally
 if kind==5:return 1 if y<4 else -1 # half vertically
 return (1 if (x//2+y//2)%2==0 else -1) # 2x2 checker
@njit(cache=True)
def feature(x,rr,cc,kind,pred):
 h=0
 for y in range(8):
  for xx in range(8):
   a=sgn(y,xx,kind)
   r=int(x[rr+y,cc+xx,0]);g=int(x[rr+y,cc+xx,1]); b=int(x[rr+y,cc+xx,2])
   if pred==0:v=g
   elif pred==1:v=g-r
   elif pred==2:v=2*g-r-b
   else:v=pred*g-r-b
   h+=a*v
 return h
@njit(cache=True)
def cost(x,rr,cc,kind,sg,n):
 if n<=0:return 0
 totalcap=0;one=0
 for y in range(8):
  for xx in range(8):
   v=int(x[rr+y,cc+xx,1]); d=sg*sgn(y,xx,kind)
   cap=255-v if d>0 else v
   totalcap+=cap
   if cap>0:one+=1
 if n>totalcap:return -1
 if n<=one:return n
 lo=0;hi=255
 while lo<hi:
  mid=(lo+hi+1)//2
  k=0
  for y in range(8):
   for xx in range(8):
    v=int(x[rr+y,cc+xx,1]); d=sg*sgn(y,xx,kind)
    cap=255-v if d>0 else v
    k+=min(cap,mid)
  if k<=n:lo=mid
  else:hi=mid-1
 l=lo
 k=0;total=0
 for y in range(8):
  for xx in range(8):
   v=int(x[rr+y,cc+xx,1]); d=sg*sgn(y,xx,kind)
   cap=255-v if d>0 else v
   q=min(l,cap);k+=q;total+=q*q
 return total+(n-k)*(2*l+1)
@njit(cache=True)
def write_edit(out,rr,cc,kind,sg,n,seed,i):
 if n<=0:return
 one=0
 for y in range(8):
  for xx in range(8):
   v=int(out[rr+y,cc+xx,1]);d=sg*sgn(y,xx,kind)
   if (v<255 if d>0 else v>0):one+=1
 stride=2*((seed+i*17)%32)+1
 off=(seed*7+i*19)&63
 if n<=one:
  for j in range(64):
   if n==0:break
   loc=(stride*j+off)&63; yy=loc//8;xx=loc%8
   d=sg*sgn(yy,xx,kind);v=int(out[rr+yy,cc+xx,1])
   if (v<255 if d>0 else v>0):out[rr+yy,cc+xx,1]=v+d;n-=1
  return
 # level formula
 lo=0;hi=255
 while lo<hi:
  mid=(lo+hi+1)//2;k=0
  for yy in range(8):
   for xx in range(8):
    v=int(out[rr+yy,cc+xx,1]);d=sg*sgn(yy,xx,kind)
    cap=255-v if d>0 else v;k+=min(cap,mid)
  if k<=n:lo=mid
  else:hi=mid-1
 l=lo
 for j in range(64):
  loc=(stride*j+off)&63;yy=loc//8;xx=loc%8;d=sg*sgn(yy,xx,kind)
  v=int(out[rr+yy,cc+xx,1]);cap=255-v if d>0 else v;t=min(l,cap)
  out[rr+yy,cc+xx,1]=v+d*t;n-=t
 for j in range(64):
  if n==0:break
  loc=(stride*j+off)&63; yy=loc//8;xx=loc%8;d=sg*sgn(yy,xx,kind)
  v=int(out[rr+yy,cc+xx,1]);
  if (v<255 if d>0 else v>0):out[rr+yy,cc+xx,1]=v+d;n-=1
 if n!=0:raise ValueError('allocation fail')
# Keyed SHAKE256 ranking; deterministic, domain separated, no affine period.
from functools import lru_cache
@lru_cache(maxsize=128)
def schedule(height,width,seed):
 K=(height//8)*(width//8)
 key=int(seed).to_bytes(16,'big')
 raw=hashlib.shake_256(b'OMIP/block/v1'+key+height.to_bytes(8,'big')+width.to_bytes(8,'big')).digest(8*K)
 order=np.argsort(np.frombuffer(raw,dtype='<u8'),kind='stable').astype(np.int64)
 order.flags.writeable=False
 return order

@njit(cache=True)
def forward(x,bits,kind,pred,order):
 N=bits.size;NC=x.shape[1]//8;alpha=1 if pred in (0,1) else pred
 H=np.zeros(N,np.int32);ones=np.zeros(N,np.int16);caps=np.zeros(N,np.int32)
 for i in range(N):
  j=order[i];rr=(j//NC)*8;cc=(j%NC)*8;s=2*int(bits[i])-1
  H[i]=s*feature(x,rr,cc,kind,pred)
  for yy in range(8):
   for xx in range(8):
    v=int(x[rr+yy,cc+xx,1]);direction=s*sgn(yy,xx,kind)
    cap=255-v if direction>0 else v
    ones[i]+=(cap>0);caps[i]+=cap
 return H,ones,caps,alpha

@njit(cache=True)
def cyclic_two_unit_plan(x,rr,cc,kind,sg,n):
 # Nonoptimal deterministic control: cycle in raster order, take up to two
 # units per visit. No content-dependent pixel ordering or alternative carrier.
 edits=np.zeros(64,np.int32);caps=np.empty(64,np.int32)
 for loc in range(64):
  yy=loc//8;xx=loc%8;d=sg*sgn(yy,xx,kind);v=int(x[rr+yy,cc+xx,1]);caps[loc]=255-v if d>0 else v
 while n>0:
  before=n
  for loc in range(64):
   t=min(2,n,caps[loc]-edits[loc]);edits[loc]+=t;n-=t
   if n==0:break
  if n==before:return edits,-1
 total=0
 for loc in range(64):total+=edits[loc]*edits[loc]
 return edits,total

@njit(cache=True)
def sequential_cost(x,rr,cc,kind,sg,n):
 edits,total=cyclic_two_unit_plan(x,rr,cc,kind,sg,n)
 return total

@njit(cache=True)
def sequential_write(x,rr,cc,kind,sg,n):
 edits,total=cyclic_two_unit_plan(x,rr,cc,kind,sg,n)
 if total<0:raise ValueError('allocation fail')
 for loc in range(64):
  yy=loc//8;xx=loc%8;d=sg*sgn(yy,xx,kind);v=int(x[rr+yy,cc+xx,1]);x[rr+yy,cc+xx,1]=v+d*edits[loc]

@njit(cache=True)
def allcost(x,bits,kind,margin,H,ones,caps,order,alpha,budget,optimal=True):
 NC=x.shape[1]//8;total=0
 for i in range(bits.size):
  j=order[i];n=max(0,(margin-int(H[i])+alpha-1)//alpha)
  if n>caps[i]:return budget+1
  if not optimal:c=sequential_cost(x,(j//NC)*8,(j%NC)*8,kind,2*int(bits[i])-1,n)
  elif n<=ones[i]:c=n
  else:c=cost(x,(j//NC)*8,(j%NC)*8,kind,2*int(bits[i])-1,n)
  if c<0:return budget+1
  total+=c
  if total>budget:return total
 return total

@njit(cache=True)
def _run_embed(x,bits,kind,pred,order,tie_seed,budget,min_margin,optimize,optimal):
 H,ones,caps,alpha=forward(x,bits,kind,pred,order)
 lo=min_margin;hi=int(np.min(H+alpha*caps)) if optimize else lo
 if hi<lo or allcost(x,bits,kind,lo,H,ones,caps,order,alpha,budget,optimal)>budget:return x.copy(),-1,-1
 while lo<hi:
  mid=(lo+hi+1)//2
  if allcost(x,bits,kind,mid,H,ones,caps,order,alpha,budget,optimal)<=budget:lo=mid
  else:hi=mid-1
 out=x.copy();NC=x.shape[1]//8
 for i in range(bits.size):
  j=order[i];n=max(0,(lo-int(H[i])+alpha-1)//alpha)
  if n>0:
   if optimal:write_edit(out,(j//NC)*8,(j%NC)*8,kind,2*int(bits[i])-1,n,tie_seed,i)
   else:sequential_write(out,(j//NC)*8,(j%NC)*8,kind,2*int(bits[i])-1,n)
 return out,lo,allcost(x,bits,kind,lo,H,ones,caps,order,alpha,budget,optimal)

def run_embed(x,bits,kind,pred,seed,budget,min_margin,optimize=True,optimal=True):
 order=schedule(x.shape[0],x.shape[1],int(seed))
 tie=int.from_bytes(hashlib.sha256(int(seed).to_bytes(16,'big')+b'OMIP/ties').digest()[:4],'big')
 return _run_embed(x,bits,kind,pred,order,tie,budget,min_margin,optimize,optimal)

@njit(cache=True)
def _features(x,length,kind,pred,order):
 NC=x.shape[1]//8;v=np.empty(length,np.int32)
 for i in range(length):
  j=order[i];v[i]=feature(x,(j//NC)*8,(j%NC)*8,kind,pred)
 return v

def features(x,length,kind,pred,seed):
 return _features(x,length,kind,pred,schedule(x.shape[0],x.shape[1],int(seed)))

def decode(x,length,kind,pred,seed):return (features(x,length,kind,pred,seed)>0).astype(np.uint8)
