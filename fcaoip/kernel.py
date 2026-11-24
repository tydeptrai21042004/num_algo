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
@njit(cache=True)
def forward(x,bits,kind,pred,seed):
 N=bits.size;K=(x.shape[0]//8)*(x.shape[1]//8);NC=x.shape[1]//8
 a=2*(seed%65536)+1
 while np.gcd(a,K)!=1:a+=2
 b=(seed*23+59)%K
 alpha=1 if pred in (0,1) else pred
 H=np.zeros(N,np.int32);ones=np.zeros(N,np.int16);caps=np.zeros(N,np.int32)
 j=b
 for i in range(N):
  rr=(j//NC)*8;cc=(j%NC)*8;s=2*int(bits[i])-1
  H[i]=s*feature(x,rr,cc,kind,pred)
  c0=0;capall=0
  for yy in range(8):
   for xx in range(8):
    v=int(x[rr+yy,cc+xx,1]);direction=s*sgn(yy,xx,kind)
    cap=255-v if direction>0 else v
    c0+=(cap>0);capall+=cap
  ones[i]=c0;caps[i]=capall
  j=(j+a)%K
 return H,ones,caps,a,b,alpha
@njit(cache=True)
def allcost(x,bits,kind,pred,margin,H,ones,caps,a,b,alpha,budget):
 NC=x.shape[1]//8;K=(x.shape[0]//8)*NC;j=b;total=0
 for i in range(bits.size):
  n=max(0,(margin*alpha-int(H[i])+alpha-1)//alpha)
  if n>caps[i]:return budget+1
  if n<=ones[i]:c=n
  else:c=cost(x,(j//NC)*8,(j%NC)*8,kind,2*int(bits[i])-1,n)
  if c<0:return budget+1
  total+=c
  if total>budget:return total
  j=(j+a)%K
 return total
@njit(cache=True)
def run_embed(x,bits,kind,pred,seed,lim,budget,min_margin):
 H,ones,caps,a,b,alpha=forward(x,bits,kind,pred,seed)
 if allcost(x,bits,kind,pred,min_margin,H,ones,caps,a,b,alpha,budget)>budget:return x.copy(),-1,-1
 lo=min_margin;hi=lim
 while lo<hi:
  mid=(lo+hi+1)//2
  if allcost(x,bits,kind,pred,mid,H,ones,caps,a,b,alpha,budget)<=budget:lo=mid
  else:hi=mid-1
 margin=lo
 out=x.copy();NC=x.shape[1]//8;K=(x.shape[0]//8)*NC;j=b
 for i in range(bits.size):
  n=max(0,(margin*alpha-int(H[i])+alpha-1)//alpha)
  if n>0:write_edit(out,(j//NC)*8,(j%NC)*8,kind,2*int(bits[i])-1,n,seed,i)
  j=(j+a)%K
 return out,margin,allcost(x,bits,kind,pred,margin,H,ones,caps,a,b,alpha,budget)
@njit(cache=True)
def decode(x,length,kind,pred,seed):
 K=(x.shape[0]//8)*(x.shape[1]//8);NC=x.shape[1]//8
 a=2*(seed%65536)+1
 while np.gcd(a,K)!=1:a+=2
 j=(seed*23+59)%K;bits=np.empty(length,np.uint8)
 for i in range(length):
  rr=(j//NC)*8;cc=(j%NC)*8
  bits[i]=1 if feature(x,rr,cc,kind,pred)>0 else 0
  j=(j+a)%K
 return bits

