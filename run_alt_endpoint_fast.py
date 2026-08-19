import numpy as np, os, csv, time
from scipy.special import gamma
from scipy.stats import linregress
MU=2.0; LAM=1.2; A=0.384; B=0.095; C=0.0025; ETA=1.0; XI0=0.0; T=1.0
M=30000; NS=np.array([16,32,64,128,256,512,1024]); HS=[0.1,0.2,0.3,0.4]; BATCH=3000; SEED=20260818
OUT='/mnt/data/alt_endpoint_fast'; os.makedirs(OUT,exist_ok=True)
def sig(y): return np.sqrt(A*(y-B)**2+C)
def endpoint_both(dW,dt,H):
    m,n=dW.shape; alpha=H+0.5
    y=np.zeros(m)
    proj=np.zeros(m); naive=np.zeros(m)
    # weights indexed by k=0..n-1 for lag n-k
    j=np.arange(n,0,-1,dtype=float)
    wn=(j*dt)**(alpha-1)/gamma(alpha)
    wp=(dt**alpha)*(j**alpha-(j-1.0)**alpha)/gamma(alpha+1)
    for k in range(n):
        b=MU-LAM*y; s=sig(y); dw=dW[:,k]
        naive += wn[k]*(b*dt+ETA*s*dw)
        proj += wp[k]*(b+ETA*s*dw/dt)
        y = y + dt*b + ETA*s*dw
    return XI0+proj, XI0+naive
def run(H,N,seed):
    rng=np.random.default_rng(seed); ssP=0.;ssN=0.;cnt=0
    dt2=T/(2*N);dt=T/N
    for st in range(0,M,BATCH):
        m=min(BATCH,M-st)
        dw2=rng.normal(0,np.sqrt(dt2),size=(m,2*N)); dw=dw2.reshape(m,N,2).sum(2)
        p2,n2=endpoint_both(dw2,dt2,H); p,n=endpoint_both(dw,dt,H)
        ep=p-p2; en=n-n2
        ssP+=float(np.dot(ep,ep)); ssN+=float(np.dot(en,en));cnt+=m
    return (ssP/cnt)**0.5,(ssN/cnt)**0.5
for hi,H in enumerate(HS):
    rows=[]; t=time.time()
    for ni,N in enumerate(NS):
        p,n=run(H,int(N),SEED+hi*100+ni);rows.append([int(N),p,n]);print(H,N,p,n,flush=True)
    with open(f'{OUT}/Alt_Endpoint_H{H:.1f}.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['N','Projected_endpoint','Naive_endpoint']);w.writerows(rows)
    arr=np.array(rows,float)
    for col,name in [(1,'Projected'),(2,'Naive')]:
        fit=linregress(np.log(arr[:,0]),np.log(arr[:,col])); print('SLOPE',H,name,-fit.slope,fit.stderr,fit.rvalue**2,flush=True)
    print('time',H,time.time()-t,flush=True)
