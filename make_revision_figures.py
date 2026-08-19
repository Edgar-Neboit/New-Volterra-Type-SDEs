import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

OUT='/mnt/data/revision_figures'
os.makedirs(OUT,exist_ok=True)
N=np.array([16,32,64,128,256,512,1024],dtype=float)
Hs=[0.1,0.2,0.3,0.4]

def slope_ref(errors, slope):
    return errors[0]*(N[0]/N)**slope

def plot_smart(kind,H):
    df=pd.read_csv(f'/mnt/data/results_unzip/results/{kind}_Errors_H{H:.1f}.csv')
    # In the supplied simulation library, internal mode 2 is the quadratic coefficient
    # sigma_1(x)=sqrt(a(x-b)^2+c) used in the paper.
    e=df['Error_X_sig2'].to_numpy(float)
    fig,ax=plt.subplots(figsize=(6.2,4.6))
    ax.loglog(N,e,'o-',label='Smart-Euler')
    ax.loglog(N,slope_ref(e,0.5),'--',label='slope 1/2')
    fit=linregress(np.log(N[N>=256]),np.log(e[N>=256]))
    ax.set_xlabel(r'$N$'); ax.set_ylabel(r'$L^2$ error')
    ax.set_title(rf'$H={H:.1f}$, $\sigma_1$; fitted slope $={-fit.slope:.3f}$')
    ax.set_xticks(N); ax.set_xticklabels([str(int(x)) for x in N])
    ax.grid(True,which='both',alpha=.25); ax.legend(); fig.tight_layout()
    name=f'Smart_{kind}_H{int(round(10*H)):02d}.png'
    fig.savefig(os.path.join(OUT,name),dpi=220,bbox_inches='tight'); plt.close(fig)

def plot_alt(H):
    df=pd.read_csv(f'/mnt/data/alt_endpoint_fast/Alt_Endpoint_H{H:.1f}.csv')
    p=df['Projected_endpoint'].to_numpy(float); n=df['Naive_endpoint'].to_numpy(float)
    fig,ax=plt.subplots(figsize=(6.2,4.6))
    ax.loglog(N,p,'o-',label='Projected-Euler')
    ax.loglog(N,n,'s-',label='Naive-Euler')
    ax.loglog(N,slope_ref(p,H),'--',label=rf'slope $H={H:.1f}$')
    ax.loglog(N,slope_ref(p,0.5),':',label='slope 1/2')
    fitp=linregress(np.log(N[N>=256]),np.log(p[N>=256]))
    fitn=linregress(np.log(N[N>=256]),np.log(n[N>=256]))
    ax.set_xlabel(r'$N$'); ax.set_ylabel(r'endpoint $L^2$ error')
    ax.set_title(rf'$H={H:.1f}$; fitted slopes: P $={-fitp.slope:.3f}$, N $={-fitn.slope:.3f}$')
    ax.set_xticks(N); ax.set_xticklabels([str(int(x)) for x in N])
    ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    name=f'Alternative_Endpoint_H{int(round(10*H)):02d}.png'
    fig.savefig(os.path.join(OUT,name),dpi=220,bbox_inches='tight'); plt.close(fig)

for H in Hs:
    plot_smart('Max',H); plot_smart('Endpoint',H); plot_alt(H)
