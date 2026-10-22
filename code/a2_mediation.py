"""Model A (hypoxia -> ER stress -> ECM) vs Model B (hypoxia -> ECM directly).
All variables are rank-residualised on purity, proliferation and CAF content, then z-scored.
Indirect effect = a*b (product of coefficients), 95% percentile bootstrap CI (5000 resamples)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from load import load; from core import groups
from modules import *
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index)
S=all_scores(p); prol=S['Proliferation']
rng=np.random.default_rng(7)
def prep(s,cov): 
    r=resid(s,cov); return (r-r.mean())/r.std()
def ols(y,X):
    X=np.column_stack([np.ones(len(y)),X]); b,*_=np.linalg.lstsq(X,y,rcond=None); return b[1:]
def mediate(x,M,y,B=5000,idx=None):
    """x: (n,), M: (n,k), y: (n,). returns total, direct, indirect per mediator & sum, with CIs"""
    def est(ii):
        xx,MM,yy=x[ii],M[ii],y[ii]
        a=np.array([ols(MM[:,j],xx[:,None])[0] for j in range(MM.shape[1])])
        bb=ols(yy,np.column_stack([xx,MM])); cprime=bb[0]; b=bb[1:]
        ctot=ols(yy,xx[:,None])[0]
        return np.concatenate([[ctot,cprime],a*b,[np.sum(a*b)]])
    n=len(y); base=est(np.arange(n))
    boot=np.array([est(rng.integers(0,n,n)) for _ in range(B)])
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
    p_ind=2*min((boot[:,-1]<=0).mean(),(boot[:,-1]>=0).mean())
    p_dir=2*min((boot[:,1]<=0).mean(),(boot[:,1]>=0).mean())
    return base,lo,hi,p_ind,p_dir

def run(sub,label,cov_names=('purity','proliferation'),B=5000):
    idx=sub
    cov=[pur[idx],prol[idx],S['CAF'][idx]]
    X=prep(S['Hypoxia consensus'][idx],cov)
    meds=['UPR canonical chaperones','UPR PERK/ATF4 targets','UPR ATF6 targets','ER redox']
    Mz=pd.concat([prep(S[k][idx],cov) for k in meds],axis=1)
    outs={'PLOD2':p['PLOD2'],'P4HA1':p['P4HA1'],'P4HA2':p['P4HA2'],'SERPINH1':p['SERPINH1'],
          'Maturation':S['Maturation'],'Maturation - deposition':S['Maturation - deposition'],
          'Fibrillar collagen':S['Fibrillar collagen'],'Collagen export':S['Collagen export']}
    rows=[]
    for yn,yv in outs.items():
        Y=prep(yv[idx],cov)
        ok=X.notna()&Y.notna()&Mz.notna().all(1)
        base,lo,hi,pi,pd_=mediate(X[ok].values,Mz[ok].values,Y[ok].values,B=B)
        prop=base[-1]/base[0] if abs(base[0])>0.05 else np.nan
        rows.append(dict(set=label,outcome=yn,n=ok.sum(),total=base[0],direct=base[1],direct_lo=lo[1],direct_hi=hi[1],p_direct=pd_,
             indirect=base[-1],ind_lo=lo[-1],ind_hi=hi[-1],p_indirect=pi,prop_mediated=prop,
             **{f'ind_{k}':base[2+j] for j,k in enumerate(meds)},**{f'ind_{k}_lo':lo[2+j] for j,k in enumerate(meds)},**{f'ind_{k}_hi':hi[2+j] for j,k in enumerate(meds)}))
    return pd.DataFrame(rows)

allidx=p.index
R1=run(allidx,'All tumours (n=122)')
R2=run(g[g=='TNBC'].index,'TNBC only',B=3000)
R=pd.concat([R1,R2]); R.to_csv('../out/mediation.csv',index=False)
cols=['set','outcome','n','total','direct','direct_lo','direct_hi','indirect','ind_lo','ind_hi','prop_mediated']
print(R[cols].round(3).to_string(index=False))
print(R1[[c for c in R1.columns if c.startswith('ind_') and not c.endswith(('_lo','_hi'))]+['outcome']].round(3).to_string(index=False))
