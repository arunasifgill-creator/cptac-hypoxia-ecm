"""Permutation test and incremental value over proliferation for the compact Hypoxia + ECM model."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('a3_ml.py').read().split("lab=g.notna()")[0])
from joblib import Parallel, delayed
lab=g.notna(); y=(g[lab]=='TNBC').astype(int).values
cols=SETS['Hypoxia + ECM']; X=F.loc[lab,cols].values
def lr(): return models()['Logistic (L2)']
obs,_=cv_auc(X,y,lr(),R=5)
def one(i):
    yp=np.random.default_rng(i).permutation(y); a,_=cv_auc(X,yp,lr(),R=1,seed0=i); return a[0]
null=np.array(Parallel(n_jobs=-1)(delayed(one)(i) for i in range(200)))
print('Hypoxia+ECM AUC',obs.mean().round(3),'perm null mean',null.mean().round(3),'p=',((null>=obs.mean()).sum()+1)/(len(null)+1))
# incremental over proliferation, paired repeats
a0,_=cv_auc(F.loc[lab,['Proliferation']].values,y,lr(),R=20)
a1,_=cv_auc(F.loc[lab,['Proliferation']+cols].values,y,lr(),R=20)
d=a1-a0; print('Prolif',a0.mean().round(3),'Prolif+Hyp+ECM',a1.mean().round(3),'delta',d.mean().round(3),np.percentile(d,[2.5,97.5]).round(3))
pd.DataFrame({'prolif':a0,'prolif_hyp_ecm':a1}).to_csv('../out/ml_incremental.csv',index=False)
np.save('../out/ml_perm_null.npy',null)
