import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from load import load
from core import groups
from modules import *
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index)
S=all_scores(p)
# coverage + coherence
rows=[]
for k,v in PROT.items():
    used=[x for x in v if x in p and p[x].notna().sum()>=60]
    A=pd.DataFrame({x:purity_adjust(p[x],pur) for x in used})
    R=A.corr('spearman').values; iu=np.triu_indices(len(used),1)
    rows.append((k,len(v),len(used),np.nanmedian(R[iu]) if len(used)>1 else np.nan, ','.join(sorted(set(v)-set(used)))))
Q=pd.DataFrame(rows,columns=['module','genes','used','median_rho','dropped']); print(Q.round(2).to_string(index=False))
Q.to_csv('../out/module_qc.csv',index=False)
# subtype comparisons, purity adjusted
rows=[]
for k,s in S.items():
    x=purity_adjust(s,pur)
    for b in ['HER2+','HR+/HER2-']:
        pv,d=compare(x,g,'TNBC',b); rows.append((k,b,d,pv))
T=pd.DataFrame(rows,columns=['module','vs','d','p']); T['q']=np.nan
for b in T.vs.unique():
    i=T.vs==b; T.loc[i,'q']=bh(T.loc[i,'p'].values)
print(T.pivot(index='module',columns='vs',values=['d','q']).round(3).to_string())
T.to_csv('../out/subtype_effects.csv',index=False)
# hypoxia axis across all 122 (purity and purity+proliferation)
H=S['Hypoxia consensus']
rows=[]
for k,s in S.items():
    if k.startswith('Hypoxia'): continue
    r1,p1,_=pspear(s,H,[pur]); r2,p2,_=pspear(s,H,[pur,S['Proliferation']])
    rows.append((k,r1,p1,r2,p2))
A=pd.DataFrame(rows,columns=['module','rho_pur','p_pur','rho_pur_prolif','p_pur_prolif'])
A['q_pur_prolif']=bh(A.p_pur_prolif.values)
print(A.sort_values('rho_pur_prolif',ascending=False).round(3).to_string(index=False))
A.to_csv('../out/hypoxia_axis.csv',index=False)
# hypoxia sub-scores vs each other and prolif
for k in ['Hypoxia (core 7)','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']:
    print(k,'vs expanded',round(pspear(S[k],H,[pur])[0],2),'vs prolif',round(pspear(S[k],S['Proliferation'],[pur])[0],2))
