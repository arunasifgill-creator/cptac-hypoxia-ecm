"""Phosphosite validation: which sites track the hypoxia score, PERK score and subtype.
Sites with >= 60 quantified tumors. Where total protein is available, the phosphosite is also
rank-residualised on its own protein (occupancy proxy)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from load import load; from core import groups
from modules import *
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index); S=all_scores(p); prol=S['Proliferation']
P=pd.read_csv('../data/phos/phospho_all.csv',index_col=0).reindex(p.index)
P=P[[x for x in P.columns if P[x].notna().sum()>=60]]
rows=[]
for site in P.columns:
    gene=site.split('_')[0].replace('s','',0)
    x=P[site]; has=gene in p.columns and p[gene].notna().sum()>=60
    r1,p1,n=pspear(x,S['Hypoxia consensus'],[pur,prol])
    r2,p2,_=pspear(x,S['UPR PERK/ATF4 targets'],[pur,prol])
    pv,d=compare(purity_adjust(x,pur),g)
    if has:
        r3,p3,_=pspear(x,S['Hypoxia consensus'],[pur,prol,p[gene]]); pv4,d4=compare(resid(x,[pur,p[gene]]),g)
    else: r3=p3=d4=pv4=np.nan
    rows.append(dict(site=site,gene=gene,n=n,rho_hyp=r1,p_hyp=p1,rho_perk=r2,p_perk=p2,d_TNBC_HER2=d,p_d=pv,protein=has,rho_hyp_occ=r3,p_hyp_occ=p3,d_occ=d4,p_d_occ=pv4))
R=pd.DataFrame(rows); R['q_hyp']=bh(R.p_hyp.values); R['q_perk']=bh(R.p_perk.values); R['q_d']=bh(R.p_d.values)
R.to_csv('../out/phospho_sites.csv',index=False)
# occupancy subset: sites whose own total protein was quantified, used for Figure 7
O=R[R.protein].copy(); O['q_occ']=bh(O.p_hyp_occ.values)
O.to_csv('../out/phospho_occupancy.csv',index=False)
print('occupancy sites',len(O),'sig q<.05',(O.q_occ<.05).sum())
print('sites tested',len(R),'sig hyp q<.05',(R.q_hyp<.05).sum())
key=['EIF4EBP1_T37t','EIF4EBP1_T46t','EIF4EBP1_S65s','EIF4EBP1_T70t','RPS6_S235s_S236s','RPS6_S236s','RPS6_S240s','RPS6_S244s','RPS6KB1_T421t_S424s','MTOR_S2481s',
 'ACACA_S80s','ACACA_S29s','PRKAA1_S501s_T505t','PRKAA1_T505t','PDHA1_S270s','PDHA1_S331s','EIF2S1_S52s','EIF2AK3_S555s','EIF2AK3_S715s','EIF2AK3_S1094s','ERN1_S724s',
 'EIF2AK4_T667t','YAP1_S128s','YAP1_S386s','WWTR1_S90s','PTK2_S910s','PTK2_Y576y','MIA3_S1741s','SEC31A_S527s','SEC24D_S267s','LDHA_Y268y','PKM_S37s','HSP90B1_S306s','SMAD3_T8t']
K=R.set_index('site').reindex([k for k in key if k in R.site.values])
print(K[['n','rho_hyp','q_hyp','rho_perk','q_perk','d_TNBC_HER2','q_d','rho_hyp_occ','d_occ']].round(3).to_string())
print('\nTop hypoxia-associated sites:'); print(R.sort_values('rho_hyp',ascending=False).head(15)[['site','n','rho_hyp','q_hyp','d_TNBC_HER2']].round(3).to_string(index=False))
print(R.sort_values('rho_hyp').head(10)[['site','n','rho_hyp','q_hyp','d_TNBC_HER2']].round(3).to_string(index=False))
