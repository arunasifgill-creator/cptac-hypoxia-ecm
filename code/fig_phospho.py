"""Figure 7: phosphorylation of the collagen export machinery along the hypoxia axis."""
import os, re, numpy as np, pandas as pd, matplotlib, warnings; warnings.filterwarnings('ignore')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from load import load; from core import groups
from modules import *
OUT='../figures'; COL={'TNBC':'#eb6834','HER2+':'#2a78d6','HR+/HER2-':'#1baf7a'}; ORDER=['TNBC','HER2+','HR+/HER2-']
INK,INK2,GRID='#1f1f1e','#5f5e5a','#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
 'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9,'axes.titleweight':'bold','axes.titlelocation':'left',
 'legend.frameon':False,'pdf.fonttype':42,'savefig.dpi':600})
rng=np.random.default_rng(4)
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index); S=all_scores(p); prol=S['Proliferation']
P=pd.read_csv('../data/phos/phospho_all.csv',index_col=0).reindex(p.index)
X=pd.read_csv('../out/phospho_occupancy.csv')
def fmtp(v): return 'p < 0.001' if v<.001 else f'p = {v:.3f}' if v<.01 else f'p = {v:.2f}'
def pos(site):
    m_=re.findall(r'[STY](\d+)',site.split('_',1)[1]); return int(m_[0]) if m_ else 0
fig=plt.figure(figsize=(7.4,6.6)); gs=fig.add_gridspec(2,3,height_ratios=[1.15,1],hspace=.6,wspace=.6)
# A: TANGO1 / cTAGE5 / SEC31A sites ordered by residue
ax=fig.add_subplot(gs[0,:2])
E=X[X.gene.isin(['MIA3','MIA2','SEC31A'])].copy(); E['pos']=E.site.map(pos); E=E.sort_values(['gene','pos'])
gcol={'MIA3':'#a63d12','MIA2':'#eda100','SEC31A':'#2a78d6'}; gname={'MIA3':'MIA3 (TANGO1)','MIA2':'MIA2 (cTAGE5/TALI)','SEC31A':'SEC31A'}
for gn,dd in E.groupby('gene',sort=False):
    sig=dd.q_occ<.05
    ax.scatter(dd.pos[~sig],dd.rho_hyp_occ[~sig],s=16,facecolor='white',edgecolor=gcol[gn],lw=1,zorder=3)
    ax.scatter(dd.pos[sig],dd.rho_hyp_occ[sig],s=22,color=gcol[gn],zorder=4,label=gname[gn])
    if not sig.any(): ax.scatter([],[],s=22,color=gcol[gn],label=gname[gn])
for _,r in E[E.q_occ<.05].iterrows():
    ax.annotate(r.site.split('_',1)[1].replace('s','').replace('t','').replace('_','/'),(r.pos,r.rho_hyp_occ),xytext=(3,3 if r.rho_hyp_occ>0 else -8),textcoords='offset points',fontsize=5.6,color=INK2)
ax.axhline(0,color=INK2,lw=.6); ax.set_xlabel('Residue position in the CPTAC reference sequence'); ax.set_ylabel('Partial ρ with hypoxia\n(occupancy: adjusted for total protein,\npurity, proliferation)')
ax.set_title('A  Export machinery phosphosites along the hypoxia axis'); ax.legend(fontsize=6.4,loc='lower left',ncol=3,bbox_to_anchor=(0,-.36))
ax.yaxis.grid(True,color=GRID,lw=.5); ax.set_axisbelow(True); ax.text(.99,.97,f'filled = BH q < 0.05 across the {len(X)} sites tested\nn per site 62–122 (Table S14)',transform=ax.transAxes,ha='right',va='top',fontsize=6.0,color=INK2)
# B/C: subtype occupancy for TANGO1 S189 and SEC31A S799
def occ(site,gene): return resid(P[site],[pur,p[gene]])
for k,(site,gene,lab) in enumerate([('MIA3_S189s','MIA3','MIA3 S189 (TANGO1)'),('SEC31A_S799s','SEC31A','SEC31A S799')]):
    ax=fig.add_subplot(gs[1,k]); x=occ(site,gene); data=[x[g==kk].dropna() for kk in ORDER]
    bp=ax.boxplot(data,widths=.5,showfliers=False,patch_artist=True,medianprops=dict(color=INK,lw=1.2),whiskerprops=dict(color=INK2,lw=.6),capprops=dict(color=INK2,lw=.6))
    for pt,kk in zip(bp['boxes'],ORDER): pt.set_facecolor(COL[kk]+'33'); pt.set_edgecolor(COL[kk])
    for i,(kk,d) in enumerate(zip(ORDER,data),1): ax.scatter(i+rng.uniform(-.13,.13,len(d)),d,s=8,color=COL[kk],alpha=.85,lw=.3,edgecolor='white',zorder=3)
    a=compare(x,g); b=compare(x,g,'TNBC','HR+/HER2-')
    qq=bh(np.array([a[0],b[0]])); fq=lambda v:'q < 0.001' if v<.001 else f'q = {v:.3f}' if v<.01 else f'q = {v:.2f}'
    ax.set_xticks([1,2,3]); ax.set_xticklabels([f'{s}\n(n={len(d)})' for s,d in zip(['TNBC','HER2+','HR+/\nHER2−'],data)],fontsize=6.4)
    ax.set_ylabel('Occupancy (adjusted rank)'); ax.set_title(f'{"BC"[k]}  {lab}',fontsize=8.4)
    ax.text(.03,.98,f'vs HER2+ δ={a[1]:+.2f}, {fmtp(a[0])}, {fq(qq[0])}\nvs HR+ δ={b[1]:+.2f}, {fmtp(b[0])}, {fq(qq[1])}\nn = {int(x.notna().sum())} tumors with this site',transform=ax.transAxes,va='top',fontsize=5.7)
    ax.set_ylim(top=ax.get_ylim()[1]+.35*(ax.get_ylim()[1]-ax.get_ylim()[0])); ax.yaxis.grid(True,color=GRID,lw=.5); ax.set_axisbelow(True)
# D: canonical acute-hypoxia readouts
ax=fig.add_subplot(gs[1,2])
K=['EIF4EBP1_T37t','EIF4EBP1_T46t','EIF4EBP1_S65s','EIF4EBP1_T70t','RPS6_S236s','RPS6_S240s','ACACA_S80s','PRKAA1_S501s_T505t','PDHA1_S331s']
L=['4E-BP1 T37','4E-BP1 T46','4E-BP1 S65','4E-BP1 T70','RPS6 S236','RPS6 S240','ACC S80','AMPKα1 S501/T505','PDHA1 S331']
D=X.set_index('site').reindex(K)
for j,(s,l) in enumerate(zip(K,L)):
    r=D.loc[s,'rho_hyp_occ']; q=D.loc[s,'q_occ']
    ax.barh(-j,r,height=.62,color='#5f5e5a' if q<.05 else '#c3c2b7')
ax.set_yticks(range(0,-len(K),-1)); ax.set_yticklabels(L,fontsize=6.3); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.axvline(0,color=INK2,lw=.6); ax.set_xlim(-.5,.5); ax.set_xlabel('Partial ρ with hypoxia (occupancy)\nAdjusted for total protein,\npurity, proliferation'); ax.set_title('D  mTORC1 / AMPK\n    readouts')
ax.text(.02,.02,'dark = BH q < 0.05',transform=ax.transAxes,va='bottom',fontsize=5.8,color=INK2)
for e in ('png','pdf'): fig.savefig(f'{OUT}/Fig7_export_phospho.{e}',bbox_inches='tight',facecolor='white')
print(D[['rho_hyp_occ','q_occ']].round(3))
for site,gene in [('MIA3_S189s','MIA3'),('SEC31A_S799s','SEC31A'),('MIA3_S727s','MIA3')]:
    x=occ(site,gene); print(site,compare(x,g),compare(x,g,'TNBC','HR+/HER2-'))
