"""Draft figures for the v3 analyses (hypoxia axis, two-model mediation). 600 dpi PNG + PDF."""
import os, numpy as np, pandas as pd, matplotlib, warnings; warnings.filterwarnings('ignore')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from load import load; from core import groups
from modules import *
OUT='../figures'; os.makedirs(OUT,exist_ok=True)
COL={'TNBC':'#eb6834','HER2+':'#2a78d6','HR+/HER2-':'#1baf7a'}; ORDER=['TNBC','HER2+','HR+/HER2-']
INK,INK2,GRID='#1f1f1e','#5f5e5a','#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
 'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9,'axes.titleweight':'bold','axes.titlelocation':'left',
 'legend.frameon':False,'pdf.fonttype':42,'savefig.dpi':600})
rng=np.random.default_rng(11)
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index); S=all_scores(p)
def save(fig,n):
    for e in ('png','pdf'): fig.savefig(f'{OUT}/{n}.{e}',bbox_inches='tight',facecolor='white')
def fmtp(v): return 'p < 0.001' if v<.001 else f'p = {v:.3f}' if v<.01 else f'p = {v:.2f}'

# ===== Figure: hypoxia axis =====
fig=plt.figure(figsize=(7.4,6.4)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.15],hspace=.62,wspace=.75,width_ratios=[1,1.1,1.05])
ax=fig.add_subplot(gs[0,0]); H=S['Hypoxia consensus']
data=[H[g==k].dropna() for k in ORDER]
bp=ax.boxplot(data,widths=.52,showfliers=False,patch_artist=True,medianprops=dict(color=INK,lw=1.2),whiskerprops=dict(color=INK2,lw=.6),capprops=dict(color=INK2,lw=.6))
for pt,k in zip(bp['boxes'],ORDER): pt.set_facecolor(COL[k]+'33'); pt.set_edgecolor(COL[k])
for i,(k,d) in enumerate(zip(ORDER,data),1): ax.scatter(i+rng.uniform(-.14,.14,len(d)),d,s=9,color=COL[k],alpha=.85,lw=.3,edgecolor='white',zorder=3)
nA=[len(d) for d in data]
ax.set_xticks([1,2,3]); ax.set_xticklabels([f'TNBC\n(n={nA[0]})',f'HER2+\n(n={nA[1]})',f'HR+/\nHER2−\n(n={nA[2]})'],fontsize=6.3); ax.set_ylabel('Hypoxia consensus score')
Ha=purity_adjust(H,pur); a=compare(Ha,g); b=compare(Ha,g,'TNBC','HR+/HER2-')
qA=bh(np.array([a[0],b[0]]))
def fmtq(v): return 'q < 0.001' if v<.001 else f'q = {v:.3f}' if v<.01 else f'q = {v:.2f}'
ax.set_title('A  Hypoxia by subtype'); ax.text(.03,.98,f'vs HER2+ δ={a[1]:+.2f},\n  {fmtp(a[0])}, {fmtq(qA[0])}\nvs HR+ δ={b[1]:+.2f},\n  {fmtp(b[0])}, {fmtq(qA[1])}\nAdjusted for purity',transform=ax.transAxes,va='top',fontsize=5.6,color=INK2)
ax.set_ylim(top=ax.get_ylim()[1]+.1); ax.yaxis.grid(True,color=GRID,lw=.5); ax.set_axisbelow(True)
# B: PLOD2 vs hypoxia scatter, all tumors colored by subtype
ax=fig.add_subplot(gs[0,1]); x=resid(H,[pur,S['Proliferation'],S['CAF']]); y=resid(p['PLOD2'],[pur,S['Proliferation'],S['CAF']])
for k in ORDER[::-1]:
    i=(g==k)&x.notna()&y.notna(); ax.scatter(x[i],y[i],s=10,color=COL[k],alpha=.8,lw=.3,edgecolor='white',label=k)
ok=x.notna()&y.notna(); r,pv=stats.spearmanr(x[ok],y[ok]); bb=np.polyfit(x[ok],y[ok],1); xx=np.linspace(x[ok].min(),x[ok].max(),20); ax.plot(xx,np.polyval(bb,xx),color=INK2,lw=1)
ax.set_xlabel('Hypoxia (adjusted rank)'); ax.set_ylabel('PLOD2 protein (adjusted rank)'); ax.set_title('B  PLOD2 tracks hypoxia')
ax.text(.03,.98,f'ρ = {r:+.2f}, {fmtp(pv)}, n = {int(ok.sum())}\nAdjusted for purity,\nproliferation, CAF',transform=ax.transAxes,va='top',fontsize=5.8,color=INK2)
ax.legend(fontsize=5.8,loc='lower right',handletextpad=.2,borderpad=0)
# C: module ~ hypoxia partial correlations
ax=fig.add_subplot(gs[0,2]); A=pd.read_csv('../out/hypoxia_axis.csv')
show=['Maturation','UPR PERK/ATF4 targets','MMP turnover','ER redox','TGF-b response','UPR canonical chaperones','Mechanotransduction','NADPH supply','Peroxide detox','NRF2 targets','UPR ATF6 targets','Collagen export','Fibrillar collagen','ER-phagy / ERAD']
A=A.set_index('module').loc[show[::-1]]
# primary specification for Figures 3 and 4: purity + proliferation + CAF (CAF module itself excluded)
A['rho_pur_prolif']=A['rho_pur_prolif_caf']; A['q_pur_prolif']=A['q_pur_prolif_caf']
cc=['#a63d12' if k in ('Maturation','MMP turnover','Mechanotransduction','Collagen export','Fibrillar collagen','ER-phagy / ERAD') else '#2a78d6' if k.startswith(('UPR','ER')) else '#1baf7a' if k in ('NADPH supply','Peroxide detox','NRF2 targets') else '#8f8e88' for k in A.index]
ax.barh(range(len(A)),A.rho_pur_prolif,color=cc,height=.65)
for j,(v,q) in enumerate(zip(A.rho_pur_prolif,A.q_pur_prolif)):
    if q<.05: ax.text(v+.01 if v>0 else v-.01,j,'*',va='center',ha='left' if v>0 else 'right',fontsize=7)
ax.set_yticks(range(len(A))); ax.set_yticklabels([s.replace('UPR canonical chaperones','Chaperones').replace('UPR ','').replace(' targets','').replace('TGF-b','TGF-β') for s in A.index],fontsize=6.3); ax.tick_params(axis='y',length=0)
ax.axvline(0,color=INK2,lw=.6); ax.set_xlabel('Partial ρ with hypoxia, n = 122\nAdjusted for purity,\nproliferation, CAF'); ax.set_title('C  Programs on the\n    hypoxia axis'); ax.set_xlim(-.25,.85)
ax.text(.97,.02,'* BH q < 0.05',transform=ax.transAxes,va='bottom',ha='right',fontsize=5.8,color=INK2)
# D: mediation forest
ax=fig.add_subplot(gs[1,:]); R=pd.read_csv('../out/mediation.csv'); R=R[R.set.str.startswith('All')]
outs=['PLOD2','P4HA1','P4HA2','SERPINH1','Maturation','Maturation - deposition','Collagen export']
R=R.set_index('outcome').loc[outs]
y0=np.arange(len(outs))[::-1]
for k,(col,lo,hi,cc_,lab) in enumerate([('direct','direct_lo','direct_hi','#eb6834','Direct: hypoxia → ECM (Model B)'),('indirect','ind_lo','ind_hi','#2a78d6','Indirect via ER stress (Model A)')]):
    yy=y0+(.17 if k==0 else -.17)
    ax.hlines(yy,R[lo],R[hi],color=cc_,lw=1.6); ax.scatter(R[col],yy,s=26,color=cc_,zorder=3,label=lab)
for j,o in enumerate(outs):
    pm=R.loc[o,'prop_mediated']; 
    if pd.notna(pm) and 0<pm<1: ax.text(.86,y0[j],f'{pm*100:.0f}% mediated',va='center',fontsize=6.4,color=INK2)
ax.axvline(0,color=INK2,lw=.6); ax.set_yticks(y0); ax.set_yticklabels([o.replace('Maturation - deposition','Maturation − deposition') for o in outs],fontsize=7.2); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(-.6,1.05); ax.set_xlabel('Standardized effect of hypoxia (95% bootstrap CI, 5000 resamples), n = 122\nAll variables rank-residualized on purity, proliferation and CAF content')
ax.set_title('D  Two models side by side: direct and ER stress-mediated effects of hypoxia (all 122 tumors)')
ax.legend(loc='upper center',fontsize=6.6,bbox_to_anchor=(.5,-.27),ncol=2)
ax.xaxis.grid(True,color=GRID,lw=.5); ax.set_axisbelow(True)
save(fig,'FigH_hypoxia_axis_mediation'); print('saved')
