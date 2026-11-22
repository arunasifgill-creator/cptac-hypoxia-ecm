"""Correlation architecture: clustered heatmap + within-subtype module preservation."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import squareform
from scipy import stats
from core import *
import warnings; warnings.filterwarnings('ignore')
OUT=os.path.join(os.path.dirname(__file__),'..','figures')
INK,INK2,GRID='#1f1f1e','#5f5e5a','#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':INK2,'axes.labelcolor':INK,
 'xtick.color':INK2,'ytick.color':INK2,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titlesize':9,'axes.titleweight':'bold','axes.titlelocation':'left','legend.frameon':False,
 'pdf.fonttype':42,'savefig.dpi':600})
DIV=LinearSegmentedColormap.from_list('bwr',['#1c5cab','#2a78d6','#9ec5f4','#f0efec','#eda7a0','#c0392b','#8c2317'])

p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index)
MOD={'Fibrillar collagen':['COL1A1','COL1A2','COL3A1','COL5A1'],
 'Maturation / modification':['PLOD1','PLOD2','PLOD3','P4HA1','P4HA2','SERPINH1','P3H1','PPIB','CRTAP','FKBP10','COLGALT1'],
 'Crosslink & processing':['LOX','LOXL2','BMP1','PCOLCE'],
 'ER export':['MIA3','MIA2','SEC23A','SEC24D'],
 'UPR chaperones':['HSPA5','P4HB','PDIA4','PDIA6','HYOU1','CALR','CANX'],
 'UPR targets':['HSP90B1','SDF2L1','MANF','ASNS','ERO1A'],
 'Hypoxia':['SLC2A1','PGK1','LDHA','NDRG1','CA9'],
 'Iron':['SLC11A2','TFRC','STEAP3','FTH1','FTL'],
 'MMP turnover':['MMP2','MMP9','MMP14']}
genes=[x for v in MOD.values() for x in v]
gene2mod={x:k for k,v in MOD.items() for x in v}
A=pd.DataFrame({x:purity_adjust(p[x],pur) for x in genes})
R=A.corr(method='spearman')
D=squareform(np.clip(1-R.values,0,2),checks=False)
Z=hc.linkage(D,method='average'); order=hc.leaves_list(Z); gl=[genes[i] for i in order]
Rc=R.loc[gl,gl]
Rc.to_csv(f'{OUT}/Fig5_correlation_matrix.csv')

MCOL={'Fibrillar collagen':'#1baf7a','Maturation / modification':'#a63d12','Crosslink & processing':'#eda100',
 'ER export':'#008300','UPR chaperones':'#2a78d6','UPR targets':'#1c5cab','Hypoxia':'#e87ba4',
 'Iron':'#eb6834','MMP turnover':'#5f5e5a'}

fig=plt.figure(figsize=(7.5,9.4))
gs=fig.add_gridspec(2,2,height_ratios=[3.05,1],width_ratios=[1,1],hspace=.30,wspace=.5)
axH=fig.add_subplot(gs[0,:])
im=axH.imshow(Rc.values,cmap=DIV,vmin=-1,vmax=1,interpolation='nearest')
axH.set_xticks(range(len(gl))); axH.set_yticks(range(len(gl)))
axH.set_xticklabels(gl,rotation=90,fontsize=5.4); axH.set_yticklabels(gl,fontsize=5.4)
for t,gn in zip(axH.get_xticklabels(),gl): t.set_color(MCOL[gene2mod[gn]])
for t,gn in zip(axH.get_yticklabels(),gl): t.set_color(MCOL[gene2mod[gn]])
axH.tick_params(length=0); [sp.set_visible(False) for sp in axH.spines.values()]
cb=fig.colorbar(im,ax=axH,fraction=.022,pad=.015,ticks=[-1,-.5,0,.5,1])
cb.set_label('Spearman ρ',fontsize=7); cb.ax.tick_params(labelsize=6.5,length=2); cb.outline.set_visible(False)
axH.legend(handles=[plt.Rectangle((0,0),1,1,color=MCOL[k],label=k) for k in MOD],
           loc='lower center',bbox_to_anchor=(.5,1.025),ncol=5,fontsize=6.2,columnspacing=1.2,handlelength=1.1)
axH.set_title('A  Co-expression architecture (protein, adjusted for purity, n = 122 tumors)',pad=34)

# B: module preservation TNBC vs HR+
axB=fig.add_subplot(gs[1,0])
core_genes=MOD['Fibrillar collagen']+MOD['Maturation / modification']+MOD['Crosslink & processing']
i1,i2=g=='TNBC',g=='HR+/HER2-'
R1=pd.DataFrame({x:purity_adjust(p[x][i1],pur[i1]) for x in core_genes}).corr('spearman')
R2=pd.DataFrame({x:purity_adjust(p[x][i2],pur[i2]) for x in core_genes}).corr('spearman')
iu=np.triu_indices(len(core_genes),1); x1,y1=R1.values[iu],R2.values[iu]
rr,pp=stats.spearmanr(x1,y1)
# Gene pairs are not independent, so the correlation between the two pairwise-rho vectors cannot be
# tested against a null of zero. The null here permutes SUBTYPE LABELS ACROSS TUMOURS, keeping the
# group sizes, and asks how well two random splits of the same sizes agree (see a11_robustness.py).
PV=pd.read_csv(os.path.join(OUT,'..','out','preservation_permutation.csv')).iloc[0]
null_mean,null_lo,null_hi=PV.perm_null_mean,PV.perm_lo,PV.perm_hi
perm_p,obs_lo,obs_hi=PV.perm_p,PV.boot_lo,PV.boot_hi
pct_obs=PV.percentile_of_observed
assert int(PV.n_proteins)==len(core_genes), 'Figure 5B and a11_robustness.py must use the same genes'
axB.axhline(0,color=GRID,lw=.6); axB.axvline(0,color=GRID,lw=.6)
axB.plot([-.25,1],[-.25,1],color=INK2,lw=.8,ls=(0,(3,2)),zorder=1)
axB.scatter(x1,y1,s=11,color='#a63d12',alpha=.7,lw=.3,edgecolor='white',zorder=3)
axB.set_xlim(-.25,1); axB.set_ylim(-.25,1); axB.set_aspect('equal')
axB.set_xlabel(f'Pairwise ρ within TNBC (n={int(i1.sum())})'); axB.set_ylabel(f'Pairwise ρ within\nHR+/HER2− (n={int(i2.sum())})')
axB.set_title('B  Module structure differs between subtypes')
axB.text(.03,.97,f'observed ρ = {rr:+.2f} ({obs_lo:.2f}–{obs_hi:.2f})\nnull, permuted subtype labels: {null_mean:.2f} ({null_lo:.2f}–{null_hi:.2f})\nobserved at the {pct_obs:.1f}th percentile of the null; {len(x1)} pairs',
          transform=axB.transAxes,va='top',fontsize=5.6,color=INK2,zorder=6,bbox=dict(fc='white',ec='none',alpha=.85,pad=1.5))

# C: module set-point
axC=fig.add_subplot(gs[1,1])
S={k:score(p,v) for k,v in SETS_PROT.items()}
mods=['Fibrillar collagen','Maturation','UPR chaperones','UPR targets','Hypoxia']
sc={'Fibrillar collagen':score(p,MOD['Fibrillar collagen']),'Maturation':score(p,MOD['Maturation / modification']),
    'UPR chaperones':score(p,MOD['UPR chaperones']),'UPR targets':score(p,MOD['UPR targets']),'Hypoxia':score(p,MOD['Hypoxia'])}
COLG={'TNBC':'#eb6834','HER2+':'#2a78d6','HR+/HER2-':'#1baf7a'}
w=.26
for j,k in enumerate(mods):
    for q,grp in enumerate(['TNBC','HER2+','HR+/HER2-']):
        v=sc[k][g==grp].dropna()
        axC.bar(j+(q-1)*w,v.median(),width=w*.9,color=COLG[grp],linewidth=0)
        lo,hi=np.percentile(v,[25,75]); axC.plot([j+(q-1)*w]*2,[lo,hi],color=INK,lw=.9)
axC.set_xticks(range(len(mods)))
axC.set_xticklabels(['Fibrillar\ncollagen','Maturation','UPR\nchaperones','UPR\ntargets','Hypoxia'],fontsize=6.5)
axC.set_ylabel('Module score\n(median, IQR)\nn = 28 / 15 / 74'); axC.set_ylim(0,.95)
axC.yaxis.grid(True,color=GRID,lw=.5); axC.set_axisbelow(True)
axC.set_title('C  Module set-points by subtype')
axC.legend(handles=[plt.Rectangle((0,0),1,1,color=COLG[k],label=k) for k in COLG],loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,fontsize=6.2,columnspacing=1.0,handlelength=1.1)
(lambda *a, **k: None)('Figure 5. Collagen processing and ER stress form separable modules whose set-point and internal structure both differ in TNBC',
             x=.02,ha='left',fontsize=9.5,fontweight='bold',y=.995)
for ext in ('png','pdf','jpg'): fig.savefig(f'{OUT}/Fig5_correlation_architecture.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
print('fig5 done; module preservation rho=',round(rr,3))
