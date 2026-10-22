"""Core ECM shift at gene level: raw vs purity-adjusted, plus composite."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from core import *
import warnings; warnings.filterwarnings('ignore')
OUT=os.path.join(os.path.dirname(__file__),'..','figures')
COL={'TNBC':'#eb6834','HER2+':'#2a78d6','HR+/HER2-':'#1baf7a'}; ORDER=['TNBC','HER2+','HR+/HER2-']
RAW,ADJ='#8f8e88','#eb6834'
INK,INK2,GRID='#1f1f1e','#5f5e5a','#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':INK2,'axes.labelcolor':INK,
 'xtick.color':INK2,'ytick.color':INK2,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
 'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9,'axes.titleweight':'bold',
 'axes.titlelocation':'left','legend.frameon':False,'pdf.fonttype':42,'savefig.dpi':600})
rng=np.random.default_rng(3)
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index)
S={k:score(p,v) for k,v in SETS_PROT.items()}; comp=S['Maturation']-S['Fibrillar collagen']

def d_ci(x,a,b,n=2000):
    xa=x[g==a].dropna().values; xb=x[g==b].dropna().values
    d=cliff(xa,xb); bs=[cliff(rng.choice(xa,len(xa)),rng.choice(xb,len(xb))) for _ in range(n)]
    return d,np.percentile(bs,2.5),np.percentile(bs,97.5),stats.mannwhitneyu(xa,xb).pvalue
def fmt_p(pv): return 'p < 0.001' if pv<.001 else (f'p = {pv:.3f}' if pv<.01 else f'p = {pv:.2f}')

fig=plt.figure(figsize=(7.4,7.6)); gs=fig.add_gridspec(2,2,height_ratios=[1.5,1],hspace=.46,wspace=.42)

def forest(ax,genes,title,note):
    rows=[]
    for j,gene in enumerate(genes):
        for k,(lab,x,cc) in enumerate([('Raw',p[gene],RAW),('Purity-adjusted',purity_adjust(p[gene],pur),ADJ)]):
            d,lo,hi,pv=d_ci(x,'TNBC','HER2+'); yy=-j+(.19 if k==0 else -.19)
            ax.plot([lo,hi],[yy,yy],color=cc,lw=1.3,solid_capstyle='round')
            ax.scatter(d,yy,s=25,color=cc if pv<.05 else 'white',edgecolor=cc,lw=1.2,zorder=3)
            rows.append((gene,lab,d,lo,hi,pv))
    ax.axvline(0,color=INK2,lw=.7)
    ax.set_yticks(range(0,-len(genes),-1)); ax.set_yticklabels(genes,fontsize=7.5)
    ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False); ax.set_xlim(-1,1)
    ax.yaxis.grid(True,color=GRID,lw=.5); ax.set_axisbelow(True)
    ax.set_title(title); ax.set_xlabel("Cliff's δ, TNBC vs HER2+")
    ax.set_ylim(-len(genes)-0.9,0.6)
    ax.text(-0.97,-len(genes)-0.5,note,fontsize=6.4,color=INK2,va='center')
    return rows

axA=fig.add_subplot(gs[0,0])
rA=forest(axA,['COL1A1','COL1A2','COL3A1','COL5A1','COL6A1','COL6A3','COL11A1'],
  'A  Fibrillar collagen protein','Differences weaken after purity adjustment\n→ stromal in origin')
axB=fig.add_subplot(gs[0,1])
rB=forest(axB,['PLOD1','PLOD2','PLOD3','P4HA1','P4HA2','SERPINH1','P3H1','PPIB','CRTAP','FKBP10','COLGALT1','LOX','LOXL2'],
  'B  Collagen maturation enzymes','Differences strengthen after adjustment\n→ tumor-cell intrinsic')
axB.set_xlabel(''); axB.legend(handles=[plt.Line2D([],[],marker='o',color=RAW,lw=1.3,ms=5,label='Raw'),
                    plt.Line2D([],[],marker='o',color=ADJ,lw=1.3,ms=5,label='Purity-adjusted'),
                    plt.Line2D([],[],marker='o',color=INK2,mfc='white',lw=0,ms=5,label='open = p ≥ 0.05')],
            loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=3,fontsize=6.3,columnspacing=1.0,handletextpad=.4,handlelength=1.4)
pd.DataFrame(rA+rB,columns=['gene','adjustment','d','lo','hi','p']).to_csv(f'{OUT}/Fig1_gene_level.csv',index=False)

# C: composite boxplot
axC=fig.add_subplot(gs[1,0])
data=[comp[g==k].dropna() for k in ORDER]
bp=axC.boxplot(data,widths=.52,showfliers=False,patch_artist=True,medianprops=dict(color=INK,lw=1.2),
               whiskerprops=dict(color=INK2,lw=.6),capprops=dict(color=INK2,lw=.6))
for patch,k in zip(bp['boxes'],ORDER):
    patch.set_facecolor(COL[k]+'33'); patch.set_edgecolor(COL[k]); patch.set_linewidth(1)
for i,(k,d) in enumerate(zip(ORDER,data),1):
    axC.scatter(i+rng.uniform(-.14,.14,len(d)),d,s=11,color=COL[k],alpha=.85,lw=.4,edgecolor='white',zorder=3)
axC.set_xticks([1,2,3]); axC.set_xticklabels([f'{s}\n(n={len(d)})' for s,d in zip(['TNBC','HER2+','HR+/\nHER2−'],data)],fontsize=7)
axC.axhline(0,color=INK2,lw=.7,ls=(0,(3,2)))
axC.set_ylabel('Maturation − deposition score'); axC.yaxis.grid(True,color=GRID,lw=.5); axC.set_axisbelow(True)
a=compare(comp,g); b=compare(comp,g,'TNBC','HR+/HER2-')
axC.set_title('C  Composite score separates TNBC')
axC.text(.03,.97,f'vs HER2+: δ={a[1]:+.2f}, {fmt_p(a[0])}\nvs HR+: δ={b[1]:+.2f}, {fmt_p(b[0])}',
         transform=axC.transAxes,va='top',fontsize=6.8,color=INK2)
axC.set_ylim(top=axC.get_ylim()[1]+.12)

# D: per-sample ranked composite
axD=fig.add_subplot(gs[1,1])
ss=comp.dropna().sort_values(); cols=[COL.get(g[i],'#cccccc') for i in ss.index]
axD.bar(range(len(ss)),ss.values,color=cols,width=1.0,linewidth=0)
axD.axhline(0,color=INK2,lw=.7)
axD.set_xlabel('Tumors, ranked by composite score'); axD.set_ylabel('Maturation − deposition score')
axD.set_title('D  Per-tumor distribution')
axD.set_xlim(-1,len(ss)); axD.set_xticks([])
axD.legend(handles=[plt.Rectangle((0,0),1,1,color=COL[k],label=k) for k in ORDER],loc='upper left',fontsize=6.5)
axD.yaxis.grid(True,color=GRID,lw=.5); axD.set_axisbelow(True)
(lambda *a, **k: None)('Figure 1. TNBC shifts from collagen deposition toward enzymatic maturation',
             x=.02,ha='left',fontsize=10,fontweight='bold',y=.995)
for ext in ('png','pdf'): fig.savefig(f'{OUT}/Fig1_core_ECM_shift.{ext}',bbox_inches='tight',facecolor='white')
print('fig1 done')
