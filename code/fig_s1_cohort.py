"""Supplementary: cohort composition and data quality."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from core import *
import warnings; warnings.filterwarnings('ignore')
OUT=os.path.join(os.path.dirname(__file__),'..','figures')
COL={'TNBC':'#eb6834','HER2+':'#2a78d6','HR+/HER2-':'#1baf7a'}; ORDER=['TNBC','HER2+','HR+/HER2-']
INK,INK2,GRID='#1f1f1e','#5f5e5a','#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':INK2,'axes.labelcolor':INK,
 'xtick.color':INK2,'ytick.color':INK2,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titlesize':9,'axes.titleweight':'bold','axes.titlelocation':'left','legend.frameon':False,
 'pdf.fonttype':42,'savefig.dpi':600})
rng=np.random.default_rng(5)
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index)
fig=plt.figure(figsize=(7.4,5.4)); gs=fig.add_gridspec(2,3,hspace=.62,wspace=.42)

# A: PAM50 composition per group
axA=fig.add_subplot(gs[0,0])
pam=c['PAM50'].reindex(p.index)
cats=['Basal','Her2','LumA','LumB','Normal-like']
shades=['#1f1f1e','#5f5e5a','#8f8e88','#c3c2b7','#e6e5e0']
bot=np.zeros(3)
for cat,sh in zip(cats,shades):
    vals=[((g==k)&(pam==cat)).sum() for k in ORDER]
    axA.bar(range(3),vals,bottom=bot,color=sh,width=.62,linewidth=.5,edgecolor='white',label=cat); bot+=vals
axA.set_xticks(range(3)); axA.set_xticklabels(['TNBC\n(28)','HER2+\n(15)','HR+/\nHER2−\n(74)'],fontsize=6.8)
axA.set_ylabel('Tumors'); axA.set_title('A  PAM50 composition')
axA.legend(fontsize=5.8,loc='upper left',bbox_to_anchor=(0,1.02),handlelength=1.0,labelspacing=.25)

# B: purity by group
axB=fig.add_subplot(gs[0,1])
data=[pur[g==k].dropna() for k in ORDER]
bp=axB.boxplot(data,widths=.5,showfliers=False,patch_artist=True,medianprops=dict(color=INK,lw=1.2),
               whiskerprops=dict(color=INK2,lw=.6),capprops=dict(color=INK2,lw=.6))
for patch,k in zip(bp['boxes'],ORDER):
    patch.set_facecolor(COL[k]+'33'); patch.set_edgecolor(COL[k]); patch.set_linewidth(1)
for i,(k,d) in enumerate(zip(ORDER,data),1):
    axB.scatter(i+rng.uniform(-.13,.13,len(d)),d,s=9,color=COL[k],alpha=.85,lw=.3,edgecolor='white',zorder=3)
axB.set_xticks([1,2,3]); axB.set_xticklabels(['TNBC','HER2+','HR+/\nHER2−'],fontsize=6.8)
axB.set_ylabel('ESTIMATE tumor purity'); axB.set_title('B  Tumor purity')
a=compare(pur,g); b=compare(pur,g,'TNBC','HR+/HER2-')
axB.set_xlabel(f'vs HER2+ p = {a[0]:.2f}; vs HR+ p = {b[0]:.2f}',fontsize=6.3)
axB.yaxis.grid(True,color=GRID,lw=.5); axB.set_axisbelow(True)

# C: stromal score vs collagen (validation)
axC=fig.add_subplot(gs[0,2])
st=c['ESTIMATE Stromal Score'].reindex(p.index)
ok=st.notna()&p['COL1A1'].notna(); r1,_=stats.spearmanr(st[ok],p['COL1A1'][ok])
ok2=st.notna()&m['COL1A1'].notna(); r2,_=stats.spearmanr(st[ok2],m['COL1A1'][ok2])
axC.scatter(st[ok],p['COL1A1'][ok],s=11,color='#a63d12',alpha=.75,lw=.3,edgecolor='white')
b1=np.polyfit(st[ok],p['COL1A1'][ok],1); xx=np.linspace(st[ok].min(),st[ok].max(),20)
axC.plot(xx,np.polyval(b1,xx),color=INK2,lw=1)
axC.set_xlabel('ESTIMATE stromal score'); axC.set_ylabel('COL1A1 protein (log2 ratio)')
axC.set_title('C  Raw collagen tracks stroma')
axC.text(.03,.97,f'protein ρ = {r1:+.2f}\nmRNA ρ = {r2:+.2f}',transform=axC.transAxes,va='top',fontsize=6.5,color=INK2)
axC.yaxis.grid(True,color=GRID,lw=.5); axC.set_axisbelow(True)

# D: protein coverage
axD=fig.add_subplot(gs[1,0])
cov=p.notna().sum()/len(p)*100
axD.hist(cov,bins=np.arange(0,105,5),color='#2a78d6',edgecolor='white',linewidth=.5)
axD.set_xlabel(f'Samples with a value (%), {p.shape[1]} genes'); axD.set_ylabel('Genes')
axD.set_title('D  Protein coverage')
axD.text(.03,.95,f'{(cov==100).sum()} genes complete\n{(cov==0).sum()} genes absent',transform=axD.transAxes,va='top',fontsize=6.5,color=INK2)
axD.yaxis.grid(True,color=GRID,lw=.5); axD.set_axisbelow(True)

# E: mRNA-protein concordance
axE=fig.add_subplot(gs[1,1])
conc=[]
for gene in p.columns:
    if p[gene].notna().sum()<60 or m[gene].notna().sum()<60: continue
    ok=p[gene].notna()&m[gene].notna(); r,_=stats.spearmanr(p[gene][ok],m[gene][ok]); conc.append((gene,r))
C=pd.DataFrame(conc,columns=['gene','rho']).sort_values('rho')
C.to_csv(f'{OUT}/FigS1_mrna_protein_concordance.csv',index=False)
axE.hist(C.rho,bins=np.arange(-.6,1.05,.1),color='#1baf7a',edgecolor='white',linewidth=.5)
axE.axvline(C.rho.median(),color=INK,lw=1,ls=(0,(3,2)))
axE.set_xlabel('Spearman ρ, mRNA vs protein'); axE.set_ylabel('Genes')
axE.set_title('E  mRNA–protein ρ')
axE.text(.03,.95,f'median ρ = {C.rho.median():.2f}\nn = {len(C)} genes',transform=axE.transAxes,va='top',fontsize=6.5,color=INK2)
axE.yaxis.grid(True,color=GRID,lw=.5); axE.set_axisbelow(True)

# F: TMT plex batch check
axF=fig.add_subplot(gs[1,2])
plex=c['TMT Plex'].reindex(p.index).astype(str).str.split('|').str[0]
S={k:score(p,v) for k,v in SETS_PROT.items()}; comp=S['Maturation']-S['Fibrillar collagen']
pl=sorted(plex.unique(),key=lambda x:(len(x),x))
vals=[comp[plex==x].dropna() for x in pl]
axF.boxplot(vals,widths=.6,showfliers=False,patch_artist=True,
            boxprops=dict(facecolor='#e6e5e0',edgecolor=INK2,linewidth=.7),
            medianprops=dict(color=INK,lw=1),whiskerprops=dict(color=INK2,lw=.6),capprops=dict(color=INK2,lw=.6))
axF.set_xticks(range(1,len(pl)+1)); axF.set_xticklabels(pl,fontsize=5.5,rotation=90)
axF.set_xlabel('TMT plex'); axF.set_ylabel('Maturation − deposition')
H,pv=stats.kruskal(*[v for v in vals if len(v)>2])
axF.set_title('F  TMT batch check')
axF.text(.03,.97,f'Kruskal–Wallis p = {pv:.2f}',transform=axF.transAxes,va='top',fontsize=6.5,color=INK2)
axF.yaxis.grid(True,color=GRID,lw=.5); axF.set_axisbelow(True)
(lambda *a, **k: None)('Figure S1. Cohort composition and data quality',x=.02,ha='left',fontsize=10,fontweight='bold',y=1.0)
for ext in ('png','pdf'): fig.savefig(f'{OUT}/FigS1_cohort_quality.{ext}',bbox_inches='tight',facecolor='white')
print('S-cohort done; concordance median',round(C.rho.median(),2),'plex p=',round(pv,3))
