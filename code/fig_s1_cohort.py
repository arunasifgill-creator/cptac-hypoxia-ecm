"""Supplementary Figure S1: cohort composition, data quality and batch structure.

Revision changes:
  - panel F now tests every module across TMT plexes with BH correction, instead of the
    maturation-minus-deposition composite alone (reviewer 1, minor point 3)
  - new panel G shows how subtypes are distributed across the 15 TMT plexes (reviewer 2, point 15)
  - new panel H shows the hypoxia consensus score by plex, since hypoxia is the main exposure
  - every panel now prints its sample size, and panels with several tests print BH q (reviewer 1, minor point 5)
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from core import *
from modules import all_scores, PROT
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
plex=c['TMT Plex'].reindex(p.index).astype(str).str.split('|').str[0]
S=all_scores(p)
def fmtq(v): return 'q < 0.001' if v<.001 else f'q = {v:.3f}' if v<.01 else f'q = {v:.2f}'
def fmtp(v): return 'p < 0.001' if v<.001 else f'p = {v:.3f}' if v<.01 else f'p = {v:.2f}'

fig=plt.figure(figsize=(7.4,8.6)); gs=fig.add_gridspec(3,3,hspace=.72,wspace=.46)

# A: PAM50 composition per group
axA=fig.add_subplot(gs[0,0])
pam=c['PAM50'].reindex(p.index)
cats=['Basal','Her2','LumA','LumB','Normal-like']
shades=['#1f1f1e','#5f5e5a','#8f8e88','#c3c2b7','#e6e5e0']
bot=np.zeros(3)
for cat,sh in zip(cats,shades):
    vals=[((g==k)&(pam==cat)).sum() for k in ORDER]
    axA.bar(range(3),vals,bottom=bot,color=sh,width=.62,linewidth=.5,edgecolor='white',label=cat); bot+=vals
axA.set_xticks(range(3)); axA.set_xticklabels(['TNBC\n(n=28)','HER2+\n(n=15)','HR+/\nHER2−\n(n=74)'],fontsize=6.8)
axA.set_ylabel('Tumors'); axA.set_title('A  PAM50 composition')
axA.text(.98,.98,'n = 117 assigned\n(5 Normal-like set aside)',transform=axA.transAxes,va='top',ha='right',fontsize=5.8,color=INK2)
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
axB.set_xticks([1,2,3]); axB.set_xticklabels([f'TNBC\n(n={len(data[0])})',f'HER2+\n(n={len(data[1])})',f'HR+/\nHER2−\n(n={len(data[2])})'],fontsize=6.4)
axB.set_ylabel('ESTIMATE tumor purity'); axB.set_title('B  Tumor purity')
a=compare(pur,g); b=compare(pur,g,'TNBC','HR+/HER2-')
qB=bh(np.array([a[0],b[0]]))
axB.text(.03,.98,f'vs HER2+: {fmtp(a[0])}, {fmtq(qB[0])}\nvs HR+: {fmtp(b[0])}, {fmtq(qB[1])}',transform=axB.transAxes,va='top',fontsize=5.9,color=INK2)
axB.set_ylim(top=axB.get_ylim()[1]+.12)
axB.yaxis.grid(True,color=GRID,lw=.5); axB.set_axisbelow(True)

# C: stromal score vs collagen (validation)
axC=fig.add_subplot(gs[0,2])
st=c['ESTIMATE Stromal Score'].reindex(p.index)
ok=st.notna()&p['COL1A1'].notna(); r1,pc1=stats.spearmanr(st[ok],p['COL1A1'][ok])
ok2=st.notna()&m['COL1A1'].notna(); r2,pc2=stats.spearmanr(st[ok2],m['COL1A1'][ok2])
axC.scatter(st[ok],p['COL1A1'][ok],s=11,color='#a63d12',alpha=.75,lw=.3,edgecolor='white')
b1=np.polyfit(st[ok],p['COL1A1'][ok],1); xx=np.linspace(st[ok].min(),st[ok].max(),20)
axC.plot(xx,np.polyval(b1,xx),color=INK2,lw=1)
axC.set_xlabel('ESTIMATE stromal score'); axC.set_ylabel('COL1A1 protein (log2 ratio)')
axC.set_title('C  Raw collagen tracks stroma')
axC.text(.03,.97,f'protein ρ = {r1:+.2f}, {fmtp(pc1)} (n={ok.sum()})\nmRNA ρ = {r2:+.2f}, {fmtp(pc2)} (n={ok2.sum()})',transform=axC.transAxes,va='top',fontsize=5.9,color=INK2)
axC.yaxis.grid(True,color=GRID,lw=.5); axC.set_axisbelow(True)

# D: protein coverage
axD=fig.add_subplot(gs[1,0])
cov=p.notna().sum()/len(p)*100
axD.hist(cov,bins=np.arange(0,105,5),color='#2a78d6',edgecolor='white',linewidth=.5)
axD.set_xlabel(f'Tumors with a value (%)\nn = {p.shape[1]} genes queried'); axD.set_ylabel('Genes')
axD.set_title('D  Protein coverage')
axD.text(.03,.95,f'{(cov>=80).sum()} genes ≥ 80%\n{(cov==100).sum()} genes complete\n{(cov==0).sum()} genes absent',transform=axD.transAxes,va='top',fontsize=6.2,color=INK2)
axD.yaxis.grid(True,color=GRID,lw=.5); axD.set_axisbelow(True)

# E: mRNA-protein concordance
axE=fig.add_subplot(gs[1,1])
conc=[]
for gene in p.columns:
    if p[gene].notna().sum()<60 or m[gene].notna().sum()<60: continue
    ok=p[gene].notna()&m[gene].notna(); r,pv=stats.spearmanr(p[gene][ok],m[gene][ok]); conc.append((gene,r,pv,ok.sum()))
C=pd.DataFrame(conc,columns=['gene','rho','p','n']); C['q']=bh(C['p'].values); C=C.sort_values('rho')
C.to_csv(f'{OUT}/FigS1_mrna_protein_concordance.csv',index=False)
axE.hist(C.rho,bins=np.arange(-.6,1.05,.1),color='#1baf7a',edgecolor='white',linewidth=.5)
axE.axvline(C.rho.median(),color=INK,lw=1,ls=(0,(3,2)))
axE.set_xlabel('Spearman ρ, mRNA vs protein'); axE.set_ylabel('Genes')
axE.set_title('E  mRNA–protein ρ')
axE.text(.03,.95,f'median ρ = {C.rho.median():.2f}\nn = {len(C)} genes\n{(C.q<0.05).sum()} with BH q < 0.05',transform=axE.transAxes,va='top',fontsize=6.2,color=INK2)
axE.yaxis.grid(True,color=GRID,lw=.5); axE.set_axisbelow(True)

# F: TMT plex batch check across ALL modules (reviewer 1, minor 3)
axF=fig.add_subplot(gs[1,2])
B=pd.read_csv(f'{OUT}/../out/batch_qc.csv')
showF=['Hypoxia consensus','UPR canonical chaperones','UPR ATF6 targets','UPR IRE1/XBP1 targets',
       'UPR PERK/ATF4 targets','ER redox','Maturation','Fibrillar collagen','Collagen export',
       'Maturation - deposition','NADPH supply']
lbl={'UPR canonical chaperones':'Chaperones','UPR ATF6 targets':'ATF6 targets','UPR IRE1/XBP1 targets':'IRE1/XBP1 targets',
     'UPR PERK/ATF4 targets':'PERK/ATF4 targets','Maturation - deposition':'Maturation − deposition','Hypoxia consensus':'Hypoxia (consensus)'}
BF=B.set_index('module').loc[showF[::-1]]
cf=['#eb6834' if q<.05 else '#2a78d6' if k.startswith(('UPR','ER')) else '#a63d12' if k in ('Maturation','Fibrillar collagen','Collagen export','Maturation - deposition') else '#8f8e88'
    for k,q in zip(BF.index,BF['q'])]
axF.barh(range(len(BF)),BF.eta2_plex*100,color=cf,height=.68)
for j,(v,q) in enumerate(zip(BF.eta2_plex*100,BF['q'])):
    axF.text(v+.6,j,fmtq(q),va='center',fontsize=5.4,color=INK2)
axF.set_yticks(range(len(BF))); axF.set_yticklabels([lbl.get(s,s) for s in BF.index],fontsize=5.9)
axF.tick_params(axis='y',length=0); axF.spines['left'].set_visible(False)
axF.set_xlim(0,54); axF.set_xlabel('Rank variance explained by\nTMT plex (%), n = 122')
axF.set_title(f'F  Batch check, all modules\n    0 of {len(B)} reach BH q < 0.05',fontsize=8)

# G: subtype distribution across plexes (reviewer 2, point 15)
axG=fig.add_subplot(gs[2,0:2])
CT=pd.read_csv(f'{OUT}/../out/plex_subtype.csv',index_col=0).drop(index='TOTAL')
CT.index=CT.index.astype(str)
pl=sorted(CT.index,key=lambda x:(len(x),x))
CT=CT.loc[pl]
stack=['TNBC','HER2+','HR+/HER2-','Normal-like (excluded)']
cg={'TNBC':'#eb6834','HER2+':'#2a78d6','HR+/HER2-':'#1baf7a','Normal-like (excluded)':'#c3c2b7'}
bot=np.zeros(len(CT))
for k in stack:
    v=CT[k].values if k in CT else np.zeros(len(CT))
    axG.bar(range(len(CT)),v,bottom=bot,color=cg[k],width=.7,linewidth=.5,edgecolor='white',
            label=k.replace('HR+/HER2-','HR+/HER2−')); bot+=v
# The test uses the three assigned clinical groups only. The Normal-like tumors are drawn in the bars but are not a group.
CT3=CT[['TNBC','HER2+','HR+/HER2-']]; CT3=CT3.loc[CT3.sum(1)>0]
chi2,p_chi,dof,_exp=stats.chi2_contingency(CT3.values)
MC=pd.read_csv(f'{OUT}/../out/plex_subtype_test.csv').iloc[0]
axG.set_xticks(range(len(CT))); axG.set_xticklabels(pl,fontsize=6)
axG.set_xlabel('TMT plex'); axG.set_ylabel('Tumors')
axG.set_title('G  Subtypes are spread across the 15 TMT plexes')
axG.text(.01,.97,f'three assigned groups, n = {int(CT3.values.sum())}: χ² = {chi2:.1f}, df = {dof}\nMonte Carlo p = {MC.p_monte_carlo:.2f} (5000 label permutations)',transform=axG.transAxes,va='top',fontsize=6,color=INK2)
axG.set_ylim(top=bot.max()+2.6); axG.set_yticks(range(0,int(bot.max())+1,2))
axG.legend(fontsize=6,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.20),handlelength=1.0,columnspacing=.9)

# H: hypoxia consensus by plex
axH=fig.add_subplot(gs[2,2])
H=S['Hypoxia consensus']
vals=[H[plex==x].dropna() for x in pl]
bp=axH.boxplot([v for v in vals],widths=.6,showfliers=False,patch_artist=True,
            boxprops=dict(facecolor='#e6e5e0',edgecolor=INK2,linewidth=.7),
            medianprops=dict(color=INK,lw=1),whiskerprops=dict(color=INK2,lw=.6),capprops=dict(color=INK2,lw=.6))
axH.set_xticks(range(1,len(pl)+1)); axH.set_xticklabels(pl,fontsize=5.2,rotation=90)
axH.set_xlabel('TMT plex'); axH.set_ylabel('Hypoxia consensus score')
rowH=B.set_index('module').loc['Hypoxia consensus']
axH.set_title('H  Hypoxia by plex')
axH.text(.03,.97,f'Kruskal–Wallis {fmtp(rowH["p"])}\n{fmtq(rowH["q"])}, n = 122',transform=axH.transAxes,va='top',fontsize=6,color=INK2)
axH.yaxis.grid(True,color=GRID,lw=.5); axH.set_axisbelow(True)

for ext in ('png','pdf','jpg'): fig.savefig(f'{OUT}/FigS1_cohort_quality.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
print('S1 done; concordance median',round(C.rho.median(),2),'| plex-subtype p=',round(p_chi,3),
      '| modules q<0.05 across plexes:',int((B.q<0.05).sum()),'of',len(B))
