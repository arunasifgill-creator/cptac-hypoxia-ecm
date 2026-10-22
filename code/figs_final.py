"""Final-set figures that changed for the hypoxia-axis manuscript: Fig2 (pathway overview, expanded modules),
Fig4 (export, TGF-b, ER redox), FigS2 (ER/PR split), FigS3 (iron import vs hypoxia)."""
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
rng=np.random.default_rng(1)
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index); S=all_scores(p); prol=S['Proliferation']
def save(fig,n):
    for e in ('png','pdf'): fig.savefig(f'{OUT}/{n}.{e}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def fmt_p(v): return 'p < 0.001' if v<.001 else f'p = {v:.3f}' if v<.01 else f'p = {v:.2f}'
def d_ci(x,a,b,grp=None,n=2000):
    grp=g if grp is None else grp
    xa=x[grp==a].dropna().values; xb=x[grp==b].dropna().values
    d=cliff(xa,xb); bs=[cliff(rng.choice(xa,len(xa)),rng.choice(xb,len(xb))) for _ in range(n)]
    return d,np.percentile(bs,2.5),np.percentile(bs,97.5),stats.mannwhitneyu(xa,xb).pvalue
def boxstrip(ax,x,title,ylabel,note=None,grp=None,order=ORDER,cols=COL,labels=None):
    grp=g if grp is None else grp
    data=[x[grp==k].dropna() for k in order]
    bp=ax.boxplot(data,widths=.5,showfliers=False,patch_artist=True,medianprops=dict(color=INK,lw=1.2),whiskerprops=dict(color=INK2,lw=.6),capprops=dict(color=INK2,lw=.6))
    for pt,k in zip(bp['boxes'],order): pt.set_facecolor(cols[k]+'33'); pt.set_edgecolor(cols[k]); pt.set_linewidth(1)
    for i,(k,d) in enumerate(zip(order,data),1): ax.scatter(i+rng.uniform(-.13,.13,len(d)),d,s=9,color=cols[k],alpha=.85,lw=.4,edgecolor='white',zorder=3)
    labels=labels or ['TNBC','HER2+','HR+/\nHER2−']
    ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels([f'{s}\n(n={len(d)})' for s,d in zip(labels,data)],fontsize=6.6)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.yaxis.grid(True,color=GRID,lw=.5); ax.set_axisbelow(True)
    if note: ax.text(.02,.98,note,transform=ax.transAxes,va='top',fontsize=6.3,color=INK2)
def adjp(x):
    xx=purity_adjust(x,pur); a=compare(xx,g); b=compare(xx,g,'TNBC','HR+/HER2-')
    return f'vs HER2+: δ={a[1]:+.2f}, {fmt_p(a[0])}\nvs HR+: δ={b[1]:+.2f}, {fmt_p(b[0])}'

# ---------- Figure 2: pathway overview ----------
sections=[('Collagen & ECM',['Fibrillar collagen','Maturation','Maturation - deposition','MMP turnover','Collagen export','ER-phagy / ERAD','Mechanotransduction']),
 ('Hypoxia',['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']),
 ('ER stress',['UPR canonical chaperones','UPR ATF6 targets','UPR IRE1/XBP1 targets','UPR PERK/ATF4 targets','ER redox']),
 ('Antioxidant',['NRF2 targets','NADPH supply','Peroxide detox','Ferroptosis defence']),
 ('Context',['Proliferation','TGF-b response','CAF'])]
label={'TGF-b response':'TGF-β response','UPR canonical chaperones':'Canonical chaperones','UPR ATF6 targets':'ATF6 targets','UPR IRE1/XBP1 targets':'IRE1/XBP1 targets',
 'UPR PERK/ATF4 targets':'PERK/ATF4 targets','Maturation - deposition':'Maturation − deposition','Hypoxia consensus':'Hypoxia (consensus)','Hypoxia: glycolysis':'Glycolysis',
 'Hypoxia: mitochondrial':'Mitochondrial adaptation','Hypoxia: HIF targets':'Other HIF targets','CAF':'CAF content','Ferroptosis defence':'Ferroptosis defense'}
rows=[]
for sec,ks in sections:
    for k in ks:
        x=purity_adjust(S[k],pur)
        for cp in ['HER2+','HR+/HER2-']:
            d,lo,hi,pv=d_ci(x,'TNBC',cp); rows.append((sec,k,cp,d,lo,hi,pv))
R=pd.DataFrame(rows,columns=['sec','mod','comp','d','lo','hi','p']); R['q']=np.nan
for cp in R.comp.unique(): i=R.comp==cp; R.loc[i,'q']=bh(R.loc[i,'p'].values)
R.to_csv(f'{OUT}/Fig2_effect_sizes.csv',index=False)
fig,ax=plt.subplots(figsize=(5.4,7.4)); y=0; yt,yl=[],[]
for sec,ks in sections:
    ax.text(-.02,y+.1,sec.upper(),fontsize=6.5,fontweight='bold',color=INK2,va='bottom',ha='right',transform=ax.get_yaxis_transform()); y-=.55
    for k in ks:
        for j,cp in enumerate(['HER2+','HR+/HER2-']):
            r=R[(R['mod']==k)&(R.comp==cp)].iloc[0]; yy=y+(.17 if j==0 else -.17)
            ax.plot([r.lo,r.hi],[yy,yy],color=COL[cp],lw=1.3,solid_capstyle='round')
            ax.scatter(r.d,yy,s=24,color=COL[cp] if r.q<.05 else 'white',edgecolor=COL[cp],lw=1.2,zorder=3)
        yt.append(y); yl.append(label.get(k,k)); y-=1
    y-=.35
ax.axvline(0,color=INK2,lw=.7)
for v in (-.474,.474): ax.axvline(v,color=GRID,lw=.7,ls=(0,(3,2)))
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False); ax.set_xlim(-1,1)
ax.set_xlabel("Cliff's δ, TNBC vs comparator (purity-adjusted protein)\n← lower in TNBC          higher in TNBC →")
ax.legend(handles=[plt.Line2D([],[],marker='o',color=COL['HER2+'],label='vs HER2+ (n=15)',ms=5,lw=1.3),plt.Line2D([],[],marker='o',color=COL['HR+/HER2-'],label='vs HR+/HER2− (n=74)',ms=5,lw=1.3),
 plt.Line2D([],[],marker='o',color=INK2,mfc='white',label='open = BH q ≥ 0.05',ms=5,lw=0)],loc='upper center',bbox_to_anchor=(.45,-.09),ncol=3,fontsize=6.4,handlelength=1.4)
ax.text(.474,ax.get_ylim()[1],' large',fontsize=6,color=INK2,va='bottom')
save(fig,'Fig2_pathway_overview')

# ---------- Figure 4: export, TGF-b, ER redox ----------
fig=plt.figure(figsize=(7.4,6.8)); gs=fig.add_gridspec(2,6,hspace=.62,wspace=1.3); gbot=gs[1,:].subgridspec(1,2,wspace=.75)
for k,(nm,x,lab) in enumerate([('CREB3L1 (mRNA)',m['CREB3L1'],'log2 FPKM'),('TANGO1 / MIA3',p['MIA3'],'log2 protein ratio'),('cTAGE5 / MIA2',p['MIA2'],'log2 protein ratio')]):
    ax=fig.add_subplot(gs[0,2*k:2*k+2]); boxstrip(ax,x,f'{"ABC"[k]}  {nm}',lab,adjp(x)); ax.set_ylim(top=ax.get_ylim()[1]+.25*(ax.get_ylim()[1]-ax.get_ylim()[0]))
ax=fig.add_subplot(gbot[0]); genes=[('PLOD2','enz'),('P4HA1','enz'),('P4HA2','enz'),('SERPINH1','enz'),('LOX','enz'),('COL1A1','col'),('COL1A2','col'),('COL3A1','col')]
vals=[(gn,kd)+pspear(p[gn],S['TGF-b response'],[pur,S['CAF']])[:2] for gn,kd in genes]
pd.DataFrame(vals,columns=['gene','class','rho','p']).to_csv(f'{OUT}/Fig4D_tgfb_partial.csv',index=False)
for j,(gn,kd,r,pv) in enumerate(vals):
    ax.barh(-j,r,height=.62,color='#a63d12' if kd=='enz' else '#8f8e88'); ax.text(r+(.02 if r>=0 else -.02),-j,f'{r:+.2f}',va='center',ha='left' if r>=0 else 'right',fontsize=6.5)
ax.axvline(0,color=INK2,lw=.7); ax.set_yticks(range(0,-len(vals),-1)); ax.set_yticklabels([v[0] for v in vals]); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(-.5,.95); ax.set_xlabel('Partial Spearman ρ with TGF-β response\n(adjusted for purity + CAF; all 122)'); ax.set_title('D  TGF-β tracks enzymes,\n    not collagen')
ax.legend(handles=[plt.Rectangle((0,0),1,1,color='#a63d12',label='Maturation enzymes'),plt.Rectangle((0,0),1,1,color='#8f8e88',label='Fibrillar collagens')],loc='lower right',fontsize=6.3)
ax=fig.add_subplot(gbot[1]); ev=[]
for gn in ['PRDX4','GPX7','GPX8','ERO1B','QSOX1','ERO1A']:
    r,pv,_=pspear(p[gn],S['Hypoxia consensus'],[pur,prol,S['CAF']]); d=compare(purity_adjust(p[gn],pur),g)[1]; ev.append((gn,r,pv,d))
pd.DataFrame(ev,columns=['gene','rho_hypoxia','p','d_vs_HER2']).to_csv(f'{OUT}/Fig4E_er_redox.csv',index=False)
for j,(gn,r,pv,d) in enumerate(ev):
    ax.barh(-j,r,height=.62,color='#2a78d6' if pv<.05 else '#9ec5f4'); ax.text(max(r,0)+.01,-j,f'{r:+.2f} ({fmt_p(pv)})',va='center',fontsize=6.3)
ax.axvline(0,color=INK2,lw=.7); ax.set_yticks(range(0,-len(ev),-1)); ax.set_yticklabels([e[0] for e in ev]); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(-.15,.95); ax.set_xlabel('Partial Spearman ρ with hypoxia\n(adjusted for purity, proliferation, CAF; all 122)'); ax.set_title('E  ER redox enzymes\n    rise with hypoxia')
save(fig,'Fig4_export_tgfb_redox')

# ---------- Figure S2: ER/PR split ----------
er=c['ER Updated Clinical Status'].str.lower().reindex(p.index); pr=c['PR Clinical Status'].str.lower().reindex(p.index)
g4=g.copy().astype(object); hr=g=='HR+/HER2-'
g4[hr&(er=='positive')&(pr=='positive')]='ER+/PR+'; g4[hr&(er=='positive')&(pr=='negative')]='ER+/PR−'
g4[hr&~((er=='positive')&pr.isin(['positive','negative']))]=np.nan
O4=['TNBC','HER2+','ER+/PR−','ER+/PR+']; C4={'TNBC':'#eb6834','HER2+':'#2a78d6','ER+/PR−':'#7fcdb0','ER+/PR+':'#1baf7a'}
fig,axs=plt.subplots(2,2,figsize=(6.6,6.4)); fig.subplots_adjust(wspace=.35,hspace=.55); axs=axs.ravel()
rowsS2=[]
for ax,(k,lab) in zip(axs,[('Hypoxia consensus','Hypoxia'),('Maturation - deposition','Maturation − deposition'),('UPR PERK/ATF4 targets','PERK/ATF4 targets'),('Proliferation','Proliferation')]):
    xa=purity_adjust(S[k],pur); a=compare(xa,g4,'ER+/PR−','ER+/PR+'); b=compare(xa,g4,'TNBC','ER+/PR−')
    rowsS2.append((k,a[1],a[0],b[1],b[0]))
    boxstrip(ax,S[k],f'{"ABCD"[list(axs).index(ax)]}  {lab}','Module score',f'PR− vs PR+: δ={a[1]:+.2f}, {fmt_p(a[0])}\nTNBC vs PR−: δ={b[1]:+.2f}, {fmt_p(b[0])}',grp=g4,order=O4,cols=C4,labels=['TNBC','HER2+','ER+/\nPR−','ER+/\nPR+'])
    ax.set_ylim(top=ax.get_ylim()[1]+.3*(ax.get_ylim()[1]-ax.get_ylim()[0])); ax.title.set_fontsize(8.2)
    if list(axs).index(ax)%2: ax.set_ylabel('')
pd.DataFrame(rowsS2,columns=['module','d_PRneg_vs_PRpos','p1','d_TNBC_vs_PRneg','p2']).to_csv(f'{OUT}/FigS2_er_pr.csv',index=False)
save(fig,'FigS2_er_pr_split')

# ---------- Figure S3: iron import vs hypoxia ----------
fig,ax=plt.subplots(figsize=(3.6,2.6)); iv=[]
for gn in ['SLC11A2','TFRC','STEAP3','FTH1','FTL']:
    r,pv,_=pspear(p[gn],S['Hypoxia consensus'],[pur,prol]); iv.append((gn,r,pv))
pd.DataFrame(iv,columns=['gene','rho','p']).to_csv(f'{OUT}/FigS3_iron_hypoxia.csv',index=False)
for j,(gn,r,pv) in enumerate(iv):
    ax.barh(-j,r,height=.62,color='#2a78d6' if gn in ('SLC11A2','TFRC','STEAP3') else '#8f8e88'); ax.text(max(r,0)+.01,-j,f'{r:+.2f} ({fmt_p(pv)})',va='center',fontsize=6.3)
ax.axvline(0,color=INK2,lw=.7); ax.set_yticks(range(0,-len(iv),-1)); ax.set_yticklabels(['SLC11A2','TFRC','STEAP3','FTH1','FTL']); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(-.1,.95); ax.set_xlabel('Partial Spearman ρ with hypoxia\n(adjusted for purity + proliferation; all 122)')
ax.legend(handles=[plt.Rectangle((0,0),1,1,color='#2a78d6',label='Iron import'),plt.Rectangle((0,0),1,1,color='#8f8e88',label='Iron storage')],loc='lower right',fontsize=6.3)
save(fig,'FigS3_iron_hypoxia')
print(R.round(3).to_string(index=False)); print(pd.DataFrame(ev)); print(pd.DataFrame(iv)); print(pd.DataFrame(rowsS2))
