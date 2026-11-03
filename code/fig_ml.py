"""Figure 6. Interpretable classifiers, fold-safe pipeline.
All preprocessing fitted inside the training folds (reviewer comments R1-M14, R2-5).
Both fold-assignment and patient-level bootstrap intervals shown (R1-M14, R2-7)."""
import os, numpy as np, pandas as pd, matplotlib, warnings; warnings.filterwarnings('ignore')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='../figures'; INK,INK2,GRID='#1f1f1e','#5f5e5a','#e6e5e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':INK2,'axes.labelcolor':INK,
 'xtick.color':INK2,'ytick.color':INK2,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titlesize':9,'axes.titleweight':'bold','axes.titlelocation':'left','legend.frameon':False,
 'pdf.fonttype':42,'savefig.dpi':600})

T=pd.read_csv('../out/ml_auc_foldsafe.csv'); C=pd.read_csv('../out/ml_coefs_foldsafe.csv')
order=['Proliferation (baseline)','Antioxidant','ER stress','Hypoxia','ECM','Hypoxia + ECM',
       'Hypoxia + ER stress + ECM','All mechanistic modules','All modules + proliferation']
gl=[f for f in T.features.unique() if f.startswith('All proteins')][0]; order.append(gl)

fig=plt.figure(figsize=(7.4,6.3)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],hspace=.44,wspace=.78)
for k,(task,col) in enumerate([('TNBC vs all other','#eb6834'),('TNBC vs HER2+','#2a78d6')]):
    ax=fig.add_subplot(gs[0,k])
    D=T[(T.task==task)&(T.model=='Logistic (L2)')].set_index('features').loc[order]
    yy=np.arange(len(order))[::-1]
    cc=[INK2 if f.startswith(('Proliferation','All proteins')) else col for f in order]
    ax.hlines(yy,D.boot_lo,D.boot_hi,color=cc,lw=3.2,alpha=.22)          # patient-level bootstrap
    ax.hlines(yy,D.rep_lo,D.rep_hi,color=cc,lw=1.5)                      # across CV repeats
    ax.scatter(D.auc,yy,s=24,color=cc,zorder=3)
    for y_,a in zip(yy,D.auc): ax.text(1.015,y_,f'{a:.2f}',va='center',fontsize=6.5,color=INK)
    ax.axvline(.5,color=GRID,lw=.8,ls=(0,(3,2)))
    ax.set_yticks(yy)
    ax.set_yticklabels([f.replace(' (gene-level, ','\n(gene-level, ') for f in order] if k==0 else ['']*len(order),fontsize=6.6)
    ax.tick_params(axis='y',length=0)
    ax.set_xlim(.40,1.0); ax.set_xticks([.4,.5,.6,.7,.8,.9,1.0])
    ax.set_xlabel('Cross-validated AUC')
    n={'TNBC vs all other':'n = 28 vs 89','TNBC vs HER2+':'n = 28 vs 15'}[task]
    ax.set_title(f'{"AB"[k]}  {task} ({n})'); ax.xaxis.grid(True,color=GRID,lw=.5); ax.set_axisbelow(True)

for k,task in enumerate(['TNBC vs all other','TNBC vs HER2+']):
    ax=fig.add_subplot(gs[1,k]); D=C[C.task==task].sort_values('coef')
    grp=lambda s: ('#a63d12' if s in ('Fibrillar collagen','Maturation','Collagen export','ER-phagy / ERAD','MMP turnover','Mechanotransduction')
                   else '#e87ba4' if s.startswith('Hypoxia')
                   else '#2a78d6' if s.startswith(('UPR','ER redox'))
                   else '#1baf7a' if s in ('NRF2 targets','NADPH supply','Peroxide detox','Ferroptosis defence')
                   else '#8f8e88')
    ax.barh(range(len(D)),D.coef,xerr=D.sd,color=[grp(s) for s in D.module],height=.65,
            error_kw=dict(lw=.5,ecolor=INK2))
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels([s.replace('Ferroptosis defence','Ferroptosis defense').replace('UPR canonical chaperones','Chaperones').replace('UPR ','').replace(' targets','') for s in D.module],fontsize=6.4)
    ax.tick_params(axis='y',length=0)
    ax.axvline(0,color=INK2,lw=.6); ax.set_xlabel('Standardised L2 coefficient\n(mean ± SD over 100 outer folds)')
    ax.set_title(f'{"CD"[k]}  Feature weights, {task}')

h=[plt.Rectangle((0,0),1,1,color=c_,label=l) for c_,l in
   [('#e87ba4','Hypoxia'),('#2a78d6','ER stress'),('#a63d12','ECM'),('#1baf7a','Antioxidant'),('#8f8e88','Proliferation')]]
h+= [plt.Line2D([],[],color=INK2,lw=1.5,label='across CV repeats'),
     plt.Line2D([],[],color=INK2,lw=3.2,alpha=.22,label='patient-level bootstrap')]
fig.legend(handles=h,loc='lower center',ncol=4,fontsize=6.6,bbox_to_anchor=(.5,-.045))
for e in ('png','pdf'): fig.savefig(f'{OUT}/FigML_classifiers.{e}',bbox_inches='tight',facecolor='white')
print('saved FigML_classifiers')
