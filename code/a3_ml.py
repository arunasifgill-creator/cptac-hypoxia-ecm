"""Interpretable ML: which biological axes discriminate TNBC? Nested, repeated, stratified CV.
Features are purity-adjusted protein module scores (label-free transformation done once on the full cohort).
Outer: 5-fold x 20 repeats; inner: 5-fold grid search for regularisation."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from load import load; from core import groups
from modules import *
p,m,ph,c=load(); g=groups(c).reindex(p.index); pur=c['ESTIMATE TumorPurity'].reindex(p.index)
S=all_scores(p)
F=pd.DataFrame({k:purity_adjust(v,pur) for k,v in S.items()})
SETS={
 'Proliferation (baseline)':['Proliferation'],
 'Hypoxia':['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets'],
 'ER stress':GROUP['ER stress'],
 'Antioxidant':GROUP['Antioxidant'],
 'ECM':GROUP['ECM'],
 'Hypoxia + ECM':['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ECM'],
 'Hypoxia + ER stress + ECM':['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ER stress']+GROUP['ECM'],
 'All mechanistic modules':['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ER stress']+GROUP['Antioxidant']+GROUP['ECM'],
 'All modules + proliferation':['Proliferation','Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ER stress']+GROUP['Antioxidant']+GROUP['ECM'],
}
genes_ok=[x for x in p.columns if p[x].notna().mean()>=0.8]
G=pd.DataFrame({x:purity_adjust(p[x],pur) for x in genes_ok})

def models():
    lr=GridSearchCV(make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=5000)),
                    {'logisticregression__C':[0.01,0.03,0.1,0.3,1,3]},cv=StratifiedKFold(5,shuffle=True,random_state=0),scoring='roc_auc')
    rf=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=300,min_samples_leaf=3,random_state=0,n_jobs=-1))
    return {'Logistic (L2)':lr,'Random forest':rf}

def cv_auc(X,y,model,R=20,seed0=0):
    aucs=[]; coefs=[]
    for r in range(R):
        oof=np.zeros(len(y))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed0+r).split(X,y):
            model.fit(X[tr],y[tr]); oof[te]=model.predict_proba(X[te])[:,1]
            if hasattr(model,'best_estimator_'): coefs.append(model.best_estimator_[-1].coef_[0])
        aucs.append(roc_auc_score(y,oof))
    return np.array(aucs), (np.array(coefs) if coefs else None)

def task(name,mask,pos='TNBC'):
    y=(g[mask]==pos).astype(int).values; rows=[]; coefrows=[]
    for sname,cols in SETS.items():
        X=F.loc[mask,cols].values
        for mname,mod in models().items():
            a,cf=cv_auc(X,y,mod)
            rows.append(dict(task=name,features=sname,n_features=len(cols),model=mname,auc=a.mean(),lo=np.percentile(a,2.5),hi=np.percentile(a,97.5),aucs=a))
            if cf is not None and sname=='All modules + proliferation':
                for j,cn in enumerate(cols): coefrows.append(dict(task=name,module=cn,coef=cf[:,j].mean(),sd=cf[:,j].std(),pos_frac=(cf[:,j]>0).mean()))
    # gene-level benchmark
    X=G.loc[mask].values
    for mname,mod in models().items():
        a,_=cv_auc(X,y,mod,R=10)
        rows.append(dict(task=name,features=f'All proteins (gene-level, {G.shape[1]})',n_features=G.shape[1],model=mname,auc=a.mean(),lo=np.percentile(a,2.5),hi=np.percentile(a,97.5),aucs=a))
    return pd.DataFrame(rows),pd.DataFrame(coefrows)

lab=g.notna()
T1,C1=task('TNBC vs all other',lab)
T2,C2=task('TNBC vs HER2+',g.isin(['TNBC','HER2+']))
T=pd.concat([T1,T2]); C=pd.concat([C1,C2])
T.drop(columns='aucs').to_csv('../out/ml_auc.csv',index=False); C.to_csv('../out/ml_coefs.csv',index=False)
T.to_pickle('../out/ml_auc.pkl')
print(T.drop(columns='aucs').round(3).to_string(index=False))
print(C.round(3).sort_values(['task','coef']).to_string(index=False))
