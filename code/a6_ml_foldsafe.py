"""Fold-safe ML: every transformation (rank-percentile, purity residualisation, standardisation)
is fitted on the training fold only and applied to the held-out fold. Addresses reviewer comments
R1-M14 and R2-5. Also reports balanced accuracy, sensitivity, specificity, and patient-level
bootstrap confidence intervals for AUC."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from load import load
from core import groups
from modules import PROT, GROUP

p, m, ph, c = load()
g = groups(c).reindex(p.index)
pur = c['ESTIMATE TumorPurity'].reindex(p.index)

SETS = {
 'Proliferation (baseline)': ['Proliferation'],
 'Hypoxia': ['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets'],
 'ER stress': GROUP['ER stress'],
 'Antioxidant': GROUP['Antioxidant'],
 'ECM': GROUP['ECM'],
 'Hypoxia + ECM': ['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ECM'],
 'Hypoxia + ER stress + ECM': ['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ER stress']+GROUP['ECM'],
 'All mechanistic modules': ['Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ER stress']+GROUP['Antioxidant']+GROUP['ECM'],
 'All modules + proliferation': ['Proliferation','Hypoxia consensus','Hypoxia: glycolysis','Hypoxia: mitochondrial','Hypoxia: HIF targets']+GROUP['ER stress']+GROUP['Antioxidant']+GROUP['ECM'],
}
GENES_OK = [x for x in p.columns if p[x].notna().mean() >= 0.8]

def ecdf_pct(train_vals, vals):
    """Percentile of `vals` within the TRAIN distribution (fold-safe rank transform)."""
    tv = np.asarray(train_vals, float); tv = tv[~np.isnan(tv)]
    if tv.size == 0: return np.full(len(vals), np.nan)
    tv = np.sort(tv)
    out = np.searchsorted(tv, np.asarray(vals, float), side='right') / tv.size
    out[np.isnan(np.asarray(vals, float))] = np.nan
    return out

def build_features(tr_idx, te_idx, module_names=None, gene_list=None):
    """Return (Xtr, Xte). All transforms fitted on tr_idx only."""
    ptr = p.iloc[tr_idx]
    pur_tr_pct = ecdf_pct(pur.iloc[tr_idx], pur.iloc[tr_idx])
    pur_te_pct = ecdf_pct(pur.iloc[tr_idx], pur.iloc[te_idx])
    cols_tr, cols_te = [], []
    if gene_list is not None:
        feats = [(gn, [gn]) for gn in gene_list]
    else:
        feats = [(nm, PROT[nm]) for nm in module_names]
    for nm, genes in feats:
        gs = [x for x in genes if x in p.columns and ptr[x].notna().sum() >= 20]
        if not gs: 
            cols_tr.append(np.zeros(len(tr_idx))); cols_te.append(np.zeros(len(te_idx))); continue
        tr_pct = np.nanmean(np.column_stack([ecdf_pct(ptr[x], p.iloc[tr_idx][x]) for x in gs]), axis=1)
        te_pct = np.nanmean(np.column_stack([ecdf_pct(ptr[x], p.iloc[te_idx][x]) for x in gs]), axis=1)
        ok = ~np.isnan(tr_pct) & ~np.isnan(pur_tr_pct)
        if ok.sum() > 3:
            b = np.polyfit(pur_tr_pct[ok], tr_pct[ok], 1)
            tr_r = tr_pct - np.polyval(b, pur_tr_pct)
            te_r = te_pct - np.polyval(b, pur_te_pct)
        else:
            tr_r, te_r = tr_pct, te_pct
        cols_tr.append(tr_r); cols_te.append(te_r)
    return np.column_stack(cols_tr), np.column_stack(cols_te)

def models():
    lr = GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                    LogisticRegression(max_iter=5000)),
                      {'logisticregression__C': [0.01,0.03,0.1,0.3,1,3]},
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')
    rf = make_pipeline(SimpleImputer(strategy='median'),
                       RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))
    return {'Logistic (L2)': lr, 'Random forest': rf}

def boot_ci(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed); out=[]
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        out.append(roc_auc_score(y[b], pred[b]))
    return (np.percentile(out,2.5), np.percentile(out,97.5)) if out else (np.nan,np.nan)

def run(name, mask, R=20, module_names=None, gene_list=None, label=''):
    idx = np.where(mask.values)[0]
    y = (g[mask] == 'TNBC').astype(int).values
    rows=[]
    for mname, mod in models().items():
        aucs=[]; oof_all=np.zeros((R, len(y)))
        for r in range(R):
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y):
                Xtr, Xte = build_features(idx[tr], idx[te], module_names, gene_list)
                mod.fit(Xtr, y[tr]); oof[te] = mod.predict_proba(Xte)[:,1]
            aucs.append(roc_auc_score(y, oof)); oof_all[r]=oof
        aucs=np.array(aucs); mean_oof = oof_all.mean(axis=0)
        lo, hi = boot_ci(y, mean_oof)
        yhat = (mean_oof >= 0.5).astype(int)
        rows.append(dict(task=name, features=label, n_features=(len(gene_list) if gene_list else len(module_names)),
                         model=mname, auc=aucs.mean(), rep_lo=np.percentile(aucs,2.5), rep_hi=np.percentile(aucs,97.5),
                         boot_lo=lo, boot_hi=hi,
                         bal_acc=balanced_accuracy_score(y, yhat),
                         sens=recall_score(y, yhat, pos_label=1), spec=recall_score(y, yhat, pos_label=0)))
        print(f'  {label[:34]:36s} {mname:15s} AUC {aucs.mean():.3f} '
              f'(rep {np.percentile(aucs,2.5):.3f}-{np.percentile(aucs,97.5):.3f}; '
              f'boot {lo:.3f}-{hi:.3f})  balAcc {rows[-1]["bal_acc"]:.3f} sens {rows[-1]["sens"]:.3f} spec {rows[-1]["spec"]:.3f}', flush=True)
    return rows

if __name__ == '__main__':
    allrows=[]
    for tname, mask in [('TNBC vs all other', g.notna()), ('TNBC vs HER2+', g.isin(['TNBC','HER2+']))]:
        print('='*70); print(tname, flush=True)
        for sname, cols in SETS.items():
            allrows += run(tname, mask, module_names=cols, label=sname)
        allrows += run(tname, mask, R=10, gene_list=GENES_OK,
                       label=f'All proteins (gene-level, {len(GENES_OK)})')
    T=pd.DataFrame(allrows)
    T.to_csv('../out/ml_auc_foldsafe.csv', index=False)
    print('\nsaved ../out/ml_auc_foldsafe.csv')
