"""Fold-safe standardized logistic coefficients for Figure 6C,D."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from a6_ml_foldsafe import build_features, models, g, SETS
COLS = SETS['All modules + proliferation']
rows=[]
for task, mask in [('TNBC vs all other', g.notna()), ('TNBC vs HER2+', g.isin(['TNBC','HER2+']))]:
    idx = np.where(mask.values)[0]; y = (g[mask]=='TNBC').astype(int).values
    coefs=[]
    for r in range(20):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y):
            Xtr, Xte = build_features(idx[tr], idx[te], module_names=COLS)
            m = models()['Logistic (L2)']; m.fit(Xtr, y[tr])
            coefs.append(m.best_estimator_[-1].coef_[0])
    A = np.array(coefs)
    for j, cn in enumerate(COLS):
        rows.append(dict(task=task, module=cn, coef=A[:,j].mean(), sd=A[:,j].std(),
                         pos_frac=(A[:,j]>0).mean()))
    print(f'{task}: {A.shape[0]} folds', flush=True)
pd.DataFrame(rows).to_csv('../out/ml_coefs_foldsafe.csv', index=False)
print('saved ../out/ml_coefs_foldsafe.csv')
