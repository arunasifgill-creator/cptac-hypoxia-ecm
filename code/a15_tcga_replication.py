"""Transcript-level replication of the TNBC classifier in TCGA-BRCA (editor decision, item 31).

Inputs (cBioPortal datahub, not stored in this repository because of size, see data/tcga/README.txt):
  rsem.txt                              brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt
  brca_tcga_data_clinical_patient.txt   brca_tcga/data_clinical_patient.txt (ER, PR, HER2 by IHC and FISH)
  brca_tcga_pan_can_atlas_2018_data_clinical_patient.txt   (PAM50 subtype)
  SI_geneset.gmt                        ESTIMATE stromal and immune gene sets

Three analyses:
  1. frozen transfer, models trained on CPTAC mRNA module scores and applied unchanged to TCGA
  2. within-TCGA nested cross-validation with fold-safe preprocessing (same pipeline as a6)
  3. the hypoxia associations at the transcript level in TCGA
"""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from load import load
from core import groups, pspear, cliff, resid
from modules import PROT, GROUP

T = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'data', 'tcga')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'rev3'); os.makedirs(OUT, exist_ok=True)
RENAME = {'COXFA4L2': 'NDUFA4L2', 'CTGF': 'CCN2', 'CYR61': 'CCN1', 'ERO1L': 'ERO1A', 'ERO1LB': 'ERO1B',
          'LEPRE1': 'P3H1', 'KIAA1199': 'CEMIP', 'C10orf10': 'DEPP1', 'FAM213A': 'PRXL2A', 'AIFM2': 'AIFM2',
          'MIA3': 'MIA3', 'CTAGE5': 'MIA2', 'WWTR1': 'WWTR1', 'TAZ': 'TAFAZZIN'}

# ------------------------------------------------------------------ TCGA expression
e = pd.read_csv(f'{T}/rsem.txt', sep='\t')
e = e[e.Hugo_Symbol.notna()].drop(columns='Entrez_Gene_Id')
e['Hugo_Symbol'] = e.Hugo_Symbol.replace(RENAME)
e = e.groupby('Hugo_Symbol').mean()
e = np.log2(e.clip(lower=0) + 1)
e = e.loc[:, [c for c in e.columns if c.endswith('-01')]]
print('TCGA expression', e.shape)

# ------------------------------------------------------------------ ESTIMATE score (stromal + immune ssGSEA)
gs = {l.split('\t')[0]: [x for x in l.strip().split('\t')[2:] if x] for l in open(f'{T}/SI_geneset.gmt')}
ev = e.dropna()
def estimate_es(col, genes):
    r = stats.rankdata(col.values) * 10000 / len(col)
    order = np.argsort(-r); rs = r[order]
    tag = np.isin(col.index.values[order], genes).astype(float)
    w = np.abs(rs) ** 0.25
    Fn = np.cumsum(tag * w / (tag * w).sum()); F0 = np.cumsum((1 - tag) / (len(tag) - tag.sum()))
    return (Fn - F0).sum()
est = pd.DataFrame({k: ev.apply(lambda c: estimate_es(c, [g for g in v if g in ev.index])) for k, v in gs.items()})
est['ESTIMATEScore'] = est.sum(axis=1)
print('ESTIMATE genes used', {k: len([g for g in v if g in ev.index]) for k, v in gs.items()})
purity_proxy = -est['ESTIMATEScore']        # ESTIMATE purity is a decreasing function of this score, ranks are identical

# ------------------------------------------------------------------ TCGA clinical groups
cl = pd.read_csv(f'{T}/brca_tcga_data_clinical_patient.txt', sep='\t', comment='#').set_index('PATIENT_ID')
pc = pd.read_csv(f'{T}/brca_tcga_pan_can_atlas_2018_data_clinical_patient.txt', sep='\t', comment='#').set_index('PATIENT_ID')
pid = pd.Series([s[:12] for s in e.columns], index=e.columns)
er = pid.map(cl['ER_STATUS_BY_IHC']); pr = pid.map(cl['PR_STATUS_BY_IHC'])
ihc = pid.map(cl['IHC_HER2']); fish = pid.map(cl['HER2_FISH_STATUS']); pam = pid.map(pc['SUBTYPE'])
her2_pos = (fish == 'Positive') | ((ihc == 'Positive') & (fish != 'Negative'))
her2_neg = ~her2_pos & ((fish == 'Negative') | (ihc == 'Negative'))
tn = (er == 'Negative') & (pr == 'Negative') & her2_neg
hr = ((er == 'Positive') | (pr == 'Positive')) & her2_neg
gt = pd.Series(np.nan, index=e.columns, dtype=object)
gt[hr] = 'HR+/HER2-'; gt[her2_pos] = 'HER2+'; gt[tn] = 'TNBC'
gt[(pam == 'BRCA_Normal') & (gt != 'TNBC')] = np.nan          # mirrors the CPTAC definition
print('TCGA groups', gt.value_counts(dropna=False).to_dict())
print('TNBC by PAM50', pam[gt == 'TNBC'].value_counts().to_dict())

# ------------------------------------------------------------------ CPTAC mRNA
p, m, ph, c = load(); gc = groups(c).reindex(m.index); purc = c['ESTIMATE TumorPurity'].reindex(m.index)
X_T = e.T; X_C = m

HYP4 = ['Hypoxia consensus', 'Hypoxia: glycolysis', 'Hypoxia: mitochondrial', 'Hypoxia: HIF targets']
SETS = {'Proliferation (baseline)': ['Proliferation'], 'Hypoxia': HYP4, 'ER stress': GROUP['ER stress'],
        'Antioxidant': GROUP['Antioxidant'], 'ECM': GROUP['ECM'], 'Hypoxia + ECM': HYP4 + GROUP['ECM'],
        'All modules + proliferation': ['Proliferation'] + HYP4 + GROUP['ER stress'] + GROUP['Antioxidant'] + GROUP['ECM']}
def genes_for(nm):                       # genes measured in BOTH cohorts, so the two scores are built identically
    return [x for x in PROT[nm] if x in X_C.columns and X_C[x].notna().sum() >= 60 and x in X_T.columns and X_T[x].notna().mean() > 0.9]
cov = pd.DataFrame([dict(module=k, n_defined=len(PROT[k]), n_used=len(genes_for(k)),
                         missing=';'.join(sorted(set(PROT[k]) - set(genes_for(k))))) for k in PROT])
cov.to_csv(f'{OUT}/S31a_tcga_module_coverage.csv', index=False)

def ecdf_pct(train_vals, vals):
    tv = np.asarray(train_vals, float); tv = np.sort(tv[~np.isnan(tv)])
    out = np.searchsorted(tv, np.asarray(vals, float), side='right') / tv.size
    out[np.isnan(np.asarray(vals, float))] = np.nan
    return out

def feats(X, pur, tr, te, names):
    """Module scores as percentiles within the TRAIN rows, residualized on purity fitted in TRAIN rows."""
    Xtr = X.iloc[tr]; ptr = ecdf_pct(pur.iloc[tr], pur.iloc[tr]); pte = ecdf_pct(pur.iloc[tr], pur.iloc[te])
    A, B = [], []
    for nm in names:
        gl = genes_for(nm)
        a = np.nanmean(np.column_stack([ecdf_pct(Xtr[x], Xtr[x]) for x in gl]), axis=1)
        b = np.nanmean(np.column_stack([ecdf_pct(Xtr[x], X.iloc[te][x]) for x in gl]), axis=1)
        ok = ~np.isnan(a) & ~np.isnan(ptr); k = np.polyfit(ptr[ok], a[ok], 1)
        A.append(a - np.polyval(k, ptr)); B.append(b - np.polyval(k, pte))
    return np.column_stack(A), np.column_stack(B)

def lr():
    return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=5000)),
                        {'logisticregression__C': [0.01, 0.03, 0.1, 0.3, 1, 3]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')

def boot(y, pr_, n=2000, seed=0):
    rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2: o.append(roc_auc_score(y[b], pr_[b]))
    return np.percentile(o, [2.5, 97.5])

def metrics(y, pr_):
    yh = (pr_ >= 0.5).astype(int)
    return dict(bal_acc=balanced_accuracy_score(y, yh), sens=recall_score(y, yh, pos_label=1), spec=recall_score(y, yh, pos_label=0))

TASKS = {'TNBC vs all other': lambda g: g.notna(), 'TNBC vs HER2+': lambda g: g.isin(['TNBC', 'HER2+'])}
rows = []
for task, sel in TASKS.items():
    ic = np.where(sel(gc).values)[0]; yc = (gc.iloc[ic] == 'TNBC').astype(int).values
    it = np.where(sel(gt).values)[0]; yt = (gt.iloc[it] == 'TNBC').astype(int).values
    for sname, names in SETS.items():
        # 1. frozen transfer: fit on all assigned CPTAC tumors, apply unchanged to TCGA.
        #    Each cohort's scores are percentiles within that cohort, since FPKM and RSEM are on different scales.
        Ac, _ = feats(X_C, purc, ic, ic, names); At, _ = feats(X_T, purity_proxy, it, it, names)
        mod = lr().fit(Ac, yc); pt = mod.predict_proba(At)[:, 1]
        lo, hi = boot(yt, pt)
        rows.append(dict(task=task, features=sname, analysis='frozen CPTAC model applied to TCGA',
                         n_train=len(yc), n_test=len(yt), n_tnbc_test=int(yt.sum()),
                         auc=roc_auc_score(yt, pt), lo=lo, hi=hi, interval='patient bootstrap', **metrics(yt, pt)))
        # reference: the same transcript-level model cross-validated inside CPTAC
        a = []
        for r in range(20):
            oof = np.zeros(len(yc))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(yc, yc):
                A, B = feats(X_C, purc, ic[tr], ic[te], names); oof[te] = lr().fit(A, yc[tr]).predict_proba(B)[:, 1]
            a.append(roc_auc_score(yc, oof))
        rows.append(dict(task=task, features=sname, analysis='CPTAC mRNA, nested CV (reference)', n_train=len(yc),
                         n_test=len(yc), n_tnbc_test=int(yc.sum()), auc=np.mean(a), lo=np.percentile(a, 2.5),
                         hi=np.percentile(a, 97.5), interval='fold assignment'))
        # 2. within-TCGA nested CV
        a = []; O = np.zeros((5, len(yt)))
        for r in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(yt, yt):
                A, B = feats(X_T, purity_proxy, it[tr], it[te], names); O[r, te] = lr().fit(A, yt[tr]).predict_proba(B)[:, 1]
            a.append(roc_auc_score(yt, O[r]))
        lo, hi = boot(yt, O.mean(0))
        rows.append(dict(task=task, features=sname, analysis='TCGA mRNA, nested CV', n_train=len(yt), n_test=len(yt),
                         n_tnbc_test=int(yt.sum()), auc=np.mean(a), lo=lo, hi=hi, interval='patient bootstrap', **metrics(yt, O.mean(0))))
        print(task, sname, [round(x['auc'], 3) for x in rows[-3:]], flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/S31_tcga_classifier.csv', index=False)

# ------------------------------------------------------------------ 3. associations at the transcript level in TCGA
def sc(X, nm):
    return X[genes_for(nm)].rank(pct=True).mean(axis=1)
S = {k: sc(X_T, k) for k in PROT}; S['Maturation - deposition'] = S['Maturation'] - S['Fibrillar collagen']
COVT = [purity_proxy, S['Proliferation'], S['CAF']]
ar = []
for on, ov in [('Maturation', S['Maturation']), ('Maturation - deposition', S['Maturation - deposition']),
               ('Fibrillar collagen', S['Fibrillar collagen']), ('PLOD2', X_T['PLOD2']), ('P4HA1', X_T['P4HA1']),
               ('SERPINH1', X_T['SERPINH1']), ('UPR PERK/ATF4 targets', S['UPR PERK/ATF4 targets']),
               ('Collagen export', S['Collagen export'])]:
    r, pv, n = pspear(S['Hypoxia consensus'], ov, COVT); ar.append(dict(kind='partial Spearman with hypoxia', item=on, value=r, p=pv, n=n))
for k in ['Hypoxia consensus', 'Maturation - deposition', 'Maturation', 'Fibrillar collagen']:
    x = resid(S[k], [purity_proxy])
    for b in ['HER2+', 'HR+/HER2-']:
        xa, xb = x[gt == 'TNBC'].dropna(), x[gt == b].dropna()
        ar.append(dict(kind=f'Cliff delta TNBC vs {b}', item=k, value=cliff(xa, xb), p=stats.mannwhitneyu(xa, xb).pvalue, n=len(xa) + len(xb)))
pd.DataFrame(ar).to_csv(f'{OUT}/S31b_tcga_associations.csv', index=False)
pd.DataFrame({'group': gt, 'PAM50': pam, 'ESTIMATEScore': est['ESTIMATEScore']}).to_csv(f'{OUT}/S31c_tcga_samples.csv')
pd.set_option('display.width', 220)
print(R.round(3).to_string(index=False)); print(pd.DataFrame(ar).round(4).to_string(index=False)); print(cov.to_string(index=False))
