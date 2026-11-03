"""Tumour-level permutation for the module-preservation analysis (R1-M16, R2-16)
and a sensitivity analysis on the HER2+ group definition (R2-14)."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
from load import load
from core import groups, purity_adjust, score, compare, bh, pspear, resid
from modules import all_scores, PROT

p,m,ph,c = load(); g = groups(c).reindex(p.index)
pur = c['ESTIMATE TumorPurity'].reindex(p.index); S = all_scores(p)

# ---------------- 1. module preservation, tumour-level permutation ----------------
# Exactly the 19 collagen-module genes drawn in Figure 5B (fig_cluster.py MOD), so that the
# permutation test corrects the statistic the submitted manuscript reported.
core = (['COL1A1','COL1A2','COL3A1','COL5A1']
        + ['PLOD1','PLOD2','PLOD3','P4HA1','P4HA2','SERPINH1','P3H1','PPIB','CRTAP','FKBP10','COLGALT1']
        + ['LOX','LOXL2','BMP1','PCOLCE'])
core = [x for x in core if x in p.columns and p[x].notna().sum() >= 60]
def pairwise(mask):
    R = pd.DataFrame({x: purity_adjust(p[x][mask], pur[mask]) for x in core}).corr('spearman')
    iu = np.triu_indices(len(core), 1); return R.values[iu]
obs_r, _ = stats.spearmanr(pairwise(g=='TNBC'), pairwise(g=='HR+/HER2-'))
npairs = len(core)*(len(core)-1)//2
lab = g[g.isin(['TNBC','HR+/HER2-'])].copy()
rng = np.random.default_rng(3); null=[]
for _ in range(2000):
    sh = pd.Series(rng.permutation(lab.values), index=lab.index)
    a = pairwise(sh.reindex(g.index) == 'TNBC'); b = pairwise(sh.reindex(g.index) == 'HR+/HER2-')
    r,_ = stats.spearmanr(a,b)
    if np.isfinite(r): null.append(r)
null = np.array(null)
p_perm = (np.sum(null >= obs_r) + 1) / (len(null) + 1)
# bootstrap CI on the observed statistic, resampling tumours
bt=[]
tn = np.where((g=='TNBC').values)[0]; hr = np.where((g=='HR+/HER2-').values)[0]
for _ in range(1000):
    i1 = rng.choice(tn, len(tn), True); i2 = rng.choice(hr, len(hr), True)
    m1 = pd.Series(False, index=g.index); m1.iloc[i1]=True
    m2 = pd.Series(False, index=g.index); m2.iloc[i2]=True
    try:
        r,_ = stats.spearmanr(pairwise(m1), pairwise(m2))
        if np.isfinite(r): bt.append(r)
    except Exception: pass
print('='*92); print('MODULE PRESERVATION, TUMOUR-LEVEL PERMUTATION'); print('='*92)
print(f'{len(core)} proteins, {npairs} gene pairs')
print(f'observed between-subtype correlation of pairwise rho: {obs_r:+.3f}')
print(f'tumour-label permutation null (n={len(null)}): mean {null.mean():+.3f}, '
      f'2.5-97.5th {np.percentile(null,2.5):+.3f} to {np.percentile(null,97.5):+.3f}')
print(f'permutation p = {p_perm:.4f}')
print(f'tumour bootstrap 95% CI for the observed statistic: {np.percentile(bt,2.5):+.3f} to {np.percentile(bt,97.5):+.3f}')
pd.DataFrame([dict(n_proteins=len(core), n_pairs=npairs, observed=obs_r,
                   perm_null_mean=null.mean(), perm_lo=np.percentile(null,2.5), perm_hi=np.percentile(null,97.5),
                   perm_p=p_perm, boot_lo=np.percentile(bt,2.5), boot_hi=np.percentile(bt,97.5))]
            ).to_csv('../out/preservation_permutation.csv', index=False)

# ---------------- 2. HER2 group definition sensitivity ----------------
def _pos(s): return s.astype(str).str.strip().str.lower().isin(['positive','1','1.0'])
tn = _pos(c['TNBC Updated Clinical Status']).reindex(p.index)
clin = _pos(c['ERBB2 Updated Clinical Status']).reindex(p.index)
pgx  = _pos(c['ERBB2 Proteogenomic Status']).reindex(p.index)
amp  = _pos(c['ERBB2 Gene Amplified']).reindex(p.index)
nl   = c['PAM50'].reindex(p.index).eq('Normal-like')
print(); print('='*92); print('HER2+ GROUP DEFINITION'); print('='*92)
print(f'  clinical status positive:      {int((clin & ~tn).sum())}')
print(f'  proteogenomic positive:        {int((pgx & ~tn).sum())}')
print(f'  gene amplified:                {int((amp & ~tn).sum())}')
print(f'  union (as published):          {int(((clin|pgx|amp) & ~tn).sum())}')
print(f'  all three concordant:          {int((clin & pgx & amp & ~tn).sum())}')
DEFS = {'union (as published)': (clin|pgx|amp) & ~tn,
        'clinical status only':  clin & ~tn,
        'all three concordant':  clin & pgx & amp & ~tn}
rows=[]
for lab_, her in DEFS.items():
    gg = pd.Series('HR+/HER2-', index=p.index)
    gg[her]='HER2+'; gg[tn]='TNBC'; gg[nl & ~tn]=np.nan
    n_her = int((gg=='HER2+').sum())
    for nm, sc in [('Hypoxia consensus',S['Hypoxia consensus']),
                   ('Maturation - deposition',S['Maturation - deposition']),
                   ('UPR PERK/ATF4 targets',S['UPR PERK/ATF4 targets'])]:
        adj = purity_adjust(sc, pur)
        pv,d = compare(adj, gg, 'TNBC','HER2+')
        rows.append(dict(definition=lab_, n_HER2=n_her, module=nm, d_TNBC_vs_HER2=d, p=pv))
H = pd.DataFrame(rows); H['q']=bh(H.p.values); H.to_csv('../out/her2_definition.csv', index=False)
print()
print(H.pivot_table(index=['definition','n_HER2'], columns='module', values='d_TNBC_vs_HER2').round(2).to_string())
print()
print('Normal-like tumours:', int(nl.sum()), '| excluded from all three definitions')
print('\nsaved preservation_permutation.csv, her2_definition.csv')
