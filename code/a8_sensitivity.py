"""Sensitivity analyses requested by reviewers R1-M5, R1-M6, R1-M11, R1-M13, R2-2, R2-9.

Tests whether the hypoxia associations survive (a) adjustment for molecular subtype,
(b) an alternative stromal estimate (xCell) in place of ESTIMATE purity,
(c) additional adjustment for TGF-beta response,
(d) substitution of the non-glycolytic hypoxia score,
and whether they hold within the largest subtype stratum. Also runs leave-one-protein-out
for the maturation-minus-deposition composite.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from load import load
from core import groups, pspear, resid, score, rankpct, bh
from modules import PROT, all_scores

p, m, ph, c = load()
g = groups(c).reindex(p.index)
S = all_scores(p)
pur   = c['ESTIMATE TumorPurity'].reindex(p.index)
estr  = c['ESTIMATE Stromal Score'].reindex(p.index)
xstr  = c['xCell Stromal Score'].reindex(p.index)
prol, caf, tgf = S['Proliferation'], S['CAF'], S['TGF-b response']
is_tnbc = (g == 'TNBC').astype(float); is_her2 = (g == 'HER2+').astype(float)

HYP  = S['Hypoxia consensus']
HYPN = S['Hypoxia non-glycolytic']
OUT = {'Maturation': S['Maturation'],
       'Maturation - deposition': S['Maturation - deposition'],
       'Fibrillar collagen': S['Fibrillar collagen'],
       'Collagen export': S['Collagen export'],
       'PLOD2': p['PLOD2'], 'P4HA1': p['P4HA1'], 'P4HA2': p['P4HA2'], 'SERPINH1': p['SERPINH1']}

SPECS = [
 ('1. purity + proliferation + CAF (as published)', HYP,  [pur, prol, caf]),
 ('2. + molecular subtype',                         HYP,  [pur, prol, caf, is_tnbc, is_her2]),
 ('3. + TGF-beta response',                          HYP,  [pur, prol, caf, tgf]),
 ('4. + subtype + TGF-beta',                         HYP,  [pur, prol, caf, is_tnbc, is_her2, tgf]),
 ('5. xCell stromal instead of ESTIMATE purity',     HYP,  [xstr, prol, caf]),
 ('6. ESTIMATE stromal instead of purity',           HYP,  [estr, prol, caf]),
 ('7. non-glycolytic hypoxia score',                 HYPN, [pur, prol, caf]),
 ('8. non-glycolytic + subtype',                     HYPN, [pur, prol, caf, is_tnbc, is_her2]),
]

rows = []
ok = g.notna()
for label, expo, cov in SPECS:
    for oname, ov in OUT.items():
        r, pv, n = pspear(expo[ok], ov[ok], [x[ok] for x in cov])
        rows.append(dict(spec=label, outcome=oname, rho=r, p=pv, n=n))
R = pd.DataFrame(rows); R['q'] = bh(R.p.values)
R.to_csv('../out/sens_covariates.csv', index=False)

print('=' * 104)
print('HYPOXIA ASSOCIATIONS UNDER ALTERNATIVE COVARIATE SPECIFICATIONS (partial Spearman rho)')
print('=' * 104)
piv = R.pivot(index='spec', columns='outcome', values='rho')[list(OUT)]
print(piv.round(2).to_string())

# ---- stratified within the largest stratum ----
print()
print('=' * 104)
print('WITHIN-STRATUM ESTIMATES (purity + proliferation + CAF)')
print('=' * 104)
srows = []
for grp_name, mask in [('HR+/HER2- only (n=74)', g == 'HR+/HER2-'),
                       ('TNBC only (n=28)',      g == 'TNBC'),
                       ('non-TNBC (n=89)',       g.isin(['HER2+', 'HR+/HER2-']))]:
    for oname, ov in OUT.items():
        r, pv, n = pspear(HYP[mask], ov[mask], [pur[mask], prol[mask], caf[mask]])
        srows.append(dict(stratum=grp_name, outcome=oname, rho=r, p=pv, n=n))
Sdf = pd.DataFrame(srows); Sdf['q'] = bh(Sdf.p.values)
Sdf.to_csv('../out/sens_stratified.csv', index=False)
print(Sdf.pivot(index='stratum', columns='outcome', values='rho')[list(OUT)].round(2).to_string())

# ---- leave-one-protein-out for the composite ----
print()
print('=' * 104)
print('LEAVE-ONE-PROTEIN-OUT, maturation minus deposition vs hypoxia (purity + proliferation + CAF)')
print('=' * 104)
mat, dep = PROT['Maturation'], PROT['Fibrillar collagen']
full = [x for x in mat + dep if x in p.columns and p[x].notna().sum() >= 60]
base_r, _, _ = pspear(HYP[ok], S['Maturation - deposition'][ok], [pur[ok], prol[ok], caf[ok]])
lrows = []
for drop in full:
    mm = [x for x in mat if x != drop]; dd = [x for x in dep if x != drop]
    comp = score(p, mm) - score(p, dd)
    r, pv, n = pspear(HYP[ok], comp[ok], [pur[ok], prol[ok], caf[ok]])
    lrows.append(dict(dropped=drop, rho=r, delta=r - base_r, p=pv))
L = pd.DataFrame(lrows).sort_values('rho')
L.to_csv('../out/sens_leave_one_out.csv', index=False)
print(f'full composite rho = {base_r:.3f}')
print(f'leave-one-out range: {L.rho.min():.3f} to {L.rho.max():.3f} '
      f'(max |change| {L.delta.abs().max():.3f}, driven by dropping {L.loc[L.delta.abs().idxmax(),"dropped"]})')
print(f'all {len(L)} leave-one-out estimates p < 0.05: {(L.p < 0.05).all()}')
print()
print(L.head(4).round(3).to_string(index=False)); print('  ...'); print(L.tail(3).round(3).to_string(index=False))
