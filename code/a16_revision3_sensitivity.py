"""Third-round revision: covariate sensitivity, leave-one-out, mediator collinearity, tumor stage and the
module-preservation percentile, all continuous analyses on the 122 tumors of Methods 2.2."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from load import load
from core import groups, pspear, resid, score, bh, purity_adjust
from modules import PROT, all_scores
OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'rev3'); os.makedirs(OUT, exist_ok=True)
p, m, ph, c = load(); g = groups(c).reindex(p.index); S = all_scores(p)
pur = c['ESTIMATE TumorPurity'].reindex(p.index); estr = c['ESTIMATE Stromal Score'].reindex(p.index)
xstr = c['xCell Stromal Score'].reindex(p.index)
prol, caf, tgf = S['Proliferation'], S['CAF'], S['TGF-b response']
SUB = [(g == 'TNBC').astype(float), (g == 'HER2+').astype(float), g.isna().astype(float)]
HYP, HYPN = S['Hypoxia consensus'], S['Hypoxia non-glycolytic']
O = {'Maturation': S['Maturation'], 'Maturation - deposition': S['Maturation - deposition'],
     'Fibrillar collagen': S['Fibrillar collagen'], 'Collagen export': S['Collagen export'],
     'PLOD2': p['PLOD2'], 'P4HA1': p['P4HA1'], 'P4HA2': p['P4HA2'], 'SERPINH1': p['SERPINH1']}
SPECS = [('1. purity + proliferation + CAF (as published)', HYP, [pur, prol, caf]),
         ('2. + molecular subtype', HYP, [pur, prol, caf] + SUB),
         ('3. + TGF-beta response', HYP, [pur, prol, caf, tgf]),
         ('4. + subtype + TGF-beta', HYP, [pur, prol, caf, tgf] + SUB),
         ('5. xCell stromal instead of ESTIMATE purity', HYP, [xstr, prol, caf]),
         ('6. ESTIMATE stromal instead of purity', HYP, [estr, prol, caf]),
         ('7. non-glycolytic hypoxia score', HYPN, [pur, prol, caf]),
         ('8. non-glycolytic + subtype', HYPN, [pur, prol, caf] + SUB)]
R = pd.DataFrame([dict(spec=l, outcome=on, **dict(zip(['rho', 'p', 'n'], pspear(e, ov, cv))))
                  for l, e, cv in SPECS for on, ov in O.items()])
R['q'] = bh(R.p.values); R.to_csv(f'{OUT}/S6_sens_covariates_n122.csv', index=False)
print(R.pivot(index='spec', columns='outcome', values='rho')[list(O)].round(2).to_string())
print(R.pivot(index='spec', columns='outcome', values='q')[list(O)].round(3).to_string())

rows = []
for nm, mask in [('HR+/HER2- only (n=74)', g == 'HR+/HER2-'), ('TNBC only (n=28)', g == 'TNBC'),
                 ('non-TNBC (n=89)', g.isin(['HER2+', 'HR+/HER2-']))]:
    for on, ov in O.items():
        r, pv, n = pspear(HYP[mask], ov[mask], [pur[mask], prol[mask], caf[mask]]); rows.append(dict(stratum=nm, outcome=on, rho=r, p=pv, n=n))
Sd = pd.DataFrame(rows); Sd['q'] = bh(Sd.p.values); Sd.to_csv(f'{OUT}/S7_sens_stratified.csv', index=False)
print(Sd.round(3).to_string(index=False))

mat, dep = PROT['Maturation'], PROT['Fibrillar collagen']
base = pspear(HYP, S['Maturation - deposition'], [pur, prol, caf])[0]
L = pd.DataFrame([dict(dropped=d, **dict(zip(['rho', 'p'], pspear(HYP, score(p, [x for x in mat if x != d]) - score(p, [x for x in dep if x != d]), [pur, prol, caf])[:2])))
                  for d in mat + dep]); L['delta'] = L.rho - base
L.sort_values('rho').to_csv(f'{OUT}/S8_leave_one_out_n122.csv', index=False)
print(f'LOO base {base:.3f} range {L.rho.min():.3f} to {L.rho.max():.3f} max|d| {L.delta.abs().max():.3f} all p<.05 {(L.p<.05).all()} n={len(L)}')

MEDS = ['UPR canonical chaperones', 'UPR PERK/ATF4 targets', 'UPR ATF6 targets', 'ER redox']
Mz = pd.concat([resid(S[k], [pur, prol, caf]).rename(k) for k in MEDS], axis=1); Mz = (Mz - Mz.mean()) / Mz.std()
Mz.corr('spearman').to_csv(f'{OUT}/S14_mediator_corr_n122.csv'); print(Mz.corr('spearman').round(2).to_string())
v = []
for j in range(4):
    X = np.column_stack([np.ones(len(Mz)), np.delete(Mz.values, j, 1)]); b, *_ = np.linalg.lstsq(X, Mz.values[:, j], rcond=None)
    r2 = 1 - ((Mz.values[:, j] - X @ b) ** 2).sum() / ((Mz.values[:, j] - Mz.values[:, j].mean()) ** 2).sum(); v.append(1 / (1 - r2))
pd.DataFrame({'mediator': MEDS, 'VIF': v}).to_csv(f'{OUT}/S19_vif_n122.csv', index=False); print('VIF', np.round(v, 2))

cov = [pur, prol, caf]
print('ASNS, ERO1A vs hypoxia', [round(pspear(HYP, p[x], cov)[0], 2) for x in ['ASNS', 'ERO1A']])
ch = PROT['UPR canonical chaperones']
print('chaperones vs maturation', round(pspear(S['UPR canonical chaperones'], S['Maturation'], cov)[0], 3),
      'without P4HB', round(pspear(score(p, [x for x in ch if x != 'P4HB']), S['Maturation'], cov)[0], 3),
      '| purity only', round(pspear(S['UPR canonical chaperones'], S['Maturation'], [pur])[0], 3), round(pspear(score(p, [x for x in ch if x != 'P4HB']), S['Maturation'], [pur])[0], 3))
print('hyp vs prolif', round(pspear(HYP, prol, [pur])[0], 2), '| seed vs consensus', round(pspear(S['Hypoxia (core 7)'], HYP, [pur])[0], 2))

# tumor stage
st = c['Tumor Stage'].reindex(p.index).str.extract(r'Stage (I{1,3})')[0]
rows = []
for k in ['Hypoxia consensus', 'Maturation - deposition', 'Maturation', 'Fibrillar collagen', 'UPR PERK/ATF4 targets']:
    x = purity_adjust(S[k], pur); a, b = x[st == 'II'].dropna(), x[st == 'III'].dropna()
    kw = stats.kruskal(*[x[st == s].dropna() for s in ['I', 'II', 'III']]).pvalue
    d = np.sign(b.values[:, None] - a.values[None, :]).mean()
    rows.append(dict(module=k, n_stage_I=int((st == 'I').sum()), n_stage_II=len(a), n_stage_III=len(b), kruskal_p=kw,
                     cliff_delta_III_vs_II=d, mannwhitney_p_III_vs_II=stats.mannwhitneyu(a, b).pvalue))
T = pd.DataFrame(rows); T['q_III_vs_II'] = bh(T.mannwhitney_p_III_vs_II.values); T.to_csv(f'{OUT}/S32_tumor_stage.csv', index=False)
print(T.round(3).to_string(index=False)); print('stage missing', int(st.isna().sum()))

# module preservation percentile (same seed and procedure as a11_robustness.py)
core = ['COL1A1','COL1A2','COL3A1','COL5A1','PLOD1','PLOD2','PLOD3','P4HA1','P4HA2','SERPINH1','P3H1','PPIB','CRTAP','FKBP10','COLGALT1','LOX','LOXL2','BMP1','PCOLCE']
def pw(mask):
    Rr = pd.DataFrame({x: purity_adjust(p[x][mask], pur[mask]) for x in core}).corr('spearman'); return Rr.values[np.triu_indices(len(core), 1)]
obs = stats.spearmanr(pw(g == 'TNBC'), pw(g == 'HR+/HER2-'))[0]
lab = g[g.isin(['TNBC', 'HR+/HER2-'])]; rng = np.random.default_rng(3); null = []
for _ in range(2000):
    sh = pd.Series(rng.permutation(lab.values), index=lab.index).reindex(g.index)
    null.append(stats.spearmanr(pw(sh == 'TNBC'), pw(sh == 'HR+/HER2-'))[0])
null = np.array(null)
P = dict(observed=obs, null_mean=null.mean(), null_p2_5=np.percentile(null, 2.5), null_p97_5=np.percentile(null, 97.5),
         n_perm=len(null), n_perm_below_observed=int((null < obs).sum()), percentile_of_observed=100 * (null < obs).mean(),
         lower_tail_p=(1 + (null <= obs).sum()) / (1 + len(null)), upper_tail_p=(1 + (null >= obs).sum()) / (1 + len(null)))
pd.DataFrame([P]).to_csv(f'{OUT}/S26_preservation_percentile.csv', index=False); print(P)
_pp = os.path.join(OUT, '..', 'preservation_permutation.csv')   # read by fig_cluster.py
if os.path.exists(_pp):
    _a = pd.read_csv(_pp)
    for _k in ['n_perm', 'n_perm_below_observed', 'percentile_of_observed', 'lower_tail_p']: _a[_k] = P[_k]
    _a.to_csv(_pp, index=False)
