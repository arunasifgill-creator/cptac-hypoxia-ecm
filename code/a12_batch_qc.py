"""Batch and covariate QC requested at revision.

(1) TMT plex vs subtype: is subtype confounded with plex? (chi-square on the contingency table,
    plus a Monte Carlo exact test because many cells are small).
(2) Every module score tested across TMT plexes by Kruskal-Wallis, with BH correction across
    modules, so the batch check is not limited to the maturation-minus-deposition composite.
(3) The same check for the hypoxia consensus score and each ER stress module explicitly.

Writes ../out/plex_subtype.csv and ../out/batch_qc.csv.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from load import load
from core import groups, bh
from modules import all_scores, PROT

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
rng = np.random.default_rng(7)

p, m, ph, c = load()
g = groups(c).reindex(p.index)
plex = c['TMT Plex'].reindex(p.index).astype(str).str.split('|').str[0]
S = all_scores(p)

# ---------- (1) subtype by plex ----------
ct = pd.crosstab(plex, g.fillna('Normal-like (excluded)'))
chi2, p_chi, dof, exp = stats.chi2_contingency(ct.values)

# Monte Carlo exact test: permute subtype labels across tumours, keep plex sizes
obs = ct.values
lab = g.fillna('Normal-like (excluded)').values
plv = plex.values
cats = list(ct.columns)
plexes = list(ct.index)


def chi2_stat(labels):
    t = pd.crosstab(pd.Series(plv), pd.Series(labels)).reindex(index=plexes, columns=cats).fillna(0).values
    with np.errstate(invalid='ignore', divide='ignore'):
        e = t.sum(1)[:, None] * t.sum(0)[None, :] / t.sum()
        v = np.where(e > 0, (t - e) ** 2 / np.where(e > 0, e, 1), 0)
    return v.sum()


obs_stat = chi2_stat(lab)
null = np.array([chi2_stat(rng.permutation(lab)) for _ in range(5000)])
p_mc = (1 + (null >= obs_stat).sum()) / (1 + len(null))

ct_out = ct.copy()
ct_out.loc['TOTAL'] = ct.sum()
ct_out.to_csv(f'{OUT}/plex_subtype.csv')
print(ct_out.to_string())
print(f'chi2 = {chi2:.1f}, dof = {dof}, asymptotic p = {p_chi:.3f}, Monte Carlo p = {p_mc:.3f}')
print(f'{len(plexes)} plexes, {(exp < 5).mean()*100:.0f}% of cells with expected count < 5')

# ---------- (2,3) every module across plexes ----------
pl = sorted(plex.dropna().unique(), key=lambda x: (len(x), x))
rows = []
for k in list(PROT) + ['Maturation - deposition']:
    x = S[k]
    vals = [x[plex == q].dropna() for q in pl]
    vals = [v for v in vals if len(v) > 2]
    if len(vals) < 3:
        continue
    H, pv = stats.kruskal(*vals)
    # variance explained by plex on ranks (eta-squared, bounded below by 0)
    r = x.rank()
    ok = r.notna() & plex.notna()
    grand = r[ok].mean()
    ssb = sum(len(r[ok][plex[ok] == q]) * (r[ok][plex[ok] == q].mean() - grand) ** 2 for q in pl if (plex[ok] == q).sum())
    sst = ((r[ok] - grand) ** 2).sum()
    rows.append((k, int(ok.sum()), len(vals), H, pv, ssb / sst))

B = pd.DataFrame(rows, columns=['module', 'n_tumors', 'n_plexes', 'H', 'p', 'eta2_plex'])
B['q'] = bh(B['p'].values)
B = B.sort_values('p')
B.to_csv(f'{OUT}/batch_qc.csv', index=False)
print()
print(B.round(4).to_string(index=False))
print()
print(f'modules with BH q < 0.05 across plexes: {(B.q < 0.05).sum()} of {len(B)}')
print(f'largest variance explained by plex: {B.eta2_plex.max()*100:.1f}% ({B.loc[B.eta2_plex.idxmax(),"module"]})')
