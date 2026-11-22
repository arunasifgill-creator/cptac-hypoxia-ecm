"""Third-round revision analyses (editor decision, items 1, 2, 5, 7, 8, 9, 16, 21).

Every continuous analysis is run on all 122 tumors, the population stated in Methods 2.2,
so the robustness tables share one baseline with Table 2. Earlier versions of
a9_mediation_robust.py and a10_signatures.py restricted to the 117 assigned tumors.
Where molecular subtype is a covariate, the five PAM50 Normal-like tumors keep their own
indicator (unassigned), so n stays at 122.
"""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from load import load
from core import groups, resid, score, pspear, compare
from modules import all_scores, PROT

OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'rev3'); os.makedirs(OUT, exist_ok=True)
p, m, ph, c = load(); g = groups(c).reindex(p.index)
pur = c['ESTIMATE TumorPurity'].reindex(p.index)
S = all_scores(p); prol = S['Proliferation']; caf = S['CAF']
ALL = p.index
COV = [pur, prol, caf]
SUBTYPE = [(g == 'TNBC').astype(float), (g == 'HER2+').astype(float), g.isna().astype(float)]
MEDS = ['UPR canonical chaperones', 'UPR PERK/ATF4 targets', 'UPR ATF6 targets', 'ER redox']
SHORT = dict(zip(MEDS, ['Chaperones', 'PERK/ATF4', 'ATF6', 'ER redox']))
HYP = S['Hypoxia consensus']
rng = np.random.default_rng(7)


def prep(s, cov):
    r = resid(s, cov); return (r - r.mean()) / r.std()


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None); return b[1:]


def paths(x, M, y, B=5000):
    def est(ii):
        xx, MM, yy = x[ii], M[ii], y[ii]
        a = np.array([ols(MM[:, j], xx[:, None])[0] for j in range(MM.shape[1])])
        bb = ols(yy, np.column_stack([xx, MM])); cp = bb[0]; b = bb[1:]
        return np.concatenate([a, b, [cp], a * b, [np.sum(a * b)], [ols(yy, xx[:, None])[0]]])
    n = len(y); base = est(np.arange(n))
    boot = np.array([est(rng.integers(0, n, n)) for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return base, lo, hi


def fit(expo, meds, outcome, cov, B=5000):
    X = prep(expo, cov); Y = prep(outcome, cov)
    M = pd.concat([prep(s, cov) for s in meds], axis=1)
    k = X.notna() & Y.notna() & M.notna().all(1)
    base, lo, hi = paths(X[k].values, M[k].values, Y[k].values, B)
    return base, lo, hi, int(k.sum()), len(meds)


OUTS = {'PLOD2': p['PLOD2'], 'P4HA1': p['P4HA1'], 'P4HA2': p['P4HA2'], 'SERPINH1': p['SERPINH1'],
        'Maturation': S['Maturation'], 'Maturation - deposition': S['Maturation - deposition']}

# ---- S15: parallel path coefficients, n = 122 ----
rows = []
for yn, yv in OUTS.items():
    base, lo, hi, n, K = fit(HYP, [S[x] for x in MEDS], yv, COV)
    for j, md in enumerate(MEDS):
        rows.append(dict(outcome=yn, mediator=SHORT[md], n=n, a=base[j], a_lo=lo[j], a_hi=hi[j],
                         b=base[K + j], b_lo=lo[K + j], b_hi=hi[K + j],
                         indirect=base[2 * K + 1 + j], ind_lo=lo[2 * K + 1 + j], ind_hi=hi[2 * K + 1 + j]))
    rows.append(dict(outcome=yn, mediator='TOTAL indirect', n=n, indirect=base[-2], ind_lo=lo[-2], ind_hi=hi[-2],
                     total=base[-1], direct=base[2 * K], prop=base[-2] / base[-1]))
S15 = pd.DataFrame(rows); S15.to_csv(f'{OUT}/S15_med_paths_n122.csv', index=False)

# ---- S16: single-mediator models ----
rows = []
for yn, yv in OUTS.items():
    for md in MEDS:
        base, lo, hi, n, K = fit(HYP, [S[md]], yv, COV, B=3000)
        rows.append(dict(outcome=yn, mediator=SHORT[md], n=n, a=base[0], b=base[1], direct=base[2],
                         indirect=base[3], ind_lo=lo[3], ind_hi=hi[3], total=base[-1], prop=base[3] / base[-1]))
S16 = pd.DataFrame(rows); S16.to_csv(f'{OUT}/S16_single_mediator_n122.csv', index=False)

# ---- S17: alternative specifications ----
rows = []
for lab, expo, cov in [('parallel, as published', HYP, COV),
                       ('parallel + subtype', HYP, COV + SUBTYPE),
                       ('non-glycolytic exposure', S['Hypoxia non-glycolytic'], COV),
                       ('non-glycolytic + subtype', S['Hypoxia non-glycolytic'], COV + SUBTYPE)]:
    for yn in ['PLOD2', 'Maturation - deposition']:
        base, lo, hi, n, K = fit(expo, [S[x] for x in MEDS], OUTS[yn], cov, B=2000)
        rows.append(dict(spec=lab, outcome=yn, n=n, total=base[-1], direct=base[2 * K],
                         indirect=base[-2], ind_lo=lo[-2], ind_hi=hi[-2], prop_mediated=base[-2] / base[-1]))
S17 = pd.DataFrame(rows); S17.to_csv(f'{OUT}/S17_alt_specs_n122.csv', index=False)

# ---- S18: reversed ordering ----
base, lo, hi, n, K = fit(HYP, [S['Maturation']], S['UPR PERK/ATF4 targets'], COV, B=3000)
S18 = pd.DataFrame([dict(model='hypoxia -> maturation -> PERK/ATF4', n=n, total=base[-1], direct=base[2],
                         indirect=base[3], ind_lo=lo[3], ind_hi=hi[3], prop=base[3] / base[-1])])
S18.to_csv(f'{OUT}/S18_reversed_n122.csv', index=False)

# ---- S24: alternative PERK/ATF4 readouts (single-mediator) ----
ATF4 = ['ASNS', 'ERO1A', 'TRIB3', 'PPP1R15A', 'ATF3', 'DDIT3', 'CHAC1', 'DDIT4', 'SESN2', 'ATF5']
atf4_ok = [x for x in ATF4 if x in m.columns and m[x].notna().sum() >= 60]
atf4_8 = [x for x in atf4_ok if x not in ('ASNS', 'ERO1A')]
READ = {'PERK module as published (ASNS + ERO1A protein)': S['UPR PERK/ATF4 targets'],
        'ASNS protein alone': p['ASNS'], 'ERO1A protein alone': p['ERO1A'],
        f'ATF4 target mRNA ({len(atf4_ok)} genes)': score(m, atf4_ok),
        f'ATF4 target mRNA, ASNS and ERO1A removed ({len(atf4_8)} genes)': score(m, atf4_8)}
rows = []
for yn in ['PLOD2', 'Maturation - deposition']:
    for lab, med in READ.items():
        base, lo, hi, n, K = fit(HYP, [med], OUTS[yn], COV, B=5000)
        rows.append(dict(outcome=yn, mediator=lab, n=n, indirect=base[3], lo=lo[3], hi=hi[3],
                         total=base[-1], prop=base[3] / base[-1]))
S24 = pd.DataFrame(rows); S24.to_csv(f'{OUT}/S24_perk_variants_n122.csv', index=False)

# ---- S20, S21, S23: signatures on 122 ----
def sig_row(lab, sc):
    r, pv, n = pspear(HYP, sc, COV)
    x = resid(sc, COV)
    p1, d1 = compare(x, g, 'TNBC', 'HER2+'); p2, d2 = compare(x, g, 'TNBC', 'HR+/HER2-')
    return dict(signature=lab, n=n, rho_hypoxia=r, p_hypoxia=pv, d_TNBC_vs_HER2=d1, p_her2=p1, d_TNBC_vs_HR=d2, p_hr=p2)
pd.DataFrame([sig_row('PERK/ATF4 protein (2)', S['UPR PERK/ATF4 targets']),
              sig_row(f'ATF4 target mRNA ({len(atf4_ok)})', score(m, atf4_ok)),
              sig_row(f'ATF4 target mRNA without ASNS/ERO1A ({len(atf4_8)})', score(m, atf4_8))]
             ).to_csv(f'{OUT}/S20_sig_perk_n122.csv', index=False)
XBP = ['DNAJB9', 'SEL1L', 'HERPUD1', 'DNAJC3', 'EDEM1', 'HSPA5', 'PDIA6', 'HYOU1', 'SDF2L1', 'DNAJB11', 'PDIA5', 'MANF', 'CRELD2']
xbp_ok = [x for x in XBP if x in m.columns and m[x].notna().sum() >= 60]
pd.DataFrame([sig_row('IRE1/XBP1 protein module (4)', S['UPR IRE1/XBP1 targets']),
              sig_row(f'XBP1s target mRNA ({len(xbp_ok)})', score(m, xbp_ok)),
              sig_row('XBP1 total mRNA', m['XBP1'])]).to_csv(f'{OUT}/S21_sig_xbp1_n122.csv', index=False)
BUFFA = ['VEGFA', 'SLC2A1', 'PGAM1', 'ENO1', 'LDHA', 'TPI1', 'ADM', 'NDRG1', 'ALDOA', 'MIF', 'GPI', 'DDIT4',
         'BNIP3', 'HILPDA', 'CHCHD2', 'ANKRD37', 'PFKP', 'SLC16A1', 'HK2', 'CA9', 'PGK1', 'EGLN3']
bp = [x for x in BUFFA if x in p.columns and p[x].notna().sum() >= 60]
buffa = score(p, bp)
rows = [dict(signature='agreement consensus vs Buffa', outcome='-', rho=pspear(HYP, buffa, [pur])[0])]
for lab, sc in [('consensus (19)', HYP), (f'Buffa overlap ({len(bp)})', buffa)]:
    for on, ov in [('Maturation', S['Maturation']), ('Maturation - deposition', S['Maturation - deposition']),
                   ('PLOD2', p['PLOD2']), ('SERPINH1', p['SERPINH1'])]:
        r, pv, n = pspear(sc, ov, COV); rows.append(dict(signature=lab, outcome=on, rho=r, p=pv, n=n))
pd.DataFrame(rows).to_csv(f'{OUT}/S23_buffa_n122.csv', index=False)

# ---- S22: signature bootstrap, leave-self-out rule for seed proteins ----
SEEDS = ['SLC2A1', 'PGK1', 'LDHA', 'NDRG1', 'BNIP3', 'CA9', 'VEGFA']
CAND = sorted(set(PROT['Hypoxia: glycolysis'] + PROT['Hypoxia: mitochondrial'] + PROT['Hypoxia: HIF targets']))
cand = [x for x in CAND if x in p.columns and p[x].notna().sum() >= 60 and x not in ('CA12', 'STC2')]

def selected(sub, pr):
    out = []
    for x in cand:
        seeds = [s for s in SEEDS if s != x]
        if pspear(sub[x], score(sub, seeds), [pr])[0] >= 0.30: out.append(x)
    return out
full = selected(p, pur)
print('Selection on full cohort reproduces the published list:', sorted(full) == sorted(PROT['Hypoxia consensus']))
keep = {x: 0 for x in cand}; Bb = 1000; r2 = np.random.default_rng(11)
for _ in range(Bb):
    bi = r2.integers(0, len(p), len(p)); sub = p.iloc[bi].reset_index(drop=True); pr = pur.iloc[bi].reset_index(drop=True)
    for x in selected(sub, pr): keep[x] += 1
K = pd.DataFrame({'protein': list(keep), 'retention_pct': [100 * keep[x] / Bb for x in keep]}).sort_values('retention_pct', ascending=False)
K['in_published_signature'] = K.protein.isin(PROT['Hypoxia consensus'])
K.to_csv(f'{OUT}/S22_bootstrap_leave_self_out_n122.csv', index=False)

# ---- item 5: plex by subtype, three assigned groups only ----
plex = c['TMT Plex'].reindex(p.index).astype(str).str.split('|').str[0]
ok = g.notna(); ct = pd.crosstab(plex[ok], g[ok]); chi2, pc, dof, exp = stats.chi2_contingency(ct.values)
lab, pl = g[ok].values, plex[ok].values
def st(l):
    t = pd.crosstab(pd.Series(pl), pd.Series(l)).values; e = t.sum(1)[:, None] * t.sum(0)[None, :] / t.sum()
    return ((t - e) ** 2 / e).sum()
r3 = np.random.default_rng(7); o = st(lab); null = np.array([st(r3.permutation(lab)) for _ in range(5000)])
pd.DataFrame([dict(groups=3, n=int(ok.sum()), chi2=chi2, dof=dof, p_asymptotic=pc,
                   p_monte_carlo=(1 + (null >= o).sum()) / 5001, pct_cells_expected_lt5=100 * (exp < 5).mean())]
             ).to_csv(f'{OUT}/S30_chi2_three_groups.csv', index=False)
pd.read_csv(f'{OUT}/S30_chi2_three_groups.csv').to_csv(os.path.join(OUT, '..', 'plex_subtype_test.csv'), index=False)   # read by fig_s1_cohort.py

pd.set_option('display.width', 200)
for f in sorted(os.listdir(OUT)):
    print('\n==', f); print(pd.read_csv(f'{OUT}/{f}').round(3).to_string(index=False))
