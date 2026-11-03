"""Mediation robustness: path coefficients, mediator collinearity, single-mediator models,
alternative exposures, subtype adjustment and a reversed pathway ordering.
Addresses reviewer comments R1-M13 and R2-11."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from load import load
from core import groups, resid
from modules import all_scores, PROT

p,m,ph,c = load(); g = groups(c).reindex(p.index)
pur = c['ESTIMATE TumorPurity'].reindex(p.index)
S = all_scores(p); prol = S['Proliferation']; caf = S['CAF']
is_tnbc = (g=='TNBC').astype(float); is_her2 = (g=='HER2+').astype(float)
rng = np.random.default_rng(7)
MEDS = ['UPR canonical chaperones','UPR PERK/ATF4 targets','UPR ATF6 targets','ER redox']
SHORT = {'UPR canonical chaperones':'Chaperones','UPR PERK/ATF4 targets':'PERK/ATF4',
         'UPR ATF6 targets':'ATF6','ER redox':'ER redox'}

def prep(s, cov):
    r = resid(s, cov); return (r - r.mean())/r.std()
def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b,*_ = np.linalg.lstsq(X, y, rcond=None); return b[1:]

def paths(x, M, y, B=5000):
    """Return a-paths, b-paths, direct, per-mediator indirect, with bootstrap CIs."""
    def est(ii):
        xx, MM, yy = x[ii], M[ii], y[ii]
        a = np.array([ols(MM[:,j], xx[:,None])[0] for j in range(MM.shape[1])])
        bb = ols(yy, np.column_stack([xx, MM])); cp = bb[0]; b = bb[1:]
        return np.concatenate([a, b, [cp], a*b, [np.sum(a*b)], [ols(yy, xx[:,None])[0]]])
    n = len(y); base = est(np.arange(n))
    boot = np.array([est(rng.integers(0,n,n)) for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5,97.5], axis=0)
    return base, lo, hi

def vif(M):
    out=[]
    for j in range(M.shape[1]):
        other = np.delete(M, j, axis=1)
        X = np.column_stack([np.ones(len(M)), other])
        b,*_ = np.linalg.lstsq(X, M[:,j], rcond=None)
        r2 = 1 - ((M[:,j] - X@b)**2).sum()/((M[:,j]-M[:,j].mean())**2).sum()
        out.append(1/max(1e-9, 1-r2))
    return np.array(out)

OUTS = {'PLOD2':p['PLOD2'],'P4HA2':p['P4HA2'],'SERPINH1':p['SERPINH1'],
        'Maturation':S['Maturation'],'Maturation - deposition':S['Maturation - deposition']}
ok_idx = g.notna()
BASECOV = [pur[ok_idx], prol[ok_idx], caf[ok_idx]]

# ---------- 1. mediator correlations and collinearity ----------
Mz = pd.concat([prep(S[k][ok_idx], BASECOV).rename(SHORT[k]) for k in MEDS], axis=1).dropna()
corr = Mz.corr(method='spearman')
v = vif(Mz.values)
print('='*96); print('MEDIATOR INTERCORRELATION (Spearman, after residualisation) and VIF'); print('='*96)
print(corr.round(2).to_string())
print(); print('VIF: ' + '  '.join(f'{SHORT[k]} {x:.2f}' for k,x in zip(MEDS, v)))
corr.to_csv('../out/med_mediator_corr.csv')
pd.DataFrame({'mediator':[SHORT[k] for k in MEDS],'VIF':v}).to_csv('../out/med_vif.csv', index=False)

# ---------- 2. path coefficients, parallel model ----------
print(); print('='*96); print('PATH COEFFICIENTS, PARALLEL MODEL (exposure = hypoxia consensus)'); print('='*96)
rows=[]
X = prep(S['Hypoxia consensus'][ok_idx], BASECOV)
for yn, yv in OUTS.items():
    Y = prep(yv[ok_idx], BASECOV)
    msk = X.notna() & Y.notna() & Mz.reindex(X.index).notna().all(1)
    base, lo, hi = paths(X[msk].values, Mz.reindex(X.index)[msk].values, Y[msk].values)
    k = len(MEDS)
    for j,md in enumerate(MEDS):
        rows.append(dict(outcome=yn, mediator=SHORT[md],
                         a=base[j], a_lo=lo[j], a_hi=hi[j],
                         b=base[k+j], b_lo=lo[k+j], b_hi=hi[k+j],
                         indirect=base[2*k+1+j], ind_lo=lo[2*k+1+j], ind_hi=hi[2*k+1+j]))
    rows.append(dict(outcome=yn, mediator='TOTAL / DIRECT',
                     a=np.nan, a_lo=np.nan, a_hi=np.nan, b=np.nan, b_lo=np.nan, b_hi=np.nan,
                     indirect=base[-2], ind_lo=lo[-2], ind_hi=hi[-2]))
P = pd.DataFrame(rows); P.to_csv('../out/med_paths.csv', index=False)
show = P[P.outcome.isin(['PLOD2','Maturation - deposition'])]
print(show[['outcome','mediator','a','b','indirect','ind_lo','ind_hi']].round(3).to_string(index=False))

# ---------- 3. single-mediator models ----------
print(); print('='*96); print('SINGLE-MEDIATOR MODELS (each mediator alone)'); print('='*96)
srows=[]
for yn, yv in OUTS.items():
    Y = prep(yv[ok_idx], BASECOV)
    for md in MEDS:
        Mi = prep(S[md][ok_idx], BASECOV).to_frame()
        msk = X.notna() & Y.notna() & Mi.notna().all(1)
        base, lo, hi = paths(X[msk].values, Mi[msk].values, Y[msk].values, B=3000)
        srows.append(dict(outcome=yn, mediator=SHORT[md], a=base[0], b=base[1],
                          direct=base[2], indirect=base[3], ind_lo=lo[3], ind_hi=hi[3],
                          prop=base[3]/base[-1] if abs(base[-1])>.05 else np.nan))
SM = pd.DataFrame(srows); SM.to_csv('../out/med_single.csv', index=False)
print(SM[SM.outcome.isin(['PLOD2','Maturation - deposition'])].round(3).to_string(index=False))

# ---------- 4. alternative exposures / covariates / ordering ----------
print(); print('='*96); print('ALTERNATIVE SPECIFICATIONS (indirect share of total)'); print('='*96)
def parallel_share(expo, cov, outcome):
    Xl = prep(expo[ok_idx], cov); Yl = prep(outcome[ok_idx], cov)
    Ml = pd.concat([prep(S[k][ok_idx], cov) for k in MEDS], axis=1)
    msk = Xl.notna() & Yl.notna() & Ml.notna().all(1)
    base,_,_ = paths(Xl[msk].values, Ml[msk].values, Yl[msk].values, B=2000)
    tot = base[-1]; ind = base[-2]
    return tot, base[2*len(MEDS)], ind, (ind/tot if abs(tot)>.05 else np.nan)
SPECS=[('parallel, as published', S['Hypoxia consensus'], BASECOV),
       ('parallel + subtype',     S['Hypoxia consensus'], BASECOV+[is_tnbc[ok_idx], is_her2[ok_idx]]),
       ('non-glycolytic exposure',S['Hypoxia non-glycolytic'], BASECOV),
       ('non-glycolytic + subtype',S['Hypoxia non-glycolytic'], BASECOV+[is_tnbc[ok_idx], is_her2[ok_idx]])]
arows=[]
for lab, expo, cov in SPECS:
    for yn in ['PLOD2','Maturation - deposition']:
        tot, dir_, ind, sh = parallel_share(expo, cov, OUTS[yn])
        arows.append(dict(spec=lab, outcome=yn, total=tot, direct=dir_, indirect=ind, prop_mediated=sh))
A = pd.DataFrame(arows); A.to_csv('../out/med_altspecs.csv', index=False)
print(A.round(3).to_string(index=False))

# reversed ordering: ECM as mediator, ER stress as outcome
print(); print('--- reversed ordering: hypoxia -> maturation -> ER stress (PERK/ATF4) ---')
Xr = prep(S['Hypoxia consensus'][ok_idx], BASECOV)
Mr = prep(S['Maturation'][ok_idx], BASECOV).to_frame()
Yr = prep(S['UPR PERK/ATF4 targets'][ok_idx], BASECOV)
msk = Xr.notna() & Yr.notna() & Mr.notna().all(1)
base, lo, hi = paths(Xr[msk].values, Mr[msk].values, Yr[msk].values, B=3000)
print(f'  total {base[-1]:.3f} | direct {base[2]:.3f} | indirect via maturation {base[3]:.3f} '
      f'({lo[3]:.3f} to {hi[3]:.3f}) | share {base[3]/base[-1]:.2f}')
pd.DataFrame([dict(model='hypoxia -> maturation -> PERK/ATF4', total=base[-1], direct=base[2],
                   indirect=base[3], ind_lo=lo[3], ind_hi=hi[3], prop=base[3]/base[-1])]
             ).to_csv('../out/med_reversed.csv', index=False)
print('\nsaved med_*.csv')
