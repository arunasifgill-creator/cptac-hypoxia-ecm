"""Shared analysis functions: groups, scores, purity adjustment, stats."""
import numpy as np, pandas as pd
from scipy import stats
from load import load

def _pos(s):
    return s.astype(str).str.strip().str.lower().isin(['positive', '1', '1.0'])

def groups(clin):
    tn = _pos(clin['TNBC Updated Clinical Status'])
    her2 = (_pos(clin['ERBB2 Updated Clinical Status']) | _pos(clin['ERBB2 Proteogenomic Status'])
            | _pos(clin['ERBB2 Gene Amplified'])) & ~tn
    nl = clin['PAM50'].eq('Normal-like')
    g = pd.Series('HR+/HER2-', index=clin.index)
    g[her2] = 'HER2+'; g[tn] = 'TNBC'; g[nl & ~tn] = np.nan
    return g

SETS_PROT = {
 'Fibrillar collagen': ['COL1A1','COL1A2','COL3A1','COL5A1'],
 'Maturation': ['P4HA1','P4HA2','PLOD1','PLOD2','PLOD3','P3H1','SERPINH1','PCOLCE','BMP1','LOX','LOXL2'],
 'MMP turnover': ['MMP2','MMP9','MMP14'],
 'UPR canonical chaperones': ['HSPA5','HYOU1','PDIA4','PDIA6','P4HB','CALR','CANX'],
 'UPR ATF6 targets': ['HSP90B1','DNAJB11','SDF2L1','CRELD2','MANF','PDIA5'],
 'UPR IRE1/XBP1 targets': ['DNAJB9','SEL1L','HERPUD1','DNAJC3'],
 'UPR PERK/ATF4 targets': ['ASNS','ERO1A'],
 'Hypoxia': ['SLC2A1','PGK1','LDHA','NDRG1','BNIP3','CA9','VEGFA'],
 'Collagen export': ['MIA3','MIA2','SEC23A','SEC24D','SAR1A','KLHL12'],
 'TGF-b response': ['SERPINE1','TGFBI','CCN2','JUNB','TPM1','TAGLN','IGFBP3','FSTL3','COMP','LRRC15'],
 'CAF': ['FAP','PDGFRB','ACTA2','POSTN','DCN','LUM'],
 'Mesenchymal (EMT)': ['VIM','CDH2','ZEB1'],
 'Proliferation': ['MKI67','TOP2A'],
}
SETS_MRNA = dict(SETS_PROT)
SETS_MRNA.update({
 'UPR PERK/ATF4 targets': ['ASNS','ERO1A','TRIB3','PPP1R15A','ATF3','DDIT3'],
 'UPR IRE1/XBP1 targets': ['DNAJB9','SEL1L','HERPUD1','DNAJC3','EDEM1'],
 'Hypoxia': ['SLC2A1','PGK1','LDHA','NDRG1','BNIP3','CA9','VEGFA','EGLN3','ADM'],
 'TGF-b response': ['SERPINE1','TGFBI','CCN2','JUNB','TPM1','TAGLN','IGFBP3','FSTL3','COMP','LRRC15','PMEPA1','SMAD7','SKIL'],
 'Mesenchymal (EMT)': ['VIM','CDH2','ZEB1','SNAI2','TWIST1'],
})

def rankpct(df):
    return df.rank(pct=True)

def score(df, genes):
    g = [x for x in genes if x in df and df[x].notna().sum() >= 60]
    return rankpct(df[g]).mean(axis=1, skipna=True)

def purity_adjust(s, purity):
    """Rank-residualise s on purity (as in the manuscript)."""
    ok = s.notna() & purity.notna()
    r = s[ok].rank(); p = purity[ok].rank()
    b = np.polyfit(p, r, 1)
    out = pd.Series(np.nan, index=s.index); out[ok] = r - np.polyval(b, p)
    return out

def resid(s, covars):
    """Rank-residualise s on several covariates (multiple regression on ranks)."""
    X = pd.concat([c.rank() for c in covars], axis=1)
    ok = s.notna() & X.notna().all(axis=1)
    A = np.column_stack([np.ones(ok.sum()), X[ok].values])
    y = s[ok].rank().values
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    out = pd.Series(np.nan, index=s.index); out[ok] = y - A @ beta
    return out

def cliff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return (np.sign(a[:, None] - b[None, :])).mean()

def compare(x, g, a='TNBC', b='HER2+'):
    xa = x[g == a].dropna(); xb = x[g == b].dropna()
    if len(xa) < 5 or len(xb) < 5: return np.nan, np.nan
    return stats.mannwhitneyu(xa, xb).pvalue, cliff(xa, xb)

def bh(p):
    p = np.asarray(p, float); n = np.sum(~np.isnan(p)); q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]; o = idx[np.argsort(p[idx])]
    ranked = p[o] * n / np.arange(1, n + 1)
    q[o] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q

def pspear(x, y, covars=()):
    """Spearman (partial if covars given, via rank residuals)."""
    if covars:
        x = resid(x, covars); y = resid(y, covars)
    ok = x.notna() & y.notna()
    r, p = stats.spearmanr(x[ok], y[ok]); return r, p, ok.sum()
