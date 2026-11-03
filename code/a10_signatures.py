"""Broader stress-pathway signatures, hypoxia signature stability, and an external comparison.
Addresses R1-M7, R1-M8, R1-M10, R2-4, R2-13."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import numpy as np, pandas as pd
from load import load
from core import groups, pspear, resid, score, compare, bh
from modules import all_scores, PROT

p,m,ph,c = load(); g = groups(c).reindex(p.index)
pur = c['ESTIMATE TumorPurity'].reindex(p.index)
S = all_scores(p); prol=S['Proliferation']; caf=S['CAF']
ok = g.notna(); COV=[pur[ok], prol[ok], caf[ok]]
HYP = S['Hypoxia consensus']

# ---------- 1. broader PERK/ATF4 signature at the mRNA level ----------
ATF4 = ['ASNS','ERO1A','TRIB3','PPP1R15A','ATF3','DDIT3','CHAC1','DDIT4','SESN2','ATF5']
atf4_ok = [x for x in ATF4 if x in m.columns and m[x].notna().sum() >= 60]
perk_mrna = score(m, atf4_ok)
perk_prot = S['UPR PERK/ATF4 targets']
print('='*94); print('BROADER PERK/ATF4 SIGNATURE'); print('='*94)
print(f'protein module: 2 proteins (ASNS, ERO1A) | mRNA module: {len(atf4_ok)} genes ({", ".join(atf4_ok)})')
r,pv,n = pspear(perk_prot[ok], perk_mrna[ok], COV)
print(f'  agreement between the two: rho = {r:.2f} (p = {pv:.1e}, n = {n})')
rows=[]
for lab, sc in [('PERK/ATF4 protein (2)', perk_prot), (f'PERK/ATF4 mRNA ({len(atf4_ok)})', perk_mrna)]:
    r1,p1,_ = pspear(HYP[ok], sc[ok], COV)
    pv2,d2 = compare(resid(sc[ok], COV), g[ok], 'TNBC','HER2+')
    pv3,d3 = compare(resid(sc[ok], COV), g[ok], 'TNBC','HR+/HER2-')
    rows.append(dict(signature=lab, rho_hypoxia=r1, p_hypoxia=p1,
                     d_TNBC_vs_HER2=d2, p1=pv2, d_TNBC_vs_HR=d3, p2=pv3))
    print(f'  {lab:26s} vs hypoxia rho={r1:.2f} (p={p1:.1e}) | TNBC vs HER2+ d={d2:+.2f} | vs HR+ d={d3:+.2f}')
# per-gene, to show no single gene drives it
print('  per-gene mRNA association with hypoxia:')
for gn in atf4_ok:
    r1,p1,_ = pspear(HYP[ok], m[gn][ok], COV); print(f'     {gn:10s} rho={r1:+.2f}  p={p1:.1e}')
pd.DataFrame(rows).to_csv('../out/sig_perk.csv', index=False)

# ---------- 2. XBP1s-responsive transcriptional signature ----------
XBP = ['DNAJB9','SEL1L','HERPUD1','DNAJC3','EDEM1','HSPA5','PDIA6','HYOU1','SDF2L1','DNAJB11','PDIA5','MANF','CRELD2']
xbp_ok = [x for x in XBP if x in m.columns and m[x].notna().sum() >= 60]
xbp_mrna = score(m, xbp_ok)
xbp1_tot = m['XBP1']
print(); print('='*94); print('XBP1s-RESPONSIVE TRANSCRIPTIONAL SIGNATURE'); print('='*94)
print(f'{len(xbp_ok)} genes: {", ".join(xbp_ok)}')
xrows=[]
for lab, sc in [('IRE1/XBP1 protein module', S['UPR IRE1/XBP1 targets']),
                (f'XBP1s target mRNA ({len(xbp_ok)})', xbp_mrna),
                ('XBP1 total mRNA', xbp1_tot)]:
    r1,p1,_ = pspear(HYP[ok], sc[ok], COV)
    pv2,d2 = compare(resid(sc[ok], COV), g[ok], 'TNBC','HER2+')
    pv3,d3 = compare(resid(sc[ok], COV), g[ok], 'TNBC','HR+/HER2-')
    xrows.append(dict(signature=lab, rho_hypoxia=r1, p_hypoxia=p1, d_TNBC_vs_HER2=d2, p_her2=pv2,
                      d_TNBC_vs_HR=d3, p_hr=pv3))
    print(f'  {lab:30s} vs hypoxia rho={r1:+.2f} (p={p1:.1e}) | TNBC vs HER2+ d={d2:+.2f} (p={pv2:.3f}) | vs HR+ d={d3:+.2f} (p={pv3:.3f})')
pd.DataFrame(xrows).to_csv('../out/sig_xbp1.csv', index=False)

# ---------- 3. hypoxia signature bootstrap stability ----------
SEEDS = ['SLC2A1','PGK1','LDHA','NDRG1','BNIP3','CA9','VEGFA']
CAND = sorted(set(PROT['Hypoxia: glycolysis']+PROT['Hypoxia: mitochondrial']+PROT['Hypoxia: HIF targets']))
cand = [x for x in CAND if x in p.columns and p[x].notna().sum()>=60 and x not in ('CA12','STC2')]
seedscore_full = score(p, [s for s in SEEDS if s in p.columns])
rng = np.random.default_rng(11); keep = {x:0 for x in cand}; B=1000
idx_all = np.where(ok.values)[0]
for b in range(B):
    bi = rng.choice(idx_all, len(idx_all), replace=True)
    sub = p.iloc[bi]
    ss = score(sub, [s for s in SEEDS if s in p.columns])
    pr = pur.iloc[bi]
    for x in cand:
        r,_,_ = pspear(sub[x], ss, [pr])
        if r >= 0.30: keep[x] += 1
K = pd.DataFrame({'protein':list(keep), 'retention_pct':[100*keep[x]/B for x in keep]}).sort_values('retention_pct', ascending=False)
K['in_published_signature'] = K.protein.isin(PROT['Hypoxia consensus'])
K.to_csv('../out/sig_hypoxia_bootstrap.csv', index=False)
print(); print('='*94); print(f'HYPOXIA SIGNATURE STABILITY ({B} bootstrap resamples of tumours)'); print('='*94)
print(f'published signature: {len(PROT["Hypoxia consensus"])} proteins')
inc = K[K.in_published_signature]
print(f'  retention of published members: median {inc.retention_pct.median():.0f}%, '
      f'min {inc.retention_pct.min():.0f}% ({inc.iloc[-1].protein}), '
      f'{(inc.retention_pct>=80).sum()} of {len(inc)} retained in >=80% of resamples')
out = K[~K.in_published_signature]
print(f'  non-members: max retention {out.retention_pct.max():.0f}% ({out.iloc[0].protein})')
print(K.head(22).to_string(index=False))

# ---------- 4. external signature comparison (Buffa metagene overlap) ----------
BUFFA = ['VEGFA','SLC2A1','PGAM1','ENO1','LDHA','TPI1','ADM','NDRG1','ALDOA','MIF','GPI','DDIT4',
         'BNIP3','HILPDA','CHCHD2','ANKRD37','PFKP','SLC16A1','HK2','CA9','PGK1','EGLN3','P4HA1']
bp = [x for x in BUFFA if x in p.columns and p[x].notna().sum()>=60]
bp_noenz = [x for x in bp if x != 'P4HA1']
buffa = score(p, bp_noenz)
print(); print('='*94); print('COMPARISON WITH AN EXTERNAL SIGNATURE (Buffa metagene overlap)'); print('='*94)
print(f'{len(bp_noenz)} of {len(BUFFA)} Buffa genes quantified here (P4HA1 excluded to avoid circularity): {", ".join(bp_noenz)}')
r,pv,n = pspear(HYP[ok], buffa[ok], [pur[ok]])
print(f'  agreement with our consensus score: rho = {r:.2f} (p = {pv:.1e})')
brows=[]
for lab, sc in [('consensus (this study, 19)', HYP), (f'Buffa overlap ({len(bp_noenz)})', buffa)]:
    for oname, ov in [('Maturation', S['Maturation']), ('Maturation - deposition', S['Maturation - deposition']),
                      ('PLOD2', p['PLOD2']), ('SERPINH1', p['SERPINH1'])]:
        r1,p1,_ = pspear(sc[ok], ov[ok], COV)
        brows.append(dict(signature=lab, outcome=oname, rho=r1, p=p1))
B2 = pd.DataFrame(brows); B2.to_csv('../out/sig_buffa.csv', index=False)
print(B2.pivot(index='signature', columns='outcome', values='rho').round(2).to_string())
print('\nsaved sig_*.csv')
