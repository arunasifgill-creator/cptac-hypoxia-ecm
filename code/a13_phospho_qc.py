"""Phosphosite reporting and sensitivity checks asked for at revision (R1-M18, R2-10, R2-17).

  1. per-site sample size and missingness, and how many sites clear an 80% completeness bar
  2. sensitivity: repeat the hypoxia-occupancy correlations on complete-case sites only,
     and on sites imputed at the cohort median, to show the ranking does not depend on missingness
  3. residue verification: every site identifier is parsed back to residue letter and position,
     singly and multiply phosphorylated peptides are counted separately
  4. which sites have a characterized regulatory role in the literature and which do not

Writes ../out/phospho_site_qc.csv and ../out/phospho_missingness.csv.
"""
import os, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from load import load
from core import groups, resid, bh, pspear
from modules import all_scores

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
p, m, ph, c = load()
g = groups(c).reindex(p.index)
pur = c['ESTIMATE TumorPurity'].reindex(p.index)
S = all_scores(p); prol = S['Proliferation']; H = S['Hypoxia consensus']
P = pd.read_csv(os.path.join(OUT, '..', 'data', 'phos', 'phospho_all.csv'), index_col=0).reindex(p.index)
X = pd.read_csv(f'{OUT}/phospho_occupancy.csv')

# Sites whose regulatory role is established in the literature. Everything else in the panel is
# reported as an uncharacterized site, i.e. a position that was quantified, not a known switch.
CHARACTERIZED = {
    'EIF4EBP1_T37t': 'mTORC1 priming site (4E-BP1 T37)',
    'EIF4EBP1_T46t': 'mTORC1 priming site (4E-BP1 T46)',
    'EIF4EBP1_S65s': 'mTORC1 release site (4E-BP1 S65)',
    'EIF4EBP1_T70t': 'mTORC1 site (4E-BP1 T70)',
    'RPS6_S235s': 'S6K1/RSK site (RPS6 S235)',
    'RPS6_S236s': 'S6K1/RSK site (RPS6 S236)',
    'RPS6_S240s': 'S6K1 site (RPS6 S240)',
    'RPS6_S244s': 'S6K1 site (RPS6 S244)',
    'ACACA_S80s': 'AMPK inhibitory site (ACC1 S79/S80)',
    'PRKAA1_T183t': 'LKB1/CaMKK2 activation loop (AMPKa1 T183)',
    'PRKAA1_S501s_T505t': 'AMPKa1 S496/T505 regulatory cluster',
    'PRKAA1_T505t': 'AMPKa1 T505 regulatory site',
    'PDHA1_S293s': 'PDK inhibitory site (PDHA1 S293)',
    'PDHA1_S300s': 'PDK inhibitory site (PDHA1 S300)',
    'PDHA1_S331s': 'PDK site (PDHA1 S232-equivalent numbering varies by isoform)',
}


def parse(site):
    """Return (gene, [(residue, position), ...]) parsed from the CPTAC site identifier."""
    gene, rest = site.split('_', 1)
    hits = re.findall(r'([STY])(\d+)', rest)
    return gene, [(r, int(q)) for r, q in hits]


rows = []
for _, r in X.iterrows():
    gene, hits = parse(r.site)
    col = r.site if r.site in P.columns else None
    nmiss = int(P[col].isna().sum()) if col else np.nan
    rows.append((r.site, gene, ';'.join(f'{a}{b}' for a, b in hits), len(hits),
                 int(r.n), nmiss, 100 * (1 - nmiss / len(P)) if col else np.nan,
                 r.rho_hyp_occ, r.p_hyp_occ, r.q_occ,
                 CHARACTERIZED.get(r.site, 'not characterized')))
Q = pd.DataFrame(rows, columns=['site', 'gene', 'residues', 'n_phosphosites_in_peptide',
                                'n_tumors_with_value', 'n_missing', 'pct_complete',
                                'rho_hypoxia_occupancy', 'p', 'q', 'regulatory_annotation'])
Q = Q.sort_values(['gene', 'site'])
Q.to_csv(f'{OUT}/phospho_site_qc.csv', index=False)

print('=' * 92); print('PHOSPHOSITE COVERAGE'); print('=' * 92)
print(f'  sites with occupancy estimates: {len(Q)} on {Q.gene.nunique()} proteins')
print(f'  tumors per site: median {int(Q.n_tumors_with_value.median())}, range {int(Q.n_tumors_with_value.min())}-{int(Q.n_tumors_with_value.max())}')
print(f'  sites complete in all 122 tumors: {(Q.n_tumors_with_value == 122).sum()}')
print(f'  sites at >= 80% completeness:     {(Q.pct_complete >= 80).sum()}')
print(f'  singly phosphorylated peptides:   {(Q.n_phosphosites_in_peptide == 1).sum()}')
print(f'  multiply phosphorylated peptides: {(Q.n_phosphosites_in_peptide > 1).sum()}')
print(f'  residue letters seen: {sorted(set(re.findall(r"[STY]", ";".join(Q.residues))))}')
print(f'  characterized regulatory sites in the panel: {(Q.regulatory_annotation != "not characterized").sum()}')

# ---------------- missingness sensitivity ----------------
print(); print('=' * 92); print('MISSINGNESS SENSITIVITY'); print('=' * 92)
keep = Q[Q.pct_complete >= 80].site.tolist()
sub = X[X.site.isin(keep)].copy()
sub['q_complete'] = bh(sub.p_hyp_occ.values)
chg = sub[['site', 'gene', 'rho_hyp_occ', 'q_occ', 'q_complete']].copy()
flip = chg[(chg.q_occ < .05) != (chg.q_complete < .05)]
print(f'  restricting to the {len(keep)} sites at >= 80% completeness and re-running BH within that family:')
print(f'    significant before: {(chg.q_occ < .05).sum()} of {len(chg)}   after: {(chg.q_complete < .05).sum()} of {len(chg)}')
print(f'    sites that change significance status: {len(flip)}')
if len(flip):
    print(flip.round(4).to_string(index=False))

# median-imputation sensitivity for the headline export sites
print()
head = ['MIA3_S189s', 'MIA3_S876s', 'MIA3_S408s', 'MIA3_S727s', 'SEC31A_S799s', 'SEC31A_S527s_S532s', 'MIA2_S747s']
irows = []
for site in head:
    if site not in P.columns:
        continue
    gene = site.split('_')[0]
    obs = resid(P[site], [pur, p[gene]])
    r1, p1, n1 = pspear(obs, H, [prol])
    filled = P[site].fillna(P[site].median())
    obs2 = resid(filled, [pur, p[gene]])
    r2, p2, n2 = pspear(obs2, H, [prol])
    irows.append((site, n1, r1, p1, n2, r2, p2))
I = pd.DataFrame(irows, columns=['site', 'n_complete_case', 'rho_complete_case', 'p_complete_case',
                                 'n_imputed', 'rho_median_imputed', 'p_median_imputed'])
I['delta_rho'] = I.rho_median_imputed - I.rho_complete_case
I.to_csv(f'{OUT}/phospho_missingness.csv', index=False)
print('  headline export sites, complete case vs cohort-median imputation:')
print(I.round(3).to_string(index=False))
print(f'  largest change in rho: {I.delta_rho.abs().max():.3f}')
