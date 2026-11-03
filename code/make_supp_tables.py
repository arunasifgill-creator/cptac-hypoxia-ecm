"""Build the supplementary numerical tables requested by reviewers
(R1-M9, R1-m2, R2-8, R2-10, R2-12)."""
import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import pandas as pd, numpy as np
from load import load
from core import groups
from modules import PROT, GROUP

OUT='/home/claude/Supplementary_Tables.xlsx'
p,m,ph,c = load(); g = groups(c).reindex(p.index)

# ---- S1 gene panel and coverage ----
cov = p.notna().mean()
panel = pd.DataFrame({'gene': p.columns,
                      'protein_n': p.notna().sum().values,
                      'protein_fraction': cov.round(3).values,
                      'mRNA_n': [m[x].notna().sum() if x in m.columns else 0 for x in p.columns]})
panel['protein_used'] = panel.protein_n >= 60
panel['coverage_class'] = np.where(panel.protein_fraction>=0.8,'>=80%',
                           np.where(panel.protein_fraction>0,'<80%','none'))
mod_of = {}
for mod, genes in PROT.items():
    for gn in genes: mod_of.setdefault(gn, []).append(mod)
panel['modules'] = panel.gene.map(lambda x: '; '.join(mod_of.get(x, [])))
panel = panel.sort_values('gene')

# ---- S2 module definitions ----
grp_of = {mm:k for k,v in GROUP.items() for mm in v}
mods = pd.DataFrame([{'module':k,'program':grp_of.get(k,'other'),'n_genes':len(v),
                      'n_quantified':sum(1 for x in v if x in p.columns and p[x].notna().sum()>=60),
                      'genes':'; '.join(v)} for k,v in PROT.items()])

# ---- S3 sample-level module scores ----
from modules import all_scores
S = all_scores(p)
scores = pd.DataFrame(S); scores.insert(0,'subtype', g.values)
scores.insert(1,'ESTIMATE_purity', c['ESTIMATE TumorPurity'].reindex(p.index).values)
scores.insert(0,'sample_id', p.index)

sheets = [
 ('S1_gene_panel_coverage', panel),
 ('S2_module_definitions', mods),
 ('S3_sample_module_scores', scores),
 ('S4_subtype_effects', pd.read_csv('../out/subtype_effects.csv')),
 ('S5_mediation', pd.read_csv('../out/mediation.csv')),
 ('S6_sens_covariates', pd.read_csv('../out/sens_covariates.csv')),
 ('S7_sens_stratified', pd.read_csv('../out/sens_stratified.csv')),
 ('S8_sens_leave_one_out', pd.read_csv('../out/sens_leave_one_out.csv')),
 ('S9_ML_foldsafe', pd.read_csv('../out/ml_auc_foldsafe.csv')),
 ('S10_ML_coefficients', pd.read_csv('../out/ml_coefs_foldsafe.csv')),
 ('S11_phosphosites_all', pd.read_csv('../out/phospho_sites.csv')),
 ('S12_phospho_protein_adj', pd.read_csv('../out/phospho_occupancy.csv')),
 ('S13_hypoxia_axis', pd.read_csv('../out/hypoxia_axis.csv')),
 ('S14_med_mediator_corr', pd.read_csv('../out/med_mediator_corr.csv')),
 ('S15_med_paths', pd.read_csv('../out/med_paths.csv')),
 ('S16_med_single_mediator', pd.read_csv('../out/med_single.csv')),
 ('S17_med_alt_specs', pd.read_csv('../out/med_altspecs.csv')),
 ('S18_med_reversed_order', pd.read_csv('../out/med_reversed.csv')),
 ('S19_med_vif', pd.read_csv('../out/med_vif.csv')),
 ('S20_sig_perk', pd.read_csv('../out/sig_perk.csv')),
 ('S21_sig_xbp1', pd.read_csv('../out/sig_xbp1.csv')),
 ('S22_hypoxia_bootstrap', pd.read_csv('../out/sig_hypoxia_bootstrap.csv')),
 ('S23_buffa_comparison', pd.read_csv('../out/sig_buffa.csv')),
 ('S24_med_perk_variants', pd.read_csv('../out/med_perk_variants.csv')),
 ('S25_her2_definition', pd.read_csv('../out/her2_definition.csv')),
 ('S26_preservation_perm', pd.read_csv('../out/preservation_permutation.csv')),
 ('S27_phospho_site_qc', pd.read_csv('../out/phospho_site_qc.csv')),
 ('S28_phospho_missingness', pd.read_csv('../out/phospho_missingness.csv')),
 ('S29_batch_qc_modules', pd.read_csv('../out/batch_qc.csv')),
 ('S30_plex_by_subtype', pd.read_csv('../out/plex_subtype.csv')),
]
DESC={
 'S1_gene_panel_coverage':'All 318 genes queried, protein and mRNA coverage, module membership',
 'S2_module_definitions':'Module names, parent program and member genes',
 'S3_sample_module_scores':'Per-tumor module scores with subtype and ESTIMATE purity',
 'S4_subtype_effects':'All module-level subtype comparisons with Cliff delta, p and BH q',
 'S5_mediation':'Parallel mediation estimates with 95% bootstrap intervals',
 'S6_sens_covariates':'Principal associations under eight covariate specifications',
 'S7_sens_stratified':'Within-subtype estimates',
 'S8_sens_leave_one_out':'Leave-one-protein-out results for the hypoxia signature',
 'S9_ML_foldsafe':'Fold-safe classifier performance, both interval types, balanced accuracy, sensitivity, specificity',
 'S10_ML_coefficients':'Standardized classifier coefficients averaged over outer folds',
 'S11_phosphosites_all':'All 385 phosphosites measured in at least 60 tumors',
 'S12_phospho_protein_adj':'The 193 protein-adjusted phosphosite estimates with BH q',
 'S13_hypoxia_axis':'Module correlations with hypoxia under three covariate specifications',
 'S14_med_mediator_corr':'Intercorrelations among the four ER stress mediators',
 'S15_med_paths':'Mediation path coefficients with bootstrap intervals',
 'S16_med_single_mediator':'Single-mediator sensitivity models',
 'S17_med_alt_specs':'Mediated fractions under alternative exposures and covariates',
 'S18_med_reversed_order':'The reversed pathway ordering',
 'S19_med_vif':'Variance inflation factors for the mediators',
 'S20_sig_perk':'Broader PERK/ATF4 mRNA signature',
 'S21_sig_xbp1':'XBP1s target signature',
 'S22_hypoxia_bootstrap':'Bootstrap retention of each hypoxia signature gene',
 'S23_buffa_comparison':'Comparison with the external Buffa hypoxia signature',
 'S24_med_perk_variants':'Mediated fraction under alternative PERK/ATF4 readouts',
 'S25_her2_definition':'Effect sizes under four definitions of the HER2+ group',
 'S26_preservation_perm':'Module preservation: observed value, permutation null and bootstrap interval',
 'S27_phospho_site_qc':'Per-site residues, peptide multiplicity, sample size, missingness and regulatory annotation',
 'S28_phospho_missingness':'Export phosphosites: complete case versus median imputation',
 'S29_batch_qc_modules':'Every module score tested across the 15 TMT plexes with BH q',
 'S30_plex_by_subtype':'Contingency table of clinical group by TMT plex',
}
with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    idx = pd.DataFrame([{'sheet':n,'description':DESC.get(n,''),'rows':len(df),'columns':df.shape[1]} for n,df in sheets])
    assert all(DESC.get(n) for n,_ in sheets), 'every sheet needs a description'
    idx.to_excel(w, sheet_name='S0_index', index=False)
    for n, df in sheets: df.to_excel(w, sheet_name=n, index=False)
print('saved', OUT)
for n, df in sheets: print(f'  {n:26s} {len(df):5d} rows x {df.shape[1]} cols')
