"""Build the final supplementary workbook for the third revision.

Starts from the second-revision workbook written by make_supp_tables.py and replaces or adds the sheets
recomputed in this round from ../out/rev3 (written by a14_revision3.py, a15_tcga_replication.py and
a16_revision3_sensitivity.py). Writes ../supplement/Supplementary_Tables.xlsx.

Usage: python3 make_supp_tables_rev3.py [path to the second-revision workbook]
"""
import sys, os, copy, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, openpyxl
from load import load
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/Supplementary_Tables.xlsx'
O = os.path.join(HERE, '..', 'out', 'rev3') + os.sep
DST = os.path.join(HERE, '..', 'supplement', 'Supplementary_Tables.xlsx'); os.makedirs(os.path.dirname(DST), exist_ok=True)
wb = openpyxl.load_workbook(SRC)
HSTYLE = copy.copy(wb['S6_sens_covariates'].cell(1, 1)._style)

def put(name, df, note=None):
    if name in wb.sheetnames:
        ws = wb[name]; ws.delete_rows(1, ws.max_row + 5)
    else:
        ws = wb.create_sheet(name)
    for j, c in enumerate(df.columns, 1):
        ws.cell(1, j, c)._style = copy.copy(HSTYLE)
        col = openpyxl.utils.get_column_letter(j)
        ws.column_dimensions[col].width = max(ws.column_dimensions[col].width or 0, min(48, max(12, len(str(c)) + 2)))
    for i, row in enumerate(df.itertuples(index=False), 2):
        for j, v in enumerate(row, 1):
            if isinstance(v, (float, np.floating)) and np.isnan(v): v = None
            elif isinstance(v, np.generic): v = v.item()
            ws.cell(i, j, v)
    if note: ws.cell(len(df) + 3, 1, note)

rd = lambda f: pd.read_csv(O + f)
SH = {'UPR canonical chaperones': 'Chaperones', 'UPR PERK/ATF4 targets': 'PERK/ATF4', 'UPR ATF6 targets': 'ATF6', 'ER redox': 'ER redox'}
put('S6_sens_covariates', rd('S6_sens_covariates_n122.csv'))
put('S7_sens_stratified', rd('S7_sens_stratified.csv'))
put('S8_sens_leave_one_out', rd('S8_leave_one_out_n122.csv')[['dropped', 'rho', 'delta', 'p']])
M = pd.read_csv(O + 'S14_mediator_corr_n122.csv', index_col=0).rename(index=SH, columns=SH).reset_index().rename(columns={'index': 'mediator'})
put('S14_med_mediator_corr', M)
V = rd('S19_vif_n122.csv'); V['mediator'] = V.mediator.map(SH); put('S19_med_vif', V)
put('S15_med_paths', rd('S15_med_paths_n122.csv')[['outcome', 'mediator', 'n', 'a', 'a_lo', 'a_hi', 'b', 'b_lo', 'b_hi', 'indirect', 'ind_lo', 'ind_hi', 'total', 'direct', 'prop']].rename(columns={'prop': 'prop_mediated'}))
put('S16_med_single_mediator', rd('S16_single_mediator_n122.csv')[['outcome', 'mediator', 'n', 'a', 'b', 'direct', 'indirect', 'ind_lo', 'ind_hi', 'total', 'prop']])
put('S17_med_alt_specs', rd('S17_alt_specs_n122.csv'))
put('S18_med_reversed_order', rd('S18_reversed_n122.csv'))
A10 = 'ASNS; ERO1A; TRIB3; PPP1R15A; ATF3; DDIT3; CHAC1; DDIT4; SESN2; ATF5'
s20 = rd('S20_sig_perk_n122.csv'); s20['genes'] = ['ASNS; ERO1A (protein)', A10 + ' (mRNA)', A10.replace('ASNS; ERO1A; ', '') + ' (mRNA)']
s20['scoring'] = 'mean of per-gene rank percentiles across all 122 tumors'; put('S20_sig_perk', s20)
s21 = rd('S21_sig_xbp1_n122.csv')
s21['genes'] = ['DNAJB9; SEL1L; HERPUD1; DNAJC3 (protein)', 'DNAJB9; SEL1L; HERPUD1; DNAJC3; EDEM1; HSPA5; PDIA6; HYOU1; SDF2L1; DNAJB11; PDIA5; MANF; CRELD2 (mRNA)', 'XBP1 (mRNA, total, not spliced)']
put('S21_sig_xbp1', s21)
b = rd('S22_bootstrap_leave_self_out_n122.csv'); b['is_seed'] = b.protein.isin(['SLC2A1', 'PGK1', 'LDHA', 'NDRG1', 'BNIP3', 'CA9', 'VEGFA'])
put('S22_hypoxia_bootstrap', b, 'Selection rule as in Methods 2.3, reapplied in each of 1000 resamples of the 122 tumors: purity-adjusted Spearman rho >= 0.30 with the seed score. For a seed protein the seed score is built from the other six seeds.')
p, m, ph, c = load()
BUFFA = ['VEGFA', 'SLC2A1', 'PGAM1', 'ENO1', 'LDHA', 'TPI1', 'ADM', 'NDRG1', 'ALDOA', 'MIF', 'GPI', 'DDIT4', 'BNIP3', 'HILPDA', 'CHCHD2', 'ANKRD37', 'PFKP', 'SLC16A1', 'HK2', 'CA9', 'PGK1', 'EGLN3']
bp = [x for x in BUFFA if x in p.columns and p[x].notna().sum() >= 60]
put('S23_buffa_comparison', rd('S23_buffa_n122.csv'), f'Buffa overlap ({len(bp)} proteins quantified in at least 60 tumors, P4HA1 excluded to avoid circularity): ' + '; '.join(bp) + '. First row: purity-adjusted agreement between the two scores.')
put('S24_med_perk_variants', rd('S24_perk_variants_n122.csv').rename(columns={'prop': 'prop_mediated'}))
pv = pd.read_csv(os.path.join(HERE, '..', 'out', 'preservation_permutation.csv')).drop(columns=['perm_p'], errors='ignore')
put('S26_preservation_perm', pv, 'percentile_of_observed: percentage of the 2000 tumor-label permutations whose statistic was below the observed value. lower_tail_p = (1 + permutations at or below observed) / (1 + permutations).')
ws = wb['S27_phospho_site_qc']; hdr = [x.value for x in ws[1]]; ca = hdr.index('regulatory_annotation') + 1; cs = hdr.index('site') + 1
ANN = {'SEC31A_S799s': 'CK2 site, characterized (Koreishi et al. 2013)', 'SEC31A_S527s': 'CK2 site, characterized (Koreishi et al. 2013)',
       'SEC31A_S527s_S532s': 'doubly phosphorylated peptide containing the characterized CK2 site S527 (Koreishi et al. 2013); not resolved to one residue',
       'SEC31A_S1163s': 'phosphorylation site identified by Koreishi et al. 2013 (C-terminal Sec23-binding region)'}
for r in range(2, ws.max_row + 1):
    sv = ws.cell(r, cs).value
    if sv in ANN: ws.cell(r, ca, ANN[sv])
    elif sv and sv.startswith(('MIA3', 'MIA2')): ws.cell(r, ca, 'not mapped to a characterized residue')
ws = wb['S30_plex_by_subtype']; t = rd('S30_chi2_three_groups.csv').iloc[0]; r0 = ws.max_row + 2
for k, (a, v) in enumerate([('Test on the three assigned groups (3 x 15 table; Normal-like tumors are not a group)', None), ('n', int(t.n)),
                            ('chi-square', round(t.chi2, 2)), ('df', int(t.dof)), ('asymptotic p (not relied on)', round(t.p_asymptotic, 3)),
                            ('Monte Carlo p, 5000 label permutations (reported test)', round(t.p_monte_carlo, 3)),
                            ('cells with expected count < 5 (%)', round(t.pct_cells_expected_lt5, 1))]):
    ws.cell(r0 + k, 1, a); ws.cell(r0 + k, 2, v)
if os.path.exists(O + 'S31_tcga_classifier.csv'):
    cl = rd('S31_tcga_classifier.csv'); cl.loc[cl.analysis.str.startswith('frozen'), ['bal_acc', 'sens', 'spec']] = np.nan
    put('S31_tcga_classifier', cl.rename(columns={'lo': 'auc_lo', 'hi': 'auc_hi'}), 'Transcript-level module scores. Frozen model: trained on all assigned CPTAC tumors, applied to TCGA unchanged; threshold metrics are not reported for it because scores are within-cohort percentiles.')
    put('S31b_tcga_associations', rd('S31b_tcga_associations.csv'), 'TCGA-BRCA, mRNA. Partial Spearman adjusted for the ESTIMATE score, proliferation and CAF content (n = 1082). Cliff delta on scores residualized on the ESTIMATE score.')
    cv = rd('S31a_tcga_module_coverage.csv'); cv['module'] = cv.module.replace({'Ferroptosis defence': 'Ferroptosis defense'})
    put('S31c_tcga_module_coverage', cv, 'Genes of each module available at the mRNA level in both CPTAC (>= 60 tumors) and TCGA.')
put('S32_tumor_stage', rd('S32_tumor_stage.csv'), 'CPTAC pathological stage, available for 107 of 122 tumors (15 missing). Scores are purity-adjusted. No grade or follow-up is provided in this release.')
ws = wb['S0_index']; desc = {r[0].value: r[1].value for r in ws.iter_rows(min_row=2) if r[0].value}
desc.update({
 'S6_sens_covariates': 'Principal associations under eight covariate specifications, all 122 tumors',
 'S8_sens_leave_one_out': 'Leave-one-protein-out results for the maturation-minus-deposition composite, all 122 tumors',
 'S15_med_paths': 'Mediation path coefficients with bootstrap intervals, parallel model, all 122 tumors',
 'S16_med_single_mediator': 'Single-mediator sensitivity models, all 122 tumors',
 'S17_med_alt_specs': 'Mediated fractions under alternative exposures and covariates, all 122 tumors',
 'S20_sig_perk': 'PERK/ATF4 protein module, ten-gene ATF4 target mRNA signature and eight-gene variant, with gene lists',
 'S21_sig_xbp1': 'IRE1/XBP1 protein module, 13-gene XBP1s target mRNA signature and total XBP1 mRNA, with gene lists',
 'S22_hypoxia_bootstrap': 'Bootstrap retention of each hypoxia candidate under the published selection rule (seed proteins tested against the other six seeds)',
 'S26_preservation_perm': 'Module preservation: observed value, permutation null, percentile of the observed value, lower-tail p and bootstrap interval',
 'S27_phospho_site_qc': 'Per-site residues, peptide multiplicity, sample size, missingness and regulatory annotation (SEC31A CK2 sites marked)',
 'S30_plex_by_subtype': 'Contingency table of clinical group by TMT plex, with the three-group test',
 'S31_tcga_classifier': 'Transcript-level replication of the classifier in TCGA-BRCA: frozen CPTAC model, CPTAC mRNA reference and within-TCGA nested cross-validation',
 'S31b_tcga_associations': 'TCGA-BRCA: hypoxia associations and subtype effect sizes at the mRNA level',
 'S31c_tcga_module_coverage': 'Genes of each module available in both cohorts at the mRNA level',
 'S32_tumor_stage': 'Hypoxia, composite and related module scores by pathological tumor stage (CPTAC)'})
ws.delete_rows(2, ws.max_row + 5)
for i, sname in enumerate([x for x in wb.sheetnames if x != 'S0_index'], 2):
    w = wb[sname]; ws.cell(i, 1, sname); ws.cell(i, 2, desc.get(sname, '')); ws.cell(i, 3, w.max_row - 1); ws.cell(i, 4, w.max_column)
wb.save(DST); print('saved', DST, len(wb.sheetnames), 'sheets')
