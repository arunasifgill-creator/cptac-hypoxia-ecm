"""Reproduce every analysis, table and figure in the manuscript.

Usage, from this directory:
    python3 run_all.py

Writes result tables to ../out, 600 dpi figures (PNG + PDF) to ../figures, and the
supplementary workbook to the path set at the top of make_supp_tables.py.

Expected runtime: roughly 2 to 3 hours. Almost all of it is in the cross-validated
classifiers (a3_ml.py and a6_ml_foldsafe.py, which refits every transformation inside
each training fold) and the permutation tests (a4_ml_extra.py, a11_robustness.py).
"""
import subprocess, sys, os, time

ANALYSES = [
    ('a1_modules.py',        'module scores, subtype effect sizes, hypoxia axis under three covariate sets'),
    ('a2_mediation.py',      'parallel mediation (Table 2)'),
    ('a3_ml.py',             'nested cross-validated classifiers'),
    ('a4_ml_extra.py',       'permutation test and incremental AUC'),
    ('a5_phospho.py',        'phosphosite and protein-adjusted abundance analysis (Figure 7)'),
    ('a6_ml_foldsafe.py',    'fold-safe classifiers with patient-level bootstrap (Figure 6)'),
    ('a7_coefs_foldsafe.py', 'fold-safe classifier coefficients (Figure 6C,D)'),
    ('a8_sensitivity.py',    'eight covariate specifications, stratified and leave-one-out'),
    ('a9_mediation_robust.py', 'path coefficients, VIF, single-mediator and reversed ordering'),
    ('a10_signatures.py',    'broader ATF4 and XBP1s signatures, hypoxia bootstrap, Buffa comparison'),
    ('a11_robustness.py',    'module-preservation permutation test and HER2 definition sensitivity'),
    ('a12_batch_qc.py',      'TMT plex by subtype, and every module score tested across plexes'),
    ('a13_phospho_qc.py',    'per-site coverage, missingness sensitivity and residue verification'),
]
FIGURES = [
    ('fig_core.py',       'Figure 1'),
    ('figs_final.py',     'Figures 2, 4, S2, S3'),
    ('fig_v3.py',         'Figure 3'),
    ('fig_cluster.py',    'Figure 5'),
    ('fig_ml.py',         'Figure 6'),
    ('fig_phospho.py',    'Figure 7'),
    ('fig8_summary.py',   'Figure 8'),
    ('fig_s1_cohort.py',  'Figure S1'),
    ('fig_supp_large.py', 'Figures S4, S5, S6'),
]
TABLES = [
    ('make_supp_tables.py', 'Supplementary_Tables.xlsx, 30 sheets'),
]

def run(stage, items):
    print(f'\n{"="*68}\n{stage}\n{"="*68}')
    for script, what in items:
        print(f'\n--- {script}  ({what})')
        t0 = time.time()
        subprocess.run([sys.executable, script], check=True)
        print(f'--- done in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for p in ('../out', '../figures'):
        os.makedirs(p, exist_ok=True)
    run('ANALYSIS', ANALYSES)
    run('FIGURES', FIGURES)
    run('TABLES', TABLES)
    print('\nAll analyses, figures and tables regenerated.')
    print('  tables  -> ../out')
    print('  figures -> ../figures')
