"""Reproduce every analysis, table and figure in the manuscript.

Usage, from this directory:
    python3 run_all.py

Writes result tables to ../out and 600 dpi figures (PNG + PDF) to ../figures.
Expected runtime: roughly 45-90 minutes, almost all of it in a3_ml.py (nested
    cross-validation) and a4_ml_extra.py (200-label permutation test).
"""
import subprocess, sys, os, time

ANALYSES = [
    ('a1_modules.py',   'module scores, subtype effect sizes, hypoxia axis'),
    ('a2_mediation.py', 'parallel mediation (Table 2)'),
    ('a3_ml.py',        'nested cross-validated classifiers (Figure 6)'),
    ('a4_ml_extra.py',  'permutation test and incremental AUC'),
    ('a5_phospho.py',   'phosphosite and occupancy analysis (Figure 7)'),
]
FIGURES = [
    ('fig_s1_cohort.py', 'Figure S1'),
    ('fig_core.py',      'Figure 1'),
    ('figs_final.py',    'Figures 2, 4, S2, S3'),
    ('fig_v3.py',        'Figure 3'),
    ('fig_cluster.py',   'Figure 5'),
    ('fig_ml.py',        'Figure 6'),
    ('fig_phospho.py',   'Figure 7'),
]

def run(stage, items):
    print(f'\n{"="*64}\n{stage}\n{"="*64}')
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
    print('\nAll analyses and figures regenerated.')
    print('  tables  -> ../out')
    print('  figures -> ../figures')
