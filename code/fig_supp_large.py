"""Full-page versions of the three dense panels, added at revision because reviewer 2 (point 17)
could not read them at the size they occupy in the main figures.

  Figure S4  the co-expression matrix of Figure 5A, one page, with readable gene labels
  Figure S5  the classifier coefficients of Figure 6C,D, one page, every feature labelled
  Figure S6  every export-machinery phosphosite of Figure 7A, with n and BH q printed per site
"""
import os, re, numpy as np, pandas as pd, matplotlib, warnings; warnings.filterwarnings('ignore')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
INK, INK2, GRID = '#1f1f1e', '#5f5e5a', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': .6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 600})
DIV = LinearSegmentedColormap.from_list('bwr', ['#1c5cab', '#2a78d6', '#9ec5f4', '#f0efec', '#eda7a0', '#c0392b', '#8c2317'])


def save(fig, n):
    for e in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{n}.{e}', bbox_inches='tight', facecolor='white')
    plt.close(fig)


MOD = {'Fibrillar collagen': ['COL1A1', 'COL1A2', 'COL3A1', 'COL5A1'],
       'Maturation / modification': ['PLOD1', 'PLOD2', 'PLOD3', 'P4HA1', 'P4HA2', 'SERPINH1', 'P3H1', 'PPIB', 'CRTAP', 'FKBP10', 'COLGALT1'],
       'Crosslink & processing': ['LOX', 'LOXL2', 'BMP1', 'PCOLCE'],
       'ER export': ['MIA3', 'MIA2', 'SEC23A', 'SEC24D'],
       'UPR chaperones': ['HSPA5', 'P4HB', 'PDIA4', 'PDIA6', 'HYOU1', 'CALR', 'CANX'],
       'UPR targets': ['HSP90B1', 'SDF2L1', 'MANF', 'ASNS', 'ERO1A'],
       'Hypoxia': ['SLC2A1', 'PGK1', 'LDHA', 'NDRG1', 'CA9'],
       'Iron': ['SLC11A2', 'TFRC', 'STEAP3', 'FTH1', 'FTL'],
       'MMP turnover': ['MMP2', 'MMP9', 'MMP14']}
gene2mod = {x: k for k, v in MOD.items() for x in v}
MCOL = {'Fibrillar collagen': '#1baf7a', 'Maturation / modification': '#a63d12', 'Crosslink & processing': '#eda100',
        'ER export': '#008300', 'UPR chaperones': '#2a78d6', 'UPR targets': '#1c5cab', 'Hypoxia': '#e87ba4',
        'Iron': '#eb6834', 'MMP turnover': '#5f5e5a'}

# ---------------- Figure S4: the matrix, full page ----------------
Rc = pd.read_csv(f'{OUT}/Fig5_correlation_matrix.csv', index_col=0)
gl = list(Rc.index)
fig, ax = plt.subplots(figsize=(7.4, 7.9))
im = ax.imshow(Rc.values, cmap=DIV, vmin=-1, vmax=1, interpolation='nearest')
ax.set_xticks(range(len(gl))); ax.set_yticks(range(len(gl)))
ax.set_xticklabels(gl, rotation=90, fontsize=7.2); ax.set_yticklabels(gl, fontsize=7.2)
for t, gn in zip(ax.get_xticklabels(), gl): t.set_color(MCOL[gene2mod[gn]])
for t, gn in zip(ax.get_yticklabels(), gl): t.set_color(MCOL[gene2mod[gn]])
ax.set_xticks(np.arange(-.5, len(gl), 1), minor=True); ax.set_yticks(np.arange(-.5, len(gl), 1), minor=True)
ax.grid(which='minor', color='white', lw=.35); ax.tick_params(which='minor', length=0)
ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=.026, pad=.015, ticks=[-1, -.5, 0, .5, 1])
cb.set_label('Spearman ρ', fontsize=8); cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MCOL[k], label=k) for k in MOD],
          loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=5, fontsize=7, columnspacing=1.2, handlelength=1.1)
ax.set_title(f'Co-expression architecture: {len(gl)} proteins, adjusted for purity, n = 122 tumors\n'
             'Average-linkage clustering on (1 − ρ) distance. Same data as Figure 5A, drawn full page.', pad=40, fontsize=9)
save(fig, 'FigS4_correlation_matrix_large')
print('S4 done,', len(gl), 'proteins')

# ---------------- Figure S5: classifier coefficients, full page ----------------
C = pd.read_csv('../out/ml_coefs_foldsafe.csv')
col = ['task']; tasks = list(dict.fromkeys(C['task']))
mcol, scol, fcol = 'coef', ['sd'], 'module'
fig, axs = plt.subplots(1, len(tasks), figsize=(7.4, 7.2), sharex=True)
axs = np.atleast_1d(axs)
lab = {'UPR canonical chaperones': 'Chaperones', 'UPR ATF6 targets': 'ATF6 targets',
       'UPR IRE1/XBP1 targets': 'IRE1/XBP1 targets', 'UPR PERK/ATF4 targets': 'PERK/ATF4 targets',
       'Maturation - deposition': 'Maturation − deposition', 'TGF-b response': 'TGF-β response'}
for ax, t in zip(axs, tasks):
    D = C[C['task'] == t]
    D = D.reindex(D[mcol].abs().sort_values().index)
    cc = ['#a63d12' if str(f).startswith(('Fibrillar', 'Maturation', 'Collagen', 'MMP', 'Mechano', 'ER-phagy'))
          else '#2a78d6' if str(f).startswith(('UPR', 'ER redox'))
          else '#e87ba4' if str(f).startswith('Hypoxia')
          else '#1baf7a' if str(f) in ('NRF2 targets', 'NADPH supply', 'Peroxide detox', 'Ferroptosis defence')
          else '#8f8e88' for f in D[fcol]]
    ax.barh(range(len(D)), D[mcol], color=cc, height=.7)
    if scol:
        ax.errorbar(D[mcol], range(len(D)), xerr=D[scol[0]], fmt='none', ecolor=INK2, elinewidth=.7, capsize=1.6)
    ax.set_yticks(range(len(D))); ax.set_yticklabels([lab.get(str(f), str(f)) for f in D[fcol]], fontsize=7)
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.axvline(0, color=INK2, lw=.7)
    ax.set_title(f'{"AB"[list(tasks).index(t)]}  {t}', fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=.5); ax.set_axisbelow(True)
axs[0].set_xlabel('Standardized coefficient (mean ± SD across outer folds)')
fig.suptitle('Classifier coefficients, drawn full page. Same models as Figure 6C,D; every feature is labelled.\n'
             'All transformations were fitted inside training folds only.', x=.02, ha='left', fontsize=9, y=1.015)
fig.subplots_adjust(wspace=.75)
save(fig, 'FigS5_classifier_coefficients_large')
print('S5 done,', len(tasks), 'tasks')

# ---------------- Figure S6: every export phosphosite, with n and q ----------------
X = pd.read_csv('../out/phospho_occupancy.csv')
E = X[X.gene.isin(['MIA3', 'MIA2', 'SEC31A', 'SEC23A', 'SEC24D', 'SAR1A', 'TFG', 'PREB'])].copy()


def pos(site):
    mm = re.findall(r'[STY](\d+)', site.split('_', 1)[1])
    return int(mm[0]) if mm else 0


E['pos'] = E.site.map(pos)
E['label'] = [s.split('_', 1)[1].replace('s', '').replace('t', '').replace('y', '').replace('_', '/') for s in E.site]
E = E.sort_values(['gene', 'pos'])
gname = {'MIA3': 'MIA3 (TANGO1)', 'MIA2': 'MIA2 (cTAGE5/TALI)', 'SEC31A': 'SEC31A', 'SEC23A': 'SEC23A',
         'SEC24D': 'SEC24D', 'SAR1A': 'SAR1A', 'TFG': 'TFG', 'PREB': 'PREB'}
gcol = {'MIA3': '#a63d12', 'MIA2': '#eda100', 'SEC31A': '#2a78d6', 'SEC23A': '#1baf7a', 'SEC24D': '#008300',
        'SAR1A': '#e87ba4', 'TFG': '#5f5e5a', 'PREB': '#8f8e88'}
fig, ax = plt.subplots(figsize=(7.4, max(5.0, .20 * len(E) + 1.6)))
y = np.arange(len(E))[::-1]
ax.barh(y, E.rho_hyp_occ, color=[gcol[g] for g in E.gene], height=.68,
        alpha=1.0, edgecolor='none')
for yy, (_, r) in zip(y, E.iterrows()):
    if r.q_occ >= .05:
        ax.barh(yy, r.rho_hyp_occ, color='white', height=.68, alpha=.55, edgecolor='none')
    xt = r.rho_hyp_occ + (.015 if r.rho_hyp_occ >= 0 else -.015)
    q = r.q_occ
    qs = 'q < 0.001' if q < .001 else f'q = {q:.3f}' if q < .01 else f'q = {q:.2f}'
    ax.text(xt, yy, f'{r.rho_hyp_occ:+.2f}  ({qs}, n = {int(r.n)})', va='center',
            ha='left' if r.rho_hyp_occ >= 0 else 'right', fontsize=5.8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([f'{gname[r.gene]}  {r.label}' for _, r in E.iterrows()], fontsize=6.2)
for t, gn in zip(ax.get_yticklabels(), E.gene): t.set_color(gcol[gn])
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.axvline(0, color=INK2, lw=.7); ax.set_xlim(-.75, .95)
ax.set_xlabel('Partial Spearman ρ of site occupancy with the hypoxia consensus score\n'
              'Occupancy adjusted for total protein and purity; correlation further adjusted for proliferation')
ax.set_title(f'Every collagen-export phosphosite quantified in the cohort ({len(E)} sites on {E.gene.nunique()} proteins)\n'
             f'Pale bars have BH q ≥ 0.05 across the {len(X)} sites tested. Residue numbering follows the CPTAC reference sequence.',
             fontsize=9)
ax.xaxis.grid(True, color=GRID, lw=.5); ax.set_axisbelow(True)
save(fig, 'FigS6_export_phosphosites_large')
print('S6 done,', len(E), 'sites,', E.gene.nunique(), 'proteins; site n range', int(E.n.min()), '-', int(E.n.max()))
