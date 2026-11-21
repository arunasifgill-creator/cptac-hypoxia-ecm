"""Figure 8. Summary schematic of the relationships examined in this study.
Addresses reviewer comment R3-7."""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID = '#1f1f1e', '#5f5e5a', '#e6e5e0'
TN, UP, DN = '#eb6834', '#c2381a', '#2a6fb0'
HYP, ERS, ECM = '#f0a500', '#8e6bbf', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': INK})

fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.34], hspace=0.04)

def box(ax, x, y, w, h, label, fc, ec=None, fs=8, bold=False, tc=None, pad=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad={pad},rounding_size=0.02',
                                fc=fc, ec=ec or INK2, lw=0.8, zorder=3))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fs, zorder=4,
            fontweight='bold' if bold else 'normal', color=tc or INK, linespacing=1.35)

def arrow(ax, p1, p2, color=INK2, lw=1.3, style='-|>', ls='-', rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11, lw=lw,
                                 color=color, linestyle=ls, zorder=z,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))

# ================= PANEL A: reciprocal model =================
axA = fig.add_subplot(gs[0]); axA.set_xlim(0, 10); axA.set_ylim(0, 5.0); axA.axis('off')
axA.text(-0.1, 4.90, 'A', fontsize=12, fontweight='bold', va='top')
axA.text(0.42, 4.78, 'A reciprocal model, not a sequence', fontsize=9, fontweight='bold', va='top', color=INK)

box(axA, 0.55, 3.05, 2.25, 0.95, 'Tumor\nhypoxia', '#fdf0d0', HYP, fs=9, bold=True)
box(axA, 0.55, 1.05, 2.25, 0.95, 'ER stress\nPERK/ATF4, ER redox', '#efe7f8', ERS, fs=7.8, bold=False)

# mutual activation
arrow(axA, (1.35, 3.02), (1.35, 2.03), color=INK2, lw=1.5)
arrow(axA, (2.00, 2.03), (2.00, 3.02), color=INK2, lw=1.5)
axA.text(2.98, 2.52, 'each activates\nthe other', fontsize=7.0, color=INK2, va='center', ha='left', linespacing=1.4)

box(axA, 6.35, 1.95, 3.15, 1.25,
    'Shift in the collagen program\nmore processing enzymes\nless deposited collagen\nlower export capacity',
    '#e2f5ec', ECM, fs=7.6)

arrow(axA, (2.80, 3.40), (6.35, 2.95), color=INK, lw=2.4, rad=0.05)
axA.text(4.55, 3.62, 'dominant component', ha='center', fontsize=7.8, color=INK, fontweight='bold')
arrow(axA, (2.80, 1.45), (6.35, 2.25), color=ERS, lw=1.2, ls=(0,(4,2)), rad=-0.05)
axA.text(4.55, 1.38, 'smaller, magnitude unresolved', ha='center', fontsize=7.0, color=ERS)

axA.add_patch(Rectangle((0.3, 0.10), 9.4, 0.72, fc='#faf9f6', ec=GRID, lw=0.8, zorder=1))
axA.text(0.52, 0.46, 'Cross-sectional associations in all 122 tumors. Fitting the model in either orientation gives a comparable fit, '
                     'so the\ndirection is not identified; the arrows mark the relationships tested, not a demonstrated sequence.',
         fontsize=6.9, va='center', color=INK2, linespacing=1.5)

# ================= PANEL B: molecular map =================
axB = fig.add_subplot(gs[1]); axB.set_xlim(0, 10); axB.set_ylim(0.14, 6.05); axB.axis('off')
axB.text(-0.1, 5.98, 'B', fontsize=12, fontweight='bold', va='top')
axB.text(0.42, 5.86, 'Where the measured proteins sit', fontsize=9, fontweight='bold', va='top')

# cell body
axB.add_patch(FancyBboxPatch((0.3, 0.30), 6.55, 5.05, boxstyle='round,pad=0.02,rounding_size=0.12',
                             fc='#fcfcfa', ec=INK2, lw=1.0, zorder=1))
axB.text(2.45, 5.12, 'Tumor cell', fontsize=7.6, color=INK2, style='italic')
# ER
axB.add_patch(FancyBboxPatch((0.62, 1.52), 5.95, 3.30, boxstyle='round,pad=0.02,rounding_size=0.10',
                             fc='#f4f1fa', ec=ERS, lw=1.0, zorder=2))
axB.text(5.78, 1.76, 'Endoplasmic reticulum', fontsize=7.6, color=ERS, style='italic', zorder=4)

box(axB, 0.95, 3.55, 1.72, 0.78, 'Procollagen\nsynthesis', '#ffffff', INK2, fs=7.4)
box(axB, 3.05, 3.55, 1.95, 0.78, 'Hydroxylation\nP4HA1/2  PLOD1-3', '#e2f5ec', ECM, fs=7.4)
box(axB, 5.28, 3.55, 1.10, 0.78, 'Folding\nSERPINH1', '#e2f5ec', ECM, fs=7.4)
arrow(axB, (2.67, 3.94), (3.05, 3.94)); arrow(axB, (5.00, 3.94), (5.28, 3.94))

box(axB, 0.95, 2.18, 2.30, 0.80, 'ER redox\nPRDX4  GPX7/8  ERO1A', '#efe7f8', ERS, fs=7.2)
box(axB, 3.62, 2.18, 2.00, 0.80, 'UPR\nPERK/ATF4  ATF6\nIRE1/XBP1', '#efe7f8', ERS, fs=7.2)
arrow(axB, (2.10, 2.98), (2.10, 3.55), color=ERS, lw=1.2, ls=(0,(3,2)))
arrow(axB, (4.62, 2.98), (4.20, 3.55), color=ERS, lw=1.2, ls=(0,(3,2)))

# export
box(axB, 2.52, 0.60, 2.98, 0.92, 'COPII export\nMIA3 (TANGO1)  MIA2 (cTAGE5)\nSEC31A', '#eaf1fb', DN, fs=7.0)
arrow(axB, (4.01, 3.55), (4.01, 1.52), color=INK2, lw=1.2)

# ECM compartment
axB.add_patch(FancyBboxPatch((7.15, 0.30), 2.70, 5.05, boxstyle='round,pad=0.02,rounding_size=0.12',
                             fc='#f3fbf7', ec=ECM, lw=1.0, zorder=1))
axB.text(7.36, 5.12, 'Extracellular matrix', fontsize=7.6, color=ECM, style='italic')
box(axB, 7.45, 3.40, 2.10, 0.95, 'Deposited\nfibrillar collagen\nCOL1A1/1A2/3A1', '#ffffff', ECM, fs=7.2)
box(axB, 7.45, 1.95, 2.10, 0.80, 'Crosslinking\nLOX  LOXL2', '#ffffff', ECM, fs=7.2)
arrow(axB, (5.52, 1.06), (8.50, 1.06), color=INK2, lw=1.2)
arrow(axB, (8.50, 1.06), (8.50, 1.95), color=INK2, lw=1.2)
arrow(axB, (8.50, 2.75), (8.50, 3.40), color=INK2, lw=1.2)

# hypoxia input
axB.add_patch(Ellipse((1.02, 5.03), 1.18, 0.46, fc='#fdf0d0', ec=HYP, lw=1.0, zorder=5))
axB.text(1.02, 5.03, 'Hypoxia', ha='center', va='center', fontsize=7.6, fontweight='bold', zorder=6)
arrow(axB, (1.56, 4.88), (3.62, 4.33), color=HYP, lw=1.6, rad=-0.12, z=5)
arrow(axB, (1.00, 4.80), (1.70, 2.98), color=HYP, lw=1.4, rad=0.10, z=5)

# direction annotations in TNBC
def tag(x, y, txt, col):
    axB.text(x, y, txt, fontsize=7.0, color=col, fontweight='bold', ha='center', va='center', zorder=7,
             bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=col, lw=0.7))
tag(4.02, 4.58, 'higher in TNBC', UP)
tag(2.10, 1.90, 'higher in TNBC', UP)
tag(1.48, 1.06, 'lower in TNBC', DN)
tag(8.50, 4.62, 'lower in TNBC', DN)

axB.text(4.01, 0.44, 'phosphosite shifts: TANGO1, SEC31A', fontsize=6.6, color=DN, style='italic', ha='center')

fig.text(0.5, -0.012,
         'Boxes mark protein modules measured here. Arrows show the direction of the relationships tested, '
         'not demonstrated flux.\nDirection labels refer to purity-adjusted differences between TNBC and the other subtypes.',
         ha='center', fontsize=6.8, color=INK2, linespacing=1.5)

for e in ('png', 'pdf'):
    fig.savefig(f'{OUT}/Fig8_summary.{e}', dpi=600, bbox_inches='tight', facecolor='white')
print('saved Fig8_summary.png / .pdf')
