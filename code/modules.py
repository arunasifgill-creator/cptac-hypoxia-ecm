"""Module definitions for the 306-gene analysis (v3). Collagen enzymes are kept OUT of hypoxia scores
to avoid circularity, even though several (P4HA1/2, PLOD2, LOX) are HIF targets."""
from core import score, purity_adjust, resid, cliff, compare, bh, pspear  # noqa

PROT = {
 # --- ECM / collagen ---
 'Fibrillar collagen': ['COL1A1', 'COL1A2', 'COL3A1', 'COL5A1'],
 'Maturation': ['P4HA1', 'P4HA2', 'PLOD1', 'PLOD2', 'PLOD3', 'P3H1', 'SERPINH1', 'PCOLCE', 'BMP1', 'LOX', 'LOXL2'],
 'Collagen export': ['MIA3', 'MIA2', 'SEC23A', 'SEC24D', 'SAR1A', 'KLHL12', 'SEC31A', 'SEC13', 'TFG', 'PREB', 'SEC24A', 'SEC24B'],
 'ER-phagy / ERAD': ['SEC62', 'TEX264', 'CALCOCO1', 'SYVN1', 'OS9', 'ERLEC1'],
 'MMP turnover': ['MMP2', 'MMP9', 'MMP14'],
 'Mechanotransduction': ['ITGA1', 'ITGA5', 'ITGA11', 'ITGB5', 'PTK2', 'YAP1', 'WWTR1', 'CCN1'],
 # --- hypoxia ---
 'Hypoxia (core 7)': ['SLC2A1', 'PGK1', 'LDHA', 'NDRG1', 'BNIP3', 'CA9', 'VEGFA'],
 'Hypoxia: glycolysis': ['ALDOA', 'ALDOC', 'PGAM1', 'TPI1', 'GPI', 'PFKP', 'PGK1', 'LDHA', 'ENO1', 'HK2', 'PFKFB3', 'PDK1', 'SLC2A1', 'SLC16A3'],
 'Hypoxia: mitochondrial': ['NDUFA4L2', 'BNIP3', 'BNIP3L', 'FAM162A', 'AK4'],
 'Hypoxia: HIF targets': ['CA9', 'CA12', 'NDRG1', 'VEGFA', 'ANGPTL4', 'STC2', 'PLIN2', 'ENO2', 'MIF', 'ADM', 'EGLN3', 'EGLN1', 'BHLHE40', 'KDM3A', 'KDM4B', 'MXI1', 'ANKRD37'],
 # --- ER stress ---
 'UPR canonical chaperones': ['HSPA5', 'HYOU1', 'PDIA4', 'PDIA6', 'P4HB', 'CALR', 'CANX'],
 'UPR ATF6 targets': ['HSP90B1', 'DNAJB11', 'SDF2L1', 'CRELD2', 'MANF', 'PDIA5'],
 'UPR IRE1/XBP1 targets': ['DNAJB9', 'SEL1L', 'HERPUD1', 'DNAJC3'],
 'UPR PERK/ATF4 targets': ['ASNS', 'ERO1A'],
 'ER redox': ['ERO1B', 'PRDX4', 'GPX7', 'GPX8', 'QSOX1'],
 # --- antioxidant ---
 'NRF2 targets': ['NQO1', 'GCLC', 'GCLM', 'SRXN1', 'TXNRD1', 'AKR1C1', 'AKR1B10', 'GSR', 'SLC7A11'],
 'NADPH supply': ['G6PD', 'PGD', 'ME1', 'IDH1'],
 'Peroxide detox': ['CAT', 'GPX1', 'GPX3', 'PRDX1', 'PRDX2', 'PRDX3', 'PRDX5', 'PRDX6', 'TXN', 'SOD1', 'SOD2', 'GSTP1'],
 'Ferroptosis defence': ['GPX4', 'AIFM2', 'GCH1', 'DHODH'],
 # --- context ---
 'Proliferation': ['MKI67', 'TOP2A'],
 'CAF': ['FAP', 'PDGFRB', 'ACTA2', 'POSTN', 'DCN', 'LUM'],
 'TGF-b response': ['SERPINE1', 'TGFBI', 'CCN2', 'JUNB', 'TPM1', 'TAGLN', 'IGFBP3', 'FSTL3', 'COMP', 'LRRC15'],
}
# Broad hypoxia score used as the main exposure: union of the three hypoxia sub-modules
PROT['Hypoxia (expanded)'] = sorted(set(PROT['Hypoxia: glycolysis']) | set(PROT['Hypoxia: mitochondrial']) | set(PROT['Hypoxia: HIF targets']))

MRNA_EXTRA = {
 'UPR PERK/ATF4 targets': ['ASNS', 'ERO1A', 'TRIB3', 'PPP1R15A', 'ATF3', 'DDIT3', 'CHAC1', 'DDIT4', 'SESN2', 'ATF5'],
 'Hypoxia: HIF targets': PROT['Hypoxia: HIF targets'] + ['HILPDA', 'INSIG2', 'TMEM45A', 'EDN1', 'ADM'],
}

GROUP = {
 'ECM': ['Fibrillar collagen', 'Maturation', 'Collagen export', 'ER-phagy / ERAD', 'MMP turnover', 'Mechanotransduction'],
 'Hypoxia': ['Hypoxia: glycolysis', 'Hypoxia: mitochondrial', 'Hypoxia: HIF targets'],
 'ER stress': ['UPR canonical chaperones', 'UPR ATF6 targets', 'UPR IRE1/XBP1 targets', 'UPR PERK/ATF4 targets', 'ER redox'],
 'Antioxidant': ['NRF2 targets', 'NADPH supply', 'Peroxide detox', 'Ferroptosis defence'],
 'Context': ['Proliferation', 'CAF', 'TGF-b response'],
}


def all_scores(p):
    S = {k: score(p, v) for k, v in PROT.items()}
    S['Maturation - deposition'] = S['Maturation'] - S['Fibrillar collagen']
    return S

# Data-driven consensus hypoxia score (Buffa-style): candidates whose purity-adjusted protein level
# correlates rho >= 0.30 with the 7 canonical seed genes (label-free selection; CA12 and STC2 excluded
# as oestrogen-receptor targets, which correlate negatively with the seed).
PROT['Hypoxia consensus'] = ['PFKP', 'ANGPTL4', 'NDRG1', 'ENO1', 'CA9', 'GPI', 'SLC16A3', 'KDM3A', 'PLIN2', 'PGAM1',
                             'PGK1', 'EGLN1', 'HK2', 'PDK1', 'NDUFA4L2', 'VEGFA', 'ALDOC', 'LDHA', 'SLC2A1']
# Sensitivity version without glycolytic enzymes
PROT['Hypoxia non-glycolytic'] = ['ANGPTL4', 'NDRG1', 'CA9', 'KDM3A', 'PLIN2', 'EGLN1', 'NDUFA4L2', 'VEGFA']
