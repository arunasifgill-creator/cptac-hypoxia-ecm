# Direct and ER Stress-Associated Components of the Hypoxia-Collagen Relationship in Breast Cancer

Analysis code and data for:

> **Direct and ER Stress-Associated Components of the Hypoxia-Collagen Relationship in Breast Cancer: An Interpretable Proteogenomic and Machine Learning Analysis**
> Arun Asif, Zeeshan, Jaeyoung Choi. *Genes* (under revision).

Every figure, table and number from the CPTAC analysis is regenerated from the raw exports in `data/` by a single command. The transcript-level replication in TCGA-BRCA (Section 3.13) draws on a separate cohort and is not reproduced by this repository.

---

## What the study does

The collagen-modifying enzymes sit inside the endoplasmic reticulum, which is also where hypoxia triggers the unfolded protein response. That raises a question nobody had tested in patient tissue at the protein level: is the association between hypoxia and collagen biosynthesis mediated by ER stress, or not?

This repository answers it two independent ways on 122 treatment-naive primary breast tumors:

1. **Parallel mediation.** A hypoxia score is the exposure, ER stress modules are competing mediators, collagen enzymes are the outcome. Bootstrap confidence intervals separate the component mediated by the measured ER stress modules from the component that is not.
2. **Interpretable classifiers.** Nested cross-validation asks whether ER stress features carry any information that hypoxia and ECM features do not.

The two methods agree that the component not mediated by the measured ER stress modules is the larger of the two. The size of the mediated component is unresolved, since it is 28 to 42% with the protein modules and falls to 4 to 9% for the PERK/ATF4 arm when ATF4 output is measured more specifically. The classifiers find no information in ER stress features beyond hypoxia and ECM features for TNBC against all other tumors, though ECM features do add information against HER2+ tumors.

These are cross-sectional associations between protein abundances. They do not establish causal direction and they do not measure collagen maturation or secretion.

---

## Quick start

```bash
git clone https://github.com/arunasifgill-creator/cptac-hypoxia-ecm.git
cd cptac-hypoxia-ecm
pip install -r requirements.txt
cd code
python3 run_all.py
```

**Runtime is roughly 2 to 3 hours on a laptop**, almost all of it in the cross-validated classifiers (`a3_ml.py` and `a6_ml_foldsafe.py`, which refits every transformation inside each training fold) and the permutation tests (`a4_ml_extra.py`, `a11_robustness.py`). The figure scripts take seconds. To regenerate only the figures from result tables already in `out/`, run the `fig_*.py` scripts directly.

Result tables land in `out/`, figures in `figures/` as 600 dpi PNG and vector PDF.

---

## Layout

```
code/
  core.py            shared statistics: purity adjustment, rank residualization,
                     Cliff's delta, Benjamini-Hochberg, partial Spearman
  load.py            merges the three cBioPortal exports into 318-gene tables
  modules.py         the gene panel, grouped into biological modules
  a1_modules.py      module scores, subtype effect sizes, the hypoxia axis
  a2_mediation.py    parallel multiple-mediator model (Table 2)
  a3_ml.py           nested cross-validated classifiers
  a4_ml_extra.py     permutation test, incremental AUC over proliferation
  a5_phospho.py      phosphosite and protein-adjusted abundance analysis

  revision analyses added in response to peer review:
  a6_ml_foldsafe.py       classifiers with every transformation fitted inside the
                          training folds, plus patient-level bootstrap intervals
  a7_coefs_foldsafe.py    fold-safe classifier coefficients
  a8_sensitivity.py       eight covariate specifications, stratified, leave-one-out
  a9_mediation_robust.py  path coefficients, VIF, single-mediator, reversed ordering
  a10_signatures.py       broader ATF4 and XBP1s signatures, hypoxia signature
                          bootstrap, comparison with the external Buffa signature
  a11_robustness.py       tumour-level permutation test for module preservation,
                          and sensitivity to the HER2+ group definition
  a12_batch_qc.py         subtype by TMT plex, and every module score tested across
                          plexes with BH correction
  a13_phospho_qc.py       per-site coverage, missingness sensitivity, residue
                          verification and regulatory annotation
  make_supp_tables.py     builds the 30-sheet supplementary workbook

  fig_*.py           one script per figure
  run_all.py         runs all of the above in order
data/                raw cBioPortal and CPTAC exports (see Data sources)
out/                 result tables written by the analysis scripts
figures/             600 dpi PNG + PDF, plus the source values behind each panel
```

---

## Figures

| Figure | Script | Content |
|---|---|---|
| 1 | `fig_core.py` | Collagen deposition falls while modifying enzymes rise in TNBC |
| 2 | `figs_final.py` | Pathway-wide effect sizes across all modules |
| 3 | `fig_v3.py` | Hypoxia axis and the mediation decomposition |
| 4 | `figs_final.py` | Collagen export, TGF-beta response and ER redox |
| 5 | `fig_cluster.py` | Correlation architecture of the gene panel |
| 6 | `fig_ml.py` | Classifier performance by feature set |
| 7 | `fig_phospho.py` | Phosphosite abundance along the export pathway |
| 8 | `fig8_summary.py` | Summary schematic of the relationships examined |
| S1 | `fig_s1_cohort.py` | Cohort composition, data quality and batch structure |
| S2 | `figs_final.py` | Hormone receptor subgroups |
| S3 | `figs_final.py` | Iron handling proteins along the hypoxia axis |
| S4 | `fig_supp_large.py` | Correlation matrix of Figure 5A at full page |
| S5 | `fig_supp_large.py` | Classifier coefficients of Figure 6C,D at full page |
| S6 | `fig_supp_large.py` | Every collagen-export phosphosite with n and adjusted q |

Each figure also writes a CSV of the values behind its panels, so any number in the paper can be traced without rerunning the analysis.

---

## Methods in brief

- **Panel.** 318 genes queried; 249 quantified at the protein level in at least 80% of tumors, 21 partially, 48 not at all.
- **Cohort.** 122 treatment-naive primary tumors from CPTAC Breast Cancer 2020 with matched TMT proteomics, phosphoproteomics and RNA-seq. Five PAM50 Normal-like tumors are excluded. Groups: TNBC (n = 28), HER2+ (n = 15), HR+/HER2- (n = 74).
- **Purity adjustment.** Module scores are rank-residualized on ESTIMATE tumor purity. Correlations additionally adjust for proliferation and CAF content, so associations are not driven by stromal fraction or growth rate.
- **Hypoxia score.** A 19-protein consensus signature selected without using subtype labels: candidates correlating at rho >= 0.30 with seven canonical hypoxia seeds. Building it label-free keeps the subtype comparison from being circular.
- **Effect sizes.** Cliff's delta with bootstrap confidence intervals; |delta| >= 0.474 is treated as a large effect. Two-sided Mann-Whitney U tests, Benjamini-Hochberg across each family.
- **Mediation.** Parallel multiple-mediator model with 5000 bootstrap percentile intervals, all variables rank-residualized then standardized.
- **Machine learning.** L2 logistic regression and random forests, nested stratified 5-fold cross-validation repeated 20 times, so hyperparameter tuning never sees the held-out fold. Tested against 200 label permutations. Every transformation, including the percentile mapping, module scoring and purity residualization, is fitted inside each outer training split, so no held-out tumor contributes to any transformation. Uncertainty is reported two ways: a percentile range across cross-validation repeats, and a patient-level bootstrap interval, which is the wider and more honest of the two.
- **Phosphosites.** Protein-adjusted phosphosite abundance is each site residualized on its own total protein and on purity, which separates a phosphorylation shift from a change in protein abundance. This is a ratio, not a measure of stoichiometry, and we do not call it occupancy.
- **Multiple testing.** Benjamini-Hochberg is applied within explicitly defined families and never across the whole study. The families are listed in the Methods of the paper and each figure panel states which one it belongs to.
- **Module preservation.** Agreement between within-subtype correlation matrices is tested against a tumor-label permutation null, not against a null of zero correlation, because the gene pairs share genes and are not independent.

Collagen-modifying enzymes are deliberately kept out of the hypoxia scores even though several are HIF targets, to avoid building the result into the exposure.

---

## Data sources

All data are public. No new patient data were generated.

- Protein, mRNA and phosphoprotein tables: [cBioPortal](https://www.cbioportal.org/), study `brca_cptac_2020`, exported as transposed matrices.
- Clinical annotations including ESTIMATE purity and PAM50 calls: the same study.
- Primary publication for the cohort: Krug et al., *Cell* 2020, [10.1016/j.cell.2020.10.036](https://doi.org/10.1016/j.cell.2020.10.036).
- CPTAC Data Portal: https://proteomics.cancer.gov/

The exports in `data/` are included verbatim so the analysis runs without any manual download step.

---

## Reproducibility notes

- Random seeds are fixed in the cross-validation and bootstrap routines, so repeated runs give identical numbers.
- `load.py` maps the deprecated gene symbol `COXFA4L2` back to `NDUFA4L2`, which is how cBioPortal currently exports it.
- The phosphosite residue numbering follows CPTAC's own annotation. Confirm against the primary literature before drawing conclusions about a specific site.
- The `MIA3` locus encodes TANGO1 and the shorter TANGO1-Short, and the `MIA2` locus encodes cTAGE5 and the longer TALI. The long and short products share their C-terminal regions, so a tryptic peptide from that region cannot be assigned to one product, and a residue number is only meaningful against a named sequence. The code and the paper use the HGNC gene symbols throughout for this reason.
- `out/phospho_site_qc.csv` annotates every site with its residues, peptide multiplicity, sample size, missingness, and whether it has a characterized regulatory role in the literature.

## Citation

If you use this code, please cite the paper above. The code itself is MIT licensed.

## Contact

Arun Asif, Department of Molecular and Cellular Physiology, Albany Medical College: asifa1@amc.edu
