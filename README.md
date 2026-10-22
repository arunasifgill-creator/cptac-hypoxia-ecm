# Hypoxia and collagen biosynthesis in breast cancer (CPTAC BRCA 2020)

Analysis code and data for:

> **Hypoxia Reprograms Collagen Biosynthesis in Breast Cancer through Direct and ER Stress-Mediated Routes: An Interpretable Proteogenomic and Machine Learning Analysis**
> Arun Asif, Zeeshan, Jaeyoung Cho. *Genes* (submitted).

Everything in the paper, every figure, every table and every number quoted in the text, is regenerated from the raw exports in `data/` by a single command.

---

## What the study does

The collagen-modifying enzymes sit inside the endoplasmic reticulum, which is also where hypoxia triggers the unfolded protein response. That raises a question nobody had tested in patient tissue at the protein level: does hypoxia reshape collagen biosynthesis directly, or does it work through ER stress?

This repository answers it two independent ways on 122 treatment-naive primary breast tumors:

1. **Parallel mediation.** A hypoxia score is the exposure, ER stress modules are competing mediators, collagen enzymes are the outcome. Bootstrap confidence intervals separate the direct route from the mediated one.
2. **Interpretable classifiers.** Nested cross-validation asks whether ER stress features carry any information that hypoxia and ECM features do not.

The two methods agree, which is the point: most of the effect is direct, and ER stress is largely redundant once hypoxia is known.

---

## Quick start

```bash
git clone https://github.com/arunasifgill-creator/cptac-hypoxia-ecm.git
cd cptac-hypoxia-ecm
pip install -r requirements.txt
cd code
python3 run_all.py
```

**Runtime is roughly 45 to 90 minutes on a laptop**, almost all of it in the nested cross-validation (`a3_ml.py`, about 25 minutes) and the 200-label permutation test (`a4_ml_extra.py`). The figure scripts take seconds. To regenerate only the figures from result tables already in `out/`, run the `fig_*.py` scripts directly.

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
  a5_phospho.py      phosphosite and occupancy analysis
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
| 7 | `fig_phospho.py` | Phosphosite occupancy along the export route |
| S1 | `fig_s1_cohort.py` | Cohort composition and data quality |
| S2 | `figs_final.py` | Hormone receptor subgroups |
| S3 | `figs_final.py` | Iron handling proteins along the hypoxia axis |

Each figure also writes a CSV of the values behind its panels, so any number in the paper can be traced without rerunning the analysis.

---

## Methods in brief

- **Panel.** 318 genes queried; 249 quantified at the protein level in at least 80% of tumors, 21 partially, 48 not at all.
- **Cohort.** 122 treatment-naive primary tumors from CPTAC Breast Cancer 2020 with matched TMT proteomics, phosphoproteomics and RNA-seq. Five PAM50 Normal-like tumors are excluded. Groups: TNBC (n = 28), HER2+ (n = 15), HR+/HER2- (n = 74).
- **Purity adjustment.** Module scores are rank-residualized on ESTIMATE tumor purity. Correlations additionally adjust for proliferation and CAF content, so associations are not driven by stromal fraction or growth rate.
- **Hypoxia score.** A 19-protein consensus signature selected without using subtype labels: candidates correlating at rho >= 0.30 with seven canonical hypoxia seeds. Building it label-free keeps the subtype comparison from being circular.
- **Effect sizes.** Cliff's delta with bootstrap confidence intervals; |delta| >= 0.474 is treated as a large effect. Two-sided Mann-Whitney U tests, Benjamini-Hochberg across each family.
- **Mediation.** Parallel multiple-mediator model with 5000 bootstrap percentile intervals, all variables rank-residualized then standardized.
- **Machine learning.** L2 logistic regression and random forests, nested stratified 5-fold cross-validation repeated 20 times, so hyperparameter tuning never sees the held-out fold. Tested against 200 label permutations.
- **Phosphosites.** Occupancy is each site residualized on its own total protein and on purity, which separates a genuine phosphorylation shift from a change in protein abundance.

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

## Citation

If you use this code, please cite the paper above. The code itself is MIT licensed.

## Contact

Arun Asif, Department of Molecular and Cellular Physiology, Albany Medical College: asifa1@amc.edu
