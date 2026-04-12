# Liposome-Water Partitioning: Mechanistic Fragment-Based QSPR Model

This repository contains the data and code for the study:

> **SORPTION OF ORGANIC MICROPOLLUTANTS BY LIPOSOMES AND CHEMOINFORMATICS ANALYSIS OF THEIR INTERACTION MECHANISM**

The work presents a novel, interpretable QSPR model for predicting the liposome-water partition coefficient (`log D_lip/w`, pH 7.4) for 306 diverse organic micropollutants. By applying a stability-selection framework to NASAWIN fragment descriptors, we identify **10 mechanistically meaningful fragments** whose coefficients directly correspond to physical processes (pi-stacking, halogen bonding, H-bond penalties, hydrophobic matching). This shifts the paradigm from pure statistical prediction to *mechanistic understanding*, enabling quantitative rules for "green" molecular design to reduce bioaccumulation risk.

** Computational Workflow for Mechanistic Fragment Selection
The development of the interpretable 10-fragment QSPR model followed a two-stage procedure combining data-driven stability selection with domain-informed mechanistic curation:

## Stage 1: Stability Selection on the Full Fragment Set

We began with a descriptor matrix of 306 organic micropollutants × 50 NASAWIN molecular fragments, where each fragment count was treated as a continuous variable (reflecting multiplicity in the molecular structure). To identify robust predictors of liposome–water partitioning (log D_lip/w, pH 7.4), we applied weighted stability selection [Meinshausen & Bühlmann, 2010]:

1. Resampling: Generated 1,000 random subsamples (80% of the dataset).
2. Modeling: On each subsample, fitted a LASSO regression model with internal 5-fold cross-validation to optimize the regularization parameter (α).
3. Selection Frequency: For each of the 50 fragments, computed the proportion of subsamples in which it received a non-zero coefficient.
4. Stable Fragment Identification: Retained all fragments with selection frequency ≥ 80%. This yielded 50 stable fragments, confirming 
that nearly the entire initial descriptor set contained reproducible signal.

This step was implemented in weighted_stability_selection.py and served as an exploratory filter to eliminate noisy or unstable descriptors before full model validation.

## Stage 2: Nested Cross-Validation and Mechanistic Curation

Using the 50 stable fragments, we performed a rigorous model validation via nested cross-validation (fragment_analysis.py):
Outer loop (5-fold): Simulated external prediction by splitting data into training/test sets (80/20).
Inner loop (5-fold): Tuned LASSO hyperparameters on the training set only.
Performance metrics: Reported mean and standard deviation of R²<sub>train</sub>, Q²<sub>EXT</sub>, and RMSE<sub>test</sub> across outer folds.
While this 50-fragment model achieved high predictive accuracy (Q²<sub>EXT</sub> = 0.923 ± 0.018), its interpretability was limited by descriptor redundancy and ambiguous chemical meaning.
To address this, we curated a minimal subset of 10 fragments based on:
High stability (selection frequency ≥ 84.7%),
Clear physicochemical mechanisms (e.g., π-stacking, halogen bonding, H-bond penalties),
Coverage of key structural motifs across chemical classes (pharmaceuticals, PFAS, halogenated aromatics, etc.).
This curated set formed the basis of the final 10-fragment mechanistic model, whose coefficients were statistically characterized via 5-fold nested CV (10-fragment_model_statistics.py) and validated against membrane biophysics principles.

## Key Findings

| Mechanism | Dominant Fragment | Coefficient | Environmental Implication |
|-----------|-------------------|-------------|---------------------------|
| **pi-Stacking** | `p2.CB2CB_.4` (aromatic C-C) | +1.143 | Drives extreme partitioning of PAHs (e.g., benzo[a]pyrene, logD = 7.15) |
| **H-bond Penalty** | `p1.O__` (ether/carbonyl O) | -0.489 | Explains low partitioning of esters/ketones vs. hydrocarbons |
| **Steric Shielding** | `p3.CB1C__Cl_.41` (ortho-Cl-phenol) | +0.210 | Rationalizes higher logD of ortho- vs. para-chlorophenols |
| **Halogen Bonding** | `p3.Cl1C__Cl1.11` (para-dichloro) | +0.090 | Small but significant; explains PCB anomalies vs. logK<sub>OW</sub> |
| **Electrostatic Attraction** | PFAS class (UMAP cluster) | - | Resolves anomaly of perfluorinated acids via anion-choline interaction |

Model performance: **Q2<sub>EXT</sub> = 0.681**, RMSE = 1.066 (5-fold nested CV).

## Licensing

This repository contains two types of content, each under a separate license:

- **Software (Python scripts)**: Licensed under the [BSD 3-Clause License](LICENSE).
- **Supplementary scholarly material** (`Supplementary.pdf`, figures, tables, documentation): Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0.txt).

You may use the code under BSD terms and the scientific content under CC BY 4.0—just provide appropriate credit in each case.

## Supplementary Material
- [`Supplementary.pdf`](Supplementary.pdf): Additional illustrations, extended discussion of fragment mechanisms, and validation details.
- Licensed under [CC BY 4.0](LICENSE-CC-BY-4.0.txt).

## Acknowledgments

This research and code development benefited from interactive assistance provided by **Qwen** (https://chat.qwen.ai), a large language model developed by Tongyi Lab, Alibaba Cloud. Qwen supported tasks including Python scripting, error debugging, Markdown formatting, and scientific writing refinement.

## Citation

If you use this repository in your work, please cite:  
> Telegin F.Y. Liposome-Water Partitioning: Mechanistic Fragment-Based QSPR Model. GitHub; 2026. https://github.com/ferntea/liposome_qspr_mechanism


## Repository Structure
```
liposome_qspr_mechanism/
├── README.md                 # This file
├── CONTRIBUTING.md           # How to contribute
├── LICENSE                   # BSD 3-Clause
├── LICENSE-CC-BY-4.0.txt     # CC BY 4.0 
├── requirements.txt          # Python dependencies
├── Supplementary.pdf         # Supplementary comments on results and additional illustrations
│
├── data/
│ ├── 2019-S_Lin-pH7_4.sdf            # Original SDF file from Lin et al. (2019); input for NASAWIN fragment generation
│ ├── 2019-S_Lin-pH7_4_decoded.csv    # Decoded dataset containing SMILES strings and experimental logD_lip/w (pH 7.40)
│ └── liposome_fragments.csv          # Fragment descriptor matrix generated by NASAWIN (306 compounds × 51 fragments)
│
├── scripts/
│ ├── sdf_to_smiles.py                          # Converts input SDF to SMILES + target property (logD_lip/w) CSV
│ ├── fragment_analysis.py                      # Full pipeline: stability selection → nested CV
│ ├── weighted_stability_selection.py           # Core stability algorithm
│ ├── 10-fragment_model.py                      # Final mechanistic model builder
│ ├── 10-fragment_model_statistics.py           # Statistical characterization (p-values, CI)
│ ├── dataset_and_correlation_visualisation.py  # Enhanced Observed vs. Predicted plot
│ └── umap_comparison_50_vs_10.py               # Side-by-side UMAP visualization
│
└── results/
  ├── liposome_analysis_stability_results.csv             # Analysis of 50-descriptor model
  ├── liposome_analysis_cv_results.csv
  ├── liposome_analysis_mechanistic_interpretation.csv
  ├── liposome_analysis_selection_frequency.png
  ├── stability_results.csv
  ├── top_fragments.csv
  ├── mechanistic_model_coefficients.csv                  # Analysis of 10-descriptor model
  ├── mechanistic_model_performance.png
  ├── statistical_characterization.csv
  ├── enhanced_observed_vs_predicted.png
  ├── enhanced_observed_vs_predicted.pdf
  ├── table-2b51272b-bc34-4e63-a106-6e5406f6a4e6.csv
  ├── table-2b51272b-bc34-4e63-a106-6e5406f6a4e6 (1).csv
  ├── umap_comparison_50_vs_10.png                        # Comparison of 50- and 10-descriptor models
  └── umap_comparison_50_vs_10.pdf
```