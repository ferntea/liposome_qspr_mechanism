# Contributing to the Liposome QSPR Project

Thank you for your interest in improving this mechanistic chemoinformatics model! This project welcomes contributions that enhance interpretability, expand chemical space, or validate mechanisms.

## How to Contribute

### 1. Report an Issue
- Use the GitHub Issues tab for bugs, unclear documentation, or requests for new features.
- Include the script name, Python version, and a minimal example if possible.

### 2. Propose a New Mechanistic Fragment
We encourage adding fragments with clear biophysical interpretations (e.g. `pX.F__F__` for CF₃ groups, `pY.NA_OA` for zwitterionic motifs). To propose one:

1.  Add the fragment definition to your local `NASAWIN` descriptor generation (or provide SMARTS).
2.  Run `weighted_stability_selection.py` with your expanded fragment set.
3.  If the fragment has >80% selection frequency and a statistically significant coefficient (p < 0.05), prepare a PR with:
    - The new fragment’s SMARTS and structural meaning.
    - Its coefficient and p-value from the 10-fragment model re-fit.
    - A brief mechanistic rationale (1–2 sentences) citing literature (e.g., halogen σ-hole theory, membrane dipole potential effects).

### 3. Extend the Dataset
We welcome additions of new experimental `log D_lip/w` values for:
- Ionic liquids
- Microplastic additives (e.g., UV stabilizers like benzotriazoles)
- Emerging contaminants (e.g., bisphenol analogs, novel PFAS)

To add data:
1.  Format your CSV to match `data/liposome_fragments.csv` (same column names, comma-separated).
2.  Submit a PR with the new file in `data/` and a short note in `CONTRIBUTING.md` under "New Compounds".

### 4. Improve Visualization
Enhance the UMAP or correlation plots (e.g., add interactive Plotly versions, annotate outlier compounds). Ensure new scripts are placed in `scripts/` and documented.

## Code Style
- Python 3.8+ only.
- Follow PEP 8; use `black` for formatting.
- All functions must have docstrings (Google style).
- Avoid hard-coded paths; use `os.path.join()`.

## License
By contributing, you agree that your code will be licensed under the [BSD 3-Clause License](LICENSE), the same as the main project.