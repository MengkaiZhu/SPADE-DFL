# Interpretation and reproduction limits

- **Alzheimer main comparison.** Configuration selection preceded five-fold evaluation of the full cohort. The 344 records used in selection later appeared in the test folds, with overlaps of 0, 93, 86, 85 and 80. Patient-level independence also remains unverified for the Brain Tumor MRI splits.
- **Sensitivity analysis.** The results are exploratory validation estimates after prior test exposure. The Alzheimer representation was selected across folds. The Brain Tumor MRI sensitivity runs use ordinary Dirichlet allocation with varying client sizes.
- **Privacy scope.** The sensitivity trainer clips the entire local model displacement and releases one perturbed state per client per round. The implementation records record replacement within a client. Its budget is conditional on fixed prepared data and partitions. Data-fitted PCA, scaling, partitioning and private training diagnostics are outside that accounting.
- **Resource comparisons.** Main methods use different function-query budgets and privacy settings. DP-SGD uses pooled data. Figure 10 compares communication at equal local-update budgets, with different stationarity–consensus residuals.
- **External inputs.** Full retraining and MRI feature extraction require the [exact inputs](../data/README.md).
