# Known limitations

The following ten limitations define how the saved results may be interpreted. Numerical agreement and package integrity do not resolve them.

1. **Alzheimer main evaluation is non-nested.** Configurations were selected on an earlier validation split before five-fold evaluation of the full cohort. All 344 selection-validation records later entered the reported test folds, with overlaps 0, 93, 86, 85, and 80. Training-fold standardization does not remove this reuse. The available evidence neither quantifies the resulting bias nor proves a ranking reversal.

2. **Sensitivity results are exploratory validation.** Prior test exposure and Alzheimer representation selection across folds prevent an independent-test interpretation of Tables III–IV. The frozen metadata therefore retains `validation_only=true`, `formal_authorized=false`, and `previous_test_exposure=true`. `formal_authorized` is a historical result-suite compatibility field, not a legal or software permission; see the [implementation note](../IMPLEMENTATION_NOTE.md).

3. **Brain sensitivity uses an unbalanced partition.** Its 45 runs use ordinary `dirichlet` allocation with variable client capacities. The paper's general balanced-capacity description does not apply to these runs, so they are not a fixed-client-capacity experiment.

4. **Alzheimer sensitivity uses a different representation.** Five principal components plus an intercept give six model parameters, unlike the 39-dimensional main-comparison representation. The saved results and preprocessing follow the six-parameter implementation.

5. **Memory and noise differ by experiment family.** Main SPADE retains memory across rounds and uses zero Gaussian release noise. Sensitivity MNIST/Fashion uses `full_refresh`; Alzheimer/Brain uses `persistent_saga`. Query noise and privacy release noise are distinct.

6. **The historical implementations differ from Algorithm 1.** Classification and sensitivity batches sample with replacement rather than as uniform subsets without replacement. Private trainers clip the complete round-level model displacement, while the displayed algorithm separates the data-update and penalty displacements. Classification schedules also differ from the communication horizon schedule. The saved experiments are not a line-by-line execution of every theoretical update.

7. **Privacy accounting is conditional on fixed prepared data.** Shared data-fitted PCA/scaling and label-based partitioning can change multiple clients after a raw-record change and are not automatically covered by the training-mechanism budget. Raw-data-to-result privacy is not certified; `end_to_end_privacy_certified=false` remains applicable. Data-dependent logs, unperturbed accuracies, and label histograms do not automatically inherit the same budget. Record replacement and whole-client replacement are different adjacency definitions. This limitation is not evidence of an observed privacy attack.

8. **Main-method rankings are not an equal-budget comparison.** Methods use different query budgets and privacy settings, and DP-SGD is centralized. Main SPADE results do not inherit privacy guarantees from other experiment families.

9. **The reported communication reduction has a specific denominator.** Figure 10's 96.8% reduction is measured at equal local-update budget. Optimization criteria differ, so it is not a 96.8% reduction at equal accuracy, stationarity, or query budget.

10. **Complete data and MRI reproduction remains unverified.** Several exact public acquisition routes and versions are unresolved. Sensitivity preparation additionally requires eight historical upstream caches beyond the 23 runtime placements; Brain main and sensitivity caches with the same filename have different hashes and are not interchangeable. The exact raw MRI archive was located and hash-matched in retained materials, but public access to that version and the complete extraction-to-training chain have not been validated. No byte-identical image crossed the historical Training/Testing split, but near duplicates or different slices from one patient were not excluded. Patient-level independence, raw-to-prepared array equivalence, full MRI L3, cross-platform bitwise equality, full-suite runtime, and peak memory remain unverified. See the [data guide](../data/README.md).

The current-PDF comparison is retained separately in [reproducibility](REPRODUCIBILITY.md). Code-license and third-party-rights status is described in [NOTICE](../NOTICE.md).
