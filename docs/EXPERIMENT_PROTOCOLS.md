# Experiment protocols

Frozen configurations define every parameter reported below. A seed fixes a pseudorandom sequence; a fold is one cross-validation split. A partition assigns observations to clients. IID means identically distributed client sampling. A Dirichlet partition produces unequal class proportions. PCA (principal component analysis) is a fitted linear reduction in feature dimension.

## Main comparison: Figures 3–6 and Table II

Four datasets each use four partitions and seven methods. MNIST, Fashion and Brain each contribute 840 runs (30 seeds per condition); Alzheimer contributes 4,200 (five folds × 30 seeds). All 6,720 trajectories have rounds 0–50, including initialization. Final values use round 50, not the best observed round.

MNIST classifies digits 6/7; Fashion classifies labels 0/1. Pixels are divided by 255; PCA10 and score scaling are fitted on training data. Alzheimer uses the frozen 39-dimensional clinical representation and training-fold standardization. Brain preserves official Training/Testing splits: frozen ResNet-18 `IMAGENET1K_V1`, its `weights.transforms()`, removed final classification layer and evaluation mode yield 512 features, followed by training-fitted PCA10 and a 40-parameter classifier (10 features × four classes).

Main SPADE uses 31 clients, 50 rounds, persistent memory and zero Gaussian release noise. Local steps per round are two for MNIST/Fashion/Brain and eight for Alzheimer. The decentralized graph is Erdős–Rényi with edge probability 0.3. DP-SGD is centralized (`N=1`). Methods do not share equal query budgets or privacy protection. Alzheimer evaluation is non-nested because earlier selection samples later enter its test folds; see [limitations](KNOWN_LIMITATIONS.md).

Final and roundwise summaries use population SD (`ddof=0`, dividing squared deviations by the trajectory count). Alzheimer directly summarizes 150 fold-seed trajectories per condition; it does not first pool folds within each seed. Public names `mnist67`, `fashion01`, `alzheimer`, `brain_tumor_mri` correspond to CSV names `mnist`, `fashion`, `alzheimer`, `brain`.

## Sensitivity: Tables III–IV

The 360-run plan covers 36 unique conditions: client counts 30/40/50/75/100 at ε=32 and ε=4/8/16/24/32 at 100 clients, with δ=10⁻⁵. The shared condition is reused, giving 40 displayed cells across the tables. Each condition has five seeds; Alzheimer additionally has five folds. ε and δ describe the conditional training-mechanism privacy budget, not a certified raw-data workflow.

Dirichlet concentration is 0.3. Capacities are balanced for MNIST, Fashion and Alzheimer; Brain uses ordinary, unbalanced allocation. MNIST uses 10 rounds, 25 local steps, batch 1308 and `full_refresh`; Fashion uses 10 rounds, 20 local steps, batch 1322 and `full_refresh`; Alzheimer uses 25 rounds, 25 local steps, batch 256 and `persistent_saga`; Brain uses 10 rounds, 25 local steps, batch 250 and `persistent_saga`. Alzheimer retains five principal components plus an intercept, giving six parameters, unlike the main representation. Increasing clients also increases total queries; this is not fixed-total-compute comparison.

Accuracy is reconstructed from `correct / eval_count`. For Alzheimer, sum correct predictions and evaluation counts over five folds within each seed, divide once, then summarize five seed accuracies. Use sample SD (`ddof=1`, denominator one less than the seed count). This intentionally differs from main aggregation. These are exploratory validation results with prior test exposure and cross-fold representation selection, not independent test confirmation.

## Heterogeneity: Figures 7–9

Figures 7–8 use MNIST 6/7, 1,000 clients on a degree-two ring, 50 rounds, eight local steps, ε=8, δ=10⁻⁵ and persistent memory. The 150 runs cover five partitions and 30 seeds. Figure 7 uses two-sided 95% Student t confidence intervals, sample SD and 29 degrees of freedom. Figure 8 uses all 150,000 client-label proportions for distribution statistics; its fixed 1,400-point sample per partition affects displayed scatter only.

Figure 9 uses a separate 840-run method suite with 31 clients; centralized DP-SGD is the `N=1` exception. Its round is an optimizer step with zero communication. Six methods are each compared with LT-ADMM-VR using 30 paired seeds, four partitions and rounds 1–50. The 1,200 two-sided Wilcoxon signed-rank tests use `zero_method=wilcox`, `method=auto`; all-zero differences have p=1. Benjamini–Hochberg (BH) correction jointly adjusts all 1,200 p values to control the false discovery rate. The [implementation note](../IMPLEMENTATION_NOTE.md) retains two exact numerical-library exceptions.

## Communication: Figure 10

This non-private synthetic diagnostic uses 20 clients and dimension 20. Thirty dataset seeds, four local-update budgets `T={1000,3375,10648,29791}`, and two schedules give 240 records and eight summaries. Local schedules use `tau={10,15,22,31}`; the comparison uses `tau=1`, with rounds `K=T/tau`. The horizon schedule sets η=T⁻¹ᐟ², μ=T⁻¹ᐟ⁶ and β=a₀/(τημ), as recorded in the frozen configuration. Memory is initialized each round and batches are drawn without replacement.

Communication is `2|E|dK` scalars (on this ring, `2NdK`), not bytes or elapsed time. The maximum budget yields 768,800 versus 23,832,800 scalars: 96.8% fewer at equal local-update budget. This is not an equal-accuracy claim; optimization criteria differ. Dataset seeds are paired, but queries differ and query randomness is not fully shared. There are no hypothesis tests. Summary intervals use two-sided 95% Student t intervals over 30 seeds, sample SD and 29 degrees of freedom. The four exact float64 exceptions are listed in the [implementation note](../IMPLEMENTATION_NOTE.md).
