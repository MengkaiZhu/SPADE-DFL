# Numerical verification

## Exact numerical exceptions

ULP means the spacing between adjacent representable floating-point values. It is not an adjustable tolerance. Saved and recomputed heterogeneity calculations agree exactly except for the checked adjacent float64 values below, shown as saved/recomputed.

- `dir_0.6 / spade_dfl / round 34`: p `0x1.28e316b1cf5f2p-17` / `0x1.28e316b1cf5f1p-17`; q `0x1.084a893023b6bp-15` / `0x1.084a893023b6ap-15`.
- `pathological / spade_dfl / round 12`: p `0x1.918f055507013p-13` / `0x1.918f055507014p-13`; q `0x1.dc28127709516p-12` / `0x1.dc28127709517p-12`.

All 1,200 identities, paired differences, test statistics, and classifications agree; classification counts are 424/410/366. These exceptions do not imply bitwise equality across all numerical libraries.

Communication has no hypothesis tests. Four `stationarity_ci95_low` values differ by one adjacent float64 value, again shown as saved/recomputed:

- `T=1000, tau=10`: `0x1.cf7c490912a28p-17` / `0x1.cf7c490912a29p-17`.
- `T=10648, tau=22`: `0x1.c702be51097e0p-17` / `0x1.c702be51097e1p-17`.
- `T=29791, tau=1`: `0x1.b81e72e884775p-17` / `0x1.b81e72e884776p-17`.
- `T=29791, tau=31`: `0x1.b7030cdd6e819p-17` / `0x1.b7030cdd6e81ap-17`.
