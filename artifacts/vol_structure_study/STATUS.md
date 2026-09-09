# Volatility-structure classical study — frozen status

Frozen: 2026-09-07. Branch: `study/daily-volatility-structure` (uncommitted).
Protocol: 286-stock panel, global-date 70/15/15 splits with label purge,
QLIKE primary, MAE/RMSE secondary, Ridge α=100, validation-only selection,
Bartlett HAC (L=h-1 primary), H ∈ {5, 10, 20}. Test never scored.

## Executed rungs (validation QLIKE vs Arm A, common scored sample)

| rung | h=5 | h=10 | h=20 | verdict |
|---|---|---|---|---|
| A rolling champion | 0.74918 | 0.58190 | 0.46066 | baseline (bit-reproduced) |
| B1-parkinson | −0.011, p=0.44 | −0.014, p=0.32 | −0.006, p=0.70 | null |
| B1-garman_klass | +0.001, p=0.82 | −0.002, p=0.95 | +0.006, p=0.57 | null |
| B1-rogers_satchell | +0.001, p=0.85 | −0.001, p=0.97 | +0.007, p=0.49 | null |
| B1-yang_zhang | +0.110, p=4e-14 | +0.117, p=2e-11 | +0.119, p=4e-11 | **candidate (banked)** |
| B2 Ridge+range | +0.040, p=0.19 | +0.038, p=0.31 | +0.019, p=0.62 | null |
| C production HAR | −5.40 | −6.32 | −10.12 | null (recursive log-median underforecast ×5–8) |
| D HAR+range Ridge | cliff-invalid | cliff-invalid | cliff-invalid | null (see below) |

Cells show ΔQLIKE vs A (positive favors candidate) with HAC p-values.
Artifacts: `vol_structure_v1` (A), `vol_structure_b` (B), `vol_structure_c`
(C), `vol_structure_d` (D).

## D verdict detail (null, mechanism understood)

D's primary-metric comparison tripped the preregistered suspicious flag:
8.6%/6.7%/3.2% of Ridge predictions go negative and the 0→1e-12 double
floor turns each into a ~1e9 penalty. A scale-aware epsilon-floor
recomputation (diagnostic only, protocol unchanged) still shows D 2.5–3.4×
worse than A (2.53/2.04/1.21 vs 0.75/0.58/0.46), and MAE agrees (+7%), so
the cliff explains the magnitude but not the verdict. Diagnostics show an
intercept-dominated fit (pred/real 1.13–1.22) with a Parkinson/RS
collinearity seesaw and zeroed YZ/HAR weights: no genuine structure.
Unconstrained-linear regression is additionally an invalid estimator for
variance targets — the log-link design of the GPU study answers exactly this.

## Not executed (not nulls)

- **E (asymmetry Ridge)**: not executed — study stopped for compute/time
  prioritization after D.
- **F (GJR/EGARCH panel)**: not executed — same reason. Reference
  implementations and DGP-recovery tests remain in
  `research/volatility_structure/asymmetric_garch.py`.

## Banked

- **Yang–Zhang**: +15–26% relative QLIKE improvement, HAC p ~ 1e-11..1e-14
  at all horizons, broad-based across 78–89% of validation dates, coherent
  overnight-variance mechanism (+23–26% level vs realized). MAE disagrees
  (favors A); both reported. First serious validation candidate found.
