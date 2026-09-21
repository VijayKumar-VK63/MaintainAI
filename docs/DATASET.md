# Dataset — NASA CMAPSS FD001 (selected)

- **Source**: NASA PCoE CMAPSS Jet Engine Simulated Data
  (https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data); local copy in `CMAPSSData/`.
- **License**: public (NASA Open Data Portal, accessLevel=public). Code in this repo is MIT;
  dataset retains its NASA public terms. Verify portal terms before redistribution.
- **Citation**: Saxena, Goebel, Simon & Eklund, "Damage Propagation Modeling for
  Aircraft Engine Run-to-Failure Simulation", PHM08.
- **Machine type**: simulated commercial turbofan fleet, run-to-failure trajectories.
- **FD001 scope**: 100 train engines + 100 test engines; ONE operating condition
  (sea level); ONE fault mode (HPC degradation).
- **Columns**: unit, cycle, 3 operational settings, 21 sensors (s1..s21).
- **Target**: Remaining Useful Life (RUL) in cycles. Train trajectories run to failure
  (RUL computable per cycle); test trajectories end before failure with true RUL in
  `RUL_FD001.txt`. Training uses piecewise-linear capped RUL (cap=125); evaluation
  reports against uncapped official RUL.
- **Sensor notes (FD001)**: s1, s5, s6, s10, s16, s18, s19 ~constant — exclude from
  features. Informative: s2, s3, s4, s7–s9, s11–s15, s17, s20, s21.
- **Limitations**: simulated (not physical engines); single condition/fault mode —
  no generalization claim to other faults; sensor noise injected; operating-setting
  variation minimal in FD001.
- **Preprocessing**: raw immutable → processed (RUL join, capped RUL, causal rolling
  features with shift(1)) → features. Documented per-phase in notebooks 02–04.
- **Leakage policy (mandatory)**: split by ENGINE (unit), never by row; scalers /
  baselines fit on train engines only; NASA test trajectories + RUL file touched
  only for final evaluation; SLM train/val/test separated by engine with no test
  leakage into targets, thresholds, or tuning.
- **Leakage check (measured Phase 2)**: train/val split is 80/20 engines,
  disjoint (asserted in tests). IMPORTANT: unit IDs 1–100 repeat across the train
  and test files — they are distinct engine fleets sharing an ID namespace, so
  cross-file ID comparison is meaningless; test engines are namespaced as
  `test_<unit>` and never used for fitting. All rolling features use `.shift(1)`
  (causal, verified by unit test); first-cycle history backfilled from current
  value so no NaNs reach the scaler.
- **Measured stats**: train 20,631 rows / 100 engines (max cycle 362, mean RUL
  107.8, capped-125 mean 86.8); test 13,096 rows / 100 engines, true RUL mean
  75.5 (min 7, max 145). Constant sensors verified std≈0: s1, s5, s10, s16, s18,
  s19 (s6 std 0.0014, also excluded).
