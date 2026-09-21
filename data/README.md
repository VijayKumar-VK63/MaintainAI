# data/ — raw stays immutable; processed is generated (git-ignored if large)

- `../CMAPSSData/` — raw NASA files (tracked locally, not in git).
- `processed/train_features.parquet` — 20,631 rows × 84 cols (26 raw + RUL/capped + 56 causal features).
- `processed/test_last_cycle.parquet` — 100 rows (last cycle per test engine + true RUL).
- Regenerate: run `notebooks/03_preprocessing.ipynb` or `src/features.py` pipeline.
