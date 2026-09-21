# SLM dataset — grounded instruction examples (measured Phase 4)

- **Size**: train 640 / val 140 / test 100 (880 total), JSONL in `data/slm/`.
- **Source**: 100% derived from real engine windows through the frozen
  `rul-xgb-clf-v1` pipeline — zero hand-written answers, zero LLM-generated targets.
  `source` field per example: `derived-train-engines` / `derived-val-engines` /
  `derived-test-engines`. No synthetic examples in v1 (field reserved).
- **Composition**:
  - train: LOW 480 / HIGH 6 / CRITICAL 154; healthy 375 / hpc 125 / unknown 140
  - val: LOW 101 / MEDIUM 1 / HIGH 4 / CRITICAL 34
  - test (last cycle per NASA test engine = decision point): LOW 74 / MEDIUM 2 /
    HIGH 4 / CRITICAL 20; healthy 52 / hpc 26 / unknown 22
- **Schema**: `{system, user, target, meta}`; target validated against
  `SLMAnalysis` (risk_level, likely_condition, confidence, evidence[1..10],
  recommended_action, urgency). Every target schema-validated at generation.
- **Label rules** (`src/slm_dataset.py::derive_target`): evidence tokens from a
  closed set, triggered by measured quantities only (causal baseline deviation
  k=2, predicted RUL ≤30, failure prob vs val-tuned threshold 0.65);
  `hpc_degradation` requires ≥2 drifting watch sensors; high risk without HPC
  pattern → `unknown_anomaly` (confidence 0.40); confidence tiers from
  probability sharpness + evidence count.
- **Generation methodology**: stratified sampling across true-RUL life stages
  (early/mid/late) for train/val; decision-point (last cycle) sampling for test.
  User prompts rendered by `ContextBuilder.render_prompt` — the same code path
  used at inference, so training format matches deployment format.
- **Validation**: all 880 targets pass `SLMAnalysis`; engine sets disjoint
  (train M-engines ⊂ predictive-train 80; val ⊂ predictive-val 20; test =
  NASA test fleet, namespaced `T`; asserted in tests).
- **Limitations**: targets are rule-derived (consistent, not human-expert);
  classifier sharpness (ROC 0.994) makes MEDIUM/HIGH rare — honest reflection of
  the model, but thin coverage of borderline cases; single fault mode (HPC).
