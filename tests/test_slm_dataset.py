"""Phase 4 tests: SLM dataset schema, grounding, engine separation."""
import json

from src.schemas import SLMAnalysis

ALLOWED_EVIDENCE = {
    "rul_low", "rul_healthy_margin", "failure_probability_high",
    "failure_probability_low", "sensors_near_baseline",
    "trend_increasing_multiple_sensors", "insufficient_evidence",
    "s3_above_baseline", "s4_above_baseline", "s9_above_baseline",
    "s11_above_baseline", "s12_above_baseline",
}


def _load(split):
    return [json.loads(line) for line in open(f"data/slm/{split}.jsonl", encoding="utf-8")]


def test_all_targets_schema_valid():
    n = 0
    for split in ("train", "val", "test"):
        for e in _load(split):
            SLMAnalysis.model_validate_json(e["target"])
            assert set(e) == {"system", "user", "target", "meta"}
            n += 1
    assert n == 880


def test_evidence_from_closed_vocabulary():
    for split in ("train", "val", "test"):
        for e in _load(split):
            for tok in json.loads(e["target"])["evidence"]:
                assert tok in ALLOWED_EVIDENCE, tok


def test_engine_separation_no_leakage():
    engines = {s: {e["meta"]["engine"] for e in _load(s)} for s in ("train", "val", "test")}
    assert engines["train"].isdisjoint(engines["val"])
    assert engines["train"].isdisjoint(engines["test"])
    assert engines["val"].isdisjoint(engines["test"])
    assert len(engines["train"]) == 80 and len(engines["val"]) == 20 and len(engines["test"]) == 100


def test_user_prompts_contain_measured_context():
    e = _load("test")[0]
    assert "failure_probability=" in e["user"] and "Machine:" in e["user"]
    assert e["meta"]["source"] == "derived-test-engines"
