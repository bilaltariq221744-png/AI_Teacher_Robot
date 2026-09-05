"""Tests for src/eval.py (Phase 9: PDF §11, SVG N4)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.chunking import Chunk, make_lexical_text
from src.cleaning import compile_watermark_patterns
from src.eval import (
    ANSWER_CATEGORIES,
    CATEGORIES,
    DECLINE_CATEGORIES,
    EvalResult,
    EvalRow,
    _apply_threshold_to_config,
    _stub_llm,
    _watermark_leak,
    build_report,
    calibrate_threshold,
    compute_metrics,
    evaluate_question,
    load_eval_set,
    write_report,
)
from src.lancedb_store import LanceStore
from src.retriever import ContextBlock

DIM = 8


def _unit(values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


V_PLANTS = _unit([1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
V_BIRDS = _unit([-1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])


def _chunk(chunk_id, parent_id, text, *, unit="Unit 3", grade="5", page=1, kind="child"):
    return Chunk(
        chunk_id=chunk_id, parent_id=parent_id, kind=kind,
        text_clean=text, text_lexical=make_lexical_text(text), embed_text=text,
        page_start=page, page_end=page, book="Test Book", board="SNC",
        grade=grade, subject="English", unit=unit, chapter="Chapter 2",
        topic="", language="english", ocr_confidence=0.9,
        pipeline_version="test-v1",
    )


class _FixedBackend:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, texts):
        return np.repeat(self.vector[np.newaxis, :], len(texts), axis=0)


@pytest.fixture
def store(tmp_path):
    s = LanceStore.create(tmp_path / "lancedb")
    s.add_parents([
        _chunk("p_plants", "p_plants", "Plants make their own food using sunlight. This process is called photosynthesis.", kind="parent"),
        _chunk("p_birds", "p_birds", "Birds build nests in trees and feed their young.", unit="Unit 4", grade="6", kind="parent"),
    ])
    s.add_children(
        [
            _chunk("c_plants", "p_plants", "Plants make their own food using sunlight.", page=30),
            _chunk("c_photo", "p_plants", "Photosynthesis needs sunlight, water and carbon dioxide.", page=31),
            _chunk("c_birds", "p_birds", "Birds build nests in trees and feed their young.", unit="Unit 4", grade="6", page=40),
        ],
        np.stack([V_PLANTS, V_PLANTS * 0.9, V_BIRDS]),
    )
    s.build_indexes()
    return s


CFG = {"vector_top_k": 20, "bm25_top_k": 20, "rrf_k": 60,
       "final_context_blocks": 5, "context_tokens": 1200}


# --- eval set file ---------------------------------------------------------


def test_eval_set_has_six_categories_and_ten_each():
    rows = load_eval_set("evaluation/eval_set.jsonl")
    assert len(rows) == 60
    counts = {cat: 0 for cat in CATEGORIES}
    for row in rows:
        counts[row.category] += 1
        assert row.expected in ("answer", "decline")
    assert {c: counts[c] for c in CATEGORIES} == {c: 10 for c in CATEGORIES}
    assert len({r.id for r in rows}) == 60


def test_load_eval_set_validation(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "category": "bogus", "question": "q", "expected": "answer"}\n')
    with pytest.raises(ValueError, match="unknown category"):
        load_eval_set(p)
    p.write_text('{"id": "x", "category": "unrelated", "question": "q", "expected": "maybe"}\n')
    with pytest.raises(ValueError, match="answer.*decline"):
        load_eval_set(p)


# --- metrics ---------------------------------------------------------------


def _result(cat, expected, declined=False, grounded=True, correct=None, score=0.1):
    if correct is None:
        correct = (expected == "decline") == declined
    return EvalResult(
        id=f"{cat}-{score}-{declined}", category=cat, question="q",
        expected=expected, top_score=score, num_blocks=1,
        declined=declined, grounded=grounded, correct=correct,
    )


def test_compute_metrics_all_correct():
    results = [
        _result("answerable", "answer"),
        _result("paraphrased", "answer"),
        _result("roman_urdu", "answer"),
        _result("ocr_typo", "answer"),
        _result("wrong_chapter", "decline", declined=True),
        _result("unrelated", "decline", declined=True),
    ]
    m = compute_metrics(results)
    assert m["total"] == 6
    assert m["hit_rate"] == 1.0
    assert m["groundedness"] == 1.0
    assert m["decline_accuracy"] == 1.0
    assert m["false_decline_rate"] == 0.0
    assert m["overall_accuracy"] == 1.0


def test_compute_metrics_counts_failures():
    results = [
        _result("answerable", "answer", declined=True, correct=False),   # false decline
        _result("answerable", "answer"),                                 # answered, grounded
        _result("wrong_chapter", "decline"),                             # wrongly answered
        _result("unrelated", "decline", declined=True),                  # correct decline
    ]
    m = compute_metrics(results)
    assert m["hit_rate"] == 0.5          # 1 of 2 answerable answered
    assert m["groundedness"] == 1.0      # the 1 answered row has sources
    assert m["decline_accuracy"] == 0.5  # 1 of 2 decline rows declined
    assert m["false_decline_rate"] == 0.5
    assert m["overall_accuracy"] == 0.5


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m["total"] == 0
    assert m["hit_rate"] is None
    assert m["overall_accuracy"] is None


# --- threshold calibration -------------------------------------------------


def _row(rid, category, score, expected):
    return EvalResult(
        id=rid, category=category, question="q", expected=expected,
        top_score=score, num_blocks=1, declined=False, grounded=True, correct=True,
    )


def test_calibrate_finds_separating_threshold():
    results = [
        _row("a1", "answerable", 0.08, "answer"),
        _row("a2", "answerable", 0.07, "answer"),
        _row("d1", "unrelated", 0.02, "decline"),
        _row("d2", "wrong_chapter", 0.01, "decline"),
    ]
    t, cal = calibrate_threshold(results)
    assert 0.02 < t <= 0.07  # separates the groups
    assert cal["overall_accuracy"] == 1.0
    assert cal["false_decline_rate"] == 0.0
    assert cal["decline_accuracy"] == 1.0
    # sweep table covers every observed score plus the sentinels
    assert {row["threshold"] for row in cal["sweep"]} == {
        0.0, 0.01, 0.02, 0.07, 0.08, 0.08 + 1e-6,
    }


def test_calibrate_tie_breaks_fewer_false_declines():
    # t=0.04 and t=0.05 both reach 2/3 accuracy; t=0.04 wrongly declines a2
    # (false decline) while t=0.05 wrongly declines d1 — so 0.04 wins.
    results = [
        _row("a1", "answerable", 0.05, "answer"),
        _row("a2", "answerable", 0.04, "answer"),
        _row("d1", "unrelated", 0.04, "decline"),
    ]
    t, cal = calibrate_threshold(results)
    assert t == 0.04
    assert cal["overall_accuracy"] == round(2 / 3, 4)
    assert cal["false_decline_rate"] == 0.0


def test_calibrate_tie_breaks_higher_threshold():
    # t=0.05 and t=0.06 are both perfect; prefer the higher threshold.
    results = [
        _row("a1", "answerable", 0.06, "answer"),
        _row("a2", "answerable", 0.06, "answer"),
        _row("d1", "unrelated", 0.04, "decline"),
        _row("d2", "wrong_chapter", 0.04, "decline"),
    ]
    t, _ = calibrate_threshold(results)
    assert t == 0.06


def test_calibrate_empty():
    assert calibrate_threshold([]) == (0.0, {})


def test_calibrate_respects_candidates():
    # Candidate thresholds only: 0.07 answers the 0.1 row and declines the 0.05
    # row (both correct); 0.12 wrongly declines the answerable row. 0.07 wins.
    results = [_row("a1", "answerable", 0.1, "answer"), _row("d1", "unrelated", 0.05, "decline")]
    t, cal = calibrate_threshold(results, candidates=[0.07, 0.12])
    assert t == 0.07
    assert cal["overall_accuracy"] == 1.0


# --- report -----------------------------------------------------------------


def test_write_report_round_trip(tmp_path):
    results = [_row("a1", "answerable", 0.1, "answer")]
    report = build_report(results, compute_metrics(results), threshold=0.05, calibrated=True)
    out = write_report(report, tmp_path / "evaluation_report.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["calibrated"] is True
    assert loaded["threshold"] == 0.05
    assert loaded["metrics"]["overall_accuracy"] == 1.0
    assert loaded["results"][0]["id"] == "a1"


def test_apply_threshold_to_config_preserves_format(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "pc:\n  dpi: 300\n\npi:\n  rrf_k: 60\n  relevance_threshold: 0.55   # comment\n",
        encoding="utf-8",
    )
    assert _apply_threshold_to_config(p, 0.42) is True
    text = p.read_text(encoding="utf-8")
    assert "  relevance_threshold: 0.42" in text
    assert "rrf_k: 60" in text
    assert "# calibrated by src/eval.py" in text


def test_apply_threshold_missing_line(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("pi:\n  rrf_k: 60\n", encoding="utf-8")
    assert _apply_threshold_to_config(p, 0.4) is False


# --- end-to-end evaluation --------------------------------------------------


def test_evaluate_question_answerable(store):
    row = EvalRow(id="a1", category="answerable", question="How do plants make food?", expected="answer")
    result = evaluate_question(row, store, _FixedBackend(V_PLANTS), CFG, llm=_stub_llm)
    assert not result.declined
    assert result.grounded
    assert result.correct
    assert result.top_score > 0


def test_evaluate_question_unrelated_uncalibrated_is_false_positive(store):
    # With no relevance threshold, vector search always returns blocks, so an
    # unrelated question is (wrongly) answered — the exact failure mode that
    # Step 10 calibration exists to fix.
    row = EvalRow(id="u1", category="unrelated", question="What is the price of petrol?", expected="decline")
    result = evaluate_question(row, store, _FixedBackend(V_PLANTS), CFG, llm=_stub_llm)
    assert not result.declined
    assert not result.correct
    assert result.top_score > 0


def test_evaluate_question_calibrated_threshold_separates(store):
    # A threshold between the two groups' RRF scores (measured above: 0.033 vs
    # 0.016) makes the unrelated question decline and the answerable one pass.
    back = _FixedBackend(V_PLANTS)
    cfg = dict(CFG, relevance_threshold=0.02)
    a = evaluate_question(
        EvalRow(id="a1", category="answerable", question="How do plants make food?", expected="answer"),
        store, back, cfg, llm=_stub_llm,
    )
    u = evaluate_question(
        EvalRow(id="u1", category="unrelated", question="What is the price of petrol?", expected="decline"),
        store, back, cfg, llm=_stub_llm,
    )
    assert not a.declined and a.correct
    assert u.declined and u.correct and not u.grounded


def test_evaluate_question_honours_cfg_threshold(store):
    # A fixed threshold of 0.9 makes everything decline even though the raw
    # top_score is high — the cfg threshold wins over the probe measurement.
    row = EvalRow(id="a1", category="answerable", question="How do plants make food?", expected="answer")
    cfg = dict(CFG, relevance_threshold=0.9)
    result = evaluate_question(row, store, _FixedBackend(V_PLANTS), cfg)
    assert result.top_score > 0      # raw retrieval was strong ...
    assert result.declined           # ... but the gate declined anyway
    assert not result.correct


# --- CLI ---------------------------------------------------------------------


def test_eval_cli_help():
    result = subprocess.run(
        [sys.executable, "src/eval.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--apply" in result.stdout
    assert "--threshold" in result.stdout


# --- watermark leak detection -----------------------------------------------


def test_watermark_leak_detects_keyword_in_context_block():
    block = ContextBlock("c1", "p1", 0.5, "Plants make food. SAMPLE COPY", metadata={})
    assert _watermark_leak([block], "some answer text", keywords=["sample copy"], patterns=[]) is True


def test_watermark_leak_detects_keyword_in_answer_text():
    block = ContextBlock("c1", "p1", 0.5, "Plants make food.", metadata={})
    assert _watermark_leak([block], "SAMPLE COPY appears here", keywords=["sample copy"], patterns=[]) is True


def test_watermark_leak_false_when_nothing_configured():
    block = ContextBlock("c1", "p1", 0.5, "Plants make food. SAMPLE COPY", metadata={})
    assert _watermark_leak([block], "answer", keywords=[], patterns=[]) is False


def test_watermark_leak_false_when_no_match():
    block = ContextBlock("c1", "p1", 0.5, "Plants make food.", metadata={})
    assert _watermark_leak([block], "clean answer", keywords=["sample copy"], patterns=[]) is False


def test_watermark_leak_matches_regex_pattern():
    block = ContextBlock("c1", "p1", 0.5, "Downloaded by ali92 on 2026-01-02", metadata={})
    patterns = compile_watermark_patterns([r"downloaded by .+ on \d{4}-\d{2}-\d{2}"])
    assert _watermark_leak([block], "answer", keywords=[], patterns=patterns) is True


def test_compute_metrics_includes_watermark_leak_rate():
    results = [
        EvalResult(id="a", category="answerable", question="q", expected="answer",
                   top_score=1.0, num_blocks=1, declined=False, grounded=True,
                   correct=True, watermark_leak=False),
        EvalResult(id="b", category="answerable", question="q", expected="answer",
                   top_score=1.0, num_blocks=1, declined=False, grounded=True,
                   correct=True, watermark_leak=True),
    ]
    metrics = compute_metrics(results)
    assert metrics["watermark_leak_rate"] == 0.5


def test_evaluate_question_flags_watermark_leak_when_configured(store):
    """End-to-end: a configured watermark keyword that shows up in retrieved
    content must be flagged on the EvalResult, even when the question is
    otherwise answered correctly — this is the regression guard eval.py's
    watermark_leak metric exists for."""
    row = EvalRow(id="a1", category="answerable", question="How do plants make food?", expected="answer")
    result = evaluate_question(
        row, store, _FixedBackend(V_PLANTS), CFG, llm=_stub_llm,
        watermark_keywords=["sunlight"],  # deliberately matches real retrieved content
    )
    assert result.watermark_leak is True


def test_evaluate_question_no_leak_when_keyword_absent(store):
    row = EvalRow(id="a1", category="answerable", question="How do plants make food?", expected="answer")
    result = evaluate_question(
        row, store, _FixedBackend(V_PLANTS), CFG, llm=_stub_llm,
        watermark_keywords=["not for sale"],
    )
    assert result.watermark_leak is False
