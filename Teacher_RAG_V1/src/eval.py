"""Evaluation runner (PDF §11; SVG N4).

Runs the eval set against a built LanceDB store and measures:

* **hit rate**        — answerable categories (1-4) that actually retrieve,
* **groundedness**    — answered rows whose context carries page sources,
* **decline accuracy** — wrong-chapter / unrelated rows (5-6) that decline,
* **false decline rate** — answerable rows wrongly declined,
* **overall accuracy**  — every row's decline-vs-answer behavior matches the
  eval set's ``expected`` label,
* **watermark leak rate** — rows where a configured watermark keyword/regex
  surfaced in retrieved context or the final answer (should always be 0;
  a nonzero rate means the watermark defense in ocr.py/cleaning.py regressed).

Threshold calibration: each question records its best pre-gate RRF score
(``top_score``). ``calibrate_threshold`` sweeps thresholds across the observed
score distribution and picks the value that maximizes overall accuracy,
tie-breaking toward fewer false declines, then a higher threshold (a teacher
robot must never confidently answer out-of-scope questions). The calibrated
value is printed (and written back to config.yaml with ``--apply``) — it is
dataset-specific and must not be treated as universal.

Usage:
    python src/eval.py --set evaluation/eval_set.jsonl --db work/lancedb
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import compile_watermark_patterns
from src.config import DEFAULT_CONFIG_PATH, Config
from src.rag_chain import answer, decline
from src.retriever import ContextBlock, retrieve

CATEGORIES = (
    "answerable",
    "paraphrased",
    "roman_urdu",
    "ocr_typo",
    "wrong_chapter",
    "unrelated",
)
ANSWER_CATEGORIES = CATEGORIES[:4]
DECLINE_CATEGORIES = CATEGORIES[4:]
DEFAULT_EVAL_SET = Path(__file__).resolve().parent.parent / "evaluation" / "eval_set.jsonl"
DEFAULT_REPORT = Path(__file__).resolve().parent.parent / "evaluation_report.json"


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    """One eval-set line: a question plus its expected behavior."""

    id: str
    category: str
    question: str
    expected: str  # "answer" | "decline"


@dataclass
class EvalResult:
    """Outcome of evaluating one question against a store."""

    id: str
    category: str
    question: str
    expected: str
    top_score: float  # best pre-gate RRF score (the calibration signal)
    num_blocks: int
    declined: bool
    grounded: bool  # the answer carried page sources
    correct: bool  # declined/answered matches ``expected``
    watermark_leak: bool = False  # a configured watermark keyword/pattern
    # surfaced in retrieved context or the final answer text — this should
    # always be False; a regression here means a watermark defense broke.


def load_eval_set(path: str | Path) -> list[EvalRow]:
    """Parse the JSONL eval set, validating categories and expected labels."""
    rows: list[EvalRow] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            row = EvalRow(
                id=str(data.get("id", "")),
                category=str(data.get("category", "")),
                question=str(data.get("question", "")),
                expected=str(data.get("expected", "")),
            )
            if not row.id or not row.question:
                raise ValueError(f"eval set line {lineno}: id and question are required")
            if row.category not in CATEGORIES:
                raise ValueError(
                    f"eval set line {lineno}: unknown category {row.category!r} "
                    f"(expected one of {CATEGORIES})"
                )
            if row.expected not in ("answer", "decline"):
                raise ValueError(
                    f"eval set line {lineno}: expected must be 'answer' or 'decline', "
                    f"got {row.expected!r}"
                )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _watermark_leak(
    blocks: Sequence[ContextBlock],
    answer_text: str,
    keywords: Sequence[str],
    patterns: Sequence["re.Pattern"],
) -> bool:
    """True if any configured watermark keyword/regex shows up in retrieved
    context or the final answer — the class of bug this eval check exists
    to catch (a broken/regressed watermark defense letting a stamp through
    to the student)."""
    if not keywords and not patterns:
        return False
    texts = [b.text for b in blocks] + [answer_text or ""]
    kw_lower = [k.lower() for k in keywords if k]
    for text in texts:
        low = (text or "").lower()
        if any(k in low for k in kw_lower):
            return True
        if any(p.search(text or "") for p in patterns):
            return True
    return False


def evaluate_question(
    row: EvalRow,
    store,
    backend,
    cfg: Optional[dict] = None,
    llm=None,
    watermark_keywords: Sequence[str] = (),
    watermark_patterns: Sequence["re.Pattern"] = (),
) -> EvalResult:
    """Evaluate one question end to end.

    ``top_score`` is the best RRF score with the relevance gate off (the raw
    retrieval signal used for calibration); the actual answer uses ``cfg`` as
    given (so it honours a calibrated ``relevance_threshold`` when present).
    """
    cfg = dict(cfg or {})
    probe_cfg = dict(cfg)
    probe_cfg["relevance_threshold"] = None  # ungated: measure raw retrieval
    blocks = retrieve(row.question, store, backend, probe_cfg)
    top_score = float(blocks[0].score) if blocks else 0.0

    result = answer(row.question, store, backend, cfg, llm=llm)
    return EvalResult(
        id=row.id,
        category=row.category,
        question=row.question,
        expected=row.expected,
        top_score=top_score,
        num_blocks=len(blocks),
        declined=result.declined,
        grounded=bool(result.sources),
        correct=(row.expected == "decline") == result.declined,
        watermark_leak=_watermark_leak(blocks, result.text, watermark_keywords, watermark_patterns),
    )


def compute_metrics(results: Sequence[EvalResult]) -> dict[str, Any]:
    """Aggregate per-question results into the report metrics.

    ``groundedness`` is computed only over rows that were actually answered
    (declined rows have no sources by design).
    """
    total = len(results)
    answerable = [r for r in results if r.category in ANSWER_CATEGORIES]
    declineable = [r for r in results if r.category in DECLINE_CATEGORIES]
    answered = [r for r in answerable if not r.declined]

    def rate(numer: int, denom: int) -> Optional[float]:
        return round(numer / denom, 4) if denom else None

    return {
        "total": total,
        "hit_rate": rate(len(answered), len(answerable)),
        "groundedness": rate(sum(r.grounded for r in answered), len(answered)),
        "decline_accuracy": rate(sum(r.correct for r in declineable), len(declineable)),
        "false_decline_rate": rate(sum(r.declined for r in answerable), len(answerable)),
        "overall_accuracy": rate(sum(r.correct for r in results), total),
        "watermark_leak_rate": rate(sum(r.watermark_leak for r in results), total),
        "per_category": {
            cat: {
                "count": n,
                "correct": rate(
                    sum(r.correct for r in results if r.category == cat), n
                ),
            }
            for cat, n in (
                (cat, sum(1 for r in results if r.category == cat)) for cat in CATEGORIES
            )
        },
    }


# ---------------------------------------------------------------------------
# threshold calibration
# ---------------------------------------------------------------------------


def _predict_answered(results: Sequence[EvalResult], threshold: float) -> dict[str, bool]:
    """Would each question be answered (not declined) at ``threshold``?

    Pure retrieval-level prediction from ``top_score`` — no LLM call needed,
    which makes the sweep cheap and deterministic.
    """
    return {r.id: r.top_score >= threshold for r in results}


def calibrate_threshold(
    results: Sequence[EvalResult], candidates: Optional[Sequence[float]] = None
) -> tuple[float, dict[str, Any]]:
    """Pick the relevance threshold that maximizes eval-set agreement.

    Sweeps every observed ``top_score`` (plus 0.0 and a value above the max so
    "always decline" is a candidate). Tie-breaks: fewer false declines, then
    the higher threshold — out-of-scope answers are the worst failure mode for
    a teacher robot. Returns ``(best_threshold, metrics_at_best)``.
    """
    results = list(results)
    if not results:
        return 0.0, {}
    observed = sorted({r.top_score for r in results})
    sweep = sorted(set([0.0, observed[-1] + 1e-6] + list(candidates or observed)))

    best_key = (-1.0, -1, -1.0)  # (accuracy, -false_declines, threshold)
    best_t = sweep[0]
    table = []
    for t in sweep:
        pred = _predict_answered(results, t)
        correct = sum(1 for r in results if pred[r.id] == (r.expected == "answer"))
        acc = correct / len(results)
        false_declines = sum(
            1 for r in results if r.expected == "answer" and not pred[r.id]
        )
        correct_declines = sum(
            1 for r in results if r.expected == "decline" and not pred[r.id]
        )
        decline_total = sum(1 for r in results if r.expected == "decline")
        answer_total = sum(1 for r in results if r.expected == "answer")
        key = (acc, -false_declines, t)
        if key > best_key:
            best_key, best_t = key, t
        table.append(
            {
                "threshold": round(t, 6),
                "overall_accuracy": round(acc, 4),
                "false_decline_rate": round(false_declines / answer_total, 4)
                if answer_total
                else None,
                "decline_accuracy": round(correct_declines / decline_total, 4)
                if decline_total
                else None,
            }
        )

    # Metrics at the winning threshold, taken from the sweep table.
    best_row = next(r for r in table if r["threshold"] == round(best_t, 6))
    return best_t, {"sweep": table, **best_row}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def build_report(
    results: Sequence[EvalResult],
    metrics: dict[str, Any],
    threshold: Optional[float],
    calibrated: bool = False,
    sweep: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Assemble the JSON report written next to the eval set (N4)."""
    return {
        "calibrated": calibrated,
        "threshold": threshold,
        "metrics": metrics,
        "calibration_sweep": sweep or [],
        "results": [asdict(r) for r in results],
    }


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _apply_threshold_to_config(config_path: str | Path, threshold: float) -> bool:
    """Rewrite ``pi.relevance_threshold`` in config.yaml (via ``--apply``).

    Line-level regex edit so YAML comments/formatting are preserved. Returns
    True when the line was found and updated.
    """
    p = Path(config_path)
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"^(\s*relevance_threshold\s*:\s*)[^\n]+",
        rf"\g<1>{threshold}   # calibrated by src/eval.py (Step 10)",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n:
        p.write_text(new_text, encoding="utf-8")
    return bool(n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _stub_llm(messages: list, cfg: Optional[dict] = None) -> str:
    """Retrieval-focused mode: answer without a live Ollama instance."""
    return "[stub answer: retrieval succeeded]"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the eval set against a built LanceDB store and calibrate "
        "the relevance threshold (PDF section 11, SVG N4)."
    )
    ap.add_argument("--set", default=str(DEFAULT_EVAL_SET), help="JSONL eval set")
    ap.add_argument("--db", default=None, help="override pi.db_path (built store)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml")
    ap.add_argument("--onnx-dir", default=None, help="ONNX embedder dir (Pi mode)")
    ap.add_argument("--device", default=None, help="torch device for sentence-transformers")
    ap.add_argument(
        "--threshold", type=float, default=None,
        help="use a fixed threshold; skip calibration",
    )
    ap.add_argument("--llm", choices=("off", "ollama"), default="off",
                    help="off: stub answers (retrieval metrics only); ollama: real LLM")
    ap.add_argument("--out", default=str(DEFAULT_REPORT), help="report output path")
    ap.add_argument("--apply", action="store_true",
                    help="write the calibrated threshold back into config.yaml")
    args = ap.parse_args(list(argv) if argv is not None else None)

    cfg = Config.load(args.config)
    pi = dict(cfg.pi)
    if args.db:
        pi["db_path"] = args.db
    else:
        # Local-testing fallback: pi.db_path is the deployed Pi location,
        # which usually won't exist on a laptop before a Phase 10 deployment
        # bundle exists. Fall back to pc.db_path — where build_db.py
        # actually wrote the store locally — using the same "actually try
        # opening it" check as query_rag.py's resolve_db_path/_is_valid_store
        # (a bare directory-exists check isn't reliable: lancedb.connect()
        # silently creates its target directory as a side effect of just
        # connecting, even with no tables inside, so a stale empty directory
        # from an earlier failed/exploratory attempt would look "present" —
        # confirmed as a real bug via testing).
        from src.lancedb_store import LanceStore as _ProbeStore

        def _is_valid_store(path: str) -> bool:
            try:
                _ProbeStore.open(path)
                return True
            except Exception:
                return False

        if not _is_valid_store(pi.get("db_path", "")):
            pc_db_path = cfg.pc.get("db_path", "work/lancedb")
            if _is_valid_store(pc_db_path):
                print(f"note: pi.db_path not found on this machine; using pc.db_path instead: {pc_db_path}")
                pi["db_path"] = pc_db_path
    if args.onnx_dir:
        pi["onnx_model"] = args.onnx_dir

    rows = load_eval_set(args.set)
    print(f"eval set: {len(rows)} question(s) from {args.set}")

    try:
        from src.lancedb_store import LanceStore

        store = LanceStore.open(pi["db_path"])
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        from src.embeddings import load_embedder

        backend = load_embedder(
            model_name=cfg.pc.get("embed_model"),
            onnx_dir=args.onnx_dir,
            device=args.device,
        )
        # Touch the backend now so a missing model fails fast, before 60 evals.
        backend.encode(["query: probe"])
    except ImportError as exc:
        print(
            f"error: embedding backend unavailable — install sentence-transformers "
            f"(or pass --onnx-dir) ({exc})",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, OSError) as exc:
        print(f"error: embedding backend failed to load: {exc}", file=sys.stderr)
        return 1

    llm = None if args.llm == "ollama" else _stub_llm

    eval_cfg = dict(pi)
    if args.threshold is None:
        eval_cfg["relevance_threshold"] = None  # measure raw retrieval first

    watermark_cfg = dict(cfg.pc.get("watermark") or {})
    wm_keywords = list(watermark_cfg.get("keywords", []) or [])
    wm_patterns = compile_watermark_patterns(watermark_cfg.get("regex_patterns", []) or [])

    results = [
        evaluate_question(
            row, store, backend, eval_cfg, llm=llm,
            watermark_keywords=wm_keywords, watermark_patterns=wm_patterns,
        )
        for row in rows
    ]

    metrics = compute_metrics(results)
    print(
        f"hit_rate={metrics['hit_rate']} groundedness={metrics['groundedness']} "
        f"decline_accuracy={metrics['decline_accuracy']} "
        f"false_decline_rate={metrics['false_decline_rate']} "
        f"overall_accuracy={metrics['overall_accuracy']} "
        f"watermark_leak_rate={metrics['watermark_leak_rate']}"
    )
    if metrics["watermark_leak_rate"]:
        print(
            f"warning: watermark_leak_rate={metrics['watermark_leak_rate']} — "
            "a configured watermark keyword/pattern surfaced in retrieved context "
            "or an answer. Check the watermark: section in config.yaml and rebuild.",
            file=sys.stderr,
        )

    calibrated = False
    threshold = args.threshold
    sweep = None
    if threshold is None:
        threshold, cal = calibrate_threshold(results)
        metrics = {**metrics, **cal}
        metrics["threshold"] = threshold
        sweep = cal.get("sweep")
        calibrated = True
        print(
            f"calibration: best relevance_threshold = {threshold} "
            f"(overall_accuracy={metrics['overall_accuracy']})"
        )
        if args.apply:
            if _apply_threshold_to_config(args.config, threshold):
                print(f"applied: pi.relevance_threshold = {threshold} in {args.config}")
            else:
                print(
                    f"warning: could not find 'relevance_threshold' in {args.config}",
                    file=sys.stderr,
                )
    else:
        metrics["threshold"] = threshold

    report = build_report(
        results, metrics, threshold, calibrated=calibrated, sweep=sweep,
    )
    out = write_report(report, args.out)
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())