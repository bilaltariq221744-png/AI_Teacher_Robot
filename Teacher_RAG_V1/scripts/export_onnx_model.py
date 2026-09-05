#!/usr/bin/env python
"""Export ``intfloat/multilingual-e5-small`` to ONNX for the Pi runtime.

Usage:
    python scripts/export_onnx_model.py --output models/multilingual-e5-small-onnx-int8 [--int8]

Writes into the output directory:
    model.onnx            # deployed artifact (INT8 when --int8, else float32)
    model_fp32.onnx       # float32 copy (for comparison)
    tokenizer.json
    config.json

After export it runs a tolerance check: mean-pooled + normalized embeddings
from the ONNX artifact vs the PyTorch model on a small sample set, and fails
(exit 1) when the worst cosine distance exceeds --tolerance.

Requires torch, transformers, and onnxruntime (PC build deps only).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.embeddings import mean_pool, normalize

SAMPLE_TEXTS = [
    "Photosynthesis is the process plants use to make their own food.",
    "Plants need sunlight, water and carbon dioxide to grow.",
    "What is photosynthesis?",
    "Birds build nests in trees and feed their young.",
    "How do plants make food?",
    # Urdu-script samples: this is a bilingual system (English + Urdu), so the
    # tolerance check needs at least one non-Latin sample — an English-only
    # sample set would silently miss a tokenizer/ONNX regression that only
    # shows up on Urdu script.
    "پودے سورج کی روشنی سے اپنی خوراک خود بناتے ہیں۔",
    "روشنی سنتھیسز کیا ہے؟",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export multilingual-e5-small to ONNX (INT8).")
    p.add_argument("--model", default="intfloat/multilingual-e5-small", help="HF model id")
    p.add_argument("--output", required=True, help="output directory (created if missing)")
    p.add_argument("--int8", action="store_true", help="also write a dynamically quantized INT8 model.onnx")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    p.add_argument("--max-length", type=int, default=512, dest="max_length")
    p.add_argument(
        "--tolerance", type=float, default=0.02,
        help="max acceptable cosine distance of the artifact vs PyTorch (0.02 for INT8, 1e-3 for fp32)",
    )
    p.add_argument("--skip-check", action="store_true", help="skip the tolerance check")
    return p.parse_args(argv)


def export_model(
    model_name: str,
    output_dir: Path,
    int8: bool = False,
    opset: int = 14,
    max_length: int = 512,
) -> Path:
    """Export the HF model to ONNX; returns the path of the artifact model.onnx."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    tokenizer.save_pretrained(str(output_dir))
    model.config.save_pretrained(str(output_dir))

    dummy = tokenizer(
        ["Plants make their own food.", "What is photosynthesis?"],
        padding="max_length", truncation=True, max_length=16, return_tensors="pt",
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(output_dir / "model_fp32.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
        )

    artifact = output_dir / "model.onnx"
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(output_dir / "model_fp32.onnx"),
            str(artifact),
            weight_type=QuantType.QInt8,
        )
    else:
        shutil.copyfile(output_dir / "model_fp32.onnx", artifact)
    return artifact


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Worst-case (max over rows) cosine distance between two embedding sets."""
    a, b = normalize(a), normalize(b)
    return float(np.max(1.0 - np.sum(a * b, axis=1)))


def check_onnx_vs_torch(
    onnx_path: Path,
    model_name: str,
    tolerance: float,
    sample_texts: list[str],
) -> float:
    """Mean-pooled, normalized embeddings: ONNX artifact vs PyTorch model."""
    import onnxruntime as ort
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_model = AutoModel.from_pretrained(model_name)
    torch_model.eval()

    inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = torch_model(**inputs).last_hidden_state.numpy()
    torch_vecs = normalize(mean_pool(hidden, inputs["attention_mask"].numpy()))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}
    feed = {name: inputs[name].numpy() for name in input_names if name in inputs}
    onnx_vecs = normalize(mean_pool(session.run(["last_hidden_state"], feed)[0], feed["attention_mask"]))

    worst = cosine_distance(torch_vecs, onnx_vecs)
    if worst > tolerance:
        raise SystemExit(
            f"FAIL: worst cosine distance {worst:.4f} exceeds tolerance {tolerance} "
            f"({onnx_path.name})"
        )
    return worst


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = export_model(
        args.model, Path(args.output), int8=args.int8, opset=args.opset, max_length=args.max_length
    )
    print(f"exported {artifact}")

    if not args.skip_check:
        worst = check_onnx_vs_torch(artifact, args.model, args.tolerance, SAMPLE_TEXTS)
        kind = "INT8" if args.int8 else "float32"
        print(f"tolerance check ({kind} vs PyTorch): worst cosine distance {worst:.4f} "
              f"(limit {args.tolerance}) — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
