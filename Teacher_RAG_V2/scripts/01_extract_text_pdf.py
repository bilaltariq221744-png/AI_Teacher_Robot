#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF


def extract_pdf_text(input_pdf: Path, output_dir: Path, lang_tag: str = "en") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(input_pdf)
    manifest = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        page_file = output_dir / f"{input_pdf.stem}_page_{i:04d}.txt"
        page_file.write_text(text, encoding="utf-8")

        manifest.append(
            {
                "source_file": input_pdf.name,
                "page": i,
                "chars": len(text),
                "lang": lang_tag,
                "extraction_method": "text_layer",
                "text_file": page_file.name,
            }
        )

        print(f"{input_pdf.name} page {i:4d}: {len(text):5d} chars")

    manifest_path = output_dir / f"{input_pdf.stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    empty_pages = sum(1 for item in manifest if item["chars"] == 0)
    print(f"\nDone: {input_pdf.name}")
    print(f"Pages extracted: {len(manifest)}")
    print(f"Empty pages: {empty_pages}")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf", help="Path to PDF file")
    parser.add_argument("output_dir", help="Folder where page text files will be saved")
    parser.add_argument("--lang-tag", default="en", help="Language tag, e.g. en, ur")
    args = parser.parse_args()

    extract_pdf_text(
        input_pdf=Path(args.input_pdf),
        output_dir=Path(args.output_dir),
        lang_tag=args.lang_tag,
    )


if __name__ == "__main__":
    main()