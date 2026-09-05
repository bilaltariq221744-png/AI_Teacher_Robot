from pathlib import Path
import fitz

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "source_pdfs"
PAGES_DIR = BASE_DIR / "data" / "pages"

PAGES_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path: Path):
    print(f"\nOpening: {pdf_path.name}")

    doc = fitz.open(pdf_path)

    print(f"Total pages: {len(doc)}")

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        output_file = PAGES_DIR / f"page_{page_number:04d}.txt"

        output_file.write_text(
            text,
            encoding="utf-8"
        )

        print(
            f"Page {page_number:4d}/{len(doc)} "
            f"→ {len(text):6d} characters"
        )

    doc.close()

    print("\nExtraction complete.")
    print(f"Pages saved to: {PAGES_DIR}")


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    if len(pdf_files) > 1:
        print("Multiple PDFs found:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")

        raise RuntimeError(
            "Please keep only the textbook PDF in source_pdfs "
            "for this extraction step."
        )

    extract_pdf(pdf_files[0])


if __name__ == "__main__":
    main()