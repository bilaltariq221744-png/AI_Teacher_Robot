# AI Teacher Robot RAG — Offline Textbook-Based Voice Assistant

## Project Overview

This project is an **offline AI Teacher Robot pipeline** that answers student questions from local textbook PDFs using **RAG (Retrieval-Augmented Generation)**.

The system is designed for low-cost classroom use, especially on a **Raspberry Pi 5** or a small mini PC. It does not depend on cloud APIs once all models and files are downloaded.

The core idea is:

```text
Student voice / text question
        ↓
Speech-to-Text using Faster-Whisper (Windows)
        ↓
RAG retrieval from textbook vector database
        ↓
Local LLM response using Ollama
        ↓
Speech output using Piper TTS
```

> **Note:** The current development/testing environment is **Windows**,
> using **Faster-Whisper** for speech-to-text instead of `whisper.cpp`.
> The original `whisper.cpp` + Raspberry Pi path further below in this
> README is still the intended low-cost classroom deployment target,
> but is being validated on Windows first. See **Latest Changes**
> below.

The important design decision is:

```text
Laptop = build the vector database once
Raspberry Pi = only run the already-built RAG + voice pipeline
```

This avoids rebuilding embeddings again and again on Raspberry Pi.

---

## Latest Changes

- Increased voice recording time from **6 → 15 seconds**.
- Switched Speech-to-Text from `whisper.cpp` to **Faster-Whisper** on Windows.
- Added Windows microphone recording using **SoundDevice**.
- Added Windows audio playback using **Winsound**.
- Installed Piper TTS with the **`en_US-lessac-medium`** voice.
- Fixed Piper model paths for the Windows environment.
- Increased response generation capacity to support up to around **300 words** when a detailed answer requires it.
- Updated the AI Teacher prompt to intelligently decide the appropriate answer length:
  - Simple question → short and direct.
  - Detailed question → detailed answer, up to around 300 words.
  - Multi-part question → answer all parts.
  - No unnecessary padding or repetition.
- Kept answers strictly grounded in the textbook context.
- Improved the prompt to make answers student-friendly, clear, relevant, and accurate.

---

## What This Project Does

This project can:

- Extract text from textbook PDFs.
- Split textbook text into small, focused chunks.
- Create embeddings using a local Ollama embedding model.
- Store those embeddings permanently in a Chroma vector database.
- Save BM25 keyword-search data for hybrid retrieval.
- Transfer the prepared vector database to Raspberry Pi.
- Ask questions from the existing vector database.
- Generate answers using a local Ollama LLM.
- Take voice input using Whisper.cpp.
- Speak answers using Piper TTS.
- Work offline after all models are installed.

---

## How the Pipeline Works

### 1. Laptop Indexing Pipeline

Run this part only on your laptop.

```text
PDF books
    ↓
Extract text page-wise
    ↓
Chunk the text into small Q&A chunks
    ↓
Create embeddings using Ollama bge-m3
    ↓
Store vectors in Chroma DB
    ↓
Store BM25 documents in docs_for_bm25.json
    ↓
Copy the index folder to Raspberry Pi
```

### 2. Raspberry Pi Runtime Pipeline

Run this part on Raspberry Pi.

```text
Student speaks
    ↓
Microphone records audio
    ↓
Whisper.cpp converts speech to text
    ↓
LangChain opens existing Chroma DB
    ↓
Hybrid retriever finds relevant textbook chunks
    ↓
Ollama LLM generates a short answer
    ↓
Piper converts answer to speech
    ↓
Speaker plays the answer
```

---

## Folder Structure

```text
teacher_robot_rag/
│
├── README.md
├── requirements.txt
├── config.py
├── main_text.py
├── main_voice.py
│
├── source_pdfs/
│   └── put_your_book_here.pdf
│
├── data/
│   ├── pages/
│   ├── extracted/
│   └── audio/
│
├── index/
│   ├── chroma_db/
│   └── docs_for_bm25.json
│
├── scripts/
│   ├── 01_extract_text_pdf.py
│   └── build_vector_db.py
│
└── src/
    ├── chunking.py
    ├── rag_engine.py
    └── voice_io.py
```

---

## File Responsibilities

### `requirements.txt`

Contains Python libraries required for LangChain, Ollama, Chroma, BM25, and PDF text extraction.

### `config.py`

Central configuration file.

It controls:

- Project paths
- Chroma DB path
- BM25 JSON path
- Ollama embedding model
- Ollama LLM model
- Retrieval settings
- Whisper.cpp binary path
- Whisper.cpp model path
- Piper binary path
- Piper voice model path
- Audio recording settings

### `scripts/01_extract_text_pdf.py`

Extracts text from a textbook PDF that already contains a selectable text layer.

**Current behavior (updated):**

- Takes **no command-line arguments**.
- Automatically looks for a single PDF inside `source_pdfs/`.
- If zero PDFs are found, it raises a `FileNotFoundError`.
- If more than one PDF is found, it raises a `RuntimeError` and asks you to keep only one PDF in `source_pdfs/` for that extraction run (this keeps the pipeline unambiguous — one textbook per pass).
- Uses `pymupdf` (the modern import) instead of the deprecated `fitz` alias — this removes the `fitz API is deprecated` warning while using the same PyMuPDF library underneath.
- Writes one `.txt` file per page into `data/pages/`, named `page_0001.txt`, `page_0002.txt`, etc.
- Prints per-page character counts as it runs, and a final `Extraction complete.` message.

It creates page-wise `.txt` files inside:

```text
data/pages/
```

Use this for English PDFs or any PDF where text can be selected/highlighted.

For scanned/image-based PDFs, OCR is needed separately — pages with `0 characters` in the output usually indicate a scanned/image page with no text layer.

**Script contents:**

```python
from pathlib import Path
import pymupdf

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "source_pdfs"
PAGES_DIR = BASE_DIR / "data" / "pages"

PAGES_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path: Path):
    print(f"\nOpening: {pdf_path.name}")

    doc = pymupdf.open(pdf_path)

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
```

> **Note:** This replaces the older two-argument version of the script
> (`01_extract_text_pdf.py input_pdf output_dir --lang-tag en`). If you
> still see `error: the following arguments are required: input_pdf,
> output_dir` when running it, your local file has not been replaced
> with the version above yet.

### `src/chunking.py`

Splits extracted textbook text into small chunks.

Default chunk size:

```text
40–90 words
```

Small chunks improve retrieval quality because each chunk stays focused on one idea.

### `scripts/build_vector_db.py`

Builds the permanent vector database.

It:

- Reads all `.txt` files from `data/pages/`
- Chunks the text
- Embeds chunks using Ollama `bge-m3`
- Stores vectors in Chroma DB
- Saves BM25 data into `index/docs_for_bm25.json`

Run this only when:

- You add new books
- You change extracted text
- You change chunking logic
- You want to rebuild the complete knowledge base

### `src/rag_engine.py`

Main RAG engine.

It:

- Opens existing Chroma DB
- Loads BM25 docs
- Builds hybrid retriever
- Sends retrieved context to Ollama LLM
- Returns an answer from textbook context, with length adapted to the question

It does **not** create the vector DB again.

**Answer length behavior (updated):**

The LLM's response generation capacity was increased to support answers
of up to around **300 words** when a question genuinely needs a
detailed explanation. The AI Teacher prompt was updated so the model
decides the appropriate length itself:

- **Simple question** → short and direct answer.
- **Detailed question** → a fuller, detailed answer, up to ~300 words.
- **Multi-part question** → every part of the question gets answered.
- No unnecessary padding, filler, or repetition either way.

Answers remain **strictly grounded in the retrieved textbook context**
regardless of length — the prompt was also improved to keep answers
student-friendly, clear, relevant, and accurate rather than generic.

### `src/voice_io.py`

Handles voice input and voice output.

**Current Windows implementation:**

- Records audio using **SoundDevice** (`sounddevice` Python package) instead of `arecord`.
- Records for **15 seconds** per question (increased from the original 6 seconds, to give students more time to speak).
- Transcribes audio using **Faster-Whisper** instead of `whisper.cpp`.
- Converts answer text to speech using **Piper** (`en_US-lessac-medium` voice).
- Plays answer audio using **Winsound** instead of `aplay`.

> The Raspberry Pi deployment path (`arecord` / `aplay` / `whisper.cpp`)
> described later in this README is still the target for the final
> low-cost classroom device, but is not what's currently being run
> during Windows development/testing.

### `main_text.py`

Text-only testing mode.

Use this first before testing voice.

### `main_voice.py`

Full voice mode.

It records the student's voice, converts it to text, sends the question to RAG, and speaks the answer.

---

## Required Models and Tools

## 1. Ollama

Ollama is used for:

- Embeddings
- Local LLM answer generation

### Required Ollama Models

On laptop:

```bash
ollama pull bge-m3
ollama pull qwen2.5
```

On Raspberry Pi:

```bash
ollama pull bge-m3
ollama pull qwen2.5:1.5b
```

Optional stronger model if Raspberry Pi can handle it:

```bash
ollama pull qwen2.5:3b
```

### Model Roles

| Model | Role |
|---|---|
| `bge-m3` | Embedding model for vector database and query embedding |
| `qwen2.5` | Main LLM for laptop testing |
| `qwen2.5:1.5b` | Lightweight LLM for Raspberry Pi |
| `qwen2.5:3b` | Better quality but heavier Pi model |

---

## 2. Speech-to-Text

Two speech-to-text backends are used depending on platform.

### Windows (current dev/testing environment) — Faster-Whisper

On Windows, speech-to-text now uses **Faster-Whisper** (a Python
package) instead of `whisper.cpp`, since it's simpler to install and
run directly inside the existing venv without a separate C++ build
step.

```bash
pip install faster-whisper
```

Microphone recording on Windows uses **SoundDevice**:

```bash
pip install sounddevice
```

Audio playback of the spoken answer on Windows uses **Winsound**
(built into the Python standard library on Windows — no extra install
needed).

Recording duration was increased from 6 seconds to **15 seconds** per
question, so students have enough time to finish speaking.

### Raspberry Pi (deployment target) — Whisper.cpp

Whisper.cpp is still the intended **speech-to-text** engine for the
final low-cost Raspberry Pi deployment.

It converts the student's spoken question into text.

Expected paths in `config.py`:

```python
WHISPER_CPP_BIN = "/home/pi/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/pi/whisper.cpp/models/ggml-base.en.bin"
```

Recommended model for Raspberry Pi:

```text
ggml-base.en.bin
```

For faster but lower-accuracy transcription:

```text
ggml-tiny.en.bin
```

For multilingual use, use non-English-specific models such as:

```text
ggml-base.bin
```

---

## 3. Piper TTS

Piper is used for **text-to-speech**.

It converts the generated answer into voice.

Piper has been installed with the **`en_US-lessac-medium`** voice
model.

### Raspberry Pi paths

```python
PIPER_BIN = "/home/pi/piper/piper"
PIPER_MODEL = "/home/pi/piper/voices/en_US-lessac-medium.onnx"
```

### Windows paths (current dev/testing environment)

Piper model paths in `config.py` were fixed for the Windows
environment — use Windows-style paths pointing to wherever Piper and
the voice model were extracted, for example:

```python
PIPER_BIN = r"F:\subhan\embotics\Langchain Bot\piper\piper.exe"
PIPER_MODEL = r"F:\subhan\embotics\Langchain Bot\piper\voices\en_US-lessac-medium.onnx"
```

> Adjust these two paths to match wherever you actually extracted the
> Piper Windows release and voice model on your machine — the exact
> folder above is an example, not a fixed requirement.

You can replace the Piper voice model with another voice if needed.

---

## 4. Audio Tools

### Windows (current dev/testing environment)

The voice pipeline on Windows uses:

- **SoundDevice** for microphone recording
- **Winsound** for audio playback

```bash
pip install sounddevice
```

`winsound` ships with Python on Windows, so no separate install is
needed for playback.

Recording length is currently **15 seconds** per question (see
Latest Changes).

### Raspberry Pi (deployment target)

The voice pipeline on Raspberry Pi uses:

```bash
arecord
aplay
```

Install ALSA utilities:

```bash
sudo apt update
sudo apt install -y alsa-utils
```

Test microphone:

```bash
arecord -D default -f S16_LE -r 16000 -c 1 -d 5 test.wav
```

Play recorded audio:

```bash
aplay test.wav
```

---

## Python Dependencies

Install using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
langchain>=0.3,<0.4
langchain-core>=0.3,<0.4
langchain-community>=0.3,<0.4
langchain-ollama>=0.2
langchain-chroma>=0.1
chromadb>=0.5
rank-bm25>=0.2
PyMuPDF>=1.24
faster-whisper
sounddevice
```

> `winsound` is part of the Python standard library on Windows and
> does not need to be listed in `requirements.txt`. `faster-whisper`
> and `sounddevice` are only needed on the Windows dev/testing path —
> the Raspberry Pi deployment path uses `whisper.cpp` (built
> separately, not via pip) plus `arecord`/`aplay` instead.

> **Note:** Your currently installed environment may resolve newer
> LangChain/Chroma versions than pinned above (a 1.x-era LangChain has
> been observed working). That's fine — write and test code against
> what `pip show` / `pip list` actually reports in your venv, not
> blindly against older examples online.

---

## Laptop Setup

### 1. Create Project Environment

```bash
cd teacher_robot_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd teacher_robot_rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Install Ollama Models on Laptop

```bash
ollama pull bge-m3
ollama pull qwen2.5
```

Make sure Ollama is running:

```bash
ollama list
```

---

### 3. Add the Textbook PDF

Put **one** textbook PDF inside:

```text
source_pdfs/
```

Example:

```text
source_pdfs/updated_6 English-1.pdf
```

> Keep only a single PDF in `source_pdfs/` when running the extraction
> step below — the script will refuse to run if it finds more than
> one, to avoid mixing pages from two different books into the same
> `data/pages/` output.

---

### 4. Extract PDF Text

Run the script with **no arguments** — it auto-detects the PDF in `source_pdfs/`:

```bash
python3 scripts/01_extract_text_pdf.py
```

Windows:

```powershell
python .\scripts\01_extract_text_pdf.py
```

Expected output:

```text
Opening: updated_6 English-1.pdf
Total pages: XXX
Page    1/XXX →   XXXX characters
Page    2/XXX →   XXXX characters
...

Extraction complete.
Pages saved to: F:\subhan\embotics\Langchain Bot\data\pages
```

Pages that show `0 characters` are usually scanned/image pages with no
selectable text layer — these will need OCR handled separately later.

After extraction, check:

```text
data/pages/
```

You should see page-wise `.txt` files.

---

### 5. Build Vector Database Once

```bash
python3 scripts/build_vector_db.py --reset
```

Windows:

```powershell
python scripts/build_vector_db.py --reset
```

This creates:

```text
index/chroma_db/
index/docs_for_bm25.json
```

This `index/` folder is the permanent knowledge base.

---

### 6. Test Text RAG on Laptop

```bash
python3 main_text.py
```

Windows:

```powershell
python main_text.py
```

Ask questions like:

```text
What is a noun?
What is the main idea of this chapter?
اسم کیا ہے؟
```

Use this step to confirm that retrieval and answers are working before moving to Raspberry Pi.

---

## Transfer to Raspberry Pi

Copy the project to Raspberry Pi:

```bash
scp -r teacher_robot_rag pi@raspberrypi.local:/home/pi/
```

Or copy only the index folder if code is already present:

```bash
scp -r teacher_robot_rag/index pi@raspberrypi.local:/home/pi/teacher_robot_rag/
```

Alternative transfer options:

- USB drive
- SCP over Wi-Fi/LAN
- GitHub for code only
- Manual folder copy
- microSD card copy

Important folder to transfer:

```text
index/
```

Without the `index/` folder, Raspberry Pi cannot answer from the books.

---

## Raspberry Pi Setup

### 1. Enter Project

```bash
cd /home/pi/teacher_robot_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Install ALSA Audio Tools

```bash
sudo apt update
sudo apt install -y alsa-utils
```

---

### 3. Install Ollama Models on Raspberry Pi

```bash
ollama pull bge-m3
ollama pull qwen2.5:1.5b
```

If performance is acceptable and you want better answers:

```bash
ollama pull qwen2.5:3b
```

Update `config.py` if needed:

```python
LLM_MODEL = "qwen2.5:1.5b"
```

or:

```python
LLM_MODEL = "qwen2.5:3b"
```

---

### 4. Test Text Mode First

Before voice mode, always test text mode:

```bash
python3 main_text.py
```

If this works, your Chroma DB, BM25 file, Ollama embeddings, and local LLM are working.

---

## Whisper.cpp Setup

### 1. Install Build Tools

```bash
sudo apt update
sudo apt install -y git build-essential cmake
```

### 2. Clone Whisper.cpp

```bash
cd /home/pi
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```

### 3. Build Whisper.cpp

```bash
cmake -B build
cmake --build build -j4
```

### 4. Download Whisper Model

```bash
bash ./models/download-ggml-model.sh base.en
```

Expected model path:

```text
/home/pi/whisper.cpp/models/ggml-base.en.bin
```

Check `config.py`:

```python
WHISPER_CPP_BIN = "/home/pi/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/pi/whisper.cpp/models/ggml-base.en.bin"
```

---

## Piper Setup

### 1. Create Piper Folder

```bash
cd /home/pi
mkdir -p piper/voices
cd piper
```

### 2. Download Piper

Download the correct Piper release for your Raspberry Pi OS and CPU architecture from the Piper releases page.

After extracting, make sure the binary path matches:

```text
/home/pi/piper/piper
```

Make it executable if needed:

```bash
chmod +x /home/pi/piper/piper
```

### 3. Add Piper Voice Model

Place your `.onnx` voice model inside:

```text
/home/pi/piper/voices/
```

Example expected path:

```text
/home/pi/piper/voices/en_US-lessac-medium.onnx
```

Check `config.py`:

```python
PIPER_BIN = "/home/pi/piper/piper"
PIPER_MODEL = "/home/pi/piper/voices/en_US-lessac-medium.onnx"
```

---

## Audio Test on Raspberry Pi

Test microphone:

```bash
arecord -D default -f S16_LE -r 16000 -c 1 -d 5 test.wav
```

Play audio:

```bash
aplay test.wav
```

If you cannot hear or record audio, fix microphone/speaker first before running voice mode.

---

## Run Full Voice Mode

```bash
python3 main_voice.py
```

Flow:

```text
Press Enter
    ↓
System records your voice
    ↓
Whisper.cpp converts voice to text
    ↓
RAG answers from textbook DB
    ↓
Piper speaks the answer
```

---

## Main Commands Summary

### Laptop

```bash
cd teacher_robot_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull bge-m3
ollama pull qwen2.5

python3 scripts/01_extract_text_pdf.py
python3 scripts/build_vector_db.py --reset
python3 main_text.py
```

### Transfer to Raspberry Pi

```bash
scp -r teacher_robot_rag pi@raspberrypi.local:/home/pi/
```

### Raspberry Pi

```bash
cd /home/pi/teacher_robot_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull bge-m3
ollama pull qwen2.5:1.5b

python3 main_text.py
python3 main_voice.py
```

---

## When to Rebuild the Vector Database

Rebuild only when:

- You add new books.
- You change textbook text.
- You improve OCR output.
- You change chunk size.
- You want to remove old content.
- You want a clean fresh index.

Command:

```bash
python3 scripts/build_vector_db.py --reset
```

Do **not** rebuild the DB every time on Raspberry Pi.

---

## Troubleshooting

### Problem: Ollama connection error

Cause:

```text
Ollama is not running.
```

Fix:

```bash
ollama serve
```

or open the Ollama app on your system.

---

### Problem: Model not found

Fix:

```bash
ollama pull bge-m3
ollama pull qwen2.5
```

On Raspberry Pi:

```bash
ollama pull qwen2.5:1.5b
```

---

### Problem: Chroma DB not found

Cause:

```text
index/chroma_db/ is missing.
```

Fix:

Build the DB on laptop:

```bash
python3 scripts/build_vector_db.py --reset
```

Then copy the `index/` folder to Raspberry Pi.

---

### Problem: BM25 file not found

Cause:

```text
index/docs_for_bm25.json is missing.
```

Fix:

Rebuild vector DB on laptop and copy the complete `index/` folder.

---

### Problem: Answers are generic or not from book

Possible causes:

- Wrong book text was extracted.
- Vector DB was not rebuilt after adding pages.
- Question is outside the textbook.
- Chunking needs improvement.
- Retrieval did not find relevant context.

Fix:

Run text mode and check sources:

```bash
python3 main_text.py
```

Inspect the shown chunks.

---

### Problem: `01_extract_text_pdf.py` asks for `input_pdf output_dir`

Cause:

```text
The file on disk is still the old two-argument version of the script.
```

Fix:

Replace the full contents of `scripts/01_extract_text_pdf.py` with the
current version shown above (uses `pymupdf`, no CLI arguments, reads
the PDF automatically from `source_pdfs/`).

---

### Problem: `fitz API is deprecated` warning

Cause:

```text
The script imports `fitz` instead of `pymupdf`.
```

Fix:

This is just a warning, not an error — PyMuPDF still works via `fitz`.
To silence it, use `import pymupdf` and `pymupdf.open(...)` instead of
`import fitz` and `fitz.open(...)`, as shown in the current script
above.

---

### Problem: Multiple PDFs found in `source_pdfs/`

Cause:

```text
More than one .pdf file exists in source_pdfs/.
```

Fix:

Keep only one textbook PDF in `source_pdfs/` per extraction run, then
re-run:

```bash
python3 scripts/01_extract_text_pdf.py
```

---

### Problem: Answers get cut off or feel too short for a detailed question

Cause:

```text
Response generation capacity was set too low for the question type.
```

Fix:

Confirm the LLM generation limit in `config.py` / `rag_engine.py`
allows up to around 300 words for detailed questions, and that the
AI Teacher prompt includes the length-adaptation instructions (short
for simple questions, detailed up to ~300 words for complex ones,
answer every part of multi-part questions).

---

### Problem: No sound recorded on Windows / recording cuts off too early

Possible causes:

- Wrong microphone selected as the default input device for `sounddevice`.
- Recording duration too short for the question.

Fix:

- Check available input devices with `sounddevice.query_devices()`.
- Confirm recording duration is set to **15 seconds** (increased from
  the original 6 seconds).

---

### Problem: Whisper does not create transcript

**On Raspberry Pi (whisper.cpp):**

Possible causes:

- Wrong Whisper binary path
- Wrong Whisper model path
- Audio file not recorded properly

Fix:

Check `config.py` paths:

```python
WHISPER_CPP_BIN = "/home/pi/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/pi/whisper.cpp/models/ggml-base.en.bin"
```

Test audio separately:

```bash
arecord -D default -f S16_LE -r 16000 -c 1 -d 5 test.wav
```

**On Windows (Faster-Whisper):**

Possible causes:

- `faster-whisper` not installed in the active venv.
- Recorded audio file from `sounddevice` is empty, silent, or the wrong sample rate.
- Wrong or unavailable Faster-Whisper model size configured.

Fix:

- Confirm `faster-whisper` is installed: `pip show faster-whisper`.
- Play back the recorded `.wav` file with Winsound (or any player) to confirm it actually captured audio before it reaches transcription.
- Double check the microphone recording duration (currently 15 seconds) is long enough for the question being asked.

---

### Problem: Piper does not speak

Possible causes:

- Wrong Piper binary path
- Wrong voice model path
- Speaker not configured

Fix:

**On Raspberry Pi:**

Check `config.py`:

```python
PIPER_BIN = "/home/pi/piper/piper"
PIPER_MODEL = "/home/pi/piper/voices/en_US-lessac-medium.onnx"
```

Test speaker:

```bash
aplay test.wav
```

**On Windows:**

Check `config.py` points to the Windows Piper executable (`piper.exe`)
and the `en_US-lessac-medium.onnx` voice model with correct
Windows-style paths, e.g.:

```python
PIPER_BIN = r"F:\subhan\embotics\Langchain Bot\piper\piper.exe"
PIPER_MODEL = r"F:\subhan\embotics\Langchain Bot\piper\voices\en_US-lessac-medium.onnx"
```

Test speaker output using Winsound directly on a known-good `.wav`
file to rule out a Piper problem versus a Windows audio/output-device
problem.

---

## Recommended Development Order

Follow this order:

```text
1. Run text-only RAG on laptop.
2. Confirm answers and sources.
3. Build final vector DB once.
4. Copy project and index folder to Raspberry Pi.
5. Run text-only RAG on Raspberry Pi.
6. Install and test microphone.
7. Install and test Whisper.cpp.
8. Install and test Piper.
9. Run full voice mode.
10. Measure speed, RAM, CPU temperature, and answer quality.
```

Do not start with voice mode directly. First make sure text mode works.

---

## Current Limitations

This project currently does not include:

- Wake word detection
- Student memory
- Quiz mode
- Grade/subject/chapter auto-routing
- Web UI
- Robot face animation
- Multi-student profiles
- Teacher dashboard

These can be added later after the RAG + voice pipeline is stable.

---

## Future Improvements

Possible next features:

- Wake word using openWakeWord
- Grade and subject selection
- Chapter-specific retrieval
- Quiz generation
- Student answer evaluation
- Roman Urdu response tuning
- Urdu Piper voice
- Better OCR cleaning
- SQLite learning memory
- Classroom UI states
- Teacher dashboard
- Mini PC deployment version

---

## Final Concept

This project is the **brain and voice pipeline** of the AI Teacher Robot.

The final idea is:

```text
Offline textbook knowledge
        +
Local LLM
        +
Voice input/output
        =
Low-cost AI teacher assistant
```

The system should behave like a curriculum-grounded teacher assistant, not a general chatbot.

It should answer only from selected textbook content, keep answers simple, and run locally on affordable hardware.
