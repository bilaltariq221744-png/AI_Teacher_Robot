Here is the full `README.md` content:

```markdown
# AI Teacher Robot RAG — Offline Textbook-Based Voice Assistant

## Project Overview

This project is an **offline AI Teacher Robot pipeline** that answers student questions from local textbook PDFs using **RAG (Retrieval-Augmented Generation)**.

The system is designed for low-cost classroom use, especially on a **Raspberry Pi 5** or a small mini PC. It does not depend on cloud APIs once all models and files are downloaded.

The core idea is:

```text
Student voice / text question
        ↓
Speech-to-Text using Whisper.cpp
        ↓
RAG retrieval from textbook vector database
        ↓
Local LLM response using Ollama
        ↓
Speech output using Piper TTS
```

The important design decision is:

```text
Laptop = build the vector database once
Raspberry Pi = only run the already-built RAG + voice pipeline
```

This avoids rebuilding embeddings again and again on Raspberry Pi.

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
│   └── put_your_books_here.pdf
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

Extracts text from PDFs that already contain selectable text.

It creates page-wise `.txt` files inside:

```text
data/pages/
```

Use this for English PDFs or any PDF where text can be selected/highlighted.

For scanned/image-based PDFs, OCR is needed separately.

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
- Returns a short answer from textbook context

It does **not** create the vector DB again.

### `src/voice_io.py`

Handles voice input and voice output.

It:

- Records audio using `arecord`
- Transcribes audio using Whisper.cpp
- Converts answer text to speech using Piper
- Plays answer audio using `aplay`

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

## 2. Whisper.cpp

Whisper.cpp is used for **speech-to-text**.

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

Expected paths in `config.py`:

```python
PIPER_BIN = "/home/pi/piper/piper"
PIPER_MODEL = "/home/pi/piper/voices/en_US-lessac-medium.onnx"
```

You can replace the Piper voice model with another voice if needed.

---

## 4. Linux Audio Tools

The voice pipeline uses:

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
```

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

### 3. Add PDFs

Put your textbook PDFs inside:

```text
source_pdfs/
```

Example:

```text
source_pdfs/book1.pdf
source_pdfs/book2.pdf
```

---

### 4. Extract PDF Text

For PDFs with selectable text:

```bash
python3 scripts/01_extract_text_pdf.py source_pdfs/book1.pdf data/pages --lang-tag en
python3 scripts/01_extract_text_pdf.py source_pdfs/book2.pdf data/pages --lang-tag en
```

For Windows:

```powershell
python scripts/01_extract_text_pdf.py source_pdfs/book1.pdf data/pages --lang-tag en
python scripts/01_extract_text_pdf.py source_pdfs/book2.pdf data/pages --lang-tag en
```

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

python3 scripts/01_extract_text_pdf.py source_pdfs/book1.pdf data/pages --lang-tag en
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

### Problem: Whisper does not create transcript

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

---

### Problem: Piper does not speak

Possible causes:

- Wrong Piper binary path
- Wrong voice model path
- Speaker not configured

Fix:

Check `config.py`:

```python
PIPER_BIN = "/home/pi/piper/piper"
PIPER_MODEL = "/home/pi/piper/voices/en_US-lessac-medium.onnx"
```

Test speaker:

```bash
aplay test.wav
```

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
```