# Embotics AI Teacher Robot Internship — Consolidated Team Work Report

## 1. Project Overview

The Embotics AI internship team has been working on an **offline AI Teacher Robot** intended to support school students when a teacher is unavailable or busy.

The long-term target is a voice-based educational assistant that can:

- Listen to a student's spoken question.
- Convert speech to text locally.
- Understand the question and its language.
- Retrieve the correct information from school textbooks and curriculum material.
- Generate a simple, student-friendly answer using a local LLM.
- Speak the answer back to the student.
- Operate primarily offline on a **Raspberry Pi 5**, with no mandatory cloud dependency.
- Support curriculum-grounded answers instead of relying only on an LLM's pretrained knowledge.
- Eventually support multiple grades, subjects, English, Urdu, and Roman Urdu.
- Support quizzes, student learning progress, teacher assistance, and classroom interaction.

The target grade range discussed for the full project is **Grades 2–10**, while the practical experiments have intentionally started with smaller scopes such as **Class 5 textbooks** so that the complete system can first be validated on manageable data.

A simplified end goal is:

```text
Student
   ↓
Microphone
   ↓
Wake Word / Voice Activity Detection
   ↓
Speech-to-Text
   ↓
Educational Router
   ↓
Curriculum Retrieval / RAG
   ↓
Local LLM
   ↓
Grounding / Safety Validation
   ↓
Text-to-Speech
   ↓
Speaker / Robot UI
```

---

# 2. Overall Project Strategy

The work across the internship has been divided into several technical areas:

1. **Local LLM research and Raspberry Pi benchmarking**
2. **PDF processing and OCR**
3. **Text cleaning and curriculum-aware chunking**
4. **Embeddings and semantic search**
5. **Vector databases**
6. **Hybrid retrieval**
7. **Retrieval-Augmented Generation (RAG)**
8. **Speech-to-Text**
9. **Text-to-Speech and audio hardware planning**
10. **Embedded hardware architecture**
11. **Multilingual retrieval**
12. **Evaluation and benchmarking**
13. **Offline deployment on Raspberry Pi 5**
14. **Architecture research and integration**

The project did not begin with one final architecture. Different interns explored different implementations so that the team could understand the strengths and weaknesses of FAISS, ChromaDB, classical retrieval, semantic embeddings, hybrid retrieval, OCR approaches, local LLM runtimes, and speech pipelines.

This experimentation gradually led toward a more unified architecture.

---

# 3. Moez Ul Haq — Team Lead, Research, Architecture, and Raspberry Pi AI Work

## 3.1 Role

Moez worked as the **AI intern team lead** for the Teacher Robot project.

The main responsibilities included:

- Defining the overall technical direction.
- Researching the Teacher Robot use case.
- Breaking the project into smaller experiments.
- Studying local LLM options.
- Benchmarking local models on Raspberry Pi 5.
- Comparing inference runtimes.
- Reviewing RAG and VectorDB approaches.
- Researching voice, STT, TTS, and edge deployment.
- Reviewing other interns' work.
- Designing and refining the overall system architecture.
- Investigating repositories and existing Raspberry Pi voice-agent architectures.
- Moving the team from isolated prototypes toward a unified Teacher Robot architecture.

---

## 3.2 Initial Teacher Robot Research

The Teacher Robot was defined as an offline educational system for students from approximately **Grade 2 to Grade 10**.

Two primary use cases were established:

### Teacher Assistance Mode

When a teacher is present but occupied, students can ask the robot questions independently.



## 3.3 Core Design Requirements Identified

Important requirements established during the project discussions included:

- Raspberry Pi 5 as the main edge device.
- 8 GB RAM as the practical deployment target.
- Local LLM inference.
- Local curriculum retrieval.
- Voice input and output.
- English support.
- Future Urdu and Roman Urdu support.
- Short and simple student-facing responses.
- Fast perceived response time.
- Avoiding hallucinated answers when textbook evidence is weak.
- Modular architecture so individual components can be replaced independently.

A desired interaction target discussed during early research was to begin responding quickly rather than forcing a student to wait for a long complete generation.

---

# 4. Local LLM Research and Benchmarking

Before adding RAG, voice, databases, and curriculum data, the team separated the **model benchmarking phase** from the complete Teacher Robot.

The purpose was to first determine:

> Can a useful local LLM actually run fast enough on Raspberry Pi 5?

The benchmark design focused on:

- Time to first token
- Total response time
- Tokens per second
- RAM usage
- CPU temperature
- Stability over repeated runs
- Answer quality
- Model loading time
- Prompt processing time

---

## 4.1 Models Investigated

Models discussed and tested during the broader experimentation included:

- TinyLlama
- Qwen2.5 0.5B
- Qwen2.5 1.5B
- Qwen2.5 3B Instruct
- Qwen3 4B

The planned controlled comparison for the stronger Teacher Robot model stage centered on:

```text
Qwen2.5-1.5B-Instruct Q4_K_M
vs
Qwen2.5-0.5B Q4_K_M
```


---

## 4.2 Raspberry Pi 5 Observations

During actual Raspberry Pi experiments, different runtimes and model sizes were tested.

Observed results from the experimentation included approximately:

### Ollama

```text
TinyLlama:
~6.43–6.52 tokens/sec
~27–28 sec for around 180 generated tokens

Qwen2.5 smaller models:
~10.25–10.65 tokens/sec
~16–17 sec for comparable short generations
```

Additional retained measurements included:

```text
TinyLlama 1.1B Q5_K_M
Load time: ~50.03 sec
Generation: ~6.43 tok/s

Qwen2.5 1.5B Q6_K
Load time: ~8.43 sec
Generation: ~10.25 tok/s
```

---

## 4.3 llama.cpp Experiments

llama.cpp-style local tests were also performed.

Examples from retained experiments included:

```text
TinyLlama:
~5.07–5.41 tok/s
RAM around 1.48 GB in one test

Qwen2.5 0.5B:
~4.96–4.99 tok/s
RAM around 1.06 GB in one test

Qwen2.5 1.5B Q2_K:
~3.66–4.61 tok/s
RAM around 0.85 GB
```

One important observation was that a model can be fast or memory-efficient but still produce a factually poor answer. One Qwen 1.5B experiment gave an incorrect answer to a Mars-related question, reinforcing the need to evaluate **quality and grounding**, not only speed.

The broader lesson from these experiments was:

```text
Model selection ≠ choose highest tokens/sec only.

Model selection =
speed
+ RAM
+ latency
+ answer quality
+ stability
+ curriculum grounding
```

---

# 5. RAG Strategy Designed for the Teacher Robot

The project moved from raw local LLM testing toward **Retrieval-Augmented Generation**.

The core idea became:

```text
Student Question
      ↓
Retrieve Relevant Textbook Content
      ↓
Give Retrieved Content to Local LLM
      ↓
Generate Student-Friendly Answer
```

The key architectural decision was that **RAG should contain the curriculum knowledge**, while fine-tuning, prompt engineering, or lightweight adaptation can later control teaching behavior.

Recommended responsibility split:

| Requirement | Main Mechanism |
|---|---|
| Textbook knowledge | RAG |
| New syllabus/book edition | Update RAG data |
| Teaching style | Prompting / possible LoRA |
| Student-friendly language | Prompting / grade adapter |
| Current conversation | Session memory |
| Student progress | SQLite |
| Retrieval knowledge | Vector database |

---

# 6. PiBot Repository Architecture Analysis

The repository:

```text
mayukh4/pibot_local_agent
```

was studied as a reference for Raspberry Pi voice-agent architecture.

The useful architectural pattern was:

```text
USB Microphone
      ↓
Wake Word
      ↓
Audio Manager
      ↓
Whisper.cpp
      ↓
Router
      ↓
Local LLM / Tools
      ↓
Piper TTS
      ↓
Speaker + UI
```

The repository was valuable because it separated the project into modules such as:

```text
audio/
brain/
senses/
ui/
config/
tools/
tests/
```

The Teacher Robot architecture reused the **modular pattern**, not the generic assistant features.

PiBot concepts recommended for reuse included:

- Central orchestrator
- Wake-word pause/resume
- STT wrapper
- TTS wrapper
- Local Ollama client
- State-based UI
- Tool router
- Configuration management
- Component testing

Generic tools such as weather/news/jokes were considered irrelevant to the core Teacher Robot and should be replaced by educational tools.

---

# 7. Educational Architecture Designed from the Repository Research

The proposed Teacher Robot modules included:

```text
teacher_robot/
│
├── orchestrator.py
│
├── audio/
│   ├── audio_manager.py
│   ├── stt_engine.py
│   ├── tts_engine.py
│   ├── vad_engine.py
│   └── noise_filter.py
│
├── senses/
│   ├── wake_word_detector.py
│   └── push_to_talk.py
│
├── brain/
│   ├── educational_router.py
│   ├── local_llm_client.py
│   ├── prompt_builder.py
│   ├── response_validator.py
│   └── session_manager.py
│
├── rag/
│   ├── ingestion_pipeline.py
│   ├── chunker.py
│   ├── embedding_engine.py
│   ├── retriever.py
│   └── vector_store.py
│
├── education/
│   ├── grade_adapter.py
│   ├── curriculum_manager.py
│   ├── quiz_engine.py
│   └── answer_evaluator.py
│
├── memory/
│   ├── conversation_memory.py
│   └── student_memory.py
│
└── ui/
    ├── ui_manager.py
    └── states.py
```

Possible educational intents identified included:

```text
GREETING
CURRICULUM_QUESTION
CONCEPT_EXPLANATION
FOLLOW_UP
EXAMPLE_REQUEST
REPEAT_REQUEST
TRANSLATION_REQUEST
QUIZ_START
QUIZ_ANSWER
CHAPTER_SUMMARY
CALCULATION
TEACHER_COMMAND
SYSTEM_STATUS
OUT_OF_SCOPE
```

---

# 8. Vector Database Experimentation and LanceDB Research

As the team built several independent VectorDB pipelines, a recurring problem appeared:

- FAISS pipelines required separate storage/metadata handling.
- Chroma pipelines solved some problems but introduced another stack.
- Different books and interns used different chunking and retrieval logic.
- Keyword retrieval and semantic retrieval were often separate.
- Multiple pipelines became harder to standardize.

This motivated research into **Lance / LanceDB** as a unified retrieval foundation.

---

## 8.1 Why LanceDB Became Important

LanceDB was investigated because a single local database can hold structured columns containing:

- Text
- Embeddings
- Vectors
- Metadata
- Images
- Multimodal data

It also supports a combination of:

- Vector similarity search
- Full-text search
- BM25-style lexical retrieval
- Metadata filtering
- Hybrid search

This matched the Teacher Robot requirement better than maintaining multiple disconnected storage systems.

---

## 8.2 Proposed LanceDB Teacher Robot Architecture

The architecture separates **heavy database building** from **lightweight Raspberry Pi querying**.

### Windows / Development PC

```text
Textbook PDF
   ↓
PDF Extraction / OCR
   ↓
Cleaning
   ↓
Curriculum Structure Detection
   ↓
Parent-Child Chunking
   ↓
Passage Embeddings
   ↓
LanceDB Build
   ↓
Hybrid Index / Evaluation
   ↓
Copy Database Bundle to Pi
```

### Raspberry Pi 5

```text
Student Question
   ↓
Question Embedding
   ↓
Metadata Filter
   ↓
Vector Search + BM25
   ↓
Rank Fusion
   ↓
Top Context
   ↓
Qwen via Ollama
   ↓
Answer
```

Heavy preprocessing remains on the PC, while Raspberry Pi performs only query-time work.

---

## 8.3 Proposed LanceDB Metadata

Recommended curriculum fields included:

```text
chunk_id
parent_id
board
grade
subject
book
edition
unit
chapter
topic
page_start
page_end
language
text_clean
text_lexical
vector
ocr_confidence
pipeline_version
```

This supports queries such as:

```text
Grade = 5
Subject = Science
Board = Federal
```

before semantic retrieval.

This reduces the chance that a Grade 5 student receives an explanation from the wrong grade or textbook.

---

## 8.4 Parent-Child Chunking

A parent-child chunking architecture was proposed:

```text
Parent chunk:
~180–300 words

Child retrieval chunks:
~60–120 words
```

The small child chunks improve retrieval precision.

After a child chunk matches the question, the system can return:

```text
matching child
+
parent context
+
optional neighboring child
```

to the LLM.

These sizes are starting values and require real evaluation rather than being treated as fixed universal rules.

---

## 8.5 Multilingual Embedding Direction

For English + Urdu + Roman Urdu retrieval, the architecture investigated:

```text
intfloat/multilingual-e5-small
```

with 384-dimensional embeddings.

The E5 retrieval format uses:

```text
passage: <textbook chunk>
query: <student question>
```

This is relevant for future multilingual Teacher Robot deployment.

---

## 8.6 Hybrid Retrieval

The proposed retrieval system became:

```text
Student Question
      ↓
Metadata / Language Filtering
      ↓
 ┌───────────────┬───────────────┐
 ↓                               ↓
Vector Search                  BM25 Search
 ↓                               ↓
 └───────────────┬───────────────┘
                 ↓
             Rank Fusion
                 ↓
           Deduplication
                 ↓
        Parent/Neighbor Expansion
                 ↓
          Confidence Threshold
                 ↓
             Best Context
```

Vector search captures semantic meaning.

BM25 helps with:

- Exact terminology
- Scientific names
- Poem/book titles
- Chapter-specific vocabulary
- OCR-sensitive words

---

## 8.7 Grounding / Decline Rule

A major project principle is:

> The Teacher Robot should not confidently guess when retrieval is weak.

Proposed flow:

```text
Strong retrieval
      ↓
Generate answer

Weak retrieval
      ↓
Return deterministic not-covered response
```

This is especially important in education because a fluent but wrong answer is more harmful than admitting that the system could not find the topic.

---

# 9. Babar — Social Studies VectorDB, RAG, Evaluation, and Audio Research

Babar's work focused on building a practical Class 5 Social Studies retrieval and RAG pipeline.

## 9.1 OCR and Vector Index Architecture Review

Babar first reviewed an OCR/vector indexing analysis and identified improvement areas including:

- Image preprocessing
- Deskewing
- Denoising
- Contrast improvement
- Header/footer/watermark removal
- Sentence reconstruction
- Better embeddings
- Smarter chunking
- Rich metadata

This review became the basis for the implementation.

---

## 9.2 Social Studies Vector Database

Source:

- Class 5 Social Studies textbook
- Scanned PDF
- Approximately 112 pages

Pipeline:

```text
PDF
 ↓
Page Images
 ↓
Deskew / Denoise / Contrast / Threshold
 ↓
Tesseract OCR
 ↓
Cleaning
 ↓
Sentence Reconstruction
 ↓
Overlapping Chunks
 ↓
BGE Embeddings
 ↓
FAISS
```

Implementation details:

- Chunk size: approximately 250–350 words
- Approximate overlap: 20%
- Total chunks: 111
- Embedding model: `BAAI/bge-small-en-v1.5`
- Embedding dimensions: 384
- Vector index: FAISS

Metadata included:

- Page
- Chunk ID
- Position
- Previous/next chunk references
- Source file
- Word count

Test queries included topics such as:

- Rights of a citizen
- Diversity and tolerance

Reported similarity scores for successful retrievals were approximately:

```text
0.53–0.77
```

---

## 9.3 Ollama Troubleshooting

Babar also installed Ollama locally and encountered a Windows crash:

```text
0xc0000005
```

The investigation included:

- Disk-space checks
- Model storage path investigation
- GPU/Vulkan investigation

The issue was traced to Ollama attempting Vulkan offloading through unsupported Intel integrated graphics.

The system was fixed by forcing **CPU-only inference**.

Models subsequently tested included:

```text
qwen2.5:0.5b
qwen2.5:1.5b
```

---

## 9.4 Full Local RAG

A local RAG script was then built:

```text
Question
 ↓
FAISS Retrieval
 ↓
Relevant Book Chunks
 ↓
Prompt
 ↓
Qwen2.5 1.5B via Ollama
 ↓
Generated Book-Grounded Answer
```

The pipeline was expanded to show:

- Retrieved pages
- Similarity scores
- Embedding time
- FAISS search time
- LLM generation time
- Total response time
- Tokens generated
- Tokens/sec
- CPU/RAM usage

---

## 9.5 RAG vs No-RAG Evaluation

A separate `no_rag.py` system was created to compare the same questions without textbook retrieval.

A key observation was that a No-RAG answer could introduce generic information not present in the textbook, while the RAG answer followed the book more closely.

This provided practical evidence for why the Teacher Robot should use curriculum-grounded retrieval.

---

## 9.6 Evaluation Dataset

Babar created:

- 50 structured questions
- Expected answers
- Expected page references
- Coverage across six Social Studies units

The questions were saved for automated evaluation.

A batch script was built to run:

```text
50 questions through RAG
+
50 questions through No-RAG
```

and export structured CSV results.

This enabled systematic comparison rather than manual testing.

---

## 9.7 Microphone and Speaker Research

Babar also researched microphone and speaker options for eventual voice interaction, considering:

- Cost
- Raspberry Pi compatibility
- Ease of integration
- Classroom usability

---

# 10. Faris — Baseline TeacherBot and Hybrid Chroma RAG

Faris's work was organized into two phases.

---

## 10.1 Phase 1 — Baseline TeacherBot

The first system tested the raw Qwen model without textbook access.

The question dataset covered school questions from:

```text
Class 2–10
```

The baseline pipeline included:

- Excel-based question dataset
- One worksheet per class
- Rule-based English / Roman Urdu detection
- Student-friendly prompting
- Qwen2.5 1.5B through Ollama
- Performance metrics
- CSV result export
- Semantic similarity scoring

Files included:

```text
teacherbot.py
prompts.py
language_detect.py
similarity.py
```

Metrics included:

- TTFT
- Total response time
- Tokens generated
- Tokens/sec

Generated answers were compared with expected answers using:

```text
all-MiniLM-L6-v2
```

for semantic similarity.

The baseline highlighted an important limitation:

> A raw local LLM is not grounded in the actual textbook.

---

## 10.2 Phase 2 — RAG Upgrade

The system was then rebuilt using:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
BGE-M3 Embeddings
 ↓
ChromaDB
 ↓
Vector Retrieval + BM25
 ↓
LangChain EnsembleRetriever
 ↓
Qwen2.5 1.5B
 ↓
Answer
```

Technologies included:

- LangChain
- Chroma
- BGE-M3
- BM25
- EnsembleRetriever
- Ollama
- Qwen2.5 1.5B

Environment troubleshooting included moving from an incompatible Python version to:

```text
Python 3.11.9
LangChain 0.3.27
Pydantic 2.13.4
```

and correcting an incorrect project path in `rag.py`.

---

## 10.3 Retrieval Tests

Reported results included:

| Question | Result |
|---|---|
| What is matter? | Correct, page 99 |
| What is photosynthesis? | Correct, pages 90 and 111 |
| What is a microorganism? | Correct, pages 24 and 88 |
| What is a noun? | Reasonable answer, but irrelevant retrieval |

The "noun" case exposed a major RAG problem:

```text
Irrelevant retrieval
      +
LLM general knowledge
      =
Plausible answer that is not properly grounded
```

This became part of the motivation for stronger grounding checks and retrieval thresholds.

---

# 11. Rayyan — Embedded Hardware Architecture

Rayyan worked on the physical and embedded design requirements for the Teacher Robot.

---

## 11.1 Main Compute Platform

Primary recommendation:

```text
Raspberry Pi 5
```

with:

```text
8 GB RAM sufficient for prototype
16 GB provides additional headroom
```

Storage recommendation:

```text
NVMe SSD
or
USB SSD
```

rather than relying entirely on microSD storage.

---

## 11.2 Main Hardware Components

The proposed embedded system included:

- Raspberry Pi 5
- NVMe / USB SSD
- ReSpeaker USB microphone array
- Powered speaker / USB speaker
- Raspberry Pi 5 5V/5A power supply
- Active cooling
- Pi 5 enclosure

The hardware design considered both a recommended prototype and a lower-cost prototype.

---

## 11.3 Storage Architecture

The SSD layout was designed to contain:

```text
Operating System
LLM Models
SQLite Database
Vector Database
Knowledge Base
Logs
Configuration
```

This aligns with a complete edge-AI deployment rather than storing only the LLM model.

---

## 11.4 Voice Hardware

Options investigated included:

### Recommended

```text
ReSpeaker USB Mic Array
USB Speaker
```

### Budget

```text
I2S 2-Mic HAT
USB DAC / powered speaker
```

The work also identified an important hardware conflict:

> An I2S microphone HAT and I2S speaker HAT may share the same I2S bus and therefore should not simply be combined without careful design.

---

## 11.5 Raspberry Pi 4 vs Pi 5 Considerations

The report also documented:

- Pi 5 RP1 I/O architecture
- GPIO differences
- Pi 5 lack of onboard 3.5 mm audio jack
- Pi 5 fan header
- Pi 5 PCIe/NVMe capability
- `gpiod` requirement compared with older `RPi.GPIO` approaches

This work provided the physical deployment foundation for the AI stack.

---

# 12. Rayyan — General Science Vector Database

Rayyan also implemented a separate General Science VectorDB pipeline.

Source material:

```text
FBISE General Science
Class 4 and Class 5
```

The pipeline was designed specifically for an offline Raspberry Pi Teacher Robot.

---

## 12.1 Build Pipeline

```text
PDF
 ↓
PyMuPDF
 ↓
Native text extraction where possible
 ↓
OCR fallback with Tesseract
 ↓
Cleaning
 ↓
Overlapping chunks
 ↓
all-MiniLM-L6-v2 embeddings
 ↓
FAISS IndexFlatIP
```

Important details:

- OCR fallback when extracted page text was under approximately 100 characters
- OCR rendered around 200 DPI
- Chunk size: 400 words
- Overlap: 60 words
- Chunks below 25 words removed
- Embedding model: `all-MiniLM-L6-v2`
- Vector dimension: 384
- Embeddings normalized
- FAISS `IndexFlatIP`
- Cosine-style similarity through normalized inner product

---

## 12.2 Query Pipeline

```text
Student Question
 ↓
Embedding
 ↓
FAISS Search
 ↓
Top-K Chunks
 ↓
RAG Prompt
 ↓
Qwen2.5
```

Default:

```text
TOP_K = 3
```

The purpose was to keep the prompt short because Pi generation speed is limited.

The design intentionally builds the database on a PC and copies only the resulting VectorDB artifacts to Raspberry Pi.

---

## 12.3 Known Limitations

Documented limitations included:

- Remaining OCR noise
- Watermark text in some chunks
- Not yet integrated end-to-end with the complete STT/TTS pipeline
- Limited initial textbook coverage

---

# 13. Talha — Whisper.cpp Speech-to-Text Pipeline

Talha focused on creating a local microphone-to-text pipeline using `whisper.cpp`.

---

## 13.1 Whisper.cpp CLI Testing

The local GGML Whisper model was tested with:

```bash
./main -m models/ggml-base.bin -f testdata/farish_16k.wav
```

Continuous microphone tests used the `stream` executable.

The pipeline successfully detected the USB microphone and generated transcriptions.

---

## 13.2 Microphone Investigation

The USB device was identified as:

```text
PCM2902 Audio Codec Analog Mono
```

An important discovery was that the device index differed between:

- whisper.cpp / SDL
- Python `sounddevice` / PortAudio

This is a practical integration detail that can otherwise create device-selection bugs.

---

## 13.3 Audio Format Problem

The microphone supported:

```text
S16_LE
Mono
44.1 kHz or 48 kHz
```

It did **not** directly provide the 16 kHz input expected by the chosen Whisper pipeline.

Attempting:

```python
samplerate=16000
```

caused:

```text
PortAudioError: Invalid sample rate
```

---

## 13.4 Resampling Solution

The solution was:

```text
Microphone
   ↓
Record at 48 kHz
   ↓
Python Resampling
   ↓
16 kHz
   ↓
Mono PCM-16 WAV
   ↓
whisper.cpp
   ↓
Transcription
```

Libraries used:

- sounddevice
- scipy.signal.resample_poly
- soundfile
- subprocess
- numpy

The Python wrapper successfully automated:

```text
record
→ resample
→ save WAV
→ invoke whisper.cpp
→ obtain text
```

---

## 13.5 Next Technical Direction

The completed version used short fixed-duration recordings.

The next stage identified was:

```text
Continuous Microphone
      ↓
VAD / Audio Buffer
      ↓
48 kHz → 16 kHz
      ↓
Whisper.cpp
      ↓
Continuous Transcription
      ↓
Teacher Robot
```

---

# 14. Additional Voice Experimentation — Vosk

The Embotics Teacher Robot work also included Vosk investigation on Raspberry Pi.

A lightweight model used in setup work was:

```text
vosk-model-small-en-us-0.15
```

A larger model considered for improved vocabulary/children's speech was:

```text
vosk-model-en-us-0.22
```

A major issue again involved microphone sample-rate mismatch.

Example:

```text
Microphone: 44.1 kHz
Vosk target: 16 kHz
```

Possible solutions investigated included:

- Software resampling
- ALSA `plughw` resampling
- Microphone gain adjustment through ALSA
- Avoiding clipping

Python 3.13 also introduced compatibility difficulties because the old built-in `audioop` module was removed.

The voice work reinforced a general project lesson:

> Audio hardware compatibility and sample-rate handling must be treated as core engineering tasks, not afterthoughts.

---

# 15. Subhan — NOME VectorDB, Classical Retrieval, Urdu VDB, and RAG Research

Subhan's work covered a broad progression from classical information retrieval to modern RAG concepts.

---

## 15.1 NOME Teaching Guide Vector Database

The NOME work created a local document retrieval pipeline.

Conceptual architecture:

```text
PDF
 ↓
Text Extraction / OCR
 ↓
Cleaning
 ↓
Chunking
 ↓
TF-IDF
 ↓
Truncated SVD / LSA
 ↓
20-Dimensional Vectors
 ↓
L2 Normalization
 ↓
FAISS IndexFlatIP
 ↓
Top-K Retrieval
```

This pipeline was primarily a **retrieval system**, not necessarily a complete RAG generation system.

---

## 15.2 Classical Retrieval Concepts Implemented

The work explored:

- TF-IDF
- Truncated SVD
- Latent Semantic Analysis
- 20-dimensional reduced representations
- L2 normalization
- Cosine-similarity-style retrieval
- FAISS IndexFlatIP
- Top-k search

This was useful for understanding the foundations beneath modern vector search.

---

## 15.3 Urdu Vector Database

Subhan also worked on an Urdu educational book pipeline.

Source material included an Urdu FBISE educational PDF of approximately 130 rendered pages.

The intended architecture was:

```text
Urdu PDF
 ↓
Page Rendering
 ↓
OCR
 ↓
Urdu Text
 ↓
Cleaning
 ↓
Chunking
 ↓
Multilingual Representation
 ↓
FAISS
 ↓
Semantic Retrieval
```

---

## 15.4 Urdu OCR Challenges

Tesseract was explored but Urdu OCR quality was not sufficiently reliable.

This demonstrated:

```text
Bad OCR
 ↓
Bad Text
 ↓
Bad Chunks
 ↓
Bad Vectors
 ↓
Bad Retrieval
 ↓
Bad Answer
```

PaddleOCR was also investigated but encountered:

```text
Illegal instruction
```

on an Intel i3-3110M machine.

This highlighted the importance of matching AI libraries and models to the actual hardware and CPU instruction set.

---

## 15.5 Retrieval vs RAG

An important conceptual distinction documented in Subhan's work was:

### Retrieval

```text
Query
 ↓
Vector Representation
 ↓
FAISS
 ↓
Relevant Chunks
```

### Full RAG

```text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Relevant Chunks
 ↓
Prompt Construction
 ↓
LLM
 ↓
Generated Answer
```

This distinction became useful for evaluating other intern pipelines as well.

---

# 16. Zahan Zahid — Local LLM Deployment, RAG, Interface Development, and Raspberry Pi Voice Pipeline

Zahan's contribution focused on **local/offline small-language-model deployment**, model comparison, inference optimization, model-specific chat interfaces, RAG experimentation, and extending the same work toward a Raspberry Pi 5 voice-assistant pipeline.

---

## 16.1 Local Model Deployment and Comparison

Three model configurations were downloaded, prepared, and evaluated:

```text
Qwen2.5-1.5B-Instruct FP16
Qwen2.5-1.5B-Instruct INT8
Gemma 3 1B
```

The purpose was to compare:

- Full precision vs quantized inference
- Setup complexity
- CPU responsiveness
- Response quality
- Instruction-following behavior
- Output tone and verbosity
- Practical offline deployment behavior

For the Qwen2.5 FP16 and INT8 variants, **OpenVINO GenAI** was used for model preparation/conversion and local inference experiments.

A parallel **Ollama-based workflow** was also used for quick comparison runs.

This work provided practical experience with:

- Model conversion
- Quantization
- Tokenizers
- Chat templates
- Stop tokens
- Runtime dependencies
- CPU-oriented local inference

---

## 16.2 Separate Model-Specific Chat Interfaces

Instead of forcing all models through one shared interface, Zahan built a separate prompt/chat window for each model.

This was useful because different model families can require different:

- Chat templates
- Token formatting
- Stop tokens
- Generation behavior

The separate interfaces made it easier to compare models side by side without one model's formatting requirements affecting another.

The interfaces also supported **live token-by-token streaming**, allowing partial responses to appear immediately instead of waiting for the full answer.

Conceptually:

```text
User Prompt
    ↓
Correct Model-Specific Chat Template
    ↓
Local Inference Runtime
    ↓
Token Streaming
    ↓
Live Chat Window
```

---

## 16.3 Generation Speed and Response-Quality Optimization

Several iterations were performed to improve both responsiveness and answer quality.

### Speed-related work

- Compared Qwen2.5 FP16 against INT8.
- Tuned generation limits.
- Experimented with batching/runtime parameters.
- Used streaming generation to improve perceived latency.
- Diagnosed Python/dependency/environment issues that slowed experimentation.

An important lesson was that **streaming improves perceived responsiveness even when raw tokens-per-second remain unchanged**.

### Quality-related work

- Corrected model-specific chat-template formatting.
- Compared instruction following between model families.
- Tested multi-turn interaction.
- Compared tone and verbosity.
- Reused similar prompts across the separate model windows for qualitative comparison.

This reinforced that poor prompt/chat formatting can make an otherwise capable local model appear significantly worse.

---

## 16.4 LLM Deployment Concepts Learned

The work developed practical understanding of:

- FP16 vs INT8
- Quantization trade-offs
- Model size vs hardware capability
- Tokenization
- Chat templates
- Stop tokens
- CPU vs GPU execution paths
- OpenVINO GenAI
- Ollama
- Streaming generation
- Memory/storage constraints
- Local/offline model serving
- Differences between model families at similar parameter counts

The practical comparison can be summarized as:

```text
Model
  +
Precision / Quantization
  +
Runtime
  +
Prompt Formatting
  +
Hardware
  =
Actual Deployment Behavior
```

---

## 16.5 LangChain RAG Pipeline

Zahan also implemented a **Retrieval-Augmented Generation pipeline using LangChain**.

The workflow included:

```text
Documents
    ↓
Document Loading
    ↓
Text Splitting
    ↓
Embeddings
    ↓
Vector Store
    ↓
Semantic Retrieval
    ↓
Retrieved Context
    ↓
Local Qwen2.5 1.5B
    ↓
Grounded Answer
```

The RAG work included:

- Loading source documents
- Chunking documents
- Generating embeddings
- Storing embeddings in a vector store
- Semantic retrieval
- Injecting retrieved chunks into the local model prompt
- Comparing grounded RAG behavior with plain prompting
- Experimenting with chunk size
- Experimenting with number of retrieved chunks
- Adjusting prompt structure to reduce irrelevant context and hallucination

This provided an end-to-end understanding of how LangChain can connect document retrieval with a locally hosted LLM.

---

## 16.6 Raspberry Pi 5 Deployment

Zahan extended the local AI work from a Windows workstation to a **Raspberry Pi 5**.

The Pi was operated primarily in headless mode through:

```text
PuTTY
+
SSH
```

Work included:

- Remote Raspberry Pi administration
- Python environment setup
- Dependency installation
- ARM-based runtime troubleshooting
- Local model deployment
- Hardware-aware model selection

The target pipeline was designed as an offline bilingual voice assistant:

```text
Microphone
    ↓
Speech-to-Text / Whisper
    ↓
Local Qwen2.5 3B-Instruct via Ollama
    ↓
Sentence-Level Streaming
    ↓
Piper TTS
    ↓
Speaker
```

The pipeline was intended to support:

```text
English
+
Urdu
```

while remaining offline.

---

## 16.7 Sentence-Level Streaming Voice Output

One particularly relevant contribution for the Teacher Robot was the use of **sentence-level streaming**.

Instead of:

```text
Generate Complete LLM Answer
        ↓
Wait
        ↓
Run TTS
```

the intended behavior was:

```text
LLM Starts Generating
        ↓
First Complete Sentence
        ↓
Send Sentence to Piper
        ↓
Robot Starts Speaking
        ↓
LLM Continues Generating
```

This improves the perceived responsiveness of a local voice assistant on constrained hardware.

For the Teacher Robot, this idea is especially useful because students should not have to wait for the entire LLM response before hearing the beginning of the explanation.

---

## 16.8 Main Contribution Summary

Zahan's work contributed experience in:

- Qwen2.5 1.5B FP16
- Qwen2.5 1.5B INT8
- Gemma 3 1B
- OpenVINO GenAI
- Ollama
- Quantization comparison
- Model-specific chat templates
- Token-by-token streaming
- Local chat-interface development
- LangChain RAG
- Embeddings and vector retrieval
- Raspberry Pi 5 deployment
- PuTTY / SSH administration
- Whisper-based voice input architecture
- Qwen2.5 3B-Instruct Pi deployment direction
- Piper TTS
- English/Urdu offline voice-assistant architecture
- Sentence-level streamed speech output

This contribution complements the rest of the Teacher Robot work by covering the **local-model deployment and user-interface layer**, while also connecting RAG and voice interaction to the Raspberry Pi edge deployment target.

---

# 17. Cross-Team Vector Database Experiments

Across the team, several VectorDB/retrieval approaches were explored.

## FAISS

Used in:

- Babar's Social Studies work
- Rayyan's General Science work
- Subhan's NOME/Urdu work
- Other Teacher Robot experiments

Strengths observed:

- Lightweight
- Fast
- Good for small local vector collections
- Easy exact similarity search

Limitations:

- Vector search is its main responsibility
- Text/metadata often have to be managed separately
- Hybrid BM25 retrieval needs additional implementation
- Different projects can easily develop incompatible metadata formats

---

## ChromaDB

Used in Faris's RAG implementation.

Strengths:

- Integrated VectorDB workflow
- LangChain support
- Persistent local database
- Easier RAG development than raw FAISS in some cases

The implementation also demonstrated hybrid retrieval by combining Chroma vector search with BM25 through LangChain.

---

## LanceDB

Investigated as the direction for a more unified architecture.

Potential advantages for this project:

- Embedded/local
- Text + vectors + metadata in the same table
- Full-text search
- BM25-style lexical retrieval
- Vector search
- Metadata filters
- Hybrid search
- Multimodal-friendly storage
- Portable PC-to-Pi workflow

The current architectural research therefore treats LanceDB as a strong candidate for consolidating previously separate VectorDB pipelines.

---

# 18. OCR Lessons Across the Team

OCR appeared repeatedly in the work of multiple interns.

Technologies included:

- Tesseract
- PyMuPDF rendering
- Native PDF text extraction
- OCR fallback
- Image preprocessing
- Urdu OCR experiments
- PaddleOCR investigation

Common problems:

- Scanned textbooks
- Watermarks
- Broken sentences
- Decorative symbols
- Low-quality page images
- Headers/footers
- Urdu recognition errors
- Diagram labels
- CPU-heavy processing

The project demonstrated that OCR is one of the most important parts of educational RAG.

A strong LLM cannot compensate for completely corrupted source text.

---

# 19. Chunking Experiments Across the Team

Different chunking configurations were tested:

### Babar

```text
~250–350 words
~20% overlap
```

### Rayyan General Science

```text
400 words
60-word overlap
```

### Proposed LanceDB architecture

```text
Child: ~60–120 words
Parent: ~180–300 words
```

These differences are useful because they provide multiple experimental baselines.

The team has not treated one universal chunk size as proven.

The correct chunking strategy should eventually be selected through retrieval evaluation.

---

# 20. Embedding Models Explored

Models used or investigated across the work included:

```text
BAAI/bge-small-en-v1.5
all-MiniLM-L6-v2
BGE-M3
intfloat/multilingual-e5-small
paraphrase-multilingual-MiniLM-L12-v2
bge-m3
```

Different models served different purposes:

- Lightweight English retrieval
- Multilingual semantic retrieval
- English/Urdu/Roman Urdu support
- General semantic similarity
- Retrieval-focused embedding

The final model for Raspberry Pi should be chosen through actual latency + retrieval-quality testing.

---

# 21. RAG Evaluation Work

The team moved beyond simply checking whether code runs.

Evaluation ideas and implementations included:

- Expected-answer datasets
- Expected-page references
- Semantic similarity
- RAG vs No-RAG comparisons
- Retrieved-page inspection
- Similarity/confidence scores
- Retrieval timing
- LLM timing
- Tokens/sec
- RAM
- CPU load
- Grounding checks
- Weak-retrieval decline
- Real textbook question testing

A future standardized evaluation set should cover cases such as:

```text
Direct textbook question
Paraphrased question
Roman Urdu query
OCR typo
Wrong chapter
Wrong subject
Out-of-book question
Ambiguous question
Exact keyword question
Conceptual semantic question
```

---

# 22. Combined Hardware + Software Architecture

The work across all interns can be combined into the following end-to-end architecture.

```text
                           AI TEACHER ROBOT
                                  │
                                  ↓
                           Student Question
                                  │
                                  ↓
                             Microphone
                                  │
                                  ↓
                    Wake Word / Push-to-Talk / VAD
                                  │
                                  ↓
                         Audio Capture Layer
                                  │
                                  ↓
                        Resampling / Filtering
                                  │
                                  ↓
                          Whisper.cpp / STT
                                  │
                                  ↓
                         Recognized Question
                                  │
                                  ↓
                       Educational Intent Router
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
               ↓                                     ↓
        Simple Local Tool                        Curriculum RAG
                                                     │
                                                     ↓
                                               Grade / Subject
                                                     │
                                                     ↓
                                              Metadata Filter
                                                     │
                                      ┌──────────────┴──────────────┐
                                      ↓                             ↓
                                Vector Search                    BM25
                                      │                             │
                                      └──────────────┬──────────────┘
                                                     ↓
                                                Rank Fusion
                                                     ↓
                                            Best Textbook Context
                                                     ↓
                                          Local Qwen / Ollama
                                                     ↓
                                          Grounding Validation
                                                     ↓
                                             Simple Response
                                                     │
                                  ┌──────────────────┴──────────────────┐
                                  ↓                                     ↓
                              Piper TTS                             Robot UI
                                  ↓
                               Speaker
                                  ↓
                               Student
```

---

# 23. Proposed Storage Architecture

The final project can separate curriculum knowledge from application state.

```text
Teacher Robot
│
├── LanceDB / Vector Retrieval Layer
│   ├── Textbook chunks
│   ├── Embeddings
│   ├── Grade
│   ├── Subject
│   ├── Board
│   ├── Chapter
│   ├── Topic
│   ├── Page
│   ├── Language
│   └── OCR / pipeline metadata
│
└── SQLite
    ├── Student profiles
    ├── Session state
    ├── Quiz results
    ├── Progress
    ├── Preferences
    └── Robot/application state
```

---

# 24. PC Builder vs Raspberry Pi Runtime

A major consolidated design principle is to avoid doing expensive one-time preparation on Raspberry Pi.

## Development PC

Run:

- PDF rendering
- OCR
- Cleaning
- Structure detection
- Chunking
- Passage embeddings
- VectorDB creation
- Full-text indexing
- Evaluation
- Database validation

## Raspberry Pi 5

Run:

- Microphone input
- STT
- Question embedding
- Metadata filtering
- Retrieval
- Local LLM inference
- TTS
- UI
- Session memory

This keeps the runtime system lightweight and more stable.

---

# 25. Main Technologies Used / Investigated

## Languages

- Python

## LLM / Runtime

- Qwen2.5
- Qwen3
- TinyLlama
- Gemma 3 1B
- Ollama
- llama.cpp
- OpenVINO GenAI

## RAG / Retrieval

- FAISS
- ChromaDB
- LanceDB
- LangChain
- BM25
- Hybrid retrieval

## Embeddings

- BGE Small
- BGE-M3
- MiniLM
- multilingual E5

## Document Processing

- PyMuPDF
- Tesseract OCR
- PaddleOCR investigation
- PDF rendering
- Text cleaning
- Chunking

## Speech

- whisper.cpp
- Vosk
- sounddevice
- ALSA
- PortAudio
- scipy resampling

## TTS / Voice Output Direction

- Piper TTS
- Sentence-level streamed TTS direction
- USB / powered speakers

## Local Deployment / Interface Tooling

- OpenVINO GenAI
- Model-specific local chat interfaces
- Token-by-token streaming
- PuTTY
- SSH

## Storage

- FAISS indexes
- Chroma persistent DB
- LanceDB architecture
- SQLite
- NVMe / USB SSD

## Hardware

- Raspberry Pi 5
- Raspberry Pi 4 comparison
- USB microphone
- ReSpeaker microphone array
- PCM2902 USB audio device
- Active cooling
- NVMe SSD / USB SSD

---

# 26. Major Engineering Problems Encountered

The internship was not only theoretical research. Multiple practical issues were encountered.

## Local AI Performance

- Small models can still be slow.
- Faster models may answer incorrectly.
- Quantization changes speed, memory, and quality.
- Loading time can be significant.
- Runtime choice affects performance.

## OCR

- Scanned books require OCR.
- Watermarks pollute chunks.
- Urdu OCR remains difficult.
- OCR errors propagate to retrieval.

## Audio

- Microphone sample rates may not match model requirements.
- Device IDs differ between libraries.
- Raspberry Pi audio configuration needs ALSA-level debugging.
- Python version changes can break older audio modules.

## RAG

- A correct-sounding answer does not guarantee correct retrieval.
- LLMs can answer from pretrained knowledge even when sources are irrelevant.
- Vector-only retrieval can miss exact textbook terms.
- Too many retrieved chunks increase latency.
- Too little context can remove essential explanation.

## Environment

- Python/library compatibility issues
- Windows Ollama/Vulkan crash
- PaddleOCR CPU compatibility
- Hardcoded paths
- Platform-specific dependencies

---

# 27. Main Lessons Learned

## Lesson 1 — The project is a system, not a single AI model

The complete robot depends on:

```text
Hardware
+
Audio
+
STT
+
OCR
+
Chunking
+
Embeddings
+
VectorDB
+
Retrieval
+
LLM
+
TTS
+
UI
+
Evaluation
```

Improving only the LLM does not guarantee a better Teacher Robot.

---

## Lesson 2 — Retrieval quality is critical

```text
Bad source
 ↓
Bad OCR
 ↓
Bad chunk
 ↓
Bad embedding
 ↓
Bad retrieval
 ↓
Bad context
 ↓
Bad answer
```

---

## Lesson 3 — RAG must be evaluated against No-RAG

A model may already know a general answer.

The important question is whether the response is:

- Grounded in the target textbook
- Appropriate for the student's grade
- Consistent with the curriculum
- Traceable to a source

---

## Lesson 4 — Edge AI requires hardware-aware design

Desktop assumptions cannot simply be transferred to Raspberry Pi.

The team had to consider:

- ARM CPU performance
- RAM
- thermal throttling
- model size
- storage
- audio interfaces
- sample rates
- library compatibility

---

## Lesson 5 — Hybrid retrieval is promising for textbooks

Vector search and BM25 solve different retrieval problems.

Combining:

```text
semantic meaning
+
exact terminology
+
metadata filtering
```

is well aligned with textbook Q&A.

---

# 28. Evolution of the Internship Work

The team's technical journey can be summarized as:

```text
Local LLM Research
        ↓
Raspberry Pi Model Testing
        ↓
PDF / OCR Experiments
        ↓
Classical Text Retrieval
        ↓
FAISS Vector Search
        ↓
Semantic Embeddings
        ↓
Multiple Textbook VectorDB Pipelines
        ↓
RAG with Qwen
        ↓
RAG vs No-RAG Evaluation
        ↓
Chroma + BM25 Hybrid Retrieval
        ↓
Voice / Whisper / Vosk Experiments
        ↓
Local Model Interface + Streaming Experiments
        ↓
OpenVINO FP16 / INT8 Comparison
        ↓
Raspberry Pi Voice Pipeline Work
        ↓
Embedded Hardware Architecture
        ↓
Unified VectorDB Requirement
        ↓
LanceDB Hybrid Architecture Research
        ↓
Toward Integrated Offline Teacher Robot
```

---

# 29. Work Status by Contributor

| Contributor | Main Work |
|---|---|
| **Moez Ul Haq** | Team lead, project research, local LLM benchmarking, Raspberry Pi model/runtime experiments, Teacher Robot architecture, PiBot architecture mapping, RAG strategy, VectorDB comparison, LanceDB/hybrid architecture research |
| **Babar** | Social Studies OCR + FAISS VectorDB, Qwen/Ollama RAG, RAG vs No-RAG, 50-question evaluation dataset, automated batch tests, microphone/speaker research |
| **Faris** | Baseline TeacherBot evaluation, language detection, semantic similarity, BGE-M3 + Chroma RAG, BM25 hybrid retrieval, LangChain integration |
| **Rayyan** | Raspberry Pi embedded hardware design, storage/audio/power/cooling architecture, General Science FAISS VectorDB for FBISE Class 4–5 |
| **Talha** | whisper.cpp STT, USB microphone testing, 48 kHz → 16 kHz resampling, Python STT wrapper, continuous STT direction |
| **Subhan** | NOME VectorDB, TF-IDF/SVD/LSA retrieval, FAISS, Urdu VDB, OCR research, multilingual retrieval concepts, RAG foundations |
| **Zahan Zahid** | Local LLM deployment and comparison, Qwen2.5 FP16/INT8, Gemma 3 1B, OpenVINO GenAI, model-specific streaming chat interfaces, LangChain RAG, Raspberry Pi 5 voice pipeline, Piper TTS, PuTTY/SSH |

---

# 30. Current Consolidated Technical Direction

Based on the accumulated experiments, the project is moving toward an architecture with:

```text
Raspberry Pi 5 8GB
+
Local Qwen model
+
Ollama / optimized local runtime
+
Whisper.cpp or validated local STT
+
Piper/local TTS
+
Curriculum-aware RAG
+
Rich metadata
+
Hybrid vector + BM25 retrieval
+
LanceDB candidate
+
SQLite for student/session state
```

The important point is that LanceDB is not being selected merely because it is newer.

It is being investigated because the team has already experienced the maintenance and integration cost of multiple disconnected FAISS, Chroma, metadata, and BM25 pipelines.

---

# 31. Current Recommended RAG Flow

```text
Textbook PDF
      ↓
Best Available Text Extraction
      ↓
OCR When Needed
      ↓
Cleaning
      ↓
Curriculum Parsing
      ↓
Parent-Child Chunking
      ↓
Multilingual Passage Embeddings
      ↓
LanceDB
      ↓
Vector + BM25 + Metadata
      ↓
Portable DB Bundle
      ↓
Raspberry Pi
      ↓
Student Question
      ↓
STT
      ↓
Question Embedding
      ↓
Hybrid Retrieval
      ↓
Threshold / Grounding Check
      ↓
Top 3–5 Context Blocks
      ↓
Qwen
      ↓
Student-Level Answer
      ↓
TTS
```

---

# 32. Work That Has Been Demonstrated vs Work Still Being Unified

## Demonstrated in Individual Pipelines

- Local Qwen inference
- Raspberry Pi model benchmarking
- FAISS retrieval
- Chroma retrieval
- BM25 hybrid retrieval
- OCR-based textbook processing
- RAG answer generation
- RAG vs No-RAG evaluation
- Semantic similarity evaluation
- Whisper.cpp microphone transcription
- Audio resampling
- Raspberry Pi embedded hardware planning
- Urdu OCR experimentation
- Multilingual retrieval research
- Vosk setup experimentation
- OpenVINO-based FP16/INT8 local model deployment
- Gemma 3 1B comparison
- Model-specific streaming chat interfaces
- LangChain-based local RAG experimentation
- Raspberry Pi 5 headless deployment through PuTTY/SSH
- Sentence-level LLM-to-Piper streaming voice architecture

## Still Being Unified Into One Final Teacher Robot

- One standard OCR pipeline
- One standard chunking strategy
- One final embedding model
- One final VectorDB architecture
- One unified metadata schema
- One final STT engine
- One final model/runtime configuration
- End-to-end microphone → RAG → Qwen → TTS on the final hardware
- Standardized evaluation across all books
- Student memory
- Quiz system
- Teacher dashboard/interface
- Multi-grade curriculum expansion

---

# 33. Source Reports Consolidated

This report combines the work documented in the following internship files:

```text
Babar_Internship_Work_Report.md
faris(rag+vectordb).md
RAYYAN_EMBEDDED_DESIGN.md
RAYYAN_VECTOR_DB_GENERAL_SCIENCE.md
talha_whisper_cpp_contribution.md
Subhan.md
zahan_Report.md
```

It also incorporates the Teacher Robot project work captured across the Embotics project discussions, including:

- Teacher Robot requirements and research
- Local LLM model analysis
- Raspberry Pi benchmarking
- Ollama token analysis
- llama.cpp experimentation
- RAG architecture
- PiBot repository analysis
- VectorDB comparison
- LanceDB research
- Hybrid retrieval architecture
- Offline deployment planning

---

# 34. Final Project Summary

The Embotics AI internship team has progressed from isolated AI experiments toward a complete offline educational AI architecture.

The work began with questions such as:

```text
Which LLM can run on Raspberry Pi?
Which VectorDB should we use?
How should scanned textbooks be processed?
Can the model answer quickly enough?
Can speech work locally?
```

Multiple pipelines were then implemented independently to answer those questions.

The team now has practical experience with:

```text
OCR
FAISS
ChromaDB
LanceDB research
BM25
Hybrid Search
Embeddings
Semantic Search
RAG
Qwen
Gemma
Ollama
llama.cpp
OpenVINO GenAI
LangChain
Whisper.cpp
Vosk
Raspberry Pi 5
Audio hardware
Multilingual retrieval
Evaluation
```

The central direction that emerged is:

> Build the knowledge database carefully on a development PC, deploy a portable retrieval bundle to Raspberry Pi 5, retrieve only the most relevant curriculum context locally, and use a small quantized local LLM to generate short, grounded educational answers.

The final Teacher Robot is therefore not just an LLM running on Raspberry Pi.

It is an integrated **offline educational AI system** composed of:

```text
Student Voice
+
Local Speech Recognition
+
Curriculum Retrieval
+
Hybrid Vector Database
+
Local LLM
+
Grounding
+
Voice Output
+
Embedded Hardware
```

This consolidated architecture is the result of the combined research, experimentation, debugging, implementation, and evaluation performed by the Embotics internship team.
