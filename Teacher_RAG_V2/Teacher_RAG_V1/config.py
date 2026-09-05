from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Permanent index folder.
# Build this on laptop once, then copy the whole index/ folder to Raspberry Pi.
INDEX_DIR = ROOT_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma_db"
BM25_DOCS_PATH = INDEX_DIR / "docs_for_bm25.json"

# Data folders.
SOURCE_PDFS_DIR = ROOT_DIR / "source_pdfs"
PAGES_DIR = ROOT_DIR / "data" / "pages"

# Chroma collection name.
COLLECTION_NAME = "teacher_robot_books"

# Ollama models.
# Use same embedding model on laptop and Raspberry Pi.
EMBEDDING_MODEL = "bge-m3"

# On laptop you can use qwen2.5.
# On Raspberry Pi 5, start with qwen2.5:1.5b or qwen2.5:3b.
LLM_MODEL = "qwen2.5:1.5b"

# Retrieval settings.
RETRIEVAL_K = 5
VECTOR_WEIGHT = 0.6

# Answer settings.
MAX_WORDS = 300
TEMPERATURE = 0.1

# Voice settings.
AUDIO_DIR = ROOT_DIR / "data" / "audio"
RECORDED_WAV = AUDIO_DIR / "student_input.wav"
ANSWER_WAV = AUDIO_DIR / "answer.wav"

# Windows voice setup
PIPER_MODEL = ROOT_DIR / "models" / "piper" / "en_US-lessac-medium.onnx"

RECORD_SECONDS = 15

# Linux audio commands.
RECORD_SECONDS = 15
AUDIO_DEVICE = "default"
PLAYER_COMMAND = "aplay"