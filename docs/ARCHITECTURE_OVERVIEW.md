# AI Teacher Robot — Architectural Overview

**Project:** AI Teacher Robot
**Version:** 0.1.0 (Pre-Alpha)
**Date:** 2026-07-25
**Author:** Bilal Tariq
**Status:** Scaffold / Placeholder Phase

---

## 1. Repository Structure

### Complete Folder Tree

```
AI_Teacher_Robot/
├── .gitignore                          # Python + Raspberry Pi exclusions
├── .pre-commit-config.yaml             # Pre-commit hooks (ruff, black, isort, pytest)
├── .env.example                        # Environment variable template
├── README.md                           # Professional project README with badges
├── CONTRIBUTING.md                     # Coding guidelines, branch conventions, PR process
├── CHANGELOG.md                        # Keep a Changelog format
├── LICENSE                             # MIT License
├── requirements.txt                    # Production dependencies (placeholder)
├── requirements-dev.txt                # Dev dependencies (pytest, ruff, black, mypy, etc.)
├── setup.py                            # Package installation script
├── pyproject.toml                      # Build config + tool configs (black, ruff, mypy, pytest)
├── src/
│   ├── __init__.py                     # Package init with __version__, __author__
│   ├── main.py                         # Application entry point (AITeacherRobot orchestrator)
│   ├── config/
│   │   ├── __init__.py                 # Exports Settings class
│   │   ├── settings.py                 # Settings class (Singleton pattern, YAML + env loading)
│   │   ├── paths.py                    # Centralized project file path definitions
│   │   └── config.yaml                 # Default YAML configuration
│   ├── speech/
│   │   ├── __init__.py                 # Exports AudioListener, SpeechToText, LanguageDetector
│   │   ├── listener.py                 # AudioListener — captures audio from microphone
│   │   ├── stt.py                      # SpeechToText — converts audio to text (Vosk/Whisper)
│   │   └── language_detector.py        # LanguageDetector — detects English vs Urdu
│   ├── llm/
│   │   ├── __init__.py                 # Exports LLMClient, PromptManager, ResponseHandler
│   │   ├── client.py                   # LLMClient — HTTP client for local LLM API
│   │   ├── prompt_manager.py           # PromptManager — builds prompts per language
│   │   └── response_handler.py         # ResponseHandler — cleans/validates LLM responses
│   ├── tts/
│   │   ├── __init__.py                 # Exports TextToSpeechSynthesizer, Speaker
│   │   ├── synthesizer.py              # TextToSpeechSynthesizer — text to audio (pyttsx3/Coqui)
│   │   └── speaker.py                  # Speaker — plays audio through speaker
│   ├── hardware/
│   │   ├── __init__.py                 # Exports HardwareController, EyeController, etc.
│   │   ├── controller.py               # HardwareController — main GPIO/servo controller
│   │   ├── eyes.py                     # EyeController — controls eye servos
│   │   ├── mouth.py                    # MouthController — controls mouth servo
│   │   └── animations.py               # AnimationController — predefined animation sequences
│   └── utils/
│       ├── __init__.py                 # Exports APP_NAME, APP_VERSION, SUPPORTED_LANGUAGES
│       ├── constants.py                # Project-wide constants
│       ├── logger.py                   # Loguru logging configuration
│       └── helpers.py                  # Helper functions (timer, truncate, merge, etc.)
├── docs/
│   ├── SRS.md                          # Software Requirements Specification
│   ├── Architecture.md                 # System architecture, data flow, design patterns
│   └── Research.md                     # STT/TTS/LLM engine evaluations
├── tests/
│   ├── __init__.py                     # Test package init
│   ├── conftest.py                     # pytest fixtures (mock_settings, mock_hardware, etc.)
│   ├── test_speech.py                  # Tests for AudioListener, SpeechToText, LanguageDetector
│   ├── test_llm.py                     # Tests for LLMClient, PromptManager, ResponseHandler
│   ├── test_tts.py                     # Tests for TextToSpeechSynthesizer, Speaker
│   ├── test_hardware.py                # Tests for HardwareController, EyeController, etc.
│   ├── test_config.py                  # Tests for Settings, paths
│   └── test_utils.py                   # Tests for constants, logger, helpers
├── scripts/
│   ├── setup.sh                        # Initial environment setup on Raspberry Pi
│   ├── install.sh                      # Model file installation
│   └── run.sh                          # Start the AI Teacher Robot application
├── models/                             # Downloaded/quantized model files (gitignored)
│   ├── .gitkeep
│   ├── stt/                            # STT model directory
│   ├── tts/                            # TTS model directory
│   └── llm/                            # LLM model directory
├── assets/
│   ├── images/.gitkeep                 # Image assets
│   ├── sounds/.gitkeep                 # Sound effect files
│   └── fonts/.gitkeep                  # Font files (e.g., Urdu fonts)
└── logs/
    └── .gitkeep                        # Runtime log files (gitignored)
```

### Folder and File Purposes

| Path | Purpose |
|---|---|
| `src/` | Root source package containing all application code |
| `src/main.py` | Application entry point; orchestrates the conversation loop |
| `src/config/` | Configuration management (YAML + env vars, Singleton pattern) |
| `src/speech/` | Audio capture, speech-to-text, language detection |
| `src/llm/` | LLM API client, prompt management, response processing |
| `src/tts/` | Text-to-speech synthesis and audio playback |
| `src/hardware/` | GPIO/servo control for robot animations (eyes, mouth) |
| `src/utils/` | Shared utilities (constants, logging, helpers) |
| `docs/` | Documentation (SRS, Architecture, Research) |
| `tests/` | Unit and integration tests (118 tests, all passing) |
| `scripts/` | Setup, install, and run scripts for Raspberry Pi |
| `models/` | Downloaded model files (STT, TTS, LLM) — gitignored |
| `assets/` | Static assets (images, sounds, fonts) — gitignored contents |
| `logs/` | Runtime log files — gitignored contents |
| `pyproject.toml` | Build system, tool configs (black, ruff, mypy, pytest, coverage) |
| `setup.py` | Package installation script (pip install -e .) |
| `requirements.txt` | Production dependencies |
| `requirements-dev.txt` | Development dependencies |
| `.pre-commit-config.yaml` | Pre-commit hooks for code quality |
| `.env.example` | Template for environment variable overrides |
| `.gitignore` | Excludes Python/Raspberry Pi/OS-specific files |

---

## 2. Module Overview

### 2.1 `src/main.py` — Application Entry Point

| Attribute | Detail |
|---|---|
| **Purpose** | Main orchestrator coordinating the end-to-end conversation flow |
| **Responsibilities** | Initialize all subsystems, run the conversation loop, handle shutdown |
| **Main files** | `main.py` (AITeacherRobot class, main() function) |
| **Status** | **Placeholder** — `initialize()`, `run()`, and `shutdown()` contain TODO comments; no actual subsystem wiring |
| **Dependencies** | `config.settings.Settings`, `utils.constants`, `utils.logger` |

### 2.2 `src/config/` — Configuration Module

| Attribute | Detail |
|---|---|
| **Purpose** | Centralized configuration management from YAML and environment variables |
| **Responsibilities** | Load YAML config, override with env vars, provide typed access to all settings |
| **Main files** | `settings.py` (Settings class, Singleton), `paths.py` (path constants), `config.yaml` (default config) |
| **Status** | **Complete** — Settings class fully implemented with YAML loading, env var override, Singleton pattern, and `get_system_prompt()` method |
| **Dependencies** | `utils.constants`, `utils.logger`, PyYAML, python-dotenv |

### 2.3 `src/speech/` — Speech Module

| Attribute | Detail |
|---|---|
| **Purpose** | Audio capture, speech-to-text conversion, and language detection |
| **Responsibilities** | Microphone input, STT transcription, English/Urdu language detection |
| **Main files** | `listener.py` (AudioListener), `stt.py` (SpeechToText), `language_detector.py` (LanguageDetector) |
| **Status** | **Partial** — LanguageDetector fully implemented (script-based detection). AudioListener and SpeechToText have class structure and method signatures but audio capture and STT are TODO placeholders |
| **Dependencies** | `config.settings.Settings`, `utils.constants`, `utils.logger` |

### 2.4 `src/llm/` — LLM Module

| Attribute | Detail |
|---|---|
| **Purpose** | Communication with the local LLM and prompt/response management |
| **Responsibilities** | Send prompts to LLM API, build language-specific prompts, clean/validate responses |
| **Main files** | `client.py` (LLMClient), `prompt_manager.py` (PromptManager), `response_handler.py` (ResponseHandler) |
| **Status** | **Partial** — LLMClient has HTTP client structure and response extraction logic (supports Ollama, OpenAI, llama.cpp formats) but `send()` method is not fully wired to actual API calls. PromptManager and ResponseHandler are fully implemented |
| **Dependencies** | `config.settings.Settings`, `utils.constants`, `utils.helpers`, `utils.logger`, `requests` |

### 2.5 `src/tts/` — Text-to-Speech Module

| Attribute | Detail |
|---|---|
| **Purpose** | Convert text to speech and play it through a speaker |
| **Responsibilities** | Text synthesis, audio playback, volume/rate/voice control |
| **Main files** | `synthesizer.py` (TextToSpeechSynthesizer), `speaker.py` (Speaker) |
| **Status** | **Partial** — Class structures, method signatures, and validation logic are implemented. Actual TTS engine initialization and audio synthesis/playback are TODO placeholders |
| **Dependencies** | `config.settings.Settings`, `utils.constants`, `utils.logger` |

### 2.6 `src/hardware/` — Hardware Module

| Attribute | Detail |
|---|---|
| **Purpose** | Control robot physical features (eyes, mouth) via GPIO and PWM |
| **Responsibilities** | Servo control, eye animations, mouth animations, predefined animation sequences |
| **Main files** | `controller.py` (HardwareController), `eyes.py` (EyeController), `mouth.py` (MouthController), `animations.py` (AnimationController) |
| **Status** | **Partial** — Class structures, angle validation, PWM conversion math, and animation method signatures are implemented. Actual GPIO/PWM initialization and servo control are TODO placeholders |
| **Dependencies** | `config.settings.Settings`, `utils.logger` |

### 2.7 `src/utils/` — Utilities Module

| Attribute | Detail |
|---|---|
| **Purpose** | Shared utilities used across all modules |
| **Responsibilities** | Constants, logging, helper functions |
| **Main files** | `constants.py` (APP_NAME, SUPPORTED_LANGUAGES, GPIO pins, etc.), `logger.py` (Loguru setup), `helpers.py` (timer, truncate, merge, safe_get, etc.) |
| **Status** | **Complete** — All utilities fully implemented and tested |
| **Dependencies** | `loguru` |

---

## 3. Data Flow

### Expected End-to-End Flow

```
Student Voice
    │
    ▼
┌──────────────────┐     (Placeholder — PyAudio capture with VAD)
│  speech/listener │
│  AudioListener   │
    │
    ▼
┌──────────────────┐     (Placeholder — Vosk/Whisper STT)
│  speech/stt      │
│  SpeechToText    │
    │
    ▼
┌──────────────────────────┐  (Implemented — script-based detection)
│  speech/language_        │
│  detector.py             │
│  LanguageDetector        │
    │
    ▼
┌──────────────────┐     (Partial — HTTP client structure exists)
│  llm/client.py   │
│  LLMClient       │
    │
    ▼
┌──────────────────┐     (Implemented — prompt building)
│  llm/prompt_     │
│  manager.py      │
│  PromptManager   │
    │
    ▼
┌──────────────────┐     (Implemented — response cleaning/validation)
│  llm/response_   │
│  handler.py      │
│  ResponseHandler │
    │
    ▼
┌──────────────────┐     (Placeholder — pyttsx3/Coqui synthesis)
│  tts/synthesizer │
│  TextToSpeech    │
    │
    ▼
┌──────────────────┐     (Placeholder — PyAudio playback)
│  tts/speaker.py  │
│  Speaker         │
    │
    ▼
┌──────────────────┐
│  Speaker (HW)    │
    │
    ▼
┌──────────────────┐     (Placeholder — GPIO/PWM servo control)
│  hardware/       │
│  animations.py   │
│  AnimationController│
    │
    ▼
┌──────────────────┐
│  Robot Face      │
│  (Eyes + Mouth)  │
└──────────────────┘
```

### Module Status Summary

| Module | Data Flow Stage | Current Status |
|---|---|---|
| `speech/listener.py` | Audio capture | Placeholder (PyAudio TODO) |
| `speech/stt.py` | Speech-to-Text | Partial (class structure, no STT engine) |
| `speech/language_detector.py` | Language detection | **Complete** (script-based) |
| `llm/prompt_manager.py` | Prompt building | **Complete** |
| `llm/client.py` | LLM API communication | Partial (HTTP structure, no actual send) |
| `llm/response_handler.py` | Response cleaning | **Complete** |
| `tts/synthesizer.py` | Text-to-speech | Partial (class structure, no TTS engine) |
| `tts/speaker.py` | Audio playback | Partial (class structure, no PyAudio) |
| `hardware/animations.py` | Robot animation | Partial (method signatures, no GPIO) |
| `main.py` | Orchestration | Placeholder (TODO comments) |

### Missing Data Flow Stages

The following stages are referenced in the data flow but have **no corresponding module**:

- **Wake Word Detection** — No module exists for detecting a wake word (e.g., "Hey Teacher")
- **Voice Activity Detection (VAD)** — No dedicated VAD module; the `AudioListener.listen()` method mentions VAD in its docstring but does not implement it

---

## 4. Configuration

### 4.1 Configuration Files

| File | Description |
|---|---|
| `src/config/config.yaml` | Default YAML configuration with sections for: app, audio, stt, language_detection, llm, tts, hardware, logging |
| `.env.example` | Template for environment variable overrides (APP_DEBUG, AUDIO_SAMPLE_RATE, STT_ENGINE, LLM_API_URL, TTS_ENGINE, LOG_LEVEL, etc.) |
| `.env` | Actual environment variables (not committed — gitignored) |

### 4.2 Configuration Loading Hierarchy

The `Settings` class (Singleton pattern) loads configuration in this order of precedence:

1. **Environment variables** (highest priority) — checked via `os.getenv()`
2. **YAML config file** (`src/config/config.yaml`) — loaded via `yaml.safe_load()`
3. **Default values** (lowest priority) — defined in `src/utils/constants.py`

### 4.3 Key Configuration Sections

| Section | Key Settings |
|---|---|
| `app` | name, version, debug |
| `audio` | sample_rate (16000), chunk_size (1024), channels (1), device_index |
| `stt` | engine ("vosk"), model_path ("models/stt"), language ("en"), timeout (5) |
| `language_detection` | enabled (true), default_language ("en") |
| `llm` | api_url ("http://localhost:11434/api/chat"), model ("llama3"), timeout (30), max_tokens (500), temperature (0.7), system_prompt_en, system_prompt_ur |
| `tts` | engine ("pyttsx3"), language ("en"), rate (200), volume (1.0), voice_id |
| `hardware` | enabled (true), eye_left_gpio (18), eye_right_gpio (19), mouth_gpio (13), pwm_frequency (50), animation_speed (1.0) |
| `logging` | level ("INFO"), log_file ("logs/ai_teacher_robot.log"), rotation ("10 MB"), retention ("7 days") |

### 4.4 Model Locations

| Model Type | Directory | Description |
|---|---|---|
| STT models | `models/stt/` | Speech-to-text model files (Vosk, Whisper) |
| TTS models | `models/tts/` | Text-to-speech model files (Coqui, etc.) |
| LLM models | `models/llm/` | LLM model files (GGUF quantized) |

These directories are created by `src/config/paths.py:ensure_directories()` at runtime.

### 4.5 Log Locations

- **Log file:** `logs/ai_teacher_robot.log`
- **Rotation:** 10 MB per file
- **Retention:** 7 days
- **Format:** `YYYY-MM-DD HH:mm:ss | LEVEL | module:function:line | message`
- **Console:** Colored output to stderr via Loguru

### 4.6 Scripts

| Script | Purpose |
|---|---|
| `scripts/setup.sh` | Initial environment setup: apt-get update, install system deps (Python, PortAudio, ffmpeg), create venv, install pip packages, create directories, install pre-commit hooks |
| `scripts/install.sh` | Model file installation: download Vosk STT models, install TTS voices (TODO placeholders) |
| `scripts/run.sh` | Start the application: activate venv, run `python -m src.main` |

---

## 5. External Dependencies

### 5.1 Python Packages (Production — `requirements.txt`)

| Package | Version | Purpose | Status |
|---|---|---|---|
| PyYAML | >=6.0,<7.0 | YAML configuration parsing | Installed |
| loguru | >=0.7,<0.8 | Structured logging | Installed |
| python-dotenv | >=1.0,<2.0 | Environment variable management | Installed |
| requests | >=2.31,<3.0 | HTTP client for LLM API | Installed |
| python-dateutil | >=2.8,<3.0 | Date/time handling | Installed |
| vosk | (commented) | Speech recognition (STT) | **Not installed** |
| pyttsx3 | (commented) | Text-to-speech (TTS) | **Not installed** |
| RPi.GPIO | (commented) | GPIO control for Raspberry Pi | **Not installed** |
| pigpio | (commented) | PWM/servo control | **Not installed** |

### 5.2 Python Packages (Development — `requirements-dev.txt`)

| Package | Version | Purpose |
|---|---|---|
| pytest | >=7.4,<8.0 | Testing framework |
| pytest-cov | >=4.1,<5.0 | Test coverage |
| pytest-mock | >=3.11,<4.0 | Mocking for tests |
| ruff | >=0.1.0,<0.2.0 | Linter |
| black | >=23.0,<24.0 | Code formatter |
| isort | >=5.12,<6.0 | Import sorter |
| mypy | >=1.5,<2.0 | Type checker |
| Sphinx | >=7.0,<8.0 | Documentation generator |
| sphinx-rtd-theme | >=1.3,<2.0 | Sphinx theme |
| pre-commit | >=3.3,<4.0 | Pre-commit hooks |

### 5.3 AI Models

| Model | Engine | Language | Status |
|---|---|---|---|
| Vosk English model | Vosk | English | **Not downloaded** (install.sh has TODO) |
| Vosk Urdu model | Vosk | Urdu | **Not downloaded** (install.sh has TODO) |
| LLM model (e.g., llama3) | Ollama/llama.cpp | Multilingual | **Not downloaded** |
| TTS voices | pyttsx3/Coqui | English, Urdu | **Not installed** |

### 5.4 Raspberry Pi Dependencies

| Dependency | Purpose | Status |
|---|---|---|
| Raspberry Pi OS (64-bit) | Operating system | **Not configured** |
| Python 3.11+ | Runtime | **Not configured** |
| RPi.GPIO | GPIO pin control | **Not installed** |
| pigpio | PWM/servo control | **Not installed** |
| PortAudio | Audio I/O | **Not installed** |
| ffmpeg | Audio processing | **Not installed** |

### 5.5 Hardware Libraries

| Library | Purpose | Status |
|---|---|---|
| RPi.GPIO | GPIO pin setup and control | **Not installed** (commented in requirements) |
| pigpio | PWM servo control | **Not installed** (commented in requirements) |

### 5.6 Audio Libraries

| Library | Purpose | Status |
|---|---|---|
| PyAudio | Microphone input and speaker output | **Not installed** (not in requirements) |
| PortAudio | System-level audio I/O | **Not installed** (system dependency in setup.sh) |

---

## 6. Missing Components

### 6.1 Missing Modules

| Missing Module | Purpose | Suggested Location |
|---|---|---|
| Wake word detection | Detect a wake word (e.g., "Hey Teacher") to start listening | `src/speech/wake_word.py` |
| Voice Activity Detection (VAD) | Detect when speech starts and ends in audio stream | `src/speech/vad.py` or integrate into `listener.py` |
| Audio recording/playback | PyAudio wrapper for microphone and speaker | `src/audio/` package |
| Conversation manager | Maintain conversation history and context | `src/conversation.py` or `src/llm/conversation.py` |
| Error handling module | Centralized error handling and recovery | `src/utils/errors.py` |
| Internationalization (i18n) | UI text localization | `src/i18n/` |

### 6.2 Missing Documentation

| Missing Document | Purpose |
|---|---|
| `docs/DEPLOYMENT.md` | Deployment guide for Raspberry Pi |
| `docs/API.md` | API reference (auto-generated via Sphinx) |
| `docs/TROUBLESHOOTING.md` | Common issues and solutions |
| `docs/CODE_OF_CONDUCT.md` | Code of conduct for contributors |
| `docs/PR_TEMPLATE.md` | Pull request template |
| `docs/ISSUE_TEMPLATE.md` | Issue template |

### 6.3 Missing Scripts

| Missing Script | Purpose |
|---|---|
| `scripts/download_models.sh` | Download all required AI models |
| `scripts/test.sh` | Run tests with coverage |
| `scripts/lint.sh` | Run all linters and formatters |
| `scripts/deploy.sh` | Deploy to Raspberry Pi |
| `scripts/start_llm.sh` | Start the local LLM server (Ollama/llama.cpp) |
| `scripts/calibrate_servos.sh` | Calibrate servo motors |

### 6.4 Missing Configuration

| Missing Config | Purpose |
|---|---|
| `models/llm/` model files | LLM model files (GGUF) |
| `models/stt/` model files | STT model files (Vosk) |
| `models/tts/` model files | TTS model files |
| `assets/fonts/` Urdu fonts | Urdu font files for display |
| `assets/sounds/` sound effects | UI sound effects |
| `assets/images/` robot images | Robot face/image assets |
| Systemd service file | Auto-start on boot (`ai-teacher-robot.service`) |
| Dockerfile | Container-based deployment |

### 6.5 Missing Testing

| Missing Test Coverage | Purpose |
|---|---|
| Integration tests | End-to-end flow tests (STT → LLM → TTS) |
| Hardware tests | Physical servo/GPIO tests (marked with `@pytest.mark.hardware`) |
| Performance tests | Response time and latency tests |
| Stress tests | Long-running stability tests |
| Mock LLM server | Test LLM client without a real LLM |
| Audio test fixtures | Sample audio files for STT testing |

### 6.6 Missing Deployment Files

| Missing File | Purpose |
|---|---|
| `Dockerfile` | Container-based deployment |
| `docker-compose.yml` | Multi-container deployment |
| `ai-teacher-robot.service` | systemd service for auto-start |
| `Makefile` | Common development commands |
| `.github/workflows/ci.yml` | CI pipeline (GitHub Actions) |
| `.github/workflows/release.yml` | Release pipeline |
| `.github/ISSUE_TEMPLATE/` | Issue templates |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR template |

---

## 7. Suggested Improvements

### 7.1 Structural Improvements

1. **Add an `audio/` package** — Create a dedicated `src/audio/` package for PyAudio wrappers, separating audio I/O from speech processing. This would include `recorder.py`, `player.py`, and `device.py`.

2. **Add a `conversation/` module** — Create `src/conversation.py` or `src/llm/conversation.py` to manage conversation history, context, and session state. This would track the dialogue between the student and the robot.

3. **Add a `wake_word/` module** — Create `src/speech/wake_word.py` for wake word detection, allowing the robot to listen for a specific phrase before activating.

4. **Add an `i18n/` module** — Create `src/i18n/` for internationalization, supporting UI text in both English and Urdu.

5. **Add a `models/` manager** — Create `src/models/manager.py` to handle model downloading, caching, and lifecycle management.

### 7.2 Configuration Improvements

1. **Add config validation** — Use Pydantic or JSON Schema to validate the YAML configuration at load time, providing clear error messages for invalid values.

2. **Add config hot-reload** — Support reloading configuration without restarting the application (useful for adjusting settings on the Raspberry Pi).

3. **Add per-environment configs** — Support `config.dev.yaml`, `config.prod.yaml` for different deployment environments.

### 7.3 Testing Improvements

1. **Add integration tests** — Create `tests/integration/` for end-to-end flow tests that verify the complete pipeline.

2. **Add hardware test markers** — Use `@pytest.mark.hardware` for tests that require physical hardware, and exclude them from CI runs.

3. **Add audio fixtures** — Create sample audio files in `tests/fixtures/` for STT testing.

4. **Add mock LLM server** — Create a mock LLM server for testing the LLM client without a real LLM.

### 7.4 Documentation Improvements

1. **Add API documentation** — Use Sphinx to auto-generate API documentation from docstrings.

2. **Add deployment guide** — Create `docs/DEPLOYMENT.md` with step-by-step Raspberry Pi setup instructions.

3. **Add troubleshooting guide** — Create `docs/TROUBLESHOOTING.md` for common issues.

### 7.5 DevOps Improvements

1. **Add CI/CD pipeline** — Create GitHub Actions workflows for CI (lint, test) and CD (release).

2. **Add Dockerfile** — Create a Dockerfile for containerized deployment.

3. **Add systemd service** — Create a systemd service file for auto-starting on boot.

4. **Add Makefile** — Create a Makefile for common development commands (test, lint, format, run).

---

## 8. Development Roadmap

### Phase 1: Foundation (Current — Completed)

- [x] Project structure and scaffolding
- [x] Configuration management (Settings, paths, config.yaml)
- [x] Utilities (constants, logger, helpers)
- [x] Test framework (pytest, conftest, 118 tests)
- [x] Documentation (README, CONTRIBUTING, CHANGELOG, SRS, Architecture, Research)
- [x] Build and tooling (setup.py, pyproject.toml, pre-commit)

### Phase 2: Core Modules (Next)

**Priority order based on data flow dependencies:**

1. **Language Detector** (already complete) → **Audio Listener**
   - Implement PyAudio microphone capture
   - Add Voice Activity Detection (VAD)
   - Add wake word detection module

2. **Speech-to-Text** → **LLM Client**
   - Integrate Vosk STT engine
   - Download and configure STT models
   - Implement `LLMClient.send()` with actual HTTP requests
   - Set up local LLM server (Ollama or llama.cpp)

3. **Prompt Manager & Response Handler** (already complete) → **Text-to-Speech**
   - Integrate pyttsx3 or Coqui TTS engine
   - Download and configure TTS models/voices
   - Implement audio playback via PyAudio

4. **Speaker** → **Hardware Controller**
   - Integrate RPi.GPIO for GPIO control
   - Implement PWM servo control
   - Test with physical servos

5. **Animation Controller** → **Main Orchestrator**
   - Wire all modules together in `main.py`
   - Implement the conversation loop
   - Add error handling and recovery

### Phase 3: Enhancement

1. **Conversation management** — Track conversation history and context
2. **Multilingual improvements** — Better Urdu support for STT and TTS
3. **Performance optimization** — Optimize for Raspberry Pi CPU/memory
4. **UI/UX improvements** — Add visual feedback (LEDs, display)
5. **Deployment** — systemd service, Dockerfile, CI/CD

### Phase 4: Production

1. **Comprehensive testing** — Integration, stress, and hardware tests
2. **Documentation** — API docs, deployment guide, troubleshooting
3. **Monitoring** — Health checks, metrics, alerting
4. **Security** — Audit, secure configuration, data privacy
5. **Release** — Version tagging, release notes, distribution

---

*This architectural overview was generated by inspecting the existing project structure and source code. No files were modified during this analysis.*
