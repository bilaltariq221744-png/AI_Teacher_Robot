# Architecture

**Project:** AI Teacher Robot
**Version:** 0.1.0 (Draft)
**Date:** 2026-07-25

---

## 1. Overview

This document describes the **system architecture** for the AI Teacher Robot.
It covers the high-level design, component diagram, module responsibilities,
data flow, and key design decisions.

The AI Teacher Robot is a modular Python application designed to run on a
Raspberry Pi. It follows a **layered, modular architecture** with clear
separation of concerns between speech processing, LLM communication, TTS,
hardware control, configuration, and utilities.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Teacher Robot                         │
│                    (Raspberry Pi)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Speech  │    │   LLM    │    │   TTS    │             │
│  │  Module  │    │  Module  │    │  Module  │             │
│  │ (STT +   │    │ (Client  │    │ (Synth + │             │
│  │  Lang)   │    │  + Prompt)│    │  Speaker)│             │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘             │
│       │               │               │                   │
│       │ Transcribed   │ LLM           │ Audio             │
│       │ Text          │ Response      │ Samples           │
│       ▼               ▼               ▼                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Main Orchestrator (main.py)            │   │
│  │  Coordinates the flow between all modules          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Hardware    │    │   Config     │    │   Utils      │ │
│  │  Module      │    │  Module      │    │  Module      │ │
│  │ (Animations) │    │ (Settings)   │    │ (Logging,    │ │
│  └──────────────┘    └──────────────┘    │  Helpers)    │ │
│                                          └──────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Module Architecture

### 3.1 `src/main.py` — Application Entry Point

The **Main Orchestrator** coordinates the end-to-end flow:

1. Initialize all modules (config, logger, speech, LLM, TTS, hardware).
2. Enter the main listening loop.
3. Capture audio → STT → detect language → send to LLM → TTS → speak + animate.
4. Handle errors and log all events.

### 3.2 `src/speech/` — Speech Module

**Responsibility:** Capture audio, perform speech-to-text, and detect language.

| File | Responsibility |
|---|---|
| `listener.py` | Captures audio from the microphone using PyAudio or similar. |
| `stt.py` | Converts audio to text using a local STT engine (e.g., Vosk). |
| `language_detector.py` | Detects whether the input is English or Urdu. |

### 3.3 `src/llm/` — LLM Module

**Responsibility:** Communicate with the locally-hosted LLM and manage prompts.

| File | Responsibility |
|---|---|
| `client.py` | HTTP client that sends text to the LLM API and receives responses. |
| `prompt_manager.py` | Builds and manages prompts, including system prompts per language. |
| `response_handler.py` | Processes and validates LLM responses before passing to TTS. |

### 3.4 `src/tts/` — Text-to-Speech Module

**Responsibility:** Convert text to speech and play it through a speaker.

| File | Responsibility |
|---|---|
| `synthesizer.py` | Converts text to audio samples using a TTS engine. |
| `speaker.py` | Plays audio samples through the connected speaker. |

### 3.5 `src/hardware/` — Hardware Module

**Responsibility:** Control physical robot features (eyes, mouth) via GPIO.

| File | Responsibility |
|---|---|
| `controller.py` | Main hardware controller that manages all servos. |
| `eyes.py` | Controls the eye servos (e.g., blinking, looking around). |
| `mouth.py` | Controls the mouth servo to animate during speech. |
| `animations.py` | Predefined animation sequences (e.g., idle, listening, speaking). |

### 3.6 `src/config/` — Configuration Module

**Responsibility:** Manage application configuration from files and environment.

| File | Responsibility |
|---|---|
| `settings.py` | Loads and validates configuration from YAML and env vars. |
| `paths.py` | Defines all project file paths (models, assets, logs, etc.). |
| `config.yaml` | Default configuration file (YAML format). |

### 3.7 `src/utils/` — Utilities Module

**Responsibility:** Shared utilities used across all modules.

| File | Responsibility |
|---|---|
| `logger.py` | Configures structured logging using Loguru. |
| `helpers.py` | General-purpose helper functions (e.g., timing, formatting). |
| `constants.py` | Project-wide constants (e.g., supported languages, version). |

---

## 4. Data Flow

```
Student speaks
    │
    ▼
┌──────────────────┐
│  Microphone      │
└────────┬─────────┘
         │ Audio
         ▼
┌──────────────────┐
│  speech/listener │
└────────┬─────────┘
         │ Audio
         ▼
┌──────────────────┐
│  speech/stt      │
└────────┬─────────┘
         │ Transcribed Text
         ▼
┌──────────────────────────┐
│  speech/language_detector│
└────────┬─────────────────┘
         │ Language + Text
         ▼
┌──────────────────┐
│  llm/client      │
└────────┬─────────┘
         │ LLM Response Text
         ▼
┌──────────────────┐
│  llm/response_   │
│  handler         │
└────────┬─────────┘
         │ Processed Text + Language
         ▼
┌──────────────────┐
│  tts/synthesizer │
└────────┬─────────┘
         │ Audio Samples
         ▼
┌──────────────────┐
│  tts/speaker     │
└────────┬─────────┘
         │ Audio
         ▼
┌──────────────────┐
│  Speaker         │
└──────────────────┘

         │ (parallel)
         ▼
┌──────────────────┐
│  hardware/       │
│  animations      │
└──────────────────┘
         │ Servo Commands
         ▼
┌──────────────────┐
│  Robot Face      │
│  (Eyes + Mouth) │
└──────────────────┘
```

---

## 5. Design Patterns

| Pattern | Where Used | Purpose |
|---|---|---|
| **Singleton** | `config/settings.py` | Ensure a single configuration instance. |
| **Factory** | `tts/synthesizer.py`, `speech/stt.py` | Create engine instances based on config. |
| **Strategy** | `llm/prompt_manager.py` | Switch prompts based on language. |
| **Observer** | `hardware/animations.py` | React to speech events for animations. |
| **Facade** | `main.py` | Simplify the interface to all subsystems. |

---

## 6. Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **STT Engine** | Vosk (placeholder) |
| **LLM** | Local LLM via HTTP API (e.g., Ollama, llama.cpp) |
| **TTS Engine** | pyttsx3 or Coqui TTS (placeholder) |
| **Hardware** | RPi.GPIO / pigpio |
| **Config** | YAML + python-dotenv |
| **Logging** | Loguru |
| **Testing** | pytest, pytest-mock |
| **Linting** | ruff, black, isort, mypy |

---

## 7. Deployment

- The application runs as a systemd service on Raspberry Pi OS.
- Model files are stored in the `models/` directory.
- Logs are written to the `logs/` directory.
- Configuration is loaded from `src/config/config.yaml` and `.env`.

---

*This architecture document is a living document and will be updated as the
project evolves.*
