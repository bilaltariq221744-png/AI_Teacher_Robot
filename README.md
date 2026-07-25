# AI Teacher Robot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Platform: Raspberry Pi OS](https://img.shields.io/badge/platform-Raspberry%20Pi%20OS-red.svg)](https://www.raspberrypi.org/software/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](#contributing)
[![Code of Conduct](https://img.shields.io/badge/%E2%80%9Chey%20%E2%80%93%20be%20kind%20to%20your%20mind%E2%80%9D-brightgreen.svg)](CODE_OF_CONDUCT.md)

> **An AI-powered educational robot for students in Grade 2 through Grade 10, running entirely
> locally on a Raspberry Pi.**

---

## 🎯 Overview

The **AI Teacher Robot** is a locally-running, voice-interactive educational companion designed to
help students from **Grade 2 to Grade 10** with their studies. The robot listens to a student's
spoken questions (in **English** or **Urdu**), converts the speech to text, sends it to an
on-device or locally-hosted Large Language Model (LLM), converts the LLM's response back to speech,
and speaks the answer aloud — all while animating physical robot features such as moving eyes and
a moving mouth.

Because everything runs **locally on a Raspberry Pi**, the system respects student privacy: no
audio or text data ever leaves the device.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multilingual Voice Input** | Supports **English** and **Urdu** speech recognition. |
| **Speech-to-Text (STT)** | Converts student voice input into text using a local or lightweight STT engine. |
| **Local LLM Inference** | Sends transcribed text to a locally-hosted LLM for educational responses. |
| **Text-to-Speech (TTS)** | Converts the LLM response back into natural-sounding speech. |
| **Robot Animations** | Controls servo-driven eyes and mouth to animate the robot's face. |
| **Offline-First** | Designed to run entirely on a Raspberry Pi without cloud dependencies. |
| **Modular Architecture** | Clean, scalable Python package structure for easy extension and maintenance. |

---

## 🏗️ Project Structure

```
AI_Teacher_Robot/
├── .gitignore
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   ├── speech/              # Speech-to-text & language detection
│   ├── llm/                 # LLM client & prompt management
│   ├── tts/                 # Text-to-speech synthesis
│   ├── hardware/            # Robot animation & servo control
│   ├── config/              # Configuration management
│   └── utils/               # Shared utilities & helpers
├── docs/
│   ├── SRS.md               # Software Requirements Specification
│   ├── Architecture.md      # System architecture & design
│   └── Research.md          # Research notes & references
├── models/                  # Downloaded / quantized model files
├── tests/                   # Unit & integration tests
├── assets/                  # Images, sounds, fonts
├── scripts/                 # Setup, install & run scripts
└── logs/                    # Runtime log files
```

---

## 🚀 Quick Start

> **Prerequisites:** Raspberry Pi 4 (4 GB+ RAM recommended), Raspberry Pi OS (64-bit), Python 3.11+.

### 1. Clone the Repository

```bash
git clone https://github.com/bilaltariq221744-png/AI_Teacher_Robot.git
cd AI_Teacher_Robot
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional: for development
```

### 3. Run the Robot

```bash
python -m src.main
```

---

## 🛠️ Development

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our coding guidelines, branch
naming conventions, and the pull-request process.

### Useful Scripts

| Script | Description |
|---|---|
| `scripts/setup.sh` | Initial environment setup on a fresh Raspberry Pi. |
| `scripts/install.sh` | Install Python dependencies and system packages. |
| `scripts/run.sh` | Start the AI Teacher Robot application. |

---

## 📚 Documentation

- **[Software Requirements Specification (SRS)](docs/SRS.md)** — Detailed functional and
  non-functional requirements.
- **[Architecture](docs/Architecture.md)** — System design, component diagram, and data flow.
- **[Research](docs/Research.md)** — Research notes on STT/TTS/LLM engines and hardware choices.

---

## 🤝 Contributing

Contributions are welcome! Please read our
[CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📄 License

This project is licensed under the MIT License — see the
[LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Project Lead:** Bilal Tariq
- **Repository:** [github.com/bilaltariq221744-png/AI_Teacher_Robot](https://github.com/bilaltariq221744-png/AI_Teacher_Robot)
- **Issues:** [GitHub Issues](https://github.com/bilaltariq221744-png/AI_Teacher_Robot/issues)

---

*Made with ❤️ for students everywhere.*
