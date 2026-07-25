# Software Requirements Specification (SRS)

**Project:** AI Teacher Robot
**Version:** 0.1.0 (Draft)
**Date:** 2026-07-25
**Author:** Bilal Tariq

---

## 1. Introduction

### 1.1 Purpose

This document describes the **Software Requirements Specification (SRS)** for the
**AI Teacher Robot** — an AI-powered educational robot that runs locally on a
Raspberry Pi and provides voice-interactive tutoring for students in Grade 2
through Grade 10.

The purpose of this SRS is to define the functional and non-functional
requirements, system interfaces, user characteristics, and constraints that the
development team will use to build the system.

### 1.2 Scope

The AI Teacher Robot listens to a student's spoken question (in English or
Urdu), converts it to text, sends it to a locally-hosted Large Language Model
(LLM), converts the LLM's response back to speech, and speaks the answer
through a speaker — all while animating physical robot features (moving eyes
and mouth).

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|---|---|
| **STT** | Speech-to-Text |
| **TTS** | Text-to-Speech |
| **LLM** | Large Language Model |
| **GPIO** | General Purpose Input/Output |
| **PWM** | Pulse Width Modulation |
| **Pi** | Raspberry Pi |
| **SRS** | Software Requirements Specification |

### 1.4 References

- IEEE Std 830-1998 — IEEE Recommended Practice for Software Requirements
  Specification.
- Raspberry Pi 4 Model B Documentation.
- Python 3.11 Language Reference.

### 1.5 Overview

This SRS is organized as follows:
- Section 2: Overall Description
- Section 3: Specific Requirements
- Section 4: Appendices

---

## 2. Overall Description

### 2.1 Product Perspective

The AI Teacher Robot is a standalone embedded application that runs on a
Raspberry Pi. It integrates the following subsystems:

1. **Speech Recognition (STT)** — Captures audio from a microphone and converts
   it to text.
2. **Language Detection** — Identifies whether the input is English or Urdu.
3. **LLM Client** — Sends the transcribed text to a locally-hosted LLM and
   receives a response.
4. **Text-to-Speech (TTS)** — Converts the LLM response to audio.
5. **Hardware Controller** — Drives servo motors to animate the robot's eyes
   and mouth.

### 2.2 Product Functions

- Listen to voice input from students.
- Detect the language of the input (English or Urdu).
- Convert speech to text.
- Send text to the LLM and receive a response.
- Convert the response text to speech.
- Speak the response through a speaker.
- Animate the robot's eyes and mouth during speech.

### 2.3 User Characteristics

- **Primary Users:** Students in Grade 2–10.
- **Secondary Users:** Teachers and parents (for setup and configuration).
- **Technical Users:** Developers and maintainers of the system.

### 2.4 Constraints

- Must run on a Raspberry Pi 4 (4 GB+ RAM recommended).
- Must operate offline (no cloud dependencies).
- Must support English and Urdu languages.
- Must use Python 3.11+.
- Must not store or transmit student data externally.

### 2.5 Assumptions and Dependencies

- The Raspberry Pi has a microphone and speaker connected.
- Servo motors are connected to GPIO pins for eye and mouth animation.
- A locally-hosted LLM is available (e.g., via Ollama or llama.cpp).

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### 3.1.1 FR-01: Voice Input

- **Description:** The system shall capture audio input from a microphone.
- **Priority:** High
- **Source:** User requirement

#### 3.1.2 FR-02: Language Detection

- **Description:** The system shall detect whether the spoken language is
  English or Urdu.
- **Priority:** High
- **Source:** User requirement

#### 3.1.3 FR-03: Speech-to-Text

- **Description:** The system shall convert captured audio to text in the
  detected language.
- **Priority:** High
- **Source:** User requirement

#### 3.1.4 FR-04: LLM Communication

- **Description:** The system shall send the transcribed text to a locally-hosted
  LLM and receive a text response.
- **Priority:** High
- **Source:** User requirement

#### 3.1.5 FR-05: Text-to-Speech

- **Description:** The system shall convert the LLM response text to speech in
  the appropriate language.
- **Priority:** High
- **Source:** User requirement

#### 3.1.6 FR-06: Audio Output

- **Description:** The system shall play the generated speech through a speaker.
- **Priority:** High
- **Source:** User requirement

#### 3.1.7 FR-07: Robot Animations

- **Description:** The system shall animate the robot's eyes and mouth during
  speech output.
- **Priority:** Medium
- **Source:** User requirement

#### 3.1.8 FR-08: Configuration

- **Description:** The system shall load configuration from a YAML file and
  environment variables.
- **Priority:** Medium
- **Source:** Developer requirement

### 3.2 Non-Functional Requirements

#### 3.2.1 NFR-01: Performance

- **Description:** The system shall respond to a student's question within
  5 seconds (end-to-end) under normal conditions.
- **Priority:** High

#### 3.2.2 NFR-02: Reliability

- **Description:** The system shall handle errors gracefully and provide
  meaningful error messages.
- **Priority:** High

#### 3.2.3 NFR-03: Privacy

- **Description:** The system shall not transmit any student data to external
  servers.
- **Priority:** Critical

#### 3.2.4 NFR-04: Maintainability

- **Description:** The system shall follow a modular architecture with clear
  separation of concerns.
- **Priority:** High

#### 3.2.5 NFR-05: Portability

- **Description:** The system shall run on Raspberry Pi OS (64-bit) and be
  portable to other Linux-based ARM platforms.
- **Priority:** Medium

---

## 4. Appendices

### 4.1 Appendix A: Use Case Diagram (Placeholder)

> A visual use case diagram will be added here once the design phase begins.

### 4.2 Appendix B: Data Flow Diagram (Placeholder)

> A DFD illustrating the flow of data between STT, LLM, TTS, and Hardware
> modules will be added here.

---

*This document is a living document and will be updated as the project
evolves.*
