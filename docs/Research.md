# Research

**Project:** AI Teacher Robot
**Version:** 0.1.0 (Draft)
**Date:** 2026-07-25

---

## Overview

This document captures research findings, technology evaluations, and
references that inform the design and implementation of the AI Teacher Robot.

---

## 1. Speech-to-Text (STT) Engines

### 1.1 Vosk

- **Pros:**
  - Runs entirely offline.
  - Lightweight and suitable for Raspberry Pi.
  - Supports multiple languages including Urdu.
  - Small model sizes (50 MB for English, similar for Urdu).
- **Cons:**
  - Accuracy may be lower than cloud-based alternatives.
  - Limited continuous recognition without additional logic.
- **Status:** **Recommended** for initial implementation.

### 1.2 Whisper (whisper.cpp)

- **Pros:**
  - High accuracy.
  - Supports many languages including Urdu.
  - Can run on Raspberry Pi with quantized models.
- **Cons:**
  - Larger model sizes (even quantized).
  - Slower inference on Raspberry Pi CPU.
- **Status:** **Consider for future enhancement** if accuracy is critical.

### 1.3 Picovoice Leopard / Cheetah

- **Pros:**
  - Commercial-grade accuracy.
  - Designed for edge devices.
- **Cons:**
  - Requires a license key for production use.
  - May have limited Urdu support.
- **Status:** **Not recommended** due to licensing costs.

### 1.4 Recommendation

Use **Vosk** for the initial implementation due to its offline capability,
small footprint, and Urdu language support. Whisper.cpp can be evaluated as a
fallback if accuracy is insufficient.

---

## 2. Text-to-Speech (TTS) Engines

### 2.1 pyttsx3

- **Pros:**
  - Works offline.
  - Cross-platform.
  - Simple API.
- **Cons:**
  - Limited naturalness.
  - Urdu language support is basic (depends on system voices).
- **Status:** **Recommended for initial prototype.**

### 2.2 Coqui TTS

- **Pros:**
  - High-quality neural TTS.
  - Supports many languages.
  - Can run on edge devices with quantized models.
- **Cons:**
  - Larger model sizes.
  - May be slow on Raspberry Pi CPU.
- **Status:** **Consider for future enhancement.**

### 2.3 eSpeak NG

- **Pros:**
  - Very lightweight.
  - Supports Urdu.
- **Cons:**
  - Robotic-sounding output.
- **Status:** **Fallback option.**

### 2.4 Recommendation

Start with **pyttsx3** for rapid prototyping. Evaluate **Coqui TTS** or
**eSpeak NG** for better Urdu support if needed.

---

## 3. Local LLM Options

### 3.1 Ollama

- **Pros:**
  - Easy to install and manage models.
  - REST API for communication.
  - Supports GGUF quantized models.
- **Cons:**
  - May not be officially supported on Raspberry Pi OS.
  - Memory usage can be high.
- **Status:** **Recommended** for LLM serving.

### 3.2 llama.cpp (server mode)

- **Pros:**
  - Lightweight and optimized for CPU.
  - Direct GGUF support.
  - Can run on Raspberry Pi.
- **Cons:**
  - Requires manual model management.
  - No built-in model management API.
- **Status:** **Recommended** as an alternative to Ollama.

### 3.3 GPT4All

- **Pros:**
  - Pre-quantized models.
  - Simple API.
- **Cons:**
  - Limited model selection.
  - May not be optimized for Raspberry Pi.
- **Status:** **Consider for evaluation.**

### 3.4 Recommendation

Use **llama.cpp** in server mode or **Ollama** for serving the local LLM.
Both support GGUF quantized models that can run on a Raspberry Pi 4 with
4 GB+ RAM.

---

## 4. Hardware Components

### 4.1 Servo Motors

- **Type:** Micro servo motors (e.g., SG90).
- **Quantity:**
  - 2 servos for eyes (left and right).
  - 1 servo for mouth.
- **Control:** PWM via GPIO using `RPi.GPIO` or `pigpio`.

### 4.2 Microphone

- **Type:** USB microphone or I2S microphone (e.g., ReSpeaker 2-Mic Array).
- **Recommendation:** USB microphone for simplicity.

### 4.3 Speaker

- **Type:** USB speaker or I2S amplifier + speaker.
- **Recommendation:** USB speaker for simplicity.

### 4.4 Raspberry Pi

- **Model:** Raspberry Pi 4 Model B (4 GB or 8 GB RAM).
- **OS:** Raspberry Pi OS (64-bit).
- **Power:** 5V/3A power adapter.

---

## 5. Multilingual Support (English + Urdu)

### 5.1 English

- Well-supported by all STT and TTS engines.
- No special considerations needed.

### 5.2 Urdu

- **STT:** Vosk has Urdu language models available.
- **TTS:** pyttsx3 may require Urdu system voices. eSpeak NG has basic Urdu
  support. Coqui TTS has Urdu models.
- **LLM:** Most LLMs support Urdu text natively.
- **Font:** Ensure the system has Urdu font support (e.g., Nafees, Jameel Noori
  Nastaleeq).

---

## 6. References

- Vosk API: https://alphacephei.com/vosk/
- Whisper.cpp: https://github.com/ggerganov/whisper.cpp
- Ollama: https://ollama.com/
- llama.cpp: https://github.com/ggerganov/llama.cpp
- pyttsx3: https://github.com/nateshmbhat/pyttsx3
- Coqui TTS: https://github.com/coqui-ai/TTS
- RPi.GPIO: https://pypi.org/project/RPi.GPIO/
- pigpio: http://abyz.me.uk/rpi/pigpio/

---

*This research document is a living document and will be updated as new
technologies and findings emerge.*
