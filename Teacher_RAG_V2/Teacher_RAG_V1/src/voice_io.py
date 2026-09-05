import subprocess
from pathlib import Path

from config import (
    AUDIO_DIR,
    RECORDED_WAV,
    ANSWER_WAV,
    WHISPER_CPP_BIN,
    WHISPER_MODEL,
    PIPER_BIN,
    PIPER_MODEL,
    RECORD_SECONDS,
    AUDIO_DEVICE,
    PLAYER_COMMAND,
)


def ensure_audio_dir() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def record_audio(seconds: int = RECORD_SECONDS) -> Path:
    """
    Records microphone audio using arecord.
    Works on Raspberry Pi OS/Linux.
    """
    ensure_audio_dir()

    if RECORDED_WAV.exists():
        RECORDED_WAV.unlink()

    command = [
        "arecord",
        "-D",
        AUDIO_DEVICE,
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(seconds),
        str(RECORDED_WAV),
    ]

    print(f"Listening for {seconds} seconds...")
    subprocess.run(command, check=True)

    return RECORDED_WAV


def speech_to_text(audio_path: Path = RECORDED_WAV) -> str:
    """
    Uses whisper.cpp CLI to transcribe recorded audio.
    """
    output_base = AUDIO_DIR / "stt_output"
    output_txt = output_base.with_suffix(".txt")

    if output_txt.exists():
        output_txt.unlink()

    command = [
        WHISPER_CPP_BIN,
        "-m",
        WHISPER_MODEL,
        "-f",
        str(audio_path),
        "-nt",
        "-otxt",
        "-of",
        str(output_base),
    ]

    subprocess.run(command, check=True)

    if not output_txt.exists():
        raise RuntimeError("Whisper did not create transcript file.")

    text = output_txt.read_text(encoding="utf-8").strip()
    return clean_transcript(text)


def clean_transcript(text: str) -> str:
    """
    Cleans common whisper.cpp transcript output.
    """
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove timestamp-like lines if any appear.
        if line.startswith("[") and "-->" in line:
            continue

        lines.append(line)

    return " ".join(lines).strip()


def text_to_speech(text: str, output_path: Path = ANSWER_WAV) -> Path:
    """
    Uses Piper CLI to convert answer text to WAV.
    """
    ensure_audio_dir()

    if output_path.exists():
        output_path.unlink()

    command = [
        PIPER_BIN,
        "--model",
        PIPER_MODEL,
        "--output_file",
        str(output_path),
    ]

    subprocess.run(
        command,
        input=text,
        text=True,
        check=True,
    )

    return output_path


def play_audio(audio_path: Path = ANSWER_WAV) -> None:
    subprocess.run([PLAYER_COMMAND, str(audio_path)], check=True)


def speak(text: str) -> None:
    wav_path = text_to_speech(text)
    play_audio(wav_path)