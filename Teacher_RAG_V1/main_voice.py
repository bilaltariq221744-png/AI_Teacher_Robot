from pathlib import Path
import sys
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from src.rag_engine import TeacherRAG

from config import (
    AUDIO_DIR,
    RECORDED_WAV,
    ANSWER_WAV,
    PIPER_MODEL,
    RECORD_SECONDS,
)


# ---------------------------------------------------------
# Audio setup
# ---------------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1

# PD200X microphone
MIC_DEVICE = 1


def ensure_audio_dir() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Record microphone
# ---------------------------------------------------------

def record_audio(seconds: int = RECORD_SECONDS) -> Path:

    ensure_audio_dir()

    if RECORDED_WAV.exists():
        RECORDED_WAV.unlink()

    print(f"\nListening for {seconds} seconds...")
    print("Speak now...")

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=MIC_DEVICE,
    )

    sd.wait()

    sf.write(
        str(RECORDED_WAV),
        audio,
        SAMPLE_RATE,
    )

    print(f"Recording saved: {RECORDED_WAV}")

    return RECORDED_WAV


# ---------------------------------------------------------
# Whisper speech-to-text
# ---------------------------------------------------------

def speech_to_text(
    audio_path: Path,
    whisper_model: WhisperModel,
) -> str:

    print("\nTranscribing...")

    segments, info = whisper_model.transcribe(
        str(audio_path),
        beam_size=5,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    if not text:
        raise RuntimeError(
            "Whisper could not understand the recording."
        )

    print(f"\nYou said: {text}")

    return text


# ---------------------------------------------------------
# Piper text-to-speech
# ---------------------------------------------------------

def text_to_speech(
    text: str,
    output_path: Path = ANSWER_WAV,
) -> Path:

    ensure_audio_dir()

    if output_path.exists():
        output_path.unlink()

    command = [
        sys.executable,
        "-m",
        "piper",
        "-m",
        str(PIPER_MODEL),
        "-f",
        str(output_path),
    ]

    import subprocess

    subprocess.run(
        command,
        input=text,
        text=True,
        check=True,
    )

    return output_path


# ---------------------------------------------------------
# Play audio
# ---------------------------------------------------------

def play_audio(audio_path: Path) -> None:

    import winsound

    winsound.PlaySound(
        str(audio_path),
        winsound.SND_FILENAME,
    )


# ---------------------------------------------------------
# Speak
# ---------------------------------------------------------

def speak(text: str) -> None:

    wav_path = text_to_speech(text)

    play_audio(wav_path)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\n================================")
    print("       AI TEACHER VOICE MODE")
    print("================================")

    print("\nLoading RAG system...")

    rag = TeacherRAG()

    print("RAG system ready.")

    print("\nLoading Whisper...")
    
    whisper_model = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
    )

    print("Whisper ready.")

    print("\nAI Teacher is ready.")

    # Test Piper + speaker first
    try:
        speak(
            "AI Teacher is ready. "
            "Ask your question."
        )
    except Exception as e:
        print("\nTTS ERROR:")
        print(e)
        return

    while True:

        print("\n--------------------------------")
        command = input(
            "Press Enter to ask a question, "
            "or type q to quit: "
        ).strip().lower()

        if command == "q":
            print("\nGoodbye!")
            break

        try:

            # 1. Record
            audio_path = record_audio()

            # 2. Speech → text
            question = speech_to_text(
                audio_path,
                whisper_model,
            )

            # 3. RAG
            print("\nSearching textbook...")

            answer, sources = rag.ask(
                question,
                show_sources=True,
            )

            # 4. Display answer
            print("\nAI Teacher:")
            print(answer)

            # 5. Answer → speech
            print("\nGenerating voice...")

            speak(answer)

        except KeyboardInterrupt:

            print("\n\nStopped.")
            break

        except Exception as e:

            print("\nERROR:")
            print(type(e).__name__)
            print(e)


if __name__ == "__main__":
    main()