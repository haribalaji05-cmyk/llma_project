import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None

try:
    from TTS.api import TTS
except ImportError:  # pragma: no cover
    TTS = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None


def configure_ffmpeg_path() -> str:
    ffmpeg_env = os.getenv("FFMPEG_PATH", "").strip()
    candidate_paths = [
        ffmpeg_env,
        r"C:\Users\harir\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe",
        r"C:\Users\harir\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin",
    ]

    for candidate in candidate_paths:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file() and candidate_path.name.lower() == "ffmpeg.exe":
            ffmpeg_dir = str(candidate_path.parent)
        elif candidate_path.is_dir():
            ffmpeg_dir = str(candidate_path)
        else:
            continue

        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path.split(os.pathsep):
            os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current_path}" if current_path else ffmpeg_dir
        return ffmpeg_dir

    return ""


class SpeechToText:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("WHISPER_MODEL", "small")
        self.model = None
        self.load_error = ""
        self.ffmpeg_dir = configure_ffmpeg_path()

        if whisper is None:
            self.load_error = "openai-whisper is not installed"
        else:
            try:
                self.model = whisper.load_model(self.model_name)
            except Exception as exc:
                self.load_error = str(exc)

    def transcribe(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        if self.model is None:
            return {
                "text": "",
                "language": language or "en",
                "error": self.load_error,
            }

        try:
            options = {"fp16": False, "task": "transcribe"}
            if language:
                options["language"] = language
            result = self.model.transcribe(file_path, **options)
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", language or "en"),
                "error": "",
            }
        except Exception as exc:
            error_message = str(exc)
            if "[WinError 2]" in error_message and not self.ffmpeg_dir:
                error_message = (
                    "ffmpeg was not found. Install ffmpeg or set FFMPEG_PATH to your ffmpeg.exe/bin folder."
                )
            return {
                "text": "",
                "language": language or "en",
                "error": error_message,
            }


class TextToSpeech:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
        self.tts = None
        self.pyttsx3_engine = None
        self.load_error = ""

        if TTS is None:
            self.load_error = "Coqui TTS is not installed"
        else:
            try:
                self.tts = TTS(self.model_name, progress_bar=False, gpu=False)
            except Exception as exc:
                self.load_error = str(exc)

        if self.tts is None and pyttsx3 is not None:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty("rate", 170)
                self.load_error = ""
            except Exception as exc:
                if not self.load_error:
                    self.load_error = str(exc)

    def synthesize(self, text: str, language: str = "en", voice: Optional[str] = None) -> Dict[str, Any]:
        if self.tts is None and self.pyttsx3_engine is None:
            return {"audio": b"", "sample_rate": 0, "error": self.load_error}

        if not text.strip():
            return {"audio": b"", "sample_rate": 0, "error": "Text cannot be empty."}

        try:
            if self.tts is not None:
                if voice:
                    audio = self.tts.tts(text, speaker=voice, language=language)
                else:
                    audio = self.tts.tts(text, language=language)

                if isinstance(audio, tuple) and len(audio) == 2:
                    signal, sample_rate = audio
                else:
                    signal = audio
                    sample_rate = getattr(self.tts.synthesizer, "output_sample_rate", 22050)

                out_buffer = io.BytesIO()
                if sf is not None:
                    sf.write(out_buffer, signal, sample_rate, format="WAV")
                else:
                    from scipy.io.wavfile import write as wav_write

                    wav_write(out_buffer, sample_rate, signal)
                return {"audio": out_buffer.getvalue(), "sample_rate": sample_rate, "error": ""}

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                temp_path = tmp.name
            try:
                if voice:
                    self.pyttsx3_engine.setProperty("voice", voice)
                self.pyttsx3_engine.save_to_file(text, temp_path)
                self.pyttsx3_engine.runAndWait()
                audio_bytes = open(temp_path, "rb").read()
                return {"audio": audio_bytes, "sample_rate": 22050, "error": ""}
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        except Exception as exc:
            return {"audio": b"", "sample_rate": 0, "error": str(exc)}
