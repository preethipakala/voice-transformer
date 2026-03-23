# asr.py  -  speech -> text via faster-whisper
import numpy as np
from faster_whisper import WhisperModel
from config import WHISPER_MODEL

print(f"[asr] Loading Whisper '{WHISPER_MODEL}' (CPU, int8) ...")
_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("[asr] Whisper ready.")


def transcribe(audio_blocks: list) -> str:
    audio = np.concatenate(audio_blocks, axis=0).flatten().astype(np.float32)
    segments, _ = _model.transcribe(
        audio, beam_size=1, language="en", vad_filter=True
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
