# audio_io.py  -  mic capture + speaker output
import queue, io
import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, CHANNELS, BLOCK_SIZE

audio_queue: queue.Queue = queue.Queue()


def _mic_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio_io] {status}")
    audio_queue.put(indata.copy())


def start_input_stream() -> sd.InputStream:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        dtype="float32", blocksize=BLOCK_SIZE,
        callback=_mic_callback,
    )
    stream.start()
    return stream


def play_audio(raw_bytes: bytes, device_index=None) -> None:
    try:
        import soundfile as sf
        audio_np, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception:
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 22050
    sd.play(audio_np, samplerate=sr, device=device_index, blocking=True)


def list_devices() -> None:
    print("\n-- Audio devices --")
    for i, dev in enumerate(sd.query_devices()):
        ins  = "IN " if dev["max_input_channels"] > 0 else "   "
        outs = "OUT" if dev["max_output_channels"] > 0 else "   "
        print(f"  [{i:2d}] {ins}{outs}  {dev['name']}")
