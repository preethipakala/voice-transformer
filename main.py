# main.py  -  headless entry point (no GUI)
# Run: python main.py  |  python main.py --list-devices
import sys, time, threading
from config import BUFFER_BLOCKS, DEBUG_TRANSCRIBE, OUTPUT_DEVICE_NAME
from audio_io import audio_queue, start_input_stream, play_audio, list_devices
from asr import transcribe
from tts import synthesize


def process_loop(out):
    buf = []
    print("[main] Listening... Ctrl+C to stop.")
    while True:
        try:
            buf.append(audio_queue.get(timeout=0.5))
        except Exception:
            continue
        if len(buf) < BUFFER_BLOCKS:
            continue
        blocks, buf = buf.copy(), []
        text = transcribe(blocks)
        if not text:
            continue
        if DEBUG_TRANSCRIBE:
            print(f'[asr] -> "{text}"')

        def _play(t):
            t0 = time.time()
            ab = synthesize(t)
            if ab:
                print(f"[tts] {len(ab)//1024}KB in {time.time()-t0:.2f}s")
                play_audio(ab, device_index=out)

        threading.Thread(target=_play, args=(text,), daemon=True).start()


def main():
    if "--list-devices" in sys.argv:
        list_devices()
        return
    import sounddevice as sd
    out = next(
        (i for i, d in enumerate(sd.query_devices())
         if OUTPUT_DEVICE_NAME and OUTPUT_DEVICE_NAME.lower() in d["name"].lower()
         and d["max_output_channels"] > 0),
        None,
    )
    stream = start_input_stream()
    try:
        process_loop(out)
    except KeyboardInterrupt:
        print("\n[main] Stopping...")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
