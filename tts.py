# tts.py  -  text -> ElevenLabs voice synthesis
import requests
import config

_BASE = "https://api.elevenlabs.io/v1"
_HEADERS = {
    "xi-api-key": config.ELEVENLABS_API_KEY,
    "Content-Type": "application/json",
    "Accept": "audio/mpeg",
}


def synthesize(text: str) -> bytes:
    if not text:
        return b""
    url = f"{_BASE}/text-to-speech/{config.ELEVENLABS_VOICE_ID}/stream"
    payload = {
        "text": text,
        "model_id": "eleven_flash_v2",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.85,
            "style": 0.0,
            "use_speaker_boost": True,
        },
        "output_format": "mp3_22050_32",
    }
    response = requests.post(url, headers=_HEADERS, json=payload, stream=True, timeout=15)
    if response.status_code != 200:
        print(f"[tts] ElevenLabs error {response.status_code}: {response.text[:200]}")
        return b""
    return b"".join(c for c in response.iter_content(chunk_size=2048) if c)
