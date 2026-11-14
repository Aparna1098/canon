"""Render each scene's narration to demo/out/audio/, and write manifest.json with the clip durations.

Provider: TTS_PROVIDER=openai | elevenlabs | say. When it is unset, the first one with a key in .env is used, with
macOS `say` as an offline stand-in. Clips are cached by a hash of provider, voice and text, so editing one line
re-renders only that scene.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import httpx
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "audio")
sys.path.insert(0, HERE)
from scenes import SCENES  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))

OPENAI_VOICE = os.environ.get("OPENAI_TTS_VOICE", "sage")
OPENAI_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_STYLE = os.environ.get(
    "OPENAI_TTS_INSTRUCTIONS",
    "Narrating a polished product demo for a hackathon. Warm, confident and clear, at a natural conversational pace. "
    "Slight emphasis on key phrases; no exaggerated excitement.")
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us")  # "Bella"
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
SAY_VOICE = os.environ.get("SAY_VOICE", "Samantha")


def provider():
    p = os.environ.get("TTS_PROVIDER")
    if p:
        return p
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    return "say"


def voice_of(p):
    return {"openai": f"{OPENAI_MODEL}/{OPENAI_VOICE}/{OPENAI_STYLE}", "elevenlabs": f"{ELEVEN_MODEL}/{ELEVEN_VOICE}",
            "say": SAY_VOICE}[p]


def render(p, text, path):
    if p == "openai":
        r = httpx.post("https://api.openai.com/v1/audio/speech", timeout=120,
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                       json={"model": OPENAI_MODEL, "voice": OPENAI_VOICE, "input": text,
                             "instructions": OPENAI_STYLE, "response_format": "mp3"})
        r.raise_for_status()
        open(path, "wb").write(r.content)
    elif p == "elevenlabs":
        r = httpx.post(f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}", timeout=120,
                       headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                       params={"output_format": "mp3_44100_128"},
                       json={"text": text, "model_id": ELEVEN_MODEL,
                             "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}})
        r.raise_for_status()
        open(path, "wb").write(r.content)
    else:
        aiff = path + ".aiff"
        subprocess.run(["say", "-v", SAY_VOICE, "-r", "175", "-o", aiff, text], check=True)
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", aiff, "-q:a", "2", path], check=True)
        os.remove(aiff)


def ffmpeg():
    """System ffmpeg if installed, else the static binary from the imageio-ffmpeg package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def duration(path):
    """Decode the clip and read the final timestamp (no ffprobe needed)."""
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", path, "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    h, m, s = re.findall(r"time=(\d+):(\d+):([\d.]+)", err)[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    os.makedirs(OUT, exist_ok=True)
    p = provider()
    if p == "say":
        print("tts: no OPENAI_API_KEY or ELEVENLABS_API_KEY in .env, using macOS `say` as a stand-in voice")
    clips, total = {}, 0.0
    for sid, _, text, _ in SCENES:
        h = hashlib.sha1(f"{p}|{voice_of(p)}|{text}".encode()).hexdigest()[:10]
        path = os.path.join(OUT, f"{sid}-{h}.mp3")
        if not os.path.exists(path):
            print(f"tts: {p} → {sid}")
            render(p, text, path)
        clips[sid] = {"path": path, "duration": duration(path)}
        total += clips[sid]["duration"]
    json.dump({"provider": p, "clips": clips}, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(f"tts: {len(clips)} clips, {total:.0f}s of narration ({p})")


if __name__ == "__main__":
    main()
