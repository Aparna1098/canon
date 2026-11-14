"""Build the narrated demo video in one command.

    .venv/bin/uvicorn api:app --port 8000     # in another terminal
    .venv/bin/python demo/build.py            # → demo/out/canon_demo.mp4 (+ .srt, youtube.txt)

    --skip-record   re-mux with new audio over the last recording (only if scene timings still fit)
    --burn-subs     burn the captions into the picture as well
"""
import json
import os
import re
import subprocess
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PY = sys.executable
sys.path.insert(0, HERE)
from scenes import SCENES  # noqa: E402
from tts import ffmpeg  # noqa: E402

REPO = "https://github.com/Aparna1098/canon"
LEAD_IN = 0.6  # seconds of the loaded page before the first word


def run(*cmd):
    subprocess.run(cmd, check=True)


def ts(t, srt=False):
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    return f"{int(m)}:{int(s):02d}" if not h else f"{int(h)}:{int(m):02d}:{int(s):02d}"


def keep_segments(tl, shift, margin=0.5, min_cut=1.0):
    """Raw-time segments to keep: drop the parts of each live-API wait where no narration is playing."""
    cuts = []
    for ws, we in tl.get("waits", []):
        sc = max((s for s in tl["scenes"] if s["start"] <= ws), key=lambda s: s["start"])
        a, b = max(ws, sc["voice_end"]) + margin, we - margin
        if b - a >= min_cut:
            cuts.append([a, b])
    cuts.sort()
    keep, t = [], shift
    for a, b in cuts:
        if a > t:
            keep.append((t, a))
        t = max(t, b)
    keep.append((t, tl["end"]))
    return keep


def remap(keep):
    """Raw recording time → time in the cut video."""
    def f(t):
        out = 0.0
        for a, b in keep:
            if t >= b:
                out += b - a
            else:
                return out + max(t - a, 0)
        return out
    return f


def captions(scenes, clips, at):
    """Sentences timed by their share of each clip's characters."""
    cues = []
    for sc, (_, _, text, _) in zip(scenes, SCENES):
        dur, t = clips[sc["id"]]["duration"], at(sc["start"])
        parts = [p for p in re.split(r"(?<=[.?!:])\s+", text) if p]
        total = sum(len(p) for p in parts)
        for p in parts:
            d = dur * len(p) / total
            cues.append((t, t + d, p))
            t += d
    return "\n".join(f"{i}\n{ts(a, True)} --> {ts(b, True)}\n{txt}\n" for i, (a, b, txt) in enumerate(cues, 1))


def main():
    args = set(sys.argv[1:])
    try:
        httpx.get("http://localhost:8000/stats", timeout=5).raise_for_status()
    except Exception:
        sys.exit("Canon API isn't up. Start it: .venv/bin/uvicorn api:app --port 8000")

    run(PY, os.path.join(HERE, "tts.py"))
    if "--skip-record" not in args or not os.path.exists(os.path.join(OUT, "raw.webm")):
        run(PY, os.path.join(HERE, "record.py"))

    clips = json.load(open(os.path.join(OUT, "audio", "manifest.json")))["clips"]
    tl = json.load(open(os.path.join(OUT, "timeline.json")))
    scenes = tl["scenes"]
    shift = max(scenes[0]["start"] - LEAD_IN, 0)  # trim the page-load frames
    keep = keep_segments(tl, shift)
    at = remap(keep)
    length = at(tl["end"])
    print(f"cuts: removed {tl['end'] - shift - length:.1f}s of waiting in {len(keep) - 1} places")

    srt = os.path.join(OUT, "canon_demo.srt")
    open(srt, "w").write(captions(scenes, clips, at))

    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-stats", "-i", os.path.join(OUT, "raw.webm")]
    filt = [f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[s{k}]" for k, (a, b) in enumerate(keep)]
    filt.append("".join(f"[s{k}]" for k in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0[cut]")
    for i, sc in enumerate(scenes, 1):
        cmd += ["-i", clips[sc["id"]]["path"]]
        ms = int(at(sc["start"]) * 1000)
        filt.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
    n = len(scenes)
    filt.append("".join(f"[a{i}]" for i in range(1, n + 1)) +
                f"amix=inputs={n}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")
    vf = "fps=30,format=yuv420p"
    if "--burn-subs" in args:
        vf += f",subtitles={srt}:force_style='FontName=Inter,FontSize=22,MarginV=40'"
    filt.append(f"[cut]{vf}[vout]")
    out = os.path.join(OUT, "canon_demo.mp4")
    cmd += ["-filter_complex", ";".join(filt), "-map", "[vout]", "-map", "[aout]", "-t", f"{length:.3f}",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", out]
    run(*cmd)

    chapters = [(at(sc["start"]), title) for sc, (_, title, _, _) in zip(scenes, SCENES) if title]
    chapters[0] = (0, chapters[0][1])
    open(os.path.join(OUT, "youtube.txt"), "w").write(
        "TITLE\nCanon: verified facts and shared memory for AI agents (built on Telem)\n\n"
        "DESCRIPTION\nTelem finds the sources. Canon keeps the facts.\n\n"
        "Canon answers an agent's factual question from the official source (PyPI/npm, SEC EDGAR, PubMed, Google "
        "Maps), with an exact quote checked against the document, then remembers the fact so the next agent pays "
        f"0 searches for it.\n\nCode: {REPO}\n\n" +
        "\n".join(f"{ts(t)} {title}" for t, title in chapters) +
        "\n\nTAGS\nAI agents, Telem, search, RAG, hackathon, Claude, verification, agent memory\n\n"
        "VISIBILITY\nUnlisted (anyone with the link can watch). Upload canon_demo.srt under Subtitles.\n")
    print(f"\nbuilt {out} ({length:.0f}s, {os.path.getsize(out) / 1e6:.1f} MB)\n"
          f"captions {srt}\nupload text {os.path.join(OUT, 'youtube.txt')}")


if __name__ == "__main__":
    main()
