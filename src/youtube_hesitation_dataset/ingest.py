"""Step 1: YouTube -> 16k mono wav."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import RAW, SETTINGS


def download_and_convert(url: str, max_duration_s: int = 0) -> dict:
    """Download bestaudio with yt-dlp, convert to 16k mono wav. Returns meta dict.

    max_duration_s>0 trims the wav to first N seconds (quick test on long videos).
    Idempotent: skips download if wav already exists.
    """
    # resolve id/title first (cheap, no download)
    probe = subprocess.run(
        ["yt-dlp", "--skip-download", "--print", "%(id)s\t%(title)s\t%(duration)s", url],
        capture_output=True, text=True, check=True,
    )
    vid, title, dur = (probe.stdout.strip().split("\t") + ["", "", "0"])[:3]
    out_dir = RAW / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio_16k.wav"
    meta_path = out_dir / "meta.json"

    meta = {"video_id": vid, "title": title, "duration_s": int(float(dur or 0)),
            "source_url": url, "max_duration_s": max_duration_s}
    if wav.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    raw_audio = out_dir / "audio_src.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-f", "bestaudio/best", "-o", str(raw_audio), url],
        check=True,
    )
    # find downloaded src file
    srcs = sorted(out_dir.glob("audio_src.*"))
    if not srcs or (len(srcs) == 1 and srcs[0].name == wav.name):
        raise FileNotFoundError(f"yt-dlp produced nothing in {out_dir}")
    src = srcs[0]
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", str(SETTINGS.sample_rate),
           "-ac", "1", "-c:a", "pcm_s16le"]
    if max_duration_s > 0:
        cmd += ["-t", str(max_duration_s)]
    cmd.append(str(wav))
    subprocess.run(cmd, check=True, capture_output=True)
    try:
        src.unlink()  # keep only normalized wav locally
    except OSError:
        pass
    meta["wav_path"] = str(wav)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
