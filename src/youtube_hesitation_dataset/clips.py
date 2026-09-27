"""Step 5: cut padded opus clips + RMS filter -> manifest.jsonl."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import soundfile as sf
import numpy as np


def rms_db(wav_path: Path, start: float, end: float) -> float:
    wav, sr = sf.read(str(wav_path), dtype="float32", start=int(start * 16_000),
                      frames=max(1, int((end - start) * 16_000)))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    rms = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
    return 20 * math.log10(rms + 1e-12)


def cut_clips(wav_path: Path, candidates: list[dict], clips_dir: Path,
              video_id: str, pad_ms: int = 300, bitrate: str = "32k",
              rms_floor_db: float = -38.0) -> list[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    pad = pad_ms / 1000.0
    rows = []
    for c in candidates:
        s = max(0.0, c["start"] - pad)
        e = c["end"] + pad
        db = rms_db(wav_path, c["start"], c["end"])
        name = f"{video_id}_{c['start']:.2f}-{c['end']:.2f}.opus"
        dst = clips_dir / name
        if not dst.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                 "-i", str(wav_path), "-ar", "16000", "-ac", "1",
                 "-c:a", "libopus", "-b:a", bitrate, str(dst)],
                check=True,
            )
        rows.append({**c, "clip_file": name, "clip_start": round(s, 3),
                     "clip_end": round(e, 3), "rms_db": round(db, 1),
                     "dropped_silence": bool(db < rms_floor_db),
                     "video_id": video_id, "status": "unlabeled"})
    return rows


def write_manifest(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path
