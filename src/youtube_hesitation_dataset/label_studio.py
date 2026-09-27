"""Step 7: Label Studio task generation.

Taxonomy (4 groups from user):
- acoustic-phonetic: filled_ee_mm, prolongation
- lexico-semantic: filler_word, repetition, self_repair
- structural-dynamic: silent_pause_long, truncation_break
- paralinguistic: breath_laugh_cough, other_noise_music
- not_hesitation (negative)
"""
from __future__ import annotations

import json
from pathlib import Path

LABELS = ["filled_ee_mm", "prolongation", "filler_word", "repetition",
          "self_repair", "silent_pause_long", "truncation_break",
          "breath_laugh_cough", "other_noise_music", "not_hesitation"]


def make_tasks(manifest_path: Path, out_path: Path,
               s3_prefix: str = "") -> Path:
    """manifest.jsonl -> Label Studio JSON tasks (audio + context)."""
    tasks = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dropped_silence"):
            continue
        audio = f"{s3_prefix}/clips/{r['video_id']}/{r['clip_file']}" if s3_prefix else r["clip_file"]
        tasks.append({"data": {
            "audio": audio,
            "video_id": r["video_id"],
            "start": r["start"], "end": r["end"],
            "context": r.get("context", ""),
            "left_word": r.get("left_word", ""), "right_word": r.get("right_word", ""),
        }})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_path
