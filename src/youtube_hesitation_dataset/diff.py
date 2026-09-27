"""Step 4: VAD minus Whisper-words -> hesitation candidates.

Core idea: where VAD hears speech but Whisper outputs no word (within tolerance),
there is likely a non-lexical vocalization: filled pause, breath, truncation, noise.
Separately we collect regex weak-labels of transcribed fillers (ээ/мм/ну/вот...).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FILLER_RE = re.compile(
    r"\b(э+|а+|м+|хм+|эм+|ну+|вот)\b|это самое|как бы|так сказать",
    re.IGNORECASE)


def _subtract_interval(vad_s: float, vad_e: float,
                       words: list[dict], tol: float) -> list[tuple[float, float]]:
    busy = sorted((w["start"] - tol, w["end"] + tol) for w in words
                  if w["end"] > vad_s - tol and w["start"] < vad_e + tol)
    gaps, cur = [], vad_s
    for bs, be in busy:
        if bs > cur:
            gaps.append((cur, min(bs, vad_e)))
        cur = max(cur, be)
        if cur >= vad_e:
            break
    if cur < vad_e:
        gaps.append((cur, vad_e))
    return [(s, e) for s, e in gaps if e > s]


def find_candidates(vad_segs: list[dict], words: list[dict],
                    segments: list[dict],
                    tol_ms: int = 180,
                    min_ms: int = 150,
                    max_ms: int = 2000) -> tuple[list[dict], list[dict]]:
    tol = tol_ms / 1000.0
    words_sorted = sorted(words, key=lambda w: w["start"])
    cands: list[dict] = []
    for v in vad_segs:
        for gs, ge in _subtract_interval(v["start"], v["end"], words_sorted, tol):
            dur = ge - gs
            if (min_ms / 1000.0) <= dur <= (max_ms / 1000.0):
                left = next((w["word"] for w in reversed(words_sorted) if w["end"] <= gs + tol), "")
                right = next((w["word"] for w in words_sorted if w["start"] >= ge - tol), "")
                ctx = " ".join(s["text"] for s in segments
                               if s["end"] > gs - 2.0 and s["start"] < ge + 2.0)[:300]
                cands.append({"start": round(gs, 3), "end": round(ge, 3),
                              "dur": round(dur, 3), "left_word": left,
                              "right_word": right, "context": ctx})
    weak = [s for s in segments if FILLER_RE.search(s["text"])]
    return cands, [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in weak]


def run_diff(vad_path: Path, transcript_path: Path, out_path: Path,
             tol_ms: int = 180, min_ms: int = 150, max_ms: int = 2000) -> dict:
    vad = json.loads(vad_path.read_text(encoding="utf-8"))
    tr = json.loads(transcript_path.read_text(encoding="utf-8"))
    cands, weak = find_candidates(vad, tr.get("words", []), tr.get("segments", []),
                                  tol_ms, min_ms, max_ms)
    out = {"n_vad": len(vad), "n_words": len(tr.get("words", [])),
           "n_candidates": len(cands), "n_filler_segments": len(weak),
           "candidates": cands, "filler_segments": weak}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
