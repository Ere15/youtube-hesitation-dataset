"""Step 2: Silero VAD (ONNX, CPU-friendly) -> vad.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def run_vad(wav_path: Path, out_path: Path,
            threshold: float = 0.5,
            min_speech_ms: int = 150,
            min_silence_ms: int = 120,
            speech_pad_ms: int = 60,
            sr: int = 16_000) -> list[dict]:
    from silero_vad import load_silero_vad, get_speech_timestamps  # lazy: heavy import

    wav, file_sr = sf.read(str(wav_path), dtype="float32")
    if file_sr != sr:
        raise ValueError(f"expected {sr}Hz, got {file_sr}Hz for {wav_path}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    model = load_silero_vad(onnx=True)
    stamps = get_speech_timestamps(
        wav, model, threshold=threshold, sampling_rate=sr,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    segs = [{"start": float(s["start"]), "end": float(s["end"])} for s in stamps]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(segs, indent=1), encoding="utf-8")
    return segs
