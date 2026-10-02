"""Step 3: faster-whisper (CPU int8) with word timestamps -> transcript.json."""
from __future__ import annotations

import json
from pathlib import Path


def run_asr(wav_path: Path, out_path: Path,
            model_size: str = "small", compute_type: str = "int8",
            device: str = "cpu",
            language: str = "ru", beam_size: int = 1) -> dict:
    from faster_whisper import WhisperModel  # lazy import

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wav_path), language=language, beam_size=beam_size,
        word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
    )
    segs, words = [], []
    for s in segments:
        segs.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip(),
                     "avg_logprob": float(s.avg_logprob)})
        for w in (s.words or []):
            words.append({"word": w.word, "start": float(w.start), "end": float(w.end),
                          "prob": float(w.probability)})
    out = {"language": info.language, "language_prob": float(info.language_probability),
           "duration": float(info.duration), "segments": segs, "words": words}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
