"""Central config: .env, paths, funnel hyperparams."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in cwd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"          # data/raw/<video_id>/
CLIPS = DATA / "clips"      # data/clips/<video_id>/
MANIFESTS = DATA / "manifests"
LABELS = DATA / "labels"


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n, "")
        if v:
            return v
    return default


@dataclass
class Settings:
    s3_endpoint: str = field(default_factory=lambda: _env("S3_ENDPOINT_URL"))
    s3_access: str = field(default_factory=lambda: _env("S3_ACCESS_KEY", "AWS_ACCESS_KEY_ID"))
    s3_secret: str = field(default_factory=lambda: _env("S3_SECRET_KEY", "AWS_SECRET_ACCESS_KEY"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", default="hesitation-ru"))

    # funnel hyperparams (CPU-tuned defaults)
    sample_rate: int = 16_000
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 150
    vad_min_silence_ms: int = 120
    vad_speech_pad_ms: int = 60
    whisper_model: str = "small"   # small-int8 on CPU; bump to medium if WER bad
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", default="cpu"))
    whisper_compute: str = field(default_factory=lambda: _env("WHISPER_COMPUTE", default="int8"))
    whisper_lang: str = "ru"
    diff_tol_ms: int = 180         # tolerance around whisper words
    cand_min_ms: int = 150
    cand_max_ms: int = 2000
    clip_pad_ms: int = 300
    clip_bitrate: str = "32k"      # opus
    rms_floor_db: float = -38.0    # below -> likely silence, drop

    def ensure_dirs(self) -> None:
        for p in (DATA, RAW, CLIPS, MANIFESTS, LABELS):
            p.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
