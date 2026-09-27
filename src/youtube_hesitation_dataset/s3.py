"""Step 6: S3-compatible storage (custom endpoint, compressed only)."""
from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from .config import SETTINGS


def client():
    if not SETTINGS.s3_endpoint or not SETTINGS.s3_access:
        raise RuntimeError("S3 not configured: fill .env (S3_ENDPOINT_URL/S3_ACCESS_KEY/...)")
    return boto3.client(
        "s3", endpoint_url=SETTINGS.s3_endpoint,
        aws_access_key_id=SETTINGS.s3_access,
        aws_secret_access_key=SETTINGS.s3_secret,
    )


def ensure_bucket(name: str = "") -> str:
    name = name or SETTINGS.s3_bucket
    c = client()
    try:
        c.head_bucket(Bucket=name)
    except ClientError:
        c.create_bucket(Bucket=name)
    return name


def upload_file(local: Path, key: str, bucket: str = "") -> str:
    bucket = bucket or SETTINGS.s3_bucket
    c = client()
    try:
        c.head_object(Bucket=bucket, Key=key)
        return f"s3://{bucket}/{key} (exists, skipped)"
    except ClientError:
        pass
    extra = {"ContentType": "audio/ogg"} if local.suffix == ".opus" else {}
    c.upload_file(str(local), bucket, key, ExtraArgs=extra or None)
    return f"s3://{bucket}/{key}"


def sync_video(video_id: str, wav_path: Path, clips_dir: Path,
               vad_path: Path, transcript_path: Path, manifest_path: Path,
               bucket: str = "") -> list[str]:
    """Upload compressed artifacts. Full wav -> opus (never raw wav to S3)."""
    import subprocess
    import tempfile

    bucket = bucket or ensure_bucket()
    out = []
    full_opus = clips_dir / f"{video_id}_full.opus"
    if not full_opus.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(wav_path),
             "-ar", "16000", "-ac", "1", "-c:a", "libopus", "-b:a", "32k",
             str(full_opus)], check=True,
        )
    out.append(upload_file(full_opus, f"audio_full/{video_id}.opus", bucket))
    for opus in sorted(clips_dir.glob(f"{video_id}_*.opus")):
        if opus.name.endswith("_full.opus"):
            continue
        out.append(upload_file(opus, f"clips/{video_id}/{opus.name}", bucket))
    out.append(upload_file(vad_path, f"vad/{video_id}.json", bucket))
    out.append(upload_file(transcript_path, f"transcripts/{video_id}.json", bucket))
    out.append(upload_file(manifest_path, f"manifests/{video_id}.jsonl", bucket))
    return out
