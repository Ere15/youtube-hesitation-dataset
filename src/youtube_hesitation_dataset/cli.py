"""CLI: single-video pipeline + batch over urls.txt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CLIPS, MANIFESTS, RAW, SETTINGS


def run_one(url: str, max_duration_s: int = 0, model: str = "",
            no_s3: bool = False, tol_ms: int = 0) -> dict:
    from . import ingest, vad as vad_m, asr as asr_m, diff as diff_m, clips as clips_m

    SETTINGS.ensure_dirs()
    model = model or SETTINGS.whisper_model
    tol_ms = tol_ms or SETTINGS.diff_tol_ms

    meta = ingest.download_and_convert(url, max_duration_s=max_duration_s)
    vid = meta["video_id"]
    wav = Path(meta.get("wav_path", RAW / vid / "audio_16k.wav"))
    vdir = RAW / vid
    vad_path, tr_path = vdir / "vad.json", vdir / "transcript.json"
    diff_path = vdir / "diff.json"

    vad_segs = (json.loads(vad_path.read_text(encoding="utf-8"))
                if vad_path.exists()
                else vad_m.run_vad(wav, vad_path, threshold=SETTINGS.vad_threshold,
                                   min_speech_ms=SETTINGS.vad_min_speech_ms,
                                   min_silence_ms=SETTINGS.vad_min_silence_ms,
                                   speech_pad_ms=SETTINGS.vad_speech_pad_ms))
    tr = (json.loads(tr_path.read_text(encoding="utf-8"))
          if tr_path.exists()
          else asr_m.run_asr(wav, tr_path, model_size=model,
                             compute_type=SETTINGS.whisper_compute,
                             language=SETTINGS.whisper_lang))
    d = diff_m.run_diff(vad_path, tr_path, diff_path, tol_ms=tol_ms,
                        min_ms=SETTINGS.cand_min_ms, max_ms=SETTINGS.cand_max_ms)
    rows = clips_m.cut_clips(wav, d["candidates"], CLIPS / vid, vid,
                             pad_ms=SETTINGS.clip_pad_ms, bitrate=SETTINGS.clip_bitrate,
                             rms_floor_db=SETTINGS.rms_floor_db)
    kept = [r for r in rows if not r["dropped_silence"]]
    mpath = clips_m.write_manifest(rows, MANIFESTS / f"{vid}.jsonl")

    from .label_studio import make_tasks
    lpath = make_tasks(mpath, MANIFESTS / f"{vid}_ls_tasks.json")

    s3log: list[str] = []
    if not no_s3:
        from . import s3 as s3_m
        s3log = s3_m.sync_video(vid, wav, CLIPS / vid, vad_path, tr_path, mpath)

    print(f"video={vid} dur={meta.get('duration_s')}s "
          f"vad={d['n_vad']} words={d['n_words']} "
          f"cands={d['n_candidates']} kept={len(kept)} dropped={len(rows)-len(kept)} "
          f"fillers={d['n_filler_segments']}")
    print(f"manifest: {mpath}\nls-tasks: {lpath}")
    for line in s3log:
        print(" ", line)
    return {"video_id": vid, **d, "n_kept": len(kept),
            "manifest": str(mpath), "ls_tasks": str(lpath)}


def run_batch(urls_file: str, **kw) -> None:
    urls = [l.strip() for l in Path(urls_file).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    print(f"{len(urls)} urls from {urls_file}")
    for i, u in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {u}")
        try:
            run_one(u, **kw)
        except Exception as e:  # batch must not stop on one failure
            print(f"  FAILED: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hesitation",
                                description="RU hesitation dataset funnel")
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one", help="process single video")
    o.add_argument("url")
    o.add_argument("--max-duration", type=int, default=0, help="trim to first N sec (test)")
    o.add_argument("--model", default="", help="whisper size (default: config small)")
    o.add_argument("--no-s3", action="store_true")
    o.add_argument("--tol-ms", type=int, default=0)
    b = sub.add_parser("batch", help="process urls.txt")
    b.add_argument("--urls", default="urls.txt")
    b.add_argument("--max-duration", type=int, default=0)
    b.add_argument("--model", default="")
    b.add_argument("--no-s3", action="store_true")
    b.add_argument("--tol-ms", type=int, default=0)
    s = sub.add_parser("ensure-bucket", help="create S3 bucket if missing")
    a = p.parse_args(argv)
    if a.cmd == "one":
        run_one(a.url, max_duration_s=a.max_duration, model=a.model,
                no_s3=a.no_s3, tol_ms=a.tol_ms)
    elif a.cmd == "batch":
        run_batch(a.urls, max_duration_s=a.max_duration, model=a.model,
                  no_s3=a.no_s3, tol_ms=a.tol_ms)
    elif a.cmd == "ensure-bucket":
        from . import s3 as s3_m
        print("bucket:", s3_m.ensure_bucket())


if __name__ == "__main__":
    main()
