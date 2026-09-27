# youtube-hesitation-dataset

Воронка для датасета хезитаций в русской речи с YouTube:
`yt-dlp` → нормализация в 16k mono → **VAD (Silero)** + **ASR (faster-whisper, word timestamps)** →
разность `VAD − Whisper` = кандидаты → нарезка сжатых `opus`-клипов → **S3** → разметка в **Label Studio**.

## Таксономия (Label Studio)

- акустико-фонетические: `filled_ee_mm`, `prolongation`
- лексико-семантические: `filler_word`, `repetition`, `self_repair`
- структурно-динамические: `silent_pause_long`, `truncation_break`
- паралингвистические: `breath_laugh_cough`, `other_noise_music`
- `not_hesitation` (негатив)

## Конфиг

`.env` (уже есть, не коммитится):
```
S3_ENDPOINT_URL=http://s3.jimmyworm.duckdns.org/
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_BUCKET=hesitation-ru
```

`urls.txt` — по 1 YouTube-URL на строку.

## Запуск

```bash
uv sync
uv run hesitation ensure-bucket
# быстрый тест: первые 600 сек одного видео, без заливки
uv run hesitation one "https://www.youtube.com/watch?v=RBM2RIvW8Ug" --max-duration 600 --no-s3
# полный прогон одного видео + заливка на S3
uv run hesitation one "https://www.youtube.com/watch?v=RBM2RIvW8Ug"
# батч по всем urls.txt
uv run hesitation batch --urls urls.txt
```

## Артефакты (локально `data/`, на S3 только сжатое)

- `data/raw/<id>/audio_16k.wav` — локально; на S3 как `audio_full/<id>.opus` (32k)
- `data/raw/<id>/{vad.json,transcript.json,diff.json}` → `vad/`, `transcripts/`
- `data/clips/<id>/*.opus` — кандидаты ±300ms паддинга → `clips/<id>/`
- `data/manifests/<id>.jsonl` — строки `{video_id,start,end,dur,left_word,right_word,context,rms_db,status}`
- `data/manifests/<id>_ls_tasks.json` — импорт в Label Studio

Строка манифеста: `video_id, clip_file, start, end, dur, rms_db, dropped_silence, context, status=unlabeled`.

## Железо

Только CPU: `faster-whisper small int8` (при плохом качестве слов — `medium`),
Silero VAD через ONNX. Пороги см. `src/youtube_hesitation_dataset/config.py`.
