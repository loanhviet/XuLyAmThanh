#!/usr/bin/env python3
"""Generate a clearly labeled synthetic command dataset; never replaces recordings.

One independent TTS request/utterance, 6 train + 2 test/word/voice. The synthetic
voice is shared across splits: this is not a human/speaker-independent benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
WORDS = {"mo": "mở", "dong": "đóng", "len": "lên", "xuong": "xuống", "dung": "dừng"}
# Fixed split/settings, no selection by test recognition accuracy.
STYLES = [(-15, -4), (-8, 2), (0, 0), (8, -2), (15, 4), (3, 1), (-4, -1), (11, 3)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_to_wav(source, target, seed):
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-f", "f32le",
        "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", "pipe:1"], check=True, capture_output=True)
    y = np.frombuffer(result.stdout, dtype="<f4").astype(float)
    if not y.size or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-5:
        raise ValueError(f"TTS trả audio rỗng/im lặng: {source}")
    peak = float(np.max(np.abs(y)))
    # Remove native TTS padding, retain all samples above -60 dB relative peak.
    active = np.flatnonzero(np.abs(y) >= peak * 1e-3)
    y = y[active[0]:active[-1] + 1]
    if len(y) < 1600:
        raise ValueError(f"TTS ngắn bất thường (<100ms): {source}")
    rng = np.random.default_rng(seed)
    head = int(rng.integers(3200, 8001))
    tail = int(rng.integers(3200, 8001))
    target_peak = float(rng.uniform(0.65, 0.85))
    y = y / peak * target_peak
    padded = np.pad(y, (head, tail))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.wav")
    try:
        sf.write(temporary, padded, 16000, subtype="PCM_16")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"head_silence_seconds": head / 16000, "tail_silence_seconds": tail / 16000,
        "speech_seconds": len(y) / 16000, "duration_seconds": len(padded) / 16000,
        "target_peak": target_peak, "postprocess_seed": seed}


async def generate(args):
    if not shutil.which("ffmpeg"):
        raise ValueError("Cần ffmpeg để chuyển MP3 thành WAV PCM16/16kHz/mono.")
    if args.output.exists() and any(args.output.iterdir()):
        manifest_path = args.output / "manifest.json"
        if not (args.output / "generation_config.json").exists():
            if not manifest_path.exists() or not json.loads(manifest_path.read_text(encoding="utf-8")).get("is_synthetic"):
                raise ValueError("Từ chối tạo TTS trong thư mục không có metadata tổng hợp; chọn output mới.")
    if len(set(args.voices)) != len(args.voices):
        raise ValueError("Voice bị lặp.")
    raw = args.output / "tts_sources"
    raw.mkdir(parents=True, exist_ok=True)
    config = {"voices": args.voices, "words": WORDS, "styles": STYLES, "seed": args.seed,
              "engine": "edge-tts", "engine_version": version("edge-tts")}
    config_path = args.output / "generation_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != json.loads(json.dumps(config)):
            raise ValueError("Cấu hình khác lần tạo trước; chọn output mới để tránh trộn nguồn.")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(voice_idx, voice, label, text, idx, rate, pitch):
        split = "train" if idx < 6 else "test"
        speaker_id = voice.removeprefix("vi-VN-").removesuffix("Neural").lower()
        stem = f"{label}_{speaker_id}_{idx + 1:02d}"
        mp3 = raw / f"{stem}.mp3"
        wav = args.output / split / label / f"{stem}.wav"
        rate_text, pitch_text = f"{rate:+d}%", f"{pitch:+d}Hz"
        async with semaphore:
            if not mp3.exists():
                temporary = mp3.with_suffix(".tmp.mp3")
                for attempt in range(3):
                    try:
                        communicate = edge_tts.Communicate(text, voice=voice, rate=rate_text,
                                                          pitch=pitch_text, connect_timeout=15, receive_timeout=30)
                        await asyncio.wait_for(communicate.save(str(temporary)), timeout=50)
                        if not temporary.stat().st_size:
                            raise ValueError("MP3 rỗng")
                        temporary.replace(mp3)
                        break
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        if attempt == 2:
                            raise
                        await asyncio.sleep(1 + attempt)
            seed = args.seed + voice_idx * 1000 + list(WORDS).index(label) * 10 + idx
            metadata = convert_to_wav(mp3, wav, seed)
            print(f"[{split}] {wav.name}: {metadata['duration_seconds']:.2f}s", flush=True)
            return {"file": str(wav.relative_to(args.output)), "split": split, "label": label,
                "spoken_text": text, "voice": voice, "speaker_type": "synthetic_voice",
                "rate": rate_text, "pitch": pitch_text, "source_mp3": str(mp3.relative_to(args.output)),
                "source_sha256": digest(mp3), "sha256": digest(wav), "independent_tts_request": True, **metadata}

    tasks = [one(vi, voice, label, text, idx, rate, pitch)
        for vi, voice in enumerate(args.voices) for label, text in WORDS.items()
        for idx, (rate, pitch) in enumerate(STYLES)]
    records = await asyncio.gather(*tasks)
    manifest = {"data_source": "synthetic_tts", "is_synthetic": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "engine": "edge-tts",
        "engine_version": version("edge-tts"), "voices": args.voices,
        "words": WORDS, "sample_rate": 16000, "channels": 1, "subtype": "PCM_16",
        "split_rule": "6 train + 2 test per word per synthetic voice; predeclared independent rate/pitch settings",
        "human_recording": False, "license_note": "Generated via online TTS; no claim of a human-recorded corpus.",
        "limitation": "Synthetic voices shared across splits; useful for pipeline demonstration, not evidence of real-speaker generalization.",
        "records": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    from lab3 import Config, extract_features, read_dataset
    labels, samples, audit, notices = read_dataset(args.output)
    for sample in samples:
        F = extract_features(sample.path, Config())
        assert F.shape[1] == 39 and np.isfinite(F).all()
    if notices:
        raise ValueError(f"Audit không đạt: {notices}")
    count = len(args.voices)
    text = f"""# Dataset tổng hợp — 5 lệnh Lab 3

**Đây là tiếng nói tổng hợp, không phải bản ghi người thật.**

- Từ: mở (`mo`), đóng (`dong`), lên (`len`), xuống (`xuong`), dừng (`dung`).
- {count} giọng tổng hợp: {', '.join(args.voices)}.
- Mỗi từ, mỗi giọng: **6 train + 2 test**, tổng **{len(records)} WAV**.
- WAV mono, 16.000 Hz, PCM 16-bit; khoảng lặng đầu/cuối 0,2–0,5 giây.
- Mỗi utterance được yêu cầu TTS riêng với rate/pitch định trước; không chia bản
  sao/biến đổi hậu kỳ của một file gốc sang hai split.
- Tất cả WAV đã qua kiểm tra định dạng, hash train/test và đặc trưng T × 39.
- `manifest.json` ghi nhãn, chữ được đọc, giọng, split, thông số, silence và hash.
- `tts_sources/` giữ MP3 gốc để kiểm tra nguồn; `generation_config.json` ghi cấu hình.

Đúng số lượng/nhãn/định dạng trong PDF để minh họa pipeline. Cùng TTS engine/voice
ở train/test làm bài toán dễ hơn dữ liệu người thật; accuracy trên bộ này không
đại diện chất lượng nhận dạng thực tế. Việc dùng dữ liệu tổng hợp cho bài nộp cần
phù hợp với yêu cầu giảng viên.

## Chạy pipeline từ thư mục gốc dự án

```bash
.venv/bin/python lab3.py run --dataset {args.output.name}
```

## Tạo lại

Cần mạng, ffmpeg và gói trong `requirements-tts.txt`. Mỗi file TTS gốc được cache;
chạy lại dùng cache. Dịch vụ có thể thay đổi đầu ra ở những lần tạo mới.

```bash
uv pip install --python .venv/bin/python -r requirements-tts.txt
.venv/bin/python scripts/generate_synthetic_dataset.py --output {args.output.name}
```

Tài liệu công cụ: https://github.com/rany2/edge-tts
"""
    (args.output / "README.md").write_text(text, encoding="utf-8")
    print(f"Hoàn tất: {len(samples)} WAV, {len(labels)} lớp; mọi đặc trưng có 39 chiều.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--voices", nargs="+", default=["vi-VN-NamMinhNeural"])
    parser.add_argument("--seed", type=int, default=695)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1 or args.concurrency > 4:
        parser.error("concurrency cần nằm trong 1..4")
    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
