#!/usr/bin/env python3
"""Import named phone recordings as WAV mono PCM-16 at 16 kHz."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("khong", "mot", "hai", "ba", "bon")
EXTENSIONS = {".m4a", ".mp3", ".aac", ".wav", ".ogg", ".opus", ".flac", ".amr", ".3gp", ".mp4", ".webm"}
NAME = re.compile(r"(khong|mot|hai|ba|bon)_(\d{1,2})", re.IGNORECASE)


def import_audio(source, output, overwrite=False):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise ValueError(f"Không tìm thấy thư mục ghi âm: {source}")
    executable = shutil.which("ffmpeg")
    if not executable:
        raise ValueError("Cần FFmpeg: sudo apt install ffmpeg")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError("Thư mục chưa có file audio hỗ trợ (M4A, MP3, WAV…).")
    planned, invalid = {}, []
    for path in files:
        match = NAME.fullmatch(path.stem)
        if not match or int(match[2]) == 0:
            invalid.append(path.name)
            continue
        label, index = match[1].lower(), int(match[2])
        destination = output / label / f"{label}_{index:02d}.wav"
        if destination in planned:
            raise ValueError(f"Trùng lần ghi âm: {planned[destination]} và {path}. Giữ một file cho mỗi lần.")
        planned[destination] = path
    if invalid:
        raise ValueError("Đổi tên file theo mẫu khong_01.m4a, mot_02.mp3… trước khi nhập: " + ", ".join(invalid))
    converted, skipped = [], []
    for destination, path in planned.items():
        if destination.exists() and not overwrite:
            skipped.append(destination)
            print(f"Bỏ qua WAV đã có: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Conversion failure cannot leave a corrupt or partial final WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=destination.parent, delete=False) as temporary:
            stage = Path(temporary.name)
        try:
            result = subprocess.run([
                executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(path.resolve()), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(stage),
            ], capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError(f"Không chuyển được {path.name}: {result.stderr.strip()}")
            stage.replace(destination)
        finally:
            stage.unlink(missing_ok=True)
        converted.append(destination)
        print(f"{path.name} → {destination.parent.name}/{destination.name}")
    return converted, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Thư mục chứa 25 bản ghi từ điện thoại.")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--overwrite", action="store_true", help="Thay các WAV đã có bằng bản ghi mới.")
    args = parser.parse_args()
    try:
        converted, skipped = import_audio(args.source, args.output, args.overwrite)
    except ValueError as error:
        parser.exit(1, f"Lỗi: {error}\n")
    print(f"\nĐã chuyển {len(converted)} file; bỏ qua {len(skipped)} file đã có.")
    complete = True
    for label in LABELS:
        count = len(list((args.output / label).glob("*.wav")))
        complete &= count >= 5
        print(f"{label}: {count} WAV — {'đủ' if count >= 5 else 'cần ghi thêm'}")
    print("Mở notebook → Restart Kernel → Run All." if complete else "Cần đủ ít nhất 5 file cho từng từ trước khi chạy notebook.")


if __name__ == "__main__":
    main()
