#!/usr/bin/env python3
"""Interactive Linux recording; starts microphone only when the user runs this script."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab2 import LABELS, VI_LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--label", choices=LABELS, help="Chỉ ghi một từ; mặc định ghi đủ 5 từ.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--device", default="default", help="Thiết bị ALSA, ví dụ default hoặc pulse.")
    args = parser.parse_args()
    if args.repeats < 5 or args.seconds < 2:
        parser.error("Cần ít nhất 5 lần/từ và 2 giây/file.")
    if not shutil.which("arecord"):
        parser.error("Cần arecord: sudo apt install alsa-utils; hoặc tự ghi WAV rồi đặt vào dataset/.")
    labels = [args.label] if args.label else LABELS
    for label in labels:
        directory = args.output / label
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(1, args.repeats + 1):
            target = directory / f"{label}_{i:02d}.wav"
            if target.exists():
                print(f"Bỏ qua file đã có: {target}")
                continue
            input(f"\n[{label} {i}/{args.repeats}] Enter để ghi '{VI_LABELS[label]}'. "
                  "Sau khi bắt đầu, chờ 0,3 s, nói một lần rồi giữ im lặng: ")
            temporary = target.with_suffix(".tmp.wav")
            print(f"Đang ghi {args.seconds} giây…", flush=True)
            try:
                subprocess.run(["arecord", "-q", "-D", args.device, "-t", "wav", "-f", "S16_LE",
                                "-c", "1", "-r", "16000", "-d", str(args.seconds), str(temporary)], check=True)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Đã lưu {target}")
    print("Mở notebook, Restart Kernel → Run All để dùng bộ ghi âm.")


if __name__ == "__main__":
    main()
