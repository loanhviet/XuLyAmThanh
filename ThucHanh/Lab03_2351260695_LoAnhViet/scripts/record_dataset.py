#!/usr/bin/env python3
"""Interactive recording. Microphone starts only after the user presses Enter."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

DIGITS = ["khong", "mot", "hai", "ba", "bon"]
COMMANDS = ["mo", "dong", "len", "xuong", "dung"]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dataset")
    parser.add_argument("--vocabulary", choices=["digits", "commands"], default="digits")
    parser.add_argument("--label", help="Chỉ thu nhãn này trong vocabulary đã chọn")
    parser.add_argument("--train-count", type=int, default=6, help="Số file train đích mỗi lớp, gồm file đã có")
    parser.add_argument("--test-count", type=int, default=2, help="Số file test đích mỗi lớp, gồm file đã có")
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--device", default="default")
    args = parser.parse_args()
    vocabulary = DIGITS if args.vocabulary == "digits" else COMMANDS
    if args.label and args.label not in vocabulary:
        parser.error("Nhãn không nằm trong vocabulary đã chọn.")
    if args.train_count < 2 or args.test_count < 1 or args.seconds < 2:
        parser.error("Cần train>=2, test>=1 và seconds>=2.")
    if not shutil.which("arecord"):
        parser.error("Không có arecord; cài alsa-utils hoặc tự chép WAV vào các thư mục.")
    for label in [args.label] if args.label else vocabulary:
        all_files = [p for split in ("train", "test") for p in (args.output / split / label).glob("*.wav")]
        indices = [int(m.group(1)) for p in all_files if (m := re.search(r"_(\d+)\.wav$", p.name))]
        index = max(indices, default=0) + 1
        for split, desired in [("train", args.train_count), ("test", args.test_count)]:
            folder = args.output / split / label
            folder.mkdir(parents=True, exist_ok=True)
            existing = len(list(folder.glob("*.wav")))
            for n in range(existing, desired):
                target = folder / f"{label}_{index:02d}.wav"
                temporary = target.with_suffix(".tmp.wav")
                if target.exists():
                    raise FileExistsError(target)
                input(f"{split}/{label} [{n + 1}/{desired}]: Enter để thu {args.seconds}s; chờ 0,3s rồi nói một lần: ")
                try:
                    subprocess.run(["arecord", "-q", "-D", args.device, "-t", "wav", "-f", "S16_LE",
                                    "-c", "1", "-r", "16000", "-d", str(args.seconds), str(temporary)], check=True)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                index += 1
                print("Đã lưu", target)
    print("Thu xong. Chạy lại Lab03.ipynb hoặc python lab3.py run.")


if __name__ == "__main__":
    main()
