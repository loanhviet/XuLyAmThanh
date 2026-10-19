#!/usr/bin/env python3
"""Copy raw WAVs from Lab 2; preserve its first-3 train / last-2 test split."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root.parent / "Lab02_2351260695_LoAnhViet/dataset")
    parser.add_argument("--output", type=Path, default=root / "dataset")
    parser.add_argument("--train-per-class", type=int, default=3)
    args = parser.parse_args()
    if args.train_per_class < 2:
        parser.error("Cần ít nhất 2 train/lớp.")
    labels = ["khong", "mot", "hai", "ba", "bon"]
    plan, records = [], []
    for label in labels:
        files = sorted(p for p in (args.source / label).glob("*") if p.suffix.lower() == ".wav")
        if len(files) < args.train_per_class + 2:
            parser.error(f"{label}: cần ít nhất {args.train_per_class + 2} WAV.")
        for i, source in enumerate(files):
            split = "train" if i < args.train_per_class else "test"
            target = args.output / split / label / source.name
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                parser.error(f"Từ chối ghi đè file khác nội dung: {target}")
            # Refuse to silently relocate files already assigned to another split.
            other = args.output / ("test" if split == "train" else "train") / label / source.name
            if other.exists():
                parser.error(f"Đã tồn tại ở split khác: {other}; cần tạo dataset mới.")
            plan.append((source, target))
            records.append({"source": str(source.resolve()), "target": str(target.relative_to(args.output)),
                            "sha256": digest, "split": split, "label": label})
    for source, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    source_manifest = args.source / "manifest.json"
    provenance = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else {}
    manifest = {"data_source": "Lab 2 raw user phone recordings", "source": str(args.source.resolve()),
        "split_rule": f"sorted first {args.train_per_class} files/class train, remaining test",
        "note": "Bộ hiện tại 3 train + 2 test/lớp; cần bổ sung train để đạt tối thiểu đề Lab 3.",
        "source_provenance": provenance, "records": records}
    (args.output / "import_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã sao chép/kiểm tra {len(records)} WAV vào {args.output}. Nguồn Lab 2 giữ nguyên.")


if __name__ == "__main__":
    main()
