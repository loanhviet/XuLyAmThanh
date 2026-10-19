"""CSE457 Lab 2: short-time analysis, explicit MFCC, and hand-written DTW.

All features use (time, coefficient) order. No ASR service/model is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.ndimage import median_filter
from scipy.signal import lfilter
from sklearn.metrics import accuracy_score, confusion_matrix

LABELS = ("khong", "mot", "hai", "ba", "bon")
VI_LABELS = dict(zip(LABELS, ("không", "một", "hai", "ba", "bốn")))
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    sr: int = 16000
    frame: int = 400
    hop: int = 160
    n_fft: int = 512
    n_mels: int = 24
    n_mfcc: int = 13
    alpha: float = 0.97
    top_db: float = 35.0
    endpoint_method: str = "adaptive"
    noise_window_ms: float = 200.0
    noise_on_db: float = 8.0
    noise_off_db: float = 3.0
    smooth_ms: float = 30.0
    min_speech_ms: float = 40.0
    max_gap_ms: float = 80.0
    margin_ms: float = 200.0
    endpoint: bool = True
    delta: bool = False
    cmn: bool = True

    def __post_init__(self):
        if not (self.sr > 0 and 1 < self.hop <= self.frame <= self.n_fft):
            raise ValueError("Cần 1 < hop <= frame <= n_fft và sr > 0.")
        if not (1 <= self.n_mfcc <= self.n_mels < self.n_fft // 2):
            raise ValueError("Số MFCC/Mel không hợp lệ.")
        if not (0 <= self.alpha < 1 and self.top_db > 0 and self.margin_ms >= 0):
            raise ValueError("Tham số tiền nhấn/endpoint không hợp lệ.")
        if self.endpoint_method not in ("adaptive", "relative"):
            raise ValueError("endpoint_method phải là adaptive hoặc relative.")
        if not (self.noise_window_ms > 0 and 0 <= self.noise_off_db < self.noise_on_db
                and self.smooth_ms > 0 and self.min_speech_ms > 0 and self.max_gap_ms >= 0):
            raise ValueError("Tham số ước lượng nền/hysteresis không hợp lệ.")


def load_audio(path, config=Config(), normalize=True):
    """Read finite mono audio, resample, optionally normalize peak amplitude."""
    y, source_sr = sf.read(str(path), dtype="float64", always_2d=True)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError(f"Audio rỗng hoặc chứa NaN/Inf: {path}")
    y = y.mean(axis=1)
    if source_sr != config.sr:
        y = librosa.resample(y, orig_sr=source_sr, target_sr=config.sr)
    peak = np.max(np.abs(y))
    if normalize and peak > EPS:
        y = y / peak
    return y


def frame_signal(y, config=Config()):
    """Frames of exactly 25 ms; zero-pad only the last incomplete frame.

    FFT padding is separate: 400-sample frames are transformed with NFFT=512.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("Tín hiệu phải là vector hữu hạn, không rỗng.")
    count = 1 + max(0, int(np.ceil((len(y) - config.frame) / config.hop)))
    padded = np.pad(y, (0, (count - 1) * config.hop + config.frame - len(y)))
    indices = np.arange(count)[:, None] * config.hop + np.arange(config.frame)
    return padded[indices]


def time_features(y, config=Config()):
    frames = frame_signal(y, config)
    windowed = frames * np.hamming(config.frame)
    energy = np.sum(windowed ** 2, axis=1)
    # sgn(0) = +1. Rectangular frames for ZCR, exactly crossings/L.
    signs = np.where(frames >= 0, 1, -1)
    return {
        "frames": frames,
        "time": (np.arange(len(frames)) * config.hop + config.frame / 2) / config.sr,
        "energy": energy,
        "log_energy": 10 * np.log10(energy + EPS),
        "magnitude": np.sum(np.abs(windowed), axis=1),
        "rms": np.sqrt(energy / config.frame),
        "zcr": np.sum(signs[:, 1:] != signs[:, :-1], axis=1) / config.frame,
    }


def detect_endpoints(y, config=Config()):
    """Noise-aware endpoints for one isolated word, measured before pre-emphasis.

    Estimate background from the first/last 200 ms (assumed silence), interpolate
    to accommodate slow gain changes. Sustained high-threshold runs seed speech;
    a lower threshold preserves weak boundaries, and short gaps are bridged.
    Keep the component with greatest peak contrast and add an explicit margin.
    If silence estimates are unavailable or no sustained speech is found, retain
    the recording and report the fallback instead of inventing a speech boundary.
    ZCR is plotted, not used as a speech gate because noise also has high ZCR.
    """
    f = time_features(y, config)
    if np.max(f["energy"]) <= EPS:
        raise ValueError("Không tìm thấy tín hiệu âm thanh (file im lặng).")
    raw = f["log_energy"]
    times = f["time"]
    hop_ms = 1000 * config.hop / config.sr
    width = max(1, round(config.smooth_ms / hop_ms))
    if width % 2 == 0:
        width += 1
    smoothed = median_filter(raw, size=width, mode="nearest")
    duration = len(y) / config.sr
    window = config.noise_window_ms / 1000
    head = np.flatnonzero(times <= window)
    # Exclude the last zero-padded frame from the background estimate.
    tail = np.flatnonzero((times >= duration - window)
                         & (times + config.frame / (2 * config.sr) <= duration))
    usable_noise = duration > 2 * window and len(head) > 0 and len(tail) > 0
    if usable_noise:
        noise_start, noise_end = float(np.median(smoothed[head])), float(np.median(smoothed[tail]))
        noise = np.interp(times, [float(np.median(times[head])), float(np.median(times[tail]))],
                          [noise_start, noise_end])
    else:
        noise_start = noise_end = float(np.median(smoothed))
        noise = np.full(len(raw), noise_start)

    def runs(mask):
        edges = np.diff(np.r_[False, mask, False].astype(int))
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    high, low = noise + config.noise_on_db, noise + config.noise_off_db
    status = "detected"
    if config.endpoint_method == "relative":
        high = np.full(len(raw), float(raw.max() - config.top_db))
        low = high.copy()
        active = np.flatnonzero(raw >= high)
        first, last = int(active[0]), int(active[-1])
    elif not usable_noise:
        first, last = 0, len(raw) - 1
        status = "too_short_for_noise_estimate"
    else:
        seeds = np.zeros(len(raw), dtype=bool)
        minimum = max(1, int(np.ceil(config.min_speech_ms / hop_ms)))
        for start, stop in runs(smoothed >= high):
            if stop - start >= minimum:
                seeds[start:stop] = True
        support = smoothed >= low
        gap = int(np.floor(config.max_gap_ms / hop_ms))
        for start, stop in runs(~support):
            if start > 0 and stop < len(support) and stop - start <= gap:
                support[start:stop] = True
        candidates = [(start, stop) for start, stop in runs(support) if seeds[start:stop].any()]
        if not candidates:
            first, last = 0, len(raw) - 1
            status = "no_confident_speech"
        else:
            start, stop = max(candidates, key=lambda span: float(np.max(
                (smoothed - noise)[span[0]:span[1]])))
            first, last = int(start), int(stop - 1)
    margin = round(config.sr * config.margin_ms / 1000)
    start = max(0, first * config.hop - margin)
    end = min(len(y), last * config.hop + config.frame + margin)
    if status == "detected":
        status = "trimmed" if start > 0 or end < len(y) else "unchanged"
    return y[start:end], {
        "start": start, "end": end,
        "first_frame": first, "last_frame": last,
        "method": config.endpoint_method, "status": status,
        "noise_db": float(np.median(noise)), "noise_start_db": noise_start, "noise_end_db": noise_end,
        "threshold_db": float(np.median(high)), "low_threshold_db": float(np.median(low)),
        "noise_curve_db": noise, "high_threshold_curve_db": high, "low_threshold_curve_db": low,
        "smoothed_log_energy": smoothed,
    }


def mel_filterbank(config=Config()):
    hz_to_mel = lambda f: 1125 * np.log1p(f / 700)
    mel_to_hz = lambda m: 700 * np.expm1(m / 1125)
    centers = mel_to_hz(np.linspace(0, hz_to_mel(config.sr / 2), config.n_mels + 2))
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sr)
    filters = np.zeros((config.n_mels, len(frequencies)))
    for m in range(config.n_mels):
        left, middle, right = centers[m:m + 3]
        filters[m] = np.maximum(0, np.minimum(
            (frequencies - left) / (middle - left),
            (right - frequencies) / (right - middle)))
    return filters, frequencies


def mfcc_feature(y, config=Config()):
    """Explicit pre-emphasis -> Hamming -> FFT -> Mel -> ln -> DCT-II.

    Orthonormal DCT scaling is fixed for every utterance. Keep c0..c12.
    """
    emphasized = lfilter([1, -config.alpha], [1], y)
    frames = frame_signal(emphasized, config) * np.hamming(config.frame)
    power = np.abs(np.fft.rfft(frames, n=config.n_fft, axis=1)) ** 2 / config.n_fft
    filters, _ = mel_filterbank(config)
    log_mel = np.log(power @ filters.T + EPS)
    mfcc = dct(log_mel, type=2, norm="ortho", axis=1)[:, :config.n_mfcc]
    if config.cmn:
        mfcc -= mfcc.mean(axis=0, keepdims=True)
    if config.delta:
        # Librosa requires odd width >= 3. Very short signals have no reliable Δ.
        width = min(9, len(mfcc) if len(mfcc) % 2 else len(mfcc) - 1)
        delta = (librosa.feature.delta(mfcc.T, width=width, mode="nearest").T
                 if width >= 3 else np.zeros_like(mfcc))
        mfcc = np.concatenate([mfcc, delta], axis=1)
    return mfcc


def extract_feature(path, config=Config()):
    y = load_audio(path, config)
    if np.max(np.abs(y)) <= EPS:
        raise ValueError(f"File im lặng: {path}")
    if config.endpoint:
        y, _ = detect_endpoints(y, config)
    return mfcc_feature(y, config)


def local_distances(X, Y):
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if (X.ndim != 2 or Y.ndim != 2 or len(X) == 0 or len(Y) == 0
            or X.shape[1] != Y.shape[1] or X.shape[1] == 0
            or not np.isfinite(X).all() or not np.isfinite(Y).all()):
        raise ValueError("X/Y phải có shape (T,D), không rỗng, cùng D và hữu hạn.")
    # Each cell is Euclidean distance, not squared distance.
    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)


def dtw_distance(X, Y):
    """Minimize TOTAL cost, backtrack that path, then divide by its length.

    This is the lab's normalization; it does not optimize average cost directly.
    Tie priority: diagonal, vertical, horizontal, to make paths reproducible.
    """
    C = local_distances(X, Y)
    N, M = C.shape
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0
    back = np.zeros((N, M), dtype=np.uint8)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            previous = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            step = int(np.argmin(previous))
            D[i, j] = C[i - 1, j - 1] + previous[step]
            back[i - 1, j - 1] = step
    path = []
    i, j = N - 1, M - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        step = back[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(D[N, M] / len(path)), path, C, D[1:, 1:]


def split_dataset(root, train_count=3):
    root = Path(root)
    split = {"train": {}, "test": {}}
    for label in LABELS:
        files = sorted(p for p in (root / label).glob("*") if p.suffix.lower() == ".wav")
        if len(files) < max(5, train_count + 1):
            raise ValueError(f"{root / label}: cần ít nhất {max(5, train_count + 1)} WAV, có {len(files)}.")
        split["train"][label] = files[:train_count]
        split["test"][label] = files[train_count:]
    train = {p.resolve() for files in split["train"].values() for p in files}
    test = {p.resolve() for files in split["test"].values() for p in files}
    if train & test:
        raise ValueError("Train/test trùng đường dẫn thực.")
    return split


def data_source(root):
    manifest = Path(root) / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8")).get("data_source", "provided_wav")
    return "provided_wav"


def choose_dataset(project):
    project = Path(project)
    dataset = project / "dataset"
    split_dataset(dataset)
    return dataset, project


def build_templates(train_files, config=Config()):
    templates = {label: [extract_feature(p, config) for p in train_files[label]] for label in LABELS}
    if any(not refs for refs in templates.values()):
        raise ValueError("Mỗi nhãn cần ít nhất một template.")
    return templates


def recognize(path, templates, config=Config()):
    X = extract_feature(path, config)
    scores = {label: min(dtw_distance(X, ref)[0] for ref in templates[label]) for label in LABELS}
    ranked = sorted(scores.items(), key=lambda item: item[1])
    return ranked[0][0], dict(ranked)


def save_templates(path, templates, config, provenance):
    payload = {f"{label}_{i}": ref for label in LABELS for i, ref in enumerate(templates[label])}
    payload["metadata"] = np.array(json.dumps({
        "config": asdict(config), "data_source": provenance,
        "counts": {label: len(templates[label]) for label in LABELS}}))
    np.savez_compressed(path, **payload)


def load_templates(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        templates = {label: [archive[f"{label}_{i}"].copy()
                            for i in range(meta["counts"][label])] for label in LABELS}
    # Older archives were extracted with the peak-relative endpoint method.
    config = dict(meta["config"])
    config.setdefault("endpoint_method", "relative")
    return templates, Config(**config), meta["data_source"]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Không có hàng để xuất CSV.")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(split, templates, config, experiment, provenance):
    rows = []
    for label in LABELS:
        for path in split["test"][label]:
            prediction, scores = recognize(path, templates, config)
            ranked = list(scores.items())
            row = {"experiment": experiment, "data_source": provenance,
                   "file": f"{label}/{path.name}", "true_label": label,
                   "predicted_label": prediction, "correct": prediction == label}
            for i, (name, score) in enumerate(ranked[:3], 1):
                row[f"top{i}_label"] = name
                row[f"top{i}_score"] = score
            rows.append(row)
    truth = [r["true_label"] for r in rows]
    predictions = [r["predicted_label"] for r in rows]
    return rows, float(accuracy_score(truth, predictions)), confusion_matrix(truth, predictions, labels=LABELS)


def run_experiments(root, output, base=Config()):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split = split_dataset(root)
    provenance = data_source(root)
    (output / "split.json").write_text(json.dumps({
        "data_source": provenance, "dataset": str(root.resolve()),
        **{part: {label: [f"{label}/{p.name}" for p in files] for label, files in groups.items()}
           for part, groups in split.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    configs = {
        "baseline": replace(base, endpoint=True, delta=False),
        "E1_no_endpoint": replace(base, endpoint=False, delta=False),
        "E2_mfcc_delta": replace(base, endpoint=True, delta=True),
    }
    experiments, summaries, all_rows = {}, [], []
    for name, config in configs.items():
        templates = build_templates(split["train"], config)
        rows, accuracy, matrix = evaluate(split, templates, config, name, provenance)
        if name == "baseline":
            save_templates(output / "templates_baseline.npz", templates, config, provenance)
        summaries.append({"experiment": name, "data_source": provenance,
                          "endpoint": config.endpoint, "feature_dim": config.n_mfcc * (2 if config.delta else 1),
                          "correct": sum(r["correct"] for r in rows), "test_count": len(rows),
                          "accuracy_percent": 100 * accuracy})
        experiments[name] = {"config": config, "templates": templates, "rows": rows,
                             "accuracy": accuracy, "matrix": matrix}
        all_rows.extend(rows)
    write_csv(output / "results.csv", experiments["baseline"]["rows"])
    write_csv(output / "results_all.csv", all_rows)
    write_csv(output / "experiments.csv", summaries)
    (output / "configurations.json").write_text(json.dumps(
        {name: asdict(c) for name, c in configs.items()}, indent=2), encoding="utf-8")
    reference = output / "endpoint_reference.json"
    if reference.exists():
        previous = json.loads(reference.read_text(encoding="utf-8"))
        fingerprints = {f"{label}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                        for group in split.values() for label, files in group.items() for p in files}
        feature_fields = ("sr", "frame", "hop", "n_fft", "n_mels", "n_mfcc", "alpha", "cmn")
        if (fingerprints == previous["dataset_files_sha256"]
                and all(asdict(base)[k] == previous["feature_config"][k] for k in feature_fields)):
            old = {r["experiment"]: r for r in previous["metrics"]}
            experiments["baseline"]["endpoint_comparison"] = [{
                "experiment": row["experiment"], "test_count": row["test_count"],
                "relative_35db_accuracy_percent": float(old[row["experiment"]]["accuracy_percent"]),
                "adaptive_accuracy_percent": row["accuracy_percent"],
                "change_percentage_points": row["accuracy_percent"] - float(old[row["experiment"]]["accuracy_percent"]),
            } for row in summaries]
    return split, experiments, summaries


def main():
    parser = argparse.ArgumentParser(description="Nhận dạng từ đơn bằng MFCC + DTW tự cài.")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="Chạy baseline và E1/E2, lưu template/kết quả.")
    evaluation.add_argument("--dataset", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict", help="Nhận dạng WAV bằng template đã lưu.")
    prediction.add_argument("wav", type=Path)
    prediction.add_argument("--templates", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        _, _, summaries = run_experiments(args.dataset, args.output)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        templates, config, provenance = load_templates(args.templates)
        label, scores = recognize(args.wav, templates, config)
        print(json.dumps({"prediction": VI_LABELS[label], "label": label,
                          "template_data_source": provenance,
                          "top3": list(scores.items())[:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
