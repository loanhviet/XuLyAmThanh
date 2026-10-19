"""Lab 3 — isolated-word left-to-right Gaussian HMM. All features are T x D.

No test data is used by fit, initialization, feature normalization or LM counts.
The 5-state/39-D baseline is fixed before looking at test outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import platform
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.hmm import GaussianHMM
from numba import njit
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
VI = {"khong": "không", "mot": "một", "hai": "hai", "ba": "ba", "bon": "bốn",
      "mo": "mở", "dong": "đóng", "len": "lên", "xuong": "xuống", "dung": "dừng"}


@dataclass(frozen=True)
class Config:
    sr: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mfcc: int = 13
    n_mels: int = 26
    top_db: float = 30.0
    dynamic: bool = True
    n_states: int = 5
    covariance_type: str = "diag"
    n_iter: int = 80
    tol: float = 1e-3
    seed: int = 695


@dataclass(frozen=True)
class Sample:
    label: str
    path: Path
    split: str


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_dataset(root):
    """Validate split identities, exact file duplicates and identical decoded audio."""
    root = Path(root)
    labels = sorted(p.name for p in (root / "train").iterdir() if p.is_dir()) if (root / "train").is_dir() else []
    if len(labels) < 2:
        raise ValueError(f"Cần ít nhất 2 lớp tại {root}/train. Xem README để nhập/thu WAV.")
    test_labels = sorted(p.name for p in (root / "test").iterdir() if p.is_dir()) if (root / "test").is_dir() else []
    if labels != test_labels:
        raise ValueError("Train và test phải có cùng các lớp.")
    samples, rows, seen, notices = [], [], {}, []
    for split in ("train", "test"):
        for label in labels:
            files = sorted(p for p in (root / split / label).iterdir() if p.suffix.lower() == ".wav")
            if len(files) < (2 if split == "train" else 1):
                raise ValueError(f"{split}/{label}: cần ít nhất 2 train và 1 test để chạy thí nghiệm.")
            if len(files) < (6 if split == "train" else 2):
                notices.append(f"{split}/{label}: {len(files)} file; đề gợi ý tối thiểu 6 train + 2 test/lớp.")
            for path in files:
                audio, sr = sf.read(path, dtype="float64", always_2d=True)
                if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-8:
                    raise ValueError(f"File rỗng, im lặng hoặc không hữu hạn: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                pcm_digest = hashlib.sha256(str(sr).encode() + audio.tobytes()).hexdigest()
                for identity in (str(path.resolve()), digest, pcm_digest):
                    if identity in seen and seen[identity] != split:
                        raise ValueError(f"Trùng dữ liệu train/test: {path}")
                    seen[identity] = split
                info = sf.info(path)
                if sr != 16000 or info.channels != 1 or info.subtype != "PCM_16":
                    notices.append(f"{path.name}: đầu vào {sr} Hz/{info.channels} kênh/{info.subtype}; đọc sẽ chuyển mono/16k.")
                samples.append(Sample(label, path, split))
                rows.append({"split": split, "label": label, "file": str(path.relative_to(root)),
                             "sha256": digest, "audio_sha256": pcm_digest, "sample_rate": sr,
                             "channels": info.channels, "subtype": info.subtype,
                             "duration_seconds": len(audio) / sr})
    return labels, samples, rows, notices


def load_audio(path, sr=16000):
    y, source_sr = sf.read(path, dtype="float64", always_2d=True)
    if not y.size or not np.isfinite(y).all():
        raise ValueError(f"Audio rỗng hoặc NaN/Inf: {path}")
    y = y.mean(axis=1)
    if source_sr != sr:
        y = librosa.resample(y, orig_sr=source_sr, target_sr=sr)
    if np.max(np.abs(y)) < 1e-8:
        raise ValueError(f"Audio im lặng: {path}")
    return y


def extract_array(y, config=Config(), return_parts=False):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-8:
        raise ValueError("Tín hiệu phải là vector mono hữu hạn, có tiếng.")
    y, bounds = librosa.effects.trim(y, top_db=config.top_db, frame_length=400, hop_length=160)
    if len(y) < config.win_length:
        raise ValueError("Audio sau cắt silence ngắn hơn 25 ms; cần thu lại.")
    y = librosa.effects.preemphasis(y, coef=0.97)
    mfcc = librosa.feature.mfcc(y=y, sr=config.sr, n_mfcc=config.n_mfcc,
        n_fft=config.n_fft, win_length=config.win_length, hop_length=config.hop_length,
        window="hamming", n_mels=config.n_mels, fmin=20, fmax=config.sr / 2)
    # mode=nearest supports short words. Delta-delta is a second derivative.
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1)
    if width < 3:
        raise ValueError("Cần ít nhất 3 frame để tính delta.")
    d1 = librosa.feature.delta(mfcc, width=width, order=1, mode="nearest")
    d2 = librosa.feature.delta(mfcc, width=width, order=2, mode="nearest")
    raw = np.vstack([mfcc, d1, d2]).T if config.dynamic else mfcc.T
    # Per-utterance CMVN: no shared test statistics, same transform at inference.
    features = (raw - raw.mean(axis=0)) / np.maximum(raw.std(axis=0), 1e-3)
    if not np.isfinite(features).all():
        raise ValueError("Đặc trưng không hữu hạn.")
    if return_parts:
        return features, (mfcc, d1, d2), bounds
    return features


def extract_features(path, config=Config()):
    return extract_array(load_audio(path, config.sr), config)


def lr_transition(n):
    if n < 2:
        raise ValueError("Cần >=2 state.")
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i], A[i, i + 1] = 0.7, 0.3
    A[-1, -1] = 1
    return A


def check_topology(model):
    n = model.n_components
    allowed = np.eye(n, dtype=bool) | np.eye(n, k=1, dtype=bool)
    if (not np.allclose(model.startprob_, np.eye(n)[0])
        or not np.isfinite(model.transmat_).all()
        or np.any(model.transmat_ < 0)
        or not np.allclose(model.transmat_.sum(axis=1), 1)
        or np.any(model.transmat_[~allowed] != 0)):
        raise ValueError("Topology trái-sang-phải đã bị phá hoặc hàng transition không hợp lệ.")


class TraceMonitor(ConvergenceMonitor):
    def __init__(self, tol, n_iter):
        super().__init__(tol, n_iter, False)
        self.trace = []

    def report(self, log_prob):
        self.trace.append(float(log_prob))
        super().report(log_prob)


def make_lr_hmm(sequences, config=Config()):
    """Temporal initialization keeps emissions ordered; fitted on train only."""
    if config.covariance_type not in ("diag", "full"):
        raise ValueError("Chỉ hỗ trợ diag/full.")
    if any(len(x) < config.n_states for x in sequences):
        raise ValueError("Utterance quá ngắn so với số state.")
    n, d = config.n_states, sequences[0].shape[1]
    model = GaussianHMM(n_components=n, covariance_type=config.covariance_type,
        n_iter=config.n_iter, tol=config.tol, min_covar=1e-3,
        init_params="", params="tmc", random_state=config.seed,
        implementation="log", covars_prior=0.01 if config.covariance_type == "diag" else np.eye(d) * 0.1,
        covars_weight=1 if config.covariance_type == "diag" else d + 2)
    model.startprob_ = np.eye(n)[0]
    model.transmat_ = lr_transition(n)
    pools = [np.vstack([np.array_split(x, n)[i] for x in sequences]) for i in range(n)]
    model.means_ = np.vstack([x.mean(axis=0) for x in pools])
    if config.covariance_type == "diag":
        model.covars_ = np.vstack([np.maximum(x.var(axis=0), 1e-2) for x in pools])
    else:
        model.covars_ = np.stack([np.atleast_2d(np.cov(x, rowvar=False)) + np.eye(d) * 0.1 for x in pools])
    model.monitor_ = TraceMonitor(config.tol, config.n_iter)
    return model


def fit_models(samples, feature_map, labels, config=Config()):
    if any(s.split != "train" for s in samples):
        raise ValueError("fit_models chỉ nhận samples có split=train.")
    models, diagnostics = {}, []
    for label in labels:
        seq = [feature_map[str(s.path)] for s in samples if s.label == label]
        if not seq:
            raise ValueError(f"Không có train cho lớp {label}.")
        model = make_lr_hmm(seq, config)
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        logger = logging.getLogger("hmmlearn.base")
        logger.addHandler(handler)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(np.vstack(seq), [len(x) for x in seq])
            messages = [str(w.message) for w in caught]
            messages.extend(log_buffer.getvalue().strip().splitlines())
        finally:
            logger.removeHandler(handler)
        check_topology(model)
        train_score = float(model.score(np.vstack(seq), [len(x) for x in seq]))
        if not np.isfinite(train_score):
            raise ValueError(f"{label}: model không có train score hữu hạn.")
        trace = model.monitor_.trace
        gain = trace[-1] - trace[-2] if len(trace) > 1 else None
        stop = ("likelihood_decreased" if gain is not None and gain < -1e-6 else
                "tolerance" if gain is not None and 0 <= gain < config.tol else "max_iter")
        diagnostics.append({"label": label, "utterances": len(seq), "frames": sum(map(len, seq)),
            "iterations": len(trace), "stop_reason": stop, "last_gain": gain,
            "library_converged": bool(model.monitor_.converged), "train_log_likelihood": train_score,
            "trace": trace, "startprob": model.startprob_.tolist(),
            "transmat": model.transmat_.tolist(), "warnings": messages})
        models[label] = model
    return models, diagnostics


def acoustic_scores(features, models):
    scores = {label: float(model.score(features)) for label, model in models.items()}
    if not all(np.isfinite(s) for s in scores.values()):
        raise ValueError("Có acoustic score NaN/-inf. Kiểm tra model/dữ liệu.")
    return scores


def recognize(path, models, config=Config()):
    scores = acoustic_scores(extract_features(path, config), models)
    return max(scores, key=scores.get), scores


def evaluate(samples, feature_map, models, labels):
    rows = []
    begin = time.perf_counter()
    for sample in samples:
        F = feature_map[str(sample.path)]
        scores = acoustic_scores(F, models)
        ranked = sorted(scores, key=scores.get, reverse=True)
        rows.append({"file": f"{sample.split}/{sample.label}/{sample.path.name}", "true_label": sample.label,
            "prediction": ranked[0], "correct": ranked[0] == sample.label, "frames": len(F),
            "top2": ranked[1], "margin_per_frame": (scores[ranked[0]] - scores[ranked[1]]) / len(F),
            "best_score_per_frame": scores[ranked[0]] / len(F),
            **{f"score_{label}": scores[label] for label in labels}})
    elapsed = time.perf_counter() - begin
    accuracy = float(accuracy_score([r["true_label"] for r in rows], [r["prediction"] for r in rows]))
    matrix = confusion_matrix([r["true_label"] for r in rows], [r["prediction"] for r in rows], labels=labels)
    return rows, accuracy, matrix, elapsed / len(samples)


def safe_log(probabilities):
    p = np.asarray(probabilities, dtype=float)
    out = np.full(p.shape, -np.inf)
    np.log(p, out=out, where=p > 0)
    return out


def forward_backward(log_emission, start, transitions):
    """Educational log-domain Forward/Backward, gamma and summed xi; no training."""
    b, a, pi = np.asarray(log_emission), safe_log(transitions), safe_log(start)
    T, n = b.shape
    alpha = np.empty((T, n))
    alpha[0] = pi + b[0]
    for t in range(1, T):
        alpha[t] = b[t] + logsumexp(alpha[t - 1, :, None] + a, axis=0)
    ll = float(logsumexp(alpha[-1]))
    if not np.isfinite(ll):
        raise ValueError("Không có đường trạng thái hợp lệ.")
    beta = np.zeros((T, n))
    for t in range(T - 2, -1, -1):
        beta[t] = logsumexp(a + b[t + 1][None, :] + beta[t + 1][None, :], axis=1)
    gamma = np.exp(alpha + beta - ll)
    xi_sum = np.zeros((n, n))
    for t in range(T - 1):
        xi_sum += np.exp(alpha[t, :, None] + a + b[t + 1][None, :] + beta[t + 1][None, :] - ll)
    return ll, gamma, xi_sum


def diag_log_emission(F, model):
    if model.covariance_type != "diag":
        raise ValueError("Hàm minh họa chỉ dùng diag.")
    variances = np.diagonal(model.covars_, axis1=1, axis2=2)
    return -0.5 * (F.shape[1] * np.log(2 * np.pi) + np.log(variances).sum(axis=1)[None, :]
                   + ((F[:, None, :] - model.means_[None, :, :]) ** 2 / variances[None, :, :]).sum(axis=2))


@njit(cache=True)
def _dtw_core(cost):
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    L = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, length = D[i - 1, j - 1], L[i - 1, j - 1]
            if D[i - 1, j] < best:
                best, length = D[i - 1, j], L[i - 1, j]
            if D[i, j - 1] < best:
                best, length = D[i, j - 1], L[i, j - 1]
            D[i, j], L[i, j] = best + cost[i - 1, j - 1], length + 1
    return D[n, m] / L[n, m]


def dtw_distance(X, Y):
    return float(_dtw_core(cdist(X, Y, metric="euclidean")))


def evaluate_dtw(samples, feature_map, templates, labels):
    dtw_distance(np.zeros((3, 2)), np.zeros((3, 2)))  # exclude JIT compilation from timing
    rows = []
    begin = time.perf_counter()
    for s in samples:
        distances = {w: min(dtw_distance(feature_map[str(s.path)], ref) for ref in templates[w]) for w in labels}
        pred = min(distances, key=distances.get)
        rows.append({"file": f"{s.split}/{s.label}/{s.path.name}", "true_label": s.label,
            "prediction": pred, "correct": pred == s.label,
            **{f"distance_{w}": distances[w] for w in labels}})
    elapsed = time.perf_counter() - begin
    accuracy = float(accuracy_score([r["true_label"] for r in rows], [r["prediction"] for r in rows]))
    cm = confusion_matrix([r["true_label"] for r in rows], [r["prediction"] for r in rows], labels=labels)
    return rows, accuracy, cm, elapsed / len(samples)


def parse_lexicon(path):
    lexicon = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.partition("#")[0].strip().split()
        if not tokens:
            continue
        if len(tokens) < 2 or tokens[0] in lexicon:
            raise ValueError(f"Lexicon dòng {number}: thiếu phone hoặc từ trùng.")
        lexicon[tokens[0]] = tokens[1:]
    return lexicon


def build_lm(path, vocabulary, k=0.5):
    if k <= 0:
        raise ValueError("k phải >0.")
    uni, bi, context = Counter(), Counter(), Counter()
    vocabulary = list(vocabulary)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        words = line.partition("#")[0].strip().split()
        if not words:
            continue
        unknown = set(words) - set(vocabulary)
        if unknown:
            raise ValueError(f"Corpus chứa từ ngoài vocabulary: {sorted(unknown)}")
        uni.update(words)
        # Predict words only; do not count </s> in a vocabulary that excludes it.
        for previous, current in zip(["<s>"] + words[:-1], words):
            bi[previous, current] += 1
            context[previous] += 1
    if not uni:
        raise ValueError("Corpus không có câu/lệnh.")
    return {"unigram": uni, "bigram": bi, "context": context, "vocabulary": vocabulary, "k": k}


def bigram_prob(lm, previous, word):
    if word not in lm["vocabulary"]:
        raise ValueError(f"Từ ngoài vocabulary: {word}")
    return (lm["bigram"][previous, word] + lm["k"]) / (lm["context"][previous] + lm["k"] * len(lm["vocabulary"]))


def rerank(scores_per_frame, previous, lm, weight):
    totals = {w: score + weight * math.log(bigram_prob(lm, previous, w)) for w, score in scores_per_frame.items()}
    return max(totals, key=totals.get), totals


def add_noise(y, snr_db, seed):
    rng = np.random.default_rng(seed)
    # Estimate reference speech power on active region so silence doesn't lower SNR.
    active, _ = librosa.effects.trim(y, top_db=30, frame_length=400, hop_length=160)
    power = float(np.mean(active ** 2))
    noise = rng.normal(size=len(y))
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    return y + noise


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(matrix, labels, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(matrix, display_labels=[VI.get(w, w) for w in labels]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    save_figure(fig, path)


def save_visuals(samples, features, models, diagnostics, labels, output, config):
    plots = output / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    selected = [next(s for s in samples if s.label == label) for label in labels[:2]]
    alignment_rows = []
    for idx, s in enumerate(selected, 1):
        F, parts, bounds = extract_array(load_audio(s.path), config, return_parts=True)
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        for ax, part, name in zip(axes, parts, ["MFCC", "Delta MFCC", "Delta² MFCC"]):
            im = ax.imshow(part, origin="lower", aspect="auto", cmap="magma")
            ax.set_ylabel(name)
            fig.colorbar(im, ax=ax, pad=0.01)
        axes[-1].set_xlabel("Frame (bước 10 ms)")
        fig.suptitle(f"{VI.get(s.label, s.label)} — {s.path.name} (trước CMVN)")
        fig.tight_layout()
        save_figure(fig, plots / f"features_{idx}.png")
        logp, states = models[s.label].decode(F, algorithm="viterbi")
        if np.any(np.diff(states) < 0) or np.any(np.diff(states) > 1):
            raise ValueError("Viterbi path vi phạm topology.")
        ll, gamma, xi_sum = forward_backward(diag_log_emission(F, models[s.label]),
                                             models[s.label].startprob_, models[s.label].transmat_)
        if not np.isclose(ll, models[s.label].score(F), atol=1e-6):
            raise ValueError("Forward tự viết khác hmmlearn.")
        fig, axes = plt.subplots(2, 1, figsize=(9, 4.7), sharex=True)
        t = np.arange(len(states)) * config.hop_length / config.sr
        axes[0].step(t, states, where="post")
        axes[0].set_yticks(range(config.n_states))
        axes[0].set_ylabel("Viterbi state")
        axes[0].set_title(f"{s.path.name} / model {s.label}; Forward={ll:.1f}, Viterbi={logp:.1f}")
        axes[1].imshow(gamma.T, aspect="auto", origin="lower", extent=[0, t[-1], -0.5, config.n_states - 0.5], cmap="Blues", vmin=0, vmax=1)
        axes[1].set_ylabel("State; màu = gamma")
        axes[1].set_xlabel("Giây kể từ đầu đoạn sau trim")
        fig.tight_layout()
        save_figure(fig, plots / f"viterbi_{idx}.png")
        np.savez_compressed(output / f"alignment_{idx}.npz", states=states, gamma=gamma, xi_sum=xi_sum)
        for state, frames in enumerate(np.bincount(states, minlength=config.n_states)):
            alignment_rows.append({"file": str(s.path), "label": s.label, "state": state,
                "frames": int(frames), "occupancy_ms_approx": int(frames) * 10,
                "trim_start_sample": int(bounds[0]), "forward_logp": ll, "viterbi_logp": float(logp)})
    write_csv(output / "state_durations.csv", alignment_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    for diag in diagnostics:
        ax.plot(np.arange(1, len(diag["trace"]) + 1), diag["trace"], label=diag["label"])
    ax.set(xlabel="Vòng EM", ylabel="Train log-likelihood", title="Baum–Welch: lịch sử tối ưu")
    ax.legend()
    save_figure(fig, plots / "convergence.png")
    fig, axes = plt.subplots(1, len(labels), figsize=(15, 3))
    for ax, label in zip(axes, labels):
        ax.imshow(models[label].transmat_, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(label)
        ax.set_xlabel("State đến")
    axes[0].set_ylabel("State đi")
    fig.tight_layout()
    save_figure(fig, plots / "transitions.png")


def run_lab(dataset=ROOT / "dataset", output=ROOT, model_dir=ROOT / "models",
            corpus=ROOT / "corpus.txt", lexicon=ROOT / "lexicon.txt", config=Config(), snr_db=20):
    dataset, output, model_dir = Path(dataset), Path(output), Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    labels, samples, manifest, notices = read_dataset(dataset)
    provenance_path = dataset / "manifest.json"
    if not provenance_path.exists():
        provenance_path = dataset / "import_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    train = [s for s in samples if s.split == "train"]
    test = [s for s in samples if s.split == "test"]
    print(f"Dữ liệu: {len(labels)} lớp, {len(train)} train, {len(test)} test", flush=True)
    for notice in notices:
        print("Lưu ý:", notice)
    write_csv(output / "dataset_audit.csv", manifest)
    write_json(output / "split.json", {"dataset": str(dataset.resolve()), "files": manifest})
    print("Trích MFCC 39-D...", flush=True)
    begin = time.perf_counter()
    features = {str(s.path): extract_features(s.path, config) for s in samples}
    feature_seconds = time.perf_counter() - begin
    write_csv(output / "feature_shapes.csv", [{"file": str(s.path), "frames": len(features[str(s.path)]),
        "dimensions": features[str(s.path)].shape[1]} for s in samples])
    print("Huấn luyện baseline 5-state...", flush=True)
    begin = time.perf_counter()
    models, diagnostics = fit_models(train, features, labels, config)
    fit_seconds = time.perf_counter() - begin
    joblib.dump({"models": models, "labels": labels, "config": asdict(config),
                 "training_files": [r for r in manifest if r["split"] == "train"],
                 "hmmlearn_version": version("hmmlearn")}, model_dir / "hmm_baseline.joblib")
    write_json(output / "training_diagnostics.json", diagnostics)
    train_rows, train_acc, _, _ = evaluate(train, features, models, labels)
    rows, test_acc, matrix, hmm_seconds = evaluate(test, features, models, labels)
    write_csv(output / "hmm_train_predictions.csv", train_rows)
    write_csv(output / "hmm_test_predictions.csv", rows)
    write_csv(output / "likelihood_examples.csv", [next(r for r in rows if r["true_label"] == w) for w in labels[:3]])
    plot_confusion(matrix, labels, f"HMM 39-D — accuracy {test_acc:.0%}", output / "figures/confusion_hmm.png")
    save_visuals(test, features, models, diagnostics, labels, output, config)
    # Predeclared ablations: baseline remains fixed; no best-test-model selection.
    experiments = []
    variants = [("E1_states3", replace(config, n_states=3)),
                ("baseline_states5_39D", config),
                ("E1_states7", replace(config, n_states=7)),
                ("E2_mfcc13", replace(config, dynamic=False))]
    for name, cfg in variants:
        print(f"Thí nghiệm {name}...", flush=True)
        fmap = features if cfg.dynamic else {str(s.path): features[str(s.path)][:, :13] for s in samples}
        try:
            fitted, diag = (models, diagnostics) if cfg == config else fit_models(train, fmap, labels, cfg)
            tr, ta, _, _ = evaluate(train, fmap, fitted, labels)
            te, va, _, _ = evaluate(test, fmap, fitted, labels)
            write_csv(output / f"{name}_predictions.csv", te)
            write_json(output / f"{name}_diagnostics.json", diag)
            experiments.append({"experiment": name, "states": cfg.n_states, "dimensions": 39 if cfg.dynamic else 13,
                "covariance": cfg.covariance_type, "train_accuracy": ta, "test_accuracy": va,
                "max_iter_models": sum(d["stop_reason"] == "max_iter" for d in diag), "status": "ok"})
        except (ValueError, np.linalg.LinAlgError) as exc:
            experiments.append({"experiment": name, "states": cfg.n_states, "dimensions": 39 if cfg.dynamic else 13,
                "covariance": cfg.covariance_type, "train_accuracy": None, "test_accuracy": None,
                "max_iter_models": None, "status": str(exc)})
    # Full covariance needs substantially more observations/state than feature dims.
    minimum_frames = min(sum(len(features[str(s.path)]) for s in train if s.label == w) for w in labels)
    full_reason = "skipped: quá ít frame/state so với D=39; E3 chỉ yêu cầu nếu đủ dữ liệu"
    if minimum_frames >= config.n_states * 39 * 2:
        try:
            full, fdiag = fit_models(train, features, labels, replace(config, covariance_type="full"))
            _, ftrain, _, _ = evaluate(train, features, full, labels)
            frows, ftest, _, _ = evaluate(test, features, full, labels)
            write_csv(output / "E3_full_predictions.csv", frows)
            write_json(output / "E3_full_diagnostics.json", fdiag)
            full_reason = "ok"
        except (ValueError, np.linalg.LinAlgError) as exc:
            ftrain = ftest = None
            full_reason = f"failed: {exc}"
    else:
        ftrain = ftest = None
    experiments.append({"experiment": "E3_full", "states": config.n_states, "dimensions": 39,
        "covariance": "full", "train_accuracy": ftrain, "test_accuracy": ftest,
        "max_iter_models": None, "status": full_reason})
    write_csv(output / "experiments.csv", experiments)
    print("So sánh DTW dùng chung front-end và mã Lab 2...", flush=True)
    begin = time.perf_counter()
    templates = {w: [features[str(s.path)] for s in train if s.label == w] for w in labels}
    dtw_fit_seconds = time.perf_counter() - begin
    drows, dacc, dcm, dseconds = evaluate_dtw(test, features, templates, labels)
    write_csv(output / "dtw_shared_predictions.csv", drows)
    plot_confusion(dcm, labels, f"DTW cùng front-end 39-D — {dacc:.0%}", output / "figures/confusion_dtw_shared.png")
    import lab2_baseline as lab2
    begin = time.perf_counter()
    legacy = {str(s.path): lab2.extract_feature(s.path, lab2.Config()) for s in samples}
    legacy_feature_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    legacy_templates = {w: [legacy[str(s.path)] for s in train if s.label == w] for w in labels}
    legacy_fit = time.perf_counter() - begin
    # Same Lab 2 recurrence, tie rule and min-template decision; compiled for timing.
    lrows, lacc, lcm, lseconds = evaluate_dtw(test, legacy, legacy_templates, labels)
    write_csv(output / "dtw_lab2_predictions.csv", lrows)
    plot_confusion(lcm, labels, f"DTW front-end Lab 2 — {lacc:.0%}", output / "figures/confusion_dtw_lab2.png")
    # Verify compiled recurrence against the original implementation.
    x, y = legacy[str(train[0].path)], legacy[str(test[0].path)]
    if not np.isclose(dtw_distance(x, y), lab2.dtw_distance(x, y)[0], atol=1e-9):
        raise AssertionError("DTW tối ưu khác thuật toán Lab 2.")
    comparisons = [
        {"method": "HMM_39D", "train_accuracy": train_acc, "test_accuracy": test_acc, "model_fit_seconds": fit_seconds,
         "frontend_seconds_per_file": feature_seconds / len(samples), "decode_seconds_per_trial": hmm_seconds},
        {"method": "DTW_shared_39D", "train_accuracy": None, "test_accuracy": dacc, "model_fit_seconds": dtw_fit_seconds,
         "frontend_seconds_per_file": feature_seconds / len(samples), "decode_seconds_per_trial": dseconds},
        {"method": "DTW_Lab2_13D", "train_accuracy": None, "test_accuracy": lacc, "model_fit_seconds": legacy_fit,
         "frontend_seconds_per_file": legacy_feature_seconds / len(samples), "decode_seconds_per_trial": lseconds}]
    write_csv(output / "dtw_vs_hmm.csv", comparisons)
    cases = [{"file": h["file"], "truth": h["true_label"], "hmm": h["prediction"], "dtw_shared": d["prediction"],
              "dtw_lab2": l["prediction"], "hmm_correct": h["correct"], "dtw_correct": l["correct"]}
             for h, d, l in zip(rows, drows, lrows)]
    write_csv(output / "comparison_cases.csv", cases)
    print("Thử số mẫu train và mismatch nhiễu...", flush=True)
    size_results = []
    counts = min(sum(s.label == w for s in train) for w in labels)
    sizes = sorted(set([1, min(2, counts), counts]))
    for count in sizes:
        subset = [s for w in labels for s in [z for z in train if z.label == w][:count]]
        small, _ = fit_models(subset, features, labels, config) if count != counts else (models, diagnostics)
        _, ha, _, _ = evaluate(test, features, small, labels)
        st = {w: [features[str(s.path)] for s in subset if s.label == w] for w in labels}
        _, da, _, _ = evaluate_dtw(test, features, st, labels)
        size_results.append({"train_per_class": count, "hmm_accuracy": ha, "dtw_shared_accuracy": da})
    write_csv(output / "training_sizes.csv", size_results)
    noisy_features, noisy_legacy, noise_analysis = {}, {}, []
    for i, s in enumerate(test):
        clean_y = load_audio(s.path)
        noisy_y = add_noise(clean_y, snr_db, config.seed + i)
        noisy_F, _, noisy_bounds = extract_array(noisy_y, config, return_parts=True)
        _, clean_bounds = librosa.effects.trim(clean_y, top_db=config.top_db, frame_length=400, hop_length=160)
        noisy_features[str(s.path)] = noisy_F
        # Apply the same Lab 2 peak normalization/endpoint/MFCC to the same noise.
        old_y = noisy_y / np.max(np.abs(noisy_y))
        old_y, _ = lab2.detect_endpoints(old_y, lab2.Config())
        noisy_legacy[str(s.path)] = lab2.mfcc_feature(old_y, lab2.Config())
        noise_analysis.append({"file": f"test/{s.label}/{s.path.name}",
            "clean_frames": len(features[str(s.path)]), "noisy_frames": len(noisy_F),
            "clean_trim_seconds": float(np.diff(clean_bounds)[0] / config.sr),
            "noisy_trim_seconds": float(np.diff(noisy_bounds)[0] / config.sr),
            "input_seconds": len(clean_y) / config.sr})
    noisy_rows, noisy_acc, ncm, _ = evaluate(test, noisy_features, models, labels)
    write_csv(output / "noise_predictions.csv", noisy_rows)
    plot_confusion(ncm, labels, f"HMM + nhiễu {snr_db:g} dB — {noisy_acc:.0%}", output / "figures/confusion_noise.png")
    ndrows, ndacc, _, _ = evaluate_dtw(test, noisy_features, templates, labels)
    nlrows, nlacc, _, _ = evaluate_dtw(test, noisy_legacy, legacy_templates, labels)
    write_csv(output / "dtw_shared_noise_predictions.csv", ndrows)
    write_csv(output / "dtw_lab2_noise_predictions.csv", nlrows)
    write_csv(output / "noise_methods.csv", [
        {"method": "HMM_39D", "clean_accuracy": test_acc, "noise_accuracy": noisy_acc},
        {"method": "DTW_shared_39D", "clean_accuracy": dacc, "noise_accuracy": ndacc},
        {"method": "DTW_Lab2_13D", "clean_accuracy": lacc, "noise_accuracy": nlacc}])
    write_csv(output / "comparison_cases_noise.csv", [
        {"file": h["file"], "truth": h["true_label"], "hmm": h["prediction"], "dtw_shared": d["prediction"],
         "dtw_lab2": l["prediction"], "hmm_correct": h["correct"], "dtw_correct": d["correct"]}
        for h, d, l in zip(noisy_rows, ndrows, nlrows)])
    for info, sample, prediction in zip(noise_analysis, test, noisy_rows):
        _, path = models[prediction["prediction"]].decode(noisy_features[str(sample.path)], algorithm="viterbi")
        info.update({"true_label": sample.label, "prediction": prediction["prediction"],
            "states_visited": ",".join(map(str, np.unique(path))),
            "dominant_state_fraction": float(np.bincount(path).max() / len(path)),
            "margin_per_frame": prediction["margin_per_frame"]})
    write_csv(output / "noise_frontend_analysis.csv", noise_analysis)
    # Train-only holdout for an illustrative reject criterion. Never calibrate on test.
    validation = [next(s for s in reversed(train) if s.label == w) for w in labels]
    held_paths = {s.path for s in validation}
    calibration_train = [s for s in train if s.path not in held_paths]
    calibration_models, _ = fit_models(calibration_train, features, labels, config)
    val_rows, val_acc, _, _ = evaluate(validation, features, calibration_models, labels)
    write_csv(output / "train_holdout_predictions.csv", val_rows)
    correct_val = [r for r in val_rows if r["correct"]]
    threshold = float(np.quantile([r["best_score_per_frame"] for r in correct_val], 0.05)) if correct_val else None
    margin_threshold = float(np.quantile([r["margin_per_frame"] for r in correct_val], 0.05)) if correct_val else None
    reject = {"calibration_source": "last train utterance/class held out; zero test used",
        "calibration_files": [str(s.path) for s in validation], "validation_accuracy": val_acc,
        "correct_calibration_count": len(correct_val), "score_threshold": threshold,
        "margin_threshold": margin_threshold, "status": "proposal_only",
        "limitation": "Rất ít validation; model train lại trên toàn train có thể đổi phân bố score. Cần validation/unknown độc lập trước khi áp dụng."}
    write_json(output / "reject_proposal.json", reject)
    distributions = []
    for condition, condition_rows in [("clean", rows), ("noise", noisy_rows)]:
        for r in condition_rows:
            correct_score = r[f"score_{r['true_label']}"] / r["frames"]
            wrong = max(r[f"score_{w}"] / r["frames"] for w in labels if w != r["true_label"])
            distributions.append({"condition": condition, "file": r["file"], "true_class_score_per_frame": correct_score,
                "best_wrong_class_score_per_frame": wrong})
    write_csv(output / "score_distributions.csv", distributions)
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, name in [("true_class_score_per_frame", "Đúng lớp"), ("best_wrong_class_score_per_frame", "Sai lớp tốt nhất")]:
        ax.hist([r[key] for r in distributions if r["condition"] == "clean"], bins=8, alpha=0.55, label=name)
    ax.set(xlabel="Log-likelihood/frame", ylabel="Số utterance", title="Phân bố score trên test sạch")
    ax.legend()
    save_figure(fig, output / "figures/score_distributions.png")
    print("Lexicon và bigram re-ranking...", flush=True)
    parsed = parse_lexicon(lexicon)
    missing = sorted(set(labels) - set(parsed))
    lex_status = {"entries": parsed, "missing_words": missing,
                  "status": "waiting_for_instructor_inventory" if missing else "format_valid_inventory_not_verified"}
    write_json(output / "lexicon_status.json", lex_status)
    lm = build_lm(corpus, labels)
    write_csv(output / "unigram_counts.csv", [{"word": w, "count": lm["unigram"][w]} for w in labels])
    write_csv(output / "bigram_probabilities.csv", [{"previous": prev, "word": w, "count": lm["bigram"][prev, w],
        "context_count": lm["context"][prev], "probability": bigram_prob(lm, prev, w)}
        for prev in ["<s>"] + labels for w in labels])
    ambiguous = sorted(rows, key=lambda r: r["margin_per_frame"])[:3]
    reranking = []
    for r in ambiguous:
        scores = {w: r[f"score_{w}"] / r["frames"] for w in labels}
        # Enumerate contexts instead of choosing context from the true test label.
        for previous in ["<s>"] + labels:
            for weight in [0, 0.1, 0.3, 1.0]:
                pred, totals = rerank(scores, previous, lm, weight)
                reranking.append({"file": r["file"], "true_label": r["true_label"], "previous": previous,
                    "context_source": "hypothetical_not_observed", "lm_weight": weight,
                    "acoustic_prediction": r["prediction"], "prediction": pred,
                    "changed": pred != r["prediction"], "correct": pred == r["true_label"],
                    **{f"total_{w}": totals[w] for w in labels}})
    write_csv(output / "lm_reranking.csv", reranking)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: version(p) for p in ["numpy", "scipy", "librosa", "soundfile", "scikit-learn", "hmmlearn", "numba"]}}
    write_json(output / "environment.json", environment)
    summary = {"labels": labels, "config": asdict(config), "train_count": len(train), "test_count": len(test),
        "dataset_path": str(dataset.resolve()), "data_source": provenance.get("data_source", "provided_wav"),
        "is_synthetic": provenance.get("is_synthetic", False), "voices": provenance.get("voices", []),
        "train_accuracy": train_acc, "test_accuracy": test_acc, "noise_accuracy": noisy_acc, "snr_db": snr_db,
        "noise_drop_percentage_points": 100 * (test_acc - noisy_acc), "dtw_shared_accuracy": dacc, "dtw_lab2_accuracy": lacc,
        "dtw_shared_noise_accuracy": ndacc, "dtw_lab2_noise_accuracy": nlacc,
        "notices": notices, "lexicon_missing": missing,
        "protocol": "Fixed 5-state 39-D baseline; ablations descriptive, no hyperparameter selection using test.",
        "lm_note": "Corpus tự tạo; previous-word contexts giả định, không báo accuracy nhận dạng chuỗi thực.",
        "speaker_note": ("Cùng synthetic voice ở train/test; không đánh giá khái quát sang người nói thật."
                         if provenance.get("is_synthetic", False) else
                         "Split theo file; không có speaker metadata để tuyên bố speaker-independent.")}
    write_json(output / "summary.json", summary)
    from report import build_report
    build_report(output)
    print(f"Hoàn tất: HMM={test_acc:.1%}, DTW Lab2={lacc:.1%}, noise={noisy_acc:.1%}. Xem {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Huấn luyện, đánh giá, xuất hình/bảng/báo cáo")
    run.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    run.add_argument("--output", type=Path, default=ROOT)
    run.add_argument("--models", type=Path, default=ROOT / "models")
    run.add_argument("--corpus", type=Path, default=ROOT / "corpus.txt")
    run.add_argument("--lexicon", type=Path, default=ROOT / "lexicon.txt")
    run.add_argument("--snr-db", type=float, default=20)
    pred = sub.add_parser("predict", help="Nhận dạng WAV bằng model đã lưu")
    pred.add_argument("wav", type=Path)
    pred.add_argument("--model", type=Path, default=ROOT / "models/hmm_baseline.joblib")
    args = parser.parse_args()
    if args.command == "run":
        run_lab(args.dataset, args.output, args.models, args.corpus, args.lexicon, snr_db=args.snr_db)
    else:
        payload = joblib.load(args.model)
        prediction, scores = recognize(args.wav, payload["models"], Config(**payload["config"]))
        print("Dự đoán:", VI.get(prediction, prediction))
        for label, score in sorted(scores.items(), key=lambda item: -item[1]):
            print(f"{label:8s} {score:12.3f}")


if __name__ == "__main__":
    # Use the importable module name so joblib never serializes __main__.TraceMonitor.
    from lab3 import main as entrypoint
    entrypoint()
