"""Figures, audio audit, and a report based only on measured Lab 2 outputs."""
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from lab2 import (Config, LABELS, VI_LABELS, detect_endpoints,
                  dtw_distance, extract_feature, load_audio, mel_filterbank,
                  time_features, write_csv)


def time_caption(path, config):
    return (f"Waveform, log-energy và ZCR của tệp {Path(path).name}; "
            f"tín hiệu mono {config.sr / 1000:g} kHz, frame {1000 * config.frame / config.sr:g} ms, "
            f"hop {1000 * config.hop / config.sr:g} ms. Ba panel lần lượt là biên độ chuẩn hóa, "
            "log-energy (dB) cùng mức nền/ngưỡng endpoint, và ZCR (crossing/sample); "
            "trục ngang là thời gian (s), vùng xanh là đoạn được giữ lại.")


def endpoint_caption(path, config):
    return (f"So sánh waveform trước và sau endpoint của tệp {Path(path).name}; "
            f"hai vạch đỏ đánh dấu biên cắt, giữ đệm {config.margin_ms:g} ms mỗi phía. "
            "Panel trên là tín hiệu gốc, panel dưới là đoạn đã cắt với gốc thời gian mới; "
            "trục ngang là thời gian (s), trục dọc là biên độ chuẩn hóa.")


def autocorrelation_caption(pitch):
    return (f"Frame có energy lớn nhất của tệp {pitch['file']} và tự tương quan ngắn hạn; "
            "panel trái biểu diễn frame đã bỏ DC và nhân Hamming theo thời gian (ms), "
            "panel phải biểu diễn R[k]/R[0] theo lag (mẫu). "
            f"Tìm đỉnh trong miền 70–400 Hz, lag {pitch['lag']} mẫu cho F0 xấp xỉ {pitch['pitch_hz']:.1f} Hz.")


def mel_caption(config):
    return (f"Mel filterbank gồm {config.n_mels} bộ lọc tam giác tại Fs = {config.sr / 1000:g} kHz "
            f"và NFFT = {config.n_fft}; trục ngang là tần số (Hz), trục dọc là trọng số. "
            "Khoảng cách theo Hz giữa các bộ lọc tăng dần ở vùng tần số cao.")


def mfcc_caption(paths, config):
    names = ", ".join(Path(p).name for p in paths)
    return (f"Heatmap MFCC của các tệp {names}; mỗi frame có {config.n_mfcc} hệ số "
            "và đã chuẩn hóa trung bình cepstral (CMN). Trục ngang là thời gian tương đối "
            "sau trim (s), trục dọc là chỉ số hệ số; các panel dùng chung thang màu.")


def dtw_caption(file_x, file_y):
    same_word = Path(file_x).stem.rsplit("_", 1)[0] == Path(file_y).stem.rsplit("_", 1)[0]
    comparison = "Hai lần nói cùng từ" if same_word else "Hai từ khác nhau"
    return (f"{comparison}: căn chỉnh DTW giữa {Path(file_x).name} và {Path(file_y).name}; "
            "panel trái là ma trận khoảng cách Euclid cục bộ C, panel phải là chi phí tích lũy D. "
            "Đường xanh cyan là đường đi tối ưu; trục ngang là chỉ số frame Y, trục dọc là "
            "chỉ số frame X. DTW_norm và số cặp frame trên đường đi được ghi trên hình.")


def confusion_caption():
    return ("Ma trận nhầm lẫn của baseline, E1 không endpoint và E2 MFCC + Δ trên cùng 10 file test; "
            "trục ngang là nhãn dự đoán, trục dọc là nhãn thật, số trong mỗi ô là số mẫu. "
            "Accuracy của từng cấu hình được ghi phía trên mỗi panel.")


def save_figure(fig, output, name, caption):
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / name, dpi=140, bbox_inches="tight")
    fig.lab2_caption = caption
    return fig


def audit_dataset(root, split, output, config=Config()):
    rows, trim_rows = [], []
    for partition in ("train", "test"):
        for label in LABELS:
            for path in split[partition][label]:
                raw, sr = sf.read(path, always_2d=True)
                info = sf.info(path)
                y = load_audio(path, config)
                trimmed, boundaries = detect_endpoints(y, config)
                previous, _ = detect_endpoints(y, replace(config, endpoint_method="relative", top_db=35, margin_ms=50))
                relative = f"{label}/{path.name}"
                rows.append({"file": relative, "split": partition, "sr_hz": sr,
                             "channels": info.channels, "subtype": info.subtype,
                             "duration_s": len(raw) / sr, "peak_original": float(np.max(np.abs(raw))),
                             "clipping_fraction": float(np.mean(np.abs(raw) >= 0.999)),
                             "input_format_ok": sr == config.sr and info.channels == 1 and info.subtype.startswith("PCM")})
                trim_rows.append({"file": relative, "split": partition,
                                  "before_s": len(y) / config.sr, "after_s": len(trimmed) / config.sr,
                                  "relative_35db_after_s": len(previous) / config.sr,
                                  "start_s": boundaries["start"] / config.sr,
                                  "end_s": boundaries["end"] / config.sr,
                                  "first_frame": boundaries["first_frame"], "last_frame": boundaries["last_frame"],
                                  "method": boundaries["method"], "status": boundaries["status"],
                                  "noise_start_db": boundaries["noise_start_db"], "noise_end_db": boundaries["noise_end_db"],
                                  "threshold_db": boundaries["threshold_db"],
                                  "low_threshold_db": boundaries["low_threshold_db"]})
                destination = Path(output) / "audio_trimmed" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, trimmed, config.sr, subtype="PCM_16")
    write_csv(Path(output) / "endpoint.csv", trim_rows)
    return rows, trim_rows


def plot_time_analysis(path, output, config=Config()):
    y = load_audio(path, config)
    f = time_features(y, config)
    _, boundaries = detect_endpoints(y, config)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(np.arange(len(y)) / config.sr, y, lw=0.7)
    axes[0].set_ylabel("Biên độ")
    axes[0].set_title(f"Waveform, log-energy, ZCR — {Path(path).name}")
    axes[1].plot(f["time"], f["log_energy"], alpha=.4, label="Log-energy gốc")
    axes[1].plot(f["time"], boundaries["smoothed_log_energy"], color="tab:blue", label=f"Median {config.smooth_ms:g} ms")
    axes[1].plot(f["time"], boundaries["noise_curve_db"], color="gray", ls=":", label="Nền ước lượng")
    axes[1].plot(f["time"], boundaries["high_threshold_curve_db"], color="red", ls="--", label="Ngưỡng xác nhận")
    axes[1].plot(f["time"], boundaries["low_threshold_curve_db"], color="orange", ls="--", label="Ngưỡng mở rộng")
    axes[1].set_ylabel("Energy (dB)")
    axes[1].legend(ncol=2, fontsize=8)
    axes[2].plot(f["time"], f["zcr"], color="tab:orange")
    axes[2].set_ylabel("ZCR (crossing/sample)")
    axes[2].set_xlabel("Thời gian (s)")
    for ax in axes:
        ax.axvspan(boundaries["start"] / config.sr, boundaries["end"] / config.sr,
                   color="green", alpha=0.10)
        ax.grid(alpha=0.2)
    return save_figure(fig, output, f"time_{Path(path).stem}.png", time_caption(path, config))


def plot_endpoint(path, output, config=Config()):
    y = load_audio(path, config)
    trimmed, bounds = detect_endpoints(y, config)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    axes[0].plot(np.arange(len(y)) / config.sr, y, lw=0.7)
    for index in ("start", "end"):
        axes[0].axvline(bounds[index] / config.sr, color="red", ls="--")
    axes[0].set_title(f"Trước trim: {len(y) / config.sr:.3f} s — {Path(path).name}")
    axes[1].plot(np.arange(len(trimmed)) / config.sr, trimmed, lw=0.7)
    axes[1].set_title(f"Sau trim: {len(trimmed) / config.sr:.3f} s; margin {config.margin_ms:g} ms")
    for ax in axes:
        ax.set_xlabel("Thời gian (s)")
        ax.set_ylabel("Biên độ")
    return save_figure(fig, output, f"endpoint_{Path(path).stem}.png", endpoint_caption(path, config))


def plot_autocorrelation(path, output, config=Config()):
    y = load_audio(path, config)
    f = time_features(y, config)
    index = int(np.argmax(f["energy"]))
    frame = f["frames"][index].copy()
    frame = (frame - frame.mean()) * np.hamming(config.frame)
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    corr /= max(corr[0], 1e-12)
    # Search only plausible voiced pitches; this is an illustrative estimator.
    first, last = int(config.sr / 400), min(int(config.sr / 70), len(corr) - 1)
    lag = first + int(np.argmax(corr[first:last + 1]))
    pitch = config.sr / lag
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].plot(np.arange(len(frame)) / config.sr * 1000, frame)
    axes[0].set(xlabel="Thời gian trong frame (ms)", ylabel="Biên độ", title=f"Frame energy lớn nhất: {index}")
    axes[1].plot(np.arange(len(corr)), corr)
    axes[1].axvspan(first, last, alpha=0.1)
    axes[1].axvline(lag, color="red", ls="--")
    axes[1].set(xlabel="Lag (mẫu)", ylabel="R[k]/R[0]", title=f"Pitch minh họa ≈ {pitch:.1f} Hz")
    result = {"file": Path(path).name, "frame": index, "lag": lag,
              "pitch_hz": pitch, "peak_correlation": float(corr[lag])}
    save_figure(fig, output, "autocorrelation.png", autocorrelation_caption(result))
    return fig, result


def plot_mel(output, config=Config()):
    filters, frequencies = mel_filterbank(config)
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.plot(frequencies, filters.T)
    ax.set(xlabel="Tần số (Hz)", ylabel="Trọng số", title=f"Mel filterbank — {config.n_mels} bộ lọc tam giác")
    return save_figure(fig, output, "mel_filterbank.png", mel_caption(config))


def plot_mfcc(paths, output, config=Config(), name="mfcc_comparison.png"):
    features = [extract_feature(path, config) for path in paths]
    low = min(float(x.min()) for x in features)
    high = max(float(x.max()) for x in features)
    fig, axes = plt.subplots(len(paths), 1, figsize=(10, 3 * len(paths)), squeeze=False, constrained_layout=True)
    for ax, path, X in zip(axes[:, 0], paths, features):
        im = ax.imshow(X.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=low, vmax=high,
                       extent=[0, len(X) * config.hop / config.sr, -0.5, config.n_mfcc - 0.5])
        ax.set(xlabel="Thời gian tương đối sau trim (s)", ylabel="Hệ số MFCC",
               title=f"{Path(path).name} — shape {X.shape}")
        fig.colorbar(im, ax=ax, label="MFCC sau CMN")
    return save_figure(fig, output, name, mfcc_caption(paths, config))


def plot_dtw(path_x, path_y, output, name, config=Config()):
    X, Y = extract_feature(path_x, config), extract_feature(path_y, config)
    score, path, local, accumulated = dtw_distance(X, Y)
    p = np.array(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, values, title in zip(axes, (local, accumulated), ("Local distance C", "Accumulated cost D")):
        im = ax.imshow(values, origin="lower", aspect="auto", cmap="magma")
        ax.plot(p[:, 1], p[:, 0], color="cyan", lw=1.3, label="Optimal path")
        ax.set(xlabel=f"Frame Y: {Path(path_y).name}", ylabel=f"Frame X: {Path(path_x).name}", title=title)
        ax.legend()
        fig.colorbar(im, ax=ax)
    description = {"same_word": "Hai lần nói cùng từ", "different_word": "Hai từ khác nhau"}.get(name, "Phân tích mẫu nhận sai")
    fig.suptitle(f"{description} — DTW_norm = {score:.4f}; số cặp frame = {len(path)}")
    save_figure(fig, output, f"dtw_{name}.png", dtw_caption(path_x, path_y))
    return fig, {"pair": name, "file_x": Path(path_x).name, "file_y": Path(path_y).name,
                 "frames_x": len(X), "frames_y": len(Y), "path_length": len(path), "dtw_norm": score}


def plot_confusion(experiments, output):
    fig, axes = plt.subplots(1, len(experiments), figsize=(14, 4.5), constrained_layout=True)
    for ax, (name, result) in zip(np.atleast_1d(axes), experiments.items()):
        matrix = result["matrix"]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(int(r["matrix"].max()) for r in experiments.values()))
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="black", bbox={"facecolor": "white", "alpha": .65, "edgecolor": "none"})
        labels = [VI_LABELS[k] for k in LABELS]
        ax.set_xticks(range(len(labels)), labels, rotation=35)
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thật", title=f"{name}\nAccuracy: {result['accuracy']:.1%}")
    return save_figure(fig, output, "confusion_matrix.png", confusion_caption())


def most_confused(matrix):
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    if not off.any():
        return None
    i, j = np.unravel_index(np.argmax(off), off.shape)
    return LABELS[i], LABELS[j], int(off[i, j])


def write_report(root, output, split, experiments, summaries, quality, trim_rows, pairs, pitch, error_figures=None):
    output = Path(output)
    baseline = experiments["baseline"]
    errors = [r for r in baseline["rows"] if not r["correct"]]
    confusion = most_confused(baseline["matrix"])
    config = baseline["config"]
    figure_number = 0

    def add_figure(filename, caption):
        nonlocal figure_number
        figure_number += 1
        title = caption.split(";", 1)[0]
        lines.extend(["", f"![{title}](figures/{filename})", "", f"*Hình {figure_number}. {caption}*", ""])

    lines = ["# Lab 2 — Đặc trưng tiếng nói và nhận dạng bằng DTW", "",
             "**Sinh viên:** Lỗ Anh Việt — **MSSV:** 2351260695", "",
             "## 1. Dữ liệu và phạm vi kết quả", "",
             "Dữ liệu WAV do người dùng cung cấp. Cần ghi rõ người nói, thiết bị và môi trường thu trong báo cáo.", "",
             f"Bộ dữ liệu: `{Path(root).name}`; 5 lớp, {len(quality)} file; "
             f"{sum(len(v) for v in split['train'].values())} template và {len(baseline['rows'])} test.", "",
             "Chia theo tên file đã sắp xếp: 3 file đầu/lớp là train, phần còn lại là test. "
             "Danh sách được lưu ở `split.json`; dùng cùng split cho baseline, E1 và E2.", "",
             f"File có mẫu gần clipping (|x| ≥ 0,999): {sum(r['clipping_fraction'] > 0 for r in quality)}. "
             f"File đúng WAV mono PCM 16 kHz ban đầu: {sum(r['input_format_ok'] for r in quality)}/{len(quality)}. "
             "Pipeline chuyển mono và resample khi cần; bảng kiểm tra trong notebook dùng dữ liệu trước chuẩn hóa.", "",
             "## 2. Cấu hình và thuật toán", "", "| Tham số | Giá trị |", "|---|---|"]
    for k, v in asdict(baseline["config"]).items():
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "Frame 400 mẫu, hop 160 mẫu, Hamming; frame cuối thiếu mẫu được zero-pad. "
                  "FFT được padding riêng lên 512. Mel dùng 1125 ln(1+f/700), 24 bộ lọc tam giác. "
                  "DCT-II trực chuẩn, giữ c0–c12; CMN theo từng utterance. "
                  "Khoảng cách Euclid, DTW ba bước; tối ưu tổng chi phí rồi chia độ dài path.", "",
                  "## 3. Energy, ZCR, endpoint và autocorrelation", "",
                  "Energy/RMS đo mức biên độ; ZCR đếm đổi dấu trên cửa sổ chữ nhật. "
                  "Silence thường có energy thấp; voiced thường có energy cao và ZCR thấp hơn unvoiced. "
                  "ZCR cao cũng có thể do nhiễu nên không dùng riêng làm điều kiện xác định speech.", ""])
    for label in ("khong", "hai", "ba"):
        path = split["train"][label][0]
        add_figure(f"time_{path.stem}.png", time_caption(path, config))
    lines.extend(["",
                  f"Endpoint `{baseline['config'].endpoint_method}` ước lượng nền bằng median "
                  f"log-energy ở {baseline['config'].noise_window_ms:g} ms đầu/cuối, "
                  "nội suy nền theo thời gian để theo thay đổi gain chậm. "
                  f"Làm trơn median {baseline['config'].smooth_ms:g} ms; "
                  f"ngưỡng xác nhận = nền + {baseline['config'].noise_on_db:g} dB, "
                  f"ngưỡng mở rộng = nền + {baseline['config'].noise_off_db:g} dB. "
                  f"Cần ít nhất {baseline['config'].min_speech_ms:g} ms liên tiếp trên ngưỡng xác nhận; "
                  f"nối khoảng ngắt ≤ {baseline['config'].max_gap_ms:g} ms, "
                  "chọn vùng có đỉnh tương phản nền lớn nhất. "
                  f"Giữ margin {baseline['config'].margin_ms:g} ms mỗi phía; chưa dùng ZCR tinh chỉnh. "
                  "Xem/nghe các file khong và hai đã trim để kiểm tra phụ âm năng lượng thấp. "
                  "Không thể kết luận giữ đủ phụ âm chỉ từ thời lượng.", "",
                  "| File | Trước (s) | Sau (s) |", "|---|---:|---:|"])
    for row in trim_rows:
        lines.append(f"| {row['file']} | {row['before_s']:.3f} | {row['after_s']:.3f} |")
    unchanged = sum(abs(row["before_s"] - row["after_s"]) < 1 / baseline["config"].sr for row in trim_rows)
    lines.extend(["", f"Với cấu hình này, {unchanged}/{len(trim_rows)} file giữ nguyên thời lượng sau endpoint."])
    removed = [row["before_s"] - row["after_s"] for row in trim_rows]
    fallback = [row["file"] for row in trim_rows if row["status"] in ("no_confident_speech", "too_short_for_noise_estimate")]
    lines.extend(["", f"Đã cắt nền ở {len(trim_rows) - unchanged}/{len(trim_rows)} file; "
                  f"thời lượng loại bỏ trung bình {np.mean(removed):.3f} s/file. "
                  f"Có {len(fallback)} file fallback giữ nguyên do chưa xác nhận được speech/clip quá ngắn.", "",
                  "So sánh trực tiếp với endpoint cũ (đỉnh trừ 35 dB, margin 50 ms) trong "
                  "bảng endpoint của notebook và cột `relative_35db_after_s` ở `endpoint.csv`. "
                  "Tham số endpoint mới được chọn bằng kiểm tra biên "
                  "trên 15 file train; giữ cố định trước khi chạy đánh giá 10 file test. "
                  "Điểm test không dùng để chọn ngưỡng."])
    if unchanged == len(trim_rows):
        lines.extend(["", "Endpoint hiện tại chưa loại được khoảng nền trên bộ dữ liệu này. "
                      "Vì đầu vào sau trim giống trước trim, E1 có/không endpoint chưa tạo đối chứng "
                      "về vùng speech; accuracy bằng nhau không chứng minh endpoint vô ích. "
                      "Kiểm tra trạng thái fallback, ước lượng nền và đồ thị log-energy. "
                      "Có thể thu lại ở nơi yên hơn; "
                      "chỉ chọn tham số bằng train/validation, giữ test để đánh giá cuối."])
    for label in ("khong", "hai"):
        path = split["train"][label][0]
        add_figure(f"endpoint_{path.stem}.png", endpoint_caption(path, config))
    lines.extend(["",
                  f"Autocorrelation minh họa: file {pitch['file']}, lag {pitch['lag']} mẫu, "
                  f"F0 ≈ {pitch['pitch_hz']:.1f} Hz, đỉnh chuẩn hóa {pitch['peak_correlation']:.3f}. "
                  "Ước lượng trên frame energy lớn nhất trong miền 70–400 Hz; có thể nhầm bội/ước pitch, "
                  "không dùng pitch làm đầu vào recognizer.", ""])
    add_figure("autocorrelation.png", autocorrelation_caption(pitch))
    lines.extend(["## 4. MFCC và DTW", "",
                  "Mỗi frame luôn có 13 hệ số; số frame phụ thuộc độ dài từ sau trim. "
                  "Cấu trúc màu theo thời gian của heatmap phản ánh biến thiên đường bao phổ.", ""])
    add_figure("mel_filterbank.png", mel_caption(config))
    add_figure("mfcc_comparison.png", mfcc_caption([split["train"][label][0] for label in ("khong", "ba")], config))
    lines.extend(["",
                  "| Cặp so sánh | Frames X/Y | Path length | DTW_norm |", "|---|---:|---:|---:|"])
    for p in pairs:
        lines.append(f"| {p['pair']} | {p['frames_x']}/{p['frames_y']} | {p['path_length']} | {p['dtw_norm']:.4f} |")
    pair_scores = {p["pair"]: p["dtw_norm"] for p in pairs}
    if pair_scores.get("same_word", -np.inf) >= pair_scores.get("different_word", np.inf):
        lines.extend(["", "Ở cặp minh họa này, score cùng từ lớn hơn hoặc bằng score khác từ. "
                      "Do đó không thể dùng riêng hai cặp này để kết luận cùng từ luôn gần hơn. "
                      "Nên kiểm tra biến thiên phát âm, vùng nền và biên speech; "
                      "không tự thay cặp minh họa hoặc chọn lại split để làm đẹp kết quả."])
    for pair in pairs:
        add_figure(f"dtw_{pair['pair']}.png", dtw_caption(pair["file_x"], pair["file_y"]))
    lines.extend(["",
                  "Hai chuỗi có độ dài khác nhau nên đường đi có thể lệch đường chéo hình học. "
                  "Bước ngang/dọc biểu diễn kéo giãn theo thời gian; không bảo đảm mọi cặp cùng từ "
                  "đều có chi phí nhỏ hơn mọi cặp khác từ.", "", "## 5. Nhận dạng và thí nghiệm E1–E2", "",
                  "Mỗi nhãn lấy khoảng cách nhỏ nhất trong 3 template; chọn nhãn có score thấp nhất. "
                  "Top-3 là ba nhãn, không phải ba template. Không áp dụng reject unknown.", "",
                  "| Thí nghiệm | Endpoint | Số chiều | Đúng/test | Accuracy |", "|---|---|---:|---:|---:|"])
    for row in summaries:
        lines.append(f"| {row['experiment']} | {row['endpoint']} | {row['feature_dim']} | "
                     f"{row['correct']}/{row['test_count']} | {row['accuracy_percent']:.1f}% |")
    comparisons = baseline.get("endpoint_comparison", [])
    if comparisons:
        lines.extend(["", "Đối chiếu với lần chạy endpoint cũ trên cùng WAV/split và cấu hình MFCC:", "",
                      "| Cấu hình | Endpoint cũ | Endpoint mới | Thay đổi (điểm %) |", "|---|---:|---:|---:|"])
        for row in comparisons:
            lines.append(f"| {row['experiment']} | {float(row['relative_35db_accuracy_percent']):.1f}% | "
                         f"{float(row['adaptive_accuracy_percent']):.1f}% | {float(row['change_percentage_points']):+.1f} |")
        lines.extend(["", "Trim loại vùng nền làm thay đổi MFCC, CMN và Δ theo thời gian. "
                      "Giảm thời lượng không bảo đảm mọi cấu hình nhận dạng đều tốt hơn; "
                      "giữ nguyên các kết quả tăng/giảm và không chọn lại ngưỡng bằng test. "
                      "`endpoint_reference.json` lưu số đo cũ và hash WAV; chỉ tạo bảng đối chứng "
                      "khi dữ liệu và cấu hình feature còn khớp."])
    a, e1, e2 = [r["accuracy_percent"] for r in summaries]
    lines.extend(["", f"E1 bỏ endpoint: accuracy thay đổi {e1 - a:+.1f} điểm phần trăm so với baseline. "
                  f"E2 thêm Δ: thay đổi {e2 - a:+.1f} điểm phần trăm. "
                  "Feature được tính lại cho cả template và test trong từng cấu hình; "
                  "E2 dùng đạo hàm Savitzky–Golay qua Librosa, cửa sổ tối đa 9 frame.", "",
                  "Cùng accuracy không đồng nghĩa cùng dự đoán; đối chiếu results_all.csv. "
                  "Tập test nhỏ nên chưa đủ suy rộng về độ chính xác trên người nói mới.", ""])
    add_figure("confusion_matrix.png", confusion_caption())
    if confusion:
        truth, predicted, count = confusion
        lines.append(f"Nhầm có hướng nhiều nhất: **{VI_LABELS[truth]} → {VI_LABELS[predicted]}**, {count} mẫu. "
                     "Cần nghe lại, kiểm tra biên trim và đối chiếu waveform/MFCC/path; "
                     "chưa thể khẳng định nguyên nhân chỉ từ confusion matrix.")
    else:
        lines.append("Baseline không có lỗi trên tập test này; không có cặp nhầm để phân tích từ số liệu hiện tại.")
    if errors:
        lines.extend(["", "| File lỗi | Nhãn thật | Dự đoán | Top-1 score |", "|---|---|---|---:|"])
        for r in errors:
            lines.append(f"| {r['file']} | {r['true_label']} | {r['predicted_label']} | {r['top1_score']:.4f} |")
    if error_figures:
        lines.extend(["", "Các hình dưới phân tích mẫu lỗi đầu tiên: waveform/energy/ZCR, "
                      "MFCC so với hai nhãn và DTW tới template gần nhất của từng nhãn."])
        for filename, caption in error_figures:
            add_figure(filename, caption)
    lines.extend(["", "## 6. Trả lời câu hỏi báo cáo", "",
                  "1. Waveform nhạy với pha, tốc độ nói và số mẫu. MFCC mô tả đường bao phổ; DTW căn chỉnh "
                  "chuỗi frame có độ dài khác nhau, phù hợp hơn so khớp mẫu thô.", "",
                  "2. Energy xác định vùng có mức tín hiệu đáng kể; ZCR hỗ trợ phân biệt hữu thanh/vô thanh "
                  "và xem xét phụ âm yếu ở biên. Nhiễu có thể có ZCR cao nên cần phối hợp và nghe kiểm tra.", "",
                  "3. Thang Mel có độ dốc giảm khi Hz tăng; các điểm cách đều theo Mel sẽ cách xa hơn "
                  "theo Hz ở vùng tần số cao.", "",
                  "4. Log nén dynamic range và biến quan hệ nhân thành cộng. DCT biểu diễn log-energy "
                  "Mel bằng các hệ số cepstral; hệ số thấp mô tả biến thiên phổ chậm/đường bao phổ.", "",
                  "5. Với trục ngang Y và dọc X: bước ngang giữ frame X, tiến Y; bước dọc tiến X, giữ Y; "
                  "bước chéo tiến cả hai. Các bước cho phép một frame ghép nhiều frame chuỗi kia.", "",
                  "6. Tổng cost thường tăng khi path dài. Chia path length giảm thiên lệch chiều dài; "
                  "đây là chuẩn hóa sau khi tối ưu tổng, không phải tối ưu cost trung bình.", "",
                  "7. Tốc độ/phát âm, pitch và trạng thái giọng, mức âm lượng/khoảng cách micro, "
                  "tiếng ồn/kênh thu, và vị trí biên trim đều có thể làm MFCC thay đổi.", "",
                  "8. Xem cặp nhầm ở mục 5. Kiểm tra audio và hình trước khi quy nguyên nhân; nếu không "
                  "có lỗi thì báo không có cặp nhầm trên tập hiện tại, không tự tạo kết luận.", "",
                  "9. Template ít người không bao phủ biến thiên người nói. HMM và mô hình âm học "
                  "được huấn luyện trên dữ liệu nhiều người có thể mô hình hóa biến thiên theo thời gian "
                  "và người nói tốt hơn; hiệu quả còn phụ thuộc dữ liệu huấn luyện.", "",
                  "## 7. Giới hạn và việc cần hoàn thiện", "",
                  "Chưa có thí nghiệm cross-speaker, chưa hiệu chỉnh ngưỡng reject. "
                  "Endpoint giả định 200 ms đầu/cuối chủ yếu là nền, và một file có một từ. "
                  "Tiếng người khác kéo dài hoặc nền thay đổi đột ngột vẫn có thể làm sai biên. "
                  "Chỉ chọn tham số trên train/validation; giữ test làm đánh giá cuối. "
                  "Trước khi nộp cần nghe biên trim và bổ sung mô tả thu âm.", ""])
    destination = output / "report_Lab02.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
