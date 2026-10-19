"""Generate the submission Markdown report from measured CSV/JSON artifacts."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from theory import theory_markdown


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def percent(value):
    return f"{float(value):.0%}" if value not in (None, "", "None") else "—"


def build_report(output):
    from lab3 import VI
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output / "training_diagnostics.json").read_text(encoding="utf-8"))
    reject = json.loads((output / "reject_proposal.json").read_text(encoding="utf-8"))
    lexicon = json.loads((output / "lexicon_status.json").read_text(encoding="utf-8"))
    experiments = read_csv(output / "experiments.csv")
    comparisons = read_csv(output / "dtw_vs_hmm.csv")
    cases = read_csv(output / "comparison_cases.csv")
    noise_cases = read_csv(output / "comparison_cases_noise.csv")
    likelihoods = read_csv(output / "likelihood_examples.csv")
    lm_rows = read_csv(output / "lm_reranking.csv")
    bigrams = read_csv(output / "bigram_probabilities.csv")
    sizes = read_csv(output / "training_sizes.csv")
    noise_methods = read_csv(output / "noise_methods.csv")
    noise_frontend = read_csv(output / "noise_frontend_analysis.csv")
    clean_rows = read_csv(output / "hmm_test_predictions.csv")
    labels = summary["labels"]
    synthetic = summary.get("is_synthetic", False)
    cfg = summary["config"]
    lines = []

    def section(text, level=2):
        lines.extend(["#" * level + " " + text, ""])

    def body(text):
        lines.extend([text, ""])

    def table(headers, rows):
        def cell(value):
            return str(value).replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join(map(cell, headers)) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
        lines.append("")

    def image(name, caption):
        body(f"![{caption}](figures/{name})\n\n*{caption}*")

    section("LAB 3 — NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM", 1)
    body("**Sinh viên:** Lỗ Anh Việt  \n**MSSV:** 2351260695  \n**Học phần:** CSE457 — Xử lý âm thanh và tiếng nói")
    section("1. Mục tiêu và thiết kế thực nghiệm")
    body("Xây dựng hệ nhận dạng từ/lệnh tách rời bằng một Gaussian HMM cho mỗi từ; quan sát likelihood và Viterbi alignment; so sánh với MFCC + DTW; thử số state, đặc trưng động, nhiễu và language-model re-ranking. Các hình và bảng dưới đây được sinh từ lần chạy notebook trên dữ liệu đã chỉ định.")
    body("Các từ sử dụng: **" + ", ".join(VI.get(w, w) for w in labels) + "**. Baseline được định trước: 5 state, Gaussian diagonal và đặc trưng 39 chiều. Không chọn cấu hình cuối theo test accuracy; các ablation là so sánh mô tả trên cùng split.")
    if synthetic:
        body("**Nguồn dữ liệu: tiếng nói tổng hợp TTS, không phải ghi âm người thật.** Giọng: "
             + ", ".join(f"`{v}`" for v in summary.get("voices", []))
             + ". Mỗi utterance có yêu cầu TTS riêng với rate/pitch định trước. Train/test dùng cùng engine và giọng, nên thí nghiệm chưa đo khả năng khái quát sang người nói thực; kết quả sạch cao có thể phản ánh tính đồng nhất của TTS.")
    else:
        body("Nguồn dữ liệu: WAV do người dùng cung cấp; nguồn/hash nằm trong manifest. Không có thông tin đủ để khẳng định speaker-independent.")
    counts = Counter((r["split"], r["label"]) for r in read_csv(output / "dataset_audit.csv"))
    table(["Nhãn", "Từ", "Train", "Test"], [[w, VI.get(w, w), counts["train", w], counts["test", w]] for w in labels])
    body(f"Tổng cộng **{summary['train_count']} train + {summary['test_count']} test**. WAV mono, 16 kHz, PCM 16-bit; bộ tổng hợp có 0,2–0,5 giây khoảng lặng ở đầu/cuối. `dataset/manifest.json` ghi nguồn, split, setting và SHA-256. `dataset/format_audit.csv` kiểm tra silence thực tế và định dạng. Pipeline kiểm tra trùng đường dẫn, hash file và hash mẫu âm thanh đã giải mã giữa train/test; không dùng test trong fit, khởi tạo Gaussian hay calibration reject.")
    table(["Thành phần", "Cấu hình"], [
        ["Front-end", "Trim 30 dB; pre-emphasis 0,97; Hamming 25 ms; hop 10 ms; FFT 512; 26 Mel"],
        ["Đặc trưng", "13 MFCC + 13 Δ + 13 Δ²; CMVN theo từng utterance"],
        ["HMM", f"{cfg['n_states']} state/từ; diagonal covariance; tối đa {cfg['n_iter']} vòng; tol={cfg['tol']}"],
        ["Tái lập", f"Seed {cfg['seed']}; manifest SHA-256; requirements-lock.txt; environment.json"],
        ["Sản phẩm", "Lab03_2351260695.ipynb; report_Lab03.md; figures/; models/; CSV/JSON ở thư mục gốc"]])

    section("2. Đặc trưng MFCC động — phần A")
    body("Pipeline: WAV → mono/16 kHz → cắt silence → tiền nhấn → MFCC → Δ/Δ² → CMVN. Vector mỗi frame có 39 chiều; giữ độ dài T khác nhau giữa các utterance. Delta dùng hồi quy lân cận, delta-delta dùng đạo hàm bậc hai. File không có tiếng hoặc quá ngắn được báo lỗi. CMVN chỉ dùng thống kê của chính utterance đang biến đổi; không học thống kê chung từ test.")
    image("features_1.png", "Hình 1. MFCC, delta và delta-delta của từ thứ nhất, trước CMVN.")
    image("features_2.png", "Hình 2. MFCC, delta và delta-delta của từ thứ hai, trước CMVN.")
    body("MFCC thể hiện hình dạng phổ theo thời gian; Δ làm rõ xu hướng thay đổi và Δ² làm rõ thay đổi của xu hướng. Những dải phổ, onset/offset và số frame khác nhau cho thấy hai từ có tiến trình âm học khác nhau. HMM đã mô hình hóa chuỗi state nhưng vẫn giả định output độc lập khi biết state; đặc trưng động bổ sung biến thiên cục bộ. Chưa có nhãn phoneme nên không gán tên âm vị chỉ dựa vào màu heatmap.")

    section("3. Cấu hình HMM và huấn luyện — phần B, C")
    body("HMM λ = (A, B, π): A là xác suất chuyển trạng thái, B là mật độ Gaussian phát xạ, π là phân bố bắt đầu. Giả thiết Markov bậc một: state hiện tại chỉ phụ thuộc state trước; quan sát phụ thuộc state hiện tại. Đặt π = [1,0,…,0], chỉ cho self-loop và next-state; aᵢᵢ=0,7, aᵢ,ᵢ₊₁=0,3 ban đầu, state cuối tự lặp với xác suất 1.")
    body("Khởi tạo mean/covariance từ các phần thời gian liên tiếp của train utterance để emissions có thứ tự ban đầu. Mỗi lớp ghép các ma trận train thành `X_concat`, truyền `lengths` để thư viện không tạo chuyển trạng thái giữa hai bản ghi. `init_params=''` giữ tham số đã khởi tạo, `params='tmc'` cập nhật transition/mean/covariance và giữ π. Sau fit, kiểm tra tổng hàng A bằng 1 và mọi transition ngược/skip bằng 0.")
    body("Với D=39, diagonal covariance cần 39 phương sai/state; full covariance cần 39×40/2=780 tham số độc lập/state. Diagonal giảm số tham số, thích hợp với dữ liệu ít. Full covariance chỉ được thử khi mỗi lớp đạt ít nhất 2×D×n_states frame; đây là tiêu chí thận trọng của triển khai, không phải điều kiện đảm bảo đủ dữ liệu.")
    image("transitions.png", "Hình 3. Transition matrix sau huấn luyện của năm model.")
    table(["Model", "Train frames", "Vòng EM", "Dừng vì", "Gain cuối"], [
        [d["label"], d["frames"], d["iterations"], d["stop_reason"], f"{d['last_gain']:.5g}" if d["last_gain"] is not None else "—"] for d in diagnostics])
    image("convergence.png", "Hình 4. Train log-likelihood theo vòng Baum–Welch.")
    body("Baum–Welch là EM: E-step tính kỳ vọng state/transition bằng Forward–Backward, M-step cập nhật tham số theo trọng số trách nhiệm. EM chỉ đảm bảo nghiệm cục bộ, phụ thuộc khởi tạo, số state, covariance, lượng dữ liệu và môi trường. Cờ `monitor.converged=True` có thể xuất hiện khi hết số vòng; vì vậy báo cáo riêng `tolerance`, `max_iter`, `likelihood_decreased` và gain cuối trong `training_diagnostics.json`.")

    section("4. Likelihood và Viterbi — phần D")
    body("Forward cộng xác suất của mọi đường hợp lệ để tính P(X|λ), còn Viterbi thay tổng bằng max để tìm đường trạng thái tốt nhất. Mã thực hiện trong miền log bằng logsumexp. Forward–Backward minh họa được đối chiếu với `model.score`; log-likelihood Forward ≥ log-probability Viterbi. γₜ(i) là posterior trách nhiệm state i cho frame t; ξₜ(i,j) là posterior trách nhiệm chuyển i→j tại t. Thư viện thực hiện huấn luyện EM đầy đủ.")
    table(["Test utterance", "Dự đoán"] + labels + ["Margin/frame"], [
        [Path(r["file"]).name, r["prediction"]] + [f"{float(r[f'score_{w}']):.1f}" for w in labels]
        + [f"{float(r['margin_per_frame']):.2f}"] for r in likelihoods])
    body("Bảng gồm ba test utterance khác lớp, chấm bằng toàn bộ năm model. Chọn log-likelihood lớn nhất; score âm hơn nghĩa là thấp hơn. Chia các score cho cùng T không đổi argmax trên một utterance. Score/frame hữu ích để quan sát phân bố giữa các file khác độ dài và thiết kế reject.")
    image("viterbi_1.png", "Hình 5. Viterbi path và posterior gamma của utterance thứ nhất.")
    image("viterbi_2.png", "Hình 6. Viterbi path và posterior gamma của utterance thứ hai.")
    durations = read_csv(output / "state_durations.csv")
    table(["Utterance", "State", "Frame", "Occupancy xấp xỉ (ms)"], [
        [Path(r["file"]).name, r["state"], r["frames"], r["occupancy_ms_approx"]] for r in durations])
    body("Path không giảm và chỉ tăng tối đa một state/frame. Self-loop kéo dài occupancy khi nói chậm; số frame × 10 ms là ước lượng thời lượng, không phải biên phoneme chính xác do các cửa sổ chồng lấn. Model không bắt buộc mọi utterance kết thúc ở state cuối: likelihood kết thúc bằng tổng qua mọi state, đúng công thức đề. DTW warping path ánh xạ frame test sang frame template; Viterbi ánh xạ frame vào state của model.")

    section("5. Đánh giá sạch và DTW baseline — phần E")
    body(f"HMM train accuracy **{percent(summary['train_accuracy'])}**, test accuracy **{percent(summary['test_accuracy'])}** trên {summary['test_count']} file. Mỗi lỗi tương ứng {100 / summary['test_count']:.0f} điểm phần trăm, nên tập test còn nhỏ. Accuracy sạch cao trên cùng TTS voice chưa chứng minh nhận dạng tốt tiếng nói thực.")
    image("confusion_hmm.png", "Hình 7. Confusion matrix HMM trên test sạch.")
    image("confusion_dtw_lab2.png", "Hình 8. Confusion matrix DTW dùng front-end Lab 2 trên cùng test.")
    table(["Phương pháp", "Test accuracy", "Fit model (s)", "Decode (ms/file)"], [
        [r["method"], percent(r["test_accuracy"]), f"{float(r['model_fit_seconds']):.6f}", f"{float(r['decode_seconds_per_trial']) * 1000:.3f}"] for r in comparisons])
    body("DTW Lab 2 giữ endpoint adaptive, 24 Mel, 13 MFCC và CMN cũ; DTW_shared dùng cùng front-end 39-D/CMVN như HMM. Mỗi test được so với mọi train template rồi lấy min distance, chuẩn hóa tổng chi phí theo độ dài path. Recurrence và quy tắc tie được đối chiếu với mã Lab 2. Decode timing loại trừ front-end và JIT lần đầu; thời gian fit DTW chỉ gồm tạo danh sách template. Các thời gian là một lần đo trên máy hiện tại.")
    different = [r for r in cases if r["hmm_correct"] != r["dtw_correct"]]
    if different:
        r = different[0]
        body(f"Trường hợp khác biệt sạch: `{r['file']}`, nhãn `{r['truth']}`, HMM chọn `{r['hmm']}`, DTW Lab 2 chọn `{r['dtw_lab2']}`. Cần nghe lại và xem alignment trước khi khẳng định nguyên nhân lỗi.")
    else:
        body("Trên test sạch, chưa có mẫu khác nhau về đúng/sai giữa HMM và DTW Lab 2. Không thể kết luận ưu thế từ accuracy đang bão hòa; phần mismatch bên dưới cung cấp trường hợp khác biệt trên cùng front-end.")

    section("6. Thí nghiệm tham số và độ bền — E1–E5, phần H")
    table(["Thí nghiệm", "State", "Chiều", "Train", "Test", "Model hết vòng", "Trạng thái"], [
        [r["experiment"], r["states"], r["dimensions"], percent(r["train_accuracy"]), percent(r["test_accuracy"]), r["max_iter_models"] or "—", r["status"]] for r in experiments])
    baseline = next(r for r in experiments if r["experiment"] == "baseline_states5_39D")
    static = next(r for r in experiments if r["experiment"] == "E2_mfcc13")
    delta_gain = 100 * (float(baseline["test_accuracy"]) - float(static["test_accuracy"]))
    body(f"39-D so với 13-D thay đổi test accuracy **{delta_gain:+.0f} điểm phần trăm**. Khi cả hai cùng đạt trần trên dữ liệu TTS nhỏ, chưa có bằng chứng định lượng rằng delta giúp phân biệt một lớp cụ thể. Với số state cũng cần xem diagnostics: cùng accuracy không có nghĩa cùng chất lượng hội tụ. Không gán underfit/overfit chỉ dựa vào số state hoặc một accuracy sạch.")
    table(["Train/lớp", "HMM 39-D", "DTW cùng 39-D"], [
        [r["train_per_class"], percent(r["hmm_accuracy"]), percent(r["dtw_shared_accuracy"])] for r in sizes])
    body("Các tập train nhỏ lấy prefix cố định từ train ban đầu, giữ nguyên test. Đây là kiểm tra nhạy với lượng mẫu trên một nguồn TTS đồng nhất; chưa đại diện quan hệ dữ liệu–accuracy của bản ghi nhiều người nói.")
    table(["Phương pháp", "Sạch", f"Nhiễu {summary['snr_db']:g} dB"], [
        [r["method"], percent(r["clean_accuracy"]), percent(r["noise_accuracy"])] for r in noise_methods])
    body(f"Nhiễu Gaussian được thêm trước trích đặc trưng, SNR {summary['snr_db']:g} dB tính theo công suất vùng tiếng nói sau trim, seed cố định. HMM giảm từ **{percent(summary['test_accuracy'])}** xuống **{percent(summary['noise_accuracy'])}**, giảm **{summary['noise_drop_percentage_points']:.0f} điểm phần trăm**. Không dùng noisy test trong fit hoặc chọn baseline.")
    image("confusion_noise.png", "Hình 9. Confusion matrix HMM khi thêm nhiễu.")
    median_clean = float(np.median([float(r["clean_trim_seconds"]) for r in noise_frontend]))
    median_noisy = float(np.median([float(r["noisy_trim_seconds"]) for r in noise_frontend]))
    one_state = sum(len(r["states_visited"].split(",")) == 1 for r in noise_frontend)
    predictions = Counter(r["prediction"] for r in noise_frontend)
    body(f"Phân tích front-end: trung vị đoạn được giữ sau trim là **{median_clean:.2f}s sạch** và **{median_noisy:.2f}s nhiễu**; noise làm phép trim tương đối giữ thêm vùng nền. Có **{one_state}/{len(noise_frontend)}** noisy utterance chỉ đi qua một state của model được chọn. Phân bố dự đoán nhiễu: {dict(predictions)}. Các dấu hiệu này cho thấy mismatch ở cả endpoint/đặc trưng và mô hình duration; chưa cô lập được riêng nguyên nhân nào quyết định toàn bộ lỗi.")
    different_noise = [r for r in noise_cases if r["hmm_correct"] != r["dtw_correct"]]
    if different_noise:
        table(["Mẫu nhiễu", "Nhãn thật", "HMM 39-D", "DTW cùng 39-D"], [
            [Path(r["file"]).name, r["truth"], r["hmm"], r["dtw_shared"]] for r in different_noise[:3]])
        body("Đây là các trường hợp hai phương pháp khác nhau về đúng/sai với cùng front-end, cùng noise realization và split. DTW không chịu phân bố Gaussian của state đã học nên có thể phản ứng khác với mismatch; quan sát này không chứng minh DTW luôn bền nhiễu hơn HMM.")
    image("score_distributions.png", "Hình 10. Phân bố score/frame của đúng lớp và sai lớp tốt nhất trên test sạch.")
    body(f"Argmax vẫn chọn một lớp khi gặp unknown. Đề xuất reject nếu best score/frame thấp hoặc margin top-1/top-2 nhỏ. Holdout lấy một file train/lớp, còn lại dùng fit; có {reject['correct_calibration_count']} mẫu calibration đúng. Ngưỡng gợi ý score={reject['score_threshold']}, margin={reject['margin_threshold']}. Chưa áp dụng ngưỡng vào baseline: calibration nhỏ, chưa có unknown thật, final model được fit lại có thể đổi score distribution. Một mẫu sai vẫn có margin lớn, nên margin đơn lẻ không đảm bảo độ tin cậy.")

    section("7. Lexicon, bigram và MAP re-ranking — phần F, G, E6")
    body("Whole-word HMM ánh xạ trực tiếp từ → model nên không cần pronunciation lexicon để nhận dạng. Khi chuyển sang phone/subword HMM, lexicon chỉ định chuỗi phone model cần nối để tạo word HMM. Parser `parse_lexicon` đọc `WORD PHONE_1 ... PHONE_K`, bỏ chú thích và báo lỗi từ trùng/thiếu phone; đã có kiểm thử bằng inventory giả.")
    body("**Lexicon chưa có phiên âm chuẩn** vì chưa nhận phone inventory của giảng viên. Các từ còn thiếu: " + ", ".join(lexicon["missing_words"]) + ". Không coi ký hiệu tự đặt là phiên âm tiếng Việt được xác nhận.")
    body("`corpus.txt` là corpus minh họa tự tạo cho năm lệnh, không lấy nhãn test để đếm. Đếm unigram và bigram; add-k=0,5: P(w|prev) = [C(prev,w)+k]/[C_out(prev)+kV]. C_out chỉ đếm context có từ kế tiếp trong vocabulary; mỗi dòng có context bắt đầu `<s>`, không gộp sang câu khác. Tổng xác suất mọi ứng viên bằng 1.")
    table(["Previous", "Current", "Bigram count", "Context count", "P(current\u007cprevious)"], [
        [r["previous"], r["word"], r["count"], r["context_count"], f"{float(r['probability']):.4f}"]
        for r in bigrams if int(r["count"]) > 0][:8])
    body("Re-ranking: Score(w)=log P(X|w)/T + λ log P(w|prev), thử λ=0; 0,1; 0,3; 1. Chọn ba utterance có acoustic margin nhỏ nhất và liệt kê mọi previous-word context, không suy context từ nhãn đúng. Context là giả định, nên không gộp thành accuracy nhận dạng chuỗi có context thật.")
    changed = [r for r in lm_rows if r["changed"] == "True"]
    minimum_margin = min(float(r["margin_per_frame"]) for r in clean_rows)
    body(f"Có **{len(changed)}/{len(lm_rows)}** tình huống giả định đổi acoustic argmax trong dải λ đang thử. Acoustic margin nhỏ nhất trên test là **{minimum_margin:.2f}/frame**; “nhỏ nhất” không đồng nghĩa hai ứng viên thực sự gần nhau.")
    example = changed[0] if changed else lm_rows[0]
    related = [r for r in lm_rows if r["file"] == example["file"] and r["previous"] == example["previous"]]
    table(["File", "Prev giả định", "λ", "Acoustic", "Kết hợp"], [
        [Path(r["file"]).name, r["previous"], r["lm_weight"], r["acoustic_prediction"], r["prediction"]] for r in related])
    if not changed:
        body("Trong thí nghiệm này LM chưa làm đổi quyết định, nên không quan sát được ví dụ sửa lỗi hoặc làm sai trong dải λ≤1. Về nguyên lý, λ lớn có thể lấn át âm học nếu prior ưu tiên một ứng viên khác. Cần dữ liệu âm học khó hơn và context thật để đánh giá lợi ích; không tạo ví dụ giả để tuyên bố LM cải thiện.")
    else:
        body("Khi quyết định đổi, cần đối chiếu context và nhãn để phân biệt prior sửa lỗi hay gây lỗi. λ lớn làm tăng tác động prior; corpus và context không phù hợp có thể lấn át bằng chứng âm học.")

    section("8. Trả lời câu hỏi lý thuyết")
    body(theory_markdown())

    section("9. Kết luận và giới hạn")
    body(f"Pipeline huấn luyện và giải mã chạy lại được, có kiểm tra topology, likelihood, Viterbi và DTW baseline. Trên bộ hiện tại HMM sạch đạt {percent(summary['test_accuracy'])}, nhiễu {summary['snr_db']:g} dB đạt {percent(summary['noise_accuracy'])}. Các ablation cần được đọc cùng diagnostics và giới hạn dữ liệu; kết quả sạch không thay thế đánh giá robustness.")
    body("Dữ liệu tổng hợp có cùng nguồn giọng ở hai split, test chỉ 10 file, chưa có nhiều người nói/unknown/context chuỗi thật. Cần thêm ghi âm thực và inventory/corpus giảng viên. Hướng cải thiện có thể kiểm chứng tiếp: endpoint dựa nền nhiễu, silence model hoặc train đa điều kiện, chọn/hiệu chỉnh trên validation trước khi đánh giá test mới.")
    section("10. Tài liệu và file đối chiếu")
    body("- Đề `Lab 3.pdf`, mục 4–7, case study và câu hỏi phân tích.\n"
         "- [hmmlearn 0.3.3 API](https://hmmlearn.readthedocs.io/en/stable/api.html).\n"
         "- [librosa feature.delta](https://librosa.org/doc/latest/generated/librosa.feature.delta.html).\n"
         "- [edge-tts](https://github.com/rany2/edge-tts): công cụ tạo WAV tổng hợp qua TTS.\n"
         "- Rabiner & Schafer, *Theory and Applications of Digital Speech Processing*.\n"
         "- Mã MFCC/DTW Lab 2 của cùng sinh viên: `lab2_baseline.py`.")
    body("Bảng nguồn: `dataset_audit.csv`, `split.json`, `experiments.csv`, `training_diagnostics.json`, `likelihood_examples.csv`, `dtw_vs_hmm.csv`, `noise_methods.csv`, `noise_frontend_analysis.csv`, `lm_reranking.csv`. Notebook chính: `Lab03_2351260695.ipynb`; model: `models/hmm_baseline.joblib`.")
    (output / "report_Lab03.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build_report(Path(__file__).resolve().parent)
