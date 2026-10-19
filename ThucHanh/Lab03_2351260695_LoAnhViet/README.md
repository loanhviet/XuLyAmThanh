# Lab 3 — Gaussian HMM nhận dạng 5 lệnh tiếng Việt

**Lỗ Anh Việt — MSSV 2351260695 — CSE457**

Notebook chính dùng 40 WAV tổng hợp cho các lệnh **mở, đóng, lên, xuống, dừng**.
Đầu ra được đặt giống Lab 1/Lab 2: báo cáo Markdown, bảng CSV/JSON ở thư mục gốc;
hình trong `figures/`, dữ liệu trong `dataset/`, model trong `models/`.

## Chạy ngay

Mở `Lab03_2351260695.ipynb`, chọn Python của `.venv` trong thư mục Lab 3, rồi
**Restart Kernel → Run All**. Báo cáo `report_Lab03.md` được sinh lại và hiển thị
ở cuối notebook. Bài hiện tại **chỉ xuất Markdown, không xuất PDF**.

Chạy bằng terminal:

```bash
source .venv/bin/activate
python lab3.py run
```

Hoặc mở JupyterLab:

```bash
.venv/bin/python -m jupyterlab Lab03_2351260695.ipynb
```

Nhận dạng WAV khác:

```bash
.venv/bin/python lab3.py predict dataset/test/mo/mo_namminh_07.wav
```

Tạo lại môi trường (Python 3.12 khuyến nghị):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

Không có `uv`: dùng `python3 -m venv .venv` và `.venv/bin/python -m pip install
-r requirements-lock.txt`. Chạy pipeline không cần mạng, GPU hay tải model ASR.

## Cấu trúc bài nộp

```text
Lab03_2351260695_LoAnhViet/
├── Lab03_2351260695.ipynb     # Notebook chính đã chạy toàn bộ
├── report_Lab03.md           # Báo cáo chính để nộp
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── lab3.py                  # HMM, đặc trưng, giải mã và thí nghiệm
├── lab2_baseline.py          # MFCC/DTW Lab 2 để đối chiếu
├── report.py                # Sinh báo cáo Markdown từ số đo
├── theory.py                # Trả lời 10 câu hỏi dùng chung cho report/notebook
├── dataset/
│   ├── train/{mo,dong,len,xuong,dung}/*.wav
│   ├── test/{mo,dong,len,xuong,dung}/*.wav
│   ├── manifest.json
│   ├── generation_config.json
│   ├── format_audit.csv
│   └── tts_sources/          # MP3 nguồn để kiểm tra provenance
├── figures/                 # Heatmap, Viterbi, confusion, convergence
├── models/hmm_baseline.joblib
├── corpus.txt
├── lexicon.txt
├── experiments.csv          # Các bảng CSV/JSON nằm ở gốc như Lab 2
├── hmm_test_predictions.csv
├── dtw_vs_hmm.csv
├── noise_methods.csv
├── ...
├── scripts/
└── tests/
```

`dataset_backups/` chỉ giữ dữ liệu và các lần chạy cũ để đối chiếu ở máy hiện tại,
được bỏ qua bằng `.gitignore` và **không cần nộp**. `.venv/`, cache, file ZIP dữ
liệu và tests là tiện ích; không cần gửi môi trường ảo/cache khi nộp.

## Commit lên Git để nộp

`.gitignore` đã lọc các file trung gian. Commit các phần sau trong thư mục Lab 3:

- `Lab03_2351260695.ipynb`, **`report_Lab03.md`**, `README.md`, `.gitignore`.
- `lab3.py`, `lab2_baseline.py`, `report.py`, `theory.py`, `requirements*.txt`,
  `scripts/` và `tests/` để chạy lại và kiểm chứng bài.
- `dataset/train/`, `dataset/test/`: 40 WAV; kèm README, manifest, generation
  config và format audit trong `dataset/`.
- `figures/`, `models/`, `corpus.txt`, `lexicon.txt`.
- Bảy bảng chính: `hmm_test_predictions.csv`, `likelihood_examples.csv`,
  `experiments.csv`, `dtw_vs_hmm.csv`, `training_sizes.csv`, `noise_methods.csv`,
  `lm_reranking.csv`.
- Bốn JSON: `split.json`, `summary.json`, `training_diagnostics.json`, `environment.json`.

Các CSV/JSON còn lại, posterior NPZ, MP3 TTS nguồn, ZIP dữ liệu, backup, môi trường
ảo và cache được giữ ở máy và bỏ qua khi `git add`. **Run All** sẽ tạo lại các bảng
trung gian cần cho phần phân tích và sinh lại báo cáo. MP3 cache không cần để chạy
notebook vì WAV đã được commit; tổng hợp lại từ TTS cần mạng/gói tùy chọn.

Chạy ngay **trong thư mục Lab 3**:

```bash
git add .
git diff --cached --stat
git commit -m "Hoan thanh Lab 3 - nhan dang 5 lenh bang HMM"
```

Lệnh `git add .` ở thư mục Lab 3 chỉ thêm nội dung bên dưới thư mục này; xem bảng
thống kê trước khi commit để kiểm tra các thay đổi khác đã stage từ trước nếu có.

## Đối chiếu nội dung thực hành

| Yêu cầu | File/hình tương ứng |
|---|---|
| A — MFCC/Δ/Δ², CMVN, hai heatmap | `lab3.py`, `feature_shapes.csv`, `figures/features_1.png`, `features_2.png` |
| B — HMM trái-sang-phải, kiểm tra topology | `training_diagnostics.json`, `figures/transitions.png` |
| C — Huấn luyện, lengths, convergence, số state | `models/`, `experiments.csv`, `figures/convergence.png` |
| D — Likelihood của 3 test, hai Viterbi path | `likelihood_examples.csv`, `state_durations.csv`, `figures/viterbi_*.png` |
| E — DTW cùng split, accuracy/time, phân tích lỗi | `dtw_vs_hmm.csv`, `comparison_cases.csv`, `comparison_cases_noise.csv` |
| F — Lexicon và parser | `lexicon.txt`, `parse_lexicon()`; phiên âm chuẩn còn chờ inventory giảng viên |
| G — Unigram/bigram và 4 trọng số LM | `corpus.txt`, `unigram_counts.csv`, `bigram_probabilities.csv`, `lm_reranking.csv` |
| H — Mismatch, score distribution, reject đề xuất | `noise_methods.csv`, `noise_frontend_analysis.csv`, `score_distributions.csv`, `reject_proposal.json` |
| Báo cáo, câu hỏi lý thuyết và kết luận | **`report_Lab03.md`**; cùng nội dung hiển thị cuối notebook |

## Dữ liệu đang sử dụng

- **40 WAV tổng hợp TTS**, một giọng `vi-VN-NamMinhNeural`, không phải ghi âm người thật.
- Mỗi lệnh **6 train + 2 test**: tổng 30 train + 10 test.
- WAV mono, 16 kHz, PCM 16-bit; silence đầu/cuối 0,2–0,5 giây.
- Mỗi utterance có yêu cầu TTS riêng với tốc độ/cao độ định trước. Không lấy các
  bản biến đổi hậu kỳ của cùng audio nguồn để chia train/test.
- Nguồn, setting, split và hash được lưu trong `dataset/manifest.json`; định dạng
  và silence đo trực tiếp ở `dataset/format_audit.csv`. Pipeline kiểm tra trùng
  đường dẫn, hash file và hash decoded audio giữa hai split.
- Cùng engine và giọng ở hai split; accuracy trên TTS chưa chứng minh khả năng
  nhận dạng người nói thật. Báo cáo ghi rõ giới hạn này.

Các WAV đã có sẵn, **không cần tạo lại** để chạy notebook. Muốn tái tạo TTS cần
mạng, ffmpeg và gói tùy chọn:

```bash
uv pip install --python .venv/bin/python -r requirements-tts.txt
.venv/bin/python scripts/generate_synthetic_dataset.py
```

Script dùng [edge-tts](https://github.com/rany2/edge-tts), tái dùng MP3 cache khi
có. Dịch vụ có thể thay đổi đầu ra nếu tổng hợp lại từ đầu. `corpus.txt` là ví dụ
tự tạo; `lexicon.txt` chưa có phiên âm vì cần phone inventory giảng viên.

Nếu thu người thật thay TTS, nên dùng thư mục riêng rồi chạy bằng `--dataset`:

```bash
.venv/bin/python scripts/record_dataset.py --vocabulary commands --output dataset_recorded --train-count 6 --test-count 2
.venv/bin/python lab3.py run --dataset dataset_recorded --output results_recorded --models models_recorded
```

Script thu dùng `arecord` trên Linux và chỉ mở microphone sau khi người dùng chạy,
nhấn Enter. Mặc định mỗi file 3 giây, im khoảng 0,3 giây đầu/cuối, nói một lệnh.

## Thí nghiệm và cách đọc kết quả

Baseline chốt trước test: **5 state, diagonal, 39-D**. Cửa sổ Hamming 25 ms,
hop 10 ms, FFT 512, 26 Mel, tiền nhấn 0,97, trim 30 dB, CMVN theo utterance.
Gaussian khởi tạo bằng các phần thời gian của train. EM cập nhật A/mean/covariance,
giữ π=[1,0,…,0]. Chỉ cho self-loop/next-state, kiểm tra sau fit.

- E1: state 3/5/7; E2: 13/39-D. Giữ split/seed còn lại, báo cáo mô tả,
  **không chọn tham số cuối theo test**. Kiểm tra gain/lý do dừng riêng với cờ hội tụ.
- E3: full covariance chỉ thử nếu mỗi lớp đủ ít nhất `2 × D × n_states` frame;
  thiếu thì ghi skipped. Đây là tiêu chí triển khai, không đảm bảo đủ dữ liệu.
- E4: DTW Lab 2 (front-end cũ) và DTW_shared (cùng 39-D với HMM), cùng train/test.
  Recurrence, tie rule và min-template decision được đối chiếu mã gốc.
- Số mẫu train: 1, 2, toàn bộ mỗi lớp; cùng test, không dùng chọn baseline.
- E5: nhiễu Gaussian 20 dB, cùng noisy waveform cho các phương pháp, seed cố định.
  Bảng endpoint/state path bổ sung giải thích vì sao accuracy sạch có thể cao
  nhưng noisy test giảm mạnh. Giữ nguyên pipeline khi đo, không sửa theo test.
- E6: add-k=0,5, λ=0/0,1/0,3/1; context được liệt kê giả định trên ba file có margin
  nhỏ nhất. Nếu LM không đổi quyết định thì báo đúng quan sát; không báo accuracy
  chuỗi có context thật khi chưa có dữ liệu như vậy.
- Reject: gợi ý từ holdout train một file/lớp; chưa kiểm chứng unknown thật và
  chưa áp dụng vào baseline. Margin lớn vẫn có thể đi cùng dự đoán sai.

`decode_seconds_per_trial` loại trừ front-end và JIT lần đầu. Fit DTW chỉ gồm tạo
danh sách template; front-end timing báo riêng. Thời gian là một lần đo trên máy
hiện tại. Toàn bộ báo cáo lấy số đo CSV/JSON từ lần chạy vừa hoàn tất.

## Kiểm chứng

```bash
.venv/bin/python -m pytest -q
```

Kiểm tra Forward–Backward bằng liệt kê mọi path trên HMM nhỏ; gamma/xi chuẩn hóa;
topology sau EM; save/load giữ score; trùng audio khác header; delta file ngắn;
DTW khớp Lab 2; bigram tổng xác suất 1 và λ=0 giữ acoustic argmax. Notebook chính
được thực thi từ kernel sạch bằng `nbclient`.

## Tài liệu

- Đề `Lab 3.pdf` do người dùng cung cấp.
- [hmmlearn API](https://hmmlearn.readthedocs.io/en/stable/api.html).
- [librosa feature.delta](https://librosa.org/doc/latest/generated/librosa.feature.delta.html).
- Mã MFCC/DTW Lab 2 của cùng sinh viên: `lab2_baseline.py`.
