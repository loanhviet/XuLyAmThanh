# Lab 2 — MFCC + DTW nhận dạng 5 từ tiếng Việt

Lỗ Anh Việt — MSSV 2351260695. Triển khai theo Lab 2 CSE457, gồm A–G và E1–E2.

## Chạy ngay

Mở VS Code ở thư mục `Lab02_2351260695_LoAnhViet`, mở `Lab2_2351260695.ipynb`, chọn kernel
Python của môi trường `.venv` tại gốc `XuLyAmThanh`, rồi **Restart Kernel → Run All**.

Notebook chính đã được cấu hình dùng **25 bản ghi điện thoại từ `data.zip`** trong `dataset/`.
Các file được chuyển từ M4A ALAC mono 48 kHz sang WAV PCM-16 mono 16 kHz;
giữ khoảng im lặng để làm endpoint. Kết quả ghi âm thật nằm ngay trong thư mục Lab 2.

## Đối chiếu sản phẩm nộp với đề

Theo mục 7, trang 14 của `Lab 2.pdf`:

| Sản phẩm bắt buộc | File/thư mục trong bài |
|---|---|
| Notebook chạy từ đầu đến cuối | `Lab2_2351260695.ipynb` |
| Tập WAV đã dùng | `dataset/`: 25 WAV, 5 từ × 5 lần |
| Waveform, energy/ZCR, MFCC, DTW path, confusion matrix | `figures/`: 15 PNG, kèm số hình và chú thích trong notebook/báo cáo |
| Tên file test, nhãn thật/dự đoán, top-1/top-2 DTW score | `results.csv`: 10 file test, có thêm top-3 |
| Báo cáo PDF hoặc Markdown cuối notebook | Báo cáo Markdown hiển thị cuối notebook; bản riêng `report_Lab02.md` có bảng tham số, E1–E2, 9 câu trả lời và kết luận |

Phần C còn yêu cầu xuất WAV sau endpoint: `audio_trimmed/` chứa 25 file.
Đầu ra bắt buộc ở mục 1 gồm demo nhận WAV chưa biết, trả nhãn và khoảng cách DTW:
ô **Nhận dạng một WAV chưa biết** và CLI `lab2.py predict` dùng `templates_baseline.npz`.

Commit kèm các module Python, script và `requirements.txt` để chạy lại notebook.
Chỉ `results.csv` được đề yêu cầu thành một CSV riêng; `results_all.csv`,
`experiments.csv`, `endpoint.csv` là bảng bổ trợ cho E1–E2 và kiểm tra trim.
Các JSON lưu cấu hình, split, nguồn dữ liệu và đối chứng endpoint.
Không cần báo cáo PDF riêng khi đã dùng lựa chọn Markdown cuối notebook.

## Bộ ghi âm đã nhập

- `dataset/`: 25 WAV đã chuyển đổi, được đưa vào Git để chạy lại bài lab.
- `dataset/manifest.json`: ánh xạ tên, metadata và SHA-256 nguồn.
- `phone_audio/` và `dataset_backups/`: bản M4A gốc và bản sao ghi thử, chỉ giữ ở máy
  đã nhập dữ liệu và được bỏ qua bằng `.gitignore`.

Ánh xạ: `khong.m4a` → `khong_01.wav`, `khong1.m4a` → `khong_02.wav`, …,
`khong4.m4a` → `khong_05.wav`; tương tự với `mot`, `hai`, `ba`, `bon`.
Vẫn lấy 3 file đầu mỗi lớp làm template, 2 file cuối làm test.

Nếu còn bản nguồn ở máy, có thể nhập lại bằng lệnh dưới từ thư mục Lab 2.
Thay đường dẫn bằng thư mục chứa các bản ghi có tên chuẩn. Mặc định bỏ qua WAV đã có:

```bash
python scripts/import_phone_audio.py /duong/dan/ban_ghi_dien_thoai
```

Nếu cần môi trường riêng, chạy từ thư mục Lab 2:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Nếu dùng môi trường sẵn có của dự án, thay `python` trong các lệnh dưới bằng
`../../.venv/bin/python` khi chưa kích hoạt môi trường.

## Thu âm và chạy với dữ liệu thật

```bash
python scripts/record_dataset.py
```

Script dùng `arecord` trên Linux. Enter để bắt đầu mỗi file, chờ khoảng 0,3 giây rồi
nói đúng một từ, giữ im lặng đến khi hết 3 giây. Giữ micro/mức âm lượng ổn định.
Chỉ mở micro khi bạn tự chạy script. File đã có được bỏ qua; muốn ghi lại thì xóa file đó trước.
Nếu thiết bị mặc định lỗi, thử `--device pulse` hoặc chọn thiết bị bằng `arecord -L`.
Có thể ghi từng lớp bằng `--label khong`.

Hoặc dùng công cụ ghi âm khác và đặt WAV theo cấu trúc:

```text
dataset/
  khong/khong_01.wav ... khong_05.wav
  mot/mot_01.wav     ... mot_05.wav
  hai/hai_01.wav     ... hai_05.wav
  ba/ba_01.wav       ... ba_05.wav
  bon/bon_01.wav     ... bon_05.wav
```

Sau khi có đủ 5 file/lớp, chạy lại notebook. Nó tự chọn `dataset/`, lấy 3 file đầu
mỗi lớp làm train và phần còn lại làm test. Khi bộ thật còn thiếu, notebook báo lỗi
để hoàn thiện dữ liệu.

Pipeline chuyển mono/resample khi đọc; kiểm tra clipping trên tín hiệu gốc, chuẩn hóa peak
rồi dùng cùng cấu hình cho train và test. Nên thu/chuẩn bị WAV mono PCM 16 kHz ngay từ đầu.

## Ghi bằng điện thoại

Mở ứng dụng Ghi âm có sẵn (Voice Memos/Ghi âm trên iPhone hoặc ứng dụng ghi âm của Android).
Đặt điện thoại cách miệng khoảng 10–15 cm ở nơi yên tĩnh. Mỗi file: bắt đầu ghi,
chờ 0,3–0,5 giây, nói một từ, chờ 0,3–0,5 giây rồi dừng. Ghi mỗi từ 5 lần.
Nghe thử trước để kiểm tra âm lượng và tiếng người khác chen vào.

Đổi tên thành `khong_01` … `khong_05`, `mot_01` … `mot_05`, `hai_01` … `hai_05`,
`ba_01` … `ba_05`, `bon_01` … `bon_05`. Giữ nguyên phần mở rộng do điện thoại xuất ra:
M4A/MP3/WAV đều được; đổi phần mở rộng không chuyển được định dạng audio.

Chuyển 25 file vào thư mục `/home/viet/Downloads/Lab2_phone` trên laptop, bằng USB
hoặc tải từ nơi bạn đã lưu/chia sẻ bản ghi. Nếu ứng dụng không cho đổi tên, đổi trên laptop.
Từ thư mục gốc `XuLyAmThanh`, chạy:

```bash
.venv/bin/python ThucHanh/Lab02_2351260695_LoAnhViet/scripts/import_phone_audio.py /home/viet/Downloads/Lab2_phone
```

Script dùng FFmpeg để chuyển sang WAV mono PCM-16 16 kHz và chia đúng thư mục `dataset/`.
Giữ nguyên file nguồn và khoảng im lặng; file WAV đã có được bỏ qua.
Chỉ dùng `--overwrite` khi muốn thay các WAV đã có bằng bản ghi mới.
Sau khi đủ 5 file/từ, mở notebook, **Restart Kernel → Run All**.

## Đầu ra

Đầu ra nằm ngay trong thư mục Lab 2.

- `figures/`: waveform/energy/ZCR của 3 từ, endpoint, autocorrelation, Mel, MFCC, DTW path, confusion matrix.
- `audio_trimmed/`: WAV sau endpoint.
- `results.csv`: baseline, nhãn thật/dự đoán, top-3 và khoảng cách.
- `results_all.csv`: dự đoán từng file của baseline/E1/E2.
- `experiments.csv`: cấu hình và accuracy của baseline/E1/E2.
- `endpoint.csv`: biên cắt, trạng thái và thời lượng trước/sau endpoint;
  có thời lượng của endpoint cũ (đỉnh −35 dB) để đối chiếu.
- Các bảng kiểm tra âm thanh, pitch, cặp DTW, confusion matrix và so sánh dự đoán
  hiển thị trực tiếp trong notebook/báo cáo, không xuất CSV riêng.
- `endpoint_design.json`: tham số endpoint và 15 file train dùng để kiểm tra biên trước đánh giá test.
- `endpoint_reference.json`: số đo trước sửa endpoint; bảng so sánh accuracy trong
  báo cáo chỉ xuất hiện khi hash WAV và cấu hình MFCC còn khớp.
- `split.json`, `configurations.json`: lưu split và cấu hình.
- `templates_baseline.npz`: template baseline có lưu cấu hình, đọc không dùng pickle.
  Template E1/E2 được tính trong bộ nhớ khi chạy đánh giá.
- `report_Lab02.md`: báo cáo sinh từ số đo, có đáp án 9 câu hỏi. Báo cáo cũng hiển thị cuối notebook.

Hình trong notebook và báo cáo dùng cùng số thứ tự, chú thích nghiêng dưới ảnh theo dạng
`Hình 1. ...`, như Lab 1. Chú thích nêu tên bản ghi, nội dung các panel, trục, đơn vị
và tham số liên quan. Heatmap phân tích lỗi được lưu riêng trong `figures/mfcc_error.png`.

Trước khi nộp: dùng bộ ghi âm thật, nghe lại biên “không” và “hai”, bổ sung mô tả thu âm
và nhận xét trên kết quả thực tế. Chạy lại notebook từ kernel sạch; nộp cả module Python
cùng notebook và dữ liệu. Đề cho phép phần Markdown cuối notebook thay báo cáo PDF.

## Nhận dạng WAV từ dòng lệnh

Với template từ bộ ghi âm đã có:

```bash
python lab2.py predict dataset/khong/khong_04.wav --templates templates_baseline.npz
```

Đánh giá bộ ghi âm và tạo template:

```bash
python lab2.py evaluate --dataset dataset --output .
python lab2.py predict /duong/dan/tu_moi.wav --templates templates_baseline.npz
```

CLI `evaluate` xuất CSV và template; notebook bổ sung hình và báo cáo.
Notebook **Run All** tạo đúng 4 CSV chính nêu trên. `.gitignore` loại các CSV phụ
của những lần chạy cũ, bản nguồn điện thoại và bản sao lưu khỏi commit.
Recognizer luôn chọn một trong 5 nhãn; chưa có reject unknown. File im lặng bị báo lỗi.

## Cấu hình và mã nguồn

`lab2.py` chứa pipeline: frame 400/hop 160; Hamming; pre-emphasis 0,97; FFT 512;
24 bộ lọc Mel; ln; DCT-II trực chuẩn; 13 MFCC; CMN từng utterance.
MFCC được tự cài, Δ dùng Librosa. DTW Euclid được tự cài DP và backtracking;
tối ưu tổng chi phí rồi chia số cặp frame trên path. Không dùng mô hình ASR.

Endpoint mặc định dùng nền median 200 ms đầu/cuối và nội suy theo thời gian;
median smoothing 30 ms; ngưỡng xác nhận nền +8 dB trong ít nhất 40 ms;
ngưỡng mở rộng nền +3 dB; nối gap tối đa 80 ms; giữ đệm 200 ms hai phía.
Các giá trị được chọn từ kiểm tra biên trên train, không tối ưu bằng accuracy test.
Nếu clip quá ngắn hoặc không có vùng speech đủ tin cậy, giữ nguyên và xuất trạng thái fallback.
Giả định mỗi file có một từ và đầu/cuối chủ yếu là nền; vẫn cần nghe kiểm tra phụ âm yếu.
Chế độ `endpoint_method="relative"` giữ để đọc template cũ; template mới lưu đủ cấu hình adaptive.
`top_db=35` chỉ dùng cho endpoint relative/đối chứng cũ; adaptive dùng các ngưỡng theo nền.

`lab2_visuals.py` tạo hình và báo cáo. `scripts/build_notebook.py` tái tạo notebook
từ mã nguồn (ghi đè notebook và xóa outputs đã chạy, chỉ dùng khi cần cập nhật khung).

```bash
python scripts/check_pipeline.py
```

Kiểm tra DTW bằng duyệt tất cả path của ma trận nhỏ, identity/time stretching,
endpoint không mất vùng có tín hiệu, ZCR, MFCC/Δ, resample stereo và template roundtrip.
Có thêm kiểm tra nền nhiễu, âm đầu yếu, xung ngắn, khoảng ngắt trong từ và fallback.
