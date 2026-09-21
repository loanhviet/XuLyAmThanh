# Lab 01 — Phân tích và xử lý tín hiệu âm thanh số

Khung nộp bài cho MSSV `2351260695` (Lỗ Anh Việt), có sẵn audio test có cấu trúc cố định để dùng tiếp cho waveform, RMS, FFT, STFT, filtering, quantization và resampling.

## Tạo audio test

Từ thư mục Lab 01, chạy:

```bash
python3 scripts/generate_test_conversation.py --seed 42
```

Script tạo `audio/generated_conversation.wav` ở định dạng 44.1 kHz, mono, PCM 16-bit, dài 15.2 giây. Cấu trúc theo thứ tự: speaker A nói bình thường (3 s), silence (1.2 s), speaker B nói lớn (3 s), silence (1 s), speaker A nói nhỏ (3 s), silence (1 s), speaker B nói kèm white noise (3 s).

Hai speaker dùng voice, tốc độ và pitch khác nhau. Câu thoại được chọn ngẫu nhiên; `--seed` giúp tái tạo đúng cùng nội dung. Các đoạn WAV riêng và `manifest.json` được lưu trong `audio/generated_parts/`.

Script ưu tiên lệnh `espeak-ng`. Nếu máy chỉ có thư viện eSpeak NG cục bộ, script tự dùng thư viện này; nếu không có cả hai, cài bằng:

```bash
sudo apt install espeak-ng
```

## Môi trường

- Python 3.10 trở lên (đã kiểm tra máy có Python 3.12.3)
- FFmpeg để Pydub có thể đọc/ghi MP3 (đã có tại `/usr/bin/ffmpeg`)

Tạo môi trường ảo và cài các thư viện từ thư mục này:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Mở notebook bằng VS Code (chọn kernel `.venv`) hoặc:

```bash
python -m jupyter lab
```

## Quy ước thư mục

- `Lab01_2351260695.ipynb`: notebook sẽ chứa phần thực hành sau này.
- `report_Lab01.md`: khung báo cáo Markdown.
- `audio/input/`: tệp âm thanh đầu vào, không đưa dữ liệu lớn lên Git. Cell A tự tạo thư mục này nếu cần và chọn tệp đầu tiên theo thứ tự tên.
- `audio/output/`: WAV/MP3 sinh ra khi làm bài.
- `figures/`: các hình xuất từ notebook.

Khi bắt đầu làm bài, đặt một tệp đầu vào vào `audio/input/` và giữ tên các tệp đầu ra thể hiện rõ cấu hình (ví dụ: `music_lpf_2k.wav`). Nếu chưa đặt tệp, notebook tự dùng `audio/generated_conversation.wav` làm dữ liệu test. Nên mở VS Code tại thư mục `Lab01_2351260695_LoAnhViet` để `Path.cwd()` trỏ đúng project.
