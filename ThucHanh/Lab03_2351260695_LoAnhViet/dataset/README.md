# Dataset tổng hợp — 5 lệnh Lab 3

**Đây là tiếng nói tổng hợp, không phải bản ghi người thật.**

- Từ: mở (`mo`), đóng (`dong`), lên (`len`), xuống (`xuong`), dừng (`dung`).
- 1 giọng tổng hợp: vi-VN-NamMinhNeural.
- Mỗi từ, mỗi giọng: **6 train + 2 test**, tổng **40 WAV**.
- WAV mono, 16.000 Hz, PCM 16-bit; khoảng lặng đầu/cuối 0,2–0,5 giây.
- Mỗi utterance được yêu cầu TTS riêng với rate/pitch định trước; không chia bản
  sao/biến đổi hậu kỳ của một file gốc sang hai split.
- Tất cả WAV đã qua kiểm tra định dạng, hash train/test và đặc trưng T × 39.
- `manifest.json` ghi nhãn, chữ được đọc, giọng, split, thông số, silence và hash.
- `tts_sources/` giữ MP3 gốc để kiểm tra nguồn; `generation_config.json` ghi cấu hình.

Đúng số lượng/nhãn/định dạng trong PDF để minh họa pipeline. Cùng TTS engine/voice
ở train/test làm bài toán dễ hơn dữ liệu người thật; accuracy trên bộ này không
đại diện chất lượng nhận dạng thực tế. Việc dùng dữ liệu tổng hợp cho bài nộp cần
phù hợp với yêu cầu giảng viên.

## Chạy pipeline từ thư mục gốc dự án

```bash
.venv/bin/python lab3.py run --dataset dataset
```

## Tạo lại

Cần mạng, ffmpeg và gói trong `requirements-tts.txt`. Mỗi file TTS gốc được cache;
chạy lại dùng cache. Dịch vụ có thể thay đổi đầu ra ở những lần tạo mới.

```bash
uv pip install --python .venv/bin/python -r requirements-tts.txt
.venv/bin/python scripts/generate_synthetic_dataset.py --output dataset
```

Tài liệu công cụ: https://github.com/rany2/edge-tts
