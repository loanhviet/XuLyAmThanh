# Báo cáo Lab 01 — Phân tích và xử lý tín hiệu âm thanh số

**Họ và tên:** Lỗ Anh Việt  
**MSSV:** 2351260695  
**Học phần:** CSE457 — Xử lý âm thanh và tiếng nói

## 1. Dữ liệu và môi trường

Notebook sử dụng `audio/generated_conversation.wav`, tín hiệu mono PCM 16-bit, sampling rate 44.100 Hz và thời lượng 15,2 s. Tệp có 670.320 mẫu, kích thước 1.340.684 byte và tốc độ bit PCM lý thuyết 705,6 kbps. Mẫu được chuẩn hóa về miền `[-1, 1]` trước khi xử lý.

## 2. Phân tích miền thời gian

Tín hiệu mono có `Peak = 0,849976`, `RMS = 0,062291` và `Energy = 2600,96`. Peak nhỏ hơn 1 nên không phát hiện clipping rõ ràng. Waveform toàn tệp và waveform chi tiết 1 giây (`figures/waveform_segment.png`) được lưu từ notebook. Waveform cho thấy các vùng im lặng xen kẽ các đoạn nói bình thường, nói lớn, nói nhỏ và đoạn nói kèm white noise ở phần cuối.

| Đoạn | Peak | RMS | Energy |
|---|---:|---:|---:|
| Nói lớn (4,2–7,2 s) | 0,849976 | 0,091347 | 1103,94 |
| Nói nhỏ (8,2–11,2 s) | 0,199982 | 0,022787 | 68,69 |
| Có white noise (12,2–15,2 s) | 0,607391 | 0,083190 | 915,59 |

Đoạn nói lớn có RMS và năng lượng cao hơn rõ rệt so với đoạn nói nhỏ; đoạn có white noise có nền năng lượng cao và phổ rộng hơn.

## 3. FFT và phân tích miền tần số

Đoạn 4,5–5,5 s được nhân cửa sổ Hamming. Với `NFFT = 2048`, `Δf = 21,533 Hz`; với `NFFT = 8192`, `Δf = 5,383 Hz`. Một số đỉnh được phát hiện tự động quanh 193,8 Hz, 387,6 Hz, 495,3 Hz và 882,9 Hz. Tăng `NFFT` làm phổ được lấy mẫu dày hơn nhờ zero-padding, nhưng không tự làm tăng độ phân giải vật lý khi độ dài frame không đổi.

## 4. STFT, spectrogram và cửa sổ

Spectrogram được tính trên toàn bộ 15,2 s với hop 10 ms, các frame length 10/25/50 ms và `NFFT = 4096`. Kết quả được lưu tại `figures/spectrogram.png`. Frame ngắn bám sát transient tốt hơn nhưng dải tần rộng hơn; frame dài cho dải tần sắc hơn nhưng làm thay đổi nhanh theo thời gian bị nhòe. So sánh rectangular/Hamming trên cùng frame và `NFFT = 8192` cho thấy Hamming giảm side-lobe và spectral leakage, đổi lại main-lobe rộng hơn.

## 5. Lọc số

Hai FIR đối xứng 201 taps được thiết kế bằng cửa sổ Hamming:

- Low-pass cutoff 2 kHz: `audio/output/speech_lpf_2k.wav`.
- High-pass cutoff 3 kHz: `audio/output/speech_hpf_3k.wav`.

Độ trễ nhóm lý thuyết là `(201 - 1)/2 = 100` mẫu, tương đương khoảng 2,27 ms tại 44.100 Hz. Hình đáp ứng tần số và phổ trước/sau lọc nằm ở `figures/filter_response.png`.

## 6. Lượng tử hóa, resampling và mã hóa

### Lượng tử hóa

| Số bit | Số mức | SNR đo được |
|---:|---:|---:|
| 4 | 16 | 7,65 dB |
| 8 | 256 | 30,56 dB |
| 16 | 65.536 | 90,61 dB |

Các file lượng tử hóa nằm trong `audio/output/`. Giá trị được lượng tử hóa ở mức 4/8/16 bit, sau đó lưu trong container WAV PCM16 để bảo đảm khả năng phát trên thiết bị thông thường.

### Resampling

Resampling dùng `scipy.signal.resample_poly`, có lọc chống aliasing, và tạo:

- `audio/output/speech_16000Hz.wav`
- `audio/output/speech_8000Hz.wav`

Nyquist tương ứng là 8 kHz và 4 kHz. Phổ so sánh nằm ở `figures/resampling_spectrum.png`.

### Mã hóa và compression ratio

Với tín hiệu mono 44.100 Hz, 16 bit, tốc độ bit PCM lý thuyết là `705,6 kbps`; kích thước PCM lý thuyết là khoảng `1,341 MB`. File MP3 128 kbps được tạo tại `audio/output/speech_compressed_128kbps.mp3`, kích thước thực tế khoảng `0,244 MB`, bitrate đo được `128,5 kbps`. Compression ratio PCM/MP3 là khoảng `5,49:1`.

## 7. Kết luận

Pipeline A→G đã được triển khai bằng code có thể chạy lại từ đầu: đọc và chuẩn hóa audio, phân tích waveform/FFT/STFT, so sánh cửa sổ, lọc FIR, lượng tử hóa, resampling và mã hóa. Các kết quả cho thấy tham số xử lý quyết định trực tiếp sự cân bằng giữa độ phân giải thời gian–tần số, chất lượng nghe, SNR và kích thước dữ liệu.
