#!/usr/bin/env python3
"""Build the teaching notebook from the checked Python implementation."""
import inspect
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lab2

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


def implementation(*functions):
    code("\n\n".join(inspect.getsource(function).strip() for function in functions))


md(r"""
# CSE457 — Lab 2: Đặc trưng tiếng nói và nhận dạng bằng DTW
**Lỗ Anh Việt — MSSV 2351260695**

Bài toán: nhận dạng 5 từ **không, một, hai, ba, bốn** bằng MFCC + đối sánh DTW.
Notebook thực hiện A–G, hai thí nghiệm bắt buộc và xuất WAV đã trim, hình, CSV, template, báo cáo.

**Nguồn dữ liệu được hiển thị ở ô cấu hình.** Khi `dataset/` có đủ 5 WAV/từ,
notebook dùng bộ ghi âm đó.
Nếu bộ ghi âm mới có một phần, notebook báo thiếu dữ liệu để bạn hoàn thiện.

Chọn kernel có thư viện trong `requirements.txt`, sau đó **Restart Kernel → Run All**.
Nộp toàn bộ thư mục để notebook có các module `lab2.py` và `lab2_visuals.py`.
""")
code("""
from pathlib import Path
from dataclasses import replace
import csv
import json
import sys
from io import BytesIO

# Chạy khi mở tại thư mục Lab 2 hoặc thư mục gốc XuLyAmThanh.
candidates = [Path.cwd(), Path.cwd() / 'ThucHanh' / 'Lab02_2351260695_LoAnhViet']
ROOT = next((p.resolve() for p in candidates if (p / 'lab2.py').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Mở VS Code/Jupyter ở thư mục Lab 2 hoặc thư mục gốc dự án.')
sys.path.insert(0, str(ROOT))

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from scipy.ndimage import median_filter
from scipy.signal import lfilter
from IPython.display import Audio, Image, Markdown, display
from lab2 import (Config, EPS, LABELS, VI_LABELS, load_audio, choose_dataset, data_source,
                  split_dataset, extract_feature, recognize, run_experiments, write_csv,
                  load_templates)
from lab2_visuals import (audit_dataset, plot_time_analysis, plot_endpoint,
                         plot_autocorrelation, plot_mel, plot_mfcc, plot_dtw,
                         plot_confusion, most_confused, write_report)

CONFIG = Config()
DATASET, OUTPUT = choose_dataset(ROOT)
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCE = data_source(DATASET)
print('Dataset:', DATASET)
print('Đầu ra:', OUTPUT)
print('Nguồn dữ liệu:', SOURCE)
print(CONFIG)
FIGURE_NUMBER = 0

def show_table(rows):
    keys = list(rows[0])
    lines = ['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |']
    for row in rows:
        values = [f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]) for k in keys]
        lines.append('| ' + ' | '.join(values) + ' |')
    display(Markdown('\\n'.join(lines)))

def show_figure(fig):
    global FIGURE_NUMBER
    # Embed PNG explicitly, including when execution uses the headless Agg backend.
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    FIGURE_NUMBER += 1
    display(Markdown(f'*Hình {FIGURE_NUMBER}. {fig.lab2_caption}*'))
    plt.close(fig)
""")
md(r"""
## A. Thu dữ liệu, kiểm tra chất lượng và chia train/test

WAV mono PCM, 16 kHz; mỗi từ tối thiểu 5 lần, có 0,2–0,5 s silence đầu/cuối.
Tên ASCII ánh xạ sang nhãn tiếng Việt. Lấy 3 file đầu/lớp làm template và các file còn lại làm test.
Chia trước mọi thí nghiệm; không dùng file test làm template.
Peak/clipping được kiểm tra trên **audio gốc**, trước chuẩn hóa.
""")
code("""
split = split_dataset(DATASET)
show_table([{'label': label, 'nhãn': VI_LABELS[label],
             'train': ', '.join(p.name for p in split['train'][label]),
             'test': ', '.join(p.name for p in split['test'][label])} for label in LABELS])
quality, endpoint_rows = audit_dataset(DATASET, split, OUTPUT, CONFIG)
show_table(quality)
if any(not row['input_format_ok'] for row in quality):
    print('Có file khác mono PCM 16 kHz: pipeline chuyển mono/resample khi đọc; xem bảng kiểm tra phía trên.')
if any(row['clipping_fraction'] > 0 for row in quality):
    print('Có mẫu gần clipping: nên nghe và ghi lại nếu méo tiếng.')
""")
md(r"""
## B. Đặc trưng miền thời gian

Frame 25 ms = 400 mẫu, hop 10 ms = 160 mẫu. Frame cuối thiếu mẫu được zero-pad.
Energy, magnitude và RMS dùng frame nhân Hamming:

$$E_r=\sum_n x_r[n]^2,\quad M_r=\sum_n|x_r[n]|,\quad RMS_r=\sqrt{E_r/L}.$$

ZCR dùng frame chữ nhật, quy ước dấu tại 0 bằng +1, đếm số lần đổi dấu chia cho L.
Silence có năng lượng thấp; voiced thường energy cao và ZCR thấp hơn unvoiced.
Nhiễu nền cũng có thể có ZCR cao; không suy ra speech chỉ từ ZCR.
""")
implementation(lab2.frame_signal, lab2.time_features)
code("""
for label in ('khong', 'hai', 'ba'):
    show_figure(plot_time_analysis(split['train'][label][0], OUTPUT, CONFIG))
example = time_features(load_audio(split['train']['khong'][0], CONFIG), CONFIG)
show_table([{'frame': i, 'time_s': float(example['time'][i]),
             'energy': float(example['energy'][i]), 'magnitude': float(example['magnitude'][i]),
             'rms': float(example['rms'][i]), 'zcr': float(example['zcr'][i])}
            for i in np.linspace(0, len(example['energy']) - 1, 8, dtype=int)])
""")
md(r"""
## C. Endpoint detection

Ước lượng nền bằng median log-energy trong **200 ms đầu/cuối**, nội suy mức nền theo thời gian.
Làm trơn median 30 ms; ngưỡng cao **nền + 8 dB** xác nhận speech khi kéo dài ít nhất 40 ms.
Ngưỡng thấp **nền + 3 dB** mở rộng biên; nối khoảng ngắt ≤80 ms và chọn vùng có đỉnh tương phản
nền lớn nhất. Giữ **200 ms đệm** hai phía để bảo vệ âm /kh/, /h/ yếu.
Energy được tính **trước pre-emphasis**; ZCR dùng để quan sát, chưa tinh chỉnh biên bằng ZCR.

Cấu hình được chọn bằng kiểm tra biên trên 15 file train, giữ cố định trước đánh giá test.
Đối chứng với endpoint cũ (đỉnh −35 dB, margin 50 ms) được lưu cùng bảng `endpoint.csv`.
Nếu clip quá ngắn để ước lượng nền hoặc không có speech đủ tin cậy, giữ nguyên và ghi rõ trạng thái fallback.
Phương pháp giả định một từ/file và đầu/cuối chủ yếu là nền; tiếng người khác vẫn có thể làm sai biên.
""")
implementation(lab2.detect_endpoints)
code("""
show_table(endpoint_rows)
show_table([{'file': row['file'], 'before_s': row['before_s'],
             'relative_35db_after_s': row['relative_35db_after_s'],
             'adaptive_after_s': row['after_s']} for row in endpoint_rows])
for label in ('khong', 'hai'):
    path = split['train'][label][0]
    show_figure(plot_endpoint(path, OUTPUT, CONFIG))
    original = load_audio(path, CONFIG)
    trimmed, _ = detect_endpoints(original, CONFIG)
    print(f'{VI_LABELS[label]}: nghe trước rồi sau trim để kiểm tra mất âm đầu/cuối')
    display(Audio(original, rate=CONFIG.sr))
    display(Audio(trimmed, rate=CONFIG.sr))
""")
md(r"""
### Autocorrelation và pitch minh họa

$$R[k]=\sum_n x[n]x[n+k],\qquad F_0\approx F_s/k_0.$$

Chọn frame energy lớn nhất của “ba”, bỏ DC, nhân Hamming; tìm đỉnh trong miền tương ứng 70–400 Hz.
Không lấy lag 0. Đây là minh họa tính tuần hoàn, có thể nhầm bội/ước pitch;
không dùng pitch để nhận dạng ở bài này.
""")
code("""
fig, pitch_result = plot_autocorrelation(split['train']['ba'][0], OUTPUT, CONFIG)
show_figure(fig)
show_table([pitch_result])
""")
md(r"""
## D. MFCC tự triển khai

Pre-emphasis α=0,97 → frame/Hamming → FFT/power → 24 Mel filters → ln → DCT-II → 13 hệ số.

$$P[k]=|FFT(x_r)[k]|^2/512,\quad S[m]=\ln(\sum_k P[k]H_m[k]+\varepsilon).$$

Thang Mel: $1125\ln(1+f/700)$. DCT-II trực chuẩn; giữ c0–c12; CMN theo utterance.
Frame thực sự dài 400 mẫu, sau đó FFT padding lên 512, tránh nhầm độ dài frame và NFFT.
Mỗi hàng feature là một frame: **shape (T,13)**. T thay đổi theo thời lượng, số chiều cố định.
Đây là một cài đặt MFCC nhất quán theo công thức đề, không yêu cầu trùng số với mặc định Librosa.
""")
implementation(lab2.mel_filterbank, lab2.mfcc_feature)
code("""
show_figure(plot_mel(OUTPUT, CONFIG))
mfcc_examples = [split['train']['khong'][0], split['train']['ba'][0]]
show_figure(plot_mfcc(mfcc_examples, OUTPUT, CONFIG))
show_table([{'file': p.name, 'shape': str(extract_feature(p, CONFIG).shape)} for p in mfcc_examples])
""")
md(r"""
## E. Euclidean local distance và DTW tự cài

$$C[i,j]=\|X_i-Y_j\|_2,$$
$$D[i,j]=C[i,j]+\min(D[i-1,j],D[i,j-1],D[i-1,j-1]).$$

Ma trận có biên ∞, chỉ D[0,0]=0 trong ma trận có padding. Lưu predecessor và truy vết
từ frame cuối về frame đầu. Khi hòa, ưu tiên chéo → dọc → ngang để kết quả lặp lại được.
Chuẩn hóa **sau** khi tối ưu tổng: DTW_norm = total/path length.
""")
implementation(lab2.local_distances, lab2.dtw_distance)
code("""
X = extract_feature(split['train']['khong'][0], CONFIG)
self_score, self_path, _, _ = dtw_distance(X, X)
assert np.isclose(self_score, 0), 'DTW(X,X) phải gần 0'
print('DTW(X,X) =', self_score)
pair_rows = []
for name, other in [('same_word', split['train']['khong'][1]),
                    ('different_word', split['train']['ba'][0])]:
    fig, row = plot_dtw(split['train']['khong'][0], other, OUTPUT, name, CONFIG)
    show_figure(fig)
    pair_rows.append(row)
show_table(pair_rows)
print('Cùng từ thường có score thấp hơn; cần căn cứ số đo của bộ dữ liệu hiện tại.')
""")
md(r"""
## F–G. Nearest-template, đánh giá và hai thí nghiệm bắt buộc

Mỗi nhãn w có 3 template: $D_w(X)=\min_r DTW_{norm}(X,T_{w,r})$;
dự đoán nhãn có D_w nhỏ nhất. Top-3 gồm **nhãn** đã gộp template.

| Cấu hình | Endpoint | Feature |
|---|---|---|
| Baseline | Có | 13 MFCC |
| E1 | Không | 13 MFCC |
| E2 | Có | 13 MFCC + 13 Δ |

Giữ cùng split, vocabulary, số template và thuật toán DTW; trích lại features train/test mỗi cấu hình.
Δ dùng Librosa, cửa sổ tối đa 9 frame. Chưa dùng reject unknown và chưa có cross-speaker.
""")
implementation(lab2.build_templates, lab2.recognize)
code("""
split, experiments, summaries = run_experiments(DATASET, OUTPUT, CONFIG)
show_table(summaries)
baseline = experiments['baseline']
show_table(baseline['rows'])  # Đủ 10 file test, bao gồm top-3 và score.
show_figure(plot_confusion(experiments, OUTPUT))
print('Accuracy = số dự đoán đúng / số test. Với 10 test, mỗi lỗi thay đổi 10 điểm phần trăm.')
confused = most_confused(baseline['matrix'])
if confused:
    a, b, n = confused
    print('Nhầm có hướng nhiều nhất:', VI_LABELS[a], '→', VI_LABELS[b], f'({n} mẫu)')
else:
    print('Không có lỗi baseline trên tập test này; chưa có cặp nhầm để kết luận.')
""")
md(r"""
### Đối chiếu lỗi giữa các cấu hình

Không kết luận Δ hay trim luôn tốt hơn. Đối chiếu từng file vì hai cấu hình có thể
cùng accuracy nhưng sai các mẫu khác nhau. Với mẫu lỗi, nghe lại và xem biên trim,
MFCC và DTW path trước khi đề xuất nguyên nhân.
""")
code("""
comparisons = []
by_config = {name: {row['file']: row for row in result['rows']} for name, result in experiments.items()}
for row in baseline['rows']:
    name = row['file']
    comparisons.append({'file': name, 'true': row['true_label'],
                        **{config: results[name]['predicted_label'] for config, results in by_config.items()}})
show_table(comparisons)

# Tự tạo hình phân tích lỗi baseline nếu có; không bịa cặp nhầm khi accuracy=100%.
for previous_plot in (OUTPUT / 'figures').glob('dtw_error_*.png'):
    previous_plot.unlink()
for paths in split['test'].values():
    for path in paths:
        (OUTPUT / 'figures' / f'time_{path.stem}.png').unlink(missing_ok=True)
first_error = next((r for r in baseline['rows'] if not r['correct']), None)
error_figures = []
if first_error:
    error_path = DATASET / first_error['file']
    fig = plot_time_analysis(error_path, OUTPUT, CONFIG)
    error_figures.append((f'time_{error_path.stem}.png', fig.lab2_caption))
    show_figure(fig)
    fig = plot_mfcc([error_path, split['train'][first_error['true_label']][0],
                     split['train'][first_error['predicted_label']][0]],
                    OUTPUT, CONFIG, name='mfcc_error.png')
    error_figures.append(('mfcc_error.png', fig.lab2_caption))
    show_figure(fig)
    for label in (first_error['true_label'], first_error['predicted_label']):
        refs = split['train'][label]
        best_ref = min(refs, key=lambda p: dtw_distance(extract_feature(error_path, CONFIG),
                                                      extract_feature(p, CONFIG))[0])
        fig, _ = plot_dtw(error_path, best_ref, OUTPUT, 'error_' + label, CONFIG)
        error_figures.append((f'dtw_error_{label}.png', fig.lab2_caption))
        show_figure(fig)
    display(Audio(load_audio(error_path, CONFIG), rate=CONFIG.sr))
""")
md(r"""
## Nhận dạng một WAV chưa biết

Đổi `UNKNOWN_WAV` thành đường dẫn WAV cần nhận dạng. Mặc định dùng một file test,
đã tách khỏi template. Template NPZ lưu cả cấu hình để predict không dùng khác pipeline.
""")
code("""
UNKNOWN_WAV = split['test']['khong'][0]  # Ví dụ: ROOT / 'my_word.wav'
saved_templates, saved_config, template_source = load_templates(OUTPUT / 'templates_baseline.npz')
prediction, scores = recognize(UNKNOWN_WAV, saved_templates, saved_config)
print('File:', UNKNOWN_WAV)
print('Nhãn dự đoán:', VI_LABELS[prediction])
print('Khoảng cách DTW:', scores[prediction])
print('Nguồn template:', template_source)
show_table([{'rank': i, 'label': VI_LABELS[label], 'dtw_norm': score}
            for i, (label, score) in enumerate(list(scores.items())[:3], 1)])
display(Audio(load_audio(UNKNOWN_WAV, CONFIG), rate=CONFIG.sr))
""")
md(r"""
## Báo cáo và sản phẩm nộp

Sinh báo cáo Markdown từ kết quả thực đo, có bảng cấu hình, thời lượng trim,
DTW cùng/khác từ, E1/E2, confusion matrix, giới hạn và trả lời 9 câu hỏi.
Trước khi nộp, bổ sung nhận xét nghe biên trim và mô tả người nói/micro/môi trường thu.
Phần báo cáo trong notebook và file Markdown đáp ứng lựa chọn báo cáo của đề.
""")
code("""
report_path = write_report(DATASET, OUTPUT, split, experiments, summaries,
                           quality, endpoint_rows, pair_rows, pitch_result, error_figures)
print('Báo cáo:', report_path)
print('Baseline CSV:', OUTPUT / 'results.csv')
print('Thí nghiệm:', OUTPUT / 'experiments.csv')
print('Hình:', OUTPUT / 'figures')
report_text = report_path.read_text(encoding='utf-8')
display(Markdown(report_text))
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3 (Lab 2)", "language": "python", "name": "python3"}
notebook.metadata.language_info = {"name": "python", "version": "3.12"}
nbf.write(notebook, ROOT / "Lab2_2351260695.ipynb")
print(f"Đã tạo notebook với {len(cells)} cells.")
