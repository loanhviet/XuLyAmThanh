#!/usr/bin/env python3
"""Generate the reproducible Vietnamese teaching notebook."""
from pathlib import Path
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from theory import theory_markdown
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# LAB 3 — NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
**Lỗ Anh Việt — 2351260695 — CSE457**

Nhận dạng 5 lệnh **mở, đóng, lên, xuống, dừng** theo case study trong PDF.
Chạy **Restart Kernel → Run All** trong môi trường của dự án. Notebook gọi mã trong
`lab3.py`; hình ở `figures/`, model ở `models/`, bảng CSV và báo cáo Markdown ở thư mục gốc.

**Dữ liệu:** 40 WAV tổng hợp TTS giọng NamMinh, 6 train + 2 test mỗi lệnh,
mono/16 kHz/PCM16 và silence đầu/cuối 0,2–0,5 giây. Nguồn/setting/hash được ghi
trong `dataset/manifest.json`. Cùng engine/giọng ở train/test; đây là kiểm chứng
pipeline trên TTS, chưa đại diện nhận dạng người nói thật.

`lexicon.txt` đang chờ phone inventory của giảng viên. Corpus LM tự tạo chỉ để minh
họa; các previous-word context không phải context đã ghi nhận trong test.
""")
md("""
## 1. Môi trường và giao thức đánh giá

Baseline được định trước: 5 state, diagonal covariance, 39-D và CMVN theo utterance.
Không chọn model cuối bằng test accuracy. Các ablation dùng cùng split và được báo
cáo mô tả. Mỗi utterance là một yêu cầu TTS riêng với setting rate/pitch định trước;
train/test cùng giọng, không gọi là speaker-independent. SHA-256 và decoded-audio
hash kiểm tra trùng train/test; không đảm bảo sự độc lập như bản ghi người thật.

Cài đặt ở terminal: `uv venv --python 3.12 .venv`, sau đó
`uv pip install --python .venv/bin/python -r requirements-lock.txt`.
""")
code("""
from pathlib import Path
import inspect
import json
import sys
import joblib
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown, Code, Audio

ROOT = Path.cwd()
if not (ROOT / 'lab3.py').exists():
    raise RuntimeError('Mở notebook từ thư mục gốc Lab03 có lab3.py.')
sys.path.insert(0, str(ROOT))
from lab3 import (Config, VI, read_dataset, load_audio, extract_array, extract_features,
    make_lr_hmm, fit_models, check_topology, acoustic_scores, recognize,
    diag_log_emission, forward_backward, parse_lexicon, build_lm, bigram_prob, rerank, run_lab)

DATASET = ROOT / 'dataset'
OUT = ROOT
CONFIG = Config()
print('Python:', sys.version)
print(CONFIG)
labels, samples, audit, notices = read_dataset(DATASET)
display(pd.DataFrame(audit).groupby(['split', 'label']).size().unstack(fill_value=0))
for notice in notices:
    print('Lưu ý:', notice)
""")
md("""
## 2. A — Trích đặc trưng MFCC + Δ + Δ²

WAV → mono/16 kHz → trim silence → pre-emphasis 0,97 → cửa sổ Hamming 25 ms,
hop 10 ms, FFT 512, 26 Mel → 13 MFCC + 13 delta + 13 delta-delta → CMVN.
Mỗi utterance có shape **T × 39**, T được giữ nguyên theo độ dài âm thanh.
Delta ước lượng bằng hồi quy lân cận; file quá ngắn/không có tiếng bị từ chối.

CMVN theo utterance không học thống kê chung từ test. Thống kê của một file test
chỉ được dùng để biến đổi chính file đó giống như khi suy luận.
""")
code("""
display(Code(inspect.getsource(extract_array), language='python'))
for sample in samples[:3]:
    F = extract_features(sample.path, CONFIG)
    print(sample.path.name, F.shape, 'finite:', np.isfinite(F).all())
    assert F.shape[1] == 39
""")
md("""
## 3. B–H — Chạy pipeline và lưu kết quả

Lệnh dưới chạy toàn bộ thí nghiệm: huấn luyện, likelihood, Viterbi, DTW, số state,
13/39-D, số mẫu train, nhiễu 20 dB, bigram và báo cáo Markdown.
Thời gian decode tách khỏi front-end; DTW đã warm-up JIT trước khi đo.
""")
code("""
summary = run_lab(DATASET, OUT, ROOT / 'models', ROOT / 'corpus.txt', ROOT / 'lexicon.txt', CONFIG)
display(pd.Series(summary)[['train_count', 'test_count', 'train_accuracy',
    'test_accuracy', 'dtw_lab2_accuracy', 'dtw_shared_accuracy', 'noise_accuracy']])
""")
code("""
display(Image(filename=str(OUT / 'figures/features_1.png')))
display(Image(filename=str(OUT / 'figures/features_2.png')))
""")
md("""
MFCC biểu diễn phổ; delta và delta-delta làm rõ chuyển động cục bộ quanh onset,
offset và các chuyển tiếp. Vị trí/số frame khác nhau phản ánh độ dài từng từ và cách
nói. Đọc hình cùng audio để nhận xét; không suy ra âm vị chỉ từ màu heatmap.
""")
md("""
## 4. B–C — HMM trái-sang-phải và Baum–Welch

HMM gồm π (phân bố bắt đầu), A (chuyển trạng thái), B (mật độ Gaussian phát xạ).
π = [1,0,…,0]; A chỉ có self-loop/next-state, state cuối hấp thụ. Khởi tạo emission
bằng các phần thời gian liên tiếp của **train** để state có thứ tự âm học ban đầu.

`init_params=''` vì đã khởi tạo tất cả; `params='tmc'` chỉ cập nhật A, mean, covariance.
Các utterance cùng lớp được ghép, truyền `lengths` để không tạo chuyển trạng thái
giả giữa hai bản ghi. Sau fit kiểm tra π, tổng từng hàng A và các transition cấm.
""")
code("""
display(Code(inspect.getsource(make_lr_hmm), language='python'))
payload = joblib.load(ROOT / 'models/hmm_baseline.joblib')
models = payload['models']
for label, model in models.items():
    check_topology(model)
    print(label, 'start:', model.startprob_)
    print(np.round(model.transmat_, 3))
diagnostics = json.loads((OUT / 'training_diagnostics.json').read_text())
display(pd.DataFrame(diagnostics)[['label', 'utterances', 'frames', 'iterations', 'stop_reason', 'last_gain']])
display(Image(filename=str(OUT / 'figures/convergence.png')))
""")
md("""
Diagonal covariance có 39 phương sai/state; full covariance có 780 phần tử độc lập
cho D=39. Với ít dữ liệu, ước lượng full dễ bất ổn. `monitor.converged` có thể True
khi hết số vòng lặp; cần đọc gain cuối và lý do dừng. Baum–Welch là EM và chỉ tìm
nghiệm cục bộ, phụ thuộc khởi tạo, số state, lượng dữ liệu và độ khớp môi trường.
""")
md("""
## 5. D — Likelihood, Forward, Viterbi và trách nhiệm EM

Forward dùng tổng qua mọi đường để tính likelihood. Viterbi dùng max để tìm đường
trạng thái tốt nhất. Trong miền log, tổng xác suất là `logsumexp`.

γₜ(i) = P(qₜ=i | X): trách nhiệm state i cho frame t.
ξₜ(i,j) = P(qₜ=i, qₜ₊₁=j | X): trách nhiệm chuyển i→j tại t.
Forward–Backward bên dưới minh họa E-step; không tự thay thế Baum–Welch của thư viện.
""")
code("""
display(Code(inspect.getsource(forward_backward), language='python'))
test_samples = [s for s in samples if s.split == 'test']
s = test_samples[0]
F = extract_features(s.path, CONFIG)
scores = acoustic_scores(F, models)
display(pd.DataFrame({'label': list(scores), 'log_likelihood': list(scores.values())}))
ll, gamma, xi_sum = forward_backward(diag_log_emission(F, models[s.label]),
    models[s.label].startprob_, models[s.label].transmat_)
viterbi_logp, states = models[s.label].decode(F, algorithm='viterbi')
assert np.isclose(ll, models[s.label].score(F), atol=1e-6)
assert ll >= viterbi_logp - 1e-6
assert np.allclose(gamma.sum(axis=1), 1)
assert np.isclose(xi_sum.sum(), len(F) - 1)
print('Forward:', ll, 'Viterbi:', viterbi_logp)
print('Prediction:', max(scores, key=scores.get), '| reference:', s.label)
display(pd.read_csv(OUT / 'likelihood_examples.csv'))
""")
code("""
display(Image(filename=str(OUT / 'figures/viterbi_1.png')))
display(Image(filename=str(OUT / 'figures/viterbi_2.png')))
display(pd.read_csv(OUT / 'state_durations.csv'))
""")
md("""
Self-loop biểu diễn duration ngầm: nói chậm có thể tăng số frame giữ cùng state.
Occupancy xấp xỉ số frame × 10 ms. HMM này không bắt buộc mọi utterance đi tới state
cuối; tổng likelihood kết thúc ở mọi state như công thức trong đề. State chưa được
giám sát theo phoneme nên không được gán tên âm vị cụ thể.
""")
md("""
## 6. E — Đánh giá và so sánh DTW

DTW Lab 2 giữ front-end cũ: endpoint adaptive, 24 Mel, 13 MFCC và CMN; mã nằm ở
`lab2_baseline.py`. DTW chung front-end dùng cùng 39-D/CMVN như HMM. Cả hai lấy min
distance qua tất cả template train, chuẩn hóa tổng chi phí theo độ dài đường DTW.
Recurrence đã được đối chiếu với mã gốc. Các phương pháp dùng đúng cùng train/test.
""")
code("""
display(pd.read_csv(OUT / 'dtw_vs_hmm.csv'))
display(Image(filename=str(OUT / 'figures/confusion_hmm.png')))
display(Image(filename=str(OUT / 'figures/confusion_dtw_lab2.png')))
comparison_cases = pd.read_csv(OUT / 'comparison_cases.csv')
display(comparison_cases)
different = comparison_cases[comparison_cases.hmm_correct != comparison_cases.dtw_correct]
display(different if len(different) else Markdown('Chưa có trường hợp khác nhau về đúng/sai; cần thêm test.'))
display(pd.read_csv(OUT / 'training_sizes.csv'))
""")
md("""
DTW lưu template, xử lý biến thiên bằng warping; HMM học phân bố và duration qua
transition, cần dữ liệu để ước lượng tham số. HMM có thể khái quát từ nhiều bản ghi;
DTW có thể thuận lợi khi ít mẫu và test gần template. Ví dụ lỗi trong bảng là quan
sát, chưa đủ để xác định nguyên nhân; nên nghe lại trước khi giải thích.
""")
md("""
## 7. E1–E3 — Kiểm soát số state và đặc trưng

Thay một biến mỗi lần, giữ split/seed/front-end còn lại cố định. 13-D là chính phần
MFCC tĩnh của 39-D, nên đối chiếu đặc trưng không đổi chuẩn hóa. Full covariance
chỉ chạy nếu tối thiểu mỗi lớp đủ 2 × D × n_states frame; nếu không thì ghi skipped.
Đây là tiêu chí thận trọng của bài triển khai, không phải định lý đủ dữ liệu.
""")
code("""
experiments = pd.read_csv(OUT / 'experiments.csv')
display(experiments)
baseline_accuracy = experiments.loc[experiments.experiment == 'baseline_states5_39D', 'test_accuracy'].iloc[0]
static_accuracy = experiments.loc[experiments.experiment == 'E2_mfcc13', 'test_accuracy'].iloc[0]
print(f'39-D trừ 13-D: {(baseline_accuracy - static_accuracy) * 100:.0f} điểm phần trăm trên test hiện tại.')
""")
md("""
## 8. F — Lexicon mini

Whole-word HMM ánh xạ nhãn trực tiếp sang model nên không cần pronunciation lexicon
để nhận dạng. Với subword HMM, từ được tạo bằng cách nối phone HMM theo lexicon.
Chưa có inventory của giảng viên: giữ `lexicon.txt` là template chú thích, không tự
đặt phiên âm rồi coi là chuẩn. Parser được kiểm thử bằng inventory giả trong tests.
""")
code("""
lexicon = parse_lexicon(ROOT / 'lexicon.txt')
print('Lexicon hiện tại:', lexicon)
print('Các từ cần phone sequence từ giảng viên:', sorted(set(labels) - set(lexicon)))
display(Code(inspect.getsource(parse_lexicon), language='python'))
""")
md("""
## 9. G — Unigram, bigram và MAP re-ranking

Corpus tự tạo trong `corpus.txt`, không lấy nhãn test làm corpus.
P(w|prev) = [C(prev,w)+k] / [C_out(prev)+kV], k=0,5.
`C_out` chỉ đếm context có từ kế tiếp trong vocabulary. Không đưa `</s>` vào mẫu số
nếu `</s>` không thuộc tập ứng viên; mỗi dòng độc lập, có context bắt đầu `<s>`.

Score(w) = log P(X|w)/T + λ log P(w|prev). Thử λ=0; 0,1; 0,3; 1,0 trên ba file có
margin âm học nhỏ nhất, liệt kê mọi previous-word context để tránh suy ngữ cảnh từ
nhãn thật. Đây là phân tích giả định, không báo accuracy LM trên chuỗi nói thực.
""")
code("""
lm = build_lm(ROOT / 'corpus.txt', labels)
for previous in ['<s>'] + labels:
    assert np.isclose(sum(bigram_prob(lm, previous, w) for w in labels), 1)
display(pd.read_csv(OUT / 'unigram_counts.csv'))
reranking = pd.read_csv(OUT / 'lm_reranking.csv')
changed = reranking[reranking.changed]
display(changed if len(changed) else reranking.head(12))
print('Số tình huống giả định đổi quyết định:', len(changed))
print('Không gộp những tình huống này thành accuracy của một test set có context thật.')
""")
md("""
λ=0 phải giữ nguyên acoustic argmax; λ lớn có thể giúp khi context phù hợp hoặc làm
sai khi context không phù hợp. Nếu không có thay đổi trong dải λ đang thử, báo đúng
quan sát đó, không tạo kết quả giả để minh họa ưu thế của LM.
""")
md("""
## 10. H — Robustness và unknown/reject

Nhiễu Gaussian cộng vào tín hiệu trước trích đặc trưng, SNR 20 dB theo công suất phần
tiếng nói sau trim; seed cố định từng file. Không đưa nhiễu test vào fit.
Với từ ngoài vocabulary, argmax vẫn chọn một nhãn vì chỉ biết các model đã học.
Có thể reject khi score/frame thấp hoặc margin top-1/top-2 nhỏ.

Ngưỡng trong `reject_proposal.json` là gợi ý lấy phân vị 5% của các mẫu đúng trên
holdout train (một file/lớp); không hiệu chỉnh trên test và chưa dùng để đổi dự đoán
baseline. Vì quá ít calibration và final model được train lại với nhiều mẫu hơn,
cần validation độc lập cùng unknown thật trước khi áp dụng.
""")
code("""
print('Clean:', summary['test_accuracy'], 'Noise:', summary['noise_accuracy'])
display(pd.read_csv(OUT / 'noise_methods.csv'))
display(Image(filename=str(OUT / 'figures/confusion_noise.png')))
display(pd.read_csv(OUT / 'noise_frontend_analysis.csv'))
display(pd.read_csv(OUT / 'comparison_cases_noise.csv'))
display(Image(filename=str(OUT / 'figures/score_distributions.png')))
display(pd.read_csv(OUT / 'score_distributions.csv'))
display(json.loads((OUT / 'reject_proposal.json').read_text()))
""")
md("## 11. Câu hỏi lý thuyết và kết luận\n\n" + theory_markdown() + "\n\n" + """
Tổng hợp số đo có trong `report_Lab03.md` ở thư mục gốc.
Cần inventory/corpus giảng viên và đánh giá thêm bằng bản ghi người thật.
Không tuyên bố kết quả TTS đại diện cho nhiều người nói hoặc ASR tiếng Việt nói chung.
""")
code("""
display(Markdown((OUT / 'report_Lab03.md').read_text(encoding='utf-8')))
display(Markdown('[Mở báo cáo Markdown](report_Lab03.md)'))
print('Báo cáo:', OUT / 'report_Lab03.md')
print('Model:', ROOT / 'models/hmm_baseline.joblib')
print('Nhận dạng file khác: .venv/bin/python lab3.py predict đường_dẫn.wav')
""")
md("""
## 12. Tài liệu

- Đề Lab 3 của học phần CSE457, mục 4–7 và case study.
- [hmmlearn 0.3.3 — API Reference](https://hmmlearn.readthedocs.io/en/stable/api.html).
- [librosa — feature.delta](https://librosa.org/doc/latest/generated/librosa.feature.delta.html).
- Mã MFCC/DTW Lab 2 của cùng sinh viên: bản sao `lab2_baseline.py`.
- Rabiner & Schafer, *Theory and Applications of Digital Speech Processing*.
""")
nb.cells = cells
for index, cell in enumerate(cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'Lab03 cell {index + 1}', 'exec')
nb.metadata = {"kernelspec": {"display_name": "Python (Lab03)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
nbf.write(nb, ROOT / "Lab03_2351260695.ipynb")
print(f"Đã tạo Lab03_2351260695.ipynb ({len(cells)} cells).")
