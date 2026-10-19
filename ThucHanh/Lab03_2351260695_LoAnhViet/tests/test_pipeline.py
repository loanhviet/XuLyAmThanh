"""Meaningful checks for probability recurrences, leakage and constrained learning."""
import itertools
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from scipy.special import logsumexp

from lab3 import (Config, Sample, acoustic_scores, add_noise, bigram_prob, build_lm,
                  check_topology, dtw_distance, extract_array, fit_models,
                  forward_backward, lr_transition, parse_lexicon, read_dataset, rerank)


def test_forward_backward_matches_all_paths():
    start = np.array([0.8, 0.2])
    trans = np.array([[0.7, 0.3], [0, 1]])
    emission = np.array([[0.4, 0.3], [0.2, 0.8], [0.7, 0.1]])
    paths = list(itertools.product(range(2), repeat=3))
    weights = np.array([start[q[0]] * emission[0, q[0]] *
        np.prod([trans[q[t - 1], q[t]] * emission[t, q[t]] for t in range(1, 3)]) for q in paths])
    ll, gamma, xi = forward_backward(np.log(emission), start, trans)
    assert np.isclose(ll, np.log(weights.sum()))
    expected = np.array([[sum(w for q, w in zip(paths, weights) if q[t] == state) / weights.sum()
                          for state in range(2)] for t in range(3)])
    assert np.allclose(gamma, expected)
    assert np.allclose(gamma.sum(axis=1), 1)
    assert np.isclose(xi.sum(), 2)
    assert xi[1, 0] == 0
    assert ll >= np.log(weights.max())


def test_constrained_fit_score_and_save(tmp_path):
    import joblib
    rng = np.random.default_rng(20)
    samples, fmap = [], {}
    for label, sign in [("a", 1), ("b", -1)]:
        for i in range(3):
            path = tmp_path / f"{label}{i}.wav"
            samples.append(Sample(label, path, "train"))
            fmap[str(path)] = np.vstack([rng.normal(sign * mu, 0.15, (12, 2)) for mu in [-2, 0, 2]])
    cfg = Config(n_states=3, n_iter=20)
    models, diag = fit_models(samples, fmap, ["a", "b"], cfg)
    for model in models.values():
        check_topology(model)
    F = fmap[str(samples[0].path)]
    scores = acoustic_scores(F, models)
    assert max(scores, key=scores.get) == "a"
    forward = models["a"].score(F)
    viterbi, path = models["a"].decode(F, algorithm="viterbi")
    assert forward >= viterbi - 1e-8
    assert np.all(np.diff(path) >= 0)
    assert np.all(np.diff(path) <= 1)
    joblib.dump(models, tmp_path / "model.joblib")
    assert acoustic_scores(F, joblib.load(tmp_path / "model.joblib")) == scores
    with pytest.raises(ValueError, match="split=train"):
        fit_models([replace(samples[0], split="test")], fmap, ["a"], cfg)


def test_features_short_audio_cmvn_and_silence():
    t = np.arange(1600) / 16000
    y = np.sin(2 * np.pi * (300 * t + 1200 * t ** 2))
    F = extract_array(y)
    assert F.shape[1] == 39 and np.isfinite(F).all()
    assert np.allclose(F.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(extract_array(y, Config(dynamic=False)), F[:, :13])
    with pytest.warns(UserWarning, match="n_fft=512 is too large"):
        assert extract_array(y[:480]).shape[1] == 39
    with pytest.raises(ValueError):
        extract_array(np.zeros(16000))


def test_leakage_detected_even_with_different_headers(tmp_path):
    t = np.arange(3200) / 16000
    for li, label in enumerate(["a", "b"]):
        for split, count in [("train", 2), ("test", 1)]:
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for i in range(count):
                frequency = 200 + 50 * li + 10 * i + (30 if split == "test" else 0)
                sf.write(folder / f"{i}.wav", np.sin(2 * np.pi * frequency * t) * 0.3, 16000, subtype="PCM_16")
    read_dataset(tmp_path)
    y, sr = sf.read(tmp_path / "train/a/0.wav")
    # Float WAV has different file bytes but identical decoded samples.
    sf.write(tmp_path / "test/a/0.wav", y, sr, subtype="FLOAT")
    with pytest.raises(ValueError, match="Trùng dữ liệu"):
        read_dataset(tmp_path)


def test_bigram_normalized_and_lm_weight_zero(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("a b\na b\nb a\n", encoding="utf-8")
    lm = build_lm(path, ["a", "b"])
    for previous in ["a", "b", "<s>", "unseen"]:
        assert np.isclose(sum(bigram_prob(lm, previous, w) for w in ["a", "b"]), 1)
    scores = {"a": -10, "b": -10.1}
    assert rerank(scores, "a", lm, 0)[0] == "a"
    assert rerank(scores, "a", lm, 1)[0] == "b"


def test_lexicon_rejects_duplicates(tmp_path):
    path = tmp_path / "lexicon.txt"
    path.write_text("# inventory fixture, not Vietnamese phones\na P1 P2\nb P3\n", encoding="utf-8")
    assert parse_lexicon(path) == {"a": ["P1", "P2"], "b": ["P3"]}
    path.write_text("a P1\na P2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_lexicon(path)


def test_dtw_matches_original_and_noise_is_reproducible():
    import lab2_baseline
    rng = np.random.default_rng(695)
    for n, m in [(1, 4), (4, 1), (7, 5), (9, 9)]:
        x, y = rng.normal(size=(n, 3)), rng.normal(size=(m, 3))
        assert np.isclose(dtw_distance(x, y), lab2_baseline.dtw_distance(x, y)[0])
    y = np.sin(np.arange(16000) * 0.1)
    assert np.array_equal(add_noise(y, 20, 4), add_noise(y, 20, 4))
    assert not np.array_equal(add_noise(y, 20, 4), add_noise(y, 20, 5))
