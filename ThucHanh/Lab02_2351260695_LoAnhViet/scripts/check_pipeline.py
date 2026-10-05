#!/usr/bin/env python3
"""Small correctness checks for the numerical algorithms and template persistence."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab2 import (Config, LABELS, detect_endpoints, dtw_distance, load_audio,
                  local_distances, mfcc_feature, save_templates, load_templates, time_features)


class PipelineChecks(unittest.TestCase):
    def test_known_euclidean_distance(self):
        self.assertEqual(local_distances([[1, 2]], [[4, 6]])[0, 0], 5)

    def test_identity_and_time_stretch(self):
        x = np.array([[0.], [1.], [3.]])
        self.assertEqual(dtw_distance(x, x)[0], 0)
        self.assertEqual(dtw_distance(x, np.repeat(x, 2, axis=0))[0], 0)

    def test_global_optimum_by_exhaustive_small_grid(self):
        x, y = np.array([[0., 1.], [2., 0.]]), np.array([[1., 2.], [3., 1.], [2., 0.]])
        score, path, cost, _ = dtw_distance(x, y)
        totals = []

        def enumerate_paths(i, j, total):
            total += cost[i, j]
            if (i, j) == (len(x) - 1, len(y) - 1):
                totals.append(total)
                return
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                if i + di < len(x) and j + dj < len(y):
                    enumerate_paths(i + di, j + dj, total)

        enumerate_paths(0, 0, 0)
        self.assertAlmostEqual(score * len(path), min(totals))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (1, 2))
        self.assertTrue(all((b[0] - a[0], b[1] - a[1]) in ((1, 0), (0, 1), (1, 1))
                            for a, b in zip(path, path[1:])))
        self.assertAlmostEqual(score, dtw_distance(y, x)[0])

    def test_invalid_and_single_frame_dtw(self):
        self.assertEqual(dtw_distance([[1]], [[4]])[0], 3)
        for x, y in ((np.empty((0, 2)), np.ones((1, 2))), ([[1, 2]], [[1]]), ([[np.nan]], [[1]])):
            with self.assertRaises(ValueError):
                dtw_distance(x, y)

    def test_zcr_zero_convention(self):
        c = Config(frame=4, hop=2, n_fft=64, n_mels=24)
        f = time_features(np.array([-1., 0., -1., 1.]), c)
        self.assertEqual(f['zcr'][0], 3 / 4)

    def test_endpoint_preserves_speech_and_rejects_silence(self):
        c = Config()
        voice = .5 * np.sin(2 * np.pi * 180 * np.arange(6400) / c.sr)
        y = np.concatenate([np.zeros(4800), voice, np.zeros(4800)])
        trimmed, boundary = detect_endpoints(y, c)
        self.assertLessEqual(boundary['start'], 4800)
        self.assertGreaterEqual(boundary['end'], 11200)
        self.assertLess(len(trimmed), len(y))
        with self.assertRaises(ValueError):
            detect_endpoints(np.zeros(16000), c)

    def test_noisy_endpoint_keeps_weak_onset_and_ignores_short_impulse(self):
        c = Config()
        rng = np.random.default_rng(32)
        y = rng.normal(0, .01, round(2.4 * c.sr))
        first, last = round(.8 * c.sr), round(1.2 * c.sr)
        y[first:last] += .25 * np.sin(2 * np.pi * 180 * np.arange(last - first) / c.sr)
        weak_start = round(.69 * c.sr)
        y[weak_start:first] += rng.normal(0, .014, first - weak_start)
        impulse = round(2 * c.sr)
        y[impulse:impulse + 20] += .9
        trimmed, boundary = detect_endpoints(y, c)
        self.assertEqual(boundary['status'], 'trimmed')
        self.assertLessEqual(boundary['start'], weak_start)
        self.assertGreaterEqual(boundary['end'], last)
        self.assertGreater(boundary['start'], round(.3 * c.sr))
        self.assertLess(boundary['end'], round(1.5 * c.sr))
        self.assertLess(len(trimmed), len(y))

    def test_endpoint_bridges_short_pause(self):
        c = Config()
        y = np.random.default_rng(7).normal(0, .005, 2 * c.sr)
        for start_s, end_s in ((.6, .8), (.86, 1.1)):
            s, e = round(start_s * c.sr), round(end_s * c.sr)
            y[s:e] += .2 * np.sin(2 * np.pi * 160 * np.arange(e - s) / c.sr)
        _, boundary = detect_endpoints(y, c)
        self.assertLessEqual(boundary['start'], round(.6 * c.sr))
        self.assertGreaterEqual(boundary['end'], round(1.1 * c.sr))

    def test_endpoint_reports_uncertain_and_short_audio(self):
        c = Config()
        noise = np.random.default_rng(41).normal(0, .01, c.sr)
        unchanged, boundary = detect_endpoints(noise, c)
        self.assertEqual(boundary['status'], 'no_confident_speech')
        np.testing.assert_array_equal(unchanged, noise)
        short = np.ones(80)
        unchanged, boundary = detect_endpoints(short, c)
        self.assertEqual(boundary['status'], 'too_short_for_noise_estimate')
        np.testing.assert_array_equal(unchanged, short)

    def test_legacy_template_endpoint_configuration(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'old.npz'
            refs = {f'{label}_0': np.ones((2, 13)) for label in LABELS}
            meta = {'config': {'margin_ms': 50, 'top_db': 35}, 'data_source': 'phone_recordings',
                    'counts': {label: 1 for label in LABELS}}
            refs['metadata'] = np.array(json.dumps(meta))
            np.savez_compressed(p, **refs)
            _, config, _ = load_templates(p)
            self.assertEqual(config.endpoint_method, 'relative')
            self.assertEqual(config.margin_ms, 50)

    def test_feature_dimensions_and_cmn(self):
        y = np.random.default_rng(1).normal(0, .1, 3200)
        for config, dims in ((Config(), 13), (Config(delta=True), 26)):
            x = mfcc_feature(y, config)
            self.assertEqual(x.shape[1], dims)
            self.assertTrue(np.isfinite(x).all())
            np.testing.assert_allclose(x[:, :13].mean(axis=0), 0, atol=1e-10)
        self.assertEqual(mfcc_feature(np.ones(80), Config(delta=True)).shape, (1, 26))

    def test_resample_stereo_and_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            sf.write(p / 'stereo.wav', np.full((8000, 2), .25), 8000, subtype='PCM_16')
            y = load_audio(p / 'stereo.wav')
            self.assertEqual(y.shape, (16000,))
            self.assertTrue(np.isfinite(y).all())
            templates = {label: [np.arange(26).reshape(2, 13)] for label in LABELS}
            save_templates(p / 'templates.npz', templates, Config(), 'test')
            restored, config, source = load_templates(p / 'templates.npz')
            self.assertEqual(config, Config())
            self.assertEqual(source, 'test')
            for label in LABELS:
                np.testing.assert_array_equal(restored[label][0], templates[label][0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
