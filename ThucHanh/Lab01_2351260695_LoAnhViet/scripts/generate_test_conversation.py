#!/usr/bin/env python3
"""Create a local TTS conversation with controlled sections for Lab 01."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import random
import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path


SAMPLE_RATE = 44_100
PART_SECONDS = 3.0
SILENCES = (1.2, 1.0, 1.0)
SPEAKERS = {
    "speaker_a": {"voice": "vi", "speed": 145, "pitch": 42},
    "speaker_b": {"voice": "vi-VN-x-south", "speed": 165, "pitch": 65},
}
SENTENCES = {
    "speaker_a": [
        "Chào bạn. Hôm nay chúng ta kiểm tra tín hiệu âm thanh.",
        "Mình sẽ đọc một câu để quan sát phổ tần số.",
        "Đây là đoạn nói dùng cho bài thực hành xử lý âm thanh.",
    ],
    "speaker_b": [
        "Được rồi. Hãy xem dạng sóng và năng lượng của tín hiệu.",
        "Sau đó chúng ta có thể so sánh FFT và spectrogram.",
        "Tôi đã sẵn sàng thử bộ lọc, lượng tử hóa và lấy mẫu lại.",
    ],
}


def clamp(value: float) -> int:
    return max(-32768, min(32767, round(value)))


def as_samples(raw: bytes) -> array:
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def as_bytes(samples: array) -> bytes:
    encoded = array("h", samples)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def write_wav(path: Path, samples: array, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(as_bytes(samples))


def read_wav(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise RuntimeError(f"TTS output must be mono PCM-16: {path}")
        return as_samples(wav_file.readframes(wav_file.getnframes())), wav_file.getframerate()


def resample_linear(samples: array, source_rate: int) -> array:
    """eSpeak is normally 22.05 kHz; this only upsamples it to 44.1 kHz."""
    if source_rate == SAMPLE_RATE:
        return array("h", samples)
    result = array("h")
    ratio = source_rate / SAMPLE_RATE
    for index in range(round(len(samples) * SAMPLE_RATE / source_rate)):
        position = index * ratio
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        result.append(clamp(samples[left] + fraction * (samples[right] - samples[left])))
    return result


def make_part(samples: array, peak: float, with_noise: bool, rng: random.Random) -> array:
    """Normalize, trim/pad to three seconds, then optionally add white noise."""
    current_peak = max((abs(value) for value in samples), default=1)
    gain = peak * 32767 / current_peak
    target_count = round(PART_SECONDS * SAMPLE_RATE)
    result = array("h", (clamp(value * gain) for value in samples[:target_count]))
    result.extend([0] * (target_count - len(result)))
    if with_noise:
        sigma = 0.035 * 32767  # white-noise RMS, relative to full scale
        result = array("h", (clamp(value + rng.gauss(0, sigma)) for value in result))
    return result


class LocalEspeak:
    """Use `espeak-ng` when present; otherwise use the installed shared library."""

    def __init__(self) -> None:
        self.command = shutil.which("espeak-ng")
        self.library = None
        self.callback = None  # Keep the C callback alive for the program lifetime.
        self.chunks: list[bytes] = []
        self.sample_rate = 0
        if not self.command:
            self._init_library()

    def _init_library(self) -> None:
        library_name = ctypes.util.find_library("espeak-ng") or "libespeak-ng.so.1"
        try:
            self.library = ctypes.CDLL(library_name)
        except OSError as error:
            raise RuntimeError("Không tìm thấy espeak-ng. Cài `espeak-ng` rồi chạy lại.") from error

        callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
        )

        def receive_audio(wav_samples: object, count: int, _events: object) -> int:
            if wav_samples and count > 0:
                self.chunks.append(ctypes.string_at(wav_samples, count * 2))
            return 0

        self.callback = callback_type(receive_audio)
        self.library.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.library.espeak_Initialize.restype = ctypes.c_int
        self.library.espeak_SetSynthCallback.argtypes = [callback_type]
        self.library.espeak_SetSynthCallback.restype = ctypes.c_int
        self.library.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self.library.espeak_SetVoiceByName.restype = ctypes.c_int
        self.library.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.library.espeak_Synth.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.library.espeak_Synth.restype = ctypes.c_int
        self.library.espeak_Synchronize.restype = ctypes.c_int
        self.sample_rate = self.library.espeak_Initialize(1, 500, None, 0)  # AUDIO_OUTPUT_RETRIEVAL
        if self.sample_rate <= 0 or self.library.espeak_SetSynthCallback(self.callback) != 0:
            raise RuntimeError("Không thể khởi tạo eSpeak NG.")

    def render(self, text: str, settings: dict[str, int | str], output: Path) -> None:
        if self.command:
            subprocess.run(
                [self.command, "-v", str(settings["voice"]), "-s", str(settings["speed"]),
                 "-p", str(settings["pitch"]), "-w", str(output), text],
                check=True,
                capture_output=True,
                text=True,
            )
            return

        assert self.library is not None
        self.chunks = []
        if self.library.espeak_SetVoiceByName(str(settings["voice"]).encode()) != 0:
            raise RuntimeError(f"Voice không khả dụng: {settings['voice']}")
        self.library.espeak_SetParameter(1, int(settings["speed"]), 0)  # espeakRATE
        self.library.espeak_SetParameter(3, int(settings["pitch"]), 0)  # espeakPITCH
        payload = text.encode("utf-8")
        result = self.library.espeak_Synth(payload, len(payload) + 1, 0, 1, 0, 1, None, None)
        if result != 0 or self.library.espeak_Synchronize() != 0 or not self.chunks:
            raise RuntimeError("eSpeak NG không thể tổng hợp câu nói.")
        write_wav(output, as_samples(b"".join(self.chunks)), self.sample_rate)


def build_audio(project_dir: Path, seed: int | None) -> Path:
    rng = random.Random(seed)
    audio_dir = project_dir / "audio"
    parts_dir = audio_dir / "generated_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    renderer = LocalEspeak()
    timeline = [
        ("01_normal_speaker_a", "speaker_a", 0.55, False),
        ("02_loud_speaker_b", "speaker_b", 0.85, False),
        ("03_quiet_speaker_a", "speaker_a", 0.20, False),
        ("04_speech_white_noise_speaker_b", "speaker_b", 0.55, True),
    ]
    conversation = array("h")
    manifest = []
    for index, (name, speaker, peak, with_noise) in enumerate(timeline):
        sentence = rng.choice(SENTENCES[speaker])
        temporary = parts_dir / f"{name}_tts.wav"
        output_part = parts_dir / f"{name}.wav"
        renderer.render(sentence, SPEAKERS[speaker], temporary)
        source, source_rate = read_wav(temporary)
        source = resample_linear(source, source_rate)
        part = make_part(source, peak, with_noise, rng)
        write_wav(output_part, part)
        temporary.unlink()
        conversation.extend(part)
        manifest.append({"file": output_part.name, "speaker": speaker, "text": sentence,
                         "duration_seconds": PART_SECONDS, "white_noise": with_noise})
        if index < len(SILENCES):
            silence_seconds = SILENCES[index]
            conversation.extend([0] * round(silence_seconds * SAMPLE_RATE))
            manifest.append({"silence_seconds": silence_seconds})

    output = audio_dir / "generated_conversation.wav"
    write_wav(output, conversation)
    (parts_dir / "manifest.json").write_text(
        json.dumps({"seed": seed, "sample_rate_hz": SAMPLE_RATE, "channels": 1,
                    "bit_depth": 16, "timeline": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def print_metadata(path: Path) -> None:
    with wave.open(str(path), "rb") as wav_file:
        print(f"duration: {wav_file.getnframes() / wav_file.getframerate():.3f} s")
        print(f"sample rate: {wav_file.getframerate()} Hz")
        print(f"channels: {wav_file.getnchannels()}")
        print(f"bit depth: {wav_file.getsampwidth() * 8}-bit PCM")
    print(f"output: {path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Seed để tái tạo đúng câu thoại đã chọn.")
    args = parser.parse_args()
    output_path = build_audio(Path(__file__).resolve().parents[1], args.seed)
    print_metadata(output_path)
