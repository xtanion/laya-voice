"""Offline contracts for the ESP32 push-to-talk receiver. No Bluetooth hardware."""

import struct

import numpy as np

from laya_voice import config
from laya_voice.remote import AUDIO, INDEX_SHIFT, MAX_FRAME_BYTES, START, STEPS, STOP, PacketStream, Recording, adpcm_decode, prepare


def tone(seconds: float, amp: float = 0.01) -> np.ndarray:
    t = np.arange(int(seconds * config.SAMPLE_RATE)) / config.SAMPLE_RATE
    return (np.sin(2 * np.pi * 440 * t) * amp * 32767).astype(np.int16)


def adpcm_encode(samples: np.ndarray, predictor: int, index: int) -> tuple[bytes, int, int]:
    codes = []
    for s in samples.astype(int):
        step = STEPS[index]
        diff = s - predictor
        code = 8 if diff < 0 else 0
        diff = abs(diff)
        delta = step >> 3
        if diff >= step:
            code |= 4; diff -= step; delta += step  # noqa: E702
        if diff >= step >> 1:
            code |= 2; diff -= step >> 1; delta += step >> 1  # noqa: E702
        if diff >= step >> 2:
            code |= 1; delta += step >> 2  # noqa: E702
        predictor = max(-32768, min(32767, predictor - delta if code & 8 else predictor + delta))
        index = max(0, min(88, index + INDEX_SHIFT[code & 7]))
        codes.append(code)
    out = bytes(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2))
    return out, predictor, index


def packets(samples: np.ndarray, per_packet: int = 320) -> list[bytes]:
    out = [bytes([START, 0, 0])]
    pred, idx, seq = 0, 0, 0
    for i in range(0, len(samples), per_packet):
        seq += 1
        data, npred, nidx = adpcm_encode(samples[i:i + per_packet], pred, idx)
        out.append(struct.pack("<BHhB", AUDIO, seq, pred, idx) + data)
        pred, idx = npred, nidx
    out.append(struct.pack("<BH", STOP, seq + 1))
    return out


class Sink:
    def __init__(self, max_seconds: float = 5) -> None:
        self.presses, self.audio, self.events = 0, [], []
        self.stream = PacketStream(self.press, self.audio.append, self.events.append, max_seconds)

    def press(self) -> None:
        self.presses += 1

    def feed(self, pkts: list[bytes]) -> None:
        for p in pkts:
            self.stream.handle(p)


def test_adpcm_roundtrip_tracks_signal():
    x = tone(0.1, amp=0.3)
    data, _, _ = adpcm_encode(x, 0, 0)
    y = np.frombuffer(adpcm_decode(data, 0, 0), dtype="<i2")
    assert len(y) == len(x)
    assert np.corrcoef(x[200:], y[200:])[0, 1] > 0.98


def test_recording_rejects_audio_outside_session():
    rec = Recording(5)
    assert rec.feed(b"\x00\x00") == "audio outside a recording"
    assert rec.stop() is None


def test_recording_drops_malformed_and_oversized_frames():
    rec = Recording(5)
    rec.start()
    assert rec.feed(b"\x00") == "malformed audio frame"
    assert rec.stop() is None
    rec.start()
    assert rec.feed(b"\x00" * (MAX_FRAME_BYTES + 2)) == "malformed audio frame"
    assert not rec.active


def test_prepare_normalizes_quiet_audio_and_trims_tail():
    pcm = prepare(tone(1.0, amp=0.01).astype("<i2").tobytes())
    assert pcm.dtype == np.float32
    assert len(pcm) == config.SAMPLE_RATE - config.SAMPLE_RATE * 80 // 1000
    assert 0.25 < float(np.max(np.abs(pcm))) <= 0.71


def test_prepare_does_not_amplify_silence():
    assert float(np.max(np.abs(prepare(b"\x00\x00" * 8000)))) == 0.0


def test_start_audio_stop_delivers_one_recording():
    sink = Sink()
    sink.feed(packets(tone(0.5, amp=0.2)))
    assert sink.presses == 1
    assert len(sink.audio) == 1
    assert len(sink.audio[0]) == 8000 - 1280


def test_sequence_gap_drops_the_recording():
    pkts = packets(tone(0.5, amp=0.2))
    del pkts[3]
    sink = Sink()
    sink.feed(pkts)
    assert sink.audio == []
    assert any("lost audio packet" in e for e in sink.events)


def test_disconnect_mid_recording_delivers_nothing():
    sink = Sink()
    sink.feed(packets(tone(0.5, amp=0.2))[:-1])
    sink.stream.disconnected()
    sink.stream.handle(struct.pack("<BH", STOP, 26))
    assert sink.audio == []


def test_bad_step_index_and_unknown_type_drop_the_recording():
    for bad in (struct.pack("<BHhB", AUDIO, 1, 0, 89) + b"\x00", struct.pack("<BH", 0x7F, 1)):
        sink = Sink()
        sink.feed([bytes([START, 0, 0]), bad, struct.pack("<BH", STOP, 2)])
        assert sink.audio == []


def test_audio_without_start_is_ignored():
    sink = Sink()
    sink.feed(packets(tone(0.2))[1:])
    assert sink.presses == 0 and sink.audio == []


def test_over_long_recording_is_dropped():
    sink = Sink(max_seconds=0.2)
    sink.feed(packets(tone(0.5)))
    assert sink.audio == []
    assert any("too long" in e for e in sink.events)
