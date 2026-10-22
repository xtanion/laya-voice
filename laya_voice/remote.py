"""Wireless push-to-talk module (ESP32 + INMP441) over Bluetooth LE.

The module notifies one characteristic with packets; byte 0 is the type, bytes 1-2 a little-endian
sequence number that restarts at 0 on every start:
    0x01 start   button pressed
    0x02 audio   int16 predictor, uint8 step index, then IMA-ADPCM nibbles (low nibble first), 16 kHz mono
    0x03 stop    button released
A recording with a gap in the sequence, a malformed packet, or one cut off by a disconnect is
dropped, never transcribed. The module only streams over an encrypted (bonded) link.
"""
from __future__ import annotations

import asyncio
import struct
import threading
from typing import Callable

import numpy as np

from . import config

SERVICE_UUID = "3e1c1517-cff3-47bc-bcb3-b0276cdbfdfd"
AUDIO_UUID = "3e1c1518-cff3-47bc-bcb3-b0276cdbfdfd"
DEVICE_NAME = "Laya Voice"

START, AUDIO, STOP = 0x01, 0x02, 0x03
MAX_FRAME_BYTES = 4096
TRIM_TAIL_MS = 80
TARGET_PEAK = 0.7
MAX_GAIN = 30.0
RETRY_SECONDS = 3.0

STEPS = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97,
    107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428,
    4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350,
    22385, 24623, 27086, 29794, 32767,
)
INDEX_SHIFT = (-1, -1, -1, -1, 2, 4, 6, 8)


def adpcm_decode(data: bytes, predictor: int, index: int) -> bytes:
    out = np.empty(len(data) * 2, dtype="<i2")
    i = 0
    for byte in data:
        for code in (byte & 0x0F, byte >> 4):
            step = STEPS[index]
            diff = step >> 3
            if code & 1:
                diff += step >> 2
            if code & 2:
                diff += step >> 1
            if code & 4:
                diff += step
            predictor = predictor - diff if code & 8 else predictor + diff
            predictor = max(-32768, min(32767, predictor))
            index = max(0, min(88, index + INDEX_SHIFT[code & 7]))
            out[i] = predictor
            i += 1
    return out.tobytes()


def prepare(raw: bytes) -> np.ndarray:
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    trim = config.SAMPLE_RATE * TRIM_TAIL_MS // 1000
    if len(x) > 2 * trim:
        x = x[:-trim]
    if not len(x):
        return x
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak > 1e-4:
        x = x * min(TARGET_PEAK / peak, MAX_GAIN)
    return x.astype(np.float32)


class Recording:
    def __init__(self, max_seconds: float) -> None:
        self.limit = int(max_seconds * config.SAMPLE_RATE)
        self.parts: list[bytes] = []
        self.samples = 0
        self.active = False

    def start(self) -> None:
        self.parts, self.samples, self.active = [], 0, True

    def abort(self) -> None:
        self.parts, self.samples, self.active = [], 0, False

    def feed(self, data: bytes) -> str | None:
        if not self.active:
            return "audio outside a recording"
        if len(data) > MAX_FRAME_BYTES or len(data) % 2:
            self.abort()
            return "malformed audio frame"
        n = len(data) // 2
        if self.samples + n > self.limit:
            self.abort()
            return "recording too long"
        self.parts.append(data)
        self.samples += n
        return None

    def stop(self) -> np.ndarray | None:
        if not self.active:
            return None
        raw = b"".join(self.parts)
        self.abort()
        return prepare(raw)


class PacketStream:
    """Turns module packets into on_press / on_audio calls. Transport-free so it can be tested."""

    def __init__(self, on_press: Callable[[], None], on_audio: Callable[[np.ndarray], None],
                 on_event: Callable[[str], None], max_seconds: float) -> None:
        self.on_press, self.on_audio, self.on_event = on_press, on_audio, on_event
        self.rec = Recording(max_seconds)
        self.seq = 0

    def _drop(self, reason: str) -> None:
        if self.rec.active:
            self.on_event(f"{reason}, recording dropped")
        self.rec.abort()

    def handle(self, packet: bytes) -> None:
        if len(packet) < 3:
            self._drop("short packet")
            return
        kind = packet[0]
        seq = packet[1] | packet[2] << 8
        if kind == START:
            if seq != 0:
                self._drop("bad start packet")
                return
            self._drop("no stop for the previous press")
            self.rec.start()
            self.seq = 0
            self.on_press()
            return
        if not self.rec.active:
            return
        if seq != (self.seq + 1) & 0xFFFF:
            self._drop("lost audio packet")
            return
        self.seq = seq
        if kind == AUDIO:
            if len(packet) < 7:
                self._drop("malformed audio packet")
                return
            predictor, index = struct.unpack_from("<hB", packet, 3)
            if index > 88:
                self._drop("malformed audio packet")
                return
            err = self.rec.feed(adpcm_decode(packet[6:], predictor, index))
            if err:
                self.on_event(f"{err}, recording dropped")
        elif kind == STOP:
            pcm = self.rec.stop()
            if pcm is not None:
                self.on_audio(pcm)
        else:
            self._drop("unknown packet")

    def disconnected(self) -> None:
        self._drop("module disconnected")


class BleReceiver:
    def __init__(self, on_press: Callable[[], None], on_audio: Callable[[np.ndarray], None],
                 on_event: Callable[[str], None] = print, device: str | None = None,
                 max_seconds: float | None = None) -> None:
        self.on_event = on_event
        self.device = device if device is not None else config.BLE_DEVICE
        self.stream = PacketStream(on_press, on_audio, on_event, max_seconds or config.REMOTE_MAX_SECONDS)
        self.stopping = threading.Event()

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True, name="ble-ptt").start()

    def stop(self) -> None:
        self.stopping.set()

    async def _find(self):  # noqa: ANN202
        from bleak import BleakScanner

        if self.device:
            return await BleakScanner.find_device_by_address(self.device, timeout=10.0)
        return await BleakScanner.find_device_by_filter(
            lambda d, ad: SERVICE_UUID in [u.lower() for u in ad.service_uuids] or ad.local_name == DEVICE_NAME,
            timeout=10.0,
        )

    async def _main(self) -> None:
        from bleak import BleakClient

        while not self.stopping.is_set():
            try:
                dev = await self._find()
                if dev is None:
                    await asyncio.sleep(RETRY_SECONDS)
                    continue
                gone = asyncio.Event()
                async with BleakClient(dev, disconnected_callback=lambda c: gone.set()) as client:
                    await client.start_notify(AUDIO_UUID, lambda ch, data: self.stream.handle(bytes(data)))
                    self.on_event(f"connected to {dev.name or DEVICE_NAME} ({dev.address})"
                                  + ("" if self.device else "; set BLE_DEVICE to this address to pin it"))
                    while not gone.is_set() and not self.stopping.is_set():
                        await asyncio.sleep(0.2)
                self.stream.disconnected()
                self.on_event("module disconnected")
            except Exception as e:  # noqa: BLE001
                self.stream.disconnected()
                self.on_event(f"bluetooth: {e}")
                await asyncio.sleep(RETRY_SECONDS)
