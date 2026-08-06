"""Cancellable PCM playback: a persistent, clock-free FIFO into pulsesink.

The pipeline is created once and stays PLAYING for the process lifetime, and
audio is treated as a plain byte FIFO: `sync=false`, no buffer timestamps.
The PulseAudio ring buffer paces consumption; the appsrc queue level (bytes)
is the only scheduling state. `play()` completion is byte-accounted: it
returns once cumulative pushed bytes minus the queue level shows the sound
has left appsrc, plus a short residual for the pulse ring — not after a
computed worst-case sleep. Clock-based scheduling (sync=true + PTS) was
abandoned after repeated stack-specific failures: PipeWire drains a freshly
connected stream before it is linked (per-play pipelines lose 150-800 ms),
pipewiresink's provided clock freezes on idle, gst pipewiresink on PipeWire
1.2.x wedges persistent streams and poisons the process-shared connection
(taking the capture stream down), and PTS timelines drift into silence.

Between sounds the sink streams keepalive dither — random noise at a few
LSB (~-78 dBFS), topped up whenever the queue level falls below a low-water
mark. An idle stream lets the session manager suspend the downstream chain
(echo-cancel nodes, USB DAC), and hardware watches content too: speaker
amps auto-standby on sustained digital zeros and mute their first few
hundred ms of wake-up, clipping the first sound after idle. Inaudible
nonzero samples keep every stage — stream, graph, and amplifier — awake.

Output goes through pulsesink (pipewire-pulse), not pipewiresink — the pulse
layer reaches the same graph nodes (pulse sink names equal node names) and
is hardened for long-lived streams. Capture stays on pipewiresrc.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np

from .gst_common import CLIENT_NAME, gst_init, install_sync_handler, s16_mono_caps
from .gst_devices import resolve_node

log = logging.getLogger(__name__)

PUSH_CHUNK_MS = 100
KEEPALIVE_TICK_S = 0.1  # queue-level check interval between sounds
KEEPALIVE_LOW_WATER_S = 0.2  # top up dither when less than this is queued
# keep-alive dither amplitude in LSB: ~-78 dBFS, inaudible, but nonzero so
# content-based silence detectors (amp auto-standby, AEC gating) stay awake
KEEPALIVE_DITHER = 4
PREAMBLE_PEAK = 4000  # ramp target; loud enough to register as signal
PLAYOUT_POLL_S = 0.05  # appsrc level poll cadence while a sound drains
# audio that has left appsrc but not the speaker: the pulsesink ring
# (buffer-time default ~200 ms, kept saturated by the keepalive FIFO)
# plus one in-flight PUSH_CHUNK_MS buffer
SINK_RESIDUAL_S = 0.3
# appsrc level not draining without a bus error (e.g. dead device):
# give up backlog + duration + this much later instead of hanging
PLAYOUT_FAILSAFE_EXTRA_S = 2.0


class PipewireSink:
    """Plays mono int16 buffers through one persistent appsrc pipeline.

    `duck()` scales output via the volume element (barge-in confirmation
    phase). `stop()` flushes pending audio immediately from any thread.
    A `play()` that interrupts a still-playing sound flushes it; otherwise
    the new sound is appended behind the queued keepalive silence.
    """

    def __init__(
        self,
        device: str | None = None,
        wakeup_preamble_ms: int = 0,
        wakeup_preamble_idle_s: float = 60.0,
    ) -> None:
        Gst = gst_init()
        self._Gst = Gst
        self._target = resolve_node(device, "output")
        self.target = self._target  # resolved node name; None = default sink
        self._preamble_ms = wakeup_preamble_ms
        self._preamble_idle_s = wakeup_preamble_idle_s
        self._gain = 1.0
        self._lock = threading.Lock()  # guards _errors/_waiter across threads
        self._play_lock = asyncio.Lock()
        self._errors: list[str] = []
        self._waiter: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None
        self._rate = 16000  # caps rate currently set on appsrc
        # event-loop time when the last queued sound ends; None = never played
        self._sound_until: float | None = None
        # cumulative bytes ever pushed into appsrc (utterances, preamble,
        # keepalive dither); only mutated on the event-loop thread
        self._pushed_bytes = 0

        self._pipeline = Gst.parse_launch(self._describe(self._target))
        self._src = self._pipeline.get_by_name("src")
        self._volume = self._pipeline.get_by_name("vol")
        install_sync_handler(Gst, self._pipeline.get_bus(), self._on_error, self._on_eos)
        self._src.set_property("caps", Gst.Caps.from_string(s16_mono_caps(self._rate)))
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("playback pipeline failed to start (is PipeWire running?)")
        self._keepalive_task = asyncio.get_running_loop().create_task(
            self._keepalive(), name="sink-keepalive"
        )
        log.info("audio output open: node=%s", self._target or "default")

    @staticmethod
    def _describe(target: str | None) -> str:
        # is-live: reach PLAYING without preroll; sync=false: pure FIFO, the
        # pulse ring buffer paces consumption; max-bytes=0: the appsrc queue
        # must hold whole utterances; buffer-time pins the pulse ring to the
        # documented default so SINK_RESIDUAL_S matches a known ring size
        t = f'device="{target}" ' if target else ""
        return (
            "appsrc name=src format=time is-live=true max-bytes=0 block=false "
            "! audioconvert ! audioresample ! volume name=vol "
            f"! pulsesink sync=false buffer-time=200000 client-name={CLIENT_NAME} {t}"
        )

    def _on_error(self, err, debug) -> None:
        log.error("playback pipeline error: %s (%s)", err.message, debug)
        with self._lock:
            self._errors.append(err.message)
            waiter = self._waiter
        if waiter is not None:
            loop, event = waiter
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass

    def _on_eos(self) -> None:
        # never sent by us; a spontaneous EOS means the stream is gone
        log.warning("playback pipeline reached end of stream")

    def duck(self, factor: float) -> None:
        self._gain = factor
        self._volume.set_property("volume", factor)

    def unduck(self) -> None:
        self.duck(1.0)

    def _flush(self) -> None:
        self._src.send_event(self._Gst.Event.new_flush_start())
        self._src.send_event(self._Gst.Event.new_flush_stop(False))

    def stop(self) -> None:
        """Drop pending audio and release the current play() waiter.

        _sound_until is left as scheduled: the amp was just playing, so the
        next sound must not get a wake-up preamble.
        """
        self._flush()
        with self._lock:
            waiter = self._waiter
        if waiter is not None:
            loop, event = waiter
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass

    def _push(self, samples: bytes) -> None:
        # the increment must stay in this single choke point so playout
        # accounting can never miss a push site
        self._pushed_bytes += len(samples)
        self._src.emit("push-buffer", self._Gst.Buffer.new_wrapped(samples))

    def _queued_bytes(self) -> int:
        return self._src.get_property("current-level-bytes")

    async def _keepalive(self) -> None:
        """Keep low-level dither queued so the output chain never idles.

        An idle stream lets PipeWire suspend the downstream nodes/device,
        and pure digital zeros trigger amplifier auto-standby; both mute
        the first few hundred ms of the next sound after an idle period.
        """
        while True:
            await asyncio.sleep(KEEPALIVE_TICK_S)
            async with self._play_lock:
                with self._lock:
                    if self._errors:
                        return
                low_water = int(KEEPALIVE_LOW_WATER_S * self._rate) * 2
                queued = self._queued_bytes()
                while queued < low_water:
                    n = int(KEEPALIVE_TICK_S * self._rate)
                    dither = np.random.randint(
                        -KEEPALIVE_DITHER, KEEPALIVE_DITHER + 1, n, dtype=np.int16
                    )
                    self._push(dither.tobytes())
                    queued += 2 * n

    def _prepare(self, pcm: np.ndarray, sample_rate: int, now: float) -> np.ndarray:
        """Flush an interrupted sound, or prepend the wake-up preamble after idle."""
        if self._sound_until is not None and now < self._sound_until:
            # interrupting a playing sound: drop the queue and restart
            self._flush()
        elif self._preamble_ms and (
            self._sound_until is None
            or now - self._sound_until >= self._preamble_idle_s
        ):
            # a powered speaker in auto-standby needs real signal for a
            # beat before it plays anything; give it a soft ramped hiss
            n = sample_rate * self._preamble_ms // 1000
            ramp = np.linspace(0.0, 1.0, n) ** 2
            noise = np.random.randint(-PREAMBLE_PEAK, PREAMBLE_PEAK + 1, n)
            pcm = np.concatenate([(noise * ramp).astype(np.int16), pcm])
        return pcm

    def _retune(self, sample_rate: int) -> None:
        if sample_rate != self._rate:
            self._rate = sample_rate
            self._src.set_property(
                "caps", self._Gst.Caps.from_string(s16_mono_caps(sample_rate))
            )
        self._volume.set_property("volume", self._gain)

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        if not len(pcm):
            return
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        async with self._play_lock:
            with self._lock:
                if self._errors:
                    raise RuntimeError(f"playback pipeline error: {self._errors[0]}")
                if self._waiter is not None:
                    # release the play we are replacing; its audio is flushed
                    self._waiter[1].set()
                self._waiter = (loop, done)
            pcm = self._prepare(pcm, sample_rate, loop.time())
            # queued bytes still play out at the pre-retune rate
            backlog_s = self._queued_bytes() / (self._rate * 2)
            self._retune(sample_rate)

            chunk_samples = sample_rate * PUSH_CHUNK_MS // 1000
            for start in range(0, len(pcm), chunk_samples):
                self._push(pcm[start:start + chunk_samples].tobytes())
            target = self._pushed_bytes  # everything up to and incl. this sound
            deadline = (
                loop.time()
                + backlog_s + len(pcm) / sample_rate
                + PLAYOUT_FAILSAFE_EXTRA_S
            )
            self._sound_until = loop.time() + backlog_s + len(pcm) / sample_rate

        try:
            await self._await_playout(done, target, deadline)
        except asyncio.CancelledError:
            self._flush()
            raise
        finally:
            with self._lock:
                if self._waiter is not None and self._waiter[1] is done:
                    self._waiter = None
        with self._lock:
            if self._errors:
                raise RuntimeError(f"playback pipeline error: {self._errors[0]}")

    async def _await_playout(
        self, done: asyncio.Event, target: int, deadline: float
    ) -> None:
        """Return once this play's last byte has left appsrc and the sink
        ring has had time to play it; early on stop()/replacement/error.

        Keepalive refills raise pushed and level equally, so they cannot move
        the watermark. Flushed bytes count as dequeued — correct, they never
        play — and every flush touching a live waiter also sets its `done`;
        a future flush path that forgets would only cause an early return.
        """
        loop = asyncio.get_running_loop()
        while not done.is_set():
            if self._pushed_bytes - self._queued_bytes() >= target:
                break
            if loop.time() >= deadline:
                log.warning("playout wait hit failsafe deadline; appsrc level not draining")
                return
            await asyncio.sleep(PLAYOUT_POLL_S)
        try:
            # done fires early on stop() or pipeline error; otherwise let the
            # pulsesink ring (+ one in-flight chunk) finish playing out
            await asyncio.wait_for(done.wait(), timeout=SINK_RESIDUAL_S)
        except asyncio.TimeoutError:
            pass  # normal completion

    def close(self) -> None:
        self._keepalive_task.cancel()
        self._pipeline.set_state(self._Gst.State.NULL)
