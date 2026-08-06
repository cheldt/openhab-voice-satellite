"""One voice interaction as a single cancellable coroutine."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .audio.broadcast import AudioBroadcaster
from .audio.earcons import Earcons
from .audio.sink import AudioSink
from .config import Config
from .openhab import OpenHABClient
from .recorder import NoSpeechError, record_utterance
from .state import Event, State
from .stt import Transcript
from .vad import SpeechEndpointer

log = logging.getLogger(__name__)


class SpeakerProtocol(Protocol):
    async def speak(self, text: str, language: str) -> None: ...


class TranscriberProtocol(Protocol):
    async def transcribe(self, pcm: np.ndarray) -> Transcript: ...


class Pipeline:
    """LISTENING -> THINKING -> SPEAKING for one interaction.

    With dialog mode enabled the whole interaction is one server-side
    conversation: a uuid is generated per wake word and sent with every
    interpreter request, and after each answer SPEAKING loops back to
    LISTENING (no wakeword). The conversation ends when no follow-up
    arrives within `dialog.followup_timeout_s` or the task is cancelled
    (barge-in); either way the server-side conversation is deleted.

    The whole run is one asyncio task; barge-in cancels it and the `finally`
    blocks release the recorder subscription and abort playback.
    """

    def __init__(
        self,
        config: Config,
        broadcaster: AudioBroadcaster,
        endpointer: SpeechEndpointer,
        transcriber: TranscriberProtocol,
        openhab: OpenHABClient,
        speaker: SpeakerProtocol,
        sink: AudioSink,
        earcons: Earcons,
        set_state: Callable[[State], None],
    ) -> None:
        self._config = config
        self._broadcaster = broadcaster
        self._endpointer = endpointer
        self._transcriber = transcriber
        self._openhab = openhab
        self._speaker = speaker
        self._sink = sink
        self._earcons = earcons
        self._set_state = set_state
        self._cleanup_tasks: set[asyncio.Task] = set()

    async def run_interaction(self, play_wake_earcon: bool = True) -> Event:
        """Returns the terminal event (PLAYBACK_DONE / NO_SPEECH / ERROR)."""
        dialog = self._config.dialog
        conversation_id = str(uuid.uuid4()) if dialog.enabled else None
        conversation_started = False
        language: str | None = None
        round_no = 0
        try:
            while True:
                # LISTENING
                self._set_state(State.LISTENING)
                if round_no > 0:
                    await self._earcons.play(dialog.earcon)
                    timeout = dialog.followup_timeout_s
                else:
                    if play_wake_earcon:
                        await self._earcons.play("wake")
                    timeout = None  # first round uses vad.no_speech_timeout_s

                transcript = await self._capture_utterance(no_speech_timeout_s=timeout)
                if transcript is None:
                    if round_no == 0:
                        return Event.NO_SPEECH
                    log.info(
                        "no follow-up within %.1fs, conversation over",
                        dialog.followup_timeout_s,
                    )
                    return Event.PLAYBACK_DONE
                # Lock TTS to the first round's language; follow-ups are often
                # too short for reliable language detection.
                language = language or transcript.language

                # Set before the await: a cancel mid-POST must still delete
                # the conversation the server may have created.
                conversation_started = conversation_id is not None
                response = await self._openhab.send_command(transcript.text, conversation_id)
                log.info("openHAB answered: %s", response[:200])

                # SPEAKING
                self._set_state(State.SPEAKING)
                await self._speaker.speak(response, language)

                if not dialog.enabled:
                    return Event.PLAYBACK_DONE
                round_no += 1
        except TimeoutError:
            log.error("openHAB response timed out")
            await self._earcons.play("error")
            return Event.ERROR
        except Exception:
            log.exception("pipeline failed")
            await self._earcons.play("error")
            return Event.ERROR
        finally:
            if conversation_started:
                self._schedule_conversation_end(conversation_id)

    def _dump_utterance(self, pcm: np.ndarray) -> None:
        """Write the recorded utterance to $OVS_DUMP_UTTERANCES for field debugging."""
        dump_dir = os.environ.get("OVS_DUMP_UTTERANCES")
        if not dump_dir:
            return
        path = Path(dump_dir) / f"utterance-{time.strftime('%H%M%S')}.wav"
        try:
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self._config.audio.sample_rate)
                f.writeframes(pcm.tobytes())
            log.info("utterance dumped: %s", path)
        except OSError:
            log.exception("utterance dump failed")

    def _schedule_conversation_end(self, conversation_id: str) -> None:
        """Fire-and-forget server-side conversation DELETE.

        Runs as its own task so a cancelled pipeline (barge-in) neither
        blocks on the round-trip nor loses the cleanup.
        """
        task = asyncio.create_task(
            self._openhab.end_conversation(conversation_id),
            name=f"end-conversation-{conversation_id[:8]}",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _capture_utterance(
        self, no_speech_timeout_s: float | None = None
    ) -> Transcript | None:
        """LISTENING record + THINKING transcribe. None = no/empty speech.

        The caller sets State.LISTENING and plays the entry earcon.
        """
        frames = self._broadcaster.subscribe()
        try:
            pcm = await record_utterance(
                frames,
                self._endpointer,
                self._config.vad,
                no_speech_timeout_s=no_speech_timeout_s,
            )
        except NoSpeechError:
            log.info("no speech detected, back to idle")
            return None
        finally:
            self._broadcaster.unsubscribe(frames)

        rms = int(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if len(pcm) else 0
        log.info(
            "utterance captured: %.1fs rms=%d",
            len(pcm) / self._config.audio.sample_rate, rms,
        )
        self._dump_utterance(pcm)

        self._set_state(State.THINKING)
        await self._earcons.play("ack")
        transcript = await self._transcriber.transcribe(pcm)
        if not transcript.text:
            log.info("empty transcript, back to idle")
            return None
        log.info("heard [%s]: %s", transcript.language, transcript.text)
        return transcript
