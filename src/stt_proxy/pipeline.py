"""One voice interaction as a single cancellable coroutine."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Protocol

import numpy as np

from .audio.broadcast import AudioBroadcaster
from .audio.earcons import Earcons
from .audio.sink import AudioSink
from .config import Config
from .openhab import OpenHABClient
from .recorder import NoSpeechError, record_utterance
from .state import Event, State
from .stt import Transcriber
from .vad import SpeechEndpointer

log = logging.getLogger(__name__)


class SpeakerProtocol(Protocol):
    async def speak(self, text: str, language: str) -> None: ...


class Pipeline:
    """LISTENING -> THINKING -> SPEAKING for one interaction.

    The whole run is one asyncio task; barge-in cancels it and the `finally`
    blocks release the recorder subscription and abort playback.
    """

    def __init__(
        self,
        config: Config,
        broadcaster: AudioBroadcaster,
        endpointer: SpeechEndpointer,
        transcriber: Transcriber,
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

    async def run_interaction(self, play_wake_earcon: bool = True) -> Event:
        """Returns the terminal event (PLAYBACK_DONE / NO_SPEECH / ERROR)."""
        try:
            # LISTENING
            self._set_state(State.LISTENING)
            if play_wake_earcon:
                await self._earcons.play("wake")
            frames = self._broadcaster.subscribe()
            try:
                pcm = await record_utterance(frames, self._endpointer, self._config.vad)
            except NoSpeechError:
                log.info("no speech detected, back to idle")
                return Event.NO_SPEECH
            finally:
                self._broadcaster.unsubscribe(frames)

            # THINKING
            self._set_state(State.THINKING)
            await self._earcons.play("ack")
            transcript = await self._transcriber.transcribe(pcm)
            if not transcript.text:
                log.info("empty transcript, back to idle")
                return Event.NO_SPEECH
            log.info("heard [%s]: %s", transcript.language, transcript.text)

            response = await self._openhab.send_command(transcript.text)
            log.info("openHAB answered: %s", response[:200])

            # SPEAKING
            self._set_state(State.SPEAKING)
            await self._speaker.speak(response, transcript.language)
            return Event.PLAYBACK_DONE
        except TimeoutError:
            log.error("openHAB response timed out")
            await self._earcons.play("error")
            return Event.ERROR
        except Exception:
            log.exception("pipeline failed")
            await self._earcons.play("error")
            return Event.ERROR
