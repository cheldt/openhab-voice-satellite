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
from .stt import Transcriber, Transcript
from .vad import SpeechEndpointer

log = logging.getLogger(__name__)


def stitch_context(history: list[tuple[str, str]], answer: str) -> str:
    """Compose prior (user_text, question) pairs plus the new answer into one
    dialogue transcript so a stateless interpreter can disambiguate."""
    lines: list[str] = []
    for user_text, question in history:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {question}")
    lines.append(f"User: {answer}")
    return "\n".join(lines)


class SpeakerProtocol(Protocol):
    async def speak(self, text: str, language: str) -> None: ...


class Pipeline:
    """LISTENING -> THINKING -> SPEAKING for one interaction.

    When the openHAB answer contains a question and dialog mode is enabled,
    SPEAKING loops back to LISTENING (no wakeword) for up to
    `dialog.max_turns` follow-up rounds.

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
        dialog = self._config.dialog
        history: list[tuple[str, str]] = []  # (user_text, question) per round
        language: str | None = None
        max_rounds = 1 + (dialog.max_turns if dialog.enabled else 0)
        try:
            for round_no in range(max_rounds):
                # LISTENING
                self._set_state(State.LISTENING)
                if round_no > 0:
                    await self._earcons.play(dialog.earcon)
                elif play_wake_earcon:
                    await self._earcons.play("wake")

                transcript = await self._capture_utterance()
                if transcript is None:
                    return Event.NO_SPEECH
                # Lock TTS to the first round's language; follow-ups are often
                # too short for reliable language detection.
                language = language or transcript.language

                outgoing = (
                    stitch_context(history, transcript.text)
                    if history and dialog.context_mode == "stitch"
                    else transcript.text
                )
                response = await self._openhab.send_command(outgoing)
                log.info("openHAB answered: %s", response[:200])

                # SPEAKING
                self._set_state(State.SPEAKING)
                await self._speaker.speak(response, language)

                if not (dialog.enabled and "?" in response):
                    return Event.PLAYBACK_DONE
                if round_no == max_rounds - 1:
                    log.info("dialog turn limit reached, not re-listening")
                    return Event.PLAYBACK_DONE
                history.append((transcript.text, response))
            return Event.PLAYBACK_DONE
        except TimeoutError:
            log.error("openHAB response timed out")
            await self._earcons.play("error")
            return Event.ERROR
        except Exception:
            log.exception("pipeline failed")
            await self._earcons.play("error")
            return Event.ERROR

    async def _capture_utterance(self) -> Transcript | None:
        """LISTENING record + THINKING transcribe. None = no/empty speech.

        The caller sets State.LISTENING and plays the entry earcon.
        """
        frames = self._broadcaster.subscribe()
        try:
            pcm = await record_utterance(frames, self._endpointer, self._config.vad)
        except NoSpeechError:
            log.info("no speech detected, back to idle")
            return None
        finally:
            self._broadcaster.unsubscribe(frames)

        self._set_state(State.THINKING)
        await self._earcons.play("ack")
        transcript = await self._transcriber.transcribe(pcm)
        if not transcript.text:
            log.info("empty transcript, back to idle")
            return None
        log.info("heard [%s]: %s", transcript.language, transcript.text)
        return transcript
