"""States and terminal pipeline events. No I/O here."""

from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    IDLE = auto()       # waiting for wakeword
    LISTENING = auto()  # recording utterance until VAD endpoint
    THINKING = auto()   # STT + openHAB roundtrip
    SPEAKING = auto()   # TTS playback


class Event(Enum):
    """Terminal outcome of one interaction (Pipeline.run_interaction)."""

    NO_SPEECH = auto()        # listening timed out without speech
    PLAYBACK_DONE = auto()    # answer spoken (or dialog ended without follow-up)
    ERROR = auto()            # any pipeline failure
