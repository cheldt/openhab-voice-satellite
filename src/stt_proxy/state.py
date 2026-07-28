"""Pure state machine: states, events, transition table. No I/O here."""

from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    IDLE = auto()       # waiting for wakeword
    LISTENING = auto()  # recording utterance until VAD endpoint
    THINKING = auto()   # STT + openHAB roundtrip
    SPEAKING = auto()   # TTS playback


class Event(Enum):
    WAKE = auto()             # wakeword detected
    VAD_ENDPOINT = auto()     # utterance finished
    NO_SPEECH = auto()        # listening timed out without speech
    RESPONSE = auto()         # openHAB answered
    PLAYBACK_DONE = auto()    # TTS finished
    FOLLOW_UP = auto()        # dialog mode: mic re-opens after the answer
    STOP = auto()             # stop/barge-in trigger
    ERROR = auto()            # any pipeline failure


# (state, event) -> next state. STOP from SPEAKING is resolved by the caller
# via barge_in.resume_listening (IDLE or LISTENING); the table holds the IDLE default.
TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.IDLE, Event.WAKE): State.LISTENING,
    (State.LISTENING, Event.VAD_ENDPOINT): State.THINKING,
    (State.LISTENING, Event.NO_SPEECH): State.IDLE,
    (State.LISTENING, Event.STOP): State.IDLE,
    (State.LISTENING, Event.ERROR): State.IDLE,
    (State.THINKING, Event.RESPONSE): State.SPEAKING,
    (State.THINKING, Event.STOP): State.IDLE,
    (State.THINKING, Event.ERROR): State.IDLE,
    (State.SPEAKING, Event.PLAYBACK_DONE): State.IDLE,
    (State.SPEAKING, Event.FOLLOW_UP): State.LISTENING,
    (State.SPEAKING, Event.STOP): State.IDLE,
    (State.SPEAKING, Event.ERROR): State.IDLE,
}


def next_state(state: State, event: Event) -> State | None:
    """Return the follow-up state, or None if the event is ignored in this state."""
    return TRANSITIONS.get((state, event))
