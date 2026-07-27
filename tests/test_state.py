from stt_proxy.state import TRANSITIONS, Event, State, next_state


def test_happy_path():
    state = State.IDLE
    for event, expected in [
        (Event.WAKE, State.LISTENING),
        (Event.VAD_ENDPOINT, State.THINKING),
        (Event.RESPONSE, State.SPEAKING),
        (Event.PLAYBACK_DONE, State.IDLE),
    ]:
        state = next_state(state, event)
        assert state is expected


def test_follow_up_reopens_listening():
    assert next_state(State.SPEAKING, Event.FOLLOW_UP) is State.LISTENING


def test_dialog_round_trip():
    state = State.SPEAKING
    for event, expected in [
        (Event.FOLLOW_UP, State.LISTENING),
        (Event.VAD_ENDPOINT, State.THINKING),
        (Event.RESPONSE, State.SPEAKING),
        (Event.PLAYBACK_DONE, State.IDLE),
    ]:
        state = next_state(state, event)
        assert state is expected


def test_stop_from_every_active_state():
    for state in (State.LISTENING, State.THINKING, State.SPEAKING):
        assert next_state(state, Event.STOP) is State.IDLE


def test_error_from_every_active_state():
    for state in (State.LISTENING, State.THINKING, State.SPEAKING):
        assert next_state(state, Event.ERROR) is State.IDLE


def test_no_speech_timeout():
    assert next_state(State.LISTENING, Event.NO_SPEECH) is State.IDLE


def test_unknown_transitions_ignored():
    assert next_state(State.IDLE, Event.STOP) is None
    assert next_state(State.IDLE, Event.RESPONSE) is None
    assert next_state(State.SPEAKING, Event.WAKE) is None


def test_all_transitions_lead_to_valid_states():
    for (state, event), target in TRANSITIONS.items():
        assert isinstance(state, State)
        assert isinstance(event, Event)
        assert isinstance(target, State)
