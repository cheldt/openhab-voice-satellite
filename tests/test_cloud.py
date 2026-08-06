from openhab_voice_satellite.cloud import pick_voice

VOICES = {"de": "viktoria", "en": "thalia"}


def test_pick_voice_prefers_language():
    assert pick_voice(VOICES, "en", "de") == "thalia"


def test_pick_voice_unknown_language_uses_default():
    assert pick_voice(VOICES, "fr", "de") == "viktoria"


def test_pick_voice_empty_map_returns_none():
    assert pick_voice({}, "de", "de") is None
