from openhab_voice_satellite.config import Config
from openhab_voice_satellite.selftest import select_checks


def _names(config: Config) -> list[str]:
    return [name for name, _ in select_checks(config)]


def test_local_config_selects_base_checks():
    assert _names(Config()) == [
        "audio devices",
        "wakeword model",
        "vad model",
        "whisper model (incl. warmup)",
        "piper voices",
        "openHAB REST",
    ]


def test_gemini_engine_appends_api_check_and_keeps_piper():
    config = Config.model_validate(
        {"stt": {"engine": "gemini"}, "gemini": {"api_key": "k"}}
    )
    names = _names(config)
    assert "gemini API" in names
    assert "piper voices" in names  # cloud engines fall back to piper
    assert "deepgram API" not in names


def test_both_cloud_engines_append_both_checks():
    config = Config.model_validate(
        {
            "stt": {"engine": "gemini"},
            "tts": {"engine": "deepgram"},
            "gemini": {"api_key": "k"},
            "deepgram": {"api_key": "k"},
        }
    )
    names = _names(config)
    assert "gemini API" in names
    assert "deepgram API" in names
