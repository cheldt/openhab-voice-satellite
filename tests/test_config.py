import pytest

from openhab_voice_satellite.config import Config, load_config


def test_load_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
openhab:
  url: "http://oh:8080/"
  llm_tools: null
wakeword:
  threshold: 0.6
"""
    )
    config = load_config(path)
    assert config.openhab.url == "http://oh:8080"  # trailing slash stripped
    assert config.openhab.llm_tools is None
    assert config.wakeword.threshold == 0.6


def test_sample_rate_is_not_configurable():
    # the whole stack (VAD, wakeword, whisper) is hardwired to 16 kHz;
    # an old config setting the key must load and be ignored
    config = Config.model_validate({"audio": {"sample_rate": 48000}})
    assert config.audio.sample_rate == 16000


def test_env_token_wins(monkeypatch):
    config = Config.model_validate({"openhab": {"api_token": "file-token"}})
    assert config.openhab.token == "file-token"
    monkeypatch.setenv("OPENHAB_TOKEN", "env-token")
    assert config.openhab.token == "env-token"


def test_engine_defaults_local():
    # the engine selection is a decision, not a literal: local by default
    config = Config()
    assert config.stt.engine == "local"
    assert config.tts.engine == "kokoro"


def test_gemini_env_key_wins(monkeypatch):
    config = Config.model_validate({"gemini": {"api_key": "file-key"}})
    assert config.gemini.key == "file-key"
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert config.gemini.key == "env-key"


def test_gemini_engine_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Config.model_validate({"stt": {"engine": "gemini"}})
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Config.model_validate({"tts": {"engine": "gemini"}})
    # key present (either source) -> valid
    Config.model_validate({"stt": {"engine": "gemini"}, "gemini": {"api_key": "k"}})
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    Config.model_validate({"tts": {"engine": "gemini"}})


def test_deepgram_env_key_wins(monkeypatch):
    config = Config.model_validate({"deepgram": {"api_key": "file-key"}})
    assert config.deepgram.key == "file-key"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "env-key")
    assert config.deepgram.key == "env-key"


def test_deepgram_engine_requires_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        Config.model_validate({"stt": {"engine": "deepgram"}})
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        Config.model_validate({"tts": {"engine": "deepgram"}})
    # key present (either source) -> valid
    Config.model_validate({"stt": {"engine": "deepgram"}, "deepgram": {"api_key": "k"}})
    monkeypatch.setenv("DEEPGRAM_API_KEY", "env-key")
    Config.model_validate({"tts": {"engine": "deepgram"}})


def test_default_language_must_have_kokoro_voice():
    en_only = {
        "voices": {
            "en": {
                "model": "x.onnx",
                "voices": "v.bin",
                "voice": "bf_emma",
                "lang": "en-gb",
            }
        }
    }
    with pytest.raises(ValueError, match="kokoro voice"):
        Config.model_validate({"tts": {"default_language": "de"}, "kokoro": en_only})
    # also enforced for cloud engines (kokoro is their fallback)
    with pytest.raises(ValueError, match="kokoro voice"):
        Config.model_validate(
            {
                "tts": {"engine": "deepgram", "default_language": "de"},
                "kokoro": en_only,
                "deepgram": {"api_key": "k"},
            }
        )
    # engine piper does not require kokoro coverage
    Config.model_validate(
        {"tts": {"engine": "piper", "default_language": "de"}, "kokoro": en_only}
    )


def test_piper_engine_accepted():
    config = Config.model_validate({"tts": {"engine": "piper"}})
    assert config.tts.engine == "piper"


def test_tts_engine_local_no_longer_valid():
    with pytest.raises(ValueError):
        Config.model_validate({"tts": {"engine": "local"}})


def test_piper_default_language_must_have_voice():
    with pytest.raises(ValueError, match="piper voice"):
        Config.model_validate(
            {
                "tts": {"engine": "piper"},
                "piper": {"voices": {"en": "models/piper/en_GB-alba-medium.onnx"}},
            }
        )
    # not enforced when piper is not the engine
    Config.model_validate(
        {"piper": {"voices": {"en": "models/piper/en_GB-alba-medium.onnx"}}}
    )


def test_empty_languages_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"stt": {"languages": []}})


def test_zero_followup_timeout_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"dialog": {"followup_timeout_s": 0}})
