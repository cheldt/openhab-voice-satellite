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


def test_load_resolves_paths_relative_to_config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
piper:
  voices:
    de: models/de.onnx
    en: /abs/en.onnx
earcons:
  wake: sounds/wake.wav
"""
    )
    config = load_config(path)
    # relative paths anchor at the config file's directory, absolute stay
    assert config.piper.voices["de"] == str(tmp_path / "models/de.onnx")
    assert config.piper.voices["en"] == "/abs/en.onnx"
    assert config.earcons.wake == str(tmp_path / "sounds/wake.wav")
    assert config.earcons.ack == str(tmp_path / "sounds/ack.wav")  # default too


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
    assert config.tts.engine == "piper"


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


def test_tts_engine_local_no_longer_valid():
    with pytest.raises(ValueError):
        Config.model_validate({"tts": {"engine": "local"}})


def test_tts_engine_kokoro_no_longer_valid():
    with pytest.raises(ValueError):
        Config.model_validate({"tts": {"engine": "kokoro"}})


def test_stale_kokoro_block_ignored():
    # configs from the kokoro era keep loading; the block is just dropped
    config = Config.model_validate({"kokoro": {"voices": {"de": {"model": "x"}}}})
    assert config.tts.engine == "piper"


def test_piper_default_language_must_have_voice():
    en_only = {"voices": {"en": "models/piper/en_GB-alba-medium.onnx"}}
    with pytest.raises(ValueError, match="piper voice"):
        Config.model_validate({"tts": {"engine": "piper"}, "piper": en_only})
    # also enforced for cloud engines (piper is their fallback)
    with pytest.raises(ValueError, match="piper voice"):
        Config.model_validate(
            {
                "tts": {"engine": "deepgram", "default_language": "de"},
                "piper": en_only,
                "deepgram": {"api_key": "k"},
            }
        )
    # default language covered -> valid
    Config.model_validate({"tts": {"default_language": "en"}, "piper": en_only})


def test_empty_languages_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"stt": {"languages": []}})


def test_zero_followup_timeout_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"dialog": {"followup_timeout_s": 0}})
