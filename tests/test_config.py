import pytest

from stt_proxy.config import Config, load_config


def test_defaults():
    config = Config()
    assert config.audio.sample_rate == 16000
    assert config.audio.frame_samples == 1280
    assert config.wakeword.model == "hey_jarvis"
    assert config.openhab.llm_tools == "item-send-command"
    assert config.openhab.verify_ssl is True
    assert config.tts.default_language == "de"
    assert config.dialog.enabled is True
    assert config.dialog.max_turns == 3
    assert config.dialog.context_mode == "verbatim"
    assert config.dialog.earcon == "wake"


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


def test_env_token_wins(monkeypatch):
    config = Config.model_validate({"openhab": {"api_token": "file-token"}})
    assert config.openhab.token == "file-token"
    monkeypatch.setenv("OPENHAB_TOKEN", "env-token")
    assert config.openhab.token == "env-token"


def test_default_language_must_have_voice():
    with pytest.raises(ValueError):
        Config.model_validate({"tts": {"voices": {"en": "x.onnx"}, "default_language": "de"}})


def test_empty_languages_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"stt": {"languages": []}})


def test_invalid_dialog_context_mode_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"dialog": {"context_mode": "telepathy"}})
