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
    assert config.dialog.followup_timeout_s == 6.0
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
        Config.model_validate(
            {
                "tts": {
                    "voices": {
                        "en": {
                            "model": "x.onnx",
                            "voices": "v.bin",
                            "voice": "bf_emma",
                            "lang": "en-gb",
                        }
                    },
                    "default_language": "de",
                }
            }
        )


def test_tts_voice_defaults():
    config = Config()
    assert config.tts.voices["de"].voice == "martin"
    assert config.tts.voices["de"].lang == "de"
    assert config.tts.voices["en"].voice == "bf_emma"
    assert config.tts.voices["en"].speed == 1.0


def test_empty_languages_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"stt": {"languages": []}})


def test_zero_followup_timeout_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"dialog": {"followup_timeout_s": 0}})


def test_legacy_dialog_keys_ignored():
    config = Config.model_validate({"dialog": {"max_turns": 3, "context_mode": "stitch"}})
    assert config.dialog.enabled is True
