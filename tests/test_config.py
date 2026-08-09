import logging

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


def test_stt_cpu_threads_default_leaves_a_core_free():
    # 3, not 4: whisper must not starve the wakeword loop on a 4-core Pi 5
    assert Config().stt.cpu_threads == 3


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


def test_wakeword_engine_defaults_to_openwakeword():
    assert Config().wakeword.engine == "openwakeword"


def test_violawake_needs_frames_it_can_actually_score():
    # violawake scores whole 20 ms units; a remainder makes it return 0.0 for
    # every frame, so a dead detector has to be a config error instead
    viola = {"engine": "violawake", "model": "wake.onnx"}
    with pytest.raises(ValueError, match="frame_ms"):
        Config.model_validate({"wakeword": viola, "audio": {"frame_ms": 30}})
    with pytest.raises(ValueError, match="frame_ms"):
        Config.model_validate({"wakeword": viola, "audio": {"frame_ms": 240}})
    # multiples of 20 up to violawake's 200 ms ceiling are fine
    Config.model_validate({"wakeword": viola, "audio": {"frame_ms": 80}})
    Config.model_validate({"wakeword": viola, "audio": {"frame_ms": 20}})
    # openwakeword is unaffected by any of it
    Config.model_validate({"audio": {"frame_ms": 30}})


def test_violawake_rejects_openwakeword_only_settings():
    with pytest.raises(ValueError, match="verifier_model"):
        Config.model_validate(
            {"wakeword": {"engine": "violawake", "model": "w.onnx",
                          "verifier_model": "v.pkl"}}
        )
    with pytest.raises(ValueError, match="verifier_model"):
        Config.model_validate(
            {"wakeword": {"engine": "violawake", "model": "w.onnx",
                          "stop_verifier_model": "v.pkl"}}
        )


def test_violawake_rejects_the_openwakeword_default_model():
    # switching engine without touching model would otherwise look for an
    # openwakeword phrase name in violawake's registry
    with pytest.raises(ValueError, match="openwakeword phrase"):
        Config.model_validate({"wakeword": {"engine": "violawake"}})


def test_violawake_adaptive_band_must_be_ordered():
    with pytest.raises(ValueError, match="min_threshold"):
        Config.model_validate(
            {
                "wakeword": {
                    "engine": "violawake",
                    "model": "w.onnx",
                    "viola": {"adaptive": {"min_threshold": 0.9, "max_threshold": 0.6}},
                }
            }
        )


def test_speaking_threshold_below_the_idle_one_warns(caplog):
    # raising `threshold` and leaving `threshold_speaking` at its default
    # inverts the echo margin: the bar drops while our own output is audible
    with caplog.at_level(logging.WARNING):
        config = Config.model_validate(
            {"wakeword": {"threshold": 0.9, "threshold_speaking": 0.7}}
        )
    assert "threshold_speaking (0.70) is below wakeword.threshold (0.90)" in caplog.text
    # a warning, not a rejection — a lower bar is a legitimate barge-in choice
    assert config.wakeword.threshold_speaking == 0.7


def test_stop_threshold_speaking_below_the_idle_one_warns(caplog):
    with caplog.at_level(logging.WARNING):
        Config.model_validate(
            {"wakeword": {"stop_threshold": 0.6, "stop_threshold_speaking": 0.3}}
        )
    assert "stop_threshold_speaking (0.30)" in caplog.text


def test_raised_speaking_thresholds_are_silent(caplog):
    # the shipped defaults (0.5 / 0.7) and the stop_threshold_speaking=None
    # fallback both keep the speaking bar at or above the idle one
    with caplog.at_level(logging.WARNING):
        Config.model_validate({"wakeword": {"stop_threshold": 0.5}})
    assert "is below wakeword." not in caplog.text


def test_wakeword_model_paths_resolve_relative_to_the_config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
wakeword:
  model: models/wake.onnx
  stop_model: /abs/stop.onnx
  verifier_model: models/verifier.pkl
"""
    )
    config = load_config(path)
    assert config.wakeword.model == str(tmp_path / "models/wake.onnx")
    assert config.wakeword.stop_model == "/abs/stop.onnx"
    assert config.wakeword.verifier_model == str(tmp_path / "models/verifier.pkl")


def test_pretrained_wakeword_names_are_not_treated_as_paths(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("wakeword:\n  model: hey_jarvis\n")
    # openwakeword phrases and violawake registry names must stay verbatim
    assert load_config(path).wakeword.model == "hey_jarvis"


def test_empty_languages_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"stt": {"languages": []}})


def test_zero_followup_timeout_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"dialog": {"followup_timeout_s": 0}})
