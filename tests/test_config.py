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


def test_a_removed_engine_is_rejected_rather_than_ignored():
    # a config carrying `engine: violawake` must not quietly run openWakeWord
    # against a model trained for something else
    with pytest.raises(ValueError, match="engine"):
        Config.model_validate({"wakeword": {"engine": "violawake"}})


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
    # pretrained openwakeword phrase names must stay verbatim
    assert load_config(path).wakeword.model == "hey_jarvis"


def test_empty_languages_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"stt": {"languages": []}})


def test_zero_followup_timeout_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"dialog": {"followup_timeout_s": 0}})


def test_cloud_tts_default_language_must_have_cloud_voice():
    with pytest.raises(ValueError, match="gemini.tts_voices"):
        Config.model_validate(
            {
                "tts": {"engine": "gemini", "default_language": "de"},
                "gemini": {"api_key": "k", "tts_voices": {"en": "Puck"}},
            }
        )
    with pytest.raises(ValueError, match="deepgram.tts_voices"):
        Config.model_validate(
            {
                "tts": {"engine": "deepgram", "default_language": "de"},
                "deepgram": {"api_key": "k", "tts_voices": {"en": "aura-2-thalia-en"}},
            }
        )
    # default language covered -> valid; STT-only cloud needs no TTS voice
    Config.model_validate(
        {
            "tts": {"engine": "gemini", "default_language": "de"},
            "gemini": {"api_key": "k", "tts_voices": {"de": "Kore"}},
        }
    )
    Config.model_validate(
        {
            "stt": {"engine": "deepgram"},
            "deepgram": {"api_key": "k", "tts_voices": {}},
        }
    )


# --- bounds: fields that used to accept nonsense their siblings reject -----


@pytest.mark.parametrize("section, values", [
    ("vad", {"silence_ms": 0}),
    ("vad", {"silence_ms": -500}),
    ("vad", {"no_speech_timeout_s": 0}),
    ("vad", {"no_speech_timeout_s": -1}),
    ("vad", {"max_utterance_s": 0}),
    ("audio", {"frame_ms": 0}),
    ("audio", {"frame_ms": 40}),   # legal-looking, but under one oww chunk
    ("audio", {"frame_ms": 120}),  # not a multiple of 80
    ("stt", {"cpu_threads": 0}),   # ctranslate2 reads 0 as "all cores"
])
def test_nonsense_values_are_rejected_at_load(section, values):
    with pytest.raises(ValueError):
        Config.model_validate({section: values})


def test_the_documented_frame_sizes_still_validate():
    # 80 ms is the shipped value; multiples of it stay legal
    for frame_ms in (80, 160, 240):
        assert Config.model_validate(
            {"audio": {"frame_ms": frame_ms}}
        ).audio.frame_ms == frame_ms


# --- stt.model: a name, a repo id, or a path -------------------------------


def test_stt_model_directory_resolves_against_the_config_file(tmp_path):
    (tmp_path / "models" / "whisper-de-ct2").mkdir(parents=True)
    path = tmp_path / "config.yaml"
    path.write_text('stt:\n  model: "models/whisper-de-ct2"\n')
    assert load_config(path).stt.model == str(tmp_path / "models/whisper-de-ct2")


@pytest.mark.parametrize("model", ["small", "Systran/faster-whisper-small"])
def test_stt_model_names_and_repo_ids_pass_through_verbatim(tmp_path, model):
    # a repo id contains a separator but is not a path; rewriting it would
    # hand faster-whisper an absolute path it then treats as a repo id
    path = tmp_path / "config.yaml"
    path.write_text(f'stt:\n  model: "{model}"\n')
    assert load_config(path).stt.model == model


def test_openhab_ca_cert_resolves_against_the_config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text('openhab:\n  ca_cert: "certs/openhab-ca.pem"\n')
    assert load_config(path).openhab.ca_cert == str(tmp_path / "certs/openhab-ca.pem")


def test_the_shipped_example_config_validates():
    """`cp config.example.yaml config.yaml` is the documented first step.

    config.py rejects removed enum values at load by design, and this repo's
    history removes them regularly (Kokoro TTS, the violawake engine), so
    example-vs-schema drift is a live risk that CI could not see: every new
    install would crash at first startup on the very file the docs told the
    user to copy.
    """
    from pathlib import Path

    example = Path(__file__).parent.parent / "config.example.yaml"
    config = load_config(example)
    assert config.wakeword.engine == "openwakeword"
    assert config.stt.engine == "local"  # the example must not need cloud keys
