import pytest

from pydantic import ValidationError

from openhab_voice_satellite.config import Config, WakewordConfig, load_config


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


# -- wakeword engine selection ------------------------------------------


def test_livekit_rejects_the_openwakeword_model_default():
    """'hey_jarvis' is an openwakeword phrase name, not a path.

    Left unset it would otherwise surface as a missing-file error from inside
    the engine, naming a path the user never wrote.
    """
    with pytest.raises(ValidationError, match="openwakeword default"):
        WakewordConfig(engine="livekit")


def test_livekit_rejects_openwakeword_verifiers():
    """Silently-ignored settings are the failure this guards against.

    Custom verifiers are an openwakeword feature; livekit has no equivalent,
    so the setting would do nothing at all and the thresholds tuned around it
    would be wrong.
    """
    with pytest.raises(ValidationError, match="verifier_model"):
        WakewordConfig(engine="livekit", model="m.onnx", verifier_model="v.pkl")
    with pytest.raises(ValidationError, match="stop_verifier_model"):
        WakewordConfig(engine="livekit", model="m.onnx", stop_verifier_model="v.pkl")


def test_livekit_accepts_a_model_path():
    config = WakewordConfig(engine="livekit", model="models/wakeword/livekit/x.onnx")
    assert config.livekit.hop_frames == 4  # the measured default


def test_openwakeword_keeps_the_pretrained_name_default():
    assert WakewordConfig().engine == "openwakeword"
    assert WakewordConfig().model == "hey_jarvis"


def test_unknown_engine_is_rejected():
    with pytest.raises(ValidationError):
        WakewordConfig(engine="porcupine")
