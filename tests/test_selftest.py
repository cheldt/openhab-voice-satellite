import pytest
from aiohttp.test_utils import TestServer

from openhab_voice_satellite.config import Config
from openhab_voice_satellite.selftest import check_deepgram, run_checks, select_checks

from .fakes import FakeDeepgram


@pytest.fixture
async def fake_deepgram():
    fake = FakeDeepgram()
    server = TestServer(fake.build_app())
    await server.start_server(shutdown_timeout=0.2)
    yield fake, server
    await server.close()


def _deepgram_config(server: TestServer, **engines) -> Config:
    return Config.model_validate(
        {
            **{k: {"engine": v} for k, v in engines.items()},
            "deepgram": {
                "api_key": "k",
                "base_url": str(server.make_url("")),
                "stt_model": "nova-test",
                "tts_voices": {"de": "aura-test-de", "en": "aura-test-en"},
            },
        }
    )


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


# --- run_checks: dispatch, failure handling, exit code ----------------------


def _sync_ok(config):
    pass


async def _async_ok(config):
    pass


def _raising(config):
    raise RuntimeError("model file missing")


async def test_run_checks_all_pass_returns_zero(capsys):
    code = await run_checks(
        Config(), checks=[("sync check", _sync_ok), ("async check", _async_ok)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "ok   sync check" in out
    assert "ok   async check" in out  # async checks must actually be awaited
    assert "all checks passed" in out


async def test_run_checks_failure_reported_and_exit_code_one(capsys):
    code = await run_checks(
        Config(),
        checks=[("good", _sync_ok), ("bad", _raising), ("also good", _async_ok)],
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL bad: model file missing" in out
    assert "ok   also good" in out  # a failure must not stop later checks
    assert "1 check(s) failed" in out


# --- check_deepgram: per-direction model probes on top of check_auth --------


async def test_check_deepgram_probes_models_per_direction(fake_deepgram):
    fake, server = fake_deepgram
    await check_deepgram(_deepgram_config(server, stt="deepgram", tts="deepgram"))
    # a silence /v1/listen validates stt_model
    assert any(("model", "nova-test") in query for query, _ in fake.listen_requests)
    # a one-word /v1/speak validates every configured voice name
    voices = sorted(dict(query)["model"] for query, _ in fake.speak_requests)
    assert voices == ["aura-test-de", "aura-test-en"]


async def test_check_deepgram_tts_only_skips_the_listen_probe(fake_deepgram):
    fake, server = fake_deepgram
    await check_deepgram(_deepgram_config(server, tts="deepgram"))
    assert fake.listen_requests == []
    assert len(fake.speak_requests) == 2
    assert fake.auth_headers[0] == "Token k"  # check_auth still ran first


# --- check_wakeword: every head that has a threshold read against it -------


class _ProbeDetector:
    """Detector stand-in with scriptable per-head scores and a verifier."""

    def __init__(self, wake=0.0, stop=0.0, verify=0.5):
        self._scores = {"wake": wake, "stop": stop}
        self._verify_score = verify
        self.verify_calls = 0

    def process(self, frame, speaking=False):
        return None

    def score(self, key="wake"):
        return self._scores[key]

    def _verify(self):
        self.verify_calls += 1
        return self._verify_score


def _check(monkeypatch, detector, **wakeword):
    # check_wakeword imports build_detector lazily, so patch it at the source
    from openhab_voice_satellite import selftest, wakeword as wakeword_module

    monkeypatch.setattr(wakeword_module, "build_detector", lambda config: detector)
    selftest.check_wakeword(
        Config.model_validate({"wakeword": {"model": "w.onnx", **wakeword}})
    )


STAGE2 = {"model": "v.onnx", "mel_basis": "m.npy"}


def test_a_healthy_two_stage_config_passes(monkeypatch):
    detector = _ProbeDetector(verify=0.42)
    _check(monkeypatch, detector, stage2=STAGE2)
    assert detector.verify_calls == 1  # the verifier really ran


def test_the_verifier_inference_is_exercised_not_just_its_shape(monkeypatch):
    """--check feeds silence, which never crosses stage 1.

    So _verify() was the one thing the self-test could not reach: a graph that
    fails at run time, or a head emitting a logit instead of a probability,
    passed a green --check and then broke the monitor loop at the first real
    wake, hours later.
    """
    detector = _ProbeDetector(verify=7.3)  # a raw logit, not a probability
    with pytest.raises(ValueError, match="stage-2 verifier scored 7.3"):
        _check(monkeypatch, detector, stage2=STAGE2)


def test_a_verifier_that_raises_fails_the_check(monkeypatch):
    class Boom(_ProbeDetector):
        def _verify(self):
            raise RuntimeError("ONNX node failed at run time")

    with pytest.raises(RuntimeError, match="run time"):
        _check(monkeypatch, Boom(), stage2=STAGE2)


def test_a_single_stage_config_never_calls_the_verifier(monkeypatch):
    detector = _ProbeDetector()
    _check(monkeypatch, detector)
    assert detector.verify_calls == 0


def test_the_stop_head_is_range_checked_too(monkeypatch):
    """stop_threshold is read against these scores exactly like the wake bar.

    openwakeword passes raw model output through unclamped, so a stop model
    exported without its sigmoid used to pass --check and then either fire on
    nearly any speech during playback or never fire at all.
    """
    detector = _ProbeDetector(stop=4.5)
    with pytest.raises(ValueError, match="stop model scored 4.5"):
        _check(monkeypatch, detector, stop_model="s.onnx")


def test_the_stop_head_is_ignored_when_unconfigured(monkeypatch):
    # score("stop") falls back to the wake trigger, so checking it without a
    # stop model would assert the wake score twice under the wrong name
    _check(monkeypatch, _ProbeDetector(stop=99.0))
