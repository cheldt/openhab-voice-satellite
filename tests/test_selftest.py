from openhab_voice_satellite.config import Config
from openhab_voice_satellite.selftest import run_checks, select_checks


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
