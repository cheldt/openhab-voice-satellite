"""CLI dispatch: mode selection, rejected flag sets, exit-code plumbing.

`main()` is the one module with no test file of its own, yet it owns two
contracts other things depend on: the `CaptureClosedError -> exit 1` mapping
the systemd unit's `Restart=on-failure` is built on, and the `nargs="*"`
`--score-wav` condition ultiwake's gate.sh/smoke.sh drive.
"""

from __future__ import annotations

import pytest

from openhab_voice_satellite import __main__ as cli


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("logging:\n  level: WARNING\n")
    return path


def _run(monkeypatch, config_file, *argv):
    monkeypatch.setattr(
        "sys.argv",
        ["openhab-voice-satellite", "--config", str(config_file), *argv],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


# --- rejected flag sets ---------------------------------------------------


@pytest.mark.parametrize("argv, message", [
    (["--model", "cand.onnx"], "--model applies to a scoring run only"),
    (["--engine", "openwakeword"], "--engine applies to a scoring run only"),
    (["--compare", "b.onnx"], "--compare applies to a scoring run only"),
    (["--model", "a", "--compare", "b"], "--model, --compare apply"),
])
def test_scoring_overrides_without_a_scoring_flag_are_an_error(
    monkeypatch, config_file, capsys, argv, message
):
    """A dropped override means field-testing a model that never loaded.

    Dispatch is a first-match if-chain, so these fell through to launching the
    full app with the config's old model and no warning at all.
    """
    assert _run(monkeypatch, config_file, *argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--check", "--probe-mic"],
    ["--list-devices", "--check"],
    ["--check", "--score-wav"],
    ["--probe-mic", "--negatives", "neg"],
])
def test_combining_modes_is_an_error(monkeypatch, config_file, capsys, argv):
    # the if-chain ran only the first branch, so the user believed both had run
    assert _run(monkeypatch, config_file, *argv) == 2
    assert "pick one mode" in capsys.readouterr().err


def test_a_scoring_override_with_a_scoring_flag_is_accepted(
    monkeypatch, config_file
):
    seen = {}

    def fake_score_wavs(config, paths, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("openhab_voice_satellite.bench.score_wavs", fake_score_wavs)
    assert _run(
        monkeypatch, config_file, "--score-wav", "a.wav", "--model", "cand.onnx"
    ) == 0
    assert seen["model"] == "cand.onnx"


# --- dispatch and exit codes ---------------------------------------------


def test_bare_score_wav_still_enters_the_scoring_branch(monkeypatch, config_file):
    # ultiwake's gate.sh passes --positives/--negatives with no --score-wav at
    # all, so the branch keys off `is not None`, not truthiness: an empty list
    # must still be a scoring run
    called = {}
    monkeypatch.setattr(
        "openhab_voice_satellite.bench.score_wavs",
        lambda config, paths, **kw: called.setdefault("paths", paths) or 2,
    )
    assert _run(monkeypatch, config_file, "--score-wav") == 2
    assert called["paths"] == []


def test_positives_alone_enters_the_scoring_branch(monkeypatch, config_file):
    monkeypatch.setattr(
        "openhab_voice_satellite.bench.score_wavs", lambda *a, **kw: 2
    )
    assert _run(monkeypatch, config_file, "--positives", "pos") == 2


def test_check_exit_code_is_the_failure_count_verdict(monkeypatch, config_file):
    async def fake_run_checks(config):
        return 1

    monkeypatch.setattr(
        "openhab_voice_satellite.selftest.run_checks", fake_run_checks
    )
    assert _run(monkeypatch, config_file, "--check") == 1


def test_probe_mic_exit_code_is_passed_through(monkeypatch, config_file):
    monkeypatch.setattr(
        "openhab_voice_satellite.probe.probe_mic", lambda config: 1
    )
    assert _run(monkeypatch, config_file, "--probe-mic") == 1


def test_list_devices_reports_a_broken_stack_instead_of_a_traceback(
    monkeypatch, config_file, capsys
):
    def boom():
        raise RuntimeError("no PipeWire")

    monkeypatch.setattr(
        "openhab_voice_satellite.audio.gst_devices.list_audio_nodes", boom
    )
    assert _run(monkeypatch, config_file, "--list-devices") == 1
    out = capsys.readouterr().out
    assert "cannot list PipeWire nodes" in out
    assert "deploy/install.md" in out


def test_capture_death_exits_nonzero_for_systemd(monkeypatch, config_file, caplog):
    """The other half of `Restart=on-failure`.

    A clean exit here would leave the satellite silently gone until somebody
    noticed, so the unit could never restart it.
    """
    from openhab_voice_satellite.app import CaptureClosedError

    class DyingApp:
        def __init__(self, config):
            pass

        async def run(self):
            raise CaptureClosedError("capture closed mid-run")

    monkeypatch.setattr("openhab_voice_satellite.app.App", DyingApp)
    assert _run(monkeypatch, config_file) == 1
    assert "exiting for restart" in caplog.text


def test_ctrl_c_is_a_clean_exit(monkeypatch, config_file):
    class Interrupted:
        def __init__(self, config):
            pass

        async def run(self):
            raise KeyboardInterrupt

    monkeypatch.setattr("openhab_voice_satellite.app.App", Interrupted)
    monkeypatch.setattr(
        "sys.argv", ["openhab-voice-satellite", "--config", str(config_file)]
    )
    cli.main()  # returns, no SystemExit


def test_a_missing_config_file_is_not_swallowed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["openhab-voice-satellite", "--config", str(tmp_path / "nope.yaml"),
         "--check"],
    )
    with pytest.raises(FileNotFoundError):
        cli.main()


def test_the_default_config_is_config_yaml_in_the_cwd(monkeypatch, tmp_path):
    # the README quick start invokes the CLI with no --config at all
    (tmp_path / "config.yaml").write_text("logging:\n  level: WARNING\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "openhab_voice_satellite.probe.probe_mic", lambda config: 0
    )
    monkeypatch.setattr("sys.argv", ["openhab-voice-satellite", "--probe-mic"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
