"""Shared test setup."""

from __future__ import annotations

import pytest

# Env vars the app reads straight from the process environment. Credentials
# override config-file values, so a developer's exported keys would leak into
# assertions; the dump directories are worse than that — app.py reads
# OVS_DUMP_WAKE on every frame, so a developer running pytest with the
# documented field-tuning vars exported had test_app's ScriptedDetector wakes
# write synthetic all-zero WAVs into their real dump corpus, and those became
# hard negatives in the next verifier retrain. The dedicated guard test cannot
# catch it: it asserts tmp_path is empty, which is true wherever the dumps
# actually landed.
_LEAKY_ENV = (
    "OPENHAB_TOKEN",
    "GEMINI_API_KEY",
    "DEEPGRAM_API_KEY",
    "OVS_DUMP_WAKE",
    "OVS_DUMP_WAKE_SCORE",
    "OVS_DUMP_UTTERANCES",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _LEAKY_ENV:
        monkeypatch.delenv(var, raising=False)
