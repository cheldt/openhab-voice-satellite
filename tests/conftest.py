"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Credential env vars override config-file values (config.py key/token
    # properties); a developer's exported keys must not leak into assertions.
    # The dump switches belong here for the same reason: a developer who left
    # one exported would have the suite scatter WAVs into their dump directory
    # and pass tests that are meant to prove the env var controls it.
    for var in (
        "OPENHAB_TOKEN", "GEMINI_API_KEY", "DEEPGRAM_API_KEY",
        "OVS_DUMP_WAKE", "OVS_DUMP_UTTERANCES",
    ):
        monkeypatch.delenv(var, raising=False)
