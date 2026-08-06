"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Credential env vars override config-file values (config.py key/token
    # properties); a developer's exported keys must not leak into assertions.
    for var in ("OPENHAB_TOKEN", "GEMINI_API_KEY", "DEEPGRAM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
