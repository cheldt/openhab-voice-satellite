import asyncio

import pytest

gi = pytest.importorskip("gi")

from openhab_voice_satellite.audio.gst_common import s16_mono_caps  # noqa: E402
from openhab_voice_satellite.audio.gst_sink import PipewireSink  # noqa: E402
from openhab_voice_satellite.audio.gst_source import PipewireSource  # noqa: E402
from openhab_voice_satellite.audio.io import audio_io  # noqa: E402
from openhab_voice_satellite.config import AudioConfig  # noqa: E402

FAKESINK = (
    "appsrc name=src format=time is-live=true max-bytes=0 block=false "
    "! audioconvert ! audioresample ! volume name=vol ! fakesink sync=false"
)


def _testsrc_describe(target, sample_rate):
    return (
        f"audiotestsrc is-live=true samplesperbuffer=800 ! audioconvert ! audioresample "
        f"! {s16_mono_caps(sample_rate)} "
        f"! appsink name=sink emit-signals=true sync=false max-buffers=8 drop=true"
    )


@pytest.fixture
def fake_pipelines(monkeypatch):
    monkeypatch.setattr(PipewireSource, "_describe", staticmethod(_testsrc_describe))
    monkeypatch.setattr(PipewireSink, "_describe", staticmethod(lambda target: FAKESINK))


async def test_closes_both_on_exception_in_block(fake_pipelines):
    handles = {}
    with pytest.raises(RuntimeError, match="boom"):
        async with audio_io(AudioConfig()) as (source, sink):
            handles["source"], handles["sink"] = source, sink
            raise RuntimeError("boom")
    # closed: the source ends its frame stream, the sink drops its keepalive
    Gst = handles["source"]._Gst
    assert handles["source"]._pipeline.current_state == Gst.State.NULL
    await asyncio.sleep(0)  # let the cancelled keepalive task finish
    assert handles["sink"]._keepalive_task.cancelled()


async def test_closes_source_when_sink_construction_fails(fake_pipelines, monkeypatch):
    # the sink constructor raises after the source is already PLAYING
    monkeypatch.setattr(
        PipewireSink, "_describe",
        staticmethod(
            lambda target: "appsrc name=src format=time is-live=true ! volume name=vol "
            "! filesink location=/nonexistent-dir/out.raw"
        ),
    )
    created = []
    original_init = PipewireSource.__init__

    def spy_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(PipewireSource, "__init__", spy_init)
    with pytest.raises(RuntimeError, match="playback"):
        async with audio_io(AudioConfig()):
            pass  # pragma: no cover - never entered
    assert len(created) == 1
    Gst = created[0]._Gst
    assert created[0]._pipeline.current_state == Gst.State.NULL
