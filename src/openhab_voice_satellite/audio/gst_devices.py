"""PipeWire node enumeration and name resolution (GStreamer DeviceMonitor).

`AudioNode` and `match_node` are pure and importable without PyGObject;
everything touching GStreamer imports `gi` lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_KIND_CLASS = {"input": "Audio/Source", "output": "Audio/Sink"}


@dataclass(frozen=True)
class AudioNode:
    name: str  # PipeWire node.name — valid as pipewiresrc/pipewiresink target-object
    description: str  # human-readable node/device description
    media_class: str  # "Audio/Source", "Audio/Sink" (possibly with suffix, e.g. /Virtual)


def match_node(
    nodes: list[AudioNode],
    substring: str | None,
    kind: Literal["input", "output"],
) -> str | None:
    """Resolve a node by case-insensitive substring of name or description.

    None = default node (caller omits target-object). No match raises.
    """
    if substring is None:
        return None
    want = _KIND_CLASS[kind]
    needle = substring.lower()
    for node in nodes:
        if not node.media_class.startswith(want):
            continue
        if needle in node.name.lower() or needle in node.description.lower():
            return node.name
    raise ValueError(f"no {kind} device matching {substring!r}")


def list_audio_nodes() -> list[AudioNode]:
    """Enumerate PipeWire audio nodes via a one-shot Gst.DeviceMonitor."""
    from .gst_common import gst_init

    Gst = gst_init()
    monitor = Gst.DeviceMonitor.new()
    monitor.add_filter("Audio/Source", None)
    monitor.add_filter("Audio/Sink", None)
    if not monitor.start():
        raise RuntimeError("failed to start GStreamer device monitor")
    try:
        nodes = []
        for device in monitor.get_devices():
            props = device.get_properties()
            name = props.get_string("node.name") if props else None
            if not name:
                continue
            description = None
            if props:
                description = props.get_string("node.description") or props.get_string(
                    "device.description"
                )
            nodes.append(
                AudioNode(
                    name=name,
                    description=description or device.get_display_name() or name,
                    media_class=device.get_device_class(),
                )
            )
        return nodes
    finally:
        monitor.stop()


def resolve_node(substring: str | None, kind: Literal["input", "output"]) -> str | None:
    if substring is None:
        return None
    return match_node(list_audio_nodes(), substring, kind)


def probe_capture(node_name: str | None, sample_rate: int, timeout_s: float = 5.0) -> None:
    """Open the real capture pipeline and require one sample within timeout.

    Catches both an unreachable PipeWire daemon and a target node that
    WirePlumber cannot link (which stalls without a bus error).
    """
    import threading

    from .gst_common import gst_init, s16_mono_caps

    Gst = gst_init()
    target = f'target-object="{node_name}" ' if node_name else ""
    pipeline = Gst.parse_launch(
        f"pipewiresrc {target}! audioconvert ! audioresample ! "
        f"{s16_mono_caps(sample_rate)} ! appsink name=sink emit-signals=true sync=false"
    )
    got_sample = threading.Event()

    def on_sample(sink):
        sink.emit("pull-sample")
        got_sample.set()
        return Gst.FlowReturn.OK

    pipeline.get_by_name("sink").connect("new-sample", on_sample)
    try:
        pipeline.set_state(Gst.State.PLAYING)
        state = pipeline.get_state(int(timeout_s * Gst.SECOND))[0]
        if state == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("capture pipeline failed to start (PipeWire running?)")
        if not got_sample.wait(timeout_s):
            raise RuntimeError(
                f"no audio from {node_name or 'default source'} within {timeout_s:.0f}s"
            )
    finally:
        pipeline.set_state(Gst.State.NULL)
