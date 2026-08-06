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


# The pipewire device provider can emit device-removed events during process
# teardown, after Python has already finalized the GstDevice wrappers —
# GStreamer then logs "invalid unclassed pointer" criticals (cosmetic, seen on
# PipeWire 1.2.x). Keeping the monitor and its devices referenced for the
# process lifetime avoids the premature finalization.
_monitor_refs: list[object] = []


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
        devices = monitor.get_devices()
        _monitor_refs.append((monitor, devices))
        for device in devices:
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


def parse_stream_peers(dump: list[dict], client_name: str) -> dict[str, str]:
    """Extract which nodes a client's streams are actually linked to.

    `dump` is parsed `pw-dump` output. Returns {"input": <peer node.name>,
    "output": <peer node.name>} for the client's capture/playback streams
    (keys absent when the stream or its link doesn't exist). PipeWire links
    streams silently — a target that cannot be linked falls back to the
    default node with no error anywhere — so the only way to know what a
    stream records from or plays to is to look at the live graph.
    """
    nodes = {
        obj["id"]: obj["info"]["props"]
        for obj in dump
        if obj.get("type") == "PipeWire:Interface:Node" and obj.get("info", {}).get("props")
    }
    streams = {}  # node id -> "input"|"output"
    for node_id, props in nodes.items():
        if props.get("node.name") != client_name:
            continue
        media_class = props.get("media.class", "")
        if media_class == "Stream/Input/Audio":
            streams[node_id] = "input"
        elif media_class == "Stream/Output/Audio":
            streams[node_id] = "output"
    peers: dict[str, str] = {}
    for obj in dump:
        if obj.get("type") != "PipeWire:Interface:Link":
            continue
        info = obj.get("info", {})
        out_node, in_node = info.get("output-node-id"), info.get("input-node-id")
        if streams.get(in_node) == "input" and out_node in nodes:
            peers["input"] = nodes[out_node].get("node.name", str(out_node))
        elif streams.get(out_node) == "output" and in_node in nodes:
            peers["output"] = nodes[in_node].get("node.name", str(in_node))
    return peers


async def verify_stream_links(
    client_name: str,
    expected_input: str | None,
    expected_output: str | None,
) -> None:
    """Log the nodes the app's streams actually linked to; warn on mismatch.

    Best-effort: silently skips when pw-dump is unavailable or unparseable.
    """
    import asyncio
    import json
    import logging

    log = logging.getLogger(__name__)
    try:
        proc = await asyncio.create_subprocess_exec(
            "pw-dump",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        peers = parse_stream_peers(json.loads(stdout), client_name)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the app
        log.debug("stream link verification skipped: %s", exc)
        return
    for kind, expected in (("input", expected_input), ("output", expected_output)):
        actual = peers.get(kind)
        if actual is None:
            log.warning("audio %s stream has no link — capture/playback will be dead", kind)
        elif expected is not None and actual != expected:
            log.warning(
                "audio %s stream linked to %r, NOT the configured %r "
                "(PipeWire fell back silently — check the node with wpctl status)",
                kind, actual, expected,
            )
        else:
            log.info("audio %s stream linked to %r", kind, actual)


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
