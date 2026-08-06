import pytest

from openhab_voice_satellite.audio.gst_devices import AudioNode, match_node

NODES = [
    AudioNode(
        name="alsa_input.usb-MIC.mono-fallback",
        description="USB Microphone",
        media_class="Audio/Source",
    ),
    AudioNode(
        name="alsa_output.usb-DAC.analog-stereo",
        description="USB DAC Analog Stereo",
        media_class="Audio/Sink",
    ),
    AudioNode(
        name="echo-cancel-source",
        description="Echo-Cancel Source",
        media_class="Audio/Source/Virtual",
    ),
]


def test_none_means_default_node():
    assert match_node(NODES, None, "input") is None
    assert match_node(NODES, None, "output") is None


def test_matches_substring_of_node_name():
    assert match_node(NODES, "usb-mic", "input") == "alsa_input.usb-MIC.mono-fallback"


def test_matches_substring_of_description():
    assert match_node(NODES, "Analog Stereo", "output") == "alsa_output.usb-DAC.analog-stereo"


def test_match_is_case_insensitive():
    assert match_node(NODES, "ECHO-CANCEL", "input") == "echo-cancel-source"


def test_kind_filters_media_class():
    # a sink must never match an input lookup, even on substring hit
    with pytest.raises(ValueError, match="no input device matching 'DAC'"):
        match_node(NODES, "DAC", "input")


def test_virtual_source_counts_as_input():
    assert match_node(NODES, "echo-cancel", "input") == "echo-cancel-source"


def test_no_match_raises_like_find_device():
    with pytest.raises(ValueError, match="no output device matching 'bogus'"):
        match_node(NODES, "bogus", "output")


def _dump_node(node_id, name, media_class=None):
    props = {"node.name": name}
    if media_class:
        props["media.class"] = media_class
    return {"id": node_id, "type": "PipeWire:Interface:Node", "info": {"props": props}}


def _dump_link(link_id, out_node, in_node):
    return {
        "id": link_id,
        "type": "PipeWire:Interface:Link",
        "info": {"output-node-id": out_node, "input-node-id": in_node},
    }


def test_parse_stream_peers_finds_both_directions():
    from openhab_voice_satellite.audio.gst_devices import parse_stream_peers

    dump = [
        _dump_node(1, "echo-cancel-source"),
        _dump_node(2, "echo-cancel-sink"),
        _dump_node(10, "app", "Stream/Input/Audio"),
        _dump_node(11, "app", "Stream/Output/Audio"),
        _dump_link(20, 1, 10),   # source -> capture stream
        _dump_link(21, 11, 2),   # playback stream -> sink
    ]
    assert parse_stream_peers(dump, "app") == {
        "input": "echo-cancel-source",
        "output": "echo-cancel-sink",
    }


def test_parse_stream_peers_missing_links():
    from openhab_voice_satellite.audio.gst_devices import parse_stream_peers

    dump = [_dump_node(10, "app", "Stream/Input/Audio")]
    assert parse_stream_peers(dump, "app") == {}
