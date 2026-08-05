"""Shared GStreamer helpers. Import only where gi is genuinely needed."""

from __future__ import annotations

_gst = None


def gst_init():
    """Import gi + Gst and initialize once; returns the Gst module."""
    global _gst
    if _gst is None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        _gst = Gst
    return _gst


def s16_mono_caps(rate: int) -> str:
    return f"audio/x-raw,format=S16LE,rate={rate},channels=1,layout=interleaved"


def install_sync_handler(Gst, bus, on_error, on_eos) -> None:
    """Route bus ERROR/EOS to callbacks from GStreamer's posting threads.

    Handlers must not call set_state() (deadlock); only log and hand off
    to the event loop via call_soon_threadsafe.
    """

    def handler(_bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            on_error(err, debug)
        elif message.type == Gst.MessageType.EOS:
            on_eos()
        return Gst.BusSyncReply.DROP

    bus.set_sync_handler(handler)
