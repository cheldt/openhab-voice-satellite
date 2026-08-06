# Installing openhab-voice-satellite on a Raspberry Pi 5

Target: Raspberry Pi OS Bookworm 64-bit, Python 3.11–3.13 (kokoro-onnx does
not support 3.14 yet).

## 1. System packages

```bash
sudo apt update
sudo apt install python3-venv python3-gi \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-plugins-base gstreamer1.0-pipewire \
  gstreamer1.0-pulseaudio \
  pipewire pipewire-audio pipewire-pulse wireplumber
```

Audio runs on PipeWire: capture natively via GStreamer `pipewiresrc`,
playback via `pulsesink` through pipewire-pulse (same graph, same node
names — gst `pipewiresink` wedges long-lived streams on PipeWire 1.2.x).
`python3-gi` provides PyGObject from apt — it has no manylinux wheels, so the
venv below uses `--system-site-packages` instead of pip-building it. (On
non-Debian systems: install the girepository + cairo dev headers and use
`pip install -e '.[gst]'` in a normal venv instead.)

## 2. Get the code and install

```bash
sudo mkdir -p /opt/openhab-voice-satellite && sudo chown $USER /opt/openhab-voice-satellite
git clone https://github.com/cheldt/openhab-voice-satellite.git /opt/openhab-voice-satellite   # or rsync the project over
cd /opt/openhab-voice-satellite
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
# openwakeword is installed without dependencies on purpose — its metadata
# demands tflite-runtime, which has no wheels for current Pythons; the ONNX
# backend used here does not need it. Version pinned: >=0.6.0 is required for
# the ncpu kwarg that keeps its ONNX sessions single-threaded (unbounded
# sessions spin-wait and burn ~1 core per worker thread at idle):
.venv/bin/pip install --no-deps 'openwakeword==0.6.0'
```

## 3. Download models (~1.6 GB total)

```bash
export HF_HOME=/opt/openhab-voice-satellite/models/hf
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/make_earcons.py   # generates sounds/*.wav
```

The two Kokoro TTS models are ~326 MB each; the Piper voices (~60 MB each)
land in `models/piper/`. espeak-ng (used for phonemization) ships inside
the `espeakng-loader` Python wheel — no apt package needed.

Note: espeak-ng silently ignores data paths longer than ~147 characters
(fixed internal buffer) and falls back to a non-existent build path. Keep
the install prefix short (`/opt/openhab-voice-satellite` is fine); a deeply nested venv
will make TTS fail with `Error processing file '...phontab'`.

## 4. Configure

```bash
cp config.example.yaml config.yaml
.venv/bin/openhab-voice-satellite --list-devices     # lists PipeWire sources/sinks
$EDITOR config.yaml                    # devices, openHAB url + token
```

`--list-devices` shows every PipeWire node with its name and description;
`audio.input_device` / `audio.output_device` match a case-insensitive
substring of either. Must run inside the PipeWire user session (not via
`sudo`). For debugging, `pw-dump` and `wpctl status` show the same nodes.

Create the openHAB side (see README section "openHAB setup"): a configured
voice interpreter that answers free text, plus an API token (openHAB UI ->
profile -> API tokens).

## 5. Self-test

```bash
OPENHAB_TOKEN=... HF_HOME=/opt/openhab-voice-satellite/models/hf .venv/bin/openhab-voice-satellite --check
```

All six checks (audio devices, wakeword, VAD, whisper, kokoro/piper, openHAB REST)
must print `ok`. The audio check opens the real capture pipeline and requires
an actual sample, so it also catches a device name that PipeWire cannot link.
The whisper line also warms the model cache, so the first real interaction is
not slow.

## 6. Run as a service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/openhab-voice-satellite.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now openhab-voice-satellite
sudo loginctl enable-linger $USER
journalctl --user -u openhab-voice-satellite -f
```

## Optional: robust barge-in with echo cancellation

The mic hears the speaker during TTS playback. The service mitigates this in
software (raised wakeword threshold + volume ducking), but for reliable
"stop"-while-speaking use PipeWire's echo canceller:

```bash
mkdir -p ~/.config/pipewire/pipewire.conf.d
cat > ~/.config/pipewire/pipewire.conf.d/echo-cancel.conf <<'EOF'
context.modules = [
  { name = libpipewire-module-echo-cancel
    args = {
      library.name = aec/libspa-aec-webrtc
      source.props = { node.name = "echo-cancel-source" }
      sink.props   = { node.name = "echo-cancel-sink" }
    }
  }
]
EOF
systemctl --user restart pipewire
```

Verify the new nodes with `--list-devices`, then set in `config.yaml`:

```yaml
audio:
  input_device: "echo-cancel-source"
  output_device: "echo-cancel-sink"
```

This requires running openhab-voice-satellite as a **user** unit (it must live in the same
session as PipeWire) — which is the default unit shipped here. Do not set
`PIPEWIRE_NODE` in the unit environment; it would silently redirect all
streams.

## Fixed-rate USB microphones (ReSpeaker and similar)

Many USB voice-assistant mics (e.g. the SEEED ReSpeaker 4 Mic Array, UAC1.0)
only support a single sample rate — typically 16 kHz — while PipeWire's graph
clock defaults to 48 kHz. When such a device drives the graph, the per-cycle
sample ratio becomes fractional (quantum 256 @ 48 kHz = 85.33 device samples)
and capture chronically under-delivers: the service logs
`degraded capture: N of 125 expected mic frames` and wakeword detection goes
deaf, often only after a service restart while the first start works.
`pw-top` shows the mismatch (driver RATE 48000 vs device FORMAT `... 16000`)
and climbing ERR counters on the affected streams.

Pin the graph clock to the device's native rate, and use generous quanta —
this service processes 80 ms frames, so tiny low-latency cycles only add
deadline pressure (full-duplex USB on a small board misses 8 ms cycles):

```bash
mkdir -p ~/.config/pipewire/pipewire.conf.d
cat > ~/.config/pipewire/pipewire.conf.d/10-clock-rate-16k.conf <<'EOF'
context.properties = {
  default.clock.rate = 16000
  default.clock.allowed-rates = [ 16000 ]
  default.clock.quantum = 512
  default.clock.min-quantum = 256
  default.clock.max-quantum = 1024
}
EOF
systemctl --user restart pipewire pipewire-pulse wireplumber
```

When combining this with the echo canceller above, align the AEC with the
graph as well, or it will miss its (default 10 ms / 48 kHz) deadline every
cycle and glitch — visible as fast-climbing ERR counters on the echo-cancel
nodes in `pw-top`:

```
    args = {
      library.name = aec/libspa-aec-webrtc
      audio.rate = 16000
      audio.channels = 1
      node.latency = 512/16000
      source.props = { node.name = "echo-cancel-source" }
      sink.props   = { node.name = "echo-cancel-sink" }
    }
```

This removes all resampling on a 16 kHz-native device (the WebRTC echo
canceller works at 16 kHz too, buffering its 10 ms blocks internally).
Verify with `pw-top`: the driver row should show RATE 16000 and ERR should
stay 0 on every row; the service heartbeat (DEBUG log) should report ~125
frames per 10 s window.

## Powered speakers with auto-standby

Active speakers often power down after minutes of low signal and mute the
first few hundred ms after wake-up — the start of the first earcon after an
idle period goes missing even though the digital path delivers it completely
(verifiable on the sink's pulse monitor). The service already streams
inaudible keep-alive dither, but many amps ignore signal that quiet. If the
speaker's eco/auto-standby mode cannot be disabled, enable the wake-up
preamble in `config.yaml`:

```yaml
audio:
  wakeup_preamble_ms: 500     # soft ramped noise before a sound after an idle gap
  wakeup_preamble_idle_s: 5   # gap that counts as idle; match the speaker's mute delay
```

The preamble is a quiet rising hiss played only when a sound starts after
the configured quiet gap, so the amp is awake before the actual earcon or
answer begins. Speakers with fast signal-sensing mutes need a small
`wakeup_preamble_idle_s` (`0` = preamble before every sound); classic
minutes-scale auto-standby is fine with the 60 s default.
