# Installing openhab-voice-satellite on a Raspberry Pi 5

Target: Raspberry Pi OS Bookworm 64-bit, Python 3.11+.

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
sudo chmod 750 /opt/openhab-voice-satellite   # config.yaml will hold live credentials
git clone https://github.com/cheldt/openhab-voice-satellite.git /opt/openhab-voice-satellite   # or rsync the project over
cd /opt/openhab-voice-satellite
python3 -m venv --system-site-packages .venv

# Install the *locked* dependency set, with hashes. `pip install -e .` alone
# resolves the range specifiers in pyproject.toml fresh at install time, so
# what runs on the Pi is neither the version set the nightly pip-audit scans
# (it audits uv.lock) nor integrity-verified against a PyPI substitution — on
# a device holding the openHAB token and an always-on microphone.
pipx run uv export --frozen --format requirements-txt --no-emit-project \
  > /tmp/requirements-deploy.txt          # or: uv export ... on a dev box, then copy
.venv/bin/pip install --require-hashes -r /tmp/requirements-deploy.txt
.venv/bin/pip install -e . --no-deps      # the project itself, deps already in

# openwakeword is installed without dependencies on purpose — its metadata
# demands tflite-runtime, which has no wheels for current Pythons; the ONNX
# backend used here does not need it. Version pinned: >=0.6.0 is required for
# the ncpu kwarg that keeps its ONNX sessions single-threaded (unbounded
# sessions spin-wait and burn ~1 core per worker thread at idle). Hash-pinned
# too, since this one is outside the lock the line above verifies:
.venv/bin/pip install --no-deps 'openwakeword==0.6.0' \
  --hash=sha256:6f423a4e3ae9dd0e3cd12b50ff8abf69679f687b4ab349d7c82c021c0e2abc9d
```

The exact pinned set the nightly `pip-audit` scans is `uv.lock`, and that
workflow appends the same `openwakeword==0.6.0` pin, so the audited set and the
installed set are the same one.

## 3. Download models (~400 MB total)

```bash
export HF_HOME=/opt/openhab-voice-satellite/models/hf
.venv/bin/python scripts/download_models.py
```

The Piper voices (~60 MB each) land in `models/piper/`; whisper `small`
(~250 MB) is cached under `HF_HOME`. The earcon WAVs (`sounds/*.wav`) are
tracked in the repo; `scripts/make_earcons.py` only exists to regenerate them.

A custom wakeword model (`wakeword.model` pointing at a local `.onnx`, e.g.
`models/wakeword/hey_shodan.onnx`) is **not** provisioned by
`download_models.py` — copy the file into place yourself before `--check`,
which otherwise fails with a missing-model error.

### The stage-2 verifier (`wakeword.stage2`)

A mel-PCEN CNN that re-scores the 1.5 s window behind every stage-1 trigger,
so stage 1 can run low for recall while false accepts are somebody else's
problem. It rejects ~99 % of what reaches it. How *often* it runs depends
entirely on the room: roughly once a minute with nobody talking, but four
times in nine seconds during a conversation (measured on the reference Pi) —
stage 1 crosses its deliberately low bar on ordinary speech, which is the
load the verifier exists to absorb.
On the shipped model that is the difference between 0.7 false accepts an hour
at 99 % recall and 37.6 an hour at 94 %.

**Engine-neutral**, and demonstrably so: measured on the same 5.48 h, the same
verifier rejects ~99 % of stage-1 triggers regardless of which architecture
produced them.

`model` and `mel_basis` come as a pair — a retrained verifier with a stale
filterbank scores features it was never trained on, silently, so the config
rejects one without the other. Training, calibration and promotion live in the
ultiwake pipeline; its `./run.sh deploy` copies both files here and re-exports
the filterbank and golden fixtures in the same step.

### Optional: per-speaker verifier models

If the base model false-triggers on the TV, the radio or passers-by, a custom
verifier is the cheapest fix available. It is a small logistic regression that
re-scores a candidate detection using the embeddings openWakeWord already
computed, so it costs nothing at idle and one `predict_proba` on a hot frame.
It is speaker-*dependent* by design: it learns your household's voices and
rejects everyone else — guests included.

Training needs only sklearn/scipy/tqdm, which the `--no-deps` install above
already provides (openWakeWord's full `train.py` does not run here — it needs
torch and an external `piper_sample_generator` checkout). Record a handful of
16 kHz mono WAVs of each person saying the wakeword, plus some of them saying
other things, then:

```python
from openwakeword.custom_verifier_model import train_custom_verifier
train_custom_verifier(
    positive_reference_clips=["me-wake-1.wav", "me-wake-2.wav", ...],
    negative_reference_clips=["me-other-1.wav", ...],
    output_path="models/wakeword/shodan_listen_verifier.pkl",
    model_path="models/wakeword/shodan_listen.onnx",
)
```

Point `wakeword.verifier_model` at the result. Two warnings:

- **The verifier replaces the score, it does not gate it.** Every threshold in
  `config.yaml` then applies to a logistic probability with a different
  distribution than the base model's output. Re-run `--probe-mic` and re-tune
  `threshold`, `threshold_speaking` and `stop_threshold` after enabling one.
- **Verifier files are unpickled at startup, which is arbitrary code
  execution.** Only load files you trained yourself; treat one appearing in
  `models/` the way you would treat a new executable on the box.

## 4. Configure

```bash
cp config.example.yaml config.yaml
chmod 600 config.yaml                  # it is about to hold live credentials
.venv/bin/openhab-voice-satellite --list-devices     # lists PipeWire sources/sinks
$EDITOR config.yaml                    # devices, openHAB url + token
```

`chmod 600` is not optional. Under the default umask the copy is world-readable
inside a world-readable directory, and it holds the openHAB API token (full
smart-home control, including physical actuation) plus any Gemini/Deepgram keys
(billable). On a Pi that also runs another service, compromising that service is
then enough to read all three. If you would rather the credentials never sit in
a file the service reads, put them in an `EnvironmentFile=` (also `chmod 600`)
or systemd credentials, and leave `api_token: null` — the env vars win over the
config file either way.

For TLS, prefer `openhab.ca_cert` over `verify_ssl: false`: a self-signed
openHAB reached with verification off accepts *any* certificate, so anyone who
can intercept the connection harvests the bearer token from the first request.
Point `ca_cert` at the server's certificate (or its CA) and the token stays
protected.

`--list-devices` shows every PipeWire node with its name and description;
`audio.input_device` / `audio.output_device` match a case-insensitive
substring of either. Must run inside the PipeWire user session (not via
`sudo`). For debugging, `pw-dump` and `wpctl status` show the same nodes.

Create the openHAB side (see README section "openHAB setup"): a configured
voice interpreter that answers free text, plus an API token (openHAB UI ->
profile -> API tokens).

`stt.cpu_threads` defaults to 3 on purpose: whisper (ctranslate2) runs with
the GIL released and saturates every core it is given, but the always-on
wakeword monitor must keep its 80 ms cadence during transcription or
barge-in ("stop" while THINKING/SPEAKING) goes deaf. On the 4-core Pi 5
leave at least one core free; only raise this on machines with more cores.

## 5. Self-test

```bash
OPENHAB_TOKEN=... HF_HOME=/opt/openhab-voice-satellite/models/hf .venv/bin/openhab-voice-satellite --check
```

All checks (audio devices, wakeword, VAD, whisper, piper, cloud APIs
when a cloud engine is configured, openHAB REST)
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
systemd-analyze --user security openhab-voice-satellite   # check the sandbox
journalctl --user -u openhab-voice-satellite -f
```

The unit carries a sandboxing block (`ProtectSystem=strict`, `ProtectHome`,
`RestrictAddressFamilies`, an empty `CapabilityBoundingSet`,
`SystemCallFilter=@system-service`, …). It matters because the process parses
network responses, runs large native parsers on untrusted audio, and optionally
unpickles verifier files, so a memory-corruption bug in onnxruntime,
ctranslate2 or aiohttp would otherwise run with full access to the user's
session. Which directives a *user* unit actually honors depends on the systemd
version in the session, which is what `systemd-analyze --user security` reports
— run it after installing and loosen only what it shows to be breaking audio.
On Debian 12 / systemd 252 the reference install scores **5.4 MEDIUM**.

Three directives are commented out in the unit rather than set, all for the
same reason: a user manager has no `CAP_SETPCAP`, so anything shrinking the
capability bounding set fails the whole unit with `status=218`
(`EXIT_CAPABILITIES`) — `CapabilityBoundingSet=` itself and, less obviously,
`ProtectKernelModules=` and `ProtectClock=`, which imply a bounding-set drop.
They cost nothing here: the process is unprivileged and `NoNewPrivileges=yes`
keeps it that way. `MemoryDenyWriteExecute` is left off for a different reason
— onnxruntime and ctranslate2 map executable pages of their own.

If the install does not live in `/opt`, two lines need adapting:
`ReadWritePaths=` must name the install tree **and** the Hugging Face cache
(`huggingface_hub` writes lock files there even on a cache hit, so a cache
outside `ReadWritePaths=` under `ProtectHome=read-only` breaks startup), and
`WorkingDirectory=`/`ExecStart=` follow the tree. Verify before restarting by
copying the unit to a `Type=oneshot` probe whose `ExecStart` ends in `--check`:
that opens the real capture pipeline, loads piper and reaches the network from
inside the sandbox.

Two things the sandbox needs from the layout: `/opt/openhab-voice-satellite` is
the only writable path (`ReadWritePaths=`), so `HF_HOME` must stay inside it,
and PipeWire is reached over `AF_UNIX` in `$XDG_RUNTIME_DIR`, not through the
home directory `ProtectHome=read-only` covers.

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
