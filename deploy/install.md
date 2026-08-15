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

### Optional: the violawake engine (`wakeword.engine: violawake`)

[ViolaWake](https://github.com/GeeIHadAGoodTime/ViolaWake) (Apache-2.0) runs a
TemporalCNN head on top of the same openWakeWord melspectrogram + embedding
backbone, and benchmarks better than openWakeWord's own heads (5.49 % vs
8.24 % EER on its published comparison). The trade-offs are real, so read all
four before switching:

- **It ships no pretrained phrases.** openWakeWord gives you `hey_jarvis`,
  `alexa` and friends for free; violawake gives you a training pipeline
  (`violawake-train`, or their browser console) and expects a `.onnx` of your
  own. `wakeword.model` must point at one, or at a registry name like
  `temporal_cnn`.
- **`stop_model` costs roughly double here.** openWakeWord shares one feature
  backbone across every model; violawake builds one per `WakeDetector`, so a
  stop model means the melspectrogram and embedding networks run twice per
  frame. Measure with `--probe-mic` before committing to it.
- **`audio.frame_ms` must be a multiple of 20** (and at most 200). Anything
  else is rejected at config load, because violawake would otherwise score
  0.0 on every frame and never fire.
- **It is a young project** (v0.2.10 at the time of writing). Pin the version.

```bash
# --no-deps for the same reason as openwakeword: the [oww] extra pulls plain
# openwakeword, whose metadata demands the unbuildable tflite-runtime. Its
# other runtime needs (onnxruntime, numpy, scipy) are already installed; pysbd
# is the only one missing.
.venv/bin/pip install --no-deps 'violawake==0.2.10'
.venv/bin/pip install pysbd
```

The app forces violawake's ONNX sessions to a single non-spinning thread
(`src/openhab_voice_satellite/violawake_ort.py`), because upstream builds them
with no `SessionOptions` at all and ORT would otherwise size its intra-op pool
to the core count — the same idle multi-core burn the `ncpu=1` pin above
avoids on the openWakeWord path. Measured on an 8-core x86 box, loading one
detector: **+9 OS threads stock, +0 patched**.

Two layers do it, because a seam that still exists is not a seam that is still
used: the patch replaces `OnnxBackend.load`, and detector construction
additionally runs inside a block that forces session options onto every ORT
session built anywhere, then reports what happened:

```
violawake bound 3 ONNX sessions, no extra OS threads
```

That line is the one to check. `violawake ONNX thread patch skipped` means the
seam moved; a `gained N OS threads` warning means ORT sized a pool anyway. Both
keep the app running, and both mean idle CPU is about to be bad. `--probe-mic`
prints the per-frame cost directly (`ms/frame` and `cpu_ms` columns) — on an
8-core x86 box one violawake detector costs ~1.0 ms per 80 ms frame; `cpu_ms`
far above `ms/frame` is a spinning thread pool.

Model provisioning: `scripts/download_models.py` reads `wakeword.engine` from
your config and fetches registry models automatically; custom `.onnx` files it
only checks for. The shared openWakeWord backbone is downloaded for this engine
and for openWakeWord itself, but not for wakeforge, which ships its own
featurizer.
Set the cache location for the service user, since the default is `~/.violawake`:

```bash
export VIOLAWAKE_MODEL_DIR=/opt/openhab-voice-satellite/models/violawake
```

### Optional: the wakeforge engine (`wakeword.engine: wakeforge`)

[wakeforge](https://github.com/TigreGotico/wakeforge) (Apache-2.0) trains a
featurizer and a classifier head and exports both as ONNX. At runtime that is
all it is, so unlike the other two engines there is **nothing to install** —
`onnxruntime` and `numpy` are already dependencies, and the inference lives in
`wakeword_wakeforge.py`. Four things to know before switching:

- **The trainer does not run on the device.** `ww_trainer` needs PyTorch,
  torchaudio, librosa and several GB of corpora, and it is not published on
  PyPI. Train on a workstation and copy the two `.onnx` files over:

  ```bash
  pip install "ww_trainer @ git+https://github.com/TigreGotico/wakeforge@dev"
  python scripts/train_wakeforge.py "showdaan listen" showdaan_v1
  ```

  The wrapper writes to `models/wakeword/wakeforge/<name>/` and finishes by
  printing the `--compare` line that judges the result.
- **`wakeword.model` is a directory**, not a file. The featurizer and head only
  work as the pair they were trained as, so they are addressed by the thing
  that keeps them together. Override the filenames with
  `wakeword.wakeforge.featurizer` / `.head` if you used `ww_trainer-train`
  rather than the quickstart, which names them differently.
- **The head scores a 500 ms window** — 50 feature frames at a 10 ms hop, which
  is shorter than most wake phrases. That is upstream's trained window, not a
  knob. The rate is verified against the real model at startup and logged:
  `featurizer 100.0 fps, 50-frame window ≈ 500 ms`.
- **`stop_model` costs roughly double**, as with violawake: the stop pair runs
  its own featurizer, there is no shared backbone. Upstream suggests a live
  threshold of 0.5–0.6 rather than whatever a per-clip sweep says is optimal.

Startup rejects a pair that scores digital silence at or above your threshold.
That is not a hypothetical: a head exported with its own sigmoid gets squashed
into [0.5, 0.73] by the sigmoid applied here, which fires constantly rather
than never — the failure that looks like working software.

### Optional: the Silero VAD gate (`wakeword.vad_gate`)

Silero decides which frames are worth scoring, so an idle room stops paying for
the wakeword model. Off by default, because it trades CPU against recall and
only one side of that trade is cheap to measure.

What it costs, measured on an x86 dev box with single-threaded ORT:

| per 80 ms frame | |
|---|---|
| `pysilero-vad`, 2.5 × 512-sample chunks | 0.168 ms |
| violawake `process` | 0.887 ms |
| openWakeWord `predict` | 1.192 ms |

Silero runs on **every** frame, so the gate is only ahead above roughly a 19 %
skip fraction against violawake (14 % against openWakeWord). A quiet room is
far past that — a 30 s recording of a low noise floor scores 12 of 375 frames,
97 % suppressed — but a room with a television in it is not, because Silero
calls that speech and the gate stays open. **Re-measure on your own hardware
with `--probe-mic` before enabling it**; if `cpu_ms` does not drop, it is not
earning its place there.

Two things are not tunable, for the same reason:

- **`preroll_ms` cannot go below the engine's context** (violawake 1480 ms,
  wakeforge 500 ms; `null` takes the right one). Skipping a frame does not
  pause these models, it *splices* them: the streaming melspectrogram is
  computed over `tail(n_samples + 480)`, so a frame that never entered the ring
  puts audio from 80 ms earlier directly against the next one, and the mel
  frames across that seam look like a plosive. It takes 76 mel frames plus the
  head's own window to flush that out — longer than the wake phrase itself. The
  gate therefore replays a full context when it opens, and a short pre-roll
  would score the whole phrase on spliced audio while passing every test.
- **openWakeWord is rejected outright.** Its 2040 ms of context is 26 frames of
  replay, which on a Pi is longer than one frame period.

The gate never runs during playback (`bypass_while_speaking`): Silero calls our
own TTS speech, so it would be open anyway, and barge-in cannot afford the
32 ms of onset granularity that opening costs.

Recall is the half you have to measure yourself. `--score-wav` marks gated
frames so the sweep cannot silently credit the gate's suppression to the model,
and prints how much it skipped; A/B two config files over the same corpus and
keep the gate only if recall holds.

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
