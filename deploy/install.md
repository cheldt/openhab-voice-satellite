# Installing openhab-voice-satellite on a Raspberry Pi 5

Target: Raspberry Pi OS Bookworm 64-bit, Python 3.11–3.13 (kokoro-onnx does
not support 3.14 yet).

## 1. System packages

```bash
sudo apt update
sudo apt install python3-venv libportaudio2 alsa-utils
```

## 2. Get the code and install

```bash
sudo mkdir -p /opt/openhab-voice-satellite && sudo chown $USER /opt/openhab-voice-satellite
git clone https://github.com/cheldt/openhab-voice-satellite.git /opt/openhab-voice-satellite   # or rsync the project over
cd /opt/openhab-voice-satellite
python3 -m venv .venv
.venv/bin/pip install -e .
# openwakeword is installed without dependencies on purpose — its metadata
# demands tflite-runtime, which has no wheels for current Pythons; the ONNX
# backend used here does not need it:
.venv/bin/pip install --no-deps openwakeword
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
.venv/bin/openhab-voice-satellite --list-devices     # find your mic + speaker names
$EDITOR config.yaml                    # devices, openHAB url + token
```

Create the openHAB side (see README section "openHAB setup"): a configured
voice interpreter that answers free text, plus an API token (openHAB UI ->
profile -> API tokens).

## 5. Self-test

```bash
OPENHAB_TOKEN=... HF_HOME=/opt/openhab-voice-satellite/models/hf .venv/bin/openhab-voice-satellite --check
```

All six checks (audio devices, wakeword, VAD, whisper, kokoro/piper, openHAB REST)
must print `ok`. The whisper line also warms the model cache, so the first
real interaction is not slow.

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
    args = { library.name = aec/libspa-aec-webrtc }
  }
]
EOF
systemctl --user restart pipewire
```

Then point `audio.input_device` in `config.yaml` at the new
"Echo-Cancel Source" and `audio.output_device` at the "Echo-Cancel Sink".
This requires running openhab-voice-satellite as a **user** unit (it must live in the same
session as PipeWire) — which is the default unit shipped here.
