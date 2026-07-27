# stt-proxy

Local voice assistant for openHAB on a Raspberry Pi 5. Fully offline audio
processing: wakeword → speech-to-text → openHAB chat → text-to-speech.

```
 mic ──▶ openWakeWord ──▶ record (Silero VAD) ──▶ faster-whisper (de/en auto)
                                                        │ transcript
                                                        ▼
 speaker ◀── Kokoro TTS ◀── answer text ◀── openHAB voice interpreter (HLI)
```

- **Wakeword**: openWakeWord (`hey_jarvis` by default), always listening.
- **STT**: faster-whisper `small` int8, auto-detects German/English.
- **Chat**: posts the transcript to openHAB's voice interpreter endpoint
  (`/rest/voice/interpreters`) and speaks the plain-text answer.
- **TTS**: Kokoro-82M (via kokoro-onnx), per-language model + voice,
  sentence-streamed (starts speaking while the rest is still synthesizing).
  English uses the official model (`bf_emma` by default); German uses the
  community [Kokoro-82M-ONNX-German-Martin](https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin)
  model (Apache 2.0, single voice `martin`). Synthesis on a Pi 5 runs at
  roughly real-time speed, so expect a short delay before the first sentence.
- **Barge-in**: saying the wakeword during processing or playback cancels the
  current interaction immediately (and by default listens for a new command).
  A custom-trained "stop" model can be added via `wakeword.stop_model`.

## State machine

```
IDLE ──wake──▶ LISTENING ──endpoint──▶ THINKING ──response──▶ SPEAKING ──done──▶ IDLE
                  │ 8s silence             │ wake/stop            │ wake/stop
                  ▼                        ▼                      ▼
                IDLE                     IDLE            LISTENING or IDLE
```

## Quick start (development machine)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install --no-deps openwakeword   # see note in pyproject.toml
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/make_earcons.py
cp config.example.yaml config.yaml             # edit devices + openHAB url/token
.venv/bin/stt-proxy --list-devices
.venv/bin/stt-proxy --check
.venv/bin/stt-proxy
```

Tests: `.venv/bin/pytest -m "not slow"` (fast, no models needed beyond VAD).

Pi installation + systemd service: see [deploy/install.md](deploy/install.md).

## openHAB setup

1. Configure a voice interpreter (human language interpreter) in openHAB
   that answers free-text questions — e.g. an LLM-backed interpreter.
   `openhab.llm_tools` is passed through as the `llmTools` query parameter;
   set it to `null` to omit. openHAB 5.2 ships `item-send-command`,
   `item-get-state` and `get-date-time`; unknown tool ids are ignored with
   a server-side warning.

2. Create an API token (profile → API tokens) and put it in `config.yaml`
   (`openhab.api_token`) or the `OPENHAB_TOKEN` env var.

3. Optional — web search: the `web-search` tool id is provided by the
   separate [openhab-llmtool-websearch](../openhab-llmtool-websearch)
   add-on. Build it (`mvn package`), drop the jar into openHAB's `addons/`
   folder and configure the SearxNG base URL in MainUI under
   Settings → Other Services → Web Search LLM Tool. The SearxNG instance
   must have the JSON output format enabled (`settings.yml`:
   `search.formats` includes `json`). The LLM then decides per request
   whether to answer from item states or search the web.

The service posts the transcript as `text/plain` to
`/rest/voice/interpreters?llmTools=...` and speaks the plain-text response.

## Configuration

Everything lives in one YAML file — see the extensively commented
[config.example.yaml](config.example.yaml). Highlights:

| Key | Meaning |
|---|---|
| `audio.input_device` / `output_device` | substring of the device name (`--list-devices`) |
| `wakeword.model` | pretrained openWakeWord name or path to custom `.onnx` |
| `wakeword.threshold_speaking` | raised threshold while TTS is audible (echo mitigation) |
| `stt.model` | `small` (default) or `base` for lower latency |
| `openhab.verify_ssl` | set `false` for self-signed HTTPS certificates |
| `barge_in.resume_listening` | wakeword during playback → listen for new command |
| `dialog.enabled` | interpreter answer containing `?` re-opens the mic for a follow-up (no wakeword) |
| `dialog.max_turns` | max follow-up rounds per interaction |
| `dialog.context_mode` | `verbatim` sends the follow-up as-is; `stitch` sends the full dialogue transcript (for stateless interpreters) |
| `tts.voices` | Kokoro model/voices/voice/lang/speed per language code |

## Barge-in and echo

During playback the mic hears the speaker. Mitigations built in: raised
wakeword threshold in SPEAKING, automatic volume ducking when the detector
starts to trigger. For robust hands-free interruption, run PipeWire's WebRTC
echo canceller and point the audio devices at it — setup in
[deploy/install.md](deploy/install.md).
