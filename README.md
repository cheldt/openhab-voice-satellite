# openhab-voice-satellite

Local voice assistant for openHAB on a Raspberry Pi 5. Offline by default:
wakeword → speech-to-text → openHAB chat → text-to-speech, all on-device.
Cloud STT/TTS (Gemini, Deepgram) can be enabled per direction and falls back
to the local engines automatically on any error.

```
 mic ──▶ openWakeWord ──▶ record (Silero VAD) ──▶ faster-whisper (de/en auto)
                                                        │ transcript
                                                        ▼
 speaker ◀── Piper TTS ◀─── answer text ◀── openHAB voice interpreter (HLI)
```

(The STT and TTS boxes can each be swapped for a cloud engine; the local
engine stays loaded and takes over per request when the cloud call fails.)

- **Wakeword**: openWakeWord (`hey_jarvis` by default), always listening,
  with an optional stage-2 verifier re-scoring every trigger
  (see [deploy/install.md](deploy/install.md)).
- **STT**: faster-whisper `small` int8, auto-detects German/English.
  Optional cloud STT via `stt.engine: "gemini"` or `"deepgram"` (Nova-3);
  whisper remains loaded and any cloud failure degrades to it for that
  request.
- **Chat**: posts the transcript to openHAB's voice interpreter endpoint
  (`/rest/voice/interpreters`) and speaks the plain-text answer.
- **TTS**: Piper, per-language voice, sentence-streamed (starts speaking
  while the rest is still synthesizing) and much faster than real time on a
  Pi 5. Default voices are `de_DE-thorsten-medium` and `en_GB-alba-medium`
  (female, Scottish English).
  Cloud TTS via `tts.engine: "gemini"` or `"deepgram"` (Aura-2) with piper
  as the automatic local fallback; if Deepgram fails mid-utterance, only the
  unspoken remainder is re-synthesized locally.
- **Dialog mode**: after each answer the mic reopens for a follow-up — no
  wakeword needed (`dialog.enabled`, ends after `dialog.followup_timeout_s`
  of silence). Chat context is kept server-side via a conversation id.
- **Barge-in**: saying the wakeword during processing or playback cancels the
  current interaction immediately (and by default listens for a new command).
  A custom-trained "stop" model can be added via `wakeword.stop_model`.
- **Earcons**: short wake/ack/error/idle tones (`sounds/*.wav`, tracked in the
  repo and regenerable via `scripts/make_earcons.py`; swappable, disable via
  `earcons.enabled`).

## State machine

```
IDLE ──wake──▶ LISTENING ──endpoint──▶ THINKING ──response──▶ SPEAKING ──done──▶ IDLE
                  │ 8s silence             │ wake/stop            │ wake/stop     │ dialog mode:
                  ▼                        ▼                      ▼               ▼ follow-up
                IDLE                     IDLE            LISTENING or IDLE     LISTENING
```

With dialog mode enabled, "done" loops back to LISTENING for a follow-up
question; the conversation ends (→ IDLE) after `dialog.followup_timeout_s`
of silence.

## Quick start (development machine)

Audio I/O needs GStreamer + PyGObject; install the system packages first
(see [deploy/install.md](deploy/install.md) for the apt list).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,gst]"
.venv/bin/pip install --no-deps 'openwakeword==0.6.0'   # see note in pyproject.toml
.venv/bin/python scripts/download_models.py
cp config.example.yaml config.yaml             # edit devices + openHAB url/token
chmod 600 config.yaml                          # it will hold live credentials
.venv/bin/openhab-voice-satellite --list-devices
.venv/bin/openhab-voice-satellite --check
.venv/bin/openhab-voice-satellite
```

If the wakeword or audio path misbehaves in the field,
`openhab-voice-satellite --probe-mic` runs a 30 s diagnostic: per-second mic
RMS/peak plus the wakeword scores (stop and stage-2 verifier columns appear
when configured), WAKE/STOP marks taken from the detector's actual verdicts —
verifier included, so a mark is what the app would have acted on — earcon
playback through the configured output, stream-link verification, and a
`diagnose_capture.wav` dump of exactly what the app heard. It exits non-zero
when the capture stream stalls, so it can gate scripts like `--check` does. Speech at the intended distance should read roughly
1000–5000 RMS — openWakeWord does no input normalization, so a quiet mic
degrades recall in a way no threshold can compensate for.

To judge a wakeword *model* rather than the audio path, `--score-wav` replays
recorded 16 kHz mono WAVs through the configured engine and prints, per file,
the score distribution and how many detections the app's own decision rule
would have emitted:

```bash
# what does this model do on audio that must never fire?
.venv/bin/openhab-voice-satellite --score-wav recordings/negatives/

# same frames through a second model, so two candidates are compared fairly
.venv/bin/openhab-voice-satellite --score-wav recordings/negatives/ \
    --compare models/wakeword/my_wake.onnx

# the promotion gate: recall against false accepts per hour
.venv/bin/openhab-voice-satellite --positives recordings/wake/ \
    --negatives recordings/room/
```

The gate reports the best threshold/patience pair, or says
`NO CLEAN OPERATING POINT` when no setting reaches recall without false
accepts — which is what a model needs retraining, not retuning, looks like. A
threshold high enough to reject everything is never counted as clean. Record
positives in the voice and room that will actually use them: a model trained
on your voice scores near zero on synthesized speech, so a TTS corpus is only
valid as the negative half.

Two things to read carefully. The sweep replays `EdgeTrigger` over stage-1
scores, so on a config with `wakeword.stage2` set it counts stage-1 triggers,
most of which the verifier then swallows — every report therefore also carries
a `live` count, which is what the detector itself returned, and `--compare`
refuses to name a winner from the sweep, pointing at the live lines instead.
And `--compare` paths resolve against the working directory, not against the
config file.

`--compare` carries `wakeword.stage2` onto both columns: two models are
compared as the two-stage systems they would actually be deployed as. Measured
on 5.48 h of continuous speech, the same verifier rejects 99 % of stage-1
triggers whichever head produced them.

Tests: `.venv/bin/pytest` (fast; the GStreamer tests skip without PyGObject).

Three env vars help debug in the field. Two take a directory:
`OVS_DUMP_UTTERANCES` writes the recorded utterance (everything after the
wakeword), and `OVS_DUMP_WAKE` writes the 2.5 s of audio *leading up to* a
detection — pre-roll only, nothing after the verdict frame, which is the only
way to collect real false accepts. The third, `OVS_DUMP_WAKE_SCORE=0.3`, takes
a **score**, not a path: it adds near misses (frames that almost fired) and
verifier rejections to what `OVS_DUMP_WAKE` writes, and does nothing on its
own. Those rejections are the hard negatives a retrain needs.

Note on logs: at the shipped `logging.level: INFO` every transcript and every
openHAB answer is written to the journal, so the household's spoken-command
history accumulates there under the journal's own retention. Set
`logging.level: WARNING` to stop that (at the cost of the transcript lines
that make field debugging possible), or bound it with
`journalctl --user --vacuum-time=`.

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
   separate `openhab-llmtool-websearch` add-on (companion repo). Build it (`mvn package`), drop the jar into openHAB's `addons/`
   folder and configure the SearxNG base URL in MainUI under
   Settings → Other Services → Web Search LLM Tool. The SearxNG instance
   must have the JSON output format enabled (`settings.yml`:
   `search.formats` includes `json`). The LLM then decides per request
   whether to answer from item states or search the web.

The service posts the transcript as `text/plain` to
`/rest/voice/interpreters?llmTools=...` and speaks the plain-text response.
With dialog mode enabled every wake word starts a server-side conversation:
a fresh uuid is sent as the `conversation` query parameter with each request
so openHAB keeps the chat context, and when the conversation ends (follow-up
silence or barge-in) it is deleted via
`DELETE /rest/voice/conversations/<id>`.

## Configuration

Everything lives in one YAML file — see the extensively commented
[config.example.yaml](config.example.yaml). Highlights:

| Key | Meaning |
|---|---|
| `audio.input_device` / `output_device` | substring of a PipeWire node name or description (`--list-devices`); `null` = default node |
| `audio.wakeup_preamble_ms` / `wakeup_preamble_idle_s` | ramped-noise lead-in that wakes powered speakers whose signal-sensing mute swallows the first sound after an idle period (details in [deploy/install.md](deploy/install.md)) |
| `wakeword.engine` | `openwakeword`, the only engine; kept so a config naming a removed one is rejected rather than silently reinterpreted |
| `wakeword.model` | pretrained openWakeWord name, or a path to a `.onnx` you trained |
| `wakeword.threshold_speaking` | raised threshold while our own output is audible (echo mitigation) |
| `wakeword.stop_threshold_speaking` | same for the stop model, which by definition runs during playback; `null` = reuse `stop_threshold` |
| `wakeword.patience` / `stop_patience` | consecutive frames above threshold before firing; `2` rejects single-frame spikes for 80 ms of latency |
| `wakeword.verifier_model` / `stop_verifier_model` | optional per-speaker openWakeWord custom verifier; **unpickled at startup**, see [deploy/install.md](deploy/install.md) |
| `wakeword.stage2.*` | engine-neutral second stage: a mel-PCEN CNN re-scoring the 1.5 s behind each trigger, so stage 1 can run low for recall. Unset = single stage |
| `stt.engine` | `local` (faster-whisper), `gemini` or `deepgram` (cloud STT, falls back to local on failure) |
| `stt.model` | `small` (default) or `base` for lower latency |
| `stt.languages` | language candidates for detection (default `[de, en]`); a single entry skips whisper's per-utterance language-detection pass — recommended on constrained boxes |
| `tts.default_language` | fallback language/voice when detection is inconclusive (default `de`) |
| `openhab.verify_ssl` | set `false` for self-signed HTTPS certificates |
| `tts.engine` | `piper` (local), `gemini` or `deepgram` (cloud TTS, falls back to piper on failure) |
| `gemini.api_key` | Google Gemini API key; env var `GEMINI_API_KEY` wins over the file |
| `gemini.tts_voices` | prebuilt Gemini voice name per language (e.g. `de: Kore`) |
| `deepgram.api_key` | Deepgram API key; env var `DEEPGRAM_API_KEY` wins over the file |
| `deepgram.tts_voices` | Aura-2 model per language (e.g. `de: aura-2-viktoria-de`) |
| `barge_in.resume_listening` | wakeword during playback → listen for new command |
| `dialog.enabled` | every answer re-opens the mic for a follow-up (no wakeword); context is kept server-side via a conversation id |
| `dialog.followup_timeout_s` | follow-up silence that ends the conversation (first turn uses `vad.no_speech_timeout_s`) |
| `piper.voices` | Piper `.onnx` model path per language code (engine `piper` + cloud fallback) |

## Barge-in and echo

During playback the mic hears the speaker. Mitigations built in: raised
wakeword threshold for as long as the sink reports audio is audible (which
covers earcons in any state, and stops the moment a barge-in flushes
playback), automatic volume ducking when the detector starts to trigger, and
abandoning the mic backlog that piled up during a cancel so our own tail is
never re-scored. For robust hands-free interruption, run PipeWire's WebRTC
echo canceller and point `audio.input_device` / `audio.output_device` at its
echo-cancel nodes (they appear in `--list-devices` like any other PipeWire
node) — setup in [deploy/install.md](deploy/install.md).

## License

[MIT](LICENSE)
