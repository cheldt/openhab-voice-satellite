# Architecture Overview — openhab-voice-satellite

An offline-first voice satellite for openHAB: wakeword → record → STT → openHAB voice
interpreter → TTS, running as a single Python process on a Raspberry Pi 5 (4 cores) next
to the PipeWire daemon in the same user session.

- **Package:** `openhab_voice_satellite` (`src/` layout, `pyproject.toml`, Python ≥ 3.11)
- **Entry point:** console script `openhab-voice-satellite` → `__main__:main`
- **Config:** one YAML file validated by pydantic; secrets overridable by env
- **Audio:** capture is 16 kHz mono int16 in 80 ms frames (1280 samples), non-negotiable
  (Silero VAD, openWakeWord and whisper are all hardwired to 16 kHz); playback is
  rate-agnostic by design — earcon WAVs keep their file rate, cloud TTS arrives at 24 kHz
- **Concurrency:** one asyncio event loop, one GStreamer callback thread, a default
  executor for blocking model calls (whisper, piper)

---

## 1. Runtime topology

```
                        ┌───────────────────────── one Python process ─────────────────────────┐
                        │                                                                      │
  mic ── PipeWire ──────┼─▶ PipewireSource (gst thread)                                        │
        (pipewiresrc)   │        │ 80 ms int16 frames                                          │
                        │        ▼                                                             │
                        │   AudioBroadcaster ──┬──▶ wake_queue ──▶ App._interrupt_monitor       │
                        │   (fan-out, bounded) │                     │ wakeword / stop / duck   │
                        │                      │                     ▼                         │
                        │                      └──▶ round queue ──▶ Pipeline.run_interaction    │
                        │                          (per LISTENING)   │                         │
                        │                                            ├─ recorder + Silero VAD   │
                        │                                            ├─ Transcriber (executor) ─┼──▶ [Gemini | Deepgram]
                        │                                            ├─ OpenHABClient ──────────┼──▶ openHAB /rest/voice
                        │                                            └─ Speaker (piper/cloud) ──┼──▶ [Gemini | Deepgram]
                        │                                                       │               │
  speaker ◀── PipeWire ─┼──── PipewireSink (persistent FIFO + keepalive) ◀───────┘               │
        (pulsesink)     │                                                                       │
                        └───────────────────────────────────────────────────────────────────────┘
```

External dependencies: PipeWire/WirePlumber graph, openHAB REST API, optionally the
Gemini and Deepgram HTTP APIs. Everything else (wakeword, VAD, STT, TTS) runs locally.

## 2. Control flow

State machine (`state.py`), owned by `app.py`, advanced by `pipeline.py`:

```
IDLE ──wake──▶ LISTENING ──endpoint──▶ THINKING ──answer──▶ SPEAKING ──┐
  ▲                │                       │                    │     │ dialog.enabled
  └────────────────┴───────────────────────┴────────────────────┴─────┘  loops back to LISTENING
                        no-speech / error / barge-in                     (no wakeword needed)
```

- The wakeword monitor never stops running — it is the only always-on loop, and it is
  what makes barge-in possible during `THINKING` and `SPEAKING`.
- One interaction is exactly one cancellable `asyncio.Task`. Barge-in = `sink.stop()` +
  `task.cancel()`; the `finally` blocks release the mic subscription and delete the
  server-side conversation.
- Terminal events: `NO_SPEECH`, `PLAYBACK_DONE`, `ERROR` (`state.Event`).

---

## 3. Component catalog

All modules live under `src/openhab_voice_satellite/`.

### 3.1 Core / orchestration

| Component | Responsibility | Notable contract |
|---|---|---|
| `__main__.py` | CLI: `--config`, `--list-devices`, `--check`, `--probe-mic`, `--score-wav`, `--model`, `--engine`, `--compare`, `--positives`, `--negatives` | Every mode imports its stack lazily, so a missing GStreamer/openwakeword shows up as one failed step, not an import crash |
| `app.py` | Process wiring and the always-on wakeword monitor: builds detector/endpointer/transcriber, opens audio via `AsyncExitStack`, wraps local engines with cloud primaries, starts/cancels the interaction task | Capture starts **after** model loading (`start_capture=False` → `source.start()`): a live stream nobody services xruns itself out of PipeWire scheduling. After a barge-in cancel, listening resumes only when the trigger was `wake`, the state was SPEAKING **and** `barge_in.resume_listening` is set — otherwise the idle earcon plays and the system returns to IDLE |
| `pipeline.py` | One interaction: LISTENING → THINKING → SPEAKING, dialog follow-up rounds, earcon echo guard, utterance dumps, conversation lifecycle | A dialog is one server-side conversation (uuid per wake); TTS language locks to the first round's detection (follow-ups are too short to detect reliably). `conversation_started` is set **before** the interpreter POST is awaited, so a barge-in landing mid-POST still deletes a conversation the server may already have created; `close()` — called from the monitor's `finally` before the exit stack tears down the session the DELETEs need — drains those fire-and-forget tasks, bounded at 5 s |
| `state.py` | `State` and `Event` enums. No I/O | Single source of truth for the four states |
| `config.py` | YAML + pydantic models, `SAMPLE_RATE = 16000`, config-relative path resolution, cross-field validation | Env wins over file for `OPENHAB_TOKEN`, `GEMINI_API_KEY`, `DEEPGRAM_API_KEY`; cloud engine without a key and a `default_language` without a Piper voice are load-time errors. Unknown keys are silently ignored (pydantic's default `extra="ignore"`): a misspelled key falls back to its default without a word, and `audio.sample_rate` is a `ClassVar` deliberately so old configs carrying the key still load |

Helpers inside `app.py` worth naming, because they carry behavior rather than plumbing:

- `_CaptureHealth` — 10 s heartbeat: frame count vs expected fps, peak RMS, peak wake
  score, queue evictions, and `CaptureStats` deltas. Distinguishes "the graph under-fed
  us" from "we lost frames ourselves" (three distinct loss channels — see §3.2). A separate 10 s timeout (`MIC_STALL_WARN_S`) on
  the wake queue logs a loud stall warning and restarts the health window so the dead
  gap is not charged to the next one — two different timers that happen to share a value.
- `_DuckController` — duck-and-confirm: a pre-threshold score (0.35) during playback
  ducks the sink to 0.2 for ~1 s so the follow-up frames reach the detector cleanly.
- `_dump_wake_audio` — `$OVS_DUMP_WAKE` writes the 2.5 s of audio leading up to a
  detection (pre-roll only, nothing after it); `$OVS_DUMP_WAKE_SCORE` — only in addition
  to `$OVS_DUMP_WAKE`, it is inert alone — also captures near misses and verifier
  rejections, the latter gated on the candidate's stage-1 peak (the hard negatives for
  retraining).
- `_log_wake_detection` — logs the candidate's stage-1 peak, not the decayed score at
  the verdict frame, and the verifier's score with it, since accepts are otherwise
  invisible.
- `_resync_detector` — after a reset, abandons the mic backlog that outran the detector,
  otherwise our own TTS echo gets re-scored against the IDLE threshold.
- `_build_engines` / `_build_speaker` — cloud primaries get their own aiohttp session
  (openHAB's may have TLS verification disabled). Local STT stays loaded as the fallback
  (whisper is constructed unconditionally); local TTS becomes a `LazySpeaker` that only
  constructs piper on the first fallback, saving its RAM and ~3.5 s startup block.

### 3.2 Audio I/O (`audio/` subpackage)

| Component | Responsibility | Notable contract |
|---|---|---|
| `io.py` | `audio_io()` async context manager owning the source/sink pair; `verify_links()` | Sink construction failure closes the source even when it is already PLAYING (the probe path — the app opens with `start_capture=False`, so its source is not live yet); link check waits 3 s for WirePlumber to settle |
| `gst_source.py` | `PipewireSource`: `pipewiresrc → appsink`, chunked to fixed frames, `CaptureStats` (buffers, samples, PTS gaps, drops) | 5 s first-frame **warning** — capture itself never fails; `--check`'s `probe_capture` is the path that raises. PTS gaps under 1 ms are jitter, not lost audio; deferred, idempotent `start()`. Has its own 50-frame drop-oldest queue and counter (`CaptureStats.dropped`), independent of the broadcaster's — see the loss-accounting note below. Bus ERROR/EOS ends the frame stream with the `None` sentinel — see the capture-death note below |
| `gst_sink.py` | `PipewireSink`: persistent clock-free byte FIFO into `pulsesink`, byte-accounted `play()` completion, keepalive dither, wake-up preamble, `duck`/`unduck`/`stop` | `sync=false` with unstamped buffers (the appsrc's `format=time` is inert under `sync=false`) — clock-based scheduling was abandoned after repeated stack failures; `pipewiresink` is avoided because on PipeWire 1.2.x it wedges persistent streams and takes the capture stream down with it. `is_playing` is a deadline estimate, not a sink query: sound end + 0.3 s residual for the *requested* (not guaranteed) 200 ms pulse ring, cleared by flush so a barge-in reads quiet immediately — every echo-mitigation decision rides on it. A second `play()` flushes the first, whose `await` then returns normally having played an arbitrary prefix — this is what makes barge-in cheap, and what makes accidental overlap a silent truncation rather than a crash. A bus error is sticky: every later `play()` raises and the keepalive exits (logged), but the process keeps listening — see the capture-death note below. The keepalive task runs from construction, before `source.start()` |
| `gst_common.py` | `gst_init()`, `CLIENT_NAME`, s16 mono caps, `capture_description`, sync bus handler | `gi` imported only where genuinely needed. The sync bus handler must never call `set_state()` (GStreamer deadlocks); the only way off the posting thread is `call_soon_threadsafe`. `capture_description` is shared byte-for-byte by the app, `--check` and `--probe-mic` — which is what makes `--check` diagnostic of the real path — and pins the appsink to `max-buffers=0 drop=false` with no `queue` element: a queue in front of the appsink could only lose buffers where nothing counts them |
| `gst_devices.py` | `AudioNode`, `match_node` (pure, importable without PyGObject), `list_audio_nodes`, `resolve_node`, `verify_stream_links` (pw-dump peers; `parse_stream_peers` is its pure, unit-testable half), `probe_capture` | Device config is a case-insensitive substring of node name or description; `null` = default node; an unmatched name raises `ValueError` at startup — a typo aborts instead of silently falling back to the default node (that silent fallback is PipeWire's own behavior, which is why `verify_stream_links` exists). `_monitor_refs` keeps every DeviceMonitor alive for the process lifetime on purpose: premature finalization triggers GStreamer teardown criticals on PipeWire 1.2.x |
| `broadcast.py` | `AudioBroadcaster` fan-out to bounded `SubscriberQueue`s; `drain_stale()` | Drop-oldest under backpressure with a `dropped` counter — gaps are never zero-filled, since fabricated silence would feed the endpointer's silence window. A `None` sentinel ends every subscriber stream: pushed when the source ends and again by `stop()`, never consumed by `drain_stale`, so a consumer blocked in `get()` (the recorder) cannot hang past shutdown. `drain_stale` deliberately leaves `dropped` alone: lost-to-backpressure and abandoned-on-purpose are separate accounting |
| `chunker.py` | `FrameChunker`: arbitrary buffers → exact `frame_samples` frames | Copies, because gst buffers are unmapped after the callback returns |
| `wav.py` | `rms`, `read_wav_mono`, `write_wav`, `pcm_to_wav_bytes` | `rms` casts to float64: squaring int16 overflows |
| `earcons.py` | `Earcons`: wake/ack/error/idle, pre-decoded to PCM at startup | Missing file = warning + silent skip, but a file that is present yet corrupt (not 16-bit PCM, not a WAV) raises in `__init__` — a hard startup crash. `play()` catches `Exception` only, so `CancelledError` propagates: that is what makes an earcon barge-in-able |
| `source.py`, `sink.py` | `AudioSource` / `AudioSink` Protocols | The seam that keeps the app testable without a live graph. `frames()` is a plain `def` on purpose: an `async def` in the Protocol would demand a coroutine returning the iterator, not an async generator |

**Frame loss has three independent channels**, and the split is what makes
`_CaptureHealth`'s "the graph under-fed us" vs "we lost frames ourselves"
decidable: PTS gaps counted at the appsink (the graph skipped audio), the
source's own queue drops (`CaptureStats.dropped` — the event loop stalled),
and per-subscriber evictions (`SubscriberQueue.dropped` — one consumer fell
behind). Both queues default to 50 frames = 4 s at 80 ms. `drain_stale`
losses appear in none of these counters, by design.

**Capture death is fatal on purpose.** A capture bus ERROR or EOS pushes the
`None` sentinel through the broadcaster; the monitor raises
`CaptureClosedError`, `main()` exits non-zero, and the systemd unit
(`Restart=on-failure`) restarts the process with a fresh graph connection.
In-process recovery was rejected: the sync bus handler cannot call
`set_state()` (deadlock), and a PipeWire stream that has lost its scheduling
rarely comes back. Playback is deliberately asymmetric — a sink bus error is
sticky and fails every subsequent response, but the satellite keeps
listening.

### 3.3 Wakeword

| Component | Responsibility | Notable contract |
|---|---|---|
| `wakeword.py` | The engine-neutral contract: `WakewordProtocol`, `EdgeTrigger`, `BaseWakewordDetector`, `build_detector`, shared ORT `_session_options` | Engines supply **raw scores only**; thresholds, edge/patience and the speaking-threshold raise live here, because barge-in is tuned against all three. ORT sessions are single-threaded with spinning disabled |
| `wakeword_oww.py` | `OpenWakewordDetector`: openWakeWord ONNX backend, wake + optional stop model, `ncpu=1`, per-speaker verifier pickles | Startup fails loudly on the two ways openwakeword's positional key mapping breaks (shared basenames, multi-output models); a custom verifier **replaces** the base score, so thresholds change meaning |
| `wakeword_buffer.py` | `Int16Ring` plus `patch_preprocessor()`: two perf patches on openwakeword 0.6.0 internals — numpy ring instead of a deque of Python ints, and a snapshot restore instead of re-embedding 4 s of noise on every `reset()` | Version-gated and **fails open**: a mismatch logs a warning and keeps stock behavior |
| `verifier_mel.py` | `MelPcenFrontend`: librosa's STFT → mel → PCEN pipeline reimplemented byte-for-byte in numpy + scipy, output `(40, 151)` | The mel filterbank ships as data (`.npy`), not code; parity is asserted against golden fixtures exported where librosa exists |

**Two-stage detection.** Stage 1 (openWakeWord, per 80 ms frame) crosses a deliberately
low threshold for recall. Each `wake` is then held for `stage2.delay_ms` of further audio
and the last 1.5 s from the detector's own ring is re-scored by the mel-PCEN CNN
(`stage2.model` + `stage2.mel_basis`, always a pair). Measured on 5.5 h of continuous
speech, stage 2 rejects ~99 % of stage-1 triggers. `stop` is never deferred — a late stop
defeats the purpose — and it also wins over a verifier accept landing on the same frame.
The detector exposes `last_trigger_score` (the candidate's stage-1 peak — the score at
the verdict frame has already decayed), `last_verifier_score`, and `last_rejection` (set
only on the frame a rejection lands, for the dump gate).

The raw-audio ring behind `tail()` belongs to `BaseWakewordDetector`, not the engine:
it has to serve both the wake-audio dump and the verifier's window, and `process()` fills
it before anything else runs.

### 3.4 Speech (record / STT / TTS)

| Component | Responsibility | Notable contract |
|---|---|---|
| `recorder.py` | `record_utterance()`: drain the frame queue until VAD endpoint, no-speech timeout, or `max_utterance_s`; `NoSpeechError` on no speech and on a closed source | Round 0 passes `no_speech_timeout_s=None`, meaning `vad.no_speech_timeout_s`; follow-up rounds pass `dialog.followup_timeout_s` (§3.1). Two stall guards with distinct messages: a per-`get` 10 s wall-clock cap ("mic stalled" — the endpointer's own timeouts count *received* samples and can never fire on a silent queue) and an overall deadline of `no_speech_timeout_s + max_utterance_s + 10 s` ("listening window exhausted" — for a trickling mic whose rare frames keep resetting the per-get cap while the sample clock barely advances). A stall *mid-utterance* is not an error: with speech started and frames collected, it breaks and transcribes the partial utterance under a warning. Reads `frames.dropped` on exit and warns how much audio was spliced out — §3.2's never-zero-filled policy, enforced at the consumer; the counter is per-utterance because the queue is subscribed per round. Uses `asyncio.timeout`, not `wait_for` (3.11 gh-86296 swallows a cancel that races a completed `get()` — that cancel is a barge-in) |
| `vad.py` | `SpeechEndpointer`: Silero VAD over 512-sample (32 ms) chunks, trailing-silence endpointing, residual carry | Exposes `speech_started`, `endpoint_reached`, `elapsed_s` for the recorder loop. The engine is `pysilero-vad` 2.1.1's bundled silero v5 ONNX, whose own session already runs single-threaded — §5's ORT posture holds for VAD for free. One endpointer lives for the process lifetime (`app.py`); that is safe because `record_utterance` calls `reset()` on entry (silero state, residual, flags). `speech_started` is sticky: one 32 ms chunk over threshold disables the no-speech timeout for the round — endpointing, not the timeout, then ends it. `probability()` is public as the monkeypatch seam for `test_vad.py` |
| `stt.py` | `Transcriber`: faster-whisper on CPU, run in the executor, `Transcript(text, language)` | With more than one configured language the decode runs under whisper's own detection (`language=None`); only the *reported* language is remapped into `stt.languages` afterwards (best allowed entry of `all_language_probs`, else `tts.default_language`) — an out-of-set utterance returns foreign text under an allowed label, and that label locks the TTS voice for the whole dialog (§3.1). A single configured language skips the detection pass entirely; `cpu_threads` ≥ core count warns but is never coerced |
| `tts.py` | Shared TTS machinery: `split_sentences`, `tts_chunks` (400 chars), `play_pipelined`, `stream_synthesis` | Chunk N plays while N+1 is fetched/synthesized; a cloud failure after audio already played raises `PartialSpeechError` with the unspoken remainder. `play_pipelined` touches `chunks[0]` before any guard — a non-empty list is a caller-side invariant (all three callers check first). `split_sentences` splits on whitespace after `.!?:;`, so spaced German abbreviations ("z. B.", "u. a.") fragment into sub-sentence pieces — a prosody cost (and one cloud request per fragment), not a failure; "21.30 Uhr" has no whitespace and is safe. A prefetch that failed before a barge-in cancels the loop is never retrieved — asyncio logs "Task exception was never retrieved" at GC; log noise only |
| `piper_tts.py` | `PiperSpeaker`: per-language voices, sentence-level overlapped synthesis, RTF debug logging | `__init__` loads every configured voice up front — the RAM figure behind §3.1's `LazySpeaker`. Unknown language falls back to the default language's voice via a bare dict access that cannot `KeyError` only because a `default_language` without a voice is a load-time error (§3.1). Empty synthesis returns `rate=0`, which `play_pipelined` skips via `if len(pcm)` |

**Cancellation stops the await, never the executor thread.** `Transcriber.transcribe`
and `stream_synthesis`'s fetch both `run_in_executor`. A barge-in during THINKING
cancels the await while ctranslate2 keeps decoding on `stt.cpu_threads` threads; one
during piper playback lets the next sentence finish synthesizing. Both burn cores
exactly when the 80 ms wakeword cadence matters most. §5's "every interaction is one
task it can cancel" is true of the event-loop task only.

### 3.5 Cloud engines (optional, per direction)

| Component | Responsibility | Notable contract |
|---|---|---|
| `gemini.py` | `GeminiClient` + `GeminiTranscriber` / `GeminiSpeaker`: plain REST `generateContent` — STT via JSON mode with inline base64 WAV, TTS via the AUDIO modality; `check_model()` validates each configured model per direction for `--check` | API key travels in the `x-goog-api-key` header, never in a URL. The TTS rate is *parsed* from the response `mimeType` (`rate=`, default 24000) — deepgram's, by contrast, is *requested* via `tts_sample_rate`; the intro's "cloud TTS arrives at 24 kHz" is true of both only by default |
| `deepgram.py` | `DeepgramClient` + transcriber/speaker: Nova-3 `/v1/listen` (one configured language → `language=`, several → repeated `detect_language=`), Aura-2 `/v1/speak` raw linear16 | Key in the `Authorization` header; the per-language voice *is* the whole Aura-2 model name (`aura-2-viktoria-de`), sent as the `model` param. `check_auth()` validates the key only, so `--check` adds per-direction probes: 0.1 s of silence through `/v1/listen` for `stt_model`, one word through `/v1/speak` per configured voice |
| `cloud.py` | `raise_for_status`, `pick_voice` | Deliberately free functions: providers differ in auth, endpoints and error types, and keeping that visible beats a base class. `pick_voice` returning `None` (no voice for the language *or* the default) is a config error the speakers turn into a `CloudEngineError` — a silent, permanent local fallback — so, like `piper.voices`, the selected engine's `tts_voices` must contain `tts.default_language` at load time |
| `fallback.py` | `CloudEngineError`, `PartialSpeechError`, `FALLBACK_ERRORS`, `FallbackTranscriber`, `FallbackSpeaker`, `LazySpeaker` | `CancelledError` (barge-in) is not an `Exception` and passes through both wrappers untouched; `LazySpeaker` defers the multi-second local model load to the executor, memoized and shielded so a barge-in mid-load does not start a second one. `FALLBACK_ERRORS` is a closed set (HTTP, timeout, JSON decode); payload-decode failures inside the providers are wrapped in their own error types so they stay inside it |

Cloud is per direction: `stt.engine` and `tts.engine` are chosen independently. Local
STT stays loaded as the fallback; local TTS becomes a `LazySpeaker` and loads on the
first fallback.

**Cloud language contract.** Both transcribers clamp a detected label outside
`stt.languages` to `tts.default_language`, but they differ when detection is missing
entirely: gemini (non-JSON response) falls back to `default_language`, deepgram
(missing or non-string `detected_language`) to `languages[0]`. Both are coarser than
whisper's remap, which picks the best *allowed* entry of `all_language_probs` (§3.4),
and the resulting label locks the TTS voice for the whole dialog (§3.1). A gemini STT
response that fails JSON parsing is used verbatim as the transcript under one warning —
an apology or a markdown fence then reaches the openHAB interpreter as a command; that
beats losing the utterance, but it is a deliberate trade.

### 3.6 openHAB integration

| Component | Responsibility | Notable contract |
|---|---|---|
| `openhab.py` | `make_session` (honors `verify_ssl`, sets a session-wide `response_timeout_s` bound so no request can inherit aiohttp's 5-minute default), `OpenHABClient.ping` / `send_command` / `end_conversation`, `OpenHABTimeoutError` | `POST /rest/voice/interpreters`, `text/plain; charset=utf-8` in, `text/plain` out; the answer is the HTTP body, and an empty 200 is a silent round (every speaker returns early on empty text), not an error. Both query params are conditional: `llm_tools: null` omits `?llmTools=`, and `?conversation=` is sent only when `dialog.enabled` produced a uuid. Auth is `Authorization: Bearer` from `OPENHAB_TOKEN`-over-`api_token`; with neither, no header at all and no load-time error — unlike the cloud keys, because anonymous openHAB is a legitimate deployment. `DELETE /rest/voice/conversations/{id}` is best-effort and never raises except `CancelledError`, re-raised so `Pipeline.close()`'s drain bound (one `CONVERSATION_END_TIMEOUT_S` plus 1 s) can cancel a hung DELETE; a 404 — barge-in before the server created the conversation — logs at debug, other errors at warning |

**Failure posture — the inverse of §3.5's.** openHAB is the brain, so no fallback
exists. `OpenHABTimeoutError` is the only named failure (error earcon +
`Event.ERROR`); every other HTTP/connection error reaches the pipeline's generic
handler as an untyped crash behind the same earcon. The interpreter's error body is
logged (truncated at 500 chars) but never spoken; 401/403 get a dedicated log line
naming `api_token`/`OPENHAB_TOKEN`, the single most likely misconfiguration. `ping` —
`--check`'s probe — GETs `/rest/voice/interpreters`, not `/rest/`: the interpreters
list requires auth when security is enabled and proves the voice subsystem is present,
so a bad token fails at `--check` instead of at the first utterance. The shipped
`response_timeout_s: 30` is the knob to raise when the interpreter is an LLM doing
several tool calls.

### 3.7 Diagnostics and evaluation

| Component | Responsibility |
|---|---|
| `selftest.py` | `--check`: audio devices + real capture probe, wakeword (2.5 s of frames so the engine's context window fills), VAD, whisper, piper, cloud probes (gemini: model metadata per configured direction; deepgram: key check plus a silence `/v1/listen` and a one-word `/v1/speak` per voice), openHAB interpreters probe (proves the token and the voice subsystem, not just reachability). Each check imports lazily and reports independently |
| `probe.py` | `--probe-mic`: 30 s field diagnostic on the same source/sink the app uses — per-second RMS/peak/wake score plus wall and process CPU ms per frame against the `frame_ms` budget, earcons at t=8 s and t=18 s, capture written to `diagnose_capture.wav`. The module docstring is the interpretation guide (silent node, wrong node, clock mismatch, playback poisoning capture, sleeping speaker) |
| `bench.py` | Offline wakeword evaluation through `build_detector` and the real `EdgeTrigger`: `--score-wav` (per-file distribution + threshold × patience sweep over 15 thresholds × patience 1–3), `--compare MODEL` (two candidates on identical frames), `--positives/--negatives` (recall vs false accepts per hour — the promotion gate). Appends silence after each file so a deferred stage-2 verdict still flushes |

---

## 4. Repository assets

| Path | Role |
|---|---|
| `config.example.yaml` | Annotated template for every config key; `config.yaml` is gitignored |
| `scripts/download_models.py` | Fetch openWakeWord models, Piper voices, whisper warmup |
| `scripts/make_earcons.py` | Generate the sine-sweep earcons into `sounds/` |
| `sounds/` | `wake.wav`, `ack.wav`, `error.wav`, `idle.wav` |
| `models/piper/` | Piper voices (de_DE thorsten, en_GB alba, en_US lessac) |
| `models/wakeword/` | openWakeWord wake/stop models, custom `shodan`/`showdaan`/`shohdaan` candidates, stage-2 verifiers with their `.config.json`, `verifier_melfb_40.npy`, `verifier_frontend_golden.npz`, plus the `wakeforge/` training artifacts |
| `deploy/openhab-voice-satellite.service` | systemd **user** unit (PipeWire lives in the user session; needs `loginctl enable-linger`). Sets `HF_HOME`, `OMP_WAIT_POLICY=PASSIVE`, `OPENBLAS_NUM_THREADS=1`. `Restart=on-failure` pairs with the non-zero capture-death exit (§3.2) |
| `deploy/install.md` | apt packages, openwakeword `--no-deps` pin, AEC config, 16 kHz graph clock pinning, powered-speaker preamble notes |
| `tests/` | 28 `test_*.py` modules (~one per source module) plus `conftest.py`, `fakes.py`, `wakeword_stubs.py`; GStreamer tests skip without PyGObject; `test_verifier_mel.py` checks librosa parity against golden fixtures |
| `.github/workflows/python-app.yml` | uv, Python 3.11–3.14 matrix, flake8 + pytest (runner has no PipeWire; live-audio tests skip) |
| `.github/workflows/security-scan.yml` | Nightly `pip-audit` over the locked dependency set |

External but coupled: **ultiwake** (the training pipeline that produces the stage-2
verifier `.onnx`, its mel filterbank `.npy` and the golden frontend fixtures, and refuses
to export a pair that has not passed its false-accept gate). `--score-wav` is driven by
its `gate.sh` / `smoke.sh`, which is why `--engine` stays a flag despite having one
choice.

---

## 5. Cross-cutting concerns

**CPU budget.** Four cores on a Pi 5, and the 80 ms wakeword cadence must never miss —
missing it means going deaf to barge-in. Hence: `ncpu=1` for openWakeWord, explicit
single-thread ORT sessions with spinning disabled, `stt.cpu_threads: 3` (ctranslate2 runs
with the GIL released and saturates its threads), `OMP_WAIT_POLICY=PASSIVE` and
`OPENBLAS_NUM_THREADS=1` in the unit file, and blocking model calls pushed to the
executor.

**Echo mitigation, in layers.** `threshold_speaking` raised (raised, not disabled — the
stop word and barge-in must survive playback); duck-and-confirm on a pre-threshold score;
the post-earcon echo guard that drops the backlog captured while an earcon was audible
(keeping the last 300 ms, which may hold the user's speech onset); stale-backlog
abandonment after every detector reset; keepalive dither so the amp never auto-standbys
and clips the first sound. Hardware AEC (PipeWire `libpipewire-module-echo-cancel`) is
documented in `deploy/install.md` as the optional real fix.

**Cancellation model.** The wakeword monitor is the only long-lived loop; every
interaction is one task it can cancel. The cancel reaches the await, not executor
threads already decoding or synthesizing (§3.4). Cleanups that must outlive a cancel
(conversation DELETE, the lazy TTS load) run as their own shielded/tracked tasks.

**Failure posture.** Cloud errors fall back to local, per request, with only the unspoken
remainder re-spoken. Frame loss is counted and reported, never hidden behind fabricated
silence. Perf patches on third-party internals are version-gated and fail open. Config
mistakes that would be silent at runtime (removed engine name, missing cloud key, voice
without a language, a stage-2 model without its filterbank, an audio device name matching
no PipeWire node) are load-time errors, and a speaking threshold below the idle one is a
warning. Capture death exits non-zero for systemd to restart (§3.2); playback death fails
responses but keeps listening.

**Security notes.** openWakeWord custom verifier `.pkl` files are unpickled at startup —
treat them as executable code. Cloud keys go in headers, never query strings.
`openhab.verify_ssl: false` logs a warning. Debug dumps (`$OVS_DUMP_UTTERANCES`,
`$OVS_DUMP_WAKE`, `$OVS_DUMP_WAKE_SCORE`) write raw room audio to disk.

## 6. Config surface

`audio` (devices, `frame_ms`, wake-up preamble) · `wakeword` (model, thresholds, patience,
per-speaker verifiers, `stage2`) · `vad` (threshold, `silence_ms`, `no_speech_timeout_s`,
`max_utterance_s`) · `stt` (engine, model, `compute_type`, `cpu_threads`, `beam_size`,
`languages`) · `openhab` (url, token, `llm_tools`, `response_timeout_s`, `verify_ssl`) ·
`tts` (engine, `default_language`) · `piper.voices` · `gemini.*` · `deepgram.*` ·
`barge_in.resume_listening` · `dialog` (enabled, `followup_timeout_s`, earcon) ·
`earcons.*` · `logging.level`.

Relative paths in the config resolve against the config file's directory (pretrained
openWakeWord phrase names like `hey_jarvis` pass through verbatim); a `Config` built
directly in tests keeps its paths as written.
