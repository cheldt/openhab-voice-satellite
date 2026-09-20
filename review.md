# Project review — openhab-voice-satellite (branch feat/two-stage-wakeword)

*Full-sweep review, 2026-08-23. Method: 10 parallel lens reviewers (wakeword, audio, orchestration, config, speech, cloud/openHAB, diagnostics, tests, docs, security) over the complete repo at commit 608ab47; every non-info finding adversarially verified against the code (two independent verifiers for high severity); refuted claims dropped. Report only — no fixes applied.*

This was a full-project sweep of a codebase that has already absorbed several per-area review passes, and the results reflect that: the core runtime paths (capture, wakeword decision layer, pipeline orchestration, fallback machinery) are largely sound, and most of what survived verification is contract drift — code that quietly disagrees with its own documented derivations — plus hardening gaps and test blind spots rather than routine crashes. Still, 34 findings survive (1 high, 10 medium, 23 low, after merging five remaining near-duplicate pairs), plus 6 info observations. The three most important: **(1)** a module-import call to `metadata.version("openwakeword")` that kills pytest collection in every CI job on every PR (masked on the dev box, where the package is manually installed); **(2)** the end-of-stream sentinel in the capture source is the only queue put that can be lost to `QueueFull`, silently defeating the deliberate "exit non-zero so systemd restarts us" contract and leaving the satellite permanently deaf; **(3)** a cluster of arithmetic-vs-documentation mismatches in the two-stage wake path — the stage-2 verifier scores one frame earlier than every doc and the ultiwake calibration replay say it does, and the earcon echo guard keeps ~200 ms of still-audible earcon tail that its own comment claims it excludes.

## Statistics

| Stage | Count |
| --- | --- |
| Raw findings | 55 |
| After dedup | 52 |
| Refuted in verification | 7 |
| **Survivors** | **45** — 1 high, 11 medium, 27 low, 6 info |

This report merges five remaining near-duplicate pairs (same defect seen through different lenses), yielding **34 numbered findings**: 1 high, 10 medium, 23 low, plus the 6 info observations. All survivors are verdict CONFIRMED.

## High

### 1. Module-import call to `metadata.version` crashes pytest collection in CI

**Where:** `tests/test_wakeword_buffer.py:208` · **Lens:** tests · **Verdict:** CONFIRMED

The `skipif` decorator evaluates `_real_models()` at import time, and `_real_models()` (line 202) calls `metadata.version("openwakeword")` without catching `PackageNotFoundError`. openwakeword is deliberately excluded from pyproject.toml and uv.lock (its metadata hard-requires tflite-runtime), so the CI environment provisioned by `uv sync --extra dev --extra gst` does not have it and the whole test module fails to collect. Verified by live simulation: pytest reports "Interrupted: 1 error during collection" and runs zero tests. The function is new on this branch (main has no `_real_models`), and the dev box masks it because openwakeword 0.6.0 is manually installed there.

**Failure scenario:** This branch is pushed as a PR to main → the Python application workflow runs `uv run pytest` in an env without openwakeword → importing `tests/test_wakeword_buffer.py` raises `importlib.metadata.PackageNotFoundError` → the entire run is interrupted at collection (299 collected tests never execute) → CI is red on every matrix job (3.11–3.14) for every PR until fixed, and no actual test result is produced.

**Fix direction:** Catch `importlib.metadata.PackageNotFoundError` inside `_real_models()` and return `[]` so the test skips instead of killing collection.

## Medium

### 2. End-of-stream sentinel lost to QueueFull; capture death never reaches consumers

**Where:** `src/openhab_voice_satellite/audio/gst_source.py:218` · **Lens:** audio · **Verdict:** CONFIRMED

`_end_stream` delivers the `None` sentinel via `call_soon_threadsafe(self._queue.put_nowait, None)` with no fallback for a full queue. Every frame-delivery path (`_put_frames`, `AudioBroadcaster._put_drop_oldest`) evicts the oldest entry before putting, but the one item that must never be lost — the sentinel signaling bus ERROR/EOS — is the only put that can raise `asyncio.QueueFull`. The exception is swallowed by the loop's callback exception handler, nothing retries, and the stream never ends: `broadcaster._run` blocks forever in `queue.get()`, subscribers never see `None`, and app.py's `CaptureClosedError` exit path (the deliberate "exit non-zero so systemd restarts us" contract from commit 908eda1) is silently defeated. The process stays alive but permanently deaf, logging only periodic "no mic frames for 10s" warnings.

**Failure scenario:** The event loop stalls ~4 s+ while capture runs (the exact condition the drop-oldest code in `_put_frames` documents as real); pending `_put_frames` callbacks execute FIFO on resume and fill the 50-slot queue via eviction. A bus ERROR or EOS arriving during/just after the stall (USB mic unplug, PipeWire node loss) enqueues `_end_stream` behind them; its `put_nowait(None)` hits a full queue, raises QueueFull, and the sentinel is dropped. The broadcaster drains the 50 stale frames and waits forever; `CaptureClosedError` is never raised, systemd never restarts the unit, and the satellite is deaf until someone notices.

**Fix direction:** Route the sentinel through a drop-oldest put (evict one frame when full before `put_nowait(None)`, mirroring `_put_frames` / `_put_drop_oldest`) so end-of-stream delivery can never raise.

### 3. `verify_ssl: false` sends the openHAB bearer token over unauthenticated TLS

**Where:** `src/openhab_voice_satellite/openhab.py:33` · **Lens:** security · **Verdict:** CONFIRMED

`make_session` builds `aiohttp.TCPConnector(ssl=False)` when `openhab.verify_ssl` is false, which disables both certificate validation and hostname checking; every request then carries `Authorization: Bearer <api_token>` (openhab.py:51) plus all voice-command traffic. The live config.yaml runs exactly this (`verify_ssl: false` against `https://ohab.shodan.io`), and the codebase offers no secure alternative for self-signed servers (no `ca_cert` or fingerprint-pinning option; the OS trust store is a theoretical workaround the docs never mention), so any self-signed deployment is forced into fully unverified TLS. Docs in three places frame the flag as "accepts self-signed certificates," understating that it accepts *any* certificate.

**Failure scenario:** Attacker on the same LAN ARP-spoofs or DNS-poisons the path to the openHAB host and terminates the TLS connection with their own certificate; aiohttp accepts it silently, the attacker harvests the bearer token from the first request and replays it for full openHAB REST control (item-send-command = physical actuation of the house), plus reads every spoken command and answer.

**Fix direction:** Add an `openhab.ca_cert` (and/or certificate-fingerprint) option and build the connector with `ssl.create_default_context(cafile=...)` or `aiohttp.Fingerprint` so self-signed servers stay authenticated instead of turning verification off entirely.

### 4. Earcon echo guard keeps ~200 ms of still-audible earcon tail; residual math is wrong

**Where:** `src/openhab_voice_satellite/pipeline.py:30` · **Lens:** audio · **Verdict:** CONFIRMED

`EARCON_ECHO_GUARD_MS = 300` is justified as "matches the sink's playout residual so it postdates the earcon's audible end," and `_listen_round`'s docstring claims the kept frames "postdate the earcon." That premise contradicts the sink's own accounting: `play()` returns `SINK_RESIDUAL_S` (300 ms) after the last byte leaves appsrc, and gst_sink.py defines the first ~200 ms of that residual as the saturated pulsesink ring still playing the earcon's tail — "exactly when echo self-triggers," in its own words. So the earcon is audible until roughly `return_time − 100 ms`, while the guard keeps the newest 320 ms (4 frames) of mic backlog: ~200+ ms of the earcon's loudest trailing audio survives the trim and is fed to the endpointer as the start of the utterance. Only a ~100 ms keep actually postdates the audible end. The shipped wake earcon is 280 ms with near-full-scale energy in its last 200 ms, so the kept slice is the loud part.

**Failure scenario:** Device without echo cancellation, dialog-mode follow-up round: the entry earcon plays, `_drain_earcon_echo` keeps 4 frames of which ~2–3 contain audible tail echo, which ships to STT (including cloud) at the head of every guarded recording. The worse branch — echo flipping `SpeechEndpointer.speech_started` and producing a phantom empty utterance that terminates the conversation — did not reproduce with the shipped chime earcons (max VAD probability 0.24–0.27 even under simulated heavy reverb, below the 0.5 threshold), but becomes trivially reachable with user-configured speech-like earcon WAVs, which the config permits. The misleading documented derivation stands unconditionally.

**Fix direction:** Keep only the slice of the residual that postdates the audible end (`SINK_RESIDUAL_S` minus the 200 ms ring, i.e. ~100 ms / 1–2 frames) or trim against a sink-exposed audible-end estimate, and correct the two comments' derivation.

### 5. PiperSpeaker synthesizes an unpunctuated long answer as one monolithic call

**Where:** `src/openhab_voice_satellite/piper_tts.py:55` · **Lens:** speech · **Verdict:** CONFIRMED

`PiperSpeaker.speak` splits only with `split_sentences`, never `tts_chunks`, so the exact case `TTS_CHUNK_CHARS` was built for — a "list all items" answer that is one giant comma-separated sentence — reaches piper as a single `synthesize()` call. `_synthesize_sync` materializes the whole sentence (`list(voice.synthesize(...))` then concatenate) before any audio can play, so pipelining degenerates to nothing. The same happens on the fallback path: a `PartialSpeechError` remainder is `' '.join` of comma-list chunks with no sentence punctuation, so `FallbackSpeaker` hands local piper one monolithic sentence too.

**Failure scenario:** Default config (tts.engine=piper). User asks openHAB to list all items; the interpreter answers with one ~2000-char comma-separated sentence. `split_sentences` returns a single sentence; piper synthesizes ~2 minutes of audio in one executor call (tens of seconds on a Pi 5 at typical RTF) while the satellite sits in SPEAKING in total silence — the user assumes it crashed. A barge-in cancels the await but the executor thread keeps synthesizing (documented), burning a core on audio that will never play.

**Fix direction:** Feed piper through `tts_chunks` (or an equivalent length cap on sentences) in `PiperSpeaker.speak` so oversized sentences stream in bounded pieces like the cloud engines do.

### 6. Stage-2 countdown decremented on its init frame: verifier runs one frame early

**Where:** `src/openhab_voice_satellite/wakeword.py:268` · **Lens:** wakeword · **Verdict:** CONFIRMED

On the trigger frame, `_stage2` sets `_verify_countdown = _verify_delay_frames` (line 260) and immediately decrements it in the same call (line 268), so the verifier scores the ring after only `ceil(delay_ms/frame_ms) − 1` frames of post-trigger audio — 240 ms for the shipped 300 ms/80 ms, and zero further audio for any `delay_ms <= frame_ms`. Every documented contract says otherwise: wakeword.py:57 and :249–250, config.example.yaml:33, and config.yaml's own "total wake latency gains ~320 ms (4 frames)." It also diverges from the calibration replay that produced the deployed operating point: ultiwake's `two_stage.py` resolves on the 4th frame (320 ms) after the trigger, so the v7 threshold 0.5 and its published recall/FA numbers were measured on a capture window ending 80 ms later than the window the satellite actually scores. (The in-repo bench.py replays through the real detector, so recall measured there does reflect runtime timing — the harness divergence and doc-vs-code mismatch stand regardless.)

**Failure scenario:** User speaks the wakeword; stage 1 crosses at frame k; the verifier scores audio ending at k+3 (240 ms post-trigger) instead of the calibrated 320 ms — the 1.5 s window shifts 80 ms toward the phrase onset, so near-boundary phrases whose tail the calibration window contained are truncated at runtime and can be rejected (recall below the measured 99.5–100 %). Degenerate case: an operator sets `stage2.delay_ms: 80` expecting one frame of settling audio and the verifier instead runs on the trigger frame itself, scoring exactly the truncated phrase the delay exists to avoid.

**Fix direction:** Skip the decrement on the frame that starts the countdown (or initialize to `delay_frames + 1`) so the verdict lands after the full `ceil(delay_ms/frame_ms)` frames of post-trigger audio, matching the docs and the ultiwake replay; update the test comment pinning "trigger frame included."

### 7. Install flow creates secrets-bearing config.yaml world-readable; no perms guidance

**Where:** `deploy/install.md:112` · **Lens:** security · **Verdict:** CONFIRMED

Section 4 says `cp config.example.yaml config.yaml` and put the openHAB API token (plus optional Gemini/Deepgram keys) into it; under the default umask this yields a 644 file inside the 755 `/opt/openhab-voice-satellite` directory created in section 2. Neither install.md, README.md (line 67), nor the systemd unit (whose comment explicitly blesses "keep the token in config.yaml") ever mentions `chmod 600` or tightening the install directory, so a by-the-book install leaves three live credentials readable by every local account.

**Failure scenario:** Operator follows install.md verbatim on a Pi that also runs another service; that service is compromised, and its unprivileged user simply cats `/opt/openhab-voice-satellite/config.yaml`, obtaining the openHAB token (full smart-home control) and the cloud STT/TTS API keys (billable abuse).

**Fix direction:** Add `chmod 600 config.yaml` (and `chmod 750` on the install dir, or an `EnvironmentFile=`/systemd-credentials alternative) to the configure step in install.md and the README quick start.

### 8. `--check` never executes the stage-2 verifier inference or validates its output

**Where:** `src/openhab_voice_satellite/selftest.py:34` · **Lens:** diagnostics · **Verdict:** CONFIRMED

`check_wakeword` feeds 2.5 s of silence, so on a two-stage config the stage-1 trigger never fires and `BaseWakewordDetector._verify` (`session.run` + `flatten()[0]`) is never executed; the only compensation is the build-time *input*-shape check in `_build_verifier`. The verifier's output is never run and never range-checked, while stage-1 scores get a per-frame is-a-probability assertion in this very function. The live deployment runs a two-stage config, and models/wakeword/ shows verifier ONNX files re-exported repeatedly from the external pipeline, so a bad export is a plausible operator error.

**Failure scenario:** A verifier ONNX with a valid `('features', [batch, 40, 151])` input but a `(1, 2)` softmax output (or raw logits, or a graph that fails only at run time) passes `--check`. Live, `_verify` returns `flatten()[0]` — the wrong class probability or a logit — so the verifier rejects real wakes (satellite goes deaf) or accepts everything (false-accept storm), or the first real wake raises inside the monitor loop hours after a green self-test.

**Fix direction:** In `check_wakeword` (or `_build_verifier`), run the verifier session once on a zeros `(1, 40, 151)` feature block and assert the output is a single finite value in [0, 1], mirroring the stage-1 probability check.

### 9. All verifier frontend parity tests are permanently skipped in CI; docs claim the guarantee holds

**Where:** `tests/test_verifier_mel.py:38` (merged with `infrastructure.md:242–246`) · **Lens:** tests + docs · **Verdict:** CONFIRMED

The module-level `pytestmark` gates every test on `models/wakeword/verifier_melfb_40.npy` and `verifier_frontend_golden.npz`, and the two deployed-model tests additionally read gitignored config.yaml; `.gitignore` excludes both `models/` and `config.yaml`, and CI fetches no artifacts, so none of these four tests has ever run or can run in CI or on any checkout but the author's machine — the golden fixtures cannot even be committed while `.gitignore` blankets `models/`. Meanwhile two documents assert the opposite: verifier_mel.py's docstring claims parity "is asserted by tests/test_verifier_mel.py," and infrastructure.md §4 catalogs the untracked `models/piper/` and `models/wakeword/` files (including the golden fixtures, and a lessac voice nothing provisions) as "repository assets" while claiming the parity tests check golden fixtures. Same class of gap: `test_wakeword_buffer.py`'s `test_real_model_reset_matches_stock_trajectory` — self-described as "the only test that actually proves the primed restore is equivalent" — also never runs in CI. Nothing gates or surfaces the skips (no `-rs`, no skip-count assertion); a local run shows 299 passed / 0 skipped, so the blind spot is invisible.

**Failure scenario:** A scipy/numpy upgrade or refactor subtly changes `_stft_power`/`_pcen` output (e.g. lfilter zi seeding) → CI stays green because the parity tests skip → the deployed stage-2 verifier scores features it was never trained on → wake recall collapses on the Pi with no test ever having failed, while the architecture doc asserts the parity guarantee is CI-verified.

**Fix direction:** Commit small golden fixtures (un-ignore a fixtures path or move them under tests/) so the frontend parity tests run in CI; make the deployed-pair tests skip visibly; mark the §4 rows as gitignored local artifacts and note where the parity tests actually run.

### 10. stt.model local path not resolved against the config dir despite the blanket contract

**Where:** `src/openhab_voice_satellite/config.py:311` (merged with `config.example.yaml:2`) · **Lens:** config + docs · **Verdict:** CONFIRMED

`_resolve_config_paths` rewrites piper voices, wakeword model/verifier paths, stage2 model/mel_basis, and all four earcons, but not `stt.model` — which config.example.yaml:43 documents as "faster-whisper model name or local path" directly under the header "Relative paths are resolved against this file's directory" (line 2); infrastructure.md §6 makes the same blanket claim with only the pretrained-phrase-name carve-out. A relative `stt.model` therefore resolves against the process CWD, and stt.py:38 passes it verbatim to `WhisperModel`, whose `dir/name`-shaped nonexistent relative path falls into the HuggingFace repo-id branch. The shipped systemd unit happens to set `WorkingDirectory` to the config dir, masking the defect only in the stock deploy.

**Failure scenario:** User places a locally converted CTranslate2 whisper model next to the config and sets `stt.model: "models/whisper-de-ct2"`; the process runs with any other working directory (e.g. `--check`/`--probe-mic` from a home directory), `os.path.isdir` fails, and faster-whisper tries to download HF repo `models/whisper-de-ct2` — startup dies with a RepositoryNotFoundError/404 (or hangs on network) instead of loading the local model the documented path contract says should resolve.

**Fix direction:** Resolve `stt.model` against the config directory when it looks like a path (contains a separator or exists relative to the config dir), or explicitly document the carve-out in both config.example.yaml and infrastructure.md §6.

### 11. Non-UTF-8/undecodable response bodies raise UnicodeDecodeError outside FALLBACK_ERRORS — no local fallback

**Where:** `src/openhab_voice_satellite/cloud.py:18` (merged with `fallback.py:37`; also `gemini.py:47`, `deepgram.py:51`) · **Lens:** cloud + speech · **Verdict:** CONFIRMED

`raise_for_status()` reads error bodies with `resp.text()`, and `GeminiClient.generate` / `DeepgramClient.listen` read 200 bodies the same way; installed aiohttp 3.14 decodes strictly (declared charset, or UTF-8 when undeclared), so an undecodable body raises `UnicodeDecodeError` — a ValueError that is neither wrapped into GeminiError/DeepgramError nor a member of `FALLBACK_ERRORS`, unlike every other malformed-response shape (JSONDecodeError, KeyError/IndexError/TypeError, and the binascii/frombuffer decode failures fixed on this very branch). It bypasses FallbackTranscriber/FallbackSpeaker and hits the pipeline's broad except — contradicting infrastructure.md:182's contract that payload-decode failures stay inside the fallback machinery. No test covers non-UTF-8 bodies. Reachability caveat: default base_urls are HTTPS, so a plain captive portal fails at TLS (correctly caught); the decode path needs a TLS-terminating middlebox, an overridden `base_url` (a supported plain-string config field with no https requirement), or a non-UTF-8 provider-edge error body.

**Failure scenario:** A gateway/proxy answers the Deepgram/Gemini request with HTTP 403/502 and an ISO-8859-1 HTML body (0xE4/0xFC umlauts, no honored charset). `resp.text()` raises UnicodeDecodeError instead of the provider error; it passes through the fallback wrappers untouched, the pipeline logs "pipeline failed" and plays the error earcon, and the utterance is lost — the exact "cloud unusable" condition where local whisper/piper should have taken over.

**Fix direction:** Add UnicodeDecodeError to FALLBACK_ERRORS, or decode cloud response bodies with `errors="replace"` / wrap the `resp.text()` call sites in the provider error type so body-decode failures stay inside the fallback taxonomy as documented.

## Low

### 12. VadConfig, audio.frame_ms and stt.cpu_threads accept nonsense values siblings would reject

**Where:** `src/openhab_voice_satellite/config.py:151` (merged with `stt.py:30`) · **Lens:** config + speech · **Verdict:** CONFIRMED

`VadConfig.silence_ms/no_speech_timeout_s/max_utterance_s` and `AudioConfig.frame_ms` carry no bounds while neighboring fields do (thresholds ge/le, patience ge=1/le=10, delay_ms le=1000, followup_timeout_s gt=0) — verified: `silence_ms=-500`, `no_speech_timeout_s=-1`, `max_utterance_s=0` and `frame_ms=0` all validate. `silence_ms<=0` makes `endpoint_reached` true on the first VAD chunk after speech starts (vad.py:66, `0 >= 0`); `frame_ms=0` yields a bare ZeroDivisionError at detector build (wakeword.py:147). `stt.cpu_threads` has no `ge=1`, and the saturation advisory at stt.py:30 fires only on `cpu_threads >= cores` — but faster-whisper/ctranslate2 treat 0 as "auto" (all cores on a Pi 5), so `cpu_threads: 0` slips past the warning while causing exactly the wakeword-cadence starvation and deaf barge-in that advisory exists to flag.

**Failure scenario:** Config with `vad.silence_ms: 0` (or a negative typo) loads without complaint; every utterance is endpointed on the first frame after speech begins, whisper receives a fraction of a word, and every voice command fails with no hint that the config was the cause — contradicting the documented posture that config mistakes are load-time errors. Separately, `stt.cpu_threads: 0` set expecting "auto" saturates all 4 cores during THINKING with no diagnostic.

**Fix direction:** Add bounds consistent with sibling fields (silence_ms gt=0, no_speech_timeout_s gt=0, max_utterance_s gt=0, frame_ms gt=0, cpu_threads ge=1), and/or extend the stt.py advisory to `cpu_threads < 1` explaining that 0 means all cores.

### 13. Verifier output contract unchecked at load; `_verify` blindly reads `flatten()[0]`

**Where:** `src/openhab_voice_satellite/wakeword.py:292` · **Lens:** wakeword · **Verdict:** CONFIRMED

`_build_verifier` validates only the input contract (single 'features' input, `(batch, 40, 151)`), but the output is never validated: `_verify` takes `session.run(...)[0].flatten()[0]`, so a verifier exported with a two-logit/softmax head (or extra outputs) passes the load-time gate and the self-test, then silently scores every wake with the wrong tensor element — no crash, just wrong verdicts, the silent sibling of the failure mode the load-time input check (added in the earlier review pass) exists to prevent. Runtime companion to finding 8.

**Failure scenario:** An operator deploys a verifier ONNX whose head emits `(batch, 2)` class probabilities; load and `--check` pass; at runtime `flatten()[0]` reads class 0 (the negative class), so genuine wakes score near 0 and are all rejected (or, with class order flipped, noise is accepted) — the satellite goes deaf to its wakeword with nothing in the logs but "rejected by verifier."

**Fix direction:** Extend the load-time contract check in `_build_verifier` to require a single output of size 1 per batch element (mirroring the `(batch, 1)` sigmoid the ultiwake exporter produces).

### 14. Unvalidated frame_ms below 80 silently defeats EdgeTrigger patience

**Where:** `src/openhab_voice_satellite/config.py:35` · **Lens:** wakeword · **Verdict:** CONFIRMED

`audio.frame_ms` has no multiple-of-80 constraint, but openwakeword only produces a new prediction per accumulated 1280 samples and returns the previous score for shorter calls: with `frame_ms < 80` (or any non-multiple of 80), `_scores` feeds EdgeTrigger duplicated/stale scores, so `patience` counts the same inference twice and single-frame transients — the thing patience exists to reject — fire detections. The stub-based tests (wakeword_stubs.py pops one fresh score per call) can never expose this. (Adjacent to finding 12's `frame_ms=0` crash; this is the subtler semantic failure at legal-looking values.)

**Failure scenario:** An operator lowers frame_ms to 40 chasing latency; every 640-sample `predict()` between 1280-boundaries returns the prior score, so a single 80 ms transient spike above threshold appears as two consecutive frames and satisfies patience=2 — the calibrated trigger shape silently degrades to patience=1 false-accept behavior while the config still reads `patience: 2`.

**Fix direction:** Constrain frame_ms in AudioConfig (`ge=80` and a multiple-of-80 validator, citing the 1280-sample openWakeWord chunk) so the score-per-frame assumption behind patience holds by construction.

### 15. Wake trigger landing on a rejection's verdict frame is silently swallowed

**Where:** `src/openhab_voice_satellite/wakeword.py:259` · **Lens:** wakeword · **Verdict:** CONFIRMED

A stage-1 trigger that fires on the exact frame a pending verification expires is folded into that expiring countdown (`_verify_countdown is not None`, so no new countdown starts), and when that verdict is a rejection the new candidate is discarded without ever being verified — its EdgeTrigger is already disarmed, so the rest of the phrase cannot re-fire either. The existing tests cover a re-trigger one frame *before* the verdict (accept path), not a WAKE trigger on a rejection's verdict frame.

**Failure scenario:** Background speech trips stage 1 at frame k (rejections run ~once a minute per config.yaml); the user starts the real wakeword so that its stage-1 crossing lands exactly on frame k+3 where the pending rejection resolves; the crossing is folded into the dying countdown, the candidate peak is thrown away with the rejection, and no new countdown is scheduled — the spoken wakeword is dropped and the user must repeat it. Requires an 80 ms alignment plus a specific score dip, so rare but concretely constructible.

**Fix direction:** Resolve an expiring countdown before folding a same-frame trigger, and let a trigger that coincides with (or follows) a rejection start a fresh countdown instead of being absorbed into the finished one.

### 16. Patience window re-judges playback-era frames at the idle threshold

**Where:** `src/openhab_voice_satellite/wakeword.py:105` · **Lens:** wakeword · **Verdict:** CONFIRMED

`EdgeTrigger.fired()` evaluates the whole patience window against the *current* frame's threshold, but the threshold flips per-frame with `sink.is_playing`: scores observed while TTS/earcons were audible — which only had to clear `threshold_speaking` — are retroactively judged at the lower idle threshold on the first frame after playback ends, defeating the documented echo-mitigation intent for up to patience−1 frames.

**Failure scenario:** Live config (threshold 0.35, threshold_speaking 0.55, patience 2): the last playing frame carries an echo score of 0.45 (correctly below the 0.55 speaking bar); the sink drains; the next frame's residual echo also scores 0.45; `fired(0.35, 2)` sees window [0.45, 0.45] all ≥ 0.35 and emits a wake built half from a frame the speaking threshold existed to reject — false barge-in/wake pressure at every playback boundary where echo hovers between the two bars. Stage 2 usually catches it, but the stop model has no second stage (stop_model is currently null live, so the aggravation is latent).

**Fix direction:** Track per-frame whether the observed score met that frame's own threshold (e.g. `observe(score, threshold)`) and require patience consecutive per-frame passes instead of re-judging history against the current threshold.

### 17. Wake loop uses asyncio.wait_for on 3.11 despite the project's own gh-86296 rule

**Where:** `src/openhab_voice_satellite/app.py:416` · **Lens:** audio + orchestration · **Verdict:** CONFIRMED

The interrupt monitor awaits `wake_queue.get()` via `asyncio.wait_for(..., timeout=MIC_STALL_WARN_S)`. recorder.py:33–36 and gst_sink.py:307–309 both deliberately use `asyncio.timeout` instead, citing that on Python 3.11 (the deployment target) `wait_for` swallows an external cancel that races a completed inner await (gh-86296). The monitor is the coroutine the main task's cancellation must reach at shutdown, and it receives a completed `get()` every ~80 ms, so the same race the other two sites guard against exists here unguarded — a violation of the codebase's own stated rule.

**Failure scenario:** On a 3.11 deployment the user hits Ctrl-C exactly as a mic frame completes `wake_queue.get()`: `wait_for` returns the frame instead of propagating CancelledError and the monitor keeps looping. Verification tempers the original consequence: the app simply continues running normally and a second Ctrl-C still performs full cleanup (finally blocks and conversation DELETEs run via the Runner's cancel-all) — the symptom is "first Ctrl-C occasionally does nothing," in a microsecond-scale window per 80 ms frame.

**Fix direction:** Wrap the `get()` in `async with asyncio.timeout(MIC_STALL_WARN_S)` like `recorder._next_frame` and `gst_sink._await_playout` already do, for the same documented reason.

### 18. `_cancel_pipeline` swallows the monitor's own cancellation while awaiting the cancelled task

**Where:** `src/openhab_voice_satellite/app.py:375` · **Lens:** orchestration · **Verdict:** CONFIRMED

In `_cancel_pipeline`, `await task` after `task.cancel()` is wrapped in `except asyncio.CancelledError: pass`. If the monitor task itself is cancelled (Ctrl-C via asyncio.Runner) while parked on that await, `Task.cancel` forwards the cancel to the fut_waiter (the pipeline task) and the resulting CancelledError raised at the await is indistinguishable from the child's — the blanket except suppresses it, and the monitor resumes its loop. No test covers a monitor cancel landing during a live barge-in unwind (the existing cancellation test only cancels while the monitor waits for a frame).

**Failure scenario:** User barge-ins (wakeword during playback) and hits Ctrl-C while the pipeline task is unwinding inside `_cancel_pipeline`'s `await task`; the shutdown cancel is silently absorbed, the satellite continues running ("interaction cancelled" logged, loop resumes), and a second Ctrl-C is needed to exit.

**Fix direction:** Record `asyncio.current_task().cancelling()` before the await and re-raise CancelledError when the count increased across it (only then was the cancel meant for the monitor), keeping the shutdown-finally path — which enters with the count already raised — unaffected.

### 19. `_CaptureHealth.restart` leaks pre-stall frame counts into the post-stall window

**Where:** `src/openhab_voice_satellite/app.py:151` · **Lens:** orchestration · **Verdict:** CONFIRMED

`restart()` resets the window clock and the CaptureStats baseline but not `_frames`, `_rms`, or `_score`, so monitor-side frames counted before a mic stall are attributed to the window that starts at restart — contradicting its own docstring ("a window that is not counted must not lend its buffers to the next one"), which is honored only for the graph-side stats. The frame count is inflated against a shortened window, muting the degraded-capture check; `test_a_stall_does_not_lend_its_buffers_to_the_next_window` asserts only the CaptureStats baseline, so the counter leak is a test blind spot too.

**Failure scenario:** Capture runs normally for 8 s (100 frames counted, no heartbeat yet), stalls 10 s (wait_for timeout → restart), then resumes at 50% delivery; the next heartbeat reports the impossible "162 frames in 10.0s" and 162 > 0.8×125, so the "degraded capture" warning stays silent for exactly the window where the operator needs it, and pre-stall peak rms/score pollute the debug line.

**Fix direction:** Zero `_frames`, `_rms`, and `_score` (and move the `_dropped` baseline) in `restart()` alongside the clock and capture baseline, and extend the stall test to cover the frame counter.

### 20. Empty `--positives`/`--negatives` dir skips the gate with a wrong-cause message and exit 0

**Where:** `src/openhab_voice_satellite/bench.py:349` · **Lens:** diagnostics · **Verdict:** CONFIRMED

The gate runs on `positive_files and negative_files` (expanded lists) but the fallback message condition is `positives or negatives` (the flags), so a directory that exists but contains no `*.wav` files silently drops the gate and prints "the gate needs both --positives and --negatives" even when both flags were supplied — and the run still exits 0. A nonexistent dir *is* caught (exit 2); the gap is precisely the existing-but-WAV-empty dir.

**Failure scenario:** User runs `--positives recordings/wake --negatives recordings/room` where recordings/wake holds only .flac files (or is freshly created and empty). The tool prints the sweep for the negatives, claims the gate needs both flags (both were given), and exits 0 — a scripted promotion pipeline treating exit 0 as "evaluation ran" proceeds without any recall measurement.

**Fix direction:** When a flag was given but its expansion is empty, print "no WAV files under <dir>" and return exit code 2 instead of falling into the generic needs-both message.

### 21. A wrong-rate or unreadable WAV aborts the whole scoring run with a raw traceback

**Where:** `src/openhab_voice_satellite/bench.py:344` · **Lens:** diagnostics · **Verdict:** CONFIRMED

`score_wavs` pre-validates only file existence (message + exit 2); the per-file errors `score_file` can raise — the crafted "resample first" ValueError for non-16 kHz files, `wave.Error` for non-WAV/corrupt files, "expected 16-bit PCM" — propagate uncaught through `_report_files` to `main()`, aborting mid-report with a traceback (exit 1) instead of the handled-error path.

**Failure scenario:** A corpus directory contains one 44.1 kHz phone recording among fifty valid files (`_expand` recursively globs `*.wav`). The run prints per-file rows up to that file, then dies with a Python traceback; the sweep, gate, and live lines for the entire corpus are never produced, and the crafted guidance message is buried at the bottom of the traceback.

**Fix direction:** Catch ValueError/wave.Error per file in `score_wavs` (or `_report_files`), print the message, and either skip the file with a warning or exit 2 cleanly.

### 22. Unguarded diagnose_capture.wav write can discard the entire probe summary

**Where:** `src/openhab_voice_satellite/probe.py:313` · **Lens:** diagnostics · **Verdict:** CONFIRMED

`DUMP_WAV` is a CWD-relative path and `write_wav` is called before any of the summary output (peak scores, verdict counts, fps, engine cost, capture accounting). An unwritable or read-only working directory raises PermissionError there, so the 30 s run's whole diagnostic yield is lost to a traceback — contradicting the module's own "a probe reports, never crashes" stance, which it applies (with noqa comments) to earcon playback and node listing.

**Failure scenario:** On the Pi, the user runs the probe as the dedicated service user from a directory it cannot write (e.g. `/etc/openhab-voice-satellite` or `/`). After the full 30 s capture, `wave.open` raises PermissionError; peak wake score, wake/stop/rejection counts, capture rate and cost lines are never printed and the process exits with a traceback (per-second live rows survive; only the summary is lost).

**Fix direction:** Wrap the `write_wav` call in try/except (print a "could not write dump" line and continue to the summary), and/or print the summary before attempting the dump.

### 23. Mode and modifier flags are silently ignored in non-matching modes

**Where:** `src/openhab_voice_satellite/__main__.py:80` · **Lens:** diagnostics · **Verdict:** CONFIRMED

Dispatch is a first-match if-chain with no mutual-exclusion or dependency validation: `--model`/`--engine`/`--compare` without any scoring flag fall through to launching the full app, and combining modes (`--check --probe-mic`) silently runs only the first branch. argparse accepts all combinations without complaint.

**Failure scenario:** A user field-testing a candidate model runs `openhab-voice-satellite --model candidate.onnx`. The scoring branch is not entered, the override is dropped, and the full app starts with the config's old model — the user evaluates wake behavior of the wrong model with no warning. Similarly `--check --probe-mic` runs only the self-test and the user believes the probe passed.

**Fix direction:** Error out (`parser.error`) when `--model`/`--engine`/`--compare` are given without a scoring flag, and when more than one of `--list-devices`/`--check`/`--probe-mic`/scoring flags are combined.

### 24. `check_wakeword` validates only the WAKE score; a configured stop model is unchecked

**Where:** `src/openhab_voice_satellite/selftest.py:45` · **Lens:** diagnostics · **Verdict:** CONFIRMED

The probability-range assertion loops over `detector.score(WAKE)` only. With `wakeword.stop_model` configured, the stop head's scores are computed (via `_scores`) but never checked against the same "is a probability" contract the check exists to enforce, even though `stop_threshold` is read against them at runtime — and openwakeword passes raw model output through unclamped.

**Failure scenario:** A stop model exported without its sigmoid (scores are raw logits) passes `--check`. Live, `stop_threshold` 0.5 is compared against logits: the stop word either fires on nearly any speech during playback (constant barge-in aborts) or never fires, and nothing in the self-test flagged it.

**Fix direction:** Extend the per-frame assertion in `check_wakeword` to `detector.score(STOP)` whenever `config.wakeword.stop_model` is set.

### 25. OVS_DUMP_WAKE env vars leak from the developer shell into the whole test suite

**Where:** `tests/conftest.py:12` · **Lens:** tests · **Verdict:** CONFIRMED

The autouse `_clean_env` fixture clears only the three credential vars; app.py reads `OVS_DUMP_WAKE` / `OVS_DUMP_WAKE_SCORE` from the process environment on every frame, so a developer running pytest with the documented field-tuning vars exported has ScriptedDetector wakes/stops in test_app.py write synthetic all-zero 2.5 s WAVs (~80 KB each; reproduced: 9 files per run) into their real dump corpus. The dedicated guard test `test_wake_audio_dump_is_off_without_the_env_var` cannot catch this because it only asserts `tmp_path` is empty — an assertion that passes wherever the dumps actually land.

**Failure scenario:** Developer tuning wakewords on the Pi has `OVS_DUMP_WAKE=~/wake-dumps` exported (the documented workflow) and runs pytest → synthetic wake-/near-/rejected-* WAVs pollute the corpus used to retrain the verifier → hard negatives fed to the training loop are test artifacts, and the suite reports all green.

**Fix direction:** Add OVS_DUMP_WAKE and OVS_DUMP_WAKE_SCORE to the autouse delenv loop in conftest.py.

### 26. `Transcriber._transcribe_sync` language-remap contract is untested

**Where:** `tests/test_stt.py:1` · **Lens:** speech · **Verdict:** CONFIRMED

test_stt.py covers only the cpu_threads advisory (its own docstring says so). The documented contract of `_transcribe_sync` — single-language skips detection, out-of-set detection is remapped to the best allowed entry of `all_language_probs`, empty/None probs fall back to `default_language`, and segment texts are stripped and joined — has zero test coverage, so a regression in the remap (e.g. inverted max, or crash when `all_language_probs` is None) would ship unnoticed.

**Failure scenario:** A future edit changes the `probs = info.all_language_probs or []` handling or the `max()` key and breaks the remap; the suite stays green, and bilingual users only notice when a misdetected utterance locks the wrong TTS voice for a whole dialog round (pipeline locks language on round 0).

**Fix direction:** Add unit tests for `_transcribe_sync` with a stubbed model: fixed-language path, out-of-set detection remapped via all_language_probs, empty probs defaulting, and segment join/strip behavior.

### 27. No test parses config.example.yaml — documented quickstart is unguarded

**Where:** `tests/test_config.py:183` · **Lens:** tests · **Verdict:** CONFIRMED

The README quickstart is `cp config.example.yaml config.yaml`, config.py rejects removed enum values at load by design, and this repo's history removes such values regularly (Kokoro TTS, wakeword engine swap) — yet no test runs `load_config` on the shipped example, so example-vs-schema drift ships with green CI. (The example parses today; verified manually.)

**Failure scenario:** A future engine/field removal adds a rejecting Literal or validator while config.example.yaml still names the old value → CI green → every new install crashes at first startup with a pydantic ValidationError on the file the docs told the user to copy.

**Fix direction:** Add a test that runs `load_config(REPO_ROOT / "config.example.yaml")` and asserts it validates.

### 28. No warnings policy: a real unraisable-exception warning already passes silently

**Where:** `pyproject.toml:53` · **Lens:** tests · **Verdict:** CONFIRMED

`[tool.pytest.ini_options]` sets only asyncio_mode and testpaths — no `filterwarnings = error` and CI passes no `-W` flag, so warnings never fail the build. The suite already emits a PytestUnraisableExceptionWarning (TypeError in asyncio `_SelectorTransport.__del__`, surfacing under test_vad from a prior test's aiohttp TestServer teardown — a transport outliving its closed server) plus a PyGIDeprecationWarning; both pass silently, as would unawaited-coroutine RuntimeWarnings — the classic `asyncio_mode=auto` failure mode where a forgotten await makes a test vacuously pass.

**Failure scenario:** A test drops an await on an async helper (or a fake server leaks a transport) → the only symptom is a RuntimeWarning/PytestUnraisableExceptionWarning → pytest exits 0, CI green, and the assertion the coroutine would have made never runs.

**Fix direction:** Add `filterwarnings = ["error", ...]` with targeted ignores (PyGIDeprecationWarning) to `[tool.pytest.ini_options]` and triage the existing `_SelectorTransport.__del__` unraisable.

### 29. Nightly pip-audit never scans openwakeword, the one dependency pinned forever

**Where:** `.github/workflows/security-scan.yml:24` · **Lens:** tests + security · **Verdict:** CONFIRMED

The scan audits requirements exported from uv.lock, but openwakeword is intentionally absent from project metadata (installed in production via `pip install --no-deps openwakeword` per pyproject's note) and is pinned to exactly 0.6.0 forever by `wakeword_buffer.PATCHED_VERSION` — so the single production dependency guaranteed to grow stale is the one the vulnerability scan structurally cannot see.

**Failure scenario:** A CVE is published against openwakeword ≤ 0.6.0 (or its vendored model-download path) → nightly pip-audit stays green because the package is not in requirements-audit.txt → the vulnerable pinned release keeps running on the satellite indefinitely.

**Fix direction:** Append the out-of-lock production pins (openwakeword==0.6.0) to requirements-audit.txt in the workflow before running pip-audit.

### 30. OVS_DUMP_WAKE documented as capturing audio "around" a detection; dump is pre-roll only

**Where:** `src/openhab_voice_satellite/app.py:102` (merged with `README.md:124–126`) · **Lens:** orchestration + docs · **Verdict:** CONFIRMED

The `_dump_wake_audio` docstring and README.md:126 both say the dump is "the audio around a detection," but the code dumps `detector.tail(WAKE_DUMP_PREROLL_S)` (app.py:129) — 2.5 s of mic audio ending exactly at the verdict frame, nothing after it. The constant's own comment at line 44 and infrastructure.md:87–88 (corrected on this branch: "pre-roll only, nothing after it") contradict the docstring and README, so the docs now disagree with each other. The same README paragraph (line 124) claims "Three env vars dump audio for debugging, each taking a directory" while enumerating only two, and the third (`OVS_DUMP_WAKE_SCORE`) takes a float score. For stage-1-only configurations (the default; no verifier), the edge trigger fires before the phrase ends, so a "wake" dump can even clip the tail of the wakeword phrase itself.

**Failure scenario:** A field debugger enables OVS_DUMP_WAKE to collect false accepts, expecting post-trigger context per the docs; every WAV ends at the detection frame, mis-slicing analysis scripts written for "around" audio, and on a verifier-less setup the dumped positives are missing the last few hundred ms of the phrase, degrading the retraining data the feature exists to feed. A reader following "each taking a directory" sets OVS_DUMP_WAKE_SCORE to a directory path; `float()` fails and near-miss/rejection dumps are silently disabled behind one warning.

**Fix direction:** Reword the docstring and README (two directory-valued vars plus one score-valued var; the dump is the 2.5 s of audio leading up to the detection/verdict frame), or defer the dump a few frames to include post-detection audio.

### 31. Production install ignores uv.lock: unpinned ranges and no hash verification

**Where:** `deploy/install.md:31` · **Lens:** security · **Verdict:** CONFIRMED

The documented deploy path is `.venv/bin/pip install -e .` (range specifiers like piper-tts>=1.3, aiohttp>=3.10 resolved fresh at install time) plus a version-pinned but hash-less openwakeword install, so what runs on the Pi is neither the set of versions the nightly pip-audit scans (which audits uv.lock) nor integrity-verified against PyPI substitution.

**Failure scenario:** A dependency inside an allowed range publishes a compromised release (or a PyPI account takeover re-uploads one) between CI's last audit and a Pi (re)install; pip resolves and installs it unaudited onto a device holding the openHAB token and an always-on microphone, and no `--hash` check exists to catch the substitution.

**Fix direction:** Deploy from a hash-pinned `uv export --format requirements-txt` (or `uv sync --frozen`) instead of bare `pip install -e .`, and add the `--hash` for the openwakeword wheel to its `--no-deps` install line.

### 32. Model downloads land in runtime-loaded paths with no checksum and a mutable ref

**Where:** `scripts/download_models.py:38` · **Lens:** security · **Verdict:** CONFIRMED

`download()` fetches Piper voices from huggingface.co `.../resolve/main` — a mutable branch, not a pinned revision — via urlretrieve with no SHA-256 verification, and `download_openwakeword()` delegates to `openwakeword.utils.download_models()`, which also fetches its shared feature models unverified; all files are then parsed by onnxruntime inside the long-running service. (TLS itself is verified by urllib defaults; the gap is content integrity/pinning.)

**Failure scenario:** The rhasspy/piper-voices HF repo (or an upstream openwakeword release asset) is compromised and its main branch force-updated; the next install fetches the trojaned .onnx, nothing detects the change, and the malicious model is loaded by native ONNX-parsing code in the always-listening process.

**Fix direction:** Pin the Hugging Face URLs to a specific commit revision and verify a recorded SHA-256 for each downloaded file before renaming the .part into place.

### 33. systemd unit has zero sandboxing directives

**Where:** `deploy/openhab-voice-satellite.service:16` · **Lens:** security · **Verdict:** CONFIRMED

The `[Service]` section sets only WorkingDirectory/ExecStart/Environment/Restart — no `NoNewPrivileges=`, `PrivateTmp=`, `ProtectSystem=`, `ProtectHome=` (with ReadWritePaths for the install dir and HF cache), `RestrictAddressFamilies=`, `CapabilityBoundingSet=`, or `SystemCallFilter=`. The process parses network responses (aiohttp), runs large native parsers (onnxruntime, ctranslate2) on untrusted audio, and optionally unpickles files (which config.py itself says to treat as executable code), so it is a realistic exploitation target, yet a compromise yields the entire unconfined user session.

**Failure scenario:** A memory-corruption bug in onnxruntime/ctranslate2/aiohttp is triggered (crafted network response or model file); the payload runs with full user privileges and reads ~/.ssh keys, browser profiles, and every user file — restrictions that cost nothing here would have contained it.

**Fix direction:** Add a hardening block (NoNewPrivileges=yes, PrivateTmp=yes, ProtectSystem=strict + ReadWritePaths for the install/model dirs, ProtectHome as far as PipeWire's socket allows, RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6) and validate with `systemd-analyze --user security`.

### 34. GitHub Actions referenced by mutable tags instead of commit SHAs

**Where:** `.github/workflows/python-app.yml:35` (also lines 25; `security-scan.yml:18,21`) · **Lens:** security · **Verdict:** CONFIRMED

`actions/checkout@v4` and third-party `astral-sh/setup-uv@v5` are pinned by tag in both workflows; tags are mutable, so a compromised maintainer account can silently repoint them at malicious code that runs with checkout write access to the workspace and the job's GITHUB_TOKEN (real-world precedent: the tj-actions/changed-files tag-repoint compromise, March 2025). Workflow-level `permissions: contents: read` is correctly set, which limits the blast radius.

**Failure scenario:** The astral-sh org is compromised and the v5 tag is moved to a commit that exfiltrates the workspace and poisons the uv cache; every subsequent push/PR run executes it unnoticed, whereas a full-SHA pin would have kept running the audited commit.

**Fix direction:** Pin all four action references to full commit SHAs with a trailing version comment, optionally letting Dependabot bump them.

## Observations (info)

- `src/openhab_voice_satellite/app.py:360` — the interaction task is untracked while its trailing idle earcon still plays: `_pipeline_task` is cleared before `earcons.play("idle")`, so the tail runs outside the cancel path and can touch the sink after the exit stack closes it (currently benign GStreamer no-ops, but outside every documented lifetime guarantee).
- `src/openhab_voice_satellite/__main__.py:33` — `main()` dispatch has zero test coverage: flag combinations, exit-code plumbing, and the lazy-import failure paths are all unverified while every other diagnostics module has a dedicated test file.
- `.github/workflows/python-app.yml:43` — the CI gate is flake8-only: a fully annotated codebase with a typing Protocol as its central engine contract has no mypy/pyright step, and artifact-gated skips never appear in CI output (no `-rs`).
- `tests/fakes.py:195` — `ScriptedDetector.reset()` keeps the `last_*` outputs the real detector clears, so the fake drifts from the WakewordProtocol contract at exactly the seam app.py resets after every dispatch.
- `src/openhab_voice_satellite/__main__.py:100` — the CLI entry point's exit-code contracts are untested, including the `CaptureClosedError → sys.exit(1)` mapping that systemd `Restart=on-failure` depends on and the subtle `nargs="*"` dispatch condition that ultiwake's gate.sh/smoke.sh drive.
- `src/openhab_voice_satellite/pipeline.py:257` — full voice transcripts (line 257) and openHAB answers (line 110) are logged at INFO, the shipped default, so the household's complete spoken-command history persists in the systemd journal with no documented retention note or opt-out short of raising the global log level.

## What was NOT re-reviewed

Nothing was deliberately skipped — this was a full sweep across wakeword, audio, orchestration, config, speech, cloud, diagnostics, tests, docs, and security lenses, on top of the branch's earlier per-area passes. Two areas remain out of scope by nature rather than omission: offline model quality (the ultiwake training/calibration pipeline lives in a separate repo and its models are judged here only by their runtime contracts) and live-hardware behavior on the Pi 5 (PipeWire/GStreamer timing, acoustic echo levels, and real-microphone recall were reasoned about from the code and configs, not measured on the device).
