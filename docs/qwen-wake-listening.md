# Qwen3-ASR engine and Wake-word listening

This document describes two features added on the `feature/qwen-wake-listening`
branch: the **Qwen3-ASR transcription engine** (a bundled Python sidecar running
the official `qwen-asr` package) and the **wake-word listening mode** built on
top of it.

## Overview

**Qwen3-ASR engine.** Echo gains a new engine, `EngineType::Qwen`, backed by the
Qwen3-ASR models (0.6B and 1.7B). Unlike the existing engines (Whisper,
Parakeet, GigaAM) which run natively in Rust, Qwen runs the official
[`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) package (`Qwen3ASRModel`)
inside an isolated embedded Python 3.12 + PyTorch 2.7.1+cu128 runtime. The
runtime is installed as a component into `<app data>/qwen-runtime`
(~2.8 GB download, ~5.4 GB installed) and the weights are downloaded per-file
from Hugging Face into `<app data>/models/qwen-0.6b|qwen-1.7b`. Both the runtime
and the models are pinned byte-for-byte (URL + SHA256, exact HF revisions), so
an update can never silently change behavior. The models appear in the Model
Manager next to Parakeet/Whisper/GigaAM; downloading one auto-installs the
runtime first.

**Wake-word listening.** Instead of holding a hotkey, Echo can listen
continuously for a wake phrase (default «эхо», user-editable) and only then
transcribe and paste. There is no dedicated wake-word neural network: the
existing Silero VAD gates capture, finished utterances are transcribed by the
*current* model (designed and tested with Qwen3-ASR), and a prefix matcher
(`AsrPrefixWakeDetector`) decides whether the recognized text starts with the
wake phrase. Everything after the prefix flows through the existing Echo
pipeline — heuristics, voice commands, snippets, paste — so wake mode adds only
prefix stripping to normal dictation. Three input modes are offered under
**Settings → Advanced → Wake listening**: normal dictation (unchanged behavior),
always-on wake listening, and a hotkey toggle (Ctrl+Space).

## Architecture

### Qwen engine: Rust ↔ Python sidecar

The engine spawns a Python worker process and speaks to it with JSONL over
stdin/stdout. The worker's stdout is reserved for the protocol — the worker
`dup2`s its own stdout to stderr at startup, so library noise (tqdm, warnings)
can never corrupt the JSON channel.

```text
┌────────────────────────── Rust (Echo) ──────────────────────────┐
│                                                                 │
│  EngineType::Qwen ──► Qwen worker manager                       │
│     │  JSONL over stdin/stdout                                  │
│     │  request:  {"id", "op": "load"|"transcribe"|"shutdown", …}
│     │  response: {"id", "result": {"text", "segments"}}         │
│     │           | {"id", "error"}                               │
│     │  audio: temp raw f32le 16 kHz mono file                   │
└─────┼───────────────────────────────────────────────────────────┘
      ▼
┌──────────── Python worker (embedded 3.12) ──────────────────────┐
│  qwen-asr (Qwen3ASRModel)                                       │
│    └─ PyTorch 2.7.1+cu128  ──► CUDA (NVIDIA GPU) or CPU         │
│  runtime installed at <app data>/qwen-runtime                   │
└─────────────────────────────────────────────────────────────────┘
```

Worker lifecycle details:

- Stale response ids are tolerated (a timed-out request's late reply is
  dropped, not fatal).
- A 300 s per-request timeout kills the worker.
- A dead worker auto-respawns and reloads the model once on the next
  transcribe.
- **Device selection** (**Settings → Advanced → "Qwen (ASR) device"**):
  `Auto` uses CUDA when an NVIDIA GPU is present (probed via `nvidia-smi`),
  otherwise CPU; `CUDA` and `CPU` are explicit. There is no silent fallback —
  requesting CUDA on a machine without a torch-CUDA build fails loudly.
  Changing the device unloads the loaded model.
- **Keep-warm:** while wake listening is armed the idle-unload watcher never
  unloads the model (see below); otherwise the normal `model_unload_timeout`
  applies.

### Wake mode: state machine

The logic is a pure, unit-testable state machine in `src-tauri/src/wake.rs`
(states `Off / Armed / Capturing / Transcribing / Activated`, plus a
`WakeDetector` trait so a dedicated KWS detector can replace the ASR-prefix
matcher later). Runtime glue lives in `src-tauri/src/managers/wake.rs`; the
microphone is integrated via `AudioRecorder::set_wake_listener`
(`src-tauri/src/audio_toolkit/audio/recorder.rs`); the hotkey branch is in
`src-tauri/src/shortcut/handler.rs`.

```text
            Ctrl+Space (hotkey-toggle mode) / app start (always-on)
   Off ────────────────────────────────────────────► Armed
   ▲                                                  │ speech starts
   │                                                  ▼
   │               Ctrl+Space from ANY             Capturing
   │               wake state (cancels                │ ~0.75 s silence → utterance done
   │               in-flight work)                    ▼
   │◄───────────── ──────────────────────────── Transcribing
   │              │                                   │ wake phrase matched as prefix
   │              │ no wake phrase in text            ▼
   │              │ (utterance discarded,          Activated ── 7 s window
   │              │  nothing is pasted)               │  (1–60 s configurable)
   │              │                                   │ one follow-up utterance
   │              ▼                                   ▼ accepted WITHOUT the wake phrase
   └──────────────┴─────────────────────────────► (back to Armed via Capturing)
```

How a captured utterance is handled:

- The Silero VAD forwards 30 ms speech/silence frames to the `WakeManager`
  worker thread while armed.
- Utterances shorter than **0.3 s** are dropped without ASR — no GPU inference
  on noise. A too-short noise burst also **reopens** the `Activated` window.
- A finished utterance (~0.75 s of silence) is transcribed by the current
  model, then matched by `AsrPrefixWakeDetector`: the wake phrase (plus the
  user's alternative spellings) must be the **prefix** of the recognized text.
  The comparison is token-level, lowercased, ё→е folded, with edge punctuation
  stripped — deliberately **not fuzzy**. The payload is cut from the original
  text so its original case and punctuation survive.
- «эхо» alone opens a 7 s (configurable 1–60) **Activated** window in which
  one utterance is accepted *without* the wake phrase.
- If no wake phrase is found, the utterance is discarded and nothing is pasted.
- The payload flows through the existing Echo pipeline
  (`process_transcription_output`: heuristics, voice commands, snippets) and
  the existing paste mechanism.
- Overlay states: armed («Слушаю фразу активации…»), wake-listening,
  transcribing, wake-activated; the overlay hides when wake mode is off.

## Data & download pinning

Everything the Qwen engine runs is pinned so updates can never silently change
weights or binaries.

**Runtime** (`<app data>/qwen-runtime`; ~2.8 GB download, ~5.4 GB installed).
The pinned wheel manifest is carried over from OpenWhisper
(`services/local_asr/qwen_runtime.json`) and lives at
`src-tauri/src/managers/qwen_runtime.json`: 94 archives (embedded Python
3.12.10, PyTorch 2.7.1+cu128, CUDA libraries, `qwen-asr`, …), every one pinned
by URL + SHA256.

**Models** (`<app data>/models/qwen-0.6b|qwen-1.7b`). Downloaded per-file from
Hugging Face, each file locked to an exact revision and verified with a
per-file SHA256:

| Model | HF revision |
|---|---|
| Qwen3-ASR 0.6B | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| Qwen3-ASR 1.7B | `7278e1e70fe206f11671096ffdd38061171dd6e5` |

Downloading a Qwen model auto-installs the runtime first; runtime progress is
surfaced on the model's download events in the UI.

## Settings reference

### Settings → Advanced → "Qwen (ASR) device"

| Option | Behavior |
|---|---|
| `Auto` | CUDA when an NVIDIA GPU is present (nvidia-smi probe), else CPU |
| `CUDA` | Force CUDA; fails loudly if torch-CUDA is unavailable — no silent fallback |
| `CPU` | Force CPU |

Changing the device unloads the currently loaded model.

### Settings → Advanced → Wake listening → Input mode

| Option | Behavior |
|---|---|
| `Normal dictation` | Unchanged Echo behavior (hotkey-held dictation) |
| `Wake word — always listening` | Armed at app start |
| `Wake word — hotkey toggle` | Ctrl+Space arms; Ctrl+Space again disarms from ANY wake state, canceling in-flight work. While armed the hotkey does *not* start a dictation. |

Additional wake options: the wake phrase itself (default «эхо», editable) with
user-defined alternative spellings, and the Activated-window length
(1–60 s, default 7 s).

## Attribution & licensing

- The worker, its JSONL protocol, and the runtime wheel manifest are ported
  from [OpenWhisper](https://github.com/Knuckles92/OpenWhisper)
  (MIT, Copyright (c) 2025 Knuckles92); the MIT notice is preserved in the
  source.
- The `qwen-asr` package and the Qwen3-ASR weights are
  [Apache-2.0](https://github.com/QwenLM/Qwen3-ASR).
- The bundled runtime includes Python (PSF license), PyTorch, and their
  transitive dependencies under their own licenses.
- These licenses are not changed by Echo's MIT.

## Windows build notes

Local Windows builds need **LLVM** (libclang for bindgen), the **Vulkan SDK**
(whisper-vulkan), and **Ninja**. The environment block used for local builds
(from a shell with the VS C++ dev environment imported):

```bat
set "CC=clang-cl"
set "CXX=clang-cl"
set "RC=llvm-rc"
set "CMAKE_GENERATOR=Ninja"
set "CMAKE_LINKER_TYPE=LLD"
set "CMAKE_POLICY_VERSION_MINIMUM=3.5"
set "CXXFLAGS=/EHsc"
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
set "VULKAN_SDK=C:\VulkanSDK\<version>"
:: unset MSVC leftovers that break the Ninja generator
set "CL="
set "CMAKE_GENERATOR_INSTANCE="
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
```

See also the general Windows build instructions in
[`BUILD.md`](../BUILD.md).

## Manual test checklist

From the project's Definition of Done:

1. **Normal dictation via Qwen** — dictate a Russian phrase and a mixed
   RU/EN phrase with Input mode set to `Normal dictation`; both are pasted
   correctly.
2. **Armed + ordinary speech** — arm wake mode and speak without the wake
   phrase; nothing is pasted.
3. **«Эхо, напиши …»** — the payload is pasted without the wake word, with
   case and punctuation preserved.
4. **Bare «Эхо»** — the Activated window opens; a follow-up voice command is
   accepted without the wake phrase.
5. **Second Ctrl+Space** — wake mode is fully off (overlay hidden, capture
   stopped, in-flight work canceled).
6. **Model switch 0.6B ↔ 1.7B** — switching models keeps wake listening
   working with the newly loaded model.
7. **CUDA off** — with the device forced to CUDA on a machine without
   torch-CUDA, the failure is loud and explicit, not a silent CPU fallback.
8. **App restart** — with `Wake word — always listening`, wake mode is armed
   again at startup; with the hotkey-toggle mode it starts off.

The wake state machine and prefix detector are covered by unit tests
(`cargo test -p echo --lib`: 19 wake tests among 183).
