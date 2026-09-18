//! Runtime glue for wake-word listening: connects the pure state machine
//! (`crate::wake`) to the microphone VAD stream, the ASR engine, the overlay
//! indicator and the hotkey (plan steps 6–14).
//!
//! A single worker thread owns the state machine. Audio frames, hotkey
//! toggles, in-flight transcription results and activation timeouts are all
//! serialized through one channel, so no locking of wake state is needed.

use crate::audio_toolkit::WakeAudioFrame;
use crate::managers::audio::AudioRecordingManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::get_settings;
use crate::wake::{
    AsrPrefixWakeDetector, WakeEffect, WakeEvent, WakeOverlay, WakeState, WakeStateMachine,
};
use log::{info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

/// Utterances shorter than this (0.3 s) are dropped without ASR — plan step 11:
/// no GPU inference on noise.
const MIN_UTTERANCE_SAMPLES: usize = 4800;
/// Extra silence (300 ms of 30 ms frames) after the VAD hangover before an
/// utterance is considered finished.
const HANGOVER_FRAMES: u32 = 10;

enum LoopMsg {
    Command(WakeCommand),
    Audio(WakeAudioFrame),
    Transcript {
        generation: u64,
        result: Result<String, String>,
    },
    ActivationTimeoutFired {
        generation: u64,
    },
}

enum WakeCommand {
    Toggle,
    SetOn(bool),
}

/// Public handle: hotkey and settings commands funnel in here.
pub struct WakeManager {
    tx: Sender<LoopMsg>,
    listening: Arc<AtomicBool>,
}

impl WakeManager {
    pub fn new(app: AppHandle) -> Self {
        let (tx, rx) = mpsc::channel();
        let listening = Arc::new(AtomicBool::new(false));
        let listening_thread = listening.clone();
        let tx_thread = tx.clone();
        thread::spawn(move || run(app, tx_thread, rx, listening_thread));
        Self { tx, listening }
    }

    /// Hotkey toggle: Off ↔ Armed (plan step 6, "hotkey toggle" mode).
    pub fn toggle(&self) {
        let _ = self.tx.send(LoopMsg::Command(WakeCommand::Toggle));
    }

    /// Programmatic switch used by the input-mode setting and app startup.
    pub fn set_on(&self, on: bool) {
        let _ = self.tx.send(LoopMsg::Command(WakeCommand::SetOn(on)));
    }

    /// Whether wake listening is currently enabled (any state ≠ Off).
    pub fn is_listening(&self) -> bool {
        self.listening.load(Ordering::SeqCst)
    }
}

fn make_machine(app: &AppHandle) -> WakeStateMachine {
    let settings = get_settings(app);
    let detector =
        AsrPrefixWakeDetector::new(&settings.wake_phrase, &settings.wake_alternative_phrases);
    WakeStateMachine::new(detector)
}

fn run(app: AppHandle, tx: Sender<LoopMsg>, rx: Receiver<LoopMsg>, listening: Arc<AtomicBool>) {
    let mut machine: Option<WakeStateMachine> = None;
    // Monotonic token: bumped on turn-off / timeout cancel so stale in-flight
    // transcripts and timers are ignored.
    let mut generation: u64 = 0;
    let mut utterance: Vec<f32> = Vec::new();
    let mut silence_frames: u32 = 0;
    let mut in_speech = false;
    let mut max_utterance_samples: usize = 30 * 16000;

    while let Ok(msg) = rx.recv() {
        match msg {
            LoopMsg::Command(cmd) => {
                let want_on = match cmd {
                    WakeCommand::Toggle => !listening.load(Ordering::SeqCst),
                    WakeCommand::SetOn(on) => on,
                };
                let is_on = listening.load(Ordering::SeqCst);
                if want_on == is_on {
                    continue;
                }
                if want_on {
                    let mut m = make_machine(&app);
                    max_utterance_samples =
                        (get_settings(&app).wake_max_utterance_secs.max(1) as usize) * 16000;
                    listening.store(true, Ordering::SeqCst);
                    generation += 1;
                    let effects = m.handle(WakeEvent::ToggleOn);
                    for effect in effects {
                        execute_effect(
                            &app,
                            &tx,
                            effect,
                            generation,
                            &mut utterance,
                            &mut in_speech,
                            &mut silence_frames,
                        );
                    }
                    info!(
                        "Wake listening ARMED (phrase: {:?})",
                        m.detector().phrases()
                    );
                    machine = Some(m);
                } else {
                    listening.store(false, Ordering::SeqCst);
                    generation += 1; // invalidate in-flight transcripts/timers
                    if let Some(ref mut m) = machine {
                        for effect in m.handle(WakeEvent::ToggleOff) {
                            execute_effect(
                                &app,
                                &tx,
                                effect,
                                generation,
                                &mut utterance,
                                &mut in_speech,
                                &mut silence_frames,
                            );
                        }
                    }
                    utterance.clear();
                    in_speech = false;
                    silence_frames = 0;
                    info!("Wake listening OFF");
                }
            }
            LoopMsg::Audio(frame) => {
                let Some(ref mut m) = machine else { continue };
                if m.state() == WakeState::Off {
                    continue;
                }
                let max_len = max_utterance_samples;
                let effects = match frame {
                    WakeAudioFrame::Speech(samples) => {
                        if !in_speech {
                            in_speech = true;
                            silence_frames = 0;
                            utterance.clear();
                            m.handle(WakeEvent::SpeechStarted)
                        } else {
                            utterance.extend_from_slice(&samples);
                            if utterance.len() >= max_len {
                                // Force-finalize an overly long utterance.
                                let buf = std::mem::take(&mut utterance);
                                in_speech = false;
                                silence_frames = 0;
                                m.handle(WakeEvent::UtteranceCaptured(buf))
                            } else {
                                Vec::new()
                            }
                        }
                    }
                    WakeAudioFrame::Silence => {
                        if !in_speech {
                            Vec::new()
                        } else {
                            silence_frames += 1;
                            if silence_frames >= HANGOVER_FRAMES {
                                let buf = std::mem::take(&mut utterance);
                                in_speech = false;
                                silence_frames = 0;
                                if buf.len() >= MIN_UTTERANCE_SAMPLES {
                                    m.handle(WakeEvent::UtteranceCaptured(buf))
                                } else {
                                    m.handle(WakeEvent::UtteranceDropped)
                                }
                            } else {
                                Vec::new()
                            }
                        }
                    }
                };
                for effect in effects {
                    execute_effect(
                        &app,
                        &tx,
                        effect,
                        generation,
                        &mut utterance,
                        &mut in_speech,
                        &mut silence_frames,
                    );
                }
            }
            LoopMsg::Transcript {
                generation: g,
                result,
            } => {
                if g != generation {
                    continue; // stale (listening turned off meanwhile)
                }
                match result {
                    Ok(text) => {
                        if let Some(ref mut m) = machine {
                            let effects = m.handle(WakeEvent::Transcribed(text));
                            for effect in effects {
                                execute_effect(
                                    &app,
                                    &tx,
                                    effect,
                                    generation,
                                    &mut utterance,
                                    &mut in_speech,
                                    &mut silence_frames,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Wake transcription failed: {}", e);
                        // Return to Armed; no paste, no state downgrade.
                        if let Some(ref mut m) = machine {
                            if m.state() == WakeState::Transcribing {
                                let effects = m.handle(WakeEvent::UtteranceDropped);
                                for effect in effects {
                                    execute_effect(
                                        &app,
                                        &tx,
                                        effect,
                                        generation,
                                        &mut utterance,
                                        &mut in_speech,
                                        &mut silence_frames,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            LoopMsg::ActivationTimeoutFired { generation: g } => {
                if g != generation {
                    continue;
                }
                if let Some(ref mut m) = machine {
                    let effects = m.handle(WakeEvent::ActivationTimeout);
                    for effect in effects {
                        execute_effect(
                            &app,
                            &tx,
                            effect,
                            generation,
                            &mut utterance,
                            &mut in_speech,
                            &mut silence_frames,
                        );
                    }
                }
            }
        }
    }
}

/// Execute one state-machine effect against the real world.
#[allow(clippy::too_many_arguments)]
fn execute_effect(
    app: &AppHandle,
    tx: &Sender<LoopMsg>,
    effect: WakeEffect,
    generation: u64,
    _utterance: &mut Vec<f32>,
    _in_speech: &mut bool,
    _silence_frames: &mut u32,
) {
    match effect {
        WakeEffect::StartListening => {
            // Channel: recorder frames → forwarder → our loop.
            let (audio_tx, audio_rx) = mpsc::channel::<WakeAudioFrame>();
            let forward_tx = tx.clone();
            thread::spawn(move || {
                for frame in audio_rx {
                    if forward_tx.send(LoopMsg::Audio(frame)).is_err() {
                        break;
                    }
                }
            });
            let audio = app.state::<Arc<AudioRecordingManager>>();
            if let Err(e) = audio.set_wake_listener(Some(audio_tx)) {
                warn!("Failed to attach wake listener: {}", e);
            }
        }
        WakeEffect::StopListening => {
            let audio = app.state::<Arc<AudioRecordingManager>>();
            if let Err(e) = audio.set_wake_listener(None) {
                warn!("Failed to detach wake listener: {}", e);
            }
        }
        WakeEffect::SetOverlay(state) => {
            let state_str = match state {
                WakeOverlay::Hidden => {
                    crate::platform::overlay::hide_recording_overlay(app);
                    return;
                }
                WakeOverlay::Armed => "armed",
                WakeOverlay::Listening => "wake-listening",
                WakeOverlay::Processing => "transcribing",
                WakeOverlay::Activated => "wake-activated",
            };
            crate::platform::overlay::show_overlay_state(app, state_str);
        }
        WakeEffect::Transcribe(samples) => {
            let app = app.clone();
            let tx = tx.clone();
            thread::spawn(move || {
                let started = Instant::now();
                let tm = app.state::<Arc<TranscriptionManager>>();
                let result = tm.transcribe(samples).map_err(|e| e.to_string());
                info!("Wake ASR finished in {:?}", started.elapsed());
                let _ = tx.send(LoopMsg::Transcript { generation, result });
            });
        }
        WakeEffect::PastePayload(payload) => {
            let app = app.clone();
            thread::spawn(move || {
                // Reuse the existing Echo pipeline (heuristics, voice
                // commands, snippets) — wake adds only prefix stripping
                // (plan step 14).
                let processed = tauri::async_runtime::block_on(
                    crate::actions::process_transcription_output(&app, &payload, false),
                );
                if processed.final_text.is_empty() {
                    return;
                }
                let app_for_main = app.clone();
                let _ = app.run_on_main_thread(move || {
                    if let Err(e) = crate::utils::paste(
                        processed.final_text,
                        app_for_main.clone(),
                        processed.force_submit,
                    ) {
                        log::error!("Wake paste failed: {}", e);
                    }
                });
            });
        }
        WakeEffect::DiscardUtterance => {
            // The utterance had no wake phrase: nothing to do — the machine
            // already transitioned back to Armed and updated the overlay.
        }
        WakeEffect::ScheduleActivationTimeout => {
            let tx = tx.clone();
            let app = app.clone();
            let secs = get_settings(&app).wake_activation_timeout_secs.max(1);
            thread::spawn(move || {
                thread::sleep(Duration::from_secs(secs));
                let _ = tx.send(LoopMsg::ActivationTimeoutFired { generation });
            });
        }
        WakeEffect::CancelActivationTimeout => {
            // Caller bumps `generation` right before dispatching effects, so
            // stale timeout messages are ignored on arrival.
        }
    }
}
