//! Wake-word listening core: a pure state machine plus the ASR-prefix wake
//! detector. No Tauri/audio dependencies here so every transition is unit
//! testable (see the `tests` module at the bottom).
//!
//! Design (per project plan):
//! - No dedicated wake-word neural network. Silero VAD gates capture, the
//!   selected ASR transcribes the utterance, and the wake phrase is checked
//!   as a prefix of the recognized text (`AsrPrefixWakeDetector`).
//! - States: Off → Armed → Capturing → Transcribing → (Armed | Activated);
//!   a bare wake phrase ("эхо") opens a timed `Activated` window in which
//!   the next utterance is accepted without repeating the phrase.

/// How the detector classified a transcribed utterance.
#[derive(Debug, Clone, PartialEq)]
pub enum WakeResult {
    /// No wake phrase at the start — the utterance is discarded.
    NoMatch,
    /// Wake phrase followed by a payload in the same breath.
    Command { payload: String },
    /// The utterance was just the wake phrase (or one of its variants).
    BareWake,
}

/// Pluggable wake detection (plan step 15): the first implementation checks
/// the ASR text prefix; a future dedicated KWS model can slot in here
/// without touching the state machine.
pub trait WakeDetector: Send {
    fn detect(&self, utterance: &str) -> WakeResult;
}

/// Collapse a phrase for prefix comparison: lowercase, ё→е, trimmed leading
/// punctuation, collapsed whitespace. Deliberately *not* fuzzy — users add
/// their own alternative spellings instead.
pub fn normalize_phrase(text: &str) -> String {
    let lowered = text.trim().to_lowercase().replace('ё', "е");
    let stripped = lowered.trim_start_matches(|c: char| !c.is_alphanumeric());
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Wake phrase matcher that runs on the ASR output. Matches only at the
/// *start* of the utterance: "я тестировал эхо" must NOT activate.
pub struct AsrPrefixWakeDetector {
    phrases: Vec<String>,
}

impl AsrPrefixWakeDetector {
    pub fn new(phrase: &str, alternatives: &[String]) -> Self {
        let mut phrases = Vec::new();
        for p in std::iter::once(phrase).chain(alternatives.iter().map(String::as_str)) {
            let normalized = normalize_phrase(p);
            if !normalized.is_empty() && !phrases.contains(&normalized) {
                phrases.push(normalized);
            }
        }
        Self { phrases }
    }

    /// Normalized phrase variants (for logging/debugging).
    pub fn phrases(&self) -> &[String] {
        &self.phrases
    }
}

impl WakeDetector for AsrPrefixWakeDetector {
    fn detect(&self, utterance: &str) -> WakeResult {
        // Token-level match: tokens are stripped of edge punctuation and
        // normalized (lowercase, ё→е) for COMPARISON, while the payload is
        // cut from the ORIGINAL tokens so capitalization and punctuation of
        // the remainder survive ("Эхо, напиши…" → "напиши…", not "напиши…"
        // with lost case).
        let norm_token = |t: &str| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
                .replace('ё', "е")
        };
        let norm_tokens: Vec<String> = utterance.split_whitespace().map(norm_token).collect();
        let raw_tokens: Vec<&str> = utterance.split_whitespace().collect();
        // strip+lowercase never changes token count, so raw and norm align 1:1.

        for phrase in &self.phrases {
            let p_tokens: Vec<String> = phrase.split_whitespace().map(String::from).collect();
            if p_tokens.is_empty() || norm_tokens.len() < p_tokens.len() {
                continue;
            }
            if norm_tokens[..p_tokens.len()] == p_tokens[..] {
                let payload = raw_tokens[p_tokens.len()..]
                    .join(" ")
                    .trim_start_matches(|c: char| " ,.:;-!—…\"'".contains(c))
                    .trim()
                    .to_string();
                return if payload.is_empty() {
                    WakeResult::BareWake
                } else {
                    WakeResult::Command { payload }
                };
            }
        }
        WakeResult::NoMatch
    }
}

/* ───────────────────────── state machine ───────────────────────── */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeState {
    /// Wake listening disabled; normal dictation owns the hotkey.
    Off,
    /// Microphone open, VAD running, waiting for a wake phrase.
    Armed,
    /// Speech detected, capturing the utterance.
    Capturing,
    /// Utterance handed to ASR.
    Transcribing,
    /// Bare wake phrase heard; next utterance is accepted without the phrase.
    Activated,
}

#[derive(Debug, Clone)]
pub enum WakeEvent {
    /// Hotkey pressed / app start in always-listening mode.
    ToggleOn,
    /// Hotkey pressed again from any wake state — returns to Off, canceling
    /// any in-flight capture/transcription.
    ToggleOff,
    /// VAD reported speech onset.
    SpeechStarted,
    /// End of speech: the complete utterance buffer.
    UtteranceCaptured(Vec<f32>),
    /// End of speech, but the captured audio was too short/noisy to run ASR
    /// (plan step 11: no inference on noise).
    UtteranceDropped,
    /// ASR result for the captured utterance.
    Transcribed(String),
    /// The `Activated` grace window elapsed without new speech.
    ActivationTimeout,
}

/// Overlay states pushed to the floating indicator (plan step 13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeOverlay {
    Hidden,
    Armed,
    Listening,
    Processing,
    Activated,
}

impl From<WakeState> for WakeOverlay {
    fn from(state: WakeState) -> Self {
        match state {
            WakeState::Off => WakeOverlay::Hidden,
            WakeState::Armed => WakeOverlay::Armed,
            WakeState::Capturing => WakeOverlay::Listening,
            WakeState::Transcribing => WakeOverlay::Processing,
            WakeState::Activated => WakeOverlay::Activated,
        }
    }
}

/// Side effects the host must perform for a transition. Keeping them as data
/// makes the machine testable without a running app.
#[derive(Debug, Clone, PartialEq)]
pub enum WakeEffect {
    StartListening,
    StopListening,
    SetOverlay(WakeOverlay),
    Transcribe(Vec<f32>),
    /// Payload passed the wake check — run it through the existing Echo
    /// pipeline (heuristics, voice commands) and paste it.
    PastePayload(String),
    /// The captured utterance had no wake phrase — nothing is pasted.
    DiscardUtterance,
    ScheduleActivationTimeout,
    CancelActivationTimeout,
}

pub struct WakeStateMachine {
    state: WakeState,
    detector: AsrPrefixWakeDetector,
    /// Set while capturing an utterance that began in the `Activated` window
    /// (the payload must not require a wake phrase).
    activated_capture: bool,
}

impl WakeStateMachine {
    pub fn new(detector: AsrPrefixWakeDetector) -> Self {
        Self {
            state: WakeState::Off,
            detector,
            activated_capture: false,
        }
    }

    pub fn state(&self) -> WakeState {
        self.state
    }

    pub fn detector(&self) -> &AsrPrefixWakeDetector {
        &self.detector
    }

    pub fn handle(&mut self, event: WakeEvent) -> Vec<WakeEffect> {
        match (&self.state, event) {
            /* ── global hotkey semantics (plan step 7) ────────────────── */
            (_, WakeEvent::ToggleOff) if self.state != WakeState::Off => {
                self.activated_capture = false;
                self.state = WakeState::Off;
                vec![
                    WakeEffect::CancelActivationTimeout,
                    WakeEffect::StopListening,
                    WakeEffect::SetOverlay(WakeOverlay::Hidden),
                ]
            }
            (WakeState::Off, WakeEvent::ToggleOn) => {
                self.activated_capture = false;
                self.state = WakeState::Armed;
                vec![
                    WakeEffect::StartListening,
                    WakeEffect::SetOverlay(WakeOverlay::Armed),
                ]
            }
            /* ── Armed: wait for speech ───────────────────────────────── */
            (WakeState::Armed, WakeEvent::SpeechStarted) => {
                self.activated_capture = false;
                self.state = WakeState::Capturing;
                vec![WakeEffect::SetOverlay(WakeOverlay::Listening)]
            }
            /* ── Capturing: collect until silence hands us the buffer ── */
            (WakeState::Capturing, WakeEvent::UtteranceCaptured(samples)) => {
                self.state = WakeState::Transcribing;
                vec![
                    WakeEffect::SetOverlay(WakeOverlay::Processing),
                    WakeEffect::Transcribe(samples),
                ]
            }
            /* ── Too short to transcribe: back to waiting ────────────── */
            (WakeState::Capturing, WakeEvent::UtteranceDropped) => {
                if self.activated_capture {
                    // Noise inside the activation window: reopen the grace period.
                    self.activated_capture = false;
                    self.state = WakeState::Activated;
                    vec![
                        WakeEffect::ScheduleActivationTimeout,
                        WakeEffect::SetOverlay(WakeOverlay::Activated),
                    ]
                } else {
                    self.activated_capture = false;
                    self.state = WakeState::Armed;
                    vec![WakeEffect::SetOverlay(WakeOverlay::Armed)]
                }
            }
            /* ── Wake check on the ASR result (plan step 8) ──────────── */
            (WakeState::Transcribing, WakeEvent::Transcribed(text)) => {
                let was_activated_capture = self.activated_capture;
                self.activated_capture = false;
                match self.detector.detect(&text) {
                    WakeResult::Command { payload } => {
                        self.state = WakeState::Armed;
                        vec![
                            WakeEffect::PastePayload(payload),
                            WakeEffect::SetOverlay(WakeOverlay::Armed),
                        ]
                    }
                    WakeResult::BareWake if !was_activated_capture => {
                        self.state = WakeState::Activated;
                        vec![
                            WakeEffect::ScheduleActivationTimeout,
                            WakeEffect::SetOverlay(WakeOverlay::Activated),
                        ]
                    }
                    // A bare "эхо" inside an already-activated window is just
                    // a repeated wake — stay armed either way.
                    WakeResult::BareWake => {
                        self.state = WakeState::Armed;
                        vec![WakeEffect::SetOverlay(WakeOverlay::Armed)]
                    }
                    WakeResult::NoMatch if was_activated_capture => {
                        // Inside the activation window the phrase requirement
                        // is waived — treat the whole utterance as the payload.
                        self.state = WakeState::Armed;
                        let payload = text.trim().to_string();
                        if payload.is_empty() {
                            vec![WakeEffect::SetOverlay(WakeOverlay::Armed)]
                        } else {
                            vec![
                                WakeEffect::PastePayload(payload),
                                WakeEffect::SetOverlay(WakeOverlay::Armed),
                            ]
                        }
                    }
                    WakeResult::NoMatch => {
                        self.state = WakeState::Armed;
                        vec![
                            WakeEffect::DiscardUtterance,
                            WakeEffect::SetOverlay(WakeOverlay::Armed),
                        ]
                    }
                }
            }
            /* ── Activated: one phrase accepted without the wake word ── */
            (WakeState::Activated, WakeEvent::SpeechStarted) => {
                self.activated_capture = true;
                self.state = WakeState::Capturing;
                vec![
                    WakeEffect::CancelActivationTimeout,
                    WakeEffect::SetOverlay(WakeOverlay::Listening),
                ]
            }
            (WakeState::Activated, WakeEvent::ActivationTimeout) => {
                self.activated_capture = false;
                self.state = WakeState::Armed;
                vec![WakeEffect::SetOverlay(WakeOverlay::Armed)]
            }
            /* ── anything else is a no-op (keeps the machine total) ──── */
            _ => Vec::new(),
        }
    }
}

/* ───────────────────────── tests ───────────────────────── */

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> WakeStateMachine {
        WakeStateMachine::new(AsrPrefixWakeDetector::new(
            "эхо",
            &["эко".to_string(), "эй эхо".to_string()],
        ))
    }

    fn arm() -> WakeStateMachine {
        let mut m = machine();
        m.handle(WakeEvent::ToggleOn);
        m
    }

    fn transcribe(m: &mut WakeStateMachine, text: &str) -> Vec<WakeEffect> {
        m.handle(WakeEvent::SpeechStarted);
        m.handle(WakeEvent::UtteranceCaptured(vec![0.0; 1600]));
        m.handle(WakeEvent::Transcribed(text.to_string()))
    }

    /* detector */

    #[test]
    fn detector_matches_prefix_and_extracts_payload() {
        let d = AsrPrefixWakeDetector::new("эхо", &[]);
        assert_eq!(
            d.detect("Эхо, напиши в Z-code, что нужно исправить обработку ошибок"),
            WakeResult::Command {
                payload: "напиши в Z-code, что нужно исправить обработку ошибок".to_string()
            }
        );
    }

    #[test]
    fn detector_bare_wake() {
        let d = AsrPrefixWakeDetector::new("эхо", &[]);
        assert_eq!(d.detect("эхо"), WakeResult::BareWake);
        assert_eq!(d.detect("Эхо!"), WakeResult::BareWake);
    }

    #[test]
    fn detector_ignores_mid_sentence_wake() {
        let d = AsrPrefixWakeDetector::new("эхо", &[]);
        assert_eq!(d.detect("Я сегодня тестировал Эхо"), WakeResult::NoMatch);
    }

    #[test]
    fn detector_uses_alternatives_and_normalization() {
        let d = AsrPrefixWakeDetector::new("Эхо", &["эко".to_string(), "эй эхо".to_string()]);
        assert_eq!(
            d.detect("Эко, сделай новый тест"),
            WakeResult::Command {
                payload: "сделай новый тест".to_string()
            }
        );
        // ё/е and case collapse
        assert_eq!(
            d.detect("ЭЙ ЭХО, запусти сборку"),
            WakeResult::Command {
                payload: "запусти сборку".to_string()
            }
        );
    }

    #[test]
    fn detector_no_match_on_ordinary_speech() {
        let d = AsrPrefixWakeDetector::new("эхо", &[]);
        assert_eq!(
            d.detect("тут вообще какой-то странный баг"),
            WakeResult::NoMatch
        );
    }

    /* state machine — the scenarios from plan step 16 */

    #[test]
    fn off_plus_hotkey_arms() {
        let mut m = machine();
        let effects = m.handle(WakeEvent::ToggleOn);
        assert_eq!(m.state(), WakeState::Armed);
        assert!(effects.contains(&WakeEffect::StartListening));
    }

    #[test]
    fn armed_plus_hotkey_goes_off() {
        let mut m = arm();
        let effects = m.handle(WakeEvent::ToggleOff);
        assert_eq!(m.state(), WakeState::Off);
        assert!(effects.contains(&WakeEffect::StopListening));
    }

    #[test]
    fn armed_ordinary_speech_discarded_no_paste() {
        let mut m = arm();
        let effects = transcribe(&mut m, "тут вообще какой-то странный баг");
        assert_eq!(m.state(), WakeState::Armed);
        assert!(effects.contains(&WakeEffect::DiscardUtterance));
        assert!(!effects
            .iter()
            .any(|e| matches!(e, WakeEffect::PastePayload(_))));
    }

    #[test]
    fn armed_wake_command_pastes_payload() {
        let mut m = arm();
        let effects = transcribe(&mut m, "Эхо, привет");
        assert_eq!(m.state(), WakeState::Armed);
        assert_eq!(
            effects,
            vec![
                WakeEffect::PastePayload("привет".to_string()),
                WakeEffect::SetOverlay(WakeOverlay::Armed),
            ]
        );
    }

    #[test]
    fn armed_bare_wake_activates() {
        let mut m = arm();
        let effects = transcribe(&mut m, "эхо");
        assert_eq!(m.state(), WakeState::Activated);
        assert!(effects.contains(&WakeEffect::ScheduleActivationTimeout));
    }

    #[test]
    fn activated_accepts_phrase_without_wake_then_returns_armed() {
        let mut m = arm();
        transcribe(&mut m, "эхо");
        let effects = transcribe(&mut m, "привет");
        assert_eq!(m.state(), WakeState::Armed);
        assert_eq!(
            effects,
            vec![
                WakeEffect::PastePayload("привет".to_string()),
                WakeEffect::SetOverlay(WakeOverlay::Armed),
            ]
        );
    }

    #[test]
    fn activated_timeout_returns_armed() {
        let mut m = arm();
        transcribe(&mut m, "эхо");
        let effects = m.handle(WakeEvent::ActivationTimeout);
        assert_eq!(m.state(), WakeState::Armed);
        assert_eq!(effects, vec![WakeEffect::SetOverlay(WakeOverlay::Armed)]);
    }

    #[test]
    fn processing_plus_hotkey_cancels_to_off() {
        let mut m = arm();
        m.handle(WakeEvent::SpeechStarted);
        m.handle(WakeEvent::UtteranceCaptured(vec![0.0; 1600]));
        assert_eq!(m.state(), WakeState::Transcribing);
        let effects = m.handle(WakeEvent::ToggleOff);
        assert_eq!(m.state(), WakeState::Off);
        assert!(effects.contains(&WakeEffect::StopListening));
        assert!(effects.contains(&WakeEffect::CancelActivationTimeout));
    }

    #[test]
    fn capturing_plus_hotkey_cancels_to_off() {
        let mut m = arm();
        m.handle(WakeEvent::SpeechStarted);
        let effects = m.handle(WakeEvent::ToggleOff);
        assert_eq!(m.state(), WakeState::Off);
        assert!(effects.contains(&WakeEffect::StopListening));
    }

    #[test]
    fn dropped_utterance_returns_armed() {
        let mut m = arm();
        m.handle(WakeEvent::SpeechStarted);
        let effects = m.handle(WakeEvent::UtteranceDropped);
        assert_eq!(m.state(), WakeState::Armed);
        assert_eq!(effects, vec![WakeEffect::SetOverlay(WakeOverlay::Armed)]);
    }

    #[test]
    fn dropped_utterance_in_activated_reopens_window() {
        let mut m = arm();
        transcribe(&mut m, "эхо");
        m.handle(WakeEvent::SpeechStarted);
        let effects = m.handle(WakeEvent::UtteranceDropped);
        assert_eq!(m.state(), WakeState::Activated);
        assert!(effects.contains(&WakeEffect::ScheduleActivationTimeout));
    }

    #[test]
    fn activated_speech_started_cancels_timeout() {
        let mut m = arm();
        transcribe(&mut m, "эхо");
        let effects = m.handle(WakeEvent::SpeechStarted);
        assert_eq!(m.state(), WakeState::Capturing);
        assert!(effects.contains(&WakeEffect::CancelActivationTimeout));
    }

    #[test]
    fn events_ignored_in_wrong_state_are_noops() {
        let mut m = machine();
        assert!(m.handle(WakeEvent::SpeechStarted).is_empty());
        assert!(m.handle(WakeEvent::ActivationTimeout).is_empty());
        assert!(m.handle(WakeEvent::ToggleOff).is_empty()); // already off
        let mut a = arm();
        assert!(a.handle(WakeEvent::ActivationTimeout).is_empty());
    }

    #[test]
    fn toggle_on_in_armed_is_noop() {
        let mut m = arm();
        assert!(m.handle(WakeEvent::ToggleOn).is_empty());
        assert_eq!(m.state(), WakeState::Armed);
    }
}
