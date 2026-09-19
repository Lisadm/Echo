#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use crate::platform::audio_feedback::{
    play_feedback_sound, play_feedback_sound_blocking, SoundType,
};
use crate::platform::tray::{change_tray_icon, TrayIconState};
use crate::settings::{get_settings, AppSettings, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::utils::{
    self, show_preparing_overlay, show_processing_overlay, show_recording_overlay,
    show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::Manager;
use tauri::{AppHandle, Emitter};

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Keep only the last `max_chars` characters (char-safe) of a subtitle so the
/// on-screen overlay stays a readable tail instead of growing unbounded.
fn tail_chars(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        s.to_string()
    } else {
        s.chars().skip(count - max_chars).collect()
    }
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

async fn post_process_transcription(settings: &AppSettings, transcription: &str) -> Option<String> {
    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return None;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    run_llm_prompt(settings, &prompt, transcription).await
}

/// Run an arbitrary instruction `prompt` over `transcription` using the
/// configured post-processing provider/model. Shared by selected-prompt
/// post-processing and Command Mode (translate / shorten / formal).
async fn run_llm_prompt(
    settings: &AppSettings,
    prompt: &str,
    transcription: &str,
) -> Option<String> {
    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("LLM step enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "LLM step skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    debug!(
        "Starting LLM step with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    // Disable reasoning for providers where post-processing rarely benefits from it.
    // - custom: top-level reasoning_effort (works for local OpenAI-compat servers)
    // - openrouter: nested reasoning object; exclude:true also keeps reasoning text
    //   out of the response so it can't pollute structured-output JSON parsing
    let (reasoning_effort, reasoning) = match provider.id.as_str() {
        "custom" => (Some("none".to_string()), None),
        "openrouter" => (
            None,
            Some(crate::llm_client::ReasoningConfig {
                effort: Some("none".to_string()),
                exclude: Some(true),
            }),
        ),
        _ => (None, None),
    };

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = build_system_prompt(prompt);
        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            Some(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
            reasoning_effort.clone(),
            reasoning.clone(),
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = strip_invisible_chars(transcription_value);
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            return Some(result);
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            return Some(strip_invisible_chars(&content));
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        return Some(strip_invisible_chars(&content));
                    }
                }
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let processed_prompt = prompt.replace("${output}", transcription);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt,
        reasoning_effort,
        reasoning,
    )
    .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(&content);
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            Some(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

async fn maybe_convert_chinese_variant(
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    // Check if language is set to Simplified or Traditional Chinese
    let is_simplified = settings.selected_language == "zh-Hans";
    let is_traditional = settings.selected_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("selected_language is not Simplified or Traditional Chinese; skipping translation");
        return None;
    }

    debug!(
        "Starting Chinese translation using OpenCC for language: {}",
        settings.selected_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
    /// Set by the Command Mode "press enter" suffix — submit after pasting even
    /// if the global auto-submit setting is off.
    pub force_submit: bool,
}

#[cfg(target_os = "windows")]
fn get_active_window_title() -> Option<String> {
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowTextW};
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.0 == std::ptr::null_mut() {
            return None;
        }
        let mut buffer = [0u16; 512];
        let len = GetWindowTextW(hwnd, &mut buffer);
        if len == 0 {
            return None;
        }
        Some(String::from_utf16_lossy(&buffer[..len as usize]))
    }
}

#[cfg(not(target_os = "windows"))]
fn get_active_window_title() -> Option<String> {
    None
}

/// Optionally translate `text` into `target`. Translation is strictly additive and
/// must NEVER break dictation: on any translator error (server down, timeout, empty
/// result) the ORIGINAL text is returned, so the user always gets their words —
/// untranslated at worst, never lost.
fn maybe_translate(
    text: &str,
    enabled: bool,
    target: crate::translate::Lang,
    translator: &dyn crate::translate::Translator,
) -> String {
    if !enabled || text.trim().is_empty() {
        return text.to_string();
    }
    match translator.translate(text, target) {
        Ok(translated) => translated,
        Err(e) => {
            warn!("Translation failed; pasting original dictation verbatim: {e:#}");
            text.to_string()
        }
    }
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
) -> ProcessedTranscription {
    use crate::voice_commands::{
        apply_self_correction, detect_prefix_command, expand_snippets, format_spoken_list,
        strip_submit_command,
    };

    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;
    let mut force_submit = false;

    // Voice capture: a configured trigger phrase saves the remainder to a folder
    // (as a markdown note) instead of pasting it into the active app.
    let capture_folder = settings.capture_folder.trim().to_string();
    if !capture_folder.is_empty() {
        let phrases =
            crate::voice_commands::parse_capture_phrases(&settings.capture_trigger_phrases);
        let phrase_refs: Vec<&str> = phrases.iter().map(|s| s.as_str()).collect();
        if let Some(body) = crate::voice_commands::detect_capture(&final_text, &phrase_refs) {
            let now = chrono::Local::now();
            let lang = Some(settings.selected_language.as_str());
            let filename = crate::capture::capture_filename(now);
            let note = crate::capture::build_note(&body, now, lang);
            match crate::capture::write_capture(
                std::path::Path::new(&capture_folder),
                &filename,
                &note,
            ) {
                Ok(path) => {
                    let _ = app.emit(
                        "voice-capture",
                        serde_json::json!({ "ok": true, "path": path.to_string_lossy() }),
                    );
                    return ProcessedTranscription {
                        final_text: String::new(),
                        post_processed_text: None,
                        post_process_prompt: None,
                        force_submit: false,
                    };
                }
                Err(e) => {
                    error!("Voice capture failed: {e:#}");
                    let _ = app.emit(
                        "voice-capture",
                        serde_json::json!({ "ok": false, "error": e.to_string() }),
                    );
                    // fall through: dictate normally so words aren't lost
                }
            }
        }
    }

    // ── Command Mode ────────────────────────────────────────────────────────
    // A trailing "press enter" submits after paste; a leading instruction
    // ("translate to english", "make shorter", ...) routes the remainder
    // through the LLM and replaces the dictated text with the result.
    if settings.command_mode_enabled {
        let (stripped, submit) = strip_submit_command(&final_text);
        force_submit = submit;
        final_text = stripped;

        if let Some(detected) = detect_prefix_command(&final_text) {
            let instruction = detected.command.instruction();
            debug!("Command Mode: detected {:?}", detected.command);
            if let Some(result) = run_llm_prompt(&settings, &instruction, &detected.remainder).await
            {
                return ProcessedTranscription {
                    post_processed_text: Some(result.clone()),
                    post_process_prompt: Some(instruction),
                    final_text: result,
                    force_submit,
                };
            }
            // LLM unavailable/failed: fall back to the dictated payload sans the
            // command prefix, and continue the normal pipeline below.
            warn!("Command Mode LLM step failed; using dictated payload verbatim");
            final_text = detected.remainder;
        }
    }

    if settings.auto_punctuate || settings.auto_capitalize {
        let h = crate::heuristics::Heuristics::new();
        if settings.auto_punctuate {
            final_text = h.auto_punctuate(&final_text);
        }
        // Self-correction runs between punctuation and capitalization: it relies
        // on the comma the auto-punctuator inserts before the correction cue,
        // and leaves a clean string for the capitalizer to finish.
        if settings.self_correction_enabled {
            final_text = apply_self_correction(&final_text);
        }
        if settings.auto_capitalize {
            final_text = h.auto_capitalize(&final_text);
        }
    } else if settings.self_correction_enabled {
        final_text = apply_self_correction(&final_text);
    }

    if settings.spoken_lists_enabled {
        final_text = format_spoken_list(&final_text);
    }

    // Context-Aware Formatting
    if let Some(title) = get_active_window_title() {
        let title_lower = title.to_lowercase();
        debug!("Active window detected: {}", title);

        let is_code_editor = title_lower.contains("visual studio code")
            || title_lower.contains("cursor")
            || title_lower.contains("zed")
            || title_lower.contains("sublime")
            || title_lower.contains("intellij")
            || title_lower.contains("webstorm")
            || title_lower.contains("pycharm")
            || title_lower.contains("clion")
            || title_lower.contains("neovim")
            || title_lower.contains("visual studio");

        if is_code_editor {
            // Strip trailing punctuation for code editors
            final_text = final_text
                .trim_end_matches(|c: char| c.is_ascii_punctuation())
                .to_string();
            debug!("Applying 'Code Editor' formatting: stripped trailing punctuation");

            // Explicit CamelCase trigger: "camel case [text]"
            let lower_final = final_text.to_lowercase();
            if let Some(rest) = lower_final.strip_prefix("camel case ") {
                // Feed the lowercased remainder: to_camel_case re-cases every
                // word itself, so this is equivalent to slicing the original and
                // avoids byte-index panics on multibyte input.
                let h = crate::heuristics::Heuristics::new();
                final_text = h.to_camel_case(rest);
                debug!("Applying 'CamelCase' formatting via trigger");
            }
        }
    }

    if let Some(converted_text) = maybe_convert_chinese_variant(&settings, &final_text).await {
        final_text = converted_text;
    }

    if post_process {
        if let Some(processed_text) = post_process_transcription(&settings, &final_text).await {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;

            if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| &prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    }

    // Optional translation: speak in one language, paste another. Runs after
    // post-processing but BEFORE snippet expansion so canned snippets (URLs,
    // signatures) stay verbatim. Offloaded to a blocking thread so it never stalls
    // the async pipeline, and graceful on failure (returns the original text).
    if settings.translate_enabled {
        let target = settings.translate_target;
        let translator = crate::translate::ServerTranslator {
            provider: crate::settings::PostProcessProvider {
                id: "translate-local".to_string(),
                label: "Translate".to_string(),
                base_url: settings.translate_base_url.clone(),
                allow_base_url_edit: false,
                models_endpoint: None,
                supports_structured_output: false,
            },
            model: settings.translate_model.clone(),
            api_key: String::new(),
        };
        let src = final_text.clone();
        final_text =
            tokio::task::spawn_blocking(move || maybe_translate(&src, true, target, &translator))
                .await
                .unwrap_or(final_text);
    }

    // Snippet expansion is the final transform so canned text (URLs, signatures)
    // is inserted verbatim and never reflowed by the LLM or the capitalizer.
    if !settings.snippets.is_empty() {
        final_text = expand_snippets(&final_text, &settings.snippets);
    }

    // Record the processed text whenever the pipeline changed the raw output.
    if post_processed_text.is_none() && final_text != transcription {
        post_processed_text = Some(final_text.clone());
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
        force_submit,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Load ASR model and VAD model in parallel
        tm.initiate_model_load();
        let rm_clone = Arc::clone(&rm);
        std::thread::spawn(move || {
            if let Err(e) = rm_clone.preload_vad() {
                debug!("VAD pre-load failed: {}", e);
            }
        });

        let binding_id = binding_id.to_string();
        change_tray_icon(app, TrayIconState::Recording);
        // Show the pill in "preparing" state first; we flip it to "recording"
        // only once the audio stream is actually live (below), so the user
        // doesn't start talking into a mic that isn't capturing yet.
        show_preparing_overlay(app);

        // Get the microphone mode to determine audio feedback timing
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<String> = None;
        if is_always_on {
            // Always-on mode: Play audio feedback immediately, then apply mute after sound finishes
            debug!("Always-on mode: Playing audio feedback immediately");
            let rm_clone = Arc::clone(&rm);
            let app_clone = app.clone();
            // The blocking helper exits immediately if audio feedback is disabled,
            // so we can always reuse this thread to ensure mute happens right after playback.
            std::thread::spawn(move || {
                // Always-on mic is already capturing, so flip to "recording"
                // together with the start cue.
                show_recording_overlay(&app_clone);
                play_feedback_sound_blocking(&app_clone, SoundType::Start);
                rm_clone.apply_mute();
            });

            if let Err(e) = rm.try_start_recording(&binding_id) {
                debug!("Recording failed: {}", e);
                recording_error = Some(e);
            }
        } else {
            // On-demand mode: Start recording first, then play audio feedback, then apply mute
            // This allows the microphone to be activated before playing the sound
            debug!("On-demand mode: Starting recording first, then audio feedback");
            let recording_start_time = Instant::now();
            match rm.try_start_recording(&binding_id) {
                Ok(()) => {
                    debug!("Recording started in {:?}", recording_start_time.elapsed());
                    // Small delay to ensure microphone stream is active
                    let app_clone = app.clone();
                    let rm_clone = Arc::clone(&rm);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        debug!("Handling delayed audio feedback/mute sequence");
                        // The stream is active now: flip the pill to "recording"
                        // in lock-step with the audible start cue.
                        show_recording_overlay(&app_clone);
                        // Helper handles disabled audio feedback by returning early, so we reuse it
                        // to keep mute sequencing consistent in every mode.
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                        rm_clone.apply_mute();
                    });
                }
                Err(e) => {
                    debug!("Failed to start recording: {}", e);
                    recording_error = Some(e);
                }
            }
        }

        if recording_error.is_none() {
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);

            // Phase 4: Streaming Subtitles
            if settings.subtitle_overlay {
                let app_handle = app.clone();
                let rm_clone = Arc::clone(&rm);
                let tm_clone = Arc::clone(&tm);
                // User-tunable cadence + on-screen length (clamped to sane bounds).
                let refresh = Duration::from_millis(settings.subtitle_refresh_ms.max(100) as u64);
                let max_chars = settings.subtitle_max_chars.max(20) as usize;

                tauri::async_runtime::spawn(async move {
                    debug!("Starting subtitle streaming task");
                    let mut last_peek_time = Instant::now();

                    while rm_clone.is_recording() {
                        if last_peek_time.elapsed() >= refresh {
                            if let Some(samples) = rm_clone.peek_recording() {
                                if !samples.is_empty() {
                                    // Transcribe a slice of the audio for speed, or the whole thing
                                    // Parakeet is fast enough for the whole thing usually
                                    if let Ok(partial_text) = tm_clone.transcribe(samples) {
                                        if !partial_text.is_empty() {
                                            let _ = app_handle.emit(
                                                "subtitle-update",
                                                tail_chars(&partial_text, max_chars),
                                            );
                                        }
                                    }
                                }
                            }
                            last_peek_time = Instant::now();
                        }
                        tokio::time::sleep(refresh.min(Duration::from_millis(100))).await;
                    }
                    debug!("Subtitle streaming task stopped");
                });
            }
        } else {
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            utils::hide_recording_overlay(app);
            change_tray_icon(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        change_tray_icon(app, TrayIconState::Transcribing);
        show_transcribing_overlay(app);

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id) {
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save
                    let transcription_time = Instant::now();
                    let transcription_result = tm.transcribe(samples);

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                transcription
                            );

                            if post_process {
                                show_processing_overlay(&ah);
                            }
                            let processed =
                                process_transcription_output(&ah, &transcription, post_process)
                                    .await;

                            // Emit the processed text as a subtitle update for the overlay
                            // (trimmed to the configured on-screen length). Only when the
                            // subtitle overlay is enabled — the overlay window is visible
                            // during transcribing, and an unconditional emit flashes the
                            // full recognized text for a moment right before the paste.
                            if get_settings(&ah).subtitle_overlay {
                                let sub_max =
                                    get_settings(&ah).subtitle_max_chars.max(20) as usize;
                                let _ = ah.emit(
                                    "subtitle-update",
                                    tail_chars(&processed.final_text, sub_max),
                                );
                            }

                            // Save to history if WAV was saved
                            if wav_saved {
                                // Coach v0 + progress: delivery-report metrics.
                                let coach_duration_ms = (sample_count as u64) * 1000
                                    / crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE as u64;
                                let coach_metrics =
                                    crate::coach::analyze(&transcription, coach_duration_ms);
                                // Durable progress row (skip empty dictations).
                                if coach_metrics.word_count > 0 {
                                    let weak_total: u32 =
                                        coach_metrics.weak_words.iter().map(|w| w.count).sum();
                                    if let Err(e) = hm.insert_coach_session(
                                        chrono::Utc::now().timestamp(),
                                        coach_metrics.word_count,
                                        coach_duration_ms,
                                        coach_metrics.wpm,
                                        coach_metrics.filler_total,
                                        weak_total,
                                    ) {
                                        error!("Failed to insert coach session: {}", e);
                                    }
                                }
                                let coach_metrics_json = serde_json::to_string(&coach_metrics).ok();
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription,
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                    Some(coach_duration_ms as i64),
                                    coach_metrics_json,
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            if processed.final_text.is_empty() {
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let final_text = processed.final_text;
                                let force_submit = processed.force_submit;
                                ah.run_on_main_thread(move || {
                                    match utils::paste(final_text, ah_clone.clone(), force_submit) {
                                        Ok(()) => debug!(
                                            "Text pasted successfully in {:?}",
                                            paste_time.elapsed()
                                        ),
                                        Err(e) => {
                                            error!("Failed to paste transcription: {}", e);
                                            let _ = ah_clone.emit("paste-error", ());
                                        }
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    change_tray_icon(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    change_tray_icon(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            debug!("Global Shortcut Transcription error: {}", err);
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                    None, // TODO(B3): pass real duration_ms + coach metrics
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            utils::hide_recording_overlay(&ah);
                            change_tray_icon(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                utils::hide_recording_overlay(&ah);
                change_tray_icon(&ah, TrayIconState::Idle);
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod translate_tests {
    use super::*;
    use crate::translate::{Lang, Translator};

    struct OkTranslator;
    impl Translator for OkTranslator {
        fn translate(&self, _text: &str, _target: Lang) -> anyhow::Result<String> {
            Ok("ПЕРЕВОД".to_string())
        }
    }
    struct ErrTranslator;
    impl Translator for ErrTranslator {
        fn translate(&self, _text: &str, _target: Lang) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("server down"))
        }
    }

    #[test]
    fn translates_when_enabled() {
        assert_eq!(
            maybe_translate("hello", true, Lang::Russian, &OkTranslator),
            "ПЕРЕВОД"
        );
    }

    #[test]
    fn passthrough_when_disabled() {
        assert_eq!(
            maybe_translate("hello", false, Lang::Russian, &OkTranslator),
            "hello"
        );
    }

    #[test]
    fn graceful_on_translator_error_returns_original() {
        assert_eq!(
            maybe_translate("hello", true, Lang::Russian, &ErrTranslator),
            "hello"
        );
    }

    #[test]
    fn empty_text_is_passthrough() {
        assert_eq!(
            maybe_translate("   ", true, Lang::Russian, &OkTranslator),
            "   "
        );
    }
}
