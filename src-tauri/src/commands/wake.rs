//! Tauri commands for wake-listening settings (plan steps 6, 10).

use crate::managers::wake::WakeManager;
use crate::settings::{get_settings, write_settings, InputMode};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

#[tauri::command]
#[specta::specta]
pub fn change_input_mode_setting(app: AppHandle, new_mode: InputMode) -> Result<(), String> {
    let mut s = get_settings(&app);
    s.input_mode = new_mode;
    write_settings(&app, s);

    // React immediately: always-listening arms at once; the other two modes
    // (re)start in the off state, the hotkey toggles from there.
    if let Some(wake) = app.try_state::<Arc<WakeManager>>() {
        wake.set_on(matches!(new_mode, InputMode::WakeAlwaysListening));
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_wake_phrase_setting(app: AppHandle, phrase: String) -> Result<(), String> {
    let normalized = crate::wake::normalize_phrase(&phrase);
    if normalized.is_empty() {
        return Err("Wake phrase must not be empty".to_string());
    }
    let mut s = get_settings(&app);
    s.wake_phrase = phrase;
    write_settings(&app, s);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_wake_alternative_phrases_setting(
    app: AppHandle,
    phrases: Vec<String>,
) -> Result<(), String> {
    let mut s = get_settings(&app);
    s.wake_alternative_phrases = phrases;
    write_settings(&app, s);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_wake_activation_timeout_setting(app: AppHandle, seconds: u64) -> Result<(), String> {
    let mut s = get_settings(&app);
    s.wake_activation_timeout_secs = seconds.clamp(1, 60);
    write_settings(&app, s);
    Ok(())
}
