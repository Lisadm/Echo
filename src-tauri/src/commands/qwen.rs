//! Tauri commands for the Qwen3-ASR sidecar runtime and device setting.

use crate::managers::qwen_runtime::{QwenRuntimeManager, QwenRuntimeStatus};
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{get_settings, write_settings, QwenDeviceSetting};
use std::sync::Arc;
use tauri::{AppHandle, Manager, State};

#[tauri::command]
#[specta::specta]
pub fn get_qwen_runtime_status(
    qwen_runtime: State<'_, Arc<QwenRuntimeManager>>,
) -> QwenRuntimeStatus {
    qwen_runtime.status()
}

#[tauri::command]
#[specta::specta]
pub async fn download_qwen_runtime(
    qwen_runtime: State<'_, Arc<QwenRuntimeManager>>,
) -> Result<(), String> {
    qwen_runtime
        .install()
        .await
        .map_err(|e| format!("Qwen runtime install failed: {}", e))
}

#[tauri::command]
#[specta::specta]
pub fn cancel_qwen_runtime_download(qwen_runtime: State<'_, Arc<QwenRuntimeManager>>) {
    qwen_runtime.cancel_download();
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen_device_setting(
    app: AppHandle,
    new_setting: QwenDeviceSetting,
) -> Result<(), String> {
    let mut s = get_settings(&app);
    s.qwen_device = new_setting;
    write_settings(&app, s);

    // Unload so the next load spawns the worker on the requested device
    // (the sidecar bakes the device in at process start).
    if let Some(tm) = app.try_state::<Arc<TranscriptionManager>>() {
        tm.unload_model().map_err(|e| e.to_string())?;
    }
    Ok(())
}
