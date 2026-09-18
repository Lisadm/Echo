//! Bundled Qwen3-ASR runtime component: an embedded Python 3.12 + PyTorch
//! (CUDA 12.4) + `qwen-asr` environment installed into `<app data>/qwen-runtime`.
//!
//! The pinned wheel manifest is carried over verbatim from OpenWhisper's
//! `services/local_asr/qwen_runtime.json` (MIT, Copyright (c) 2025 Knuckles92);
//! the install mechanics (resume + SHA256 + staged flat extraction + atomic
//! swap) mirror both OpenWhisper's component installer and Echo's own
//! `ModelManager::download_model`.

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use specta::Type;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

const MANIFEST_JSON: &str = include_str!("qwen_runtime.json");
const THROTTLE: Duration = Duration::from_millis(100);

#[derive(Deserialize)]
struct RuntimeManifest {
    version: String,
    #[allow(dead_code)]
    install_bytes: u64,
    archives: Vec<RuntimeArchive>,
}

#[derive(Clone, Deserialize)]
struct RuntimeArchive {
    name: String,
    url: String,
    sha256: String,
    size_bytes: u64,
    extract: String,
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct QwenRuntimeStatus {
    pub installed: bool,
    pub version: Option<String>,
    pub installing: bool,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Serialize, Clone, Debug)]
struct RuntimeProgressEvent {
    downloaded: u64,
    total: u64,
    percentage: f64,
}

#[derive(Serialize, Clone, Debug)]
struct RuntimeStatusEvent {
    state: String,
    error: Option<String>,
}

/// Path of the bundled Python interpreter. Windows-only for now — the
/// pinned manifest is win_amd64.
pub fn python_executable(app: &AppHandle) -> Result<PathBuf> {
    #[cfg(not(windows))]
    {
        let _ = app;
        Err(anyhow!(
            "The bundled Qwen runtime is currently Windows-only"
        ))
    }
    #[cfg(windows)]
    {
        let base = crate::portable::app_data_dir(app)
            .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
        let exe = base.join("qwen-runtime").join("python.exe");
        if !exe.is_file() {
            return Err(anyhow!(
                "Qwen runtime is not installed. Download it from the model settings."
            ));
        }
        Ok(exe)
    }
}

pub struct QwenRuntimeManager {
    app_handle: AppHandle,
    base_dir: PathBuf,
    installing: Arc<AtomicBool>,
    cancel_flag: Arc<AtomicBool>,
    progress: Mutex<(u64, u64)>,
}

impl QwenRuntimeManager {
    pub fn new(app_handle: &AppHandle) -> Result<Self> {
        let base_dir = crate::portable::app_data_dir(app_handle)
            .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
        Ok(Self {
            app_handle: app_handle.clone(),
            base_dir,
            installing: Arc::new(AtomicBool::new(false)),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            progress: Mutex::new((0, 0)),
        })
    }

    fn manifest(&self) -> Result<RuntimeManifest> {
        serde_json::from_str(MANIFEST_JSON).context("Failed to parse qwen_runtime.json")
    }

    fn runtime_dir(&self) -> PathBuf {
        self.base_dir.join("qwen-runtime")
    }

    fn staging_dir(&self) -> PathBuf {
        self.base_dir.join("qwen-runtime.staging")
    }

    fn cache_dir(&self) -> PathBuf {
        self.base_dir.join("qwen-runtime.cache")
    }

    fn sentinel(&self) -> PathBuf {
        self.runtime_dir().join(".installed")
    }

    pub fn is_installed(&self) -> bool {
        self.sentinel().is_file()
    }

    pub fn status(&self) -> QwenRuntimeStatus {
        let (downloaded, total) = *self.progress.lock().unwrap();
        QwenRuntimeStatus {
            installed: self.is_installed(),
            version: self.installed_version().ok(),
            installing: self.installing.load(Ordering::SeqCst),
            downloaded_bytes: downloaded,
            total_bytes: total,
        }
    }

    fn installed_version(&self) -> Result<String> {
        Ok(std::fs::read_to_string(self.sentinel())?.trim().to_string())
    }

    pub fn cancel_download(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// Download and install the runtime if it isn't already present.
    /// `on_progress` receives cumulative (downloaded, total) archive bytes.
    pub async fn install_with_progress<F>(&self, mut on_progress: F) -> Result<()>
    where
        F: FnMut(u64, u64) + Send,
    {
        if self.installing.swap(true, Ordering::SeqCst) {
            return Err(anyhow!("Qwen runtime install is already in progress"));
        }
        let result = self.install_inner(&mut on_progress).await;
        self.installing.store(false, Ordering::SeqCst);
        match &result {
            Ok(()) => {
                *self.progress.lock().unwrap() = (0, 0);
                self.emit_status("installed", None);
            }
            Err(e) => {
                if self.cancel_flag.load(Ordering::SeqCst) {
                    self.emit_status("cancelled", Some("Download cancelled".into()));
                    return Err(anyhow!("Download cancelled"));
                }
                warn!("Qwen runtime install failed: {}", e);
                self.emit_status("failed", Some(e.to_string()));
            }
        }
        result
    }

    pub async fn install(&self) -> Result<()> {
        self.install_with_progress(|_, _| {}).await
    }

    async fn install_inner<F>(&self, on_progress: &mut F) -> Result<()>
    where
        F: FnMut(u64, u64) + Send,
    {
        let manifest = self.manifest()?;
        if self.is_installed() {
            info!(
                "Qwen runtime already installed ({}), skipping",
                manifest.version
            );
            return Ok(());
        }

        self.cancel_flag.store(false, Ordering::SeqCst);
        self.emit_status("installing", None);

        let staging = self.staging_dir();
        let cache = self.cache_dir();
        let total: u64 = manifest.archives.iter().map(|a| a.size_bytes).sum();

        // Clean any leftover staging from an interrupted install.
        if staging.exists() {
            std::fs::remove_dir_all(&staging)
                .with_context(|| format!("Failed to clean staging dir {}", staging.display()))?;
        }
        std::fs::create_dir_all(&staging)?;
        std::fs::create_dir_all(&cache)?;

        let client = reqwest::Client::new();
        let mut completed = 0u64;
        for archive in &manifest.archives {
            if self.cancel_flag.load(Ordering::SeqCst) {
                return Err(anyhow!("Download cancelled"));
            }
            let part = cache.join(format!("{}.part", sanitize_component(&archive.name)));
            let staging_path = staging.clone();
            let part_path = part.clone();
            let archive_clone = archive.clone();
            let cancel = self.cancel_flag.clone();
            let app = self.app_handle.clone();

            // Closure shared by the streaming loop for throttled progress.
            let mut last_emit = Instant::now();
            {
                let on_progress = &mut *on_progress;
                self.download_verified(
                    &client,
                    &archive_clone.url,
                    &part,
                    archive_clone.size_bytes,
                    &archive_clone.sha256,
                    cancel,
                    |downloaded, file_total| {
                        on_progress(completed + downloaded, total);
                        if last_emit.elapsed() >= THROTTLE {
                            last_emit = Instant::now();
                            let _ = app.emit(
                                "qwen-runtime-progress",
                                &RuntimeProgressEvent {
                                    downloaded: completed + downloaded,
                                    total,
                                    percentage: if total > 0 {
                                        ((completed + downloaded) as f64 / total as f64) * 100.0
                                    } else {
                                        0.0
                                    },
                                },
                            );
                        }
                        let _ = file_total;
                    },
                )
                .await?;
            }

            if archive_clone.extract == "zip" {
                tokio::task::spawn_blocking(move || extract_zip_flat(&part_path, &staging_path))
                    .await
                    .map_err(|e| anyhow!("Extraction task panicked: {}", e))?
                    .with_context(|| format!("Failed to extract {}", archive_clone.name))?;
            } else {
                return Err(anyhow!(
                    "Unsupported archive type in runtime manifest: {}",
                    archive_clone.name
                ));
            }
            let _ = std::fs::remove_file(&part); // keep disk usage reasonable
            completed += archive_clone.size_bytes;
        }

        // Finalize the staging directory (blocking FS work off the executor).
        let version = manifest.version.clone();
        let finalize_staging = staging.clone();
        tokio::task::spawn_blocking(move || finalize_runtime_dir(&finalize_staging, &version))
            .await
            .map_err(|e| anyhow!("Finalize task panicked: {}", e))??;

        // Atomic-ish swap with rollback, tolerating transient Windows file locks
        // (antivirus/indexer) like OpenWhisper's `_replace_speech_runtime`.
        let runtime_dir = self.runtime_dir();
        let old_dir = self.base_dir.join("qwen-runtime.old");
        if old_dir.exists() {
            let _ = std::fs::remove_dir_all(&old_dir);
        }
        if runtime_dir.exists() {
            rename_with_retry(&runtime_dir, &old_dir).await?;
        }
        if let Err(e) = std::fs::rename(&staging, &runtime_dir) {
            // Roll the old install back so the app keeps a working runtime.
            if old_dir.exists() {
                let _ = rename_with_retry(&old_dir, &runtime_dir).await;
            }
            return Err(anyhow!(
                "Failed to activate Qwen runtime (is the model still loaded? Unload it and retry): {}",
                e
            ));
        }
        let _ = std::fs::remove_dir_all(&old_dir);
        let _ = std::fs::remove_dir_all(&cache);

        std::fs::write(self.sentinel(), &manifest.version)?;
        info!("Qwen runtime {} installed successfully", manifest.version);
        Ok(())
    }

    /// Streaming download with Range resume, size check and SHA256 verify.
    /// `on_chunk(downloaded, total)` is called as bytes arrive.
    #[allow(clippy::too_many_arguments)]
    async fn download_verified<F>(
        &self,
        client: &reqwest::Client,
        url: &str,
        part_path: &Path,
        expected_size: u64,
        expected_sha256: &str,
        cancel_flag: Arc<AtomicBool>,
        mut on_chunk: F,
    ) -> Result<()>
    where
        F: FnMut(u64, u64) + Send,
    {
        let mut resume_from = if part_path.exists() {
            part_path.metadata()?.len()
        } else {
            0
        };

        let mut request = client.get(url);
        if resume_from > 0 {
            request = request.header("Range", format!("bytes={}-", resume_from));
        }
        let mut response = request.send().await?;

        // Server ignored the Range request — restart from scratch.
        if resume_from > 0 && response.status() == reqwest::StatusCode::OK {
            let _ = std::fs::remove_file(part_path);
            resume_from = 0;
            response = client.get(url).send().await?;
        }
        if !response.status().is_success()
            && response.status() != reqwest::StatusCode::PARTIAL_CONTENT
        {
            return Err(anyhow!(
                "Failed to download {}: HTTP {}",
                url,
                response.status()
            ));
        }

        let content_length = response.content_length().unwrap_or(0);
        let total = if resume_from > 0 {
            resume_from + content_length
        } else {
            content_length
        };
        if expected_size > 0 && total > 0 && total != expected_size {
            return Err(anyhow!(
                "Download size mismatch for {}: expected {} bytes, server reports {}",
                url,
                expected_size,
                total
            ));
        }

        let mut file = if resume_from > 0 {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(part_path)?
        } else {
            std::fs::File::create(part_path)?
        };

        let mut downloaded = resume_from;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if cancel_flag.load(Ordering::SeqCst) {
                return Err(anyhow!("Download cancelled"));
            }
            let chunk = chunk?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            on_chunk(downloaded, total);
        }
        file.flush()?;
        drop(file);

        let actual = part_path.metadata()?.len();
        if expected_size > 0 && actual != expected_size {
            let _ = std::fs::remove_file(part_path);
            return Err(anyhow!(
                "Download incomplete for {}: expected {} bytes, got {}",
                url,
                expected_size,
                actual
            ));
        }

        // SHA256 (blocking hash of up to 2.5 GB — keep off the async executor).
        let verify_path = part_path.to_path_buf();
        let expected = expected_sha256.to_string();
        let actual_hash = tokio::task::spawn_blocking(move || {
            let mut file = std::fs::File::open(&verify_path)?;
            let mut hasher = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buffer[..n]);
            }
            Ok::<String, std::io::Error>(format!("{:x}", hasher.finalize()))
        })
        .await
        .map_err(|e| anyhow!("SHA256 task panicked: {}", e))??;

        if actual_hash != expected.to_lowercase() {
            let _ = std::fs::remove_file(part_path);
            return Err(anyhow!(
                "SHA256 mismatch for {}: file is corrupt. Please retry.",
                url
            ));
        }
        Ok(())
    }

    fn emit_status(&self, state: &str, error: Option<String>) {
        let _ = self.app_handle.emit(
            "qwen-runtime-status-changed",
            &RuntimeStatusEvent {
                state: state.to_string(),
                error,
            },
        );
    }
}

/// Per-archive finalization: embedded-python path config, integrity check and
/// install marker — mirrors OpenWhisper's `_validate_component_payload`.
fn finalize_runtime_dir(staging: &Path, version: &str) -> Result<()> {
    let required = [
        "python.exe",
        "python312.dll",
        "python312.zip",
        "qwen_asr/__init__.py",
        "torch/__init__.py",
    ];
    for rel in required {
        if !staging.join(rel).is_file() {
            return Err(anyhow!(
                "Qwen runtime payload is incomplete: missing {}",
                rel
            ));
        }
    }
    // The embedded interpreter must only ever search this directory.
    std::fs::write(
        staging.join("python312._pth"),
        "python312.zip\n.\nimport site\n",
    )?;
    std::fs::write(staging.join(".runtime-version"), version)?;
    Ok(())
}

async fn rename_with_retry(from: &Path, to: &Path) -> Result<()> {
    let mut last_err = None;
    for _ in 0..16 {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    Err(anyhow!(
        "Failed to rename {} → {}: {:?}",
        from.display(),
        to.display(),
        last_err
    ))
}

fn sanitize_component(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Flat zip extraction with zip-slip protection. Wheels and the embedded
/// Python zip both extract directly into the component root.
fn extract_zip_flat(archive_path: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("Failed to open archive {}", archive_path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip {}", archive_path.display()))?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let Some(rel) = entry.enclosed_name() else {
            return Err(anyhow!(
                "Unsafe path in archive {}: {}",
                archive_path.display(),
                entry.name()
            ));
        };
        let out_path = dest.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = std::fs::File::create(&out_path)?;
        std::io::copy(&mut entry, &mut out)?;
    }
    Ok(())
}
