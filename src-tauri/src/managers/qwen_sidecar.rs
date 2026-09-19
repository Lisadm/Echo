//! JSONL stdin/stdout sidecar around the bundled Python + PyTorch runtime
//! running the official `qwen-asr` package (Qwen3-ASR 0.6B / 1.7B).
//!
//! The worker process and its protocol are ported from OpenWhisper's
//! `services/local_asr/` (MIT, Copyright (c) 2025 Knuckles92). One JSON
//! object per line: `load` / `transcribe` / `shutdown` requests, responses
//! matched by monotonically increasing ids, stale ids tolerated.

use anyhow::{anyhow, Context, Result};
use log::{debug, info, warn};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::AppHandle;

/// Embedded copy of the worker script; materialized into the app data dir so
/// the bundled runtime's Python can execute it without resource-path probing.
pub const WORKER_PY: &str = include_str!("qwen_worker.py");

const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const WORKER_DEAD_POLL: Duration = Duration::from_millis(200);
const STDERR_TAIL_LINES: usize = 25;

pub struct QwenSidecar {
    stdin: Mutex<Option<ChildStdin>>,
    child: Mutex<Option<Child>>,
    pending: Arc<Mutex<HashMap<u64, mpsc::SyncSender<Value>>>>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    alive: Arc<AtomicBool>,
    serial: AtomicU64,
    // Stored so a dead worker can be respawned with the same model/device.
    python_exe: PathBuf,
    worker_py: PathBuf,
    model_path: Option<PathBuf>,
    device: Option<String>,
}

impl QwenSidecar {
    /// Write the embedded worker script next to the app data and return its path.
    /// Rewritten on every launch so app updates always ship the current script.
    pub fn materialize_worker(app: &AppHandle) -> Result<PathBuf> {
        let dir = crate::portable::app_data_dir(app)
            .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?
            .join("qwen");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("qwen_worker.py");
        std::fs::write(&path, WORKER_PY)?;
        Ok(path)
    }

    pub fn spawn(python_exe: &Path, worker_py: &Path) -> Result<Self> {
        let mut cmd = Command::new(python_exe);
        cmd.arg("-u")
            .arg(worker_py)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("PYTHONUTF8", "1");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn Qwen worker at {:?}", python_exe))?;
        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let stderr = child.stderr.take().context("no stderr")?;

        let pending: Arc<Mutex<HashMap<u64, mpsc::SyncSender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let stderr_tail: Arc<Mutex<VecDeque<String>>> =
            Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));
        let alive = Arc::new(AtomicBool::new(true));

        // Protocol reader: every stdout line is a JSON response (the worker
        // redirects its real stdout to stderr, so library noise can't leak in).
        {
            let pending = pending.clone();
            let alive = alive.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let Ok(value) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    let id = value.get("id").and_then(Value::as_u64);
                    let Some(id) = id else { continue };
                    if let Some(tx) = pending.lock().unwrap().remove(&id) {
                        // Receiver already gone (stale reply) is fine — send fails silently.
                        let _ = tx.send(value);
                    }
                }
                alive.store(false, Ordering::SeqCst);
                // Wake every waiter with a disconnected channel.
                pending.lock().unwrap().clear();
            });
        }

        // stderr tail for crash diagnostics.
        {
            let stderr_tail = stderr_tail.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let mut tail = stderr_tail.lock().unwrap();
                    if tail.len() >= STDERR_TAIL_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(line);
                }
            });
        }

        info!("Qwen worker spawned (pid: {:?})", child.id());
        Ok(Self {
            stdin: Mutex::new(Some(stdin)),
            child: Mutex::new(Some(child)),
            pending,
            stderr_tail,
            alive,
            serial: AtomicU64::new(1),
            python_exe: python_exe.to_path_buf(),
            worker_py: worker_py.to_path_buf(),
            model_path: None,
            device: None,
        })
    }

    pub fn load(&mut self, model_dir: &Path, device: &str) -> Result<()> {
        self.model_path = Some(model_dir.to_path_buf());
        self.device = Some(device.to_string());
        self.request(
            "load",
            json!({
                "model_path": model_dir.to_string_lossy(),
                "device": device,
            }),
            REQUEST_TIMEOUT,
        )
        .map(|_| ())
    }

    /// Transcribe 16 kHz mono f32 samples. Writes a temp raw-PCM file, sends a
    /// `transcribe` request, returns the recognized text. Respawns the worker
    /// once if the previous process died (e.g. after a timeout kill).
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        language: &str,
        context: Option<&str>,
    ) -> Result<String> {
        if !self.is_alive() {
            warn!("Qwen worker is not running; respawning");
            self.respawn()?;
        }

        let temp_path = write_temp_audio(audio)?;
        let request = self.request(
            "transcribe",
            json!({
                "audio_path": temp_path.to_string_lossy(),
                "language": map_language(language),
                "context": context.unwrap_or(""),
            }),
            REQUEST_TIMEOUT,
        );
        let _ = std::fs::remove_file(&temp_path);
        let response = request?;

        response
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("Qwen worker returned no text field"))
    }

    fn respawn(&mut self) -> Result<()> {
        let model_path = self
            .model_path
            .clone()
            .ok_or_else(|| anyhow!("Qwen worker died before any model was loaded"))?;
        let device = self.device.clone().unwrap_or_else(|| "cpu".to_string());
        // Drop the dead process handles first.
        self.shutdown_and_kill();
        let mut fresh = Self::spawn(&self.python_exe, &self.worker_py)?;
        fresh.load(&model_path, &device)?;
        *self = fresh;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        match self.child.lock().unwrap().as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    fn request(&mut self, op: &str, payload: Value, timeout: Duration) -> Result<Value> {
        let id = self.serial.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::sync_channel(1);
        self.pending.lock().unwrap().insert(id, tx);

        {
            let mut stdin_guard = self.stdin.lock().unwrap();
            let stdin = stdin_guard
                .as_mut()
                .ok_or_else(|| anyhow!("Qwen worker stdin is closed"))?;
            let mut request = payload;
            let object = request
                .as_object_mut()
                .ok_or_else(|| anyhow!("payload must be a JSON object"))?;
            object.insert("id".into(), json!(id));
            object.insert("op".into(), json!(op));
            let line = serde_json::to_string(&object)?;
            writeln!(stdin, "{}", line).map_err(|e| {
                self.kill();
                anyhow!("Failed to write to Qwen worker: {}", e)
            })?;
            stdin.flush()?;
        }
        debug!("Qwen request #{} op={}", id, op);

        let started = Instant::now();
        loop {
            match rx.recv_timeout(WORKER_DEAD_POLL) {
                Ok(response) => {
                    if let Some(error) = response.get("error").and_then(Value::as_str) {
                        return Err(anyhow!("{}", error));
                    }
                    return response
                        .get("result")
                        .cloned()
                        .ok_or_else(|| anyhow!("Qwen worker returned no result"));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(anyhow!("Qwen worker stopped: {}", self.stderr_summary()));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if !self.is_alive() {
                        return Err(anyhow!("Qwen worker stopped: {}", self.stderr_summary()));
                    }
                    if started.elapsed() > timeout {
                        // Donor semantics: a timed-out engine is killed and must reload.
                        self.kill();
                        return Err(anyhow!(
                            "Qwen worker timed out after {}s and was stopped. Reload the engine.",
                            timeout.as_secs()
                        ));
                    }
                }
            }
        }
    }

    fn stderr_summary(&self) -> String {
        let tail = self.stderr_tail.lock().unwrap();
        tail.iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(1600)
            .collect()
    }

    /// Best-effort graceful shutdown, then hard kill (used by Drop and respawn).
    fn shutdown_and_kill(&mut self) {
        if let Some(stdin) = self.stdin.lock().unwrap().as_mut() {
            let _ = writeln!(stdin, r#"{{"id":0,"op":"shutdown"}}"#);
            let _ = stdin.flush();
        }
        self.stdin.lock().unwrap().take();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut child_guard = self.child.lock().unwrap();
        if let Some(child) = child_guard.as_mut() {
            while Instant::now() < deadline {
                if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        *child_guard = None;
        self.alive.store(false, Ordering::SeqCst);
        self.pending.lock().unwrap().clear();
    }

    fn kill(&self) {
        self.stdin.lock().unwrap().take();
        let mut child_guard = self.child.lock().unwrap();
        if let Some(child) = child_guard.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *child_guard = None;
        self.alive.store(false, Ordering::SeqCst);
        self.pending.lock().unwrap().clear();
    }
}

impl Drop for QwenSidecar {
    fn drop(&mut self) {
        debug!("Shutting down Qwen worker");
        self.shutdown_and_kill();
    }
}

fn write_temp_audio(audio: &[f32]) -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let name = format!(
        "echo-qwen-{}-{}.f32",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let path = std::env::temp_dir().join(name);
    let mut bytes = Vec::with_capacity(audio.len() * 4);
    for sample in audio {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(&path, bytes)
        .with_context(|| format!("Failed to write temp audio to {:?}", path))?;
    Ok(path)
}

/// Map Echo's ISO language codes onto the English names `qwen-asr` expects
/// ("auto" → auto-detect; the worker maps "en"/"en-US" → "English" itself).
/// Unknown codes pass through verbatim, matching the donor behavior.
fn map_language(code: &str) -> String {
    const LANGUAGE_NAMES: &[(&str, &str)] = &[
        ("zh", "Chinese"),
        ("zh-Hans", "Chinese"),
        ("zh-Hant", "Chinese"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("de", "German"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("it", "Italian"),
        ("pt", "Portuguese"),
        ("nl", "Dutch"),
        ("pl", "Polish"),
        ("sv", "Swedish"),
        ("da", "Danish"),
        ("no", "Norwegian"),
        ("nb", "Norwegian"),
        ("fi", "Finnish"),
        ("el", "Greek"),
        ("cs", "Czech"),
        ("ro", "Romanian"),
        ("hu", "Hungarian"),
        ("ar", "Arabic"),
        ("ru", "Russian"),
        ("tr", "Turkish"),
        ("hi", "Hindi"),
        ("vi", "Vietnamese"),
        ("id", "Indonesian"),
        ("th", "Thai"),
        ("ms", "Malay"),
        ("uk", "Ukrainian"),
        ("he", "Hebrew"),
    ];
    if code == "auto" {
        return "auto".to_string();
    }
    LANGUAGE_NAMES
        .iter()
        .find(|(iso, _)| *iso == code)
        .map(|(_, name)| name.to_string())
        .unwrap_or_else(|| code.to_string())
}

/// Resolve the configured device to what the worker understands.
/// `Auto` picks CUDA when an NVIDIA GPU with a working driver is present.
pub fn resolve_device(device_setting: crate::settings::QwenDeviceSetting) -> &'static str {
    use crate::settings::QwenDeviceSetting;
    match device_setting {
        QwenDeviceSetting::Cuda => "cuda",
        QwenDeviceSetting::Cpu => "cpu",
        QwenDeviceSetting::Auto => {
            if detect_cuda() {
                "cuda"
            } else {
                "cpu"
            }
        }
    }
}

fn detect_cuda() -> bool {
    let mut cmd = Command::new("nvidia-smi");
    cmd.arg("-L").stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    match cmd.output() {
        Ok(output) => output.status.success(),
        Err(e) => {
            debug!("nvidia-smi probe failed ({}); assuming no CUDA", e);
            false
        }
    }
}
