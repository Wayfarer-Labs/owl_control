use std::{
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
    fs::OpenOptions,
    io,
};

use color_eyre::Result;
use game_process::{Pid, windows::Win32::Foundation::HWND};
use serde::Serialize;
use tokio_util::task::AbortOnDropHandle;

#[cfg(feature = "real-video")]
use video_audio_recorder::{WindowRecorder, PerformanceMetrics, MetricsEvent, MetricsCollector};
#[cfg(feature = "real-video")]
use tokio::sync::mpsc;

use crate::{hardware_id, input_recorder::InputRecorder};

pub(crate) struct Recording {
    #[cfg(feature = "real-video")]
    window_recorder: WindowRecorder,
    #[cfg(feature = "real-video")]
    window_recorder_listener: AbortOnDropHandle<Result<()>>,
    #[cfg(feature = "real-video")]
    metrics_collector: MetricsCollector,
    #[cfg(feature = "real-video")]
    metrics_rx: mpsc::UnboundedReceiver<MetricsEvent>,
    input_recorder: InputRecorder,

    metadata_path: PathBuf,
    game_exe: String,
    start_time: SystemTime,
    start_instant: Instant,

    pid: Pid,
    hwnd: HWND,
}

pub(crate) struct MetadataParameters {
    pub(crate) path: PathBuf,
    pub(crate) game_exe: String,
}

pub(crate) struct WindowParameters {
    pub(crate) path: PathBuf,
    pub(crate) pid: Pid,
    pub(crate) hwnd: HWND,
}

pub(crate) struct InputParameters {
    pub(crate) path: PathBuf,
}

impl Recording {
    pub(crate) async fn start(
        MetadataParameters {
            path: metadata_path,
            game_exe,
        }: MetadataParameters,
        #[cfg_attr(not(feature = "real-video"), expect(unused_variables))] WindowParameters {
            path: video_path,
            pid,
            hwnd,
        }: WindowParameters,
        InputParameters { path: csv_path }: InputParameters,
        debug_level: Option<String>,
    ) -> Result<Self> {
        let start_time = SystemTime::now();
        let start_instant = Instant::now();

        #[cfg(feature = "real-video")]
        let debug_log_file = if debug_level.is_some() {
            let debug_log_path = video_path.with_extension("debug.log");
            redirect_stdout_stderr_to_file(&debug_log_path)?;
            
            // Also set up GStreamer to log to the same file
            video_audio_recorder::set_gst_log_file(&debug_log_path)?;
            
            Some(debug_log_path)
        } else {
            None
        };
        let metrics_collector = MetricsCollector::new(debug_level)?;
        #[cfg(feature = "real-video")]
        let (window_recorder, metrics_rx) =
            WindowRecorder::start_recording(&video_path, pid.0, hwnd.0.expose_provenance())?;
        #[cfg(feature = "real-video")]
        let window_recorder_listener =
            AbortOnDropHandle::new(tokio::task::spawn(window_recorder.listen_to_messages()));

        let input_recorder = InputRecorder::start(&csv_path).await?;

        Ok(Self {
            #[cfg(feature = "real-video")]
            window_recorder,
            #[cfg(feature = "real-video")]
            window_recorder_listener,
            #[cfg(feature = "real-video")]
            metrics_collector,
            #[cfg(feature = "real-video")]
            metrics_rx,

            input_recorder,

            metadata_path,
            game_exe,
            start_time,
            start_instant,

            pid,
            hwnd,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn game_exe(&self) -> &str {
        &self.game_exe
    }

    #[allow(dead_code)]
    pub(crate) fn start_time(&self) -> SystemTime {
        self.start_time
    }

    #[allow(dead_code)]
    pub(crate) fn start_instant(&self) -> Instant {
        self.start_instant
    }

    #[allow(dead_code)]
    pub(crate) fn elapsed(&self) -> std::time::Duration {
        self.start_instant.elapsed()
    }

    #[allow(dead_code)]
    pub(crate) fn pid(&self) -> Pid {
        self.pid
    }

    #[allow(dead_code)]
    pub(crate) fn hwnd(&self) -> HWND {
        self.hwnd
    }

    pub(crate) async fn seen_input(&mut self, e: raw_input::Event) -> Result<()> {
        self.input_recorder.seen_input(e).await
    }

    pub(crate) fn sample_system_resources(&mut self) -> Result<()> {
        #[cfg(feature = "real-video")]
        {
            self.metrics_collector.sample_system_resources()?;
        }
        Ok(())
    }

    pub(crate) fn handle_metrics_event(&mut self, event: MetricsEvent) {
        #[cfg(feature = "real-video")]
        {
            match event {
                MetricsEvent::FrameDrop => self.metrics_collector.record_frame_drop(),
                MetricsEvent::EncodingError => self.metrics_collector.record_encoding_error(),
                MetricsEvent::PipelineStateChange => self.metrics_collector.record_pipeline_state_change(),
            }
        }
    }

    pub(crate) fn try_recv_metrics_event(&mut self) -> Option<MetricsEvent> {
        #[cfg(feature = "real-video")]
        {
            self.metrics_rx.try_recv().ok()
        }
        #[cfg(not(feature = "real-video"))]
        {
            None
        }
    }

    pub(crate) async fn stop(self) -> Result<()> {
        tracing::debug!("Recording::stop() called");
        #[cfg(feature = "real-video")]
        {
            tracing::debug!("Dropping metrics receiver");
            // Close the metrics receiver to signal we're done collecting metrics
            drop(self.metrics_rx);

            tracing::debug!("Calling window_recorder.stop_recording()");
            self.window_recorder.stop_recording();
            tracing::debug!("Awaiting window_recorder_listener");
            self.window_recorder_listener.await.unwrap()?;
            tracing::debug!("window_recorder_listener completed");

            // Try to save performance metrics, but don't fail the recording if this fails
            let file_size = self.window_recorder.get_file_size();
            let performance_metrics = self.metrics_collector.finalize_metrics(file_size);
            let metrics_path = self.metadata_path.parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("performance_metrics.json");
            if let Err(e) = Self::write_performance_metrics(&metrics_path, &performance_metrics).await {
                tracing::warn!("Failed to save performance metrics: {}", e);
            }
        }

        self.input_recorder.stop().await?;

        Self::write_metadata(
            &self.metadata_path,
            self.game_exe,
            self.start_instant,
            self.start_time,
        )
        .await?;
        Ok(())
    }

    async fn write_metadata(
        path: &Path,
        game_exe: String,
        start_instant: Instant,
        start_time: SystemTime,
    ) -> Result<()> {
        let metadata = Self::final_metadata(game_exe, start_instant, start_time).await?;
        let metadata = serde_json::to_string_pretty(&metadata)?;
        tokio::fs::write(path, &metadata).await?;
        Ok(())
    }

    #[cfg(feature = "real-video")]
    async fn write_performance_metrics(
        path: &Path,
        metrics: &PerformanceMetrics,
    ) -> Result<()> {
        let metrics_json = serde_json::to_string_pretty(metrics)?;
        tokio::fs::write(path, &metrics_json).await?;
        Ok(())
    }

    async fn final_metadata(
        game_exe: String,
        start_instant: Instant,
        start_time: SystemTime,
    ) -> Result<Metadata> {
        let duration = start_instant.elapsed().as_secs_f32();

        let start_timestamp = start_time.duration_since(UNIX_EPOCH).unwrap().as_secs();
        let end_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let hardware_id = hardware_id::get()?;

        Ok(Metadata {
            game_exe,
            session_id: uuid::Uuid::new_v4().to_string(),
            hardware_id,
            start_timestamp,
            end_timestamp,
            duration,
        })
    }
}

#[derive(Serialize)]
struct Metadata {
    game_exe: String,
    session_id: String,
    hardware_id: String,
    start_timestamp: u64,
    end_timestamp: u64,
    duration: f32,
}

#[cfg(target_os = "windows")]
fn redirect_stdout_stderr_to_file(log_path: &Path) -> Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::System::Console::{SetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE};
    use windows::Win32::Foundation::HANDLE;
    
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    
    let handle = HANDLE(file.as_raw_handle() as *mut std::ffi::c_void);
    
    unsafe {
        SetStdHandle(STD_OUTPUT_HANDLE, handle)?;
        SetStdHandle(STD_ERROR_HANDLE, handle)?;
    }
    
    // Don't close the file handle here - it needs to stay open for stdout/stderr
    std::mem::forget(file);
    
    tracing::info!("Redirected stdout/stderr to {}", log_path.display());
    
    Ok(())
}

#[cfg(target_os = "linux")]
fn redirect_stdout_stderr_to_file(log_path: &Path) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    
    let fd = file.as_raw_fd();
    
    unsafe {
        libc::dup2(fd, libc::STDOUT_FILENO);
        libc::dup2(fd, libc::STDERR_FILENO);
    }
    
    // Don't close the file handle here - it needs to stay open for stdout/stderr
    std::mem::forget(file);
    
    tracing::info!("Redirected stdout/stderr to {}", log_path.display());
    
    Ok(())
}
