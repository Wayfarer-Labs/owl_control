use std::time::{Duration, Instant, SystemTime};
use std::fs::OpenOptions;
use std::io::{Write, BufWriter};
use std::sync::{Arc, Mutex};
use std::path::Path;
use serde::{Deserialize, Serialize};
use gstreamer::glib;
use chrono;

use sysinfo::{
    System
};

// Debug level is now handled as a GST_DEBUG format string
// Examples: "*:3", "audiotestsrc:6,*:2", "audio*:5"

// Global log writer for GStreamer logs
static GST_LOG_WRITER: std::sync::OnceLock<Arc<Mutex<Option<BufWriter<std::fs::File>>>>> = std::sync::OnceLock::new();

// Initialize the global log writer
fn init_gst_log_writer() {
    GST_LOG_WRITER.set(Arc::new(Mutex::new(None))).ok();
}

// Set the log file for GStreamer output
pub fn set_gst_log_file(log_path: &Path) -> color_eyre::Result<()> {
    let writer = GST_LOG_WRITER.get_or_init(|| Arc::new(Mutex::new(None)));

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    let mut writer_guard = writer.lock().unwrap();
    *writer_guard = Some(BufWriter::new(file));

    Ok(())
}

// Custom GStreamer log function
fn gst_log_function(
    category: gstreamer::DebugCategory,
    level: gstreamer::DebugLevel,
    file: &glib::GStr,
    function: &glib::GStr,
    line: u32,
    _object: Option<&gstreamer::LoggedObject>,
    message: &gstreamer::DebugMessage,
) {
    if let Some(writer_arc) = GST_LOG_WRITER.get() {
        if let Ok(mut writer_guard) = writer_arc.lock() {
            if let Some(ref mut writer) = *writer_guard {
                let timestamp_str = chrono::DateTime::<chrono::Utc>::from(SystemTime::now())
                    .format("%Y-%m-%dT%H:%M:%S%.6fZ")
                    .to_string();

                let file_str = file.as_str();
                let func_str = function.as_str();
                let msg_str = message.get().map(|s| s.as_str().to_string()).unwrap_or_default();

                let log_line = format!(
                    "{} {:?} {}: {} ({}:{}:{})\n",
                    timestamp_str, level, category.name(), msg_str, file_str, func_str, line
                );

                let _ = writer.write_all(log_line.as_bytes());
                let _ = writer.flush();
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub recording_duration: Duration,
    pub frame_drops: u64,
    pub encoding_errors: u64,
    pub average_cpu_usage: f32,
    pub peak_memory_usage: u64,
    pub average_memory_usage: f64,
    pub file_size_bytes: u64,
    pub bitrate_kbps: f64,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub os_version: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub total_memory_mb: u64,
    pub gpu_name: Option<String>,
    pub gpu_memory_mb: Option<u64>,
    pub gpu_driver_version: Option<String>,
    pub gstreamer_version: String,
}

pub struct MetricsCollector {
    start_time: Instant,
    frame_drops: u64,
    encoding_errors: u64,
    cpu_samples: Vec<f32>,
    memory_samples: Vec<u64>,
    peak_memory: u64,
    system: Arc<System>,
    system_info: SystemInfo,
}

impl MetricsCollector {
    pub fn new(debug_level: Option<String>) -> color_eyre::Result<Self> {
        let mut system = System::new_all();
        let system_info = SystemInfo::collect(&system)?;

        // Initialize global log writer if not already done
        init_gst_log_writer();

        // Configure GStreamer debug output
        if let Some(ref debug_str) = debug_level {
            if !debug_str.is_empty() && debug_str != "0" && debug_str != "*:0" {
                gstreamer::log::set_threshold_from_string(debug_str, true);

                // Set up custom log function to capture GStreamer logs
                gstreamer::log::add_log_function(gst_log_function);
            }
        }

        let collector = Self {
            start_time: Instant::now(),
            frame_drops: 0,
            encoding_errors: 0,
            cpu_samples: Vec::new(),
            memory_samples: Vec::new(),
            peak_memory: 0,
            system: Arc::new(system),
            system_info,
        };

        Ok(collector)
    }


    pub fn record_frame_drop(&mut self) {
        self.frame_drops += 1;
    }

    pub fn record_encoding_error(&mut self) {
        self.encoding_errors += 1;
    }

    pub fn sample_system_resources(&mut self) -> color_eyre::Result<()> {
        let cpu_usage = self.system.global_cpu_usage();
        let memory_usage = self.system.used_memory() / 1024;

        self.cpu_samples.push(cpu_usage);
        self.memory_samples.push(memory_usage);

        if memory_usage > self.peak_memory {
            self.peak_memory = memory_usage;
        }
        tracing::debug!("sampled system cpu: {} memory: {}", cpu_usage, memory_usage);

        Ok(())
    }

    pub fn finalize_metrics(&self, file_size_bytes: u64) -> PerformanceMetrics {
        let recording_duration = self.start_time.elapsed();
        let bitrate_kbps = if recording_duration.as_secs() > 0 {
            (file_size_bytes * 8) as f64 / (recording_duration.as_secs() as f64 * 1000.0)
        } else {
            0.0
        };

        PerformanceMetrics {
            recording_duration,
            frame_drops: self.frame_drops,
            encoding_errors: self.encoding_errors,
            average_cpu_usage: self.cpu_samples.iter().sum::<f32>() / self.cpu_samples.len().max(1) as f32,
            peak_memory_usage: self.peak_memory,
            average_memory_usage: self.memory_samples.iter().sum::<u64>() as f64 / self.memory_samples.len().max(1) as f64,
            file_size_bytes,
            bitrate_kbps,
            system_info: self.system_info.clone(),
        }
    }
}

impl SystemInfo {
    pub fn collect(sys: &System) -> color_eyre::Result<Self> {
        let os = System::distribution_id();
        let os_version = format!("{} ({})", System::long_os_version().ok_or("Unknown").unwrap(), System::kernel_version().unwrap());
        let cpus = sys.cpus();
        let cpu_model = cpus[0].brand().to_string();
        let cpu_cores = System::physical_core_count().unwrap();
        let cpu_threads = cpus.len();
        let total_memory_mb = sys.total_memory()/1024/1024;
        let (gpu_name, gpu_memory_mb, gpu_driver_version) = Self::get_gpu_info(sys);
        let gstreamer_version = gstreamer::version_string().to_string();

        Ok(Self {
            os,
            os_version,
            cpu_model,
            cpu_cores,
            cpu_threads,
            total_memory_mb,
            gpu_name,
            gpu_memory_mb,
            gpu_driver_version,
            gstreamer_version
        })
    }
    fn get_gpu_info(sys: &System) -> (Option<String>, Option<u64>, Option<String>) {
        use std::process::Command;

        let output = Command::new("wmic")
            .args(&["path", "win32_VideoController", "get", "name,AdapterRAM,DriverVersion", "/value"])
            .output()
            .unwrap_or_else(|_| std::process::Output {
                status: std::process::ExitStatus::default(),
                stdout: Vec::new(),
                stderr: Vec::new()
            });

        let output_str = String::from_utf8_lossy(&output.stdout);
        let mut name = None;
        let mut memory_mb = None;
        let mut driver_version = None;

        for line in output_str.lines() {
            if line.starts_with("Name=") {
                let gpu_name = line.split('=').nth(1).unwrap_or("").trim();
                if !gpu_name.is_empty() && name.is_none() {
                    name = Some(gpu_name.to_string());
                }
            } else if line.starts_with("AdapterRAM=") {
                let ram_bytes = line.split('=').nth(1).unwrap_or("0").trim().parse::<u64>().unwrap_or(0);
                if ram_bytes > 0 && memory_mb.is_none() {
                    memory_mb = Some(ram_bytes / (1024 * 1024));
                }
            } else if line.starts_with("DriverVersion=") {
                let version = line.split('=').nth(1).unwrap_or("").trim();
                if !version.is_empty() && driver_version.is_none() {
                    driver_version = Some(version.to_string());
                }
            }
        }

        (name, memory_mb, driver_version)
    }
}
