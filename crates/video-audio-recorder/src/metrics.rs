use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::fs::OpenOptions;
use std::io::{Write, BufWriter};
use std::sync::{Arc, Mutex};
use std::path::Path;
use serde::{Deserialize, Serialize};
use gstreamer::glib;
use chrono;

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
    object: Option<&gstreamer::LoggedObject>,
    message: &gstreamer::DebugMessage,
) {
    if let Some(writer_arc) = GST_LOG_WRITER.get() {
        if let Ok(mut writer_guard) = writer_arc.lock() {
            if let Some(ref mut writer) = *writer_guard {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
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
    pub average_cpu_usage: f64,
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
    pub cpu_cores: u32,
    pub cpu_threads: u32,
    pub total_memory_mb: u64,
    pub gpu_name: Option<String>,
    pub gpu_memory_mb: Option<u64>,
    pub gpu_driver_version: Option<String>,
    pub gstreamer_version: String,
}

#[derive(Debug)]
pub struct MetricsCollector {
    start_time: Instant,
    frame_drops: u64,
    encoding_errors: u64,
    cpu_samples: Vec<f64>,
    memory_samples: Vec<u64>,
    peak_memory: u64,
    system_info: SystemInfo,
    debug_level: Option<String>,
}

impl MetricsCollector {
    pub fn new(debug_level: Option<String>) -> color_eyre::Result<Self> {
        let system_info = SystemInfo::collect()?;
        
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

        let mut collector = Self {
            start_time: Instant::now(),
            frame_drops: 0,
            encoding_errors: 0,
            cpu_samples: Vec::new(),
            memory_samples: Vec::new(),
            peak_memory: 0,
            system_info,
            debug_level,
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
        let cpu_usage = self.get_cpu_usage()?;
        let memory_usage = self.get_memory_usage()?;
        
        self.cpu_samples.push(cpu_usage);
        self.memory_samples.push(memory_usage);
        
        if memory_usage > self.peak_memory {
            self.peak_memory = memory_usage;
        }
        
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
            average_cpu_usage: self.cpu_samples.iter().sum::<f64>() / self.cpu_samples.len().max(1) as f64,
            peak_memory_usage: self.peak_memory,
            average_memory_usage: self.memory_samples.iter().sum::<u64>() as f64 / self.memory_samples.len().max(1) as f64,
            file_size_bytes,
            bitrate_kbps,
            system_info: self.system_info.clone(),
        }
    }

    fn get_cpu_usage(&self) -> color_eyre::Result<f64> {
        use std::process::Command;
        
        let output = Command::new("wmic")
            .args(&["cpu", "get", "loadpercentage", "/value"])
            .output()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.starts_with("LoadPercentage=") {
                let percentage = line.split('=').nth(1).unwrap_or("0").trim();
                return Ok(percentage.parse::<f64>().unwrap_or(0.0));
            }
        }
        
        Ok(0.0)
    }

    fn get_memory_usage(&self) -> color_eyre::Result<u64> {
        use std::process::Command;
        
        let output = Command::new("wmic")
            .args(&["OS", "get", "TotalVirtualMemorySize,FreeVirtualMemorySize", "/value"])
            .output()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        let mut total = 0u64;
        let mut free = 0u64;
        
        for line in output_str.lines() {
            if line.starts_with("TotalVirtualMemorySize=") {
                total = line.split('=').nth(1).unwrap_or("0").trim().parse().unwrap_or(0);
            } else if line.starts_with("FreeVirtualMemorySize=") {
                free = line.split('=').nth(1).unwrap_or("0").trim().parse().unwrap_or(0);
            }
        }
        
        Ok((total - free) * 1024) // Convert from KB to bytes
    }
}

impl SystemInfo {
    pub fn collect() -> color_eyre::Result<Self> {
        let os = std::env::consts::OS.to_string();
        let os_version = Self::get_os_version();
        let cpu_model = Self::get_cpu_model();
        let cpu_cores = num_cpus::get() as u32;
        let cpu_threads = num_cpus::get() as u32; // Note: This gets logical CPUs, not physical threads
        let total_memory_mb = Self::get_total_memory_mb()?;
        let (gpu_name, gpu_memory_mb, gpu_driver_version) = Self::get_gpu_info();
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

    fn get_os_version() -> String {
        use std::process::Command;
        
        let output = Command::new("wmic")
            .args(&["os", "get", "Caption,Version", "/value"])
            .output()
            .unwrap_or_else(|_| std::process::Output { 
                status: std::process::ExitStatus::default(), 
                stdout: Vec::new(), 
                stderr: Vec::new() 
            });
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        let mut caption = String::new();
        let mut version = String::new();
        
        for line in output_str.lines() {
            if line.starts_with("Caption=") {
                caption = line.split('=').nth(1).unwrap_or("").trim().to_string();
            } else if line.starts_with("Version=") {
                version = line.split('=').nth(1).unwrap_or("").trim().to_string();
            }
        }
        
        if !caption.is_empty() && !version.is_empty() {
            format!("{} ({})", caption, version)
        } else {
            "Windows (unknown version)".to_string()
        }
    }

    fn get_cpu_model() -> String {
        use std::process::Command;
        
        let output = Command::new("wmic")
            .args(&["cpu", "get", "name", "/value"])
            .output()
            .unwrap_or_else(|_| std::process::Output { 
                status: std::process::ExitStatus::default(), 
                stdout: Vec::new(), 
                stderr: Vec::new() 
            });
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.starts_with("Name=") {
                let name = line.split('=').nth(1).unwrap_or("").trim();
                if !name.is_empty() {
                    return name.to_string();
                }
            }
        }
        
        "Unknown CPU".to_string()
    }

    fn get_total_memory_mb() -> color_eyre::Result<u64> {
        use std::process::Command;
        
        let output = Command::new("wmic")
            .args(&["computersystem", "get", "TotalPhysicalMemory", "/value"])
            .output()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.starts_with("TotalPhysicalMemory=") {
                let bytes = line.split('=').nth(1).unwrap_or("0").trim().parse::<u64>().unwrap_or(0);
                return Ok(bytes / (1024 * 1024));
            }
        }
        
        Ok(0)
    }

    fn get_gpu_info() -> (Option<String>, Option<u64>, Option<String>) {
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
        
        // Try to get more accurate VRAM info using an alternative method
        if let Some(accurate_vram) = Self::get_gpu_memory_alternative() {
            memory_mb = Some(accurate_vram);
        }
        
        (name, memory_mb, driver_version)
    }

    fn get_gpu_memory_alternative() -> Option<u64> {
        use std::process::Command;
        
        // First try PowerShell method for more accurate results
        if let Some(memory) = Self::get_gpu_memory_powershell() {
            return Some(memory);
        }
        
        // Fallback to wmic with different query
        let output = Command::new("wmic")
            .args(&["path", "win32_VideoController", "get", "AdapterRAM,VideoMemoryType", "/value"])
            .output()
            .ok()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        let mut max_memory = 0u64;
        
        // Look for the largest AdapterRAM value (likely the dedicated GPU)
        for line in output_str.lines() {
            if line.starts_with("AdapterRAM=") {
                if let Ok(ram_bytes) = line.split('=').nth(1).unwrap_or("0").trim().parse::<u64>() {
                    let ram_mb = ram_bytes / (1024 * 1024);
                    if ram_mb > max_memory {
                        max_memory = ram_mb;
                    }
                }
            }
        }
        
        if max_memory > 0 {
            Some(max_memory)
        } else {
            None
        }
    }

    fn get_gpu_memory_powershell() -> Option<u64> {
        use std::process::Command;
        
        // Use PowerShell to get GPU memory via CIM (more reliable than wmic)
        let script = r#"
        Get-CimInstance -ClassName Win32_VideoController | 
        Where-Object { $_.AdapterRAM -gt 0 } | 
        ForEach-Object { 
            "$($_.Name):$($_.AdapterRAM)" 
        }
        "#;
        
        let output = Command::new("powershell")
            .args(&["-Command", script])
            .output()
            .ok()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        let mut max_memory = 0u64;
        
        for line in output_str.lines() {
            if let Some((name, ram_str)) = line.split_once(':') {
                // Skip integrated graphics (Intel, AMD APU, etc.)
                if name.to_lowercase().contains("intel") && 
                   (name.to_lowercase().contains("hd") || name.to_lowercase().contains("uhd") || name.to_lowercase().contains("iris")) {
                    continue;
                }
                
                if let Ok(ram_bytes) = ram_str.trim().parse::<u64>() {
                    let ram_mb = ram_bytes / (1024 * 1024);
                    if ram_mb > max_memory {
                        max_memory = ram_mb;
                    }
                }
            }
        }
        
        if max_memory > 0 {
            Some(max_memory)
        } else {
            None
        }
    }
}
