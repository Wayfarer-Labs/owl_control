use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use gstreamer::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub recording_duration: Duration,
    pub frame_drops: u64,
    pub encoding_errors: u64,
    pub pipeline_state_changes: u64,
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
    pub rust_version: String,
}

#[derive(Debug)]
pub struct MetricsCollector {
    start_time: Instant,
    frame_drops: u64,
    encoding_errors: u64,
    pipeline_state_changes: u64,
    cpu_samples: Vec<f64>,
    memory_samples: Vec<u64>,
    peak_memory: u64,
    system_info: SystemInfo,
}

impl MetricsCollector {
    pub fn new() -> color_eyre::Result<Self> {
        let system_info = SystemInfo::collect()?;
        
        Ok(Self {
            start_time: Instant::now(),
            frame_drops: 0,
            encoding_errors: 0,
            pipeline_state_changes: 0,
            cpu_samples: Vec::new(),
            memory_samples: Vec::new(),
            peak_memory: 0,
            system_info,
        })
    }

    pub fn record_frame_drop(&mut self) {
        self.frame_drops += 1;
    }

    pub fn record_encoding_error(&mut self) {
        self.encoding_errors += 1;
    }

    pub fn record_pipeline_state_change(&mut self) {
        self.pipeline_state_changes += 1;
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
            pipeline_state_changes: self.pipeline_state_changes,
            average_cpu_usage: self.cpu_samples.iter().sum::<f64>() / self.cpu_samples.len().max(1) as f64,
            peak_memory_usage: self.peak_memory,
            average_memory_usage: self.memory_samples.iter().sum::<u64>() as f64 / self.memory_samples.len().max(1) as f64,
            file_size_bytes,
            bitrate_kbps,
            system_info: self.system_info.clone(),
        }
    }

    #[cfg(target_os = "windows")]
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

    #[cfg(target_os = "linux")]
    fn get_cpu_usage(&self) -> color_eyre::Result<f64> {
        use std::fs;
        
        let stat = fs::read_to_string("/proc/stat")?;
        let first_line = stat.lines().next().unwrap_or("");
        let fields: Vec<&str> = first_line.split_whitespace().collect();
        
        if fields.len() >= 8 {
            let user: u64 = fields[1].parse().unwrap_or(0);
            let nice: u64 = fields[2].parse().unwrap_or(0);
            let system: u64 = fields[3].parse().unwrap_or(0);
            let idle: u64 = fields[4].parse().unwrap_or(0);
            let iowait: u64 = fields[5].parse().unwrap_or(0);
            let irq: u64 = fields[6].parse().unwrap_or(0);
            let softirq: u64 = fields[7].parse().unwrap_or(0);
            
            let total = user + nice + system + idle + iowait + irq + softirq;
            let active = total - idle - iowait;
            
            Ok((active as f64 / total as f64) * 100.0)
        } else {
            Ok(0.0)
        }
    }

    #[cfg(target_os = "windows")]
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

    #[cfg(target_os = "linux")]
    fn get_memory_usage(&self) -> color_eyre::Result<u64> {
        use std::fs;
        
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        let mut total = 0u64;
        let mut available = 0u64;
        
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line.split_whitespace().nth(1).unwrap_or("0").parse::<u64>().unwrap_or(0) * 1024;
            } else if line.starts_with("MemAvailable:") {
                available = line.split_whitespace().nth(1).unwrap_or("0").parse::<u64>().unwrap_or(0) * 1024;
            }
        }
        
        Ok(total - available)
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
        let gstreamer_version = gstreamer::version_string();
        let rust_version = env!("RUSTC_VERSION", "unknown").to_string();
        
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
            gstreamer_version,
            rust_version,
        })
    }

    #[cfg(target_os = "windows")]
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

    #[cfg(target_os = "linux")]
    fn get_os_version() -> String {
        use std::fs;
        
        if let Ok(release) = fs::read_to_string("/etc/os-release") {
            let mut name = String::new();
            let mut version = String::new();
            
            for line in release.lines() {
                if line.starts_with("NAME=") {
                    name = line.split('=').nth(1).unwrap_or("").trim_matches('"').to_string();
                } else if line.starts_with("VERSION=") {
                    version = line.split('=').nth(1).unwrap_or("").trim_matches('"').to_string();
                }
            }
            
            if !name.is_empty() && !version.is_empty() {
                format!("{} {}", name, version)
            } else if !name.is_empty() {
                name
            } else {
                "Linux (unknown distribution)".to_string()
            }
        } else {
            "Linux (unknown distribution)".to_string()
        }
    }

    #[cfg(target_os = "windows")]
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

    #[cfg(target_os = "linux")]
    fn get_cpu_model() -> String {
        use std::fs;
        
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
        
        "Unknown CPU".to_string()
    }

    #[cfg(target_os = "windows")]
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

    #[cfg(target_os = "linux")]
    fn get_total_memory_mb() -> color_eyre::Result<u64> {
        use std::fs;
        
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                let kb = line.split_whitespace().nth(1).unwrap_or("0").parse::<u64>().unwrap_or(0);
                return Ok(kb / 1024);
            }
        }
        
        Ok(0)
    }

    #[cfg(target_os = "windows")]
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
        
        (name, memory_mb, driver_version)
    }

    #[cfg(target_os = "linux")]
    fn get_gpu_info() -> (Option<String>, Option<u64>, Option<String>) {
        use std::process::Command;
        
        // Try to get GPU info from lspci
        let name = Self::get_gpu_name_linux();
        
        // Try to get GPU memory from nvidia-smi if available
        let memory_mb = Self::get_gpu_memory_linux();
        
        // Try to get driver version
        let driver_version = Self::get_gpu_driver_version_linux();
        
        (name, memory_mb, driver_version)
    }

    #[cfg(target_os = "linux")]
    fn get_gpu_name_linux() -> Option<String> {
        use std::process::Command;
        
        let output = Command::new("lspci")
            .args(&["-nn"])
            .output()
            .ok()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.contains("VGA compatible controller") || line.contains("3D controller") {
                let parts: Vec<&str> = line.split(": ").collect();
                if parts.len() >= 2 {
                    return Some(parts[1].to_string());
                }
            }
        }
        
        None
    }

    #[cfg(target_os = "linux")]
    fn get_gpu_memory_linux() -> Option<u64> {
        use std::process::Command;
        
        // Try nvidia-smi first
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = output_str.lines().next() {
                if let Ok(memory_mb) = line.trim().parse::<u64>() {
                    return Some(memory_mb);
                }
            }
        }
        
        // Try reading from /sys/class/drm for AMD/Intel GPUs
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("card") && !name.contains('-') {
                        let mem_path = format!("/sys/class/drm/{}/device/mem_info_vram_total", name);
                        if let Ok(content) = std::fs::read_to_string(&mem_path) {
                            if let Ok(bytes) = content.trim().parse::<u64>() {
                                return Some(bytes / (1024 * 1024));
                            }
                        }
                    }
                }
            }
        }
        
        None
    }

    #[cfg(target_os = "linux")]
    fn get_gpu_driver_version_linux() -> Option<String> {
        use std::process::Command;
        
        // Try nvidia-smi for NVIDIA drivers
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&["--query-gpu=driver_version", "--format=csv,noheader,nounits"])
            .output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = output_str.lines().next() {
                let version = line.trim();
                if !version.is_empty() {
                    return Some(format!("NVIDIA {}", version));
                }
            }
        }
        
        // Try reading from /sys/module for other drivers
        if let Ok(content) = std::fs::read_to_string("/sys/module/i915/version") {
            return Some(format!("Intel i915 {}", content.trim()));
        }
        
        if let Ok(content) = std::fs::read_to_string("/sys/module/amdgpu/version") {
            return Some(format!("AMD {}", content.trim()));
        }
        
        None
    }
}