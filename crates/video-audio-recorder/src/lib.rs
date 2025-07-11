use std::path::Path;

use color_eyre::{
    Result,
    eyre::{Context, ContextCompat as _, OptionExt as _, eyre},
};
use futures_util::StreamExt as _;
use gstreamer::{
    Pipeline,
    glib::object::Cast,
    prelude::{ElementExt as _, ElementExtManual as _, GObjectExtManualGst as _, GstBinExt as _},
};

pub use gstreamer;

pub mod metrics;
pub use metrics::{MetricsCollector, PerformanceMetrics, SystemInfo, set_gst_log_file};

pub struct DebugParameters {
    pub debug_level: Option<String>,
    pub save_debug_log: bool,
    pub gstreamer_logging_enabled: bool,
    pub gstreamer_tracing_enabled: bool,
}


fn create_pipeline(path: &Path, _pid: u32, hwnd: usize) -> Result<Pipeline> {
    // Loopback is bugged: gstreamer/gstreamer#4259
    // Add the following parameters once it's fixed: remove loopback=true and add "loopback-target-pid={pid} loopback-mode=include-process-tree"
    let video = format!(
            "
            d3d12screencapturesrc window-handle={hwnd}
            ! encoder.video_0

            wasapi2src loopback=true
            ! encoder.audio_0

            encodebin2 name=encoder profile=video/quicktime,variant=iso:video/x-raw,width=1920,height=1080,framerate=60/1->video/x-h264:audio/x-raw,channels=2,rate=48000->audio/mpeg,mpegversion=1,layer=3
            ! filesink name=filesink
        "
        );

    let pipeline = gstreamer::parse::launch(&video)?
        .dynamic_cast::<Pipeline>()
        .expect("Failed to cast element to pipeline");
    let filesink = pipeline
        .by_name("filesink")
        .wrap_err("Failed to find 'filesink' element")?;
    filesink.set_property_from_str(
        "location",
        path.to_str().ok_or_eyre("Path must be valid UTF-8")?,
    );

    tracing::debug!("Created pipeline");

    Ok(pipeline)
}

#[derive(derive_more::From, derive_more::Deref, derive_more::DerefMut)]
pub struct NullPipelineOnDrop(Pipeline);

impl Drop for NullPipelineOnDrop {
    fn drop(&mut self) {
        tracing::debug!("Setting pipeline to Null state on drop");
        if let Err(e) = self.set_state(gstreamer::State::Null) {
            tracing::error!(message = "Failed to set pipeline to Null state", error = ?e);
        } else {
            tracing::debug!("Set pipeline to Null state successfully");
        }
    }
}

pub struct WindowRecorder {
    pipeline: NullPipelineOnDrop,
    recording_path: std::path::PathBuf,
    metrics_collector: MetricsCollector,
}

impl WindowRecorder {
    pub fn start_recording(path: &Path, pid: u32, hwnd: usize, debug_params: Option<DebugParameters>) -> Result<WindowRecorder> {
        let pipeline = create_pipeline(path, pid, hwnd)?;
        let metrics_collector = MetricsCollector::new(debug_params.unwrap().debug_level)?;

        pipeline
            .set_state(gstreamer::State::Playing)
            .wrap_err("failed to set pipeline state to Playing")?;

        let recorder = WindowRecorder {
            pipeline: pipeline.into(),
            recording_path: path.to_path_buf(),
            metrics_collector,
        };

        Ok(recorder)
    }

    pub fn listen_to_messages(&self) -> impl Future<Output = Result<()>> + use<> {
        let bus = self.pipeline.bus().unwrap();

        // FIXME - a clone is absolutely not what we want here but I'm a rust n00b and can't get a
        // working mutable reference, help!
        //let &mut metrics = &mut (self.metrics_collector);

        async move {
            while let Some(msg) = bus.stream().next().await {
                use gstreamer::MessageView;

                match msg.view() {
                    MessageView::Eos(..) => {
                        tracing::debug!("Received EOS from bus");
                        break;
                    }
                    MessageView::Error(err) => {
                        //self.metrics_collector.record_encoding_error();
                        return Err(eyre!(err.error()).wrap_err("Received error message from bus"));
                    }
                    MessageView::Qos(_) => {
                        // QoS messages indicate quality issues, record as potential frame drops
                        //metrics.record_frame_drop();
                        tracing::warn!("QoS message received - potential performance issue");
                    }
                    MessageView::Warning(warning) => {
                        tracing::warn!("GStreamer warning: {}", warning.error());
                    }
                    _ => (),
                };
                // FIXME - this should probably be executed on a regular repeating timer, we can't
                // guarantee that enough events will come in to trigger this frequently enough.
                // Either way it should be rate limited, and won't work properly until I fix the
                // mutable issue above
                /*
                if let Err(e) = self.metrics_collector.sample_system_resources() {
                    tracing::warn!("Failed to sample system resources: {}", e);
                }
                */
            }
            Ok(())
        }
    }

    pub fn stop_recording(&self) {
        tracing::debug!("Sending EOS event to pipeline");
        self.pipeline.send_event(gstreamer::event::Eos::new());
        tracing::debug!("Sent EOS event to pipeline");
    }

    pub fn get_file_size(&self) -> u64 {
        std::fs::metadata(&self.recording_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }
    pub fn finalize_metrics(&self, file_size_bytes: u64) -> PerformanceMetrics {
        return self.metrics_collector.finalize_metrics(file_size_bytes);
    }
}
