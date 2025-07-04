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
}

impl WindowRecorder {
    pub fn start_recording(path: &Path, pid: u32, hwnd: usize) -> Result<WindowRecorder> {
        let pipeline = create_pipeline(path, pid, hwnd)?;
        pipeline
            .set_state(gstreamer::State::Playing)
            .wrap_err("failed to set pipeline state to Playing")?;
        Ok(WindowRecorder {
            pipeline: pipeline.into(),
        })
    }

    pub fn listen_to_messages(&self) -> impl Future<Output = Result<()>> + use<> {
        let bus = self.pipeline.bus().unwrap();
        async move {
            while let Some(msg) = bus.stream().next().await {
                use gstreamer::MessageView;

                match msg.view() {
                    MessageView::Eos(..) => {
                        tracing::debug!("Received EOS from bus");
                        break;
                    }
                    MessageView::Error(err) => {
                        return Err(eyre!(err.error()).wrap_err("Received error message from bus"));
                    }
                    _ => (),
                };
            }
            Ok(())
        }
    }

    pub fn stop_recording(&self) {
        tracing::debug!("Sending EOS event to pipeline");
        self.pipeline.send_event(gstreamer::event::Eos::new());
        tracing::debug!("Sent EOS event to pipeline");
    }
}
