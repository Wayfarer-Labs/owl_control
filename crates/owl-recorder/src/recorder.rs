use std::path::PathBuf;

use color_eyre::{Result, eyre::Context as _};
use tauri_winrt_notification::Toast;

use crate::{
    find_game::{Game, get_foregrounded_game},
    recording::{InputParameters, MetadataParameters, Recording, WindowParameters},
};

#[cfg(feature = "real-video")]
use video_audio_recorder::MetricsEvent;

pub(crate) struct Recorder<D> {
    recording_dir: D,
    games: Vec<Game>,
    recording: Option<Recording>,
    debug_level: Option<String>,
}

impl<D> Recorder<D>
where
    D: FnMut() -> PathBuf,
{
    pub(crate) fn new(recording_dir: D, games: Vec<Game>, debug_level: Option<String>) -> Self {
        Self {
            recording_dir,
            games,
            recording: None,
            debug_level,
        }
    }

    pub(crate) fn recording(&self) -> Option<&Recording> {
        self.recording.as_ref()
    }

    pub(crate) fn recording_mut(&mut self) -> Option<&mut Recording> {
        self.recording.as_mut()
    }

    pub(crate) async fn start(&mut self) -> Result<()> {
        if self.recording.is_some() {
            return Ok(());
        }

        let recording_location = (self.recording_dir)();

        std::fs::create_dir_all(&recording_location)
            .wrap_err("Failed to create recording directory")?;

        let Some((game_exe, pid, hwnd)) =
            get_foregrounded_game(&self.games).wrap_err("failed to get foregrounded game")?
        else {
            tracing::warn!("No game window found");
            Self::show_invalid_game_notification();
            return Ok(());
        };

        tracing::info!(
            game_exe,
            ?pid,
            ?hwnd,
            recording_location=%recording_location.display(),
            "Starting recording"
        );

        let recording = Recording::start(
            MetadataParameters {
                path: recording_location.join("metadata.json"),
                game_exe,
            },
            WindowParameters {
                path: recording_location.join("recording.mp4"),
                pid,
                hwnd,
            },
            InputParameters {
                path: recording_location.join("inputs.csv"),
            },
            self.debug_level.clone(),
        )
        .await?;

        Self::show_start_notification(recording.game_exe());

        self.recording = Some(recording);

        Ok(())
    }

    pub(crate) async fn seen_input(&mut self, e: raw_input::Event) -> Result<()> {
        let Some(recording) = self.recording.as_mut() else {
            return Ok(());
        };
        recording.seen_input(e).await?;
        Ok(())
    }

    pub(crate) fn sample_system_resources(&mut self) -> Result<()> {
        let Some(recording) = self.recording.as_mut() else {
            return Ok(());
        };
        recording.sample_system_resources()
    }

    pub(crate) fn handle_metrics_events(&mut self) {
        let Some(recording) = self.recording.as_mut() else {
            return;
        };

        // Process all available metrics events
        while let Some(event) = recording.try_recv_metrics_event() {
            recording.handle_metrics_event(event);
        }
    }

    pub(crate) async fn stop(&mut self) -> Result<()> {
        tracing::debug!("Recorder::stop() called");
        let Some(recording) = self.recording.take() else {
            tracing::debug!("No recording to stop");
            return Ok(());
        };

        tracing::debug!("Showing stop notification");
        Self::show_stop_notification(recording.game_exe());

        tracing::debug!("Calling recording.stop()");
        recording.stop().await?;
        tracing::debug!("Recording.stop() completed");

        Ok(())
    }

    fn show_start_notification(exe_name: &str) {
        if let Err(e) = Toast::new(Toast::POWERSHELL_APP_ID)
            .title("Started recording")
            .text1(&format!("Recording {exe_name}"))
            .sound(None)
            .show()
        {
            tracing::error!("Failed to show start notification: {e}");
        };
    }

    fn show_invalid_game_notification() {
        if let Err(e) = Toast::new(Toast::POWERSHELL_APP_ID)
            .title("Invalid game")
            .text1(&format!("Not recording foreground window."))
            .text2("It's either not a supported game or not fullscreen.")
            .sound(None)
            .show()
        {
            tracing::error!("Failed to show invalid game notification: {e}");
        };
    }

    fn show_stop_notification(exe_name: &str) {
        if let Err(e) = Toast::new(Toast::POWERSHELL_APP_ID)
            .title("Stopped recording")
            .text1(&format!("No longer recording {exe_name}"))
            .sound(None)
            .show()
        {
            tracing::error!("Failed to show stop notification: {e}");
        };
    }
}
