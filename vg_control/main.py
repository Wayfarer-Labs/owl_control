import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Import internal modules
from input_tracking.tracker import InputTracker
from input_tracking.writer import InputWriter
from video.obs_client import OBSClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('vg_control')

class VGControl:
    def __init__(self, recording_path, output_path, api_key):
        self.recording_path = recording_path
        self.output_path = output_path
        self.api_key = api_key
        self.tracker = InputTracker()
        self.writer = InputWriter(output_path)
        self.obs_client = None
        self.recording_start_time = None
        self.recording_end_time = None
        self.running = False
        self.task = None

    def start_recording_callback(self, timestamp):
        """Called when OBS starts recording"""
        self.recording_start_time = timestamp
        logger.info(f"Recording started at {self.recording_start_time}")
        # Update the writer with the recording start time
        self.writer.set_recording_start_time(self.recording_start_time)

    def end_recording_callback(self, timestamp):
        """Called when OBS stops recording"""
        self.recording_end_time = timestamp
        logger.info(f"Recording ended at {self.recording_end_time}")
        # Finalize the output file
        self.writer.finalize(self.recording_end_time)
        
        # Stop the tracker if we're still running
        if self.running:
            self.stop()

    def setup_callbacks(self):
        """Set up the callbacks for the input tracker"""
        callbacks = {
            'mouse_move': self.writer.on_mouse_move,
            'mouse_button': self.writer.on_mouse_button,
            'mouse_scroll': self.writer.on_mouse_scroll,
            'keyboard': self.writer.on_keyboard
        }
        self.tracker.set_callbacks(callbacks)

    async def run(self):
        """Main run loop"""
        try:
            # Initialize OBS client
            self.obs_client = OBSClient(
                self.recording_path,
                self.start_recording_callback,
                self.end_recording_callback
            )
            
            # Set up callbacks
            self.setup_callbacks()
            
            # Start the input tracker
            self.running = True
            self.task = asyncio.create_task(self.tracker())
            
            # Start OBS recording
            self.obs_client.start_recording()
            
            # Run until stopped
            while self.running:
                await asyncio.sleep(0.1)  # Small sleep to reduce CPU usage
                
                # Check for input on stdin (for control from Electron)
                if sys.stdin.isatty():  # Only check if stdin is a TTY
                    for line in sys.stdin:
                        if line.strip().upper() == "STOP":
                            logger.info("Received stop command")
                            self.stop()
                            break
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()
            if self.task:
                await self.task

    def stop(self):
        """Stop recording and tracking"""
        if not self.running:
            return
            
        self.running = False
        logger.info("Stopping recording and tracking")
        
        # Stop OBS recording if it's active
        if self.obs_client:
            try:
                self.obs_client.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping OBS recording: {e}")
        
        # Stop the tracker
        self.tracker.stop()

async def main():
    parser = argparse.ArgumentParser(description='VG Control - Video Game Input Tracker')
    parser.add_argument('--recording-path', type=str, required=True, help='Path where OBS will save recordings')
    parser.add_argument('--output-path', type=str, required=True, help='Path where input data will be saved')
    parser.add_argument('--api-key', type=str, required=True, help='API key for data upload')
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting VG Control with recording path: {args.recording_path}, output path: {args.output_path}")
    
    controller = VGControl(args.recording_path, args.output_path, args.api_key)
    await controller.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        logger.info("VG Control has shut down")