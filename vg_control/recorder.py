import asyncio
import psutil
import os
from datetime import datetime
import win32gui
import win32process
import time
from typing import Optional
import uuid

from .input_tracking.writer import DataWriterClient
from .video.obs_client import OBSClient
from .constants import ROOT_DIR, GAME_LIST
from .input_tracking.rawinputlib import wait_until_input
from .metadata import Metadata

class Recorder:
    """
    Versatile recording class.

    Records after start_recording is called.
    1. Moves onto new recording after 10min of recording.
    2. Pauses recording if no input for 10sec
    3. Cancels recording if game closed minimized
    4. Only records full screen apps that are in game_list
    """
    def __init__(self):
        self.client = DataWriterClient()
        self.obs_client = None
        self.game_pid: Optional[int] = None
        self.game_name: Optional[str] = None
        
        # Tasks
        self.focus_monitor_task = None
        self.inactivity_monitor_task = None
        self.recording_duration_task = None
        self.input_wait_task = None
        
        # Control flags
        self.is_recording = False
        self.is_interrupted = False
        self.should_end = False

        self.session_id = uuid.uuid4()
        self.metadata = Metadata(self.session_id)

    def _get_game_process(self):
        """Find first matching game process from GAME_LIST"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.name().lower().endswith('.exe'):
                    for game in GAME_LIST:
                        if game.lower() in proc.name().lower():
                            return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return None

    def is_game_in_focus(self):
        """Check if game window is in focus"""
        try:
            fore_hwnd = win32gui.GetForegroundWindow()
            _, fore_pid = win32process.GetWindowThreadProcessId(fore_hwnd)
            return fore_pid == self.game_pid
        except:
            return False

    def is_game_running(self):
        """Check if game process still exists"""
        try:
            return psutil.Process(self.game_pid).is_running()
        except:
            return False

    async def _monitor_focus(self):
        """Monitor game focus and interrupt recording when not focused"""
        while not self.should_end:
            if not self.is_game_in_focus():
                if not self.is_interrupted:
                    self.is_interrupted = True
                    await self._interrupt_recording()
                
                # Wait for game to come back to focus
                while not self.is_game_in_focus() and not self.should_end:
                    await asyncio.sleep(0.1)
                    
                if not self.should_end:
                    self.is_interrupted = False
                    await self._resume_recording()
            await asyncio.sleep(0.1)

    async def _monitor_inactivity(self):
        """Monitor for input inactivity"""
        while not self.should_end:
            time_since_last = self.client.get_time_since_last_event()
            
            if time_since_last > 600:  # 10 minutes
                await self.stop_recording()
                return
                
            elif time_since_last > 10 and not self.is_interrupted:  # 10 seconds
                self.is_interrupted = True
                await self._interrupt_recording()
                
                if not self.should_end:
                    self.input_wait_task = asyncio.create_task(wait_until_input())
                    try:
                        await self.input_wait_task
                        if not self.should_end:
                            self.is_interrupted = False
                            await self._resume_recording()
                    except asyncio.CancelledError:
                        pass
                        
            await asyncio.sleep(1)

    async def _monitor_duration(self):
        """Monitor recording duration and restart after 10 minutes"""
        while not self.should_end:
            await asyncio.sleep(600)  # 10 minutes
            if not self.should_end:
                self.is_interrupted = True
                await self._interrupt_recording()
                await self._resume_recording(new_file=True)
                self.is_interrupted = False

    async def _setup_recording(self, new_file=False):
        """Setup recording directory and clients"""
        if not new_file:
            game_dir = os.path.join(ROOT_DIR, self.game_name.split('.')[0])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_dir = os.path.join(game_dir, timestamp)
        os.makedirs(recording_dir, exist_ok=True)
        
        obs_recording_path = os.path.abspath(recording_dir).replace('\\', '/')
        
        self.obs_client = OBSClient(
            recording_path=obs_recording_path,
            start_callback=self.client.writer.start_fn,
            end_callback=self.client.writer.end_fn
        )
        
        csv_filename = os.path.join(recording_dir, "inputs.csv")
        await self.client.start(csv_filename)
        self.metadata.reset(timestamp, recording_dir)
        self.obs_client.start_recording()

    async def _interrupt_recording(self):
        """Pause recording"""
        if self.obs_client:
            self.obs_client.stop_recording()
            await self.client.end()
            self.metadata.end()

    async def _resume_recording(self, new_file=False):
        """Resume recording, optionally to a new file"""
        if not self.should_end:
            await self._setup_recording(new_file)

    async def _cleanup(self):
        """Clean up all tasks and clients"""
        self.should_end = True
        
        if self.focus_monitor_task:
            self.focus_monitor_task.cancel()
        if self.inactivity_monitor_task:
            self.inactivity_monitor_task.cancel()
        if self.recording_duration_task:
            self.recording_duration_task.cancel()
        if self.input_wait_task:
            self.input_wait_task.cancel()
            
        if self.obs_client:
            self.obs_client.stop_recording()
        if self.client:
            await self.client.end()
        if self.metadata:
            self.metadata.end()
            
        self.obs_client = None
        self.game_pid = None
        self.game_name = None
        self.is_recording = False

    async def start_recording(self):
        """Start recording the active game"""
        if self.is_recording:
            return
            
        proc = self._get_game_process()
        if not proc:
            return
            
        self.game_pid = proc.pid
        self.game_name = proc.name()
        
        # Wait for game to be in focus
        while not self.is_game_in_focus():
            await asyncio.sleep(0.1)
            
        await self._setup_recording()
        
        self.is_recording = True
        self.should_end = False
        
        # Start monitoring tasks
        self.focus_monitor_task = asyncio.create_task(self._monitor_focus())
        self.inactivity_monitor_task = asyncio.create_task(self._monitor_inactivity())
        self.recording_duration_task = asyncio.create_task(self._monitor_duration())

    async def stop_recording(self):
        """Stop recording and clean up"""
        if not self.is_recording:
            return
            
        await self._cleanup()

async def main():
    from .input_tracking.rawinputlib import AsyncHotkeyManager

    recorder = Recorder()
    hotkeys = AsyncHotkeyManager()

    hotkeys.add_callback(57, recorder.start_recording)
    hotkeys.add_callback(48, recorder.stop_recording)

if __name__ == "__main__":
    asyncio.run(main())