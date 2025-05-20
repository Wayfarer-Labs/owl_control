import asyncio
import os
import shutil
from datetime import datetime
import win32gui
import win32api
import win32process
import psutil
import warnings
import uuid
from typing import Optional

from .input_tracking.writer import DataWriterClient
from .video.obs_client import OBSClient
from .metadata import Metadata
from .constants import (
    ROOT_DIR, GAME_LIST,
    INACTIVITY_TIME, MIN_FOOTAGE, MAX_FOOTAGE,
    TIME_TO_STOP
)
from .input_tracking.rawinputlib import (
    HotkeyManager,
    wait_until_input, wait_until_idle,
    RAW_INPUT
)

class SimpleRecorder:
    def __init__(self):
        self.client = DataWriterClient()
        self.obs_client = None
        self.game_pid: Optional[int] = None
        self.game_name: Optional[str] = None
        self.is_recording = False
        self.recording_dir = None
        self.metadata = Metadata(str(uuid.uuid4()))

        self.idle_task = None
        self.game_running_task = None
        self.timer_task = None

    async def detect_idleing(self):
        await wait_until_idle(INACTIVITY_TIME)
        await self.stop_recording(cancel_idle_task=False)
        # Wait until further input then go again
        await wait_until_input()
        await self.start_recording()

    def is_window_fullscreen(self, hwnd):
        try:
            window_rect = win32gui.GetWindowRect(hwnd)
            monitor = win32api.MonitorFromWindow(hwnd)
            monitor_info = win32api.GetMonitorInfo(monitor)
            monitor_rect = monitor_info['Monitor']
            return (window_rect == monitor_rect)
        except:
            return False

    def _get_game_process(self):
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.name().lower().endswith('.exe'):
                    for game in GAME_LIST:
                        if game.lower() in proc.name().lower():
                            hwnd = win32gui.FindWindow(None, win32gui.GetWindowText(win32gui.GetForegroundWindow()))
                            _, pid = win32process.GetWindowThreadProcessId(hwnd)
                            if pid == proc.pid:
                                return proc, self.is_window_fullscreen(hwnd)
            except:
                continue
        return None, False
    
    async def check_game_running(self):
        while True:
            try:
                process = psutil.Process(self.game_pid)
                if not process.is_running():
                    await self.stop_recording(cancel_gr_task=False)
                    break
            except psutil.NoSuchProcess:
                await self.stop_recording(cancel_gr_task=False)
                break
            await asyncio.sleep(0.5)
    
    async def stop_after_time(self):
        await asyncio.sleep(MAX_FOOTAGE)
        await self.stop_recording(cancel_timer_task=False)
        await self.start_recording()

    async def start_recording(self):
        if self.is_recording:
            return

        proc, is_fullscreen = self._get_game_process()
        if not proc:
            return
        if not is_fullscreen:
            warnings.warn("Game not fullscreen. Recording not started.")
            return

        self.game_pid = proc.pid
        self.game_name = proc.name()

        # Setup recording directory
        game_dir = os.path.join(ROOT_DIR, self.game_name.split('.')[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_dir = os.path.join(game_dir, timestamp)
        os.makedirs(self.recording_dir, exist_ok=True)

        # Initialize clients
        obs_path = os.path.abspath(self.recording_dir).replace('\\', '/')
        self.obs_client = OBSClient(
            recording_path=obs_path,
            start_callback=self.client.writer.start_fn,
            end_callback=self.client.writer.end_fn
        )

        csv_path = os.path.join(self.recording_dir, "inputs.csv")
        await self.client.start(csv_path)
        self.metadata.reset(timestamp, self.recording_dir)
        self.obs_client.start_recording()
        self.is_recording = True


        self.game_running_task = asyncio.create_task(
            self.check_game_running(), 
            name="game_monitor"
        )

        self.timer_task = asyncio.create_task(
            self.stop_after_time(),
            name="timer"
        )

        self.idle_task = asyncio.create_task(
            self.detect_idleing(),
            name="idle_detector"
        )

    async def stop_recording(self, cancel_gr_task = True, cancel_timer_task = True, cancel_idle_task = True):
        if not self.is_recording:
            return
        
        if cancel_gr_task and self.game_running_task is not None:
            self.game_running_task.cancel()
            self.game_running_task = None

        if cancel_timer_task and self.timer_task is not None:
            self.timer_task.cancel()
            self.timer_task = None

        if cancel_idle_task and self.idle_task is not None:
            self.idle_task.cancel()
            self.idle_task = None

        duration = self.metadata.get_duration()
        self.obs_client.stop_recording()
        await self.client.end()
        self.metadata.end()
        await asyncio.sleep(TIME_TO_STOP)
        
        if duration < MIN_FOOTAGE:
            # Mark directory as uploaded and invalid since recording was too short
            try:
                with open(os.path.join(self.recording_dir, '.uploaded'), 'w') as f:
                    pass
                with open(os.path.join(self.recording_dir, '.invalid'), 'w') as f:
                    pass
            except:
                print("Failed to mark directory as uploaded/invalid")

        self.obs_client = None
        self.game_pid = None
        self.game_name = None
        self.recording_dir = None
        self.is_recording = False

async def main():
    recorder = SimpleRecorder()
    hotkeys = HotkeyManager()

    hotkeys.add_callback('F4', recorder.start_recording)
    hotkeys.add_callback('F5', recorder.stop_recording)

    try:
        await hotkeys.event_loop()
    except KeyboardInterrupt:
        if recorder.is_recording:
            await recorder.stop_recording()
    finally:
        RAW_INPUT.close()

if __name__ == "__main__":
    asyncio.run(main())
