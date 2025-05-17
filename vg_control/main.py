import asyncio
import psutil
from datetime import datetime
import os
import win32gui
import win32process
import time

from .input_tracking.writer import DataWriterClient
from .video.obs_client import OBSClient
from .constants import ROOT_DIR

def valid_game_names(fp = "game_list.txt"):
    with open(fp, 'r') as file:
        return [line.strip() for line in file if line.strip()]

class Recorder:
    def __init__(self):
        self.client = DataWriterClient()
        self.obs_client = None
    
        self.gpu_threshold = 50  # GPU usage threshold to detect game
        self.game_pid = None
        self.game_name = None

    async def main(self):

        while True:
            # Check for valid games in the process list
            valid_games = valid_game_names()
            processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.name().lower().endswith('.exe'):
                        for game in valid_games:
                            if game.lower() in proc.name().lower():
                                processes.append(proc)
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            print("Checked for game")
            if processes:
                print("Found game")
                # Choose the first matching game process
                self.game_pid = processes[0].pid
                self.game_name = processes[0].name()

                # Wait for game to be in focus
                while not self.is_game_in_focus():
                    await asyncio.sleep(1)

                # Create directory structure
                game_dir = os.path.join(ROOT_DIR, self.game_name.split('.')[0])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                recording_dir = os.path.join(game_dir, timestamp)
                os.makedirs(recording_dir, exist_ok=True)

                # Get full path and use forward slashes
                obs_recording_path = os.path.abspath(recording_dir).replace('\\', '/')

                print("Time 1")
                self.obs_client = OBSClient(
                    recording_path=obs_recording_path,  # OBS will save the MP4 in this directory
                    start_callback=self.client.writer.start_fn,
                    end_callback=self.client.writer.end_fn
                )
                print("Time 2")

                print("Game Detected. Starting Recording")
                self.obs_client.start_recording()
                
                csv_filename = os.path.join(recording_dir, "inputs.csv")
                client_start_time = await self.client.start(csv_filename)

                # Monitor game focus
                while self.is_game_in_focus():
                    await asyncio.sleep(1)

                # Game out of focus or closed
                self.obs_client.stop_recording()
                await self.client.end()
                del self.obs_client
                self.obs_client = None
                self.game_pid = None
                self.game_name = None

            await asyncio.sleep(1)  # Check every second

    def is_game_in_focus(self):
        try:
            fore_hwnd = win32gui.GetForegroundWindow()
            _, fore_pid = win32process.GetWindowThreadProcessId(fore_hwnd)
            return fore_pid == self.game_pid
        except:
            return False
        
if __name__ == "__main__":
    import asyncio

    async def main():
        recorder = Recorder()
        try:
            await recorder.main()
        except KeyboardInterrupt:
            print("Recorder stopped by user.")
        finally:
            if recorder.obs_client:
                recorder.obs_client.stop_recording()
            if recorder.client:
                await recorder.client.end()

    asyncio.run(main())
