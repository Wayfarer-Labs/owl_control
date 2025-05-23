from typing import Callable

import obsws_python as obs
import time

import os
import subprocess

from ..constants import FPS

"""
OBS websocket association with params on OBS client is finnicky. Listing a few so I don't forget.

"SimpleOutput" "FilePath" <-- recording path
"SimpleOutput" "VBitrate" <-- video bitrate
- "Preset" = "veryfast"
- "RecQuality" = "Stream" 

You can see the rest in the %appdata% folder for obs-studio
There are ini files for the presets
"""

def try_launch_obs_if_not_open():
    # Check common installation paths
    obs_paths = [
        r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
        r"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe"
    ]
    
    for path in obs_paths:
        if os.path.exists(path):
            # Get the directory containing obs64.exe
            obs_dir = os.path.dirname(path)
            # Launch OBS from its directory
            subprocess.Popen([path], cwd=obs_dir)
            # Give OBS a moment to start up
            time.sleep(10)
            return
            
    raise FileNotFoundError("Could not find OBS installation")

class OBSClient:
    """
    Very simple client to interact with OBS. Assumes OBS is already focused on game (just set soruce to game capture and it should do this for you)
    """
    def __init__(
        self, 
        recording_path: str,
        start_callback : Callable,
        end_callback : Callable
    ):
        
        
        try:
            self.req_client = obs.ReqClient()
        except:
            try_launch_obs_if_not_open()
            self.req_client = obs.ReqClient()

        self.event_client = obs.EventClient()
        
        def on_record_state_changed(data):
            if str(data.output_state) == "OBS_WEBSOCKET_OUTPUT_STARTED":
                start_callback(time.perf_counter())
            if str(data.output_state) == "OBS_WEBSOCKET_OUTPUT_STOPPING":
                end_callback(time.perf_counter())

        self.event_client.callback.register(on_record_state_changed)
        self.req_client.set_profile_parameter("SimpleOutput", "FilePath", recording_path)

        # Monitor/resolution info
        ml = self.req_client.get_monitor_list().monitors
        primary_monitor = next(m for m in ml if m['monitorIndex'] == 0)
        monitor_height = primary_monitor['monitorHeight']
        monitor_width = primary_monitor['monitorWidth']

        self.req_client.set_video_settings(
            FPS,
            1,
            monitor_width,
            monitor_height,
            512,
            512
        )

        profile_list = self.req_client.get_profile_list()
        crnt_profile = profile_list.current_profile_name
        self.default_profile = crnt_profile
        if "owl_data_recorder" not in profile_list.profiles:
            self.req_client.create_profile("owl_data_recorder")

        # Init the scene with game
        scene_name_list = self.req_client.get_scene_list().scenes
        scene_name_list = [d['sceneName'] for d in scene_name_list]
        if not "owl_data_collection_scene" in scene_name_list:
            self.req_client.create_scene("owl_data_collection_scene")
        self.req_client.set_current_program_scene("owl_data_collection_scene")

        # Check if a game capture source exists
        inputs = self.req_client.get_input_list().inputs
        input_names = [d['inputName'] for d in inputs]
        if 'owl_game_capture' in input_names:
            # Delete it so we can recreate with right settings
            self.req_client.remove_input('owl_game_capture')
            time.sleep(0.5) # Give it a second to refresh

        self.req_client.create_input(
            "owl_data_collection_scene",
            "owl_game_capture",
            "game_capture",
            {
                'capture_mode' : 'any_fullscreen',
                'capture_audio' : True
            },
            True
        )

        # Audio settings
        self.req_client.set_input_volume('Mic/Aux', None, -100)
        self.req_client.set_input_volume('Desktop Audio', None, 0)

        # Find the owl game capture scene id
        item_list = self.req_client.get_scene_item_list("owl_data_collection_scene").scene_items
        owl_gc_id = None
        for item in item_list:
            if item['sourceName'] == 'owl_game_capture':
                owl_gc_id = item['sceneItemId']
                break
        new_tform_dict = {
            'position' : {
                'x' : 0.0, 'y' : 0.0
            },
            'rotation' : 0.0,
            'scale' : {
                'x' : 1.0, 'y' : 1.0
            }
        }
        self.req_client.set_scene_item_transform("owl_data_collection_scene", owl_gc_id, new_tform_dict)

        # Init the profile, make sure all params are right
        self.req_client.set_current_profile('owl_data_recorder')
        self.req_client.set_profile_parameter("SimpleOutput", "RecQuality", "Stream")
        self.req_client.set_profile_parameter("SimpleOutput", "VBitrate", "2000")
        self.req_client.set_profile_parameter("SimpleOutput", "Preset", "veryfast")
        self.req_client.set_profile_parameter("Output", "Mode", "Simple")
        self.req_client.set_profile_parameter("SimpleOutput", "RecFormat2", "mp4")
        #self.req_client.set_profile_parameter("SimpleOutput", "NVENCPreset2", "p7")
        self.req_client.set_profile_parameter("SimpleOutput", "StreamEncoder", "x264")
        self.req_client.set_profile_parameter("SimpleOutput", "Preset", "veryfast")
        #self.req_client.set_current_profile(self.default_profile)




    def init_configuration(self):
        self.req_client.set_current_profile('owl_data_recorder')

    def reset_configuration(self):
        self.req_client.set_current_profile(self.default_profile)
        

    def start_recording(self):
        #self.init_configuration()
        self.req_client.start_record()

    def stop_recording(self):
        #self.reset_configuration()
        self.req_client.stop_record()