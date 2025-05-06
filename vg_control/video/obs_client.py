from typing import Callable

import obsws_python as obs
import time

BITRATE = 2000
RESOLUTION = 512
ENCODING = "P7"

"""
OBS websocket association with params on OBS client is finnicky. Listing a few so I don't forget.

"SimpleOutput" "FilePath" <-- recording path
"SimpleOutput" "VBitrate" <-- video bitrate
- "Preset" = "veryfast"
- "RecQuality" = "Stream" 

You can see the rest in the %appdata% folder for obs-studio
There are ini files for the presets
"""
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
        
        self.req_client = obs.ReqClient()
        self.event_client = obs.EventClient()
        
        def on_record_state_changed(data):
            if str(data.output_state) == "OBS_WEBSOCKET_OUTPUT_STARTED":
                start_callback(time.perf_counter())
            if str(data.output_state) == "OBS_WEBSOCKET_OUTPUT_STOPPING":
                end_callback(time.perf_counter())

        self.event_client.callback.register(on_record_state_changed)
        self.req_client.set_profile_parameter("SimpleOutput", "FilePath", recording_path)

        profile_list = self.req_client.get_profile_list()
        crnt_profile = profile_list.current_profile_name
        self.default_profile = crnt_profile
        if "owl_data_recorder" not in profile_list.profiles:
            self.req_client.create_profile("owl_data_recorder")

        # Init the profile, make sure everything matches defaults
        self.req_client.set_current_profile('owl_data_recorder')
        self.req_client.set_profile_parameter("SimpleOutput", "RecQuality", "Stream")
        self.req_client.set_profile_parameter("SimpleOutput", "VBitrate", "2000")
        self.req_client.set_profile_parameter("SimpleOutput", "Preset", "veryfast")
        self.req_client.set_profile_parameter("Video", "OutputCX", "512")
        self.req_client.set_profile_parameter("Video", "OutputCY", "512")
        self.req_client.set_profile_parameter("Video", "FPSType", "0")
        self.req_client.set_profile_parameter("Video", "FPSCommon", "60")
        self.req_client.set_profile_parameter("Video", "FPSInt", "60")
        self.req_client.set_profile_parameter("Output", "Mode", "Simple")
        self.req_client.set_profile_parameter("SimpleOutput", "RecFormat2", "mp4")
        #self.req_client.set_profile_parameter("SimpleOutput", "NVENCPreset2", "p7")
        self.req_client.set_profile_parameter("SimpleOutput", "StreamEncoder", "x264")
        self.req_client.set_profile_parameter("SimpleOutput", "Preset", "veryfast")
        #self.req_client.set_current_profile(self.default_profile)
        self.req_client.set_profile_parameter("Video", "OutputCX", "512")
        self.req_client.set_profile_parameter("Video", "OutputCY", "512")

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