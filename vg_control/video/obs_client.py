from typing import Callable

import obsws_python as obs
import time

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
        

    def start_recording(self):
        self.req_client.start_record()

    def stop_recording(self):
        self.req_client.stop_record()