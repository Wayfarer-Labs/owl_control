import os
import pandas as pd
from datetime import datetime
import time
import json

def get_hwid():
    try:
        with open('/sys/class/dmi/id/product_uuid', 'r') as f:
            hardware_id = f.read().strip()
    except:
        try:
            # Fallback for Windows
            import subprocess
            output = subprocess.check_output('wmic csproduct get uuid').decode()
            hardware_id = output.split('\n')[1].strip()
        except:
            hardware_id = None
    return hardware_id

class Metadata:
    """
    When constructed, gets several pieces of information about the users setup.

    """
    def __init__(self, session_id):
        self.session_id = session_id
        self.data = {}
        self.path = None
        
        self.hardware_id = get_hwid()
        
    def reset(self, timestamp, path):
        self.data = {
            'session-id': self.session_id,
            'hardware-id': self.hardware_id,
            'start-timestamp': timestamp
        }
        self.path = path
        self.start_time = time.time()

    def get_duration(self):
        return time.time() - self.start_time
    
    def end(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data.update({
            'end-timestamp': timestamp,
            'duration' : self.get_duration()
        })

        metadata_path = os.path.join(self.path, "metadata.json")

        with open(metadata_path, 'w') as f:
            json.dump(self.data, f, indent=4)