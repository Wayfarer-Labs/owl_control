import os
import pandas as pd
from datetime import datetime

class Metadata:
    """
    When constructed, gets several pieces of information about the users setup.

    """
    def __init__(self, session_id):
        self.session_id = session_id
        self.data = {}
        self.path = None
        
        # Get hardware UUID
        try:
            with open('/sys/class/dmi/id/product_uuid', 'r') as f:
                self.hardware_id = f.read().strip()
        except:
            try:
                # Fallback for Windows
                import subprocess
                output = subprocess.check_output('wmic csproduct get uuid').decode()
                self.hardware_id = output.split('\n')[1].strip()
            except:
                self.hardware_id = None
        

    def reset(self, timestamp, path):
        self.data = {
            'session-id': self.session_id,
            'hardware-id': self.hardware_id,
            'start-timestamp': timestamp
        }
        self.path = path

    def end(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data.update({
            'end-timestamp': timestamp
        })

        import json
        metadata_path = os.path.join(self.path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.data, f, indent=4)
