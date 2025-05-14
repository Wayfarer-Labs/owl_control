"""
Constants used throughout the application
"""

# Recording settings
ROOT_DIR = "./data_dump/games/" # User should be able to set this, but we will need to use it
FPS = 60  # Frames per second for tracking
POLLS_PER_FRAME = 4  # Number of input polls per frame

# API endpoints
API_BASE_URL = "https://api.openworldlabs.com"
UPLOAD_ENDPOINT = "/v1/upload"

# File formats
DATA_FILE_FORMAT = "vgc"  # VG Control data format
