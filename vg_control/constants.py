"""
Constants used throughout the application
"""

# Recording settings
FPS = 60  # Frames per second for tracking
MIN_FOOTAGE = 30 # This many seconds needed before something is "worth saving"
MAX_FOOTAGE = 60*10 # How long should each individual clip be?
INACTIVITY_TIME = 20 # No input for this long = stop recording
TIME_TO_STOP = 5 # How long do we think it will take OBS to stop recording?

# API endpoints
API_BASE_URL = "https://api.openworldlabs.ai"
UPLOAD_ENDPOINT = "/v1/upload"

# File formats
DATA_FILE_FORMAT = "vgc"  # OWL Control data format
ROOT_DIR = "./data_dump/games/" # User should be able to set this, but we will need to use it
POLLS_PER_FRAME=20 # Hard coded

GAME_LIST = [
    'doomthedarkages',
    'DOOMEternalx64vk',
    'blackopscoldwar'
]
