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
    'DOOMEternalx64vk',
    'DOOMx64',
    'DOOMTheDarkAges',
    'FactoryGame',
    'Titanfall2',
    'SkyrimSE',
    'TESV',
    'Crysis3','Crysis3_x64','Crysis2','Crysis2_x64',
    'OblivionRemastered',
    'MCC-Win64-Shipping',
    'farcry3', 'fc3', 'farcry4', 'farcry5', 'fc3_blooddragon',
    'Cyberpunk2077'
]
