# Controls and mouse vectors are extracted on a per frame basis
FPS = 60

# Root path to where all the data (mp4s and csvs) is
ROOT_DIR = "/mnt/c/Users/samib/OneDrive/Desktop/OWL/vg_control/data_dump/games/" + 'MCC-Win64-Shipping/'

# Keys to extract from the data
KEYBINDS = ["W","A","S","D","LMB","RMB"]

ACTION_DIM = len(KEYBINDS) + 2 # +2 for mouse movement