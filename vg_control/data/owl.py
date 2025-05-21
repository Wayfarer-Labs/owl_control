from dotenv import load_dotenv
import os

load_dotenv()

import tarfile
import shutil
import json
from datetime import datetime

from ..constants import ROOT_DIR, MIN_FOOTAGE, MAX_FOOTAGE

from .input_utils.buttons import get_button_stats
from .input_utils.mouse import get_mouse_stats
from .uploader import upload_archive

# Directory structure might be nested, but the root dirs will always have a .mp4 and .csv

def filter_invalid_sample(vid_path, csv_path, meta_path, verbose = False):
    """
    Detect invalid videos
    """
    with open(meta_path) as f:
        metadata = json.load(f)
    duration = float(metadata['duration'])

    if duration < MIN_FOOTAGE:
        if verbose:
            print(f"Video length {duration:.2f} too short.")
        return True
    if duration > MAX_FOOTAGE + 10:
        if verbose:
            print(f"Video length {duration:.2f} too long.")
    
    bitrate = 2 # mbps
    # Get video file size in MB
    vid_size = os.path.getsize(vid_path) / (1024 * 1024)
    vid_size *= 8 # megabits
    expected_bits = bitrate * duration

    if vid_size < 0.25 * expected_bits:  # Less than quarter of expected size is unlikely
        if verbose:
            print(f"Video size {vid_size:.2f} Mb too small compared to expected {expected_bits:.2f} Mb")
        return True
    
    btn_stats = get_button_stats(csv_path)
    mouse_stats = get_mouse_stats(csv_path)

    # Filter out samples with too little activity
    if btn_stats['wasd_apm'] < 20:  # Less than 20 actions per minute is likely AFK/inactive
        if verbose:
            print(f"WASD actions per minute too low: {btn_stats['wasd_apm']:.1f}")
        return True
        
    if btn_stats['total_keyboard_events'] < 100:  # Too few keyboard events overall
        if verbose:
            print(f"Too few keyboard events: {btn_stats['total_keyboard_events']}")
        return True

    # Filter out samples with abnormal mouse behavior
    if mouse_stats['overall_max'] < 0.05:  # Very little mouse movement
        if verbose:
            print(f"Mouse movement too small: {mouse_stats['overall_max']:.3f}")
        return True
        
    if mouse_stats['overall_max'] > 500:  # Unreasonably large mouse movements
        if verbose:
            print(f"Mouse movement too large: {mouse_stats['overall_max']:.1f}")
        return True

    # Add stats to metadata
    extra_metadata = {
        'input_stats': {
            'wasd_apm': btn_stats['wasd_apm'],
            'unique_keys': btn_stats['unique_keys'], 
            'button_diversity': btn_stats['button_diversity'],
            'total_keyboard_events': btn_stats['total_keyboard_events'],
            'mouse_movement_std': mouse_stats['overall_std'],
            'mouse_x_std': mouse_stats['x_std'],
            'mouse_y_std': mouse_stats['y_std'],
            'mouse_max_movement': mouse_stats['overall_max'],
            'mouse_max_x': mouse_stats['max_x'],
            'mouse_max_y': mouse_stats['max_y']
        }
    }

    if not 'input_stats' in metadata:
        metadata.update(extra_metadata)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    return False

class OWLDataManager:
    def __init__(self, token):
        self.staged_files = []
        self.staging_dir = "staging"
        self.current_tar_uuid = None
        self.token = token
        os.makedirs(self.staging_dir, exist_ok=True)

    def stage(self, verbose = False):
        file_counter = 0
        for root, dirs, files in os.walk(ROOT_DIR):
            if '.uploaded' in files or '.invalid' in files:
                continue

            if verbose: 
                print(root, dirs, files)

            has_mp4 = any([fname.endswith('.mp4') for fname in files])
            has_csv = any([fname.endswith('.csv') for fname in files])
            has_metadata = any([fname == 'metadata.json' for fname in files])
                
            if has_mp4 and has_csv and has_metadata:
                mp4_file = next(f for f in files if f.endswith('.mp4'))
                csv_file = next(f for f in files if f.endswith('.csv'))
                metadata_file = 'metadata.json'
                
                new_mp4 = f"{file_counter:06d}.mp4"
                new_csv = f"{file_counter:06d}.csv"
                new_metadata = f"{file_counter:06d}.json"
                
                os.makedirs(self.staging_dir, exist_ok=True)
                
                # Copy files to staging with new names
                mp4_src_path = os.path.join(root, mp4_file)
                csv_src_path = os.path.join(root, csv_file)
                meta_src_path = os.path.join(root, metadata_file)

                try:
                    invalid = filter_invalid_sample(mp4_src_path, csv_src_path, meta_src_path, verbose = True)
                except:
                    invalid = True

                if invalid:
                    with open(os.path.join(root, '.invalid'), 'w') as f:
                        pass
                    continue

                shutil.copy2(mp4_src_path, os.path.join(self.staging_dir, new_mp4))
                shutil.copy2(csv_src_path, os.path.join(self.staging_dir, new_csv))
                shutil.copy2(meta_src_path, os.path.join(self.staging_dir, new_metadata))

                self.staged_files.append(root)
                file_counter += 1

    def compress(self):
        import uuid
        self.current_tar_uuid = uuid.uuid4().hex[:16]
        tar_name = f"{self.current_tar_uuid}.tar"
        
        with tarfile.open(tar_name, "w") as tar:
            for file in os.listdir(self.staging_dir):
                tar.add(os.path.join(self.staging_dir, file), arcname=file)
            
        return tar_name

    def upload(self):
        if not self.current_tar_uuid:
            raise Exception("Must compress before uploading")
            
        tar_name = f"{self.current_tar_uuid}.tar"
        
        try:
            OWL_TOKEN = None # TODO
            upload_archive(OWL_TOKEN, tar_name)
        finally:
            # Mark files as uploaded
            for staged_path in self.staged_files:
                with open(os.path.join(staged_path, '.uploaded'), 'w') as f:
                    f.write('')
                    
            # Cleanup
            if os.path.exists(tar_name):
                os.remove(tar_name)
            if os.path.exists(self.staging_dir):
                shutil.rmtree(self.staging_dir)
            self.staged_files = []
            self.current_tar_uuid = None

    def clear_upload_status(self):
        for root, dirs, files in os.walk(ROOT_DIR):
            if '.uploaded' in files:
                os.remove(os.path.join(root, '.uploaded'))

    def delete_uploaded(self):
        for root, dirs, files in os.walk(ROOT_DIR):
            if '.uploaded' in files:
                shutil.rmtree(root)


def upload_all_files(token, delete_uploaded=False):
    manager = OWLDataManager(token)
    manager.stage()
    manager.compress()
    manager.upload()

    if delete_uploaded:
        manager.delete_uploaded()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Error: Token argument is required")
        sys.exit(1)
        
    token = sys.argv[1]
    del_uploaded = False
    if len(sys.argv) > 2:
        del_uploaded = sys.argv[2].lower() == 'true'

    try:
        upload_all_files(token, delete_uploaded=del_uploaded)
        print("Upload completed successfully")
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        sys.exit(1)