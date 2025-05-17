from dotenv import load_dotenv
import boto3
import os

load_dotenv()

import tarfile
import shutil

s3 = boto3.client(
    service_name='s3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
)

"""
Notes on approach:

There is a folder ROOT_DIR with many folders containing a .mp4 and .csv
- If anything has already been uploaded, the folder should contain a .uploaded file
- There should be some code to perform cleanup and delete anything with a .uploaded file
- We'll get everything in the ROOT_DIR and copy it to a staging directory "staging"
- If this code fails, "staging" directory should be deleted
- When files are moved to staging directory, their names are made numerical
- I.e. the first video pulled will become 000000.mp4 and 000000.csv, then 000001.{csv|mp4}, etc.
- After all files have been staged, turn staging folder into a tar
- Name of tar should be random 16 digit alphanumeric uuid {uuid}.tar
- Uploaded to s3 after
"""

from ..constants import ROOT_DIR

# Directory structure might be nested, but the root dirs will always have a .mp4 and .csv

class TigrisDataManager:
    def __init__(self):
        self.staged_files = []
        self.staging_dir = "staging"
        self.current_tar_uuid = None
        os.makedirs(self.staging_dir, exist_ok=True)

    def stage(self):
        file_counter = 0
        for root, dirs, files in os.walk(ROOT_DIR):
            if '.uploaded' in files:
                continue

            print(root, dirs, files)

            has_mp4 = any([fname.endswith('.mp4') for fname in files])
            has_csv = any([fname.endswith('.csv') for fname in files])
                
            if has_mp4 and has_csv:
                mp4_file = next(f for f in files if f.endswith('.mp4'))
                csv_file = next(f for f in files if f.endswith('.csv'))
                
                new_mp4 = f"{file_counter:06d}.mp4"
                new_csv = f"{file_counter:06d}.csv"
                
                os.makedirs(self.staging_dir, exist_ok=True)
                
                # Copy files to staging with new names
                shutil.copy2(os.path.join(root, mp4_file), os.path.join(self.staging_dir, new_mp4))
                shutil.copy2(os.path.join(root, csv_file), os.path.join(self.staging_dir, new_csv))

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
            s3.upload_file(
                tar_name,
                "game-data", 
                tar_name
            )
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


if __name__ == "__main__":
    manager = TigrisDataManager()
    manager.clear_upload_status()
    manager.stage()
    manager.compress() 
    manager.upload()
    manager.clear_upload_status()
    #manager.delete_uploaded()