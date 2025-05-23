import os
from typing import List, Optional
from datetime import datetime
import requests
import urllib3
import httpx
import subprocess
import shlex
from tqdm import tqdm

from ..constants import API_BASE_URL
from ..metadata import get_hwid

def _get_upload_url(
    api_key: str,
    archive_path: str,
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> str:
    """Request a pre-signed S3 URL for uploading a tar archive."""

    file_size_mb = os.path.getsize(archive_path) // (1024 * 1024)
    payload = {
        "filename": os.path.basename(archive_path),
        "content_type": "application/x-tar", 
        "file_size_mb": file_size_mb,
        "expiration": 3600,
        "uploader_hwid": get_hwid(),
        "upload_timestamp": datetime.now().isoformat()
    }
    if tags:
        payload["tags"] = tags

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    url = f"{base_url}/tracker/upload/game_control"

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("url") or data.get("upload_url") or data["uploadUrl"]

def _upload_archive(
    api_key: str,
    archive_path: str,
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> None:
    """Upload an archive to the storage bucket via a pre-signed URL."""

    upload_url = get_upload_url(
        api_key,
        archive_path,
        tags=tags,
        base_url=base_url,
    )

    with open(archive_path, "rb") as f:
        put_resp = requests.put(upload_url, data=f, timeout=60, verify=False)
        put_resp.raise_for_status()

def get_upload_url(
    api_key: str,
    archive_path: str,
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> str:
    """Request a pre-signed S3 URL for uploading a tar archive."""

    file_size = os.path.getsize(archive_path)
    file_size_mb = file_size // (1024 * 1024)
    payload = {
        "filename": os.path.basename(archive_path),
        "content_type": "application/x-tar", 
        "file_size_mb": file_size_mb,
        "expiration": 3600,
        "uploader_hwid": get_hwid(),
        "upload_timestamp": datetime.now().isoformat()
    }
    if tags:
        payload["tags"] = tags

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    url = f"{base_url}/tracker/upload/game_control"

    with requests.Session() as session:
        response = session.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("url") or data.get("upload_url") or data["uploadUrl"]

def upload_archive(
    api_key: str,
    archive_path: str,
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> None:
    """Upload an archive to the storage bucket via a pre-signed URL."""

    upload_url = get_upload_url(
        api_key,
        archive_path,
        tags=tags,
        base_url=base_url,
    )

    # Get file size for progress bar
    file_size = os.path.getsize(archive_path)
    
    # Use -# for a simpler progress indicator that's easier to parse
    curl_command = f'curl -X PUT "{upload_url}" -H "Content-Type: application/x-tar" -T "{archive_path}" -# -m 1200'
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
        process = subprocess.Popen(
            shlex.split(curl_command),
            stderr=subprocess.PIPE,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        last_update = 0
        # Read curl's progress output
        while True:
            line = process.stderr.readline()
            if not line:
                break
            
            # Update progress bar based on curl's output
            if "#" in line:
                try:
                    # Extract percentage from the number of # characters
                    percent = (line.count("#") / 50) * 100  # curl uses 50 # chars for 100%
                    current = int(file_size * (percent / 100))
                    # Only update if we've made progress to avoid unnecessary refreshes
                    if current > last_update:
                        pbar.n = current
                        pbar.refresh()
                        last_update = current
                except:
                    continue

        # Wait for process to complete
        return_code = process.wait()
        if return_code != 0:
            raise Exception(f"Upload failed with return code {return_code}")
