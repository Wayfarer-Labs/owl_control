import os
from typing import List, Optional
from datetime import datetime
import requests
import urllib3
import httpx

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

    print("====")
    print(upload_url)
    print("====")
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

    file_size = os.path.getsize(archive_path)
    
    headers = {
        'Content-Type': 'application/x-tar',
        'Content-Length': str(file_size),
        'Transfer-Encoding': 'chunked'
    }

    # Use a chunk size of 5MB
    CHUNK_SIZE = 5 * 1024 * 1024  

    def file_stream():
        with open(archive_path, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    with httpx.Client(
        verify=False,
        timeout=httpx.Timeout(timeout=300.0, read=300.0, write=300.0, connect=60.0)
    ) as client:
        put_resp = client.put(
            upload_url,
            content=file_stream(),
            headers=headers
        )
        put_resp.raise_for_status()