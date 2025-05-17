import os
from typing import List, Optional

import requests

from ..constants import API_BASE_URL


def get_upload_url(
    api_key: str,
    archive_path: str,
    video_filename: str,
    control_filename: str,
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
        "video_filename": os.path.basename(video_filename),
        "control_filename": os.path.basename(control_filename),
    }
    if tags:
        payload["tags"] = tags

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    url = f"{base_url}/tracker/upload/game_control"
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("url") or data.get("upload_url") or data["uploadUrl"]


def upload_archive(
    api_key: str,
    archive_path: str,
    video_filename: str,
    control_filename: str,
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> None:
    """Upload an archive to the storage bucket via a pre-signed URL."""

    upload_url = get_upload_url(
        api_key,
        archive_path,
        video_filename,
        control_filename,
        tags=tags,
        base_url=base_url,
    )

    with open(archive_path, "rb") as f:
        put_resp = requests.put(upload_url, data=f, timeout=60)
        put_resp.raise_for_status()
