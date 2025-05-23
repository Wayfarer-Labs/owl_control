import os
import json
import hashlib
from typing import List, Optional, Dict
from datetime import datetime
import requests

from ..constants import API_BASE_URL
from ..metadata import get_hwid
from ..security import session_manager
from .uploader import get_upload_url, upload_archive as _legacy_upload


def create_secure_upload_payload(
    api_key: str,
    archive_path: str,
    session_tokens: List[Dict],
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> Dict:
    """Create a secure upload payload with session validation."""
    
    file_size_mb = os.path.getsize(archive_path) // (1024 * 1024)
    
    # Calculate archive hash
    archive_hash = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            archive_hash.update(chunk)
    
    payload = {
        "filename": os.path.basename(archive_path),
        "content_type": "application/x-tar",
        "file_size_mb": file_size_mb,
        "file_hash": archive_hash.hexdigest(),
        "expiration": 3600,
        "uploader_hwid": get_hwid(),
        "upload_timestamp": datetime.now().isoformat(),
        "session_tokens": session_tokens,
        "version": "2.0"  # New secure version
    }
    
    if tags:
        payload["tags"] = tags
        
    return payload


def get_secure_upload_url(
    api_key: str,
    archive_path: str,
    session_tokens: List[Dict],
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> str:
    """Request a pre-signed S3 URL with session validation."""
    
    payload = create_secure_upload_payload(
        api_key, archive_path, session_tokens, tags, base_url
    )
    
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    url = f"{base_url}/tracker/upload/game_control/secure"
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("url") or data.get("upload_url") or data["uploadUrl"]


def upload_archive_with_sessions(
    api_key: str,
    archive_path: str,
    session_dirs: List[str],
    tags: Optional[List[str]] = None,
    base_url: str = API_BASE_URL,
) -> None:
    """Upload an archive with session validation."""
    
    # Load sessions and generate tokens
    session_tokens = []
    for session_dir in session_dirs:
        session = session_manager.load_session_from_directory(session_dir)
        if session:
            token = session.generate_upload_token(api_key)
            session_tokens.append(token)
        else:
            raise ValueError(f"Invalid or missing session in {session_dir}")
    
    try:
        # Try secure upload first
        upload_url = get_secure_upload_url(
            api_key, archive_path, session_tokens, tags, base_url
        )
        
        # Use curl to upload with the secure URL
        import subprocess
        import shlex
        from tqdm import tqdm
        
        file_size = os.path.getsize(archive_path)
        curl_command = f'curl -X PUT "{upload_url}" -H "Content-Type: application/x-tar" -T "{archive_path}" -# -m 1200'
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading (secure)") as pbar:
            process = subprocess.Popen(
                shlex.split(curl_command),
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            
            last_update = 0
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                
                if "#" in line:
                    try:
                        percent = (line.count("#") / 50) * 100
                        current = int(file_size * (percent / 100))
                        if current > last_update:
                            pbar.n = current
                            pbar.refresh()
                            last_update = current
                    except:
                        continue

            return_code = process.wait()
            if return_code != 0:
                raise Exception(f"Secure upload failed with return code {return_code}")
                
    except Exception as e:
        print(f"Secure upload failed: {e}, falling back to legacy upload")
        # Fall back to legacy upload
        _legacy_upload(api_key, archive_path, tags, base_url)


def is_secure_upload_available(base_url: str = API_BASE_URL) -> bool:
    """Check if the server supports secure uploads."""
    try:
        url = f"{base_url}/tracker/upload/game_control/secure/status"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False