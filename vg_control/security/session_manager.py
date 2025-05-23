import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from typing import Dict, Optional, Tuple
from datetime import datetime

from ..metadata import get_hwid


class RecordingSession:
    """Represents a secure recording session with cryptographic validation."""
    
    def __init__(self, session_id: str, game_name: str, recording_dir: str):
        self.session_id = session_id
        self.game_name = game_name
        self.recording_dir = recording_dir
        self.created_at = time.time()
        self.completed = False
        self.files = {}
        
        # Generate session secret
        self.session_secret = secrets.token_hex(32)
        
        # Session metadata
        self.metadata = {
            "session_id": session_id,
            "game_name": game_name,
            "hwid": get_hwid(),
            "created_at": self.created_at,
            "recording_dir": recording_dir,
            "version": "1.0"
        }
        
    def add_file(self, filename: str, filepath: str) -> Dict:
        """Add a recorded file to the session."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        file_hash = self._calculate_file_hash(filepath)
        file_size = os.path.getsize(filepath)
        
        file_data = {
            "filename": filename,
            "path": filepath,
            "hash": file_hash,
            "size": file_size,
            "added_at": time.time()
        }
        
        self.files[filename] = file_data
        return file_data
        
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def complete_recording(self, video_path: str, csv_path: str) -> Dict:
        """Mark recording as complete and generate session manifest."""
        self.add_file("video.mp4", video_path)
        self.add_file("inputs.csv", csv_path)
        self.completed = True
        self.completed_at = time.time()
        
        # Generate session manifest
        manifest = {
            "session": self.metadata,
            "files": self.files,
            "completed_at": self.completed_at,
            "duration": self.completed_at - self.created_at
        }
        
        # Calculate manifest signature
        manifest_json = json.dumps(manifest, sort_keys=True)
        signature = hmac.new(
            self.session_secret.encode(),
            manifest_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Save session data
        session_data = {
            "manifest": manifest,
            "signature": signature,
            "session_secret": self.session_secret
        }
        
        # Save to recording directory
        session_file = os.path.join(self.recording_dir, ".session")
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
            
        return session_data
        
    def generate_upload_token(self, api_key: str) -> Dict:
        """Generate a secure upload token for this session."""
        if not self.completed:
            raise ValueError("Cannot generate upload token for incomplete session")
            
        # Create upload claim
        upload_claim = {
            "session_id": self.session_id,
            "hwid": get_hwid(),
            "timestamp": time.time(),
            "files": {
                name: {
                    "hash": data["hash"],
                    "size": data["size"]
                }
                for name, data in self.files.items()
            }
        }
        
        # Sign claim with combined secret
        combined_secret = f"{api_key}:{self.session_secret}:{self.session_id}"
        claim_json = json.dumps(upload_claim, sort_keys=True)
        claim_signature = hmac.new(
            combined_secret.encode(),
            claim_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "claim": upload_claim,
            "signature": claim_signature,
            "session_id": self.session_id
        }


class SessionManager:
    """Manages recording sessions across the application."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sessions = {}
            cls._instance.active_session = None
        return cls._instance
        
    def create_session(self, game_name: str, recording_dir: str) -> RecordingSession:
        """Create a new recording session."""
        session_id = str(uuid.uuid4())
        session = RecordingSession(session_id, game_name, recording_dir)
        self.sessions[session_id] = session
        self.active_session = session
        return session
        
    def get_session(self, session_id: str) -> Optional[RecordingSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
        
    def get_active_session(self) -> Optional[RecordingSession]:
        """Get the currently active recording session."""
        return self.active_session
        
    def complete_active_session(self, video_path: str, csv_path: str) -> Dict:
        """Complete the active recording session."""
        if not self.active_session:
            raise ValueError("No active session to complete")
            
        result = self.active_session.complete_recording(video_path, csv_path)
        self.active_session = None
        return result
        
    def load_session_from_directory(self, directory: str) -> Optional[RecordingSession]:
        """Load a session from a recording directory."""
        session_file = os.path.join(directory, ".session")
        if not os.path.exists(session_file):
            return None
            
        try:
            with open(session_file, "r") as f:
                session_data = json.load(f)
                
            # Verify signature
            manifest = session_data["manifest"]
            signature = session_data["signature"]
            session_secret = session_data["session_secret"]
            
            manifest_json = json.dumps(manifest, sort_keys=True)
            expected_signature = hmac.new(
                session_secret.encode(),
                manifest_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if signature != expected_signature:
                return None  # Invalid signature
                
            # Reconstruct session
            session_id = manifest["session"]["session_id"]
            game_name = manifest["session"]["game_name"]
            session = RecordingSession(session_id, game_name, directory)
            session.metadata = manifest["session"]
            session.files = manifest["files"]
            session.completed = True
            session.completed_at = manifest["completed_at"]
            session.session_secret = session_secret
            
            return session
        except Exception:
            return None


# Global session manager instance
session_manager = SessionManager()