import requests
import json
from ..constants import STACK_PROJECT_ID, STACK_PUBLISHABLE_CLIENT_KEY, STACK_SECRET_SERVER_KEY

class StackAuth:
    def __init__(self):
        self.project_id = STACK_PROJECT_ID
        self.client_key = STACK_PUBLISHABLE_CLIENT_KEY
        self.server_key = STACK_SECRET_SERVER_KEY
        self.base_url = "https://stack.auth.com/api"
        self.user_token = None
        
    def login(self, email, password):
        """Login with email and password"""
        url = f"{self.base_url}/v1/email/login"
        headers = {
            "Content-Type": "application/json",
            "X-Stack-Project-Id": self.project_id,
            "X-Stack-Client-Key": self.client_key
        }
        payload = {
            "email": email,
            "password": password
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            auth_data = response.json()
            self.user_token = auth_data.get("token")
            return True, auth_data
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def validate_api_key(self, api_key):
        """Validate a provided API key"""
        url = f"{self.base_url}/v1/api-keys/validate"
        headers = {
            "Content-Type": "application/json",
            "X-Stack-Project-Id": self.project_id,
            "X-Stack-Client-Key": self.client_key,
            "X-Stack-Api-Key": api_key
        }
        
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def logout(self):
        """Logout current user"""
        if not self.user_token:
            return True, "No active session"
            
        url = f"{self.base_url}/v1/sessions/logout"
        headers = {
            "Content-Type": "application/json",
            "X-Stack-Project-Id": self.project_id,
            "X-Stack-Client-Key": self.client_key,
            "Authorization": f"Bearer {self.user_token}"
        }
        
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            self.user_token = None
            return True, "Logged out successfully"
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def get_user_info(self):
        """Get current user information"""
        if not self.user_token:
            return False, "Not authenticated"
            
        url = f"{self.base_url}/v1/users/me"
        headers = {
            "Content-Type": "application/json",
            "X-Stack-Project-Id": self.project_id,
            "X-Stack-Client-Key": self.client_key,
            "Authorization": f"Bearer {self.user_token}"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.RequestException as e:
            return False, str(e)