import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from .stack_client import StackAuth

class LoginWindow:
    def __init__(self, root, on_auth_success):
        self.root = root
        self.on_auth_success = on_auth_success
        self.auth_client = StackAuth()
        
        # Configure the root window
        self.root.title("VG Control - Login")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        # Set app style
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Arial", 12))
        self.style.configure("TButton", font=("Arial", 12))
        self.style.configure("TEntry", font=("Arial", 12))
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add logo/title
        self.title_label = ttk.Label(
            self.main_frame, 
            text="VG Control", 
            font=("Arial", 24, "bold")
        )
        self.title_label.pack(pady=(0, 20))
        
        # Create notebook for login methods
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create login tab
        self.login_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.login_frame, text="Login")
        
        # Email field
        ttk.Label(self.login_frame, text="Email:").pack(anchor="w", pady=(0, 5))
        self.email_var = tk.StringVar()
        self.email_entry = ttk.Entry(self.login_frame, textvariable=self.email_var, width=40)
        self.email_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Password field
        ttk.Label(self.login_frame, text="Password:").pack(anchor="w", pady=(0, 5))
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(self.login_frame, textvariable=self.password_var, show="*", width=40)
        self.password_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Login button
        self.login_button = ttk.Button(
            self.login_frame, 
            text="Login", 
            command=self.handle_login
        )
        self.login_button.pack(fill=tk.X, pady=(0, 20))
        
        # Create API key tab
        self.api_key_frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.api_key_frame, text="API Key")
        
        # API key field
        ttk.Label(self.api_key_frame, text="API Key:").pack(anchor="w", pady=(0, 5))
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(self.api_key_frame, textvariable=self.api_key_var, width=40)
        self.api_key_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Validate API key button
        self.api_key_button = ttk.Button(
            self.api_key_frame, 
            text="Validate API Key", 
            command=self.handle_api_key
        )
        self.api_key_button.pack(fill=tk.X, pady=(0, 20))
        
        # Load saved credentials if they exist
        self.load_saved_credentials()
        
    def handle_login(self):
        """Handle login button click"""
        email = self.email_var.get()
        password = self.password_var.get()
        
        if not email or not password:
            messagebox.showerror("Error", "Please enter both email and password")
            return
        
        # Show loading state
        self.login_button.config(state=tk.DISABLED)
        self.login_button.config(text="Logging in...")
        self.root.update()
        
        # Attempt login
        success, result = self.auth_client.login(email, password)
        
        # Reset button state
        self.login_button.config(state=tk.NORMAL)
        self.login_button.config(text="Login")
        
        if success:
            # Save credentials if login successful
            self.save_credentials("email", email)
            
            # Pass auth info to callback
            self.on_auth_success(self.auth_client)
            
            # Close login window
            self.root.destroy()
        else:
            messagebox.showerror("Login Failed", f"Error: {result}")
    
    def handle_api_key(self):
        """Handle API key validation"""
        api_key = self.api_key_var.get()
        
        if not api_key:
            messagebox.showerror("Error", "Please enter an API key")
            return
        
        # Show loading state
        self.api_key_button.config(state=tk.DISABLED)
        self.api_key_button.config(text="Validating...")
        self.root.update()
        
        # Attempt to validate API key
        success, result = self.auth_client.validate_api_key(api_key)
        
        # Reset button state
        self.api_key_button.config(state=tk.NORMAL)
        self.api_key_button.config(text="Validate API Key")
        
        if success:
            # Save credentials if API key is valid
            self.save_credentials("api_key", api_key)
            
            # Pass auth info to callback
            self.on_auth_success(self.auth_client)
            
            # Close login window
            self.root.destroy()
        else:
            messagebox.showerror("Validation Failed", f"Error: {result}")
    
    def save_credentials(self, cred_type, value):
        """Save credentials to file"""
        try:
            creds_dir = os.path.expanduser("~/.vg_control")
            os.makedirs(creds_dir, exist_ok=True)
            
            creds_file = os.path.join(creds_dir, "credentials.json")
            
            # Load existing credentials or create new
            if os.path.exists(creds_file):
                with open(creds_file, "r") as f:
                    creds = json.load(f)
            else:
                creds = {}
            
            # Update with new credential
            creds[cred_type] = value
            
            # Save to file
            with open(creds_file, "w") as f:
                json.dump(creds, f)
                
        except Exception as e:
            print(f"Error saving credentials: {e}")
    
    def load_saved_credentials(self):
        """Load saved credentials if they exist"""
        try:
            creds_file = os.path.expanduser("~/.vg_control/credentials.json")
            
            if os.path.exists(creds_file):
                with open(creds_file, "r") as f:
                    creds = json.load(f)
                
                # Populate email field if saved
                if "email" in creds:
                    self.email_var.set(creds["email"])
                
                # Populate API key field if saved
                if "api_key" in creds:
                    self.api_key_var.set(creds["api_key"])
                    
        except Exception as e:
            print(f"Error loading credentials: {e}")


def show_login_window(on_auth_success):
    """Show login window and wait for authentication"""
    root = tk.Tk()
    login_window = LoginWindow(root, on_auth_success)
    root.mainloop()
    
    # If window is closed without authentication
    return None